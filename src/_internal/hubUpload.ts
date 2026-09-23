/**
 * Explicit publication of a local artifact folder to the Hugging Face Hub
 * (Python ``HfApi.create_repo(exist_ok=True)`` + ``HfApi.upload_folder``).
 * ``PretrainedModule.pushToHub`` calls {@link uploadFolder}.
 *
 * The transport is the public Hub HTTP API over ``fetch``:
 *
 * 1. ``POST {endpoint}/api/repos/create`` (``409`` means the repository exists);
 * 2. ``POST {endpoint}/api/models/{repo}/preupload/{revision}`` classifies each
 *    file as ``regular`` or ``lfs`` (weights and large files use Git LFS);
 * 3. LFS files: ``POST {endpoint}/{repo}.git/info/lfs/objects/batch`` with
 *    SHA-256 object ids, then the returned ``PUT`` upload (basic or multipart)
 *    and optional verification;
 * 4. ``POST {endpoint}/api/models/{repo}/commit/{revision}`` with an NDJSON body:
 *    a ``header`` line, base64 ``file`` lines and ``lfsFile`` pointer lines.
 *
 * Tokens come from the options, ``HF_TOKEN`` or the token file saved by
 * ``hf auth login``. Tokens are sent only to the Hub endpoint, never to
 * presigned upload URLs, and are never included in errors or logs.
 */
import { createHash } from 'node:crypto';
import { readdir, readFile, stat } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import { ValueError } from '../errors.js';
import { DEFAULT_ENDPOINT, HubError, env, hfHome, validateRepoId } from './hub.js';

export interface UploadFolderOptions {
  repoId: string;
  folderPath: string;
  private?: boolean;
  revision?: string | null;
  token?: string | null;
  commitMessage?: string;
  /** Optional commit description (NDJSON header ``description``). */
  commitDescription?: string;
  /** Hub endpoint (default ``$HF_ENDPOINT`` or https://huggingface.co). */
  endpoint?: string | null;
  /** Injected transport for tests (default ``globalThis.fetch``). */
  fetch?: typeof fetch;
}

export interface UploadResult {
  /** Commit URL or identifier reported by the Hub. */
  commit: string;
  /** Commit object id when reported. */
  oid?: string | null;
  /** Repository-relative paths uploaded, with their upload mode. */
  files?: { path: string; mode: 'regular' | 'lfs' }[];
}

interface LocalFile {
  path: string;
  bytes: Uint8Array;
  sha256: string;
}

interface LfsAction {
  href: string;
  header?: Record<string, string>;
}

/** Files at least this large always use LFS when the Hub does not classify them. */
const LFS_THRESHOLD = 10 * 1024 * 1024;
const PREUPLOAD_CHUNK = 256;

/** ``HF_TOKEN``/``HUGGING_FACE_HUB_TOKEN`` or the saved token file (``HF_TOKEN_PATH``). */
export async function resolveUploadToken(token: string | null | undefined): Promise<string | null> {
  if (token) return token;
  const fromEnv = env('HF_TOKEN') ?? env('HUGGING_FACE_HUB_TOKEN');
  if (fromEnv) return fromEnv;
  try {
    const saved = (await readFile(env('HF_TOKEN_PATH') ?? join(hfHome(), 'token'), 'utf8')).trim();
    return saved || null;
  } catch {
    return null;
  }
}

async function collectFiles(folder: string): Promise<LocalFile[]> {
  const result: LocalFile[] = [];
  const visit = async (directory: string): Promise<void> => {
    const entries = (await readdir(directory, { withFileTypes: true })).sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    for (const entry of entries) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === '.git' || entry.name === '.cache') continue;
        await visit(path);
      } else if (entry.isFile()) {
        const buffer = await readFile(path);
        const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        result.push({
          path: relative(folder, path).split(sep).join('/'), bytes, sha256: createHash('sha256').update(bytes).digest('hex'),
        });
      } else {
        throw new ValueError(`cannot upload non-regular file ${entry.name}; artifact folders contain only files`);
      }
    }
  };
  await visit(folder);
  return result;
}

function base64(bytes: Uint8Array): string {
  return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
}

async function failure(response: Response, what: string): Promise<HubError> {
  let detail = '';
  try {
    const text = await response.text();
    try {
      const body = JSON.parse(text) as { error?: unknown; message?: unknown };
      const message = body.error ?? body.message;
      if (typeof message === 'string') detail = `: ${message}`;
    } catch {
      if (text && text.length < 300) detail = `: ${text}`;
    }
  } catch {
    // ignore unreadable bodies
  }
  return new HubError(`${what} failed with HTTP ${response.status}${detail}`, response.status);
}

class HubClient {
  constructor(
    readonly endpoint: string,
    private readonly token: string,
    private readonly transport: typeof fetch,
  ) {}

  /** Authenticated request to the Hub endpoint. */
  async hub(path: string, init: RequestInit & { headers?: Record<string, string> }, what: string, accept: number[] = []): Promise<Response> {
    const response = await this.transport(`${this.endpoint}${path}`, {
      ...init, headers: { ...(init.headers ?? {}), authorization: `Bearer ${this.token}` },
    });
    if (!response.ok && !accept.includes(response.status)) throw await failure(response, what);
    return response;
  }

  /** Unauthenticated request to a presigned storage URL (optionally with action headers). */
  async external(url: string, init: RequestInit, what: string): Promise<Response> {
    const response = await this.transport(url, init);
    if (!response.ok) throw await failure(response, what);
    return response;
  }
}

async function uploadLfsObject(client: HubClient, file: LocalFile, actions: { upload?: LfsAction; verify?: LfsAction }): Promise<void> {
  const upload = actions.upload;
  if (upload) {
    const header = { ...(upload.header ?? {}) };
    const chunkSize = header.chunk_size !== undefined ? Number(header.chunk_size) : null;
    if (chunkSize !== null) {
      // Multipart: numbered part URLs in the header, then a completion POST.
      if (!Number.isInteger(chunkSize) || chunkSize <= 0) throw new HubError('malformed LFS multipart chunk size');
      const parts: { partNumber: number; etag: string }[] = [];
      const count = Math.ceil(file.bytes.byteLength / chunkSize);
      for (let part = 1; part <= count; part += 1) {
        const url = header[String(part)];
        if (typeof url !== 'string') throw new HubError(`missing LFS multipart URL for part ${part}`);
        const chunk = file.bytes.subarray((part - 1) * chunkSize, part * chunkSize);
        const response = await client.external(url, { method: 'PUT', body: chunk }, `LFS part upload of ${file.path}`);
        const etag = response.headers.get('etag');
        if (!etag) throw new HubError(`LFS part upload of ${file.path} returned no ETag`);
        parts.push({ partNumber: part, etag });
      }
      await client.external(upload.href, {
        method: 'POST', headers: { 'content-type': 'application/vnd.git-lfs+json', accept: 'application/vnd.git-lfs+json' },
        body: JSON.stringify({ oid: file.sha256, parts }),
      }, `LFS multipart completion of ${file.path}`);
    } else {
      await client.external(upload.href, { method: 'PUT', headers: header, body: file.bytes }, `LFS upload of ${file.path}`);
    }
  }
  const verify = actions.verify;
  if (verify) {
    await client.external(verify.href, {
      method: 'POST', headers: { ...(verify.header ?? {}), 'content-type': 'application/vnd.git-lfs+json', accept: 'application/vnd.git-lfs+json' },
      body: JSON.stringify({ oid: file.sha256, size: file.bytes.byteLength }),
    }, `LFS verification of ${file.path}`);
  }
}

/** Create the model repository if needed and upload every file in ``folderPath`` in one commit. */
export async function uploadFolder(options: UploadFolderOptions): Promise<UploadResult> {
  const { repoId, folderPath } = options;
  validateRepoId(repoId);
  if (typeof folderPath !== 'string' || !folderPath || !(await stat(folderPath)).isDirectory()) {
    throw new ValueError('folderPath must be an existing directory');
  }
  const revision = options.revision ?? 'main';
  if (typeof revision !== 'string' || !revision) throw new ValueError('revision must be a nonempty string');
  const token = await resolveUploadToken(options.token);
  if (!token) {
    throw new HubError('publishing requires a Hugging Face token: pass token, set HF_TOKEN, or run `hf auth login`', 401);
  }
  const transport = options.fetch ?? globalThis.fetch;
  if (typeof transport !== 'function') throw new HubError('global fetch is unavailable; supply options.fetch');
  const endpoint = (options.endpoint ?? env('HF_ENDPOINT') ?? DEFAULT_ENDPOINT).replace(/\/+$/, '');
  const client = new HubClient(endpoint, token, transport);
  const files = await collectFiles(folderPath);
  if (!files.length) throw new ValueError('folderPath contains no files to upload');

  // 1. Repository (exist_ok=True).
  const [organization, name] = repoId.includes('/') ? repoId.split('/') as [string, string] : [null, repoId];
  await client.hub('/api/repos/create', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name, ...(organization ? { organization } : {}), private: Boolean(options.private), type: 'model' }),
  }, `creating repository ${repoId}`, [409]);

  // 2. Upload modes.
  const encodedRevision = encodeURIComponent(revision);
  const modes = new Map<string, 'regular' | 'lfs'>();
  const ignored = new Set<string>();
  for (let start = 0; start < files.length; start += PREUPLOAD_CHUNK) {
    const chunk = files.slice(start, start + PREUPLOAD_CHUNK);
    const response = await client.hub(`/api/models/${repoId}/preupload/${encodedRevision}`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ files: chunk.map((file) => ({ path: file.path, sample: base64(file.bytes.subarray(0, 512)), size: file.bytes.byteLength })) }),
    }, `classifying files for ${repoId}`);
    const body = await response.json() as { files?: { path?: unknown; uploadMode?: unknown; shouldIgnore?: unknown }[] };
    for (const entry of body.files ?? []) {
      if (typeof entry.path !== 'string') continue;
      if (entry.uploadMode === 'lfs' || entry.uploadMode === 'regular') modes.set(entry.path, entry.uploadMode);
      if (entry.shouldIgnore === true) ignored.add(entry.path);
    }
  }
  const selected = files.filter((file) => !ignored.has(file.path));
  const modeOf = (file: LocalFile): 'regular' | 'lfs' => modes.get(file.path)
    ?? (file.bytes.byteLength >= LFS_THRESHOLD || file.path.endsWith('.safetensors') ? 'lfs' : 'regular');

  // 3. LFS objects.
  const lfs = selected.filter((file) => modeOf(file) === 'lfs');
  if (lfs.length) {
    const response = await client.hub(`/${repoId}.git/info/lfs/objects/batch`, {
      method: 'POST',
      headers: { accept: 'application/vnd.git-lfs+json', 'content-type': 'application/vnd.git-lfs+json' },
      body: JSON.stringify({
        operation: 'upload', transfers: ['basic', 'multipart'], hash_algo: 'sha256', ref: { name: revision },
        objects: lfs.map((file) => ({ oid: file.sha256, size: file.bytes.byteLength })),
      }),
    }, `requesting LFS upload for ${repoId}`);
    const body = await response.json() as {
      objects?: { oid?: unknown; error?: { message?: unknown }; actions?: { upload?: LfsAction; verify?: LfsAction } }[];
    };
    for (const file of lfs) {
      const object = (body.objects ?? []).find((item) => item.oid === file.sha256);
      if (!object) throw new HubError(`LFS batch response omitted ${file.path}`);
      if (object.error) throw new HubError(`LFS upload of ${file.path} rejected: ${String(object.error.message ?? 'error')}`);
      if (object.actions) await uploadLfsObject(client, file, object.actions);
    }
  }

  // 4. Commit.
  const lines: unknown[] = [{
    key: 'header', value: { summary: options.commitMessage ?? 'Upload folder using huggingface_hub', description: options.commitDescription ?? '' },
  }];
  for (const file of selected) {
    lines.push(modeOf(file) === 'lfs'
      ? { key: 'lfsFile', value: { path: file.path, algo: 'sha256', oid: file.sha256, size: file.bytes.byteLength } }
      : { key: 'file', value: { content: base64(file.bytes), path: file.path, encoding: 'base64' } });
  }
  const response = await client.hub(`/api/models/${repoId}/commit/${encodedRevision}`, {
    method: 'POST', headers: { 'content-type': 'application/x-ndjson' },
    body: `${lines.map((line) => JSON.stringify(line)).join('\n')}\n`,
  }, `committing to ${repoId}@${revision}`);
  const result = await response.json() as { commitUrl?: unknown; commitOid?: unknown };
  const oid = typeof result.commitOid === 'string' ? result.commitOid : null;
  const commit = typeof result.commitUrl === 'string' ? result.commitUrl
    : `${endpoint}/${repoId}/commit/${oid ?? revision}`;
  return { commit, oid, files: selected.map((file) => ({ path: file.path, mode: modeOf(file) })) };
}
