/**
 * Explicit Hugging Face Hub downloads through ``fetch``.
 *
 * Files are stored in the standard Hub cache layout shared with Python's
 * ``huggingface_hub``::
 *
 *     <cache>/models--<org>--<name>/refs/<revision>        (commit sha)
 *     <cache>/models--<org>--<name>/snapshots/<sha>/<file>
 *
 * so artifacts downloaded by either implementation are reused by the other.
 * Nothing here runs on import. Loading never executes downloaded code.
 */
import { mkdir, readFile, rename, stat, writeFile, rm } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join, resolve, sep } from 'node:path';
import { ValueError } from '../errors.js';

export const DEFAULT_ENDPOINT = 'https://huggingface.co';

export interface HubOptions {
  /** Branch, tag or commit sha (default ``main``). Pin a commit for reproducibility. */
  revision?: string | null;
  /** Never contact the network; resolve from the local cache only. */
  localFilesOnly?: boolean;
  /** Hub cache directory (default ``$HF_HUB_CACHE`` or ``$HF_HOME/hub`` or ``~/.cache/huggingface/hub``). */
  cacheDir?: string | null;
  /** Access token (default ``$HF_TOKEN`` or the token saved by ``hf auth login``). */
  token?: string | null;
  /** Download only files matching these glob patterns. */
  allowPatterns?: readonly string[] | null;
  /** Hub endpoint (default ``$HF_ENDPOINT`` or https://huggingface.co). */
  endpoint?: string | null;
  /** Injected transport for tests. */
  fetch?: typeof fetch;
}

export class HubError extends Error {
  readonly status: number | null;
  constructor(message: string, status: number | null = null, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'HubError';
    this.status = status;
  }
}

/** The local file or directory is absent (Python ``FileNotFoundError``). */
export class FileNotFoundError extends Error {
  readonly path: string;
  constructor(path: string, message = `No such file or directory: ${path}`) {
    super(message);
    this.name = 'FileNotFoundError';
    this.path = path;
  }
}

/** A nonempty environment variable, or ``undefined``. */
export function env(name: string): string | undefined {
  const processLike = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  const value = processLike?.env?.[name];
  return value === undefined || value === '' ? undefined : value;
}

export function expandUser(path: string): string {
  if (path === '~') return homedir();
  if (path.startsWith(`~${sep}`) || path.startsWith('~/')) return join(homedir(), path.slice(2));
  return path;
}

export function hfHome(): string {
  return expandUser(env('HF_HOME') ?? join(env('XDG_CACHE_HOME') ?? join(homedir(), '.cache'), 'huggingface'));
}

export function defaultCacheDir(): string {
  return expandUser(env('HF_HUB_CACHE') ?? env('HUGGINGFACE_HUB_CACHE') ?? join(hfHome(), 'hub'));
}

export function hubOffline(): boolean {
  const value = (env('HF_HUB_OFFLINE') ?? '').toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

async function resolveToken(token: string | null | undefined): Promise<string | null> {
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

export function validateRepoId(repoId: string): void {
  if (typeof repoId !== 'string' || !/^[A-Za-z0-9][\w.-]*(\/[\w.-]+)?$/.test(repoId) || repoId.includes('..') || repoId.includes('--')) {
    throw new ValueError(`invalid Hugging Face repository id: ${JSON.stringify(repoId)}`);
  }
}

function repoFolder(repoId: string): string {
  return `models--${repoId.replace(/\//g, '--')}`;
}

function isCommitSha(revision: string): boolean {
  return /^[0-9a-f]{40}$/.test(revision);
}

/** ``fnmatch``-style glob matching used by ``allowPatterns``. */
export function globMatch(pattern: string, name: string): boolean {
  let source = '^';
  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index]!;
    if (char === '*') source += '.*';
    else if (char === '?') source += '.';
    else if (char === '[') {
      const end = pattern.indexOf(']', index + 1);
      if (end < 0) source += '\\[';
      else {
        let body = pattern.slice(index + 1, end);
        if (body.startsWith('!')) body = `^${body.slice(1)}`;
        source += `[${body.replace(/\\/g, '\\\\')}]`;
        index = end;
      }
    } else source += char.replace(/[.+^${}()|\\/]/g, '\\$&');
  }
  return new RegExp(`${source}$`, 's').test(name);
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

/** Whether ``path`` exists and is a directory. */
export async function isDirectory(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory();
  } catch {
    return false;
  }
}

function safeRelative(file: string): string {
  if (!file || isAbsolute(file) || file.split(/[\\/]/).some((part) => part === '..' || part === '')) {
    throw new ValueError(`unsafe repository file path: ${file}`);
  }
  return file;
}

async function cachedSnapshot(repoId: string, revision: string, cacheDir: string): Promise<string | null> {
  const root = join(cacheDir, repoFolder(repoId));
  let sha = revision;
  if (!isCommitSha(revision)) {
    try {
      sha = (await readFile(join(root, 'refs', revision), 'utf8')).trim();
    } catch {
      return null;
    }
  }
  const snapshot = join(root, 'snapshots', sha);
  return (await isDirectory(snapshot)) ? snapshot : null;
}

interface RevisionInfo {
  sha: string;
  files: string[];
}

async function fetchRevision(repoId: string, revision: string, endpoint: string, token: string | null, transport: typeof fetch): Promise<RevisionInfo> {
  const url = `${endpoint}/api/models/${repoId}/revision/${encodeURIComponent(revision)}`;
  const response = await transport(url, { headers: token ? { authorization: `Bearer ${token}` } : {} });
  if (!response.ok) {
    throw new HubError(`Hub revision lookup failed for ${repoId}@${revision}: HTTP ${response.status}`, response.status);
  }
  const body = await response.json() as { sha?: unknown; siblings?: unknown };
  if (typeof body.sha !== 'string' || !isCommitSha(body.sha) || !Array.isArray(body.siblings)) {
    throw new HubError(`malformed Hub revision response for ${repoId}@${revision}`);
  }
  const files = body.siblings.map((item) => (item as { rfilename?: unknown }).rfilename)
    .filter((name): name is string => typeof name === 'string');
  return { sha: body.sha, files };
}

async function downloadFile(url: string, target: string, token: string | null, transport: typeof fetch): Promise<void> {
  const response = await transport(url, { headers: token ? { authorization: `Bearer ${token}` } : {}, redirect: 'follow' });
  if (!response.ok) throw new HubError(`Hub download failed (${url}): HTTP ${response.status}`, response.status);
  const bytes = new Uint8Array(await response.arrayBuffer());
  await mkdir(dirname(target), { recursive: true });
  const temporary = `${target}.incomplete-${process.pid}-${Date.now()}`;
  try {
    await writeFile(temporary, bytes);
    await rename(temporary, target);
  } finally {
    await rm(temporary, { force: true });
  }
}

/**
 * Download (or resolve from cache) a model repository snapshot and return its
 * local directory. ``localFilesOnly`` (or ``HF_HUB_OFFLINE=1``) never touches
 * the network. A network failure falls back to a matching cached snapshot.
 */
export async function snapshotDownload(repoId: string, options: HubOptions = {}): Promise<string> {
  validateRepoId(repoId);
  const revision = options.revision ?? 'main';
  const cacheDir = expandUser(options.cacheDir ?? defaultCacheDir());
  const endpoint = (options.endpoint ?? env('HF_ENDPOINT') ?? DEFAULT_ENDPOINT).replace(/\/+$/, '');
  const patterns = options.allowPatterns ?? null;
  const offline = options.localFilesOnly || hubOffline();
  if (offline) {
    const cached = await cachedSnapshot(repoId, revision, cacheDir);
    if (!cached) {
      throw new FileNotFoundError(join(cacheDir, repoFolder(repoId)),
        `no cached snapshot of ${repoId}@${revision}; disable localFilesOnly to download it`);
    }
    return cached;
  }
  const transport = options.fetch ?? globalThis.fetch;
  if (typeof transport !== 'function') throw new HubError('global fetch is unavailable; supply options.fetch');
  const token = await resolveToken(options.token);
  let info: RevisionInfo;
  try {
    info = await fetchRevision(repoId, revision, endpoint, token, transport);
  } catch (error) {
    if (error instanceof HubError && error.status !== null) throw error;
    const cached = await cachedSnapshot(repoId, revision, cacheDir);
    if (cached) return cached;
    throw new HubError(`could not reach the Hub for ${repoId}@${revision}`, null, { cause: error });
  }
  const root = join(cacheDir, repoFolder(repoId));
  const snapshot = join(root, 'snapshots', info.sha);
  const wanted = info.files.filter((file) => !patterns || patterns.some((pattern) => globMatch(pattern, file)));
  for (const file of wanted) {
    const relative = safeRelative(file);
    const target = join(snapshot, relative);
    if (!resolve(target).startsWith(resolve(snapshot) + sep)) throw new ValueError(`unsafe repository file path: ${file}`);
    if (await exists(target)) continue;
    const url = `${endpoint}/${repoId}/resolve/${info.sha}/${relative.split('/').map(encodeURIComponent).join('/')}`;
    await downloadFile(url, target, token, transport);
  }
  await mkdir(snapshot, { recursive: true });
  if (!isCommitSha(revision) || revision !== info.sha) {
    await mkdir(join(root, 'refs', dirname(revision)), { recursive: true });
    await writeFile(join(root, 'refs', revision), info.sha);
  }
  return snapshot;
}

const CONVERSION_PR_TITLE = 'Adding `safetensors` variant of this model';

async function hubJson(url: string, token: string | null, transport: typeof fetch): Promise<unknown> {
  const response = await transport(url, { headers: token ? { authorization: `Bearer ${token}` } : {} });
  if (!response.ok) throw new HubError(`Hub request failed (${url}): HTTP ${response.status}`, response.status);
  return response.json();
}

/**
 * transformers' safetensors auto-conversion lookup (``get_conversion_pr_reference``):
 * for a repository whose ``main`` branch has only PyTorch weights, the open
 * "Adding `safetensors` variant of this model" pull request (by SFconvertbot
 * for public repositories) based on the current ``main`` commit provides
 * ``model.safetensors``. Returns ``refs/pr/<n>``, or ``null`` offline, for a
 * pinned revision, or when no such PR exists. Unlike Python, TypeScript never
 * asks the Hub to create a conversion.
 */
export async function safetensorsConversionRevision(repoId: string, options: HubOptions = {}): Promise<string | null> {
  const revision = options.revision ?? 'main';
  const disabled = ['1', 'ON', 'YES', 'TRUE'].includes((env('DISABLE_SAFETENSORS_CONVERSION') ?? '').toUpperCase());
  if (revision !== 'main' || options.localFilesOnly || hubOffline() || disabled) return null;
  validateRepoId(repoId);
  const endpoint = (options.endpoint ?? env('HF_ENDPOINT') ?? DEFAULT_ENDPOINT).replace(/\/+$/, '');
  const transport = options.fetch ?? globalThis.fetch;
  if (typeof transport !== 'function') return null;
  const token = await resolveToken(options.token);
  const info = await hubJson(`${endpoint}/api/models/${repoId}`, token, transport) as { private?: unknown };
  const commits = async (reference: string): Promise<string[]> => {
    const list = await hubJson(`${endpoint}/api/models/${repoId}/commits/${encodeURIComponent(reference)}`, token, transport);
    return Array.isArray(list) ? list.map((item) => String((item as { id?: unknown }).id)) : [];
  };
  const mainCommit = (await commits('main'))[0];
  for (let page = 0; page < 100; page += 1) {
    const body = await hubJson(`${endpoint}/api/models/${repoId}/discussions?p=${page}`, token, transport) as { discussions?: unknown };
    const discussions = Array.isArray(body.discussions) ? body.discussions as Record<string, unknown>[] : [];
    if (!discussions.length) break;
    for (const discussion of discussions) {
      if (discussion.title !== CONVERSION_PR_TITLE || discussion.status !== 'open' || discussion.isPullRequest !== true) continue;
      const reference = `refs/pr/${String(discussion.num)}`;
      const history = await commits(reference);
      if (history[1] !== mainCommit) continue;
      const author = (discussion.author as { name?: unknown } | undefined)?.name;
      if (info.private !== true && author !== 'SFconvertbot') continue;
      return reference;
    }
  }
  return null;
}

/** Whether ``source`` must be treated as a local path rather than a Hub id. */
export function looksLikeLocalPath(source: string): boolean {
  return isAbsolute(source) || source.startsWith('.') || source.startsWith('~');
}

/**
 * Resolve a local directory or Hub repository id to a local directory
 * (Python ``Path(x).is_dir()`` else ``snapshot_download``).
 */
export async function resolveArtifactDirectory(source: string, options: HubOptions = {}): Promise<{ path: string; remote: boolean }> {
  if (typeof source !== 'string' || !source) throw new TypeError('source must be a nonempty path or repository id');
  const local = resolve(expandUser(source));
  if (await isDirectory(local)) return { path: local, remote: false };
  if (looksLikeLocalPath(source)) throw new FileNotFoundError(local, `local model directory not found: ${local}`);
  return { path: await snapshotDownload(source, options), remote: true };
}
