/** Hub publication transport (``uploadFolder``) against a fake Hub server; no network. */
import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { uploadFolder } from '../../src/_internal/hubUpload.js';
import { HubError } from '../../src/_internal/hub.js';
import { scratchDirectory } from './helpers.js';

const scratch = scratchDirectory('tensorcode-upload-');

interface Recorded {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: Uint8Array | string | null;
}

function headersOf(init?: RequestInit): Record<string, string> {
  const headers: Record<string, string> = {};
  new Headers(init?.headers).forEach((value, key) => { headers[key] = value; });
  return headers;
}

function bodyOf(init?: RequestInit): Uint8Array | string | null {
  const body = init?.body;
  if (body === undefined || body === null) return null;
  if (typeof body === 'string') return body;
  return new Uint8Array(body as ArrayBuffer);
}

/** A minimal Hub: repo creation, preupload classification, LFS batch/upload and commit. */
function fakeHub(options: { existing?: boolean; alreadyUploaded?: boolean; multipart?: number; failCommit?: number } = {}) {
  const requests: Recorded[] = [];
  const stored = new Map<string, Uint8Array>();
  const parts: Uint8Array[] = [];
  const transport = (async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    const recorded: Recorded = { method: init?.method ?? 'GET', url, headers: headersOf(init), body: bodyOf(init) };
    requests.push(recorded);
    const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } });
    if (url === 'https://hub.test/api/repos/create') {
      return options.existing ? json({ error: 'You already created this model repo' }, 409) : json({ url: 'https://hub.test/owner/model' });
    }
    if (url.startsWith('https://hub.test/api/models/owner/model/preupload/')) {
      const body = JSON.parse(recorded.body as string) as { files: { path: string; size: number }[] };
      return json({ files: body.files.map((file) => ({ path: file.path, uploadMode: file.path.endsWith('.safetensors') ? 'lfs' : 'regular', shouldIgnore: false })) });
    }
    if (url === 'https://hub.test/owner/model.git/info/lfs/objects/batch') {
      const body = JSON.parse(recorded.body as string) as { objects: { oid: string; size: number }[] };
      return json({
        transfer: 'basic',
        objects: body.objects.map((object) => (options.alreadyUploaded ? { oid: object.oid, size: object.size } : {
          oid: object.oid, size: object.size,
          actions: options.multipart ? {
            upload: { href: `https://storage.test/complete/${object.oid}`, header: { chunk_size: String(options.multipart), 1: 'https://storage.test/part/1', 2: 'https://storage.test/part/2', 3: 'https://storage.test/part/3' } },
          } : {
            upload: { href: `https://storage.test/${object.oid}`, header: { 'x-amz-signature': 'signed' } },
            verify: { href: 'https://hub.test/owner/model.git/info/lfs/objects/verify', header: { authorization: 'Basic verify' } },
          },
        })),
      });
    }
    if (url.startsWith('https://storage.test/part/')) {
      parts.push(recorded.body as Uint8Array);
      return new Response(null, { status: 200, headers: { etag: `"etag-${url.slice(-1)}"` } });
    }
    if (url.startsWith('https://storage.test/complete/')) return json({});
    if (url.startsWith('https://storage.test/')) {
      stored.set(url.slice('https://storage.test/'.length), recorded.body as Uint8Array);
      return new Response(null, { status: 200 });
    }
    if (url === 'https://hub.test/owner/model.git/info/lfs/objects/verify') return json({});
    if (url.startsWith('https://hub.test/api/models/owner/model/commit/')) {
      if (options.failCommit) return json({ error: 'Invalid revision' }, options.failCommit);
      return json({ commitUrl: 'https://hub.test/owner/model/commit/abc123', commitOid: 'abc123' });
    }
    return json({ error: `unexpected ${url}` }, 404);
  }) as typeof fetch;
  return { transport, requests, stored, parts };
}

function artifact(): string {
  const directory = scratch();
  mkdirSync(join(directory, 'nested'), { recursive: true });
  writeFileSync(join(directory, 'tensorcode_config.json'), '{"format": "tensorcode.pretrained"}\n');
  writeFileSync(join(directory, 'README.md'), '# Model\n');
  writeFileSync(join(directory, 'model.safetensors'), new Uint8Array([8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32, 1, 2, 3]));
  writeFileSync(join(directory, 'nested', 'asset.txt'), 'asset');
  mkdirSync(join(directory, '.git'), { recursive: true });
  writeFileSync(join(directory, '.git', 'HEAD'), 'ignored');
  return directory;
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('uploadFolder', () => {
  it('creates the repo, uploads LFS weights and commits regular files as NDJSON', async () => {
    const hub = fakeHub();
    const folder = artifact();
    const result = await uploadFolder({
      repoId: 'owner/model', folderPath: folder, private: true, revision: 'main', token: 'hf_secret',
      commitMessage: 'Upload TensorCode model', endpoint: 'https://hub.test/', fetch: hub.transport,
    });
    expect(result.commit).toBe('https://hub.test/owner/model/commit/abc123');
    expect(result.oid).toBe('abc123');
    expect(result.files).toEqual([
      { path: 'README.md', mode: 'regular' }, { path: 'model.safetensors', mode: 'lfs' },
      { path: 'nested/asset.txt', mode: 'regular' }, { path: 'tensorcode_config.json', mode: 'regular' },
    ]);
    const [create, preupload, batch, upload, verify, commit] = hub.requests;
    expect(hub.requests.length).toBe(6);
    expect(create!.method).toBe('POST');
    expect(JSON.parse(create!.body as string)).toEqual({ name: 'model', organization: 'owner', private: true, type: 'model' });
    expect(create!.headers.authorization).toBe('Bearer hf_secret');
    expect(preupload!.url).toBe('https://hub.test/api/models/owner/model/preupload/main');
    const weights = new Uint8Array([8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32, 1, 2, 3]);
    const oid = createHash('sha256').update(weights).digest('hex');
    expect(JSON.parse(batch!.body as string)).toEqual({
      operation: 'upload', transfers: ['basic', 'multipart'], hash_algo: 'sha256', ref: { name: 'main' },
      objects: [{ oid, size: weights.byteLength }],
    });
    expect(batch!.headers['content-type']).toBe('application/vnd.git-lfs+json');
    expect(upload!.method).toBe('PUT');
    expect(upload!.url).toBe(`https://storage.test/${oid}`);
    expect(upload!.headers.authorization).toBeUndefined();
    expect(upload!.headers['x-amz-signature']).toBe('signed');
    expect(hub.stored.get(oid)).toEqual(weights);
    expect(JSON.parse(verify!.body as string)).toEqual({ oid, size: weights.byteLength });
    expect(verify!.headers.authorization).toBe('Basic verify');
    expect(commit!.url).toBe('https://hub.test/api/models/owner/model/commit/main');
    expect(commit!.headers['content-type']).toBe('application/x-ndjson');
    const lines = (commit!.body as string).trim().split('\n').map((line) => JSON.parse(line));
    expect(lines[0]).toEqual({ key: 'header', value: { summary: 'Upload TensorCode model', description: '' } });
    expect(lines.slice(1)).toEqual([
      { key: 'file', value: { content: Buffer.from('# Model\n').toString('base64'), path: 'README.md', encoding: 'base64' } },
      { key: 'lfsFile', value: { path: 'model.safetensors', algo: 'sha256', oid, size: weights.byteLength } },
      { key: 'file', value: { content: Buffer.from('asset').toString('base64'), path: 'nested/asset.txt', encoding: 'base64' } },
      { key: 'file', value: { content: Buffer.from('{"format": "tensorcode.pretrained"}\n').toString('base64'), path: 'tensorcode_config.json', encoding: 'base64' } },
    ]);
  });

  it('accepts an existing repository, skips uploaded objects and encodes revisions', async () => {
    const hub = fakeHub({ existing: true, alreadyUploaded: true });
    const result = await uploadFolder({
      repoId: 'owner/model', folderPath: artifact(), revision: 'refs/pr/1', token: 'hf_secret', endpoint: 'https://hub.test', fetch: hub.transport,
    });
    expect(result.commit).toContain('abc123');
    expect(hub.requests.map((request) => request.url)).toEqual([
      'https://hub.test/api/repos/create',
      'https://hub.test/api/models/owner/model/preupload/refs%2Fpr%2F1',
      'https://hub.test/owner/model.git/info/lfs/objects/batch',
      'https://hub.test/api/models/owner/model/commit/refs%2Fpr%2F1',
    ]);
    expect(JSON.parse(hub.requests[0]!.body as string).private).toBe(false);
  });

  it('uploads multipart LFS objects', async () => {
    const hub = fakeHub({ multipart: 8 });
    await uploadFolder({ repoId: 'owner/model', folderPath: artifact(), token: 't', endpoint: 'https://hub.test', fetch: hub.transport });
    expect(hub.parts.map((part) => part.byteLength)).toEqual([8, 8, 3]);
    const completion = hub.requests.find((request) => request.url.startsWith('https://storage.test/complete/'))!;
    expect(JSON.parse(completion.body as string).parts).toEqual([
      { partNumber: 1, etag: '"etag-1"' }, { partNumber: 2, etag: '"etag-2"' }, { partNumber: 3, etag: '"etag-3"' },
    ]);
  });

  it('resolves tokens from the environment or token file and never leaks them in errors', async () => {
    const hub = fakeHub({ failCommit: 400 });
    vi.stubEnv('HF_TOKEN', 'hf_from_env');
    const error = await uploadFolder({ repoId: 'owner/model', folderPath: artifact(), endpoint: 'https://hub.test', fetch: hub.transport })
      .catch((caught: unknown) => caught as HubError);
    expect(error).toBeInstanceOf(HubError);
    expect((error as HubError).status).toBe(400);
    expect((error as HubError).message).toMatch(/Invalid revision/);
    expect((error as HubError).message).not.toContain('hf_from_env');
    expect(hub.requests[0]!.headers.authorization).toBe('Bearer hf_from_env');
    vi.stubEnv('HF_TOKEN', '');
    vi.stubEnv('HUGGING_FACE_HUB_TOKEN', '');
    const tokenFile = join(scratch(), 'token');
    mkdirSync(join(tokenFile, '..'), { recursive: true });
    writeFileSync(tokenFile, 'hf_from_file\n');
    vi.stubEnv('HF_TOKEN_PATH', tokenFile);
    const second = fakeHub();
    await uploadFolder({ repoId: 'owner/model', folderPath: artifact(), endpoint: 'https://hub.test', fetch: second.transport });
    expect(second.requests[0]!.headers.authorization).toBe('Bearer hf_from_file');
    vi.stubEnv('HF_TOKEN_PATH', join(scratch(), 'missing-token'));
    const third = fakeHub();
    await expect(uploadFolder({ repoId: 'owner/model', folderPath: artifact(), endpoint: 'https://hub.test', fetch: third.transport }))
      .rejects.toThrow(/token/);
    expect(third.requests.length).toBe(0);
  });

  it('validates repository ids and folders before any request', async () => {
    const hub = fakeHub();
    await expect(uploadFolder({ repoId: '../escape', folderPath: artifact(), token: 't', fetch: hub.transport })).rejects.toThrow(/repository id/);
    await expect(uploadFolder({ repoId: 'owner/model', folderPath: join(scratch(), 'missing'), token: 't', fetch: hub.transport })).rejects.toThrow();
    expect(hub.requests.length).toBe(0);
  });
});
