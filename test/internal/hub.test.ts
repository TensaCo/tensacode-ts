import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FileNotFoundError, HubError, globMatch, resolveArtifactDirectory, snapshotDownload } from '../../src/_internal/hub.js';
import { ConfigOperation, validatedConfig } from '../../src/_internal/operationConfig.js';
import { ValueError } from '../../src/errors.js';
import type { JsonObject } from '../../src/_internal/json.js';

const SHA = 'a'.repeat(40);
const cache = mkdtempSync(join(tmpdir(), 'tensorcode-hub-'));
afterAll(() => rmSync(cache, { recursive: true, force: true }));

function fakeHub(files: Record<string, string>, calls: string[]): typeof fetch {
  return (async (input: string | URL) => {
    const url = String(input);
    calls.push(url);
    if (url.includes('/api/models/')) {
      return new Response(JSON.stringify({ sha: SHA, siblings: Object.keys(files).map((rfilename) => ({ rfilename })) }), { status: 200 });
    }
    const name = decodeURIComponent(url.split(`/resolve/${SHA}/`)[1]!);
    if (!(name in files)) return new Response('missing', { status: 404 });
    return new Response(files[name], { status: 200 });
  }) as typeof fetch;
}

describe('Hugging Face Hub client', () => {
  // These tests exercise the (faked) network path; ignore a developer's offline setting.
  beforeEach(() => vi.stubEnv('HF_HUB_OFFLINE', ''));
  afterEach(() => vi.unstubAllEnvs());

  it('matches fnmatch-style patterns', () => {
    expect(globMatch('*.safetensors', 'model.safetensors')).toBe(true);
    expect(globMatch('tensorcode_config.json', 'tensorcode_config.json')).toBe(true);
    expect(globMatch('model-?.bin', 'model-1.bin')).toBe(true);
    expect(globMatch('*.json', 'weights.bin')).toBe(false);
  });

  it('downloads a pinned snapshot into the shared cache layout and resolves it offline', async () => {
    const calls: string[] = [];
    const transport = fakeHub({ 'config.json': '{"a": 1}', 'sub/dir/file.txt': 'x', 'weights.bin': 'no' }, calls);
    const path = await snapshotDownload('org/model', { cacheDir: cache, fetch: transport, allowPatterns: ['*.json', 'sub/*'] });
    expect(path).toBe(join(cache, 'models--org--model', 'snapshots', SHA));
    expect(readFileSync(join(path, 'config.json'), 'utf8')).toBe('{"a": 1}');
    expect(existsSync(join(path, 'sub/dir/file.txt'))).toBe(true);
    expect(existsSync(join(path, 'weights.bin'))).toBe(false);
    expect(readFileSync(join(cache, 'models--org--model', 'refs', 'main'), 'utf8')).toBe(SHA);
    const offline = await snapshotDownload('org/model', { cacheDir: cache, localFilesOnly: true });
    expect(offline).toBe(path);
    const count = calls.length;
    await snapshotDownload('org/model', { cacheDir: cache, fetch: transport, allowPatterns: ['*.json'] });
    expect(calls.length).toBe(count + 1); // revision lookup only; cached files are reused
  });

  it('falls back to the cache on network failure and reports HTTP errors', async () => {
    const offline = (async () => { throw new TypeError('network down'); }) as typeof fetch;
    expect(await snapshotDownload('org/model', { cacheDir: cache, fetch: offline })).toContain(SHA);
    await expect(snapshotDownload('org/absent', { cacheDir: cache, fetch: offline })).rejects.toThrow(HubError);
    const denied = (async () => new Response('no', { status: 401 })) as unknown as typeof fetch;
    await expect(snapshotDownload('org/private', { cacheDir: cache, fetch: denied })).rejects.toMatchObject({ status: 401 });
    await expect(snapshotDownload('org/missing', { cacheDir: cache, localFilesOnly: true })).rejects.toThrow(FileNotFoundError);
  });

  it('distinguishes local paths from repository ids', async () => {
    await expect(resolveArtifactDirectory('./missing-directory')).rejects.toThrow(FileNotFoundError);
    await expect(resolveArtifactDirectory('../bad..id', { cacheDir: cache, localFilesOnly: true })).rejects.toThrow();
    expect((await resolveArtifactDirectory(cache)).remote).toBe(false);
  });
});

class Selector extends ConfigOperation<number, number> {
  static override readonly qualifiedName: string = 'tests.Selector';
  constructor(config: JsonObject | null = null) {
    super(config, ['largest', 'k'], { largest: true });
  }
  forward(value: number): number {
    return value;
  }
}

describe('configuration-only operation artifacts', () => {
  it('validates JSON configuration and rejects unknown or obsolete fields', () => {
    expect(validatedConfig(null, ['a'], { a: 1 })).toEqual({ a: 1 });
    expect(() => validatedConfig({ b: 1 }, ['a'])).toThrow(/Unknown configuration fields/);
    expect(() => validatedConfig({ a: Number.NaN }, ['a'])).toThrow(ValueError);
    expect(() => validatedConfig({ a: new Date() }, ['a'])).toThrow(ValueError);
    expect(() => validatedConfig([], ['a'])).toThrow(TypeError);
  });

  it('saves and restores a configuration artifact with its identity', async () => {
    const directory = join(cache, 'selector');
    const selector = new Selector({ k: 2 });
    await selector.savePretrained(directory);
    const manifest = readFileSync(join(directory, 'tensorcode_config.json'), 'utf8');
    expect(manifest).toBe('{\n  "format": "tensorcode.operation",\n  "version": 1,\n  "operation": "tests.Selector",\n  "config": {\n    "largest": true,\n    "k": 2\n  }\n}\n');
    const restored = await Selector.fromPretrained(directory);
    expect(restored.configuration()).toEqual({ largest: true, k: 2 });
    class Other extends Selector {
      static override readonly qualifiedName: string = 'tests.Other';
    }
    await expect(Other.fromPretrained(directory)).rejects.toThrow(/identity/);
  });
});
