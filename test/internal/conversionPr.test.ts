/**
 * transformers' Hub safetensors conversion PR lookup
 * (``transformers.safetensors_conversion.get_conversion_pr_reference``):
 * a repository whose ``main`` has only PyTorch weights loads
 * ``model.safetensors`` from the open SFconvertbot pull request based on
 * ``main``.
 */
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { safetensorsConversionRevision } from '../../src/_internal/hub.js';
import { loadNativeFoundation } from '../../src/_internal/native/foundation.js';
import { ValueError } from '../../src/errors.js';

const MAIN = '1'.repeat(40);
const CONVERTED = '2'.repeat(40);
const DECOY = '3'.repeat(40);
const TITLE = 'Adding `safetensors` variant of this model';
const fixture = 'test/fixtures/vec/albert_foundation';
const cache = mkdtempSync(join(tmpdir(), 'tensorcode-conversion-'));
afterAll(() => rmSync(cache, { recursive: true, force: true }));

interface Discussion { num: number; title: string; status: string; isPullRequest: boolean; author: { name: string } }

interface SpawnRecord { data: unknown; events: string[] }

function fakeHub(options: { discussions: Discussion[]; isPrivate?: boolean; converted?: Discussion[]; spawns?: SpawnRecord[] }, calls: string[]): typeof fetch {
  let discussions = options.discussions;
  const mainFiles = readdirSync(fixture).filter((file) => file !== 'model.safetensors').concat('pytorch_model.bin');
  const revisions: Record<string, { sha: string; files: string[] }> = {
    main: { sha: MAIN, files: mainFiles },
    'refs/pr/2': { sha: CONVERTED, files: [...mainFiles, 'model.safetensors'] },
    'refs/pr/1': { sha: DECOY, files: [...mainFiles, 'model.safetensors'] },
  };
  const history: Record<string, string[]> = { main: [MAIN], 'refs/pr/2': [CONVERTED, MAIN], 'refs/pr/1': [DECOY, MAIN] };
  const json = (value: unknown) => new Response(JSON.stringify(value), { status: 200 });
  return (async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push(url);
    if (url === 'https://safetensors-convert.hf.space/call/run') {
      expect(init?.method).toBe('POST');
      options.spawns?.push({ data: JSON.parse(String(init!.body)).data, events: [] });
      return json({ event_id: 'event-1' });
    }
    if (url === 'https://safetensors-convert.hf.space/call/run/event-1') {
      // The Space opens the pull request while it streams its status.
      discussions = [...discussions, ...(options.converted ?? [])];
      const body = 'event: heartbeat\ndata: null\n\nevent: complete\ndata: ["done"]\n\n';
      return new Response(new Blob([body]).stream(), { status: 200, headers: { 'content-type': 'text/event-stream' } });
    }
    const path = url.replace('https://hub.test', '');
    let match = /^\/api\/models\/org\/model\/revision\/(.+)$/.exec(path);
    if (match) {
      const revision = revisions[decodeURIComponent(match[1]!)]!;
      return json({ sha: revision.sha, siblings: revision.files.map((rfilename) => ({ rfilename })) });
    }
    match = /^\/api\/models\/org\/model\/commits\/(.+)$/.exec(path);
    if (match) return json(history[decodeURIComponent(match[1]!)]!.map((id) => ({ id })));
    match = /^\/api\/models\/org\/model\/discussions\?p=(\d+)$/.exec(path);
    if (match) return json({ discussions: match[1] === '0' ? discussions : [] });
    if (path === '/api/models/org/model') return json({ id: 'org/model', private: options.isPrivate ?? false });
    match = /^\/org\/model\/resolve\/[0-9a-f]{40}\/(.+)$/.exec(path);
    if (match) {
      const name = decodeURIComponent(match[1]!);
      if (name === 'pytorch_model.bin') return new Response(readFileSync('test/fixtures/torch_checkpoint/albert_pytorch_model.bin'), { status: 200 });
      return new Response(readFileSync(join(fixture, name)), { status: 200 });
    }
    return new Response('missing', { status: 404 });
  }) as typeof fetch;
}

const bot: Discussion = { num: 2, title: TITLE, status: 'open', isPullRequest: true, author: { name: 'SFconvertbot' } };
const impostor: Discussion = { num: 1, title: TITLE, status: 'open', isPullRequest: true, author: { name: 'someone' } };

describe('safetensors conversion PR', () => {
  beforeEach(() => {
    vi.stubEnv('HF_HUB_OFFLINE', '');
    vi.stubEnv('DISABLE_SAFETENSORS_CONVERSION', '');
  });
  afterEach(() => vi.unstubAllEnvs());

  it('finds only open bot PRs based on main for public repositories', async () => {
    const calls: string[] = [];
    const options = { endpoint: 'https://hub.test', token: null, cacheDir: cache };
    const closed = { ...bot, status: 'closed' };
    const spawns: SpawnRecord[] = [];
    expect(await safetensorsConversionRevision('org/model', { ...options, fetch: fakeHub({ discussions: [impostor, bot], spawns }, calls) })).toBe('refs/pr/2');
    expect(spawns).toEqual([]);
    // No bot PR: the conversion Space is asked to open one (transformers' spawn_conversion), then the lookup repeats.
    expect(await safetensorsConversionRevision('org/model', {
      ...options, fetch: fakeHub({ discussions: [impostor, closed], converted: [bot], spawns }, calls),
    })).toBe('refs/pr/2');
    expect(spawns).toEqual([{ data: ['org/model', false, null], events: [] }]);
    // The explicitly supplied token (never the saved login) is sent, as transformers sends ``kwargs['token']``.
    await safetensorsConversionRevision('org/model', { ...options, token: 'hf_explicit', fetch: fakeHub({ discussions: [], converted: [bot], spawns }, calls) });
    expect(spawns[1]!.data).toEqual(['org/model', false, 'hf_explicit']);
    // A conversion that opens no pull request fails with transformers' message.
    await expect(safetensorsConversionRevision('org/model', { ...options, fetch: fakeHub({ discussions: [impostor, closed], spawns }, calls) }))
      .rejects.toThrow(/Could not create safetensors conversion PR/);
    // Private repositories accept any author (only collaborators can open PRs).
    expect(await safetensorsConversionRevision('org/model', { ...options, fetch: fakeHub({ discussions: [impostor], isPrivate: true }, calls) })).toBe('refs/pr/1');
    // Pinned revisions, offline mode and DISABLE_SAFETENSORS_CONVERSION never query the Hub.
    calls.length = 0;
    const transport = fakeHub({ discussions: [bot] }, calls);
    expect(await safetensorsConversionRevision('org/model', { ...options, fetch: transport, revision: MAIN })).toBeNull();
    expect(await safetensorsConversionRevision('org/model', { ...options, fetch: transport, localFilesOnly: true })).toBeNull();
    vi.stubEnv('DISABLE_SAFETENSORS_CONVERSION', 'true');
    expect(await safetensorsConversionRevision('org/model', { ...options, fetch: transport })).toBeNull();
    expect(calls).toEqual([]);
    vi.stubEnv('DISABLE_SAFETENSORS_CONVERSION', '');
    vi.stubEnv('HF_HUB_OFFLINE', '1');
    expect(await safetensorsConversionRevision('org/model', { ...options, fetch: transport })).toBeNull();
    expect(calls).toEqual([]);
  });

  it('loads a PyTorch-only foundation from the conversion PR with useSafetensors', async () => {
    const calls: string[] = [];
    const options = { endpoint: 'https://hub.test', token: null, cacheDir: cache, tokenizer: false, useSafetensors: true };
    const loaded = await loadNativeFoundation('org/model', { ...options, fetch: fakeHub({ discussions: [impostor, bot] }, calls) });
    expect(calls.some((url) => url.endsWith(`/resolve/${CONVERTED}/model.safetensors`))).toBe(true);
    expect(calls.some((url) => url.includes('pytorch_model.bin'))).toBe(false);
    expect(calls.some((url) => url.includes(DECOY))).toBe(false);
    const reference = await loadNativeFoundation(fixture, { tokenizer: false });
    const expected = reference.model.stateDict();
    for (const [name, value] of loaded.model.stateDict()) expect(value.equal(expected.get(name)!), name).toBe(true);

    vi.stubEnv('DISABLE_SAFETENSORS_CONVERSION', '1');
    await expect(loadNativeFoundation('org/model', { ...options, cacheDir: mkdtempSync(join(cache, 'fresh-')), fetch: fakeHub({ discussions: [bot] }, []) }))
      .rejects.toThrow('org/model does not appear to have a file named model.safetensors or model.safetensors.index.json and thus cannot be loaded with `safetensors`. Please do not set `use_safetensors=True`.');
  });

  it('loads pytorch_model.bin by default and starts the conversion in the background', async () => {
    const calls: string[] = [];
    const spawns: SpawnRecord[] = [];
    const options = { endpoint: 'https://hub.test', token: null, cacheDir: mkdtempSync(join(cache, 'bin-')), tokenizer: false };
    const loaded = await loadNativeFoundation('org/model', { ...options, fetch: fakeHub({ discussions: [impostor], converted: [bot], spawns }, calls) });
    expect(calls.some((url) => url.endsWith(`/resolve/${MAIN}/pytorch_model.bin`))).toBe(true);
    expect(calls.some((url) => url.includes(CONVERTED))).toBe(false);
    const reference = await loadNativeFoundation(fixture, { tokenizer: false });
    const expected = reference.model.stateDict();
    for (const [name, value] of loaded.model.stateDict()) expect(value.equal(expected.get(name)!), name).toBe(true);
    await vi.waitFor(() => expect(spawns).toEqual([{ data: ['org/model', false, null], events: [] }]));
  });
});
