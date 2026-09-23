/**
 * Real cached Hugging Face foundations (skipped unless the pinned snapshots
 * are in the local Hub cache): flan-T5-small text encoding/decoding and CLIP
 * perception inside Scene, against outputs recorded from Python.
 */
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { noGrad } from '../../src/nn/index.js';
import { Space, TextDecoder, TextEncoder } from '../../src/ops/vec/index.js';
import { FoundationSceneRank, Scene } from '../../src/tools/scene.js';
import { cachedSnapshot } from '../helpers/hub.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson, fromJson } from '../helpers/fixtures.js';
import { deepCopy, parseJsonStrict, pythonJsonDumps } from '../../src/_internal/json.js';

const records = fixtureJson('vec/cached.json');
const T5_SNAPSHOT = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab';
const flan = records.flan_t5_small && cachedSnapshot('google/flan-t5-small', T5_SNAPSHOT);
const clipConfig = cachedSnapshot('openai/clip-vit-base-patch32', '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268');
const clipWeights = cachedSnapshot('openai/clip-vit-base-patch32', 'c237dc49a33fc61debc9276459120b7eac67e7ef');
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-cached-vec-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

describe('cached real foundations', () => {
  it.skipIf(!flan)('google/flan-t5-small TextDecoder/TextEncoder match Python', async () => {
    const record = records.flan_t5_small;
    const options = { revision: T5_SNAPSHOT, localFilesOnly: true };
    const probe = await TextDecoder.fromFoundation('google/flan-t5-small', { ...options, inputSpace: new Space('unused', 4, { organization: 'sequence' }) });
    const decoder = await TextDecoder.fromFoundation('google/flan-t5-small', {
      ...options, inputSpace: probe.nativeInputSpace, bridge: 'identity', generation: { max_new_tokens: 12 },
    });
    const { tokenizer, ...configuration } = decoder.configuration();
    void tokenizer;
    expect(configuration).toEqual(record.decoder_configuration);
    const embedded = decoder.embedText(record.prompts);
    expect(decoder.call(embedded)).toEqual(record.generated);
    expect(noGrad(() => decoder.loss(embedded, ['Das Haus ist wunderbar.', 'Paris'])).item()).toBeCloseTo(record.loss, 4);
    const encoder = await TextEncoder.fromFoundation('google/flan-t5-small', { ...options, readout: 'pooled' });
    expect(encoder.configuration().output_space).toEqual(record.encoder_output_space);
    const pooled = noGrad(() => encoder.call(record.prompts).tensor);
    expectClose(pooled.slice(1, 0, 16).data, record.pooled.data, 1e-4, 1e-4);
    expect(pooled.abs().mean().item()).toBeCloseTo(record.pooled_abs_mean, 5);
  }, 300_000);

  it.skipIf(!records.clip || !clipConfig || !clipWeights)('openai/clip-vit-base-patch32 Scene perception matches Python', async () => {
    const record = records.clip;
    const directory = join(scratch, 'clip');
    rmSync(directory, { recursive: true, force: true });
    const { mkdirSync } = await import('node:fs');
    mkdirSync(directory);
    for (const name of ['config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt']) {
      copyFileSync(join(clipConfig!, name), join(directory, name));
    }
    copyFileSync(join(clipWeights!, 'model.safetensors'), join(directory, 'model.safetensors'));
    const scene = await Scene.fromFoundation(directory, { revision: null, dimensions: 8 });
    const { tokenizer_sha256: sha, foundation_source: source, ...configuration } = scene.configuration();
    void sha;
    const { foundation_source: expectedSource, ...expected } = record.configuration;
    void expectedSource;
    expect(source).toEqual({ repo_id: directory, revision: null });
    // Including the nested text/vision configurations, which drop legacy
    // generation keys and carry the loaded dtype as in transformers.
    // ``_name_or_path`` records each run's own temporary foundation directory.
    expect((configuration.foundation_config as any)._name_or_path).toBe(directory);
    expect({ ...configuration, foundation_config: { ...(configuration.foundation_config as object), _name_or_path: null } })
      .toEqual({ ...expected, foundation_config: { ...expected.foundation_config, _name_or_path: null } });
    // Byte for byte as Python writes it (Python float kinds read losslessly from the fixture).
    const python = (parseJsonStrict(readFileSync(new URL('../fixtures/vec/cached.json', import.meta.url), 'utf8')) as any).clip.configuration;
    const bytes = (value: any) => {
      const copy = deepCopy(value);
      delete copy.tokenizer_sha256;
      delete copy.foundation_source;
      copy.foundation_config._name_or_path = null;
      return pythonJsonDumps(copy, { indent: 2, sortKeys: true, allowNan: false });
    };
    expect(bytes(scene.configuration())).toBe(bytes(python));
    const rank = scene.rank as FoundationSceneRank;
    expect(rank.tokens('a photo of a cat').toArray()).toEqual(record.tokens);
    const [patches, imageGlobal] = rank.encodeImage(fromJson(record.pixels));
    expectClose(imageGlobal.data, record.image_global.data, 2e-4, 1e-3);
    expectClose(patches.slice(0, 0, 3).slice(1, 0, 8).data, record.patch_head.data, 2e-4, 1e-3);
    const [, questionGlobal] = rank.encodeText('a photo of a cat');
    expectClose(questionGlobal.data, record.question_global.data, 2e-4, 1e-3);
    const receipt = scene.call({
      question: 'a photo of a cat', source_id: 'image:1', pixels: fromJson(record.pixels),
      candidates: [{ id: 'cat', text: 'a cat' }, { id: 'dog', text: 'a dog' }],
    });
    expect(receipt.patch_coordinates.length).toBe(49);
  }, 300_000);
});
