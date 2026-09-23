/** Port of ``tests/models/test_scene.py`` (+ language-mode unavailability) with Python parity fixtures. */
import { mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Adam, manualSeed, noGrad, onesLike, rand, zerosLike, type Tensor } from '../../src/nn/index.js';
import { NotImplementedError, ValueError } from '../../src/errors.js';
import { FoundationSceneRank, Scene, type SceneInputs } from '../../src/tools/scene.js';
import { sha256Hex } from '../../src/_internal/json.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson, fromJson } from '../helpers/fixtures.js';
import { expectByteIdenticalResave, fingerprints, fixturePath } from '../vec/helpers.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-scene-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const records = fixtureJson('vec/scene.json');

function sample(): SceneInputs {
  return {
    question: 'object left of other', source_id: 'photo:1', pixels: rand([3, 16, 16]),
    candidates: [{ id: 'yes', text: 'supported' }, { id: 'no', text: 'unsupported' }],
  };
}

function model(): Scene {
  return new Scene({ vocabulary: ['object', 'left', 'of', 'other', 'supported', 'unsupported'], dimensions: 8, slots: 3 });
}

const FOUNDATION_TOKENIZER = JSON.stringify({
  version: '1.0', truncation: null, padding: null, added_tokens: [
    { id: 1, content: '[BOS]', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
    { id: 15, content: '[EOS]', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
  ],
  normalizer: null, pre_tokenizer: { type: 'Whitespace' },
  post_processor: {
    type: 'TemplateProcessing', single: [{ SpecialToken: { id: '[BOS]', type_id: 0 } }, { Sequence: { id: 'A', type_id: 0 } }, { SpecialToken: { id: '[EOS]', type_id: 0 } }],
    pair: [{ Sequence: { id: 'A', type_id: 0 } }, { Sequence: { id: 'B', type_id: 1 } }],
    special_tokens: { '[BOS]': { id: '[BOS]', ids: [1], tokens: ['[BOS]'] }, '[EOS]': { id: '[EOS]', ids: [15], tokens: ['[EOS]'] } },
  },
  decoder: null,
  model: { type: 'WordLevel', vocab: { '[UNK]': 0, '[BOS]': 1, object: 2, left: 3, of: 4, other: 5, supported: 6, unsupported: 7, '[EOS]': 15 }, unk_token: '[UNK]' },
});

function tinyFoundationModel(imageSize = 16): Scene {
  return new Scene({
    vocabulary: ['<foundation>'], dimensions: 8, slots: 3,
    foundation_config: {
      model_type: 'clip', projection_dim: 8,
      text_config: { vocab_size: 16, hidden_size: 8, intermediate_size: 16, num_hidden_layers: 1, num_attention_heads: 2, max_position_embeddings: 16, eos_token_id: 15, bos_token_id: 1, pad_token_id: 0 },
      vision_config: { image_size: imageSize, patch_size: 8, hidden_size: 8, intermediate_size: 16, num_hidden_layers: 1, num_attention_heads: 2 },
    },
    _tokenizer_json: FOUNDATION_TOKENIZER, tokenizer_sha256: sha256Hex(FOUNDATION_TOKENIZER),
    image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], patch_size: 8,
  });
}

function allClose(a: Tensor, b: Tensor, tolerance = 1e-8): boolean {
  return a.sub(b).abs().max().item() <= tolerance;
}

describe('Scene ranking (tests/models/test_scene.py)', () => {
  it('spatial sources and complete weight roundtrip', async () => {
    manualSeed(4);
    const tool = model();
    const inputs = sample();
    const result = tool.call(inputs);
    expect(result.patch_coordinates).toEqual([[4, 4], [4, 12], [12, 4], [12, 12]]);
    expect(result.attention_source_ids).toEqual([...Array(4).fill('photo:1'), ...Array(4).fill(null)]);
    expect(result.relations.length).toBe(3);
    expect(result.candidates.reduce((total, item) => total + (item.probability as number), 0)).toBeCloseTo(1, 6);
    await tool.savePretrained(join(scratch, 'scene'));
    const restored = await Scene.fromPretrained(join(scratch, 'scene'));
    expect(restored.call(inputs)).toEqual(result);
    expect('objective' in tool.operationBindings()).toBe(true);
  });

  it('owned visual workspace gradients and input dependence', () => {
    manualSeed(3);
    const tool = model();
    const inputs = sample();
    const parameters = new Map(tool.namedParameters());
    const first = tool.rank.call(inputs);
    const changed = { ...inputs, pixels: zerosLike(inputs.pixels) };
    expect(allClose(first, tool.rank.call(changed))).toBe(false);
    tool.trainingOperation.call({ inputs, targets: 'yes' }).backward();
    for (const [name, parameter] of tool.namedParameters()) {
      expect(parameter.grad, name).not.toBeNull();
      expect(parameter.grad!.allFinite(), name).toBe(true);
    }
    expect(tool.rank.image!.module.getParameter('weight')!.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(tool.rank.workspace.queries.grad!.abs().sum().item()).toBeGreaterThan(0);
    for (const [name, parameter] of tool.namedParameters()) expect(parameters.get(name)).toBe(parameter);
  });

  it('authored fixture optimization is not a pretrained claim', () => {
    manualSeed(12);
    const tool = model();
    const inputs = sample();
    const optimizer = new Adam(tool.parameters(), { lr: 0.02 });
    const before = tool.loss(inputs, 'no').item();
    for (let step = 0; step < 15; step += 1) {
      optimizer.zeroGrad();
      tool.loss(inputs, 'no').backward();
      optimizer.step();
    }
    expect(tool.loss(inputs, 'no').item()).toBeLessThan(before * 0.5);
  });

  const invalid: [string, Partial<SceneInputs>][] = [
    ['empty source', { source_id: '' }], ['empty question', { question: '' }], ['one channel', { pixels: rand([1, 16, 16]) }],
    ['out of range', { pixels: rand([3, 16, 16]).add(2) }], ['nan', { pixels: rand([3, 16, 16]).mul(Number.NaN) }],
    ['too small', { pixels: rand([3, 3, 3]) }], ['no candidates', { candidates: [] }],
    ['duplicate ids', { candidates: [{ id: 'x', text: 'a' }, { id: 'x', text: 'b' }] }],
  ];
  it.each(invalid)('rejects invalid inputs: %s', (_, change) => {
    expect(() => model().call({ ...sample(), ...change })).toThrow(ValueError);
  });

  it('relational text order is preserved', () => {
    manualSeed(5);
    const tool = model();
    const inputs = sample();
    inputs.candidates = [{ id: 'a', text: 'object left other' }, { id: 'b', text: 'other left object' }];
    const scores = tool.rank.call(inputs);
    expect(Math.abs(scores.get(0) - scores.get(1))).toBeGreaterThan(1e-7);
    expect(allClose(tool.rank.call({ ...inputs, question: 'other left of object' }), scores)).toBe(false);
  });

  it('workspace ablations and context rejection', () => {
    const tool = model();
    const inputs = sample();
    const base = noGrad(() => tool.rank.compute(inputs).logits);
    expect(allClose(base, noGrad(() => tool.rank.compute(inputs, { workspaceAblation: 'zero' }).logits))).toBe(false);
    noGrad(() => tool.rank.compute(inputs, { workspaceAblation: 'bypass' }));
    expect(() => tool.rank.compute(inputs, { workspaceAblation: 'other' as never })).toThrow(/workspace_ablation/);
    expect(() => tool.call(inputs, { context: { hint: 1 } })).toThrow(/context/);
    expect(() => tool.loss(inputs, 'missing')).toThrow(/supplied candidate/);
    expect(() => tool.loss(inputs, 5)).toThrow(/valid candidate/);
  });
});

describe('Scene with owned CLIP foundation perception', () => {
  it('owned foundation assets, gradients, cache and reload', async () => {
    const tool = tinyFoundationModel();
    const inputs = sample();
    tool.train();
    const rank = tool.rank as FoundationSceneRank;
    expect(rank.foundation.training).toBe(false);
    const result = tool.call(inputs);
    expect(result.patch_coordinates.length).toBe(4);
    expect(result.attention_source_ids.slice(0, 4)).toEqual(Array(4).fill('photo:1'));
    tool.loss(inputs, 'yes').backward();
    expect(rank.image_projection.module.weight.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(rank.workspace.queries.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(rank.foundation.parameters().every((parameter) => parameter.grad === null)).toBe(true);
    const changed = { ...inputs, pixels: zerosLike(inputs.pixels) };
    expect(allClose(tool.rank.call(inputs), tool.rank.call(changed))).toBe(false);
    tool.eval();
    const evaluated = tool.call(inputs);
    await tool.savePretrained(join(scratch, 'foundation'));
    expect(statSync(join(scratch, 'foundation', 'tokenizer.json')).isFile()).toBe(true);
    const restored = await Scene.fromPretrained(join(scratch, 'foundation'));
    expect((restored.rank as FoundationSceneRank).cacheSizes.vision).toBe(0);
    expect(restored.call(inputs)).toEqual(evaluated);
  });

  it('foundation requires matching assets', () => {
    const tool = tinyFoundationModel();
    expect(() => new Scene(tool.configuration())).toThrow(/tokenizer/);
  });

  it('cache invalidates on weight changes', () => {
    const first = tinyFoundationModel();
    const second = tinyFoundationModel();
    const inputs = sample();
    first.call(inputs);
    const rank = first.rank as FoundationSceneRank;
    expect(rank.cacheSizes.vision).toBe(1);
    first.loadStateDict(second.stateDict());
    expect(rank.cacheSizes).toEqual({ vision: 0, text: 0 });
    expect(allClose(first.rank.call(inputs), second.rank.call(inputs), 0)).toBe(true);
  });

  it('patch coordinates preserve a dropped border', () => {
    const tool = tinyFoundationModel(18);
    const coordinates = tool.call(sample()).patch_coordinates.flat();
    expectClose(coordinates, [4, 4, 4, 12, 12, 4, 12, 12].map((value) => (value * 16) / 18), 1e-6);
  });

  it('cache keys distinguish equal bytes with different dtypes', () => {
    const tool = tinyFoundationModel();
    const pixels = onesLike(rand([3, 16, 16])).mul(0.5);
    tool.rank.call({ ...sample(), pixels });
    tool.rank.call({ ...sample(), pixels: pixels.to('float16') });
    expect((tool.rank as FoundationSceneRank).cacheSizes.vision).toBe(2);
  });

  it('fromFoundation imports owned weights from a local CLIP snapshot', async () => {
    const path = fixturePath('clip_foundation');
    const imported = await Scene.fromFoundation(path, { revision: 'pinned', dimensions: 8 });
    const source = await Scene.fromPretrained(fixturePath('scene_foundation_python'));
    const expected = (source.rank as FoundationSceneRank).foundation.stateDict();
    for (const [name, weight] of (imported.rank as FoundationSceneRank).foundation.stateDict()) {
      expect(weight.equal(expected.get(name)!), name).toBe(true);
    }
    expect(imported.configuration().foundation_source).toEqual({ repo_id: path, revision: 'pinned' });
    expect(imported.call(sample()).candidates.length).toBe(2);
  });
});

describe('Scene artifacts written by Python', () => {
  it('ranking receipt, loss, fingerprints and byte-identical re-save', async () => {
    const record = records.rank;
    const tool = await Scene.fromPretrained(fixturePath('scene_python'));
    expect(tool.configuration()).toEqual(record.configuration);
    expect([...tool.stateDict().keys()]).toEqual(record.state_keys);
    const inputs: SceneInputs = { ...sample(), pixels: fromJson(record.pixels) };
    const receipt = noGrad(() => tool.call(inputs));
    expect(receipt.selected_id).toBe(record.receipt.selected_id);
    expect(receipt.attention_source_ids).toEqual(record.receipt.attention_source_ids);
    expect(receipt.patch_coordinates).toEqual(record.receipt.patch_coordinates);
    expectClose(receipt.candidates.map((item) => item.predicted_score as number), record.receipt.candidates.map((item: any) => item.predicted_score), 1e-5);
    expectClose(receipt.candidates.map((item) => item.probability as number), record.receipt.candidates.map((item: any) => item.probability), 1e-5);
    expectClose(receipt.attention.flat(), record.receipt.attention.flat(), 1e-5);
    expectClose(receipt.relations.flat(), record.receipt.relations.flat(), 1e-5);
    expect(noGrad(() => tool.loss(inputs, 'yes')).item()).toBeCloseTo(record.loss, 5);
    expect(fingerprints(tool.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(tool, 'scene_python', scratch);
  });

  it('foundation receipt (CLIP bicubic resize), tokens and byte-identical re-save', async () => {
    const record = records.foundation;
    const tool = await Scene.fromPretrained(fixturePath('scene_foundation_python'));
    expect(tool.configuration()).toEqual(record.configuration);
    expect([...tool.stateDict().keys()]).toEqual(record.state_keys);
    expect(tool.rank.tokens('object left of other').toArray()).toEqual(record.tokens);
    const inputs: SceneInputs = { ...sample(), pixels: fromJson(records.rank.pixels) };
    for (const [pixels, expected] of [[inputs.pixels, record.receipt], [fromJson(record.wide_pixels), record.wide_receipt]] as const) {
      const receipt = noGrad(() => tool.call({ ...inputs, pixels }));
      expect(receipt.selected_id).toBe(expected.selected_id);
      expect(receipt.attention_source_ids).toEqual(expected.attention_source_ids);
      expectClose(receipt.patch_coordinates.flat(), expected.patch_coordinates.flat(), 1e-5);
      expectClose(receipt.candidates.map((item) => item.predicted_score as number), expected.candidates.map((item: any) => item.predicted_score), 1e-5);
      expectClose(receipt.attention.flat(), expected.attention.flat(), 1e-5);
    }
    expect(fingerprints(tool.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(tool, 'scene_foundation_python', scratch);
  });
});

describe('Scene language mode is unavailable (tests/models/test_scene_language.py)', () => {
  it('construction, foundation import, interpretation and artifacts raise NotImplementedError', async () => {
    expect(() => new Scene({ mode: 'language', language_config: {} })).toThrow(NotImplementedError);
    await expect(Scene.fromLanguageFoundation('HuggingFaceTB/SmolVLM-500M-Instruct', { revision: 'main' })).rejects.toThrow(NotImplementedError);
    expect(() => model().interpret(sample())).toThrow(NotImplementedError);
    await expect(Scene.loadPretrainedConfig({ mode: 'language' }, scratch)).rejects.toThrow(NotImplementedError);
  });
});

