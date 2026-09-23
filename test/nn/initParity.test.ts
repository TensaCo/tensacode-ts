/**
 * Fresh-construction parity: ``manualSeed(n)`` followed by constructing a tool,
 * operation or native model produces weights whose bytes hash exactly like
 * Python's after ``torch.manual_seed(n)`` (``scripts/fixtures/init_fixtures.py``),
 * and leaves the generator in the same state.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { getRngState, manualSeed, pythonRandom, tensor, type Module, type Tensor } from '../../src/nn/index.js';
import { generateSeq2Seq } from '../../src/_internal/native/generation.js';
import type { T5ForConditionalGeneration } from '../../src/_internal/native/t5.js';
import { Latent, Space } from '../../src/ops/vec/index.js';
import { RetrievalEncoder } from '../../src/_internal/retrieval.js';
import { float32ToBFloat16Bits, float32ToFloat16Bits } from '../../src/nn/dtype.js';
import * as vec from '../../src/ops/vec/index.js';
import * as text from '../../src/ops/text/index.js';
import * as tools from '../../src/tools/index.js';
import { ResponseQualityAssessor } from '../../src/_internal/responseQuality.js';
import { RankOperation } from '../../src/_internal/ranking.js';
import { Workspace } from '../../src/_internal/workspace.js';
import { createNativeModel, type NativeHead } from '../../src/_internal/native/registry.js';
import { NativeConfig } from '../../src/_internal/native/config.js';
import { AutoencoderKL, UNet2DConditionModel } from '../../src/_internal/native/diffusers.js';
import type { JsonObject } from '../../src/_internal/json.js';

type Digest = [string, number[], string];
interface Case { name: string; seed: number; configuration: JsonObject; state: Record<string, Digest>; rng: string }
interface ToolCase extends Case { class: string }
interface NativeCase extends Case { head: NativeHead }
interface DiffusersCase extends Case { kind: 'unet' | 'vae' }

interface GenerationRun { settings: Record<string, unknown>; sequences: number[][]; rng: string }
interface DropoutForward { head: NativeHead; configuration: JsonObject; input_ids: number[][]; output: number[]; rng: string }

const fixture = JSON.parse(readFileSync(join(__dirname, '..', 'fixtures', 'init_parity.json'), 'utf8')) as {
  tools: ToolCase[]; native: NativeCase[]; diffusers: DiffusersCase[];
  generation: { configuration: JsonObject; seed: number; input_ids: number[][]; runs: GenerationRun[] };
  dropout_forwards: DropoutForward[];
  seeded_decode: {
    configuration: JsonObject; seed: number; value: { shape: number[]; data: number[] }; context_seed: number; pixels: number[]; shape: number[];
  };
  foundation_flows: Record<string, Case>;
};

type Constructor = new (config: JsonObject) => { stateDict(): Map<string, Tensor> };
const TOOLS: Record<string, (config: JsonObject) => { stateDict(): Map<string, Tensor> }> = {
  'tensorcode.ops.vec.transform:Transform': (c) => new vec.Transform(c as never),
  'tensorcode.ops.vec.classify:Classify': (c) => new vec.Classify(c as never),
  'tensorcode.ops.vec.decode:Decode': (c) => new vec.Decode(c as never),
  'tensorcode.ops.vec.score:Score': (c) => new vec.Score(c as never),
  'tensorcode.ops.vec.encode:VocabularyEncoder': (c) => new vec.VocabularyEncoder(c as never),
  'tensorcode.ops.vec.encode:PatchEncoder': (c) => new vec.PatchEncoder(c as never),
  'tensorcode.ops.vec.encode:TextEncoder': (c) => new vec.TextEncoder(c as never),
  'tensorcode.ops.vec.decode:TextDecoder': (c) => new vec.TextDecoder(c as never),
  'tensorcode.ops.vec.encode:ImageEncoder': (c) => new vec.ImageEncoder(c as never),
  'tensorcode.ops.vec.decode:ImageDecoder': (c) => new vec.ImageDecoder(c as never),
  'tensorcode.tools.scene:Scene': (c) => new tools.Scene(c as never),
  'tensorcode.tools.chatbot:Chatbot': (c) => new tools.Chatbot(c as never),
  'tensorcode.tools.investigator:Investigator': (c) => new tools.Investigator(c as never),
  'tensorcode.tools.planner:Planner': (c) => new tools.Planner(c as never),
  'tensorcode.tools.decision:Decision': (c) => new tools.Decision(c as never),
  'tensorcode._internal.response_quality:ResponseQualityAssessor': (c) => new ResponseQualityAssessor(c),
  'tensorcode._internal.ranking:RankOperation': (c) => new RankOperation(c, { taskKey: 'question', candidatesKey: 'hypotheses' }),
  'tensorcode._internal.workspace:Workspace': (c) => new Workspace(c.dimensions as number, c.slots as number, c.steps as number),
  'tensorcode.ops.text.transform:Transform': (c) => new (text.Transform as unknown as Constructor)(c),
  'tensorcode.ops.text.classify:Classify': (c) => new (text.Classify as unknown as Constructor)(c),
  'tensorcode.ops.text.decide:Decide': (c) => new (text.Decide as unknown as Constructor)(c),
  'tensorcode.ops.text.score:Score': (c) => new (text.Score as unknown as Constructor)(c),
  'tensorcode.ops.text.retrieve:Retrieve': (c) => new (text.Retrieve as unknown as Constructor)(c),
};

function bytesOf(value: Tensor): Uint8Array {
  const count = value.numel;
  switch (value.dtype) {
    case 'float32': return new Uint8Array(Float32Array.from(value.data).buffer);
    case 'float64': return new Uint8Array(Float64Array.from(value.data).buffer);
    case 'float16': return new Uint8Array(Uint16Array.from(value.data, float32ToFloat16Bits).buffer);
    case 'bfloat16': return new Uint8Array(Uint16Array.from(value.data, float32ToBFloat16Bits).buffer);
    case 'int64': return new Uint8Array(BigInt64Array.from(value.data, (item) => BigInt(item)).buffer);
    case 'int32': return new Uint8Array(Int32Array.from(value.data).buffer);
    case 'uint8': case 'bool': return Uint8Array.from(value.data);
    default: throw new Error(`unsupported dtype ${value.dtype} (${count} elements)`);
  }
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function expectState(name: string, model: { stateDict(): Map<string, Tensor> }, expected: Case): void {
  const state = model.stateDict();
  expect([...state.keys()].sort(), name).toEqual(Object.keys(expected.state).sort());
  const different: string[] = [];
  for (const [key, [dtype, shape, digest]] of Object.entries(expected.state)) {
    const value = state.get(key)!;
    if (value.dtype !== dtype || JSON.stringify([...value.shape]) !== JSON.stringify(shape) || sha256(bytesOf(value)) !== digest) {
      different.push(key);
    }
  }
  expect(different, `${name}: tensors differing from Python`).toEqual([]);
  expect(sha256(getRngState()), `${name}: generator state after construction`).toBe(expected.rng);
}

describe('seeded fresh construction matches Python bit for bit', () => {
  for (const testCase of fixture.tools) {
    it(`${testCase.name} (${testCase.class.split(':')[1]})`, () => {
      manualSeed(testCase.seed);
      pythonRandom.seed(testCase.seed);
      const build = TOOLS[testCase.class];
      if (!build) throw new Error(`no TypeScript constructor for ${testCase.class}`);
      expectState(testCase.name, build(testCase.configuration), testCase);
    });
  }

  for (const testCase of fixture.native) {
    it(`native ${testCase.name}`, () => {
      manualSeed(testCase.seed);
      const model = createNativeModel(NativeConfig.fromDict(testCase.configuration), testCase.head,
        { addPoolingLayer: testCase.name.startsWith('vit:') ? false : undefined }) as unknown as Module;
      expectState(testCase.name, model, testCase);
    });
  }

  for (const testCase of fixture.diffusers) {
    it(`diffusers ${testCase.name}`, () => {
      manualSeed(testCase.seed);
      const model = testCase.kind === 'unet' ? new UNet2DConditionModel(testCase.configuration) : new AutoencoderKL(testCase.configuration);
      expectState(testCase.name, model, testCase);
    });
  }
});

describe('seeded sampling and dropout match Python', () => {
  it('reproduces transformers generate() sampling (multinomial, nucleus, beam sampling)', () => {
    const record = fixture.generation;
    manualSeed(record.seed);
    const model = createNativeModel(NativeConfig.fromDict(record.configuration), 'seq2seq') as unknown as T5ForConditionalGeneration;
    model.eval();
    for (const run of record.runs) {
      manualSeed(11);
      const sequences = generateSeq2Seq(model, { inputIds: tensor(record.input_ids, { dtype: 'int64' }) }, run.settings);
      expect(sequences.tolist(), JSON.stringify(run.settings)).toEqual(run.sequences);
      expect(sha256(getRngState()), JSON.stringify(run.settings)).toBe(run.rng);
    }
  });

  for (const forward of fixture.dropout_forwards) {
    it(`draws the same dropout masks in a training-mode ${forward.head} forward`, () => {
      manualSeed(5);
      const model = createNativeModel(NativeConfig.fromDict(forward.configuration), forward.head) as unknown as Module & {
        forward(inputs: Record<string, Tensor>): { logits?: Tensor; lastHiddenState?: Tensor };
      };
      model.train();
      const ids = tensor(forward.input_ids, { dtype: 'int64' });
      manualSeed(6);
      const result = forward.head === 'seq2seq'
        ? model.forward({ inputIds: ids, decoderInputIds: tensor(forward.input_ids.map((row) => row.slice(0, 4)), { dtype: 'int64' }) }).logits!
        : model.forward({ inputIds: ids }).lastHiddenState!;
      const values = Array.from(result.data);
      expect(values).toHaveLength(forward.output.length);
      // Identical masks leave only float rounding; a different mask moves outputs by O(1).
      expect(Math.max(...values.map((value, index) => Math.abs(value - forward.output[index]!)))).toBeLessThan(1e-5);
      expect(sha256(getRngState())).toBe(forward.rng);
    });
  }

  it('decodes images from context.seed with torch.Generator().manual_seed noise', () => {
    const record = fixture.seeded_decode;
    manualSeed(record.seed);
    const decoder = new vec.ImageDecoder(record.configuration as never).eval();
    const value = new Latent(tensor(record.value.data, { shape: record.value.shape }), decoder.inputSpace);
    const pixels = decoder.call(value, { context: { seed: record.context_seed } }) as Tensor;
    expect([...pixels.shape]).toEqual(record.shape);
    const values = Array.from(pixels.data);
    expect(Math.max(...values.map((item, index) => Math.abs(item - record.pixels[index]!)))).toBeLessThan(1e-4);
  });
});

type Stateful = { stateDict(): Map<string, Tensor> };
const ROOT = 'test/fixtures';
const FLOWS: Record<string, () => Promise<Stateful>> = {
  vec_transform: () => vec.Transform.fromFoundation(`${ROOT}/vec/bert_foundation`, { inputSpace: new Space('in', 3), outputSpace: new Space('out', 2) }),
  vec_classify: () => vec.Classify.fromFoundation(`${ROOT}/vec/roberta_foundation`, { inputSpace: new Space('in', 3), labels: ['a', 'b'] }),
  vec_score: () => vec.Score.fromFoundation(`${ROOT}/vec/bert_foundation`, {
    querySpace: new Space('q', 3), candidateSpace: new Space('c', 3), meaning: 'pair utility',
  } as never),
  text_encoder_t5: () => vec.TextEncoder.fromFoundation(`${ROOT}/vec/t5_foundation`),
  text_encoder_bert: () => vec.TextEncoder.fromFoundation(`${ROOT}/vec/bert_foundation`, {
    readout: 'output_encoding', contextSpace: new Space('ctx', 8, { organization: 'sequence' }),
  }),
  text_decoder_t5: () => vec.TextDecoder.fromFoundation(`${ROOT}/vec/t5_foundation`, { inputSpace: new Space('in', 3, { organization: 'sequence' }) }),
  image_encoder: () => vec.ImageEncoder.fromFoundation(`${ROOT}/vec/vit_foundation`, { outputSpace: new Space('v', 8), readout: 'output_encoding' } as never),
  image_decoder: () => vec.ImageDecoder.fromFoundation(`${ROOT}/diffusion/foundation`, {
    inputSpace: new Space('in', 6, { organization: 'sequence' }), numInferenceSteps: 2,
  } as never),
  text_transform: () => (text.Transform as unknown as { fromFoundation(repo: string): Promise<Stateful> }).fromFoundation(`${ROOT}/text/foundation`),
  text_classify: () => (text.Classify as unknown as { fromFoundation(repo: string, options: unknown): Promise<Stateful> })
    .fromFoundation(`${ROOT}/text/foundation`, { config: { labels: ['yes', 'no'] } }),
  scene_clip: () => tools.Scene.fromFoundation(`${ROOT}/vec/clip_foundation`, { revision: 'pinned', dimensions: 8 } as never),
  scene_language: () => tools.Scene.fromLanguageFoundation(`${ROOT}/scene_language/tiny_foundation`, { revision: null, localFilesOnly: true } as never),
  chatbot: () => tools.Chatbot.fromFoundation(`${ROOT}/vec/t5_foundation`),
  planner: () => tools.Planner.fromFoundations(`${ROOT}/vec/bert_foundation`, `${ROOT}/vec/t5_foundation`),
  investigator_missing_head: () => (tools.Investigator as unknown as {
    fromFoundations(encoder: string, generator: string, verifier: string, options: unknown): Promise<Stateful>;
  }).fromFoundations(`${ROOT}/vec/bert_foundation`, `${ROOT}/vec/t5_foundation`, `${ROOT}/init/verifier_foundation`, {
    verifierLabels: { support: 0, unknown: 1, contradiction: 2 },
  }),
  retrieval: () => RetrievalEncoder.fromFoundation(`${ROOT}/vec/bert_foundation`, { pooling: 'masked_mean', normalize: true } as never),
  response_quality: () => ResponseQualityAssessor.fromFoundation(`${ROOT}/vec/bert_foundation`, { maxTokens: 8 }),
};

describe('foundation loading draws like Python', () => {
  it('covers every recorded flow', () => {
    expect(Object.keys(FLOWS).sort()).toEqual(Object.keys(fixture.foundation_flows).sort());
  });
  for (const [name, build] of Object.entries(FLOWS)) {
    it(`${name}: loaded weights draw nothing, new heads match bit for bit`, async () => {
      const expected = fixture.foundation_flows[name]!;
      manualSeed(expected.seed);
      pythonRandom.seed(expected.seed);
      expectState(name, await build(), expected);
    });
  }
});
