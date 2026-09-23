/**
 * transformers ``generate`` parity (``scripts/fixtures/generation_fixtures.py``):
 * decoding strategies, logits processors, warpers, stopping criteria and the
 * errors transformers raises, on a small random Idefics3 model.
 */
import { describe, expect, it } from 'vitest';
import { nativeConfig } from '../../src/_internal/native/config.js';
import { generationConfigFromDict } from '../../src/_internal/native/causalGeneration.js';
import { pythonJsonLoads } from '../../src/_internal/json.js';
import { Idefics3ForConditionalGeneration } from '../../src/_internal/native/idefics3.js';
import { createNativeModel } from '../../src/_internal/native/registry.js';
import * as P from '../../src/_internal/native/logitsProcessors.js';
import { Generator, loadModelFromBytes, tensor } from '../../src/nn/index.js';
import { randpermValues } from '../../src/nn/random.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, fromJson, ints, type TensorJson } from '../helpers/fixtures.js';

const records = fixtureJson('generation/records.json');

/** Non-finite floats are stored as strings (strict JSON). */
function finite(values: (number | string)[]): number[] {
  return values.map((value) => (value === 'inf' ? Infinity : value === '-inf' ? -Infinity : value === 'nan' ? Number.NaN : value as number));
}

function tensorOf(value: TensorJson): ReturnType<typeof fromJson> {
  return fromJson({ ...value, data: finite(value.data as unknown as (number | string)[]) });
}

function load(): Idefics3ForConditionalGeneration {
  const model = createNativeModel(nativeConfig(records.config), 'image-text-to-text') as Idefics3ForConditionalGeneration;
  loadModelFromBytes(model, fixtureBytes('generation/idefics3_generation.safetensors'));
  return model.eval();
}

interface CaseSpec {
  name: string;
  settings?: Record<string, unknown>;
  generation?: Record<string, unknown>;
  inputs?: string;
  prompt?: number[];
  negative?: number[][];
  seed?: number;
}

function expectScores(actual: Float32Array[][] | null, expected: TensorJson[] | undefined): void {
  if (!expected) return;
  expect(actual).not.toBeNull();
  expect(actual!.length).toBe(expected.length);
  actual!.forEach((step, index) => {
    const flat = step.flatMap((row) => Array.from(row));
    const want = finite(expected[index]!.data as unknown as (number | string)[]);
    flat.forEach((value, position) => {
      const target = want[position]!;
      if (!Number.isFinite(target)) expect(value).toBe(target);
      else expect(Math.abs(value - target)).toBeLessThanOrEqual(2e-4 + 2e-4 * Math.abs(target));
    });
  });
}

describe('Idefics3 generate matches transformers', () => {
  const model = load();
  const features = fromJson(records.image_hidden_states);
  const pixels = fromJson(records.pixel_values);
  const specs: CaseSpec[] = records.case_specs;

  for (const spec of specs) {
    const record = records.cases.find((item: { name: string }) => item.name === spec.name);
    it(spec.name, () => {
      const inputs = spec.inputs ?? 'states';
      let run: () => ReturnType<Idefics3ForConditionalGeneration['generate']>;
      const options = {
        generationConfig: { ...records.model_generation, ...(spec.generation ?? {}) },
        settings: spec.settings ?? {},
        ...(spec.negative ? { negativePromptIds: spec.negative } : {}),
        ...(spec.seed !== undefined ? { generator: new Generator(spec.seed) } : {}),
      };
      if (inputs === 'batch') {
        run = () => model.generate({
          inputIds: ints([[0, 0, 0, 1, 11, 12, 13], [1, 14, 15, 16, 17, 18, 19]]),
          attentionMask: ints([[0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]),
        }, options);
      } else {
        const prompt = spec.prompt ?? (inputs === 'text' ? records.short : records.prompt);
        run = () => model.generate({
          inputIds: ints([prompt]),
          ...(inputs === 'states' ? { imageHiddenStates: features } : {}),
          ...(inputs === 'pixels' ? { pixelValues: pixels } : {}),
        }, options);
      }
      if (record.error) {
        let caught: unknown = null;
        try {
          run();
        } catch (error) {
          caught = error;
        }
        expect(caught, `${spec.name} should raise`).toBeInstanceOf(Error);
        expect((caught as Error).name).toBe(record.error);
        expect((caught as Error).message).toBe(record.message);
        return;
      }
      const output = run();
      expect(output.sequences).toEqual(record.sequences);
      expectScores(output.scores, record.scores);
      expectScores(output.logits, record.logits);
      if (record.sequences_scores) expectClose(output.sequencesScores!, record.sequences_scores, 1e-4, 1e-4);
      if (record.beam_indices) expect(output.beamIndices).toEqual(record.beam_indices);
    });
  }
});

describe('logits processors and warpers match transformers', () => {
  const data = records.processors;
  const scores = tensorOf(data.scores);
  const special = tensorOf(data.special);
  const rowsOf = (value: ReturnType<typeof fromJson>): Float32Array[] =>
    Array.from({ length: value.shape[0]! }, (_, row) => Float32Array.from(value.data.subarray(row * value.shape[1]!, (row + 1) * value.shape[1]!)));
  const ids: number[][] = data.ids;
  const processors: Record<string, P.LogitsProcessorFn> = {
    repetition: P.repetitionPenaltyProcessor(1.3),
    temperature: P.temperatureWarper(0.7),
    top_k: P.topKWarper(5),
    top_k_keep: P.topKWarper(1, 3),
    top_p: P.topPWarper(0.8),
    top_p_keep: P.topPWarper(0.1, 4),
    top_h: P.topHWarper(0.5),
    min_p: P.minPWarper(0.1),
    min_p_keep: P.minPWarper(0.9, 3),
    typical: P.typicalWarper(0.7),
    typical_keep: P.typicalWarper(0.2, 5),
    epsilon: P.epsilonWarper(0.02),
    eta: P.etaWarper(0.03),
    no_repeat: P.noRepeatNGramProcessor(2),
    min_length: P.minLengthProcessor(8, [2, 3]),
    min_new_tokens: P.minNewTokensProcessor(2, 5, [4]),
    infnan: P.infNanRemoveProcessor(),
    normalize: P.logitNormalization(),
    exponential: P.exponentialDecayLengthPenalty([1, 1.5], [2, 3], 3),
    suppress: P.suppressTokensProcessor([1, 2, 39]),
    suppress_begin: P.suppressTokensAtBeginProcessor([4, 5], 6),
    sequence_bias: P.sequenceBiasProcessor(P.normalizeSequenceBias([[[5], 2.5], [[5, 9], -3.0], [[30, 7], 1.25], [[9, 12, 5], 0.5]])),
    bad_words: P.noBadWordsProcessor([[3], [7, 30], [5, 5, 9], [2]], [2]),
    watermark_left: P.watermarkProcessor(40, { ...P.WATERMARK_DEFAULTS }),
    watermark_self: P.watermarkProcessor(40, { ...P.WATERMARK_DEFAULTS, greenlist_ratio: 0.5, bias: 1.5, seeding_scheme: 'selfhash', context_width: 3 }),
  };
  for (const [name, processor] of Object.entries(processors)) {
    it(name, () => {
      const source = name === 'infnan' ? special : scores;
      const output = processor(ids, rowsOf(source)).flatMap((row) => Array.from(row));
      const expected = finite(data.outputs[name].data);
      output.forEach((value, index) => {
        const target = expected[index]!;
        if (Number.isNaN(target)) expect(Number.isNaN(value), `${name}[${index}]`).toBe(true);
        else if (!Number.isFinite(target)) expect(value, `${name}[${index}]`).toBe(target);
        else expect(Math.abs(value - target), `${name}[${index}]`).toBeLessThanOrEqual(1e-5 + 1e-5 * Math.abs(target));
      });
    });
  }

  it('encoder repetition penalty and encoder n-gram blocking', () => {
    const penalty = P.encoderRepetitionPenaltyProcessor(1.6, [[1, 9, 20, 20]])(ids, rowsOf(scores)).flatMap((row) => Array.from(row));
    expectClose(penalty, finite(data.encoder.penalty.data), 1e-6, 1e-6);
    const ngram = P.encoderNoRepeatNGramProcessor(2, [[5, 9, 12, 5, 30, 1, 7, 33]])(ids, rowsOf(scores)).flatMap((row) => Array.from(row));
    const want = finite(data.encoder.ngram.data);
    ngram.forEach((value, index) => expect(value).toBe(Math.fround(want[index]!)));
  });

  it("PyTorch's CPU randperm (watermark greenlists)", () => {
    expect(Array.from(randpermValues(data.randperm.n, new Generator(BigInt(data.randperm.seed))))).toEqual(data.randperm.values);
  });

  it('stopping criteria', () => {
    expect(P.maxLengthCriteria(3)([[1, 2], [1, 2, 3]])).toEqual([false, true]);
    expect(P.eosTokenCriteria([2, 7])([[1, 2], [1, 3], [7]])).toEqual([true, false, true]);
    let now = 10;
    const timer = P.maxTimeCriteria(1.5, () => now);
    expect(timer([[1]])).toEqual([false]);
    now = 11.6;
    expect(timer([[1]])).toEqual([true]);
    void tensor;
  });
});

describe('generation settings read from Python JSON keep their int/float kind', () => {
  // transformers' processors check isinstance(value, float) / isinstance(value, int), so a
  // configuration file that stores ``2`` for repetition_penalty or ``5.0`` for top_k fails in
  // Python; the same file fails the same way here.
  const model = load();
  const features = fromJson(records.image_hidden_states);
  const run = (json: string, settings: Record<string, unknown> = {}): string => {
    // One Python JSON text: the model's generation configuration updated with ``json``.
    const text = `${JSON.stringify(records.model_generation).slice(0, -1)}, ${json.slice(1)}`;
    const merged = generationConfigFromDict(pythonJsonLoads(text) as Record<string, unknown>);
    try {
      model.generate({ inputIds: ints([records.prompt]), imageHiddenStates: features }, {
        generationConfig: merged, settings: { max_new_tokens: 2, ...settings },
      });
      return 'ok';
    } catch (error) {
      return `${(error as Error).name}: ${(error as Error).message}`;
    }
  };

  it('rejects whole numbers of the wrong kind with transformers messages', () => {
    expect(run('{"repetition_penalty": 2}')).toBe('ValueError: `penalty` has to be a strictly positive float, but is 2');
    expect(run('{"repetition_penalty": 2.0}')).toBe('ok');
    // Unmarked whole numbers in JavaScript data follow the field's Python type (float here).
    expect(() => model.generate({ inputIds: ints([records.prompt]), imageHiddenStates: features }, {
      generationConfig: generationConfigFromDict({ ...records.model_generation, repetition_penalty: 2 }), settings: { max_new_tokens: 2 },
    })).not.toThrow();
    expect(run('{"encoder_repetition_penalty": 3}')).toBe('ValueError: `penalty` has to be a strictly positive float, but is 3');
    expect(run('{"no_repeat_ngram_size": 2.0}')).toBe('ValueError: `ngram_size` has to be a strictly positive integer, but is 2.0');
    expect(run('{"no_repeat_ngram_size": 2}')).toBe('ok');
    // min_length becomes min_new_tokens + prompt length, a float, and is checked first.
    expect(run('{"min_new_tokens": 2.0}')).toBe(`ValueError: \`min_length\` has to be a non-negative integer, but is ${records.prompt.length + 2}.0`);
    expect(run('{"min_length": 3.0}')).toBe('ValueError: `min_length` has to be a non-negative integer, but is 3.0');
    expect(run('{"do_sample": true, "top_k": 5.0}', { do_sample: true })).toBe('ValueError: `top_k` has to be a strictly positive integer, but is 5.0');
    expect(run('{"do_sample": true, "temperature": 0}', { do_sample: true })).toBe(
      'ValueError: `temperature` (=0) has to be a strictly positive float, otherwise your next token scores will be invalid.');
  });
});
