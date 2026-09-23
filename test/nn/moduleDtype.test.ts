/**
 * Module dtype conversion and artifacts with non-float32 parameters
 * (Python ``tests/models/test_pretrained.py`` dtype cases and
 * ``scripts/fixtures/dtype_fixtures.py``).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Linear, Parameter, Tensor, noGrad, ones, randn, tensor } from '../../src/nn/index.js';
import { PretrainedModule } from '../../src/_internal/pretrained.js';
import { T5LayerNorm } from '../../src/_internal/native/t5.js';
import { Chatbot } from '../../src/tools/chatbot.js';
import { Investigator } from '../../src/tools/investigator.js';
import type { JsonObject } from '../../src/_internal/json.js';
import { expectClose } from '../helpers/gradcheck.js';
import { scratchDirectory } from '../training/helpers.js';

const scratch = scratchDirectory('tensorcode-dtype-');
const fixtures = new URL('../fixtures/dtype/', import.meta.url).pathname;
const expected = JSON.parse(readFileSync(`${fixtures}expected.json`, 'utf8'));

class Mixed extends PretrainedModule<Tensor, Tensor> {
  static override readonly qualifiedName: string = 'tests.models.test_pretrained.Mixed';
  readonly encoder: Linear;
  readonly decoder: Linear;
  readonly other: Linear;
  scale: Tensor;

  constructor(config: JsonObject) {
    super(config);
    const width = this.config.width as number;
    this.encoder = this.registerModule('encoder', new Linear(width, width, { bias: false }));
    this.decoder = this.registerModule('decoder', new Linear(width, width, { bias: false }));
    this.decoder.setParameterAt('weight', this.encoder.weight);
    this.scale = this.registerBuffer('scale', ones([2]));
    this.registerBuffer('count', ones([2], { dtype: 'int64' }));
    this.other = this.registerModule('other', new Linear(2, 2));
  }

  forward(value: Tensor): Tensor {
    return this.decoder.forward(this.encoder.forward(value));
  }
}

describe('Module dtype conversion', () => {
  it('casts floating parameters and buffers, keeps ties and rebinds cached fields', () => {
    const norm = new T5LayerNorm(3);
    noGrad(() => norm.weight.fill_(2));
    norm.double();
    expect(norm.weight.dtype).toBe('float64');
    expect(norm.getParameter('weight')).toBe(norm.weight);
    const value = tensor([[1, 2, 3]], { dtype: 'float64' });
    expect(norm.forward(value).dtype).toBe('float64');
    expect(norm.forward(value).select(0, 0).select(0, 0).item()).toBeCloseTo(2 / Math.sqrt(14 / 3), 6);

    const model = new Mixed({ width: 2 });
    model.half();
    expect(model.encoder.weight.dtype).toBe('float16');
    expect(model.decoder.weight).toBe(model.encoder.weight);
    expect(model.scale.dtype).toBe('float16');
    expect(model.getBuffer('count')!.dtype).toBe('int64');
    expect(() => model.to('int64')).toThrow(TypeError);
  });

  it('round-trips mixed parameter and buffer dtypes', async () => {
    const model = new Mixed({ width: 2 });
    model.encoder.double();
    model.other.half();
    model.setBufferAt('scale', model.scale.to('bfloat16'));
    expect(model.decoder.weight).toBe(model.encoder.weight);
    const directory = scratch();
    await model.savePretrained(directory);
    const loaded = await Mixed.fromPretrained(directory);
    expect(loaded.decoder.weight).toBe(loaded.encoder.weight);
    expect(loaded.scale).toBe(loaded.getBuffer('scale'));
    const actual = loaded.stateDict();
    for (const [name, value] of model.stateDict()) {
      expect(actual.get(name)!.dtype, name).toBe(value.dtype);
      expect(actual.get(name)!.equal(value), name).toBe(true);
    }
  });

  it('rebinds fields after artifact dtype restoration', async () => {
    const model = new Mixed({ width: 2 });
    const directory = scratch();
    model.setBufferAt('scale', tensor([3, 4], { dtype: 'float64' }));
    await model.savePretrained(directory);
    const loaded = await Mixed.fromPretrained(directory);
    expect(loaded.scale.dtype).toBe('float64');
    expect(loaded.scale.tolist()).toEqual([3, 4]);
    const parameter = new Parameter(randn([2, 2]).to('float64'));
    loaded.encoder.setParameterAt('weight', parameter);
    expect(loaded.encoder.weight).toBe(parameter);
  });
});

describe('float64 Python artifacts reproduce Python outputs', () => {
  it('Investigator', async () => {
    const model = await Investigator.fromPretrained(`${fixtures}investigator_float64`);
    const candidates = model.call(expected.investigator.inputs).candidates as unknown as { predicted_score: number; probability: number }[];
    expectClose(candidates.map((c) => c.predicted_score), expected.investigator.scores, 1e-12, 1e-12);
    expectClose(candidates.map((c) => c.probability), expected.investigator.probabilities, 1e-12, 1e-12);
  });

  it('T5 Chatbot', async () => {
    const model = await Chatbot.fromPretrained(`${fixtures}chatbot_float64`);
    expect(model.foundation.shared.weight.dtype).toBe('float64');
    const states = noGrad(() => model.foundation.getEncoder().forward({
      inputIds: tensor(expected.chatbot.encoder_ids, { dtype: 'int64' }),
    }).lastHiddenState);
    // transformers accumulates the T5 layer-norm variance in float32.
    expectClose(states.data, expected.chatbot.encoder_states, 1e-6, 1e-6);
    expect(noGrad(() => model.lossBatch(['hello world'], ['answer'])).item()).toBeCloseTo(expected.chatbot.loss, 6);
    expect(model.generateBatch(['hello world', 'hello'])).toEqual(expected.chatbot.generation);
  });
});

