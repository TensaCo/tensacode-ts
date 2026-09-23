/** Ports of ``tests/vec/test_configuration.py``, ``test_config_lifecycle.py`` and ``test_selector_configuration.py``. */
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Identity, Linear, SGD, noGrad, ones, rand, tensor, type Tensor } from '../../src/nn/index.js';
import { NotImplementedError, ValueError } from '../../src/errors.js';
import {
  Classify, Decide, Decode, Latent, PatchEncoder, Retrieve, Score, Space, Transform, VocabularyEncoder,
} from '../../src/ops/vec/index.js';
import { Transform as GraphTransform } from '../../src/ops/graph/index.js';
import { ExplicitScale, Scale, Similarity } from './modules.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-vec-config-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

describe('operation configurations (tests/vec/test_configuration.py)', () => {
  it('configurations are JSON-safe and include constructor semantics', () => {
    const source = new Space('source', 2);
    const target = new Space('target', 3);
    const operations = [
      Transform.fromModule(new Linear(2, 3), { inputSpace: source, outputSpace: target }),
      new VocabularyEncoder({ vocabulary: ['one', 'two'], dimensions: 2, output_space: source.configuration() as any }),
      Classify.fromModule(new Linear(2, 2), { labels: ['yes', 'no'], inputSpace: source }),
      new PatchEncoder({ in_channels: 1, patch_size: 2, dimensions: 2, output_space: new Space('patches', 2, { organization: 'spatial' }).configuration() as any }),
      Decode.fromModule(new Linear(2, 1), { inputSpace: source, output: 'scalar regression' }),
      Score.fromModule(new Similarity(), { querySpace: source, candidateSpace: source, meaning: 'dot-product relevance' }),
      new Decide({ largest: false }),
      new Retrieve({ k: 2 }),
    ];
    const serialized = operations.map((operation) => JSON.parse(JSON.stringify(operation.configuration())));
    expect(serialized[0].input_space).toEqual(source.configuration());
    expect(serialized[0].output_space).toEqual(target.configuration());
    expect(serialized[0].operation).toBe('tensorcode.ops.vec.transform.Transform');
    expect(serialized[1].vocabulary).toEqual(['one', 'two']);
    expect(serialized[2].labels).toEqual(['yes', 'no']);
    expect(serialized[4].output).toBe('scalar regression');
    expect(serialized[5].meaning).toBe('dot-product relevance');
    expect(serialized[6].largest).toBe(false);
    expect(serialized[7].k).toBe(2);
  });

  it('module configuration tracks architecture but not learned values', () => {
    const operation = Transform.fromModule(new Linear(2, 3));
    const before = operation.configuration();
    noGrad(() => {
      (operation.module as Linear).weight.fill_(91);
      (operation.module as Linear).bias!.fill_(-37);
    });
    expect(operation.configuration()).toEqual(before);
    expect(Transform.fromModule(new Linear(2, 4)).configuration()).not.toEqual(before);
    const text = JSON.stringify(before);
    expect(text).not.toContain('91');
    expect(text).not.toContain('-37');
  });

  it('module configuration includes custom JSON-safe behavior attributes', () => {
    expect(Transform.fromModule(new Scale(2)).configuration()).not.toEqual(Transform.fromModule(new Scale(3)).configuration());
  });

  it('explicit module configuration is authoritative over opaque runtime attributes', () => {
    const first = Transform.fromModule(new ExplicitScale(2)).configuration();
    const second = Transform.fromModule(new ExplicitScale(3)).configuration();
    expect(first).not.toEqual(second);
    JSON.stringify(first);
  });

  it('callbacks persist only with explicit configuration metadata', () => {
    const combine = Object.assign((value: unknown, context: Record<string, unknown>) => (value as Tensor).add(context.bias as Tensor), {
      configuration: () => ({ bias_scale: 2 }),
    });
    const configured = Transform.fromModule(new Identity(), { combine }).configuration();
    expect((configured.combine as any).configuration).toEqual({ bias_scale: 2 });
    const closure = Transform.fromModule(new Identity(), { combine: (value: unknown) => value });
    expect(() => closure.configuration()).toThrow(/explicit configuration/);
  });

  it('classify space validation preserves the prediction API', () => {
    const space = new Space('classifier/features', 2);
    const classify = Classify.fromModule(new Linear(2, 2, { bias: false }), { labels: ['left', 'right'], inputSpace: space });
    const vector = ones([2]);
    vector.requiresGrad = true;
    const prediction = classify.call(new Latent(vector, space));
    expect(prediction.logits.shape).toEqual([2]);
    expect(['left', 'right']).toContain(prediction.value);
    prediction.logits.sum().backward();
    expect(vector.grad).not.toBeNull();
    expect(() => classify.call(new Latent(ones([2]), new Space('other', 2)))).toThrow(/incompatible.*space/);
  });

  it('classify label order is part of the persistable configuration', () => {
    const first = Classify.fromModule(new Linear(2, 2), { labels: ['yes', 'no'] }).configuration();
    const swapped = Classify.fromModule(new Linear(2, 2), { labels: ['no', 'yes'] }).configuration();
    expect(first).not.toEqual(swapped);
  });

  it('operations can share the same trainable backbone instance', () => {
    const backbone = new Linear(2, 2, { bias: false });
    const transform = Transform.fromModule(backbone);
    const classify = Classify.fromModule(backbone, { labels: ['a', 'b'] });
    const value = tensor([1, -1]);
    const before = noGrad(() => classify.call(value).logits.clone());
    (transform.call(value) as Tensor).sum().backward();
    new SGD(transform.parameters(), { lr: 0.2 }).step();
    expect(transform.module).toBe(classify.module);
    expect(transform.parameters()[0]).toBe(classify.parameters()[0]);
    expect(classify.call(value).logits.equal(before)).toBe(false);
  });

  it('fromModule rejects unsupported arguments and invalid labels/outputs', () => {
    expect(() => Transform.fromModule(new Linear(2, 2), { labels: ['a'] } as any)).toThrow(TypeError);
    expect(() => Classify.fromModule(new Linear(2, 2), { labels: ['a', 'a'] })).toThrow(/unique/);
    expect(() => Decode.fromModule(new Linear(2, 2), { output: ' ' })).toThrow(/nonempty/);
  });
});

describe('configuration lifecycle (tests/vec/test_config_lifecycle.py)', () => {
  it('specialized encoders construct from config and restore', async () => {
    const cases = [
      [VocabularyEncoder, { vocabulary: ['hello', 'world'], dimensions: 4, output_space: new Space('words', 4).configuration() }, 'hello world'],
      [PatchEncoder, { patch_size: 2, in_channels: 3, output_space: new Space('patches', 4, { organization: 'spatial' }).configuration() }, rand([3, 4, 4])],
    ] as const;
    for (const [cls, config, value] of cases) {
      const model = new (cls as any)(config);
      const expected = model.call(value) as Latent;
      const directory = join(scratch, cls.name);
      await model.savePretrained(directory);
      const restored = await (cls as any).fromPretrained(directory);
      expect(restored.constructor).toBe(cls);
      expect(restored.configuration()).toEqual(model.configuration());
      expect((restored.call(value) as Latent).tensor.equal(expected.tensor)).toBe(true);
      expect(JSON.parse(readFileSync(join(directory, 'tensorcode_config.json'), 'utf8')).tool).toBe(`tensorcode.ops.vec.encode.${cls.name}`);
    }
  });

  it('parameter-free selector configuration round trip', async () => {
    for (const [cls, config] of [[Decide, { largest: false }], [Retrieve, { k: 2, largest: false }]] as const) {
      const selector = new (cls as any)(config);
      await selector.savePretrained(join(scratch, cls.name));
      expect((await (cls as any).fromPretrained(join(scratch, cls.name))).configuration()).toEqual(selector.configuration());
      expect(() => new (cls as any)({ unexpected: true })).toThrow();
    }
  });

  it('graph stubs accept only declared config', async () => {
    const graph = new GraphTransform({});
    expect(() => graph.call(null as never)).toThrow(NotImplementedError);
    await expect(GraphTransform.fromPretrained('unused')).rejects.toThrow(NotImplementedError);
    expect(() => new GraphTransform({ model: 'implicit' })).toThrow(ValueError);
  });
});

describe('selector configuration (tests/vec/test_selector_configuration.py)', () => {
  it.each([[Decide, { largest: false }], [Retrieve, { k: 2, largest: false }]] as const)('%o artifacts are weightless and preserve settings', async (cls, config) => {
    const settings: Record<string, unknown> = { ...config };
    const operation = new (cls as any)(settings);
    settings.largest = true;
    expect(operation.largest).toBe(false);
    expect(operation.parameters()).toEqual([]);
    const directory = join(scratch, `selector-${cls.name}`);
    await operation.savePretrained(directory);
    expect(readdirSync(directory)).toEqual(['tensorcode_config.json']);
    const restored = await (cls as any).fromPretrained(directory);
    expect(restored.configuration()).toEqual(operation.configuration());
    expect(restored.largest).toBe(false);
  });

  for (const cls of [Decide, Retrieve]) {
    it.each([0, 1, 'true', null, [], {}])(`${cls.name} largest requires an actual boolean (%o)`, (bad) => {
      const config: Record<string, unknown> = { largest: bad };
      if (cls === Retrieve) config.k = 1;
      expect(() => new (cls as any)(config)).toThrow(/largest/);
    });
  }

  it.each([null, {}, { k: true }, { k: 0 }, { k: 1.5 }])('Retrieve requires a positive integer count (%o)', (config) => {
    expect(() => new Retrieve(config as any)).toThrow(/positive integer/);
  });

  it('selector artifact cannot be loaded as a different operation', async () => {
    const directory = join(scratch, 'decide-only');
    await new Decide().savePretrained(directory);
    await expect(Retrieve.fromPretrained(directory)).rejects.toThrow(/identity/);
    expect(existsSync(join(directory, 'tensorcode_config.json'))).toBe(true);
  });
});
