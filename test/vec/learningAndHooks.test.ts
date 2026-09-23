/** Port of ``tests/vec/test_learning_and_hooks.py`` and ``tests/runtime/test_basic_compositions.py``. */
import { describe, expect, it } from 'vitest';
import { Identity, Linear, ModuleList, SGD, manualSeed, noGrad, ones, tensor, type Tensor } from '../../src/nn/index.js';
import { crossEntropy } from '../../src/nn/functional.js';
import { Classify, Transform, type Prediction } from '../../src/ops/vec/index.js';
import { trace } from '../../src/_internal/tracing.js';

describe('learning through vector operations (tests/vec/test_learning_and_hooks.py)', () => {
  it('gradient reaches both modules through trace and replay', () => {
    manualSeed(1);
    const encoder = Transform.fromModule(new Linear(2, 3));
    const classifier = Classify.fromModule(new Linear(3, 2), { labels: ['left', 'right'] });
    const x = tensor([[1, 0], [0, 1]]);
    const y = tensor([0, 1], { dtype: 'int64' });
    const optimizer = new SGD([...encoder.parameters(), ...classifier.parameters()], { lr: 0.2 });
    const episode = trace();
    const prediction = episode.run(() => classifier.call(encoder.call(x)));
    const initial = crossEntropy(prediction.logits, y).item();
    const target = episode.ref(prediction);
    expect(episode.example(target).inputs.size).toBe(1);
    let loss: Tensor = prediction.logits;
    for (let step = 0; step < 40; step += 1) {
      optimizer.zeroGrad();
      loss = crossEntropy((episode.replay(target) as Prediction).logits, y);
      loss.backward();
      expect((encoder.module as Linear).weight.grad).not.toBeNull();
      expect((classifier.module as Linear).weight.grad).not.toBeNull();
      optimizer.step();
    }
    expect(loss.item()).toBeLessThan(initial * 0.4);
  });

  it('vector invocation keeps context gradients through an explicit combine', () => {
    const combine = Object.assign((value: unknown, context: Record<string, unknown>) => (value as Tensor).add(context.bias as Tensor), {
      configuration: () => ({ combine: 'add-bias' }),
    });
    const op = Transform.fromModule(new Identity(), { combine });
    const x = ones([2]);
    x.requiresGrad = true;
    const bias = ones([2]);
    bias.requiresGrad = true;
    (op.call(x, { context: { bias } }) as Tensor).sum().backward();
    expect(bias.grad!.toArray()).toEqual([1, 1]);
    expect(() => Transform.fromModule(new Identity()).call(x, { context: { bias } })).toThrow(/combine/);
  });

  it('a mutated intermediate tensor is rejected by trace', () => {
    const op = Transform.fromModule(new Identity());
    const session = trace();
    session.run(() => {
      const result = op.call(tensor([1])) as Tensor;
      noGrad(() => result.add_(1));
      expect(() => op.call(result)).toThrow(/mutat/);
    });
  });

  it('classifier validates the label count and keeps logits', () => {
    const op = Classify.fromModule(new Identity(), { labels: ['a', 'b'] });
    const result = op.call(tensor([1, 3]));
    expect(result.value).toBe('b');
    expect(result.probabilities.sum().item()).toBeCloseTo(1, 6);
    expect(() => op.call(ones([3]))).toThrow(/labels/);
    const batch = op.call(tensor([[1, 3], [4, 0]]));
    expect(batch.values).toEqual(['b', 'a']);
    expect(() => batch.value).toThrow(/values/);
    expect(() => result.values).toThrow(/batch/);
  });

  it('a shared backbone is registered once and receives both path gradients', () => {
    const backbone = new Linear(1, 1, { bias: false });
    noGrad(() => backbone.weight.fill_(2));
    const a = Transform.fromModule(backbone);
    const b = Transform.fromModule(backbone);
    expect(new ModuleList([a, b]).parameters().length).toBe(1);
    const session = trace();
    const output = session.run(() => b.call(a.call(ones([1]))) as Tensor);
    output.sum().backward();
    expect(backbone.weight.grad!.toArray()).toEqual([4]);
  });
});

describe('basic compositions (tests/runtime/test_basic_compositions.py)', () => {
  it('decision is a composition of public operations', () => {
    const encode = Transform.fromModule(new Identity());
    const decide = Classify.fromModule(new Identity(), { labels: ['a', 'b'] });
    expect(decide.call(encode.call(tensor([0, 2]))).value).toBe('b');
  });
});
