/** Port of Python ``tests/training/test_trainers.py``. */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Identity, Linear, SGD, Tensor, F, manualSeed, noGrad, tensor } from '../../src/nn/index.js';
import { trace, type Trace } from '../../src/_internal/tracing.js';
import { Trainer } from '../../src/training/index.js';
import { loadCheckpoint, parseCheckpoint, saveCheckpoint } from '../../src/_internal/training/checkpoint.js';
import { Codec, pythonDumps } from '../../src/_internal/training/persistence.js';
import { LabelHead, scratchDirectory, transform } from './helpers.js';

const scratch = scratchDirectory('tensorcode-trainers-');

function crossEntropyOf(logits: Tensor, targets: Tensor): number {
  return noGrad(() => F.crossEntropy(logits, targets).item());
}

function file(name: string): string {
  const directory = scratch();
  mkdirSync(directory, { recursive: true });
  return join(directory, name);
}

describe('operation trainers', () => {
  it('deduplicates shared parameters and checks checkpoint aliases', async () => {
    const module = new Linear(2, 2);
    const operations = { a: transform(module), b: transform(module) };
    const trainer = Trainer.fromOps(operations);
    expect(trainer.parameters.length).toBe(2);
    expect(trainer.optimizer.paramGroups[0]!.params.length).toBe(2);
    const path = file('checkpoint.json');
    await saveCheckpoint(path, { operations, optimizer: trainer.optimizer });
    const expected = module.weight.detach().clone();
    noGrad(() => module.weight.add_(5));
    await loadCheckpoint(path, { operations, optimizer: trainer.optimizer });
    expect(module.weight.equal(expected)).toBe(true);
    const separate = { a: transform(new Linear(2, 2)), b: transform(new Linear(2, 2)) };
    await expect(loadCheckpoint(path, { operations: separate })).rejects.toThrow(/alias/);
  });

  it('improves held-out supervision and rejects nondifferentiable paths', () => {
    manualSeed(3);
    const head = new LabelHead(new Linear(1, 2), ['negative', 'positive']);
    const experiences: Trace[] = [];
    for (const value of [-3, -1, 1, 3]) {
      const session = trace();
      const output = session.run(() => head.call(tensor([value])));
      session.supervise(output, value < 0 ? 'negative' : 'positive');
      experiences.push(session);
    }
    const heldOut = tensor([[-2], [2]]);
    const targets = tensor([0, 1], { dtype: 'int64' });
    const before = crossEntropyOf(head.call(heldOut).logits, targets);
    const losses = Trainer.fromOps({ head }, { lr: 0.1 }).fit(experiences, { epochs: 15 });
    const after = crossEntropyOf(head.call(heldOut).logits, targets);
    expect(after).toBeLessThan(before * 0.5);
    expect(losses[losses.length - 1]!).toBeLessThan(losses[0]!);
    expect(() => experiences[0]!.supervise(experiences[0]!.calls[0]!.output, 0, { source: '' })).toThrow(/source/);
    const detached = transform(new Identity());
    const session = trace();
    const out = session.run(() => detached.call(tensor([1])));
    session.supervise(out, tensor([2]), { loss: 'mse' });
    expect(() => Trainer.fromOps({ head: detached }).step(session)).toThrow(/parameter|differentiable/);
  });

  it('applies one optimizer update to a shared parameter', () => {
    const module = new Linear(1, 1, { bias: false });
    noGrad(() => module.weight.fill_(1));
    const first = transform(module);
    const second = transform(module);
    const session = trace();
    const out = session.run(() => second.call(first.call(tensor([2]))));
    session.supervise(out, tensor([0]), { loss: 'mse' });
    Trainer.fromOps({ a: first, b: second }, { lr: 0.01 }).step(session);
    // L=(2*w*w)^2, dL/dw=16 at w=1; one update reaches .84.
    expect(module.weight.item()).toBeCloseTo(0.84, 6);
  });

  it('supports custom losses and validates optimizer ownership', () => {
    const head = transform(new Linear(1, 1));
    const session = trace();
    const out = session.run(() => head.call(tensor([1])));
    session.supervise(out, tensor([0]), { loss: 'absolute', source: 'test:observed' });
    const trainer = Trainer.fromOps({ head }, {
      losses: { absolute: (actual: Tensor, target: Tensor) => actual.sub(target).abs().mean() },
    });
    expect(trainer.step(session)).toBeGreaterThanOrEqual(0);
    const unrelated = new Linear(1, 1);
    expect(() => Trainer.fromOps({ head }, { optimizer: new SGD(unrelated.parameters(), { lr: 0.1 }) })).toThrow(/exactly/);
    expect(() => Trainer.fromOps({ head }, { losses: { broken: 3 as never } })).toThrow(TypeError);
  });

  it('restores optimizer momentum from a checkpoint', async () => {
    const head = transform(new Linear(1, 1));
    const trainer = Trainer.fromOps({ head }, { optimizer: (params) => new SGD(params, { lr: 0.1, momentum: 0.9 }) });
    const session = trace();
    const out = session.run(() => head.call(tensor([1])));
    session.supervise(out, tensor([0]), { loss: 'mse' });
    trainer.step(session);
    const path = file('momentum.json');
    await saveCheckpoint(path, { operations: { head }, optimizer: trainer.optimizer });
    const expected = new Map([...trainer.optimizer.state].map(([param, slots]) => [param, slots.momentum_buffer!.clone()]));
    trainer.step(session);
    await loadCheckpoint(path, { operations: { head }, optimizer: trainer.optimizer });
    for (const [param, value] of expected) expect(trainer.optimizer.state.get(param)!.momentum_buffer!.equal(value)).toBe(true);
  });

  it('rejects a nonfinite custom loss without updating', () => {
    const head = transform(new Linear(1, 1));
    const session = trace();
    const out = session.run(() => head.call(tensor([1])));
    session.supervise(out, 0, { loss: 'broken' });
    const initial = head.module.weight.detach().clone();
    const trainer = Trainer.fromOps({ head }, { losses: { broken: (output: Tensor) => output.sum().mul(Number.NaN) } });
    expect(() => trainer.step(session)).toThrow(/finite/);
    expect(initial.equal(head.module.weight)).toBe(true);
    expect(head.module.weight.grad).toBeNull();
  });

  it('rejects malformed optimizer slots before mutating anything', async () => {
    const head = transform(new Linear(1, 1));
    const trainer = Trainer.fromOps({ head }, { optimizer: (params) => new SGD(params, { lr: 0.1, momentum: 0.9 }) });
    const session = trace();
    const out = session.run(() => head.call(tensor([1])));
    session.supervise(out, tensor([0]), { loss: 'mse' });
    trainer.step(session);
    const path = file('bad-optimizer.json');
    await saveCheckpoint(path, { operations: { head }, optimizer: trainer.optimizer });
    const payload = parseCheckpoint(readFileSync(path, 'utf8')) as Record<string, any>;
    const codec = new Codec();
    const state = codec.decode(payload.optimizer.state) as Record<string, any>;
    state.state['0'].momentum_buffer = tensor([1, 1, 1, 1, 1, 1, 1]);
    const slots = new Map<number, unknown>(Object.entries(state.state).map(([key, value]) => [Number(key), value]));
    payload.optimizer.state = codec.encode(new Map<string, unknown>([['state', slots], ['param_groups', state.param_groups]]));
    writeFileSync(path, pythonDumps(payload));
    noGrad(() => head.module.weight.add_(10));
    const expected = head.module.weight.detach().clone();
    await expect(loadCheckpoint(path, { operations: { head }, optimizer: trainer.optimizer })).rejects.toThrow(/optimizer.*shape/);
    expect(expected.equal(head.module.weight)).toBe(true);
  });

  it('validates the Python loss contract', () => {
    const head = new LabelHead(new Linear(2, 2), ['a', 'b']);
    const cases: [unknown, RegExp][] = [
      ['c', /absent/], [['a', 'c'], /absent/], [0.5, /integer indices/], [true, /integer indices/],
    ];
    for (const [target, pattern] of cases) {
      const session = trace();
      const out = session.run(() => head.call(tensor([1, 2])));
      session.supervise(out, target);
      expect(() => Trainer.fromOps({ head }).step(session)).toThrow(pattern);
    }
    const adapter = transform(new Linear(2, 2));
    const session = trace();
    const out = session.run(() => adapter.call(tensor([1, 2])));
    session.supervise(out, tensor([1, 2, 3]), { loss: 'mse' });
    expect(() => Trainer.fromOps({ adapter }).step(session)).toThrow(/MSE target shape/);
    const unknown = trace();
    const result = unknown.run(() => adapter.call(tensor([1, 2])));
    unknown.supervise(result, 0, { loss: 'hinge' });
    expect(() => Trainer.fromOps({ adapter }).step(unknown)).toThrow(/Unknown loss: hinge/);
    expect(() => Trainer.fromOps({ adapter }).fit([], { epochs: 1 })).toThrow(/No experiences/);
    expect(() => Trainer.fromOps({ adapter }).fit([unknown], { epochs: 0 })).toThrow(/epochs/);
    expect(() => Trainer.fromOps({ adapter }).step(trace())).toThrow(/no explicit supervision/);
  });
});
