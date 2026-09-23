/** Port of Python ``tests/training/test_experience_persistence.py`` (plus tracing persistence aspects). */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Linear, Tensor, noGrad, tensor } from '../../src/nn/index.js';
import { Operation, type Context } from '../../src/ops/base.js';
import { invokeAsync, trace } from '../../src/_internal/tracing.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import { ValueError } from '../../src/errors.js';
import { LabelHead, Num, Prediction, scratchDirectory, transform } from './helpers.js';

const scratch = scratchDirectory('tensorcode-experience-');

function directory(): string {
  const path = scratch();
  mkdirSync(path, { recursive: true });
  return path;
}

class Add extends Operation<Num, Num> {
  override get replayable(): boolean {
    return true;
  }

  forward(value: Num, context: Context | null): Num {
    const offset = (context?.offset as Num | undefined) ?? new Num(1);
    return new Num(value.value + offset.value);
  }
}

describe('experience persistence', () => {
  it('round-trips records, context and release; codecs are explicit', async () => {
    const op = new Add();
    const session = trace();
    const out = session.run(() => {
      const a = op.call(new Num(2));
      return op.call(a, { context: { offset: new Num(4) } });
    });
    const ref = session.ref(out);
    session.supervise(ref, new Num(8), { loss: 'custom', source: 'reviewer:42' });
    const path = join(directory(), 'experience.json');
    await expect(session.save(path, { operations: { add: op } })).rejects.toThrow(TypeError);
    await expect(session.save(path, { operations: { add: op } })).rejects.toThrow(/codec/);
    await session.save(path, { operations: { add: op }, codecs: { number: Num }, release: true });
    expect(session.calls.every((call) => call.result === null)).toBe(true);
    expect(session._objects.size).toBe(0);
    expect(session.replay(ref)).toEqual(new Num(7));
    await expect(loadExperience(path, { operations: { add: op } })).rejects.toThrow(ValueError);
    await expect(loadExperience(path, { operations: { add: op } })).rejects.toThrow(/codec/);
    const loaded = await loadExperience(path, { operations: { add: new Add() }, codecs: { number: Num } });
    expect(loaded.replay(loaded.supervisions[0]!.output)).toEqual(new Num(7));
    expect(loaded.supervisions[0]!.source).toBe('reviewer:42');
  });

  it('rejects missing or changed bindings and malformed artifacts', async () => {
    const op = new LabelHead(new Linear(2, 2), ['a', 'b']);
    const session = trace();
    const result = session.run(() => op.call(tensor([1, 2])));
    session.supervise(result, 'a');
    const path = join(directory(), 'experience.json');
    await session.save(path, { operations: { head: op } });
    await expect(loadExperience(path, { operations: {} })).rejects.toThrow(/binding/);
    const changed = new LabelHead(new Linear(2, 2), ['b', 'a']);
    await expect(loadExperience(path, { operations: { head: changed } })).rejects.toThrow(/configuration/);
    // Updated weights are intentionally not configuration changes.
    await loadExperience(path, { operations: { head: new LabelHead(new Linear(2, 2), ['a', 'b']) } });
    const data = JSON.parse(readFileSync(path, 'utf8'));
    data.calls[0].value = { kind: 'output', call: 999, path: [] };
    writeFileSync(path, JSON.stringify(data));
    await expect(loadExperience(path, { operations: { head: op } })).rejects.toThrow(ValueError);
  });

  it('loads into fresh bindings, replays gradients and trains', async () => {
    const op = new LabelHead(new Linear(2, 2), ['a', 'b']);
    const session = trace();
    const result = session.run(() => op.call(tensor([1, -1])));
    session.supervise(result, 'a');
    const path = join(directory(), 'experience.json');
    await session.save(path, { operations: { head: op }, release: true });
    const fresh = new LabelHead(new Linear(2, 2), ['a', 'b']);
    const loaded = await loadExperience(path, { operations: { head: fresh } });
    const before = fresh.module.weight.detach().clone();
    const trainer = Trainer.fromOps({ head: fresh }, { lr: 0.1 });
    const losses = trainer.fit([loaded], { epochs: 8 });
    expect(losses[losses.length - 1]!).toBeLessThan(losses[0]!);
    expect(fresh.module.weight.grad).not.toBeNull();
    expect(before.equal(fresh.module.weight)).toBe(false);
  });

  it('records effect boundaries and never re-invokes them', async () => {
    class Effect extends Operation<number, Tensor> {
      calls = 0;
      forward(value: number): Tensor {
        this.calls += 1;
        return tensor([value]);
      }
    }
    const effect = new Effect();
    const head = transform(new Linear(1, 1));
    const session = trace();
    const out = session.run(() => head.call(effect.call(2)));
    session.supervise(out, tensor([4]), { loss: 'mse' });
    const path = join(directory(), 'experience.json');
    await session.save(path, { operations: { external: effect, head } });
    const loaded = await loadExperience(path, { operations: { external: effect, head } });
    expect(() => loaded.replay(loaded.supervisions[0]!.output)).toThrow(/replay/);
    Trainer.fromOps({ external: effect, head }).step(loaded);
    expect(effect.calls).toBe(1);
  });

  it('rejects executable type tags and duplicate JSON keys', async () => {
    const op = new Add();
    const session = trace();
    const out = session.run(() => op.call(new Num(1)));
    session.supervise(out, new Num(2), { loss: 'custom' });
    const path = join(directory(), 'safe.json');
    await session.save(path, { operations: { add: op }, codecs: { number: Num } });
    const data = JSON.parse(readFileSync(path, 'utf8'));
    data.inputs['0'] = { type: 'pickle', data: 'not executable' };
    writeFileSync(path, JSON.stringify(data));
    await expect(loadExperience(path, { operations: { add: op }, codecs: { number: Num } })).rejects.toThrow(/codec/);
    writeFileSync(path, '{"format":"tensorcode.experience","format":"other","version":1}');
    await expect(loadExperience(path, { operations: { add: op } })).rejects.toThrow(/Duplicate/);
    await expect(loadExperience(path, { operations: { add: op } })).rejects.toThrow(ValueError);
  });

  it('importing tensorcode/training performs no I/O and exposes only the public API', async () => {
    const training = await import('../../src/training/index.js');
    expect(Object.keys(training).sort()).toEqual(
      ['Trainer', 'TemperatureCalibration', 'evaluateCalibration', 'fitThreshold', 'loadExperience'].sort(),
    );
  });

  it('async capture records awaited results and failures', async () => {
    class AsyncAdd extends Add {
      override async aforward(value: Num, context: Context | null): Promise<Num> {
        await Promise.resolve();
        return this.forward(value, context);
      }
    }
    const op = new AsyncAdd();
    const session = trace();
    const last = await session.run(async () => {
      const first = await invokeAsync(op, new Num(1), {}, (v, c) => op.aforward(v, c));
      return invokeAsync(op, first, {}, (v, c) => op.aforward(v, c));
    });
    expect(session.calls.length).toBe(2);
    expect(session.replay(session.ref(last))).toEqual(new Num(3));
    const failed = trace();
    await failed.run(async () => {
      await expect(invokeAsync(op, new Num(0), {}, async () => {
        throw new Error('async provider failed');
      })).rejects.toThrow(/provider/);
    });
    expect(failed.calls[0]!.error).toBe('Error: async provider failed');
  });

  it('rejects saving a mutated intermediate and leaves the destination intact', async () => {
    const op = transform(new Linear(1, 1));
    const session = trace();
    const out = session.run(() => op.call(tensor([1]))) as Tensor;
    const path = join(directory(), 'preserved.json');
    writeFileSync(path, 'original content');
    noGrad(() => out.add_(1));
    await expect(session.save(path, { operations: { head: op } })).rejects.toThrow(/mutated/);
    expect(readFileSync(path, 'utf8')).toBe('original content');
  });

  it('release rejects mutated boundaries without partially releasing', () => {
    class External extends Operation<number, Tensor> {
      forward(value: number): Tensor {
        return tensor([value]);
      }
    }
    const external = new External();
    const head = transform(new Linear(1, 1));
    const session = trace();
    let boundary!: Tensor;
    const out = session.run(() => {
      boundary = external.call(1);
      return head.call(boundary);
    });
    noGrad(() => boundary.add_(100));
    expect(() => session.release()).toThrow(/mutated/);
    expect(session._released).toBe(false);
    expect(session.calls[0]!.result).toBe(boundary);
    expect(session.calls[1]!.result).toBe(out);
    expect(session._boundaries.size).toBe(0);
  });

  it('pending async calls cannot be persisted or released', async () => {
    class External extends Operation<number, number> {
      forward(value: number): number {
        return value;
      }
    }
    const op = new External();
    let release!: () => void;
    const ready = new Promise<void>((resolve) => { release = resolve; });
    const waiting = async (value: number): Promise<number[]> => {
      await ready;
      return [value];
    };
    const session = trace();
    let task!: Promise<unknown>;
    await session.run(async () => {
      task = invokeAsync(op, 2, {}, waiting as never);
      await Promise.resolve();
    });
    const root = directory();
    await expect(session.save(join(root, 'pending.json'), { operations: { external: op } })).rejects.toThrow(/pending/);
    expect(() => session.release()).toThrow(/pending/);
    expect(() => session.ref(session.calls[0]!.output)).toThrow(/pending/);
    release();
    expect(await task).toEqual([2]);
    await session.save(join(root, 'done.json'), { operations: { external: op } });
    const loaded = await loadExperience(join(root, 'done.json'), { operations: { external: op } });
    expect(loaded.replay(loaded.calls[0]!.output, { boundary: 'recorded' })).toEqual([2]);
    // A late task retaining the closed session cannot begin a new call.
    const closed = trace();
    let delayed!: Promise<unknown>;
    await closed.run(async () => {
      delayed = (async () => {
        await new Promise((resolve) => setTimeout(resolve, 5));
        return invokeAsync(op, 3, {}, waiting as never);
      })();
    });
    await expect(delayed).rejects.toThrow(/closed/);
    expect(closed.calls.length).toBe(0);
  });

  it('round-trips immutable nested records, tuples, bytes and mappings', async () => {
    class Anchor {
      static readonly recordFields = ['source', 'target', 'location'] as const;
      constructor(readonly source: string, readonly target: string, readonly location: Readonly<Record<string, number>>) {
        Object.freeze(this.location);
        Object.freeze(this);
      }
      static fromRecord(f: Record<string, unknown>): Anchor {
        return new Anchor(f.source as string, f.target as string, f.location as Record<string, number>);
      }
      toRecord(): Record<string, unknown> {
        return { source: this.source, target: this.target, location: this.location };
      }
    }
    class Graph {
      static readonly recordFields = ['nodes', 'sources', 'attributes', 'anchors', 'digest'] as const;
      constructor(readonly nodes: readonly string[], readonly sources: readonly string[],
        readonly attributes: Readonly<Record<string, unknown>>, readonly anchors: readonly Anchor[], readonly digest: Uint8Array) {
        Object.freeze(this);
      }
      static fromRecord(f: Record<string, unknown>): Graph {
        return new Graph(f.nodes as string[], f.sources as string[], f.attributes as Record<string, unknown>,
          f.anchors as Anchor[], f.digest as Uint8Array);
      }
      toRecord(): Record<string, unknown> {
        return { nodes: this.nodes, sources: this.sources, attributes: this.attributes, anchors: this.anchors, digest: this.digest };
      }
    }
    class Decision {
      static readonly recordFields = ['label', 'distribution'] as const;
      readonly distribution: Readonly<Record<string, number>>;
      constructor(readonly label: string, distribution: Record<string, number>) {
        this.distribution = Object.freeze({ ...distribution });
        Object.freeze(this);
      }
      static fromRecord(f: Record<string, unknown>): Decision {
        return new Decision(f.label as string, f.distribution as Record<string, number>);
      }
      toRecord(): Record<string, unknown> {
        return { label: this.label, distribution: this.distribution };
      }
    }
    class Echo extends Operation<unknown, Record<string, unknown>> {
      override get replayable(): boolean {
        return true;
      }
      forward(value: unknown, context: Context | null): Record<string, unknown> {
        return { value, context };
      }
    }
    const op = new Echo();
    const graph = new Graph(Object.freeze(['a']), Object.freeze(['source:1']), { nested: { flag: true } },
      Object.freeze([new Anchor('source:1', 'a', { line: 2 })]), new Uint8Array([0, 255, 7]));
    const result = new Decision('yes', { yes: 0.75, no: 0.25 });
    const session = trace();
    const output = session.run(() => op.call(graph, { context: { decision: result } }));
    const codecs = { graph: Graph, anchor: Anchor, classification: Decision };
    session.supervise(output, result, { loss: 'custom' });
    const path = join(directory(), 'immutable.json');
    await session.save(path, { operations: { echo: op }, codecs, release: true });
    const loaded = await loadExperience(path, { operations: { echo: op }, codecs });
    const restored = loaded.replay(loaded.supervisions[0]!.output) as { value: Graph; context: { decision: Decision } };
    expect(restored.value).toEqual(graph);
    expect(Object.isFrozen(restored.value.nodes)).toBe(true);
    expect(restored.value.digest).toEqual(new Uint8Array([0, 255, 7]));
    expect(restored.context.decision).toEqual(result);
    expect(() => { (restored.context.decision.distribution as Record<string, number>).yes = 0; }).toThrow(TypeError);
  });

  it('rejects experiences whose calls reference operations outside the trainer', async () => {
    const head = new LabelHead(new Linear(2, 2), ['a', 'b']);
    const other = new LabelHead(new Linear(2, 2), ['a', 'b']);
    const session = trace();
    const out = session.run(() => head.call(tensor([1, 1])));
    session.supervise(out, 'b');
    expect(() => Trainer.fromOps({ other }).step(session)).toThrow(/outside Trainer bindings/);
    expect(out).toBeInstanceOf(Prediction);
  });
});
