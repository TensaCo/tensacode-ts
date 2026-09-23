import { describe, expect, it } from 'vitest';
import { InputRef, OutputRef, trace, type Trace } from '../../src/_internal/tracing.js';
import { Operation, type Context } from '../../src/ops/base.js';
import { ValueError } from '../../src/errors.js';
import { tensor } from '../../src/nn/index.js';

class Num {
  static readonly recordFields = ['value'] as const;
  readonly value: number;
  constructor(value: number) { this.value = value; Object.freeze(this); }
  static fromRecord(fields: Record<string, unknown>): Num { return new Num(fields.value as number); }
  toRecord(): Record<string, unknown> { return { value: this.value }; }
}

class Add extends Operation<Num, Num> {
  override get replayable(): boolean { return true; }
  forward(value: Num, context: Context | null): Num {
    const offset = (context?.offset as Num | undefined) ?? new Num(1);
    return new Num(value.value + offset.value);
  }
}

describe('trace sessions', () => {
  it('replay recomputes intermediates from external roots', () => {
    const episode = trace();
    const last = episode.run(() => {
      const first = new Add().call(new Num(2));
      return new Add().call(first);
    });
    const example = episode.example(episode.ref(last));
    expect(example.inputs.size).toBe(1);
    expect(example.calls.length).toBe(2);
    const root = [...example.inputs.keys()][0]!;
    expect(episode.replay(episode.ref(last), { inputs: new Map([[root, new Num(10)]]) })).toEqual(new Num(12));
  });

  it('treats context outputs as dependencies', () => {
    const episode = trace();
    const last = episode.run(() => {
      const offset = new Add().call(new Num(4));
      return new Add().call(new Num(2), { context: { offset } });
    });
    const example = episode.example(episode.ref(last));
    expect(example.calls.length).toBe(2);
    expect(example.inputs.size).toBe(2);
    expect(episode.replay(episode.ref(last))).toEqual(new Num(7));
  });

  it('keeps distinct producers for equal objects and requires explicit scalar refs', () => {
    class Scalar extends Operation<number, number> {
      override get replayable(): boolean { return true; }
      forward(): number { return 1; }
    }
    const episode = trace();
    const [a, b] = episode.run(() => {
      const pair = [new Add().call(new Num(0)), new Add().call(new Num(0))];
      new Scalar().call(0);
      new Scalar().call(0);
      return pair;
    });
    expect(episode.ref(a).equals(episode.ref(b))).toBe(false);
    expect(() => episode.ref(1)).toThrow(/explicit/);
    expect(episode.replay(episode.calls.at(-1)!.output)).toBe(1);
  });

  it('connects consumers through explicit scalar references', () => {
    class Twice extends Operation<number, number> {
      override get replayable(): boolean { return true; }
      forward(value: number): number { return value * 2; }
    }
    const episode = trace();
    const result = episode.run(() => {
      new Twice().call(3);
      return new Twice().call(episode.calls.at(-1)!.output as unknown as number);
    });
    expect(result).toBe(12);
    const example = episode.example(episode.calls.at(-1)!.output);
    expect(example.inputs.size).toBe(1);
    expect(example.calls.length).toBe(2);
  });

  it('records failures and restores the outer trace after nested scopes', () => {
    class Fail extends Operation<Num, Num> {
      forward(): Num { throw new Error('provider down'); }
    }
    const outer = trace();
    let inner!: Trace;
    outer.run(() => {
      new Add().call(new Num(1));
      inner = trace();
      inner.run(() => {
        expect(() => new Fail().call(new Num(2))).toThrow('provider down');
      });
      new Add().call(new Num(3));
    });
    expect(outer.calls.length).toBe(2);
    expect(inner.calls[0]!.error).toBe('Error: provider down');
    new Add().call(new Num(5));
    expect(outer.calls.length).toBe(2);
  });

  it('refuses to replay effectful operations unless recorded boundaries are requested', () => {
    class Effect extends Operation<Num, Num> {
      forward(): Num { return new Num(9); }
    }
    const episode = trace();
    const result = episode.run(() => new Effect().call(new Num(0)));
    expect(() => episode.replay(episode.ref(result))).toThrow(/replay/);
    expect(episode.replay(episode.ref(result), { boundary: 'recorded' })).toEqual(new Num(9));
  });

  it('snapshots external mutable inputs', () => {
    class Sum extends Operation<number[], Num> {
      override get replayable(): boolean { return true; }
      forward(value: number[]): Num { return new Num(value.reduce((a, b) => a + b, 0)); }
    }
    const values = [1, 2];
    const episode = trace();
    const result = episode.run(() => new Sum().call(values));
    values.push(100);
    expect(episode.replay(episode.ref(result))).toEqual(new Num(3));
  });

  it('rejects foreign references', () => {
    const first = trace();
    const result = first.run(() => new Add().call(new Num(1)));
    trace().run(() => {
      expect(() => new Add().call(first.ref(result) as unknown as Num)).toThrow(/session/);
    });
  });

  it('can be entered only once and rejects capture after closing', () => {
    const episode = trace();
    episode.run(() => new Add().call(new Num(1)));
    expect(() => episode.run(() => 0)).toThrow(/once/);
    expect(episode.closed).toBe(true);
  });

  it('detects mutation of captured intermediates', () => {
    class Grow extends Operation<number[], number[]> {
      override get replayable(): boolean { return true; }
      forward(value: number[]): number[] { return [...value, 1]; }
    }
    const episode = trace();
    episode.run(() => {
      const out = new Grow().call([0]);
      out.push(5);
      expect(() => new Grow().call(out)).toThrow(/mutated/);
    });
  });

  it('detects in-place tensor writes through version counters', () => {
    class Double extends Operation<ReturnType<typeof tensor>, ReturnType<typeof tensor>> {
      override get replayable(): boolean { return true; }
      forward(value: ReturnType<typeof tensor>) { return value.mul(2); }
    }
    const episode = trace();
    episode.run(() => {
      const out = new Double().call(tensor([1, 2]));
      out.add_(1);
      expect(() => new Double().call(out)).toThrow(/mutated/);
    });
  });

  it('distinguishes tuples (frozen arrays) from lists and preserves them on replay', () => {
    class Echo extends Operation<readonly number[], readonly number[]> {
      override get replayable(): boolean { return true; }
      forward(value: readonly number[]): readonly number[] { return value; }
    }
    const episode = trace();
    const out = episode.run(() => new Echo().call(Object.freeze([1, 2])));
    const replayed = episode.replay(episode.calls[0]!.output) as readonly number[];
    expect(Object.isFrozen(replayed)).toBe(true);
    expect(replayed).toEqual([1, 2]);
    expect(out).toEqual([1, 2]);
  });

  it('supervises outputs with explicit provenance and release keeps boundaries', () => {
    class Effect extends Operation<Num, Num> {
      forward(value: Num): Num { return new Num(value.value * 3); }
    }
    const episode = trace();
    const result = episode.run(() => new Add().call(new Effect().call(new Num(2))));
    expect(() => episode.supervise(result, new Num(1), { source: ' ' })).toThrow(ValueError);
    const supervision = episode.supervise(result, new Num(1), { source: 'human:review-1', loss: 'mse' });
    expect(supervision.output).toBeInstanceOf(OutputRef);
    expect(supervision.loss).toBe('mse');
    const ref = episode.ref(result);
    episode.release();
    expect(() => episode.ref(result)).toThrow(/Unknown/);
    expect(episode.replay(ref, { boundary: 'recorded' })).toEqual(new Num(7));
    expect(episode.calls[0]!.value).toBeInstanceOf(InputRef);
  });

  it('rejects unsupported opaque inputs', () => {
    class Take extends Operation<unknown, number> {
      forward(): number { return 0; }
    }
    trace().run(() => {
      expect(() => new Take().call(new (class Opaque {})())).toThrow(/Unsupported trace value/);
    });
  });
});
