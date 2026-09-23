/**
 * Plan validation binds keyword arguments like Python's
 * ``inspect.signature(action).bind(None, **arguments)``, and error observations
 * record Python exception names. Expected messages were produced by CPython 3
 * (``inspect.Signature.bind``) for the equivalent Python actions.
 */
import { describe, expect, it } from 'vitest';
import { ValueError } from '../../src/errors.js';
import { ActionOutcome } from '../../src/tools/actions.js';
import {
  ExecutablePlan, PlanExecutor, PlanStep, withSignature,
} from '../../src/_internal/execution/planning.js';
import { bindArguments, inferSignature } from '../../src/_internal/execution/signature.js';
import { describePythonError, pythonErrorName } from '../../src/_internal/pythonErrors.js';
import type { JsonObject } from '../../src/_internal/json.js';

type Fn = (...args: never[]) => unknown;

function bind(fn: Fn, args: Record<string, unknown>): string {
  try {
    bindArguments(inferSignature(fn), args);
    return 'ok';
  } catch (error) {
    return `${(error as Error).name}: ${(error as Error).message}`;
  }
}

describe('plan action signatures', () => {
  // Python: def a(state, amount, note='x')
  const a = (state: unknown, { amount, note = 'x' }: { amount: number; note?: string }) => [state, amount, note];
  // Python: def c(state, **kw)
  const c = (state: unknown, kw: Record<string, unknown>) => [state, kw];
  // Python: def e(state)
  const e = (state: unknown) => state;
  // Python: def g(state, *, amount, **kw)
  const g = (state: unknown, { amount, ...kw }: Record<string, unknown>) => [state, amount, kw];

  it('matches Python binding results and messages', () => {
    expect(bind(a, {})).toBe("TypeError: missing a required argument: 'amount'");
    expect(bind(a, { amount: 1 })).toBe('ok');
    expect(bind(a, { amount: 1, note: 'y' })).toBe('ok');
    expect(bind(a, { amount: 1, extra: 2 })).toBe("TypeError: got an unexpected keyword argument 'extra'");
    expect(bind(a, { amount: 1, b: 1, c: 2 })).toBe("TypeError: got an unexpected keyword argument 'b'");
    expect(bind(a, { extra: 2 })).toBe("TypeError: missing a required argument: 'amount'");
    expect(bind(a, { state: 1, amount: 1 })).toBe("TypeError: multiple values for argument 'state'");
    expect(bind(c, { z: 1 })).toBe('ok');
    expect(bind(e, { z: 1 })).toBe("TypeError: got an unexpected keyword argument 'z'");
    expect(bind(e, {})).toBe('ok');
    expect(bind(g, { z: 1 })).toBe("TypeError: missing a required argument: 'amount'");
    expect(bind(g, { amount: 1, z: 1 })).toBe('ok');
    expect(bind((...args: unknown[]) => args, { z: 1 })).toBe('ok');
  });

  it('reads function, method, async and commented parameter lists', () => {
    async function named(state: unknown, { first, second = { nested: [1, 2] }, 'quoted-key': quoted = 3 }: Record<string, unknown>) {
      return [state, first, second, quoted];
    }
    const methods = {
      step(this: void, s: unknown, /* comment, with comma */ { value, label = `a,${'b'}` }: Record<string, unknown>) {
        return [s, value, label];
      },
    };
    const arrow = async (st: unknown, { renamed: target, withDefault: other = [1, 2] }: Record<string, unknown> = {}) => [st, target, other];
    expect(inferSignature(named as Fn).parameters).toEqual([
      { name: 'first', required: true }, { name: 'second', required: false }, { name: 'quoted-key', required: false },
    ]);
    expect(inferSignature(methods.step as Fn)).toMatchObject({ state: 's', variadic: false });
    expect(inferSignature(methods.step as Fn).parameters).toEqual([{ name: 'value', required: true }, { name: 'label', required: false }]);
    expect(inferSignature(arrow as Fn).parameters).toEqual([{ name: 'renamed', required: true }, { name: 'withDefault', required: false }]);
    expect(bind(arrow as Fn, { st: 1, renamed: 2 })).toBe("TypeError: multiple values for argument 'st'");
    // Bound and native functions do not expose their parameters: every keyword binds.
    expect(bind(a.bind(null) as Fn, { anything: 1 })).toBe('ok');
    expect(bind(Math.max as Fn, { anything: 1 })).toBe('ok');
  });

  it('validates whole plans before any effect with Python messages', async () => {
    const calls: unknown[] = [];
    const pay = (state: JsonObject, { amount, note = 'none' }: { amount: number; note?: string }) => {
      calls.push([amount, note]);
      return new ActionOutcome(state, { paid: amount }, true);
    };
    const executor = new PlanExecutor({ actions: { pay }, replan: () => null, maxSteps: 1 });
    await expect(executor.call({}, new ExecutablePlan('p', [new PlanStep('pay', {})])))
      .rejects.toThrow(new TypeError("missing a required argument: 'amount'"));
    await expect(executor.call({}, new ExecutablePlan('p', [new PlanStep('pay', { amount: 1, currency: 'EUR' })])))
      .rejects.toThrow(new TypeError("got an unexpected keyword argument 'currency'"));
    expect(calls).toEqual([]);
    const result = await executor.call({}, new ExecutablePlan('p', [new PlanStep('pay', { amount: 3 })]));
    expect(result.stopReason).toBe('completed');
    expect(calls).toEqual([[3, 'none']]);
  });

  it('records replan binding failures as Python policy errors', async () => {
    const first = (state: JsonObject) => new ActionOutcome(state, 'looked');
    const second = (state: JsonObject, { amount }: { amount: number }) => new ActionOutcome(state, amount, true);
    const result = await new PlanExecutor({
      actions: { first, second }, maxSteps: 3,
      replan: () => new ExecutablePlan('next', [new PlanStep('second', { amount: 1, unknown: true })]),
    }).call({}, new ExecutablePlan('p', [new PlanStep('first')]));
    expect(result.stopReason).toBe('policy_error');
    expect(result.policyErrors).toEqual(["TypeError: got an unexpected keyword argument 'unknown'"]);
  });

  it('accepts explicit signatures', async () => {
    const opaque = withSignature(((state: JsonObject, args: JsonObject) => new ActionOutcome(state, args, true)).bind(null), {
      parameters: ['amount', 'note?', { name: 'mode', required: true, keywordOnly: true }],
    });
    const executor = new PlanExecutor({ actions: { opaque }, replan: () => null, maxSteps: 1 });
    await expect(executor.call({}, new ExecutablePlan('p', [new PlanStep('opaque', { amount: 1 })])))
      .rejects.toThrow("missing a required keyword-only argument: 'mode'");
    const loose = (state: JsonObject, args: JsonObject) => new ActionOutcome(state, args, true);
    const strict = new PlanExecutor({ actions: { loose }, replan: () => null, maxSteps: 1, signatures: { loose: { parameters: ['only'] } } });
    await expect(strict.call({}, new ExecutablePlan('p', [new PlanStep('loose', { other: 1 })])))
      .rejects.toThrow("missing a required argument: 'only'");
    expect(() => new PlanExecutor({ actions: { loose }, replan: () => null, maxSteps: 1, signatures: { missing: {} } })).toThrow(ValueError);
  });
});

describe('Python error names', () => {
  it('maps JavaScript errors to the exceptions Python raises', () => {
    class Declined extends Error {}
    class Named extends Error {
      override name = 'CustomName';
    }
    class Narrow extends TypeError {}
    const missing = Object.assign(new Error('ENOENT: no such file'), { code: 'ENOENT', errno: -2, syscall: 'open' });
    const other = Object.assign(new Error('EMFILE: too many open files'), { code: 'EMFILE', errno: -24, syscall: 'open' });
    expect(pythonErrorName(new Error('x'))).toBe('RuntimeError');
    expect(pythonErrorName(new TypeError('x'))).toBe('TypeError');
    expect(pythonErrorName(new RangeError('x'))).toBe('ValueError');
    expect(pythonErrorName(new ReferenceError('x'))).toBe('NameError');
    expect(pythonErrorName(new ValueError('x'))).toBe('ValueError');
    expect(pythonErrorName(new Declined('x'))).toBe('Declined');
    expect(pythonErrorName(new Named('x'))).toBe('CustomName');
    expect(pythonErrorName(new Narrow('x'))).toBe('Narrow');
    expect(pythonErrorName(missing)).toBe('FileNotFoundError');
    expect(pythonErrorName(other)).toBe('OSError');
    expect(pythonErrorName('plain string')).toBe('Exception');
    expect(describePythonError(new Error('after external effect'))).toBe('RuntimeError: after external effect');
  });
});
