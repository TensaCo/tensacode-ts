/** Port of python/tests/runtime/test_action_loop.py. */
import { describe, expect, it } from 'vitest';
import { ActionOutcome, actionLoop } from '../../src/tools/actions.js';

const choice = (value: string | null, abstained = false) => ({ value, abstained });

describe('action loop', () => {
  it('validates the exact choice before running an effect', async () => {
    const effects: unknown[] = [];
    const loop = actionLoop({
      chooser: () => choice(' SEND '),
      actions: { send: (state: string) => { effects.push(state); return new ActionOutcome(state, null); } },
      maxSteps: 2,
    });
    await expect(loop.call('draft')).rejects.toThrow('not one of the supplied actions');
    expect(effects).toEqual([]);
  });

  it('executes nothing on abstention', async () => {
    const effects: unknown[] = [];
    const loop = actionLoop({
      chooser: () => choice(null, true),
      actions: { send: (state: string) => { effects.push(state); return new ActionOutcome(state, null); } },
      maxSteps: 2,
    });
    const result = await loop.call('draft');
    expect(result.stopReason).toBe('abstained');
    expect(result.state).toBe('draft');
    expect(result.receipts).toEqual([]);
    expect(effects).toEqual([]);
  });

  it('stops at the budget and returns effect receipts', async () => {
    const calls: number[] = [];
    const advance = (state: number) => {
      calls.push(state + 1);
      return new ActionOutcome(state + 1, { observed_state: state + 1 });
    };
    const result = await actionLoop({ chooser: () => choice('advance'), actions: { advance }, maxSteps: 2 }).call(0);
    expect(result.stopReason).toBe('budget_exhausted');
    expect(result.state).toBe(2);
    expect(calls).toEqual([1, 2]);
    expect(result.receipts.map((receipt) => receipt.action)).toEqual(['advance', 'advance']);
    expect(result.receipts.map((receipt) => receipt.effect)).toEqual([{ observed_state: 1 }, { observed_state: 2 }]);
  });

  it('honors explicit completion', async () => {
    const result = await actionLoop({
      chooser: () => 'finish', actions: { finish: () => new ActionOutcome('done', 'effect-17', true) }, maxSteps: 3,
    }).call('ready');
    expect(result.stopReason).toBe('completed');
    expect(result.state).toBe('done');
    expect(result.receipts[0]!.effect).toBe('effect-17');
    expect(result.receipts[0]!.step).toBe(0);
  });

  it('requires a structured ActionOutcome', async () => {
    const loop = actionLoop({ chooser: () => 'bad', actions: { bad: (() => 'unverifiable effect') as never }, maxSteps: 1 });
    await expect(loop.call(null)).rejects.toThrow(/ActionOutcome/);
  });

  it('snapshots nested receipt values against later mutation', async () => {
    const shared = { measurements: [] as number[] };
    const advance = (state: number) => {
      shared.measurements.push(state + 1);
      return new ActionOutcome(state + 1, shared);
    };
    const result = await actionLoop({ chooser: () => 'advance', actions: { advance }, maxSteps: 2 }).call(0);
    expect(result.receipts.map((receipt) => receipt.effect)).toEqual([{ measurements: [1] }, { measurements: [1, 2] }]);
    shared.measurements.push(999);
    expect(result.receipts[1]!.effect).toEqual({ measurements: [1, 2] });
  });

  it('prevents choosers from rewriting prior receipt history', async () => {
    const result = await actionLoop({
      chooser: (request) => {
        if (request.receipts.length) ((request.receipts[0]!.effect as { nested: { value: unknown } }).nested).value = 'rewritten';
        return 'advance';
      },
      actions: { advance: (state: number) => new ActionOutcome(state + 1, { nested: { value: state } }) },
      maxSteps: 3,
    }).call(0);
    expect(result.receipts.map((receipt) => (receipt.effect as { nested: { value: unknown } }).nested.value)).toEqual([0, 1, 2]);
  });

  it('supports asynchronous choosers and actions, and passes context', async () => {
    const contexts: unknown[] = [];
    const result = await actionLoop({
      chooser: async (_request, { context }) => {
        contexts.push(context);
        return 'step';
      },
      actions: { step: async (state: number) => new ActionOutcome(state + 1, null, state + 1 === 2) },
      maxSteps: 5,
    }).call(0, { context: { reason: 'test' } });
    expect(result.stopReason).toBe('completed');
    expect(result.state).toBe(2);
    expect(contexts).toEqual([{ reason: 'test' }, { reason: 'test' }]);
  });

  it('validates construction', () => {
    expect(() => actionLoop({ chooser: 'no' as never, actions: {}, maxSteps: 1 })).toThrow(TypeError);
    expect(() => actionLoop({ chooser: () => 'x', actions: {}, maxSteps: -1 })).toThrow(/max_steps/);
    expect(() => actionLoop({ chooser: () => 'x', actions: { '': () => new ActionOutcome(null, null) }, maxSteps: 1 })).toThrow(/non-empty/);
    expect(() => actionLoop({ chooser: () => 'x', actions: { a: 1 as never }, maxSteps: 1 })).toThrow(TypeError);
  });
});
