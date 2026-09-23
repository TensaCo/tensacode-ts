/**
 * A bounded chooser/effect loop with explicit receipts (Python
 * ``tensorcode/_internal/execution/action_loop.py``).
 */
import { ValueError } from '../../errors.js';
import type { Context } from '../../ops/base.js';
import { ActionLoopResult, ActionOutcome, ActionReceipt, ActionRequest } from '../../tools/actions.js';
import { deepClone } from './clone.js';

/** A chooser result: an action name, or an object with ``value`` and optional ``abstained``. */
export type ChooserResult = string | { value?: string | null; abstained?: boolean } | null | undefined;

export type Chooser<S = unknown> = (request: ActionRequest<S>, options: { context: Context | null }) => ChooserResult | Promise<ChooserResult>;

export type ActionFunction<S = unknown> = (state: S) => ActionOutcome<S> | Promise<ActionOutcome<S>>;

/**
 * Choose only from supplied names and execute at most ``maxSteps``.
 *
 * State is passed directly to the supplied chooser/actions. Receipt values must
 * be structured-clonable: each observation is snapshotted, and choosers get
 * separate receipt copies. Returned receipts belong to the caller. This does
 * not roll back state mutations or external effects when a callback fails.
 */
export class ActionLoop<S = unknown> {
  static readonly qualifiedName: string = 'tensorcode._internal.execution.action_loop.ActionLoop';
  readonly chooser: Chooser<S>;
  readonly actions: ReadonlyMap<string, ActionFunction<S>>;
  readonly maxSteps: number;

  constructor(options: {
    chooser: Chooser<S>;
    actions: Record<string, ActionFunction<S>> | Map<string, ActionFunction<S>>;
    maxSteps: number;
  }) {
    const { chooser, actions, maxSteps } = options;
    if (typeof chooser !== 'function') throw new TypeError('chooser must be callable');
    if (typeof maxSteps !== 'number' || !Number.isInteger(maxSteps) || maxSteps < 0) {
      throw new ValueError('max_steps must be a non-negative integer');
    }
    const copied = new Map<string, ActionFunction<S>>(actions instanceof Map ? actions : Object.entries(actions ?? {}));
    if (![...copied.keys()].every((name) => typeof name === 'string' && name)) {
      throw new ValueError('action names must be non-empty strings');
    }
    if (![...copied.values()].every((action) => typeof action === 'function')) throw new TypeError('every action must be callable');
    this.chooser = chooser;
    this.actions = copied;
    this.maxSteps = maxSteps;
    Object.freeze(this);
  }

  /** Authorize execution from ``state``; returns the final state and receipts. */
  async call(state: S, options: { context?: Context | null } = {}): Promise<ActionLoopResult<S>> {
    const context = options.context ?? null;
    const receipts: ActionReceipt[] = [];
    const names = [...this.actions.keys()];
    let current = state;
    for (let step = 0; step < this.maxSteps; step += 1) {
      const request = new ActionRequest(current, names, step, receipts.map((receipt) => copyReceipt(receipt)));
      const choice = await this.chooser(request, { context });
      const selected = ActionLoop.selectedName(choice);
      if (selected === null) return new ActionLoopResult(current, receipts, 'abstained');
      const action = this.actions.get(selected);
      if (!action) throw new ValueError(`chosen action ${pythonRepr(selected)} is not one of the supplied actions`);
      const outcome = await action(current);
      if (!(outcome instanceof ActionOutcome)) throw new TypeError('actions must return ActionOutcome with an effect receipt');
      receipts.push(new ActionReceipt(step, selected, deepClone(outcome.receipt)));
      current = outcome.state;
      if (outcome.done) return new ActionLoopResult(current, receipts, 'completed');
    }
    return new ActionLoopResult(current, receipts, 'budget_exhausted');
  }

  static selectedName(choice: unknown): string | null {
    if (typeof choice === 'string') return choice;
    const record = (choice !== null && typeof choice === 'object' ? choice : {}) as { abstained?: unknown; value?: unknown };
    const abstained = record.abstained ?? false;
    if (typeof abstained !== 'boolean') throw new TypeError('chooser abstained flag must be boolean');
    if (abstained) return null;
    const selected = record.value ?? null;
    if (typeof selected !== 'string') throw new TypeError('chooser must return a string or a result with string value');
    return selected;
  }
}

function copyReceipt(receipt: ActionReceipt): ActionReceipt {
  return new ActionReceipt(receipt.step, receipt.action, deepClone(receipt.effect));
}

/** Python ``repr`` of a string (single quotes unless the text contains one). */
function pythonRepr(text: string): string {
  if (text.includes("'") && !text.includes('"')) return `"${text}"`;
  return `'${text.replaceAll('\\', '\\\\').replaceAll("'", "\\'")}'`;
}
