/**
 * Explicit action callback records and bounded orchestration construction
 * (Python ``tensorcode/tools/actions.py``). The plan execution records of
 * ``tensorcode._internal.execution.planning`` are re-exported for convenience.
 */
import { ActionLoop, type ActionFunction, type Chooser } from '../_internal/execution/actionLoop.js';

/** What a chooser sees: current state, allowed action names, step index and prior receipts. */
export class ActionRequest<S = unknown> {
  static readonly qualifiedName: string = 'tensorcode.tools.actions.ActionRequest';
  static readonly recordFields = ['state', 'options', 'step', 'receipts'] as const;
  readonly state: S;
  readonly options: readonly string[];
  readonly step: number;
  readonly receipts: readonly ActionReceipt[];

  constructor(state: S, options: readonly string[], step: number, receipts: readonly ActionReceipt[]) {
    this.state = state;
    this.options = Object.freeze([...options]);
    this.step = step;
    this.receipts = Object.freeze([...receipts]);
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): ActionRequest {
    return new ActionRequest(fields.state, fields.options as string[], fields.step as number, fields.receipts as ActionReceipt[]);
  }

  toRecord(): Record<string, unknown> {
    return { state: this.state, options: this.options, step: this.step, receipts: this.receipts };
  }
}

/** Returned by an action: the next state, an effect receipt, and whether the run is done. */
export class ActionOutcome<S = unknown> {
  static readonly qualifiedName: string = 'tensorcode.tools.actions.ActionOutcome';
  static readonly recordFields = ['state', 'receipt', 'done'] as const;
  readonly state: S;
  readonly receipt: unknown;
  readonly done: boolean;

  constructor(state: S, receipt: unknown, done = false) {
    this.state = state;
    this.receipt = receipt;
    this.done = done;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): ActionOutcome {
    return new ActionOutcome(fields.state, fields.receipt, (fields.done as boolean | undefined) ?? false);
  }

  toRecord(): Record<string, unknown> {
    return { state: this.state, receipt: this.receipt, done: this.done };
  }
}

/** Snapshot of one executed action and the receipt it returned. */
export class ActionReceipt {
  static readonly qualifiedName: string = 'tensorcode.tools.actions.ActionReceipt';
  static readonly recordFields = ['step', 'action', 'effect'] as const;
  readonly step: number;
  readonly action: string;
  readonly effect: unknown;

  constructor(step: number, action: string, effect: unknown) {
    this.step = step;
    this.action = action;
    this.effect = effect;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): ActionReceipt {
    return new ActionReceipt(fields.step as number, fields.action as string, fields.effect);
  }

  toRecord(): Record<string, unknown> {
    return { step: this.step, action: this.action, effect: this.effect };
  }
}

export type ActionStopReason = 'completed' | 'abstained' | 'budget_exhausted';

/** Final state, receipts, and ``stop_reason`` (completed, abstained or budget_exhausted). */
export class ActionLoopResult<S = unknown> {
  static readonly qualifiedName: string = 'tensorcode.tools.actions.ActionLoopResult';
  static readonly recordFields = ['state', 'receipts', 'stop_reason'] as const;
  readonly state: S;
  readonly receipts: readonly ActionReceipt[];
  readonly stopReason: ActionStopReason;

  constructor(state: S, receipts: readonly ActionReceipt[], stopReason: ActionStopReason) {
    this.state = state;
    this.receipts = Object.freeze([...receipts]);
    this.stopReason = stopReason;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): ActionLoopResult {
    return new ActionLoopResult(fields.state, fields.receipts as ActionReceipt[], fields.stop_reason as ActionStopReason);
  }

  toRecord(): Record<string, unknown> {
    return { state: this.state, receipts: this.receipts, stop_reason: this.stopReason };
  }
}

export interface ActionLoopOptions<S = unknown> {
  chooser: Chooser<S>;
  actions: Record<string, ActionFunction<S>> | Map<string, ActionFunction<S>>;
  maxSteps: number;
}

/**
 * Construct a bounded loop without invoking the chooser or any action.
 *
 * Choosers receive an {@link ActionRequest} and ``{ context }`` and return an
 * action name (or a result with ``value``/``abstained``); actions receive the
 * state and return an {@link ActionOutcome}. Both may be synchronous or
 * asynchronous. ``await loop.call(state, { context })`` authorizes execution and
 * returns an {@link ActionLoopResult}.
 */
export function actionLoop<S = unknown>(options: ActionLoopOptions<S>): ActionLoop<S> {
  return new ActionLoop<S>(options);
}

export type { ActionFunction, Chooser, ChooserResult } from '../_internal/execution/actionLoop.js';
export { ActionLoop } from '../_internal/execution/actionLoop.js';
export {
  PlanStep, ExecutablePlan, OutcomeExperience, ReplanRequest, PlanExecutionResult, PlanExecutor, withSignature,
  type PlanAction, type ReplanPolicy, type PlanStopReason, type ActionSignature, type ActionParameter,
} from '../_internal/execution/planning.js';
