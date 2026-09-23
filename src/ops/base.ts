/**
 * The common callable boundary: ``operation.call(value, { context })``.
 *
 * Python operations are invoked as ``op(value, context=None)``. TypeScript
 * instances are not callable, so every operation exposes:
 *
 * - ``call(value, { context })`` — synchronous invocation through the active
 *   trace boundary (Python ``op(value, context=...)``);
 * - ``acall(value, { context })`` — asynchronous invocation through the same
 *   boundary (Python ``await op.acall(...)``);
 * - ``forward(value, context)`` — the implementation hook. Callers never invoke
 *   ``forward`` directly; that would bypass tracing.
 *
 * ``replayable`` is an accessor. Pure operations opt into replay with
 * ``override get replayable() { return true; }``; I/O operations keep the
 * default ``false`` so replay cannot silently repeat external effects.
 *
 * Operations that own tensors extend {@link ModuleOperation} (an ``nn.Module``);
 * weightless operations extend {@link Operation}. Both carry the same brand, so
 * {@link isOperation} recognises either.
 */
import { Module } from '../nn/module.js';
import { invoke, invokeAsync } from '../_internal/tracing.js';

/** Conditioning data. Required operands belong in the primary value. */
export type Context = Record<string, unknown>;

export interface CallOptions {
  context?: Context | null;
}

export const OPERATION_BRAND: unique symbol = Symbol.for('tensorcode.operation');

/** Structural contract shared by {@link Operation} and {@link ModuleOperation}. */
export interface OperationLike<I = unknown, O = unknown> {
  readonly [OPERATION_BRAND]: true;
  readonly replayable: boolean;
  call(value: I, options?: CallOptions): O;
  acall(value: I, options?: CallOptions): Promise<O>;
  forward(value: I, context: Context | null): O;
  aforward(value: I, context: Context | null): Promise<O>;
  /** Optional truthful JSON metadata used by fingerprints and persistence. */
  configuration?(): unknown;
}

export function isOperation(value: unknown): value is OperationLike {
  return typeof value === 'object' && value !== null && (value as { [OPERATION_BRAND]?: unknown })[OPERATION_BRAND] === true;
}

function contextOf(options: CallOptions | undefined): Context | null {
  if (options === undefined || options === null) return null;
  if (typeof options !== 'object') throw new TypeError('call options must be an object such as { context }');
  const unknown = Object.keys(options).filter((key) => key !== 'context');
  if (unknown.length) throw new TypeError(`unknown call options: ${unknown.join(', ')}`);
  const context = options.context ?? null;
  if (context !== null && (typeof context !== 'object' || Array.isArray(context))) {
    throw new TypeError('context must be a mapping of conditioning values');
  }
  return context;
}

/** A traceable weightless operation. */
export abstract class Operation<I = unknown, O = unknown> implements OperationLike<I, O> {
  static readonly qualifiedName: string = 'tensorcode.ops.base.Operation';
  readonly [OPERATION_BRAND] = true as const;

  /** Opt in only when replay cannot perform external effects. */
  get replayable(): boolean {
    return false;
  }

  call(value: I, options?: CallOptions): O {
    return invoke(this, value, contextOf(options), (v, c) => this.forward(v, c));
  }

  acall(value: I, options?: CallOptions): Promise<O> {
    return invokeAsync(this, value, contextOf(options), (v, c) => this.aforward(v, c));
  }

  /** Asynchronous implementation hook; defaults to running {@link forward}. */
  async aforward(value: I, context: Context | null): Promise<O> {
    return this.forward(value, context);
  }

  /** Implement the transformation; callers use {@link call}, not ``forward``. */
  abstract forward(value: I, context: Context | null): O;
}

/**
 * A traceable operation that is also an ``nn.Module`` (owns parameters or
 * buffers, supports ``train()``/``eval()`` and ``stateDict()``).
 */
export abstract class ModuleOperation<I = unknown, O = unknown> extends Module implements OperationLike<I, O> {
  static override readonly qualifiedName: string = 'tensorcode.ops.base.Operation';
  readonly [OPERATION_BRAND] = true as const;

  get replayable(): boolean {
    return false;
  }

  call(value: I, options?: CallOptions): O {
    return invoke(this, value, contextOf(options), (v, c) => this.forward(v, c));
  }

  acall(value: I, options?: CallOptions): Promise<O> {
    return invokeAsync(this, value, contextOf(options), (v, c) => this.aforward(v, c));
  }

  async aforward(value: I, context: Context | null): Promise<O> {
    return this.forward(value, context);
  }

  abstract forward(value: I, context: Context | null): O;
}
