/**
 * Cross-module contracts fixed by the foundation so the operation, tool and
 * training modules can be built independently. FOUNDATION-OWNED.
 */
import type { Parameter, Tensor } from '../nn/tensor.js';
import type { Module } from '../nn/module.js';
import { ValueError } from '../errors.js';
import { isPlainObject } from './json.js';
import type { Context, OperationLike } from '../ops/base.js';

/** Anything whose trainable tensors a trainer can collect (Python ``operation.parameters()``). */
export interface HasParameters {
  parameters(): Parameter[];
}

/**
 * A tool or owned operation trainable through ``Trainer.fromTool`` (Python
 * ``TrainingEngine``/``ToolObjective`` protocol):
 *
 * - ``operationBindings()`` — stable names → registered operations (must include
 *   ``trainingOperation``);
 * - ``trainingOperation`` — the declared objective operation;
 * - ``trainingInputsIncludeTargets`` — when true (all built-in tools), the
 *   objective receives ``{inputs, targets}`` and returns the scalar loss;
 *   otherwise ``trainingLoss(output, target)`` computes it;
 * - optional ``train()``, ``stateDict()``/``configuration()`` — when both of the
 *   latter exist, checkpoints bind the whole tool as ``{tool}``.
 */
export interface TrainableTool {
  operationBindings(): Record<string, OperationLike>;
  readonly trainingOperation: OperationLike;
  readonly trainingInputsIncludeTargets?: boolean;
  trainingLoss?(output: unknown, target: unknown): Tensor;
  train?(mode?: boolean): unknown;
  stateDict?(): Map<string, Tensor>;
  configuration?(): unknown;
}

export function isTrainableTool(value: unknown): value is TrainableTool {
  return typeof value === 'object' && value !== null
    && typeof (value as { operationBindings?: unknown }).operationBindings === 'function'
    && typeof (value as { trainingOperation?: unknown }).trainingOperation === 'object';
}

export interface ObjectiveEnvelope {
  inputs: unknown;
  targets: unknown;
  context: Context | null;
}

/**
 * Parse an objective call ``{inputs, targets}``; ``inputs`` may itself be a
 * conditioning envelope ``{value, context}`` (Python owned objectives). Context
 * must appear only inside that envelope.
 */
export function parseObjectiveEnvelope(value: unknown, context: Context | null): ObjectiveEnvelope {
  if (!isPlainObject(value) || Object.keys(value).length !== 2 || !('inputs' in value) || !('targets' in value)) {
    throw new ValueError('objective envelope requires inputs and targets');
  }
  let inputs = value.inputs;
  let conditioning = context;
  if (isPlainObject(inputs)) {
    if (Object.keys(inputs).length !== 2 || !('value' in inputs) || !('context' in inputs)) {
      throw new ValueError('conditioning envelope requires exactly value and context');
    }
    if (context) throw new ValueError('context must appear only inside the conditioning envelope');
    conditioning = (inputs.context as Context | null) ?? null;
    inputs = inputs.value;
  }
  return { inputs, targets: value.targets, context: conditioning };
}

/** A module-owning object usable as a checkpoint binding (state dict protocol). */
export type Stateful = Module;
