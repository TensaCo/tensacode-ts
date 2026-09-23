/**
 * Safe JSON tensor-module checkpoints with explicit binding and alias checks
 * (Python ``tensorcode/_internal/training/checkpoint.py``).
 *
 * The ``tensorcode.checkpoint`` v1 format stores binding fingerprints, the
 * shared-parameter alias topology, module state dicts and (optionally) the
 * optimizer state in the PyTorch ``state_dict`` layout, so SGD/Adam/AdamW
 * checkpoints are interchangeable with Python.
 */
import { Tensor } from '../../nn/tensor.js';
import { noGrad } from '../../nn/autograd.js';
import { Adam, AdamW, Optimizer, SGD, type OptimizerStateDict } from '../../nn/optim.js';
import { ValueError } from '../../errors.js';
import { validateBindings } from '../fingerprint.js';
import { isPlainObject } from '../json.js';
import {
  Codec, bindingsJson, parseArtifact, pythonCompare, readArtifact, writeArtifact,
} from './persistence.js';
import { collectParameters, validateOptimizer } from './trainer.js';
import type { OperationLike } from '../../ops/base.js';

type StateMap = Map<string, Tensor>;

interface StatefulOperation {
  stateDict(): StateMap | Record<string, Tensor>;
  loadStateDict(state: StateMap | Record<string, Tensor>, options?: { strict?: boolean }): unknown;
  namedParameters(options?: { removeDuplicate?: boolean }): Iterable<[string, Tensor]>;
}

/** Whether ``operation`` implements the state dict protocol (rejecting partial implementations). */
export function isStateful(operation: unknown): operation is StatefulOperation {
  const target = operation as Record<string, unknown>;
  const present = ['stateDict', 'loadStateDict', 'namedParameters'].map((name) => typeof target?.[name] === 'function');
  if (present.every(Boolean)) return true;
  const parameters = typeof target?.parameters === 'function' ? [...(target.parameters as () => Iterable<unknown>).call(operation)] : [];
  if (present.some(Boolean) || parameters.length) {
    throw new TypeError('Checkpoint restoration requires state_dict, load_state_dict and named_parameters (stateDict, loadStateDict, namedParameters)');
  }
  return false;
}

/** Live state of a binding as an ordered map (empty for parameterless operations). */
export function stateOf(operation: unknown): StateMap {
  if (!isStateful(operation)) return new Map();
  const state = operation.stateDict();
  return state instanceof Map ? state : new Map(Object.entries(state));
}

/** Independent copies of a binding's state (rollback snapshots). */
export function snapshotState(operation: unknown): StateMap {
  return noGrad(() => new Map([...stateOf(operation)].map(([key, value]) => [key, value.detach().clone()])));
}

export function loadStateOf(operation: unknown, state: StateMap | Record<string, Tensor>): void {
  if (isStateful(operation)) {
    operation.loadStateDict(state, { strict: true });
    return;
  }
  const size = state instanceof Map ? state.size : Object.keys(state).length;
  if (size) throw new ValueError('Parameterless operation checkpoint must have empty state');
}

function namedParametersOf(operation: unknown, removeDuplicate: boolean): [string, Tensor][] {
  return isStateful(operation) ? [...operation.namedParameters({ removeDuplicate })] : [];
}

/** Sorted groups of ``[binding, key]`` names sharing one parameter tensor. */
export function aliasGroups(operations: Record<string, unknown>): string[][][] {
  const groups = new Map<Tensor, string[][]>();
  for (const [name, operation] of Object.entries(operations)) {
    for (const [key, parameter] of namedParametersOf(operation, false)) {
      const group = groups.get(parameter) ?? [];
      group.push([name, key]);
      groups.set(parameter, group);
    }
  }
  return [...groups.values()].map((group) => [...group].sort(pythonCompare)).sort(pythonCompare);
}

export function optimizerLayout(optimizer: Optimizer, operations: Record<string, unknown>): string[][][][] {
  const names = new Map<Tensor, string[][]>();
  for (const [name, operation] of Object.entries(operations)) {
    for (const [key, parameter] of namedParametersOf(operation, true)) {
      const group = names.get(parameter) ?? [];
      group.push([name, key]);
      names.set(parameter, group);
    }
  }
  return optimizer.paramGroups.map((group) => group.params.map((parameter) => {
    const entry = names.get(parameter);
    if (!entry) throw new ValueError('Optimizer parameter is not owned by a checkpoint binding');
    return [...entry].sort(pythonCompare);
  }));
}

function jsonEqualStructure(a: unknown, b: unknown): boolean {
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((item, index) => jsonEqualStructure(item, b[index]));
  return a === b;
}

function isSupportedOptimizer(optimizer: Optimizer): boolean {
  return optimizer.constructor === SGD || optimizer.constructor === Adam || optimizer.constructor === AdamW;
}

/** Validate a (decoded) optimizer state against the live optimizer (Python ``_validate_optimizer_state``). */
export function validateOptimizerState(optimizer: Optimizer, state: unknown): asserts state is OptimizerStateDict {
  if (!isSupportedOptimizer(optimizer)) throw new TypeError('Optimizer checkpoints support SGD, Adam and AdamW');
  if (!isPlainObject(state) || Object.keys(state).length !== 2 || !isPlainObject(state.state) || !Array.isArray(state.param_groups)) {
    throw new ValueError('Malformed optimizer checkpoint state');
  }
  const groups = state.param_groups as unknown[];
  if (groups.length !== optimizer.paramGroups.length) throw new ValueError('Checkpoint optimizer parameter groups differ');
  const owners = new Map<string, Tensor>();
  groups.forEach((saved, index) => {
    const current = optimizer.paramGroups[index]!;
    const currentKeys = Object.keys(current);
    if (!isPlainObject(saved) || Object.keys(saved).length !== currentKeys.length || currentKeys.some((key) => !(key in saved))
      || !Array.isArray(saved.params) || saved.params.length !== current.params.length) {
      throw new ValueError('Checkpoint optimizer parameter group fields differ');
    }
    (saved.params as unknown[]).forEach((key, position) => {
      if (typeof key !== 'number' || !Number.isInteger(key) || owners.has(String(key))) throw new ValueError('Invalid optimizer parameter identity');
      owners.set(String(key), current.params[position]!);
    });
  });
  const slotsByKey = state.state as Record<string, unknown>;
  if (Object.keys(slotsByKey).some((key) => !owners.has(key))) throw new ValueError('Unknown optimizer state parameter');
  const sgd = optimizer.constructor === SGD;
  for (const [key, slots] of Object.entries(slotsByKey)) {
    if (!isPlainObject(slots)) throw new ValueError('Malformed optimizer parameter slots');
    const names = Object.keys(slots);
    if (sgd) {
      if (names.some((name) => name !== 'momentum_buffer')) throw new ValueError('Unknown SGD optimizer state slot');
    } else if (names.length && (!['step', 'exp_avg', 'exp_avg_sq'].every((name) => names.includes(name))
      || names.some((name) => !['step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'].includes(name)))) {
      throw new ValueError('Malformed Adam optimizer state slots');
    }
    const owner = owners.get(key)!;
    for (const [name, value] of Object.entries(slots)) {
      if (name === 'step') {
        if (!(value instanceof Tensor) || value.ndim !== 0 || !value.allFinite() || value.item() < 0) {
          throw new ValueError('Malformed optimizer step counter');
        }
      } else if (!(value instanceof Tensor) || value.dtype !== owner.dtype || value.shape.length !== owner.shape.length
        || value.shape.some((size, index) => size !== owner.shape[index])) {
        throw new ValueError('Checkpoint optimizer slot shape or dtype differs from parameter');
      }
    }
  }
}

/** The optimizer state in Python's shape (integer state keys, ``betas`` tuple) for encoding. */
export function pythonOptimizerState(optimizer: Optimizer): Map<string, unknown> {
  const state = optimizer.stateDict();
  const slots = new Map<number, Map<string, Tensor>>();
  for (const [key, value] of Object.entries(state.state)) slots.set(Number(key), new Map(Object.entries(value)));
  const groups = state.param_groups.map((group) => {
    const copy: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(group)) {
      copy[key] = key === 'betas' && Array.isArray(value) ? Object.freeze([...value]) : value;
    }
    return copy;
  });
  return new Map<string, unknown>([['state', slots], ['param_groups', groups]]);
}

export interface CheckpointOptions {
  operations: Record<string, OperationLike | object>;
  optimizer?: Optimizer | null;
  /** @internal shared codec (training directories reference one tensor file). */
  codec?: Codec;
}

/** Build a ``tensorcode.checkpoint`` payload (Python ``save_checkpoint`` content). */
export function checkpointPayload(options: CheckpointOptions): Record<string, unknown> {
  const codec = options.codec ?? new Codec();
  const operations = options.operations;
  const config = bindingsJson(operations as Record<string, OperationLike>);
  const aliases = aliasGroups(operations);
  const states: Record<string, unknown> = {};
  for (const [name, operation] of Object.entries(operations)) states[name] = codec.encode(stateOf(operation));
  const payload: Record<string, unknown> = {
    format: 'tensorcode.checkpoint', version: 1, operations: config, aliases, states, optimizer: null,
  };
  const optimizer = options.optimizer ?? null;
  if (optimizer !== null) {
    validateOptimizer(optimizer, collectParameters(operations));
    validateOptimizerState(optimizer, optimizer.stateDict());
    payload.optimizer = {
      type: optimizer.identity, layout: optimizerLayout(optimizer, operations), state: codec.encode(pythonOptimizerState(optimizer)),
    };
  }
  return payload;
}

/** Atomically write a model (and optional optimizer) checkpoint. */
export async function saveCheckpoint(path: string, options: CheckpointOptions): Promise<void> {
  await writeArtifact(path, checkpointPayload(options));
}

export interface PreparedCheckpoint {
  states: Map<string, Record<string, Tensor>>;
  optimizerState: OptimizerStateDict | null;
}

const CHECKPOINT_FIELDS = ['format', 'version', 'operations', 'aliases', 'states', 'optimizer'];

/** Decode and validate a parsed payload without mutating or snapshotting bound state. */
export function prepareCheckpoint(payload: Record<string, unknown>, options: CheckpointOptions): PreparedCheckpoint {
  const operations = options.operations;
  const keys = Object.keys(payload);
  if (keys.length !== CHECKPOINT_FIELDS.length || CHECKPOINT_FIELDS.some((key) => !keys.includes(key))) {
    throw new ValueError('Malformed checkpoint fields');
  }
  const names = Object.keys(operations);
  const same = (value: unknown): boolean => isPlainObject(value) && Object.keys(value).length === names.length
    && names.every((name) => Object.prototype.hasOwnProperty.call(value, name));
  if (!same(payload.operations) || !same(payload.states)) throw new ValueError('Checkpoint operation bindings must match exactly');
  validateBindings(payload.operations, operations as Record<string, OperationLike>);
  if (!jsonEqualStructure(payload.aliases, aliasGroups(operations))) {
    throw new ValueError('Shared-parameter alias topology differs from checkpoint');
  }
  const codec = options.codec ?? new Codec();
  const states = new Map<string, Record<string, Tensor>>();
  for (const [name, encoded] of Object.entries(payload.states as Record<string, unknown>)) {
    const state = codec.decode(encoded);
    const current = stateOf(operations[name]);
    if (!isPlainObject(state) || Object.keys(state).length !== current.size || [...current.keys()].some((key) => !(key in state))) {
      throw new ValueError('Checkpoint state keys differ from bound module');
    }
    for (const [key, value] of current) {
      const other = state[key];
      if (!(other instanceof Tensor) || value.dtype !== other.dtype || value.shape.length !== other.shape.length
        || value.shape.some((size, index) => size !== other.shape[index])) {
        throw new ValueError('Checkpoint tensor shape or dtype differs');
      }
    }
    states.set(name, state as Record<string, Tensor>);
  }
  for (const group of payload.aliases as string[][][]) {
    const [firstName, firstKey] = group[0]!;
    const first = states.get(firstName!)![firstKey!]!;
    for (const [name, key] of group.slice(1)) {
      if (!first.equal(states.get(name!)![key!]!)) throw new ValueError('Contradictory shared-parameter alias values');
    }
  }
  let optimizerState: OptimizerStateDict | null = null;
  const optimizer = options.optimizer ?? null;
  if (optimizer !== null) {
    validateOptimizer(optimizer, collectParameters(operations));
    const record = payload.optimizer;
    if (!isPlainObject(record) || Object.keys(record).length !== 3 || !('type' in record) || !('layout' in record) || !('state' in record)
      || record.type !== optimizer.identity || !jsonEqualStructure(record.layout, optimizerLayout(optimizer, operations))) {
      throw new ValueError('Checkpoint optimizer type or parameter layout differs');
    }
    const decoded = codec.decode(record.state);
    validateOptimizerState(optimizer, decoded);
    optimizerState = decoded;
  }
  return { states, optimizerState };
}

/** Apply prepared states inside the caller's rollback transaction. */
export function applyCheckpoint(prepared: PreparedCheckpoint, options: CheckpointOptions): void {
  for (const [name, state] of prepared.states) loadStateOf(options.operations[name], state);
  if (options.optimizer) options.optimizer.loadStateDict(prepared.optimizerState!);
}

/**
 * Restore supported module states after configuration/alias validation. A
 * supplied optimizer also restores its saved state; omit it to restore model
 * weights only. The artifact never creates modules, optimizers or classes.
 */
export async function loadCheckpoint(path: string, options: CheckpointOptions): Promise<void> {
  restoreCheckpointPayload(await readArtifact(path, 'tensorcode.checkpoint'), options);
}

/** Validate then apply an already parsed payload with full rollback on failure. */
export function restoreCheckpointPayload(payload: Record<string, unknown>, options: CheckpointOptions): void {
  const prepared = prepareCheckpoint(payload, options);
  const originals = new Map(Object.entries(options.operations).map(([name, operation]) => [name, snapshotState(operation)]));
  const optimizer = options.optimizer ?? null;
  const originalOptimizer = optimizer ? optimizer.stateDict() : null;
  try {
    applyCheckpoint(prepared, options);
  } catch (error) {
    for (const [name, state] of originals) loadStateOf(options.operations[name], state);
    if (optimizer) optimizer.loadStateDict(originalOptimizer!);
    throw error;
  }
}

/** Parse checkpoint text (tests and tools). */
export function parseCheckpoint(text: string): Record<string, unknown> {
  return parseArtifact(text, 'tensorcode.checkpoint');
}
