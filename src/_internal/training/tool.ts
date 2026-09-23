/**
 * Explicit objective adaptation and a shared resumable training engine (Python
 * ``tensorcode/_internal/training/tool.py``).
 *
 * Directory checkpoints (``training.json`` + ``tensors-<uuid>.safetensors``)
 * preserve weights, optimizer state, per-module training modes, the step count,
 * caller progress and the TensorCode random generator.
 *
 * Cross-language note: Python directory checkpoints carry ``python_rng``,
 * ``torch_rng`` and ``cuda_rng``. TypeScript cannot restore PyTorch or CPython
 * generators, so loading such a checkpoint validates and ignores them (weights,
 * optimizer, modes, steps and progress are restored). TypeScript checkpoints
 * store ``{rng, runtime: 'typescript'}`` instead, which Python rejects.
 */
import { mkdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { Module } from '../../nn/module.js';
import { Generator, getRngState, setRngState, type GeneratorState } from '../../nn/random.js';
import { ValueError } from '../../errors.js';
import { isPlainObject } from '../json.js';
import { snapshot, trace, type Trace } from '../tracing.js';
import type { TrainableTool } from '../contracts.js';
import type { OperationLike } from '../../ops/base.js';
import {
  applyCheckpoint, checkpointPayload, loadCheckpoint as loadStandaloneCheckpoint, loadStateOf, prepareCheckpoint, snapshotState,
} from './checkpoint.js';
import { readArtifact, writeArtifact } from './persistence.js';
import { OperationTrainer, type LossFunction, type TrainerOptions } from './trainer.js';
import { TensorStore } from './tensorStore.js';

/** Adapt only a tool's explicitly declared input and loss contract. */
export class ToolObjective {
  readonly operation: OperationLike;
  readonly checkpointOperations: Record<string, OperationLike | object>;
  readonly joint: boolean;
  readonly loss: LossFunction;

  constructor(tool: TrainableTool, operations: Record<string, OperationLike>) {
    this.operation = tool.trainingOperation;
    if (!Object.values(operations).some((operation) => operation === this.operation)) {
      throw new ValueError('training_operation must appear in operation_bindings()');
    }
    const candidate = tool as unknown as { stateDict?: unknown; configuration?: unknown };
    this.checkpointOperations = typeof candidate.stateDict === 'function' && typeof candidate.configuration === 'function'
      ? { tool } : operations;
    this.joint = Boolean(tool.trainingInputsIncludeTargets);
    if (this.joint) {
      this.loss = (output) => output;
    } else {
      if (typeof tool.trainingLoss !== 'function') {
        throw new TypeError('A tool without trainingInputsIncludeTargets must define trainingLoss(output, targets)');
      }
      this.loss = (output, target) => tool.trainingLoss!(output, target);
    }
  }

  capture(inputs: unknown, targets: unknown, options: { source: string }): Trace {
    const source = options?.source;
    if (typeof source !== 'string' || !source.trim()) throw new ValueError('Feedback source must be a nonempty string');
    const value = this.joint ? { inputs, targets } : inputs;
    const experience = trace();
    const output = experience.run(() => this.operation.call(value));
    experience.supervise(output, targets, { loss: 'tool_objective', source });
    return experience;
  }
}

export interface EngineOptions extends TrainerOptions {
  tool?: TrainableTool | null;
}

const PYTHON_STATE = ['modes', 'steps', 'progress', 'python_rng', 'torch_rng', 'cuda_rng'];
const TYPESCRIPT_STATE = ['modes', 'steps', 'progress', 'rng', 'runtime'];

function hasExactly(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const own = Object.keys(value);
  return own.length === keys.length && keys.every((key) => own.includes(key));
}

function rngRecord(state: GeneratorState): Record<string, unknown> {
  return { algorithm: state.algorithm, words: [...state.words], spare_normal: state.spareNormal };
}

function parseRng(value: unknown): GeneratorState {
  if (!isPlainObject(value) || !hasExactly(value, ['algorithm', 'words', 'spare_normal'])) {
    throw new ValueError('Invalid TensorCode generator state');
  }
  const state = { algorithm: value.algorithm, words: value.words, spareNormal: value.spare_normal } as GeneratorState;
  new Generator().setState(state); // validates (TypeError on malformed state)
  return { algorithm: state.algorithm, words: [...state.words] as [number, number, number, number], spareNormal: state.spareNormal };
}

function deepCopy<T>(value: T): T {
  return snapshot(value);
}

/** One optimization and checkpoint lifecycle for explicit objective adapters. */
export class TrainingEngine extends OperationTrainer {
  readonly tool: TrainableTool | null;
  readonly objective: ToolObjective | null;
  readonly checkpointOperations: Record<string, OperationLike | object>;
  steps = 0;
  progress: Record<string, unknown> = {};

  constructor(operations: Record<string, OperationLike>, options: EngineOptions = {}) {
    const tool = options.tool ?? null;
    const objective = tool !== null ? new ToolObjective(tool, operations) : null;
    const losses = objective !== null ? { tool_objective: objective.loss } : options.losses;
    super(operations, { optimizer: options.optimizer ?? null, lr: options.lr, losses });
    this.tool = tool;
    this.objective = objective;
    this.checkpointOperations = { ...(objective !== null ? objective.checkpointOperations : operations) };
    if (tool !== null && typeof tool.train === 'function') tool.train();
  }

  private modeModules(): Record<string, Map<string, Module>> {
    const result: Record<string, Map<string, Module>> = {};
    for (const [name, operation] of Object.entries(this.checkpointOperations)) {
      result[name] = operation instanceof Module || typeof (operation as { namedModules?: unknown }).namedModules === 'function'
        ? new Map((operation as Module).namedModules()) : new Map();
    }
    return result;
  }

  /** Capture explicit tool feedback, separately from optimizer updates. */
  capture(inputs: unknown, targets: unknown, options: { source: string }): Trace {
    if (this.objective === null) {
      throw new ValueError('Operation trainers require explicit trace() capture and supervise(); '
        + 'use Trainer.fromTool for a declared objective');
    }
    return this.objective.capture(inputs, targets, options);
  }

  override step(session: Trace): number {
    const loss = super.step(session);
    this.steps += 1;
    return loss;
  }

  /**
   * Save weights, optimizer, module modes, RNG and progress without executable
   * serialization. An atomic JSON manifest selects immutable, checksummed
   * safetensors; older tensor generations remain valid for concurrent readers.
   */
  async saveCheckpoint(directory: string, options: { progress?: Record<string, unknown> | null } = {}): Promise<void> {
    await mkdir(directory, { recursive: true });
    const codec = new TensorStore();
    const progress = options.progress ?? this.progress;
    if (!isPlainObject(progress)) throw new TypeError('progress must be a dictionary');
    const modes: Record<string, Record<string, boolean>> = {};
    for (const [name, modules] of Object.entries(this.modeModules())) {
      modes[name] = Object.fromEntries([...modules].map(([key, module]) => [key, module.training]));
    }
    const state = {
      modes, steps: this.steps, progress: codec.encode(progress), rng: rngRecord(getRngState()), runtime: 'typescript',
    };
    const model = checkpointPayload({ operations: this.checkpointOperations, optimizer: this.optimizer, codec });
    const tensors = await codec.write(directory);
    await TrainingEngine.writeManifest(join(directory, 'training.json'), {
      format: 'tensorcode.tool_training', version: 1, model, state, tensors,
    });
    this.progress = deepCopy(progress);
  }

  /** @internal manifest writer (replaceable in tests to simulate interrupted writes). */
  static writeManifest: (path: string, data: unknown) => Promise<void> = writeArtifact;

  /**
   * Restore an initialized matching tool/optimizer and return progress. A
   * standalone ``tensorcode.checkpoint`` file restores model/optimizer only and
   * leaves RNG, modes, steps and progress unchanged. Callers should record
   * their own data cursor in ``progress``.
   */
  async loadCheckpoint(path: string): Promise<Record<string, unknown>> {
    if ((await stat(path)).isFile()) {
      await loadStandaloneCheckpoint(path, { operations: this.checkpointOperations, optimizer: this.optimizer });
      return deepCopy(this.progress);
    }
    const payload = await readArtifact(join(path, 'training.json'), 'tensorcode.tool_training');
    if (!hasExactly(payload, ['format', 'version', 'model', 'state', 'tensors'])) throw new ValueError('Malformed tool training checkpoint');
    const state = payload.state;
    if (!isPlainObject(state) || !(hasExactly(state, TYPESCRIPT_STATE) || hasExactly(state, PYTHON_STATE))) {
      throw new ValueError('Malformed training progress');
    }
    const steps = state.steps;
    if (typeof steps !== 'number' || !Number.isInteger(steps) || steps < 0) throw new ValueError('Invalid training step count');
    const modules = this.modeModules();
    const modes = state.modes;
    const names = Object.keys(modules);
    if (!isPlainObject(modes) || !hasExactly(modes, names)) throw new ValueError('Checkpoint module mode topology differs');
    const seen = new Map<Module, boolean>();
    for (const [name, children] of Object.entries(modules)) {
      const saved = modes[name];
      if (!isPlainObject(saved) || !hasExactly(saved, [...children.keys()])) throw new ValueError('Checkpoint module mode topology differs');
      for (const [key, module] of children) {
        const flag = saved[key];
        if (typeof flag !== 'boolean') throw new ValueError('Module training mode must be boolean');
        if (seen.has(module) && seen.get(module) !== flag) throw new ValueError('Contradictory shared module training modes');
        seen.set(module, flag);
      }
    }
    const codec = await TensorStore.read(path, payload.tensors, payload);
    const progress = codec.decode(state.progress);
    if (!isPlainObject(progress)) throw new ValueError('Invalid training progress');
    let rng: GeneratorState | null = null;
    if ('runtime' in state) {
      if (state.runtime !== 'typescript') throw new ValueError('Unknown training checkpoint runtime');
      rng = parseRng(state.rng);
    } else {
      // Python checkpoint: validate the recorded generator payloads, then ignore
      // them — CPython/PyTorch generator streams cannot be restored here.
      codec.decode(state.python_rng);
      codec.decode(state.torch_rng);
      if (!Array.isArray(codec.decode(state.cuda_rng))) throw new ValueError('Invalid CUDA RNG states');
    }
    const model = payload.model;
    if (!isPlainObject(model) || model.format !== 'tensorcode.checkpoint' || model.version !== 1) {
      throw new ValueError('Unknown artifact format or version');
    }
    // Validate before allocating rollback copies. This outer transaction owns
    // the sole model/optimizer snapshot as well as modes and RNG restoration.
    const options = { operations: this.checkpointOperations, optimizer: this.optimizer, codec };
    const prepared = prepareCheckpoint(model, options);
    const originals = new Map(Object.entries(this.checkpointOperations).map(([name, operation]) => [name, snapshotState(operation)]));
    const originalOptimizer = this.optimizer.stateDict();
    const originalModes = Object.values(modules).flatMap((children) => [...children.values()].map((module) => [module, module.training] as const));
    const originalRng = getRngState();
    try {
      applyCheckpoint(prepared, options);
      // Set exact local flags, without recursively resetting mixed modes or
      // invoking train() overrides that may force a different policy.
      for (const [name, children] of Object.entries(modules)) {
        for (const [key, module] of children) module.training = (modes[name] as Record<string, boolean>)[key]!;
      }
      if (rng !== null) TrainingEngine.restoreRng(rng);
    } catch (error) {
      for (const [name, snapshot] of originals) loadStateOf(this.checkpointOperations[name], snapshot);
      this.optimizer.loadStateDict(originalOptimizer);
      for (const [module, flag] of originalModes) module.training = flag;
      setRngState(originalRng);
      throw error;
    }
    this.steps = steps;
    this.progress = progress;
    return deepCopy(progress);
  }

  /** @internal RNG restoration hook (replaceable in tests to simulate interruption). */
  static restoreRng: (state: GeneratorState) => void = setRngState;
}
