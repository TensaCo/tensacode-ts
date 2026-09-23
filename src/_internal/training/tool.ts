/**
 * Explicit objective adaptation and a shared resumable training engine (Python
 * ``tensorcode/_internal/training/tool.py``).
 *
 * Directory checkpoints (``training.json`` + ``tensors-<uuid>.safetensors``)
 * preserve weights, optimizer state, per-module training modes, the step count,
 * caller progress and both random generators, in exactly Python's format:
 * ``python_rng`` is CPython's ``random.getstate()`` (the process-wide
 * ``pythonRandom``), ``torch_rng`` is ``torch.get_rng_state()`` (the default
 * PyTorch-compatible generator) and ``cuda_rng`` is empty. Checkpoints move
 * between the two languages in both directions and restore the same random
 * streams.
 */
import { mkdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { Module } from '../../nn/module.js';
import { Generator } from '../../nn/random.js';
import { PythonRandom, pythonRandom, type PythonRandomState } from '../../nn/randomPython.js';
import { getRngState, rngStateTensor, setRngState } from '../../nn/randomTensor.js';
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { PythonFloat, isPlainObject } from '../json.js';
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

const STATE_KEYS = ['modes', 'steps', 'progress', 'python_rng', 'torch_rng', 'cuda_rng'];

function hasExactly(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const own = Object.keys(value);
  return own.length === keys.length && keys.every((key) => own.includes(key));
}

/** ``random.getstate()`` as the codec's tuple: ``(3, (words..., index), gauss_next)``. */
function pythonRngValue(state: PythonRandomState): unknown {
  const [version, internal, gaussNext] = state;
  return Object.freeze([version, Object.freeze([...internal]), gaussNext === null ? null : new PythonFloat(gaussNext)]);
}

/** Validate a decoded ``random.getstate()`` like ``random.Random().setstate``. */
function parsePythonRng(value: unknown): PythonRandomState {
  if (!Array.isArray(value)) throw new ValueError('Invalid Python RNG state');
  if (Array.isArray(value[1]) && !Object.isFrozen(value[1])) throw new TypeError('state vector must be a tuple');
  new PythonRandom(0).setstate(value);
  const [version, internal, gaussNext] = value as [number, unknown[], number | null];
  return [version, (internal as unknown[]).map((item) => Number(item)), gaussNext === null ? null : Number(gaussNext)];
}

/** Validate a decoded ``torch.get_rng_state()`` like ``torch.Generator().set_state``. */
function parseTorchRng(value: unknown): Tensor {
  if (!(value instanceof Tensor)) throw new ValueError('Invalid PyTorch RNG state');
  new Generator().setState(value);
  return value;
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
      modes, steps: this.steps, progress: codec.encode(progress),
      python_rng: codec.encode(pythonRngValue(pythonRandom.getstate())),
      torch_rng: codec.encode(rngStateTensor(getRngState())),
      cuda_rng: codec.encode([]),
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
    if (!isPlainObject(state) || !hasExactly(state, STATE_KEYS)) {
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
    const pythonRng = parsePythonRng(codec.decode(state.python_rng));
    const torchRng = parseTorchRng(codec.decode(state.torch_rng));
    const cudaRng = codec.decode(state.cuda_rng);
    if (!Array.isArray(cudaRng)) throw new ValueError('Invalid CUDA RNG states');
    if (cudaRng.length) throw new ValueError('CUDA device topology differs from training checkpoint');
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
    const originalTorchRng = getRngState();
    const originalPythonRng = pythonRandom.getstate();
    try {
      applyCheckpoint(prepared, options);
      // Set exact local flags, without recursively resetting mixed modes or
      // invoking train() overrides that may force a different policy.
      for (const [name, children] of Object.entries(modules)) {
        for (const [key, module] of children) module.training = (modes[name] as Record<string, boolean>)[key]!;
      }
      TrainingEngine.restoreRng(torchRng, pythonRng);
    } catch (error) {
      for (const [name, snapshot] of originals) loadStateOf(this.checkpointOperations[name], snapshot);
      this.optimizer.loadStateDict(originalOptimizer);
      for (const [module, flag] of originalModes) module.training = flag;
      pythonRandom.setstate(originalPythonRng);
      setRngState(originalTorchRng);
      throw error;
    }
    this.steps = steps;
    this.progress = progress;
    return deepCopy(progress);
  }

  /** @internal RNG restoration hook (replaceable in tests to simulate interruption). */
  static restoreRng: (torchState: Tensor, pythonState: PythonRandomState) => void = (torchState, pythonState) => {
    pythonRandom.setstate(pythonState);
    setRngState(torchState);
  };
}
