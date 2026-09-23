/**
 * Explicit supervised training and portable experience (Python
 * ``tensorcode.training``).
 *
 * - {@link Trainer}: train a declared tool objective (``Trainer.fromTool``) or an
 *   explicitly supervised operation graph (``Trainer.fromOps``), with complete
 *   resumable checkpoints.
 * - {@link loadExperience}: load a ``tensorcode.experience`` artifact bound to
 *   supplied operations (``trace.save`` writes them).
 * - Calibration utilities: {@link TemperatureCalibration},
 *   {@link evaluateCalibration}, {@link fitThreshold}.
 *
 * Importing this module performs no I/O and loads no model weights.
 */
import { ValueError } from '../errors.js';
import type { Optimizer } from '../nn/optim.js';
import type { Tensor } from '../nn/tensor.js';
import type { TrainableTool } from '../_internal/contracts.js';
import type { Trace } from '../_internal/tracing.js';
import type { OperationLike } from '../ops/base.js';
import { TrainingEngine } from '../_internal/training/tool.js';
import type { LossFunction, OptimizerOption } from '../_internal/training/trainer.js';

export { loadExperience, type ExperienceOptions } from '../_internal/training/persistence.js';
export {
  TemperatureCalibration, evaluateCalibration, fitThreshold,
  type CalibrationMetrics, type CalibrationReport, type ThresholdReport,
} from './calibration.js';
export type { LossFunction, OptimizerOption } from '../_internal/training/trainer.js';

export interface FromToolOptions {
  /** An optimizer instance or a factory ``(params) => Optimizer`` (default SGD). */
  optimizer?: OptimizerOption | null;
  /** Learning rate of the default SGD optimizer (default ``0.001``). */
  lr?: number;
}

export interface FromOpsOptions extends FromToolOptions {
  /** Learning rate of the default SGD optimizer (default ``0.01``). */
  lr?: number;
  /** Custom loss callbacks keyed by supervision loss name. */
  losses?: Record<string, LossFunction> | null;
}

const CONSTRUCTION = Symbol('tensorcode.Trainer.construction');

/**
 * Train a declared tool objective or an explicitly supervised operation graph.
 *
 * Construct with {@link Trainer.fromTool} or {@link Trainer.fromOps}. Complete
 * checkpoints preserve weights, optimizer, the TensorCode random generator,
 * module modes, step count and progress. Experience and model deployment
 * artifacts are saved separately.
 */
export class Trainer {
  static readonly qualifiedName: string = 'tensorcode.training.Trainer';
  /** @internal */
  private readonly engine: TrainingEngine;

  constructor(token?: unknown, engine?: TrainingEngine) {
    if (token !== CONSTRUCTION || !engine) throw new TypeError('Use Trainer.fromTool(...) or Trainer.fromOps(...)');
    this.engine = engine;
  }

  /** Use the tool's declared objective and activate its training policy. */
  static fromTool(tool: TrainableTool, options: FromToolOptions = {}): Trainer {
    const engine = new TrainingEngine(tool.operationBindings(), {
      tool, optimizer: options.optimizer ?? null, lr: options.lr ?? 0.001,
    });
    return new Trainer(CONSTRUCTION, engine);
  }

  /** Use explicit trace supervision; retain existing operation modes. */
  static fromOps(operations: Record<string, OperationLike>, options: FromOpsOptions = {}): Trainer {
    const engine = new TrainingEngine(operations, {
      optimizer: options.optimizer ?? null, lr: options.lr ?? 0.01, losses: options.losses ?? null,
    });
    return new Trainer(CONSTRUCTION, engine);
  }

  /** Named operations whose parameters this trainer optimizes. */
  get operations(): Record<string, OperationLike> {
    return this.engine.operations;
  }

  /** Trainable parameters collected from the bound operations. */
  get parameters(): Tensor[] {
    return this.engine.parameters;
  }

  /** The optimizer (supplied, or SGD with the given ``lr``). */
  get optimizer(): Optimizer {
    return this.engine.optimizer;
  }

  /** Number of optimizer updates applied so far. */
  get steps(): number {
    return this.engine.steps;
  }

  set steps(value: number) {
    if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) throw new ValueError('steps must be a nonnegative integer');
    this.engine.steps = value;
  }

  /** Caller-defined progress mapping saved with checkpoints. */
  get progress(): Record<string, unknown> {
    return this.engine.progress;
  }

  set progress(value: Record<string, unknown>) {
    if (value === null || typeof value !== 'object' || Array.isArray(value) || Object.getPrototypeOf(value) !== Object.prototype) {
      throw new TypeError('progress must be a dictionary');
    }
    this.engine.progress = value;
  }

  /** The bound tool for ``fromTool`` trainers, otherwise ``null``. */
  get tool(): TrainableTool | null {
    return this.engine.tool;
  }

  /** Capture reviewed tool feedback; operation graphs use ``trace().supervise()``. */
  capture(inputs: unknown, targets: unknown, options: { source: string }): Trace {
    return this.engine.capture(inputs, targets, options);
  }

  /** Apply one optimizer step to one captured experience; returns the loss. */
  step(experience: Trace): number {
    return this.engine.step(experience);
  }

  /** Step through the experiences for ``epochs`` passes, in order. */
  fit(experiences: Iterable<Trace>, options: { epochs?: number } = {}): number[] {
    return this.engine.fit(experiences, options);
  }

  /** Save complete continuation state; deploy weights with ``savePretrained``. */
  saveCheckpoint(path: string, options: { progress?: Record<string, unknown> | null } = {}): Promise<void> {
    return this.engine.saveCheckpoint(path, options);
  }

  /**
   * Restore complete directories, or standalone model/optimizer checkpoint
   * files. Standalone files leave RNG, modes, steps and progress unchanged.
   * Python directory checkpoints restore everything except their PyTorch and
   * CPython random generator states, which TypeScript cannot reproduce.
   */
  loadCheckpoint(path: string): Promise<Record<string, unknown>> {
    return this.engine.loadCheckpoint(path);
  }
}
