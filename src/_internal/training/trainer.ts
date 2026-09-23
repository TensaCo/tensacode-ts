/**
 * Explicit gradient training for captured, supervised tensor paths (Python
 * ``tensorcode/_internal/training/trainer.py``).
 */
import { Optimizer, SGD } from '../../nn/optim.js';
import { Tensor, tensor as makeTensor, type NestedNumbers } from '../../nn/tensor.js';
import { isFloatingDType } from '../../nn/dtype.js';
import { crossEntropy, mseLoss } from '../../nn/ops/nn.js';
import { stack } from '../../nn/ops/shape.js';
import { ValueError } from '../../errors.js';
import { isPythonNumber, pythonKindOf, unboxNumber, unboxNumbers } from '../json.js';
import type { Supervision, Trace } from '../tracing.js';
import type { OperationLike } from '../../ops/base.js';

/** A custom loss callback ``(output, target) => scalar tensor``. */
export type LossFunction = (output: any, target: any) => Tensor;

/** A supplied optimizer, or a factory receiving the deduplicated trainable parameters. */
export type OptimizerOption = Optimizer | ((parameters: Tensor[]) => Optimizer);

/** Trainable tensors of every binding (``operation.parameters()``), deduplicated. */
export function collectParameters(operations: Record<string, unknown>): Tensor[] {
  const result: Tensor[] = [];
  const seen = new Set<Tensor>();
  for (const operation of Object.values(operations)) {
    const method = (operation as { parameters?: unknown } | null)?.parameters;
    if (typeof method !== 'function') continue;
    for (const parameter of method.call(operation) as Iterable<Tensor>) {
      if (parameter.requiresGrad && !seen.has(parameter)) {
        seen.add(parameter);
        result.push(parameter);
      }
    }
  }
  return result;
}

export function validateOptimizer(optimizer: Optimizer, params: readonly Tensor[]): void {
  const actual = optimizer.paramGroups.flatMap((group) => group.params);
  const owned = new Set(actual);
  if (owned.size !== actual.length) throw new ValueError('Optimizer contains duplicate shared parameters');
  const expected = new Set(params);
  if (owned.size !== expected.size || [...owned].some((parameter) => !expected.has(parameter))) {
    throw new ValueError('Optimizer must own exactly the deduplicated trainable parameters');
  }
}

function asTarget(value: unknown, dtype: 'int64' | Tensor['dtype']): Tensor {
  if (value instanceof Tensor) return value.detach().to(dtype);
  return makeTensor(unboxNumbers(value) as NestedNumbers, { dtype });
}

function inferredKind(value: unknown, container: object | null = null, key: unknown = null): 'float' | 'bool' | 'int' {
  // ``torch.as_tensor`` dtype inference for the cross-entropy target check. A
  // Python float (``1.0``, or ``float(1)``) is a float even when integral.
  if (value instanceof Tensor) return isFloatingDType(value.dtype) ? 'float' : value.dtype === 'bool' ? 'bool' : 'int';
  let kind: 'float' | 'bool' | 'int' | null = null;
  const visit = (item: unknown, holder: object | null, slot: unknown): void => {
    if (Array.isArray(item)) {
      item.forEach((child, index) => visit(child, item, index));
      return;
    }
    if (typeof item === 'boolean') kind ??= 'bool';
    else if (typeof item === 'number' || isPythonNumber(item)) {
      if (!Number.isInteger(unboxNumber(item)) || pythonKindOf(item, holder, slot) === 'float') kind = 'float';
      else if (kind !== 'float') kind = 'int';
    } else {
      throw new TypeError('Cross-entropy targets must be integer indices or named labels');
    }
  };
  visit(value, container, key);
  return kind ?? 'float';
}

export interface TrainerOptions {
  optimizer?: OptimizerOption | null;
  lr?: number;
  losses?: Record<string, LossFunction> | null;
}

/**
 * Train with current bound parameters and explicit target provenance.
 *
 * External operations are treated as recorded constant boundaries. No gradient
 * is claimed across them or across ordinary code outside captured operations.
 * Custom losses are explicit in-process callbacks mapped by supervision name.
 */
export class OperationTrainer {
  readonly operations: Record<string, OperationLike>;
  readonly parameters: Tensor[];
  readonly optimizer: Optimizer;
  readonly losses: Record<string, LossFunction>;

  constructor(operations: Record<string, OperationLike>, options: TrainerOptions = {}) {
    if (operations === null || typeof operations !== 'object' || Array.isArray(operations) || !Object.keys(operations).length) {
      throw new ValueError('Trainer needs named operation bindings');
    }
    this.operations = { ...operations };
    this.parameters = collectParameters(operations);
    if (!this.parameters.length) throw new ValueError('No trainable parameters in supplied operations');
    const optimizer = options.optimizer ?? null;
    if (optimizer === null) this.optimizer = new SGD(this.parameters, { lr: options.lr ?? 0.01 });
    else if (optimizer instanceof Optimizer) this.optimizer = optimizer;
    else if (typeof optimizer === 'function') this.optimizer = optimizer(this.parameters);
    else throw new TypeError('optimizer must be an Optimizer or a factory (params) => Optimizer');
    if (!(this.optimizer instanceof Optimizer)) throw new TypeError('optimizer factory must return an Optimizer');
    validateOptimizer(this.optimizer, this.parameters);
    this.losses = { ...(options.losses ?? {}) };
    if (Object.values(this.losses).some((loss) => typeof loss !== 'function')) {
      throw new TypeError('Custom losses must map names to callables');
    }
  }

  protected loss(output: unknown, supervision: Supervision): Tensor {
    let loss: unknown;
    if (Object.prototype.hasOwnProperty.call(this.losses, supervision.loss)) {
      loss = this.losses[supervision.loss]!(output, supervision.target);
    } else {
      const logits = output !== null && typeof output === 'object' && !(output instanceof Tensor) && 'logits' in output
        ? (output as { logits: unknown }).logits : output;
      if (!(logits instanceof Tensor) || !logits.requiresGrad) {
        throw new ValueError('Supervision target is not a differentiable tensor path');
      }
      const prediction = logits;
      if (supervision.loss === 'cross_entropy') {
        let target = supervision.target;
        const labels = output !== null && typeof output === 'object' && 'labels' in output
          ? (output as { labels: unknown }).labels as readonly unknown[] | null : null;
        if (typeof target === 'string') {
          if (!Array.isArray(labels) || !labels.includes(target)) throw new ValueError('Target label is absent from prediction labels');
          target = labels.indexOf(target);
        } else if (Array.isArray(target) && target.length && typeof target[0] === 'string') {
          if (!Array.isArray(labels) || target.some((item) => !labels.includes(item))) {
            throw new ValueError('Target labels are absent from prediction labels');
          }
          target = target.map((item) => labels.indexOf(item));
        }
        const raw = target === supervision.target;
        if (inferredKind(target, raw ? supervision : null, raw ? 'target' : null) !== 'int') throw new ValueError('Cross-entropy targets must be integer indices or named labels');
        loss = crossEntropy(prediction, asTarget(target, 'int64'));
      } else if (supervision.loss === 'mse') {
        const target = asTarget(supervision.target, prediction.dtype);
        if (target.shape.length !== prediction.shape.length || target.shape.some((size, index) => size !== prediction.shape[index])) {
          throw new ValueError('MSE target shape must exactly match prediction');
        }
        loss = mseLoss(prediction, target);
      } else {
        throw new ValueError(`Unknown loss: ${supervision.loss}`);
      }
    }
    if (!(loss instanceof Tensor) || loss.ndim !== 0 || !loss.requiresGrad) {
      throw new ValueError('Loss must be a differentiable scalar tensor');
    }
    if (!loss.allFinite()) throw new ValueError('Loss must be finite');
    return loss;
  }

  /** Apply one optimizer step to one supervised experience; returns the mean loss. */
  step(session: Trace): number {
    if (!session.supervisions.length) throw new ValueError('Experience has no explicit supervision');
    const allowed = new Set<unknown>(Object.values(this.operations));
    for (const supervision of session.supervisions) {
      for (const index of session.example(supervision.output).calls) {
        if (!allowed.has(session.calls[index]!.operation)) throw new ValueError('Experience uses an operation outside Trainer bindings');
      }
    }
    validateOptimizer(this.optimizer, this.parameters);
    this.optimizer.zeroGrad();
    try {
      const losses = session.supervisions.map((supervision) => (
        this.loss(session.replay(supervision.output, { boundary: 'recorded' }), supervision)));
      const loss = stack(losses).mean();
      loss.backward();
      if (!this.parameters.some((parameter) => parameter.grad !== null)) {
        throw new ValueError('Loss has no differentiable path to bound parameters');
      }
      if (this.parameters.some((parameter) => parameter.grad !== null && !parameter.grad.allFinite())) {
        throw new ValueError('Nonfinite gradients; optimizer update rejected');
      }
      this.optimizer.step();
      return loss.detach().item();
    } catch (error) {
      this.optimizer.zeroGrad();
      throw error;
    }
  }

  /** Step through the experiences for ``epochs`` passes, in order. */
  fit(sessions: Iterable<Trace>, options: { epochs?: number } = {}): number[] {
    const epochs = options.epochs ?? 1;
    if (typeof epochs !== 'number' || !Number.isInteger(epochs) || epochs < 1) throw new ValueError('epochs must be a positive integer');
    const list = [...sessions];
    if (!list.length) throw new ValueError('No experiences supplied');
    const result: number[] = [];
    for (let epoch = 0; epoch < epochs; epoch += 1) for (const session of list) result.push(this.step(session));
    return result;
  }
}
