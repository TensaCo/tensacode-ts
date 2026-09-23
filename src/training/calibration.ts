/**
 * Empirical calibration on explicit held-out scores and reviewed labels
 * (Python ``tensorcode/training/calibration.py``). These utilities never fit
 * source-model weights. Calibration describes the supplied sample; it is neither
 * a statistical guarantee nor general epistemic probability. Recalibrate after
 * changing source weights or the deployment distribution. FOUNDATION-OWNED.
 */
import { Module } from '../nn/module.js';
import { Tensor, scalar } from '../nn/tensor.js';
import { noGrad } from '../nn/autograd.js';
import { crossEntropy } from '../nn/ops/nn.js';
import { ValueError } from '../errors.js';
import { type JsonObject } from '../_internal/json.js';


function validated(logits: Tensor, labels: Tensor): [Tensor, Tensor] {
  if (!(logits instanceof Tensor) || !logits.isFloatingPoint) throw new TypeError('logits must be a floating-point tensor');
  if (logits.ndim !== 2 || logits.shape[0] === 0 || logits.shape[1]! < 2) {
    throw new ValueError('logits must have nonempty shape [samples, classes >= 2]');
  }
  if (!logits.allFinite()) throw new ValueError('logits must be finite');
  if (!(labels instanceof Tensor) || (labels.dtype !== 'int32' && labels.dtype !== 'int64')) {
    throw new TypeError('labels must be integer class indices');
  }
  if (labels.ndim !== 1 || labels.shape[0] !== logits.shape[0]) throw new ValueError('labels must have one class index per sample');
  if (labels.toArray().some((label) => label < 0 || label >= logits.shape[1]!)) throw new ValueError('label index outside class range');
  return [logits.detach().to('float64'), labels.detach().to('int64')];
}

export interface CalibrationMetrics {
  nll: number;
  brier: number;
  accuracy: number;
  ece: number;
  sample_count: number;
}

/** Empirical NLL, multiclass Brier, accuracy and equal-width ECE. */
export function evaluateCalibration(logits: Tensor, labels: Tensor, options: { nBins?: number } = {}): CalibrationMetrics {
  const bins = options.nBins ?? 15;
  if (!Number.isInteger(bins) || bins < 1) throw new ValueError('n_bins must be a positive integer');
  const [scores, targets] = validated(logits, labels);
  return noGrad(() => {
    const probabilities = scores.softmax(-1);
    const [samples, classes] = scores.shape as [number, number];
    const target = targets.toArray();
    const rows = probabilities.toArray();
    const confidence: number[] = [];
    const correct: number[] = [];
    let brier = 0;
    for (let row = 0; row < samples; row += 1) {
      let best = 0;
      for (let column = 1; column < classes; column += 1) if (rows[row * classes + column]! > rows[row * classes + best]!) best = column;
      confidence.push(rows[row * classes + best]!);
      correct.push(best === target[row] ? 1 : 0);
      for (let column = 0; column < classes; column += 1) {
        const difference = rows[row * classes + column]! - (column === target[row] ? 1 : 0);
        brier += difference * difference;
      }
    }
    let ece = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const members = confidence.map((value, index) => [value, index] as const)
        .filter(([value]) => Math.min(Math.trunc(value * bins), bins - 1) === bin);
      if (!members.length) continue;
      const meanConfidence = members.reduce((total, [value]) => total + value, 0) / members.length;
      const meanCorrect = members.reduce((total, [, index]) => total + correct[index]!, 0) / members.length;
      ece += (members.length / samples) * Math.abs(meanConfidence - meanCorrect);
    }
    return {
      nll: crossEntropy(scores, targets).item(), brier: brier / samples,
      accuracy: correct.reduce((a, b) => a + b, 0) / samples, ece, sample_count: samples,
    };
  });
}

export interface CalibrationReport {
  before: CalibrationMetrics;
  after: CalibrationMetrics;
  temperature: number;
}

/**
 * Positive bounded scalar temperature fitted only to held-out logits. Forward
 * preserves argmax and input gradients. ``calibrated`` is false until ``fit``;
 * buffers persist in ordinary state dicts.
 */
export class TemperatureCalibration extends Module {
  static override readonly qualifiedName: string = 'tensorcode.training.calibration.TemperatureCalibration';
  readonly minTemperature: number;
  readonly maxTemperature: number;
  readonly iterations: number;
  temperature: Tensor;
  calibrated: Tensor;
  sampleCount: Tensor;

  constructor(options: { minTemperature?: number; maxTemperature?: number; iterations?: number } = {}) {
    super();
    const unknown = Object.keys(options ?? {}).filter((key) => !['minTemperature', 'maxTemperature', 'iterations'].includes(key));
    // Python raises TypeError for unexpected keyword arguments.
    if (unknown.length) throw new TypeError(`TemperatureCalibration got unexpected options: ${unknown.join(', ')}`);
    const min = options.minTemperature ?? 0.05;
    const max = options.maxTemperature ?? 100;
    const iterations = options.iterations ?? 64;
    if (!(Number.isFinite(min) && Number.isFinite(max) && min > 0 && min <= 1 && max >= 1)) {
      throw new ValueError('temperature bounds must be finite, positive and contain 1');
    }
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 256) throw new ValueError('iterations must be an integer between 1 and 256');
    this.minTemperature = min;
    this.maxTemperature = max;
    this.iterations = iterations;
    this.temperature = this.registerBuffer('temperature', scalar(1, 'float64'));
    this.calibrated = this.registerBuffer('calibrated', scalar(0, 'bool'));
    this.sampleCount = this.registerBuffer('sample_count', scalar(0, 'int64'));
  }

  protected override onRegistryChange(): void {
    this.temperature = this.getBuffer('temperature') ?? this.temperature;
    this.calibrated = this.getBuffer('calibrated') ?? this.calibrated;
    this.sampleCount = this.getBuffer('sample_count') ?? this.sampleCount;
  }

  /** Constructor options as Python-compatible JSON. */
  configuration(): JsonObject {
    return { min_temperature: this.minTemperature, max_temperature: this.maxTemperature, iterations: this.iterations };
  }

  override configurationAttributes(): Record<string, unknown> {
    return { min_temperature: this.minTemperature, max_temperature: this.maxTemperature, iterations: this.iterations };
  }

  get isCalibrated(): boolean {
    return this.calibrated.item() !== 0;
  }

  /** Divide logits by the fitted temperature (identity until ``fit``). */
  forward(logits: Tensor): Tensor {
    if (!(logits instanceof Tensor) || !logits.isFloatingPoint) throw new TypeError('logits must be a floating-point tensor');
    if (logits.ndim < 1 || logits.shape[logits.ndim - 1]! < 2 || !logits.allFinite()) throw new ValueError('logits must be finite with at least two classes');
    return logits.div(this.temperature.to(logits.dtype));
  }

  /** Minimize held-out NLL by deterministic convex inverse-temperature search. */
  fit(logits: Tensor, labels: Tensor): CalibrationReport {
    return noGrad(() => {
      const [scores, targets] = validated(logits, labels);
      const before = evaluateCalibration(scores, targets);
      let lower = 1 / this.maxTemperature;
      let upper = 1 / this.minTemperature;
      const targetScores = scores.gather(1, targets.unsqueeze(1)).squeeze(1);
      for (let step = 0; step < this.iterations; step += 1) {
        const midpoint = (lower + upper) / 2;
        const derivative = scores.mul(midpoint).softmax(-1).mul(scores).sum(-1).sub(targetScores).mean().item();
        if (derivative > 0) upper = midpoint;
        else lower = midpoint;
      }
      // Include the identity and exact bounds so fit never worsens identity NLL.
      const candidates = [1, this.minTemperature, this.maxTemperature, 1 / ((lower + upper) / 2)];
      let temperature = candidates[0]!;
      let best = Number.POSITIVE_INFINITY;
      for (const candidate of candidates) {
        const nll = crossEntropy(scores.div(candidate), targets).item();
        if (nll < best) {
          best = nll;
          temperature = candidate;
        }
      }
      const after = evaluateCalibration(scores.div(temperature), targets);
      if (![after.nll, after.brier, after.ece].every(Number.isFinite)) throw new ValueError('calibration computation produced nonfinite metrics');
      this.temperature.fill_(temperature);
      this.calibrated.fill_(1);
      this.sampleCount.fill_(targets.shape[0]!);
      return { before, after, temperature };
    });
  }
}

export interface ThresholdReport {
  threshold: number | null;
  accepted_count: number;
  sample_count: number;
  coverage: number;
  error: number | null;
  max_error: number;
  empirical: true;
}

/**
 * Select maximal empirical coverage subject to a supplied sample error limit.
 * ``scores`` are confidences in [0, 1]; ``labels`` explicit boolean correctness.
 * Accept scores >= threshold, or abstain on all when the threshold is null.
 */
export function fitThreshold(scores: Tensor, labels: Tensor, options: { maxError: number }): ThresholdReport {
  const maxError = options.maxError;
  if (typeof maxError !== 'number' || !Number.isFinite(maxError) || maxError < 0 || maxError > 1) {
    throw new ValueError('max_error must be finite and in [0, 1]');
  }
  if (!(scores instanceof Tensor) || !scores.isFloatingPoint) throw new TypeError('scores must be floating-point confidence values');
  const values = scores.toArray();
  if (scores.ndim !== 1 || !values.length || !scores.allFinite() || values.some((value) => value < 0 || value > 1)) {
    throw new ValueError('scores must be nonempty finite confidence values in [0, 1]');
  }
  if (!(labels instanceof Tensor) || labels.dtype !== 'bool') throw new TypeError('labels must be explicit boolean correctness labels');
  if (labels.ndim !== 1 || labels.shape[0] !== values.length) throw new ValueError('one correctness label is required per confidence score');
  const correct = labels.toArray();
  const order = values.map((_, index) => index).sort((a, b) => (values[b]! - values[a]!) || (a - b));
  const ordered = order.map((index) => values[index]!);
  const errors: number[] = [];
  let running = 0;
  for (const index of order) {
    running += correct[index] ? 0 : 1;
    errors.push(running);
  }
  let count = 0;
  for (let index = 0; index < ordered.length; index += 1) {
    if (index + 1 < ordered.length && ordered[index] === ordered[index + 1]) continue;
    if (errors[index]! / (index + 1) <= maxError) count = index + 1;
  }
  return {
    threshold: count ? ordered[count - 1]! : null, accepted_count: count, sample_count: values.length,
    coverage: count / values.length, error: count ? errors[count - 1]! / count : null, max_error: maxError, empirical: true,
  };
}
