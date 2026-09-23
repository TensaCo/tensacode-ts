import { describe, expect, it } from 'vitest';
import { TemperatureCalibration, evaluateCalibration, fitThreshold } from '../../src/training/calibration.js';
import { tensor, ones, type Tensor } from '../../src/nn/index.js';

describe('calibration', () => {
  it('temperature fit improves NLL, preserves predictions and restores from state', () => {
    const logits = tensor([[12, 0], [0, 12], [12, 0], [0, 12]], { dtype: 'float64' });
    const labels = tensor([0, 1, 1, 0], { dtype: 'int64' });
    const calibration = new TemperatureCalibration();
    expect(calibration.isCalibrated).toBe(false);
    const report = calibration.fit(logits, labels);
    expect(report.after.nll).toBeLessThan(report.before.nll);
    expect(calibration.forward(logits).argmax(-1).toArray()).toEqual(logits.argmax(-1).toArray());
    expect(calibration.sampleCount.item()).toBe(4);
    expect(calibration.isCalibrated).toBe(true);
    const config = calibration.configuration();
    const restored = new TemperatureCalibration({
      minTemperature: config.min_temperature as number, maxTemperature: config.max_temperature as number, iterations: config.iterations as number,
    });
    restored.loadStateDict(calibration.stateDict());
    expect(restored.forward(logits).toArray()).toEqual(calibration.forward(logits).toArray());
    expect(restored.isCalibrated).toBe(true);
    const repeated = new TemperatureCalibration();
    repeated.fit(logits, labels);
    expect(repeated.temperature.item()).toBe(calibration.temperature.item());
  });

  const invalid: [string, Tensor, Tensor][] = [
    ['nonfinite', tensor([[Number.NaN, 0]]), tensor([0], { dtype: 'int64' })],
    ['empty', ones([0, 2]), tensor([], { dtype: 'int64' })],
    ['count', ones([2, 2]), tensor([0], { dtype: 'int64' })],
    ['float labels', ones([2, 2]), tensor([0, 1], { dtype: 'float32' })],
    ['range', ones([2, 2]), tensor([0, 2], { dtype: 'int64' })],
  ];
  for (const [name, logits, labels] of invalid) {
    it(`invalid fit (${name}) does not mutate state`, () => {
      const calibration = new TemperatureCalibration();
      const before = [...calibration.stateDict().values()].map((value) => value.toArray());
      expect(() => calibration.fit(logits, labels)).toThrow();
      expect([...calibration.stateDict().values()].map((value) => value.toArray())).toEqual(before);
    });
  }

  it('metrics and bin validation', () => {
    const result = evaluateCalibration(tensor([[0, 0], [0, 0]]), tensor([0, 1], { dtype: 'int64' }), { nBins: 2 });
    expect(result.nll).toBeCloseTo(0.69314718, 6);
    expect(result.brier).toBeCloseTo(0.5, 9);
    expect(result.accuracy).toBeCloseTo(0.5, 9);
    expect(result.ece).toBeCloseTo(0, 9);
    expect(() => evaluateCalibration(tensor([[0, 0], [0, 0]]), tensor([0, 1], { dtype: 'int64' }), { nBins: 0 })).toThrow();
  });

  it('threshold selects the largest empirical coverage without splitting ties', () => {
    const report = fitThreshold(tensor([0.9, 0.8, 0.8, 0.6]), tensor([true, true, false, false]), { maxError: 0 });
    expect(report.accepted_count).toBe(1);
    expect(report.coverage).toBe(0.25);
    expect(report.error).toBe(0);
    expect(report.sample_count).toBe(4);
    const empty = fitThreshold(tensor([1]), tensor([false]), { maxError: 0 });
    expect(empty.threshold).toBeNull();
    expect(empty.accepted_count).toBe(0);
    expect(empty.error).toBeNull();
    expect(() => fitThreshold(tensor([1.1]), tensor([true]), { maxError: 0 })).toThrow();
  });
});
