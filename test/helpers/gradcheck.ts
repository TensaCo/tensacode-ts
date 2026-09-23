import { expect } from 'vitest';
import { Tensor, noGrad, tensor } from '../../src/nn/index.js';

/**
 * Compare analytic gradients of ``sum(fn(...inputs) * probe)`` with central
 * finite differences. Inputs should be float64 for tight tolerances.
 */
export function gradcheck(
  fn: (...inputs: Tensor[]) => Tensor,
  inputs: Tensor[],
  options: { eps?: number; atol?: number; rtol?: number } = {},
): void {
  const eps = options.eps ?? 1e-6;
  const atol = options.atol ?? 1e-5;
  const rtol = options.rtol ?? 1e-4;
  for (const input of inputs) {
    input.grad = null;
    if (input.isFloatingPoint) input.requiresGrad = true;
  }
  const output = fn(...inputs);
  // A deterministic, non-uniform probe so every output element matters.
  const probeValues = Array.from({ length: output.numel }, (_, index) => Math.sin(index * 1.37 + 0.5) + 1.1);
  const probe = tensor(probeValues, { dtype: output.dtype, shape: output.shape });
  output.mul(probe).sum().backward();
  inputs.forEach((input, inputIndex) => {
    if (!input.isFloatingPoint) return;
    const analytic = input.grad ? Array.from(input.grad.data) : new Array<number>(input.numel).fill(0);
    const numeric: number[] = [];
    noGrad(() => {
      for (let index = 0; index < input.numel; index += 1) {
        const original = input.data[index]!;
        input.data[index] = original + eps;
        const plus = fn(...inputs).mul(probe).sum().item();
        input.data[index] = original - eps;
        const minus = fn(...inputs).mul(probe).sum().item();
        input.data[index] = original;
        numeric.push((plus - minus) / (2 * eps));
      }
    });
    analytic.forEach((value, index) => {
      const expected = numeric[index]!;
      const tolerance = atol + rtol * Math.abs(expected);
      if (!(Math.abs(value - expected) <= tolerance)) {
        expect.fail(`gradient mismatch for input ${inputIndex} element ${index}: analytic ${value}, numeric ${expected}`);
      }
    });
  });
}

export function randomTensor(shape: number[], seed: number, scale = 1, dtype: 'float64' | 'float32' = 'float64'): Tensor {
  const count = shape.reduce((a, b) => a * b, 1);
  let state = seed >>> 0 || 1;
  const values = Array.from({ length: count }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return ((state / 2 ** 32) * 2 - 1) * scale;
  });
  return tensor(values, { dtype, shape });
}

export function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, atol = 1e-5, rtol = 1e-5): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    const a = actual[index]!;
    const e = expected[index]!;
    if (!(Math.abs(a - e) <= atol + rtol * Math.abs(e))) {
      expect.fail(`element ${index}: ${a} != ${e} (atol ${atol}, rtol ${rtol})`);
    }
  }
}
