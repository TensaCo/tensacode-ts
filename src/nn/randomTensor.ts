/**
 * Tensor-level random functions (``torch.randperm``, ``torch.multinomial``,
 * ``torch.bernoulli``, ``torch.normal``, ``Tensor.exponential_``) and the
 * default generator's state bytes (``torch.get_rng_state``).
 * All draw from PyTorch's CPU generator in ``random.ts``.
 */
import { noGrad } from './autograd.js';
import { isFloatingDType, type DType } from './dtype.js';
import {
  type Generator, RNG_STATE_SIZE, fillBernoulli, fillBernoulliTensor, fillExponential, fillNormal, getDefaultGenerator,
  multinomialValues, randpermValues,
} from './random.js';
import type { Shape } from './shape.js';
import { Tensor, fromStorage, zeros, type RandomOptions } from './tensor.js';

/** The default generator's state: the 5056 bytes of ``torch.get_rng_state()``. */
export function getRngState(): Uint8Array {
  return getDefaultGenerator().getState();
}

/** Restore the default generator from ``torch.get_rng_state()`` bytes or a ``uint8`` tensor (``torch.set_rng_state``). */
export function setRngState(state: Tensor | Uint8Array): void {
  getDefaultGenerator().setState(state);
}

/** RNG state bytes as the ``uint8`` tensor ``torch.get_rng_state()`` returns. */
export function rngStateTensor(state: Uint8Array = getRngState()): Tensor {
  if (state.length !== RNG_STATE_SIZE) throw new RangeError(`RNG state must have ${RNG_STATE_SIZE} bytes`);
  return fromStorage(Float64Array.from(state), [RNG_STATE_SIZE], 'uint8');
}

/** A random permutation of ``0 .. n - 1`` (``torch.randperm``). */
export function randperm(n: number, options: RandomOptions = {}): Tensor {
  const generator = options.generator ?? getDefaultGenerator();
  const values = randpermValues(n, generator);
  const result = zeros([n], { dtype: options.dtype ?? 'int64' });
  result.data.set(values);
  return result;
}

/**
 * ``torch.multinomial(input, numSamples, replacement)`` over a 1-D or 2-D
 * probability tensor; returns int64 category indices.
 */
export function multinomial(
  input: Tensor, numSamples: number, options: { replacement?: boolean; generator?: Generator } = {},
): Tensor {
  if (input.ndim < 1 || input.ndim > 2) throw new RangeError('prob_dist must be 1 or 2 dim');
  if (!isFloatingDType(input.dtype)) {
    throw new TypeError(`multinomial only supports floating-point dtypes for input, got: ${input.dtype}`);
  }
  const categories = input.shape[input.ndim - 1]!;
  const rows = input.ndim === 2 ? input.shape[0]! : 1;
  const values = multinomialValues(
    input.data, input.dtype, rows, categories, numSamples, options.replacement ?? false,
    options.generator ?? getDefaultGenerator(),
  );
  const shape: Shape = input.ndim === 2 ? [rows, numSamples] : [numSamples];
  const result = zeros(shape, { dtype: 'int64' });
  result.data.set(values);
  return result;
}

/**
 * ``torch.bernoulli``: with a probability tensor, one draw per element; with a
 * scalar ``p``, draws shaped like ``input``.
 */
export function bernoulli(input: Tensor, p?: number, options: { generator?: Generator } = {}): Tensor {
  const generator = options.generator ?? getDefaultGenerator();
  const result = zeros(input.shape, { dtype: input.dtype });
  if (p === undefined) fillBernoulliTensor(result.data, input.data, input.dtype, generator);
  else fillBernoulli(result.data, result.dtype, p, generator);
  return result;
}

/** In-place ``Tensor.bernoulli_(p)`` with a scalar or tensor probability. */
export function bernoulli_(tensor: Tensor, p: number | Tensor = 0.5, options: { generator?: Generator } = {}): Tensor {
  const generator = options.generator ?? getDefaultGenerator();
  noGrad(() => {
    const values = zeros(tensor.shape, { dtype: tensor.dtype });
    if (typeof p === 'number') {
      fillBernoulli(values.data, tensor.dtype, p, generator);
    } else {
      const probabilities = p.shape.length === tensor.shape.length && p.shape.every((size, index) => size === tensor.shape[index])
        ? p : p.expand(tensor.shape);
      fillBernoulliTensor(values.data, probabilities.data, probabilities.dtype, generator);
    }
    tensor.copy_(values);
  });
  return tensor;
}

/** In-place ``Tensor.exponential_(lambd)``. */
export function exponential_(tensor: Tensor, lambd = 1, options: { generator?: Generator } = {}): Tensor {
  const generator = options.generator ?? getDefaultGenerator();
  noGrad(() => {
    const values = zeros(tensor.shape, { dtype: tensor.dtype });
    fillExponential(values.data, tensor.dtype, lambd, generator);
    tensor.copy_(values);
  });
  return tensor;
}

/** ``torch.normal(mean, std, size)`` with scalar mean and standard deviation. */
export function normal(mean: number, std: number, shape: Shape, options: RandomOptions = {}): Tensor {
  const dtype: DType = options.dtype ?? 'float32';
  const result = zeros(shape, { dtype });
  fillNormal(result.data, dtype, mean, std, options.generator ?? getDefaultGenerator());
  if (options.requiresGrad) result.requiresGrad = true;
  return result;
}
