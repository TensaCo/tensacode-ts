/**
 * In-place parameter initializers (``torch.nn.init``).
 *
 * Each function performs the same double-precision arithmetic as PyTorch's
 * Python implementation and draws from the PyTorch-compatible generator, so a
 * seeded initialization is bitwise identical to Python's.
 */
import { noGrad } from './autograd.js';
import { erf } from './randomMath.js';
import { roundToDType } from './dtype.js';
import { getDefaultGenerator, type Generator } from './random.js';
import type { Tensor } from './tensor.js';
import { zerosLike } from './tensor.js';

export type Nonlinearity =
  | 'linear' | 'conv1d' | 'conv2d' | 'conv3d' | 'conv_transpose1d' | 'conv_transpose2d' | 'conv_transpose3d'
  | 'sigmoid' | 'tanh' | 'relu' | 'leaky_relu' | 'selu';
export type FanMode = 'fan_in' | 'fan_out';

/** ``torch.nn.init.calculate_gain``. */
export function calculateGain(nonlinearity: Nonlinearity, param: number | null = null): number {
  switch (nonlinearity) {
    case 'linear': case 'conv1d': case 'conv2d': case 'conv3d':
    case 'conv_transpose1d': case 'conv_transpose2d': case 'conv_transpose3d': case 'sigmoid':
      return 1;
    case 'tanh':
      return 5.0 / 3;
    case 'relu':
      return Math.sqrt(2.0);
    case 'leaky_relu': {
      const negativeSlope = param === null ? 0.01 : param;
      if (typeof negativeSlope !== 'number' || !Number.isFinite(negativeSlope)) {
        throw new RangeError(`negative_slope ${String(param)} not a valid number`);
      }
      return Math.sqrt(2.0 / (1 + negativeSlope * negativeSlope));
    }
    case 'selu':
      return 3.0 / 4;
    default:
      throw new RangeError(`Unsupported nonlinearity ${String(nonlinearity)}`);
  }
}

/** ``_calculate_fan_in_and_fan_out``: ``[fanIn, fanOut]``. */
export function fanInOut(tensor: Tensor): [number, number] {
  if (tensor.ndim < 2) {
    throw new RangeError('Fan in and fan out can not be computed for tensor with fewer than 2 dimensions');
  }
  const receptive = tensor.shape.slice(2).reduce((product, size) => product * size, 1);
  return [tensor.shape[1]! * receptive, tensor.shape[0]! * receptive];
}

function correctFan(tensor: Tensor, mode: FanMode): number {
  if (mode !== 'fan_in' && mode !== 'fan_out') {
    throw new RangeError(`Mode ${String(mode)} not supported, please use one of fan_in, fan_out`);
  }
  const [fanIn, fanOut] = fanInOut(tensor);
  return mode === 'fan_in' ? fanIn : fanOut;
}

let initGuard: ((tensor: Tensor) => boolean) | null = null;

/**
 * Run ``fn`` with initializers skipping tensors for which ``skip`` returns
 * true (transformers' ``guard_torch_init_functions`` for weights that were
 * loaded and marked ``_is_hf_initialized``).
 */
export function withInitGuard<T>(skip: (tensor: Tensor) => boolean, fn: () => T): T {
  const previous = initGuard;
  initGuard = skip;
  try {
    return fn();
  } finally {
    initGuard = previous;
  }
}

/** Whether the active guard protects ``tensor`` from initialization. */
export function isInitGuarded(tensor: Tensor): boolean {
  return initGuard !== null && initGuard(tensor);
}

/** ``uniform_(tensor, a, b)``. */
export function uniform_(tensor: Tensor, a = 0, b = 1, generator?: Generator): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  return noGrad(() => tensor.uniform_(a, b, generator ?? getDefaultGenerator()));
}

/** ``normal_(tensor, mean, std)``. */
export function normal_(tensor: Tensor, mean = 0, std = 1, generator?: Generator): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  return noGrad(() => tensor.normal_(mean, std, generator ?? getDefaultGenerator()));
}

/** ``constant_(tensor, val)``. */
export function constant_(tensor: Tensor, value: number): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  return noGrad(() => tensor.fill_(value));
}

/** ``ones_(tensor)``. */
export function ones_(tensor: Tensor): Tensor {
  return constant_(tensor, 1);
}

/** ``zeros_(tensor)``. */
export function zeros_(tensor: Tensor): Tensor {
  return constant_(tensor, 0);
}

/** ``eye_(tensor)`` for a 2-D tensor. */
export function eye_(tensor: Tensor): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  if (tensor.ndim !== 2) throw new RangeError('Only tensors with 2 dimensions are supported');
  return noGrad(() => {
    const values = zerosLike(tensor);
    const [rows, columns] = tensor.shape as [number, number];
    for (let index = 0; index < Math.min(rows, columns); index += 1) values.data[index * columns + index] = 1;
    return tensor.copy_(values);
  });
}

/** ``xavier_uniform_(tensor, gain)``. */
export function xavierUniform_(tensor: Tensor, gain = 1, generator?: Generator): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  const [fanIn, fanOut] = fanInOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  const a = Math.sqrt(3.0) * std;
  return uniform_(tensor, -a, a, generator);
}

/** ``xavier_normal_(tensor, gain)``. */
export function xavierNormal_(tensor: Tensor, gain = 1, generator?: Generator): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  const [fanIn, fanOut] = fanInOut(tensor);
  const std = gain * Math.sqrt(2.0 / (fanIn + fanOut));
  return normal_(tensor, 0, std, generator);
}

/** ``kaiming_uniform_(tensor, a, mode, nonlinearity)``. */
export function kaimingUniform_(
  tensor: Tensor, a = 0, mode: FanMode = 'fan_in', nonlinearity: Nonlinearity = 'leaky_relu', generator?: Generator,
): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  if (tensor.shape.includes(0)) return tensor; // PyTorch warns and leaves empty tensors untouched.
  const fan = correctFan(tensor, mode);
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  const bound = Math.sqrt(3.0) * std;
  return uniform_(tensor, -bound, bound, generator);
}

/** ``kaiming_normal_(tensor, a, mode, nonlinearity)``. */
export function kaimingNormal_(
  tensor: Tensor, a = 0, mode: FanMode = 'fan_in', nonlinearity: Nonlinearity = 'leaky_relu', generator?: Generator,
): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  if (tensor.shape.includes(0)) return tensor;
  const fan = correctFan(tensor, mode);
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  return normal_(tensor, 0, std, generator);
}

/** ``kaiming_uniform_(a=sqrt(5))``: the default ``nn.Linear``/``nn.Conv2d`` weight init. */
export function kaimingUniformDefault_(tensor: Tensor, generator?: Generator): Tensor {
  return kaimingUniform_(tensor, Math.sqrt(5), 'fan_in', 'leaky_relu', generator);
}

function normCdf(x: number): number {
  return (1.0 + erf(x / Math.sqrt(2.0))) / 2.0;
}

/** The value ``tensor.new_tensor(x).item()`` reads back (``x`` rounded to the tensor's dtype). */
function asDType(tensor: Tensor, value: number): number {
  if (tensor.dtype === 'float64') return value;
  return roundToDType(tensor.dtype, Math.fround(value));
}

/**
 * ``trunc_normal_(tensor, mean, std, a, b)``: PyTorch 2.14's sampler. When
 * [a, b] holds more than 30% of the mass it redraws out-of-range values from
 * ``normal_`` (a whole-tensor draw per round); otherwise it uses uniform
 * proposals with a log-density acceptance test.
 */
export function truncNormal_(tensor: Tensor, mean = 0, std = 1, a = -2, b = 2, generator?: Generator): Tensor {
  if (isInitGuarded(tensor)) return tensor;
  const source = generator ?? getDefaultGenerator();
  return noGrad(() => {
    const p = normCdf((b - mean) / std) - normCdf((a - mean) / std);
    if (p > 0.3) {
      const low = asDType(tensor, a);
      const high = asDType(tensor, b);
      const result = zerosLike(tensor).normal_(mean, std, source);
      const data = result.data;
      const outOfRange = () => data.some((value) => value < low || value > high);
      while (outOfRange()) {
        const fresh = zerosLike(tensor).normal_(mean, std, source).data;
        for (let index = 0; index < data.length; index += 1) {
          if (data[index]! < low || data[index]! > high) data[index] = fresh[index]!;
        }
      }
      return tensor.copy_(result);
    }
    const mode = Math.max(a, Math.min(mean, b));
    const logPeak = -0.5 * ((mode - mean) / std) ** 2;
    const candidates = zerosLike(tensor);
    const accept = zerosLike(tensor);
    const logDensity = (values: Tensor): Tensor => values.sub(mean).div(std).pow(2).mul(-0.5).sub(logPeak);
    tensor.uniform_(a, b, source);
    let density = logDensity(tensor);
    let pending = accept.uniform_(0, 1, source).log().gt(density);
    if (!pending.data.some(Boolean)) return tensor;
    let result = tensor.clone();
    for (;;) {
      candidates.uniform_(a, b, source);
      const pendingMask = pending.data;
      const resultData = result.data;
      for (let index = 0; index < resultData.length; index += 1) if (pendingMask[index]) resultData[index] = candidates.data[index]!;
      density = logDensity(candidates);
      const draw = accept.uniform_(0, 1, source).log().gt(density);
      const next = pending.clone();
      for (let index = 0; index < next.data.length; index += 1) if (pendingMask[index]) next.data[index] = draw.data[index]!;
      pending = next;
      if (!pending.data.some(Boolean)) break;
      result = result.clone();
    }
    return tensor.copy_(result);
  });
}

/** Alias of {@link truncNormal_}. */
export const truncatedNormal_ = truncNormal_;
