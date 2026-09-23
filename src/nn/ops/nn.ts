/** Neural-network primitives: activations, normalization, lookup, convolution and losses. */
import { allocate, isFloatingDType, type DType } from '../dtype.js';
import { getDefaultGenerator, type Generator } from '../random.js';
import { formatShape, shapesEqual } from '../shape.js';
import { Tensor, attachGrad, fromStorage, tensor } from '../tensor.js';
import {
  abs, add, cast, clamp, div, erfValue, exp, log, mapUnary, maximum, mul, neg, relu, sigmoid, sqrt, sub, tanh,
} from './elementwise.js';
import { gather, transpose, unsqueeze, squeeze } from './shape.js';
import { logSoftmax, mean, softmax, sum } from './reduce.js';
import { matmul } from './linalg.js';

const SQRT_2_OVER_PI = 0.7978845608028654;
const INV_SQRT_2 = 0.7071067811865476;
const INV_SQRT_2PI = 0.3989422804014327;

// ------------------------------------------------------------------ activations

/** GELU; ``approximate='tanh'`` matches HF ``gelu_new``/``gelu_pytorch_tanh``. */
export function gelu(x: Tensor, approximate: 'none' | 'tanh' = 'none'): Tensor {
  const floatX = isFloatingDType(x.dtype) ? x : cast(x, 'float32');
  if (approximate === 'tanh') {
    const result = mapUnary(floatX, (v) => 0.5 * v * (1 + Math.tanh(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v))));
    return attachGrad(result, [floatX], (grad) => [
      mul(grad, mapUnary(floatX, (v) => {
        const inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
        const t = Math.tanh(inner);
        const derivative = SQRT_2_OVER_PI * (1 + 3 * 0.044715 * v * v);
        return 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * derivative;
      }, grad.dtype)),
    ], 'gelu_tanh');
  }
  const result = mapUnary(floatX, (v) => 0.5 * v * (1 + erfValue(v * INV_SQRT_2)));
  return attachGrad(result, [floatX], (grad) => [
    mul(grad, mapUnary(floatX, (v) => 0.5 * (1 + erfValue(v * INV_SQRT_2)) + v * INV_SQRT_2PI * Math.exp(-0.5 * v * v), grad.dtype)),
  ], 'gelu');
}

export function silu(x: Tensor): Tensor {
  return mul(x, sigmoid(x));
}

export function quickGelu(x: Tensor): Tensor {
  return mul(x, sigmoid(mul(x, 1.702)));
}

export function leakyRelu(x: Tensor, slope = 0.01): Tensor {
  return add(relu(x), mul(relu(neg(x)), -slope));
}

export function softplus(x: Tensor): Tensor {
  return add(relu(x), log(add(exp(neg(abs(x))), 1)));
}

export type ActivationName =
  | 'gelu' | 'gelu_new' | 'gelu_pytorch_tanh' | 'gelu_fast' | 'gelu_python' | 'quick_gelu'
  | 'relu' | 'silu' | 'swish' | 'tanh' | 'sigmoid' | 'linear' | 'identity';

/** Hugging Face ``ACT2FN`` activation lookup. */
export function activation(name: string): (x: Tensor) => Tensor {
  switch (name) {
    case 'gelu': case 'gelu_python': return (x) => gelu(x, 'none');
    case 'gelu_new': case 'gelu_pytorch_tanh': case 'gelu_fast': return (x) => gelu(x, 'tanh');
    case 'quick_gelu': return quickGelu;
    case 'relu': return relu;
    case 'silu': case 'swish': return silu;
    case 'tanh': return tanh;
    case 'sigmoid': return sigmoid;
    case 'linear': case 'identity': return (x) => x;
    default: throw new Error(`unsupported activation function: ${name}`);
  }
}

// ------------------------------------------------------------------ regularization

/** Inverted dropout; the identity when ``training`` is false or ``p`` is 0. */
export function dropout(x: Tensor, p: number, training: boolean, generator: Generator = getDefaultGenerator()): Tensor {
  if (!(p >= 0 && p <= 1)) throw new RangeError('dropout probability must be in [0, 1]');
  if (!training || p === 0) return x;
  if (p === 1) return mul(x, 0);
  const scale = 1 / (1 - p);
  const maskData = allocate(x.dtype, x.numel);
  for (let index = 0; index < maskData.length; index += 1) maskData[index] = generator.random() >= p ? scale : 0;
  const mask = fromStorage(maskData, x.shape, x.dtype);
  return mul(x, mask);
}

// ------------------------------------------------------------------ normalization

/** Layer normalization over the last dimension (computed in float64). */
export function layerNorm(x: Tensor, weight: Tensor | null, bias: Tensor | null, eps = 1e-5): Tensor {
  const width = x.shape[x.ndim - 1]!;
  if (weight && weight.numel !== width) throw new RangeError('layerNorm weight must match the last dimension');
  const dtype: DType = isFloatingDType(x.dtype) ? x.dtype : 'float32';
  const rows = x.numel / width;
  const data = x.data;
  const normalized = new Float64Array(x.numel);
  const inverse = new Float64Array(rows);
  const out = allocate(dtype, x.numel);
  const w = weight?.data;
  const b = bias?.data;
  for (let r = 0; r < rows; r += 1) {
    const base = r * width;
    let total = 0;
    for (let i = 0; i < width; i += 1) total += data[base + i]!;
    const average = total / width;
    let squares = 0;
    for (let i = 0; i < width; i += 1) {
      const centered = data[base + i]! - average;
      squares += centered * centered;
    }
    const inv = 1 / Math.sqrt(squares / width + eps);
    inverse[r] = inv;
    for (let i = 0; i < width; i += 1) {
      const value = (data[base + i]! - average) * inv;
      normalized[base + i] = value;
      out[base + i] = value * (w ? w[i]! : 1) + (b ? b[i]! : 0);
    }
  }
  const result = fromStorage(out, x.shape, dtype);
  return attachGrad(result, [x, weight, bias], (grad) => {
    const g = grad.data;
    let gradX: Tensor | null = null;
    if (x.requiresGrad) {
      const gx = allocate(grad.dtype, x.numel);
      for (let r = 0; r < rows; r += 1) {
        const base = r * width;
        let sumG = 0;
        let sumGX = 0;
        for (let i = 0; i < width; i += 1) {
          const gi = g[base + i]! * (w ? w[i]! : 1);
          sumG += gi;
          sumGX += gi * normalized[base + i]!;
        }
        const inv = inverse[r]!;
        for (let i = 0; i < width; i += 1) {
          const gi = g[base + i]! * (w ? w[i]! : 1);
          gx[base + i] = (inv / width) * (width * gi - sumG - normalized[base + i]! * sumGX);
        }
      }
      gradX = fromStorage(gx, x.shape, grad.dtype);
    }
    let gradW: Tensor | null = null;
    let gradB: Tensor | null = null;
    if (weight?.requiresGrad) {
      const gw = new Float64Array(width);
      for (let r = 0; r < rows; r += 1) for (let i = 0; i < width; i += 1) gw[i]! += g[r * width + i]! * normalized[r * width + i]!;
      const storage = allocate(weight.dtype, width);
      storage.set(gw);
      gradW = fromStorage(storage, weight.shape, weight.dtype);
    }
    if (bias?.requiresGrad) {
      const gb = new Float64Array(width);
      for (let r = 0; r < rows; r += 1) for (let i = 0; i < width; i += 1) gb[i]! += g[r * width + i]!;
      const storage = allocate(bias.dtype, width);
      storage.set(gb);
      gradB = fromStorage(storage, bias.shape, bias.dtype);
    }
    return [gradX, gradW, gradB];
  }, 'layerNorm');
}

/** Root-mean-square normalization (T5 layer norm): ``weight * x / sqrt(mean(x^2) + eps)``. */
export function rmsNorm(x: Tensor, weight: Tensor | null, eps = 1e-6): Tensor {
  const variance = mean(mul(x, x), -1, true);
  const normalized = mul(x, div(1, sqrt(add(variance, eps))));
  return weight ? mul(weight, normalized) : normalized;
}

/** L2-normalize along ``dim`` (``torch.nn.functional.normalize``). */
export function normalize(x: Tensor, p = 2, dim = -1, eps = 1e-12): Tensor {
  if (p !== 2) throw new RangeError('normalize supports p=2');
  const norms = sqrt(sum(mul(x, x), dim, true));
  return div(x, clamp(norms, eps, null));
}

export function cosineSimilarity(a: Tensor, b: Tensor, dim = -1, eps = 1e-8): Tensor {
  const dot = sum(mul(a, b), dim);
  const na = sqrt(sum(mul(a, a), dim));
  const nb = sqrt(sum(mul(b, b), dim));
  return div(dot, maximum(mul(na, nb), eps));
}

// ------------------------------------------------------------------ lookup

function indexList(indices: Tensor | readonly number[]): { values: ArrayLike<number>; shape: number[] } {
  if (indices instanceof Tensor) return { values: indices.data, shape: [...indices.shape] };
  return { values: indices, shape: [indices.length] };
}

/** Row lookup ``weight[indices]``; ``paddingIdx`` rows receive no gradient. */
export function embedding(indices: Tensor | readonly number[], weight: Tensor, paddingIdx: number | null = null): Tensor {
  const { values, shape } = indexList(indices);
  const [rows, width] = weight.shape as [number, number];
  const out = allocate(weight.dtype, values.length * width);
  const w = weight.data;
  for (let position = 0; position < values.length; position += 1) {
    const row = values[position]!;
    if (!Number.isInteger(row) || row < 0 || row >= rows) throw new RangeError(`embedding index ${row} out of range [0, ${rows})`);
    const base = row * width;
    for (let i = 0; i < width; i += 1) out[position * width + i] = w[base + i]!;
  }
  const result = fromStorage(out, [...shape, width], weight.dtype);
  return attachGrad(result, [weight], (grad) => {
    const gw = allocate(grad.dtype, weight.numel);
    const g = grad.data;
    for (let position = 0; position < values.length; position += 1) {
      const row = values[position]!;
      if (paddingIdx !== null && row === paddingIdx) continue;
      const base = row * width;
      for (let i = 0; i < width; i += 1) gw[base + i]! += g[position * width + i]!;
    }
    return [fromStorage(gw, weight.shape, grad.dtype)];
  }, 'embedding');
}

/** Mean-pooled bags of rows (``nn.EmbeddingBag(mode='mean')``) delimited by ``offsets``. */
export function embeddingBag(indices: readonly number[], offsets: readonly number[], weight: Tensor): Tensor {
  const [rows, width] = weight.shape as [number, number];
  const bags = offsets.length;
  const out = allocate(weight.dtype, bags * width);
  const w = weight.data;
  const bounds = offsets.map((start, bag) => [start, bag + 1 < bags ? offsets[bag + 1]! : indices.length] as const);
  bounds.forEach(([start, end], bag) => {
    if (end < start) throw new RangeError('embeddingBag offsets must be nondecreasing');
    const count = end - start;
    for (let position = start; position < end; position += 1) {
      const row = indices[position]!;
      if (!Number.isInteger(row) || row < 0 || row >= rows) throw new RangeError(`embedding index ${row} out of range`);
      for (let i = 0; i < width; i += 1) out[bag * width + i]! += w[row * width + i]! / count;
    }
  });
  const result = fromStorage(out, [bags, width], weight.dtype);
  return attachGrad(result, [weight], (grad) => {
    const gw = allocate(grad.dtype, weight.numel);
    const g = grad.data;
    bounds.forEach(([start, end], bag) => {
      const count = end - start;
      for (let position = start; position < end; position += 1) {
        const row = indices[position]!;
        for (let i = 0; i < width; i += 1) gw[row * width + i]! += g[bag * width + i]! / count;
      }
    });
    return [fromStorage(gw, weight.shape, grad.dtype)];
  }, 'embeddingBag');
}

export function oneHot(indices: Tensor, numClasses: number): Tensor {
  const values = indices.data;
  const out = new Float64Array(values.length * numClasses);
  for (let position = 0; position < values.length; position += 1) {
    const value = values[position]!;
    if (!Number.isInteger(value) || value < 0 || value >= numClasses) throw new RangeError('oneHot index out of range');
    out[position * numClasses + value] = 1;
  }
  return fromStorage(out, [...indices.shape, numClasses], 'int64');
}

// ------------------------------------------------------------------ convolution

export interface Conv2dOptions {
  stride?: number | readonly [number, number];
  padding?: number | readonly [number, number];
  dilation?: number | readonly [number, number];
}

function pairOf(value: number | readonly [number, number] | undefined, fallback: number): [number, number] {
  if (value === undefined) return [fallback, fallback];
  return typeof value === 'number' ? [value, value] : [value[0], value[1]];
}

/** 2-D cross-correlation over ``[B, C, H, W]`` with weight ``[O, C, kh, kw]``. */
export function conv2d(x: Tensor, weight: Tensor, bias: Tensor | null = null, options: Conv2dOptions = {}): Tensor {
  if (x.ndim !== 4 || weight.ndim !== 4) throw new RangeError('conv2d expects [B, C, H, W] input and [O, C, kh, kw] weight');
  const [batch, channels, height, width] = x.shape as [number, number, number, number];
  const [outChannels, weightChannels, kh, kw] = weight.shape as [number, number, number, number];
  if (channels !== weightChannels) {
    throw new RangeError(`conv2d input channels ${channels} do not match weight ${formatShape(weight.shape)}`);
  }
  const [sh, sw] = pairOf(options.stride, 1);
  const [ph, pw] = pairOf(options.padding, 0);
  const [dh, dw] = pairOf(options.dilation, 1);
  const outH = Math.floor((height + 2 * ph - dh * (kh - 1) - 1) / sh) + 1;
  const outW = Math.floor((width + 2 * pw - dw * (kw - 1) - 1) / sw) + 1;
  if (outH < 1 || outW < 1) throw new RangeError('conv2d kernel is larger than the padded input');
  const dtype: DType = isFloatingDType(x.dtype) ? x.dtype : 'float32';
  const patch = channels * kh * kw;
  const positions = outH * outW;
  // im2col: [batch, positions, patch] with -1 marking padding.
  const sourceIndex = new Int32Array(positions * patch);
  for (let oy = 0; oy < outH; oy += 1) {
    for (let ox = 0; ox < outW; ox += 1) {
      const position = oy * outW + ox;
      let column = 0;
      for (let c = 0; c < channels; c += 1) {
        for (let ky = 0; ky < kh; ky += 1) {
          for (let kx = 0; kx < kw; kx += 1, column += 1) {
            const iy = oy * sh - ph + ky * dh;
            const ix = ox * sw - pw + kx * dw;
            sourceIndex[position * patch + column] = iy < 0 || iy >= height || ix < 0 || ix >= width
              ? -1
              : (c * height + iy) * width + ix;
          }
        }
      }
    }
  }
  const xd = x.data;
  const wd = weight.data;
  const bd = bias?.data;
  const imageSize = channels * height * width;
  const out = allocate(dtype, batch * outChannels * positions);
  const column = new Float64Array(patch);
  for (let n = 0; n < batch; n += 1) {
    const imageBase = n * imageSize;
    for (let position = 0; position < positions; position += 1) {
      for (let p = 0; p < patch; p += 1) {
        const source = sourceIndex[position * patch + p]!;
        column[p] = source < 0 ? 0 : xd[imageBase + source]!;
      }
      for (let o = 0; o < outChannels; o += 1) {
        let acc = bd ? bd[o]! : 0;
        const wBase = o * patch;
        for (let p = 0; p < patch; p += 1) acc += wd[wBase + p]! * column[p]!;
        out[(n * outChannels + o) * positions + position] = acc;
      }
    }
  }
  const result = fromStorage(out, [batch, outChannels, outH, outW], dtype);
  return attachGrad(result, [x, weight, bias], (grad) => {
    const g = grad.data;
    const gx = x.requiresGrad ? new Float64Array(x.numel) : null;
    const gw = weight.requiresGrad ? new Float64Array(weight.numel) : null;
    const gb = bias?.requiresGrad ? new Float64Array(outChannels) : null;
    for (let n = 0; n < batch; n += 1) {
      const imageBase = n * imageSize;
      for (let position = 0; position < positions; position += 1) {
        for (let p = 0; p < patch; p += 1) {
          const source = sourceIndex[position * patch + p]!;
          column[p] = source < 0 ? 0 : xd[imageBase + source]!;
        }
        for (let o = 0; o < outChannels; o += 1) {
          const value = g[(n * outChannels + o) * positions + position]!;
          if (value === 0) continue;
          if (gb) gb[o]! += value;
          const wBase = o * patch;
          for (let p = 0; p < patch; p += 1) {
            if (gw) gw[wBase + p]! += value * column[p]!;
            if (gx) {
              const source = sourceIndex[position * patch + p]!;
              if (source >= 0) gx[imageBase + source]! += value * wd[wBase + p]!;
            }
          }
        }
      }
    }
    const pack = (values: Float64Array | null, like: Tensor | null): Tensor | null => {
      if (!values || !like) return null;
      const storage = allocate(like.dtype, values.length);
      storage.set(values);
      return fromStorage(storage, like.shape, like.dtype);
    };
    return [pack(gx, x), pack(gw, weight), pack(gb, bias)];
  }, 'conv2d');
}

// ------------------------------------------------------------------ losses

export type Reduction = 'mean' | 'sum' | 'none';

export interface CrossEntropyOptions {
  ignoreIndex?: number;
  reduction?: Reduction;
}

/**
 * Cross entropy of unnormalized ``logits`` (``[N, C]`` or ``[C]``) against
 * integer class indices (``[N]`` or 0-d) or class probabilities with the logits'
 * shape. ``ignoreIndex`` targets are excluded from the mean.
 */
export function crossEntropy(logits: Tensor, target: Tensor, options: CrossEntropyOptions = {}): Tensor {
  const reduction = options.reduction ?? 'mean';
  const ignoreIndex = options.ignoreIndex ?? -100;
  const logProbabilities = logSoftmax(logits, -1);
  if (isFloatingDType(target.dtype)) {
    if (!shapesEqual(target.shape, logits.shape)) throw new RangeError('probability targets must match the logits shape');
    const losses = neg(sum(mul(logProbabilities, target), -1));
    return reduce(losses, reduction);
  }
  const batched = logits.ndim === 2;
  if (logits.ndim === 1 ? target.ndim !== 0 : logits.ndim !== 2 || target.ndim !== 1 || target.shape[0] !== logits.shape[0]) {
    throw new RangeError(`cross entropy target ${formatShape(target.shape)} does not match logits ${formatShape(logits.shape)}`);
  }
  const classes = logits.shape[logits.ndim - 1]!;
  const values = Array.from(target.data);
  const valid = values.map((value) => value !== ignoreIndex);
  values.forEach((value, index) => {
    if (valid[index] && (!Number.isInteger(value) || value < 0 || value >= classes)) {
      throw new RangeError(`cross entropy target ${value} out of range for ${classes} classes`);
    }
  });
  const safe = values.map((value, index) => (valid[index] ? value : 0));
  const rows = batched ? logProbabilities : unsqueeze(logProbabilities, 0);
  const index = tensor(safe.map((value) => [value]), { dtype: 'int64' });
  const picked = squeeze(gather(rows, 1, index), 1);
  const weights = tensor(valid.map((flag) => (flag ? 1 : 0)), { dtype: picked.dtype });
  const losses = neg(mul(picked, weights));
  if (reduction === 'none') return batched ? losses : squeeze(losses, 0);
  const total = sum(losses);
  if (reduction === 'sum') return total;
  const count = valid.filter(Boolean).length;
  return div(total, count);
}

export function nllLoss(logProbabilities: Tensor, target: Tensor, options: CrossEntropyOptions = {}): Tensor {
  const reduction = options.reduction ?? 'mean';
  const ignoreIndex = options.ignoreIndex ?? -100;
  const values = Array.from(target.data);
  const valid = values.map((value) => value !== ignoreIndex);
  const index = tensor(values.map((value, position) => [valid[position] ? value : 0]), { dtype: 'int64' });
  const rows = logProbabilities.ndim === 1 ? unsqueeze(logProbabilities, 0) : logProbabilities;
  const picked = squeeze(gather(rows, 1, index), 1);
  const losses = neg(mul(picked, tensor(valid.map((flag) => (flag ? 1 : 0)), { dtype: picked.dtype })));
  if (reduction === 'none') return losses;
  const total = sum(losses);
  return reduction === 'sum' ? total : div(total, valid.filter(Boolean).length);
}

export function mseLoss(prediction: Tensor, target: Tensor, reduction: Reduction = 'mean'): Tensor {
  if (!shapesEqual(prediction.shape, target.shape)) {
    throw new RangeError(`mse target ${formatShape(target.shape)} must match prediction ${formatShape(prediction.shape)}`);
  }
  const difference = sub(prediction, target);
  return reduce(mul(difference, difference), reduction);
}

/** Numerically stable binary cross entropy on logits. */
export function binaryCrossEntropyWithLogits(logits: Tensor, target: Tensor, reduction: Reduction = 'mean'): Tensor {
  if (!shapesEqual(logits.shape, target.shape)) throw new RangeError('binary targets must match the logits shape');
  // max(x, 0) - x*y + log(1 + exp(-|x|))
  const losses = add(sub(relu(logits), mul(logits, target)), log(add(exp(neg(abs(logits))), 1)));
  return reduce(losses, reduction);
}

function reduce(losses: Tensor, reduction: Reduction): Tensor {
  if (reduction === 'none') return losses;
  if (reduction === 'sum') return sum(losses);
  return mean(losses);
}

/** Scaled dot-product attention: ``softmax(q k^T * scale + bias) v``. */
export function scaledDotProductAttention(
  query: Tensor, key: Tensor, value: Tensor,
  options: { bias?: Tensor | null; scale?: number; dropout?: number; training?: boolean } = {},
): { output: Tensor; weights: Tensor } {
  const scale = options.scale ?? 1 / Math.sqrt(query.shape[query.ndim - 1]!);
  let scores = mul(queryKey(query, key), scale);
  if (options.bias) scores = add(scores, options.bias);
  let weights = softmaxLast(scores);
  if (options.dropout && options.training) weights = dropout(weights, options.dropout, true);
  return { output: attend(weights, value), weights };
}

function queryKey(query: Tensor, key: Tensor): Tensor {
  return matmul(query, transpose(key, -2, -1));
}

function softmaxLast(scores: Tensor): Tensor {
  return softmax(scores, -1);
}

function attend(weights: Tensor, value: Tensor): Tensor {
  return matmul(weights, value);
}

