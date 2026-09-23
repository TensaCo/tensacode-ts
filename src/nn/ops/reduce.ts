/** Reductions, normalizations over a dimension, and cumulative sums. */
import { allocate, isFloatingDType, type DType } from '../dtype.js';
import { normalizeDim, normalizeDims, numelOf, shapesEqual, type Shape } from '../shape.js';
import { Tensor, attachGrad, fromStorage, aliasWithShape } from '../tensor.js';
import { expand } from './shape.js';
import { div, mul, sub, exp, mapBinary, sqrt } from './elementwise.js';

/** Split a shape around ``dim`` into outer, size and inner extents. */
export function extents(shape: Shape, dim: number): [number, number, number] {
  let outer = 1;
  for (let index = 0; index < dim; index += 1) outer *= shape[index]!;
  let inner = 1;
  for (let index = dim + 1; index < shape.length; index += 1) inner *= shape[index]!;
  return [outer, shape[dim]!, inner];
}

function keptShape(shape: Shape, dims: readonly number[]): number[] {
  return shape.map((size, index) => (dims.includes(index) ? 1 : size));
}

function outputShape(shape: Shape, dims: readonly number[], keepdim: boolean): number[] {
  return keepdim ? keptShape(shape, dims) : shape.filter((_, index) => !dims.includes(index));
}

function reduceDimRaw(
  data: ArrayLike<number>, shape: Shape, dim: number, dtype: DType,
  init: number, step: (acc: number, value: number) => number,
): { data: Float32Array | Float64Array; shape: number[] } {
  const [outer, size, inner] = extents(shape, dim);
  const out = allocate(dtype, outer * inner);
  for (let o = 0; o < outer; o += 1) {
    for (let i = 0; i < inner; i += 1) {
      let acc = init;
      let offset = o * size * inner + i;
      for (let k = 0; k < size; k += 1, offset += inner) acc = step(acc, data[offset]!);
      out[o * inner + i] = acc;
    }
  }
  const next = [...shape];
  next[dim] = 1;
  return { data: out, shape: next };
}

function accumulateDtype(dtype: DType): DType {
  return isFloatingDType(dtype) ? (dtype === 'float64' ? 'float64' : 'float64') : 'float64';
}

function reduceDims(
  x: Tensor, dims: readonly number[], keepdim: boolean, resultDtype: DType,
  init: number, step: (acc: number, value: number) => number,
): Tensor {
  let data: ArrayLike<number> = x.data;
  let shape: number[] = [...x.shape];
  const work = accumulateDtype(x.dtype);
  for (let index = dims.length - 1; index >= 0; index -= 1) {
    const reduced = reduceDimRaw(data, shape, dims[index]!, work, init, step);
    data = reduced.data;
    shape = reduced.shape;
  }
  const out = allocate(resultDtype, (data as ArrayLike<number>).length);
  for (let index = 0; index < out.length; index += 1) out[index] = (data as ArrayLike<number>)[index]!;
  return fromStorage(out, outputShape(x.shape, dims, keepdim), resultDtype);
}

/** Sum ``grad`` down to ``shape`` (inverse of broadcasting). */
export function sumToShape(grad: Tensor, shape: Shape): Tensor {
  if (shapesEqual(grad.shape, shape)) return grad;
  const lead = grad.ndim - shape.length;
  const dims: number[] = [];
  for (let index = 0; index < grad.ndim; index += 1) {
    if (index < lead) dims.push(index);
    else if (shape[index - lead] === 1 && grad.shape[index] !== 1) dims.push(index);
  }
  const reduced = dims.length ? sum(grad, dims, true) : grad;
  return aliasWithShape(reduced, shape);
}

function sumResultDtype(dtype: DType): DType {
  return isFloatingDType(dtype) ? dtype : 'int64';
}

export function sum(x: Tensor, dim?: number | readonly number[] | null, keepdim = false): Tensor {
  const dims = normalizeDims(dim, x.ndim);
  const result = reduceDims(x, dims, keepdim, sumResultDtype(x.dtype), 0, (acc, value) => acc + value);
  return attachGrad(result, [x], (grad) => [
    expand(aliasWithShape(grad, keptShape(x.shape, dims)), x.shape),
  ], 'sum');
}

export function mean(x: Tensor, dim?: number | readonly number[] | null, keepdim = false): Tensor {
  const dims = normalizeDims(dim, x.ndim);
  const count = dims.reduce((product, index) => product * x.shape[index]!, 1);
  const dtype = isFloatingDType(x.dtype) ? x.dtype : 'float32';
  const total = reduceDims(x, dims, keepdim, 'float64', 0, (acc, value) => acc + value);
  const out = allocate(dtype, total.numel);
  for (let index = 0; index < out.length; index += 1) out[index] = total.data[index]! / count;
  const result = fromStorage(out, total.shape, dtype);
  return attachGrad(result, [x], (grad) => [
    div(expand(aliasWithShape(grad, keptShape(x.shape, dims)), x.shape), count),
  ], 'mean');
}

/** Unbiased (default) or population variance. */
export function variance(x: Tensor, dim?: number | readonly number[] | null, keepdim = false, unbiased = true): Tensor {
  const dims = normalizeDims(dim, x.ndim);
  const count = dims.reduce((product, index) => product * x.shape[index]!, 1);
  const centered = sub(x, mean(x, dims, true));
  const squares = sum(mul(centered, centered), dims, keepdim);
  return div(squares, Math.max(unbiased ? count - 1 : count, 0));
}

function extreme(x: Tensor, dim: number | readonly number[] | null | undefined, keepdim: boolean, largest: boolean, name: string): Tensor {
  if (x.numel === 0) throw new RangeError(`${name}() of an empty tensor`);
  const dims = normalizeDims(dim, x.ndim);
  const result = reduceDims(
    x, dims, keepdim, x.dtype, largest ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY,
    largest
      ? (acc, value) => (Number.isNaN(acc) || Number.isNaN(value) ? Number.NaN : value > acc ? value : acc)
      : (acc, value) => (Number.isNaN(acc) || Number.isNaN(value) ? Number.NaN : value < acc ? value : acc),
  );
  return attachGrad(result, [x], (grad) => {
    // Distribute the gradient evenly among tied extreme elements.
    const kept = aliasWithShape(result.detach(), keptShape(x.shape, dims));
    const selected = mapBinary(x, kept, (value, best) => (value === best ? 1 : 0), grad.dtype);
    const counts = sum(selected, dims, true);
    const scaled = div(aliasWithShape(grad, keptShape(x.shape, dims)), counts);
    return [mul(selected, expand(scaled, x.shape))];
  }, name);
}

export function amax(x: Tensor, dim?: number | readonly number[] | null, keepdim = false): Tensor {
  return extreme(x, dim, keepdim, true, 'amax');
}

export function amin(x: Tensor, dim?: number | readonly number[] | null, keepdim = false): Tensor {
  return extreme(x, dim, keepdim, false, 'amin');
}

function argExtremeRaw(x: Tensor, dim: number, largest: boolean): { values: Float64Array; indices: Float64Array; shape: number[] } {
  const [outer, size, inner] = extents(x.shape, dim);
  if (size === 0) throw new RangeError('reduction over an empty dimension');
  const data = x.data;
  const values = new Float64Array(outer * inner);
  const indices = new Float64Array(outer * inner);
  for (let o = 0; o < outer; o += 1) {
    for (let i = 0; i < inner; i += 1) {
      let bestIndex = 0;
      let best = data[o * size * inner + i]!;
      for (let k = 1; k < size; k += 1) {
        const value = data[(o * size + k) * inner + i]!;
        if (Number.isNaN(best)) break;
        if (Number.isNaN(value) || (largest ? value > best : value < best)) {
          best = value;
          bestIndex = k;
        }
      }
      values[o * inner + i] = best;
      indices[o * inner + i] = bestIndex;
    }
  }
  const shape = [...x.shape];
  shape[dim] = 1;
  return { values, indices, shape };
}

/** Values and first indices of the maximum (``largest``) or minimum along ``dim``. */
export function maxDim(x: Tensor, dim: number, keepdim: boolean, largest: boolean): { values: Tensor; indices: Tensor } {
  const d = normalizeDim(dim, x.ndim);
  const raw = argExtremeRaw(x, d, largest);
  const shape = keepdim ? raw.shape : raw.shape.filter((_, index) => index !== d);
  const valueData = allocate(x.dtype, raw.values.length);
  valueData.set(raw.values);
  const indices = fromStorage(raw.indices, shape, 'int64');
  const values = fromStorage(valueData, shape, x.dtype);
  attachGrad(values, [x], (grad) => {
    const out = allocate(grad.dtype, x.numel);
    const [outer, size, inner] = extents(x.shape, d);
    const g = grad.data;
    for (let o = 0; o < outer; o += 1) {
      for (let i = 0; i < inner; i += 1) {
        const k = raw.indices[o * inner + i]!;
        out[(o * size + k) * inner + i] = g[o * inner + i]!;
      }
    }
    return [fromStorage(out, x.shape, grad.dtype)];
  }, largest ? 'max' : 'min');
  return { values, indices };
}

export function argExtreme(x: Tensor, dim: number | undefined, keepdim: boolean, largest: boolean): Tensor {
  if (dim === undefined) {
    const flat = aliasWithShape(x.detach(), [x.numel]);
    const raw = argExtremeRaw(flat, 0, largest);
    return fromStorage(raw.indices, keepdim ? x.shape.map(() => 1) : [], 'int64');
  }
  return maxDim(x.detach(), dim, keepdim, largest).indices;
}

export function softmax(x: Tensor, dim: number): Tensor {
  const d = normalizeDim(dim, x.ndim);
  const dtype = isFloatingDType(x.dtype) ? x.dtype : 'float32';
  const [outer, size, inner] = extents(x.shape, d);
  const data = x.data;
  const out = allocate(dtype, x.numel);
  for (let o = 0; o < outer; o += 1) {
    for (let i = 0; i < inner; i += 1) {
      const base = o * size * inner + i;
      let maximum = Number.NEGATIVE_INFINITY;
      for (let k = 0; k < size; k += 1) {
        const value = data[base + k * inner]!;
        if (value > maximum || Number.isNaN(value)) maximum = value;
      }
      if (maximum === Number.NEGATIVE_INFINITY) {
        // All entries are -inf: PyTorch returns NaN for the whole row.
        for (let k = 0; k < size; k += 1) out[base + k * inner] = Number.NaN;
        continue;
      }
      let total = 0;
      for (let k = 0; k < size; k += 1) total += Math.exp(data[base + k * inner]! - maximum);
      for (let k = 0; k < size; k += 1) out[base + k * inner] = Math.exp(data[base + k * inner]! - maximum) / total;
    }
  }
  const result = fromStorage(out, x.shape, dtype);
  return attachGrad(result, [x], (grad) => {
    const g = grad.data;
    const y = result.data;
    const gradIn = allocate(grad.dtype, x.numel);
    for (let o = 0; o < outer; o += 1) {
      for (let i = 0; i < inner; i += 1) {
        const base = o * size * inner + i;
        let dot = 0;
        for (let k = 0; k < size; k += 1) dot += g[base + k * inner]! * y[base + k * inner]!;
        for (let k = 0; k < size; k += 1) {
          const offset = base + k * inner;
          gradIn[offset] = y[offset]! * (g[offset]! - dot);
        }
      }
    }
    return [fromStorage(gradIn, x.shape, grad.dtype)];
  }, 'softmax');
}

export function logSoftmax(x: Tensor, dim: number): Tensor {
  const d = normalizeDim(dim, x.ndim);
  const dtype = isFloatingDType(x.dtype) ? x.dtype : 'float32';
  const [outer, size, inner] = extents(x.shape, d);
  const data = x.data;
  const out = allocate(dtype, x.numel);
  for (let o = 0; o < outer; o += 1) {
    for (let i = 0; i < inner; i += 1) {
      const base = o * size * inner + i;
      let maximum = Number.NEGATIVE_INFINITY;
      for (let k = 0; k < size; k += 1) {
        const value = data[base + k * inner]!;
        if (value > maximum || Number.isNaN(value)) maximum = value;
      }
      let total = 0;
      for (let k = 0; k < size; k += 1) total += Math.exp(data[base + k * inner]! - maximum);
      const logTotal = maximum + Math.log(total);
      for (let k = 0; k < size; k += 1) out[base + k * inner] = data[base + k * inner]! - logTotal;
    }
  }
  const result = fromStorage(out, x.shape, dtype);
  return attachGrad(result, [x], (grad) => {
    const g = grad.data;
    const y = result.data;
    const gradIn = allocate(grad.dtype, x.numel);
    for (let o = 0; o < outer; o += 1) {
      for (let i = 0; i < inner; i += 1) {
        const base = o * size * inner + i;
        let total = 0;
        for (let k = 0; k < size; k += 1) total += g[base + k * inner]!;
        for (let k = 0; k < size; k += 1) {
          const offset = base + k * inner;
          gradIn[offset] = g[offset]! - Math.exp(y[offset]!) * total;
        }
      }
    }
    return [fromStorage(gradIn, x.shape, grad.dtype)];
  }, 'logSoftmax');
}

export function logsumexp(x: Tensor, dim: number | readonly number[], keepdim = false): Tensor {
  const dims = normalizeDims(dim, x.ndim);
  const maximum = amax(x.detach(), dims, true);
  const finiteMax = maximum.data.map((value) => (Number.isFinite(value) ? value : 0));
  const shift = fromStorage(finiteMax, maximum.shape, maximum.dtype);
  const shifted = exp(sub(x, shift));
  const total = sum(shifted, dims, true);
  const logged = mapBinary(total, shift, (t, m) => Math.log(t) + m, total.dtype);
  const result = aliasWithShape(logged, outputShape(x.shape, dims, keepdim));
  return attachGrad(result, [x], (grad) => {
    const kept = aliasWithShape(grad, keptShape(x.shape, dims));
    const probabilities = exp(sub(x, aliasWithShape(logged, keptShape(x.shape, dims))));
    return [mul(probabilities, expand(kept, x.shape))];
  }, 'logsumexp');
}

export function cumsum(x: Tensor, dim: number): Tensor {
  const d = normalizeDim(dim, x.ndim);
  const [outer, size, inner] = extents(x.shape, d);
  const dtype = isFloatingDType(x.dtype) ? x.dtype : 'int64';
  const data = x.data;
  const out = allocate(dtype, x.numel);
  for (let o = 0; o < outer; o += 1) {
    for (let i = 0; i < inner; i += 1) {
      let acc = 0;
      for (let k = 0; k < size; k += 1) {
        const offset = (o * size + k) * inner + i;
        acc += data[offset]!;
        out[offset] = acc;
      }
    }
  }
  const result = fromStorage(out, x.shape, dtype);
  return attachGrad(result, [x], (grad) => {
    const g = grad.data;
    const gradIn = allocate(grad.dtype, x.numel);
    for (let o = 0; o < outer; o += 1) {
      for (let i = 0; i < inner; i += 1) {
        let acc = 0;
        for (let k = size - 1; k >= 0; k -= 1) {
          const offset = (o * size + k) * inner + i;
          acc += g[offset]!;
          gradIn[offset] = acc;
        }
      }
    }
    return [fromStorage(gradIn, x.shape, grad.dtype)];
  }, 'cumsum');
}

export function anyAll(x: Tensor, dim: number | null | undefined, keepdim: boolean, kind: 'any' | 'all'): Tensor {
  const dims = normalizeDims(dim, x.ndim);
  return reduceDims(
    x, dims, keepdim, 'bool', kind === 'any' ? 0 : 1,
    kind === 'any' ? (acc, value) => (acc || value ? 1 : 0) : (acc, value) => (acc && value ? 1 : 0),
  );
}

export function norm(x: Tensor, p: number, dim: number | readonly number[] | null | undefined, keepdim: boolean): Tensor {
  if (p === 2) return sqrt(sum(mul(x, x), dim, keepdim));
  if (p === 1) return sum(mapAbs(x), dim, keepdim);
  if (p === Number.POSITIVE_INFINITY) return amax(mapAbs(x), dim, keepdim);
  throw new RangeError('norm supports p = 1, 2 or Infinity');
}

function mapAbs(x: Tensor): Tensor {
  return mul(x, mapBinary(x, x, (value) => (value > 0 ? 1 : value < 0 ? -1 : 0), x.dtype));
}

export function numelAlong(shape: Shape, dims: readonly number[]): number {
  return numelOf(dims.map((dim) => shape[dim]!));
}
