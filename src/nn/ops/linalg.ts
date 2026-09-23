/** Matrix products. */
import { allocate, isFloatingDType, promoteTypes, type DType } from '../dtype.js';
import { broadcastShapes, formatShape, numelOf, stridesOf } from '../shape.js';
import { Tensor, attachGrad, fromStorage, aliasWithShape } from '../tensor.js';
import { sumToShape } from './reduce.js';
import { transpose } from './shape.js';
import { cast } from './elementwise.js';

/**
 * Batched ``[..., n, k] @ [..., k, m]`` with broadcast batch dimensions. Each
 * output row accumulates in float64 before storing.
 */
function batchedMatmul(a: Tensor, b: Tensor, dtype: DType): Tensor {
  const n = a.shape[a.ndim - 2]!;
  const k = a.shape[a.ndim - 1]!;
  const m = b.shape[b.ndim - 1]!;
  if (b.shape[b.ndim - 2] !== k) {
    throw new RangeError(`matmul shape mismatch: ${formatShape(a.shape)} @ ${formatShape(b.shape)}`);
  }
  const batchA = a.shape.slice(0, -2);
  const batchB = b.shape.slice(0, -2);
  const batch = broadcastShapes(batchA, batchB);
  const batchCount = numelOf(batch);
  const out = allocate(dtype, batchCount * n * m);
  const ad = a.data;
  const bd = b.data;
  const aStrides = batchStrides(batch, batchA);
  const bStrides = batchStrides(batch, batchB);
  const counter = new Array<number>(batch.length).fill(0);
  const row = new Float64Array(m);
  for (let flat = 0; flat < batchCount; flat += 1) {
    let aBatch = 0;
    let bBatch = 0;
    for (let axis = 0; axis < batch.length; axis += 1) {
      aBatch += counter[axis]! * aStrides[axis]!;
      bBatch += counter[axis]! * bStrides[axis]!;
    }
    const aBase = aBatch * n * k;
    const bBase = bBatch * k * m;
    const outBase = flat * n * m;
    for (let i = 0; i < n; i += 1) {
      row.fill(0);
      const aRow = aBase + i * k;
      for (let p = 0; p < k; p += 1) {
        const value = ad[aRow + p]!;
        if (value === 0) continue;
        const bRow = bBase + p * m;
        for (let j = 0; j < m; j += 1) row[j]! += value * bd[bRow + j]!;
      }
      const target = outBase + i * m;
      for (let j = 0; j < m; j += 1) out[target + j] = row[j]!;
    }
    for (let axis = batch.length - 1; axis >= 0; axis -= 1) {
      counter[axis]! += 1;
      if (counter[axis]! < batch[axis]!) break;
      counter[axis] = 0;
    }
  }
  return fromStorage(out, [...batch, n, m], dtype);
}

function batchStrides(batch: readonly number[], shape: readonly number[]): number[] {
  const offset = batch.length - shape.length;
  const strides = stridesOf(shape);
  return batch.map((_, axis) => {
    if (axis < offset) return 0;
    const size = shape[axis - offset]!;
    return size === 1 ? 0 : strides[axis - offset]!;
  });
}

function floatType(a: Tensor, b: Tensor): DType {
  const dtype = promoteTypes(a.dtype, b.dtype);
  return isFloatingDType(dtype) ? dtype : dtype;
}

/** ``torch.matmul`` semantics for 1-D, 2-D and batched operands. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.ndim === 0 || b.ndim === 0) throw new RangeError('matmul operands must have at least one dimension');
  const dtype = floatType(a, b);
  const x = cast(a, dtype);
  const y = cast(b, dtype);
  const left = x.ndim === 1 ? aliasWithShape(x, [1, x.shape[0]!]) : x;
  const right = y.ndim === 1 ? aliasWithShape(y, [y.shape[0]!, 1]) : y;
  const product = batchedMatmul(left, right, dtype);
  let shape = [...product.shape];
  if (x.ndim === 1) shape.splice(shape.length - 2, 1);
  if (y.ndim === 1) shape = shape.slice(0, -1);
  const result = aliasWithShape(product, shape);
  return attachGrad(result, [x, y], (grad) => {
    const g = aliasWithShape(grad, product.shape);
    const gradLeft = x.requiresGrad ? sumToShape(batchedMatmul(g, transpose(right, -2, -1), g.dtype), left.shape) : null;
    const gradRight = y.requiresGrad ? sumToShape(batchedMatmul(transpose(left, -2, -1), g, g.dtype), right.shape) : null;
    return [
      gradLeft ? aliasWithShape(gradLeft, x.shape) : null,
      gradRight ? aliasWithShape(gradRight, y.shape) : null,
    ];
  }, 'matmul');
}

/**
 * ``x @ weight.T + bias`` for ``weight`` of shape ``[out, in]`` without
 * materializing the transpose.
 */
export function linear(x: Tensor, weight: Tensor, bias: Tensor | null = null): Tensor {
  const inFeatures = weight.shape[1]!;
  const outFeatures = weight.shape[0]!;
  if (weight.ndim !== 2) throw new RangeError('linear weight must be 2-dimensional');
  if (x.shape[x.ndim - 1] !== inFeatures) {
    throw new RangeError(`linear input ${formatShape(x.shape)} does not match weight ${formatShape(weight.shape)}`);
  }
  let dtype = promoteTypes(x.dtype, weight.dtype);
  if (!isFloatingDType(dtype)) dtype = 'float32';
  const input = cast(x, dtype);
  const rows = input.numel / inFeatures;
  const out = allocate(dtype, rows * outFeatures);
  const xd = input.data;
  const wd = weight.data;
  const bd = bias ? bias.data : null;
  for (let r = 0; r < rows; r += 1) {
    const xBase = r * inFeatures;
    const outBase = r * outFeatures;
    for (let o = 0; o < outFeatures; o += 1) {
      const wBase = o * inFeatures;
      let acc = bd ? bd[o]! : 0;
      for (let i = 0; i < inFeatures; i += 1) acc += xd[xBase + i]! * wd[wBase + i]!;
      out[outBase + o] = acc;
    }
  }
  const result = fromStorage(out, [...x.shape.slice(0, -1), outFeatures], dtype);
  return attachGrad(result, [input, weight, bias], (grad) => {
    const g = grad.data;
    let gradInput: Tensor | null = null;
    let gradWeight: Tensor | null = null;
    let gradBias: Tensor | null = null;
    if (input.requiresGrad) {
      const gi = allocate(grad.dtype, input.numel);
      const row = new Float64Array(inFeatures);
      for (let r = 0; r < rows; r += 1) {
        row.fill(0);
        for (let o = 0; o < outFeatures; o += 1) {
          const value = g[r * outFeatures + o]!;
          if (value === 0) continue;
          const wBase = o * inFeatures;
          for (let i = 0; i < inFeatures; i += 1) row[i]! += value * wd[wBase + i]!;
        }
        gi.set(row, r * inFeatures);
      }
      gradInput = fromStorage(gi, input.shape, grad.dtype);
    }
    if (weight.requiresGrad) {
      const accumulator = new Float64Array(outFeatures * inFeatures);
      for (let r = 0; r < rows; r += 1) {
        const xBase = r * inFeatures;
        for (let o = 0; o < outFeatures; o += 1) {
          const value = g[r * outFeatures + o]!;
          if (value === 0) continue;
          const wBase = o * inFeatures;
          for (let i = 0; i < inFeatures; i += 1) accumulator[wBase + i]! += value * xd[xBase + i]!;
        }
      }
      const gw = allocate(weight.dtype, accumulator.length);
      gw.set(accumulator);
      gradWeight = fromStorage(gw, weight.shape, weight.dtype);
    }
    if (bias && bias.requiresGrad) {
      const accumulator = new Float64Array(outFeatures);
      for (let r = 0; r < rows; r += 1) for (let o = 0; o < outFeatures; o += 1) accumulator[o]! += g[r * outFeatures + o]!;
      const gb = allocate(bias.dtype, outFeatures);
      gb.set(accumulator);
      gradBias = fromStorage(gb, bias.shape, bias.dtype);
    }
    return [gradInput, gradWeight, gradBias];
  }, 'linear');
}
