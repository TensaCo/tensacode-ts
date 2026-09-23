/** Shape manipulation, indexing, joining and ordering operations. */
import { allocate, promoteTypes, type Storage } from '../dtype.js';
import {
  broadcastIndexMap, formatShape, inferShape, normalizeDim, numelOf, shapesEqual, stridesOf,
} from '../shape.js';
import { Tensor, attachGrad, fromStorage, aliasWithShape } from '../tensor.js';
import { extents, sumToShape } from './reduce.js';
import { cast } from './elementwise.js';
import { swapAxes } from '../backend/kernels.js';

// ------------------------------------------------------------------ views

export function reshape(x: Tensor, shape: readonly number[]): Tensor {
  const target = inferShape(shape, x.numel);
  if (shapesEqual(target, x.shape)) return x;
  const result = aliasWithShape(x, target);
  return attachGrad(result, [x], (grad) => [aliasWithShape(grad, x.shape)], 'reshape');
}

export function flatten(x: Tensor, startDim = 0, endDim = -1): Tensor {
  if (x.ndim === 0) return reshape(x, [1]);
  const start = normalizeDim(startDim, x.ndim);
  const end = normalizeDim(endDim, x.ndim);
  if (start > end) throw new RangeError('flatten start dimension must not follow end dimension');
  const merged = x.shape.slice(start, end + 1).reduce((product, size) => product * size, 1);
  return reshape(x, [...x.shape.slice(0, start), merged, ...x.shape.slice(end + 1)]);
}

export function unsqueeze(x: Tensor, dim: number): Tensor {
  const d = normalizeDim(dim, x.ndim, true);
  const shape = [...x.shape];
  shape.splice(d, 0, 1);
  return reshape(x, shape);
}

export function squeeze(x: Tensor, dim?: number): Tensor {
  if (dim === undefined) return reshape(x, x.shape.filter((size) => size !== 1));
  const d = normalizeDim(dim, x.ndim);
  if (x.shape[d] !== 1) return x;
  const shape = [...x.shape];
  shape.splice(d, 1);
  return reshape(x, shape);
}

// ------------------------------------------------------------------ copies

export function permute(x: Tensor, dims: readonly number[]): Tensor {
  if (dims.length !== x.ndim) throw new RangeError('permute needs one entry per dimension');
  const order = dims.map((dim) => normalizeDim(dim, x.ndim));
  if (new Set(order).size !== order.length) throw new RangeError('permute dimensions must be unique');
  if (order.every((dim, index) => dim === index)) return x;
  const shape = order.map((dim) => x.shape[dim]!);
  const inStrides = stridesOf(x.shape);
  const strides = order.map((dim) => inStrides[dim]!);
  const size = x.numel;
  const source = x.data;
  const ndim = shape.length;
  const swapped = source instanceof Float32Array ? adjacentSwap(order) : -1;
  if (swapped >= 0) {
    const extent = (from: number, to: number) => x.shape.slice(from, to).reduce((product, value) => product * value, 1);
    const fast = swapAxes(source as Float32Array, extent(0, swapped), x.shape[swapped]!, x.shape[swapped + 1]!, extent(swapped + 2, ndim));
    if (fast) return permuteResult(x, fast, shape, order);
  }
  const out = allocate(x.dtype, size);
  const counter = new Array<number>(ndim).fill(0);
  let position = 0;
  if (order[ndim - 1] === ndim - 1 && size > 0) {
    // The last axis stays last: copy contiguous rows (for example head splits).
    const inner = shape[ndim - 1]!;
    for (let flat = 0; flat < size; flat += inner) {
      for (let index = 0; index < inner; index += 1) out[flat + index] = source[position + index]!;
      for (let dim = ndim - 2; dim >= 0; dim -= 1) {
        counter[dim]! += 1;
        position += strides[dim]!;
        if (counter[dim]! < shape[dim]!) break;
        position -= strides[dim]! * counter[dim]!;
        counter[dim] = 0;
      }
    }
  }
  for (let flat = order[ndim - 1] === ndim - 1 ? size : 0; flat < size; flat += 1) {
    out[flat] = source[position]!;
    for (let dim = ndim - 1; dim >= 0; dim -= 1) {
      counter[dim]! += 1;
      position += strides[dim]!;
      if (counter[dim]! < shape[dim]!) break;
      position -= strides[dim]! * counter[dim]!;
      counter[dim] = 0;
    }
  }
  return permuteResult(x, out, shape, order);
}

function permuteResult(x: Tensor, out: Storage, shape: number[], order: number[]): Tensor {
  const inverse = new Array<number>(order.length);
  order.forEach((dim, index) => {
    inverse[dim] = index;
  });
  const result = fromStorage(out, shape, x.dtype);
  return attachGrad(result, [x], (grad) => [permute(grad, inverse)], 'permute');
}

/** ``i`` when ``order`` only swaps axes ``i`` and ``i + 1``, else -1. */
function adjacentSwap(order: readonly number[]): number {
  let found = -1;
  for (let index = 0; index < order.length; index += 1) {
    if (order[index] === index) continue;
    if (found >= 0 || order[index] !== index + 1 || order[index + 1] !== index) return -1;
    found = index;
    index += 1;
  }
  return found;
}

export function transpose(x: Tensor, dim0: number, dim1: number): Tensor {
  const order = Array.from({ length: x.ndim }, (_, index) => index);
  const a = normalizeDim(dim0, x.ndim);
  const b = normalizeDim(dim1, x.ndim);
  [order[a], order[b]] = [order[b]!, order[a]!];
  return permute(x, order);
}

/**
 * Copy ``source`` (shape ``shape``) broadcast to ``target`` into ``out``,
 * copying the longest trailing block that is not broadcast as one run.
 */
function expandInto(source: Storage, shape: readonly number[], target: readonly number[], out: Storage): void {
  const ndim = target.length;
  const offset = ndim - shape.length;
  const inputSize = (axis: number): number => (axis >= offset ? shape[axis - offset]! : 1);
  let split = ndim;
  let run = 1;
  while (split > 0 && inputSize(split - 1) === target[split - 1]) {
    split -= 1;
    run *= target[split]!;
  }
  const strides = new Array<number>(split).fill(0);
  let stride = run;
  for (let axis = split - 1; axis >= 0; axis -= 1) {
    const size = inputSize(axis);
    strides[axis] = size === 1 ? 0 : stride;
    stride *= size;
  }
  const size = out.length;
  if (size === 0) return;
  const counter = new Array<number>(split).fill(0);
  let position = 0;
  for (let flat = 0; flat < size; flat += run) {
    if (run === 1) out[flat] = source[position]!;
    else out.set(source.subarray(position, position + run), flat);
    for (let axis = split - 1; axis >= 0; axis -= 1) {
      counter[axis]! += 1;
      position += strides[axis]!;
      if (counter[axis]! < target[axis]!) break;
      position -= strides[axis]! * counter[axis]!;
      counter[axis] = 0;
    }
  }
}

/** Materialize ``x`` broadcast to ``shape`` (``-1`` keeps a dimension). */
export function expand(x: Tensor, shape: readonly number[]): Tensor {
  if (shape.length < x.ndim) throw new RangeError(`cannot expand ${formatShape(x.shape)} to ${formatShape(shape)}`);
  const offset = shape.length - x.ndim;
  const target = shape.map((size, index) => {
    const current = index >= offset ? x.shape[index - offset]! : 1;
    if (size === -1) {
      if (index < offset) throw new RangeError('expand cannot infer new leading dimensions');
      return current;
    }
    if (current !== 1 && current !== size) {
      throw new RangeError(`cannot expand ${formatShape(x.shape)} to ${formatShape(shape)}`);
    }
    return size;
  });
  if (shapesEqual(target, x.shape)) return x;
  const out = allocate(x.dtype, numelOf(target));
  expandInto(x.data, x.shape, target, out);
  const result = fromStorage(out, target, x.dtype);
  return attachGrad(result, [x], (grad) => [sumToShape(grad, x.shape)], 'expand');
}

export function repeat(x: Tensor, counts: readonly number[]): Tensor {
  if (counts.length < x.ndim) throw new RangeError('repeat needs at least one count per dimension');
  const lead = counts.length - x.ndim;
  const base = [...new Array<number>(lead).fill(1), ...x.shape];
  // Interleave: [c0, s0, c1, s1, ...] then reshape.
  const interleavedShape: number[] = [];
  const sourceShape: number[] = [];
  base.forEach((size, index) => {
    interleavedShape.push(counts[index]!, size);
    sourceShape.push(1, size);
  });
  const expanded = expand(reshape(x, sourceShape), interleavedShape);
  return reshape(expanded, base.map((size, index) => size * counts[index]!));
}

export function slice(x: Tensor, dim: number, start: number | null, end: number | null, step = 1): Tensor {
  const d = normalizeDim(dim, x.ndim);
  if (!Number.isInteger(step) || step < 1) throw new RangeError('slice step must be a positive integer');
  const size = x.shape[d]!;
  const clampIndex = (value: number | null, fallback: number): number => {
    if (value === null) return fallback;
    const resolved = value < 0 ? value + size : value;
    return Math.min(Math.max(resolved, 0), size);
  };
  const from = clampIndex(start, 0);
  const to = Math.max(clampIndex(end, size), from);
  const count = Math.ceil((to - from) / step);
  if (from === 0 && count === size && step === 1) return x;
  const [outer, , inner] = extents(x.shape, d);
  const out = allocate(x.dtype, outer * count * inner);
  const source = x.data;
  for (let o = 0; o < outer; o += 1) {
    for (let k = 0; k < count; k += 1) {
      const sourceOffset = (o * size + from + k * step) * inner;
      const targetOffset = (o * count + k) * inner;
      for (let i = 0; i < inner; i += 1) out[targetOffset + i] = source[sourceOffset + i]!;
    }
  }
  const shape = [...x.shape];
  shape[d] = count;
  const result = fromStorage(out, shape, x.dtype);
  return attachGrad(result, [x], (grad) => {
    const gradIn = allocate(grad.dtype, x.numel);
    const g = grad.data;
    for (let o = 0; o < outer; o += 1) {
      for (let k = 0; k < count; k += 1) {
        const targetOffset = (o * size + from + k * step) * inner;
        const sourceOffset = (o * count + k) * inner;
        for (let i = 0; i < inner; i += 1) gradIn[targetOffset + i] = g[sourceOffset + i]!;
      }
    }
    return [fromStorage(gradIn, x.shape, grad.dtype)];
  }, 'slice');
}

export function select(x: Tensor, dim: number, index: number): Tensor {
  const d = normalizeDim(dim, x.ndim);
  const size = x.shape[d]!;
  const resolved = index < 0 ? index + size : index;
  if (!Number.isInteger(resolved) || resolved < 0 || resolved >= size) {
    throw new RangeError(`index ${index} is out of range for dimension of size ${size}`);
  }
  return squeeze(slice(x, d, resolved, resolved + 1), d);
}

function indexValues(index: Tensor | readonly number[]): number[] {
  return index instanceof Tensor ? Array.from(index.data) : [...index];
}

export function indexSelect(x: Tensor, dim: number, index: Tensor | readonly number[]): Tensor {
  const d = normalizeDim(dim, x.ndim);
  const indices = indexValues(index);
  const [outer, size, inner] = extents(x.shape, d);
  for (const value of indices) {
    if (!Number.isInteger(value) || value < -size || value >= size) throw new RangeError(`index ${value} out of range`);
  }
  const resolved = indices.map((value) => (value < 0 ? value + size : value));
  const count = resolved.length;
  const out = allocate(x.dtype, outer * count * inner);
  const source = x.data;
  for (let o = 0; o < outer; o += 1) {
    for (let k = 0; k < count; k += 1) {
      const sourceOffset = (o * size + resolved[k]!) * inner;
      const targetOffset = (o * count + k) * inner;
      for (let i = 0; i < inner; i += 1) out[targetOffset + i] = source[sourceOffset + i]!;
    }
  }
  const shape = [...x.shape];
  shape[d] = count;
  const result = fromStorage(out, shape, x.dtype);
  return attachGrad(result, [x], (grad) => {
    const gradIn = allocate(grad.dtype, x.numel);
    const g = grad.data;
    for (let o = 0; o < outer; o += 1) {
      for (let k = 0; k < count; k += 1) {
        const targetOffset = (o * size + resolved[k]!) * inner;
        const sourceOffset = (o * count + k) * inner;
        for (let i = 0; i < inner; i += 1) gradIn[targetOffset + i]! += g[sourceOffset + i]!;
      }
    }
    return [fromStorage(gradIn, x.shape, grad.dtype)];
  }, 'indexSelect');
}

/** ``torch.gather``: ``out[i][j][k] = x[i][index[i][j][k]][k]`` for ``dim=1``. */
export function gather(x: Tensor, dim: number, index: Tensor): Tensor {
  const d = normalizeDim(dim, x.ndim);
  if (index.ndim !== x.ndim) throw new RangeError('gather index must have the same number of dimensions');
  for (let axis = 0; axis < x.ndim; axis += 1) {
    if (axis !== d && index.shape[axis]! > x.shape[axis]!) throw new RangeError('gather index shape exceeds input');
  }
  const outShape = [...index.shape];
  const size = numelOf(outShape);
  const inStrides = stridesOf(x.shape);
  const idx = index.data;
  const source = x.data;
  const positions = new Int32Array(size);
  const ndim = outShape.length;
  const counter = new Array<number>(ndim).fill(0);
  for (let flat = 0; flat < size; flat += 1) {
    let offset = 0;
    for (let axis = 0; axis < ndim; axis += 1) {
      let coordinate = counter[axis]!;
      if (axis === d) {
        coordinate = idx[flat]!;
        if (coordinate < 0) coordinate += x.shape[d]!;
        if (!Number.isInteger(coordinate) || coordinate < 0 || coordinate >= x.shape[d]!) {
          throw new RangeError(`gather index ${idx[flat]} out of range`);
        }
      }
      offset += coordinate * inStrides[axis]!;
    }
    positions[flat] = offset;
    for (let axis = ndim - 1; axis >= 0; axis -= 1) {
      counter[axis]! += 1;
      if (counter[axis]! < outShape[axis]!) break;
      counter[axis] = 0;
    }
  }
  const out = allocate(x.dtype, size);
  for (let flat = 0; flat < size; flat += 1) out[flat] = source[positions[flat]!]!;
  const result = fromStorage(out, outShape, x.dtype);
  return attachGrad(result, [x], (grad) => {
    const gradIn = allocate(grad.dtype, x.numel);
    const g = grad.data;
    for (let flat = 0; flat < size; flat += 1) gradIn[positions[flat]!]! += g[flat]!;
    return [fromStorage(gradIn, x.shape, grad.dtype)];
  }, 'gather');
}

/**
 * Select entries where a boolean ``mask`` over the leading dimensions of ``x``
 * is true (``x[mask]``). The result has shape ``[count, ...trailing]``.
 */
export function maskedSelect(x: Tensor, mask: Tensor): Tensor {
  if (mask.ndim > x.ndim || !shapesEqual(mask.shape, x.shape.slice(0, mask.ndim))) {
    throw new RangeError(`mask shape ${formatShape(mask.shape)} must match the leading shape of ${formatShape(x.shape)}`);
  }
  const inner = numelOf(x.shape.slice(mask.ndim));
  const flags = mask.data;
  const rows: number[] = [];
  for (let index = 0; index < flags.length; index += 1) if (flags[index]) rows.push(index);
  const out = allocate(x.dtype, rows.length * inner);
  const source = x.data;
  rows.forEach((row, position) => {
    for (let i = 0; i < inner; i += 1) out[position * inner + i] = source[row * inner + i]!;
  });
  const result = fromStorage(out, [rows.length, ...x.shape.slice(mask.ndim)], x.dtype);
  return attachGrad(result, [x], (grad) => {
    const gradIn = allocate(grad.dtype, x.numel);
    const g = grad.data;
    rows.forEach((row, position) => {
      for (let i = 0; i < inner; i += 1) gradIn[row * inner + i] = g[position * inner + i]!;
    });
    return [fromStorage(gradIn, x.shape, grad.dtype)];
  }, 'maskedSelect');
}

// ------------------------------------------------------------------ joining

export function cat(tensors: readonly Tensor[], dim = 0): Tensor {
  if (!tensors.length) throw new RangeError('cat needs at least one tensor');
  const parts = tensors.filter((t) => !(t.ndim === 1 && t.numel === 0 && tensors.some((u) => u.ndim > 1)));
  const first = parts[0] ?? tensors[0]!;
  const d = normalizeDim(dim, first.ndim);
  let dtype = first.dtype;
  for (const part of parts) {
    if (part.ndim !== first.ndim) throw new RangeError('cat tensors must have the same number of dimensions');
    for (let axis = 0; axis < first.ndim; axis += 1) {
      if (axis !== d && part.shape[axis] !== first.shape[axis]) {
        throw new RangeError(`cat shape mismatch: ${formatShape(part.shape)} vs ${formatShape(first.shape)}`);
      }
    }
    dtype = promoteTypes(dtype, part.dtype);
  }
  const casted = parts.map((part) => cast(part, dtype));
  const total = casted.reduce((sum, part) => sum + part.shape[d]!, 0);
  const shape = [...first.shape];
  shape[d] = total;
  const [outer, , inner] = extents(shape, d);
  const out = allocate(dtype, numelOf(shape));
  let offset = 0;
  for (const part of casted) {
    const size = part.shape[d]!;
    const source = part.data;
    for (let o = 0; o < outer; o += 1) {
      const targetBase = (o * total + offset) * inner;
      const sourceBase = o * size * inner;
      for (let i = 0; i < size * inner; i += 1) out[targetBase + i] = source[sourceBase + i]!;
    }
    offset += size;
  }
  const result = fromStorage(out, shape, dtype);
  return attachGrad(result, casted, (grad) => {
    let start = 0;
    return casted.map((part) => {
      const size = part.shape[d]!;
      const piece = slice(grad, d, start, start + size);
      start += size;
      return piece;
    });
  }, 'cat');
}

export function stack(tensors: readonly Tensor[], dim = 0): Tensor {
  if (!tensors.length) throw new RangeError('stack needs at least one tensor');
  const d = normalizeDim(dim, tensors[0]!.ndim, true);
  for (const part of tensors) {
    if (!shapesEqual(part.shape, tensors[0]!.shape)) throw new RangeError('stack tensors must have equal shapes');
  }
  return cat(tensors.map((part) => unsqueeze(part, d)), d);
}

export function split(x: Tensor, sizes: number | readonly number[], dim = 0): Tensor[] {
  const d = normalizeDim(dim, x.ndim);
  const total = x.shape[d]!;
  const list: number[] = [];
  if (typeof sizes === 'number') {
    if (!Number.isInteger(sizes) || sizes < 1) throw new RangeError('split size must be positive');
    for (let start = 0; start < total; start += sizes) list.push(Math.min(sizes, total - start));
  } else {
    if (sizes.reduce((a, b) => a + b, 0) !== total) throw new RangeError('split sizes must sum to the dimension');
    list.push(...sizes);
  }
  let start = 0;
  return list.map((size) => {
    const piece = slice(x, d, start, start + size);
    start += size;
    return piece;
  });
}

export function chunk(x: Tensor, chunks: number, dim = 0): Tensor[] {
  const d = normalizeDim(dim, x.ndim);
  const size = Math.ceil(x.shape[d]! / chunks);
  return split(x, Math.max(size, 1), d);
}

export function unbind(x: Tensor, dim = 0): Tensor[] {
  const d = normalizeDim(dim, x.ndim);
  return Array.from({ length: x.shape[d]! }, (_, index) => select(x, d, index));
}

/** Pad a list of ``[length, ...]`` tensors into ``[batch, maxLength, ...]``. */
export function padSequence(sequences: readonly Tensor[], batchFirst = true, paddingValue = 0): Tensor {
  if (!sequences.length) throw new RangeError('padSequence needs at least one sequence');
  const trailing = sequences[0]!.shape.slice(1);
  const longest = Math.max(...sequences.map((sequence) => sequence.shape[0]!));
  const rows = sequences.map((sequence) => {
    if (!shapesEqual(sequence.shape.slice(1), trailing)) throw new RangeError('padSequence trailing shapes must match');
    const missing = longest - sequence.shape[0]!;
    if (!missing) return sequence;
    const padding = fromStorage(allocate(sequence.dtype, missing * numelOf(trailing)).fill(paddingValue) as Storage, [missing, ...trailing], sequence.dtype);
    return cat([sequence, padding], 0);
  });
  const stacked = stack(rows, 0);
  return batchFirst ? stacked : transpose(stacked, 0, 1);
}

/** Coordinate grids with ``indexing='ij'``. */
export function meshgrid(...vectors: Tensor[]): Tensor[] {
  const shape = vectors.map((vector) => vector.numel);
  return vectors.map((vector, axis) => {
    const view = shape.map((size, index) => (index === axis ? size : 1));
    return expand(reshape(vector, view), shape);
  });
}

// ------------------------------------------------------------------ ordering

function orderAlong(
  x: Tensor, dim: number, compare: (a: number, b: number) => number,
): { order: Int32Array; outer: number; size: number; inner: number } {
  const [outer, size, inner] = extents(x.shape, dim);
  const order = new Int32Array(outer * size * inner);
  const data = x.data;
  const scratch = new Array<number>(size);
  for (let o = 0; o < outer; o += 1) {
    for (let i = 0; i < inner; i += 1) {
      const base = o * size * inner + i;
      for (let k = 0; k < size; k += 1) scratch[k] = k;
      scratch.sort((a, b) => compare(data[base + a * inner]!, data[base + b * inner]!) || a - b);
      for (let k = 0; k < size; k += 1) order[base + k * inner] = scratch[k]!;
    }
  }
  return { order, outer, size, inner };
}

function descendingCompare(a: number, b: number): number {
  if (Number.isNaN(a)) return Number.isNaN(b) ? 0 : -1;
  if (Number.isNaN(b)) return 1;
  return b - a;
}

function ascendingCompare(a: number, b: number): number {
  if (Number.isNaN(a)) return Number.isNaN(b) ? 0 : 1;
  if (Number.isNaN(b)) return -1;
  return a - b;
}

/** Stable sort along ``dim``; values stay differentiable through ``gather``. */
export function sort(x: Tensor, dim = -1, descending = false): { values: Tensor; indices: Tensor } {
  const d = normalizeDim(dim, x.ndim);
  const { order } = orderAlong(x, d, descending ? descendingCompare : ascendingCompare);
  const indices = fromStorage(Float64Array.from(order), x.shape, 'int64');
  return { values: gather(x, d, indices), indices };
}

export function topk(x: Tensor, k: number, dim = -1, largest = true, sorted = true): { values: Tensor; indices: Tensor } {
  const d = normalizeDim(dim, x.ndim);
  const size = x.shape[d]!;
  if (!Number.isInteger(k) || k < 0 || k > size) throw new RangeError(`topk k=${k} out of range for size ${size}`);
  const ordered = sort(x.detach(), d, largest);
  let indices = slice(ordered.indices, d, 0, k);
  if (!sorted) indices = slice(ordered.indices, d, 0, k);
  return { values: gather(x, d, indices), indices };
}
