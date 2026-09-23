/** Shape arithmetic shared by tensor operations. */

export type Shape = readonly number[];

export function numelOf(shape: Shape): number {
  let size = 1;
  for (const dimension of shape) size *= dimension;
  return size;
}

export function stridesOf(shape: Shape): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let index = shape.length - 1; index >= 0; index -= 1) {
    strides[index] = stride;
    stride *= shape[index]!;
  }
  return strides;
}

export function shapesEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) if (a[index] !== b[index]) return false;
  return true;
}

export function formatShape(shape: Shape): string {
  return `[${shape.join(', ')}]`;
}

/** Normalize a possibly negative dimension index. */
export function normalizeDim(dim: number, ndim: number, allowEnd = false): number {
  if (!Number.isInteger(dim)) throw new TypeError(`dimension must be an integer, received ${dim}`);
  const limit = allowEnd ? ndim + 1 : ndim;
  const normalized = dim < 0 ? dim + limit : dim;
  if (normalized < 0 || normalized >= Math.max(limit, 1)) {
    throw new RangeError(`dimension ${dim} is out of range for a ${ndim}-dimensional tensor`);
  }
  return normalized;
}

export function normalizeDims(dims: number | readonly number[] | undefined | null, ndim: number): number[] {
  if (dims === undefined || dims === null) return Array.from({ length: ndim }, (_, index) => index);
  const list = typeof dims === 'number' ? [dims] : [...dims];
  const result = list.map((dim) => normalizeDim(dim, ndim));
  if (new Set(result).size !== result.length) throw new RangeError('repeated reduction dimension');
  return result.sort((a, b) => a - b);
}

export function validateShape(shape: Shape): number[] {
  const result = [...shape];
  for (const dimension of result) {
    if (!Number.isInteger(dimension) || dimension < 0) {
      throw new RangeError(`invalid tensor shape ${formatShape(shape)}`);
    }
  }
  return result;
}

export function broadcastShapes(...shapes: Shape[]): number[] {
  const ndim = Math.max(0, ...shapes.map((shape) => shape.length));
  const result = new Array<number>(ndim).fill(1);
  for (const shape of shapes) {
    const offset = ndim - shape.length;
    for (let index = 0; index < shape.length; index += 1) {
      const size = shape[index]!;
      const current = result[offset + index]!;
      if (size === current || size === 1) continue;
      if (current === 1) {
        result[offset + index] = size;
        continue;
      }
      throw new RangeError(`shapes ${shapes.map(formatShape).join(' and ')} cannot be broadcast`);
    }
  }
  return result;
}

/**
 * For every element of a contiguous tensor with ``outShape``, return the flat
 * offset of the corresponding element of a contiguous tensor with ``inShape``
 * that broadcasts to it.
 */
export function broadcastIndexMap(outShape: Shape, inShape: Shape): Int32Array {
  const size = numelOf(outShape);
  const map = new Int32Array(size);
  const ndim = outShape.length;
  const offset = ndim - inShape.length;
  const inStrides = stridesOf(inShape);
  const effective = new Array<number>(ndim).fill(0);
  for (let index = 0; index < inShape.length; index += 1) {
    effective[offset + index] = inShape[index] === 1 ? 0 : inStrides[index]!;
  }
  if (size === 0) return map;
  const counter = new Array<number>(ndim).fill(0);
  let position = 0;
  for (let flat = 0; flat < size; flat += 1) {
    map[flat] = position;
    for (let dim = ndim - 1; dim >= 0; dim -= 1) {
      counter[dim]! += 1;
      position += effective[dim]!;
      if (counter[dim]! < outShape[dim]!) break;
      position -= effective[dim]! * counter[dim]!;
      counter[dim] = 0;
    }
  }
  return map;
}

/** Resolve a single ``-1`` entry in a reshape target. */
export function inferShape(shape: Shape, size: number): number[] {
  const result = [...shape];
  const unknown = result.indexOf(-1);
  if (unknown !== result.lastIndexOf(-1)) throw new RangeError('only one dimension can be inferred');
  if (unknown >= 0) {
    const known = result.reduce((product, value, index) => (index === unknown ? product : product * value), 1);
    if (known === 0 || size % known !== 0) {
      throw new RangeError(`cannot reshape ${size} elements into ${formatShape(shape)}`);
    }
    result[unknown] = size / known;
  }
  validateShape(result);
  if (numelOf(result) !== size) throw new RangeError(`cannot reshape ${size} elements into ${formatShape(shape)}`);
  return result;
}
