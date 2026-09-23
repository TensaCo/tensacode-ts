/**
 * Dense CPU tensors with reverse-mode automatic differentiation.
 *
 * Every tensor owns (or shares, for reshape-style views) a contiguous row-major
 * storage. Operations that change element order (permute, expand, slice, ...)
 * copy. Reshape, view, flatten, squeeze, unsqueeze and detach share storage, so
 * in-place writes are visible through those aliases and bump the shared
 * version counter used by tracing to detect mutation.
 */
import {
  allocate, castValue, isFloatingDType, promoteScalar, roundToDType, usesFloat32Storage,
  type DType, type Storage,
} from './dtype.js';
import { isGradEnabled, noGrad, type GradNode } from './autograd.js';
import { getDefaultGenerator, type Generator } from './random.js';
import {
  formatShape, inferShape, normalizeDim, numelOf, shapesEqual, validateShape, type Shape,
} from './shape.js';
import * as E from './ops/elementwise.js';
import * as R from './ops/reduce.js';
import * as S from './ops/shape.js';
import * as L from './ops/linalg.js';
import * as N from './ops/nn.js';

export type { DType, Storage } from './dtype.js';
export type { Shape } from './shape.js';

export type NestedNumbers = number | boolean | readonly NestedNumbers[];
export type Operand = Tensor | number;

interface TensorStorage {
  data: Storage;
  version: number;
}

let nextTensorId = 1;

export interface TensorOptions {
  dtype?: DType;
  requiresGrad?: boolean;
}

export class Tensor {
  /** Stable identity (diagnostics, alias detection). */
  readonly id: number;
  readonly shape: readonly number[];
  readonly dtype: DType;
  /** @internal shared storage record */
  _storage: TensorStorage;
  /** @internal */
  _requiresGrad = false;
  /** @internal */
  _gradFn: GradNode | null = null;
  /** @internal */
  _retainGrad = false;
  /** Accumulated gradient of a leaf tensor after ``backward``. */
  grad: Tensor | null = null;

  constructor(data: Storage, shape: Shape, dtype: DType = 'float32', storage?: TensorStorage) {
    const validated = validateShape(shape);
    if (data.length !== numelOf(validated)) {
      throw new RangeError(`storage has ${data.length} elements but shape ${formatShape(validated)} needs ${numelOf(validated)}`);
    }
    if (usesFloat32Storage(dtype) !== (data instanceof Float32Array)) {
      throw new TypeError(`dtype ${dtype} requires ${usesFloat32Storage(dtype) ? 'Float32Array' : 'Float64Array'} storage`);
    }
    this.id = nextTensorId++;
    this.shape = Object.freeze(validated);
    this.dtype = dtype;
    this._storage = storage ?? { data, version: 0 };
  }

  // -------------------------------------------------------------- metadata
  get data(): Storage {
    return this._storage.data;
  }

  /** Version counter bumped by in-place writes (shared by storage aliases). */
  get version(): number {
    return this._storage.version;
  }

  get ndim(): number {
    return this.shape.length;
  }

  get numel(): number {
    return this._storage.data.length;
  }

  get requiresGrad(): boolean {
    return this._requiresGrad;
  }

  set requiresGrad(value: boolean) {
    if (value && !isFloatingDType(this.dtype)) {
      throw new TypeError('only floating point tensors can require gradients');
    }
    if (this._gradFn !== null && !value) {
      throw new Error('cannot disable gradients of a non-leaf tensor; use detach()');
    }
    this._requiresGrad = value;
  }

  get gradFn(): GradNode | null {
    return this._gradFn;
  }

  get isLeaf(): boolean {
    return this._gradFn === null;
  }

  get isFloatingPoint(): boolean {
    return isFloatingDType(this.dtype);
  }

  size(dim?: number): number {
    if (dim === undefined) return this.numel;
    return this.shape[normalizeDim(dim, this.ndim)]!;
  }

  /** True when both tensors share the same storage record. */
  sharesStorage(other: Tensor): boolean {
    return this._storage === other._storage;
  }

  toString(): string {
    return `Tensor(shape=${formatShape(this.shape)}, dtype=${this.dtype}${this.requiresGrad ? ', requiresGrad' : ''})`;
  }

  // -------------------------------------------------------------- reading
  /** The single value of a one-element tensor. */
  item(): number {
    if (this.numel !== 1) throw new RangeError(`item() requires one element, tensor has ${this.numel}`);
    return this.data[0]!;
  }

  /** Read one element by (possibly negative) coordinates. */
  get(...indices: number[]): number {
    if (indices.length !== this.ndim) throw new RangeError('get() needs one index per dimension');
    let offset = 0;
    let stride = 1;
    for (let dim = this.ndim - 1; dim >= 0; dim -= 1) {
      const size = this.shape[dim]!;
      let index = indices[dim]!;
      if (index < 0) index += size;
      if (!Number.isInteger(index) || index < 0 || index >= size) throw new RangeError('index out of range');
      offset += index * stride;
      stride *= size;
    }
    return this.data[offset]!;
  }

  /** Nested JavaScript arrays (booleans for bool tensors). */
  tolist(): NestedNumbers {
    const data = this.data;
    const isBool = this.dtype === 'bool';
    const build = (dim: number, offset: number): NestedNumbers => {
      if (dim === this.ndim) return isBool ? data[offset] !== 0 : data[offset]!;
      const size = this.shape[dim]!;
      const stride = numelOf(this.shape.slice(dim + 1));
      const result: NestedNumbers[] = [];
      for (let index = 0; index < size; index += 1) result.push(build(dim + 1, offset + index * stride));
      return result;
    };
    return build(0, 0);
  }

  /** Flat copy of the values as a plain array. */
  toArray(): number[] {
    return Array.from(this.data);
  }

  // -------------------------------------------------------------- autograd
  /** Compute gradients of this tensor with respect to leaf tensors. */
  backward(gradient?: Tensor): void {
    runBackward(this, gradient);
  }

  /** Keep ``grad`` for this non-leaf tensor during ``backward``. */
  retainGrad(): this {
    this._retainGrad = true;
    return this;
  }

  requiresGrad_(value = true): this {
    this.requiresGrad = value;
    return this;
  }

  /**
   * @internal Replace this tensor's values and dtype in place (PyTorch
   * ``param.data = param.data.to(dtype)``): identity, and therefore module ties
   * and optimizer references, are preserved. The storage version advances so
   * traces see the change; the gradient is dropped.
   */
  _replaceData(data: Storage, dtype: DType): void {
    if (this._gradFn !== null) throw new Error('only leaf tensors can replace their data');
    if (data.length !== this.numel) throw new RangeError('replacement data must keep the element count');
    if (usesFloat32Storage(dtype) !== (data instanceof Float32Array)) {
      throw new TypeError(`dtype ${dtype} requires ${usesFloat32Storage(dtype) ? 'Float32Array' : 'Float64Array'} storage`);
    }
    const version = this._storage.version + 1;
    (this as { dtype: DType }).dtype = dtype;
    this._storage = { data, version };
    this.grad = null;
  }

  /** A tensor sharing storage but excluded from gradient recording. */
  detach(): Tensor {
    return new Tensor(this.data, this.shape, this.dtype, this._storage);
  }

  // -------------------------------------------------------------- in-place
  private checkInPlace(): void {
    if (this._requiresGrad && this._gradFn === null && isGradEnabled()) {
      throw new Error('a leaf tensor that requires grad cannot be modified in place outside noGrad()');
    }
  }

  private bump(): this {
    this._storage.version += 1;
    return this;
  }

  fill_(value: number): this {
    this.checkInPlace();
    this.data.fill(castValue(this.dtype, roundToDType(this.dtype, value)));
    return this.bump();
  }

  zero_(): this {
    return this.fill_(0);
  }

  /** Copy (broadcast) values from ``source`` into this tensor. */
  copy_(source: Tensor | NestedNumbers): this {
    this.checkInPlace();
    const tensorSource = source instanceof Tensor ? source : tensor(source as NestedNumbers, { dtype: this.dtype });
    const expanded = shapesEqual(tensorSource.shape, this.shape)
      ? tensorSource
      : noGrad(() => tensorSource.expand(this.shape));
    const target = this.data;
    const values = expanded.data;
    for (let index = 0; index < target.length; index += 1) {
      target[index] = castValue(this.dtype, roundToDType(this.dtype, values[index]!));
    }
    return this.bump();
  }

  add_(other: Operand, alpha = 1): this {
    this.checkInPlace();
    const target = this.data;
    if (typeof other === 'number') {
      for (let index = 0; index < target.length; index += 1) target[index]! += alpha * other;
    } else {
      const source = shapesEqual(other.shape, this.shape) ? other : noGrad(() => other.expand(this.shape));
      const values = source.data;
      for (let index = 0; index < target.length; index += 1) target[index]! += alpha * values[index]!;
    }
    return this.bump();
  }

  sub_(other: Operand, alpha = 1): this {
    return this.add_(other, -alpha);
  }

  mul_(other: Operand): this {
    this.checkInPlace();
    const target = this.data;
    if (typeof other === 'number') {
      for (let index = 0; index < target.length; index += 1) target[index]! *= other;
    } else {
      const source = shapesEqual(other.shape, this.shape) ? other : noGrad(() => other.expand(this.shape));
      const values = source.data;
      for (let index = 0; index < target.length; index += 1) target[index]! *= values[index]!;
    }
    return this.bump();
  }

  div_(other: Operand): this {
    if (typeof other === 'number') return this.mul_(1 / other);
    this.checkInPlace();
    const source = shapesEqual(other.shape, this.shape) ? other : noGrad(() => other.expand(this.shape));
    const target = this.data;
    const values = source.data;
    for (let index = 0; index < target.length; index += 1) target[index]! /= values[index]!;
    return this.bump();
  }

  /** Elementwise ``this += value * a * b`` (optimizer helper). */
  addcmul_(a: Tensor, b: Tensor, value = 1): this {
    this.checkInPlace();
    const target = this.data;
    const x = a.data;
    const y = b.data;
    for (let index = 0; index < target.length; index += 1) target[index]! += value * x[index]! * y[index]!;
    return this.bump();
  }

  /** Elementwise ``this += value * a / b`` (optimizer helper). */
  addcdiv_(a: Tensor, b: Tensor, value = 1): this {
    this.checkInPlace();
    const target = this.data;
    const x = a.data;
    const y = b.data;
    for (let index = 0; index < target.length; index += 1) target[index]! += value * x[index]! / y[index]!;
    return this.bump();
  }

  clampMin_(minimum: number): this {
    this.checkInPlace();
    const target = this.data;
    for (let index = 0; index < target.length; index += 1) if (target[index]! < minimum) target[index] = minimum;
    return this.bump();
  }

  maskedFill_(mask: Tensor, value: number): this {
    this.checkInPlace();
    const expanded = shapesEqual(mask.shape, this.shape) ? mask : noGrad(() => mask.expand(this.shape));
    const target = this.data;
    const flags = expanded.data;
    for (let index = 0; index < target.length; index += 1) if (flags[index]) target[index] = value;
    return this.bump();
  }

  normal_(mean = 0, std = 1, generator: Generator = getDefaultGenerator()): this {
    this.checkInPlace();
    const target = this.data;
    for (let index = 0; index < target.length; index += 1) {
      target[index] = roundToDType(this.dtype, mean + std * generator.normal());
    }
    return this.bump();
  }

  uniform_(low = 0, high = 1, generator: Generator = getDefaultGenerator()): this {
    this.checkInPlace();
    const target = this.data;
    for (let index = 0; index < target.length; index += 1) {
      target[index] = roundToDType(this.dtype, low + (high - low) * generator.random());
    }
    return this.bump();
  }

  // -------------------------------------------------------------- conversion
  clone(): Tensor { return E.clone(this); }
  to(dtype: DType): Tensor { return E.cast(this, dtype); }
  float(): Tensor { return E.cast(this, 'float32'); }
  double(): Tensor { return E.cast(this, 'float64'); }
  long(): Tensor { return E.cast(this, 'int64'); }
  bool(): Tensor { return E.cast(this, 'bool'); }
  contiguous(): Tensor { return this; }

  // -------------------------------------------------------------- arithmetic
  add(other: Operand, alpha = 1): Tensor { return E.add(this, alpha === 1 ? other : E.mulOperand(other, alpha)); }
  sub(other: Operand, alpha = 1): Tensor { return E.sub(this, alpha === 1 ? other : E.mulOperand(other, alpha)); }
  mul(other: Operand): Tensor { return E.mul(this, other); }
  div(other: Operand): Tensor { return E.div(this, other); }
  pow(exponent: Operand): Tensor { return E.pow(this, exponent); }
  neg(): Tensor { return E.neg(this); }
  exp(): Tensor { return E.exp(this); }
  log(): Tensor { return E.log(this); }
  log1p(): Tensor { return E.log1p(this); }
  sqrt(): Tensor { return E.sqrt(this); }
  rsqrt(): Tensor { return E.rsqrt(this); }
  abs(): Tensor { return E.abs(this); }
  sign(): Tensor { return E.sign(this); }
  square(): Tensor { return E.square(this); }
  reciprocal(): Tensor { return E.reciprocal(this); }
  tanh(): Tensor { return E.tanh(this); }
  sigmoid(): Tensor { return E.sigmoid(this); }
  relu(): Tensor { return E.relu(this); }
  sin(): Tensor { return E.sin(this); }
  cos(): Tensor { return E.cos(this); }
  erf(): Tensor { return E.erf(this); }
  floor(): Tensor { return E.floor(this); }
  ceil(): Tensor { return E.ceil(this); }
  round(): Tensor { return E.round(this); }
  maximum(other: Operand): Tensor { return E.maximum(this, other); }
  minimum(other: Operand): Tensor { return E.minimum(this, other); }
  clamp(min?: number | null, max?: number | null): Tensor { return E.clamp(this, min ?? null, max ?? null); }
  clampMin(min: number): Tensor { return E.clamp(this, min, null); }
  clampMax(max: number): Tensor { return E.clamp(this, null, max); }
  maskedFill(mask: Tensor, value: number): Tensor { return E.maskedFill(this, mask, value); }

  // -------------------------------------------------------------- comparison
  eq(other: Operand): Tensor { return E.compare(this, other, 'eq'); }
  ne(other: Operand): Tensor { return E.compare(this, other, 'ne'); }
  lt(other: Operand): Tensor { return E.compare(this, other, 'lt'); }
  le(other: Operand): Tensor { return E.compare(this, other, 'le'); }
  gt(other: Operand): Tensor { return E.compare(this, other, 'gt'); }
  ge(other: Operand): Tensor { return E.compare(this, other, 'ge'); }
  logicalNot(): Tensor { return E.logicalNot(this); }
  logicalAnd(other: Tensor): Tensor { return E.logical(this, other, 'and'); }
  logicalOr(other: Tensor): Tensor { return E.logical(this, other, 'or'); }
  isFinite(): Tensor { return E.isFinite(this); }
  isNan(): Tensor { return E.isNan(this); }
  /** True when every element is finite. */
  allFinite(): boolean { return E.allFinite(this); }
  /** Exact elementwise equality of shape, dtype family and values. */
  equal(other: Tensor): boolean { return E.equal(this, other); }

  // -------------------------------------------------------------- reductions
  sum(dim?: number | readonly number[] | null, keepdim = false): Tensor { return R.sum(this, dim, keepdim); }
  mean(dim?: number | readonly number[] | null, keepdim = false): Tensor { return R.mean(this, dim, keepdim); }
  /** Maximum over all elements (0-d tensor). */
  max(): Tensor;
  /** Maximum values and first indices along ``dim``. */
  max(dim: number, keepdim?: boolean): { values: Tensor; indices: Tensor };
  max(dim?: number, keepdim = false): Tensor | { values: Tensor; indices: Tensor } {
    return dim === undefined ? R.amax(this, null, false) : R.maxDim(this, dim, keepdim, true);
  }
  min(): Tensor;
  min(dim: number, keepdim?: boolean): { values: Tensor; indices: Tensor };
  min(dim?: number, keepdim = false): Tensor | { values: Tensor; indices: Tensor } {
    return dim === undefined ? R.amin(this, null, false) : R.maxDim(this, dim, keepdim, false);
  }
  amax(dim?: number | readonly number[] | null, keepdim = false): Tensor { return R.amax(this, dim, keepdim); }
  amin(dim?: number | readonly number[] | null, keepdim = false): Tensor { return R.amin(this, dim, keepdim); }
  argmax(dim?: number, keepdim = false): Tensor { return R.argExtreme(this, dim, keepdim, true); }
  argmin(dim?: number, keepdim = false): Tensor { return R.argExtreme(this, dim, keepdim, false); }
  logsumexp(dim: number | readonly number[], keepdim = false): Tensor { return R.logsumexp(this, dim, keepdim); }
  softmax(dim: number): Tensor { return R.softmax(this, dim); }
  logSoftmax(dim: number): Tensor { return R.logSoftmax(this, dim); }
  cumsum(dim: number): Tensor { return R.cumsum(this, dim); }
  any(dim?: number | null, keepdim = false): Tensor { return R.anyAll(this, dim, keepdim, 'any'); }
  all(dim?: number | null, keepdim = false): Tensor { return R.anyAll(this, dim, keepdim, 'all'); }
  norm(p = 2, dim?: number | readonly number[] | null, keepdim = false): Tensor { return R.norm(this, p, dim, keepdim); }
  var(dim?: number | readonly number[] | null, keepdim = false, unbiased = true): Tensor {
    return R.variance(this, dim, keepdim, unbiased);
  }

  // -------------------------------------------------------------- shape
  reshape(...shape: number[] | [readonly number[]]): Tensor { return S.reshape(this, flattenShapeArgs(shape)); }
  view(...shape: number[] | [readonly number[]]): Tensor { return S.reshape(this, flattenShapeArgs(shape)); }
  flatten(startDim = 0, endDim = -1): Tensor { return S.flatten(this, startDim, endDim); }
  unsqueeze(dim: number): Tensor { return S.unsqueeze(this, dim); }
  squeeze(dim?: number): Tensor { return S.squeeze(this, dim); }
  permute(...dims: number[] | [readonly number[]]): Tensor { return S.permute(this, flattenShapeArgs(dims)); }
  transpose(dim0: number, dim1: number): Tensor { return S.transpose(this, dim0, dim1); }
  /** Swap the last two dimensions (``tensor.mT``). */
  get mT(): Tensor { return S.transpose(this, -2, -1); }
  expand(...shape: number[] | [readonly number[]]): Tensor { return S.expand(this, flattenShapeArgs(shape)); }
  expandAs(other: Tensor): Tensor { return S.expand(this, other.shape); }
  repeat(...counts: number[] | [readonly number[]]): Tensor { return S.repeat(this, flattenShapeArgs(counts)); }
  /** ``tensor[start:end:step]`` along ``dim`` (negative indices allowed). */
  slice(dim: number, start?: number | null, end?: number | null, step = 1): Tensor {
    return S.slice(this, dim, start ?? null, end ?? null, step);
  }
  narrow(dim: number, start: number, length: number): Tensor { return S.slice(this, dim, start, start + length, 1); }
  /** ``tensor[..., index, ...]`` removing ``dim``. */
  select(dim: number, index: number): Tensor { return S.select(this, dim, index); }
  indexSelect(dim: number, index: Tensor | readonly number[]): Tensor { return S.indexSelect(this, dim, index); }
  gather(dim: number, index: Tensor): Tensor { return S.gather(this, dim, index); }
  /** Rows selected by a boolean mask over the leading dimensions (``x[mask]``). */
  maskedSelect(mask: Tensor): Tensor { return S.maskedSelect(this, mask); }
  split(sizes: number | readonly number[], dim = 0): Tensor[] { return S.split(this, sizes, dim); }
  chunk(chunks: number, dim = 0): Tensor[] { return S.chunk(this, chunks, dim); }
  unbind(dim = 0): Tensor[] { return S.unbind(this, dim); }
  topk(k: number, dim = -1, largest = true, sorted = true): { values: Tensor; indices: Tensor } {
    return S.topk(this, k, dim, largest, sorted);
  }
  sort(dim = -1, descending = false): { values: Tensor; indices: Tensor } { return S.sort(this, dim, descending); }
  argsort(dim = -1, descending = false): Tensor { return S.sort(this, dim, descending).indices; }

  // -------------------------------------------------------------- linear algebra / nn
  matmul(other: Tensor): Tensor { return L.matmul(this, other); }
  gelu(approximate: 'none' | 'tanh' = 'none'): Tensor { return N.gelu(this, approximate); }
}

function flattenShapeArgs(args: number[] | [readonly number[]]): number[] {
  if (args.length === 1 && Array.isArray(args[0])) return [...(args[0] as readonly number[])];
  return args as number[];
}

/** Tensor that is registered as a trainable module parameter. */
export class Parameter extends Tensor {
  constructor(source: Tensor, requiresGrad = true) {
    const data = source.data.slice() as Storage;
    super(data, source.shape, source.dtype);
    if (requiresGrad) this.requiresGrad = true;
  }
}

// ---------------------------------------------------------------------------
// Autograd plumbing used by operation implementations.
// ---------------------------------------------------------------------------

/**
 * Record ``backward`` on ``result`` when gradients are enabled and any input
 * requires them. Returns ``result`` for chaining.
 */
export function attachGrad(
  result: Tensor,
  inputs: readonly (Tensor | null)[],
  backward: (grad: Tensor) => readonly (Tensor | null | undefined)[],
  name: string,
): Tensor {
  if (!isGradEnabled() || !isFloatingDType(result.dtype)) return result;
  let needed = false;
  for (const input of inputs) if (input !== null && input._requiresGrad) needed = true;
  if (!needed) return result;
  result._requiresGrad = true;
  result._gradFn = { name, inputs, backward };
  return result;
}

function runBackward(root: Tensor, gradient?: Tensor): void {
  if (!root._requiresGrad) throw new Error('tensor does not require gradients and has no gradient function');
  let seed = gradient;
  if (seed === undefined) {
    if (root.numel !== 1) throw new Error('backward() without a gradient requires a scalar tensor');
    seed = full(root.shape, 1, { dtype: root.dtype });
  } else if (!shapesEqual(seed.shape, root.shape)) {
    throw new RangeError('gradient shape must match the tensor');
  }
  // Iterative post-order DFS over tensors with gradient functions.
  const order: Tensor[] = [];
  const visited = new Set<Tensor>();
  const stack: [Tensor, number][] = [[root, 0]];
  visited.add(root);
  while (stack.length) {
    const top = stack[stack.length - 1]!;
    const [node, childIndex] = top;
    const inputs = node._gradFn?.inputs ?? [];
    if (childIndex < inputs.length) {
      top[1] += 1;
      const child = inputs[childIndex];
      if (child && child._requiresGrad && !visited.has(child)) {
        visited.add(child);
        stack.push([child, 0]);
      }
    } else {
      stack.pop();
      order.push(node);
    }
  }
  const grads = new Map<Tensor, Tensor>();
  grads.set(root, seed);
  noGrad(() => {
    for (let index = order.length - 1; index >= 0; index -= 1) {
      const node = order[index]!;
      const grad = grads.get(node);
      if (!grad) continue;
      grads.delete(node);
      const fn = node._gradFn;
      if (fn === null || node._retainGrad) {
        const stored = grad.dtype === node.dtype ? grad : grad.to(node.dtype);
        node.grad = node.grad === null ? detachedCopy(stored) : accumulate(node.grad, stored);
      }
      if (fn === null) continue;
      const inputGrads = fn.backward(grad);
      fn.inputs.forEach((input, position) => {
        const inputGrad = inputGrads[position];
        if (!input || !input._requiresGrad || inputGrad === null || inputGrad === undefined) return;
        if (!shapesEqual(inputGrad.shape, input.shape)) {
          throw new Error(`internal gradient shape mismatch in ${fn.name}: ${formatShape(inputGrad.shape)} vs ${formatShape(input.shape)}`);
        }
        const existing = grads.get(input);
        grads.set(input, existing ? existing.add(inputGrad) : inputGrad);
      });
    }
  });
}

function detachedCopy(value: Tensor): Tensor {
  return new Tensor(value.data.slice() as Storage, value.shape, value.dtype);
}

function accumulate(existing: Tensor, addition: Tensor): Tensor {
  const result = detachedCopy(existing);
  const target = result.data;
  const values = addition.data;
  for (let index = 0; index < target.length; index += 1) target[index]! += values[index]!;
  return result;
}

// ---------------------------------------------------------------------------
// Creation.
// ---------------------------------------------------------------------------

function inferNested(values: NestedNumbers): { shape: number[]; flat: number[]; boolean: boolean } {
  const shape: number[] = [];
  let probe: NestedNumbers = values;
  while (Array.isArray(probe)) {
    shape.push(probe.length);
    probe = probe.length ? probe[0]! : 0;
  }
  const flat: number[] = [];
  let boolean = typeof probe === 'boolean';
  const walk = (value: NestedNumbers, depth: number): void => {
    if (depth === shape.length) {
      if (Array.isArray(value)) throw new RangeError('ragged nested array');
      if (typeof value === 'boolean') flat.push(value ? 1 : 0);
      else if (typeof value === 'number') {
        boolean = false;
        flat.push(value);
      } else throw new TypeError('tensor values must be numbers or booleans');
      return;
    }
    if (!Array.isArray(value) || value.length !== shape[depth]) throw new RangeError('ragged nested array');
    for (const item of value as readonly NestedNumbers[]) walk(item, depth + 1);
  };
  walk(values, 0);
  return { shape, flat, boolean };
}

/**
 * Create a tensor from nested arrays, a typed array (with ``shape``) or a
 * number. Numbers default to float32; booleans default to ``bool``.
 */
export function tensor(
  values: NestedNumbers | ArrayLike<number>,
  options: TensorOptions & { shape?: Shape } = {},
): Tensor {
  let shape: number[];
  let flat: ArrayLike<number>;
  let inferred: DType = 'float32';
  if (ArrayBuffer.isView(values)) {
    flat = values as unknown as ArrayLike<number>;
    shape = options.shape ? [...options.shape] : [flat.length];
  } else {
    const nested = inferNested(values as NestedNumbers);
    shape = options.shape ? [...options.shape] : nested.shape;
    flat = nested.flat;
    if (nested.boolean && nested.flat.length) inferred = 'bool';
  }
  const dtype = options.dtype ?? inferred;
  const data = allocate(dtype, flat.length);
  for (let index = 0; index < flat.length; index += 1) data[index] = castValue(dtype, roundToDType(dtype, flat[index]!));
  const result = new Tensor(data, shape, dtype);
  if (options.requiresGrad) result.requiresGrad = true;
  return result;
}

/** A 0-dimensional tensor. */
export function scalar(value: number, dtype: DType = 'float32'): Tensor {
  return tensor(value, { dtype });
}

export function full(shape: Shape, value: number, options: TensorOptions = {}): Tensor {
  const dtype = options.dtype ?? promoteScalar('float32', value);
  const resolved = validateShape(shape);
  const data = allocate(dtype, numelOf(resolved));
  data.fill(castValue(dtype, roundToDType(dtype, value)));
  const result = new Tensor(data, resolved, dtype);
  if (options.requiresGrad) result.requiresGrad = true;
  return result;
}

export function zeros(shape: Shape, options: TensorOptions = {}): Tensor {
  return full(shape, 0, { dtype: 'float32', ...options });
}

export function ones(shape: Shape, options: TensorOptions = {}): Tensor {
  return full(shape, 1, { dtype: 'float32', ...options });
}

export function empty(shape: Shape, options: TensorOptions = {}): Tensor {
  return zeros(shape, options);
}

export function zerosLike(like: Tensor, options: TensorOptions = {}): Tensor {
  return full(like.shape, 0, { dtype: like.dtype, ...options });
}

export function onesLike(like: Tensor, options: TensorOptions = {}): Tensor {
  return full(like.shape, 1, { dtype: like.dtype, ...options });
}

export function fullLike(like: Tensor, value: number, options: TensorOptions = {}): Tensor {
  return full(like.shape, value, { dtype: like.dtype, ...options });
}

/** Values ``start, start+step, ...`` below ``end`` (int64 unless a float step/bound). */
export function arange(start: number, end?: number, step = 1, options: TensorOptions = {}): Tensor {
  let low = start;
  let high = end;
  if (high === undefined) {
    high = low;
    low = 0;
  }
  if (step === 0) throw new RangeError('arange step must be nonzero');
  const count = Math.max(0, Math.ceil((high - low) / step));
  const integral = Number.isInteger(low) && Number.isInteger(high) && Number.isInteger(step);
  const dtype = options.dtype ?? (integral ? 'int64' : 'float32');
  const data = allocate(dtype, count);
  for (let index = 0; index < count; index += 1) data[index] = castValue(dtype, low + index * step);
  return new Tensor(data, [count], dtype);
}

export function linspace(start: number, end: number, steps: number, options: TensorOptions = {}): Tensor {
  const dtype = options.dtype ?? 'float32';
  const data = allocate(dtype, steps);
  for (let index = 0; index < steps; index += 1) {
    data[index] = steps === 1 ? start : start + ((end - start) * index) / (steps - 1);
  }
  return new Tensor(data, [steps], dtype);
}

export function eye(size: number, options: TensorOptions = {}): Tensor {
  const result = zeros([size, size], options);
  for (let index = 0; index < size; index += 1) result.data[index * size + index] = 1;
  return result;
}

export interface RandomOptions extends TensorOptions {
  generator?: Generator;
}

export function randn(shape: Shape, options: RandomOptions = {}): Tensor {
  const result = zeros(shape, { dtype: options.dtype ?? 'float32' });
  noGrad(() => result.normal_(0, 1, options.generator));
  result._storage.version = 0;
  if (options.requiresGrad) result.requiresGrad = true;
  return result;
}

export function rand(shape: Shape, options: RandomOptions = {}): Tensor {
  const result = zeros(shape, { dtype: options.dtype ?? 'float32' });
  noGrad(() => result.uniform_(0, 1, options.generator));
  result._storage.version = 0;
  if (options.requiresGrad) result.requiresGrad = true;
  return result;
}

/** Uniform integers in [low, high). */
export function randint(low: number, high: number, shape: Shape, options: RandomOptions = {}): Tensor {
  const generator = options.generator ?? getDefaultGenerator();
  const dtype = options.dtype ?? 'int64';
  const result = zeros(shape, { dtype });
  const data = result.data;
  for (let index = 0; index < data.length; index += 1) data[index] = generator.integer(low, high);
  return result;
}

export function randnLike(like: Tensor, options: RandomOptions = {}): Tensor {
  return randn(like.shape, { dtype: like.dtype, ...options });
}

/** Internal: wrap storage produced by an operation. */
export function fromStorage(data: Storage, shape: Shape, dtype: DType): Tensor {
  return new Tensor(data, shape, dtype);
}

/** Internal: a new tensor sharing ``source`` storage with a different shape. */
export function aliasWithShape(source: Tensor, shape: Shape): Tensor {
  return new Tensor(source.data, inferShape(shape, source.numel), source.dtype, source._storage);
}

export function isTensor(value: unknown): value is Tensor {
  return value instanceof Tensor;
}
