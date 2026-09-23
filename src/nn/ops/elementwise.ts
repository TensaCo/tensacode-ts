/** Elementwise arithmetic, comparison and selection with broadcasting. */
import {
  allocate, castValue, isFloatingDType, promoteScalar, promoteTypes, roundToDType, type DType, type Storage,
} from '../dtype.js';
import { broadcastIndexMap, broadcastShapes, numelOf, shapesEqual, type Shape } from '../shape.js';
import { Tensor, attachGrad, fromStorage, type Operand } from '../tensor.js';
import { sumToShape } from './reduce.js';
import { binary, unary, UnaryOp, type BinaryOp } from '../backend/kernels.js';

// ------------------------------------------------------------------ helpers

/**
 * Round float16/bfloat16 results held in float32 storage to their dtype, as
 * PyTorch's CPU kernels do after computing in float32 (other dtypes unchanged).
 */
export function roundHalf<T extends Storage>(dtype: DType, data: T): T {
  if (dtype === 'float16' || dtype === 'bfloat16') {
    for (let index = 0; index < data.length; index += 1) data[index] = roundToDType(dtype, data[index]!);
  }
  return data;
}

function roundStorage(dtype: DType, data: Storage): Storage {
  if (dtype === 'float16' || dtype === 'bfloat16') {
    for (let index = 0; index < data.length; index += 1) data[index] = roundToDType(dtype, data[index]!);
  } else if (!isFloatingDType(dtype)) {
    for (let index = 0; index < data.length; index += 1) data[index] = castValue(dtype, data[index]!);
  }
  return data;
}

/** Apply ``fn`` to every element; output dtype defaults to the input dtype. */
export function mapUnary(x: Tensor, fn: (value: number) => number, dtype: DType = x.dtype): Tensor {
  const source = x.data;
  const out = allocate(dtype, source.length);
  for (let index = 0; index < source.length; index += 1) out[index] = fn(source[index]!);
  return fromStorage(roundStorage(dtype, out), x.shape, dtype);
}

function floatUnary(x: Tensor, fn: (value: number) => number): Tensor {
  const dtype = isFloatingDType(x.dtype) ? x.dtype : 'float32';
  return mapUnary(x, fn, dtype);
}

/**
 * Broadcast walk over ``shape``: calls ``row(offsetA, offsetB, offsetOut,
 * length, strideA, strideB)`` for every run along the last dimension, where the
 * strides are 1 (or 0 for a broadcast operand).
 */
function broadcastRows(
  a: Tensor, b: Tensor, shape: Shape,
  row: (offsetA: number, offsetB: number, offsetOut: number, length: number, strideA: number, strideB: number) => void,
): void {
  const size = numelOf(shape);
  if (size === 0) return;
  const ndim = shape.length;
  if (ndim === 0) {
    row(0, 0, 0, 1, 0, 0);
    return;
  }
  const effective = (input: readonly number[]): number[] => {
    const strides = new Array<number>(ndim).fill(0);
    let stride = 1;
    for (let axis = input.length - 1; axis >= 0; axis -= 1) {
      const target = ndim - input.length + axis;
      strides[target] = input[axis] === 1 && shape[target] !== 1 ? 0 : stride;
      stride *= input[axis]!;
    }
    return strides;
  };
  const sa = effective(a.shape);
  const sb = effective(b.shape);
  const inner = shape[ndim - 1]!;
  const counter = new Array<number>(ndim).fill(0);
  let pa = 0;
  let pb = 0;
  for (let out = 0; out < size; out += inner) {
    row(pa, pb, out, inner, sa[ndim - 1]!, sb[ndim - 1]!);
    for (let axis = ndim - 2; axis >= 0; axis -= 1) {
      counter[axis]! += 1;
      pa += sa[axis]!;
      pb += sb[axis]!;
      if (counter[axis]! < shape[axis]!) break;
      pa -= sa[axis]! * counter[axis]!;
      pb -= sb[axis]! * counter[axis]!;
      counter[axis] = 0;
    }
  }
}

/** Broadcast ``a`` and ``b`` and apply ``fn`` elementwise into ``dtype``. */
export function mapBinary(a: Tensor, b: Tensor, fn: (x: number, y: number) => number, dtype: DType): Tensor {
  const ad = a.data;
  const bd = b.data;
  if (shapesEqual(a.shape, b.shape)) {
    const out = allocate(dtype, ad.length);
    for (let index = 0; index < ad.length; index += 1) out[index] = fn(ad[index]!, bd[index]!);
    return fromStorage(roundStorage(dtype, out), a.shape, dtype);
  }
  const shape = broadcastShapes(a.shape, b.shape);
  const out = allocate(dtype, numelOf(shape));
  broadcastRows(a, b, shape, (pa, pb, po, length, sa, sb) => {
    for (let index = 0; index < length; index += 1) out[po + index] = fn(ad[pa + index * sa]!, bd[pb + index * sb]!);
  });
  return fromStorage(roundStorage(dtype, out), shape, dtype);
}

const enum Arith { Add, Sub, Mul, Div }

/** ``+ - * /`` with broadcasting, without a per-element callback. */
function arith(a: Tensor, b: Tensor, op: Arith, dtype: DType): Tensor {
  const ad = a.data;
  const bd = b.data;
  const shape = shapesEqual(a.shape, b.shape) ? a.shape : broadcastShapes(a.shape, b.shape);
  if (dtype === 'float32' && ad instanceof Float32Array && bd instanceof Float32Array && ad.length === numelOf(shape)
    && (bd.length === ad.length || bd.length === 1)) {
    // Same-shape or scalar float32 arithmetic on the kernels (one IEEE float32 operation: identical results).
    const fast = binary(op as number as BinaryOp, ad, bd);
    if (fast) return fromStorage(fast, shape, dtype);
  }
  const out = allocate(dtype, numelOf(shape));
  const run = (pa: number, pb: number, po: number, length: number, sa: number, sb: number): void => {
    switch (op) {
      case Arith.Add: for (let index = 0; index < length; index += 1) out[po + index] = ad[pa + index * sa]! + bd[pb + index * sb]!; break;
      case Arith.Sub: for (let index = 0; index < length; index += 1) out[po + index] = ad[pa + index * sa]! - bd[pb + index * sb]!; break;
      case Arith.Mul: for (let index = 0; index < length; index += 1) out[po + index] = ad[pa + index * sa]! * bd[pb + index * sb]!; break;
      case Arith.Div: for (let index = 0; index < length; index += 1) out[po + index] = ad[pa + index * sa]! / bd[pb + index * sb]!; break;
    }
  };
  if (shape === a.shape) run(0, 0, 0, out.length, 1, 1);
  else broadcastRows(a, b, shape, run);
  return fromStorage(roundStorage(dtype, out), shape, dtype);
}

function scalarTensor(value: number, dtype: DType): Tensor {
  const data = allocate(dtype, 1);
  data[0] = castValue(dtype, value);
  return fromStorage(data, [], dtype);
}

function operands(a: Operand, b: Operand): [Tensor, Tensor, DType] {
  if (a instanceof Tensor && b instanceof Tensor) return [a, b, promoteTypes(a.dtype, b.dtype)];
  if (a instanceof Tensor) {
    const dtype = promoteScalar(a.dtype, b as number);
    return [a, scalarTensor(b as number, dtype), dtype];
  }
  if (b instanceof Tensor) {
    const dtype = promoteScalar(b.dtype, a);
    return [scalarTensor(a, dtype), b, dtype];
  }
  const dtype = promoteScalar('float32', a + (b as number));
  return [scalarTensor(a, dtype), scalarTensor(b as number, dtype), dtype];
}

/** Multiply an operand by a JavaScript number without changing its kind. */
export function mulOperand(value: Operand, factor: number): Operand {
  return value instanceof Tensor ? mul(value, factor) : value * factor;
}

function reduceGrad(grad: Tensor, input: Tensor): Tensor {
  return sumToShape(grad, input.shape);
}

// ------------------------------------------------------------------ copies

export function clone(x: Tensor): Tensor {
  const result = fromStorage(x.data.slice() as Storage, x.shape, x.dtype);
  return attachGrad(result, [x], (grad) => [grad], 'clone');
}

export function cast(x: Tensor, dtype: DType): Tensor {
  if (dtype === x.dtype) return x;
  const source = x.data;
  const out = allocate(dtype, source.length);
  for (let index = 0; index < source.length; index += 1) out[index] = source[index]!;
  const result = fromStorage(roundStorage(dtype, out), x.shape, dtype);
  if (!isFloatingDType(x.dtype) || !isFloatingDType(dtype)) return result;
  return attachGrad(result, [x], (grad) => [cast(grad, x.dtype)], 'cast');
}

// ------------------------------------------------------------------ arithmetic

export function add(a: Operand, b: Operand): Tensor {
  const [x, y, dtype] = operands(a, b);
  const result = arith(x, y, Arith.Add, dtype);
  return attachGrad(result, [x, y], (grad) => [reduceGrad(grad, x), reduceGrad(grad, y)], 'add');
}

export function sub(a: Operand, b: Operand): Tensor {
  const [x, y, dtype] = operands(a, b);
  const result = arith(x, y, Arith.Sub, dtype);
  return attachGrad(result, [x, y], (grad) => [reduceGrad(grad, x), reduceGrad(neg(grad), y)], 'sub');
}

export function mul(a: Operand, b: Operand): Tensor {
  const [x, y, dtype] = operands(a, b);
  const result = arith(x, y, Arith.Mul, dtype);
  return attachGrad(result, [x, y], (grad) => [
    x.requiresGrad ? reduceGrad(mul(grad, y), x) : null,
    y.requiresGrad ? reduceGrad(mul(grad, x), y) : null,
  ], 'mul');
}

export function div(a: Operand, b: Operand): Tensor {
  let [x, y, dtype] = operands(a, b);
  if (!isFloatingDType(dtype)) {
    dtype = 'float32';
    x = cast(x, dtype);
    y = cast(y, dtype);
  }
  const result = arith(x, y, Arith.Div, dtype);
  return attachGrad(result, [x, y], (grad) => [
    x.requiresGrad ? reduceGrad(div(grad, y), x) : null,
    y.requiresGrad ? reduceGrad(neg(div(mul(grad, x), mul(y, y))), y) : null,
  ], 'div');
}

export function pow(a: Tensor, exponent: Operand): Tensor {
  if (typeof exponent === 'number') {
    const dtype = isFloatingDType(a.dtype) || !Number.isInteger(exponent) ? promoteScalar(a.dtype, exponent) : a.dtype;
    const result = mapUnary(a, (value) => value ** exponent, isFloatingDType(dtype) ? dtype : dtype);
    return attachGrad(result, [a], (grad) => [
      exponent === 0 ? mul(grad, 0) : mul(grad, mul(pow(a, exponent - 1), exponent)),
    ], 'pow');
  }
  const [x, y, dtype] = operands(a, exponent);
  const result = mapBinary(x, y, (p, q) => p ** q, dtype);
  return attachGrad(result, [x, y], (grad) => [
    x.requiresGrad ? reduceGrad(mul(grad, mul(y, pow(x, sub(y, 1)))), x) : null,
    y.requiresGrad ? reduceGrad(mul(grad, mul(result, log(x))), y) : null,
  ], 'pow');
}

export function neg(x: Tensor): Tensor {
  const result = mapUnary(x, (value) => -value);
  return attachGrad(result, [x], (grad) => [neg(grad)], 'neg');
}

export function exp(x: Tensor): Tensor {
  const result = kernelUnary(x, UnaryOp.Exp) ?? floatUnary(x, Math.exp);
  return attachGrad(result, [x], (grad) => [mul(grad, result)], 'exp');
}

export function log(x: Tensor): Tensor {
  const result = floatUnary(x, Math.log);
  return attachGrad(result, [x], (grad) => [div(grad, x)], 'log');
}

export function log1p(x: Tensor): Tensor {
  const result = floatUnary(x, Math.log1p);
  return attachGrad(result, [x], (grad) => [div(grad, add(x, 1))], 'log1p');
}

export function sqrt(x: Tensor): Tensor {
  const result = floatUnary(x, Math.sqrt);
  return attachGrad(result, [x], (grad) => [div(grad, mul(result, 2))], 'sqrt');
}

export function rsqrt(x: Tensor): Tensor {
  const result = floatUnary(x, (value) => 1 / Math.sqrt(value));
  return attachGrad(result, [x], (grad) => [mul(grad, mul(pow(result, 3), -0.5))], 'rsqrt');
}

export function abs(x: Tensor): Tensor {
  const result = mapUnary(x, Math.abs);
  return attachGrad(result, [x], (grad) => [mul(grad, sign(x))], 'abs');
}

export function sign(x: Tensor): Tensor {
  return mapUnary(x, (value) => (value > 0 ? 1 : value < 0 ? -1 : 0));
}

export function square(x: Tensor): Tensor {
  const result = mapUnary(x, (value) => value * value);
  return attachGrad(result, [x], (grad) => [mul(grad, mul(x, 2))], 'square');
}

export function reciprocal(x: Tensor): Tensor {
  const result = floatUnary(x, (value) => 1 / value);
  return attachGrad(result, [x], (grad) => [neg(mul(grad, mul(result, result)))], 'reciprocal');
}

export function tanh(x: Tensor): Tensor {
  const result = kernelUnary(x, UnaryOp.Tanh) ?? floatUnary(x, Math.tanh);
  return attachGrad(result, [x], (grad) => [mul(grad, mapUnary(result, (value) => 1 - value * value))], 'tanh');
}

function sigmoidValue(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const e = Math.exp(value);
  return e / (1 + e);
}

/** Float32 ``op`` on the WebAssembly kernels (bit-identical to the JavaScript formula), or ``null``. */
function kernelUnary(x: Tensor, op: UnaryOp): Tensor | null {
  if (x.dtype !== 'float32' || !(x.data instanceof Float32Array) || x.numel < 4096) return null;
  const out = unary(op, x.data);
  return out ? fromStorage(out, x.shape, 'float32') : null;
}

export function sigmoid(x: Tensor): Tensor {
  const result = kernelUnary(x, UnaryOp.Sigmoid) ?? floatUnary(x, sigmoidValue);
  return attachGrad(result, [x], (grad) => [mul(grad, mapUnary(result, (value) => value * (1 - value)))], 'sigmoid');
}

export function relu(x: Tensor): Tensor {
  const result = mapUnary(x, (value) => (value > 0 ? value : 0));
  return attachGrad(result, [x], (grad) => [mul(grad, mapUnary(x, (value) => (value > 0 ? 1 : 0)))], 'relu');
}

export function sin(x: Tensor): Tensor {
  const result = floatUnary(x, Math.sin);
  return attachGrad(result, [x], (grad) => [mul(grad, cos(x))], 'sin');
}

export function cos(x: Tensor): Tensor {
  const result = floatUnary(x, Math.cos);
  return attachGrad(result, [x], (grad) => [neg(mul(grad, sin(x)))], 'cos');
}

const TWO_OVER_SQRT_PI = 1.1283791670955126;

/** Error function with ~1e-13 absolute accuracy (series / continued fraction). */
export function erfValue(value: number): number {
  if (Number.isNaN(value)) return value;
  const x = Math.abs(value);
  let result: number;
  if (x < 2.5) {
    // Maclaurin series: erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1))
    const x2 = x * x;
    let term = x;
    let sum = x;
    for (let n = 1; n < 60; n += 1) {
      term *= -x2 / n;
      const contribution = term / (2 * n + 1);
      sum += contribution;
      if (Math.abs(contribution) < 1e-17 * Math.abs(sum)) break;
    }
    result = TWO_OVER_SQRT_PI * sum;
  } else if (x > 6) {
    result = 1;
  } else {
    // Continued fraction for erfc (Lentz), accurate for x >= 2.5.
    const tiny = 1e-300;
    let f = x;
    let c = x;
    let d = 0;
    for (let n = 1; n < 200; n += 1) {
      const an = n / 2;
      d = x + an * d;
      d = Math.abs(d) < tiny ? tiny : d;
      c = x + an / c;
      c = Math.abs(c) < tiny ? tiny : c;
      d = 1 / d;
      const delta = c * d;
      f *= delta;
      if (Math.abs(delta - 1) < 1e-16) break;
    }
    const erfc = Math.exp(-x * x) / (f * Math.sqrt(Math.PI));
    result = 1 - erfc;
  }
  return value < 0 ? -result : result;
}

export function erf(x: Tensor): Tensor {
  const result = floatUnary(x, erfValue);
  return attachGrad(result, [x], (grad) => [
    mul(grad, mapUnary(x, (value) => TWO_OVER_SQRT_PI * Math.exp(-value * value))),
  ], 'erf');
}

export function floor(x: Tensor): Tensor { return mapUnary(x, Math.floor); }
export function ceil(x: Tensor): Tensor { return mapUnary(x, Math.ceil); }
/** Round half to even (like ``torch.round``). */
export function round(x: Tensor): Tensor {
  return mapUnary(x, (value) => {
    const rounded = Math.round(value);
    return Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
  });
}

export function maximum(a: Operand, b: Operand): Tensor {
  const [x, y, dtype] = operands(a, b);
  const result = mapBinary(x, y, (p, q) => (Number.isNaN(p) || Number.isNaN(q) ? Number.NaN : Math.max(p, q)), dtype);
  return attachGrad(result, [x, y], (grad) => [
    reduceGrad(mul(grad, mapBinary(x, y, (p, q) => (p > q ? 1 : p === q ? 0.5 : 0), grad.dtype)), x),
    reduceGrad(mul(grad, mapBinary(x, y, (p, q) => (q > p ? 1 : p === q ? 0.5 : 0), grad.dtype)), y),
  ], 'maximum');
}

export function minimum(a: Operand, b: Operand): Tensor {
  const [x, y, dtype] = operands(a, b);
  const result = mapBinary(x, y, (p, q) => (Number.isNaN(p) || Number.isNaN(q) ? Number.NaN : Math.min(p, q)), dtype);
  return attachGrad(result, [x, y], (grad) => [
    reduceGrad(mul(grad, mapBinary(x, y, (p, q) => (p < q ? 1 : p === q ? 0.5 : 0), grad.dtype)), x),
    reduceGrad(mul(grad, mapBinary(x, y, (p, q) => (q < p ? 1 : p === q ? 0.5 : 0), grad.dtype)), y),
  ], 'minimum');
}

export function clamp(x: Tensor, min: number | null, max: number | null): Tensor {
  const low = min ?? Number.NEGATIVE_INFINITY;
  const high = max ?? Number.POSITIVE_INFINITY;
  const dtype = [min, max].some((value) => value !== null && !Number.isInteger(value)) ? promoteScalar(x.dtype, 0.5) : x.dtype;
  const result = mapUnary(x, (value) => (value < low ? low : value > high ? high : value), dtype);
  return attachGrad(result, [x], (grad) => [
    mul(grad, mapUnary(x, (value) => (value >= low && value <= high ? 1 : 0), grad.dtype)),
  ], 'clamp');
}

/** Replace elements where ``mask`` is true with ``value`` (mask broadcasts). */
export function maskedFill(x: Tensor, mask: Tensor, value: number): Tensor {
  const dtype = promoteScalar(x.dtype, value);
  const result = mapBinary(x, mask, (p, flag) => (flag ? value : p), dtype);
  if (!shapesEqual(result.shape, x.shape)) {
    throw new RangeError('maskedFill mask must broadcast to the input shape');
  }
  return attachGrad(result, [x], (grad) => [
    mapBinary(grad, mask, (g, flag) => (flag ? 0 : g), grad.dtype),
  ], 'maskedFill');
}

/** Select from ``a`` where ``condition`` is true, else from ``b``. */
export function where(condition: Tensor, a: Operand, b: Operand): Tensor {
  const [x, y, dtype] = operands(a, b);
  const shape = broadcastShapes(condition.shape, x.shape, y.shape);
  const cond = expandData(condition, shape);
  const xd = expandData(x, shape);
  const yd = expandData(y, shape);
  const out = allocate(dtype, cond.length);
  for (let index = 0; index < cond.length; index += 1) out[index] = cond[index] ? xd[index]! : yd[index]!;
  const result = fromStorage(roundStorage(dtype, out), shape, dtype);
  return attachGrad(result, [x, y], (grad) => [
    x.requiresGrad ? reduceGrad(mapBinary(grad, condition, (g, flag) => (flag ? g : 0), grad.dtype), x) : null,
    y.requiresGrad ? reduceGrad(mapBinary(grad, condition, (g, flag) => (flag ? 0 : g), grad.dtype), y) : null,
  ], 'where');
}

function expandData(x: Tensor, shape: Shape): ArrayLike<number> {
  if (shapesEqual(x.shape, shape)) return x.data;
  const map = broadcastIndexMap(shape, x.shape);
  const source = x.data;
  const out = new Float64Array(map.length);
  for (let index = 0; index < map.length; index += 1) out[index] = source[map[index]!]!;
  return out;
}

// ------------------------------------------------------------------ comparison

export type Comparison = 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge';

export function compare(a: Operand, b: Operand, kind: Comparison): Tensor {
  const [x, y] = operands(a, b);
  const fn: (p: number, q: number) => number = {
    eq: (p: number, q: number) => (p === q ? 1 : 0),
    ne: (p: number, q: number) => (p !== q ? 1 : 0),
    lt: (p: number, q: number) => (p < q ? 1 : 0),
    le: (p: number, q: number) => (p <= q ? 1 : 0),
    gt: (p: number, q: number) => (p > q ? 1 : 0),
    ge: (p: number, q: number) => (p >= q ? 1 : 0),
  }[kind];
  return mapBinary(x, y, fn, 'bool');
}

export function logicalNot(x: Tensor): Tensor {
  return mapUnary(x, (value) => (value ? 0 : 1), 'bool');
}

export function logical(a: Tensor, b: Tensor, kind: 'and' | 'or' | 'xor'): Tensor {
  const fn = kind === 'and'
    ? (p: number, q: number) => (p && q ? 1 : 0)
    : kind === 'or'
      ? (p: number, q: number) => (p || q ? 1 : 0)
      : (p: number, q: number) => ((p ? 1 : 0) !== (q ? 1 : 0) ? 1 : 0);
  return mapBinary(a, b, fn, 'bool');
}

export function isFinite(x: Tensor): Tensor {
  return mapUnary(x, (value) => (Number.isFinite(value) ? 1 : 0), 'bool');
}

export function isNan(x: Tensor): Tensor {
  return mapUnary(x, (value) => (Number.isNaN(value) ? 1 : 0), 'bool');
}

export function allFinite(x: Tensor): boolean {
  const data = x.data;
  for (let index = 0; index < data.length; index += 1) if (!Number.isFinite(data[index]!)) return false;
  return true;
}

export function equal(a: Tensor, b: Tensor): boolean {
  if (!shapesEqual(a.shape, b.shape)) return false;
  const x = a.data;
  const y = b.data;
  for (let index = 0; index < x.length; index += 1) if (x[index] !== y[index]) return false;
  return true;
}

/** Elementwise closeness like ``torch.allclose``. */
export function allclose(a: Tensor, b: Tensor, rtol = 1e-5, atol = 1e-8): boolean {
  if (!shapesEqual(broadcastShapes(a.shape, b.shape), a.shape)) return false;
  const shape = a.shape;
  const x = expandData(a, shape);
  const y = expandData(b, shape);
  for (let index = 0; index < x.length; index += 1) {
    const p = x[index]!;
    const q = y[index]!;
    if (Number.isNaN(p) || Number.isNaN(q)) return false;
    if (Math.abs(p - q) > atol + rtol * Math.abs(q)) return false;
  }
  return true;
}
