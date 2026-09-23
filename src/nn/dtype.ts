/**
 * Tensor element types.
 *
 * Every tensor stores its values in a contiguous JavaScript typed array.
 * Floating types that JavaScript cannot represent natively (float16, bfloat16)
 * are computed in float32 and only rounded when serialized. Integer and boolean
 * types are stored in a Float64Array, which represents every int32 exactly and
 * every int64 up to 2**53. ``uint16`` is, as in PyTorch, mostly a storage type
 * (decoded 16-bit images): it promotes only to floating types.
 */
export type DType =
  | 'float32'
  | 'float64'
  | 'float16'
  | 'bfloat16'
  | 'int64'
  | 'int32'
  | 'int16'
  | 'int8'
  | 'uint8'
  | 'uint16'
  | 'bool';

export type Storage = Float32Array | Float64Array;

export const DTYPES: readonly DType[] = [
  'float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'uint16', 'bool',
];

export function isDType(value: unknown): value is DType {
  return typeof value === 'string' && (DTYPES as readonly string[]).includes(value);
}

export function isFloatingDType(dtype: DType): boolean {
  return dtype === 'float32' || dtype === 'float64' || dtype === 'float16' || dtype === 'bfloat16';
}

export function isIntegerDType(dtype: DType): boolean {
  return !isFloatingDType(dtype) && dtype !== 'bool';
}

/** Allocate storage for ``size`` elements of ``dtype``. */
export function allocate(dtype: DType, size: number): Storage {
  return usesFloat32Storage(dtype) ? new Float32Array(size) : new Float64Array(size);
}

export function usesFloat32Storage(dtype: DType): boolean {
  return dtype === 'float32' || dtype === 'float16' || dtype === 'bfloat16';
}

/** Byte size of one element when serialized (safetensors / checksums). */
export function itemSize(dtype: DType): number {
  switch (dtype) {
    case 'float64': case 'int64': return 8;
    case 'float32': case 'int32': return 4;
    case 'float16': case 'bfloat16': case 'int16': case 'uint16': return 2;
    case 'int8': case 'uint8': case 'bool': return 1;
  }
}

const TORCH_SCALAR_NAMES: Readonly<Record<DType, string>> = {
  float32: 'Float', float64: 'Double', float16: 'Half', bfloat16: 'BFloat16', int64: 'Long', int32: 'Int', int16: 'Short',
  int8: 'Char', uint8: 'Byte', uint16: 'UInt16', bool: 'Bool',
};

/** Result type of combining two tensors in arithmetic. */
export function promoteTypes(a: DType, b: DType): DType {
  if (a === b) return a === 'bool' ? 'bool' : a;
  if (a === 'uint16' || b === 'uint16') {
    // PyTorch promotes its limited unsigned types only to floating types.
    const other = a === 'uint16' ? b : a;
    if (isFloatingDType(other)) return other;
    throw new RangeError(`Promotion for uint16, uint32, uint64 types is not supported, attempted to promote ${TORCH_SCALAR_NAMES[a]} and ${TORCH_SCALAR_NAMES[b]}`);
  }
  if (a === 'float64' || b === 'float64') return 'float64';
  const af = isFloatingDType(a);
  const bf = isFloatingDType(b);
  if (af && bf) return 'float32';
  if (af) return a;
  if (bf) return b;
  if (a === 'bool') return b;
  if (b === 'bool') return a;
  return 'int64';
}

/** Result type of combining a tensor with a JavaScript number scalar. */
export function promoteScalar(dtype: DType, scalar: number): DType {
  if (isFloatingDType(dtype)) return dtype;
  if (!Number.isInteger(scalar)) return 'float32';
  return dtype === 'bool' ? 'int64' : dtype;
}

/** Round a value into the representable range of ``dtype``. */
export function castValue(dtype: DType, value: number): number {
  switch (dtype) {
    case 'float32': case 'float16': case 'bfloat16': case 'float64':
      return value;
    case 'bool':
      return value !== 0 && !Number.isNaN(value) ? 1 : 0;
    case 'uint8':
      return (Math.trunc(value) & 0xff) >>> 0;
    case 'uint16':
      return (Math.trunc(value) & 0xffff) >>> 0;
    case 'int8':
      return (Math.trunc(value) << 24) >> 24;
    case 'int16':
      return (Math.trunc(value) << 16) >> 16;
    case 'int32':
      return Math.trunc(value) | 0;
    case 'int64':
      return Math.trunc(value);
  }
}

/** PyTorch-style dtype names used in persisted configuration topologies. */
export function torchDTypeName(dtype: DType): string {
  return `torch.${dtype}`;
}

// ---------------------------------------------------------------------------
// Half-precision conversion helpers (IEEE binary16 and bfloat16).
// ---------------------------------------------------------------------------

const conversionBuffer = new ArrayBuffer(4);
const conversionFloat = new Float32Array(conversionBuffer);
const conversionUint = new Uint32Array(conversionBuffer);

export function float32ToBFloat16Bits(value: number): number {
  conversionFloat[0] = value;
  const bits = conversionUint[0]!;
  if ((bits & 0x7fffffff) > 0x7f800000) return ((bits >>> 16) | 0x40) & 0xffff; // quiet NaN
  const rounding = 0x7fff + ((bits >>> 16) & 1);
  return ((bits + rounding) >>> 16) & 0xffff;
}

export function bfloat16BitsToFloat32(bits: number): number {
  conversionUint[0] = (bits & 0xffff) << 16;
  return conversionFloat[0]!;
}

export function float32ToFloat16Bits(value: number): number {
  conversionFloat[0] = value;
  const x = conversionUint[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exponent = (x >>> 23) & 0xff;
  let mantissa = x & 0x7fffff;
  if (exponent === 0xff) {
    return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  }
  // Unbiased exponent for half precision.
  let halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00; // overflow to infinity
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign; // underflow to zero
    mantissa |= 0x800000;
    const shift = 14 - halfExponent;
    let half = mantissa >>> shift;
    const remainder = mantissa & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (half & 1))) half += 1;
    return sign | half;
  }
  let half = (halfExponent << 10) | (mantissa >>> 13);
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (half & 1))) half += 1;
  exponent = half; // may carry into exponent (correct rounding to infinity)
  return sign | exponent;
}

export function float16BitsToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** Round a float32 value through a narrower floating format. */
export function roundToDType(dtype: DType, value: number): number {
  if (dtype === 'float16') return float16BitsToFloat32(float32ToFloat16Bits(value));
  if (dtype === 'bfloat16') return bfloat16BitsToFloat32(float32ToBFloat16Bits(value));
  return value;
}

export const FLOAT32_MIN = -3.4028234663852886e38;
export const FLOAT32_EPSILON = 1.1920928955078125e-7;

/** Most negative finite value of a floating dtype (like ``torch.finfo(dtype).min``). */
export function finfoMin(dtype: DType): number {
  switch (dtype) {
    case 'float64': return -Number.MAX_VALUE;
    case 'float16': return -65504;
    case 'bfloat16': return -3.3895313892515355e38;
    default: return FLOAT32_MIN;
  }
}
