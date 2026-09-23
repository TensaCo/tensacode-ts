/**
 * Floating-point helpers that reproduce the reference C libraries bit for
 * bit: a correctly rounded fused multiply-add for doubles and floats
 * (Boldo-Melquiond emulation with round-to-odd), and glibc 2.39's ``sin``,
 * ``cos`` (``sysdeps/ieee754/dbl-64/s_sin.c``, as compiled for aarch64 with
 * FMA contraction) and ``sinf`` (``sysdeps/ieee754/flt-32/s_sinf.c``), which
 * PyTorch's and Pillow's resampling filters call. JavaScript's ``Math.sin``
 * differs from glibc in the last bit for a fraction of arguments.
 */
import { SINCOS_TABLE } from './sincosTable.js';

const scratch = new Float64Array(1);
const scratchBits = new BigInt64Array(scratch.buffer);
const scratchWords = new Uint32Array(scratch.buffer);

function lowWord(value: number): number {
  scratch[0] = value;
  return scratchWords[0]!;
}

function highWord(value: number): number {
  scratch[0] = value;
  return scratchWords[1]!;
}

/** ``nextafter(value, direction > 0 ? +inf : -inf)`` for finite non-zero steps. */
function nextToward(value: number, direction: number): number {
  if (value === 0) return direction > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE;
  scratch[0] = value;
  const away = (value > 0) === (direction > 0);
  scratchBits[0] = scratchBits[0]! + (away ? 1n : -1n);
  return scratch[0]!;
}

function twoSum(a: number, b: number): [number, number] {
  const s = a + b;
  const bb = s - a;
  return [s, (a - (s - bb)) + (b - bb)];
}

const SPLITTER = 134217729; // 2**27 + 1

function twoProduct(a: number, b: number): [number, number] {
  const p = a * b;
  let t = SPLITTER * a;
  const ah = t - (t - a);
  const al = a - ah;
  t = SPLITTER * b;
  const bh = t - (t - b);
  const bl = b - bh;
  return [p, ((ah * bh - p) + ah * bl + al * bh) + al * bl];
}

/** ``a + b`` rounded to odd (the neighbour with an odd last bit when inexact). */
function addRoundOdd(a: number, b: number): number {
  const [s, e] = twoSum(a, b);
  if (e === 0) return s;
  scratch[0] = s;
  return (scratchWords[0]! & 1) === 0 ? nextToward(s, e) : s;
}

/** Correctly rounded ``a * b + c`` in double precision. */
export function fma(a: number, b: number, c: number): number {
  const p = a * b;
  if (!Number.isFinite(p) || !Number.isFinite(c) || p === 0) return p + c;
  const [uh, ul] = twoProduct(a, b);
  const [th, tl] = twoSum(c, uh);
  return th + addRoundOdd(tl, ul);
}

/** Correctly rounded ``a * b + c`` in single precision (float32 operands). */
export function fmaf(a: number, b: number, c: number): number {
  return Math.fround(addRoundOdd(a * b, c));
}

// ---------------------------------------------------------------------------
// glibc double sin/cos (IBM Accurate Mathematical Library).
// ---------------------------------------------------------------------------

const S1 = -0.16666666666666666;
const S2 = 0.008333333333332329;
const S3 = -0.00019841269834414642;
const S4 = 0.000002755729806860771;
const S5 = -2.5022014848318398e-8;
const BIG = 52776558133248;
const HP0 = 1.5707963267948966;
const HP1 = 6.123233995736766e-17;
const MP1 = 1.5707963407039642;
const MP2 = -1.3909067564377153e-8;
const PP3 = -4.97899623147991e-17;
const PP4 = -1.9034889620193266e-25;
const HPINV = 0.6366197723675814;
const TOINT = 6755399441055744;
const SN3 = -1.66666666666664880952546298448555e-1;
const SN5 = 8.33333214285722277379541354343671e-3;
const CS2 = 4.99999999999999999999950396842453e-1;
const CS4 = -4.16666666666664434524222570944589e-2;
const CS6 = 1.38888874007937613028114285595617e-3;

function taylorSin(xx: number, x: number, dx: number): number {
  const poly = fma(fma(fma(S5, xx, S4), xx, S3), xx, S2) * xx + S1;
  const t = fma(fma(poly, x, -0.5 * dx), xx, dx);
  return x + t;
}

function doCos(x: number, dx: number): number {
  let d = dx;
  if (x < 0) d = -d;
  const u = BIG + Math.abs(x);
  const y = Math.abs(x) - (u - BIG) + d;
  const xx = y * y;
  const s = fma(y * xx, fma(xx, SN5, SN3), y);
  const c = xx * fma(xx, fma(xx, CS6, CS4), CS2);
  const k = lowWord(u) << 2;
  const sn = SINCOS_TABLE[k]!;
  const ssn = SINCOS_TABLE[k + 1]!;
  const cs = SINCOS_TABLE[k + 2]!;
  const ccs = SINCOS_TABLE[k + 3]!;
  const cor = fma(-sn, s, fma(-cs, c, fma(-s, ssn, ccs)));
  return cs + cor;
}

function doSin(x: number, dx: number): number {
  const xold = x;
  if (Math.abs(x) < 0.126) return taylorSin(x * x, x, dx);
  let d = dx;
  if (x <= 0) d = -d;
  const u = BIG + Math.abs(x);
  const y = Math.abs(x) - (u - BIG);
  const xx = y * y;
  const s = y + fma(y * xx, fma(xx, SN5, SN3), d);
  const c = fma(xx, fma(xx, fma(xx, CS6, CS4), CS2), y * d);
  const k = lowWord(u) << 2;
  const sn = SINCOS_TABLE[k]!;
  const ssn = SINCOS_TABLE[k + 1]!;
  const cs = SINCOS_TABLE[k + 2]!;
  const ccs = SINCOS_TABLE[k + 3]!;
  const cor = fma(cs, s, fma(-sn, c, fma(s, ccs, ssn)));
  const result = sn + cor;
  return xold < 0 || Object.is(xold, -0) ? -Math.abs(result) : Math.abs(result);
}

function reduceSinCos(x: number): [number, number, number] {
  const t = fma(x, HPINV, TOINT);
  const xn = t - TOINT;
  const y = fma(-xn, MP2, fma(-xn, MP1, x));
  const n = lowWord(t) & 3;
  let t1 = xn * PP3;
  const t2 = y - t1;
  let db = (y - t2) - t1;
  t1 = xn * PP4;
  const b = t2 - t1;
  db += (t2 - b) - t1;
  return [n, b, db];
}

function doSinCos(a: number, da: number, n: number): number {
  const value = n & 1 ? doCos(a, da) : doSin(a, da);
  return n & 2 ? -value : value;
}

/** glibc's ``sin`` for ``|x| < 105414350`` (``Math.sin`` beyond). */
export function glibcSin(x: number): number {
  const k = highWord(x) & 0x7fffffff;
  if (!Number.isFinite(x) || k >= 0x419921fb) return Math.sin(x);
  if (k < 0x3e500000) return x;
  if (k < 0x3feb6000) return doSin(x, 0);
  if (k < 0x400368fd) {
    const value = doCos(HP0 - Math.abs(x), HP1);
    return x < 0 ? -value : value;
  }
  const [n, a, da] = reduceSinCos(x);
  return doSinCos(a, da, n);
}

/** glibc's ``cos`` for ``|x| < 105414350`` (``Math.cos`` beyond). */
export function glibcCos(x: number): number {
  const k = highWord(x) & 0x7fffffff;
  if (!Number.isFinite(x) || k >= 0x419921fb) return Math.cos(x);
  if (k < 0x3e400000) return 1;
  if (k < 0x3feb6000) return doCos(x, 0);
  if (k < 0x400368fd) {
    const y = HP0 - Math.abs(x);
    const a = y + HP1;
    const da = (y - a) + HP1;
    return doSin(a, da);
  }
  const [n, a, da] = reduceSinCos(x);
  return doSinCos(a, da, n + 1);
}

// ---------------------------------------------------------------------------
// glibc float sinf (Szabolcs Nagy's implementation; aarch64 uses round/lround).
// ---------------------------------------------------------------------------

const SINF_TABLES = [
  { c0: 1, c1: -0.49999999725108224, c2: 0.041666623324344516, c3: -0.001388676379437604, c4: 2.4390450703564542e-5 },
  { c0: -1, c1: 0.49999999725108224, c2: -0.041666623324344516, c3: 0.001388676379437604, c4: -2.4390450703564542e-5 },
] as const;
const SINF_S1 = -0.16666654943701084;
const SINF_S2 = 0.008332178146138854;
const SINF_S3 = -0.00019517298981385725;
const SINF_HPI_INV = 0.6366197723675814;
const SINF_HPI = 1.5707963267948966;
const SINF_SIGN = [1, -1, -1, 1];
const floatScratch = new Float32Array(1);
const floatBits = new Uint32Array(floatScratch.buffer);

function abstop12(value: number): number {
  floatScratch[0] = value;
  return (floatBits[0]! >>> 20) & 0x7ff;
}

const PIO4_TOP = abstop12(0.7853981852531433);
const TINY_TOP = abstop12(2 ** -12);
const LIMIT_TOP = abstop12(120);

function sinfPoly(x: number, x2: number, table: (typeof SINF_TABLES)[number], n: number): number {
  if ((n & 1) === 0) {
    const x3 = x * x2;
    const s1 = SINF_S2 + x2 * SINF_S3;
    const x7 = x3 * x2;
    const s = x + x3 * SINF_S1;
    return Math.fround(s + x7 * s1);
  }
  const x4 = x2 * x2;
  const c2 = table.c3 + x2 * table.c4;
  const c1 = table.c0 + x2 * table.c1;
  const x6 = x4 * x2;
  const c = c1 + x4 * table.c2;
  return Math.fround(c + x6 * c2);
}

/** glibc's ``sinf`` for ``|y| < 120`` (float argument, float result). */
export function glibcSinf(y: number): number {
  const top = abstop12(y);
  if (top < PIO4_TOP) {
    if (top < TINY_TOP) return Math.fround(y);
    return sinfPoly(y, y * y, SINF_TABLES[0], 0);
  }
  if (top >= LIMIT_TOP || !Number.isFinite(y)) return Math.fround(Math.sin(y));
  const r = y * SINF_HPI_INV;
  const n = r >= 0 ? Math.floor(r + 0.5) : -Math.floor(-r + 0.5); // lround: ties away from zero
  const x = y - n * SINF_HPI;
  const quadrant = n & 3;
  const table = quadrant & 2 ? SINF_TABLES[1] : SINF_TABLES[0];
  const signed = x * SINF_SIGN[quadrant]!;
  return sinfPoly(signed, x * x, table, n);
}
