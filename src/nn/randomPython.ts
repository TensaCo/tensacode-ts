/**
 * CPython's ``random.Random`` (``Lib/random.py`` over ``Modules/_randommodule.c``),
 * reproduced exactly: the same Mersenne Twister seeding (``init_by_array``),
 * ``random()``, ``getrandbits``, ``randrange``/``randint``, ``choice``,
 * ``shuffle``, ``sample``, ``choices``, ``uniform``, ``gauss`` and the
 * ``getstate()``/``setstate()`` tuple ``(3, (624 words..., index), gauss_next)``.
 *
 * ``pythonRandom`` is the process-wide instance behind Python's module-level
 * functions; training checkpoints save and restore it as ``python_rng``.
 */
import { cos, log, sin } from './randomMath.js';

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;
const TWOPI = 2 * Math.PI;
const PY_HASH_BITS = 61n;
const PY_HASH_MODULUS = (1n << 61n) - 1n;

/** ``Random.getstate()``: ``[3, [624 state words, index], gauss_next]``. */
export type PythonRandomState = readonly [number, readonly number[], number | null];

/** A seed accepted by ``Random.seed``. */
export type PythonSeed = number | bigint | string | Uint8Array | null | undefined;

// --------------------------------------------------------------------------- SHA-512

const SHA512_K = [
  '428a2f98d728ae22', '7137449123ef65cd', 'b5c0fbcfec4d3b2f', 'e9b5dba58189dbbc', '3956c25bf348b538', '59f111f1b605d019',
  '923f82a4af194f9b', 'ab1c5ed5da6d8118', 'd807aa98a3030242', '12835b0145706fbe', '243185be4ee4b28c', '550c7dc3d5ffb4e2',
  '72be5d74f27b896f', '80deb1fe3b1696b1', '9bdc06a725c71235', 'c19bf174cf692694', 'e49b69c19ef14ad2', 'efbe4786384f25e3',
  '0fc19dc68b8cd5b5', '240ca1cc77ac9c65', '2de92c6f592b0275', '4a7484aa6ea6e483', '5cb0a9dcbd41fbd4', '76f988da831153b5',
  '983e5152ee66dfab', 'a831c66d2db43210', 'b00327c898fb213f', 'bf597fc7beef0ee4', 'c6e00bf33da88fc2', 'd5a79147930aa725',
  '06ca6351e003826f', '142929670a0e6e70', '27b70a8546d22ffc', '2e1b21385c26c926', '4d2c6dfc5ac42aed', '53380d139d95b3df',
  '650a73548baf63de', '766a0abb3c77b2a8', '81c2c92e47edaee6', '92722c851482353b', 'a2bfe8a14cf10364', 'a81a664bbc423001',
  'c24b8b70d0f89791', 'c76c51a30654be30', 'd192e819d6ef5218', 'd69906245565a910', 'f40e35855771202a', '106aa07032bbd1b8',
  '19a4c116b8d2d0c8', '1e376c085141ab53', '2748774cdf8eeb99', '34b0bcb5e19b48a8', '391c0cb3c5c95a63', '4ed8aa4ae3418acb',
  '5b9cca4f7763e373', '682e6ff3d6b2b8a3', '748f82ee5defb2fc', '78a5636f43172f60', '84c87814a1f0ab72', '8cc702081a6439ec',
  '90befffa23631e28', 'a4506cebde82bde9', 'bef9a3f7b2c67915', 'c67178f2e372532b', 'ca273eceea26619c', 'd186b8c721c0c207',
  'eada7dd6cde0eb1e', 'f57d4f7fee6ed178', '06f067aa72176fba', '0a637dc5a2c898a6', '113f9804bef90dae', '1b710b35131c471b',
  '28db77f523047d84', '32caab7b40c72493', '3c9ebe0a15c9bebc', '431d67c49c100d4c', '4cc5d4becb3e42b6', '597f299cfc657e2a',
  '5fcb6fab3ad6faec', '6c44198c4a475817',
].map((word) => BigInt(`0x${word}`));
const SHA512_INITIAL = [
  '6a09e667f3bcc908', 'bb67ae8584caa73b', '3c6ef372fe94f82b', 'a54ff53a5f1d36f1',
  '510e527fade682d1', '9b05688c2b3e6c1f', '1f83d9abfb41bd6b', '5be0cd19137e2179',
].map((word) => BigInt(`0x${word}`));
const MASK64 = (1n << 64n) - 1n;

function rotr64(x: bigint, n: bigint): bigint {
  return ((x >> n) | (x << (64n - n))) & MASK64;
}

/** SHA-512 digest (FIPS 180-4), used by ``Random.seed`` for str and bytes seeds. */
export function sha512(message: Uint8Array): Uint8Array {
  const length = message.length;
  const padded = new Uint8Array(Math.ceil((length + 17) / 128) * 128);
  padded.set(message);
  padded[length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setBigUint64(padded.length - 8, BigInt(length) * 8n, false);
  const hash = [...SHA512_INITIAL];
  const w = new Array<bigint>(80);
  for (let block = 0; block < padded.length; block += 128) {
    for (let t = 0; t < 16; t += 1) w[t] = view.getBigUint64(block + 8 * t, false);
    for (let t = 16; t < 80; t += 1) {
      const s0 = rotr64(w[t - 15]!, 1n) ^ rotr64(w[t - 15]!, 8n) ^ (w[t - 15]! >> 7n);
      const s1 = rotr64(w[t - 2]!, 19n) ^ rotr64(w[t - 2]!, 61n) ^ (w[t - 2]! >> 6n);
      w[t] = (w[t - 16]! + s0 + w[t - 7]! + s1) & MASK64;
    }
    let [a, b, c, d, e, f, g, h] = hash as [bigint, bigint, bigint, bigint, bigint, bigint, bigint, bigint];
    for (let t = 0; t < 80; t += 1) {
      const S1 = rotr64(e, 14n) ^ rotr64(e, 18n) ^ rotr64(e, 41n);
      const ch = (e & f) ^ (~e & MASK64 & g);
      const temp1 = (h + S1 + ch + SHA512_K[t]! + w[t]!) & MASK64;
      const S0 = rotr64(a, 28n) ^ rotr64(a, 34n) ^ rotr64(a, 39n);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) & MASK64;
      h = g; g = f; f = e; e = (d + temp1) & MASK64;
      d = c; c = b; b = a; a = (temp1 + temp2) & MASK64;
    }
    const values = [a, b, c, d, e, f, g, h];
    for (let index = 0; index < 8; index += 1) hash[index] = (hash[index]! + values[index]!) & MASK64;
  }
  const digest = new Uint8Array(64);
  const out = new DataView(digest.buffer);
  hash.forEach((word, index) => out.setBigUint64(8 * index, word, false));
  return digest;
}

// --------------------------------------------------------------------------- hashing

function frexp(value: number): [number, number] {
  if (value === 0 || !Number.isFinite(value)) return [value, 0];
  const view = new DataView(new ArrayBuffer(8));
  let scaled = value;
  let bias = 0;
  view.setFloat64(0, scaled);
  if (((view.getUint16(0) >>> 4) & 0x7ff) === 0) { // subnormal: scale into the normal range
    scaled *= 2 ** 64;
    bias = -64;
    view.setFloat64(0, scaled);
  }
  const exponent = ((view.getUint16(0) >>> 4) & 0x7ff) - 1022;
  view.setUint16(0, (view.getUint16(0) & 0x800f) | (1022 << 4));
  return [view.getFloat64(0), exponent + bias];
}

/** CPython ``hash(float)`` (``_Py_HashDouble``) as a signed 64-bit value. */
export function pythonFloatHash(value: number): bigint {
  if (Number.isNaN(value)) throw new TypeError('Cannot seed from a NaN float; its hash depends on object identity');
  if (value === Infinity) return 314159n;
  if (value === -Infinity) return -314159n;
  let [m, e] = frexp(value);
  let sign = 1n;
  if (m < 0) { sign = -1n; m = -m; }
  let x = 0n;
  while (m) {
    x = ((x << 28n) & PY_HASH_MODULUS) | (x >> (PY_HASH_BITS - 28n));
    m *= 268435456;
    e -= 28;
    const y = Math.trunc(m);
    m -= y;
    x += BigInt(y);
    if (x >= PY_HASH_MODULUS) x -= PY_HASH_MODULUS;
  }
  const shift = BigInt(e >= 0 ? e % 61 : 61 - 1 - ((-1 - e) % 61));
  x = ((x << shift) & PY_HASH_MODULUS) | (x >> (PY_HASH_BITS - shift));
  x *= sign;
  if (x === -1n) x = -2n;
  return x;
}

function bytesToBigInt(bytes: Uint8Array): bigint {
  let result = 0n;
  for (const byte of bytes) result = (result << 8n) | BigInt(byte);
  return result;
}

function bitLength(value: bigint): number {
  return value === 0n ? 0 : value.toString(2).length;
}

function toIndex(value: number | bigint, name: string): bigint {
  if (typeof value === 'bigint') return value;
  if (!Number.isInteger(value)) throw new TypeError(`'float' object cannot be interpreted as an integer (${name})`);
  if (!Number.isSafeInteger(value)) throw new RangeError(`${name} must be a safe integer; pass a bigint`);
  return BigInt(value);
}

function fromIndex(value: bigint, asBigInt: boolean): number | bigint {
  return asBigInt ? value : Number(value);
}

/** ``bisect.bisect_right(a, x, lo, hi)``. */
function bisectRight(values: readonly number[], x: number, low = 0, high = values.length): number {
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (x < values[middle]!) high = middle;
    else low = middle + 1;
  }
  return low;
}

// --------------------------------------------------------------------------- Random

/** ``random.Random``: CPython's Mersenne Twister generator. */
export class PythonRandom {
  static readonly VERSION = 3;
  private readonly mt = new Uint32Array(N);
  private index = N + 1;
  /** Cached second value of ``gauss``. */
  gaussNext: number | null = null;

  constructor(seed?: PythonSeed) {
    this.seed(seed);
  }

  private initGenrand(s: number): void {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i += 1) {
      const previous = mt[i - 1]!;
      mt[i] = (Math.imul(1812433253, (previous ^ (previous >>> 30)) >>> 0) + i) >>> 0;
    }
    this.index = N;
  }

  private initByArray(key: Uint32Array): void {
    const mt = this.mt;
    this.initGenrand(19650218);
    let i = 1;
    let j = 0;
    const keyLength = key.length;
    for (let k = N > keyLength ? N : keyLength; k; k -= 1) {
      const previous = mt[i - 1]!;
      mt[i] = ((mt[i]! ^ Math.imul((previous ^ (previous >>> 30)) >>> 0, 1664525)) + key[j]! + j) >>> 0;
      i += 1;
      j += 1;
      if (i >= N) { mt[0] = mt[N - 1]!; i = 1; }
      if (j >= keyLength) j = 0;
    }
    for (let k = N - 1; k; k -= 1) {
      const previous = mt[i - 1]!;
      mt[i] = ((mt[i]! ^ Math.imul((previous ^ (previous >>> 30)) >>> 0, 1566083941)) - i) >>> 0;
      i += 1;
      if (i >= N) { mt[0] = mt[N - 1]!; i = 1; }
    }
    mt[0] = 0x80000000;
  }

  /**
   * ``Random.seed(a, version=2)``: ``null``/``undefined`` seeds from the
   * operating system; integers use all their bits (absolute value); strings
   * and bytes are extended with their SHA-512 digest; non-integral numbers
   * seed from ``hash(float)``.
   */
  seed(a?: PythonSeed): void {
    let value: bigint;
    if (a === null || a === undefined) {
      const key = new Uint32Array(N);
      globalThis.crypto.getRandomValues(key);
      this.initByArray(key);
      this.gaussNext = null;
      return;
    }
    if (typeof a === 'string' || a instanceof Uint8Array) {
      const bytes = typeof a === 'string' ? new TextEncoder().encode(a) : a;
      const combined = new Uint8Array(bytes.length + 64);
      combined.set(bytes);
      combined.set(sha512(bytes), bytes.length);
      value = bytesToBigInt(combined);
    } else if (typeof a === 'bigint' || (typeof a === 'number' && Number.isSafeInteger(a))) {
      value = BigInt(a);
      if (value < 0n) value = -value;
    } else if (typeof a === 'number') {
      value = pythonFloatHash(a) & MASK64;
    } else {
      throw new TypeError('The only supported seed types are:\nNone, int, float, str, bytes, and bytearray.');
    }
    const words: number[] = [];
    while (value > 0n) {
      words.push(Number(value & 0xffffffffn));
      value >>= 32n;
    }
    if (words.length === 0) words.push(0);
    this.initByArray(Uint32Array.from(words));
    this.gaussNext = null;
  }

  /** ``genrand_uint32``: the next tempered 32-bit output. */
  genrandUint32(): number {
    const mt = this.mt;
    if (this.index >= N) {
      let kk = 0;
      for (; kk < N - M; kk += 1) {
        const y = ((mt[kk]! & UPPER_MASK) | (mt[kk + 1]! & LOWER_MASK)) >>> 0;
        mt[kk] = (mt[kk + M]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
      }
      for (; kk < N - 1; kk += 1) {
        const y = ((mt[kk]! & UPPER_MASK) | (mt[kk + 1]! & LOWER_MASK)) >>> 0;
        mt[kk] = (mt[kk + (M - N)]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
      }
      const y = ((mt[N - 1]! & UPPER_MASK) | (mt[0]! & LOWER_MASK)) >>> 0;
      mt[N - 1] = (mt[M - 1]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
      this.index = 0;
    }
    let y = mt[this.index]!;
    this.index += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** ``Random.random()``: a float in [0, 1) with 53 random bits. */
  random(): number {
    const a = this.genrandUint32() >>> 5;
    const b = this.genrandUint32() >>> 6;
    return (a * 67108864 + b) * (1 / 9007199254740992);
  }

  /** ``Random.getrandbits(k)``. */
  getrandbits(k: number): bigint {
    if (!Number.isInteger(k) || k < 0) throw new RangeError('number of bits must be non-negative');
    if (k === 0) return 0n;
    if (k <= 32) return BigInt(this.genrandUint32() >>> (32 - k));
    let result = 0n;
    let shift = 0n;
    for (let remaining = k; remaining > 0; remaining -= 32) {
      let r = this.genrandUint32();
      if (remaining < 32) r >>>= 32 - remaining;
      result |= BigInt(r) << shift;
      shift += 32n;
    }
    return result;
  }

  private randbelow(n: bigint): bigint {
    const k = bitLength(n);
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  /** ``Random.randrange(start, stop=None, step=1)``. */
  randrange(start: number | bigint, stop?: number | bigint | null, step: number | bigint = 1): number | bigint {
    const asBigInt = typeof start === 'bigint' || typeof stop === 'bigint' || typeof step === 'bigint';
    const istart = toIndex(start, 'start');
    if (stop === undefined || stop === null) {
      if (step !== 1 && step !== 1n) throw new TypeError('Missing a non-None stop argument');
      if (istart > 0n) return fromIndex(this.randbelow(istart), asBigInt);
      throw new RangeError('empty range for randrange()');
    }
    const istop = toIndex(stop, 'stop');
    const width = istop - istart;
    const istep = toIndex(step, 'step');
    if (istep === 1n) {
      if (width > 0n) return fromIndex(istart + this.randbelow(width), asBigInt);
      throw new RangeError(`empty range in randrange(${start}, ${stop})`);
    }
    let n: bigint;
    const floorDiv = (a: bigint, b: bigint) => {
      const q = a / b;
      return (a % b !== 0n) && ((a < 0n) !== (b < 0n)) ? q - 1n : q;
    };
    if (istep > 0n) n = floorDiv(width + istep - 1n, istep);
    else if (istep < 0n) n = floorDiv(width + istep + 1n, istep);
    else throw new RangeError('zero step for randrange()');
    if (n <= 0n) throw new RangeError(`empty range in randrange(${start}, ${stop}, ${step})`);
    return fromIndex(istart + istep * this.randbelow(n), asBigInt);
  }

  /** ``Random.randint(a, b)``: an integer in [a, b]. */
  randint(a: number | bigint, b: number | bigint): number | bigint {
    const high = typeof b === 'bigint' ? b + 1n : b + 1;
    return this.randrange(a, high);
  }

  /** ``Random.choice(seq)``. */
  choice<T>(sequence: ArrayLike<T>): T {
    if (!sequence.length) throw new RangeError('Cannot choose from an empty sequence');
    return sequence[Number(this.randbelow(BigInt(sequence.length)))]!;
  }

  /** ``Random.shuffle(x)``: shuffle an array in place. */
  shuffle<T>(values: T[]): void {
    for (let i = values.length - 1; i >= 1; i -= 1) {
      const j = Number(this.randbelow(BigInt(i + 1)));
      const swap = values[i]!;
      values[i] = values[j]!;
      values[j] = swap;
    }
  }

  /** ``Random.sample(population, k, counts=None)``. */
  sample<T>(population: ArrayLike<T>, k: number, options: { counts?: readonly number[] } = {}): T[] {
    const n = population.length;
    if (options.counts !== undefined) {
      const cumulative: number[] = [];
      let running = 0;
      for (const count of options.counts) {
        if (!Number.isInteger(count)) throw new TypeError('Counts must be integers');
        running += count;
        cumulative.push(running);
      }
      if (cumulative.length !== n) throw new RangeError('The number of counts does not match the population');
      const total = cumulative.pop() ?? 0;
      if (total < 0) throw new RangeError('Counts must be non-negative');
      const selections = this.sample(Array.from({ length: total }, (_, index) => index), k);
      return selections.map((selection) => population[bisectRight(cumulative, selection)]!);
    }
    if (!Number.isInteger(k) || !(k >= 0 && k <= n)) throw new RangeError('Sample larger than population or is negative');
    const result = new Array<T>(k);
    let setsize = 21;
    if (k > 5) setsize += 4 ** Math.ceil(log(k * 3) / log(4));
    if (n <= setsize) {
      const pool = Array.from(population);
      for (let i = 0; i < k; i += 1) {
        const j = Number(this.randbelow(BigInt(n - i)));
        result[i] = pool[j]!;
        pool[j] = pool[n - i - 1]!;
      }
    } else {
      const selected = new Set<number>();
      for (let i = 0; i < k; i += 1) {
        let j = Number(this.randbelow(BigInt(n)));
        while (selected.has(j)) j = Number(this.randbelow(BigInt(n)));
        selected.add(j);
        result[i] = population[j]!;
      }
    }
    return result;
  }

  /** ``Random.choices(population, weights=None, cum_weights=None, k=1)``. */
  choices<T>(
    population: ArrayLike<T>,
    options: { weights?: readonly number[]; cumWeights?: readonly number[]; k?: number } = {},
  ): T[] {
    const k = options.k ?? 1;
    const n = population.length;
    let cumulative = options.cumWeights;
    if (cumulative === undefined) {
      if (options.weights === undefined) {
        const size = n;
        return Array.from({ length: k }, () => population[Math.floor(this.random() * size)]!);
      }
      let running = 0;
      cumulative = options.weights.map((weight) => (running += weight));
    } else if (options.weights !== undefined) {
      throw new TypeError('Cannot specify both weights and cumulative weights');
    }
    if (cumulative.length !== n) throw new RangeError('The number of weights does not match the population');
    const total = cumulative[n - 1]! + 0.0;
    if (total <= 0) throw new RangeError('Total of weights must be greater than zero');
    if (!Number.isFinite(total)) throw new RangeError('Total of weights must be finite');
    const weights = cumulative;
    return Array.from({ length: k }, () => population[bisectRight(weights, this.random() * total, 0, n - 1)]!);
  }

  /** ``Random.uniform(a, b)``. */
  uniform(a: number, b: number): number {
    return a + (b - a) * this.random();
  }

  /** ``Random.gauss(mu, sigma)`` with the cached second sample. */
  gauss(mu = 0, sigma = 1): number {
    let z = this.gaussNext;
    this.gaussNext = null;
    if (z === null) {
      const x2pi = this.random() * TWOPI;
      const g2rad = Math.sqrt(-2 * log(1 - this.random()));
      z = cos(x2pi) * g2rad;
      this.gaussNext = sin(x2pi) * g2rad;
    }
    return mu + z * sigma;
  }

  /** ``Random.getstate()``. */
  getstate(): PythonRandomState {
    return [PythonRandom.VERSION, [...this.mt, this.index], this.gaussNext];
  }

  /** ``Random.setstate(state)`` for version 3 (and legacy version 2) states. */
  setstate(state: readonly unknown[]): void {
    if (!Array.isArray(state) || state.length === 0) throw new TypeError('state must be a sequence');
    const version = state[0];
    if (version !== 3 && version !== 2) {
      throw new RangeError(`state with version ${String(version)} passed to Random.setstate() of version 3`);
    }
    if (state.length !== 3) throw new RangeError(`too many values to unpack (expected 3, got ${state.length})`);
    const internal = state[1];
    const gaussNext = state[2];
    if (gaussNext !== null && typeof gaussNext !== 'number') throw new TypeError('gauss_next must be a float or None');
    if (!Array.isArray(internal)) throw new TypeError('state vector must be a tuple');
    if (internal.length !== N + 1) throw new RangeError('state vector is the wrong size');
    const words = new Uint32Array(N);
    for (let i = 0; i < N; i += 1) {
      const element = internal[i];
      let word: bigint;
      if (typeof element === 'bigint') word = element;
      else if (typeof element === 'number' && Number.isInteger(element)) word = BigInt(element);
      else throw new TypeError('state vector items must be integers');
      if (version === 2) word = ((word % (1n << 32n)) + (1n << 32n)) % (1n << 32n);
      if (word < 0n) throw new RangeError("can't convert negative int to unsigned");
      if (word > MASK64) throw new RangeError('Python int too large to convert to C unsigned long');
      words[i] = Number(word & 0xffffffffn);
    }
    const index = internal[N];
    if (typeof index !== 'number' || !Number.isInteger(index)) throw new TypeError('state index must be an integer');
    if (index < 0 || index > N) throw new RangeError('invalid state');
    this.mt.set(words);
    this.index = index;
    this.gaussNext = gaussNext as number | null;
  }
}

/** The process-wide ``random`` module instance (``random._inst``), seeded from the operating system. */
export const pythonRandom = new PythonRandom();
