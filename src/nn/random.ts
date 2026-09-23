/**
 * PyTorch's CPU random number generator, reproduced exactly.
 *
 * ``Generator`` is ``at::CPUGeneratorImpl``: a 32-bit Mersenne Twister
 * (``at::mt19937``) seeded like ``torch.manual_seed``, with the same cached
 * normal samples and the same 5056-byte state layout as
 * ``torch.get_rng_state()``. The sampling kernels below follow ATen's CPU
 * kernels (``uniform_``, ``normal_`` including the 16-wide Box-Muller fill,
 * ``bernoulli_``, ``random_``, ``exponential_``, ``randperm`` and
 * ``multinomial``), so ``manualSeed(n)`` followed by the same calls produces
 * bitwise-identical tensors to ``torch.manual_seed(n)`` in Python.
 *
 * Transcendental functions come from ``randomMath.ts``, which ports the C
 * library of the reference platform (glibc 2.39, AArch64). PyTorch on x86-64
 * with AVX2 uses a different vectorized ``normal_`` fill, so Python itself
 * produces different normal samples there.
 */
import { roundToDType, type DType } from './dtype.js';
import { Kernel, activeEngine } from './backend/engine.js';
import { cos, fma, fmaf, log1p, normalFill16Double, normalFill16Float, sin } from './randomMath.js';

const MERSENNE_STATE_N = 624;
const MERSENNE_STATE_M = 397;
const MATRIX_A = 0x9908b0df;
const UMASK = 0x80000000;
const LMASK = 0x7fffffff;

/** ``default_rng_seed_val``: the seed of a newly constructed ``torch.Generator()``. */
export const DEFAULT_RNG_SEED = 67280421310721n;

/** Byte size of ``torch.get_rng_state()`` (``CPUGeneratorImplState``). */
export const RNG_STATE_SIZE = 5056;

const TWO_PI = 2 * Math.PI;
const FLOAT_MASK = 0xffffff;
const FLOAT_DIVISOR = 2 ** -24;
const DOUBLE_DIVISOR = 2 ** -53;
const UINT64_MASK = (1n << 64n) - 1n;

/** A seed accepted by ``manualSeed``: an integer (negative values wrap like PyTorch). */
export type Seed = number | bigint;

function seedToUint64(seed: Seed): bigint {
  let value: bigint;
  if (typeof seed === 'bigint') {
    value = seed;
  } else {
    if (typeof seed !== 'number' || !Number.isFinite(seed) || !Number.isInteger(seed)) {
      throw new TypeError('seed must be an integer');
    }
    if (!Number.isSafeInteger(seed)) throw new RangeError('seed must be a safe integer; pass a bigint for larger seeds');
    value = BigInt(seed);
  }
  if (value < -(1n << 63n) || value > UINT64_MASK) {
    throw new RangeError('Overflow when unpacking long');
  }
  return value & UINT64_MASK;
}

function nondeterministicSeed(): bigint {
  const words = new Uint32Array(2);
  globalThis.crypto.getRandomValues(words);
  return (BigInt(words[0]!) << 32n) | BigInt(words[1]!);
}

function isStateTensor(value: unknown): value is { data: ArrayLike<number>; dtype: string; numel: number } {
  return typeof value === 'object' && value !== null && 'data' in value && 'dtype' in value;
}

/**
 * ``torch.Generator`` for the CPU: the Mersenne Twister engine plus the cached
 * second Box-Muller samples.
 */
export class Generator {
  private readonly state = new Uint32Array(MERSENNE_STATE_N);
  private left = 1;
  private next = 0;
  private seedValue = DEFAULT_RNG_SEED;
  /** Cached ``normal_distribution<double>`` sample. */
  nextDoubleNormalSample: number | null = null;
  /** Cached ``normal_distribution<float>`` sample. */
  nextFloatNormalSample: number | null = null;

  constructor(seed: Seed = DEFAULT_RNG_SEED) {
    this.manualSeed(seed);
  }

  /** Seed the engine (``Generator.manual_seed``); clears cached normal samples. */
  manualSeed(seed: Seed): this {
    const value = seedToUint64(seed);
    this.nextFloatNormalSample = null;
    this.nextDoubleNormalSample = null;
    this.seedValue = value;
    const state = this.state;
    state[0] = Number(value & 0xffffffffn);
    for (let j = 1; j < MERSENNE_STATE_N; j += 1) {
      const previous = state[j - 1]!;
      state[j] = (Math.imul(1812433253, (previous ^ (previous >>> 30)) >>> 0) + j) >>> 0;
    }
    this.left = 1;
    this.next = 0;
    return this;
  }

  /** Reseed from a nondeterministic source and return the seed (``Generator.seed``). */
  seed(): bigint {
    const value = nondeterministicSeed();
    this.manualSeed(value);
    return value;
  }

  /** The seed last used to initialize the engine (``Generator.initial_seed``). */
  initialSeed(): bigint {
    return this.seedValue;
  }

  private nextState(): void {
    const state = this.state;
    this.left = MERSENNE_STATE_N;
    this.next = 0;
    let p = 0;
    for (let j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p += 1) {
      const u = state[p]!;
      const v = state[p + 1]!;
      const mixed = ((u & UMASK) | (v & LMASK)) >>> 0;
      state[p] = (state[p + MERSENNE_STATE_M]! ^ (mixed >>> 1) ^ (v & 1 ? MATRIX_A : 0)) >>> 0;
    }
    for (let j = MERSENNE_STATE_M; --j; p += 1) {
      const u = state[p]!;
      const v = state[p + 1]!;
      const mixed = ((u & UMASK) | (v & LMASK)) >>> 0;
      state[p] = (state[p + MERSENNE_STATE_M - MERSENNE_STATE_N]! ^ (mixed >>> 1) ^ (v & 1 ? MATRIX_A : 0)) >>> 0;
    }
    const u = state[p]!;
    const v = state[0]!;
    const mixed = ((u & UMASK) | (v & LMASK)) >>> 0;
    state[p] = (state[p + MERSENNE_STATE_M - MERSENNE_STATE_N]! ^ (mixed >>> 1) ^ (v & 1 ? MATRIX_A : 0)) >>> 0;
  }

  /** Next tempered 32-bit output (``CPUGeneratorImpl::random``). */
  randomUint32(): number {
    this.left -= 1;
    if (this.left === 0) this.nextState();
    let y = this.state[this.next]!;
    this.next += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** Next 64-bit output (``CPUGeneratorImpl::random64``): the first 32-bit draw is the high word. */
  random64(): bigint {
    const high = this.randomUint32();
    const low = this.randomUint32();
    return (BigInt(high) << 32n) | BigInt(low);
  }

  /** ``uniform_real_distribution<float>(0, 1)``: 24 random bits. */
  uniformFloat(): number {
    return (this.randomUint32() & FLOAT_MASK) * FLOAT_DIVISOR;
  }

  /** ``uniform_real_distribution<double>(0, 1)``: 53 random bits of ``random64``. */
  uniformDouble(): number {
    const high = this.randomUint32();
    const low = this.randomUint32();
    return ((high & 0x1fffff) * 4294967296 + low) * DOUBLE_DIVISOR;
  }

  /** A uniform double in [0, 1) (``uniform_real_distribution<double>``). */
  random(): number {
    return this.uniformDouble();
  }

  /** ``normal_distribution<double>(mean, std)`` with the cached second sample. */
  normalDouble(mean = 0, std = 1): number {
    const cached = this.nextDoubleNormalSample;
    if (cached !== null) {
      this.nextDoubleNormalSample = null;
      return fma(cached, std, mean);
    }
    const u1 = this.uniformDouble();
    const u2 = this.uniformDouble();
    const r = Math.sqrt(-2 * log1p(-u2));
    const theta = TWO_PI * u1;
    this.nextDoubleNormalSample = r * sin(theta);
    return fma(r * cos(theta), std, mean);
  }

  /** A standard normal double (``normal_distribution<double>(0, 1)``). */
  normal(): number {
    return this.normalDouble(0, 1);
  }

  /** A uniform integer in [low, high) (``random_(low, high)``). */
  integer(low: number, high: number): number {
    if (!(high > low)) throw new RangeError('random_ expects \'from\' to be less than \'to\'');
    return randomFromTo(this, high - low, low);
  }

  /** The engine state as ``torch.Generator.get_state()`` bytes. */
  getState(): Uint8Array {
    const bytes = new Uint8Array(RNG_STATE_SIZE);
    const view = new DataView(bytes.buffer);
    view.setBigUint64(0, this.seedValue, true);
    view.setInt32(8, this.left, true);
    view.setInt32(12, 1, true);
    view.setBigUint64(16, BigInt(this.next), true);
    for (let index = 0; index < MERSENNE_STATE_N; index += 1) {
      view.setUint32(24 + 8 * index, this.state[index]!, true);
    }
    // normal_x (5016) and normal_rho (5032) are unused and stay zero.
    if (this.nextDoubleNormalSample !== null) {
      view.setFloat64(5024, this.nextDoubleNormalSample, true);
      view.setInt32(5040, 1, true);
    }
    if (this.nextFloatNormalSample !== null) {
      view.setFloat32(5048, this.nextFloatNormalSample, true);
      bytes[5052] = 1;
    }
    return bytes;
  }

  /**
   * Restore ``torch.Generator.get_state()`` bytes (a ``uint8`` tensor or a
   * ``Uint8Array`` of 5056 bytes), validated like ``set_state``.
   */
  setState(state: Uint8Array | { data: ArrayLike<number>; dtype: string; numel: number }): this {
    let bytes: Uint8Array;
    if (state instanceof Uint8Array) {
      bytes = state;
    } else if (isStateTensor(state)) {
      if (state.dtype !== 'uint8') {
        throw new TypeError(`expected a torch.ByteTensor, but got ${state.dtype}`);
      }
      bytes = Uint8Array.from(state.data as ArrayLike<number>);
    } else {
      throw new TypeError('RNG state must be a torch.ByteTensor');
    }
    if (bytes.length !== RNG_STATE_SIZE) {
      throw new RangeError(`Expected a CPUGeneratorImplState of size ${RNG_STATE_SIZE} but found the input RNG state size to be ${bytes.length}`);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const seeded = view.getInt32(12, true) !== 0;
    const left = view.getInt32(8, true);
    const next = Number(view.getBigUint64(16, true) & 0xffffffffn);
    if (!seeded || !(left > 0 && left <= MERSENNE_STATE_N) || next > MERSENNE_STATE_N) {
      throw new RangeError('Invalid mt19937 state');
    }
    for (let index = 0; index < MERSENNE_STATE_N; index += 1) {
      this.state[index] = view.getUint32(24 + 8 * index, true);
    }
    this.seedValue = view.getBigUint64(0, true);
    this.left = left;
    this.next = next;
    this.nextDoubleNormalSample = view.getInt32(5040, true) !== 0 ? view.getFloat64(5024, true) : null;
    this.nextFloatNormalSample = bytes[5052] !== 0 ? view.getFloat32(5048, true) : null;
    return this;
  }

  /** An independent copy with the same state (``Generator.clone_state``). */
  clone(): Generator {
    return new Generator().setState(this.getState());
  }
}

let defaultGenerator = new Generator(nondeterministicSeed());

/** The process-wide generator used when no explicit generator is supplied (``torch.default_generator``). */
export function getDefaultGenerator(): Generator {
  return defaultGenerator;
}

/** Seed the default generator (``torch.manual_seed``). */
export function manualSeed(seed: Seed): Generator {
  defaultGenerator.manualSeed(seed);
  return defaultGenerator;
}

/** Reseed the default generator nondeterministically and return the seed (``torch.seed``). */
export function seed(): bigint {
  return defaultGenerator.seed();
}

/** The default generator's initial seed (``torch.initial_seed``). */
export function initialSeed(): bigint {
  return defaultGenerator.initialSeed();
}

/** Replace the default generator (primarily for tests). */
export function setDefaultGenerator(generator: Generator): void {
  defaultGenerator = generator;
}

// ---------------------------------------------------------------------------
// Construction without random draws (``torch.device('meta')``).
// ---------------------------------------------------------------------------

let suppressedInit = 0;

/**
 * Run ``build`` with random initialization suppressed, like constructing
 * modules under ``torch.device('meta')`` in Python: ``normal_``/``uniform_``
 * leave tensors unchanged and draw nothing, so the generator state is the same
 * as Python's after a meta-device construction. Use it for objects whose
 * weights are replaced by loaded or supplied tensors.
 */
export function withoutRandomInit<T>(build: () => T): T {
  suppressedInit += 1;
  try {
    return build();
  } finally {
    suppressedInit -= 1;
  }
}

/** Whether random initialization is currently suppressed (see {@link withoutRandomInit}). */
export function randomInitSuppressed(): boolean {
  return suppressedInit > 0;
}

// ---------------------------------------------------------------------------
// Sampling kernels (ATen ``native/cpu/DistributionTemplates.h``).
// ---------------------------------------------------------------------------

type FloatData = Float32Array | Float64Array;

function requireFloating(dtype: DType, operation: string): void {
  if (dtype !== 'float32' && dtype !== 'float64' && dtype !== 'float16' && dtype !== 'bfloat16') {
    throw new TypeError(`"${operation}" not implemented for '${dtype}'`);
  }
}

/** Round a float32 value to ``dtype`` (float32 storage already holds float32 values). */
function narrow(dtype: DType, value: number): number {
  return dtype === 'float16' || dtype === 'bfloat16' ? roundToDType(dtype, value) : value;
}

/** ``uniform_(from, to)`` over contiguous storage. */
export function fillUniform(data: FloatData, dtype: DType, from: number, to: number, generator: Generator): void {
  requireFloating(dtype, 'uniform_kernel_cpu');
  if (!(from <= to)) throw new RangeError(`uniform_ expects to return a [from, to) range, but found from=${from} > to=${to}`);
  if (suppressedInit > 0) return;
  if (dtype === 'float64') {
    const range = to - from;
    for (let index = 0; index < data.length; index += 1) {
      const value = fma(generator.uniformDouble(), range, from);
      data[index] = value === to ? from : value;
    }
    return;
  }
  const low = Math.fround(from);
  const high = Math.fround(to);
  const range = Math.fround(high - low);
  const lowScalar = narrow(dtype, low);
  const highScalar = narrow(dtype, high);
  for (let index = 0; index < data.length; index += 1) {
    const value = narrow(dtype, fmaf(generator.uniformFloat(), range, low));
    data[index] = value === highScalar ? lowScalar : value;
  }
}

/** Elements transformed per step of the 16-wide fill (a multiple of 16). */
const NORMAL_CHUNK = 1 << 16;
/** Elements per step when the transforms run on the worker pool. */
const NORMAL_PARALLEL_CHUNK = 1 << 20;
/** Smallest fill that uses the worker pool. */
const NORMAL_PARALLEL_MIN = 1 << 15;

/**
 * ATen's vectorized ``normal_`` for a contiguous tensor of at least 16
 * elements: float32/float64 draw every uniform first and then transform
 * blocks of 16 in place; float16/bfloat16 draw each block's uniforms just
 * before its transform. Either way the transforms draw nothing, so the
 * uniforms are drawn in order a chunk at a time and each chunk's blocks are
 * transformed (on the worker pool when there is one) before the next chunk.
 * A trailing partial block is recomputed from 16 fresh uniforms.
 */
function fillNormalBlocks(data: FloatData, dtype: DType, mean: number, std: number, generator: Generator): void {
  const size = data.length;
  const double = dtype === 'float64';
  const reduced = dtype === 'float16' || dtype === 'bfloat16';
  const fill = double ? normalFill16Double : normalFill16Float;
  const opMean = double ? mean : Math.fround(mean);
  const opStd = double ? std : Math.fround(std);
  const draw = (target: Float64Array, count: number): void => {
    if (double) for (let index = 0; index < count; index += 1) target[index] = generator.uniformDouble();
    else for (let index = 0; index < count; index += 1) target[index] = generator.uniformFloat();
  };
  const store = (source: Float64Array, offset: number, count: number): void => {
    if (reduced) for (let index = 0; index < count; index += 1) data[offset + index] = roundToDType(dtype, source[index]!);
    else data.set(count === source.length ? source : source.subarray(0, count), offset);
  };
  const full = size - (size % 16);
  let engine = full >= NORMAL_PARALLEL_MIN ? activeEngine() : null;
  if (engine && engine.parallelism() <= 1) engine = null;
  const chunk = engine ? NORMAL_PARALLEL_CHUNK : NORMAL_CHUNK;
  const bytes = 16 + chunk * 8;
  let pointer = 0;
  if (engine) {
    engine.beginCall();
    pointer = engine.alloc(bytes);
    if (!pointer) engine = null;
  }
  try {
    const local = engine ? null : new Float64Array(Math.min(chunk, full));
    for (let start = 0; start < full; start += chunk) {
      const count = Math.min(chunk, full - start);
      if (engine) {
        const params = new Float64Array(engine.memory.buffer, pointer, 2);
        params[0] = opMean;
        params[1] = opStd;
        const buffer = new Float64Array(engine.memory.buffer, pointer + 16, count);
        draw(buffer, count);
        const blocks = count / 16;
        const perTask = Math.max(256, Math.ceil(blocks / (engine.parallelism() * 4)));
        engine.run(Kernel.NormalFill, [pointer, blocks, perTask, double ? 1 : 0], Math.ceil(blocks / perTask), true);
        store(buffer, start, count);
      } else {
        draw(local!, count);
        for (let offset = 0; offset < count; offset += 16) fill(local!, offset, opMean, opStd);
        store(local!, start, count);
      }
    }
  } finally {
    if (engine) engine.release(pointer, bytes);
  }
  if (size === full) return;
  // float32/float64 drew the partial block's uniforms with the rest.
  if (!reduced) draw(new Float64Array(size - full), size - full);
  const tail = new Float64Array(16);
  draw(tail, 16);
  fill(tail, 0, opMean, opStd);
  store(tail, size - 16, 16);
}

/**
 * ``normal_(mean, std)`` over storage. Contiguous tensors with at least 16
 * elements use ATen's vectorized Box-Muller fill; smaller or strided tensors
 * draw from ``normal_distribution<double>`` one element at a time.
 */
export function fillNormal(
  data: FloatData, dtype: DType, mean: number, std: number, generator: Generator, contiguous = true,
): void {
  requireFloating(dtype, 'normal_kernel_cpu');
  if (!(std >= 0)) throw new RangeError(`normal expects std >= 0.0, but found std ${std}`);
  if (suppressedInit > 0) return;
  const size = data.length;
  if (size >= 16 && contiguous) {
    fillNormalBlocks(data, dtype, mean, std, generator);
    return;
  }
  for (let index = 0; index < size; index += 1) {
    const value = generator.normalDouble(mean, std);
    data[index] = dtype === 'float64' ? value : narrow(dtype, Math.fround(value));
  }
}

/** ``bernoulli_(p)`` with a scalar probability (``bernoulli_distribution<double>``). */
export function fillBernoulli(data: FloatData, dtype: DType, p: number, generator: Generator): void {
  if (!(p >= 0 && p <= 1)) throw new RangeError(`bernoulli_ expects p to be in [0, 1], but got p=${p}`);
  for (let index = 0; index < data.length; index += 1) {
    data[index] = generator.uniformDouble() < p ? 1 : 0;
  }
  void dtype;
}

/**
 * ``bernoulli_(p)`` with a probability tensor of the same size. Float64
 * probabilities draw doubles; other floating probabilities draw floats.
 */
export function fillBernoulliTensor(
  data: FloatData, probabilities: ArrayLike<number>, probabilityDType: DType, generator: Generator,
): void {
  requireFloating(probabilityDType, 'bernoulli_tensor_cpu_p_');
  const double = probabilityDType === 'float64';
  for (let index = 0; index < data.length; index += 1) {
    const p = probabilities[index]!;
    if (!(p >= 0 && p <= 1)) throw new RangeError('Expected p_in >= 0 && p_in <= 1 to be true');
    data[index] = (double ? generator.uniformDouble() : generator.uniformFloat()) < p ? 1 : 0;
  }
}

/** One ``uniform_int_from_to_distribution`` draw in ``[base, base + range)``. */
export function randomFromTo(generator: Generator, range: number, base: number): number {
  if (range >= 2 ** 28) {
    return Number(generator.random64() % BigInt(range)) + base;
  }
  return (generator.randomUint32() % range) + base;
}

/** ``random_(from, to)`` over storage. */
export function fillRandomFromTo(data: FloatData, from: number, to: number, generator: Generator): void {
  if (!(from < to)) throw new RangeError(`random_ expects 'from' to be less than 'to', but got from=${from} >= to=${to}`);
  const range = to - from;
  for (let index = 0; index < data.length; index += 1) data[index] = randomFromTo(generator, range, from);
}

/** ``exponential_(lambd)`` over storage (``exponential_distribution<double>``). */
export function fillExponential(data: FloatData, dtype: DType, lambd: number, generator: Generator): void {
  requireFloating(dtype, 'exponential_cpu');
  if (!(lambd > 0)) throw new RangeError(`exponential_ expects lambda > 0.0, but found lambda=${lambd}`);
  const scale = -1 / lambd;
  for (let index = 0; index < data.length; index += 1) {
    const value = scale * log1p(-generator.uniformDouble());
    data[index] = dtype === 'float64' ? value : narrow(dtype, Math.fround(value));
  }
}

/** ``torch.randperm(n)`` values (``randperm_cpu``). */
export function randpermValues(n: number, generator: Generator): Float64Array {
  if (!Number.isInteger(n) || n < 0) throw new RangeError(`n must be non-negative, got${n}`);
  const result = new Float64Array(n);
  if (n < Math.floor(0xffffffff / 20)) {
    for (let index = 0; index < n; index += 1) result[index] = index;
    for (let index = 0; index < n - 1; index += 1) {
      const z = generator.randomUint32() % (n - index);
      const save = result[index]!;
      result[index] = result[z + index]!;
      result[z + index] = save;
    }
    return result;
  }
  for (let index = 0; index < n; index += 1) {
    const z = Number(generator.random64() % BigInt(index + 1));
    result[index] = result[z]!;
    result[z] = index;
  }
  return result;
}

/**
 * ``torch.multinomial`` over rows of ``probabilities`` (``rows x categories``).
 * Without replacement, or for a single sample, it takes the top ``samples`` of
 * ``p / q`` with ``q ~ Exp(1)`` (drawn for the whole input first); with
 * replacement it inverts the normalized cumulative distribution.
 */
export function multinomialValues(
  probabilities: ArrayLike<number>, dtype: DType, rows: number, categories: number, samples: number,
  replacement: boolean, generator: Generator,
): Float64Array {
  requireFloating(dtype, 'multinomial');
  if (!(samples > 0)) throw new RangeError('cannot sample n_sample <= 0 samples');
  if (!replacement && samples > categories) {
    throw new RangeError('cannot sample n_sample > prob_dist.size(-1) samples without replacement');
  }
  if (categories > 2 ** 24) throw new RangeError('number of categories cannot exceed 2^24');
  const result = new Float64Array(rows * samples);
  const double = dtype === 'float64';
  const round = (value: number) => (double ? value : narrow(dtype, Math.fround(value)));
  if (!replacement || samples === 1) {
    for (let row = 0; row < rows; row += 1) {
      let sum = 0;
      for (let column = 0; column < categories; column += 1) {
        const value = probabilities[row * categories + column]!;
        if (!(value >= 0) || value === Infinity) {
          throw new RangeError('probability tensor contains either `inf`, `nan` or element < 0');
        }
        sum += value;
      }
      if (sum === 0) throw new RangeError('invalid multinomial distribution (sum of probabilities <= 0)');
    }
    const q = new Float64Array(rows * categories);
    const scale = -1;
    for (let index = 0; index < q.length; index += 1) q[index] = round(scale * log1p(-generator.uniformDouble()));
    for (let index = 0; index < q.length; index += 1) q[index] = round(probabilities[index]! / q[index]!);
    for (let row = 0; row < rows; row += 1) {
      const offset = row * categories;
      if (samples === 1) {
        let best = 0;
        let bestValue = q[offset]!;
        for (let column = 1; column < categories; column += 1) {
          const value = q[offset + column]!;
          if (value > bestValue || (Number.isNaN(value) && !Number.isNaN(bestValue))) {
            best = column;
            bestValue = value;
          }
        }
        result[row] = best;
      } else {
        const order = Array.from({ length: categories }, (_, column) => column);
        order.sort((a, b) => (q[offset + b]! - q[offset + a]!) || a - b);
        for (let sample = 0; sample < samples; sample += 1) result[row * samples + sample] = order[sample]!;
      }
    }
    return result;
  }
  const cumulative = new Float64Array(categories);
  for (let row = 0; row < rows; row += 1) {
    let sum = 0;
    for (let column = 0; column < categories; column += 1) {
      const value = probabilities[row * categories + column]!;
      if (!(value >= 0)) throw new RangeError('invalid multinomial distribution (encountering probability entry < 0)');
      if (!Number.isFinite(value)) {
        throw new RangeError('invalid multinomial distribution (encountering probability entry = infinity or NaN)');
      }
      sum = double ? sum + value : Math.fround(sum + value);
      cumulative[column] = sum;
    }
    if (!(sum > 0)) throw new RangeError('invalid multinomial distribution (sum of probabilities <= 0)');
    for (let column = 0; column < categories; column += 1) {
      cumulative[column] = double ? cumulative[column]! / sum : Math.fround(cumulative[column]! / sum);
    }
    for (let sample = 0; sample < samples; sample += 1) {
      const uniform = generator.uniformDouble();
      let left = 0;
      let right = categories;
      cumulative[categories - 1] = 1;
      while (right - left > 0) {
        const middle = left + Math.floor((right - left) / 2);
        if (cumulative[middle]! < uniform) left = middle + 1;
        else right = middle;
      }
      result[row * samples + sample] = left;
    }
  }
  return result;
}
