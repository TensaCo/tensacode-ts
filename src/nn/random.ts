/**
 * Seedable pseudo-random numbers for initialization, dropout and sampling.
 *
 * The generator is xoshiro128** seeded through splitmix32. Its complete state is
 * a small JSON-safe object so training checkpoints can restore exact streams.
 * Streams are deterministic within this library; they do not reproduce PyTorch's
 * random sequences.
 */

export interface GeneratorState {
  readonly algorithm: 'xoshiro128**';
  readonly words: readonly [number, number, number, number];
  readonly spareNormal: number | null;
}

function splitmix32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

export class Generator {
  private s0 = 0;
  private s1 = 0;
  private s2 = 0;
  private s3 = 0;
  private spare: number | null = null;

  constructor(seed = 0x5eed) {
    this.manualSeed(seed);
  }

  manualSeed(seed: number): this {
    if (!Number.isFinite(seed)) throw new RangeError('seed must be a finite number');
    // Mix both halves of large integer seeds.
    const low = Math.trunc(seed) >>> 0;
    const high = Math.trunc(seed / 2 ** 32) >>> 0;
    const next = splitmix32(low ^ Math.imul(high, 0x9e3779b1));
    this.s0 = next();
    this.s1 = next();
    this.s2 = next();
    this.s3 = next();
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1;
    this.spare = null;
    return this;
  }

  /** Next unsigned 32-bit integer. */
  nextUint32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5) >>> 0, 7), 9) >>> 0;
    const t = (this.s1 << 9) >>> 0;
    this.s2 = (this.s2 ^ this.s0) >>> 0;
    this.s3 = (this.s3 ^ this.s1) >>> 0;
    this.s1 = (this.s1 ^ this.s2) >>> 0;
    this.s0 = (this.s0 ^ this.s3) >>> 0;
    this.s2 = (this.s2 ^ t) >>> 0;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform double in [0, 1) with 53 random bits. */
  random(): number {
    const high = this.nextUint32() >>> 5; // 27 bits
    const low = this.nextUint32() >>> 6; // 26 bits
    return (high * 67108864 + low) / 9007199254740992;
  }

  /** Standard normal sample (Box-Muller with a cached spare value). */
  normal(): number {
    if (this.spare !== null) {
      const value = this.spare;
      this.spare = null;
      return value;
    }
    let u = 0;
    while (u <= Number.MIN_VALUE) u = this.random();
    const v = this.random();
    const radius = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    this.spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  }

  /** Uniform integer in [low, high). */
  integer(low: number, high: number): number {
    if (!(high > low)) throw new RangeError('integer range must be nonempty');
    return low + Math.floor(this.random() * (high - low));
  }

  getState(): GeneratorState {
    return {
      algorithm: 'xoshiro128**',
      words: [this.s0, this.s1, this.s2, this.s3],
      spareNormal: this.spare,
    };
  }

  setState(state: GeneratorState): this {
    if (
      !state || state.algorithm !== 'xoshiro128**' || !Array.isArray(state.words) || state.words.length !== 4
      || state.words.some((word) => !Number.isInteger(word) || word < 0 || word > 0xffffffff)
      || (state.spareNormal !== null && !Number.isFinite(state.spareNormal))
    ) {
      throw new TypeError('Invalid generator state');
    }
    [this.s0, this.s1, this.s2, this.s3] = state.words as [number, number, number, number];
    this.spare = state.spareNormal;
    return this;
  }
}

let defaultGenerator = new Generator();

/** The process-wide generator used when no explicit generator is supplied. */
export function getDefaultGenerator(): Generator {
  return defaultGenerator;
}

/** Seed the default generator (like ``torch.manual_seed``). */
export function manualSeed(seed: number): Generator {
  defaultGenerator.manualSeed(seed);
  return defaultGenerator;
}

export function getRngState(): GeneratorState {
  return defaultGenerator.getState();
}

export function setRngState(state: GeneratorState): void {
  defaultGenerator.setState(state);
}

/** Replace the default generator (primarily for tests). */
export function setDefaultGenerator(generator: Generator): void {
  defaultGenerator = generator;
}
