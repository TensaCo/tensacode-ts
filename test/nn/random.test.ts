/**
 * Bitwise parity of the PyTorch CPU generator, its samplers, default layer
 * initialization and CPython's ``random`` against fixtures written by
 * ``scripts/fixtures/random_fixtures.py``.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  Conv2d, Embedding, EmbeddingBag, F, GRU, GRUCell, Generator, LayerNorm, Linear, MultiheadAttention, PythonRandom,
  arange, bernoulli, bernoulli_, exponential_, getRngState, rngStateTensor, init, manualSeed, multinomial, rand, randint,
  randn, randperm, setRngState, tensor, zeros, type Module, type Tensor,
} from '../../src/nn/index.js';
import { float32ToBFloat16Bits, float32ToFloat16Bits, type DType } from '../../src/nn/dtype.js';
import * as libm from '../../src/nn/randomMath.js';

interface Raw { dtype: string; shape: number[]; bytes: string }
interface Op {
  op: string; shape?: number[]; dtype?: string; low?: number; high?: number; mean?: number; std?: number; n?: number;
  p?: number; size?: number; lambd?: number; rows?: number; categories?: number; samples?: number; replacement?: boolean;
}

const fixture = JSON.parse(readFileSync(join(__dirname, '..', 'fixtures', 'random.json'), 'utf8')) as {
  sequence: Op[];
  generator: { seed: string; initial_state: Raw; outputs: Raw[]; final_state: Raw }[];
  generator_object: { default_state: Raw; seed_99_randn_20: Raw; state: Raw };
  large: { float32: Raw; float64: Raw; state: Raw };
  init: { states: Record<string, Record<string, Raw>>; xavier: Raw; kaiming: Raw; trunc_normal: Raw; state: Raw };
  python_random: {
    seed: { type: string; value: string | number };
    values: Record<string, unknown>;
    state: [number, number[], number | null];
  }[];
  libm: { float: Record<string, [number, number][]>; double: Record<string, [string, string][]>; fma: string[][] };
};

function bytesOf(raw: Raw): Uint8Array {
  return new Uint8Array(Buffer.from(raw.bytes, 'base64'));
}

/** Raw IEEE bits / integer values of a fixture tensor, as numbers or bigints. */
function expectedValues(raw: Raw): (number | bigint)[] {
  const bytes = bytesOf(raw);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out: (number | bigint)[] = [];
  const width = { float32: 4, float64: 8, float16: 2, bfloat16: 2, int64: 8, uint8: 1 }[raw.dtype];
  if (width === undefined) throw new Error(`unsupported fixture dtype ${raw.dtype}`);
  for (let offset = 0; offset < bytes.length; offset += width) {
    switch (raw.dtype) {
      case 'float32': out.push(view.getUint32(offset, true)); break;
      case 'float64': out.push(view.getBigUint64(offset, true)); break;
      case 'float16': case 'bfloat16': out.push(view.getUint16(offset, true)); break;
      case 'int64': out.push(view.getBigInt64(offset, true)); break;
      default: out.push(view.getUint8(offset));
    }
  }
  return out;
}

function actualValues(value: Tensor): (number | bigint)[] {
  const buffer = new DataView(new ArrayBuffer(8));
  return Array.from(value.data, (item) => {
    switch (value.dtype) {
      case 'float32': buffer.setFloat32(0, item); return buffer.getUint32(0);
      case 'float64': buffer.setFloat64(0, item); return buffer.getBigUint64(0);
      case 'float16': return float32ToFloat16Bits(item);
      case 'bfloat16': return float32ToBFloat16Bits(item);
      case 'int64': return BigInt(item);
      default: return item;
    }
  });
}

function expectBitwise(value: Tensor, raw: Raw): void {
  expect(value.dtype).toBe(raw.dtype);
  expect([...value.shape]).toEqual(raw.shape);
  expect(actualValues(value)).toEqual(expectedValues(raw));
}

function run(op: Op): Tensor {
  const dtype = op.dtype as DType | undefined;
  switch (op.op) {
    case 'rand': return rand(op.shape!, { dtype });
    case 'randn': return randn(op.shape!, { dtype });
    case 'uniform': return zeros(op.shape!, { dtype }).uniform_(op.low, op.high);
    case 'normal': return zeros(op.shape!, { dtype }).normal_(op.mean, op.std);
    case 'randint': return randint(op.low!, op.high!, op.shape!);
    case 'randperm': return randperm(op.n!);
    case 'bernoulli': return bernoulli_(zeros(op.shape!, { dtype }), op.p);
    case 'bernoulli_tensor': return bernoulli(arange(op.size!, undefined, 1, { dtype }).div(op.size! - 1));
    case 'exponential': return exponential_(zeros(op.shape!, { dtype }), op.lambd);
    case 'dropout': {
      const count = op.shape!.reduce((a, b) => a * b, 1);
      const x = arange(count, undefined, 1, { dtype }).div(4).sub(3).reshape(op.shape!);
      return F.dropout(x, op.p!, true);
    }
    case 'multinomial': {
      const row = Array.from({ length: op.categories! }, (_, index) => index + 1);
      const base = tensor(Array.from({ length: op.rows! }, () => row), { dtype });
      const weights = base.mul(base).div(8);
      return multinomial(weights, op.samples!, { replacement: op.replacement });
    }
    default: throw new Error(op.op);
  }
}

describe('PyTorch CPU generator', () => {
  for (const testCase of fixture.generator) {
    it(`reproduces torch.manual_seed(${testCase.seed}) sampling bit for bit`, () => {
      manualSeed(BigInt(testCase.seed));
      expectBitwise(rngStateTensor(), testCase.initial_state);
      fixture.sequence.forEach((op, index) => {
        const output = run(op);
        try {
          expectBitwise(output, testCase.outputs[index]!);
        } catch (error) {
          throw new Error(`${op.op} ${JSON.stringify(op)}: ${(error as Error).message}`);
        }
      });
      expectBitwise(rngStateTensor(), testCase.final_state);
    });
  }

  it('matches torch.Generator() defaults, explicit generators and get_state()', () => {
    const generator = new Generator();
    expectBitwise(rngStateTensor(generator.getState()), fixture.generator_object.default_state);
    generator.manualSeed(99);
    expectBitwise(randn([20], { generator }), fixture.generator_object.seed_99_randn_20);
    expectBitwise(rngStateTensor(generator.getState()), fixture.generator_object.state);
  });

  it('matches large vectorized normal fills in float32 and float64', () => {
    manualSeed(3);
    expectBitwise(randn([20_003]), fixture.large.float32);
    expectBitwise(randn([4_001], { dtype: 'float64' }), fixture.large.float64);
    expectBitwise(rngStateTensor(), fixture.large.state);
  });

  it('restores torch.get_rng_state() bytes, including the cached normal sample', () => {
    const last = fixture.generator[0]!;
    setRngState(bytesOf(last.final_state));
    expectBitwise(rngStateTensor(), last.final_state);
    const replay = randn([5], { dtype: 'float64' });
    setRngState(bytesOf(last.final_state));
    expect(Array.from(randn([5], { dtype: 'float64' }).data)).toEqual(Array.from(replay.data));
    const corrupt = bytesOf(last.final_state);
    corrupt[8] = 0; corrupt[9] = 0; corrupt[10] = 0; corrupt[11] = 0; // left = 0
    expect(() => setRngState(corrupt)).toThrow('Invalid mt19937 state');
    expect(() => setRngState(new Uint8Array(12))).toThrow('CPUGeneratorImplState of size 5056');
  });

  it('accepts negative and 64-bit seeds like torch.manual_seed', () => {
    expect(() => manualSeed(1.5)).toThrow(TypeError);
    expect(() => manualSeed(2n ** 64n)).toThrow('Overflow');
    expect(manualSeed(-1).initialSeed()).toBe(2n ** 64n - 1n);
  });
});

describe('default initialization', () => {
  it('matches PyTorch layer construction under the same seed', () => {
    manualSeed(5);
    const modules: Record<string, Module> = {
      linear: new Linear(37, 11),
      embedding: new Embedding(13, 6, { paddingIdx: 2 }),
      bag: new EmbeddingBag(9, 4, { mode: 'mean' }),
      conv: new Conv2d(3, 5, 3, { stride: 2 }),
      gru: new GRU(6, 4, { batchFirst: true }),
      cell: new GRUCell(5, 3),
      attention: new MultiheadAttention(8, 1, { batchFirst: true }),
      norm: new LayerNorm(7),
    };
    const xavier = init.xavierUniform_(zeros([9, 4]));
    const kaiming = init.kaimingUniform_(zeros([6, 10]), Math.sqrt(5));
    const trunc = init.truncNormal_(zeros([50]), 0, 0.02);
    for (const [name, module] of Object.entries(modules)) {
      const state = module.stateDict();
      const expected = fixture.init.states[name]!;
      expect([...state.keys()].sort(), name).toEqual(Object.keys(expected).sort());
      for (const [key, raw] of Object.entries(expected)) {
        try {
          expectBitwise(state.get(key)!, raw);
        } catch (error) {
          throw new Error(`${name}.${key}: ${(error as Error).message}`);
        }
      }
    }
    expectBitwise(xavier, fixture.init.xavier);
    expectBitwise(kaiming, fixture.init.kaiming);
    expectBitwise(trunc, fixture.init.trunc_normal);
    expectBitwise(rngStateTensor(), fixture.init.state);
  });
});

describe('CPython random', () => {
  for (const testCase of fixture.python_random) {
    it(`reproduces random.Random(${JSON.stringify(testCase.seed.value)})`, () => {
      const seed = testCase.seed;
      const value = seed.type === 'int' ? BigInt(seed.value as string)
        : seed.type === 'bytes' ? new Uint8Array(Buffer.from(seed.value as string, 'base64'))
          : seed.value;
      const rng = new PythonRandom(value);
      const expected = testCase.values;
      const shuffled = Array.from({ length: 15 }, (_, index) => index);
      const actual = {
        random: Array.from({ length: 5 }, () => rng.random()),
        getrandbits: [1, 7, 32, 33, 64, 100].map((k) => rng.getrandbits(k).toString()),
        randrange: [rng.randrange(10), rng.randrange(3, 1000), rng.randrange(-50, 50, 7), rng.randrange(100, 0, -3), rng.randint(1, 6)],
        big_randrange: rng.randrange(2n ** 80n).toString(),
        choice: rng.choice('abcdefghij'),
        shuffle: (rng.shuffle(shuffled), shuffled),
        sample_small: rng.sample(Array.from({ length: 30 }, (_, index) => index), 5),
        sample_large: rng.sample(Array.from({ length: 1000 }, (_, index) => index), 12),
        sample_counts: rng.sample(['a', 'b', 'c'], 4, { counts: [3, 1, 2] }),
        choices: rng.choices('xyz', { k: 4 }),
        weighted_choices: rng.choices('xyz', { weights: [1, 5, 2], k: 5 }),
        uniform: rng.uniform(-2, 3),
        gauss: Array.from({ length: 5 }, () => rng.gauss(0.5, 2)),
      };
      expect(actual).toEqual(expected);
      const state = rng.getstate();
      expect([state[0], [...state[1]], state[2]]).toEqual(testCase.state);
      const next = [rng.random(), rng.gauss()];
      const restored = new PythonRandom(0);
      restored.setstate(testCase.state);
      expect([restored.random(), restored.gauss()]).toEqual(next);
    });
  }

  it('validates setstate like CPython', () => {
    const rng = new PythonRandom(1);
    expect(() => rng.setstate([4, [], null])).toThrow('version 4');
    expect(() => rng.setstate([3, [1, 2], null])).toThrow('wrong size');
    const state = rng.getstate();
    expect(() => rng.setstate([3, [...state[1].slice(0, 624), 625], null])).toThrow('invalid state');
    expect(() => rng.setstate([3, [-1, ...state[1].slice(1)], null])).toThrow('negative');
  });
});

describe('C library ports', () => {
  const view = new DataView(new ArrayBuffer(8));
  const fromFloatBits = (bits: number) => { view.setUint32(0, bits); return view.getFloat32(0); };
  const floatBits = (value: number) => { view.setFloat32(0, value); return view.getUint32(0); };
  const fromDoubleBits = (hex: string) => { view.setBigUint64(0, BigInt(`0x${hex}`)); return view.getFloat64(0); };
  const doubleBits = (value: number) => { view.setFloat64(0, value); return view.getBigUint64(0).toString(16).padStart(16, '0'); };
  const floats = { logf: libm.logf, sinf: libm.sinf, cosf: libm.cosf };
  const doubles = { log: libm.log, log1p: libm.log1p, sin: libm.sin, cos: libm.cos };

  for (const [name, fn] of Object.entries(floats)) {
    it(`${name} matches glibc bit for bit`, () => {
      const cases = fixture.libm.float[name]!;
      expect(cases.filter(([input, output]) => floatBits(fn(fromFloatBits(input))) !== output)).toEqual([]);
    });
  }
  for (const [name, fn] of Object.entries(doubles)) {
    it(`${name} matches glibc bit for bit`, () => {
      const cases = fixture.libm.double[name]!;
      expect(cases.filter(([input, output]) => doubleBits(fn(fromDoubleBits(input))) !== output)).toEqual([]);
    });
  }
  it('fma rounds once', () => {
    expect(fixture.libm.fma.filter(([a, b, c, out]) => doubleBits(libm.fma(fromDoubleBits(a!), fromDoubleBits(b!), fromDoubleBits(c!))) !== out)).toEqual([]);
    expect(libm.fma(0.1, 10, -1)).toBe(5.551115123125783e-17);
    expect(libm.fma(2 ** 1000, 2 ** 1000, -Infinity)).toBe(-Infinity);
    expect(libm.fma(2 ** -600, 2 ** -600, 0)).toBe(0);
    expect(libm.fmaf(Math.fround(1.1), Math.fround(1.1), Math.fround(-1.21))).toBe(Math.fround(Math.fround(1.1) * Math.fround(1.1) - Math.fround(1.21)));
  });
});
