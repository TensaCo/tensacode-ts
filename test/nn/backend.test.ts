/**
 * WebAssembly SIMD compute backend (``src/nn/backend``): float32 kernels agree
 * with float64 references, results do not depend on batch size or thread
 * count, activations / layer norm / AdamW are bit-identical to the JavaScript
 * kernels, and fused attention equals the composed (differentiable) path.
 */
import { afterEach, describe, expect, it } from 'vitest';
import {
  AdamW, Parameter, backendInfo, enableGrad, finfoMin, getNumThreads, noGrad, setBackend, setNumThreads, tensor, type Tensor,
} from '../../src/nn/index.js';
import { linear, matmul } from '../../src/nn/ops/linalg.js';
import { conv2d, gelu, layerNorm, scaledDotProductAttention, silu } from '../../src/nn/ops/nn.js';
import { softmax } from '../../src/nn/ops/reduce.js';
import { exp, sigmoid, tanh } from '../../src/nn/ops/elementwise.js';
import { expectClose, randomTensor } from '../helpers/gradcheck.js';
import { fixtureJson, fromJson, type TensorJson } from '../helpers/fixtures.js';
import { roundToDType, type DType } from '../../src/nn/dtype.js';
import { logSoftmax } from '../../src/nn/ops/reduce.js';

const available = backendInfo().backend !== 'js';

function random32(shape: number[], seed: number, scale = 1): Tensor {
  return randomTensor(shape, seed, scale, 'float32');
}

function as64(x: Tensor): Tensor {
  return x.to('float64');
}

function bitwiseEqual(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) if (!Object.is(a[index], b[index])) return false;
  return true;
}

function withBackend<T>(name: 'wasm' | 'js', fn: () => T): T {
  setBackend(name);
  try {
    return fn();
  } finally {
    setBackend('wasm');
  }
}

function withThreads<T>(count: number, fn: () => T): T {
  const previous = getNumThreads();
  setNumThreads(count);
  try {
    return fn();
  } finally {
    setNumThreads(previous);
  }
}

afterEach(() => setBackend('wasm'));

describe.skipIf(!available)('WebAssembly matrix products', () => {
  it('linear matches a float64 reference for aligned and ragged shapes', () => {
    for (const [rows, k, n] of [[1, 5, 3], [3, 7, 9], [17, 33, 21], [64, 128, 100], [1, 576, 257]] as const) {
      const x = random32([rows, k], rows + k);
      const w = random32([n, k], n);
      const b = random32([n], 3);
      const actual = noGrad(() => linear(x, w, b));
      const expected = noGrad(() => linear(as64(x), as64(w), as64(b)));
      expect(actual.dtype).toBe('float32');
      expect(actual.shape).toEqual([rows, n]);
      expectClose(actual.data, expected.data, 1e-5, 1e-5);
    }
  });

  it('every row is computed identically whatever the batch size', () => {
    const x = random32([37, 100], 4);
    const w = random32([45, 100], 5);
    const full = noGrad(() => linear(x, w));
    for (let row = 0; row < 37; row += 6) {
      const single = noGrad(() => linear(x.narrow(0, row, 1), w));
      expect(bitwiseEqual(single.data, full.data.subarray(row * 45, (row + 1) * 45))).toBe(true);
    }
  });

  it('results do not depend on the thread count', () => {
    const x = random32([256, 192], 6);
    const w = random32([320, 192], 7);
    const a = random32([4, 64, 96], 8);
    const b = random32([4, 96, 80], 9);
    const one = withThreads(1, () => noGrad(() => [linear(x, w), matmul(a, b)]));
    const four = withThreads(4, () => noGrad(() => [linear(x, w), matmul(a, b)]));
    expect(bitwiseEqual(one[0]!.data, four[0]!.data)).toBe(true);
    expect(bitwiseEqual(one[1]!.data, four[1]!.data)).toBe(true);
  });

  it('resident weights follow in-place updates', () => {
    const x = random32([8, 40], 10);
    const w = new Parameter(random32([30, 40], 11));
    const results = [0, 1, 2].map(() => noGrad(() => linear(x, w)).data.slice());
    expect(bitwiseEqual(results[0]!, results[2]!)).toBe(true);
    noGrad(() => w.add_(0.5));
    const updated = noGrad(() => linear(x, w));
    const expected = noGrad(() => linear(as64(x), as64(w)));
    expectClose(updated.data, expected.data, 1e-5, 1e-5);
  });

  it('linear gradients match float64', () => {
    const x = random32([6, 13], 12);
    const w = random32([11, 13], 13);
    const b = random32([11], 14);
    const probe = random32([6, 11], 15);
    const grads = (inputs: Tensor[]) => {
      const leaves = inputs.map((input) => input.detach().clone().requiresGrad_());
      enableGrad(() => linear(leaves[0]!, leaves[1]!, leaves[2]!).mul(probe.to(leaves[0]!.dtype)).sum().backward());
      return leaves.map((leaf) => leaf.grad!);
    };
    const actual = grads([x, w, b]);
    const expected = grads([as64(x), as64(w), as64(b)]);
    actual.forEach((grad, index) => {
      expect(grad.dtype).toBe('float32');
      expectClose(grad.data, expected[index]!.data, 1e-5, 1e-5);
    });
  });

  it('batched, broadcast and vector matmul with gradients match float64', () => {
    const cases: [number[], number[]][] = [[[2, 1, 5, 7], [3, 7, 4]], [[5, 7], [7]], [[7], [7, 3]], [[4, 6, 9], [9, 2]], [[3], [3]]];
    for (const [shapeA, shapeB] of cases) {
      const a = random32(shapeA, shapeA.length);
      const b = random32(shapeB, shapeB.length + 7);
      const run = (left: Tensor, right: Tensor) => {
        const l = left.detach().clone().requiresGrad_();
        const r = right.detach().clone().requiresGrad_();
        const out = enableGrad(() => matmul(l, r));
        enableGrad(() => out.mul(out).sum().backward());
        return [out, l.grad!, r.grad!];
      };
      const actual = run(a, b);
      const expected = run(as64(a), as64(b));
      expect(actual[0]!.shape).toEqual(expected[0]!.shape);
      actual.forEach((value, index) => expectClose(value.data, expected[index]!.data, 1e-4, 1e-5));
    }
  });

  it('conv2d forward and gradients match float64', () => {
    const x = random32([2, 3, 9, 8], 20);
    const w = random32([4, 3, 3, 3], 21);
    const b = random32([4], 22);
    const run = (inputs: Tensor[]) => {
      const leaves = inputs.map((input) => input.detach().clone().requiresGrad_());
      const out = enableGrad(() => conv2d(leaves[0]!, leaves[1]!, leaves[2]!, { stride: [2, 1], padding: 1 }));
      enableGrad(() => out.mul(out).sum().backward());
      return [out, ...leaves.map((leaf) => leaf.grad!)];
    };
    const actual = run([x, w, b]);
    const expected = run([as64(x), as64(w), as64(b)]);
    actual.forEach((value, index) => expectClose(value.data, expected[index]!.data, 1e-4, 1e-5));
  });
});

describe.skipIf(!available)('fused attention', () => {
  const [batch, heads, queries, keys, dim] = [2, 3, 37, 41, 10];
  const q = random32([batch, heads, queries, dim], 30);
  const k = random32([batch, heads, keys, dim], 31);
  const v = random32([batch, heads, keys, 6], 32);

  /** The composed path: differentiable attention (gradients enabled) is never fused. */
  function composed(query: Tensor, key: Tensor, value: Tensor, options: Parameters<typeof scaledDotProductAttention>[3]): Tensor {
    const leaf = query.detach().clone().requiresGrad_();
    return enableGrad(() => scaledDotProductAttention(leaf, key, value, options).output).detach();
  }

  it('equals the composed path for broadcast additive biases', () => {
    const minimum = finfoMin('float32');
    const padding = tensor(Array.from({ length: batch * keys }, (_, index) => (index % 7 === 3 ? minimum : 0)), { shape: [batch, 1, 1, keys] });
    const causal = tensor(Array.from({ length: queries * keys }, (_, index) => (index % keys > Math.floor(index / keys) + 4 ? minimum : 0)), { shape: [1, 1, queries, keys] });
    const full = random32([batch, heads, queries, keys], 33);
    for (const bias of [null, padding, causal, padding.add(causal), full]) {
      const options = { bias, scale: 0.3 };
      const fused = noGrad(() => scaledDotProductAttention(q, k, v, options));
      expect(fused.output.shape).toEqual([batch, heads, queries, 6]);
      expectClose(fused.output.data, composed(q, k, v, options).data, 1e-5, 1e-5);
      const weights = noGrad(() => softmax((bias ? q.matmul(k.transpose(-2, -1)).mul(0.3).add(bias) : q.matmul(k.transpose(-2, -1)).mul(0.3)), -1));
      expectClose(fused.weights.data, weights.data, 1e-6, 1e-5);
    }
  });

  it('three-dimensional inputs and fully masked rows', () => {
    const q3 = q.reshape(batch * heads, queries, dim);
    const k3 = k.reshape(batch * heads, keys, dim);
    const v3 = v.reshape(batch * heads, keys, 6);
    const bias = tensor(Array.from({ length: queries * keys }, (_, index) => (Math.floor(index / keys) === 5 ? -Infinity : 0)), { shape: [queries, keys] });
    const fused = noGrad(() => scaledDotProductAttention(q3, k3, v3, { bias })).output;
    const reference = composed(q3, k3, v3, { bias });
    for (let index = 0; index < fused.numel; index += 1) {
      const row = Math.floor(index / 6) % queries;
      if (row === 5) expect(Number.isNaN(fused.data[index]!) && Number.isNaN(reference.data[index]!)).toBe(true);
    }
    const valid = (t: Tensor) => Array.from(t.data).filter((_, index) => Math.floor(index / 6) % queries !== 5);
    expectClose(valid(fused), valid(reference), 1e-5, 1e-5);
  });

  it('enableGqa shares key/value heads across query groups', () => {
    const kv = random32([batch, 1, keys, dim], 34);
    const value = random32([batch, 1, keys, dim], 35);
    const query = random32([batch, 4, queries, dim], 36);
    const repeat = (x: Tensor) => x.expand([batch, 4, keys, dim]);
    const fused = noGrad(() => scaledDotProductAttention(query, kv, value, { enableGqa: true })).output;
    const explicit = noGrad(() => scaledDotProductAttention(query, repeat(kv), repeat(value))).output;
    expect(bitwiseEqual(fused.data, explicit.data)).toBe(true);
    expectClose(composed(query, kv, value, { enableGqa: true }).data, explicit.data, 1e-5, 1e-5);
  });

  it('does not depend on the thread count', () => {
    const big = random32([2, 4, 128, 32], 37);
    const one = withThreads(1, () => noGrad(() => scaledDotProductAttention(big, big, big).output));
    const four = withThreads(4, () => noGrad(() => scaledDotProductAttention(big, big, big).output));
    expect(bitwiseEqual(one.data, four.data)).toBe(true);
  });
});

describe.skipIf(!available)('bit-identical elementwise kernels', () => {
  const x = tensor(Array.from({ length: 9000 }, (_, index) => {
    const t = Math.sin(index * 12.9898) * 43758.5453;
    const u = t - Math.floor(t);
    return index % 5 === 0 ? (u - 0.5) * 200 : (u - 0.5) * 12;
  }));

  it('activations equal their JavaScript formulas', () => {
    const ops: ((t: Tensor) => Tensor)[] = [(t) => gelu(t, 'tanh'), (t) => gelu(t), silu, sigmoid, tanh, exp];
    for (const op of ops) {
      const fast = noGrad(() => op(x));
      const slow = withBackend('js', () => noGrad(() => op(x)));
      expect(bitwiseEqual(fast.data, slow.data)).toBe(true);
    }
  });

  it('GELU gradients equal the JavaScript derivatives', () => {
    for (const approximate of ['none', 'tanh'] as const) {
      const run = () => {
        const leaf = x.detach().clone().requiresGrad_();
        enableGrad(() => gelu(leaf, approximate).sum().backward());
        return leaf.grad!;
      };
      expect(bitwiseEqual(run().data, withBackend('js', run).data)).toBe(true);
    }
  });

  it('large adjacent-axis permutes and float32 arithmetic equal the JavaScript kernels', () => {
    const big = random32([3, 40, 12, 800], 44);
    const other = random32([3, 40, 12, 800], 45);
    for (const order of [[0, 2, 1, 3], [1, 0, 2, 3], [0, 1, 3, 2], [0, 3, 2, 1]]) {
      const fast = noGrad(() => big.permute(order));
      const slow = withBackend('js', () => noGrad(() => big.permute(order)));
      expect(fast.shape).toEqual(slow.shape);
      expect(bitwiseEqual(fast.data, slow.data)).toBe(true);
    }
    const ops: ((a: Tensor, b: Tensor | number) => Tensor)[] = [(a, b) => a.add(b), (a, b) => a.sub(b), (a, b) => a.mul(b), (a, b) => a.div(b)];
    for (const op of ops) {
      for (const right of [other, 0.37, tensor([1.5])]) {
        const fast = noGrad(() => op(big, right));
        const slow = withBackend('js', () => noGrad(() => op(big, right)));
        expect(bitwiseEqual(fast.data, slow.data)).toBe(true);
      }
    }
    // Gradients flow through the kernel-backed permute.
    const leaf = random32([2, 700, 16, 48], 46).requiresGrad_();
    enableGrad(() => leaf.transpose(1, 2).mul(2).sum().backward());
    expect(Array.from(leaf.grad!.data.subarray(0, 5))).toEqual([2, 2, 2, 2, 2]);
  });

  it('layer normalization equals the JavaScript kernel', () => {
    const input = random32([300, 64], 40, 3);
    const weight = random32([64], 41);
    const bias = random32([64], 42);
    for (const [w, b] of [[weight, bias], [null, null], [weight, null]] as const) {
      const fast = noGrad(() => layerNorm(input, w, b, 1e-6));
      const slow = withBackend('js', () => noGrad(() => layerNorm(input, w, b, 1e-6)));
      expect(bitwiseEqual(fast.data, slow.data)).toBe(true);
    }
  });

  it('softmax matches the JavaScript kernel and keeps PyTorch edge cases', () => {
    const input = random32([50, 33], 43, 4);
    const fast = noGrad(() => softmax(input, -1));
    const slow = withBackend('js', () => noGrad(() => softmax(input, -1)));
    expectClose(fast.data, slow.data, 1e-7, 1e-6);
    const edge = tensor([[-Infinity, -Infinity, -Infinity], [0, Number.NaN, 1], [-Infinity, 0, -Infinity]]);
    const out = noGrad(() => softmax(edge, -1));
    expect(Array.from(out.data).slice(0, 6).every(Number.isNaN)).toBe(true);
    expect(Array.from(out.data).slice(6)).toEqual([0, 1, 0]);
  });
});

describe('AdamW fused update', () => {
  it('is bit-identical to the chained in-place update', () => {
    const initial = random32([257], 50);
    const grads = [random32([257], 51), random32([257], 52), random32([257], 53)];
    const param = new Parameter(initial.clone());
    const optimizer = new AdamW([param], { lr: 1e-2, weightDecay: 0.1 });
    const reference = initial.clone();
    const m = reference.mul(0);
    const v = reference.mul(0);
    grads.forEach((grad, index) => {
      param.grad = grad;
      optimizer.step();
      const step = index + 1;
      noGrad(() => {
        reference.mul_(1 - 1e-2 * 0.1);
        m.mul_(0.9).add_(grad, 1 - 0.9);
        v.mul_(0.999).addcmul_(grad, grad, 1 - 0.999);
        const denominator = v.sqrt().div(Math.sqrt(1 - 0.999 ** step)).add(1e-8);
        reference.addcdiv_(m, denominator, -(1e-2 / (1 - 0.9 ** step)));
      });
      expect(bitwiseEqual(param.data, reference.data)).toBe(true);
    });
    const state = optimizer.stateDict().state['0']!;
    expect(bitwiseEqual(state.exp_avg!.data, m.data)).toBe(true);
    expect(bitwiseEqual(state.exp_avg_sq!.data, v.data)).toBe(true);
  });
});

describe('thread controls', () => {
  it('validates and reports the thread count', () => {
    expect(() => setNumThreads(0)).toThrow(RangeError);
    expect(() => setNumThreads(1.5)).toThrow(RangeError);
    if (!available) return;
    withThreads(3, () => expect(getNumThreads()).toBe(3));
    expect(getNumThreads()).toBeGreaterThanOrEqual(1);
  });
});

describe('PyTorch parity (scripts/fixtures/backend_fixtures.py)', () => {
  const fixture = fixtureJson('backend.json');

  function checkGradients(record: { inputs: TensorJson[]; out: TensorJson; probe: TensorJson; grads: TensorJson[] }, fn: (...inputs: Tensor[]) => Tensor): void {
    const leaves = record.inputs.map((input) => fromJson(input).requiresGrad_());
    const out = enableGrad(() => fn(...leaves));
    expect(out.shape).toEqual(record.out.shape);
    expectClose(out.data, record.out.data, 1e-5, 1e-5);
    enableGrad(() => out.mul(fromJson(record.probe)).sum().backward());
    leaves.forEach((leaf, index) => expectClose(leaf.grad!.data, record.grads[index]!.data, 1e-5, 1e-5));
  }

  it('float32 linear, broadcast matmul and conv2d with gradients', () => {
    const cases = fixture.float32;
    checkGradients(cases.linear, (x, w, b) => linear(x, w, b));
    checkGradients(cases.matmul, (a, b) => matmul(a, b));
    checkGradients(cases.conv2d, (x, w, b) => conv2d(x, w, b, { stride: [2, 1], padding: 1 }));
  });

  it('float32 fused attention equals F.scaled_dot_product_attention (masks, enable_gqa)', () => {
    const record = fixture.float32.attention;
    const [q, k, v] = [fromJson(record.q), fromJson(record.k), fromJson(record.v)];
    const gqa = noGrad(() => scaledDotProductAttention(q, k, v, { bias: fromJson(record.padding), enableGqa: true })).output;
    expectClose(gqa.data, record.gqa.data, 1e-5, 1e-5);
    const repeat = (x: Tensor) => x.unsqueeze(2).expand([2, 2, 2, x.shape[2]!, x.shape[3]!]).reshape(2, 4, x.shape[2]!, x.shape[3]!);
    const biased = noGrad(() => scaledDotProductAttention(q, repeat(k), repeat(v), { bias: fromJson(record.full_bias), scale: 0.3 })).output;
    expectClose(biased.data, record.biased.data, 1e-5, 1e-5);
    // The differentiable (composed) path agrees too.
    const composed = enableGrad(() => scaledDotProductAttention(q.clone().requiresGrad_(), k, v, { bias: fromJson(record.padding), enableGqa: true })).output;
    expectClose(composed.data, record.gqa.data, 1e-5, 1e-5);
  });

  for (const dtype of ['bfloat16', 'float16'] as const) {
    it(`${dtype} results are float32 computations rounded to ${dtype}, like PyTorch`, () => {
      const record = fixture[dtype];
      const input = (name: string) => fromJson(record.inputs[name], dtype as DType);
      const outputs: Record<string, Tensor> = noGrad(() => ({
        linear: linear(input('x'), input('w'), input('b')),
        matmul: matmul(input('a'), input('m')),
        conv2d: conv2d(input('image'), input('kernel'), null, { padding: 1 }),
        softmax: softmax(input('scores'), -1),
        log_softmax: logSoftmax(input('scores'), -1),
        layer_norm: layerNorm(input('x'), input('norm_weight'), input('norm_bias'), 1e-5),
        attention: scaledDotProductAttention(input('q'), input('k'), input('v'), { scale: 0.25 }).output,
      }));
      const ulp = dtype === 'bfloat16' ? 2 ** -8 : 2 ** -11;
      // PyTorch's half log_softmax rounds intermediates, and its float16 CPU
      // convolution accumulates in half precision on CPUs with native fp16
      // arithmetic: those agree to a few units of the inputs' precision.
      const loose = new Set(dtype === 'float16' ? ['log_softmax', 'conv2d'] : ['log_softmax']);
      for (const [name, value] of Object.entries(outputs)) {
        const expected = record.outputs[name] as TensorJson;
        expect(value.dtype, name).toBe(dtype);
        expect(value.shape, name).toEqual(expected.shape);
        let exact = 0;
        value.data.forEach((actual, index) => {
          expect(roundToDType(dtype, actual), name).toBe(actual);
          const target = expected.data[index]!;
          if (actual === target) exact += 1;
          else if (loose.has(name)) expect(Math.abs(actual - target), `${name}[${index}]`).toBeLessThanOrEqual(8 * ulp * Math.max(Math.abs(target), 1));
          else expect(Math.abs(actual - target), `${name}[${index}]`).toBeLessThanOrEqual(2 * ulp * Math.max(Math.abs(target), 1e-2));
        });
        if (!loose.has(name)) expect(exact / value.numel, name).toBeGreaterThan(0.9);
      }
    });
  }
});
