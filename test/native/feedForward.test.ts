/**
 * ``feedForward`` (fused ``fc2(act(fc1(x)))``) and ``selfAttention`` (fused
 * projections, heads and attention) are bit-identical to the separate calls,
 * and fall back to them whenever gradients or dropout are needed.
 */
import { describe, expect, it } from 'vitest';
import { activationModule } from '../../src/_internal/native/activations.js';
import { attention, causalBias, feedForward, keyPaddingBias, mergeHeads, selfAttention, splitHeads } from '../../src/_internal/native/modules.js';
import { Linear, manualSeed, noGrad, randn, tensor, type Tensor } from '../../src/nn/index.js';

function bits(values: ArrayLike<number>): Uint32Array {
  return new Uint32Array(Float32Array.from(values).buffer);
}

describe('feedForward', () => {
  for (const act of ['gelu', 'gelu_pytorch_tanh', 'gelu_new', 'silu', 'relu']) {
    for (const [rows, width, hidden, bias] of [[3, 5, 12, true], [5, 7, 10, true], [700, 48, 192, false], [2048, 96, 384, true]] as const) {
      it(`${act} over [${rows}, ${width}] -> ${hidden} (bias ${bias}) equals the unfused layers`, () => {
        manualSeed(rows + width);
        const fc1 = new Linear(width, hidden, { bias });
        const fc2 = new Linear(hidden, width, { bias });
        const activation = activationModule(act);
        const x = randn([1, rows, width]);
        const expected = noGrad(() => fc2.forward(activation.forward(fc1.forward(x))));
        const fused = noGrad(() => feedForward(fc1, activation, fc2, x));
        expect(fused.shape).toEqual(expected.shape);
        expect(fused.dtype).toBe('float32');
        expect(bits(fused.data)).toEqual(bits(expected.data));
      });
    }
  }

  it('keeps the autograd graph when gradients are needed', () => {
    manualSeed(1);
    const fc1 = new Linear(8, 16);
    const fc2 = new Linear(16, 8);
    const x = randn([4, 8]);
    const output = feedForward(fc1, activationModule('gelu'), fc2, x);
    output.sum().backward();
    expect(fc1.weight.grad).not.toBeNull();
    expect(fc2.weight.grad).not.toBeNull();
  });
});

describe('selfAttention', () => {
  function layers(width: number, inner: number, bias: boolean) {
    return {
      query: new Linear(width, inner, { bias }), key: new Linear(width, inner, { bias }),
      value: new Linear(width, inner, { bias }), output: new Linear(inner, width),
    };
  }

  function unfused(p: ReturnType<typeof layers>, x: Tensor, heads: number, scale: number, bias: Tensor | null): Tensor {
    const q = splitHeads(p.query.forward(x), heads);
    const k = splitHeads(p.key.forward(x), heads);
    const v = splitHeads(p.value.forward(x), heads);
    return p.output.forward(mergeHeads(attention(q, k, v, { scale, bias })));
  }

  const masks: Record<string, (batch: number, length: number) => Tensor | null> = {
    none: () => null,
    padding: (batch, length) => keyPaddingBias(tensor(Array.from({ length: batch * length }, (_, i) => Number(i % length < length - 1 - (i % 3))), { shape: [batch, length], dtype: 'int64' })),
    causal: (_, length) => causalBias(length, length),
  };
  for (const [batch, length, width, heads, inner] of [[1, 5, 8, 2, 8], [3, 33, 48, 4, 64], [2, 300, 96, 12, 96], [2, 7, 12, 2, 12]] as const) {
    for (const mask of Object.keys(masks)) {
      it(`[${batch}, ${length}, ${width}] with ${heads} heads of ${inner / heads} (${mask}) equals the unfused layers`, () => {
        manualSeed(batch * 100 + length);
        const p = layers(width, inner, length % 2 === 1);
        const x = randn([batch, length, width]);
        const bias = masks[mask]!(batch, length);
        const scale = (inner / heads) ** -0.5;
        const expected = noGrad(() => unfused(p, x, heads, scale, bias));
        const fused = noGrad(() => selfAttention(p, x, heads, { scale, bias }));
        expect(fused.shape).toEqual(expected.shape);
        expect(bits(fused.data)).toEqual(bits(expected.data));
      });
    }
  }

  it('keeps the autograd graph when gradients are needed, and applies dropout in training', () => {
    manualSeed(3);
    const p = layers(8, 8, true);
    const x = randn([2, 4, 8]);
    selfAttention(p, x, 2, { scale: 0.5 }).sum().backward();
    expect(p.query.weight.grad).not.toBeNull();
    manualSeed(4);
    const dropped = noGrad(() => selfAttention(p, x, 2, { scale: 0.5, dropout: 0.5, training: true }));
    manualSeed(4);
    const reference = noGrad(() => {
      const q = splitHeads(p.query.forward(x), 2);
      const k = splitHeads(p.key.forward(x), 2);
      const v = splitHeads(p.value.forward(x), 2);
      return p.output.forward(mergeHeads(attention(q, k, v, { scale: 0.5, dropout: 0.5, training: true })));
    });
    expect(bits(dropped.data)).toEqual(bits(reference.data));
  });
});
