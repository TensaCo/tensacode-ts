/** Ports of ``tests/vec/test_latent.py`` and ``tests/vec/test_latent_operation.py``. */
import { describe, expect, it } from 'vitest';
import { Identity, Linear, Parameter, noGrad, ones, scalar, tensor, zeros, type Tensor } from '../../src/nn/index.js';
import { ValueError } from '../../src/errors.js';
import { Latent, Space, Transform, VocabularyEncoder } from '../../src/ops/vec/index.js';
import { LatentOperation, asSequence } from '../../src/_internal/latentOps.js';
import { trace } from '../../src/_internal/tracing.js';
import type { Context } from '../../src/ops/base.js';

describe('latent values (tests/vec/test_latent.py)', () => {
  it('latent retains tensor gradient and metadata', () => {
    const space = new Space('example/features', 3, { version: '2' });
    const source = tensor([1, 2, 3], { requiresGrad: true });
    const latent = new Latent(source, space, { sources: ['sample:7'], metadata: { split: 'train' } });
    latent.tensor.square().sum().backward();
    expect(source.grad!.toArray()).toEqual([2, 4, 6]);
    expect(latent.sources).toEqual(['sample:7']);
    expect(latent.metadata).toEqual({ split: 'train' });
  });

  it('transform rejects equal shape from an incompatible space', () => {
    const expected = new Space('model-a/text', 2);
    const transform = Transform.fromModule(new Identity(), { inputSpace: expected, outputSpace: expected });
    expect(() => transform.call(new Latent(ones([2]), new Space('model-b/text', 2)))).toThrow(/incompatible.*space/);
  });

  it('space-aware transform preserves provenance and gradients', () => {
    const inputSpace = new Space('encoder/text', 2);
    const outputSpace = new Space('projector/shared', 3);
    const source = tensor([1, -1], { requiresGrad: true });
    const transform = Transform.fromModule(new Linear(2, 3, { bias: false }), { inputSpace, outputSpace });
    const result = transform.call(new Latent(source, inputSpace, { sources: ['ticket:3'], metadata: { lang: 'en' } })) as Latent;
    expect(result).toBeInstanceOf(Latent);
    expect(result.space.equals(outputSpace)).toBe(true);
    expect(result.sources).toEqual(['ticket:3']);
    expect(result.metadata).toEqual({ lang: 'en' });
    result.tensor.sum().backward();
    expect(source.grad).not.toBeNull();
    expect((transform.module as Linear).weight.grad).not.toBeNull();
  });

  it('text encoder space is opt-in and the tensor API remains', () => {
    const ordinary = new VocabularyEncoder({ vocabulary: ['hello'], dimensions: 4 });
    const configured = new VocabularyEncoder({ vocabulary: ['hello'], dimensions: 4, output_space: new Space('local/text', 4).configuration() as any });
    expect(ordinary.call('hello')).not.toBeInstanceOf(Latent);
    const encoded = configured.call(['hello', 'unknown']) as Latent;
    expect(encoded).toBeInstanceOf(Latent);
    expect(encoded.tensor.shape).toEqual([2, 4]);
    expect(encoded.space.equals(new Space('local/text', 4))).toBe(true);
  });

  it('latent validates feature, mask and coordinate shapes', () => {
    const space = new Space('image/patches', 4, { organization: 'spatial' });
    expect(() => new Latent(ones([2, 3]), space)).toThrow(/feature dimension/);
    expect(() => new Latent(ones([2, 2, 4]), space, { mask: ones([2]) })).toThrow(/mask/);
    expect(() => new Latent(ones([2, 2, 4]), space, { coordinates: ones([2, 2]) })).toThrow(/coordinates/);
  });

  it('space can declare dtype and device expectations', () => {
    const space = new Space('typed/features', 2, { dtype: 'torch.float32', device: 'cpu' });
    expect(new Latent(ones([2]), space).space).toBe(space);
    expect(() => new Latent(ones([2], { dtype: 'float64' }), space)).toThrow(/dtype/);
  });
});

describe('latent operations (tests/vec/test_latent_operation.py)', () => {
  it('sequence contract masks and space', () => {
    const space = new Space('tokens', 3, { organization: 'sequence' });
    const x = ones([2, 3]);
    x.requiresGrad = true;
    const value = new Latent(x, space, { mask: tensor([true, false]) });
    const [sequence, mask] = asSequence(value, space);
    expect(sequence.shape).toEqual([1, 2, 3]);
    expect(mask.tolist()).toEqual([[true, false]]);
    expect(sequence.select(0, 0).select(0, 1).toArray()).toEqual([0, 0, 0]);
    sequence.sum().backward();
    expect(x.grad!.select(0, 1).toArray()).toEqual([0, 0, 0]);
    expect(() => asSequence(value, new Space('other', 3, { organization: 'sequence' }))).toThrow(/incompatible/);
    expect(() => asSequence(new Latent(x, space, { mask: zeros([2], { dtype: 'bool' }) }), space)).toThrow(/valid/);
  });

  it('owned operation calls are traced (Python forward hooks have no TypeScript counterpart)', () => {
    class ScaleOperation extends LatentOperation<Tensor, Tensor> {
      readonly weight: Parameter;
      constructor() {
        super({});
        this.weight = this.registerParameter('weight', new Parameter(scalar(2)));
      }
      forward(value: Tensor, context: Context | null): Tensor {
        void context;
        return this.weight.mul(value);
      }
    }
    const op = new ScaleOperation();
    const session = trace();
    const output = session.run(() => op.call(scalar(3)));
    expect(session.calls.length).toBe(1);
    output.backward();
    expect(op.weight.grad!.item()).toBe(3);
    expect(op.operationBindings().operation).toBe(op);
    expect(noGrad(() => op.call(scalar(1))).item()).toBe(2);
  });

  it('owned operations reject non-JSON and legacy module construction', () => {
    expect(() => new Transform(new Linear(3, 2) as any)).toThrow(ValueError);
  });
});
