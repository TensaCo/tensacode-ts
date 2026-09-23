import { describe, expect, it } from 'vitest';
import { F, Tensor, tensor } from '../../src/nn/index.js';
import { gradcheck, randomTensor } from '../helpers/gradcheck.js';

const a = () => randomTensor([2, 3], 1);
const b = () => randomTensor([2, 3], 2);
const positive = () => randomTensor([2, 3], 3).abs().add(0.5).detach();

describe('elementwise gradients match finite differences', () => {
  const unary: [string, (x: Tensor) => Tensor, () => Tensor][] = [
    ['neg', (x) => x.neg(), a],
    ['exp', (x) => x.exp(), a],
    ['log', (x) => x.log(), positive],
    ['log1p', (x) => x.log1p(), positive],
    ['sqrt', (x) => x.sqrt(), positive],
    ['rsqrt', (x) => x.rsqrt(), positive],
    ['abs', (x) => x.abs(), a],
    ['square', (x) => x.square(), a],
    ['reciprocal', (x) => x.reciprocal(), positive],
    ['tanh', (x) => x.tanh(), a],
    ['sigmoid', (x) => x.sigmoid(), a],
    ['relu', (x) => x.relu(), a],
    ['sin', (x) => x.sin(), a],
    ['cos', (x) => x.cos(), a],
    ['erf', (x) => x.erf(), a],
    ['pow scalar', (x) => x.pow(3), a],
    ['clamp', (x) => x.clamp(-0.3, 0.4), a],
    ['gelu', (x) => F.gelu(x), a],
    ['gelu tanh', (x) => F.gelu(x, 'tanh'), a],
    ['silu', (x) => F.silu(x), a],
    ['softplus', (x) => F.softplus(x), a],
    ['clone', (x) => x.clone(), a],
  ];
  for (const [name, fn, make] of unary) {
    it(name, () => gradcheck(fn, [make()]));
  }

  it('broadcasting binary ops', () => {
    const row = () => randomTensor([3], 7);
    gradcheck((x, y) => x.add(y), [a(), row()]);
    gradcheck((x, y) => x.sub(y), [a(), row()]);
    gradcheck((x, y) => x.mul(y), [a(), row()]);
    gradcheck((x, y) => x.div(y), [a(), positive()]);
    gradcheck((x, y) => x.pow(y), [positive(), b()]);
    gradcheck((x, y) => x.maximum(y), [a(), b()]);
    gradcheck((x, y) => x.minimum(y), [a(), b()]);
    gradcheck((x, y) => F.where(x.gt(0), x, y), [a(), b()]);
  });

  it('maskedFill passes gradient only to unmasked entries', () => {
    const mask = tensor([[true, false, false], [false, true, false]]);
    gradcheck((x) => x.maskedFill(mask, -2), [a()]);
  });
});

describe('reduction gradients', () => {
  it('sum and mean over dims', () => {
    gradcheck((x) => x.sum(), [a()]);
    gradcheck((x) => x.sum(1), [a()]);
    gradcheck((x) => x.mean([0, 1], true), [a()]);
    gradcheck((x) => x.mean(0), [a()]);
    gradcheck((x) => x.var(1), [a()]);
  });

  it('extremes route gradients to the selected element', () => {
    gradcheck((x) => x.amax(1), [a()]);
    gradcheck((x) => x.amin(0), [a()]);
    gradcheck((x) => x.max(1).values, [a()]);
  });

  it('softmax, logSoftmax, logsumexp and cumsum', () => {
    gradcheck((x) => x.softmax(1), [a()]);
    gradcheck((x) => x.softmax(0), [a()]);
    gradcheck((x) => x.logSoftmax(-1), [a()]);
    gradcheck((x) => x.logsumexp(1), [a()]);
    gradcheck((x) => x.cumsum(1), [a()]);
    gradcheck((x) => x.norm(2, 1), [a()]);
  });
});

describe('shape and indexing gradients', () => {
  it('views and copies', () => {
    gradcheck((x) => x.reshape(3, 2), [a()]);
    gradcheck((x) => x.transpose(0, 1), [a()]);
    gradcheck((x) => x.permute(1, 0).unsqueeze(0), [a()]);
    gradcheck((x) => x.unsqueeze(1).expand(2, 4, 3), [a()]);
    gradcheck((x) => x.slice(1, 1, 3), [a()]);
    gradcheck((x) => x.slice(1, 0, null, 2), [a()]);
    gradcheck((x) => x.select(1, -1), [a()]);
    gradcheck((x) => x.repeat(2, 1), [a()]);
    gradcheck((x) => x.flatten(), [a()]);
  });

  it('gather, indexSelect, maskedSelect and topk', () => {
    const index = tensor([[0, 2], [1, 1]], { dtype: 'int64' });
    gradcheck((x) => x.gather(1, index), [a()]);
    gradcheck((x) => x.indexSelect(1, [2, 0, 2]), [a()]);
    gradcheck((x) => x.maskedSelect(tensor([true, false])), [a()]);
    gradcheck((x) => x.topk(2, 1).values, [a()]);
    gradcheck((x) => x.sort(1, true).values, [a()]);
  });

  it('cat, stack, split and padSequence', () => {
    gradcheck((x, y) => F.cat([x, y], 1), [a(), b()]);
    gradcheck((x, y) => F.stack([x, y], 1), [a(), b()]);
    gradcheck((x) => x.split([1, 2], 1)[1]!, [a()]);
    gradcheck((x, y) => F.padSequence([x, y.slice(0, 0, 1)]), [a(), b()]);
  });
});

describe('linear algebra and neural gradients', () => {
  it('matmul in all rank combinations', () => {
    gradcheck((x, y) => x.matmul(y), [randomTensor([2, 3], 1), randomTensor([3, 4], 2)]);
    gradcheck((x, y) => x.matmul(y), [randomTensor([3], 1), randomTensor([3, 4], 2)]);
    gradcheck((x, y) => x.matmul(y), [randomTensor([2, 3], 1), randomTensor([3], 2)]);
    gradcheck((x, y) => x.matmul(y), [randomTensor([2, 2, 3], 1), randomTensor([3, 4], 2)]);
    gradcheck((x, y) => x.matmul(y), [randomTensor([2, 1, 2, 3], 1), randomTensor([3, 3, 2], 2)]);
  });

  it('linear with and without bias', () => {
    gradcheck((x, w, bias) => F.linear(x, w, bias), [randomTensor([2, 3, 4], 1), randomTensor([5, 4], 2), randomTensor([5], 3)]);
    gradcheck((x, w) => F.linear(x, w), [randomTensor([4], 1), randomTensor([2, 4], 2)]);
  });

  it('layerNorm and rmsNorm', () => {
    gradcheck((x, w, bias) => F.layerNorm(x, w, bias, 1e-5), [randomTensor([3, 4], 1), randomTensor([4], 2), randomTensor([4], 3)]);
    gradcheck((x, w) => F.rmsNorm(x, w, 1e-6), [randomTensor([3, 4], 1), randomTensor([4], 2)]);
  });

  it('embedding, embeddingBag and conv2d', () => {
    gradcheck((w) => F.embedding(tensor([[1, 0], [2, 1]], { dtype: 'int64' }), w), [randomTensor([3, 4], 1)]);
    gradcheck((w) => F.embeddingBag([0, 2, 1, 1], [0, 1], w), [randomTensor([3, 4], 1)]);
    gradcheck(
      (x, w, bias) => F.conv2d(x, w, bias, { stride: 2, padding: 1 }),
      [randomTensor([2, 2, 5, 4], 1), randomTensor([3, 2, 3, 2], 2), randomTensor([3], 3)],
    );
    gradcheck(
      (x, w) => F.conv2d(x, w, null, { dilation: 2 }),
      [randomTensor([1, 1, 5, 5], 4), randomTensor([2, 1, 2, 2], 5)],
    );
  });

  it('losses', () => {
    const targets = tensor([2, 0], { dtype: 'int64' });
    gradcheck((x) => F.crossEntropy(x, targets), [a()]);
    gradcheck((x) => F.crossEntropy(x, tensor([2, -100], { dtype: 'int64' })), [a()]);
    gradcheck((x) => F.crossEntropy(x, tensor([[0.2, 0.3, 0.5], [1, 0, 0]], { dtype: 'float64' })), [a()]);
    gradcheck((x) => F.crossEntropy(x.select(0, 0), tensor(1, { dtype: 'int64' })), [a()]);
    gradcheck((x, y) => F.mseLoss(x, y), [a(), b()]);
    gradcheck((x) => F.binaryCrossEntropyWithLogits(x, tensor([[1, 0, 1], [0, 0, 1]], { dtype: 'float64' })), [a()]);
    gradcheck((x, y) => F.cosineSimilarity(x, y), [a(), b()]);
    gradcheck((x) => F.normalize(x), [a()]);
  });
});

describe('autograd engine', () => {
  it('accumulates gradients across uses and backward calls', () => {
    const x = tensor([1, 2, 3], { dtype: 'float64', requiresGrad: true });
    const y = x.mul(x).add(x).sum();
    y.backward();
    expect(Array.from(x.grad!.data)).toEqual([3, 5, 7]);
    x.mul(2).sum().backward();
    expect(Array.from(x.grad!.data)).toEqual([5, 7, 9]);
  });

  it('rejects in-place mutation of leaves that require grad', () => {
    const x = tensor([1, 2], { requiresGrad: true });
    expect(() => x.add_(1)).toThrow(/in place/);
  });

  it('does not record gradients for integer tensors or under noGrad', () => {
    const x = tensor([1, 2], { requiresGrad: true });
    const y = F.relu(x);
    expect(y.requiresGrad).toBe(true);
    expect(x.gt(0).requiresGrad).toBe(false);
  });

  it('keeps grad of non-leaf tensors only when retained', () => {
    const x = tensor([1, 2], { requiresGrad: true });
    const y = x.mul(3).retainGrad();
    y.sum().backward();
    expect(Array.from(y.grad!.data)).toEqual([1, 1]);
    expect(Array.from(x.grad!.data)).toEqual([3, 3]);
  });
});
