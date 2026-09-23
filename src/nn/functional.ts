/** Stateless tensor functions (``torch.nn.functional`` equivalents). */
export { linear, matmul } from './ops/linalg.js';
export {
  activation, binaryCrossEntropyWithLogits, conv2d, cosineSimilarity, crossEntropy, dropout, embedding,
  embeddingBag, gelu, layerNorm, leakyRelu, mseLoss, nllLoss, normalize, oneHot, quickGelu, rmsNorm,
  scaledDotProductAttention, silu, softplus,
  type ActivationName, type Conv2dOptions, type CrossEntropyOptions, type Reduction,
} from './ops/nn.js';
export {
  abs, add, allclose, allFinite, clamp, clone, cast, compare, cos, div, equal, erf, exp, isFinite, isNan, log,
  log1p, logical, logicalNot, maskedFill, maximum, minimum, mul, neg, pow, reciprocal, relu, round, rsqrt,
  sigmoid, sign, sin, sqrt, square, sub, tanh, where,
} from './ops/elementwise.js';
export {
  amax, amin, anyAll, cumsum, logSoftmax, logsumexp, mean, norm, softmax, sum, sumToShape, variance,
} from './ops/reduce.js';
export {
  cat, chunk, expand, flatten, gather, indexSelect, maskedSelect, meshgrid, padSequence, permute, repeat,
  reshape, select, slice, sort, split, squeeze, stack, topk, transpose, unbind, unsqueeze,
} from './ops/shape.js';
