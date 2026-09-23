/**
 * TensorCode's numerical core: dense CPU tensors, reverse-mode autograd,
 * PyTorch-compatible modules and optimizers, and safetensors I/O. It has no
 * runtime dependencies and performs no I/O on import.
 */
export {
  Tensor, Parameter, tensor, scalar, zeros, ones, full, empty, zerosLike, onesLike, fullLike, arange, linspace,
  eye, randn, rand, randint, randnLike, isTensor,
  type NestedNumbers, type Operand, type TensorOptions, type RandomOptions,
} from './tensor.js';
export type { DType, Storage } from './dtype.js';
export { isFloatingDType, DTYPES, finfoMin } from './dtype.js';
export type { Shape } from './shape.js';
export { noGrad, enableGrad, isGradEnabled, setGradEnabled, type GradNode } from './autograd.js';
export {
  Generator, manualSeed, seed, initialSeed, getDefaultGenerator, withoutRandomInit, DEFAULT_RNG_SEED, RNG_STATE_SIZE, type Seed,
} from './random.js';
export {
  getRngState, setRngState, rngStateTensor, randperm, multinomial, bernoulli, bernoulli_, exponential_, normal,
} from './randomTensor.js';
export { PythonRandom, pythonRandom, type PythonRandomState, type PythonSeed } from './randomPython.js';
export { Module, parameter, type StateDict, type LoadStateDictResult } from './module.js';
export {
  Linear, Identity, Embedding, EmbeddingBag, LayerNorm, Dropout, GELU, ReLU, Tanh, Sigmoid, SiLU,
  Sequential, ModuleList, Conv2d, GRUCell, GRU, GroupNorm, MultiheadAttention, NonDynamicallyQuantizableLinear, isTensorModule, type TensorModule,
} from './layers.js';
export * as init from './init.js';
export * as F from './functional.js';
export {
  cat, stack, where, padSequence, meshgrid, topk, sort,
} from './functional.js';
export {
  Optimizer, SGD, Adam, AdamW,
  type SGDOptions, type AdamOptions, type OptimizerStateDict, type ParamGroup, type ParamGroupInput,
} from './optim.js';
export {
  backendInfo, setNumThreads, getNumThreads, setBackend, clearWeightCache, type BackendInfo,
} from './backend/index.js';
export {
  serializeSafetensors, deserializeSafetensors, readSafetensorsFile, writeSafetensorsFile, serializeModel,
  saveModel, loadModel, loadModelFromBytes, findSharedTensors, removeDuplicateNames, type SafetensorsContents,
} from './safetensors.js';
