/** Standard layers with PyTorch parameter names, shapes and default initialization. */
import { Module } from './module.js';
import { Parameter, Tensor, empty, zeros, ones, tensor as tensorOf } from './tensor.js';
import * as init from './init.js';
import { linear as linearOp } from './ops/linalg.js';
import {
  conv2d as conv2dOp, dropout as dropoutOp, embedding as embeddingOp, embeddingBag as embeddingBagOp,
  gelu as geluOp, layerNorm as layerNormOp, silu as siluOp, type Conv2dOptions,
} from './ops/nn.js';
import { add, mul, relu as reluOp, sigmoid as sigmoidOp, sub, tanh as tanhOp } from './ops/elementwise.js';
import { chunk, select, stack, unsqueeze } from './ops/shape.js';
import { groupNorm as groupNormOp, scaledDotProductAttention as scaledDotProductAttentionOp } from './ops/nn.js';

/** A module mapping one tensor to one tensor. */
export interface TensorModule extends Module {
  forward(input: Tensor): Tensor;
}

export function isTensorModule(value: unknown): value is TensorModule {
  return value instanceof Module && typeof (value as { forward?: unknown }).forward === 'function';
}

export class Linear extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.linear.Linear';

  override configurationAttributes(): Record<string, unknown> {
    return { in_features: this.inFeatures, out_features: this.outFeatures };
  }

  readonly inFeatures: number;
  readonly outFeatures: number;
  weight: Parameter;
  bias: Parameter | null;

  constructor(inFeatures: number, outFeatures: number, options: { bias?: boolean } = {}) {
    super();
    positiveInteger(inFeatures, 'inFeatures', true);
    positiveInteger(outFeatures, 'outFeatures', true);
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.weight = this.registerParameter('weight', new Parameter(empty([outFeatures, inFeatures])));
    this.bias = this.registerParameter('bias', options.bias === false ? null : new Parameter(empty([outFeatures])));
    this.resetParameters();
  }

  resetParameters(): void {
    init.kaimingUniformDefault_(this.weight);
    if (this.bias) {
      const bound = this.inFeatures > 0 ? 1 / Math.sqrt(this.inFeatures) : 0;
      init.uniform_(this.bias, -bound, bound);
    }
  }

  protected override onRegistryChange(): void {
    this.weight = (this.getParameter('weight') ?? this.weight) as Parameter;
    this.bias = this.getParameter('bias');
  }

  forward(input: Tensor): Tensor {
    return linearOp(input, this.weight, this.bias);
  }
}

export class Identity extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.linear.Identity';

  forward(input: Tensor): Tensor {
    return input;
  }
}

export class Embedding extends Module {
  static override readonly qualifiedName: string = 'torch.nn.modules.sparse.Embedding';

  override configurationAttributes(): Record<string, unknown> {
    return { embedding_dim: this.embeddingDim, max_norm: null, norm_type: 2, num_embeddings: this.numEmbeddings, padding_idx: this.paddingIdx, scale_grad_by_freq: false, sparse: false };
  }

  readonly numEmbeddings: number;
  readonly embeddingDim: number;
  readonly paddingIdx: number | null;
  weight: Parameter;

  constructor(numEmbeddings: number, embeddingDim: number, options: { paddingIdx?: number | null } = {}) {
    super();
    positiveInteger(numEmbeddings, 'numEmbeddings');
    positiveInteger(embeddingDim, 'embeddingDim');
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    let padding = options.paddingIdx ?? null;
    if (padding !== null && padding < 0) padding += numEmbeddings;
    if (padding !== null && (padding < 0 || padding >= numEmbeddings)) throw new RangeError('paddingIdx out of range');
    this.paddingIdx = padding;
    this.weight = this.registerParameter('weight', new Parameter(empty([numEmbeddings, embeddingDim])));
    this.resetParameters();
  }

  resetParameters(): void {
    init.normal_(this.weight);
    if (this.paddingIdx !== null) {
      const width = this.embeddingDim;
      this.weight.data.fill(0, this.paddingIdx * width, (this.paddingIdx + 1) * width);
    }
  }

  protected override onRegistryChange(): void {
    this.weight = (this.getParameter('weight') ?? this.weight) as Parameter;
  }

  forward(indices: Tensor | readonly number[]): Tensor {
    return embeddingOp(indices, this.weight, this.paddingIdx);
  }
}

/** ``nn.EmbeddingBag(mode='mean')``. */
export class EmbeddingBag extends Module {
  static override readonly qualifiedName: string = 'torch.nn.modules.sparse.EmbeddingBag';

  override configurationAttributes(): Record<string, unknown> {
    return { embedding_dim: this.embeddingDim, include_last_offset: false, max_norm: null, mode: this.mode, norm_type: 2, num_embeddings: this.numEmbeddings, padding_idx: null, scale_grad_by_freq: false, sparse: false };
  }

  readonly numEmbeddings: number;
  readonly embeddingDim: number;
  readonly mode = 'mean' as const;
  weight: Parameter;

  constructor(numEmbeddings: number, embeddingDim: number, options: { mode?: 'mean' } = {}) {
    super();
    if (options.mode !== undefined && options.mode !== 'mean') throw new RangeError('EmbeddingBag supports mode="mean"');
    positiveInteger(numEmbeddings, 'numEmbeddings');
    positiveInteger(embeddingDim, 'embeddingDim');
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.weight = this.registerParameter('weight', new Parameter(empty([numEmbeddings, embeddingDim])));
    init.normal_(this.weight);
  }

  protected override onRegistryChange(): void {
    this.weight = (this.getParameter('weight') ?? this.weight) as Parameter;
  }

  forward(indices: readonly number[], offsets: readonly number[]): Tensor {
    return embeddingBagOp(indices, offsets, this.weight);
  }
}

export class LayerNorm extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.normalization.LayerNorm';

  override configurationAttributes(): Record<string, unknown> {
    return { elementwise_affine: this.weight !== null, eps: this.eps, normalized_shape: [this.normalizedShape] };
  }

  readonly normalizedShape: number;
  readonly eps: number;
  weight: Parameter | null;
  bias: Parameter | null;

  constructor(normalizedShape: number, options: { eps?: number; elementwiseAffine?: boolean; bias?: boolean } = {}) {
    super();
    positiveInteger(normalizedShape, 'normalizedShape');
    this.normalizedShape = normalizedShape;
    this.eps = options.eps ?? 1e-5;
    const affine = options.elementwiseAffine ?? true;
    this.weight = this.registerParameter('weight', affine ? new Parameter(ones([normalizedShape])) : null);
    this.bias = this.registerParameter('bias', affine && options.bias !== false ? new Parameter(zeros([normalizedShape])) : null);
  }

  protected override onRegistryChange(): void {
    this.weight = this.getParameter('weight');
    this.bias = this.getParameter('bias');
  }

  forward(input: Tensor): Tensor {
    return layerNormOp(input, this.weight, this.bias, this.eps);
  }
}

export class Dropout extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.dropout.Dropout';

  override configurationAttributes(): Record<string, unknown> {
    return { inplace: false, p: this.p };
  }

  readonly p: number;

  constructor(p = 0.5) {
    super();
    if (!(p >= 0 && p <= 1)) throw new RangeError('dropout probability must be in [0, 1]');
    this.p = p;
  }

  forward(input: Tensor): Tensor {
    return dropoutOp(input, this.p, this.training);
  }
}

export class GELU extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.GELU';

  override configurationAttributes(): Record<string, unknown> {
    return { approximate: this.approximate };
  }

  readonly approximate: 'none' | 'tanh';

  constructor(options: { approximate?: 'none' | 'tanh' } = {}) {
    super();
    this.approximate = options.approximate ?? 'none';
  }

  forward(input: Tensor): Tensor {
    return geluOp(input, this.approximate);
  }
}

export class ReLU extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.ReLU';

  override configurationAttributes(): Record<string, unknown> {
    return { inplace: false };
  }

  forward(input: Tensor): Tensor {
    return reluOp(input);
  }
}

export class Tanh extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.Tanh';

  forward(input: Tensor): Tensor {
    return tanhOp(input);
  }
}

export class Sigmoid extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.Sigmoid';

  forward(input: Tensor): Tensor {
    return sigmoidOp(input);
  }
}

export class SiLU extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.SiLU';

  override configurationAttributes(): Record<string, unknown> {
    return { inplace: false };
  }

  forward(input: Tensor): Tensor {
    return siluOp(input);
  }
}

/** Ordered container named ``0``, ``1``, ... whose forward chains children. */
export class Sequential extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.container.Sequential';

  constructor(...modules: TensorModule[]) {
    super();
    modules.forEach((module, index) => this.registerModule(String(index), module));
  }

  get length(): number {
    return this._modules.size;
  }

  at(index: number): TensorModule {
    const module = this._modules.get(String(index < 0 ? index + this.length : index));
    if (!module) throw new RangeError(`no module at index ${index}`);
    return module as TensorModule;
  }

  forward(input: Tensor): Tensor {
    let value = input;
    for (const module of this._modules.values()) if (module) value = (module as TensorModule).forward(value);
    return value;
  }
}

/** Indexed module list (``nn.ModuleList``). */
export class ModuleList<T extends Module = Module> extends Module implements Iterable<T> {
  static override readonly qualifiedName: string = 'torch.nn.modules.container.ModuleList';

  constructor(modules: Iterable<T> = []) {
    super();
    for (const module of modules) this.append(module);
  }

  get length(): number {
    return this._modules.size;
  }

  append(module: T): this {
    this.registerModule(String(this._modules.size), module);
    return this;
  }

  at(index: number): T {
    const module = this._modules.get(String(index < 0 ? index + this.length : index));
    if (!module) throw new RangeError(`no module at index ${index}`);
    return module as T;
  }

  *[Symbol.iterator](): Iterator<T> {
    for (const module of this._modules.values()) if (module) yield module as T;
  }
}

export class Conv2d extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.conv.Conv2d';

  override configurationAttributes(): Record<string, unknown> {
    return { _reversed_padding_repeated_twice: [this.padding[1], this.padding[1], this.padding[0], this.padding[0]], dilation: [...this.dilation], groups: 1, in_channels: this.inChannels, kernel_size: [...this.kernelSize], out_channels: this.outChannels, output_padding: [0, 0], padding: [...this.padding], padding_mode: 'zeros', stride: [...this.stride], transposed: false };
  }

  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: [number, number];
  readonly stride: [number, number];
  readonly padding: [number, number];
  readonly dilation: [number, number];
  weight: Parameter;
  bias: Parameter | null;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | readonly [number, number],
    options: { stride?: number | readonly [number, number]; padding?: number | readonly [number, number]; dilation?: number | readonly [number, number]; bias?: boolean } = {},
  ) {
    super();
    positiveInteger(inChannels, 'inChannels');
    positiveInteger(outChannels, 'outChannels');
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = pair(kernelSize, 'kernelSize');
    this.stride = pair(options.stride ?? 1, 'stride');
    this.padding = pair(options.padding ?? 0, 'padding', true);
    this.dilation = pair(options.dilation ?? 1, 'dilation');
    this.weight = this.registerParameter('weight', new Parameter(empty([outChannels, inChannels, ...this.kernelSize])));
    this.bias = this.registerParameter('bias', options.bias === false ? null : new Parameter(empty([outChannels])));
    init.kaimingUniformDefault_(this.weight);
    if (this.bias) {
      const fanIn = inChannels * this.kernelSize[0] * this.kernelSize[1];
      init.uniform_(this.bias, -1 / Math.sqrt(fanIn), 1 / Math.sqrt(fanIn));
    }
  }

  protected override onRegistryChange(): void {
    this.weight = (this.getParameter('weight') ?? this.weight) as Parameter;
    this.bias = this.getParameter('bias');
  }

  forward(input: Tensor): Tensor {
    const options: Conv2dOptions = { stride: this.stride, padding: this.padding, dilation: this.dilation };
    return conv2dOp(input, this.weight, this.bias, options);
  }
}

function gruGates(
  input: Tensor, hidden: Tensor,
  weightIh: Tensor, weightHh: Tensor, biasIh: Tensor | null, biasHh: Tensor | null,
): Tensor {
  const gi = chunk(linearOp(input, weightIh, biasIh), 3, -1);
  const gh = chunk(linearOp(hidden, weightHh, biasHh), 3, -1);
  const reset = sigmoidOp(add(gi[0]!, gh[0]!));
  const update = sigmoidOp(add(gi[1]!, gh[1]!));
  const candidate = tanhOp(add(gi[2]!, mul(reset, gh[2]!)));
  // h' = (1 - z) * n + z * h
  return add(candidate, mul(update, sub(hidden, candidate)));
}

/** ``nn.GRUCell`` with gate order (reset, update, new). */
export class GRUCell extends Module {
  static override readonly qualifiedName: string = 'torch.nn.modules.rnn.GRUCell';

  override configurationAttributes(): Record<string, unknown> {
    return { bias: this.bias_ih !== null, hidden_size: this.hiddenSize, input_size: this.inputSize };
  }

  readonly inputSize: number;
  readonly hiddenSize: number;
  weight_ih: Parameter;
  weight_hh: Parameter;
  bias_ih: Parameter | null;
  bias_hh: Parameter | null;

  constructor(inputSize: number, hiddenSize: number, options: { bias?: boolean } = {}) {
    super();
    positiveInteger(inputSize, 'inputSize');
    positiveInteger(hiddenSize, 'hiddenSize');
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    const bias = options.bias ?? true;
    this.weight_ih = this.registerParameter('weight_ih', new Parameter(empty([3 * hiddenSize, inputSize])));
    this.weight_hh = this.registerParameter('weight_hh', new Parameter(empty([3 * hiddenSize, hiddenSize])));
    this.bias_ih = this.registerParameter('bias_ih', bias ? new Parameter(empty([3 * hiddenSize])) : null);
    this.bias_hh = this.registerParameter('bias_hh', bias ? new Parameter(empty([3 * hiddenSize])) : null);
    const bound = 1 / Math.sqrt(hiddenSize);
    for (const parameter of this.parameters()) init.uniform_(parameter, -bound, bound);
  }

  protected override onRegistryChange(): void {
    this.weight_ih = (this.getParameter('weight_ih') ?? this.weight_ih) as Parameter;
    this.weight_hh = (this.getParameter('weight_hh') ?? this.weight_hh) as Parameter;
    this.bias_ih = this.getParameter('bias_ih');
    this.bias_hh = this.getParameter('bias_hh');
  }

  /** ``input`` is ``[batch, inputSize]`` (or unbatched); ``hidden`` defaults to zeros. */
  forward(input: Tensor, hidden?: Tensor | null): Tensor {
    const state = hidden ?? zeros(input.ndim === 1 ? [this.hiddenSize] : [input.shape[0]!, this.hiddenSize], { dtype: input.dtype });
    return gruGates(input, state, this.weight_ih, this.weight_hh, this.bias_ih, this.bias_hh);
  }
}

/** Single-layer unidirectional ``nn.GRU``. */
export class GRU extends Module {
  static override readonly qualifiedName: string = 'torch.nn.modules.rnn.GRU';

  readonly inputSize: number;
  readonly hiddenSize: number;
  readonly batchFirst: boolean;
  weight_ih_l0: Parameter;
  weight_hh_l0: Parameter;
  bias_ih_l0: Parameter | null;
  bias_hh_l0: Parameter | null;

  constructor(inputSize: number, hiddenSize: number, options: { batchFirst?: boolean; bias?: boolean; numLayers?: number } = {}) {
    super();
    if ((options.numLayers ?? 1) !== 1) throw new RangeError('GRU supports numLayers=1');
    positiveInteger(inputSize, 'inputSize');
    positiveInteger(hiddenSize, 'hiddenSize');
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.batchFirst = options.batchFirst ?? false;
    const bias = options.bias ?? true;
    this.weight_ih_l0 = this.registerParameter('weight_ih_l0', new Parameter(empty([3 * hiddenSize, inputSize])));
    this.weight_hh_l0 = this.registerParameter('weight_hh_l0', new Parameter(empty([3 * hiddenSize, hiddenSize])));
    this.bias_ih_l0 = this.registerParameter('bias_ih_l0', bias ? new Parameter(empty([3 * hiddenSize])) : null);
    this.bias_hh_l0 = this.registerParameter('bias_hh_l0', bias ? new Parameter(empty([3 * hiddenSize])) : null);
    const bound = 1 / Math.sqrt(hiddenSize);
    for (const parameter of this.parameters()) init.uniform_(parameter, -bound, bound);
  }

  protected override onRegistryChange(): void {
    this.weight_ih_l0 = (this.getParameter('weight_ih_l0') ?? this.weight_ih_l0) as Parameter;
    this.weight_hh_l0 = (this.getParameter('weight_hh_l0') ?? this.weight_hh_l0) as Parameter;
    this.bias_ih_l0 = this.getParameter('bias_ih_l0');
    this.bias_hh_l0 = this.getParameter('bias_hh_l0');
  }

  /**
   * ``input`` is ``[batch, time, features]`` when ``batchFirst`` (else
   * ``[time, batch, features]``). Returns all outputs and the final hidden
   * state ``[1, batch, hidden]``.
   */
  forward(input: Tensor, hidden?: Tensor | null): { output: Tensor; hidden: Tensor } {
    if (input.ndim !== 3) throw new RangeError('GRU expects a 3-dimensional batched input');
    const timeDim = this.batchFirst ? 1 : 0;
    const batchDim = this.batchFirst ? 0 : 1;
    const batch = input.shape[batchDim]!;
    let state = hidden ? select(hidden, 0, 0) : zeros([batch, this.hiddenSize], { dtype: input.dtype });
    const outputs: Tensor[] = [];
    for (let step = 0; step < input.shape[timeDim]!; step += 1) {
      state = gruGates(select(input, timeDim, step), state, this.weight_ih_l0, this.weight_hh_l0, this.bias_ih_l0, this.bias_hh_l0);
      outputs.push(state);
    }
    return { output: stack(outputs, timeDim), hidden: unsqueeze(state, 0) };
  }
}

function positiveInteger(value: number, name: string, allowZero = false): void {
  if (!Number.isInteger(value) || value < (allowZero ? 0 : 1)) throw new RangeError(`${name} must be a positive integer`);
}

function pair(value: number | readonly [number, number], name: string, allowZero = false): [number, number] {
  const result: [number, number] = typeof value === 'number' ? [value, value] : [value[0], value[1]];
  for (const item of result) positiveInteger(item, name, allowZero);
  return result;
}

/** ``torch.nn.GroupNorm`` over ``[N, C, *]`` with per-channel affine parameters. */
export class GroupNorm extends Module implements TensorModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.normalization.GroupNorm';

  override configurationAttributes(): Record<string, unknown> {
    return { affine: this.weight !== null, eps: this.eps, num_channels: this.numChannels, num_groups: this.numGroups };
  }

  readonly numGroups: number;
  readonly numChannels: number;
  readonly eps: number;
  weight: Parameter | null;
  bias: Parameter | null;

  constructor(numGroups: number, numChannels: number, options: { eps?: number; affine?: boolean } = {}) {
    super();
    positiveInteger(numGroups, 'numGroups');
    positiveInteger(numChannels, 'numChannels');
    if (numChannels % numGroups !== 0) throw new RangeError('numChannels must be divisible by numGroups');
    this.numGroups = numGroups;
    this.numChannels = numChannels;
    this.eps = options.eps ?? 1e-5;
    const affine = options.affine ?? true;
    this.weight = this.registerParameter('weight', affine ? new Parameter(ones([numChannels])) : null);
    this.bias = this.registerParameter('bias', affine ? new Parameter(zeros([numChannels])) : null);
  }

  protected override onRegistryChange(): void {
    this.weight = this.getParameter('weight');
    this.bias = this.getParameter('bias');
  }

  forward(input: Tensor): Tensor {
    return groupNormOp(input, this.numGroups, this.weight, this.bias, this.eps);
  }
}

/** ``torch.nn.modules.linear.NonDynamicallyQuantizableLinear`` (the attention output projection). */
export class NonDynamicallyQuantizableLinear extends Linear {
  static override readonly qualifiedName: string = 'torch.nn.modules.linear.NonDynamicallyQuantizableLinear';
}

/**
 * ``torch.nn.MultiheadAttention`` with a packed input projection
 * (``in_proj_weight``/``in_proj_bias``) and the ``out_proj`` output layer.
 * ``forward`` returns ``[output, weights]``; weights are averaged over heads
 * (``need_weights=True``) or ``null``.
 */
export class MultiheadAttention extends Module {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.MultiheadAttention';
  readonly embedDim: number;
  readonly numHeads: number;
  readonly headDim: number;
  readonly dropout: number;
  readonly batchFirst: boolean;
  in_proj_weight: Parameter;
  in_proj_bias: Parameter | null;
  readonly out_proj: NonDynamicallyQuantizableLinear;

  override configurationAttributes(): Record<string, unknown> {
    return {
      embed_dim: this.embedDim, kdim: this.embedDim, vdim: this.embedDim, num_heads: this.numHeads, dropout: this.dropout,
      batch_first: this.batchFirst, head_dim: this.headDim, bias_k: null, bias_v: null, add_zero_attn: false,
    };
  }

  constructor(embedDim: number, numHeads: number, options: { bias?: boolean; batchFirst?: boolean; dropout?: number } = {}) {
    super();
    positiveInteger(embedDim, 'embedDim');
    positiveInteger(numHeads, 'numHeads');
    if (embedDim % numHeads !== 0) throw new RangeError('embed_dim must be divisible by num_heads');
    this.embedDim = embedDim;
    this.numHeads = numHeads;
    this.headDim = embedDim / numHeads;
    this.dropout = options.dropout ?? 0;
    this.batchFirst = options.batchFirst ?? false;
    const bias = options.bias !== false;
    this.in_proj_weight = this.registerParameter('in_proj_weight', new Parameter(empty([3 * embedDim, embedDim])));
    this.in_proj_bias = this.registerParameter('in_proj_bias', bias ? new Parameter(zeros([3 * embedDim])) : null);
    this.out_proj = this.registerModule('out_proj', new NonDynamicallyQuantizableLinear(embedDim, embedDim, { bias }));
    init.xavierUniform_(this.in_proj_weight);
    if (this.out_proj.bias) init.zeros_(this.out_proj.bias);
  }

  protected override onRegistryChange(): void {
    this.in_proj_weight = (this.getParameter('in_proj_weight') ?? this.in_proj_weight) as Parameter;
    this.in_proj_bias = this.getParameter('in_proj_bias');
  }

  /**
   * ``query``/``key``/``value`` are ``[batch, length, embed]`` when
   * ``batchFirst`` (else ``[length, batch, embed]``). ``keyPaddingMask``
   * (``[batch, keys]``, true = ignore) is optional.
   */
  forward(query: Tensor, key: Tensor, value: Tensor, options: { keyPaddingMask?: Tensor | null; needWeights?: boolean } = {}): [Tensor, Tensor | null] {
    const q0 = this.batchFirst ? query : query.transpose(0, 1);
    const k0 = this.batchFirst ? key : key.transpose(0, 1);
    const v0 = this.batchFirst ? value : value.transpose(0, 1);
    const e = this.embedDim;
    const part = (index: number): [Tensor, Tensor | null] => [
      this.in_proj_weight.slice(0, index * e, (index + 1) * e),
      this.in_proj_bias ? this.in_proj_bias.slice(0, index * e, (index + 1) * e) : null,
    ];
    const [wq, bq] = part(0);
    const [wk, bk] = part(1);
    const [wv, bv] = part(2);
    const split = (x: Tensor): Tensor => {
      const [batch, length] = x.shape as [number, number];
      return x.reshape(batch, length, this.numHeads, this.headDim).transpose(1, 2);
    };
    const q = split(linearOp(q0, wq, bq));
    const k = split(linearOp(k0, wk, bk));
    const v = split(linearOp(v0, wv, bv));
    let bias: Tensor | null = null;
    const mask = options.keyPaddingMask;
    if (mask) {
      const [batch, keys] = mask.shape as [number, number];
      const values = new Float32Array(batch * keys);
      for (let index = 0; index < values.length; index += 1) values[index] = mask.data[index] ? -Infinity : 0;
      bias = tensorOf(values, { shape: [batch, 1, 1, keys] });
    }
    const { output, weights } = scaledDotProductAttentionOp(q, k, v, { bias, dropout: this.dropout, training: this.training });
    const [batch, , length] = output.shape as [number, number, number];
    let result = this.out_proj.forward(output.transpose(1, 2).reshape(batch, length, e));
    if (!this.batchFirst) result = result.transpose(0, 1);
    return [result, options.needWeights === false ? null : weights.mean(1)];
  }
}
