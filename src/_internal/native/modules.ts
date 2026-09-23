/** Shared building blocks for native transformer architectures. */
import { Module } from '../../nn/module.js';
import { Parameter, Tensor, fromStorage, full, zeros, tensor } from '../../nn/tensor.js';
import { finfoMin, type DType } from '../../nn/dtype.js';
import { Embedding, Linear, LayerNorm, Conv2d } from '../../nn/layers.js';
import * as init from '../../nn/init.js';
import { isGradEnabled, noGrad } from '../../nn/autograd.js';
import { scaledDotProductAttention } from '../../nn/ops/nn.js';
import { mlpForward, selfAttentionForward, type AttentionBias, type Projection } from '../../nn/backend/kernels.js';
import type { ActivationModule } from './activations.js';
import type { NativeConfig } from './config.js';
import { ValueError } from '../../errors.js';

/** Inputs accepted by native encoders (Python keyword arguments, camelCased). */
export interface EncoderInputs {
  inputIds?: Tensor | null;
  attentionMask?: Tensor | null;
  tokenTypeIds?: Tensor | null;
  positionIds?: Tensor | null;
  inputsEmbeds?: Tensor | null;
}

export interface EncoderOutput {
  lastHiddenState: Tensor;
  poolerOutput: Tensor | null;
}

/** A native model owning a configuration (``PreTrainedModel`` equivalent). */
export abstract class NativeModel extends Module {
  config: NativeConfig;

  constructor(config: NativeConfig) {
    super();
    this.config = config;
  }

  /** Input token embedding table (``get_input_embeddings()``). */
  abstract getInputEmbeddings(): Embedding;
}

/** A native encoder (``AutoModel`` of an encoder-only architecture, or a T5 encoder stack). */
export interface NativeEncoder extends Module {
  readonly config: NativeConfig;
  forward(inputs: EncoderInputs): EncoderOutput;
  getInputEmbeddings(): Embedding;
}

export function isNativeEncoder(value: unknown): value is NativeEncoder {
  return value instanceof Module && typeof (value as { getInputEmbeddings?: unknown }).getInputEmbeddings === 'function'
    && typeof (value as { forward?: unknown }).forward === 'function';
}

/** ``[batch, length, heads * dim]`` → ``[batch, heads, length, dim]``. */
export function splitHeads(x: Tensor, heads: number): Tensor {
  const [batch, length, width] = x.shape as [number, number, number];
  return x.reshape(batch, length, heads, width / heads).transpose(1, 2);
}

/**
 * ``fc2(activation(fc1(x)))``. Without gradients, float32 inputs and a kernel
 * activation run as one fused kernel call that keeps the (large) hidden
 * activations in kernel memory; the result is bit-identical.
 */
export function feedForward(fc1: Linear, activation: ActivationModule, fc2: Linear, x: Tensor): Tensor {
  const op = activation.kernelOp();
  if (op !== null && fusable(x, [fc1, fc2]) && x.shape[x.ndim - 1] === fc1.inFeatures && fc1.outFeatures === fc2.inFeatures) {
    const first = projection(fc1);
    const second = projection(fc2);
    const out = mlpForward(
      x.data as Float32Array, x.numel / fc1.inFeatures, fc1.inFeatures, first.weight, fc1.outFeatures, first.bias, op,
      second.weight, fc2.outFeatures, second.bias,
    );
    if (out) return fromStorage(out, [...x.shape.slice(0, -1), fc2.outFeatures], 'float32');
  }
  return fc2.forward(activation.forward(fc1.forward(x)));
}

/** Whether ``layers`` applied to ``x`` can run as one fused float32 kernel call (no autograd needed). */
function fusable(x: Tensor, layers: readonly Linear[], extra: readonly (Tensor | null)[] = []): boolean {
  if (x.ndim < 1 || !layers.every((layer) => layer instanceof Linear && layer.forward === Linear.prototype.forward && layer.weight.ndim === 2)) return false;
  const operands = [x, ...extra, ...layers.flatMap((layer) => [layer.weight, layer.bias])];
  if (!operands.every((t) => t === null || (t.dtype === 'float32' && t.data instanceof Float32Array))) return false;
  return !(isGradEnabled() && operands.some((t) => t?.requiresGrad));
}

function projection(layer: Linear): Projection {
  return {
    weight: { data: layer.weight.data as Float32Array, key: layer.weight._storage, version: layer.weight._storage.version },
    bias: (layer.bias?.data ?? null) as Float32Array | null,
  };
}

/** The projections of a multi-head self-attention block. */
export interface AttentionProjections {
  query: Linear;
  key: Linear;
  value: Linear;
  output: Linear;
}

/**
 * ``output(mergeHeads(attention(splitHeads(query(x)), splitHeads(key(x)),
 * splitHeads(value(x)))))`` for ``x [batch, length, width]``. Without gradients
 * or dropout, float32 operands run as one fused kernel call that keeps the
 * projections and heads in kernel memory; the result is bit-identical.
 */
export function selfAttention(
  projections: AttentionProjections, x: Tensor, heads: number,
  options: { scale: number; bias?: Tensor | null; dropout?: number; training?: boolean },
): Tensor {
  const { query, key, value, output } = projections;
  const bias = options.bias ?? null;
  const inner = query.outFeatures;
  const dim = inner / heads;
  if (x.ndim === 3 && Number.isInteger(dim) && !(options.dropout && options.training)
    && fusable(x, [query, key, value, output], [bias])
    && x.shape[2] === query.inFeatures && key.inFeatures === query.inFeatures && value.inFeatures === query.inFeatures
    && key.outFeatures === inner && value.outFeatures === inner && output.inFeatures === inner) {
    const [batch, length, width] = x.shape as [number, number, number];
    const strides = bias ? attentionBias(bias, batch, heads, length) : null;
    if (!bias || strides) {
      const out = selfAttentionForward(
        x.data as Float32Array, batch, length, width, heads, dim,
        projection(query), projection(key), projection(value), projection(output), output.outFeatures, options.scale, strides,
      );
      if (out) return fromStorage(out, [batch, length, output.outFeatures], 'float32');
    }
  }
  const q = splitHeads(query.forward(x), heads);
  const k = splitHeads(key.forward(x), heads);
  const v = splitHeads(value.forward(x), heads);
  return output.forward(mergeHeads(attention(q, k, v, {
    scale: options.scale, bias, dropout: options.dropout ?? 0, training: options.training ?? false,
  })));
}

/** Broadcast strides of an additive ``[b, h, q, keys]`` bias (``null``: unsupported layout), as fused attention reads it. */
function attentionBias(bias: Tensor, batch: number, heads: number, length: number): AttentionBias | null {
  if (bias.ndim > 4 || bias.shape[bias.ndim - 1] !== length) return null;
  const [bb, bh, bq] = [...new Array<number>(4 - bias.ndim).fill(1), ...bias.shape] as [number, number, number, number];
  if ((bb !== 1 && bb !== batch) || (bh !== 1 && bh !== heads) || (bq !== 1 && bq !== length)) return null;
  return {
    data: bias.data as Float32Array,
    queryStride: bq === 1 ? 0 : length,
    headStride: bh === 1 ? 0 : bq * length,
    batchStride: bb === 1 ? 0 : bh * bq * length,
  };
}

/** ``[batch, heads, length, dim]`` → ``[batch, length, heads * dim]``. */
export function mergeHeads(x: Tensor): Tensor {
  const [batch, heads, length, dim] = x.shape as [number, number, number, number];
  return x.transpose(1, 2).reshape(batch, length, heads * dim);
}

/** Additive key-padding bias ``[batch, 1, 1, keys]`` (0 valid, dtype minimum masked). */
export function keyPaddingBias(mask: Tensor | null | undefined, dtype: DType = 'float32'): Tensor | null {
  if (!mask) return null;
  if (mask.ndim !== 2) throw new RangeError('attention mask must have shape [batch, length]');
  const [batch, keys] = mask.shape as [number, number];
  const minimum = finfoMin(dtype);
  const values = new Float32Array(batch * keys);
  const source = mask.data;
  for (let index = 0; index < values.length; index += 1) values[index] = source[index] ? 0 : minimum;
  return tensor(values, { shape: [batch, 1, 1, keys], dtype });
}

/** Causal bias ``[1, 1, queries, keys]`` for ``keys - queries`` cached positions. */
export function causalBias(queries: number, keys: number, dtype: DType = 'float32'): Tensor {
  const minimum = finfoMin(dtype);
  const offset = keys - queries;
  const values = new Float32Array(queries * keys);
  for (let q = 0; q < queries; q += 1) for (let k = 0; k < keys; k += 1) values[q * keys + k] = k > q + offset ? minimum : 0;
  return tensor(values, { shape: [1, 1, queries, keys], dtype });
}

/** Combine additive biases, broadcasting; ``null`` entries are skipped. */
export function combineBias(...biases: (Tensor | null | undefined)[]): Tensor | null {
  let result: Tensor | null = null;
  for (const bias of biases) if (bias) result = result ? result.add(bias) : bias;
  return result;
}

/** Multi-head attention core over pre-projected ``[batch, heads, length, dim]`` tensors. */
export function attention(
  query: Tensor, key: Tensor, value: Tensor,
  options: { scale: number; bias?: Tensor | null; dropout?: number; training?: boolean; enableGqa?: boolean },
): Tensor {
  return scaledDotProductAttention(query, key, value, {
    scale: options.scale, bias: options.bias ?? null, dropout: options.dropout ?? 0, training: options.training ?? false,
    enableGqa: options.enableGqa ?? false,
  }).output;
}

/** ``torch.arange(length)`` repeated per batch row as an int64 ``[batch, length]`` tensor. */
export function positionIds(batch: number, length: number, start = 0): Tensor {
  const values: number[] = [];
  for (let b = 0; b < batch; b += 1) for (let i = 0; i < length; i += 1) values.push(start + i);
  return tensor(values, { shape: [batch, length], dtype: 'int64' });
}

/**
 * Register transformers' non-persistent embedding buffers: ``position_ids``
 * (``arange(length).expand((1, -1))``) and optionally ``token_type_ids``
 * (zeros). They are not in state dicts, but ``named_buffers()`` (tensor
 * schemas, content fingerprints) lists them exactly as in Python.
 */
export function registerPositionBuffers(module: Module, length: number, tokenTypes: boolean): void {
  module.registerBuffer('position_ids', positionIds(1, length), false);
  if (tokenTypes) module.registerBuffer('token_type_ids', zerosLong([1, length]), false);
}

export function zerosLong(shape: readonly number[]): Tensor {
  return zeros(shape, { dtype: 'int64' });
}

export function onesLong(shape: readonly number[]): Tensor {
  return full(shape, 1, { dtype: 'int64' });
}

/** Hugging Face default initialization: normal(0, std) weights, zero biases, unit norms. */
export function initializeWeights(module: Module, std: number): void {
  noGrad(() => {
    for (const child of module.modules()) {
      if (child instanceof Linear) {
        init.normal_(child.weight, 0, std);
        if (child.bias) child.bias.zero_();
      } else if (child instanceof Embedding) {
        init.normal_(child.weight, 0, std);
        if (child.paddingIdx !== null) {
          const width = child.embeddingDim;
          child.weight.data.fill(0, child.paddingIdx * width, (child.paddingIdx + 1) * width);
        }
      } else if (child instanceof LayerNorm) {
        if (child.weight) child.weight.fill_(1);
        if (child.bias) child.bias.zero_();
      } else if (child instanceof Conv2d) {
        init.normal_(child.weight, 0, std);
        if (child.bias) child.bias.zero_();
      }
    }
  });
}

/** Register a parameter of the given shape initialized by ``fill``. */
export function newParameter(shape: number[], fill: (value: Tensor) => void = () => {}): Parameter {
  const value = new Parameter(zeros(shape));
  noGrad(() => fill(value));
  return value;
}

/**
 * Python ``_parameter_aliases``: every parameter path mapped to the first path
 * that registers the same tensor (``named_parameters(remove_duplicate=False)``).
 */
export function parameterAliases(model: Module): Record<string, string> {
  const canonical = new Map<Parameter, string>();
  const aliases: Record<string, string> = {};
  for (const [name, parameter] of model.namedParameters({ removeDuplicate: false })) {
    if (!canonical.has(parameter)) canonical.set(parameter, name);
    aliases[name] = canonical.get(parameter)!;
  }
  return aliases;
}

/**
 * Python ``_restore_parameter_aliases``: rebuild the exact native parameter
 * sharing (for example partially untied T5 embeddings) recorded in a config.
 */
export function restoreParameterAliases(model: Module, aliases: Record<string, string> | null | undefined): void {
  if (aliases === null || aliases === undefined) return;
  const params = new Map<string, Parameter>(model.namedParameters({ removeDuplicate: false }));
  const names = Object.keys(aliases);
  if (names.length !== params.size || names.some((name) => !params.has(name))) {
    throw new ValueError('native parameter topology differs from configuration');
  }
  const seen = new Set<Parameter>();
  for (const [name, source] of Object.entries(aliases)) {
    if (!params.has(source) || aliases[source] !== source) throw new ValueError('invalid native parameter alias topology');
    if (name === source) {
      let parameter = params.get(name)!;
      if (seen.has(parameter)) parameter = new Parameter(parameter.detach(), parameter.requiresGrad);
      seen.add(parameter);
      params.set(name, parameter);
    }
  }
  for (const [name, source] of Object.entries(aliases)) model.setParameterAt(name, params.get(source)!);
}
