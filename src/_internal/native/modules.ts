/** Shared building blocks for native transformer architectures. */
import { Module } from '../../nn/module.js';
import { Parameter, Tensor, full, zeros, tensor } from '../../nn/tensor.js';
import { finfoMin, type DType } from '../../nn/dtype.js';
import { Embedding, Linear, LayerNorm, Conv2d } from '../../nn/layers.js';
import * as init from '../../nn/init.js';
import { noGrad } from '../../nn/autograd.js';
import { scaledDotProductAttention } from '../../nn/ops/nn.js';
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
