/**
 * ``LlamaModel`` (transformers 5.17 names): RMSNorm, rotary position
 * embeddings (every ``ROPE_INIT_FUNCTIONS`` type: ``default``, ``linear``,
 * ``dynamic``, ``yarn``, ``longrope``, ``llama3`` and ``proportional``), grouped-query
 * attention and the gated SiLU MLP, with a key/value cache for incremental
 * decoding. Used as the Idefics3 (SmolVLM) text model.
 */
import { activationModule, type ActivationModule } from './activations.js';
import { Module } from '../../nn/module.js';
import { Parameter, Tensor, ones, tensor } from '../../nn/tensor.js';
import { Embedding, Linear, ModuleList } from '../../nn/layers.js';
import { cat } from '../../nn/ops/shape.js';
import { KeyError, ValueError } from '../../errors.js';

export { KeyError };
import { isPlainObject, type JsonObject } from '../json.js';
import { baseInitWeights, initializerStd, postInit } from './hfInit.js';
import type { NativeConfig } from './config.js';
import { NativeModel, attention, causalBias, combineBias, keyPaddingBias, mergeHeads } from './modules.js';

/** ``LlamaRMSNorm``: normalized in float32, cast back, then scaled. */
export class LlamaRMSNorm extends Module {
  static override readonly qualifiedName: string = 'transformers.models.llama.modeling_llama.LlamaRMSNorm';
  readonly weight: Parameter;
  readonly eps: number;

  constructor(hidden: number, eps = 1e-6) {
    super();
    this.weight = this.registerParameter('weight', new Parameter(ones([hidden])));
    this.eps = eps;
  }

  forward(hidden: Tensor): Tensor {
    const states = hidden.to('float32');
    const variance = states.square().mean(-1, true);
    return this.weight.mul(states.mul(variance.add(this.eps).rsqrt()).to(hidden.dtype));
  }
}

const f32 = Math.fround;

/** Standardized ``rope_parameters`` (``RotaryEmbeddingConfigMixin.standardize_rope_params``). */
export function ropeParameters(config: NativeConfig): JsonObject {
  const parameters = config.get('rope_parameters') ?? config.get('rope_scaling');
  const rope: JsonObject = isPlainObject(parameters) ? { ...(parameters as JsonObject) } : {};
  if (rope.rope_type === undefined) rope.rope_type = rope.type ?? 'default';
  if (rope.rope_theta === undefined || rope.rope_theta === null) rope.rope_theta = config.optionalNumber('rope_theta') ?? 10000;
  if (['llama3', 'yarn', 'longrope'].includes(String(rope.rope_type))) {
    const original = config.optionalNumber('original_max_position_embeddings');
    if (original !== null) rope.original_max_position_embeddings = original;
    else if (rope.original_max_position_embeddings === undefined) rope.original_max_position_embeddings = config.number('max_position_embeddings');
  }
  return rope;
}

/** ``1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))`` in float32. */
function baseFrequencies(base: number, dim: number, count = Math.ceil(dim / 2)): Float32Array {
  const out = new Float32Array(count);
  for (let index = 0; index < count; index += 1) out[index] = f32(1 / f32(base ** f32((2 * index) / dim)));
  return out;
}

function headDimension(config: NativeConfig): number {
  return config.optionalNumber('head_dim') ?? Math.floor(config.number('hidden_size') / config.number('num_attention_heads'));
}

/** ``_compute_dynamic_ntk_parameters`` for a sequence length (``null``: the configuration's). */
function dynamicFrequencies(config: NativeConfig, rope: JsonObject, seqLen: number | null): Float32Array {
  const base = Number(rope.rope_theta);
  const partial = typeof rope.partial_rotary_factor === 'number' ? rope.partial_rotary_factor : 1;
  const dim = Math.trunc(headDimension(config) * partial);
  const factor = Number(rope.factor);
  const maxPositions = config.number('max_position_embeddings');
  const exponent = dim / (dim - 2);
  let scaled: number;
  if (seqLen === null) {
    // Python floats (double precision) at construction.
    scaled = base * ((factor * maxPositions) / maxPositions - (factor - 1)) ** exponent;
  } else {
    // A tensor ``seq_len``: float32 arithmetic.
    const seq = Math.max(seqLen, maxPositions);
    let value = f32(f32(factor) * f32(seq));
    value = f32(value / f32(maxPositions));
    value = f32(value - f32(factor - 1));
    value = f32(value ** f32(exponent));
    scaled = f32(f32(base) * value);
  }
  const out = new Float32Array(Math.ceil(dim / 2));
  for (let index = 0; index < out.length; index += 1) out[index] = f32(1 / f32(scaled ** f32((2 * index) / dim)));
  return out;
}

/** ``_compute_longrope_parameters`` for a sequence length (``null``: construction). */
function longropeFrequencies(config: NativeConfig, rope: JsonObject, seqLen: number | null): { invFreq: Float32Array; scaling: number } {
  const base = Number(rope.rope_theta);
  const partial = typeof rope.partial_rotary_factor === 'number' ? rope.partial_rotary_factor : 1;
  const dim = Math.trunc(headDimension(config) * partial);
  const original = Number(rope.original_max_position_embeddings);
  let factor = typeof rope.factor === 'number' ? rope.factor : null;
  if (factor === null) factor = config.number('max_position_embeddings') / original;
  let scaling = typeof rope.attention_factor === 'number' ? rope.attention_factor : null;
  if (scaling === null) scaling = factor <= 1 ? 1 : Math.sqrt(1 + Math.log(factor) / Math.log(original));
  const factors = (seqLen !== null && seqLen > original ? rope.long_factor : rope.short_factor) as number[];
  const count = Math.ceil(dim / 2);
  if (!Array.isArray(factors) || factors.length !== count) {
    throw new ValueError(`The size of tensor a (${Array.isArray(factors) ? factors.length : 0}) must match the size of tensor b (${count}) at non-singleton dimension 0`);
  }
  const out = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    out[index] = f32(1 / f32(f32(factors[index]!) * f32(base ** f32((2 * index) / dim))));
  }
  return { invFreq: out, scaling };
}

/** ``_compute_yarn_parameters``. */
function yarnFrequencies(config: NativeConfig, rope: JsonObject): { invFreq: Float32Array; scaling: number } {
  const base = Number(rope.rope_theta);
  const partial = typeof rope.partial_rotary_factor === 'number' ? rope.partial_rotary_factor : 1;
  const dim = Math.trunc(headDimension(config) * partial);
  const original = Number(rope.original_max_position_embeddings);
  let factor = typeof rope.factor === 'number' ? rope.factor : null;
  if (factor === null) factor = config.number('max_position_embeddings') / original;
  const mscaleOf = (scale: number, mscale = 1): number => (scale <= 1 ? 1 : 0.1 * mscale * Math.log(scale) + 1);
  let scaling = typeof rope.attention_factor === 'number' ? rope.attention_factor : null;
  if (scaling === null) {
    const mscale = typeof rope.mscale === 'number' ? rope.mscale : null;
    const mscaleAll = typeof rope.mscale_all_dim === 'number' ? rope.mscale_all_dim : null;
    scaling = mscale && mscaleAll ? mscaleOf(factor, mscale) / mscaleOf(factor, mscaleAll) : mscaleOf(factor);
  }
  const betaFast = (typeof rope.beta_fast === 'number' && rope.beta_fast) || 32;
  const betaSlow = (typeof rope.beta_slow === 'number' && rope.beta_slow) || 1;
  const correction = (rotations: number): number => (dim * Math.log(original / (rotations * 2 * Math.PI))) / (2 * Math.log(base));
  // ``config.rope_parameters.get('truncate', True)`` (an explicit ``None`` disables truncation).
  const truncate = rope.truncate === undefined ? true : Boolean(rope.truncate);
  let low = correction(betaFast);
  let high = correction(betaSlow);
  if (truncate) {
    low = Math.floor(low);
    high = Math.ceil(high);
  }
  low = Math.max(low, 0);
  high = Math.min(high, dim - 1);
  if (low === high) high += 0.001;
  const count = Math.ceil(dim / 2);
  const half = Math.floor(dim / 2);
  const out = new Float32Array(count);
  const f = f32(factor);
  for (let index = 0; index < count; index += 1) {
    const position = f32(base ** f32((2 * index) / dim));
    const extrapolation = f32(1 / position);
    const interpolation = f32(1 / f32(f * position));
    const ramp = index < half ? Math.min(Math.max(f32(f32(f32(index) - f32(low)) / f32(high - low)), 0), 1) : 0;
    const extrapolationFactor = f32(1 - ramp);
    out[index] = f32(f32(interpolation * f32(1 - extrapolationFactor)) + f32(extrapolation * extrapolationFactor));
  }
  return { invFreq: out, scaling };
}

/** RoPE inverse frequencies (float32) and attention scaling of a Llama configuration at construction. */
export function ropeInverseFrequencies(config: NativeConfig): { invFreq: Float32Array; scaling: number } {
  const rope = ropeParameters(config);
  const type = String(rope.rope_type);
  const base = Number(rope.rope_theta);
  const headDim = headDimension(config);
  const partial = typeof rope.partial_rotary_factor === 'number' ? rope.partial_rotary_factor : 1;
  const dim = Math.trunc(headDim * partial);
  if (type === 'default') return { invFreq: baseFrequencies(base, headDim), scaling: 1 };
  if (type === 'linear') {
    const invFreq = baseFrequencies(base, dim);
    const factor = f32(Number(rope.factor));
    for (let index = 0; index < invFreq.length; index += 1) invFreq[index] = f32(invFreq[index]! / factor);
    return { invFreq, scaling: 1 };
  }
  if (type === 'dynamic') return { invFreq: dynamicFrequencies(config, rope, null), scaling: 1 };
  if (type === 'yarn') return yarnFrequencies(config, rope);
  if (type === 'longrope') return longropeFrequencies(config, rope, null);
  if (type === 'proportional') {
    const factor = typeof rope.factor === 'number' ? rope.factor : 1;
    const angles = Math.trunc(Math.floor((partial * headDim) / 2));
    const invFreq = new Float32Array(Math.floor(headDim / 2));
    for (let index = 0; index < angles; index += 1) invFreq[index] = f32(1 / f32(base ** f32((2 * index) / headDim)));
    for (let index = 0; index < invFreq.length; index += 1) invFreq[index] = f32(invFreq[index]! / f32(factor));
    return { invFreq, scaling: 1 };
  }
  if (type === 'llama3') {
    const invFreq = baseFrequencies(base, dim);
    const factor = Number(rope.factor);
    const low = Number(rope.low_freq_factor);
    const high = Number(rope.high_freq_factor);
    const original = Number(rope.original_max_position_embeddings);
    const lowWavelen = original / low;
    const highWavelen = original / high;
    for (let index = 0; index < invFreq.length; index += 1) {
      const frequency = invFreq[index]!;
      const wavelen = f32((2 * Math.PI) / frequency);
      let value = wavelen > lowWavelen ? f32(frequency / factor) : frequency;
      const smooth = f32(f32(original / wavelen - low) / (high - low));
      // ``(1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama``
      const smoothed = f32(f32(f32(f32(1 - smooth) * value) / factor) + f32(smooth * value));
      if (!(wavelen < highWavelen) && !(wavelen > lowWavelen)) value = smoothed;
      invFreq[index] = value;
    }
    return { invFreq, scaling: 1 };
  }
  throw new KeyError(`'${type}'`);
}

/**
 * ``LlamaRotaryEmbedding`` (the frequency buffers are not persistent). The
 * ``dynamic`` and ``longrope`` types update their frequencies from the largest
 * position of each forward pass (``dynamic_rope_update``), statefully as in
 * transformers.
 */
export class LlamaRotaryEmbedding extends Module {
  static override readonly qualifiedName: string = 'transformers.models.llama.modeling_llama.LlamaRotaryEmbedding';
  attentionScaling: number;
  readonly #config: NativeConfig;
  readonly #rope: JsonObject;
  readonly #type: string;
  #maxSeqLenCached: number;
  readonly #originalMaxSeqLen: number;

  constructor(config: NativeConfig) {
    super();
    const { invFreq, scaling } = ropeInverseFrequencies(config);
    this.#config = config;
    this.#rope = ropeParameters(config);
    this.#type = String(this.#rope.rope_type);
    this.#maxSeqLenCached = config.number('max_position_embeddings');
    this.#originalMaxSeqLen = this.#maxSeqLenCached;
    this.attentionScaling = scaling;
    this.registerBuffer('inv_freq', tensor(invFreq), false);
    this.registerBuffer('original_inv_freq', tensor(Float32Array.from(invFreq)), false);
  }

  /** ``dynamic_rope_update``. */
  #update(positionIds: Tensor): void {
    if (this.#type !== 'dynamic' && this.#type !== 'longrope') return;
    let max = -Infinity;
    for (const value of positionIds.data) if (value > max) max = value;
    const seqLen = max + 1;
    if (this.#type === 'dynamic') {
      if (seqLen > this.#maxSeqLenCached) {
        this.registerBuffer('inv_freq', tensor(dynamicFrequencies(this.#config, this.#rope, seqLen)), false);
        this.attentionScaling = 1;
        this.#maxSeqLenCached = seqLen;
      }
      if (seqLen < this.#originalMaxSeqLen && this.#maxSeqLenCached > this.#originalMaxSeqLen) {
        this.registerBuffer('inv_freq', tensor(Float32Array.from(this.getBuffer('original_inv_freq')!.data)), false);
        this.#maxSeqLenCached = this.#originalMaxSeqLen;
      }
      return;
    }
    const original = Number(this.#rope.original_max_position_embeddings);
    const invFreq = seqLen > original
      ? longropeFrequencies(this.#config, this.#rope, original + 1).invFreq
      : Float32Array.from(this.getBuffer('original_inv_freq')!.data);
    this.registerBuffer('inv_freq', tensor(invFreq), false);
  }

  /** ``cos``/``sin`` ``[batch, length, dim]`` in ``dtype`` for int64 ``positionIds`` ``[batch, length]``. */
  forward(positionIds: Tensor, dtype: Tensor['dtype']): [Tensor, Tensor] {
    this.#update(positionIds);
    const invFreq = this.getBuffer('inv_freq')!.data;
    const half = invFreq.length;
    const [batch, length] = positionIds.shape as [number, number];
    const cos = new Float32Array(batch * length * half * 2);
    const sin = new Float32Array(batch * length * half * 2);
    const scaling = f32(this.attentionScaling);
    for (let row = 0; row < batch * length; row += 1) {
      const position = f32(positionIds.data[row]!);
      for (let index = 0; index < half; index += 1) {
        const angle = f32(invFreq[index]! * position);
        const c = scaling === 1 ? f32(Math.cos(angle)) : f32(f32(Math.cos(angle)) * scaling);
        const s = scaling === 1 ? f32(Math.sin(angle)) : f32(f32(Math.sin(angle)) * scaling);
        const offset = row * half * 2;
        cos[offset + index] = c;
        cos[offset + half + index] = c;
        sin[offset + index] = s;
        sin[offset + half + index] = s;
      }
    }
    const shape = [batch, length, half * 2];
    return [tensor(cos, { shape }).to(dtype), tensor(sin, { shape }).to(dtype)];
  }
}

function rotateHalf(x: Tensor): Tensor {
  const width = x.shape[x.ndim - 1]!;
  const half = width / 2;
  return cat([x.slice(-1, half, width).neg(), x.slice(-1, 0, half)], -1);
}

/** ``apply_rotary_pos_emb`` over ``[batch, heads, length, dim]`` with ``cos``/``sin`` ``[batch, length, dim]``. */
export function applyRotary(x: Tensor, cos: Tensor, sin: Tensor): Tensor {
  const c = cos.unsqueeze(1);
  const s = sin.unsqueeze(1);
  return x.mul(c).add(rotateHalf(x).mul(s));
}

/** Per-layer key/value cache (keys and values after RoPE, before head repetition). */
export interface LlamaLayerCache {
  key: Tensor | null;
  value: Tensor | null;
}

export class LlamaAttention extends Module {
  static override readonly qualifiedName: string = 'transformers.models.llama.modeling_llama.LlamaAttention';
  readonly heads: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly dropoutRate: number;
  readonly q_proj: Linear;
  readonly k_proj: Linear;
  readonly v_proj: Linear;
  readonly o_proj: Linear;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    this.heads = config.number('num_attention_heads');
    this.kvHeads = config.number('num_key_value_heads');
    this.headDim = config.optionalNumber('head_dim') ?? Math.floor(hidden / this.heads);
    this.dropoutRate = config.optionalNumber('attention_dropout') ?? 0;
    const bias = config.get('attention_bias') === true;
    this.q_proj = this.registerModule('q_proj', new Linear(hidden, this.heads * this.headDim, { bias }));
    this.k_proj = this.registerModule('k_proj', new Linear(hidden, this.kvHeads * this.headDim, { bias }));
    this.v_proj = this.registerModule('v_proj', new Linear(hidden, this.kvHeads * this.headDim, { bias }));
    this.o_proj = this.registerModule('o_proj', new Linear(this.heads * this.headDim, hidden, { bias }));
  }

  forward(hidden: Tensor, cos: Tensor, sin: Tensor, bias: Tensor | null, cache: LlamaLayerCache | null = null): Tensor {
    const [batch, length] = hidden.shape as [number, number];
    const split = (x: Tensor, heads: number) => x.reshape(batch, length, heads, this.headDim).transpose(1, 2);
    const query = applyRotary(split(this.q_proj.forward(hidden), this.heads), cos, sin);
    let key = applyRotary(split(this.k_proj.forward(hidden), this.kvHeads), cos, sin);
    let value = split(this.v_proj.forward(hidden), this.kvHeads);
    if (cache) {
      if (cache.key) key = cat([cache.key, key], 2);
      if (cache.value) value = cat([cache.value, value], 2);
      cache.key = key;
      cache.value = value;
    }
    // ``repeat_kv`` happens inside attention (transformers' SDPA ``enable_gqa``).
    const output = attention(query, key, value, {
      scale: this.headDim ** -0.5, bias, dropout: this.dropoutRate, training: this.training, enableGqa: this.heads !== this.kvHeads,
    });
    return this.o_proj.forward(mergeHeads(output));
  }
}

export class LlamaMLP extends Module {
  static override readonly qualifiedName: string = 'transformers.models.llama.modeling_llama.LlamaMLP';
  readonly gate_proj: Linear;
  readonly up_proj: Linear;
  readonly down_proj: Linear;
  readonly act_fn: ActivationModule;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    const intermediate = config.number('intermediate_size');
    const bias = config.get('mlp_bias') === true;
    this.gate_proj = this.registerModule('gate_proj', new Linear(hidden, intermediate, { bias }));
    this.up_proj = this.registerModule('up_proj', new Linear(hidden, intermediate, { bias }));
    this.down_proj = this.registerModule('down_proj', new Linear(intermediate, hidden, { bias }));
    this.act_fn = this.registerModule('act_fn', activationModule(config.string('hidden_act')));
  }

  forward(hidden: Tensor): Tensor {
    return this.down_proj.forward(this.act_fn.forward(this.gate_proj.forward(hidden)).mul(this.up_proj.forward(hidden)));
  }
}

export class LlamaDecoderLayer extends Module {
  static override readonly qualifiedName: string = 'transformers.models.llama.modeling_llama.LlamaDecoderLayer';
  readonly self_attn: LlamaAttention;
  readonly mlp: LlamaMLP;
  readonly input_layernorm: LlamaRMSNorm;
  readonly post_attention_layernorm: LlamaRMSNorm;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    const eps = config.number('rms_norm_eps');
    this.self_attn = this.registerModule('self_attn', new LlamaAttention(config));
    this.mlp = this.registerModule('mlp', new LlamaMLP(config));
    this.input_layernorm = this.registerModule('input_layernorm', new LlamaRMSNorm(hidden, eps));
    this.post_attention_layernorm = this.registerModule('post_attention_layernorm', new LlamaRMSNorm(hidden, eps));
  }

  forward(hidden: Tensor, cos: Tensor, sin: Tensor, bias: Tensor | null, cache: LlamaLayerCache | null): Tensor {
    const attended = hidden.add(this.self_attn.forward(this.input_layernorm.forward(hidden), cos, sin, bias, cache));
    return attended.add(this.mlp.forward(this.post_attention_layernorm.forward(attended)));
  }
}

export interface LlamaInputs {
  inputIds?: Tensor | null;
  inputsEmbeds?: Tensor | null;
  /** ``[batch, past + length]`` padding mask (1 keeps a position). */
  attentionMask?: Tensor | null;
  positionIds?: Tensor | null;
  /** Per-layer caches, updated in place (``use_cache=True``). */
  cache?: LlamaLayerCache[] | null;
}

/** ``LlamaModel``. */
export class LlamaModel extends NativeModel {
  static override readonly qualifiedName: string = 'transformers.models.llama.modeling_llama.LlamaModel';
  readonly embed_tokens: Embedding;
  readonly layers: ModuleList<LlamaDecoderLayer>;
  readonly norm: LlamaRMSNorm;
  readonly rotary_emb: LlamaRotaryEmbedding;

  constructor(config: NativeConfig) {
    super(config);
    const hidden = config.number('hidden_size');
    const pad = config.optionalNumber('pad_token_id');
    const vocab = config.number('vocab_size');
    this.embed_tokens = this.registerModule('embed_tokens', new Embedding(vocab, hidden, { paddingIdx: pad !== null && pad >= 0 && pad < vocab ? pad : null }));
    this.layers = this.registerModule('layers', new ModuleList(Array.from({ length: config.number('num_hidden_layers') }, () => new LlamaDecoderLayer(config))));
    this.norm = this.registerModule('norm', new LlamaRMSNorm(hidden, config.number('rms_norm_eps')));
    this.rotary_emb = this.registerModule('rotary_emb', new LlamaRotaryEmbedding(config));
    // ``LlamaPreTrainedModel`` uses the base ``_init_weights`` (normal Linear/Embedding weights, unit RMSNorm).
    const std = initializerStd(config);
    postInit(this, (module) => baseInitWeights(module, std));
  }

  getInputEmbeddings(): Embedding {
    return this.embed_tokens;
  }

  /** Fresh per-layer key/value caches. */
  newCache(): LlamaLayerCache[] {
    return Array.from(this.layers, () => ({ key: null, value: null }));
  }

  forward(inputs: LlamaInputs): Tensor {
    const hasIds = inputs.inputIds !== null && inputs.inputIds !== undefined;
    const hasEmbeds = inputs.inputsEmbeds !== null && inputs.inputsEmbeds !== undefined;
    if (hasIds === hasEmbeds) throw new ValueError('You must specify exactly one of input_ids or inputs_embeds');
    const embeds = hasEmbeds ? inputs.inputsEmbeds! : this.embed_tokens.forward(inputs.inputIds!);
    const [batch, length] = embeds.shape as [number, number];
    const cache = inputs.cache ?? null;
    const past = cache?.[0]?.key?.shape[2] ?? 0;
    let positions = inputs.positionIds ?? null;
    if (!positions) {
      const values: number[] = [];
      for (let b = 0; b < batch; b += 1) for (let index = 0; index < length; index += 1) values.push(past + index);
      positions = tensor(values, { shape: [batch, length], dtype: 'int64' });
    }
    const bias = combineBias(causalBias(length, past + length, embeds.dtype), keyPaddingBias(inputs.attentionMask, embeds.dtype));
    const [cos, sin] = this.rotary_emb.forward(positions, embeds.dtype);
    let hidden = embeds;
    let index = 0;
    for (const layer of this.layers) {
      hidden = layer.forward(hidden, cos, sin, bias, cache ? cache[index]! : null);
      index += 1;
    }
    return this.norm.forward(hidden);
  }
}
