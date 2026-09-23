/**
 * T5 encoder-decoder (``T5ForConditionalGeneration``/``T5EncoderModel``) with
 * transformers 5.17 parameter names, teacher-forced loss and a decoder
 * key/value cache for generation.
 */
import { activationModule, type ActivationModule } from './activations.js';
import { Module } from '../../nn/module.js';
import { Tensor, tensor, zeros } from '../../nn/tensor.js';
import { Dropout, Embedding, Linear, ModuleList } from '../../nn/layers.js';
import { crossEntropy } from '../../nn/ops/nn.js';
import { cat } from '../../nn/ops/shape.js';
import { Parameter } from '../../nn/tensor.js';
import * as init from '../../nn/init.js';
import { ValueError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import { baseInitWeights, initializerStd, postInit, type InitWeights } from './hfInit.js';
import {
  NativeModel, attention, causalBias, combineBias, keyPaddingBias, mergeHeads, splitHeads,
  type EncoderInputs, type EncoderOutput, type NativeEncoder,
} from './modules.js';

export class T5LayerNorm extends Module {
  readonly weight: Parameter;
  readonly eps: number;

  constructor(hidden: number, eps = 1e-6) {
    super();
    this.weight = this.registerParameter('weight', new Parameter(zeros([hidden]).add(1)));
    this.eps = eps;
  }

  /**
   * transformers ``T5LayerNorm``: the variance is always accumulated in
   * float32 (also for float64 inputs), and half-precision weights cast the
   * normalized states to their dtype before scaling.
   */
  forward(hidden: Tensor): Tensor {
    const variance = hidden.to('float32').square().mean(-1, true);
    let normalized = hidden.mul(variance.add(this.eps).rsqrt());
    if (this.weight.dtype === 'float16' || this.weight.dtype === 'bfloat16') normalized = normalized.to(this.weight.dtype);
    return this.weight.mul(normalized);
  }
}

class T5DenseActDense extends Module {
  readonly wi: Linear;
  readonly wo: Linear;
  readonly dropout: Dropout;
  readonly act: ActivationModule;

  constructor(config: NativeConfig) {
    super();
    const model = config.number('d_model');
    const ff = config.number('d_ff');
    this.wi = this.registerModule('wi', new Linear(model, ff, { bias: false }));
    this.wo = this.registerModule('wo', new Linear(ff, model, { bias: false }));
    this.dropout = this.registerModule('dropout', new Dropout(config.number('dropout_rate')));
    this.act = this.registerModule('act', activationModule(config.string('dense_act_fn')));
  }

  forward(hidden: Tensor): Tensor {
    return this.wo.forward(this.dropout.forward(this.act.forward(this.wi.forward(hidden))));
  }
}

class T5DenseGatedActDense extends Module {
  readonly wi_0: Linear;
  readonly wi_1: Linear;
  readonly wo: Linear;
  readonly dropout: Dropout;
  readonly act: ActivationModule;

  constructor(config: NativeConfig) {
    super();
    const model = config.number('d_model');
    const ff = config.number('d_ff');
    this.wi_0 = this.registerModule('wi_0', new Linear(model, ff, { bias: false }));
    this.wi_1 = this.registerModule('wi_1', new Linear(model, ff, { bias: false }));
    this.wo = this.registerModule('wo', new Linear(ff, model, { bias: false }));
    this.dropout = this.registerModule('dropout', new Dropout(config.number('dropout_rate')));
    this.act = this.registerModule('act', activationModule(config.string('dense_act_fn')));
  }

  forward(hidden: Tensor): Tensor {
    const gated = this.act.forward(this.wi_0.forward(hidden)).mul(this.wi_1.forward(hidden));
    return this.wo.forward(this.dropout.forward(gated));
  }
}

class T5LayerFF extends Module {
  readonly DenseReluDense: T5DenseActDense | T5DenseGatedActDense;
  readonly layer_norm: T5LayerNorm;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    this.DenseReluDense = this.registerModule('DenseReluDense',
      config.boolean('is_gated_act') ? new T5DenseGatedActDense(config) : new T5DenseActDense(config));
    this.layer_norm = this.registerModule('layer_norm', new T5LayerNorm(config.number('d_model'), config.number('layer_norm_epsilon')));
    this.dropout = this.registerModule('dropout', new Dropout(config.number('dropout_rate')));
  }

  forward(hidden: Tensor): Tensor {
    return hidden.add(this.dropout.forward(this.DenseReluDense.forward(this.layer_norm.forward(hidden))));
  }
}

/** T5 relative position bucket (Mesh TensorFlow), computed in float32 like PyTorch. */
export function relativePositionBucket(relative: number, bidirectional: boolean, numBuckets: number, maxDistance: number): number {
  let buckets = 0;
  let buckets_ = numBuckets;
  let position = relative;
  if (bidirectional) {
    buckets_ = Math.floor(buckets_ / 2);
    if (position > 0) buckets += buckets_;
    position = Math.abs(position);
  } else {
    position = -Math.min(position, 0);
  }
  const maxExact = Math.floor(buckets_ / 2);
  if (position < maxExact) return buckets + position;
  const f = Math.fround;
  const ratio = f(f(Math.log(f(f(position) / maxExact))) / f(Math.log(maxDistance / maxExact)));
  const large = maxExact + Math.trunc(f(ratio * (buckets_ - maxExact)));
  return buckets + Math.min(large, buckets_ - 1);
}

export interface AttentionCache {
  key: Tensor | null;
  value: Tensor | null;
}

class T5Attention extends Module {
  readonly heads: number;
  readonly headDim: number;
  readonly isDecoder: boolean;
  readonly hasRelativeBias: boolean;
  readonly numBuckets: number;
  readonly maxDistance: number;
  readonly dropoutRate: number;
  readonly q: Linear;
  readonly k: Linear;
  readonly v: Linear;
  readonly o: Linear;
  readonly relative_attention_bias: Embedding | null;

  constructor(config: NativeConfig, hasRelativeBias: boolean, isDecoder: boolean) {
    super();
    const model = config.number('d_model');
    this.heads = config.number('num_heads');
    this.headDim = config.number('d_kv');
    this.isDecoder = isDecoder;
    this.hasRelativeBias = hasRelativeBias;
    this.numBuckets = config.number('relative_attention_num_buckets');
    this.maxDistance = config.number('relative_attention_max_distance');
    this.dropoutRate = config.number('dropout_rate');
    const inner = this.heads * this.headDim;
    this.q = this.registerModule('q', new Linear(model, inner, { bias: false }));
    this.k = this.registerModule('k', new Linear(model, inner, { bias: false }));
    this.v = this.registerModule('v', new Linear(model, inner, { bias: false }));
    this.o = this.registerModule('o', new Linear(inner, model, { bias: false }));
    this.relative_attention_bias = hasRelativeBias
      ? this.registerModule('relative_attention_bias', new Embedding(this.numBuckets, this.heads)) : null;
  }

  /** ``compute_bias``: ``[1, heads, queries, keys]``. */
  computeBias(queries: number, keys: number, past = 0): Tensor {
    if (!this.relative_attention_bias) return zeros([1, this.heads, queries, keys]);
    const buckets: number[] = [];
    for (let q = 0; q < queries; q += 1) {
      for (let k = 0; k < keys; k += 1) {
        buckets.push(relativePositionBucket(k - (q + past), !this.isDecoder, this.numBuckets, this.maxDistance));
      }
    }
    const values = this.relative_attention_bias.forward(tensor(buckets, { shape: [queries, keys], dtype: 'int64' }));
    return values.permute(2, 0, 1).unsqueeze(0);
  }

  forward(hidden: Tensor, options: {
    keyValue?: Tensor | null; mask?: Tensor | null; positionBias?: Tensor | null; cache?: AttentionCache | null; past?: number;
  } = {}): { output: Tensor; positionBias: Tensor } {
    const queries = hidden.shape[1]!;
    const q = splitHeads(this.q.forward(hidden), this.heads);
    const cross = options.keyValue !== undefined && options.keyValue !== null;
    const source = cross ? options.keyValue! : hidden;
    let k: Tensor;
    let v: Tensor;
    const cache = options.cache ?? null;
    if (cross && cache?.key && cache.value) {
      k = cache.key;
      v = cache.value;
    } else {
      k = splitHeads(this.k.forward(source), this.heads);
      v = splitHeads(this.v.forward(source), this.heads);
      if (cache) {
        if (!cross && cache.key && cache.value) {
          k = cat([cache.key, k], 2);
          v = cat([cache.value, v], 2);
        }
        cache.key = k;
        cache.value = v;
      }
    }
    let positionBias = options.positionBias ?? null;
    if (positionBias === null) {
      positionBias = this.hasRelativeBias ? this.computeBias(queries, k.shape[2]!, options.past ?? 0) : zeros([1, this.heads, queries, k.shape[2]!]);
    }
    const bias = combineBias(positionBias, options.mask);
    const output = attention(q, k, v, { scale: 1, bias, dropout: this.dropoutRate, training: this.training });
    return { output: this.o.forward(mergeHeads(output)), positionBias };
  }
}

class T5LayerSelfAttention extends Module {
  readonly SelfAttention: T5Attention;
  readonly layer_norm: T5LayerNorm;
  readonly dropout: Dropout;

  constructor(config: NativeConfig, hasRelativeBias: boolean, isDecoder: boolean) {
    super();
    this.SelfAttention = this.registerModule('SelfAttention', new T5Attention(config, hasRelativeBias, isDecoder));
    this.layer_norm = this.registerModule('layer_norm', new T5LayerNorm(config.number('d_model'), config.number('layer_norm_epsilon')));
    this.dropout = this.registerModule('dropout', new Dropout(config.number('dropout_rate')));
  }

  forward(hidden: Tensor, mask: Tensor | null, positionBias: Tensor | null, cache: AttentionCache | null, past: number): { hidden: Tensor; positionBias: Tensor } {
    const result = this.SelfAttention.forward(this.layer_norm.forward(hidden), { mask, positionBias, cache, past });
    return { hidden: hidden.add(this.dropout.forward(result.output)), positionBias: result.positionBias };
  }
}

class T5LayerCrossAttention extends Module {
  readonly EncDecAttention: T5Attention;
  readonly layer_norm: T5LayerNorm;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    this.EncDecAttention = this.registerModule('EncDecAttention', new T5Attention(config, false, true));
    this.layer_norm = this.registerModule('layer_norm', new T5LayerNorm(config.number('d_model'), config.number('layer_norm_epsilon')));
    this.dropout = this.registerModule('dropout', new Dropout(config.number('dropout_rate')));
  }

  forward(hidden: Tensor, encoder: Tensor, mask: Tensor | null, positionBias: Tensor | null, cache: AttentionCache | null): { hidden: Tensor; positionBias: Tensor } {
    const result = this.EncDecAttention.forward(this.layer_norm.forward(hidden), { keyValue: encoder, mask, positionBias, cache });
    return { hidden: hidden.add(this.dropout.forward(result.output)), positionBias: result.positionBias };
  }
}

export interface LayerCache {
  self: AttentionCache;
  cross: AttentionCache;
}

class T5Block extends Module {
  readonly layer: ModuleList;
  readonly isDecoder: boolean;

  constructor(config: NativeConfig, hasRelativeBias: boolean, isDecoder: boolean) {
    super();
    this.isDecoder = isDecoder;
    const layers: Module[] = [new T5LayerSelfAttention(config, hasRelativeBias, isDecoder)];
    if (isDecoder) layers.push(new T5LayerCrossAttention(config));
    layers.push(new T5LayerFF(config));
    this.layer = this.registerModule('layer', new ModuleList(layers));
  }
}

export interface StackState {
  hidden: Tensor;
}

/** ``T5Stack`` (encoder or decoder); ``embed_tokens`` is tied to ``shared`` when embeddings are tied. */
export class T5Stack extends Module {
  readonly embed_tokens: Embedding;
  readonly block: ModuleList<T5Block>;
  readonly final_layer_norm: T5LayerNorm;
  readonly dropout: Dropout;
  readonly isDecoder: boolean;
  readonly config: NativeConfig;

  constructor(config: NativeConfig, isDecoder: boolean) {
    super();
    this.config = config;
    this.isDecoder = isDecoder;
    const layers = isDecoder ? config.number('num_decoder_layers') : config.number('num_layers');
    this.embed_tokens = this.registerModule('embed_tokens', new Embedding(config.number('vocab_size'), config.number('d_model')));
    this.block = this.registerModule('block', new ModuleList(Array.from({ length: layers }, (_, index) => new T5Block(config, index === 0, isDecoder))));
    this.final_layer_norm = this.registerModule('final_layer_norm', new T5LayerNorm(config.number('d_model'), config.number('layer_norm_epsilon')));
    this.dropout = this.registerModule('dropout', new Dropout(config.number('dropout_rate')));
    // ``T5Stack`` is a ``T5PreTrainedModel``: its ``post_init`` runs as it is constructed.
    postInit(this, t5InitWeights(config));
  }

  getInputEmbeddings(): Embedding {
    return this.embed_tokens;
  }

  /** Encoder (or uncached decoder) pass returning final hidden states. */
  run(options: {
    inputIds?: Tensor | null; inputsEmbeds?: Tensor | null; attentionMask?: Tensor | null;
    encoderHidden?: Tensor | null; encoderMask?: Tensor | null; cache?: LayerCache[] | null; past?: number;
  }): Tensor {
    if ((options.inputIds == null) === (options.inputsEmbeds == null)) throw new ValueError('You must specify exactly one of input_ids or inputs_embeds');
    const embeds = options.inputsEmbeds ?? this.embed_tokens.forward(options.inputIds!);
    const queries = embeds.shape[1]!;
    const past = options.past ?? 0;
    const selfMask = this.isDecoder
      ? combineBias(causalBias(queries, past + queries, embeds.dtype), keyPaddingBias(options.attentionMask, embeds.dtype))
      : keyPaddingBias(options.attentionMask, embeds.dtype);
    const crossMask = this.isDecoder ? keyPaddingBias(options.encoderMask, embeds.dtype) : null;
    let hidden = this.dropout.forward(embeds);
    let positionBias: Tensor | null = null;
    let crossBias: Tensor | null = null;
    let index = 0;
    for (const block of this.block) {
      const cache = options.cache ? options.cache[index]! : null;
      const self = block.layer.at(0) as T5LayerSelfAttention;
      const attended = self.forward(hidden, selfMask, positionBias, cache ? cache.self : null, past);
      hidden = attended.hidden;
      positionBias = attended.positionBias;
      if (this.isDecoder) {
        if (!options.encoderHidden) throw new ValueError('the T5 decoder requires encoder hidden states');
        const cross = (block.layer.at(1) as T5LayerCrossAttention).forward(hidden, options.encoderHidden, crossMask, crossBias, cache ? cache.cross : null);
        hidden = cross.hidden;
        crossBias = cross.positionBias;
      }
      hidden = (block.layer.at(this.isDecoder ? 2 : 1) as T5LayerFF).forward(hidden);
      index += 1;
    }
    return this.dropout.forward(this.final_layer_norm.forward(hidden));
  }

  newCache(): LayerCache[] {
    return Array.from({ length: this.block.length }, () => ({ self: { key: null, value: null }, cross: { key: null, value: null } }));
  }
}

/** The T5 encoder stack exposed as a {@link NativeEncoder} (``model.get_encoder()``). */
class T5EncoderView implements Pick<NativeEncoder, 'forward' | 'getInputEmbeddings' | 'config'> {
  constructor(private readonly stack: T5Stack, readonly config: NativeConfig) {}

  getInputEmbeddings(): Embedding {
    return this.stack.embed_tokens;
  }

  forward(inputs: EncoderInputs): EncoderOutput {
    return { lastHiddenState: this.stack.run({ inputIds: inputs.inputIds ?? null, inputsEmbeds: inputs.inputsEmbeds ?? null, attentionMask: inputs.attentionMask ?? null }), poolerOutput: null };
  }
}

export interface Seq2SeqInputs extends EncoderInputs {
  decoderInputIds?: Tensor | null;
  labels?: Tensor | null;
  /** Precomputed encoder states (``encoder_outputs``); skips the encoder. */
  encoderHiddenStates?: Tensor | null;
}

export interface Seq2SeqOutput {
  logits: Tensor;
  loss: Tensor | null;
  encoderLastHiddenState: Tensor;
}

/**
 * ``T5PreTrainedModel._init_weights``: the base Hugging Face initialization
 * (``std = initializer_factor``) followed by T5's own scaled normals.
 */
function t5InitWeights(config: NativeConfig): InitWeights {
  const factor = config.number('initializer_factor');
  const dModel = config.number('d_model');
  const dFf = config.number('d_ff');
  const dKv = config.number('d_kv');
  const heads = config.number('num_heads');
  const std = initializerStd(config);
  const tied = config.get('tie_word_embeddings') !== false;
  return (module) => {
    baseInitWeights(module, std);
    if (module instanceof T5LayerNorm) {
      init.constant_(module.weight, factor * 1.0);
    } else if (module instanceof T5ForConditionalGeneration || module instanceof T5EncoderModel) {
      init.normal_(module.shared.weight, 0, factor * 1.0);
      if (module instanceof T5ForConditionalGeneration && !tied) init.normal_(module.lm_head.weight, 0, factor * 1.0);
    } else if (module instanceof T5DenseActDense) {
      init.normal_(module.wi.weight, 0, factor * dModel ** -0.5);
      init.normal_(module.wo.weight, 0, factor * dFf ** -0.5);
    } else if (module instanceof T5DenseGatedActDense) {
      init.normal_(module.wi_0.weight, 0, factor * dModel ** -0.5);
      init.normal_(module.wi_1.weight, 0, factor * dModel ** -0.5);
      init.normal_(module.wo.weight, 0, factor * dFf ** -0.5);
    } else if (module instanceof T5Attention) {
      init.normal_(module.q.weight, 0, factor * (dModel * dKv) ** -0.5);
      init.normal_(module.k.weight, 0, factor * dModel ** -0.5);
      init.normal_(module.v.weight, 0, factor * dModel ** -0.5);
      init.normal_(module.o.weight, 0, factor * (heads * dKv) ** -0.5);
      if (module.relative_attention_bias) init.normal_(module.relative_attention_bias.weight, 0, factor * dModel ** -0.5);
    }
  };
}

/** ``T5ForConditionalGeneration``. */
export class T5ForConditionalGeneration extends NativeModel {
  readonly shared: Embedding;
  readonly encoder: T5Stack;
  readonly decoder: T5Stack;
  readonly lm_head: Linear;
  readonly modelDim: number;

  constructor(config: NativeConfig) {
    super(config);
    if (config.modelType !== 't5') throw new ValueError('T5ForConditionalGeneration requires a t5 configuration');
    this.modelDim = config.number('d_model');
    this.shared = this.registerModule('shared', new Embedding(config.number('vocab_size'), this.modelDim));
    this.encoder = this.registerModule('encoder', new T5Stack(config, false));
    this.decoder = this.registerModule('decoder', new T5Stack(config, true));
    this.lm_head = this.registerModule('lm_head', new Linear(this.modelDim, config.number('vocab_size'), { bias: false }));
    postInit(this, t5InitWeights(config));
    // ``_tied_weights_keys`` apply only when word embeddings are tied.
    if (config.get('tie_word_embeddings') !== false) {
      for (const path of ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']) this.setParameterAt(path, this.shared.weight);
    }
  }

  getInputEmbeddings(): Embedding {
    return this.shared;
  }

  /** ``model.get_encoder()``. */
  getEncoder(): NativeEncoder {
    return new T5EncoderView(this.encoder, this.config) as unknown as NativeEncoder;
  }

  /** ``_shift_right``: prepend ``decoder_start_token_id``; ``-100`` becomes ``pad_token_id``. */
  shiftRight(labels: Tensor): Tensor {
    const start = this.config.optionalNumber('decoder_start_token_id');
    const pad = this.config.optionalNumber('pad_token_id');
    if (start === null) throw new ValueError('self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id.');
    if (pad === null) throw new ValueError('self.model.config.pad_token_id has to be defined.');
    const [batch, length] = labels.shape as [number, number];
    const values = new Array<number>(batch * length);
    for (let b = 0; b < batch; b += 1) {
      values[b * length] = start;
      for (let i = 1; i < length; i += 1) {
        const value = labels.data[b * length + i - 1]!;
        values[b * length + i] = value === -100 ? pad : value;
      }
    }
    return tensor(values, { shape: [batch, length], dtype: 'int64' });
  }

  /** Decoder hidden states → vocabulary logits (with T5 output scaling when configured). */
  project(sequence: Tensor): Tensor {
    const scaled = this.config.get('scale_decoder_outputs') === true ? sequence.mul(this.modelDim ** -0.5) : sequence;
    return this.lm_head.forward(scaled);
  }

  encode(inputs: EncoderInputs): Tensor {
    return this.encoder.run({ inputIds: inputs.inputIds ?? null, inputsEmbeds: inputs.inputsEmbeds ?? null, attentionMask: inputs.attentionMask ?? null });
  }

  forward(inputs: Seq2SeqInputs): Seq2SeqOutput {
    const encoderHidden = inputs.encoderHiddenStates ?? this.encode(inputs);
    let decoderInputIds = inputs.decoderInputIds ?? null;
    if (inputs.labels && !decoderInputIds) decoderInputIds = this.shiftRight(inputs.labels);
    if (!decoderInputIds) throw new ValueError('decoder_input_ids or labels are required');
    const sequence = this.decoder.run({ inputIds: decoderInputIds, encoderHidden, encoderMask: inputs.attentionMask ?? null });
    const logits = this.project(sequence);
    let loss: Tensor | null = null;
    if (inputs.labels) {
      const vocab = logits.shape[2]!;
      loss = crossEntropy(logits.reshape(-1, vocab), inputs.labels.reshape(-1), { ignoreIndex: -100 });
    }
    return { logits, loss, encoderLastHiddenState: encoderHidden };
  }

  /** One cached decoder step: logits ``[batch, steps, vocab]`` for ``decoderInputIds`` after ``past`` cached tokens. */
  decodeStep(decoderInputIds: Tensor, encoderHidden: Tensor, encoderMask: Tensor | null, cache: LayerCache[], past: number): Tensor {
    const sequence = this.decoder.run({ inputIds: decoderInputIds, encoderHidden, encoderMask, cache, past });
    return this.project(sequence);
  }

  newCache(): LayerCache[] {
    return this.decoder.newCache();
  }
}

/** ``T5EncoderModel`` (shared embedding + encoder only). */
export class T5EncoderModel extends NativeModel implements NativeEncoder {
  readonly shared: Embedding;
  readonly encoder: T5Stack;

  constructor(config: NativeConfig) {
    super(config);
    this.shared = this.registerModule('shared', new Embedding(config.number('vocab_size'), config.number('d_model')));
    this.encoder = this.registerModule('encoder', new T5Stack(config, false));
    postInit(this, t5InitWeights(config));
    if (config.get('tie_word_embeddings') !== false) this.setParameterAt('encoder.embed_tokens.weight', this.shared.weight);
  }

  getInputEmbeddings(): Embedding {
    return this.shared;
  }

  forward(inputs: EncoderInputs): EncoderOutput {
    return { lastHiddenState: this.encoder.run({ inputIds: inputs.inputIds ?? null, inputsEmbeds: inputs.inputsEmbeds ?? null, attentionMask: inputs.attentionMask ?? null }), poolerOutput: null };
  }
}
