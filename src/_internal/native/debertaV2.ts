/**
 * DeBERTa-v2/v3 (``model_type='deberta-v2'``) with transformers 5.17 parameter
 * names and numerics: ``DebertaV2Model`` and
 * ``DebertaV2ForSequenceClassification`` (NLI verifiers such as
 * ``cross-encoder/nli-deberta-v3-small``).
 *
 * Implements disentangled attention (content→position ``c2p`` and
 * position→content ``p2c`` terms, ``share_att_key``), log-bucketed relative
 * positions (``position_buckets``/``max_relative_positions``), relative
 * embedding layer norm (``norm_rel_ebd``), the optional first-layer
 * convolution (``conv_kernel_size``), the embedding variants
 * (``embedding_size`` projection, ``position_biased_input``, token types) and
 * the ``ContextPooler`` classification head. ``registry.ts`` routes
 * ``deberta-v2`` here.
 */
import { activationModule, type ActivationModule } from './activations.js';
import { Module } from '../../nn/module.js';
import { Parameter, Tensor, tensor, zeros } from '../../nn/tensor.js';
import { Dropout, Embedding, LayerNorm, Linear, ModuleList } from '../../nn/layers.js';
import { activation } from '../../nn/ops/nn.js';
import { cat } from '../../nn/ops/shape.js';
import { finfoMin } from '../../nn/dtype.js';
import * as init from '../../nn/init.js';
import { noGrad } from '../../nn/autograd.js';
import { ValueError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import { baseInitWeights, initializerStd, postInit, type InitWeights } from './hfInit.js';
import {
  NativeModel, registerPositionBuffers, type EncoderInputs, type EncoderOutput, type NativeEncoder,
} from './modules.js';
import type { ClassifierOutput, NativeSequenceClassifier } from './bert.js';

function num(config: NativeConfig, key: string, fallback?: number): number {
  const value = config.get(key);
  if (typeof value === 'number') return value;
  if (fallback !== undefined && (value === undefined || value === null)) return fallback;
  return config.number(key);
}

function positionAttentionTypes(config: NativeConfig): string[] {
  const value = config.get('pos_att_type');
  if (value === null || value === undefined) return [];
  if (typeof value === 'string') return value.toLowerCase().split('|').map((item) => item.trim());
  if (Array.isArray(value)) return value.map((item) => String(item));
  throw new ValueError('pos_att_type must be a list or a "|"-separated string');
}

// ---------------------------------------------------------------------------
// Relative positions.
// ---------------------------------------------------------------------------

const f32 = Math.fround;

/** transformers ``make_log_bucket_position`` for one relative distance (float32 arithmetic). */
export function logBucketPosition(relative: number, bucketSize: number, maxPosition: number): number {
  const mid = Math.floor(bucketSize / 2);
  const sign = Math.sign(relative);
  const absolute = relative < mid && relative > -mid ? mid - 1 : Math.abs(relative);
  if (absolute <= mid) return relative;
  const numerator = f32(Math.log(f32(absolute / mid)));
  const denominator = f32(Math.log(f32((maxPosition - 1) / mid)));
  const logPosition = f32(Math.ceil(f32(f32(numerator / denominator) * (mid - 1))) + mid);
  return Math.trunc(logPosition * sign);
}

/** ``build_relative_position``: ``[query, key]`` relative distances ``q - k`` (optionally log-bucketed). */
export function buildRelativePosition(querySize: number, keySize: number, bucketSize = -1, maxPosition = -1): Int32Array {
  const result = new Int32Array(querySize * keySize);
  for (let q = 0; q < querySize; q += 1) {
    for (let k = 0; k < keySize; k += 1) {
      const relative = q - k;
      result[q * keySize + k] = bucketSize > 0 && maxPosition > 0 ? logBucketPosition(relative, bucketSize, maxPosition) : relative;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Layers.
// ---------------------------------------------------------------------------

/** ``torch.nn.Conv1d`` over ``[batch, channels, length]`` (zero padding, stride 1, grouped). */
export class Conv1d extends Module {
  static override readonly qualifiedName: string = 'torch.nn.modules.conv.Conv1d';
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: number;
  readonly padding: number;
  readonly groups: number;
  weight: Parameter;
  bias: Parameter | null;

  constructor(inChannels: number, outChannels: number, kernelSize: number, options: { padding?: number; groups?: number; bias?: boolean } = {}) {
    super();
    const groups = options.groups ?? 1;
    if (inChannels % groups || outChannels % groups) throw new ValueError('Conv1d channels must be divisible by groups');
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.padding = options.padding ?? 0;
    this.groups = groups;
    this.weight = this.registerParameter('weight', new Parameter(zeros([outChannels, inChannels / groups, kernelSize])));
    this.bias = this.registerParameter('bias', options.bias === false ? null : new Parameter(zeros([outChannels])));
    noGrad(() => {
      init.kaimingUniform_(this.weight, Math.sqrt(5));
      if (this.bias) {
        const bound = 1 / Math.sqrt((inChannels / groups) * kernelSize);
        init.uniform_(this.bias, -bound, bound);
      }
    });
  }

  protected override onRegistryChange(): void {
    this.weight = (this.getParameter('weight') ?? this.weight) as Parameter;
    this.bias = this.getParameter('bias');
  }

  override configurationAttributes(): Record<string, unknown> {
    return {
      in_channels: this.inChannels, out_channels: this.outChannels, kernel_size: [this.kernelSize], stride: [1],
      padding: [this.padding], dilation: [1], transposed: false, output_padding: [0], groups: this.groups, padding_mode: 'zeros',
    };
  }

  forward(input: Tensor): Tensor {
    const [batch, channels, length] = input.shape as [number, number, number];
    if (channels !== this.inChannels) throw new RangeError('Conv1d input channels differ from the layer');
    const pad = this.padding;
    const padded = pad ? cat([zeros([batch, channels, pad], { dtype: input.dtype }), input, zeros([batch, channels, pad], { dtype: input.dtype })], 2) : input;
    const outLength = length + 2 * pad - this.kernelSize + 1;
    const inGroup = channels / this.groups;
    const outGroup = this.outChannels / this.groups;
    const outputs: Tensor[] = [];
    for (let group = 0; group < this.groups; group += 1) {
      const source = this.groups === 1 ? padded : padded.slice(1, group * inGroup, (group + 1) * inGroup);
      const weights = this.groups === 1 ? this.weight : this.weight.slice(0, group * outGroup, (group + 1) * outGroup);
      let total: Tensor | null = null;
      for (let offset = 0; offset < this.kernelSize; offset += 1) {
        const term = weights.select(2, offset).matmul(source.slice(2, offset, offset + outLength));
        total = total ? total.add(term) : term;
      }
      outputs.push(total!);
    }
    const result = outputs.length === 1 ? outputs[0]! : cat(outputs, 1);
    return this.bias ? result.add(this.bias.reshape(1, this.outChannels, 1)) : result;
  }
}

class DebertaV2SelfOutput extends Module {
  readonly dense: Linear;
  readonly LayerNorm: LayerNorm;
  readonly dropout: Dropout;

  constructor(config: NativeConfig, inWidth: number) {
    super();
    const hidden = num(config, 'hidden_size');
    this.dense = this.registerModule('dense', new Linear(inWidth, hidden));
    this.LayerNorm = this.registerModule('LayerNorm', new LayerNorm(hidden, { eps: num(config, 'layer_norm_eps') }));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'hidden_dropout_prob')));
  }

  forward(hidden: Tensor, input: Tensor): Tensor {
    return this.LayerNorm.forward(this.dropout.forward(this.dense.forward(hidden)).add(input));
  }
}

/** Disentangled self-attention (``DisentangledSelfAttention``). */
class DisentangledSelfAttention extends Module {
  readonly heads: number;
  readonly headSize: number;
  readonly query_proj: Linear;
  readonly key_proj: Linear;
  readonly value_proj: Linear;
  readonly shareAttKey: boolean;
  readonly posAttType: string[];
  readonly relativeAttention: boolean;
  readonly positionBuckets: number = -1;
  readonly maxRelativePositions: number = -1;
  readonly posEbdSize: number = 0;
  readonly pos_dropout: Dropout | null = null;
  readonly pos_key_proj: Linear | null = null;
  readonly pos_query_proj: Linear | null = null;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    const hidden = num(config, 'hidden_size');
    this.heads = num(config, 'num_attention_heads');
    if (hidden % this.heads !== 0) {
      throw new ValueError(`The hidden size (${hidden}) is not a multiple of the number of attention heads (${this.heads})`);
    }
    this.headSize = num(config, 'attention_head_size', Math.floor(hidden / this.heads));
    const allHead = this.heads * this.headSize;
    this.query_proj = this.registerModule('query_proj', new Linear(hidden, allHead));
    this.key_proj = this.registerModule('key_proj', new Linear(hidden, allHead));
    this.value_proj = this.registerModule('value_proj', new Linear(hidden, allHead));
    this.shareAttKey = config.get('share_att_key') === true;
    this.posAttType = positionAttentionTypes(config);
    this.relativeAttention = config.get('relative_attention') === true;
    if (this.relativeAttention) {
      this.positionBuckets = num(config, 'position_buckets', -1);
      let maxRelative = num(config, 'max_relative_positions', -1);
      if (maxRelative < 1) maxRelative = num(config, 'max_position_embeddings');
      this.maxRelativePositions = maxRelative;
      this.posEbdSize = this.positionBuckets > 0 ? this.positionBuckets : maxRelative;
      this.pos_dropout = this.registerModule('pos_dropout', new Dropout(num(config, 'hidden_dropout_prob')));
      if (!this.shareAttKey) {
        if (this.posAttType.includes('c2p')) this.pos_key_proj = this.registerModule('pos_key_proj', new Linear(hidden, allHead));
        if (this.posAttType.includes('p2c')) this.pos_query_proj = this.registerModule('pos_query_proj', new Linear(hidden, allHead));
      }
    }
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'attention_probs_dropout_prob')));
  }

  /** ``[batch, length, heads * d]`` → ``[batch * heads, length, d]``. */
  private transposeForScores(x: Tensor): Tensor {
    const [batch, length] = x.shape as [number, number];
    return x.reshape(batch, length, this.heads, -1).permute(0, 2, 1, 3).reshape(batch * this.heads, length, -1);
  }

  forward(hidden: Tensor, mask: Tensor, relativePos: Int32Array | null, relEmbeddings: Tensor | null): Tensor {
    const [batch, length] = hidden.shape as [number, number];
    const query = this.transposeForScores(this.query_proj.forward(hidden));
    const key = this.transposeForScores(this.key_proj.forward(hidden));
    const value = this.transposeForScores(this.value_proj.forward(hidden));
    let scaleFactor = 1;
    if (this.posAttType.includes('c2p')) scaleFactor += 1;
    if (this.posAttType.includes('p2c')) scaleFactor += 1;
    const scale = f32(Math.sqrt(f32(query.shape[2]! * scaleFactor)));
    let scores = query.matmul(key.transpose(-1, -2).div(scale));
    if (this.relativeAttention) {
      const bias = this.disentangledBias(query, key, relativePos!, this.pos_dropout!.forward(relEmbeddings!), scaleFactor, batch);
      if (bias) scores = scores.add(bias);
    }
    const shaped = scores.reshape(batch, this.heads, length, length).maskedFill(mask.logicalNot(), finfoMin(query.dtype));
    const probabilities = this.dropout.forward(shaped.softmax(-1));
    const context = probabilities.reshape(batch * this.heads, length, length).matmul(value);
    return context.reshape(batch, this.heads, length, -1).permute(0, 2, 1, 3).reshape(batch, length, -1);
  }

  private disentangledBias(query: Tensor, key: Tensor, relativePos: Int32Array, relEmbeddings: Tensor, scaleFactor: number, batch: number): Tensor | null {
    const span = this.posEbdSize;
    const length = query.shape[1]!;
    const keys = key.shape[1]!;
    const embeddings = relEmbeddings.slice(0, 0, span * 2).unsqueeze(0);
    const project = (layer: Linear): Tensor => this.transposeForScores(layer.forward(embeddings)).repeat(batch, 1, 1);
    let positionKey: Tensor | null = null;
    let positionQuery: Tensor | null = null;
    if (this.shareAttKey) {
      positionQuery = project(this.query_proj);
      positionKey = project(this.key_proj);
    } else {
      if (this.posAttType.includes('c2p')) positionKey = project(this.pos_key_proj!);
      if (this.posAttType.includes('p2c')) positionQuery = project(this.pos_query_proj!);
    }
    const rows = query.shape[0]!;
    const indexTensor = (compute: (relative: number) => number, width: number): Tensor => {
      const values = new Float64Array(rows * length * width);
      const plane = new Float64Array(length * width);
      for (let index = 0; index < plane.length; index += 1) plane[index] = compute(relativePos[index]!);
      for (let row = 0; row < rows; row += 1) values.set(plane, row * plane.length);
      return tensor(values, { dtype: 'int64', shape: [rows, length, width] });
    };
    const clamp = (value: number): number => Math.min(Math.max(value, 0), span * 2 - 1);
    let score: Tensor | null = null;
    if (this.posAttType.includes('c2p')) {
      const scale = f32(Math.sqrt(f32(positionKey!.shape[2]! * scaleFactor)));
      const c2p = query.matmul(positionKey!.transpose(-1, -2)).gather(-1, indexTensor((relative) => clamp(relative + span), keys));
      score = c2p.div(scale);
    }
    if (this.posAttType.includes('p2c')) {
      if (keys !== length) throw new ValueError('DeBERTa p2c attention requires equal query and key lengths');
      const scale = f32(Math.sqrt(f32(positionQuery!.shape[2]! * scaleFactor)));
      const p2c = key.matmul(positionQuery!.transpose(-1, -2))
        .gather(-1, indexTensor((relative) => clamp(-relative + span), keys)).transpose(-1, -2);
      score = score ? score.add(p2c.div(scale)) : p2c.div(scale);
    }
    return score;
  }
}

class DebertaV2Attention extends Module {
  readonly self: DisentangledSelfAttention;
  readonly output: DebertaV2SelfOutput;

  constructor(config: NativeConfig) {
    super();
    this.self = this.registerModule('self', new DisentangledSelfAttention(config));
    this.output = this.registerModule('output', new DebertaV2SelfOutput(config, num(config, 'hidden_size')));
  }

  forward(hidden: Tensor, mask: Tensor, relativePos: Int32Array | null, relEmbeddings: Tensor | null): Tensor {
    return this.output.forward(this.self.forward(hidden, mask, relativePos, relEmbeddings), hidden);
  }
}

class DebertaV2Intermediate extends Module {
  readonly dense: Linear;
  readonly intermediate_act_fn: ActivationModule;

  constructor(config: NativeConfig) {
    super();
    this.dense = this.registerModule('dense', new Linear(num(config, 'hidden_size'), num(config, 'intermediate_size')));
    this.intermediate_act_fn = this.registerModule('intermediate_act_fn', activationModule(config.string('hidden_act')));
  }

  forward(hidden: Tensor): Tensor {
    return this.intermediate_act_fn.forward(this.dense.forward(hidden));
  }
}

class DebertaV2Layer extends Module {
  readonly attention: DebertaV2Attention;
  readonly intermediate: DebertaV2Intermediate;
  readonly output: DebertaV2SelfOutput;

  constructor(config: NativeConfig) {
    super();
    this.attention = this.registerModule('attention', new DebertaV2Attention(config));
    this.intermediate = this.registerModule('intermediate', new DebertaV2Intermediate(config));
    this.output = this.registerModule('output', new DebertaV2SelfOutput(config, num(config, 'intermediate_size')));
  }

  forward(hidden: Tensor, mask: Tensor, relativePos: Int32Array | null, relEmbeddings: Tensor | null): Tensor {
    const attended = this.attention.forward(hidden, mask, relativePos, relEmbeddings);
    return this.output.forward(this.intermediate.forward(attended), attended);
  }
}

/** First-layer convolution branch (``ConvLayer``). */
class ConvLayer extends Module {
  readonly conv: Conv1d;
  readonly LayerNorm: LayerNorm;
  readonly dropout: Dropout;
  private readonly act: (x: Tensor) => Tensor;

  constructor(config: NativeConfig) {
    super();
    const hidden = num(config, 'hidden_size');
    const kernel = num(config, 'conv_kernel_size', 3);
    this.conv = this.registerModule('conv', new Conv1d(hidden, hidden, kernel, {
      padding: Math.floor((kernel - 1) / 2), groups: num(config, 'conv_groups', 1),
    }));
    this.LayerNorm = this.registerModule('LayerNorm', new LayerNorm(hidden, { eps: num(config, 'layer_norm_eps') }));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'hidden_dropout_prob')));
    const act = config.get('conv_act');
    this.act = activation(typeof act === 'string' ? act : 'tanh');
  }

  forward(hidden: Tensor, residual: Tensor, inputMask: Tensor): Tensor {
    const invalid = inputMask.eq(0).unsqueeze(-1);
    const convolved = this.conv.forward(hidden.permute(0, 2, 1)).permute(0, 2, 1).maskedFill(invalid, 0);
    const output = this.LayerNorm.forward(residual.add(this.act(this.dropout.forward(convolved))));
    return output.mul(inputMask.to(output.dtype).unsqueeze(-1));
  }
}

class DebertaV2Embeddings extends Module {
  readonly embeddingSize: number;
  readonly positionBiasedInput: boolean;
  readonly word_embeddings: Embedding;
  readonly position_embeddings: Embedding | null;
  readonly token_type_embeddings: Embedding | null;
  readonly embed_proj: Linear | null;
  readonly LayerNorm: LayerNorm;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    const hidden = num(config, 'hidden_size');
    this.embeddingSize = num(config, 'embedding_size', hidden);
    const pad = config.get('pad_token_id');
    this.word_embeddings = this.registerModule('word_embeddings', new Embedding(num(config, 'vocab_size'), this.embeddingSize, {
      paddingIdx: typeof pad === 'number' ? pad : pad === null ? null : 0,
    }));
    this.positionBiasedInput = config.get('position_biased_input') !== false;
    this.position_embeddings = this.positionBiasedInput
      ? this.registerModule('position_embeddings', new Embedding(num(config, 'max_position_embeddings'), this.embeddingSize)) : null;
    const types = num(config, 'type_vocab_size');
    this.token_type_embeddings = types > 0 ? this.registerModule('token_type_embeddings', new Embedding(types, this.embeddingSize)) : null;
    this.embed_proj = this.embeddingSize !== hidden
      ? this.registerModule('embed_proj', new Linear(this.embeddingSize, hidden, { bias: false })) : null;
    this.LayerNorm = this.registerModule('LayerNorm', new LayerNorm(hidden, { eps: num(config, 'layer_norm_eps') }));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'hidden_dropout_prob')));
    registerPositionBuffers(this, num(config, 'max_position_embeddings'), false);
  }

  forward(inputs: EncoderInputs, mask: Tensor): Tensor {
    const source = inputs.inputIds ?? inputs.inputsEmbeds!;
    const [batch, length] = source.shape as [number, number];
    let embeddings = inputs.inputsEmbeds ?? this.word_embeddings.forward(inputs.inputIds!);
    if (this.position_embeddings) {
      const positions = inputs.positionIds ?? tensor(Array.from({ length }, (_, index) => index), { dtype: 'int64' }).unsqueeze(0);
      embeddings = embeddings.add(this.position_embeddings.forward(positions));
    }
    if (this.token_type_embeddings) {
      const types = inputs.tokenTypeIds ?? zeros([batch, length], { dtype: 'int64' });
      embeddings = embeddings.add(this.token_type_embeddings.forward(types));
    }
    if (this.embed_proj) embeddings = this.embed_proj.forward(embeddings);
    embeddings = this.LayerNorm.forward(embeddings).mul(mask.to(embeddings.dtype).unsqueeze(2));
    return this.dropout.forward(embeddings);
  }
}

class DebertaV2Encoder extends Module {
  readonly layer: ModuleList<DebertaV2Layer>;
  readonly relativeAttention: boolean;
  readonly maxRelativePositions: number = -1;
  readonly positionBuckets: number = -1;
  readonly rel_embeddings: Embedding | null = null;
  readonly normRelEbd: string[];
  readonly LayerNorm: LayerNorm | null = null;
  readonly conv: ConvLayer | null;

  constructor(config: NativeConfig) {
    super();
    this.layer = this.registerModule('layer', new ModuleList(
      Array.from({ length: num(config, 'num_hidden_layers') }, () => new DebertaV2Layer(config)),
    ));
    this.relativeAttention = config.get('relative_attention') === true;
    const hidden = num(config, 'hidden_size');
    if (this.relativeAttention) {
      let maxRelative = num(config, 'max_relative_positions', -1);
      if (maxRelative < 1) maxRelative = num(config, 'max_position_embeddings');
      this.maxRelativePositions = maxRelative;
      this.positionBuckets = num(config, 'position_buckets', -1);
      const size = this.positionBuckets > 0 ? this.positionBuckets * 2 : maxRelative * 2;
      this.rel_embeddings = this.registerModule('rel_embeddings', new Embedding(size, hidden));
    }
    const norm = config.get('norm_rel_ebd');
    this.normRelEbd = (typeof norm === 'string' ? norm : 'none').toLowerCase().split('|').map((item) => item.trim());
    if (this.normRelEbd.includes('layer_norm')) {
      this.LayerNorm = this.registerModule('LayerNorm', new LayerNorm(hidden, { eps: num(config, 'layer_norm_eps') }));
    }
    this.conv = num(config, 'conv_kernel_size', 0) > 0 ? this.registerModule('conv', new ConvLayer(config)) : null;
  }

  relativeEmbeddings(): Tensor | null {
    if (!this.rel_embeddings) return null;
    const weight = this.rel_embeddings.weight;
    return this.LayerNorm ? this.LayerNorm.forward(weight) : weight;
  }

  forward(hidden: Tensor, attentionMask: Tensor): Tensor {
    const [batch, length] = attentionMask.shape as [number, number];
    const valid = attentionMask.ne(0);
    const mask = valid.reshape(batch, 1, 1, length).logicalAnd(valid.reshape(batch, 1, length, 1));
    const relativePos = this.relativeAttention
      ? buildRelativePosition(length, length, this.positionBuckets, this.maxRelativePositions) : null;
    const relEmbeddings = this.relativeEmbeddings();
    let state = hidden;
    let index = 0;
    for (const layer of this.layer) {
      let output = layer.forward(state, mask, relativePos, relEmbeddings);
      if (index === 0 && this.conv) output = this.conv.forward(hidden, output, attentionMask);
      state = output;
      index += 1;
    }
    return state;
  }
}

/** ``DebertaV2Model`` (embeddings + relative-attention encoder; no pooler). */
export class DebertaV2Model extends NativeModel implements NativeEncoder {
  readonly embeddings: DebertaV2Embeddings;
  readonly encoder: DebertaV2Encoder;

  constructor(config: NativeConfig, options: { initialize?: boolean } = {}) {
    super(config);
    this.embeddings = this.registerModule('embeddings', new DebertaV2Embeddings(config));
    this.encoder = this.registerModule('encoder', new DebertaV2Encoder(config));
    // ``DebertaV2Model`` is a pretrained model: its ``post_init`` runs as it is constructed,
    // also inside a task head (whose own ``post_init`` then skips these modules).
    if (options.initialize !== false) postInit(this, debertaInitWeights(config));
  }

  getInputEmbeddings(): Embedding {
    return this.embeddings.word_embeddings;
  }

  forward(inputs: EncoderInputs): EncoderOutput {
    if ((inputs.inputIds == null) === (inputs.inputsEmbeds == null)) {
      throw new ValueError('You must specify exactly one of input_ids or inputs_embeds');
    }
    const source = inputs.inputIds ?? inputs.inputsEmbeds!;
    const [batch, length] = source.shape as [number, number];
    const mask = inputs.attentionMask ?? tensor(new Float64Array(batch * length).fill(1), { dtype: 'int64', shape: [batch, length] });
    const embedded = this.embeddings.forward(inputs, mask);
    return { lastHiddenState: this.encoder.forward(embedded, mask), poolerOutput: null };
  }
}

/** ``DebertaV2PreTrainedModel._init_weights``: the base Hugging Face initialization (``nn.Conv1d`` included). */
function debertaInitWeights(config: NativeConfig): InitWeights {
  const std = initializerStd(config);
  return (module) => {
    baseInitWeights(module, std);
    if (module instanceof Conv1d) {
      init.normal_(module.weight, 0, std);
      if (module.bias) init.zeros_(module.bias);
    }
  };
}

/** ``ContextPooler``: first token → dropout → dense → activation. */
class ContextPooler extends Module {
  readonly dense: Linear;
  readonly dropout: Dropout;
  private readonly act: (x: Tensor) => Tensor;

  constructor(config: NativeConfig) {
    super();
    const width = num(config, 'pooler_hidden_size', num(config, 'hidden_size'));
    this.dense = this.registerModule('dense', new Linear(width, width));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'pooler_dropout', 0)));
    this.act = activation(typeof config.get('pooler_hidden_act') === 'string' ? config.string('pooler_hidden_act') : 'gelu');
  }

  forward(hidden: Tensor): Tensor {
    return this.act(this.dense.forward(this.dropout.forward(hidden.select(1, 0))));
  }
}

/** ``DebertaV2ForSequenceClassification`` (``ContextPooler`` → dropout → classifier). */
export class DebertaV2ForSequenceClassification extends NativeModel implements NativeSequenceClassifier {
  readonly deberta: DebertaV2Model;
  readonly pooler: ContextPooler;
  readonly classifier: Linear;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super(config);
    this.deberta = this.registerModule('deberta', new DebertaV2Model(config));
    this.pooler = this.registerModule('pooler', new ContextPooler(config));
    this.classifier = this.registerModule('classifier', new Linear(num(config, 'hidden_size'), config.numLabels));
    const clsDropout = config.get('cls_dropout');
    this.dropout = this.registerModule('dropout', new Dropout(typeof clsDropout === 'number' ? clsDropout : num(config, 'hidden_dropout_prob')));
    postInit(this, debertaInitWeights(config));
  }

  get base(): NativeEncoder {
    return this.deberta;
  }

  getInputEmbeddings(): Embedding {
    return this.deberta.getInputEmbeddings();
  }

  forward(inputs: EncoderInputs): ClassifierOutput {
    const hidden = this.deberta.forward(inputs).lastHiddenState;
    return { logits: this.classifier.forward(this.dropout.forward(this.pooler.forward(hidden))), lastHiddenState: hidden };
  }
}

export function createDebertaV2Model(config: NativeConfig): NativeModel {
  return new DebertaV2Model(config);
}

export function createDebertaV2ForSequenceClassification(config: NativeConfig): NativeModel {
  return new DebertaV2ForSequenceClassification(config);
}
