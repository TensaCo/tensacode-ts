/**
 * BERT-family encoders with transformers 5.17 parameter names: BERT, RoBERTa,
 * Electra and DistilBERT, plus their ``*ForSequenceClassification`` heads.
 */
import { activationModule, type ActivationModule } from './activations.js';
import { Module } from '../../nn/module.js';
import { Tensor, tensor } from '../../nn/tensor.js';
import { Dropout, Embedding, LayerNorm, Linear, ModuleList } from '../../nn/layers.js';
import { activation } from '../../nn/ops/nn.js';
import { ValueError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import {
  NativeModel, attention, initializeWeights, keyPaddingBias, mergeHeads, positionIds, registerPositionBuffers, splitHeads, zerosLong,
  type EncoderInputs, type EncoderOutput, type NativeEncoder,
} from './modules.js';

function num(config: NativeConfig, key: string): number {
  return config.number(key);
}

function inputShape(inputs: EncoderInputs): [number, number] {
  if ((inputs.inputIds == null) === (inputs.inputsEmbeds == null)) {
    throw new ValueError('You must specify exactly one of input_ids or inputs_embeds');
  }
  const source = inputs.inputIds ?? inputs.inputsEmbeds!;
  return [source.shape[0]!, source.shape[1]!];
}

// ---------------------------------------------------------------------------
// BERT layers (shared by BERT, RoBERTa and Electra).
// ---------------------------------------------------------------------------

class BertSelfAttention extends Module {
  readonly heads: number;
  readonly headDim: number;
  readonly query: Linear;
  readonly key: Linear;
  readonly value: Linear;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    const hidden = num(config, 'hidden_size');
    this.heads = num(config, 'num_attention_heads');
    if (hidden % this.heads !== 0) throw new ValueError('hidden_size must be a multiple of num_attention_heads');
    this.headDim = hidden / this.heads;
    this.query = this.registerModule('query', new Linear(hidden, hidden));
    this.key = this.registerModule('key', new Linear(hidden, hidden));
    this.value = this.registerModule('value', new Linear(hidden, hidden));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'attention_probs_dropout_prob')));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const q = splitHeads(this.query.forward(hidden), this.heads);
    const k = splitHeads(this.key.forward(hidden), this.heads);
    const v = splitHeads(this.value.forward(hidden), this.heads);
    return mergeHeads(attention(q, k, v, { scale: this.headDim ** -0.5, bias, dropout: this.dropout.p, training: this.training }));
  }
}

class BertResidualOutput extends Module {
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

  forward(hidden: Tensor, residual: Tensor): Tensor {
    return this.LayerNorm.forward(this.dropout.forward(this.dense.forward(hidden)).add(residual));
  }
}

class BertAttention extends Module {
  readonly self: BertSelfAttention;
  readonly output: BertResidualOutput;

  constructor(config: NativeConfig) {
    super();
    this.self = this.registerModule('self', new BertSelfAttention(config));
    this.output = this.registerModule('output', new BertResidualOutput(config, num(config, 'hidden_size')));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    return this.output.forward(this.self.forward(hidden, bias), hidden);
  }
}

class BertIntermediate extends Module {
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

class BertLayer extends Module {
  readonly attention: BertAttention;
  readonly intermediate: BertIntermediate;
  readonly output: BertResidualOutput;

  constructor(config: NativeConfig) {
    super();
    this.attention = this.registerModule('attention', new BertAttention(config));
    this.intermediate = this.registerModule('intermediate', new BertIntermediate(config));
    this.output = this.registerModule('output', new BertResidualOutput(config, num(config, 'intermediate_size')));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const attended = this.attention.forward(hidden, bias);
    return this.output.forward(this.intermediate.forward(attended), attended);
  }
}

export class BertEncoder extends Module {
  readonly layer: ModuleList<BertLayer>;

  constructor(config: NativeConfig) {
    super();
    this.layer = this.registerModule('layer', new ModuleList(
      Array.from({ length: num(config, 'num_hidden_layers') }, () => new BertLayer(config)),
    ));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    let state = hidden;
    for (const layer of this.layer) state = layer.forward(state, bias);
    return state;
  }
}

class BertPooler extends Module {
  readonly dense: Linear;
  readonly activation: ActivationModule;

  constructor(config: NativeConfig) {
    super();
    const hidden = num(config, 'hidden_size');
    this.dense = this.registerModule('dense', new Linear(hidden, hidden));
    this.activation = this.registerModule('activation', activationModule('tanh'));
  }

  forward(hidden: Tensor): Tensor {
    return this.activation.forward(this.dense.forward(hidden.select(1, 0)));
  }
}

// ---------------------------------------------------------------------------
// Embeddings.
// ---------------------------------------------------------------------------

export class BertEmbeddings extends Module {
  readonly word_embeddings: Embedding;
  readonly position_embeddings: Embedding;
  readonly token_type_embeddings: Embedding;
  readonly LayerNorm: LayerNorm;
  readonly dropout: Dropout;

  constructor(config: NativeConfig, width = num(config, 'hidden_size')) {
    super();
    const pad = config.optionalNumber('pad_token_id');
    this.word_embeddings = this.registerModule('word_embeddings', new Embedding(num(config, 'vocab_size'), width, { paddingIdx: pad }));
    this.position_embeddings = this.registerModule('position_embeddings', new Embedding(num(config, 'max_position_embeddings'), width));
    this.token_type_embeddings = this.registerModule('token_type_embeddings', new Embedding(num(config, 'type_vocab_size'), width));
    this.LayerNorm = this.registerModule('LayerNorm', new LayerNorm(width, { eps: num(config, 'layer_norm_eps') }));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'hidden_dropout_prob')));
    registerPositionBuffers(this, num(config, 'max_position_embeddings'), true);
  }

  forward(inputs: EncoderInputs): Tensor {
    const [batch, length] = inputShape(inputs);
    const embeds = inputs.inputsEmbeds ?? this.word_embeddings.forward(inputs.inputIds!);
    const positions = inputs.positionIds ?? positionIds(batch, length);
    const types = inputs.tokenTypeIds ?? zerosLong([batch, length]);
    const summed = embeds.add(this.token_type_embeddings.forward(types)).add(this.position_embeddings.forward(positions));
    return this.dropout.forward(this.LayerNorm.forward(summed));
  }
}

/** RoBERTa embeddings: positions start after the padding index and skip padding. */
export class RobertaEmbeddings extends Module {
  readonly word_embeddings: Embedding;
  readonly token_type_embeddings: Embedding;
  readonly LayerNorm: LayerNorm;
  readonly dropout: Dropout;
  readonly position_embeddings: Embedding;
  readonly paddingIdx: number;

  constructor(config: NativeConfig) {
    super();
    const hidden = num(config, 'hidden_size');
    this.paddingIdx = config.optionalNumber('pad_token_id') ?? 1;
    this.word_embeddings = this.registerModule('word_embeddings', new Embedding(num(config, 'vocab_size'), hidden, { paddingIdx: this.paddingIdx }));
    this.token_type_embeddings = this.registerModule('token_type_embeddings', new Embedding(num(config, 'type_vocab_size'), hidden));
    this.LayerNorm = this.registerModule('LayerNorm', new LayerNorm(hidden, { eps: num(config, 'layer_norm_eps') }));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'hidden_dropout_prob')));
    this.position_embeddings = this.registerModule('position_embeddings',
      new Embedding(num(config, 'max_position_embeddings'), hidden, { paddingIdx: this.paddingIdx }));
    registerPositionBuffers(this, num(config, 'max_position_embeddings'), true);
  }

  /** ``create_position_ids_from_input_ids`` / ``..._from_inputs_embeds``. */
  positions(inputs: EncoderInputs): Tensor {
    if (inputs.positionIds) return inputs.positionIds;
    const [batch, length] = inputShape(inputs);
    if (!inputs.inputIds) return positionIds(batch, length, this.paddingIdx + 1);
    const ids = inputs.inputIds.data;
    const values = new Array<number>(batch * length);
    for (let b = 0; b < batch; b += 1) {
      let count = 0;
      for (let i = 0; i < length; i += 1) {
        const index = b * length + i;
        if (ids[index] !== this.paddingIdx) {
          count += 1;
          values[index] = count + this.paddingIdx;
        } else values[index] = this.paddingIdx;
      }
    }
    return tensor(values, { shape: [batch, length], dtype: 'int64' });
  }

  forward(inputs: EncoderInputs): Tensor {
    const [batch, length] = inputShape(inputs);
    const embeds = inputs.inputsEmbeds ?? this.word_embeddings.forward(inputs.inputIds!);
    const types = inputs.tokenTypeIds ?? zerosLong([batch, length]);
    const summed = embeds.add(this.token_type_embeddings.forward(types)).add(this.position_embeddings.forward(this.positions(inputs)));
    return this.dropout.forward(this.LayerNorm.forward(summed));
  }
}

// ---------------------------------------------------------------------------
// Base models.
// ---------------------------------------------------------------------------

export interface BertModelOptions {
  addPoolingLayer?: boolean;
}

/** ``BertModel`` (``AutoModel`` for ``model_type='bert'``). */
export class BertModel extends NativeModel implements NativeEncoder {
  readonly embeddings: BertEmbeddings | RobertaEmbeddings;
  readonly encoder: BertEncoder;
  readonly pooler: BertPooler | null;

  constructor(config: NativeConfig, options: BertModelOptions = {}) {
    super(config);
    this.embeddings = this.registerModule('embeddings', this.createEmbeddings(config));
    this.encoder = this.registerModule('encoder', new BertEncoder(config));
    this.pooler = options.addPoolingLayer === false ? null : this.registerModule('pooler', new BertPooler(config));
    initializeWeights(this, num(config, 'initializer_range'));
  }

  protected createEmbeddings(config: NativeConfig): BertEmbeddings | RobertaEmbeddings {
    return new BertEmbeddings(config);
  }

  getInputEmbeddings(): Embedding {
    return this.embeddings.word_embeddings;
  }

  forward(inputs: EncoderInputs): EncoderOutput {
    const hidden = this.embeddings.forward(inputs);
    const lastHiddenState = this.encoder.forward(hidden, keyPaddingBias(inputs.attentionMask, hidden.dtype));
    return { lastHiddenState, poolerOutput: this.pooler ? this.pooler.forward(lastHiddenState) : null };
  }
}

/** ``RobertaModel``. */
export class RobertaModel extends BertModel {
  protected override createEmbeddings(config: NativeConfig): RobertaEmbeddings {
    return new RobertaEmbeddings(config);
  }
}

/** ``ElectraModel``: factorized embeddings projected to the hidden width; no pooler. */
export class ElectraModel extends NativeModel implements NativeEncoder {
  readonly embeddings: BertEmbeddings;
  readonly embeddings_project: Linear | null;
  readonly encoder: BertEncoder;

  constructor(config: NativeConfig) {
    super(config);
    const embedding = num(config, 'embedding_size');
    const hidden = num(config, 'hidden_size');
    this.embeddings = this.registerModule('embeddings', new BertEmbeddings(config, embedding));
    this.embeddings_project = embedding !== hidden ? this.registerModule('embeddings_project', new Linear(embedding, hidden)) : null;
    this.encoder = this.registerModule('encoder', new BertEncoder(config));
    initializeWeights(this, num(config, 'initializer_range'));
  }

  getInputEmbeddings(): Embedding {
    return this.embeddings.word_embeddings;
  }

  forward(inputs: EncoderInputs): EncoderOutput {
    let hidden = this.embeddings.forward(inputs);
    if (this.embeddings_project) hidden = this.embeddings_project.forward(hidden);
    return { lastHiddenState: this.encoder.forward(hidden, keyPaddingBias(inputs.attentionMask, hidden.dtype)), poolerOutput: null };
  }
}

// ---------------------------------------------------------------------------
// DistilBERT.
// ---------------------------------------------------------------------------

class DistilEmbeddings extends Module {
  readonly word_embeddings: Embedding;
  readonly position_embeddings: Embedding;
  readonly LayerNorm: LayerNorm;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    const dim = num(config, 'dim');
    this.word_embeddings = this.registerModule('word_embeddings', new Embedding(num(config, 'vocab_size'), dim, { paddingIdx: config.optionalNumber('pad_token_id') }));
    this.position_embeddings = this.registerModule('position_embeddings', new Embedding(num(config, 'max_position_embeddings'), dim));
    if (config.boolean('sinusoidal_pos_embds')) {
      const table = this.position_embeddings.weight;
      const positions = table.shape[0]!;
      for (let pos = 0; pos < positions; pos += 1) {
        for (let j = 0; j < dim; j += 1) {
          const angle = pos / 10000 ** ((2 * Math.floor(j / 2)) / dim);
          table.data[pos * dim + j] = j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
        }
      }
      table.requiresGrad = false;
    }
    this.LayerNorm = this.registerModule('LayerNorm', new LayerNorm(dim, { eps: 1e-12 }));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'dropout')));
    registerPositionBuffers(this, num(config, 'max_position_embeddings'), false);
  }

  forward(inputs: EncoderInputs): Tensor {
    const [batch, length] = inputShape(inputs);
    const embeds = inputs.inputsEmbeds ?? this.word_embeddings.forward(inputs.inputIds!);
    const positions = inputs.positionIds ?? positionIds(batch, length);
    return this.dropout.forward(this.LayerNorm.forward(embeds.add(this.position_embeddings.forward(positions))));
  }
}

class DistilAttention extends Module {
  readonly heads: number;
  readonly q_lin: Linear;
  readonly k_lin: Linear;
  readonly v_lin: Linear;
  readonly out_lin: Linear;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    const dim = num(config, 'dim');
    this.heads = num(config, 'n_heads');
    this.q_lin = this.registerModule('q_lin', new Linear(dim, dim));
    this.k_lin = this.registerModule('k_lin', new Linear(dim, dim));
    this.v_lin = this.registerModule('v_lin', new Linear(dim, dim));
    this.out_lin = this.registerModule('out_lin', new Linear(dim, dim));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'attention_dropout')));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const q = splitHeads(this.q_lin.forward(hidden), this.heads);
    const k = splitHeads(this.k_lin.forward(hidden), this.heads);
    const v = splitHeads(this.v_lin.forward(hidden), this.heads);
    const scale = (q.shape[3]!) ** -0.5;
    return this.out_lin.forward(mergeHeads(attention(q, k, v, { scale, bias, dropout: this.dropout.p, training: this.training })));
  }
}

class DistilFFN extends Module {
  readonly lin1: Linear;
  readonly lin2: Linear;
  readonly dropout: Dropout;
  readonly activation: ActivationModule;

  constructor(config: NativeConfig) {
    super();
    const dim = num(config, 'dim');
    // HF FFN registration order: dropout, lin1, lin2, activation.
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'dropout')));
    this.lin1 = this.registerModule('lin1', new Linear(dim, num(config, 'hidden_dim')));
    this.lin2 = this.registerModule('lin2', new Linear(num(config, 'hidden_dim'), dim));
    this.activation = this.registerModule('activation', activationModule(config.string('activation')));
  }

  forward(hidden: Tensor): Tensor {
    return this.dropout.forward(this.lin2.forward(this.activation.forward(this.lin1.forward(hidden))));
  }
}

class DistilLayer extends Module {
  readonly attention: DistilAttention;
  readonly sa_layer_norm: LayerNorm;
  readonly ffn: DistilFFN;
  readonly output_layer_norm: LayerNorm;

  constructor(config: NativeConfig) {
    super();
    const dim = num(config, 'dim');
    this.attention = this.registerModule('attention', new DistilAttention(config));
    this.sa_layer_norm = this.registerModule('sa_layer_norm', new LayerNorm(dim, { eps: 1e-12 }));
    this.ffn = this.registerModule('ffn', new DistilFFN(config));
    this.output_layer_norm = this.registerModule('output_layer_norm', new LayerNorm(dim, { eps: 1e-12 }));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const attended = this.sa_layer_norm.forward(this.attention.forward(hidden, bias).add(hidden));
    return this.output_layer_norm.forward(this.ffn.forward(attended).add(attended));
  }
}

class DistilTransformer extends Module {
  readonly layer: ModuleList<DistilLayer>;

  constructor(config: NativeConfig) {
    super();
    this.layer = this.registerModule('layer', new ModuleList(Array.from({ length: num(config, 'n_layers') }, () => new DistilLayer(config))));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    let state = hidden;
    for (const layer of this.layer) state = layer.forward(state, bias);
    return state;
  }
}

/** ``DistilBertModel``. */
export class DistilBertModel extends NativeModel implements NativeEncoder {
  readonly embeddings: DistilEmbeddings;
  readonly transformer: DistilTransformer;

  constructor(config: NativeConfig) {
    super(config);
    this.embeddings = this.registerModule('embeddings', new DistilEmbeddings(config));
    this.transformer = this.registerModule('transformer', new DistilTransformer(config));
    const sinusoidal = config.boolean('sinusoidal_pos_embds');
    const saved = sinusoidal ? this.embeddings.position_embeddings.weight.clone() : null;
    initializeWeights(this, num(config, 'initializer_range'));
    if (saved) this.embeddings.position_embeddings.weight.data.set(saved.data);
  }

  getInputEmbeddings(): Embedding {
    return this.embeddings.word_embeddings;
  }

  forward(inputs: EncoderInputs): EncoderOutput {
    const hidden = this.embeddings.forward(inputs);
    return { lastHiddenState: this.transformer.forward(hidden, keyPaddingBias(inputs.attentionMask, hidden.dtype)), poolerOutput: null };
  }
}

// ---------------------------------------------------------------------------
// Sequence classification heads.
// ---------------------------------------------------------------------------

export interface ClassifierOutput {
  logits: Tensor;
  lastHiddenState: Tensor;
}

export interface NativeSequenceClassifier extends Module {
  readonly config: NativeConfig;
  forward(inputs: EncoderInputs): ClassifierOutput;
  readonly base: NativeEncoder;
}

function classifierDropout(config: NativeConfig, fallbackKey: string): number {
  const value = config.get('classifier_dropout');
  return typeof value === 'number' ? value : num(config, fallbackKey);
}

/** ``BertForSequenceClassification`` (pooler → dropout → classifier). */
export class BertForSequenceClassification extends NativeModel implements NativeSequenceClassifier {
  readonly bert: BertModel;
  readonly dropout: Dropout;
  readonly classifier: Linear;

  constructor(config: NativeConfig) {
    super(config);
    this.bert = this.registerModule('bert', new BertModel(config));
    this.dropout = this.registerModule('dropout', new Dropout(classifierDropout(config, 'hidden_dropout_prob')));
    this.classifier = this.registerModule('classifier', new Linear(num(config, 'hidden_size'), config.numLabels));
    initializeWeights(this.classifier, num(config, 'initializer_range'));
  }

  get base(): NativeEncoder { return this.bert; }
  getInputEmbeddings(): Embedding { return this.bert.getInputEmbeddings(); }

  forward(inputs: EncoderInputs): ClassifierOutput {
    const output = this.bert.forward(inputs);
    return { logits: this.classifier.forward(this.dropout.forward(output.poolerOutput!)), lastHiddenState: output.lastHiddenState };
  }
}

/** Two-layer head over the first token (RoBERTa uses tanh, Electra gelu). */
class ClassificationHead extends Module {
  readonly dense: Linear;
  readonly dropout: Dropout;
  readonly out_proj: Linear;
  private readonly act: (x: Tensor) => Tensor;

  /**
   * ``registered`` mirrors HF: Electra's head registers ``activation`` as a
   * submodule, RoBERTa's applies ``torch.tanh`` inline.
   */
  constructor(config: NativeConfig, act: string, registered: boolean) {
    super();
    const hidden = num(config, 'hidden_size');
    this.dense = this.registerModule('dense', new Linear(hidden, hidden));
    if (registered) {
      const module = this.registerModule('activation', activationModule(act));
      this.act = (x) => module.forward(x);
    } else {
      this.act = activation(act);
    }
    this.dropout = this.registerModule('dropout', new Dropout(classifierDropout(config, 'hidden_dropout_prob')));
    this.out_proj = this.registerModule('out_proj', new Linear(hidden, config.numLabels));
  }

  forward(features: Tensor): Tensor {
    let x = this.dropout.forward(features.select(1, 0));
    x = this.act(this.dense.forward(x));
    return this.out_proj.forward(this.dropout.forward(x));
  }
}

/** ``RobertaForSequenceClassification`` (no pooler; tanh head). */
export class RobertaForSequenceClassification extends NativeModel implements NativeSequenceClassifier {
  readonly roberta: RobertaModel;
  readonly classifier: ClassificationHead;

  constructor(config: NativeConfig) {
    super(config);
    this.roberta = this.registerModule('roberta', new RobertaModel(config, { addPoolingLayer: false }));
    this.classifier = this.registerModule('classifier', new ClassificationHead(config, 'tanh', false));
    initializeWeights(this.classifier, num(config, 'initializer_range'));
  }

  get base(): NativeEncoder { return this.roberta; }
  getInputEmbeddings(): Embedding { return this.roberta.getInputEmbeddings(); }

  forward(inputs: EncoderInputs): ClassifierOutput {
    const hidden = this.roberta.forward(inputs).lastHiddenState;
    return { logits: this.classifier.forward(hidden), lastHiddenState: hidden };
  }
}

/** ``ElectraForSequenceClassification`` (gelu head over the first token). */
export class ElectraForSequenceClassification extends NativeModel implements NativeSequenceClassifier {
  readonly electra: ElectraModel;
  readonly classifier: ClassificationHead;

  constructor(config: NativeConfig) {
    super(config);
    this.electra = this.registerModule('electra', new ElectraModel(config));
    this.classifier = this.registerModule('classifier', new ClassificationHead(config, 'gelu', true));
    initializeWeights(this.classifier, num(config, 'initializer_range'));
  }

  get base(): NativeEncoder { return this.electra; }
  getInputEmbeddings(): Embedding { return this.electra.getInputEmbeddings(); }

  forward(inputs: EncoderInputs): ClassifierOutput {
    const hidden = this.electra.forward(inputs).lastHiddenState;
    return { logits: this.classifier.forward(hidden), lastHiddenState: hidden };
  }
}

/** ``DistilBertForSequenceClassification``. */
export class DistilBertForSequenceClassification extends NativeModel implements NativeSequenceClassifier {
  readonly distilbert: DistilBertModel;
  readonly pre_classifier: Linear;
  readonly classifier: Linear;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super(config);
    const dim = num(config, 'dim');
    this.distilbert = this.registerModule('distilbert', new DistilBertModel(config));
    this.pre_classifier = this.registerModule('pre_classifier', new Linear(dim, dim));
    this.classifier = this.registerModule('classifier', new Linear(dim, config.numLabels));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'seq_classif_dropout')));
    initializeWeights(this.pre_classifier, num(config, 'initializer_range'));
    initializeWeights(this.classifier, num(config, 'initializer_range'));
  }

  get base(): NativeEncoder { return this.distilbert; }
  getInputEmbeddings(): Embedding { return this.distilbert.getInputEmbeddings(); }

  forward(inputs: EncoderInputs): ClassifierOutput {
    const hidden = this.distilbert.forward(inputs).lastHiddenState;
    const pooled = this.pre_classifier.forward(hidden.select(1, 0)).relu();
    return { logits: this.classifier.forward(this.dropout.forward(pooled)), lastHiddenState: hidden };
  }
}

// ---------------------------------------------------------------------------
// ALBERT.
// ---------------------------------------------------------------------------

/** ``AlbertEmbeddings``: BERT embeddings at the factorized ``embedding_size``. */
export class AlbertEmbeddings extends BertEmbeddings {
  constructor(config: NativeConfig) {
    super(config, num(config, 'embedding_size'));
  }
}

class AlbertAttention extends Module {
  readonly heads: number;
  readonly headDim: number;
  readonly attention_dropout: Dropout;
  readonly output_dropout: Dropout;
  readonly query: Linear;
  readonly key: Linear;
  readonly value: Linear;
  readonly dense: Linear;
  readonly LayerNorm: LayerNorm;

  constructor(config: NativeConfig) {
    super();
    const hidden = num(config, 'hidden_size');
    this.heads = num(config, 'num_attention_heads');
    this.headDim = Math.floor(hidden / this.heads);
    const all = this.heads * this.headDim;
    this.attention_dropout = this.registerModule('attention_dropout', new Dropout(num(config, 'attention_probs_dropout_prob')));
    this.output_dropout = this.registerModule('output_dropout', new Dropout(num(config, 'hidden_dropout_prob')));
    this.query = this.registerModule('query', new Linear(hidden, all));
    this.key = this.registerModule('key', new Linear(hidden, all));
    this.value = this.registerModule('value', new Linear(hidden, all));
    this.dense = this.registerModule('dense', new Linear(hidden, hidden));
    this.LayerNorm = this.registerModule('LayerNorm', new LayerNorm(hidden, { eps: num(config, 'layer_norm_eps') }));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const q = splitHeads(this.query.forward(hidden), this.heads);
    const k = splitHeads(this.key.forward(hidden), this.heads);
    const v = splitHeads(this.value.forward(hidden), this.heads);
    const attended = mergeHeads(attention(q, k, v, {
      scale: this.headDim ** -0.5, bias, dropout: this.attention_dropout.p, training: this.training,
    }));
    return this.LayerNorm.forward(hidden.add(this.output_dropout.forward(this.dense.forward(attended))));
  }
}

class AlbertLayer extends Module {
  readonly full_layer_layer_norm: LayerNorm;
  readonly attention: AlbertAttention;
  readonly ffn: Linear;
  readonly ffn_output: Linear;
  readonly activation: ActivationModule;
  /** Registered like transformers' ``AlbertLayer.dropout``, which its forward never applies. */
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    const hidden = num(config, 'hidden_size');
    this.full_layer_layer_norm = this.registerModule('full_layer_layer_norm', new LayerNorm(hidden, { eps: num(config, 'layer_norm_eps') }));
    this.attention = this.registerModule('attention', new AlbertAttention(config));
    this.ffn = this.registerModule('ffn', new Linear(hidden, num(config, 'intermediate_size')));
    this.ffn_output = this.registerModule('ffn_output', new Linear(num(config, 'intermediate_size'), hidden));
    this.activation = this.registerModule('activation', activationModule(config.string('hidden_act')));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'hidden_dropout_prob')));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const attended = this.attention.forward(hidden, bias);
    const feedForward = this.ffn_output.forward(this.activation.forward(this.ffn.forward(attended)));
    return this.full_layer_layer_norm.forward(feedForward.add(attended));
  }
}

class AlbertLayerGroup extends Module {
  readonly albert_layers: ModuleList<AlbertLayer>;

  constructor(config: NativeConfig) {
    super();
    this.albert_layers = this.registerModule('albert_layers', new ModuleList(
      Array.from({ length: num(config, 'inner_group_num') }, () => new AlbertLayer(config)),
    ));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    let state = hidden;
    for (const layer of this.albert_layers) state = layer.forward(state, bias);
    return state;
  }
}

/** ``AlbertTransformer``: shared layer groups applied ``num_hidden_layers`` times. */
class AlbertTransformer extends Module {
  readonly embedding_hidden_mapping_in: Linear;
  readonly albert_layer_groups: ModuleList<AlbertLayerGroup>;
  readonly layers: number;
  readonly groups: number;

  constructor(config: NativeConfig) {
    super();
    this.layers = num(config, 'num_hidden_layers');
    this.groups = num(config, 'num_hidden_groups');
    this.embedding_hidden_mapping_in = this.registerModule('embedding_hidden_mapping_in',
      new Linear(num(config, 'embedding_size'), num(config, 'hidden_size')));
    this.albert_layer_groups = this.registerModule('albert_layer_groups', new ModuleList(
      Array.from({ length: this.groups }, () => new AlbertLayerGroup(config)),
    ));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    let state = this.embedding_hidden_mapping_in.forward(hidden);
    for (let index = 0; index < this.layers; index += 1) {
      const group = Math.trunc(index / (this.layers / this.groups));
      state = this.albert_layer_groups.at(group).forward(state, bias);
    }
    return state;
  }
}

/** ``AlbertModel`` (``AutoModel`` for ``model_type='albert'``). */
export class AlbertModel extends NativeModel implements NativeEncoder {
  readonly embeddings: AlbertEmbeddings;
  readonly encoder: AlbertTransformer;
  readonly pooler: Linear | null;
  readonly pooler_activation: ActivationModule | null;

  constructor(config: NativeConfig, options: BertModelOptions = {}) {
    super(config);
    this.embeddings = this.registerModule('embeddings', new AlbertEmbeddings(config));
    this.encoder = this.registerModule('encoder', new AlbertTransformer(config));
    const pooling = options.addPoolingLayer !== false;
    const hidden = num(config, 'hidden_size');
    this.pooler = pooling ? this.registerModule('pooler', new Linear(hidden, hidden)) : null;
    this.pooler_activation = pooling ? this.registerModule('pooler_activation', activationModule('tanh')) : null;
    initializeWeights(this, num(config, 'initializer_range'));
  }

  getInputEmbeddings(): Embedding {
    return this.embeddings.word_embeddings;
  }

  forward(inputs: EncoderInputs): EncoderOutput {
    const embedded = this.embeddings.forward(inputs);
    const lastHiddenState = this.encoder.forward(embedded, keyPaddingBias(inputs.attentionMask, embedded.dtype));
    const poolerOutput = this.pooler && this.pooler_activation
      ? this.pooler_activation.forward(this.pooler.forward(lastHiddenState.select(1, 0)))
      : null;
    return { lastHiddenState, poolerOutput };
  }
}

/** ``AlbertForSequenceClassification`` (pooler → dropout → classifier). */
export class AlbertForSequenceClassification extends NativeModel implements NativeSequenceClassifier {
  readonly albert: AlbertModel;
  readonly dropout: Dropout;
  readonly classifier: Linear;

  constructor(config: NativeConfig) {
    super(config);
    this.albert = this.registerModule('albert', new AlbertModel(config));
    this.dropout = this.registerModule('dropout', new Dropout(num(config, 'classifier_dropout_prob')));
    this.classifier = this.registerModule('classifier', new Linear(num(config, 'hidden_size'), config.numLabels));
    initializeWeights(this.classifier, num(config, 'initializer_range'));
  }

  get base(): NativeEncoder { return this.albert; }
  getInputEmbeddings(): Embedding { return this.albert.getInputEmbeddings(); }

  forward(inputs: EncoderInputs): ClassifierOutput {
    const output = this.albert.forward(inputs);
    return { logits: this.classifier.forward(this.dropout.forward(output.poolerOutput!)), lastHiddenState: output.lastHiddenState };
  }
}
