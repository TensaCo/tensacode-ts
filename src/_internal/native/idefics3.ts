/**
 * ``Idefics3ForConditionalGeneration`` (SmolVLM; transformers 5.17 names): a
 * SigLIP-style vision transformer over variable-resolution patches, the
 * pixel-shuffle connector, a Llama text model and a language-model head, with
 * greedy generation.
 */
import { activationModule, type ActivationModule } from './activations.js';
import { Module } from '../../nn/module.js';
import { Tensor, tensor } from '../../nn/tensor.js';
import { Conv2d, Embedding, LayerNorm, Linear, ModuleList } from '../../nn/layers.js';
import { crossEntropy } from '../../nn/ops/nn.js';
import { cat } from '../../nn/ops/shape.js';
import { noGrad } from '../../nn/autograd.js';
import { NotImplementedError, ValueError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import { NativeModel, attention, keyPaddingBias, mergeHeads, splitHeads } from './modules.js';
import { LlamaModel, type LlamaLayerCache } from './llama.js';

class Idefics3VisionEmbeddings extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3VisionEmbeddings';
  readonly patch_embedding: Conv2d;
  readonly position_embedding: Embedding;
  readonly patchSize: number;
  readonly patchesPerSide: number;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    this.patchSize = config.number('patch_size');
    this.patchesPerSide = Math.floor(config.number('image_size') / this.patchSize);
    this.patch_embedding = this.registerModule('patch_embedding',
      new Conv2d(config.number('num_channels'), hidden, this.patchSize, { stride: this.patchSize }));
    this.position_embedding = this.registerModule('position_embedding', new Embedding(this.patchesPerSide ** 2, hidden));
  }

  /** Fractional-coordinate position ids (``torch.bucketize`` over ``1/side`` boundaries). */
  positionIds(mask: Tensor): Tensor {
    const [batch, rows, columns] = mask.shape as [number, number, number];
    const side = this.patchesPerSide;
    const step = 1 / side;
    // torch.arange(1 / side, 1.0, 1 / side) in float32.
    const count = Math.ceil((1 - step) / step);
    const boundaries = Array.from({ length: count }, (_, index) => Math.fround(step + index * step));
    const bucket = (value: number): number => {
      let index = 0;
      while (index < boundaries.length && boundaries[index]! <= value) index += 1;
      return index;
    };
    const clampMax = Math.fround(1 - 1e-6);
    const ids = new Array<number>(batch * rows * columns).fill(0);
    for (let b = 0; b < batch; b += 1) {
      const offset = b * rows * columns;
      let validRows = 0;
      let validColumns = 0;
      for (let row = 0; row < rows; row += 1) if (mask.data[offset + row * columns]) validRows += 1;
      for (let column = 0; column < columns; column += 1) if (mask.data[offset + column]) validColumns += 1;
      const stepRows = Math.fround(1 / validRows);
      const stepColumns = Math.fround(1 / validColumns);
      for (let row = 0; row < rows; row += 1) {
        const bucketRow = bucket(Math.min(Math.fround(row * stepRows), clampMax));
        for (let column = 0; column < columns; column += 1) {
          if (!mask.data[offset + row * columns + column]) continue;
          const bucketColumn = bucket(Math.min(Math.fround(column * stepColumns), clampMax));
          ids[offset + row * columns + column] = bucketRow * side + bucketColumn;
        }
      }
    }
    return tensor(ids, { shape: [batch, rows * columns], dtype: 'int64' });
  }

  forward(pixels: Tensor, patchMask: Tensor): Tensor {
    const embeddings = this.patch_embedding.forward(pixels).flatten(2).transpose(1, 2);
    return embeddings.add(this.position_embedding.forward(this.positionIds(patchMask)));
  }
}

class Idefics3VisionAttention extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3VisionAttention';
  readonly heads: number;
  readonly headDim: number;
  readonly dropoutRate: number;
  readonly k_proj: Linear;
  readonly v_proj: Linear;
  readonly q_proj: Linear;
  readonly out_proj: Linear;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    this.heads = config.number('num_attention_heads');
    this.headDim = Math.floor(hidden / this.heads);
    if (this.headDim * this.heads !== hidden) throw new ValueError(`embed_dim must be divisible by num_heads (got \`embed_dim\`: ${hidden} and \`num_heads\`: ${this.heads}).`);
    this.dropoutRate = config.optionalNumber('attention_dropout') ?? 0;
    this.k_proj = this.registerModule('k_proj', new Linear(hidden, hidden));
    this.v_proj = this.registerModule('v_proj', new Linear(hidden, hidden));
    this.q_proj = this.registerModule('q_proj', new Linear(hidden, hidden));
    this.out_proj = this.registerModule('out_proj', new Linear(hidden, hidden));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const q = splitHeads(this.q_proj.forward(hidden), this.heads);
    const k = splitHeads(this.k_proj.forward(hidden), this.heads);
    const v = splitHeads(this.v_proj.forward(hidden), this.heads);
    const output = attention(q, k, v, { scale: this.headDim ** -0.5, bias, dropout: this.dropoutRate, training: this.training });
    return this.out_proj.forward(mergeHeads(output));
  }
}

class Idefics3VisionMLP extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3VisionMLP';
  readonly activation_fn: ActivationModule;
  readonly fc1: Linear;
  readonly fc2: Linear;

  constructor(config: NativeConfig) {
    super();
    this.activation_fn = this.registerModule('activation_fn', activationModule(config.string('hidden_act')));
    this.fc1 = this.registerModule('fc1', new Linear(config.number('hidden_size'), config.number('intermediate_size')));
    this.fc2 = this.registerModule('fc2', new Linear(config.number('intermediate_size'), config.number('hidden_size')));
  }

  forward(hidden: Tensor): Tensor {
    return this.fc2.forward(this.activation_fn.forward(this.fc1.forward(hidden)));
  }
}

class Idefics3EncoderLayer extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3EncoderLayer';
  readonly self_attn: Idefics3VisionAttention;
  readonly layer_norm1: LayerNorm;
  readonly mlp: Idefics3VisionMLP;
  readonly layer_norm2: LayerNorm;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    const eps = config.number('layer_norm_eps');
    this.self_attn = this.registerModule('self_attn', new Idefics3VisionAttention(config));
    this.layer_norm1 = this.registerModule('layer_norm1', new LayerNorm(hidden, { eps }));
    this.mlp = this.registerModule('mlp', new Idefics3VisionMLP(config));
    this.layer_norm2 = this.registerModule('layer_norm2', new LayerNorm(hidden, { eps }));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const attended = hidden.add(this.self_attn.forward(this.layer_norm1.forward(hidden), bias));
    return attended.add(this.mlp.forward(this.layer_norm2.forward(attended)));
  }
}

class Idefics3Encoder extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3Encoder';
  readonly layers: ModuleList<Idefics3EncoderLayer>;

  constructor(config: NativeConfig) {
    super();
    this.layers = this.registerModule('layers', new ModuleList(Array.from({ length: config.number('num_hidden_layers') }, () => new Idefics3EncoderLayer(config))));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    let state = hidden;
    for (const layer of this.layers) state = layer.forward(state, bias);
    return state;
  }
}

/** ``Idefics3VisionTransformer``. */
export class Idefics3VisionTransformer extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3VisionTransformer';
  readonly embeddings: Idefics3VisionEmbeddings;
  readonly encoder: Idefics3Encoder;
  readonly post_layernorm: LayerNorm;
  readonly patchSize: number;

  constructor(config: NativeConfig) {
    super();
    this.embeddings = this.registerModule('embeddings', new Idefics3VisionEmbeddings(config));
    this.encoder = this.registerModule('encoder', new Idefics3Encoder(config));
    this.patchSize = config.number('patch_size');
    this.post_layernorm = this.registerModule('post_layernorm', new LayerNorm(config.number('hidden_size'), { eps: config.number('layer_norm_eps') }));
  }

  /** ``last_hidden_state`` for ``[images, channels, height, width]`` and a boolean patch mask. */
  forward(pixels: Tensor, patchMask: Tensor | null = null): Tensor {
    const [batch, , height, width] = pixels.shape as [number, number, number, number];
    const mask = patchMask ?? tensor(new Array<number>(batch * Math.floor(height / this.patchSize) * Math.floor(width / this.patchSize)).fill(1), {
      shape: [batch, Math.floor(height / this.patchSize), Math.floor(width / this.patchSize)], dtype: 'bool',
    });
    const hidden = this.embeddings.forward(pixels, mask);
    const flat = mask.reshape(batch, -1);
    const allValid = flat.data.every((value) => value !== 0);
    const bias = allValid ? null : keyPaddingBias(flat, hidden.dtype);
    return this.post_layernorm.forward(this.encoder.forward(hidden, bias));
  }
}

class Idefics3SimpleMLP extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3SimpleMLP';
  readonly proj: Linear;

  constructor(config: NativeConfig) {
    super();
    const scale = config.number('scale_factor');
    const input = config.sub('vision_config').number('hidden_size') * scale ** 2;
    this.proj = this.registerModule('proj', new Linear(input, config.sub('text_config').number('hidden_size'), { bias: false }));
  }

  forward(x: Tensor): Tensor {
    return this.proj.forward(x);
  }
}

/** ``Idefics3Connector``: pixel shuffle then modality projection. */
class Idefics3Connector extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3Connector';
  readonly scaleFactor: number;
  readonly modality_projection: Idefics3SimpleMLP;

  constructor(config: NativeConfig) {
    super();
    this.scaleFactor = config.number('scale_factor');
    this.modality_projection = this.registerModule('modality_projection', new Idefics3SimpleMLP(config));
  }

  pixelShuffle(x: Tensor, scale: number): Tensor {
    const [batch, sequence, dim] = x.shape as [number, number, number];
    const height = Math.floor(Math.sqrt(sequence));
    const width = height;
    let result = x.reshape(batch, height, width, dim);
    result = result.reshape(batch, height, Math.floor(width / scale), dim * scale);
    result = result.permute(0, 2, 1, 3);
    result = result.reshape(batch, Math.floor(width / scale), Math.floor(height / scale), dim * scale ** 2);
    result = result.permute(0, 2, 1, 3);
    return result.reshape(batch, Math.floor(sequence / scale ** 2), dim * scale ** 2);
  }

  forward(hidden: Tensor): Tensor {
    return this.modality_projection.forward(this.pixelShuffle(hidden, this.scaleFactor));
  }
}

export interface Idefics3Inputs {
  inputIds: Tensor;
  attentionMask?: Tensor | null;
  pixelValues?: Tensor | null;
  pixelAttentionMask?: Tensor | null;
  /** Connector outputs (``image_hidden_states``) used instead of ``pixelValues``. */
  imageHiddenStates?: Tensor | null;
  labels?: Tensor | null;
  cache?: LlamaLayerCache[] | null;
}

export interface Idefics3Output {
  logits: Tensor;
  loss: Tensor | null;
  imageHiddenStates: Tensor | null;
}

/** ``Idefics3Model``. */
export class Idefics3Model extends Module {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3Model';
  readonly config: NativeConfig;
  readonly vision_model: Idefics3VisionTransformer;
  readonly connector: Idefics3Connector;
  readonly text_model: LlamaModel;
  readonly imageTokenId: number;

  constructor(config: NativeConfig) {
    super();
    this.config = config;
    this.vision_model = this.registerModule('vision_model', new Idefics3VisionTransformer(config.sub('vision_config')));
    this.connector = this.registerModule('connector', new Idefics3Connector(config));
    this.text_model = this.registerModule('text_model', new LlamaModel(config.sub('text_config')));
    this.imageTokenId = config.number('image_token_id');
  }

  getInputEmbeddings(): Embedding {
    return this.text_model.getInputEmbeddings();
  }

  /**
   * ``get_image_features``: vision states of the non-padding images of
   * ``[batch, images, channels, height, width]`` pixels, shuffled and projected
   * (``pooler_output``, ``[realImages, tokens, textHidden]``).
   */
  getImageFeatures(pixelValues: Tensor, pixelAttentionMask: Tensor | null = null): Tensor {
    const [batch, images, channels, height, width] = pixelValues.shape as [number, number, number, number, number];
    const flat = pixelValues.to(this.connector.modality_projection.proj.weight.dtype).reshape(batch * images, channels, height, width);
    const perImage = channels * height * width;
    const real: number[] = [];
    for (let index = 0; index < batch * images; index += 1) {
      let zeros = 0;
      for (let offset = index * perImage; offset < (index + 1) * perImage; offset += 1) if (flat.data[offset] === 0) zeros += 1;
      if (zeros !== perImage) real.push(index);
    }
    const pixels = flat.indexSelect(0, real);
    const patch = this.config.sub('vision_config').number('patch_size');
    const rows = Math.floor(height / patch);
    const columns = Math.floor(width / patch);
    const patchValues = new Array<number>(real.length * rows * columns).fill(1);
    if (pixelAttentionMask) {
      const mask = pixelAttentionMask.reshape(batch * images, height, width);
      real.forEach((image, position) => {
        for (let row = 0; row < rows; row += 1) {
          for (let column = 0; column < columns; column += 1) {
            let any = false;
            for (let y = row * patch; y < (row + 1) * patch && !any; y += 1) {
              for (let x = column * patch; x < (column + 1) * patch; x += 1) {
                if (mask.data[(image * height + y) * width + x]) { any = true; break; }
              }
            }
            patchValues[(position * rows + row) * columns + column] = any ? 1 : 0;
          }
        }
      });
    }
    const patchMask = tensor(patchValues, { shape: [real.length, rows, columns], dtype: 'bool' });
    return this.connector.forward(this.vision_model.forward(pixels, patchMask));
  }

  /** ``inputs_merger``: image token embeddings replaced, in order, by image states. */
  inputsMerger(inputIds: Tensor, embeds: Tensor, imageHiddenStates: Tensor): Tensor {
    const [batch, length, hidden] = embeds.shape as [number, number, number];
    const image = imageHiddenStates.to(embeds.dtype).reshape(-1, hidden);
    const available = image.shape[0]!;
    const index: number[] = [];
    let next = 0;
    for (let position = 0; position < batch * length; position += 1) {
      if (inputIds.data[position] === this.imageTokenId) {
        if (next >= available) throw new ValueError('masked_scatter: the image features do not cover every image token');
        index.push(batch * length + next);
        next += 1;
      } else index.push(position);
    }
    if (next === 0) return embeds;
    return cat([embeds.reshape(-1, hidden), image], 0).indexSelect(0, index).reshape(batch, length, hidden);
  }

  forward(inputs: Idefics3Inputs): { lastHiddenState: Tensor; imageHiddenStates: Tensor | null } {
    if (inputs.pixelValues && inputs.imageHiddenStates) {
      throw new ValueError('You cannot specify both pixel_values and image_hidden_states at the same time.');
    }
    let embeds = this.text_model.getInputEmbeddings().forward(inputs.inputIds);
    let image: Tensor | null = null;
    if (inputs.pixelValues) image = this.getImageFeatures(inputs.pixelValues, inputs.pixelAttentionMask ?? null);
    else if (inputs.imageHiddenStates) image = inputs.imageHiddenStates.to(embeds.dtype);
    if (image) embeds = this.inputsMerger(inputs.inputIds, embeds, image);
    const lastHiddenState = this.text_model.forward({ inputsEmbeds: embeds, attentionMask: inputs.attentionMask ?? null, cache: inputs.cache ?? null });
    return { lastHiddenState, imageHiddenStates: image };
  }
}

/** transformers ``ForCausalLMLoss``: shifted cross-entropy in float32, ``-100`` ignored. */
export function causalLmLoss(logits: Tensor, labels: Tensor): Tensor {
  const [batch, length, vocab] = logits.shape as [number, number, number];
  const shifted: number[] = [];
  for (let b = 0; b < batch; b += 1) {
    for (let position = 0; position < length; position += 1) {
      shifted.push(position + 1 < length ? labels.data[b * length + position + 1]! : -100);
    }
  }
  return crossEntropy(logits.to('float32').reshape(-1, vocab), tensor(shifted, { dtype: 'int64' }), { ignoreIndex: -100 });
}

/** ``Idefics3ForConditionalGeneration``. */
export class Idefics3ForConditionalGeneration extends NativeModel {
  static override readonly qualifiedName: string = 'transformers.models.idefics3.modeling_idefics3.Idefics3ForConditionalGeneration';
  readonly model: Idefics3Model;
  readonly lm_head: Linear;

  constructor(config: NativeConfig) {
    super(config);
    const text = config.sub('text_config');
    this.model = this.registerModule('model', new Idefics3Model(config));
    this.lm_head = this.registerModule('lm_head', new Linear(text.number('hidden_size'), text.number('vocab_size'), { bias: false }));
    if (config.get('tie_word_embeddings') === true) this.lm_head.setParameterAt('weight', this.model.text_model.embed_tokens.weight);
  }

  getInputEmbeddings(): Embedding {
    return this.model.getInputEmbeddings();
  }

  getImageFeatures(pixelValues: Tensor, pixelAttentionMask: Tensor | null = null): Tensor {
    return this.model.getImageFeatures(pixelValues, pixelAttentionMask);
  }

  forward(inputs: Idefics3Inputs): Idefics3Output {
    const output = this.model.forward(inputs);
    const logits = this.lm_head.forward(output.lastHiddenState);
    const loss = inputs.labels ? causalLmLoss(logits, inputs.labels) : null;
    return { logits, loss, imageHiddenStates: output.imageHiddenStates };
  }

  /** Last-position logits of a cached incremental forward. */
  private step(inputIds: Tensor, attentionMask: Tensor, cache: LlamaLayerCache[], image: Tensor | null): Float32Array {
    const output = this.model.forward({ inputIds, attentionMask, imageHiddenStates: image, cache });
    const last = output.lastHiddenState.select(1, output.lastHiddenState.shape[1]! - 1);
    return Float32Array.from(this.lm_head.forward(last).data);
  }

  /**
   * ``generate(..., do_sample=False)`` for one sequence: greedy decoding with a
   * key/value cache. ``image_hidden_states`` are forwarded at every step, as
   * transformers does. Returns the prompt followed by the generated ids.
   */
  generateGreedy(inputs: Idefics3Inputs, settings: GreedySettings): number[] {
    if (inputs.inputIds.shape[0] !== 1) throw new ValueError('generation supports a single sequence');
    return noGrad(() => {
      let image = inputs.imageHiddenStates ?? null;
      if (inputs.pixelValues) {
        if (image) throw new ValueError('You cannot specify both pixel_values and image_hidden_states at the same time.');
        image = this.getImageFeatures(inputs.pixelValues, inputs.pixelAttentionMask ?? null);
      }
      const sequence = Array.from(inputs.inputIds.data, Number);
      const promptLength = sequence.length;
      const mask = Array.from(inputs.attentionMask?.data ?? new Array<number>(promptLength).fill(1), Number);
      const cache = this.model.text_model.newCache();
      const processor = new LogitsProcessor(settings, promptLength);
      let current = inputs.inputIds;
      const maxLength = promptLength + settings.maxNewTokens;
      while (sequence.length < maxLength) {
        const scores = this.step(current, tensor(mask, { shape: [1, mask.length], dtype: 'int64' }), cache, image);
        processor.apply(scores, sequence);
        let best = 0;
        for (let index = 1; index < scores.length; index += 1) if (scores[index]! > scores[best]!) best = index;
        sequence.push(best);
        mask.push(1);
        if (settings.eos.includes(best)) break;
        current = tensor([best], { shape: [1, 1], dtype: 'int64' });
      }
      return sequence;
    });
  }
}

/** Greedy-decoding settings resolved from a ``GenerationConfig`` dictionary. */
export interface GreedySettings {
  maxNewTokens: number;
  eos: number[];
  repetitionPenalty: number | null;
  noRepeatNgramSize: number | null;
  badWordsIds: number[][] | null;
  minLength: number | null;
  minNewTokens: number | null;
  forcedBosTokenId: number | null;
  forcedEosTokenId: number[] | null;
  maxLength: number | null;
  removeInvalidValues: boolean;
  suppressTokens: number[] | null;
  beginSuppressTokens: number[] | null;
}

const UNSUPPORTED_GENERATION: readonly string[] = [
  'guidance_scale', 'sequence_bias', 'encoder_repetition_penalty', 'encoder_no_repeat_ngram_size', 'exponential_decay_length_penalty',
  'watermarking_config', 'stop_strings', 'force_words_ids', 'constraints', 'prompt_lookup_num_tokens', 'assistant_early_exit',
];

/**
 * Resolve ``generate(**batch, max_new_tokens=n, do_sample=False)`` settings
 * from a serialized ``GenerationConfig`` (``to_dict()``) for greedy decoding.
 */
export function greedySettings(generation: Record<string, unknown>, maxNewTokens: number): GreedySettings {
  const numbers = (value: unknown): number[] | null => {
    if (value === null || value === undefined) return null;
    return Array.isArray(value) ? value.map(Number) : [Number(value)];
  };
  const beams = generation.num_beams;
  if (typeof beams === 'number' && beams > 1) throw new NotImplementedError('beam search generation is not implemented for Idefics3');
  for (const key of UNSUPPORTED_GENERATION) {
    const value = generation[key];
    if (value === null || value === undefined) continue;
    if ((key === 'guidance_scale' || key === 'encoder_repetition_penalty') && value === 1) continue;
    if ((key === 'encoder_no_repeat_ngram_size' || key === 'prompt_lookup_num_tokens') && value === 0) continue;
    throw new NotImplementedError(`generation setting ${key} is not implemented natively`);
  }
  const number = (key: string): number | null => (typeof generation[key] === 'number' ? generation[key] as number : null);
  return {
    maxNewTokens,
    eos: numbers(generation.eos_token_id) ?? [],
    repetitionPenalty: number('repetition_penalty'),
    noRepeatNgramSize: number('no_repeat_ngram_size'),
    badWordsIds: Array.isArray(generation.bad_words_ids) ? (generation.bad_words_ids as number[][]).map((item) => item.map(Number)) : null,
    minLength: number('min_length'),
    minNewTokens: number('min_new_tokens'),
    forcedBosTokenId: number('forced_bos_token_id'),
    forcedEosTokenId: numbers(generation.forced_eos_token_id),
    maxLength: null,
    removeInvalidValues: generation.remove_invalid_values === true,
    suppressTokens: numbers(generation.suppress_tokens),
    beginSuppressTokens: numbers(generation.begin_suppress_tokens),
  };
}

/** transformers' logits processors for greedy decoding, in ``_get_logits_processor`` order. */
class LogitsProcessor {
  constructor(private readonly settings: GreedySettings, private readonly promptLength: number) {}

  apply(scores: Float32Array, sequence: readonly number[]): void {
    const s = this.settings;
    const length = sequence.length;
    const maxLength = this.promptLength + s.maxNewTokens;
    if (s.repetitionPenalty !== null && s.repetitionPenalty !== 1) {
      for (const id of new Set(sequence)) {
        if (id < 0 || id >= scores.length) continue;
        const score = scores[id]!;
        scores[id] = score < 0 ? Math.fround(score * s.repetitionPenalty) : Math.fround(score / s.repetitionPenalty);
      }
    }
    if (s.noRepeatNgramSize !== null && s.noRepeatNgramSize > 0 && length + 1 >= s.noRepeatNgramSize) {
      const n = s.noRepeatNgramSize;
      const prefix = sequence.slice(length - n + 1).join(',');
      for (let start = 0; start + n <= length; start += 1) {
        if (sequence.slice(start, start + n - 1).join(',') === prefix) scores[sequence[start + n - 1]!] = -Infinity;
      }
    }
    if (s.badWordsIds) {
      for (const word of s.badWordsIds) {
        if (!word.length || (word.length === 1 && s.eos.includes(word[0]!))) continue;
        const prefix = word.slice(0, -1);
        const matches = prefix.length === 0 || (length >= prefix.length && prefix.every((id, index) => sequence[length - prefix.length + index] === id));
        if (matches) scores[word[word.length - 1]!] = -Infinity;
      }
    }
    if (s.minLength !== null && s.minLength > 0 && s.eos.length && length < s.minLength) for (const id of s.eos) scores[id] = -Infinity;
    if (s.minNewTokens !== null && s.minNewTokens > 0 && s.eos.length && length - this.promptLength < s.minNewTokens) {
      for (const id of s.eos) scores[id] = -Infinity;
    }
    if (s.forcedBosTokenId !== null && length === 1) {
      scores.fill(-Infinity);
      scores[s.forcedBosTokenId] = 0;
    }
    if (s.forcedEosTokenId !== null && length === maxLength - 1) {
      scores.fill(-Infinity);
      for (const id of s.forcedEosTokenId) scores[id] = 0;
    }
    if (s.removeInvalidValues) {
      const max = 3.4028234663852886e38;
      for (let index = 0; index < scores.length; index += 1) {
        const value = scores[index]!;
        if (Number.isNaN(value)) scores[index] = 0;
        else if (value === Infinity) scores[index] = max;
        else if (value === -Infinity) scores[index] = -max;
      }
    }
    if (s.suppressTokens) for (const id of s.suppressTokens) if (id >= 0 && id < scores.length) scores[id] = -Infinity;
    // ``begin_index`` moves past a forced BOS token for one-token prompts.
    const beginIndex = this.promptLength > 1 || s.forcedBosTokenId === null ? this.promptLength : this.promptLength + 1;
    if (s.beginSuppressTokens && length === beginIndex) {
      for (const id of s.beginSuppressTokens) if (id >= 0 && id < scores.length) scores[id] = -Infinity;
    }
  }
}
