/** ``CLIPModel`` (text and vision towers with projections), transformers 5.17 names. */
import { Module } from '../../nn/module.js';
import { Parameter, Tensor, tensor, scalar } from '../../nn/tensor.js';
import { Conv2d, Embedding, LayerNorm, Linear, ModuleList } from '../../nn/layers.js';
import { activation, normalize } from '../../nn/ops/nn.js';
import { cat } from '../../nn/ops/shape.js';
import { noGrad } from '../../nn/autograd.js';
import * as init from '../../nn/init.js';
import { ValueError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import { NativeModel, attention, causalBias, combineBias, keyPaddingBias, mergeHeads, newParameter, positionIds, splitHeads } from './modules.js';

class CLIPAttention extends Module {
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
    this.headDim = hidden / this.heads;
    this.dropoutRate = config.number('attention_dropout');
    this.k_proj = this.registerModule('k_proj', new Linear(hidden, hidden));
    this.v_proj = this.registerModule('v_proj', new Linear(hidden, hidden));
    this.q_proj = this.registerModule('q_proj', new Linear(hidden, hidden));
    this.out_proj = this.registerModule('out_proj', new Linear(hidden, hidden));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const q = splitHeads(this.q_proj.forward(hidden), this.heads);
    const k = splitHeads(this.k_proj.forward(hidden), this.heads);
    const v = splitHeads(this.v_proj.forward(hidden), this.heads);
    return this.out_proj.forward(mergeHeads(attention(q, k, v, { scale: this.headDim ** -0.5, bias, dropout: this.dropoutRate, training: this.training })));
  }
}

class CLIPMLP extends Module {
  readonly fc1: Linear;
  readonly fc2: Linear;
  private readonly act: (x: Tensor) => Tensor;

  constructor(config: NativeConfig) {
    super();
    this.fc1 = this.registerModule('fc1', new Linear(config.number('hidden_size'), config.number('intermediate_size')));
    this.fc2 = this.registerModule('fc2', new Linear(config.number('intermediate_size'), config.number('hidden_size')));
    this.act = activation(config.string('hidden_act'));
  }

  forward(hidden: Tensor): Tensor {
    return this.fc2.forward(this.act(this.fc1.forward(hidden)));
  }
}

class CLIPEncoderLayer extends Module {
  readonly self_attn: CLIPAttention;
  readonly layer_norm1: LayerNorm;
  readonly mlp: CLIPMLP;
  readonly layer_norm2: LayerNorm;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    const eps = config.number('layer_norm_eps');
    this.self_attn = this.registerModule('self_attn', new CLIPAttention(config));
    this.layer_norm1 = this.registerModule('layer_norm1', new LayerNorm(hidden, { eps }));
    this.mlp = this.registerModule('mlp', new CLIPMLP(config));
    this.layer_norm2 = this.registerModule('layer_norm2', new LayerNorm(hidden, { eps }));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const attended = hidden.add(this.self_attn.forward(this.layer_norm1.forward(hidden), bias));
    return attended.add(this.mlp.forward(this.layer_norm2.forward(attended)));
  }
}

class CLIPEncoder extends Module {
  readonly layers: ModuleList<CLIPEncoderLayer>;

  constructor(config: NativeConfig) {
    super();
    this.layers = this.registerModule('layers', new ModuleList(Array.from({ length: config.number('num_hidden_layers') }, () => new CLIPEncoderLayer(config))));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    let state = hidden;
    for (const layer of this.layers) state = layer.forward(state, bias);
    return state;
  }
}

class CLIPTextEmbeddings extends Module {
  readonly token_embedding: Embedding;
  readonly position_embedding: Embedding;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    this.token_embedding = this.registerModule('token_embedding', new Embedding(config.number('vocab_size'), hidden));
    this.position_embedding = this.registerModule('position_embedding', new Embedding(config.number('max_position_embeddings'), hidden));
  }

  forward(inputIds: Tensor | null, inputsEmbeds: Tensor | null): Tensor {
    const embeds = inputsEmbeds ?? this.token_embedding.forward(inputIds!);
    const [batch, length] = embeds.shape as [number, number];
    if (length > this.position_embedding.numEmbeddings) throw new ValueError('Sequence length must be less than max_position_embeddings');
    return embeds.add(this.position_embedding.forward(positionIds(batch, length)));
  }
}

export interface TextOutput {
  lastHiddenState: Tensor;
  poolerOutput: Tensor;
}

export class CLIPTextTransformer extends Module {
  readonly embeddings: CLIPTextEmbeddings;
  readonly encoder: CLIPEncoder;
  readonly final_layer_norm: LayerNorm;
  readonly eosTokenId: number;

  constructor(config: NativeConfig) {
    super();
    this.embeddings = this.registerModule('embeddings', new CLIPTextEmbeddings(config));
    this.encoder = this.registerModule('encoder', new CLIPEncoder(config));
    this.final_layer_norm = this.registerModule('final_layer_norm', new LayerNorm(config.number('hidden_size'), { eps: config.number('layer_norm_eps') }));
    this.eosTokenId = config.number('eos_token_id');
  }

  forward(inputIds: Tensor, attentionMask: Tensor | null = null): TextOutput {
    const hidden = this.embeddings.forward(inputIds, null);
    const length = hidden.shape[1]!;
    const bias = combineBias(causalBias(length, length, hidden.dtype), keyPaddingBias(attentionMask, hidden.dtype));
    const lastHiddenState = this.final_layer_norm.forward(this.encoder.forward(hidden, bias));
    const [batch] = inputIds.shape as [number];
    const positions: number[] = [];
    for (let b = 0; b < batch; b += 1) {
      const row = Array.from(inputIds.data.subarray(b * length, (b + 1) * length));
      if (this.eosTokenId === 2) {
        let best = 0;
        for (let i = 1; i < row.length; i += 1) if (row[i]! > row[best]!) best = i;
        positions.push(best);
      } else {
        const index = row.indexOf(this.eosTokenId);
        positions.push(index < 0 ? 0 : index);
      }
    }
    const rows = positions.map((position, b) => lastHiddenState.select(0, b).select(0, position).unsqueeze(0));
    return { lastHiddenState, poolerOutput: cat(rows, 0) };
  }
}

class CLIPVisionEmbeddings extends Module {
  readonly class_embedding: Parameter;
  readonly patch_embedding: Conv2d;
  readonly position_embedding: Embedding;
  readonly imageSize: number;
  readonly patchSize: number;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    this.imageSize = config.number('image_size');
    this.patchSize = config.number('patch_size');
    this.class_embedding = this.registerParameter('class_embedding', newParameter([hidden], (value) => init.normal_(value)));
    this.patch_embedding = this.registerModule('patch_embedding',
      new Conv2d(config.number('num_channels'), hidden, this.patchSize, { stride: this.patchSize, bias: false }));
    const patches = (this.imageSize / this.patchSize) ** 2;
    this.position_embedding = this.registerModule('position_embedding', new Embedding(patches + 1, hidden));
  }

  forward(pixels: Tensor): Tensor {
    const [batch, , height, width] = pixels.shape as [number, number, number, number];
    if (height !== this.imageSize || width !== this.imageSize) {
      throw new ValueError(`Input image size (${height}*${width}) doesn't match model (${this.imageSize}*${this.imageSize}).`);
    }
    const patches = this.patch_embedding.forward(pixels).flatten(2).transpose(1, 2);
    const hidden = this.class_embedding.shape[0]!;
    const cls = this.class_embedding.reshape(1, 1, hidden).expand(batch, 1, hidden);
    const embeddings = cat([cls, patches], 1);
    return embeddings.add(this.position_embedding.forward(positionIds(1, embeddings.shape[1]!)));
  }
}

export class CLIPVisionTransformer extends Module {
  readonly embeddings: CLIPVisionEmbeddings;
  readonly pre_layrnorm: LayerNorm;
  readonly encoder: CLIPEncoder;
  readonly post_layernorm: LayerNorm;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    const eps = config.number('layer_norm_eps');
    this.embeddings = this.registerModule('embeddings', new CLIPVisionEmbeddings(config));
    this.pre_layrnorm = this.registerModule('pre_layrnorm', new LayerNorm(hidden, { eps }));
    this.encoder = this.registerModule('encoder', new CLIPEncoder(config));
    this.post_layernorm = this.registerModule('post_layernorm', new LayerNorm(hidden, { eps }));
  }

  forward(pixels: Tensor): TextOutput {
    const hidden = this.pre_layrnorm.forward(this.embeddings.forward(pixels));
    const lastHiddenState = this.encoder.forward(hidden, null);
    return { lastHiddenState, poolerOutput: this.post_layernorm.forward(lastHiddenState.select(1, 0)) };
  }
}

export interface CLIPOutput {
  textEmbeds: Tensor;
  imageEmbeds: Tensor;
  logitsPerImage: Tensor;
  logitsPerText: Tensor;
  text: TextOutput;
  vision: TextOutput;
}

/** ``CLIPModel``. */
export class CLIPModel extends NativeModel {
  readonly logit_scale: Parameter;
  readonly text_model: CLIPTextTransformer;
  readonly vision_model: CLIPVisionTransformer;
  readonly visual_projection: Linear;
  readonly text_projection: Linear;

  constructor(config: NativeConfig) {
    super(config);
    const text = config.sub('text_config');
    const vision = config.sub('vision_config');
    const projection = config.number('projection_dim');
    this.logit_scale = this.registerParameter('logit_scale', new Parameter(scalar(config.number('logit_scale_init_value'))));
    this.text_model = this.registerModule('text_model', new CLIPTextTransformer(text));
    this.vision_model = this.registerModule('vision_model', new CLIPVisionTransformer(vision));
    this.visual_projection = this.registerModule('visual_projection', new Linear(vision.number('hidden_size'), projection, { bias: false }));
    this.text_projection = this.registerModule('text_projection', new Linear(text.number('hidden_size'), projection, { bias: false }));
    const factor = config.number('initializer_factor');
    noGrad(() => {
      for (const module of this.modules()) {
        if (module instanceof Linear) {
          init.normal_(module.weight, 0, 0.02 * factor);
          module.bias?.zero_();
        } else if (module instanceof Embedding) init.normal_(module.weight, 0, 0.02 * factor);
        else if (module instanceof LayerNorm) {
          module.weight?.fill_(1);
          module.bias?.zero_();
        }
      }
    });
  }

  getInputEmbeddings(): Embedding {
    return this.text_model.embeddings.token_embedding;
  }

  /** transformers 5.17 ``get_text_features``: text tower output with the projected pooler output. */
  getTextFeatures(inputIds: Tensor, attentionMask: Tensor | null = null): TextOutput {
    const output = this.text_model.forward(inputIds, attentionMask);
    return { lastHiddenState: output.lastHiddenState, poolerOutput: this.text_projection.forward(output.poolerOutput) };
  }

  /** transformers 5.17 ``get_image_features``: vision tower output with the projected pooler output. */
  getImageFeatures(pixels: Tensor): TextOutput {
    const output = this.vision_model.forward(pixels);
    return { lastHiddenState: output.lastHiddenState, poolerOutput: this.visual_projection.forward(output.poolerOutput) };
  }

  forward(inputs: { inputIds: Tensor; attentionMask?: Tensor | null; pixelValues: Tensor }): CLIPOutput {
    const vision = this.getImageFeatures(inputs.pixelValues);
    const text = this.getTextFeatures(inputs.inputIds, inputs.attentionMask ?? null);
    const imageEmbeds = normalize(vision.poolerOutput, 2, -1);
    const textEmbeds = normalize(text.poolerOutput, 2, -1);
    const logitsPerText = textEmbeds.matmul(imageEmbeds.transpose(0, 1)).mul(this.logit_scale.exp());
    return { textEmbeds, imageEmbeds, logitsPerText, logitsPerImage: logitsPerText.transpose(0, 1), text, vision };
  }
}

/** Deterministic helper for tests: a tensor of pixel values in ``[0, 1]``. */
export function pixelTensor(values: number[], shape: number[]): Tensor {
  return tensor(values, { shape });
}
