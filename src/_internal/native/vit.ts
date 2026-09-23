/** ``ViTModel`` with transformers 5.17 parameter names (``layers.N.attention.q_proj`` ...). */
import { Module } from '../../nn/module.js';
import { Parameter, Tensor } from '../../nn/tensor.js';
import { Conv2d, Dropout, Embedding, LayerNorm, Linear, ModuleList } from '../../nn/layers.js';
import { activation } from '../../nn/ops/nn.js';
import { cat } from '../../nn/ops/shape.js';
import { noGrad } from '../../nn/autograd.js';
import * as init from '../../nn/init.js';
import { ValueError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import { NativeModel, attention, mergeHeads, splitHeads, newParameter } from './modules.js';

export class ViTPatchEmbeddings extends Module {
  readonly projection: Conv2d;
  readonly imageSize: number;
  readonly patchSize: number;
  readonly channels: number;
  readonly numPatches: number;

  constructor(config: NativeConfig) {
    super();
    this.imageSize = config.number('image_size');
    this.patchSize = config.number('patch_size');
    this.channels = config.number('num_channels');
    this.numPatches = Math.floor(this.imageSize / this.patchSize) ** 2;
    this.projection = this.registerModule('projection', new Conv2d(this.channels, config.number('hidden_size'), this.patchSize, { stride: this.patchSize }));
  }

  forward(pixels: Tensor): Tensor {
    if (pixels.shape[1] !== this.channels) {
      throw new ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.');
    }
    return this.projection.forward(pixels).flatten(2).transpose(1, 2);
  }
}

export class ViTEmbeddings extends Module {
  readonly cls_token: Parameter;
  readonly position_embeddings: Parameter;
  readonly patch_embeddings: ViTPatchEmbeddings;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    const std = config.number('initializer_range');
    this.cls_token = this.registerParameter('cls_token', newParameter([1, 1, hidden], (value) => init.truncatedNormal_(value, 0, std)));
    const patches = new ViTPatchEmbeddings(config);
    this.position_embeddings = this.registerParameter('position_embeddings',
      newParameter([1, patches.numPatches + 1, hidden], (value) => init.truncatedNormal_(value, 0, std)));
    this.patch_embeddings = this.registerModule('patch_embeddings', patches);
    this.dropout = this.registerModule('dropout', new Dropout(config.number('hidden_dropout_prob')));
  }

  forward(pixels: Tensor): Tensor {
    const [batch, , height, width] = pixels.shape as [number, number, number, number];
    const size = this.patch_embeddings.imageSize;
    if (height !== size || width !== size) throw new ValueError(`Input image size (${height}*${width}) doesn't match model (${size}*${size}).`);
    const patches = this.patch_embeddings.forward(pixels);
    const cls = this.cls_token.expand(batch, 1, this.cls_token.shape[2]!);
    return this.dropout.forward(cat([cls, patches], 1).add(this.position_embeddings));
  }
}

class ViTAttention extends Module {
  readonly heads: number;
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
    this.headDim = hidden / this.heads;
    this.dropoutRate = config.number('attention_probs_dropout_prob');
    const bias = config.get('qkv_bias') !== false;
    this.q_proj = this.registerModule('q_proj', new Linear(hidden, hidden, { bias }));
    this.k_proj = this.registerModule('k_proj', new Linear(hidden, hidden, { bias }));
    this.v_proj = this.registerModule('v_proj', new Linear(hidden, hidden, { bias }));
    this.o_proj = this.registerModule('o_proj', new Linear(hidden, hidden));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const q = splitHeads(this.q_proj.forward(hidden), this.heads);
    const k = splitHeads(this.k_proj.forward(hidden), this.heads);
    const v = splitHeads(this.v_proj.forward(hidden), this.heads);
    return this.o_proj.forward(mergeHeads(attention(q, k, v, { scale: this.headDim ** -0.5, bias, dropout: this.dropoutRate, training: this.training })));
  }
}

class ViTMLP extends Module {
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

class ViTLayer extends Module {
  readonly attention: ViTAttention;
  readonly layernorm_before: LayerNorm;
  readonly layernorm_after: LayerNorm;
  readonly mlp: ViTMLP;
  readonly dropout: Dropout;

  constructor(config: NativeConfig) {
    super();
    const hidden = config.number('hidden_size');
    const eps = config.number('layer_norm_eps');
    this.attention = this.registerModule('attention', new ViTAttention(config));
    this.layernorm_before = this.registerModule('layernorm_before', new LayerNorm(hidden, { eps }));
    this.layernorm_after = this.registerModule('layernorm_after', new LayerNorm(hidden, { eps }));
    this.mlp = this.registerModule('mlp', new ViTMLP(config));
    this.dropout = this.registerModule('dropout', new Dropout(config.number('hidden_dropout_prob')));
  }

  forward(hidden: Tensor, bias: Tensor | null): Tensor {
    const attended = this.dropout.forward(this.attention.forward(this.layernorm_before.forward(hidden), bias)).add(hidden);
    return this.dropout.forward(this.mlp.forward(this.layernorm_after.forward(attended))).add(attended);
  }
}

class ViTPooler extends Module {
  readonly dense: Linear;
  private readonly act: (x: Tensor) => Tensor;

  constructor(config: NativeConfig) {
    super();
    this.dense = this.registerModule('dense', new Linear(config.number('hidden_size'), config.number('pooler_output_size')));
    this.act = activation(config.string('pooler_act'));
  }

  forward(hidden: Tensor): Tensor {
    return this.act(this.dense.forward(hidden.select(1, 0)));
  }
}

export interface VisionInputs {
  pixelValues?: Tensor | null;
  /** Precomputed input embeddings ``[batch, tokens, hidden]`` (bypasses patch embedding). */
  inputsEmbeds?: Tensor | null;
  /** Additive attention bias ``[batch, 1, 1|tokens, tokens]`` (e.g. for prefixed context). */
  attentionBias?: Tensor | null;
}

export interface VisionOutput {
  lastHiddenState: Tensor;
  poolerOutput: Tensor | null;
}

export class ViTModel extends NativeModel {
  readonly embeddings: ViTEmbeddings;
  readonly layers: ModuleList<ViTLayer>;
  readonly layernorm: LayerNorm;
  readonly pooler: ViTPooler | null;

  constructor(config: NativeConfig, options: { addPoolingLayer?: boolean } = {}) {
    super(config);
    this.embeddings = this.registerModule('embeddings', new ViTEmbeddings(config));
    this.layers = this.registerModule('layers', new ModuleList(Array.from({ length: config.number('num_hidden_layers') }, () => new ViTLayer(config))));
    this.layernorm = this.registerModule('layernorm', new LayerNorm(config.number('hidden_size'), { eps: config.number('layer_norm_eps') }));
    this.pooler = options.addPoolingLayer === false ? null : this.registerModule('pooler', new ViTPooler(config));
    const std = config.number('initializer_range');
    noGrad(() => {
      for (const module of this.modules()) {
        if (module instanceof Linear || module instanceof Conv2d) {
          init.truncatedNormal_(module.weight, 0, std);
          if (module.bias) module.bias.zero_();
        } else if (module instanceof LayerNorm) {
          module.weight?.fill_(1);
          module.bias?.zero_();
        }
      }
    });
  }

  /** ViT has no token embedding table; exposes the patch projection instead. */
  getInputEmbeddings(): Embedding {
    throw new ValueError('ViT has no token input embeddings; use embeddings.patch_embeddings');
  }

  /** Embed pixels (CLS + patches + positions) without running the encoder. */
  embed(pixels: Tensor): Tensor {
    return this.embeddings.forward(pixels);
  }

  forward(inputs: VisionInputs): VisionOutput {
    const hidden0 = inputs.inputsEmbeds ?? (inputs.pixelValues ? this.embeddings.forward(inputs.pixelValues) : null);
    if (!hidden0) throw new ValueError('You have to specify pixel_values');
    let hidden = hidden0;
    for (const layer of this.layers) hidden = layer.forward(hidden, inputs.attentionBias ?? null);
    const lastHiddenState = this.layernorm.forward(hidden);
    return { lastHiddenState, poolerOutput: this.pooler ? this.pooler.forward(lastHiddenState) : null };
  }
}
