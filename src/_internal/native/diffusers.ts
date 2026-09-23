/**
 * diffusers ``UNet2DConditionModel``, ``AutoencoderKL`` and ``DDIMScheduler``
 * (diffusers 0.40 parameter names, module order and numerics) for the latent
 * diffusion ``ImageDecoder``.
 *
 * The supported UNet subset is the plain cross-attention family that
 * TensorCode's ``ImageDecoder`` accepts: ``CrossAttnDownBlock2D`` /
 * ``DownBlock2D`` / ``CrossAttnUpBlock2D`` / ``UpBlock2D`` blocks, a
 * ``UNetMidBlock2DCrossAttn`` (or no) mid block, positional timestep
 * embeddings and ``default``/``scale_shift`` ResNet time conditioning. The VAE
 * supports ``DownEncoderBlock2D``/``UpDecoderBlock2D``. Other diffusers blocks
 * raise {@link NotImplementedError}.
 */
import { NotImplementedError, ValueError } from '../../errors.js';
import { noGrad } from '../../nn/autograd.js';
import { cat, interpolateNearest } from '../../nn/functional.js';
import { Conv2d, Dropout, GELU, GroupNorm, LayerNorm, Linear, ModuleList, ReLU, SiLU } from '../../nn/layers.js';
import { Module } from '../../nn/module.js';
import { gelu } from '../../nn/ops/nn.js';
import { Tensor, tensor, zeros } from '../../nn/tensor.js';
import { deepCopy, isPlainObject, type JsonObject, type JsonValue } from '../json.js';
import { DIFFUSERS_CONFIG_DEFAULTS } from './diffusersDefaults.generated.js';
import { attention, mergeHeads, splitHeads } from './modules.js';

export type DiffusersComponent = 'UNet2DConditionModel' | 'AutoencoderKL' | 'DDIMScheduler';

/**
 * A component's resolved configuration: constructor defaults overlaid with every
 * supplied key (diffusers keeps unknown keys in ``config``), without private
 * ``_``-prefixed entries (Python ``_native_config(component.config)``).
 */
export function resolveDiffusersConfig(component: DiffusersComponent, supplied: unknown): JsonObject {
  if (!isPlainObject(supplied)) throw new TypeError(`${component} configuration must be a mapping`);
  const result: JsonObject = deepCopy(DIFFUSERS_CONFIG_DEFAULTS[component]!);
  for (const [key, value] of Object.entries(supplied as JsonObject)) {
    if (!key.startsWith('_')) result[key] = deepCopy(value as JsonValue);
  }
  return result;
}

function int(config: JsonObject, key: string): number {
  const value = config[key];
  if (typeof value !== 'number' || !Number.isInteger(value)) throw new ValueError(`${key} must be an integer`);
  return value;
}

function num(config: JsonObject, key: string): number {
  const value = config[key];
  if (typeof value !== 'number') throw new ValueError(`${key} must be a number`);
  return value;
}

function perBlock<T>(value: JsonValue | undefined, count: number, name: string): T[] {
  if (Array.isArray(value)) {
    if (value.length !== count) throw new ValueError(`Must provide the same number of \`${name}\` as \`down_block_types\``);
    return value as unknown as T[];
  }
  return Array.from({ length: count }, () => value as unknown as T);
}

/** diffusers ``get_activation``. */
function activation(name: string): Module & { forward(x: Tensor): Tensor } {
  switch (name) {
    case 'silu': case 'swish': return new SiLU();
    case 'gelu': return new GELU();
    case 'relu': return new ReLU();
    default: throw new NotImplementedError(`diffusers activation ${JSON.stringify(name)} is not available in TypeScript`);
  }
}

// ---------------------------------------------------------------------------
// Embeddings.
// ---------------------------------------------------------------------------

/** diffusers ``get_timestep_embedding`` in float32. */
export function timestepEmbedding(timesteps: Tensor, dimension: number, flipSinToCos: boolean, shift: number): Tensor {
  const half = Math.floor(dimension / 2);
  const f = Math.fround;
  const logPeriod = f(-Math.log(10000));
  const divisor = f(half - shift);
  const frequencies = Array.from({ length: half }, (_, index) => f(Math.exp(f(f(logPeriod * index) / divisor))));
  const steps = Array.from(timesteps.toArray(), (value) => f(value));
  const values: number[] = [];
  for (const step of steps) {
    const angles = frequencies.map((frequency) => f(step * frequency));
    const sin = angles.map((angle) => f(Math.sin(angle)));
    const cos = angles.map((angle) => f(Math.cos(angle)));
    const row = flipSinToCos ? [...cos, ...sin] : [...sin, ...cos];
    if (dimension % 2 === 1) row.push(0);
    values.push(...row);
  }
  return tensor(values, { shape: [steps.length, dimension], dtype: 'float32' });
}

/** ``Timesteps``: a parameter-free sinusoidal projection. */
class Timesteps extends Module {
  constructor(readonly channels: number, readonly flipSinToCos: boolean, readonly shift: number) {
    super();
  }

  forward(timesteps: Tensor): Tensor {
    return timestepEmbedding(timesteps, this.channels, this.flipSinToCos, this.shift);
  }
}

class TimestepEmbedding extends Module {
  readonly linear_1: Linear;
  readonly act: Module & { forward(x: Tensor): Tensor };
  readonly linear_2: Linear;
  readonly post_act: (Module & { forward(x: Tensor): Tensor }) | null;

  constructor(input: number, embedding: number, act: string, postAct: string | null) {
    super();
    this.linear_1 = this.registerModule('linear_1', new Linear(input, embedding));
    this.act = this.registerModule('act', activation(act));
    this.linear_2 = this.registerModule('linear_2', new Linear(embedding, embedding));
    this.post_act = postAct === null ? null : this.registerModule('post_act', activation(postAct));
  }

  forward(sample: Tensor): Tensor {
    const hidden = this.linear_2.forward(this.act.forward(this.linear_1.forward(sample)));
    return this.post_act ? this.post_act.forward(hidden) : hidden;
  }
}

// ---------------------------------------------------------------------------
// ResNet, sampling and attention blocks.
// ---------------------------------------------------------------------------

interface ResnetOptions {
  inChannels: number;
  outChannels: number;
  tembChannels: number | null;
  eps: number;
  groups: number;
  groupsOut?: number;
  dropout: number;
  timeEmbeddingNorm: string;
  nonLinearity: string;
  outputScaleFactor: number;
}

export class ResnetBlock2D extends Module {
  readonly norm1: GroupNorm;
  readonly conv1: Conv2d;
  readonly time_emb_proj: Linear | null;
  readonly norm2: GroupNorm;
  readonly dropout: Dropout;
  readonly conv2: Conv2d;
  readonly nonlinearity: Module & { forward(x: Tensor): Tensor };
  readonly conv_shortcut: Conv2d | null;
  readonly timeEmbeddingNorm: string;
  readonly outputScaleFactor: number;

  constructor(options: ResnetOptions) {
    super();
    const { inChannels, outChannels, tembChannels, eps } = options;
    if (options.timeEmbeddingNorm === 'spatial' || options.timeEmbeddingNorm === 'ada_group') {
      throw new NotImplementedError(`ResNet time embedding norm ${options.timeEmbeddingNorm} is not available in TypeScript`);
    }
    this.timeEmbeddingNorm = options.timeEmbeddingNorm;
    this.outputScaleFactor = options.outputScaleFactor;
    this.norm1 = this.registerModule('norm1', new GroupNorm(options.groups, inChannels, { eps }));
    this.conv1 = this.registerModule('conv1', new Conv2d(inChannels, outChannels, 3, { padding: 1 }));
    if (tembChannels !== null) {
      if (this.timeEmbeddingNorm === 'default') this.time_emb_proj = this.registerModule('time_emb_proj', new Linear(tembChannels, outChannels));
      else if (this.timeEmbeddingNorm === 'scale_shift') this.time_emb_proj = this.registerModule('time_emb_proj', new Linear(tembChannels, 2 * outChannels));
      else throw new ValueError(`unknown time_embedding_norm : ${this.timeEmbeddingNorm} `);
    } else {
      this.time_emb_proj = null;
    }
    this.norm2 = this.registerModule('norm2', new GroupNorm(options.groupsOut ?? options.groups, outChannels, { eps }));
    this.dropout = this.registerModule('dropout', new Dropout(options.dropout));
    this.conv2 = this.registerModule('conv2', new Conv2d(outChannels, outChannels, 3, { padding: 1 }));
    this.nonlinearity = this.registerModule('nonlinearity', activation(options.nonLinearity));
    this.conv_shortcut = inChannels !== outChannels
      ? this.registerModule('conv_shortcut', new Conv2d(inChannels, outChannels, 1))
      : null;
  }

  forward(input: Tensor, temb: Tensor | null): Tensor {
    let hidden = this.conv1.forward(this.nonlinearity.forward(this.norm1.forward(input)));
    let time: Tensor | null = null;
    if (this.time_emb_proj && temb) {
      time = this.time_emb_proj.forward(this.nonlinearity.forward(temb));
      time = time.reshape(time.shape[0]!, time.shape[1]!, 1, 1);
    }
    if (this.timeEmbeddingNorm === 'default') {
      if (time) hidden = hidden.add(time);
      hidden = this.norm2.forward(hidden);
    } else if (this.timeEmbeddingNorm === 'scale_shift') {
      if (!time) throw new ValueError(' `temb` should not be None when `time_embedding_norm` is scale_shift');
      const [scale, shift] = time.chunk(2, 1) as [Tensor, Tensor];
      hidden = this.norm2.forward(hidden).mul(scale.add(1)).add(shift);
    } else {
      hidden = this.norm2.forward(hidden);
    }
    hidden = this.conv2.forward(this.dropout.forward(this.nonlinearity.forward(hidden)));
    const shortcut = this.conv_shortcut ? this.conv_shortcut.forward(input) : input;
    const output = shortcut.add(hidden);
    return this.outputScaleFactor === 1 ? output : output.div(this.outputScaleFactor);
  }
}

/** ``Downsample2D(use_conv=True)``; padding 0 pads right/bottom by one first. */
class Downsample2D extends Module {
  readonly conv: Conv2d;

  constructor(channels: number, readonly padding: number) {
    super();
    this.conv = this.registerModule('conv', new Conv2d(channels, channels, 3, { stride: 2, padding }));
  }

  forward(hidden: Tensor): Tensor {
    let input = hidden;
    if (this.padding === 0) {
      const [batch, channels, height, width] = input.shape as [number, number, number, number];
      input = cat([input, zeros([batch, channels, height, 1], { dtype: input.dtype })], 3);
      input = cat([input, zeros([batch, channels, 1, width + 1], { dtype: input.dtype })], 2);
    }
    return this.conv.forward(input);
  }
}

/** ``Upsample2D(use_conv=True)``: nearest 2x (or to ``size``) then a 3x3 convolution. */
class Upsample2D extends Module {
  readonly conv: Conv2d;

  constructor(channels: number) {
    super();
    this.conv = this.registerModule('conv', new Conv2d(channels, channels, 3, { padding: 1 }));
  }

  forward(hidden: Tensor, size: readonly [number, number] | null = null): Tensor {
    const target: [number, number] = size ? [size[0], size[1]] : [hidden.shape[2]! * 2, hidden.shape[3]! * 2];
    return this.conv.forward(interpolateNearest(hidden, target));
  }
}

/** diffusers ``Attention`` with ``AttnProcessor2_0`` (the default processor). */
class DiffusersAttention extends Module {
  readonly heads: number;
  readonly headDim: number;
  readonly residualConnection: boolean;
  readonly rescaleOutputFactor: number;
  readonly group_norm: GroupNorm | null;
  readonly to_q: Linear;
  readonly to_k: Linear;
  readonly to_v: Linear;
  readonly to_out: ModuleList<Module>;

  constructor(options: {
    queryDim: number; heads: number; dimHead: number; crossAttentionDim?: number | null; bias?: boolean; outBias?: boolean;
    dropout?: number; groupNormGroups?: number | null; eps?: number; residualConnection?: boolean; rescaleOutputFactor?: number;
  }) {
    super();
    const inner = options.dimHead * options.heads;
    const cross = options.crossAttentionDim ?? options.queryDim;
    const bias = options.bias ?? false;
    this.heads = options.heads;
    this.headDim = options.dimHead;
    this.residualConnection = options.residualConnection ?? false;
    this.rescaleOutputFactor = options.rescaleOutputFactor ?? 1;
    this.group_norm = options.groupNormGroups
      ? this.registerModule('group_norm', new GroupNorm(options.groupNormGroups, options.queryDim, { eps: options.eps ?? 1e-5 }))
      : null;
    this.to_q = this.registerModule('to_q', new Linear(options.queryDim, inner, { bias }));
    this.to_k = this.registerModule('to_k', new Linear(cross, inner, { bias }));
    this.to_v = this.registerModule('to_v', new Linear(cross, inner, { bias }));
    this.to_out = this.registerModule('to_out', new ModuleList<Module>([
      new Linear(inner, options.queryDim, { bias: options.outBias ?? true }), new Dropout(options.dropout ?? 0),
    ]));
  }

  /** ``bias`` is an additive ``[batch, 1, keys]`` mask (diffusers' converted attention mask). */
  forward(hidden: Tensor, encoder: Tensor | null = null, bias: Tensor | null = null): Tensor {
    const residual = hidden;
    const spatial = hidden.ndim === 4;
    const [batch, channels, height, width] = hidden.shape as [number, number, number, number];
    let states = spatial ? hidden.reshape(batch, channels, height * width).transpose(1, 2) : hidden;
    if (this.group_norm) states = this.group_norm.forward(states.transpose(1, 2)).transpose(1, 2);
    const context = encoder ?? states;
    const q = splitHeads(this.to_q.forward(states), this.heads);
    const k = splitHeads(this.to_k.forward(context), this.heads);
    const v = splitHeads(this.to_v.forward(context), this.heads);
    const mask = bias ? bias.unsqueeze(1) : null;
    let output = mergeHeads(attention(q, k, v, { scale: this.headDim ** -0.5, bias: mask }));
    output = (this.to_out.at(1) as Dropout).forward((this.to_out.at(0) as Linear).forward(output));
    if (spatial) output = output.transpose(-1, -2).reshape(batch, channels, height, width);
    if (this.residualConnection) output = output.add(residual);
    return this.rescaleOutputFactor === 1 ? output : output.div(this.rescaleOutputFactor);
  }
}

class GEGLU extends Module {
  readonly proj: Linear;

  constructor(dimIn: number, dimOut: number) {
    super();
    this.proj = this.registerModule('proj', new Linear(dimIn, dimOut * 2));
  }

  forward(hidden: Tensor): Tensor {
    const [value, gate] = this.proj.forward(hidden).chunk(2, -1) as [Tensor, Tensor];
    return value.mul(gelu(gate));
  }
}

class FeedForward extends Module {
  readonly net: ModuleList<Module>;

  constructor(dim: number, dropout: number) {
    super();
    const inner = dim * 4;
    this.net = this.registerModule('net', new ModuleList<Module>([new GEGLU(dim, inner), new Dropout(dropout), new Linear(inner, dim)]));
  }

  forward(hidden: Tensor): Tensor {
    const [act, dropout, out] = [...this.net] as [GEGLU, Dropout, Linear];
    return out.forward(dropout.forward(act.forward(hidden)));
  }
}

class BasicTransformerBlock extends Module {
  readonly norm1: LayerNorm;
  readonly attn1: DiffusersAttention;
  readonly norm2: LayerNorm;
  readonly attn2: DiffusersAttention;
  readonly norm3: LayerNorm;
  readonly ff: FeedForward;

  constructor(dim: number, heads: number, headDim: number, crossAttentionDim: number, dropout: number, readonly onlyCrossAttention: boolean) {
    super();
    this.norm1 = this.registerModule('norm1', new LayerNorm(dim, { eps: 1e-5 }));
    this.attn1 = this.registerModule('attn1', new DiffusersAttention({
      queryDim: dim, heads, dimHead: headDim, dropout, crossAttentionDim: onlyCrossAttention ? crossAttentionDim : null,
    }));
    this.norm2 = this.registerModule('norm2', new LayerNorm(dim, { eps: 1e-5 }));
    this.attn2 = this.registerModule('attn2', new DiffusersAttention({ queryDim: dim, heads, dimHead: headDim, dropout, crossAttentionDim }));
    this.norm3 = this.registerModule('norm3', new LayerNorm(dim, { eps: 1e-5 }));
    this.ff = this.registerModule('ff', new FeedForward(dim, dropout));
  }

  forward(hidden: Tensor, encoder: Tensor, encoderBias: Tensor | null): Tensor {
    let states = this.attn1.forward(this.norm1.forward(hidden), this.onlyCrossAttention ? encoder : null, null).add(hidden);
    states = this.attn2.forward(this.norm2.forward(states), encoder, encoderBias).add(states);
    return this.ff.forward(this.norm3.forward(states)).add(states);
  }
}

class Transformer2DModel extends Module {
  readonly norm: GroupNorm;
  readonly proj_in: Linear | Conv2d;
  readonly transformer_blocks: ModuleList<BasicTransformerBlock>;
  readonly proj_out: Linear | Conv2d;

  constructor(heads: number, headDim: number, readonly channels: number, layers: number, crossAttentionDim: number,
    groups: number, readonly linearProjection: boolean, onlyCrossAttention: boolean, dropout: number) {
    super();
    const inner = heads * headDim;
    this.norm = this.registerModule('norm', new GroupNorm(groups, channels, { eps: 1e-6 }));
    this.proj_in = this.registerModule('proj_in', linearProjection ? new Linear(channels, inner) : new Conv2d(channels, inner, 1));
    this.transformer_blocks = this.registerModule('transformer_blocks', new ModuleList(
      Array.from({ length: layers }, () => new BasicTransformerBlock(inner, heads, headDim, crossAttentionDim, dropout, onlyCrossAttention)),
    ));
    this.proj_out = this.registerModule('proj_out', linearProjection ? new Linear(inner, channels) : new Conv2d(inner, channels, 1));
  }

  forward(hidden: Tensor, encoder: Tensor, encoderBias: Tensor | null): Tensor {
    const [batch, , height, width] = hidden.shape as [number, number, number, number];
    const residual = hidden;
    let states = this.norm.forward(hidden);
    let inner: number;
    if (!this.linearProjection) {
      states = (this.proj_in as Conv2d).forward(states);
      inner = states.shape[1]!;
      states = states.permute(0, 2, 3, 1).reshape(batch, height * width, inner);
    } else {
      inner = states.shape[1]!;
      states = (this.proj_in as Linear).forward(states.permute(0, 2, 3, 1).reshape(batch, height * width, inner));
    }
    for (const block of this.transformer_blocks) states = block.forward(states, encoder, encoderBias);
    if (!this.linearProjection) {
      states = (this.proj_out as Conv2d).forward(states.reshape(batch, height, width, inner).permute(0, 3, 1, 2));
    } else {
      states = (this.proj_out as Linear).forward(states).reshape(batch, height, width, this.channels).permute(0, 3, 1, 2);
    }
    return states.add(residual);
  }
}

// ---------------------------------------------------------------------------
// UNet blocks.
// ---------------------------------------------------------------------------

interface BlockArgs {
  layers: number;
  transformerLayers: number | number[];
  inChannels: number;
  outChannels: number;
  prevOutputChannel?: number;
  temb: number | null;
  resample: boolean;
  eps: number;
  act: string;
  groups: number;
  crossAttentionDim: number;
  heads: number;
  downsamplePadding: number;
  linearProjection: boolean;
  onlyCrossAttention: boolean;
  timeScaleShift: string;
  dropout: number;
}

function resnet(args: BlockArgs, inChannels: number, outChannels: number, outputScaleFactor = 1): ResnetBlock2D {
  return new ResnetBlock2D({
    inChannels, outChannels, tembChannels: args.temb, eps: args.eps, groups: args.groups, dropout: args.dropout,
    timeEmbeddingNorm: args.timeScaleShift, nonLinearity: args.act, outputScaleFactor,
  });
}

function transformerLayers(args: BlockArgs): number[] {
  return typeof args.transformerLayers === 'number'
    ? Array.from({ length: args.layers }, () => args.transformerLayers as number)
    : args.transformerLayers;
}

interface DownBlock extends Module {
  readonly hasCrossAttention: boolean;
  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, bias: Tensor | null): [Tensor, Tensor[]];
}

class CrossAttnDownBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = true;
  readonly attentions: ModuleList<Transformer2DModel>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Downsample2D> | null;

  constructor(args: BlockArgs) {
    super();
    const layers = transformerLayers(args);
    const resnets: ResnetBlock2D[] = [];
    const attentions: Transformer2DModel[] = [];
    for (let index = 0; index < args.layers; index += 1) {
      resnets.push(resnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels));
      attentions.push(new Transformer2DModel(args.heads, Math.floor(args.outChannels / args.heads), args.outChannels, layers[index]!,
        args.crossAttentionDim, args.groups, args.linearProjection, args.onlyCrossAttention, 0));
    }
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.downsamplers = args.resample
      ? this.registerModule('downsamplers', new ModuleList([new Downsample2D(args.outChannels, args.downsamplePadding)]))
      : null;
  }

  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, bias: Tensor | null): [Tensor, Tensor[]] {
    const outputs: Tensor[] = [];
    let states = hidden;
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = this.resnets.at(index).forward(states, temb);
      states = this.attentions.at(index).forward(states, encoder, bias);
      outputs.push(states);
    }
    if (this.downsamplers) {
      for (const downsampler of this.downsamplers) states = downsampler.forward(states);
      outputs.push(states);
    }
    return [states, outputs];
  }
}

class DownBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = false;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Downsample2D> | null;

  constructor(args: BlockArgs) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(
      Array.from({ length: args.layers }, (_, index) => resnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels)),
    ));
    this.downsamplers = args.resample
      ? this.registerModule('downsamplers', new ModuleList([new Downsample2D(args.outChannels, args.downsamplePadding)]))
      : null;
  }

  forward(hidden: Tensor, temb: Tensor | null): [Tensor, Tensor[]] {
    const outputs: Tensor[] = [];
    let states = hidden;
    for (const block of this.resnets) {
      states = block.forward(states, temb);
      outputs.push(states);
    }
    if (this.downsamplers) {
      for (const downsampler of this.downsamplers) states = downsampler.forward(states);
      outputs.push(states);
    }
    return [states, outputs];
  }
}

interface UpBlock extends Module {
  readonly resnets: ModuleList<ResnetBlock2D>;
  forward(hidden: Tensor, residuals: Tensor[], temb: Tensor | null, encoder: Tensor, bias: Tensor | null, size: readonly [number, number] | null): Tensor;
}

function upResnets(args: BlockArgs): ResnetBlock2D[] {
  return Array.from({ length: args.layers }, (_, index) => {
    const skip = index === args.layers - 1 ? args.inChannels : args.outChannels;
    const input = index === 0 ? args.prevOutputChannel! : args.outChannels;
    return resnet(args, input + skip, args.outChannels);
  });
}

class CrossAttnUpBlock2D extends Module implements UpBlock {
  readonly attentions: ModuleList<Transformer2DModel>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Upsample2D> | null;

  constructor(args: BlockArgs) {
    super();
    const layers = transformerLayers(args);
    const resnets = upResnets(args);
    const attentions = resnets.map((_, index) => new Transformer2DModel(args.heads, Math.floor(args.outChannels / args.heads), args.outChannels,
      layers[index]!, args.crossAttentionDim, args.groups, args.linearProjection, args.onlyCrossAttention, 0));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList([new Upsample2D(args.outChannels)])) : null;
  }

  forward(hidden: Tensor, residuals: Tensor[], temb: Tensor | null, encoder: Tensor, bias: Tensor | null, size: readonly [number, number] | null): Tensor {
    let states = hidden;
    const pending = [...residuals];
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = cat([states, pending.pop()!], 1);
      states = this.resnets.at(index).forward(states, temb);
      states = this.attentions.at(index).forward(states, encoder, bias);
    }
    if (this.upsamplers) for (const upsampler of this.upsamplers) states = upsampler.forward(states, size);
    return states;
  }
}

class UpBlock2D extends Module implements UpBlock {
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Upsample2D> | null;

  constructor(args: BlockArgs) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(upResnets(args)));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList([new Upsample2D(args.outChannels)])) : null;
  }

  forward(hidden: Tensor, residuals: Tensor[], temb: Tensor | null, _encoder: Tensor, _bias: Tensor | null, size: readonly [number, number] | null): Tensor {
    let states = hidden;
    const pending = [...residuals];
    for (const block of this.resnets) states = block.forward(cat([states, pending.pop()!], 1), temb);
    if (this.upsamplers) for (const upsampler of this.upsamplers) states = upsampler.forward(states, size);
    return states;
  }
}

class UNetMidBlock2DCrossAttn extends Module {
  readonly attentions: ModuleList<Transformer2DModel>;
  readonly resnets: ModuleList<ResnetBlock2D>;

  constructor(args: BlockArgs, outputScaleFactor: number) {
    super();
    const channels = args.inChannels;
    const layers = typeof args.transformerLayers === 'number' ? [args.transformerLayers] : args.transformerLayers;
    const resnets = [resnet(args, channels, channels, outputScaleFactor)];
    const attentions = [new Transformer2DModel(args.heads, Math.floor(channels / args.heads), channels, layers[0]!, args.crossAttentionDim,
      args.groups, args.linearProjection, false, 0)];
    resnets.push(resnet(args, channels, channels, outputScaleFactor));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
  }

  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, bias: Tensor | null): Tensor {
    let states = this.resnets.at(0).forward(hidden, temb);
    for (let index = 0; index < this.attentions.length; index += 1) {
      states = this.attentions.at(index).forward(states, encoder, bias);
      states = this.resnets.at(index + 1).forward(states, temb);
    }
    return states;
  }
}

// ---------------------------------------------------------------------------
// UNet2DConditionModel.
// ---------------------------------------------------------------------------

const UNSUPPORTED_UNET: [string, (value: JsonValue | undefined) => boolean][] = [
  ['time_embedding_type', (value) => value !== 'positional'],
  ['encoder_hid_dim_type', (value) => value !== null && value !== undefined],
  ['class_embed_type', (value) => value !== null && value !== undefined],
  ['addition_embed_type', (value) => value !== null && value !== undefined],
  ['num_class_embeds', (value) => value !== null && value !== undefined],
  ['time_cond_proj_dim', (value) => value !== null && value !== undefined],
  ['dual_cross_attention', (value) => value === true],
  ['attention_type', (value) => value !== 'default'],
  ['center_input_sample', (value) => typeof value !== 'boolean'],
];

export class UNet2DConditionModel extends Module {
  static override readonly qualifiedName: string = 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel';
  readonly config: JsonObject;
  readonly conv_in: Conv2d;
  readonly time_proj: Timesteps;
  readonly time_embedding: TimestepEmbedding;
  readonly time_embed_act: (Module & { forward(x: Tensor): Tensor }) | null;
  readonly down_blocks: ModuleList<DownBlock>;
  readonly up_blocks: ModuleList<UpBlock>;
  readonly mid_block: UNetMidBlock2DCrossAttn | null;
  readonly conv_norm_out: GroupNorm | null;
  readonly conv_act: (Module & { forward(x: Tensor): Tensor }) | null;
  readonly conv_out: Conv2d;
  readonly numUpsamplers: number;

  constructor(config: JsonObject) {
    super();
    this.config = resolveDiffusersConfig('UNet2DConditionModel', config);
    const c = this.config;
    for (const [key, unsupported] of UNSUPPORTED_UNET) {
      if (unsupported(c[key])) throw new NotImplementedError(`UNet2DConditionModel ${key}=${JSON.stringify(c[key])} is not available in TypeScript`);
    }
    if (c.num_attention_heads !== null && c.num_attention_heads !== undefined) {
      throw new ValueError('At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19.');
    }
    const down = c.down_block_types as string[];
    const up = c.up_block_types as string[];
    const channels = c.block_out_channels as number[];
    if (!Array.isArray(down) || !Array.isArray(up) || !Array.isArray(channels)) throw new ValueError('block types and block_out_channels must be lists');
    if (down.length !== up.length) throw new ValueError(`Must provide the same number of \`down_block_types\` as \`up_block_types\`. \`down_block_types\`: ${JSON.stringify(down)}. \`up_block_types\`: ${JSON.stringify(up)}.`);
    if (channels.length !== down.length) throw new ValueError(`Must provide the same number of \`block_out_channels\` as \`down_block_types\`. \`block_out_channels\`: ${JSON.stringify(channels)}. \`down_block_types\`: ${JSON.stringify(down)}.`);
    const groups = c.norm_num_groups;
    if (typeof groups !== 'number') throw new NotImplementedError('UNet2DConditionModel without norm_num_groups is not available in TypeScript');
    const count = down.length;
    // diffusers' naming quirk: ``attention_head_dim`` is the number of heads.
    const heads = perBlock<number>(c.attention_head_dim, count, 'attention_head_dim');
    const crossDims = perBlock<number>(c.cross_attention_dim, count, 'cross_attention_dim');
    const layersPerBlock = perBlock<number>(c.layers_per_block, count, 'layers_per_block');
    const transformerLayers = perBlock<number | number[]>(c.transformer_layers_per_block, count, 'transformer_layers_per_block');
    const onlyCross = perBlock<boolean>(c.only_cross_attention, count, 'only_cross_attention');
    const inKernel = int(c, 'conv_in_kernel');
    const outKernel = int(c, 'conv_out_kernel');
    this.conv_in = this.registerModule('conv_in', new Conv2d(int(c, 'in_channels'), channels[0]!, inKernel, { padding: Math.floor((inKernel - 1) / 2) }));
    const timeEmbedDim = (c.time_embedding_dim as number | null) ?? channels[0]! * 4;
    this.time_proj = this.registerModule('time_proj', new Timesteps(channels[0]!, c.flip_sin_to_cos === true, num(c, 'freq_shift')));
    const act = String(c.act_fn);
    this.time_embedding = this.registerModule('time_embedding', new TimestepEmbedding(channels[0]!, timeEmbedDim, act, (c.timestep_post_act as string | null) ?? null));
    this.time_embed_act = typeof c.time_embedding_act_fn === 'string' ? this.registerModule('time_embed_act', activation(c.time_embedding_act_fn)) : null;
    const base = {
      temb: timeEmbedDim, eps: num(c, 'norm_eps'), act, groups, downsamplePadding: int(c, 'downsample_padding'),
      linearProjection: c.use_linear_projection === true, timeScaleShift: String(c.resnet_time_scale_shift), dropout: num(c, 'dropout'),
    };
    this.down_blocks = this.registerModule('down_blocks', new ModuleList<DownBlock>());
    this.up_blocks = this.registerModule('up_blocks', new ModuleList<UpBlock>());
    let output = channels[0]!;
    down.forEach((type, index) => {
      const input = output;
      output = channels[index]!;
      const args: BlockArgs = {
        ...base, layers: layersPerBlock[index]!, transformerLayers: transformerLayers[index]!, inChannels: input, outChannels: output,
        resample: index !== channels.length - 1, crossAttentionDim: crossDims[index]!, heads: heads[index]!, onlyCrossAttention: onlyCross[index]!,
      };
      const name = type.startsWith('UNetRes') ? type.slice(7) : type;
      if (name === 'CrossAttnDownBlock2D') this.down_blocks.append(new CrossAttnDownBlock2D(args));
      else if (name === 'DownBlock2D') this.down_blocks.append(new DownBlock2D(args));
      else throw new NotImplementedError(`diffusers down block ${type} is not available in TypeScript`);
    });
    const midType = c.mid_block_type;
    if (midType === 'UNetMidBlock2DCrossAttn') {
      const last = transformerLayers[count - 1]!;
      this.mid_block = this.registerModule('mid_block', new UNetMidBlock2DCrossAttn({
        ...base, layers: 1, transformerLayers: last, inChannels: channels[count - 1]!, outChannels: channels[count - 1]!, resample: false,
        crossAttentionDim: crossDims[count - 1]!, heads: heads[count - 1]!, onlyCrossAttention: false,
      }, num(c, 'mid_block_scale_factor')));
    } else if (midType === null) {
      this.mid_block = null;
    } else {
      throw new NotImplementedError(`diffusers mid block ${String(midType)} is not available in TypeScript`);
    }
    const reversedChannels = [...channels].reverse();
    const reversedHeads = [...heads].reverse();
    const reversedLayers = [...layersPerBlock].reverse();
    const reversedCross = [...crossDims].reverse();
    const reversedTransformer = (c.reverse_transformer_layers_per_block as (number | number[])[] | null) ?? [...transformerLayers].reverse();
    const reversedOnlyCross = [...onlyCross].reverse();
    let upsamplers = 0;
    output = reversedChannels[0]!;
    up.forEach((type, index) => {
      const final = index === channels.length - 1;
      const previous = output;
      output = reversedChannels[index]!;
      const input = reversedChannels[Math.min(index + 1, channels.length - 1)]!;
      if (!final) upsamplers += 1;
      const args: BlockArgs = {
        ...base, layers: reversedLayers[index]! + 1, transformerLayers: reversedTransformer[index]!, inChannels: input, outChannels: output,
        prevOutputChannel: previous, resample: !final, crossAttentionDim: reversedCross[index]!, heads: reversedHeads[index]!,
        onlyCrossAttention: reversedOnlyCross[index]!,
      };
      const name = type.startsWith('UNetRes') ? type.slice(7) : type;
      if (name === 'CrossAttnUpBlock2D') this.up_blocks.append(new CrossAttnUpBlock2D(args));
      else if (name === 'UpBlock2D') this.up_blocks.append(new UpBlock2D(args));
      else throw new NotImplementedError(`diffusers up block ${type} is not available in TypeScript`);
    });
    this.numUpsamplers = upsamplers;
    this.conv_norm_out = this.registerModule('conv_norm_out', new GroupNorm(groups, channels[0]!, { eps: num(c, 'norm_eps') }));
    this.conv_act = this.registerModule('conv_act', activation(act));
    this.conv_out = this.registerModule('conv_out', new Conv2d(channels[0]!, int(c, 'out_channels'), outKernel, { padding: Math.floor((outKernel - 1) / 2) }));
  }

  /**
   * ``unet(sample, timestep, encoder_hidden_states, encoder_attention_mask).sample``.
   * ``timestep`` is an int64 scalar or ``[batch]`` tensor; the mask is boolean ``[batch, keys]``.
   */
  forward(sample: Tensor, timestep: Tensor, encoderHiddenStates: Tensor, encoderAttentionMask: Tensor | null = null): Tensor {
    const factor = 2 ** this.numUpsamplers;
    const forwardSize = sample.shape.slice(-2).some((size) => size % factor !== 0);
    const bias = encoderAttentionMask
      ? encoderAttentionMask.to(sample.dtype).neg().add(1).mul(-10000).unsqueeze(1)
      : null;
    let input = sample;
    if (this.config.center_input_sample === true) input = input.mul(2).sub(1);
    const steps = timestep.ndim === 0 ? timestep.reshape(1) : timestep;
    const expanded = steps.shape[0] === sample.shape[0] ? steps : steps.expand(sample.shape[0]!);
    let emb = this.time_embedding.forward(this.time_proj.forward(expanded).to(sample.dtype));
    if (this.time_embed_act) emb = this.time_embed_act.forward(emb);
    let states = this.conv_in.forward(input);
    let residuals: Tensor[] = [states];
    for (const block of this.down_blocks) {
      const [next, outputs] = block.forward(states, emb, encoderHiddenStates, bias);
      states = next;
      residuals.push(...outputs);
    }
    if (this.mid_block) states = this.mid_block.forward(states, emb, encoderHiddenStates, bias);
    [...this.up_blocks].forEach((block, index) => {
      const final = index === this.up_blocks.length - 1;
      const count = block.resnets.length;
      const current = residuals.slice(residuals.length - count);
      residuals = residuals.slice(0, residuals.length - count);
      const size = !final && forwardSize ? residuals[residuals.length - 1]!.shape.slice(2) as [number, number] : null;
      states = block.forward(states, current, emb, encoderHiddenStates, bias, size);
    });
    if (this.conv_norm_out && this.conv_act) states = this.conv_act.forward(this.conv_norm_out.forward(states));
    return this.conv_out.forward(states);
  }
}

// ---------------------------------------------------------------------------
// AutoencoderKL.
// ---------------------------------------------------------------------------

class DownEncoderBlock2D extends Module {
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Downsample2D> | null;

  constructor(layers: number, input: number, output: number, groups: number, act: string, downsample: boolean) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(Array.from({ length: layers }, (_, index) => new ResnetBlock2D({
      inChannels: index === 0 ? input : output, outChannels: output, tembChannels: null, eps: 1e-6, groups, dropout: 0,
      timeEmbeddingNorm: 'default', nonLinearity: act, outputScaleFactor: 1,
    }))));
    this.downsamplers = downsample ? this.registerModule('downsamplers', new ModuleList([new Downsample2D(output, 0)])) : null;
  }

  forward(hidden: Tensor): Tensor {
    let states = hidden;
    for (const block of this.resnets) states = block.forward(states, null);
    if (this.downsamplers) for (const downsampler of this.downsamplers) states = downsampler.forward(states);
    return states;
  }
}

class UpDecoderBlock2D extends Module {
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Upsample2D> | null;

  constructor(layers: number, input: number, output: number, groups: number, act: string, upsample: boolean) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(Array.from({ length: layers }, (_, index) => new ResnetBlock2D({
      inChannels: index === 0 ? input : output, outChannels: output, tembChannels: null, eps: 1e-6, groups, dropout: 0,
      timeEmbeddingNorm: 'group', nonLinearity: act, outputScaleFactor: 1,
    }))));
    this.upsamplers = upsample ? this.registerModule('upsamplers', new ModuleList([new Upsample2D(output)])) : null;
  }

  forward(hidden: Tensor): Tensor {
    let states = hidden;
    for (const block of this.resnets) states = block.forward(states, null);
    if (this.upsamplers) for (const upsampler of this.upsamplers) states = upsampler.forward(states);
    return states;
  }
}

/** ``UNetMidBlock2D`` of the VAE (no time embedding; single-head spatial attention). */
class UNetMidBlock2D extends Module {
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<ResnetBlock2D>;

  constructor(channels: number, groups: number, act: string, addAttention: boolean) {
    super();
    const block = (): ResnetBlock2D => new ResnetBlock2D({
      inChannels: channels, outChannels: channels, tembChannels: null, eps: 1e-6, groups, dropout: 0,
      timeEmbeddingNorm: 'default', nonLinearity: act, outputScaleFactor: 1,
    });
    const resnets = [block()];
    const attentions: DiffusersAttention[] = [];
    if (addAttention) {
      attentions.push(new DiffusersAttention({
        queryDim: channels, heads: 1, dimHead: channels, rescaleOutputFactor: 1, eps: 1e-6, groupNormGroups: groups,
        residualConnection: true, bias: true,
      }));
    }
    resnets.push(block());
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
  }

  forward(hidden: Tensor): Tensor {
    let states = this.resnets.at(0).forward(hidden, null);
    if (this.attentions.length) states = this.attentions.at(0).forward(states);
    return this.resnets.at(1).forward(states, null);
  }
}

class VaeEncoder extends Module {
  readonly conv_in: Conv2d;
  readonly down_blocks: ModuleList<DownEncoderBlock2D>;
  readonly mid_block: UNetMidBlock2D;
  readonly conv_norm_out: GroupNorm;
  readonly conv_act: SiLU;
  readonly conv_out: Conv2d;

  constructor(c: JsonObject) {
    super();
    const channels = c.block_out_channels as number[];
    const groups = int(c, 'norm_num_groups');
    const act = String(c.act_fn);
    this.conv_in = this.registerModule('conv_in', new Conv2d(int(c, 'in_channels'), channels[0]!, 3, { padding: 1 }));
    this.down_blocks = this.registerModule('down_blocks', new ModuleList());
    let output = channels[0]!;
    (c.down_block_types as string[]).forEach((type, index) => {
      if (type !== 'DownEncoderBlock2D') throw new NotImplementedError(`diffusers VAE down block ${type} is not available in TypeScript`);
      const input = output;
      output = channels[index]!;
      this.down_blocks.append(new DownEncoderBlock2D(int(c, 'layers_per_block'), input, output, groups, act, index !== channels.length - 1));
    });
    this.mid_block = this.registerModule('mid_block', new UNetMidBlock2D(channels[channels.length - 1]!, groups, act, c.mid_block_add_attention !== false));
    this.conv_norm_out = this.registerModule('conv_norm_out', new GroupNorm(groups, channels[channels.length - 1]!, { eps: 1e-6 }));
    this.conv_act = this.registerModule('conv_act', new SiLU());
    this.conv_out = this.registerModule('conv_out', new Conv2d(channels[channels.length - 1]!, 2 * int(c, 'latent_channels'), 3, { padding: 1 }));
  }

  forward(sample: Tensor): Tensor {
    let states = this.conv_in.forward(sample);
    for (const block of this.down_blocks) states = block.forward(states);
    states = this.mid_block.forward(states);
    return this.conv_out.forward(this.conv_act.forward(this.conv_norm_out.forward(states)));
  }
}

class VaeDecoder extends Module {
  readonly conv_in: Conv2d;
  readonly up_blocks: ModuleList<UpDecoderBlock2D>;
  readonly mid_block: UNetMidBlock2D;
  readonly conv_norm_out: GroupNorm;
  readonly conv_act: SiLU;
  readonly conv_out: Conv2d;

  constructor(c: JsonObject) {
    super();
    const channels = c.block_out_channels as number[];
    const groups = int(c, 'norm_num_groups');
    const act = String(c.act_fn);
    const last = channels[channels.length - 1]!;
    this.conv_in = this.registerModule('conv_in', new Conv2d(int(c, 'latent_channels'), last, 3, { padding: 1 }));
    // diffusers registers ``up_blocks`` before ``mid_block`` (state-dict order).
    this.up_blocks = this.registerModule('up_blocks', new ModuleList());
    this.mid_block = this.registerModule('mid_block', new UNetMidBlock2D(last, groups, act, c.mid_block_add_attention !== false));
    const reversed = [...channels].reverse();
    let output = reversed[0]!;
    (c.up_block_types as string[]).forEach((type, index) => {
      if (type !== 'UpDecoderBlock2D') throw new NotImplementedError(`diffusers VAE up block ${type} is not available in TypeScript`);
      const previous = output;
      output = reversed[index]!;
      this.up_blocks.append(new UpDecoderBlock2D(int(c, 'layers_per_block') + 1, previous, output, groups, act, index !== channels.length - 1));
    });
    this.conv_norm_out = this.registerModule('conv_norm_out', new GroupNorm(groups, channels[0]!, { eps: 1e-6 }));
    this.conv_act = this.registerModule('conv_act', new SiLU());
    this.conv_out = this.registerModule('conv_out', new Conv2d(channels[0]!, int(c, 'out_channels'), 3, { padding: 1 }));
  }

  forward(sample: Tensor): Tensor {
    let states = this.mid_block.forward(this.conv_in.forward(sample));
    for (const block of this.up_blocks) states = block.forward(states);
    return this.conv_out.forward(this.conv_act.forward(this.conv_norm_out.forward(states)));
  }
}

export class AutoencoderKL extends Module {
  static override readonly qualifiedName: string = 'diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL';
  readonly config: JsonObject;
  readonly encoder: VaeEncoder;
  readonly decoder: VaeDecoder;
  readonly quant_conv: Conv2d | null;
  readonly post_quant_conv: Conv2d | null;

  constructor(config: JsonObject) {
    super();
    this.config = resolveDiffusersConfig('AutoencoderKL', config);
    const c = this.config;
    const latent = int(c, 'latent_channels');
    this.encoder = this.registerModule('encoder', new VaeEncoder(c));
    this.decoder = this.registerModule('decoder', new VaeDecoder(c));
    this.quant_conv = c.use_quant_conv !== false ? this.registerModule('quant_conv', new Conv2d(2 * latent, 2 * latent, 1)) : null;
    this.post_quant_conv = c.use_post_quant_conv !== false ? this.registerModule('post_quant_conv', new Conv2d(latent, latent, 1)) : null;
  }

  /** The posterior mean (``encode(x).latent_dist.mode()``). */
  encodeMode(sample: Tensor): Tensor {
    let moments = this.encoder.forward(sample);
    if (this.quant_conv) moments = this.quant_conv.forward(moments);
    return moments.chunk(2, 1)[0]!;
  }

  /** ``decode(z).sample``. */
  decode(latents: Tensor): Tensor {
    const z = this.post_quant_conv ? this.post_quant_conv.forward(latents) : latents;
    return this.decoder.forward(z);
  }
}

// ---------------------------------------------------------------------------
// DDIMScheduler (float32 arithmetic, as PyTorch).
// ---------------------------------------------------------------------------

const f32 = Math.fround;

/** ``torch.linspace(start, end, steps, dtype=torch.float32)``. */
function linspace32(start: number, end: number, steps: number): number[] {
  const s = f32(start);
  const e = f32(end);
  if (steps === 1) return [s];
  const step = f32(f32(e - s) / f32(steps - 1));
  const halfway = Math.floor(steps / 2);
  return Array.from({ length: steps }, (_, index) => (index < halfway ? f32(s + f32(step * index)) : f32(e - f32(step * (steps - index - 1)))));
}

/** ``torch.cumprod`` on float32 CPU tensors (accumulated in double, stored in float32). */
function cumprod32(values: readonly number[]): number[] {
  let accumulator = 1;
  return values.map((value) => {
    accumulator *= value;
    return f32(accumulator);
  });
}

function betasForAlphaBar(count: number, maxBeta = 0.999): number[] {
  const alphaBar = (t: number): number => Math.cos((t + 0.008) / 1.008 * Math.PI / 2) ** 2;
  return Array.from({ length: count }, (_, index) => f32(Math.min(1 - alphaBar((index + 1) / count) / alphaBar(index / count), maxBeta)));
}

function rescaleZeroTerminalSnr(betas: number[]): number[] {
  const alphas = betas.map((beta) => f32(1 - beta));
  const sqrt = cumprod32(alphas).map((value) => f32(Math.sqrt(value)));
  const first = sqrt[0]!;
  const last = sqrt[sqrt.length - 1]!;
  const shifted = sqrt.map((value) => f32(value - last));
  const factor = f32(first / f32(first - last));
  const bars = shifted.map((value) => f32(value * factor)).map((value) => f32(value * value));
  const rescaled = [bars[0]!, ...bars.slice(1).map((value, index) => f32(value / bars[index]!))];
  return rescaled.map((alpha) => f32(1 - alpha));
}

/** Round half to even (``numpy.round``). */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const difference = value - floor;
  if (difference > 0.5) return floor + 1;
  if (difference < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

export class DDIMScheduler {
  readonly config: JsonObject;
  readonly betas: number[];
  readonly alphasCumprod: number[];
  readonly finalAlphaCumprod: number;
  readonly initNoiseSigma = 1;
  numInferenceSteps: number | null = null;
  timesteps: number[];

  constructor(config: JsonObject) {
    this.config = resolveDiffusersConfig('DDIMScheduler', config);
    const c = this.config;
    const count = int(c, 'num_train_timesteps');
    let betas: number[];
    if (c.trained_betas !== null && c.trained_betas !== undefined) {
      if (!Array.isArray(c.trained_betas)) throw new ValueError('trained_betas must be a list');
      betas = (c.trained_betas as number[]).map((value) => f32(value));
    } else if (c.beta_schedule === 'linear') {
      betas = linspace32(num(c, 'beta_start'), num(c, 'beta_end'), count);
    } else if (c.beta_schedule === 'scaled_linear') {
      betas = linspace32(num(c, 'beta_start') ** 0.5, num(c, 'beta_end') ** 0.5, count).map((value) => f32(value * value));
    } else if (c.beta_schedule === 'squaredcos_cap_v2') {
      betas = betasForAlphaBar(count);
    } else {
      throw new NotImplementedError(`${String(c.beta_schedule)} is not implemented for DDIMScheduler`);
    }
    if (c.rescale_betas_zero_snr === true) betas = rescaleZeroTerminalSnr(betas);
    this.betas = betas;
    this.alphasCumprod = cumprod32(betas.map((beta) => f32(1 - beta)));
    this.finalAlphaCumprod = c.set_alpha_to_one === false ? this.alphasCumprod[0]! : 1;
    this.timesteps = Array.from({ length: count }, (_, index) => count - 1 - index);
  }

  get predictionType(): string {
    return String(this.config.prediction_type);
  }

  setTimesteps(steps: number): void {
    const train = int(this.config, 'num_train_timesteps');
    if (steps > train) {
      throw new ValueError(`\`num_inference_steps\`: ${steps} cannot be larger than \`self.config.train_timesteps\`: ${train} as the unet model trained with this scheduler can only handle maximal ${train} timesteps.`);
    }
    this.numInferenceSteps = steps;
    const spacing = this.config.timestep_spacing;
    if (spacing === 'linspace') {
      const values = Array.from({ length: steps }, (_, index) => (steps === 1 ? 0 : index * (train - 1) / (steps - 1)));
      this.timesteps = values.map(roundHalfEven).reverse();
    } else if (spacing === 'leading') {
      const ratio = Math.floor(train / steps);
      const offset = int(this.config, 'steps_offset');
      this.timesteps = Array.from({ length: steps }, (_, index) => index * ratio).reverse().map((value) => value + offset);
    } else if (spacing === 'trailing') {
      const ratio = train / steps;
      const values: number[] = [];
      for (let value = train; value > 0; value -= ratio) values.push(roundHalfEven(value) - 1);
      this.timesteps = values;
    } else {
      throw new ValueError(`${String(spacing)} is not supported. Please make sure to choose one of 'leading' or 'trailing'.`);
    }
  }

  private threshold(sample: Tensor): Tensor {
    const [batch] = sample.shape as [number];
    const flat = sample.reshape(batch, -1);
    const ratio = num(this.config, 'dynamic_thresholding_ratio');
    const maximum = num(this.config, 'sample_max_value');
    const rows = noGrad(() => flat.abs().sort(1).values.toArray());
    const width = flat.shape[1]!;
    const limits = Array.from({ length: batch }, (_, row) => {
      // ``torch.quantile`` with linear interpolation.
      const position = ratio * (width - 1);
      const low = Math.floor(position);
      const high = Math.min(low + 1, width - 1);
      const values = rows.slice(row * width, (row + 1) * width);
      const quantile = f32(values[low]! + (values[high]! - values[low]!) * (position - low));
      return Math.min(Math.max(quantile, 1), maximum);
    });
    const s = tensor(limits, { shape: [batch, 1], dtype: sample.dtype });
    return flat.maximum(s.neg()).minimum(s).div(s).reshape(sample.shape);
  }

  /** ``step(model_output, timestep, sample, eta=0).prev_sample``. */
  step(modelOutput: Tensor, timestep: number, sample: Tensor): Tensor {
    if (this.numInferenceSteps === null) throw new ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler");
    const previous = timestep - Math.floor(int(this.config, 'num_train_timesteps') / this.numInferenceSteps);
    const alphaT = this.alphasCumprod[timestep]!;
    const alphaPrev = previous >= 0 ? this.alphasCumprod[previous]! : this.finalAlphaCumprod;
    const betaT = f32(1 - alphaT);
    const sqrtAlpha = f32(Math.sqrt(alphaT));
    const sqrtBeta = f32(Math.sqrt(betaT));
    let original: Tensor;
    let epsilon: Tensor;
    switch (this.predictionType) {
      case 'epsilon':
        original = sample.sub(modelOutput.mul(sqrtBeta)).div(sqrtAlpha);
        epsilon = modelOutput;
        break;
      case 'sample':
        original = modelOutput;
        epsilon = sample.sub(original.mul(sqrtAlpha)).div(sqrtBeta);
        break;
      case 'v_prediction':
        original = sample.mul(sqrtAlpha).sub(modelOutput.mul(sqrtBeta));
        epsilon = modelOutput.mul(sqrtAlpha).add(sample.mul(sqrtBeta));
        break;
      default:
        throw new ValueError(`prediction_type given as ${this.predictionType} must be one of \`epsilon\`, \`sample\`, or \`v_prediction\``);
    }
    if (this.config.thresholding === true) original = this.threshold(original);
    else if (this.config.clip_sample === true) {
      const range = num(this.config, 'clip_sample_range');
      original = original.clamp(-range, range);
    }
    const direction = epsilon.mul(f32(Math.sqrt(f32(1 - alphaPrev))));
    return original.mul(f32(Math.sqrt(alphaPrev))).add(direction);
  }

  private alphaFactors(timesteps: Tensor, ndim: number): [Tensor, Tensor] {
    const values = Array.from(timesteps.toArray());
    const shape = [values.length, ...Array.from({ length: ndim - 1 }, () => 1)];
    const alpha = values.map((t) => f32(Math.sqrt(this.alphasCumprod[t]!)));
    const beta = values.map((t) => f32(Math.sqrt(f32(1 - this.alphasCumprod[t]!))));
    return [tensor(alpha, { shape, dtype: 'float32' }), tensor(beta, { shape, dtype: 'float32' })];
  }

  addNoise(original: Tensor, noise: Tensor, timesteps: Tensor): Tensor {
    const [alpha, beta] = this.alphaFactors(timesteps, original.ndim);
    return alpha.to(original.dtype).mul(original).add(beta.to(original.dtype).mul(noise));
  }

  getVelocity(sample: Tensor, noise: Tensor, timesteps: Tensor): Tensor {
    const [alpha, beta] = this.alphaFactors(timesteps, sample.ndim);
    return alpha.to(sample.dtype).mul(noise).sub(beta.to(sample.dtype).mul(sample));
  }
}

/** Legacy VAE attention parameter names (diffusers ``_convert_deprecated_attention_blocks``). */
export function convertDeprecatedAttentionKey(key: string): string {
  return key.replace(/(mid_block\.attentions\.\d+)\.query\./, '$1.to_q.')
    .replace(/(mid_block\.attentions\.\d+)\.key\./, '$1.to_k.')
    .replace(/(mid_block\.attentions\.\d+)\.value\./, '$1.to_v.')
    .replace(/(mid_block\.attentions\.\d+)\.proj_attn\./, '$1.to_out.0.');
}

/** A float32 ``torch.arange``-style int64 helper for timesteps. */
export function timestepTensor(values: readonly number[]): Tensor {
  return tensor([...values], { dtype: 'int64' });
}

