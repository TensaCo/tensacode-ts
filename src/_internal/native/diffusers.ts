/**
 * diffusers ``UNet2DConditionModel``, ``AutoencoderKL`` and ``DDIMScheduler``
 * (diffusers 0.40 parameter names, module order and numerics) for the latent
 * diffusion ``ImageDecoder``.
 *
 * Every block type these two models build is ported with diffusers' module
 * tree: UNet down blocks ``DownBlock2D``, ``ResnetDownsampleBlock2D``,
 * ``AttnDownBlock2D``, ``CrossAttnDownBlock2D``, ``SimpleCrossAttnDownBlock2D``,
 * ``KDownBlock2D`` and ``KCrossAttnDownBlock2D``; the matching up blocks; mid
 * blocks ``UNetMidBlock2DCrossAttn``, ``UNetMidBlock2DSimpleCrossAttn``,
 * ``UNetMidBlock2D`` (or none); positional and Gaussian Fourier time
 * embeddings; ``default``/``scale_shift`` ResNet time conditioning and the
 * ``ada_group`` (``AdaGroupNorm``) and ``spatial`` conditional norms; the
 * ``silu``/``swish``/``mish``/``gelu``/``relu`` activations; VAE blocks
 * ``DownEncoderBlock2D``, ``AttnDownEncoderBlock2D``, ``UpDecoderBlock2D`` and
 * ``AttnUpDecoderBlock2D``. Block types that diffusers constructs but cannot
 * run inside these models (skip blocks, encoder blocks inside a UNet, UNet
 * blocks inside a VAE) raise ``ValueError`` when constructed. UNet
 * conditioning inputs that ``ImageDecoder`` rejects (class, addition and
 * encoder projections, timestep conditions, dual or gated attention) raise
 * ``ValueError('unsupported diffusion pipeline: ...')`` like Python's
 * ``ImageDecoder``.
 */
import { NotImplementedError, ValueError } from '../../errors.js';
import { noGrad } from '../../nn/autograd.js';
import { cat, interpolateNearest, softplus, stack, tanh } from '../../nn/functional.js';
import { Conv2d, Dropout, GELU, GroupNorm, LayerNorm, Linear, ModuleList, ReLU, SiLU } from '../../nn/layers.js';
import { Module } from '../../nn/module.js';
import { conv2d, gelu, groupNorm } from '../../nn/ops/nn.js';
import { Parameter, Tensor, randn, tensor, zeros } from '../../nn/tensor.js';
import { deepCopy, isPlainObject, orderedEntries, transferPythonNumberKind, type JsonObject, type JsonValue } from '../json.js';
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
  for (const [key, value] of orderedEntries(supplied as JsonObject)) {
    if (key.startsWith('_')) continue;
    result[key] = deepCopy(value as JsonValue);
    transferPythonNumberKind(result, key, supplied as JsonObject, key);
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

/** Python ``repr`` of a JSON value in diffusers' messages. */
function pyRepr(value: JsonValue | undefined): string {
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'string') return `'${value.replaceAll('\\', '\\\\').replaceAll("'", "\\'")}'`;
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(', ')}]`;
  if (typeof value === 'number') return String(value);
  return JSON.stringify(value);
}

type Act = Module & { forward(x: Tensor): Tensor };

/** ``torch.nn.Mish``: ``x * tanh(softplus(x))``. */
export class Mish extends Module {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.Mish';

  override configurationAttributes(): Record<string, unknown> {
    return { inplace: false };
  }

  forward(input: Tensor): Tensor {
    return input.mul(tanh(softplus(input)));
  }
}

/** ``torch.nn.AvgPool2d(kernel_size=2, stride=2)``. */
export class AvgPool2d extends Module {
  static override readonly qualifiedName: string = 'torch.nn.modules.pooling.AvgPool2d';

  constructor(readonly kernelSize: number, readonly stride: number) {
    super();
  }

  override configurationAttributes(): Record<string, unknown> {
    return { ceil_mode: false, count_include_pad: true, divisor_override: null, kernel_size: this.kernelSize, padding: 0, stride: this.stride };
  }

  forward(input: Tensor): Tensor {
    const [batch, channels, height, width] = input.shape as [number, number, number, number];
    const k = this.kernelSize;
    const outH = Math.floor((height - k) / this.stride) + 1;
    const outW = Math.floor((width - k) / this.stride) + 1;
    if (this.stride !== k) throw new ValueError('AvgPool2d here pools non-overlapping windows (stride equals kernel size)');
    const cropped = input.slice(2, 0, outH * k).slice(3, 0, outW * k);
    return cropped.reshape(batch, channels, outH, k, outW, k).mean([3, 5]);
  }
}

/** diffusers ``get_activation`` (case-insensitive). */
function activation(name: string): Act {
  switch (String(name).toLowerCase()) {
    case 'silu': case 'swish': return new SiLU();
    case 'mish': return new Mish();
    case 'gelu': return new GELU();
    case 'relu': return new ReLU();
    default: throw new ValueError(`activation function ${String(name).toLowerCase()} not found in ACT2FN mapping ['swish', 'silu', 'mish', 'gelu', 'relu']`);
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

/**
 * ``GaussianFourierProjection(embedding_size, set_W_to_weight=False, log=False)``:
 * a fixed random ``weight`` (``randn(embedding_size) * scale``, not trained).
 */
class GaussianFourierProjection extends Module {
  readonly weight: Parameter;

  constructor(embeddingSize: number, readonly flipSinToCos: boolean, readonly log = false, scale = 1) {
    super();
    const initial = noGrad(() => randn([embeddingSize]).mul(scale));
    this.weight = this.registerParameter('weight', new Parameter(initial, false));
  }

  forward(timesteps: Tensor): Tensor {
    const f = Math.fround;
    const pi = f(Math.PI);
    const weights = Array.from(this.weight.toArray(), (value) => f(value));
    const values: number[] = [];
    for (const raw of timesteps.toArray()) {
      const step = this.log ? f(Math.log(f(raw))) : f(raw);
      const projected = weights.map((weight) => f(f(f(step * weight) * 2) * pi));
      const sin = projected.map((angle) => f(Math.sin(angle)));
      const cos = projected.map((angle) => f(Math.cos(angle)));
      values.push(...(this.flipSinToCos ? [...cos, ...sin] : [...sin, ...cos]));
    }
    return tensor(values, { shape: [timesteps.numel, 2 * weights.length], dtype: 'float32' });
  }
}

class TimestepEmbedding extends Module {
  readonly linear_1: Linear;
  readonly act: Act;
  readonly linear_2: Linear;
  readonly post_act: Act | null;

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
// Normalization, resampling.
// ---------------------------------------------------------------------------

/** ``AdaGroupNorm``: group norm whose scale and shift come from the time embedding. */
class AdaGroupNorm extends Module {
  readonly act: Act | null;
  readonly linear: Linear;

  constructor(embeddingDim: number, outDim: number, readonly numGroups: number, act: string | null = null, readonly eps = 1e-5) {
    super();
    this.act = act === null ? null : this.registerModule('act', activation(act));
    this.linear = this.registerModule('linear', new Linear(embeddingDim, outDim * 2));
  }

  forward(x: Tensor, emb: Tensor): Tensor {
    let embedding = this.act ? this.act.forward(emb) : emb;
    embedding = this.linear.forward(embedding);
    embedding = embedding.reshape(embedding.shape[0]!, embedding.shape[1]!, 1, 1);
    const [scale, shift] = embedding.chunk(2, 1) as [Tensor, Tensor];
    return groupNorm(x, this.numGroups, null, null, this.eps).mul(scale.add(1)).add(shift);
  }
}

/** ``SpatialNorm``: group norm modulated by a spatial conditioning map (MoVQ). */
class SpatialNorm extends Module {
  readonly norm_layer: GroupNorm;
  readonly conv_y: Conv2d;
  readonly conv_b: Conv2d;

  constructor(fChannels: number, zqChannels: number) {
    super();
    this.norm_layer = this.registerModule('norm_layer', new GroupNorm(32, fChannels, { eps: 1e-6 }));
    this.conv_y = this.registerModule('conv_y', new Conv2d(zqChannels, fChannels, 1));
    this.conv_b = this.registerModule('conv_b', new Conv2d(zqChannels, fChannels, 1));
  }

  forward(f: Tensor, zq: Tensor | null): Tensor {
    if (zq === null || zq.ndim !== 4) {
      throw new ValueError('SpatialNorm conditioning must be a [batch, channels, height, width] map (F.interpolate input)');
    }
    const resized = interpolateNearest(zq, [f.shape[2]!, f.shape[3]!]);
    return this.norm_layer.forward(f).mul(this.conv_y.forward(resized)).add(this.conv_b.forward(resized));
  }
}

/** ``Downsample2D``: a stride-2 convolution (``padding=0`` pads right/bottom first) or 2x2 average pooling. */
class Downsample2D extends Module {
  readonly conv: Conv2d | AvgPool2d;

  constructor(channels: number, readonly useConv: boolean, readonly padding: number, outChannels: number = channels) {
    super();
    if (!useConv && outChannels !== channels) throw new ValueError('Downsample2D without a convolution keeps the channel count');
    this.conv = this.registerModule('conv', useConv
      ? new Conv2d(channels, outChannels, 3, { stride: 2, padding })
      : new AvgPool2d(2, 2));
  }

  forward(hidden: Tensor): Tensor {
    let input = hidden;
    if (this.useConv && this.padding === 0) {
      const [batch, channels, height, width] = input.shape as [number, number, number, number];
      input = cat([input, zeros([batch, channels, height, 1], { dtype: input.dtype })], 3);
      input = cat([input, zeros([batch, channels, 1, width + 1], { dtype: input.dtype })], 2);
    }
    return this.conv.forward(input);
  }
}

/** ``Upsample2D``: nearest 2x (or to ``outputSize``), then an optional 3x3 convolution. */
class Upsample2D extends Module {
  readonly conv: Conv2d | null;

  constructor(channels: number, useConv: boolean) {
    super();
    this.conv = useConv ? this.registerModule('conv', new Conv2d(channels, channels, 3, { padding: 1 })) : null;
  }

  forward(hidden: Tensor, size: readonly [number, number] | null = null): Tensor {
    const target: [number, number] = size ? [size[0], size[1]] : [hidden.shape[2]! * 2, hidden.shape[3]! * 2];
    const upsampled = interpolateNearest(hidden, target);
    return this.conv ? this.conv.forward(upsampled) : upsampled;
  }
}

/** ``F.pad(x, (p, p, p, p), mode='reflect')`` for ``[N, C, H, W]``. */
function reflectPad(x: Tensor, pad: number): Tensor {
  let result = x;
  for (const dim of [3, 2]) {
    const size = result.shape[dim]!;
    if (pad >= size) throw new ValueError(`Padding size should be less than the corresponding input dimension, but got: padding (${pad}, ${pad}) at dimension ${dim} of input ${result.ndim}`);
    const before = Array.from({ length: pad }, (_, index) => pad - index);
    const after = Array.from({ length: pad }, (_, index) => size - 2 - index);
    result = cat([result.indexSelect(dim, before), result, result.indexSelect(dim, after)], dim);
  }
  return result;
}

/** The dense ``[C, C, 4, 4]`` weight that applies ``kernel`` to each channel independently. */
function channelDiagonalWeight(channels: number, kernel: readonly number[], dtype: Tensor['dtype']): Tensor {
  const size = kernel.length;
  const values = new Float32Array(channels * channels * size * size);
  for (let c = 0; c < channels; c += 1) {
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) values[((c * channels + c) * size + y) * size + x] = Math.fround(kernel[y]! * kernel[x]!);
    }
  }
  return tensor(values, { shape: [channels, channels, size, size], dtype });
}

const K_KERNEL = [1 / 8, 3 / 8, 3 / 8, 1 / 8].map(Math.fround);

/** ``KDownsample2D``: reflect padding and a fixed ``[1, 3, 3, 1] / 8`` stride-2 filter. */
class KDownsample2D extends Module {
  constructor() {
    super();
    this.registerBuffer('kernel', tensor(K_KERNEL.flatMap((a) => K_KERNEL.map((b) => Math.fround(a * b))), { shape: [4, 4] }), false);
  }

  forward(inputs: Tensor): Tensor {
    const padded = reflectPad(inputs, 1);
    return conv2d(padded, channelDiagonalWeight(inputs.shape[1]!, K_KERNEL, inputs.dtype), null, { stride: 2 });
  }
}

/** ``KUpsample2D``: reflect padding and a fixed ``[1, 3, 3, 1] / 4`` stride-2 transposed filter. */
class KUpsample2D extends Module {
  constructor() {
    super();
    const kernel = K_KERNEL.map((value) => Math.fround(value * 2));
    this.registerBuffer('kernel', tensor(kernel.flatMap((a) => kernel.map((b) => Math.fround(a * b))), { shape: [4, 4] }), false);
  }

  forward(inputs: Tensor): Tensor {
    // conv_transpose2d(x, w, stride=2, padding=3) = conv2d(zero-interleaved x, flipped w, padding=0)
    // for a symmetric, channel-diagonal 4x4 kernel.
    const padded = reflectPad(inputs, 1);
    const [batch, channels, height, width] = padded.shape as [number, number, number, number];
    const zero = zeros(padded.shape, { dtype: padded.dtype });
    const wide = stack([padded, zero], 4).reshape(batch, channels, height, width * 2).slice(3, 0, width * 2 - 1);
    const zeroRows = zeros(wide.shape, { dtype: wide.dtype });
    const dilated = stack([wide, zeroRows], 3).reshape(batch, channels, height * 2, width * 2 - 1).slice(2, 0, height * 2 - 1);
    const kernel = K_KERNEL.map((value) => Math.fround(value * 2));
    return conv2d(dilated, channelDiagonalWeight(channels, kernel, inputs.dtype), null, { stride: 1 });
  }
}

// ---------------------------------------------------------------------------
// ResNet blocks.
// ---------------------------------------------------------------------------

interface ResnetOptions {
  inChannels: number;
  outChannels: number;
  tembChannels: number | null;
  eps: number;
  groups: number | null;
  groupsOut?: number | null;
  dropout: number;
  timeEmbeddingNorm: string;
  nonLinearity: string;
  outputScaleFactor: number;
  skipTimeAct?: boolean;
  up?: boolean;
  down?: boolean;
  convShortcutBias?: boolean;
  conv2dOutChannels?: number | null;
}

function groupCount(groups: number | null | undefined, where: string): number {
  if (typeof groups !== 'number') throw new TypeError(`${where}: GroupNorm needs an integer num_groups (norm_num_groups is None)`);
  return groups;
}

interface Resnet extends Module {
  forward(input: Tensor, temb: Tensor | null): Tensor;
}

export class ResnetBlock2D extends Module implements Resnet {
  readonly norm1: GroupNorm;
  readonly conv1: Conv2d;
  readonly time_emb_proj: Linear | null;
  readonly norm2: GroupNorm;
  readonly dropout: Dropout;
  readonly conv2: Conv2d;
  readonly nonlinearity: Act;
  readonly upsample: Upsample2D | null;
  readonly downsample: Downsample2D | null;
  readonly conv_shortcut: Conv2d | null;
  readonly timeEmbeddingNorm: string;
  readonly outputScaleFactor: number;
  readonly skipTimeAct: boolean;

  constructor(options: ResnetOptions) {
    super();
    const { inChannels, outChannels, tembChannels, eps } = options;
    if (options.timeEmbeddingNorm === 'ada_group' || options.timeEmbeddingNorm === 'spatial') {
      throw new ValueError(`This class cannot be used with \`time_embedding_norm==${options.timeEmbeddingNorm}\`, please use \`ResnetBlockCondNorm2D\` instead`);
    }
    this.timeEmbeddingNorm = options.timeEmbeddingNorm;
    this.outputScaleFactor = options.outputScaleFactor;
    this.skipTimeAct = options.skipTimeAct === true;
    const groups = groupCount(options.groups, 'ResnetBlock2D');
    const groupsOut = options.groupsOut ?? groups;
    this.norm1 = this.registerModule('norm1', new GroupNorm(groups, inChannels, { eps }));
    this.conv1 = this.registerModule('conv1', new Conv2d(inChannels, outChannels, 3, { padding: 1 }));
    if (tembChannels !== null) {
      if (this.timeEmbeddingNorm === 'default') this.time_emb_proj = this.registerModule('time_emb_proj', new Linear(tembChannels, outChannels));
      else if (this.timeEmbeddingNorm === 'scale_shift') this.time_emb_proj = this.registerModule('time_emb_proj', new Linear(tembChannels, 2 * outChannels));
      else throw new ValueError(`unknown time_embedding_norm : ${this.timeEmbeddingNorm} `);
    } else {
      this.time_emb_proj = null;
    }
    this.norm2 = this.registerModule('norm2', new GroupNorm(groupsOut, outChannels, { eps }));
    this.dropout = this.registerModule('dropout', new Dropout(options.dropout));
    const convOut = options.conv2dOutChannels ?? outChannels;
    this.conv2 = this.registerModule('conv2', new Conv2d(outChannels, convOut, 3, { padding: 1 }));
    this.nonlinearity = this.registerModule('nonlinearity', activation(options.nonLinearity));
    this.upsample = options.up ? this.registerModule('upsample', new Upsample2D(inChannels, false)) : null;
    this.downsample = !options.up && options.down ? this.registerModule('downsample', new Downsample2D(inChannels, false, 1)) : null;
    this.conv_shortcut = inChannels !== convOut
      ? this.registerModule('conv_shortcut', new Conv2d(inChannels, convOut, 1, { bias: options.convShortcutBias !== false }))
      : null;
  }

  forward(input: Tensor, temb: Tensor | null): Tensor {
    let residual = input;
    let hidden = this.nonlinearity.forward(this.norm1.forward(input));
    if (this.upsample) {
      residual = this.upsample.forward(residual);
      hidden = this.upsample.forward(hidden);
    } else if (this.downsample) {
      residual = this.downsample.forward(residual);
      hidden = this.downsample.forward(hidden);
    }
    hidden = this.conv1.forward(hidden);
    let time: Tensor | null = temb;
    if (this.time_emb_proj && temb) {
      time = this.time_emb_proj.forward(this.skipTimeAct ? temb : this.nonlinearity.forward(temb));
      time = time.reshape(time.shape[0]!, time.shape[1]!, 1, 1);
    }
    if (this.timeEmbeddingNorm === 'default') {
      if (time) hidden = hidden.add(time);
      hidden = this.norm2.forward(hidden);
    } else if (this.timeEmbeddingNorm === 'scale_shift') {
      if (!time) throw new ValueError(` \`temb\` should not be None when \`time_embedding_norm\` is ${this.timeEmbeddingNorm}`);
      const [scale, shift] = time.chunk(2, 1) as [Tensor, Tensor];
      hidden = this.norm2.forward(hidden).mul(scale.add(1)).add(shift);
    } else {
      hidden = this.norm2.forward(hidden);
    }
    hidden = this.conv2.forward(this.dropout.forward(this.nonlinearity.forward(hidden)));
    const shortcut = this.conv_shortcut ? this.conv_shortcut.forward(residual) : residual;
    const output = shortcut.add(hidden);
    return this.outputScaleFactor === 1 ? output : output.div(this.outputScaleFactor);
  }
}

/** ``ResnetBlockCondNorm2D``: a ResNet block whose norms take the time embedding (``ada_group`` or ``spatial``). */
export class ResnetBlockCondNorm2D extends Module implements Resnet {
  readonly norm1: AdaGroupNorm | SpatialNorm;
  readonly conv1: Conv2d;
  readonly norm2: AdaGroupNorm | SpatialNorm;
  readonly dropout: Dropout;
  readonly conv2: Conv2d;
  readonly nonlinearity: Act;
  readonly upsample: Upsample2D | null;
  readonly downsample: Downsample2D | null;
  readonly conv_shortcut: Conv2d | null;
  readonly outputScaleFactor: number;

  constructor(options: ResnetOptions) {
    super();
    const { inChannels, outChannels, eps } = options;
    const temb = options.tembChannels ?? 512;
    const groups = groupCount(options.groups, 'ResnetBlockCondNorm2D');
    const groupsOut = options.groupsOut ?? groups;
    this.outputScaleFactor = options.outputScaleFactor;
    const norm = (channels: number, count: number): AdaGroupNorm | SpatialNorm => {
      if (options.timeEmbeddingNorm === 'ada_group') return new AdaGroupNorm(temb, channels, count, null, eps);
      if (options.timeEmbeddingNorm === 'spatial') return new SpatialNorm(channels, temb);
      throw new ValueError(` unsupported time_embedding_norm: ${options.timeEmbeddingNorm}`);
    };
    this.norm1 = this.registerModule('norm1', norm(inChannels, groups));
    this.conv1 = this.registerModule('conv1', new Conv2d(inChannels, outChannels, 3, { padding: 1 }));
    this.norm2 = this.registerModule('norm2', norm(outChannels, groupsOut));
    this.dropout = this.registerModule('dropout', new Dropout(options.dropout));
    const convOut = options.conv2dOutChannels ?? outChannels;
    this.conv2 = this.registerModule('conv2', new Conv2d(outChannels, convOut, 3, { padding: 1 }));
    this.nonlinearity = this.registerModule('nonlinearity', activation(options.nonLinearity));
    this.upsample = options.up ? this.registerModule('upsample', new Upsample2D(inChannels, false)) : null;
    this.downsample = !options.up && options.down ? this.registerModule('downsample', new Downsample2D(inChannels, false, 1)) : null;
    this.conv_shortcut = inChannels !== convOut
      ? this.registerModule('conv_shortcut', new Conv2d(inChannels, convOut, 1, { bias: options.convShortcutBias !== false }))
      : null;
  }

  forward(input: Tensor, temb: Tensor | null): Tensor {
    let residual = input;
    let hidden = this.nonlinearity.forward(this.norm1.forward(input, temb!));
    if (this.upsample) {
      residual = this.upsample.forward(residual);
      hidden = this.upsample.forward(hidden);
    } else if (this.downsample) {
      residual = this.downsample.forward(residual);
      hidden = this.downsample.forward(hidden);
    }
    hidden = this.conv1.forward(hidden);
    hidden = this.nonlinearity.forward(this.norm2.forward(hidden, temb!));
    hidden = this.conv2.forward(this.dropout.forward(hidden));
    const shortcut = this.conv_shortcut ? this.conv_shortcut.forward(residual) : residual;
    const output = shortcut.add(hidden);
    return this.outputScaleFactor === 1 ? output : output.div(this.outputScaleFactor);
  }
}

// ---------------------------------------------------------------------------
// Attention.
// ---------------------------------------------------------------------------

interface AttentionOptions {
  queryDim: number;
  crossAttentionDim?: number | null;
  heads: number;
  dimHead: number;
  dropout?: number;
  bias?: boolean;
  crossAttentionNorm?: string | null;
  crossAttentionNormNumGroups?: number;
  addedKvProjDim?: number | null;
  normNumGroups?: number | null;
  spatialNormDim?: number | null;
  outBias?: boolean;
  onlyCrossAttention?: boolean;
  eps?: number;
  rescaleOutputFactor?: number;
  residualConnection?: boolean;
  /** ``AttnAddedKVProcessor2_0`` (simple cross-attention blocks) instead of ``AttnProcessor2_0``. */
  addedKvProcessor?: boolean;
}

/** diffusers ``Attention`` with ``AttnProcessor2_0`` or ``AttnAddedKVProcessor2_0``. */
class DiffusersAttention extends Module {
  readonly heads: number;
  readonly headDim: number;
  readonly residualConnection: boolean;
  readonly rescaleOutputFactor: number;
  readonly onlyCrossAttention: boolean;
  readonly addedKvProcessor: boolean;
  readonly group_norm: GroupNorm | null;
  readonly spatial_norm: SpatialNorm | null;
  readonly norm_cross: LayerNorm | GroupNorm | null;
  readonly to_q: Linear;
  readonly to_k: Linear | null;
  readonly to_v: Linear | null;
  readonly add_k_proj: Linear | null;
  readonly add_v_proj: Linear | null;
  readonly to_out: ModuleList<Module>;

  constructor(options: AttentionOptions) {
    super();
    const inner = options.dimHead * options.heads;
    const cross = options.crossAttentionDim ?? options.queryDim;
    const bias = options.bias ?? false;
    const added = options.addedKvProjDim ?? null;
    this.heads = options.heads;
    this.headDim = options.dimHead;
    this.residualConnection = options.residualConnection ?? false;
    this.rescaleOutputFactor = options.rescaleOutputFactor ?? 1;
    this.onlyCrossAttention = options.onlyCrossAttention ?? false;
    this.addedKvProcessor = options.addedKvProcessor ?? false;
    if (added === null && this.onlyCrossAttention) {
      throw new ValueError('`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`.');
    }
    this.group_norm = options.normNumGroups !== null && options.normNumGroups !== undefined
      ? this.registerModule('group_norm', new GroupNorm(options.normNumGroups, options.queryDim, { eps: options.eps ?? 1e-5 }))
      : null;
    this.spatial_norm = options.spatialNormDim !== null && options.spatialNormDim !== undefined
      ? this.registerModule('spatial_norm', new SpatialNorm(options.queryDim, options.spatialNormDim))
      : null;
    const crossNorm = options.crossAttentionNorm ?? null;
    if (crossNorm === null) this.norm_cross = null;
    else if (crossNorm === 'layer_norm') this.norm_cross = this.registerModule('norm_cross', new LayerNorm(cross));
    else if (crossNorm === 'group_norm') {
      this.norm_cross = this.registerModule('norm_cross', new GroupNorm(options.crossAttentionNormNumGroups ?? 32, added ?? cross, { eps: 1e-5 }));
    } else {
      throw new ValueError(`unknown cross_attention_norm: ${crossNorm}. Should be None, 'layer_norm' or 'group_norm'`);
    }
    this.to_q = this.registerModule('to_q', new Linear(options.queryDim, inner, { bias }));
    this.to_k = this.onlyCrossAttention ? null : this.registerModule('to_k', new Linear(cross, inner, { bias }));
    this.to_v = this.onlyCrossAttention ? null : this.registerModule('to_v', new Linear(cross, inner, { bias }));
    this.add_k_proj = added === null ? null : this.registerModule('add_k_proj', new Linear(added, inner));
    this.add_v_proj = added === null ? null : this.registerModule('add_v_proj', new Linear(added, inner));
    this.to_out = this.registerModule('to_out', new ModuleList<Module>([
      new Linear(inner, options.queryDim, { bias: options.outBias ?? true }), new Dropout(options.dropout ?? 0),
    ]));
  }

  private normCross(encoder: Tensor): Tensor {
    if (this.norm_cross instanceof LayerNorm) return this.norm_cross.forward(encoder);
    return this.norm_cross!.forward(encoder.transpose(1, 2)).transpose(1, 2);
  }

  /**
   * Python ``prepare_attention_mask`` for an additive ``[batch, 1, keys]``
   * mask: zero-padded by ``targetLength`` when its length differs, as a
   * ``[batch, 1, 1, keys']`` bias broadcast over heads.
   */
  private prepareMask(mask: Tensor | null, targetLength: number): Tensor | null {
    if (mask === null) return null;
    let prepared = mask;
    const current = mask.shape[mask.ndim - 1]!;
    if (current !== targetLength) {
      prepared = cat([prepared, zeros([...prepared.shape.slice(0, -1), targetLength], { dtype: prepared.dtype })], prepared.ndim - 1);
    }
    return prepared.unsqueeze(1);
  }

  private output(states: Tensor): Tensor {
    return (this.to_out.at(1) as Dropout).forward((this.to_out.at(0) as Linear).forward(states));
  }

  /** ``mask`` is the UNet's additive ``[batch, 1, keys]`` mask (``(1 - mask) * -10000``). */
  forward(hidden: Tensor, encoder: Tensor | null = null, mask: Tensor | null = null, temb: Tensor | null = null): Tensor {
    return this.addedKvProcessor ? this.addedKv(hidden, encoder, mask) : this.standard(hidden, encoder, mask, temb);
  }

  private standard(hidden: Tensor, encoder: Tensor | null, mask: Tensor | null, temb: Tensor | null): Tensor {
    const residual = hidden;
    let states = hidden;
    if (this.spatial_norm) states = this.spatial_norm.forward(states, temb);
    const spatial = states.ndim === 4;
    const [batch, channels, height, width] = states.shape as [number, number, number, number];
    if (spatial) states = states.reshape(batch, channels, height * width).transpose(1, 2);
    const sequence = (encoder ?? states).shape[1]!;
    const bias = this.prepareMask(mask, sequence);
    if (this.group_norm) states = this.group_norm.forward(states.transpose(1, 2)).transpose(1, 2);
    const query = this.to_q.forward(states);
    let context = encoder ?? states;
    if (encoder !== null && this.norm_cross) context = this.normCross(encoder);
    const q = splitHeads(query, this.heads);
    const k = splitHeads(this.to_k!.forward(context), this.heads);
    const v = splitHeads(this.to_v!.forward(context), this.heads);
    let output = this.output(mergeHeads(attention(q, k, v, { scale: (q.shape[3]!) ** -0.5, bias })));
    if (spatial) output = output.transpose(-1, -2).reshape(batch, channels, height, width);
    if (this.residualConnection) output = output.add(residual);
    return this.rescaleOutputFactor === 1 ? output : output.div(this.rescaleOutputFactor);
  }

  private addedKv(hidden: Tensor, encoder: Tensor | null, mask: Tensor | null): Tensor {
    const residual = hidden;
    const [batch, channels] = hidden.shape as [number, number];
    let states = hidden.reshape(batch, channels, -1).transpose(1, 2);
    const sequence = states.shape[1]!;
    const bias = this.prepareMask(mask, sequence);
    let context: Tensor;
    if (encoder === null) context = states;
    else context = this.norm_cross ? this.normCross(encoder) : encoder;
    states = this.group_norm!.forward(states.transpose(1, 2)).transpose(1, 2);
    const q = splitHeads(this.to_q.forward(states), this.heads);
    let k = splitHeads(this.add_k_proj!.forward(context), this.heads);
    let v = splitHeads(this.add_v_proj!.forward(context), this.heads);
    if (!this.onlyCrossAttention) {
      k = cat([k, splitHeads(this.to_k!.forward(states), this.heads)], 2);
      v = cat([v, splitHeads(this.to_v!.forward(states), this.heads)], 2);
    }
    if (bias && bias.shape[bias.ndim - 1] !== k.shape[2]) {
      throw new ValueError(`The size of tensor a (${k.shape[2]}) must match the size of tensor b (${bias.shape[bias.ndim - 1]}) at non-singleton dimension 3`);
    }
    const output = this.output(mergeHeads(attention(q, k, v, { scale: (q.shape[3]!) ** -0.5, bias })));
    return output.transpose(-1, -2).reshape(residual.shape).add(residual);
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

  forward(hidden: Tensor, encoder: Tensor, encoderMask: Tensor | null): Tensor {
    let states = this.attn1.forward(this.norm1.forward(hidden), this.onlyCrossAttention ? encoder : null, null).add(hidden);
    states = this.attn2.forward(this.norm2.forward(states), encoder, encoderMask).add(states);
    return this.ff.forward(this.norm3.forward(states)).add(states);
  }
}

class Transformer2DModel extends Module {
  readonly norm: GroupNorm;
  readonly proj_in: Linear | Conv2d;
  readonly transformer_blocks: ModuleList<BasicTransformerBlock>;
  readonly proj_out: Linear | Conv2d;

  constructor(heads: number, headDim: number, readonly channels: number, layers: number, crossAttentionDim: number,
    groups: number | null, readonly linearProjection: boolean, onlyCrossAttention: boolean, dropout: number) {
    super();
    const inner = heads * headDim;
    this.norm = this.registerModule('norm', new GroupNorm(groupCount(groups, 'Transformer2DModel'), channels, { eps: 1e-6 }));
    this.proj_in = this.registerModule('proj_in', linearProjection ? new Linear(channels, inner) : new Conv2d(channels, inner, 1));
    this.transformer_blocks = this.registerModule('transformer_blocks', new ModuleList(
      Array.from({ length: layers }, () => new BasicTransformerBlock(inner, heads, headDim, crossAttentionDim, dropout, onlyCrossAttention)),
    ));
    this.proj_out = this.registerModule('proj_out', linearProjection ? new Linear(inner, channels) : new Conv2d(inner, channels, 1));
  }

  forward(hidden: Tensor, encoder: Tensor, encoderMask: Tensor | null): Tensor {
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
    for (const block of this.transformer_blocks) states = block.forward(states, encoder, encoderMask);
    if (!this.linearProjection) {
      states = (this.proj_out as Conv2d).forward(states.reshape(batch, height, width, inner).permute(0, 3, 1, 2));
    } else {
      states = (this.proj_out as Linear).forward(states).reshape(batch, height, width, this.channels).permute(0, 3, 1, 2);
    }
    return states.add(residual);
  }
}

/** ``KAttentionBlock``: AdaGroupNorm-conditioned self- and cross-attention with residuals. */
class KAttentionBlock extends Module {
  readonly norm1: AdaGroupNorm | null;
  readonly attn1: DiffusersAttention | null;
  readonly norm2: AdaGroupNorm;
  readonly attn2: DiffusersAttention;

  constructor(dim: number, heads: number, headDim: number, options: {
    crossAttentionDim: number | null; tembChannels: number; bias: boolean; addSelfAttention: boolean;
    crossAttentionNorm: string | null; groupSize: number;
  }) {
    super();
    const groups = Math.max(1, Math.floor(dim / options.groupSize));
    if (options.addSelfAttention) {
      this.norm1 = this.registerModule('norm1', new AdaGroupNorm(options.tembChannels, dim, groups));
      this.attn1 = this.registerModule('attn1', new DiffusersAttention({ queryDim: dim, heads, dimHead: headDim, bias: options.bias }));
    } else {
      this.norm1 = null;
      this.attn1 = null;
    }
    this.norm2 = this.registerModule('norm2', new AdaGroupNorm(options.tembChannels, dim, groups));
    this.attn2 = this.registerModule('attn2', new DiffusersAttention({
      queryDim: dim, crossAttentionDim: options.crossAttentionDim, heads, dimHead: headDim, bias: options.bias,
      crossAttentionNorm: options.crossAttentionNorm,
    }));
  }

  private static to3d(hidden: Tensor): Tensor {
    const [batch, channels, height, width] = hidden.shape as [number, number, number, number];
    return hidden.permute(0, 2, 3, 1).reshape(batch, height * width, channels);
  }

  private static to4d(hidden: Tensor, height: number, width: number): Tensor {
    return hidden.permute(0, 2, 1).reshape(hidden.shape[0]!, -1, height, width);
  }

  forward(hidden: Tensor, encoder: Tensor | null, emb: Tensor, attentionMask: Tensor | null, encoderMask: Tensor | null): Tensor {
    let states = hidden;
    const [height, width] = [hidden.shape[2]!, hidden.shape[3]!];
    if (this.attn1 && this.norm1) {
      const normed = KAttentionBlock.to3d(this.norm1.forward(states, emb));
      states = KAttentionBlock.to4d(this.attn1.forward(normed, null, attentionMask), height, width).add(states);
    }
    const normed = KAttentionBlock.to3d(this.norm2.forward(states, emb));
    const output = this.attn2.forward(normed, encoder, encoder === null ? attentionMask : encoderMask);
    return KAttentionBlock.to4d(output, height, width).add(states);
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
  groups: number | null;
  crossAttentionDim: number | null;
  /** ``num_attention_heads`` for Transformer2DModel blocks. */
  heads: number;
  /** ``attention_head_dim`` for attention, simple cross-attention and K blocks. */
  headDim: number;
  downsamplePadding: number;
  linearProjection: boolean;
  onlyCrossAttention: boolean;
  timeScaleShift: string;
  dropout: number;
  skipTimeAct: boolean;
  outputScaleFactor: number;
  crossAttentionNorm: string | null;
}

/** Residuals are ``null`` where K blocks record no skip connection. */
type Residual = Tensor | null;

interface DownBlock extends Module {
  readonly hasCrossAttention: boolean;
  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, mask: Tensor | null): [Tensor, Residual[]];
}

interface UpBlock extends Module {
  readonly hasCrossAttention: boolean;
  readonly resnets: ModuleList<Module>;
  forward(hidden: Tensor, residuals: Residual[], temb: Tensor | null, encoder: Tensor, mask: Tensor | null, size: readonly [number, number] | null): Tensor;
}

function resnet(args: BlockArgs, inChannels: number, outChannels: number, extra: Partial<ResnetOptions> = {}): ResnetBlock2D {
  return new ResnetBlock2D({
    inChannels, outChannels, tembChannels: args.temb, eps: args.eps, groups: args.groups, dropout: args.dropout,
    timeEmbeddingNorm: args.timeScaleShift, nonLinearity: args.act, outputScaleFactor: 1, ...extra,
  });
}

function transformerLayers(args: BlockArgs): number[] {
  return typeof args.transformerLayers === 'number'
    ? Array.from({ length: args.layers }, () => args.transformerLayers as number)
    : args.transformerLayers;
}

function crossDim(args: BlockArgs, block: string): number {
  if (args.crossAttentionDim === null) throw new ValueError(`cross_attention_dim must be specified for ${block}`);
  return args.crossAttentionDim;
}

/** Attention modules diffusers creates with ``_from_deprecated_attn_block=True``. */
const DEPRECATED_ATTENTION = new WeakSet<DiffusersAttention>();

function deprecated(module: DiffusersAttention): DiffusersAttention {
  DEPRECATED_ATTENTION.add(module);
  return module;
}

/** A self-attention block over spatial positions; ``legacy`` marks ``_from_deprecated_attn_block`` attentions. */
function spatialAttention(channels: number, headDim: number, groups: number | null, eps: number, legacy: boolean): DiffusersAttention {
  const module = new DiffusersAttention({
    queryDim: channels, heads: Math.floor(channels / headDim), dimHead: headDim, eps,
    normNumGroups: groups, residualConnection: true, bias: true,
  });
  return legacy ? deprecated(module) : module;
}

function simpleCrossAttention(args: BlockArgs, channels: number, onlyCross: boolean): DiffusersAttention {
  return new DiffusersAttention({
    queryDim: channels, crossAttentionDim: channels, heads: Math.floor(channels / args.headDim), dimHead: args.headDim,
    addedKvProjDim: args.crossAttentionDim, normNumGroups: args.groups, bias: true, onlyCrossAttention: onlyCross,
    crossAttentionNorm: args.crossAttentionNorm, addedKvProcessor: true,
  });
}

function runDownsamplers(downsamplers: ModuleList<Module> | null, states: Tensor, temb: Tensor | null): Tensor {
  let result = states;
  if (downsamplers) {
    for (const downsampler of downsamplers) {
      result = downsampler instanceof ResnetBlock2D ? downsampler.forward(result, temb) : (downsampler as Downsample2D).forward(result);
    }
  }
  return result;
}

class CrossAttnDownBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = true;
  readonly attentions: ModuleList<Transformer2DModel>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    const cross = crossDim(args, 'CrossAttnDownBlock2D');
    const layers = transformerLayers(args);
    const resnets: ResnetBlock2D[] = [];
    const attentions: Transformer2DModel[] = [];
    for (let index = 0; index < args.layers; index += 1) {
      resnets.push(resnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels));
      attentions.push(new Transformer2DModel(args.heads, Math.floor(args.outChannels / args.heads), args.outChannels, layers[index]!,
        cross, args.groups, args.linearProjection, args.onlyCrossAttention, 0));
    }
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.downsamplers = args.resample
      ? this.registerModule('downsamplers', new ModuleList<Module>([new Downsample2D(args.outChannels, true, args.downsamplePadding)]))
      : null;
  }

  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, mask: Tensor | null): [Tensor, Residual[]] {
    const outputs: Tensor[] = [];
    let states = hidden;
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = this.resnets.at(index).forward(states, temb);
      states = this.attentions.at(index).forward(states, encoder, mask);
      outputs.push(states);
    }
    if (this.downsamplers) {
      states = runDownsamplers(this.downsamplers, states, temb);
      outputs.push(states);
    }
    return [states, outputs];
  }
}

class DownBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = false;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(
      Array.from({ length: args.layers }, (_, index) => resnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels)),
    ));
    this.downsamplers = args.resample
      ? this.registerModule('downsamplers', new ModuleList<Module>([new Downsample2D(args.outChannels, true, args.downsamplePadding)]))
      : null;
  }

  forward(hidden: Tensor, temb: Tensor | null): [Tensor, Residual[]] {
    const outputs: Tensor[] = [];
    let states = hidden;
    for (const block of this.resnets) {
      states = block.forward(states, temb);
      outputs.push(states);
    }
    if (this.downsamplers) {
      states = runDownsamplers(this.downsamplers, states, temb);
      outputs.push(states);
    }
    return [states, outputs];
  }
}

/** ``ResnetDownsampleBlock2D``: ResNets, then a downsampling ResNet (``down=True``). */
class ResnetDownsampleBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = false;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    const extra = { skipTimeAct: args.skipTimeAct, outputScaleFactor: args.outputScaleFactor };
    this.resnets = this.registerModule('resnets', new ModuleList(
      Array.from({ length: args.layers }, (_, index) => resnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels, extra)),
    ));
    this.downsamplers = args.resample
      ? this.registerModule('downsamplers', new ModuleList<Module>([resnet(args, args.outChannels, args.outChannels, { ...extra, down: true })]))
      : null;
  }

  forward(hidden: Tensor, temb: Tensor | null): [Tensor, Residual[]] {
    const outputs: Tensor[] = [];
    let states = hidden;
    for (const block of this.resnets) {
      states = block.forward(states, temb);
      outputs.push(states);
    }
    if (this.downsamplers) {
      states = runDownsamplers(this.downsamplers, states, temb);
      outputs.push(states);
    }
    return [states, outputs];
  }
}

/** ``AttnDownBlock2D``: ResNets with spatial self-attention (``downsample_type='conv'``). */
class AttnDownBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = false;
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    const resnets: ResnetBlock2D[] = [];
    const attentions: DiffusersAttention[] = [];
    for (let index = 0; index < args.layers; index += 1) {
      resnets.push(resnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels));
      attentions.push(spatialAttention(args.outChannels, args.headDim, args.groups, args.eps, true));
    }
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.downsamplers = args.resample
      ? this.registerModule('downsamplers', new ModuleList<Module>([new Downsample2D(args.outChannels, true, args.downsamplePadding)]))
      : null;
  }

  forward(hidden: Tensor, temb: Tensor | null): [Tensor, Residual[]] {
    const outputs: Tensor[] = [];
    let states = hidden;
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = this.attentions.at(index).forward(this.resnets.at(index).forward(states, temb));
      outputs.push(states);
    }
    if (this.downsamplers) {
      states = runDownsamplers(this.downsamplers, states, temb);
      outputs.push(states);
    }
    return [states, outputs];
  }
}

/** ``SimpleCrossAttnDownBlock2D``: ResNets with added-KV cross-attention (UnCLIP style). */
class SimpleCrossAttnDownBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = true;
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    crossDim(args, 'SimpleCrossAttnDownBlock2D');
    const extra = { skipTimeAct: args.skipTimeAct, outputScaleFactor: args.outputScaleFactor };
    const resnets: ResnetBlock2D[] = [];
    const attentions: DiffusersAttention[] = [];
    for (let index = 0; index < args.layers; index += 1) {
      resnets.push(resnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels, extra));
      attentions.push(simpleCrossAttention(args, args.outChannels, args.onlyCrossAttention));
    }
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.downsamplers = args.resample
      ? this.registerModule('downsamplers', new ModuleList<Module>([resnet(args, args.outChannels, args.outChannels, { ...extra, down: true })]))
      : null;
  }

  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, mask: Tensor | null): [Tensor, Residual[]] {
    const outputs: Tensor[] = [];
    let states = hidden;
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = this.attentions.at(index).forward(this.resnets.at(index).forward(states, temb), encoder, mask);
      outputs.push(states);
    }
    if (this.downsamplers) {
      states = runDownsamplers(this.downsamplers, states, temb);
      outputs.push(states);
    }
    return [states, outputs];
  }
}

/** A K-block ResNet: ``groups = in // 32`` and ``groups_out = block out_channels // 32``. */
function kResnet(args: BlockArgs, inChannels: number, outChannels: number, conv2dOutChannels: number | null = null): ResnetBlockCondNorm2D {
  return new ResnetBlockCondNorm2D({
    inChannels, outChannels, tembChannels: args.temb, eps: args.eps, groups: Math.floor(inChannels / 32), groupsOut: Math.floor(args.outChannels / 32),
    dropout: args.dropout, timeEmbeddingNorm: 'ada_group', nonLinearity: args.act, outputScaleFactor: 1, convShortcutBias: false, conv2dOutChannels,
  });
}

/** ``KDownBlock2D``: AdaGroupNorm ResNets and a K-downsampler. */
class KDownBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = false;
  readonly resnets: ModuleList<ResnetBlockCondNorm2D>;
  readonly downsamplers: ModuleList<KDownsample2D> | null;

  constructor(args: BlockArgs) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(
      Array.from({ length: args.layers }, (_, index) => kResnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels)),
    ));
    this.downsamplers = args.resample ? this.registerModule('downsamplers', new ModuleList([new KDownsample2D()])) : null;
  }

  forward(hidden: Tensor, temb: Tensor | null): [Tensor, Residual[]] {
    const outputs: Tensor[] = [];
    let states = hidden;
    for (const block of this.resnets) {
      states = block.forward(states, temb);
      outputs.push(states);
    }
    if (this.downsamplers) for (const downsampler of this.downsamplers) states = downsampler.forward(states);
    return [states, outputs];
  }
}

/** ``KCrossAttnDownBlock2D``: AdaGroupNorm ResNets with K attention (self-attention only without downsampling). */
class KCrossAttnDownBlock2D extends Module implements DownBlock {
  readonly hasCrossAttention = true;
  readonly resnets: ModuleList<ResnetBlockCondNorm2D>;
  readonly attentions: ModuleList<KAttentionBlock>;
  readonly downsamplers: ModuleList<KDownsample2D> | null;

  constructor(args: BlockArgs) {
    super();
    const resnets: ResnetBlockCondNorm2D[] = [];
    const attentions: KAttentionBlock[] = [];
    for (let index = 0; index < args.layers; index += 1) {
      resnets.push(kResnet(args, index === 0 ? args.inChannels : args.outChannels, args.outChannels));
      attentions.push(new KAttentionBlock(args.outChannels, Math.floor(args.outChannels / args.headDim), args.headDim, {
        crossAttentionDim: args.crossAttentionDim, tembChannels: args.temb!, bias: true, addSelfAttention: !args.resample,
        crossAttentionNorm: 'layer_norm', groupSize: 32,
      }));
    }
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.downsamplers = args.resample ? this.registerModule('downsamplers', new ModuleList([new KDownsample2D()])) : null;
  }

  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, mask: Tensor | null): [Tensor, Residual[]] {
    const outputs: Residual[] = [];
    let states = hidden;
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = this.resnets.at(index).forward(states, temb);
      states = this.attentions.at(index).forward(states, encoder, temb!, null, mask);
      outputs.push(this.downsamplers ? states : null);
    }
    if (this.downsamplers) for (const downsampler of this.downsamplers) states = downsampler.forward(states);
    return [states, outputs];
  }
}

function popResidual(pending: Residual[]): Tensor {
  const value = pending.pop();
  if (value === null || value === undefined) throw new TypeError('expected Tensor as element 1 in argument 0, but got NoneType');
  return value;
}

function upResnet(args: BlockArgs, index: number, extra: Partial<ResnetOptions> = {}): ResnetBlock2D {
  const skip = index === args.layers - 1 ? args.inChannels : args.outChannels;
  const input = index === 0 ? args.prevOutputChannel! : args.outChannels;
  return resnet(args, input + skip, args.outChannels, extra);
}

function upResnets(args: BlockArgs, extra: Partial<ResnetOptions> = {}): ResnetBlock2D[] {
  return Array.from({ length: args.layers }, (_, index) => upResnet(args, index, extra));
}

/**
 * Build ``count`` (resnet, attention) pairs in diffusers' order: each layer's
 * ResNet and then its attention, so seeded initialization draws match Python.
 */
function interleaved<R, A>(count: number, makeResnet: (index: number) => R, makeAttention: (index: number) => A): [R[], A[]] {
  const resnets: R[] = [];
  const attentions: A[] = [];
  for (let index = 0; index < count; index += 1) {
    resnets.push(makeResnet(index));
    attentions.push(makeAttention(index));
  }
  return [resnets, attentions];
}

function runUpsamplers(upsamplers: ModuleList<Module> | null, states: Tensor, temb: Tensor | null, size: readonly [number, number] | null): Tensor {
  let result = states;
  if (upsamplers) {
    for (const upsampler of upsamplers) {
      if (upsampler instanceof ResnetBlock2D) result = upsampler.forward(result, temb);
      else if (upsampler instanceof KUpsample2D) result = upsampler.forward(result);
      else result = (upsampler as Upsample2D).forward(result, size);
    }
  }
  return result;
}

class CrossAttnUpBlock2D extends Module implements UpBlock {
  readonly hasCrossAttention = true;
  readonly attentions: ModuleList<Transformer2DModel>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    const cross = crossDim(args, 'CrossAttnUpBlock2D');
    const layers = transformerLayers(args);
    const [resnets, attentions] = interleaved(args.layers, (index) => upResnet(args, index), (index) => new Transformer2DModel(args.heads,
      Math.floor(args.outChannels / args.heads), args.outChannels, layers[index]!, cross, args.groups, args.linearProjection, args.onlyCrossAttention, 0));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList<Module>([new Upsample2D(args.outChannels, true)])) : null;
  }

  forward(hidden: Tensor, residuals: Residual[], temb: Tensor | null, encoder: Tensor, mask: Tensor | null, size: readonly [number, number] | null): Tensor {
    let states = hidden;
    const pending = [...residuals];
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = cat([states, popResidual(pending)], 1);
      states = this.resnets.at(index).forward(states, temb);
      states = this.attentions.at(index).forward(states, encoder, mask);
    }
    return runUpsamplers(this.upsamplers, states, temb, size);
  }
}

class UpBlock2D extends Module implements UpBlock {
  readonly hasCrossAttention = false;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(upResnets(args)));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList<Module>([new Upsample2D(args.outChannels, true)])) : null;
  }

  forward(hidden: Tensor, residuals: Residual[], temb: Tensor | null, _encoder: Tensor, _mask: Tensor | null, size: readonly [number, number] | null): Tensor {
    let states = hidden;
    const pending = [...residuals];
    for (const block of this.resnets) states = block.forward(cat([states, popResidual(pending)], 1), temb);
    return runUpsamplers(this.upsamplers, states, temb, size);
  }
}

/** ``ResnetUpsampleBlock2D``: ResNets, then an upsampling ResNet (``up=True``, which ignores ``size``). */
class ResnetUpsampleBlock2D extends Module implements UpBlock {
  readonly hasCrossAttention = false;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    const extra = { skipTimeAct: args.skipTimeAct, outputScaleFactor: args.outputScaleFactor };
    this.resnets = this.registerModule('resnets', new ModuleList(upResnets(args, extra)));
    this.upsamplers = args.resample
      ? this.registerModule('upsamplers', new ModuleList<Module>([resnet(args, args.outChannels, args.outChannels, { ...extra, up: true })]))
      : null;
  }

  forward(hidden: Tensor, residuals: Residual[], temb: Tensor | null): Tensor {
    let states = hidden;
    const pending = [...residuals];
    for (const block of this.resnets) states = block.forward(cat([states, popResidual(pending)], 1), temb);
    return runUpsamplers(this.upsamplers, states, temb, null);
  }
}

/** ``AttnUpBlock2D``: ResNets with spatial self-attention and a convolutional upsampler (which ignores ``size``). */
class AttnUpBlock2D extends Module implements UpBlock {
  readonly hasCrossAttention = false;
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    const [resnets, attentions] = interleaved(args.layers, (index) => upResnet(args, index),
      () => spatialAttention(args.outChannels, args.headDim, args.groups, args.eps, false));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList<Module>([new Upsample2D(args.outChannels, true)])) : null;
  }

  forward(hidden: Tensor, residuals: Residual[], temb: Tensor | null): Tensor {
    let states = hidden;
    const pending = [...residuals];
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = cat([states, popResidual(pending)], 1);
      states = this.attentions.at(index).forward(this.resnets.at(index).forward(states, temb));
    }
    return runUpsamplers(this.upsamplers, states, temb, null);
  }
}

/** ``SimpleCrossAttnUpBlock2D``: ResNets with added-KV cross-attention and an upsampling ResNet. */
class SimpleCrossAttnUpBlock2D extends Module implements UpBlock {
  readonly hasCrossAttention = true;
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Module> | null;

  constructor(args: BlockArgs) {
    super();
    crossDim(args, 'SimpleCrossAttnUpBlock2D');
    const extra = { skipTimeAct: args.skipTimeAct, outputScaleFactor: args.outputScaleFactor };
    const [resnets, attentions] = interleaved(args.layers, (index) => upResnet(args, index, extra),
      () => simpleCrossAttention(args, args.outChannels, args.onlyCrossAttention));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.upsamplers = args.resample
      ? this.registerModule('upsamplers', new ModuleList<Module>([resnet(args, args.outChannels, args.outChannels, { ...extra, up: true })]))
      : null;
  }

  forward(hidden: Tensor, residuals: Residual[], temb: Tensor | null, encoder: Tensor, mask: Tensor | null): Tensor {
    let states = hidden;
    const pending = [...residuals];
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = cat([states, popResidual(pending)], 1);
      states = this.attentions.at(index).forward(this.resnets.at(index).forward(states, temb), encoder, mask);
    }
    return runUpsamplers(this.upsamplers, states, temb, null);
  }
}

/** ``KUpBlock2D``: concatenates the last skip (when present), AdaGroupNorm ResNets, a K-upsampler. */
class KUpBlock2D extends Module implements UpBlock {
  readonly hasCrossAttention = false;
  readonly resnets: ModuleList<ResnetBlockCondNorm2D>;
  readonly upsamplers: ModuleList<KUpsample2D> | null;

  constructor(args: BlockArgs) {
    super();
    const kIn = 2 * args.outChannels;
    const kOut = args.inChannels;
    const layers = args.layers - 1;
    this.resnets = this.registerModule('resnets', new ModuleList(Array.from({ length: layers }, (_, index) => kResnet(
      args, index === 0 ? kIn : args.outChannels, index === layers - 1 ? kOut : args.outChannels,
    ))));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList([new KUpsample2D()])) : null;
  }

  forward(hidden: Tensor, residuals: Residual[], temb: Tensor | null): Tensor {
    const skip = residuals[residuals.length - 1] ?? null;
    let states = skip === null ? hidden : cat([hidden, skip], 1);
    for (const block of this.resnets) states = block.forward(states, temb);
    return runUpsamplers(this.upsamplers as ModuleList<Module> | null, states, temb, null);
  }
}

/** ``KCrossAttnUpBlock2D``: K ResNets and K attention; the first block (in == out == temb channels) adds self-attention. */
class KCrossAttnUpBlock2D extends Module implements UpBlock {
  readonly hasCrossAttention = true;
  readonly resnets: ModuleList<ResnetBlockCondNorm2D>;
  readonly attentions: ModuleList<KAttentionBlock>;
  readonly upsamplers: ModuleList<KUpsample2D> | null;

  constructor(args: BlockArgs) {
    super();
    const first = args.inChannels === args.outChannels && args.outChannels === args.temb;
    const middle = args.inChannels !== args.outChannels;
    const kIn = first ? args.outChannels : 2 * args.outChannels;
    const kOut = args.inChannels;
    const layers = args.layers - 1;
    const resnets: ResnetBlockCondNorm2D[] = [];
    const attentions: KAttentionBlock[] = [];
    for (let index = 0; index < layers; index += 1) {
      const last = index === layers - 1;
      resnets.push(kResnet(args, index === 0 ? kIn : args.outChannels, args.outChannels, middle && last ? kOut : null));
      const dim = last ? kOut : args.outChannels;
      attentions.push(new KAttentionBlock(dim, Math.floor(dim / args.headDim), args.headDim, {
        crossAttentionDim: args.crossAttentionDim, tembChannels: args.temb!, bias: true, addSelfAttention: first,
        crossAttentionNorm: 'layer_norm', groupSize: 32,
      }));
    }
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList([new KUpsample2D()])) : null;
  }

  forward(hidden: Tensor, residuals: Residual[], temb: Tensor | null, encoder: Tensor, mask: Tensor | null): Tensor {
    const skip = residuals[residuals.length - 1] ?? null;
    let states = skip === null ? hidden : cat([hidden, skip], 1);
    for (let index = 0; index < this.resnets.length; index += 1) {
      states = this.resnets.at(index).forward(states, temb);
      states = this.attentions.at(index).forward(states, encoder, temb!, null, mask);
    }
    return runUpsamplers(this.upsamplers as ModuleList<Module> | null, states, temb, null);
  }
}

interface MidBlock extends Module {
  readonly hasCrossAttention: boolean;
  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, mask: Tensor | null): Tensor;
}

class UNetMidBlock2DCrossAttn extends Module implements MidBlock {
  readonly hasCrossAttention = true;
  readonly attentions: ModuleList<Transformer2DModel>;
  readonly resnets: ModuleList<ResnetBlock2D>;

  constructor(args: BlockArgs, outputScaleFactor: number) {
    super();
    const channels = args.inChannels;
    const groups = args.groups ?? Math.min(Math.floor(channels / 4), 32);
    const scoped = { ...args, groups };
    const layers = typeof args.transformerLayers === 'number' ? [args.transformerLayers] : args.transformerLayers;
    const resnets = [resnet(scoped, channels, channels, { outputScaleFactor })];
    const attentions = [new Transformer2DModel(args.heads, Math.floor(channels / args.heads), channels, layers[0]!, crossDim(args, 'UNetMidBlock2DCrossAttn'),
      groups, args.linearProjection, false, 0)];
    resnets.push(resnet(scoped, channels, channels, { outputScaleFactor }));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
  }

  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, mask: Tensor | null): Tensor {
    let states = this.resnets.at(0).forward(hidden, temb);
    for (let index = 0; index < this.attentions.length; index += 1) {
      states = this.attentions.at(index).forward(states, encoder, mask);
      states = this.resnets.at(index + 1).forward(states, temb);
    }
    return states;
  }
}

/** ``UNetMidBlock2DSimpleCrossAttn``: ResNets around one added-KV cross-attention. */
class UNetMidBlock2DSimpleCrossAttn extends Module implements MidBlock {
  readonly hasCrossAttention = true;
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<ResnetBlock2D>;

  constructor(args: BlockArgs, outputScaleFactor: number, onlyCross: boolean) {
    super();
    const channels = args.inChannels;
    if (!Number.isInteger(args.headDim)) throw new TypeError("unsupported operand type(s) for //: 'int' and 'NoneType'");
    const groups = args.groups ?? Math.min(Math.floor(channels / 4), 32);
    const scoped = { ...args, groups };
    const extra = { skipTimeAct: args.skipTimeAct, outputScaleFactor };
    const resnets = [resnet(scoped, channels, channels, extra)];
    const attentions = [simpleCrossAttention(scoped, channels, onlyCross)];
    resnets.push(resnet(scoped, channels, channels, extra));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
  }

  forward(hidden: Tensor, temb: Tensor | null, encoder: Tensor, mask: Tensor | null): Tensor {
    let states = this.resnets.at(0).forward(hidden, temb);
    for (let index = 0; index < this.attentions.length; index += 1) {
      states = this.attentions.at(index).forward(states, encoder, mask);
      states = this.resnets.at(index + 1).forward(states, temb);
    }
    return states;
  }
}

/**
 * ``UNetMidBlock2D``: ResNets with optional single-head-dimension spatial
 * attention. Inside a UNet it has ``num_layers=0`` (one ResNet); the VAE uses
 * one attention layer (optional) between two ResNets.
 */
class UNetMidBlock2D extends Module implements MidBlock {
  readonly hasCrossAttention = false;
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<Module>;

  constructor(options: {
    channels: number; temb: number | null; eps: number; act: string; groups: number | null; layers: number; addAttention: boolean;
    attentionHeadDim: number | null; timeScaleShift: string; outputScaleFactor: number; dropout: number;
  }) {
    super();
    const { channels } = options;
    const groups = options.groups ?? Math.min(Math.floor(channels / 4), 32);
    const attentionGroups = options.timeScaleShift === 'default' ? groups : null;
    const block = (): Module => (options.timeScaleShift === 'spatial'
      ? new ResnetBlockCondNorm2D({
        inChannels: channels, outChannels: channels, tembChannels: options.temb, eps: options.eps, groups, dropout: options.dropout,
        timeEmbeddingNorm: 'spatial', nonLinearity: options.act, outputScaleFactor: options.outputScaleFactor,
      })
      : new ResnetBlock2D({
        inChannels: channels, outChannels: channels, tembChannels: options.temb, eps: options.eps, groups, dropout: options.dropout,
        timeEmbeddingNorm: options.timeScaleShift, nonLinearity: options.act, outputScaleFactor: options.outputScaleFactor,
      }));
    const headDim = options.attentionHeadDim ?? channels;
    const resnets = [block()];
    const attentions = new ModuleList<DiffusersAttention>();
    for (let index = 0; index < options.layers; index += 1) {
      if (options.addAttention) {
        attentions.append(deprecated(new DiffusersAttention({
          queryDim: channels, heads: Math.floor(channels / headDim), dimHead: headDim, rescaleOutputFactor: options.outputScaleFactor,
          eps: options.eps, normNumGroups: attentionGroups, spatialNormDim: options.timeScaleShift === 'spatial' ? options.temb : null,
          residualConnection: true, bias: true,
        })));
      } else {
        attentions.registerModule(String(index), null);
      }
      resnets.push(block());
    }
    this.attentions = this.registerModule('attentions', attentions);
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
  }

  forward(hidden: Tensor, temb: Tensor | null = null): Tensor {
    const run = (module: Module, states: Tensor): Tensor => (module as Resnet).forward(states, temb);
    let states = run(this.resnets.at(0), hidden);
    const count = this.resnets.length - 1;
    for (let index = 0; index < count; index += 1) {
      const attention = this.attentions._modules.get(String(index)) as DiffusersAttention | null | undefined;
      if (attention) states = attention.forward(states, null, null, temb);
      states = run(this.resnets.at(index + 1), states);
    }
    return states;
  }
}

// ---------------------------------------------------------------------------
// UNet2DConditionModel.
// ---------------------------------------------------------------------------

/** Conditioning inputs Python's ``ImageDecoder`` rejects (``unsupported diffusion pipeline``). */
const IMAGE_DECODER_UNSUPPORTED: [string, (value: JsonValue | undefined) => boolean][] = [
  ['encoder_hid_dim_type', (value) => value !== null && value !== undefined],
  ['class_embed_type', (value) => value !== null && value !== undefined],
  ['addition_embed_type', (value) => value !== null && value !== undefined],
  ['num_class_embeds', (value) => value !== null && value !== undefined],
  ['time_cond_proj_dim', (value) => value !== null && value !== undefined],
  ['dual_cross_attention', (value) => value === true],
  ['attention_type', (value) => value !== 'default'],
];

const UNET_DOWN_BLOCKS: Record<string, new (args: BlockArgs) => DownBlock> = {
  DownBlock2D, ResnetDownsampleBlock2D, AttnDownBlock2D, CrossAttnDownBlock2D, SimpleCrossAttnDownBlock2D, KDownBlock2D, KCrossAttnDownBlock2D,
};

const UNET_UP_BLOCKS: Record<string, new (args: BlockArgs) => UpBlock> = {
  UpBlock2D, ResnetUpsampleBlock2D, CrossAttnUpBlock2D, SimpleCrossAttnUpBlock2D, AttnUpBlock2D, KUpBlock2D, KCrossAttnUpBlock2D,
};

/** Block types diffusers constructs but cannot run inside the named model. */
const NOT_RUNNABLE = new Set([
  'SkipDownBlock2D', 'AttnSkipDownBlock2D', 'DownEncoderBlock2D', 'AttnDownEncoderBlock2D',
  'SkipUpBlock2D', 'AttnSkipUpBlock2D', 'UpDecoderBlock2D', 'AttnUpDecoderBlock2D',
]);

function perBlock<T>(value: JsonValue | undefined, count: number): T[] {
  return Array.isArray(value) ? value as unknown as T[] : Array.from({ length: count }, () => value as unknown as T);
}

export class UNet2DConditionModel extends Module {
  static override readonly qualifiedName: string = 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel';
  readonly config: JsonObject;
  readonly conv_in: Conv2d;
  readonly time_proj: Timesteps | GaussianFourierProjection;
  readonly time_embedding: TimestepEmbedding;
  readonly time_embed_act: Act | null;
  readonly down_blocks: ModuleList<DownBlock>;
  readonly up_blocks: ModuleList<UpBlock>;
  readonly mid_block: MidBlock | null;
  readonly conv_norm_out: GroupNorm | null;
  readonly conv_act: Act | null;
  readonly conv_out: Conv2d;
  readonly numUpsamplers: number;

  constructor(config: JsonObject) {
    super();
    this.config = resolveDiffusersConfig('UNet2DConditionModel', config);
    const c = this.config;
    if (c.num_attention_heads !== null && c.num_attention_heads !== undefined) {
      throw new ValueError('At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19.');
    }
    // diffusers registers ``encoder_hid_dim_type='text_proj'`` when only ``encoder_hid_dim`` is given.
    if ((c.encoder_hid_dim_type === null || c.encoder_hid_dim_type === undefined) && c.encoder_hid_dim !== null && c.encoder_hid_dim !== undefined) {
      c.encoder_hid_dim_type = 'text_proj';
    }
    for (const [key, unsupported] of IMAGE_DECODER_UNSUPPORTED) {
      if (unsupported(c[key])) throw new ValueError('unsupported diffusion pipeline: only plain cross-attention UNets are supported');
    }
    const down = c.down_block_types as string[];
    const up = c.up_block_types as string[];
    const channels = c.block_out_channels as number[];
    if (!Array.isArray(down) || !Array.isArray(up) || !Array.isArray(channels)) throw new ValueError('block types and block_out_channels must be lists');
    const count = down.length;
    const mismatch = (name: string): ValueError => new ValueError(
      `Must provide the same number of \`${name}\` as \`down_block_types\`. \`${name}\`: ${pyRepr(c[name])}. \`down_block_types\`: ${pyRepr(down)}.`,
    );
    if (count !== up.length) throw new ValueError(`Must provide the same number of \`down_block_types\` as \`up_block_types\`. \`down_block_types\`: ${pyRepr(down)}. \`up_block_types\`: ${pyRepr(up)}.`);
    if (channels.length !== count) throw new ValueError(`Must provide the same number of \`block_out_channels\` as \`down_block_types\`. \`block_out_channels\`: ${pyRepr(channels)}. \`down_block_types\`: ${pyRepr(down)}.`);
    for (const name of ['only_cross_attention', 'attention_head_dim', 'cross_attention_dim', 'layers_per_block']) {
      if (Array.isArray(c[name]) && (c[name] as JsonValue[]).length !== count) throw mismatch(name);
    }
    if (Array.isArray(c.transformer_layers_per_block) && (c.reverse_transformer_layers_per_block ?? null) === null
      && (c.transformer_layers_per_block as JsonValue[]).some(Array.isArray)) {
      throw new ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.");
    }
    const groups = typeof c.norm_num_groups === 'number' ? c.norm_num_groups : null;
    const eps = num(c, 'norm_eps');
    const act = String(c.act_fn);
    const inKernel = int(c, 'conv_in_kernel');
    const outKernel = int(c, 'conv_out_kernel');
    this.conv_in = this.registerModule('conv_in', new Conv2d(int(c, 'in_channels'), channels[0]!, inKernel, { padding: Math.floor((inKernel - 1) / 2) }));
    let timeEmbedDim: number;
    let timestepInputDim: number;
    if (c.time_embedding_type === 'fourier') {
      timeEmbedDim = (c.time_embedding_dim as number | null) ?? channels[0]! * 2;
      if (timeEmbedDim % 2 !== 0) throw new ValueError(`\`time_embed_dim\` should be divisible by 2, but is ${timeEmbedDim}.`);
      this.time_proj = this.registerModule('time_proj', new GaussianFourierProjection(timeEmbedDim / 2, c.flip_sin_to_cos === true));
      timestepInputDim = timeEmbedDim;
    } else if (c.time_embedding_type === 'positional') {
      timeEmbedDim = (c.time_embedding_dim as number | null) ?? channels[0]! * 4;
      this.time_proj = this.registerModule('time_proj', new Timesteps(channels[0]!, c.flip_sin_to_cos === true, num(c, 'freq_shift')));
      timestepInputDim = channels[0]!;
    } else {
      throw new ValueError(`${String(c.time_embedding_type)} does not exist. Please make sure to use one of \`fourier\` or \`positional\`.`);
    }
    this.time_embedding = this.registerModule('time_embedding', new TimestepEmbedding(timestepInputDim, timeEmbedDim, act, (c.timestep_post_act as string | null) ?? null));
    this.time_embed_act = typeof c.time_embedding_act_fn === 'string' ? this.registerModule('time_embed_act', activation(c.time_embedding_act_fn)) : null;
    this.down_blocks = this.registerModule('down_blocks', new ModuleList<DownBlock>());
    this.up_blocks = this.registerModule('up_blocks', new ModuleList<UpBlock>());
    let onlyCross: boolean[];
    let midOnlyCross = c.mid_block_only_cross_attention as boolean | null;
    if (typeof c.only_cross_attention === 'boolean') {
      if (midOnlyCross === null || midOnlyCross === undefined) midOnlyCross = c.only_cross_attention;
      onlyCross = perBlock<boolean>(c.only_cross_attention, count);
    } else {
      onlyCross = c.only_cross_attention as boolean[];
    }
    if (midOnlyCross === null || midOnlyCross === undefined) midOnlyCross = false;
    // diffusers' naming quirk: ``attention_head_dim`` doubles as ``num_attention_heads``.
    const heads = perBlock<number>(c.attention_head_dim, count);
    const headDims = perBlock<number | null>(c.attention_head_dim, count);
    const crossDims = perBlock<number | null>(c.cross_attention_dim, count);
    const layersPerBlock = perBlock<number>(c.layers_per_block, count);
    const transformerLayers = perBlock<number | number[]>(c.transformer_layers_per_block, count);
    const blocksTimeEmbedDim = c.class_embeddings_concat === true ? timeEmbedDim * 2 : timeEmbedDim;
    const base = {
      temb: blocksTimeEmbedDim, eps, act, groups, downsamplePadding: int(c, 'downsample_padding'),
      linearProjection: c.use_linear_projection === true, timeScaleShift: String(c.resnet_time_scale_shift), dropout: num(c, 'dropout'),
      skipTimeAct: c.resnet_skip_time_act === true, outputScaleFactor: num(c, 'resnet_out_scale_factor'),
      crossAttentionNorm: (c.cross_attention_norm as string | null) ?? null,
    };
    let output = channels[0]!;
    down.forEach((type, index) => {
      const input = output;
      output = channels[index]!;
      const name = type.startsWith('UNetRes') ? type.slice(7) : type;
      const args: BlockArgs = {
        ...base, layers: layersPerBlock[index]!, transformerLayers: transformerLayers[index]!, inChannels: input, outChannels: output,
        resample: index !== channels.length - 1, crossAttentionDim: crossDims[index] ?? null, heads: heads[index]!,
        headDim: headDims[index] ?? output, onlyCrossAttention: onlyCross[index]!,
      };
      const Block = UNET_DOWN_BLOCKS[name];
      if (Block) this.down_blocks.append(new Block(args));
      else if (NOT_RUNNABLE.has(name)) throw new ValueError(`${name} cannot run inside UNet2DConditionModel (diffusers fails in its forward pass)`);
      else throw new ValueError(`${name} does not exist.`);
    });
    const midType = c.mid_block_type ?? null;
    const last = count - 1;
    const midArgs: BlockArgs = {
      ...base, layers: 1, transformerLayers: transformerLayers[last]!, inChannels: channels[last]!, outChannels: channels[last]!, resample: false,
      crossAttentionDim: crossDims[last] ?? null, heads: heads[last]!, headDim: headDims[last] ?? Number.NaN, onlyCrossAttention: false,
    };
    const midScale = num(c, 'mid_block_scale_factor');
    if (midType === 'UNetMidBlock2DCrossAttn') {
      this.mid_block = this.registerModule('mid_block', new UNetMidBlock2DCrossAttn(midArgs, midScale));
    } else if (midType === 'UNetMidBlock2DSimpleCrossAttn') {
      this.mid_block = this.registerModule('mid_block', new UNetMidBlock2DSimpleCrossAttn(midArgs, midScale, midOnlyCross === true));
    } else if (midType === 'UNetMidBlock2D') {
      this.mid_block = this.registerModule('mid_block', new UNetMidBlock2D({
        channels: channels[last]!, temb: blocksTimeEmbedDim, eps, act, groups, layers: 0, addAttention: false, attentionHeadDim: headDims[last] ?? null,
        timeScaleShift: base.timeScaleShift, outputScaleFactor: midScale, dropout: base.dropout,
      }));
    } else if (midType === null) {
      this.mid_block = null;
    } else {
      throw new ValueError(`unknown mid_block_type : ${String(midType)}`);
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
      const name = type.startsWith('UNetRes') ? type.slice(7) : type;
      const args: BlockArgs = {
        ...base, layers: reversedLayers[index]! + 1, transformerLayers: reversedTransformer[index]!, inChannels: input, outChannels: output,
        prevOutputChannel: previous, resample: !final, crossAttentionDim: reversedCross[index] ?? null, heads: reversedHeads[index]!,
        // diffusers passes the *unreversed* ``attention_head_dim[i]`` to up blocks.
        headDim: headDims[index] ?? output, onlyCrossAttention: reversedOnlyCross[index]!,
      };
      const Block = UNET_UP_BLOCKS[name];
      if (Block) this.up_blocks.append(new Block(args));
      else if (NOT_RUNNABLE.has(name)) throw new ValueError(`${name} cannot run inside UNet2DConditionModel (diffusers fails in its forward pass)`);
      else throw new ValueError(`${name} does not exist.`);
    });
    this.numUpsamplers = upsamplers;
    if (groups !== null) {
      this.conv_norm_out = this.registerModule('conv_norm_out', new GroupNorm(groups, channels[0]!, { eps }));
      this.conv_act = this.registerModule('conv_act', activation(act));
    } else {
      this.conv_norm_out = null;
      this.conv_act = null;
    }
    this.conv_out = this.registerModule('conv_out', new Conv2d(channels[0]!, int(c, 'out_channels'), outKernel, { padding: Math.floor((outKernel - 1) / 2) }));
  }

  /**
   * ``unet(sample, timestep, encoder_hidden_states, encoder_attention_mask).sample``.
   * ``timestep`` is an int64 scalar or ``[batch]`` tensor; the mask is boolean ``[batch, keys]``.
   */
  forward(sample: Tensor, timestep: Tensor, encoderHiddenStates: Tensor, encoderAttentionMask: Tensor | null = null): Tensor {
    const factor = 2 ** this.numUpsamplers;
    const forwardSize = sample.shape.slice(-2).some((size) => size % factor !== 0);
    const mask = encoderAttentionMask
      ? encoderAttentionMask.to(sample.dtype).neg().add(1).mul(-10000).unsqueeze(1)
      : null;
    let input = sample;
    if (this.config.center_input_sample === true) input = input.mul(2).sub(1);
    const steps = timestep.ndim === 0 ? timestep.reshape(1) : timestep;
    const expanded = steps.shape[0] === sample.shape[0] ? steps : steps.expand(sample.shape[0]!);
    let emb = this.time_embedding.forward(this.time_proj.forward(expanded).to(sample.dtype));
    if (this.time_embed_act) emb = this.time_embed_act.forward(emb);
    let states = this.conv_in.forward(input);
    let residuals: Residual[] = [states];
    for (const block of this.down_blocks) {
      const [next, outputs] = block.forward(states, emb, encoderHiddenStates, mask);
      states = next;
      residuals.push(...outputs);
    }
    if (this.mid_block) states = this.mid_block.forward(states, emb, encoderHiddenStates, mask);
    [...this.up_blocks].forEach((block, index) => {
      const final = index === this.up_blocks.length - 1;
      const count = block.resnets.length;
      const current = residuals.slice(residuals.length - count);
      residuals = residuals.slice(0, residuals.length - count);
      let size: [number, number] | null = null;
      if (!final && forwardSize) {
        const previous = residuals[residuals.length - 1];
        if (!previous) throw new ValueError("'NoneType' object has no attribute 'shape'");
        size = previous.shape.slice(2) as [number, number];
      }
      states = block.forward(states, current, emb, encoderHiddenStates, mask, size);
    });
    if (this.conv_norm_out && this.conv_act) states = this.conv_act.forward(this.conv_norm_out.forward(states));
    return this.conv_out.forward(states);
  }
}

// ---------------------------------------------------------------------------
// AutoencoderKL.
// ---------------------------------------------------------------------------

interface VaeBlockArgs {
  layers: number;
  input: number;
  output: number;
  groups: number;
  act: string;
  resample: boolean;
  timeScaleShift: string;
}

function vaeResnet(args: VaeBlockArgs, index: number): ResnetBlock2D {
  return new ResnetBlock2D({
    inChannels: index === 0 ? args.input : args.output, outChannels: args.output, tembChannels: null, eps: 1e-6, groups: args.groups,
    dropout: 0, timeEmbeddingNorm: args.timeScaleShift, nonLinearity: args.act, outputScaleFactor: 1,
  });
}

interface VaeBlock extends Module {
  forward(hidden: Tensor): Tensor;
}

class DownEncoderBlock2D extends Module implements VaeBlock {
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Downsample2D> | null;

  constructor(args: VaeBlockArgs) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(Array.from({ length: args.layers }, (_, index) => vaeResnet(args, index))));
    this.downsamplers = args.resample ? this.registerModule('downsamplers', new ModuleList([new Downsample2D(args.output, true, 0)])) : null;
  }

  forward(hidden: Tensor): Tensor {
    let states = hidden;
    for (const block of this.resnets) states = block.forward(states, null);
    if (this.downsamplers) for (const downsampler of this.downsamplers) states = downsampler.forward(states);
    return states;
  }
}

class AttnDownEncoderBlock2D extends Module implements VaeBlock {
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly downsamplers: ModuleList<Downsample2D> | null;

  constructor(args: VaeBlockArgs) {
    super();
    const [resnets, attentions] = interleaved(args.layers, (index) => vaeResnet(args, index),
      () => spatialAttention(args.output, args.output, args.groups, 1e-6, true));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.downsamplers = args.resample ? this.registerModule('downsamplers', new ModuleList([new Downsample2D(args.output, true, 0)])) : null;
  }

  forward(hidden: Tensor): Tensor {
    let states = hidden;
    for (let index = 0; index < this.resnets.length; index += 1) states = this.attentions.at(index).forward(this.resnets.at(index).forward(states, null));
    if (this.downsamplers) for (const downsampler of this.downsamplers) states = downsampler.forward(states);
    return states;
  }
}

class UpDecoderBlock2D extends Module implements VaeBlock {
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Upsample2D> | null;

  constructor(args: VaeBlockArgs) {
    super();
    this.resnets = this.registerModule('resnets', new ModuleList(Array.from({ length: args.layers }, (_, index) => vaeResnet(args, index))));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList([new Upsample2D(args.output, true)])) : null;
  }

  forward(hidden: Tensor): Tensor {
    let states = hidden;
    for (const block of this.resnets) states = block.forward(states, null);
    if (this.upsamplers) for (const upsampler of this.upsamplers) states = upsampler.forward(states);
    return states;
  }
}

class AttnUpDecoderBlock2D extends Module implements VaeBlock {
  readonly attentions: ModuleList<DiffusersAttention>;
  readonly resnets: ModuleList<ResnetBlock2D>;
  readonly upsamplers: ModuleList<Upsample2D> | null;

  constructor(args: VaeBlockArgs) {
    super();
    const [resnets, attentions] = interleaved(args.layers, (index) => vaeResnet(args, index), () => new DiffusersAttention({
      queryDim: args.output, heads: 1, dimHead: args.output, eps: 1e-6, normNumGroups: args.groups, residualConnection: true, bias: true,
    }));
    this.attentions = this.registerModule('attentions', new ModuleList(attentions));
    this.resnets = this.registerModule('resnets', new ModuleList(resnets));
    this.upsamplers = args.resample ? this.registerModule('upsamplers', new ModuleList([new Upsample2D(args.output, true)])) : null;
  }

  forward(hidden: Tensor): Tensor {
    let states = hidden;
    for (let index = 0; index < this.resnets.length; index += 1) states = this.attentions.at(index).forward(this.resnets.at(index).forward(states, null));
    if (this.upsamplers) for (const upsampler of this.upsamplers) states = upsampler.forward(states);
    return states;
  }
}

const VAE_DOWN_BLOCKS: Record<string, new (args: VaeBlockArgs) => VaeBlock> = { DownEncoderBlock2D, AttnDownEncoderBlock2D };
const VAE_UP_BLOCKS: Record<string, new (args: VaeBlockArgs) => VaeBlock> = { UpDecoderBlock2D, AttnUpDecoderBlock2D };
const UNET_ONLY = new Set([...Object.keys(UNET_DOWN_BLOCKS), ...Object.keys(UNET_UP_BLOCKS), 'SkipDownBlock2D', 'AttnSkipDownBlock2D', 'SkipUpBlock2D', 'AttnSkipUpBlock2D']);

function vaeBlock<T>(table: Record<string, T>, type: string, where: string): T {
  const name = type.startsWith('UNetRes') ? type.slice(7) : type;
  const Block = table[name];
  if (Block) return Block;
  if (UNET_ONLY.has(name)) throw new ValueError(`${name} cannot run inside the AutoencoderKL ${where} (diffusers fails in its forward pass)`);
  throw new ValueError(`${name} does not exist.`);
}

class VaeEncoder extends Module {
  readonly conv_in: Conv2d;
  readonly down_blocks: ModuleList<VaeBlock>;
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
      const Block = vaeBlock(VAE_DOWN_BLOCKS, type, 'encoder');
      const input = output;
      output = channels[index]!;
      this.down_blocks.append(new Block({
        layers: int(c, 'layers_per_block'), input, output, groups, act, resample: index !== channels.length - 1, timeScaleShift: 'default',
      }));
    });
    const last = channels[channels.length - 1]!;
    this.mid_block = this.registerModule('mid_block', new UNetMidBlock2D({
      channels: last, temb: null, eps: 1e-6, act, groups, layers: 1, addAttention: c.mid_block_add_attention !== false, attentionHeadDim: last,
      timeScaleShift: 'default', outputScaleFactor: 1, dropout: 0,
    }));
    this.conv_norm_out = this.registerModule('conv_norm_out', new GroupNorm(groups, last, { eps: 1e-6 }));
    this.conv_act = this.registerModule('conv_act', new SiLU());
    this.conv_out = this.registerModule('conv_out', new Conv2d(last, 2 * int(c, 'latent_channels'), 3, { padding: 1 }));
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
  readonly up_blocks: ModuleList<VaeBlock>;
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
    this.mid_block = this.registerModule('mid_block', new UNetMidBlock2D({
      channels: last, temb: null, eps: 1e-6, act, groups, layers: 1, addAttention: c.mid_block_add_attention !== false, attentionHeadDim: last,
      timeScaleShift: 'default', outputScaleFactor: 1, dropout: 0,
    }));
    const reversed = [...channels].reverse();
    let output = reversed[0]!;
    (c.up_block_types as string[]).forEach((type, index) => {
      const Block = vaeBlock(VAE_UP_BLOCKS, type, 'decoder');
      const previous = output;
      output = reversed[index]!;
      this.up_blocks.append(new Block({
        layers: int(c, 'layers_per_block') + 1, input: previous, output, groups, act, resample: index !== channels.length - 1, timeScaleShift: 'group',
      }));
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

/**
 * Paths of the attention modules diffusers marks ``_from_deprecated_attn_block``
 * (VAE and UNet mid-block attentions, ``AttnDownBlock2D`` and
 * ``AttnDownEncoderBlock2D`` attentions), whose legacy parameter names
 * ``query``/``key``/``value``/``proj_attn`` load as ``to_q``/``to_k``/``to_v``/``to_out.0``.
 */
export function deprecatedAttentionPaths(module: Module): Set<string> {
  const paths = new Set<string>();
  for (const [path, child] of module.namedModules()) {
    if (child instanceof DiffusersAttention && DEPRECATED_ATTENTION.has(child)) paths.add(path);
  }
  return paths;
}

const LEGACY_NAMES: Record<string, string> = { query: 'to_q', key: 'to_k', value: 'to_v', proj_attn: 'to_out.0' };

/**
 * Legacy attention parameter names (diffusers ``_fix_state_dict_keys_on_load``).
 * With ``paths`` (from {@link deprecatedAttentionPaths}) exactly those modules
 * are converted, as diffusers does; without it, mid-block attentions are.
 */
export function convertDeprecatedAttentionKey(key: string, paths?: ReadonlySet<string>): string {
  const match = /^(.*)\.(query|key|value|proj_attn)\.(weight|bias)$/.exec(key);
  if (!match) return key;
  const [, path, name, kind] = match as unknown as [string, string, string, string];
  const deprecated = paths ? paths.has(path) : /(^|\.)mid_block\.attentions\.\d+$/.test(path);
  return deprecated ? `${path}.${LEGACY_NAMES[name]}.${kind}` : key;
}

/** A float32 ``torch.arange``-style int64 helper for timesteps. */
export function timestepTensor(values: readonly number[]): Tensor {
  return tensor([...values], { dtype: 'int64' });
}

