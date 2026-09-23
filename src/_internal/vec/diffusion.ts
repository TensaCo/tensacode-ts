/**
 * Owned latent diffusion with explicit conditioning and replayable sampling
 * (Python ``tensorcode/_internal/vec/diffusion.py``).
 *
 * Only ordinary cross-attention ``UNet2DConditionModel`` + ``AutoencoderKL``
 * models are supported, sampled with DDIM (eta 0, no classifier-free
 * guidance). A linear adapter starts untrained; importing diffusion weights does
 * not establish alignment with an arbitrary input Space.
 */
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ValueError } from '../../errors.js';
import { noGrad } from '../../nn/autograd.js';
import { roundToDType } from '../../nn/dtype.js';
import { cat, mseLoss } from '../../nn/functional.js';
import { Identity, Linear } from '../../nn/layers.js';
import type { Module } from '../../nn/module.js';
import { Generator } from '../../nn/random.js';
import { deserializeSafetensors } from '../../nn/safetensors.js';
import { Tensor, randn } from '../../nn/tensor.js';
import type { Parameter } from '../../nn/tensor.js';
import { Operation, type Context, type OperationLike } from '../../ops/base.js';
import { Latent, Space } from '../../ops/vec/latent.js';
import { pathExists } from '../files.js';
import { resolveArtifactDirectory, type HubOptions } from '../hub.js';
import { qualifiedName } from '../identity.js';
import { deepCopy, isPlainObject, parseJsonStrict, type JsonObject, type JsonValue } from '../json.js';
import { LatentOperation, asSequence, spaceFromConfig } from '../latentOps.js';
import {
  AutoencoderKL, DDIMScheduler, UNet2DConditionModel, convertDeprecatedAttentionKey, resolveDiffusersConfig,
} from '../native/diffusers.js';
import { pythonList, spaceJson, unknownKeys } from './owned.js';

const ALLOWED = [
  'input_space', 'unet_config', 'vae_config', 'scheduler_config', 'bridge', 'conditioning_status', 'num_inference_steps', 'foundation',
];

interface Components {
  unet: UNet2DConditionModel;
  vae: AutoencoderKL;
  scheduler: DDIMScheduler;
}

const DIFFUSION_INTERNAL: unique symbol = Symbol('tensorcode.vec.diffusion.internal');

interface DiffusionInternals {
  readonly [DIFFUSION_INTERNAL]: true;
  readonly components: Components;
}

function nativeConfig(value: unknown, key: string): JsonObject {
  if (value === undefined) throw new ValueError(`missing ${key}`);
  if (!isPlainObject(value)) throw new TypeError(`${key} must be a mapping`);
  const result: JsonObject = {};
  // Python ``_native_config``: private diffusers keys are not architecture.
  for (const [name, item] of Object.entries(value as JsonObject)) if (!name.startsWith('_')) result[name] = deepCopy(item as JsonValue);
  return result;
}

function present(value: JsonValue | undefined): boolean {
  return value !== null && value !== undefined;
}

/** Python's checks on the resolved components, before TypeScript builds any module. */
function validateComponents(unet: JsonObject, vae: JsonObject, scheduler: JsonObject, config: JsonObject, inputSpace: Space): void {
  if (present(unet.addition_embed_type) || present(unet.class_embed_type) || present(unet.num_class_embeds)
    || present(unet.encoder_hid_dim_type) || present(unet.time_cond_proj_dim) || unet.dual_cross_attention === true
    || unet.attention_type !== 'default') {
    throw new ValueError('unsupported diffusion pipeline: only plain cross-attention UNets are supported');
  }
  const blocks = [...(unet.down_block_types as string[]), ...(unet.up_block_types as string[]), (unet.mid_block_type as string | null) ?? ''];
  if (!blocks.some((block) => block.includes('CrossAttn'))) throw new ValueError('UNet must contain cross-attention conditioning blocks');
  const width = unet.cross_attention_dim;
  if (typeof width !== 'number' || !Number.isInteger(width)) throw new ValueError('UNet cross_attention_dim must be a single integer');
  if (unet.in_channels !== vae.latent_channels || unet.out_channels !== vae.latent_channels) {
    throw new ValueError('UNet channels must match VAE latent_channels');
  }
  if (vae.in_channels !== 3 || vae.out_channels !== 3) throw new ValueError('VAE must encode and decode RGB images');
  if ((present(vae.shift_factor) && vae.shift_factor !== 0) || present(vae.latents_mean) || present(vae.latents_std)) {
    throw new ValueError('unsupported VAE latent normalization');
  }
  if (typeof vae.scaling_factor !== 'number' || !(vae.scaling_factor > 0)) throw new ValueError('VAE scaling_factor must be positive');
  if (!['epsilon', 'v_prediction', 'sample'].includes(String(scheduler.prediction_type))) throw new ValueError('unsupported scheduler prediction_type');
  const steps = config.num_inference_steps;
  if (typeof steps !== 'number' || !Number.isInteger(steps) || steps < 1 || steps > (scheduler.num_train_timesteps as number)) {
    throw new ValueError('num_inference_steps must be within the training timestep count');
  }
  if (config.bridge === 'identity') {
    if (inputSpace.dimensions !== width) throw new ValueError('identity conditioning requires native cross-attention dimensions');
  } else if (config.bridge !== 'linear') {
    throw new ValueError('bridge must be linear or identity');
  }
  const size = unet.sample_size;
  const latent = typeof size === 'number' ? [size, size] : Array.isArray(size) ? size : [];
  if (latent.length !== 2 || latent.some((item) => typeof item !== 'number' || !Number.isInteger(item) || item <= 0)) {
    throw new ValueError('UNet sample_size must specify a positive height and width');
  }
}

/** The supervised diffusion objective (Python ``_DiffusionObjective``). */
export class DiffusionObjective extends Operation<Record<string, unknown>, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.diffusion._DiffusionObjective';
  readonly #owner: ImageDecoder;

  constructor(owner: ImageDecoder) {
    super();
    this.#owner = owner;
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: Record<string, unknown>, context: Context | null): Tensor {
    if (context !== null && context !== undefined && Object.keys(context).length) {
      throw new ValueError('Pass diffusion objective context inside inputs');
    }
    const inputs = (value as { inputs?: unknown }).inputs;
    if (!isPlainObject(inputs)) throw new TypeError('diffusion objective inputs must be a mapping');
    const record = inputs as Record<string, unknown>;
    return this.#owner.loss(record.value, value.targets, {
      context: (record.context as Context | undefined) ?? null, noise: record.noise, timesteps: record.timesteps,
    });
  }

  parameters(): Parameter[] {
    return this.#owner.parameters();
  }

  operationIdentity(): string {
    return `${qualifiedName(this.#owner)}.objective`;
  }

  configuration(): JsonObject {
    return { operation: qualifiedName(this.#owner), role: 'objective', model: this.#owner.configuration() };
  }
}

export interface ImageDecoderFoundationOptions extends Omit<HubOptions, 'allowPatterns'> {
  inputSpace: Space;
  bridge?: 'linear' | 'identity';
  numInferenceSteps?: number;
}

/**
 * Generate RGB tensors from compatible latent conditioning.
 *
 * Configuration owns ``input_space``, native ``unet_config``, ``vae_config``,
 * ``scheduler_config``, ``bridge`` and ``num_inference_steps``. Construction
 * initializes weights locally; ``fromFoundation`` explicitly imports pretrained
 * weights. Sampling requires ``context.noise`` (unscaled standard Gaussian
 * noise) or ``context.seed`` (a TensorCode generator seed; its noise differs
 * from PyTorch's). ``context.latents`` prefixes conditioning in order. DDIM
 * sampling uses eta 0 and no classifier-free guidance.
 */
export class ImageDecoder extends LatentOperation<Latent, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.diffusion.ImageDecoder';
  readonly inputSpace: Space;
  readonly unet: UNet2DConditionModel;
  readonly vae: AutoencoderKL;
  /** The configured scheduler (sampling and objectives use fresh copies). */
  readonly scheduler: DDIMScheduler;
  readonly projection: Linear | Identity;
  readonly latentSize: [number, number];
  readonly vaeScaleFactor: number;
  readonly #objective: DiffusionObjective;

  constructor(config: unknown, internals?: DiffusionInternals) {
    if (isPlainObject(config) && 'conditioning_projection' in (config as object)) {
      throw new ValueError('conditioning_projection is unsupported; configure bridge instead');
    }
    super(config);
    const cfg = this.config;
    const unknown = unknownKeys(cfg, ALLOWED);
    if (unknown.length) throw new ValueError(`Unknown configuration fields: ${pythonList(unknown)}`);
    for (const key of ['unet_config', 'vae_config', 'scheduler_config']) cfg[key] = nativeConfig(cfg[key], key);
    if (!('bridge' in cfg)) cfg.bridge = 'linear';
    if (!('conditioning_status' in cfg)) cfg.conditioning_status = cfg.bridge === 'linear' ? 'requires_training' : 'caller_declared_native_identity';
    if (!('num_inference_steps' in cfg)) cfg.num_inference_steps = 20;
    this.inputSpace = spaceFromConfig(cfg.input_space);
    const unetConfig = resolveDiffusersConfig('UNet2DConditionModel', cfg.unet_config);
    const vaeConfig = resolveDiffusersConfig('AutoencoderKL', cfg.vae_config);
    const schedulerConfig = resolveDiffusersConfig('DDIMScheduler', cfg.scheduler_config);
    validateComponents(unetConfig, vaeConfig, schedulerConfig, cfg, this.inputSpace);
    const components = internals?.[DIFFUSION_INTERNAL] ? internals.components : null;
    this.unet = this.registerModule('unet', components?.unet ?? new UNet2DConditionModel(unetConfig));
    this.vae = this.registerModule('vae', components?.vae ?? new AutoencoderKL(vaeConfig));
    this.scheduler = components?.scheduler ?? new DDIMScheduler(schedulerConfig);
    const width = unetConfig.cross_attention_dim as number;
    this.projection = cfg.bridge === 'identity'
      ? this.registerModule('projection', new Identity())
      : this.registerModule('projection', new Linear(this.inputSpace.dimensions, width));
    const size = unetConfig.sample_size as number | number[];
    this.latentSize = typeof size === 'number' ? [size, size] : [size[0]!, size[1]!];
    this.vaeScaleFactor = 2 ** ((vaeConfig.block_out_channels as number[]).length - 1);
    // Complete resolved native configurations are portable without Hub access.
    cfg.unet_config = this.unet.config;
    cfg.vae_config = this.vae.config;
    cfg.scheduler_config = this.scheduler.config;
    this.#objective = new DiffusionObjective(this);
  }

  /**
   * Import known diffusion components; never execute repository code.
   * ``identity`` is an explicit caller assertion (not library-verified) that
   * input embeddings already inhabit the foundation's native conditioning
   * space. Linear projections are initialized randomly and require paired training.
   */
  static async fromFoundation<T extends ImageDecoder>(
    this: new (config: unknown, internals?: DiffusionInternals) => T, repoIdOrPath: string, options: ImageDecoderFoundationOptions,
  ): Promise<T> {
    const { inputSpace, bridge = 'linear', numInferenceSteps = 20, ...hub } = options;
    const revision = hub.revision ?? null;
    const { path } = await resolveArtifactDirectory(repoIdOrPath, {
      ...hub, allowPatterns: ['unet/*', 'vae/*', 'scheduler/*', 'model_index.json'],
    });
    const unetConfig = await readComponentConfig(path, 'unet', 'config.json');
    const vaeConfig = await readComponentConfig(path, 'vae', 'config.json');
    const schedulerConfig = await readComponentConfig(path, 'scheduler', 'scheduler_config.json');
    const unet = new UNet2DConditionModel(nativeConfig(unetConfig, 'unet_config'));
    const vae = new AutoencoderKL(nativeConfig(vaeConfig, 'vae_config'));
    await loadComponentWeights(unet, join(path, 'unet'), 'unet');
    await loadComponentWeights(vae, join(path, 'vae'), 'vae');
    const scheduler = new DDIMScheduler(nativeConfig(schedulerConfig, 'scheduler_config'));
    const config: JsonObject = {
      input_space: spaceJson(inputSpace),
      unet_config: unet.config,
      vae_config: vae.config,
      scheduler_config: scheduler.config,
      bridge,
      num_inference_steps: numInferenceSteps,
      foundation: { source: String(repoIdOrPath), revision, scheduler: 'DDIMScheduler' },
    };
    return new this(config, { [DIFFUSION_INTERNAL]: true, components: { unet, vae, scheduler } });
  }

  get trainingInputsIncludeTargets(): boolean {
    return true;
  }

  get trainingOperation(): DiffusionObjective {
    return this.#objective;
  }

  override operationBindings(): Record<string, OperationLike> {
    return { ...super.operationBindings(), objective: this.#objective };
  }

  /** Ordered, masked conditioning ``[projected embeddings, mask]`` (Python ``_conditioning``). */
  conditioning(value: unknown, context: Context | null): [Tensor, Tensor] {
    if (context !== null && context !== undefined && !isPlainObject(context)) throw new TypeError('context must be a mapping');
    const ctx = (context ?? {}) as Record<string, unknown>;
    if (Object.keys(ctx).some((key) => !['latents', 'seed', 'noise'].includes(key))) throw new ValueError('unsupported diffusion context fields');
    const extras = ctx.latents ?? [];
    if (!Array.isArray(extras)) throw new TypeError("context['latents'] must be an ordered sequence of Latent values");
    const pairs = [...extras, value].map((item) => asSequence(item, this.inputSpace));
    const batch = pairs[0]![0].shape[0];
    const dtype = this.unet.conv_in.weight.dtype;
    for (const [sequence] of pairs) {
      if (sequence.shape[0] !== batch) throw new ValueError('conditioning batch sizes must match');
      if (sequence.dtype !== dtype) throw new ValueError('conditioning dtype and device must match the decoder');
    }
    const sequence = cat(pairs.map(([item]) => item), 1);
    const mask = cat(pairs.map(([, item]) => item), 1);
    if (!mask.any(1).all().item()) throw new ValueError('each image requires at least one unmasked conditioning token');
    // Mask before projection as well: hidden NaNs cannot leak through attention.
    const masked = sequence.maskedFill(mask.logicalNot().unsqueeze(-1), 0);
    return [(this.projection as Linear).forward(masked), mask];
  }

  private noiseShape(batch: number): number[] {
    return [batch, this.vae.config.latent_channels as number, ...this.latentSize];
  }

  private validateNoise(noise: unknown, batch: number, reference: Tensor): Tensor {
    const shape = this.noiseShape(batch);
    if (!(noise instanceof Tensor) || noise.shape.length !== shape.length || noise.shape.some((size, index) => size !== shape[index])) {
      throw new ValueError(`noise shape must be (${shape.join(', ')})`);
    }
    if (noise.dtype !== reference.dtype || !noise.allFinite()) throw new ValueError('noise must be finite and match conditioning dtype and device');
    return noise;
  }

  forward(value: Latent, context: Context | null): Tensor {
    return noGrad(() => {
      const [embeddings, mask] = this.conditioning(value, context);
      const ctx = (context ?? {}) as Record<string, unknown>;
      const seed = ctx.seed ?? null;
      let noise = ctx.noise ?? null;
      if ((seed === null) === (noise === null)) throw new ValueError("supply exactly one of context['seed'] or context['noise']");
      if (seed !== null) {
        if (typeof seed !== 'number' || !Number.isInteger(seed)) throw new ValueError('seed must be an integer');
        noise = randn(this.noiseShape(embeddings.shape[0]!), { dtype: embeddings.dtype, generator: new Generator(seed) });
      }
      const checked = this.validateNoise(noise, embeddings.shape[0]!, embeddings);
      // A local scheduler owns mutable timesteps; concurrent calls do not race.
      const scheduler = new DDIMScheduler(this.scheduler.config);
      scheduler.setTimesteps(this.config.num_inference_steps as number);
      let sample = scheduler.initNoiseSigma === 1 ? checked : checked.mul(scheduler.initNoiseSigma);
      const modes = new Map<Module, boolean>(this.modules().map((module) => [module, module.training]));
      try {
        this.eval();
        for (const timestep of scheduler.timesteps) {
          const prediction = this.unet.forward(sample, new Tensor(new Float64Array([timestep]), [], 'int64'), embeddings, mask);
          sample = scheduler.step(prediction, timestep, sample);
        }
        const scaling = roundToDType('float32', this.vae.config.scaling_factor as number);
        const pixels = this.vae.decode(sample.div(scaling));
        return pixels.div(2).add(0.5).clamp(0, 1);
      } finally {
        for (const [module, mode] of modes) module.training = mode;
      }
    });
  }

  /**
   * Differentiable diffusion prediction loss using only actual targets. Targets
   * are encoded with the VAE posterior mean (no hidden random draw). Noise and
   * per-image timesteps are mandatory; targets never enter the conditioning
   * path. The caller controls train/eval mode for this objective.
   */
  loss(value: unknown, targetPixels: unknown, options: { context?: Context | null; noise: unknown; timesteps: unknown }): Tensor {
    const [embeddings, mask] = this.conditioning(value, options.context ?? null);
    const batch = embeddings.shape[0]!;
    const expected = [batch, 3, ...this.latentSize.map((size) => size * this.vaeScaleFactor)];
    if (!(targetPixels instanceof Tensor) || targetPixels.shape.length !== 4 || targetPixels.shape.some((size, index) => size !== expected[index])
      || targetPixels.dtype !== embeddings.dtype) {
      throw new ValueError(`target_pixels must have shape (${expected.join(', ')}) and decoder dtype/device`);
    }
    if (!targetPixels.allFinite() || targetPixels.min().item() < 0 || targetPixels.max().item() > 1) {
      throw new ValueError('target_pixels must be finite RGB values in [0, 1]');
    }
    const timesteps = options.timesteps;
    const train = this.scheduler.config.num_train_timesteps as number;
    if (!(timesteps instanceof Tensor) || timesteps.dtype !== 'int64' || timesteps.ndim !== 1 || timesteps.shape[0] !== batch
      || Array.from(timesteps.toArray()).some((step) => step < 0 || step >= train)) {
      throw new ValueError('timesteps must contain one valid int64 training timestep per image');
    }
    const noise = this.validateNoise(options.noise, batch, embeddings);
    const scaling = roundToDType('float32', this.vae.config.scaling_factor as number);
    const target = noGrad(() => this.vae.encodeMode(targetPixels.mul(2).sub(1)).mul(scaling));
    const scheduler = new DDIMScheduler(this.scheduler.config);
    const noisy = scheduler.addNoise(target, noise, timesteps);
    const prediction = this.unet.forward(noisy, timesteps, embeddings, mask);
    const type = scheduler.predictionType;
    const desired = type === 'epsilon' ? noise : type === 'v_prediction' ? scheduler.getVelocity(target, noise, timesteps) : target;
    return mseLoss(prediction.to('float32'), desired.to('float32'));
  }
}

async function readComponentConfig(root: string, subfolder: string, name: string): Promise<JsonObject> {
  const file = join(root, subfolder, name);
  if (!(await pathExists(file))) throw new ValueError(`foundation is missing ${subfolder}/${name}`);
  const data = parseJsonStrict(await readFile(file, 'utf8'));
  if (!isPlainObject(data)) throw new ValueError(`${subfolder}/${name} must contain a JSON object`);
  return data as JsonObject;
}

/** diffusers ``from_pretrained(use_safetensors=True)`` weight loading with strict key checks. */
async function loadComponentWeights(module: Module, directory: string, subfolder: string): Promise<void> {
  const single = join(directory, 'diffusion_pytorch_model.safetensors');
  const index = join(directory, 'diffusion_pytorch_model.safetensors.index.json');
  const files: string[] = [];
  if (await pathExists(single)) files.push(single);
  else if (await pathExists(index)) {
    const map = parseJsonStrict(await readFile(index, 'utf8')) as { weight_map?: Record<string, string> };
    files.push(...new Set(Object.values(map.weight_map ?? {}).map((file) => join(directory, file))));
  } else {
    throw new ValueError(`incomplete or incompatible foundation ${subfolder} weights: no safetensors weights`);
  }
  const tensors = new Map<string, Tensor>();
  for (const file of files) {
    const bytes = await readFile(file);
    for (const [name, value] of deserializeSafetensors(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)).tensors) {
      tensors.set(convertDeprecatedAttentionKey(name), value);
    }
  }
  const state = module.stateDict();
  const missing = [...state.keys()].filter((name) => !tensors.has(name));
  const unexpected = [...tensors.keys()].filter((name) => !state.has(name));
  const mismatched = [...state].filter(([name, value]) => {
    const source = tensors.get(name);
    return source && (source.shape.length !== value.shape.length || source.shape.some((size, position) => size !== value.shape[position]));
  }).map(([name]) => name);
  if (missing.length || unexpected.length || mismatched.length) {
    throw new ValueError(`incomplete or incompatible foundation ${subfolder} weights: ${JSON.stringify({ missing_keys: missing, unexpected_keys: unexpected, mismatched_keys: mismatched })}`);
  }
  noGrad(() => {
    for (const [name, value] of state) {
      const source = tensors.get(name)!;
      const target = value.data;
      for (let position = 0; position < target.length; position += 1) target[position] = roundToDType(value.dtype, source.data[position]!);
      value._storage.version += 1;
    }
  });
}
