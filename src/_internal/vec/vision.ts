/**
 * Owned ViT transformer with explicit preprocessing and embedding context
 * (Python ``tensorcode/_internal/vec/vision.py``).
 */
import { readFile, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Parameter, Tensor, arange, ones, tensor, zeros } from '../../nn/tensor.js';
import { normal_ } from '../../nn/init.js';
import { noGrad } from '../../nn/autograd.js';
import { cat, meshgrid, stack } from '../../nn/ops/shape.js';
import { ValueError } from '../../errors.js';
import type { Context } from '../../ops/base.js';
import { Latent, Space } from '../../ops/vec/latent.js';
import { LatentOperation, asSequence } from '../latentOps.js';
import { requireCpu } from '../pretrained.js';
import { isSymlink } from '../files.js';
import { resolveArtifactDirectory } from '../hub.js';
import { isPlainObject, jsonEqual, parseJsonStrict, pythonJsonDumps, type JsonObject } from '../json.js';
import { NativeConfig } from '../native/config.js';
import { keyPaddingBias } from '../native/modules.js';
import { ViTModel } from '../native/vit.js';
import { loadNativeFoundation } from '../native/foundation.js';
import { ImageProcessor, type ImageInputs } from './imageProcessing.js';
import type { RasterImage } from '../image/raster.js';
import { pythonList, spaceJson, unknownKeys } from './owned.js';

export type ImageReadout = 'sequence' | 'pooled' | 'output_encoding';

/** Processed input: processor-normalized ``pixel_values`` with optional provenance. */
export interface ProcessedImages {
  pixel_values: Tensor;
  sources?: readonly string[];
}

const VISION_INTERNAL: unique symbol = Symbol('tensorcode.vec.vision.internal');

interface VisionInternals {
  readonly [VISION_INTERNAL]: true;
  readonly model: ViTModel;
}

export interface ImageFoundationOptions {
  outputSpace: Space | JsonObject;
  readout?: ImageReadout;
  contextSpace?: Space | JsonObject | null;
  revision?: string | null;
  localFilesOnly?: boolean;
  cacheDir?: string | null;
  token?: string | null;
  endpoint?: string | null;
  device?: string;
}

/**
 * Encode images with a ViT (random on construction, loaded explicitly).
 *
 * Tensor inputs are CHW/BCHW floating pixels in ``[0, 1]``, already at the
 * model's spatial resolution. Normalization preserves gradients. Use
 * {@link ImageEncoder.preprocess} for other decoded images; its resize path is
 * not differentiable. A mapping with ``pixel_values`` bypasses normalization,
 * accepting finite processor-normalized BCHW tensors. Coordinates refer to
 * processed pixels.
 *
 * Sequence output contains final patch states; pooled output is the native
 * final CLS state. Ordered context latents are prepended to image embeddings
 * before transformer attention. OUTPUT_ENCODING appends an owned learnable
 * token after those embeddings and reads its final state; it starts untrained.
 */
export class ImageEncoder extends LatentOperation<Tensor | ProcessedImages, Latent> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.vision.ImageEncoder';
  readonly model: ViTModel;
  readonly processor: ImageProcessor;
  readonly readout: ImageReadout;
  readonly outputEncoding: Parameter | null = null;
  readonly outputSpace: Space;
  readonly contextSpace: Space | null;
  readonly imageSize: readonly [number, number];
  readonly patchSize: readonly [number, number];

  constructor(config: unknown, internals?: VisionInternals) {
    super(config);
    const cfg = this.config;
    const unknown = unknownKeys(cfg, ['model', 'processor', 'readout', 'output_space', 'context_space', 'foundation']);
    if (unknown.length) throw new ValueError(`Unknown configuration fields: ${pythonList(unknown)}; use output_space and readout`);
    if (!isPlainObject(cfg.model)) throw new ValueError('ImageEncoder requires a model configuration object');
    const modelConfig = { ...cfg.model } as JsonObject;
    if ((modelConfig.model_type ?? 'vit') !== 'vit') throw new ValueError('ImageEncoder supports only ViTModel architecture');
    const native = NativeConfig.fromDict({ ...modelConfig, model_type: 'vit' });
    this.model = this.registerModule('model', internals?.[VISION_INTERNAL] ? internals.model : new ViTModel(native, { addPoolingLayer: false }));
    if (!isPlainObject(cfg.processor)) throw new ValueError('ImageEncoder requires a processor configuration object');
    if ((cfg.processor.image_processor_type ?? 'ViTImageProcessor') !== 'ViTImageProcessor') {
      throw new ValueError('ImageEncoder supports only ViTImageProcessor');
    }
    this.processor = new ImageProcessor(cfg.processor);
    const readout = cfg.readout ?? 'sequence';
    if (readout !== 'sequence' && readout !== 'pooled' && readout !== 'output_encoding') {
      throw new ValueError('readout must be sequence, pooled or output_encoding');
    }
    this.readout = readout;
    const hidden = native.hiddenSize;
    if (readout === 'output_encoding') {
      const initial = normal_(zeros([1, 1, hidden]), 0, 0.02); // nn.init.normal_(std=0.02)
      this.outputEncoding = this.registerParameter('output_encoding', new Parameter(initial));
    }
    this.outputSpace = Space.fromConfig(cfg.output_space);
    const organization = readout === 'sequence' ? 'sequence' : 'feature';
    if (this.outputSpace.dimensions !== hidden || this.outputSpace.organization !== organization) {
      throw new ValueError('space must match native hidden size and output organization');
    }
    this.contextSpace = cfg.context_space ? Space.fromConfig(cfg.context_space) : null;
    if (this.contextSpace && this.contextSpace.dimensions !== hidden) throw new ValueError('context_space must match ViT hidden size');
    const size = native.get('image_size');
    this.imageSize = Object.freeze(Array.isArray(size) ? [size[0] as number, size[1] as number] : [size as number, size as number]) as readonly [number, number];
    const patch = native.get('patch_size');
    this.patchSize = Object.freeze(Array.isArray(patch) ? [patch[0] as number, patch[1] as number] : [patch as number, patch as number]) as readonly [number, number];
    if (this.imageSize.some((value, index) => value % this.patchSize[index]!)) throw new ValueError('image_size must be divisible by patch_size');
    if (this.processor.doNormalize) {
      const mean = this.processor.imageMean;
      const std = this.processor.imageStd;
      const channels = native.number('num_channels');
      if (mean.length !== channels || std.length !== channels || std.some((value) => value <= 0)) {
        throw new ValueError('processor normalization must match channels with positive std');
      }
    }
  }

  /**
   * Run the owned processor assets (``ViTImageProcessor(images=...)``) over
   * tensors, decoded {@link RasterImage}s or string sources (file paths,
   * base64 text, data URIs); output uses processed-image coordinates. Use
   * {@link apreprocess} for ``http(s)://`` URLs.
   */
  preprocess(images: ImageInputs): ProcessedImages {
    return this.processor.preprocess(images);
  }

  /** {@link preprocess} that also fetches ``http(s)://`` image URLs. */
  apreprocess(images: ImageInputs): Promise<ProcessedImages> {
    return this.processor.apreprocess(images);
  }

  private pixels(value: unknown): { pixels: Tensor; single: boolean } {
    const processed = isPlainObject(value);
    if (processed && Object.keys(value).some((key) => key !== 'pixel_values' && key !== 'sources')) {
      throw new ValueError('processed input supports only pixel_values and sources');
    }
    let pixels = processed ? value.pixel_values : value;
    if (!(pixels instanceof Tensor) || (pixels.ndim !== 3 && pixels.ndim !== 4)) throw new ValueError('expected CHW/BCHW tensor or pixel_values mapping');
    const single = pixels.ndim === 3;
    if (single) pixels = pixels.unsqueeze(0);
    const batch = pixels as Tensor;
    if (!batch.isFloatingPoint || !batch.allFinite()) throw new ValueError('pixels must be finite floating point values');
    const channels = this.model.config.number('num_channels');
    if (batch.shape[0] === 0 || batch.shape[1] !== channels || batch.shape[2] !== this.imageSize[0] || batch.shape[3] !== this.imageSize[1]) {
      throw new ValueError('pixel batch, channels and size must match ViT configuration');
    }
    if (!processed && (batch.min().item() < 0 || batch.max().item() > 1)) throw new ValueError('raw floating pixels must be in [0,1]');
    const weight = this.model.embeddings.patch_embeddings.projection.weight;
    let result = batch.dtype === weight.dtype ? batch : batch.to(weight.dtype);
    if (!processed && this.processor.doNormalize) {
      const mean = tensor(this.processor.imageMean, { dtype: result.dtype }).reshape(1, -1, 1, 1);
      const std = tensor(this.processor.imageStd, { dtype: result.dtype }).reshape(1, -1, 1, 1);
      result = result.sub(mean).div(std);
    }
    return { pixels: result, single };
  }

  forward(value: Tensor | ProcessedImages, context: Context | null): Latent {
    const ctx = context ?? {};
    if (!isPlainObject(ctx)) throw new ValueError('context must be a mapping');
    if (Object.keys(ctx).some((key) => key !== 'latents')) throw new ValueError('ImageEncoder context supports only ordered latents');
    const { pixels, single } = this.pixels(value);
    const latents = ctx.latents ?? [];
    if (!Array.isArray(latents)) throw new ValueError('context latents must be an ordered list or tuple');
    const sources: string[] = isPlainObject(value) ? [...((value as ProcessedImages).sources ?? [])] : [];
    if (!sources.every((source) => typeof source === 'string')) throw new ValueError('image sources must be strings');
    let hidden: Tensor;
    if (latents.length || this.readout === 'output_encoding') {
      if (latents.length && this.contextSpace === null) throw new ValueError('context requires an explicit context_space');
      const embedded = this.model.embed(pixels);
      const [batch, tokens] = embedded.shape as [number, number];
      const pieces: Tensor[] = [];
      const masks: Tensor[] = [];
      for (const latent of latents) {
        const [sequence, mask] = asSequence(latent, this.contextSpace!);
        if (sequence.shape[0] !== batch) throw new ValueError('context batch must match image batch');
        pieces.push(sequence.dtype === embedded.dtype ? sequence : sequence.to(embedded.dtype));
        masks.push(mask);
        sources.push(...(latent as Latent).sources);
      }
      const prefixLength = pieces.reduce((total, piece) => total + piece.shape[1]!, 0);
      pieces.push(embedded);
      masks.push(ones([batch, tokens], { dtype: 'bool' }));
      if (this.readout === 'output_encoding') {
        pieces.push(this.outputEncoding!.expand(batch, 1, this.outputEncoding!.shape[2]!));
        masks.push(ones([batch, 1], { dtype: 'bool' }));
      }
      const combined = cat(pieces, 1);
      const mask = cat(masks, 1);
      const bias = mask.all().item() ? null : keyPaddingBias(mask, combined.dtype);
      hidden = this.model.forward({ inputsEmbeds: combined, attentionBias: bias }).lastHiddenState.slice(1, prefixLength);
    } else {
      hidden = this.model.forward({ pixelValues: pixels }).lastHiddenState;
    }
    let result: Tensor;
    let coordinates: Tensor | null = null;
    if (this.readout === 'output_encoding') {
      result = hidden.select(1, hidden.shape[1]! - 1);
    } else if (this.readout === 'pooled') {
      result = hidden.select(1, 0);
    } else {
      result = hidden.slice(1, 1);
      const [ph, pw] = this.patchSize;
      const ys = arange(0, this.imageSize[0] / ph, 1, { dtype: result.dtype }).mul(ph).add(ph / 2);
      const xs = arange(0, this.imageSize[1] / pw, 1, { dtype: result.dtype }).mul(pw).add(pw / 2);
      const [y, x] = meshgrid(ys, xs);
      const grid = stack([y!, x!], -1).reshape(1, -1, 2);
      coordinates = grid.expand(result.shape[0]!, grid.shape[1]!, 2);
    }
    let mask = ones(result.shape.slice(0, -1), { dtype: 'bool' });
    if (single) {
      result = result.select(0, 0);
      mask = mask.select(0, 0);
      if (coordinates !== null) coordinates = coordinates.select(0, 0);
    }
    const readouts: Record<ImageReadout, string> = { sequence: 'patch-states', pooled: 'native-cls', output_encoding: 'output_encoding' };
    return new Latent(result, this.outputSpace, {
      mask, coordinates, sources,
      metadata: {
        readout: readouts[this.readout],
        readout_initialization: this.readout === 'output_encoding' ? 'untrained' : 'native',
        coordinate_space: 'processed-image-pixels',
        initialization: this.config.foundation ? 'foundation' : 'random',
        foundation: this.config.foundation ?? null,
        context_conditioning: 'embedding-attention',
      },
    });
  }

  /** Fingerprint current owned processor semantics with fixed architecture. */
  override configuration(): JsonObject {
    const config = super.configuration();
    config.processor = this.processor.toJson();
    return config;
  }

  protected override async savePretrainedAssets(directory: string): Promise<void> {
    const asset = join(directory, 'vision_processor.json');
    if (await isSymlink(asset)) await unlink(asset);
    await writeFile(asset, pythonJsonDumps(this.configuration().processor, { sortKeys: true }), 'utf8');
  }

  static override async loadPretrainedConfig(config: JsonObject, directory: string): Promise<JsonObject> {
    const saved = parseJsonStrict(await readFile(join(directory, 'vision_processor.json'), 'utf8'));
    if (!jsonEqual(saved, config.processor)) throw new ValueError('processor asset differs from model configuration');
    return config;
  }

  /** Load only known native ViT weights and processor, never Hub code. */
  static async fromFoundation<T extends ImageEncoder>(
    this: new (config: unknown, internals?: VisionInternals) => T, repo: string, options: ImageFoundationOptions,
  ): Promise<T> {
    const { outputSpace, readout = 'sequence', contextSpace = null, revision = null, device, ...hub } = options;
    requireCpu(device);
    const { path } = await resolveArtifactDirectory(repo, { ...hub, revision, allowPatterns: ['config.json', 'preprocessor_config.json'] });
    const raw = parseJsonStrict(await readFile(join(path, 'config.json'), 'utf8'));
    if (!isPlainObject(raw) || raw.model_type !== 'vit') throw new ValueError('foundation must be a ViT checkpoint');
    const loaded = await loadNativeFoundation(repo, { ...hub, revision, head: 'base', addPoolingLayer: false, tokenizer: false });
    if (loaded.config.modelType !== 'vit') throw new ValueError('foundation must be a ViT checkpoint');
    const processor = parseJsonStrict(await readFile(join(loaded.directory, 'preprocessor_config.json'), 'utf8'));
    const config: JsonObject = {
      model: loaded.config.toDiffDict(),
      processor: new ImageProcessor(processor).toJson(),
      readout,
      output_space: spaceJson(outputSpace),
      context_space: spaceJson(contextSpace),
      foundation: { repo: String(repo), revision, resolved_revision: loaded.commitHash },
    };
    // Python constructs the encoder (drawing its random initialization), then loads the foundation weights.
    const result = new this(config);
    (result as unknown as { model: ViTModel }).model.loadStateDict(loaded.model.stateDict(), { strict: true });
    return result.eval();
  }
}
