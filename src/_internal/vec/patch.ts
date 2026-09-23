/**
 * Trainable image patch projection that retains patch-grid coordinates
 * (Python ``tensorcode/_internal/vec/patch.py``).
 *
 * The default convolution starts from ordinary random initialization. It is a
 * trainable projection mechanism, not a pretrained image understanding model.
 */
import { Module } from '../../nn/module.js';
import { Tensor, arange } from '../../nn/tensor.js';
import { Conv2d } from '../../nn/layers.js';
import { meshgrid, stack } from '../../nn/ops/shape.js';
import { ValueError } from '../../errors.js';
import type { Context } from '../../ops/base.js';
import { Latent, Space } from '../../ops/vec/latent.js';
import { LatentOperation } from '../latentOps.js';
import { validatedConfig } from '../operationConfig.js';
import { isPlainObject, type JsonObject } from '../json.js';
import { moduleConfiguration } from './configuration.js';

/** A module mapping ``BCHW`` images to ``BCHW`` patch features. */
export type PatchModule = Module & { forward(input: Tensor): Tensor };

function isInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value);
}

function pair(value: unknown): [number, number] {
  const result = typeof value === 'number' ? [value, value] : Array.isArray(value) ? [...value] : null;
  if (!result || result.length !== 2 || result.some((item) => !isInteger(item) || item <= 0)) {
    throw new ValueError('patch_size must be a positive integer or pair');
  }
  return result as [number, number];
}

function coordinatePair(value: unknown, name: string): [number, number] {
  const result = typeof value === 'number' ? [value, value] : Array.isArray(value) ? [...value] : null;
  if (!result || result.length !== 2 || result.some((item) => typeof item !== 'number' || !Number.isFinite(item))) {
    throw new ValueError(`${name} must be a number or pair`);
  }
  return result as [number, number];
}

const PATCH_INTERNAL: unique symbol = Symbol('tensorcode.vec.patch.internal');

interface PatchInternals {
  readonly [PATCH_INTERNAL]: true;
  readonly module: PatchModule;
  readonly patchSize: [number, number];
  readonly outputSpace: Space;
  readonly inChannels: number | null;
  readonly coordinateStride: unknown;
  readonly coordinateOffset: unknown;
}

export interface PatchFromModuleOptions {
  patchSize: number | readonly [number, number];
  outputSpace: Space;
  inChannels?: number | null;
  dimensions?: number | null;
  coordinateStride?: number | readonly [number, number] | null;
  coordinateOffset?: number | readonly [number, number] | null;
}

/** Project CHW/BCHW images to channel-last spatial patch latents. */
export class PatchEncoder extends LatentOperation<Tensor, Latent> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.patch.PatchEncoder';
  readonly patchSize: readonly [number, number];
  readonly outputSpace: Space;
  /** ``'pytorch-random'`` for configured encoders, ``'supplied'`` for ``fromModule``. */
  readonly initialization: 'pytorch-random' | 'supplied';
  readonly module: PatchModule;
  coordinateStride: readonly [number, number] | null = null;
  coordinateOffset: readonly [number, number] | null = null;

  constructor(config: unknown, internals?: PatchInternals) {
    if (internals?.[PATCH_INTERNAL]) {
      const supplied: JsonObject = {
        patch_size: [...internals.patchSize],
        output_space: internals.outputSpace.configuration() as unknown as JsonObject,
        dimensions: internals.outputSpace.dimensions,
      };
      if (internals.inChannels !== null) supplied.in_channels = internals.inChannels;
      super(supplied);
      this.patchSize = Object.freeze([...internals.patchSize]) as readonly [number, number];
      this.outputSpace = internals.outputSpace;
      this.initialization = 'supplied';
      this.module = this.registerModule('module', internals.module);
      this.setGeometry(internals.coordinateStride, internals.coordinateOffset);
      return;
    }
    const validated = validatedConfig(config, [
      'patch_size', 'output_space', 'in_channels', 'dimensions', 'coordinate_stride', 'coordinate_offset',
    ]);
    if (!('patch_size' in validated) || !('output_space' in validated)) throw new ValueError('PatchEncoder requires patch_size and output_space');
    if (!isPlainObject(validated.output_space)) throw new TypeError('output_space must be a Space configuration object');
    const outputSpace = Space.fromConfig(validated.output_space);
    if (outputSpace.organization !== 'spatial') throw new ValueError('PatchEncoder requires a spatial Space');
    const patchSize = pair(validated.patch_size);
    const inChannels = validated.in_channels;
    if (!isInteger(inChannels) || inChannels <= 0) throw new ValueError('in_channels is required for a newly initialized encoder');
    const dimensions = validated.dimensions ?? outputSpace.dimensions;
    if (!isInteger(dimensions) || dimensions !== outputSpace.dimensions) throw new ValueError('PatchEncoder dimensions must match its output_space');
    validated.patch_size = [...patchSize];
    validated.dimensions = dimensions;
    super(validated);
    this.patchSize = Object.freeze([...patchSize]) as readonly [number, number];
    this.outputSpace = outputSpace;
    this.initialization = 'pytorch-random';
    this.module = this.registerModule('module', new Conv2d(inChannels, dimensions, patchSize, { stride: patchSize }));
    this.setGeometry(validated.coordinate_stride, validated.coordinate_offset);
    this.config.coordinate_stride = [...this.coordinateStride!];
    this.config.coordinate_offset = [...this.coordinateOffset!];
  }

  /** Wrap a supplied module (Python ``from_module``); it cannot be saved as an artifact. */
  static fromModule(module: PatchModule, options: PatchFromModuleOptions): PatchEncoder {
    if (!(module instanceof Module) || typeof module.forward !== 'function') throw new TypeError('PatchEncoder module must be an nn.Module');
    const { outputSpace, dimensions } = options;
    if (!(outputSpace instanceof Space) || outputSpace.organization !== 'spatial') throw new ValueError('PatchEncoder requires a spatial Space');
    if (dimensions !== undefined && dimensions !== null && (!isInteger(dimensions) || dimensions !== outputSpace.dimensions)) {
      throw new ValueError('PatchEncoder dimensions must match its output_space');
    }
    const inChannels = options.inChannels ?? null;
    if (inChannels !== null && (!isInteger(inChannels) || inChannels <= 0)) throw new ValueError('in_channels must be a positive integer');
    return new PatchEncoder({}, {
      [PATCH_INTERNAL]: true, module, patchSize: pair(options.patchSize), outputSpace, inChannels,
      coordinateStride: options.coordinateStride ?? null, coordinateOffset: options.coordinateOffset ?? null,
    });
  }

  private setGeometry(stride: unknown, offset: unknown): void {
    const hasStride = stride !== null && stride !== undefined;
    const hasOffset = offset !== null && offset !== undefined;
    if (hasStride !== hasOffset) throw new ValueError('coordinate_stride and coordinate_offset must be supplied together');
    if (hasStride) {
      this.coordinateStride = Object.freeze(coordinatePair(stride, 'coordinate_stride'));
      this.coordinateOffset = Object.freeze(coordinatePair(offset, 'coordinate_offset'));
    } else if (this.module instanceof Conv2d) {
      const conv = this.module;
      this.coordinateStride = Object.freeze([conv.stride[0], conv.stride[1]]);
      const kernel = [0, 1].map((axis) => conv.dilation[axis]! * (conv.kernelSize[axis]! - 1) + 1);
      this.coordinateOffset = Object.freeze([-conv.padding[0] + kernel[0]! / 2, -conv.padding[1] + kernel[1]! / 2]);
    } else {
      this.coordinateStride = null;
      this.coordinateOffset = null;
    }
  }

  override async savePretrained(directory: string): Promise<string> {
    if (this.initialization === 'supplied') throw new ValueError('Cannot save a supplied PatchEncoder module as a reconstructible artifact');
    return super.savePretrained(directory);
  }

  forward(value: Tensor, context: Context | null): Latent {
    if (context) throw new ValueError('PatchEncoder does not consume context');
    if (!(value instanceof Tensor) || (value.ndim !== 3 && value.ndim !== 4)) throw new ValueError('PatchEncoder expects a CHW or BCHW tensor');
    const single = value.ndim === 3;
    const batch = single ? value.unsqueeze(0) : value;
    const encoded = this.module.forward(batch);
    if (!(encoded instanceof Tensor) || encoded.ndim !== 4) throw new ValueError('PatchEncoder module must return a BCHW tensor');
    if (encoded.shape[0] !== batch.shape[0]) throw new ValueError('PatchEncoder module must preserve the input batch count');
    if (encoded.shape[1] !== this.outputSpace.dimensions) throw new ValueError('PatchEncoder module output channels must match its output_space');
    const [count, , rows, columns] = encoded.shape as [number, number, number, number];
    let coordinates: Tensor | null = null;
    if (this.coordinateStride !== null && this.coordinateOffset !== null) {
      // Coordinates are actual receptive-field centers in source pixel space.
      const rowCenters = arange(0, rows, 1, { dtype: encoded.dtype }).mul(this.coordinateStride[0]).add(this.coordinateOffset[0]);
      const columnCenters = arange(0, columns, 1, { dtype: encoded.dtype }).mul(this.coordinateStride[1]).add(this.coordinateOffset[1]);
      const [rowGrid, columnGrid] = meshgrid(rowCenters, columnCenters);
      coordinates = stack([rowGrid!, columnGrid!], -1).unsqueeze(0).expand(count, rows, columns, 2);
    }
    let result = encoded.permute(0, 2, 3, 1);
    if (single) {
      result = result.select(0, 0);
      if (coordinates !== null) coordinates = coordinates.select(0, 0);
    }
    return new Latent(result, this.outputSpace, { coordinates });
  }

  override configuration(): JsonObject {
    const result = super.configuration();
    result.coordinate_stride = this.coordinateStride === null ? null : [...this.coordinateStride];
    result.coordinate_offset = this.coordinateOffset === null ? null : [...this.coordinateOffset];
    if (this.initialization === 'supplied') {
      result.module = moduleConfiguration(this.module);
      result.initialization = 'supplied';
    }
    return result;
  }
}
