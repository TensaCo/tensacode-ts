/**
 * Image preprocessing over decoded tensors: the ``ViTImageProcessor``
 * (transformers 5 torchvision backend) equivalent used by ``ImageEncoder`` and
 * the antialiased ``F.interpolate`` resize used by ``Scene``.
 *
 * Image *file* decoding (PNG/JPEG/PIL) is out of scope: callers supply decoded
 * ``CHW``/``BCHW`` tensors, either ``uint8`` in ``[0, 255]`` (the processor's
 * native path, matching ``PIL.Image`` inputs in Python) or floating point in
 * ``[0, 1]`` (already rescaled; the processor then skips ``rescale``).
 *
 * Resizing reproduces PyTorch's separable antialiased interpolation
 * (``_upsample_{bilinear,bicubic}2d_aa``): width first, then height. ``uint8``
 * images use PyTorch's fixed-point int16 weights with a ``uint8`` intermediate,
 * exactly like torchvision's native ``uint8`` resize.
 */
import { Tensor, tensor } from '../../nn/tensor.js';
import { NotImplementedError, ValueError } from '../../errors.js';
import { deepCopy, isPlainObject, jsonEqual, type JsonObject, type JsonValue } from '../json.js';

export type InterpolationMode = 'nearest-exact' | 'bilinear' | 'bicubic' | 'lanczos';

/** PIL resampling codes accepted by processor configurations. */
export const PIL_RESAMPLING: Readonly<Record<number, InterpolationMode | null>> = Object.freeze({
  0: 'nearest-exact', 1: 'lanczos', 2: 'bilinear', 3: 'bicubic', 4: null, 5: null,
});

function filterLinear(x: number): number {
  const value = Math.abs(x);
  return value < 1 ? 1 - value : 0;
}

function filterCubic(x: number): number {
  // PyTorch's antialiased bicubic filter uses a = -0.5 (like PIL).
  const a = -0.5;
  const value = Math.abs(x);
  if (value < 1) return ((a + 2) * value - (a + 3)) * value * value + 1;
  if (value < 2) return (((value - 5) * value + 8) * value - 4) * a;
  return 0;
}

function filterLanczos(x: number): number {
  // Lanczos with a = 3: ``sinc(x) * sinc(x / 3)`` on ``(-3, 3)``.
  const sinc = (value: number): number => {
    if (value === 0) return 1;
    const scaled = value * Math.PI;
    return Math.sin(scaled) / scaled;
  };
  return x > -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0;
}

interface AxisWeights {
  starts: Int32Array;
  sizes: Int32Array;
  /** ``[output, maxInterp]`` normalized weights. */
  weights: Float64Array;
  maxInterp: number;
  maxWeight: number;
}

/** PyTorch ``HelperInterpBase::_compute_index_ranges_weights`` (antialias, ``align_corners=False``). */
function axisWeights(inputSize: number, outputSize: number, mode: 'bilinear' | 'bicubic' | 'lanczos'): AxisWeights {
  const interpSize = mode === 'bilinear' ? 2 : mode === 'bicubic' ? 4 : 6;
  const filter = mode === 'bilinear' ? filterLinear : mode === 'bicubic' ? filterCubic : filterLanczos;
  const scale = inputSize / outputSize;
  const support = scale >= 1 ? interpSize * 0.5 * scale : interpSize * 0.5;
  const maxInterp = Math.ceil(support) * 2 + 1;
  const invscale = scale >= 1 ? 1 / scale : 1;
  const starts = new Int32Array(outputSize);
  const sizes = new Int32Array(outputSize);
  const weights = new Float64Array(outputSize * maxInterp);
  let maxWeight = 0;
  for (let i = 0; i < outputSize; i += 1) {
    const center = scale * (i + 0.5);
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    let xsize = Math.min(Math.trunc(center + support + 0.5), inputSize) - xmin;
    xsize = Math.min(Math.max(xsize, 0), maxInterp);
    let total = 0;
    for (let j = 0; j < xsize; j += 1) {
      const w = filter((j + xmin - center + 0.5) * invscale);
      weights[i * maxInterp + j] = w;
      total += w;
    }
    if (total !== 0) {
      for (let j = 0; j < xsize; j += 1) {
        weights[i * maxInterp + j]! /= total;
        maxWeight = Math.max(maxWeight, weights[i * maxInterp + j]!);
      }
    }
    starts[i] = xmin;
    sizes[i] = xsize;
  }
  return { starts, sizes, weights, maxInterp, maxWeight };
}

/** Fixed-point int16 weights and precision (PyTorch ``_compute_index_ranges_int16_weights``). */
function int16Weights(axis: AxisWeights): { weights: Int32Array; precision: number } {
  let precision = 0;
  for (precision = 0; precision < 22; precision += 1) {
    const next = Math.trunc(0.5 + axis.maxWeight * 2 ** (precision + 1));
    if (next >= 2 ** 15) break;
  }
  const weights = new Int32Array(axis.weights.length);
  for (let index = 0; index < weights.length; index += 1) {
    const v = axis.weights[index]! * 2 ** precision;
    weights[index] = v < 0 ? Math.trunc(-0.5 + v) : Math.trunc(0.5 + v);
  }
  return { weights, precision };
}

/**
 * Resize ``values`` laid out as ``[planes, height, width]`` along one axis.
 * ``integer`` selects the uint8 fixed-point path.
 */
function resizeAxis(
  values: Float64Array, planes: number, height: number, width: number, output: number, horizontal: boolean,
  mode: InterpolationMode, integer: boolean,
): Float64Array {
  const inputSize = horizontal ? width : height;
  const outHeight = horizontal ? height : output;
  const outWidth = horizontal ? output : width;
  const result = new Float64Array(planes * outHeight * outWidth);
  const read = (plane: number, row: number, column: number): number => values[(plane * height + row) * width + column]!;
  if (mode === 'nearest-exact') {
    const scale = Math.fround(inputSize / output);
    for (let plane = 0; plane < planes; plane += 1) {
      for (let row = 0; row < outHeight; row += 1) {
        for (let column = 0; column < outWidth; column += 1) {
          const target = horizontal ? column : row;
          const source = Math.min(Math.floor(Math.fround((target + 0.5) * scale)), inputSize - 1);
          result[(plane * outHeight + row) * outWidth + column] = horizontal ? read(plane, row, source) : read(plane, source, column);
        }
      }
    }
    return result;
  }
  const axis = axisWeights(inputSize, output, mode);
  const fixed = integer ? int16Weights(axis) : null;
  for (let plane = 0; plane < planes; plane += 1) {
    for (let row = 0; row < outHeight; row += 1) {
      for (let column = 0; column < outWidth; column += 1) {
        const target = horizontal ? column : row;
        const start = axis.starts[target]!;
        const size = axis.sizes[target]!;
        const offset = target * axis.maxInterp;
        let value: number;
        if (fixed) {
          let acc = 2 ** (fixed.precision - 1);
          for (let j = 0; j < size; j += 1) {
            const sample = horizontal ? read(plane, row, start + j) : read(plane, start + j, column);
            acc += sample * fixed.weights[offset + j]!;
          }
          value = Math.min(Math.max(Math.floor(acc / 2 ** fixed.precision), 0), 255);
        } else {
          value = 0;
          for (let j = 0; j < size; j += 1) {
            const sample = horizontal ? read(plane, row, start + j) : read(plane, start + j, column);
            value += sample * axis.weights[offset + j]!;
          }
        }
        result[(plane * outHeight + row) * outWidth + column] = value;
      }
    }
  }
  return result;
}

/**
 * Antialiased resize of a ``CHW``/``BCHW`` tensor to ``[height, width]``
 * (``F.interpolate(..., align_corners=False, antialias=True)`` for bilinear and
 * bicubic, ``nearest-exact`` otherwise). ``uint8`` inputs return ``uint8``
 * results computed with PyTorch's fixed-point path; other inputs are computed
 * in float and returned as ``float32``. Not differentiable.
 */
export function resizeImage(image: Tensor, size: readonly [number, number], mode: InterpolationMode = 'bilinear'): Tensor {
  if (!(image instanceof Tensor) || (image.ndim !== 3 && image.ndim !== 4)) throw new ValueError('resize expects a CHW or BCHW tensor');
  const [outHeight, outWidth] = size;
  if (!Number.isInteger(outHeight) || !Number.isInteger(outWidth) || outHeight < 1 || outWidth < 1) {
    throw new ValueError('resize size must contain positive integers');
  }
  const shape = image.shape;
  const height = shape[shape.length - 2]!;
  const width = shape[shape.length - 1]!;
  const integer = image.dtype === 'uint8';
  const dtype = integer ? 'uint8' : (image.dtype === 'float64' ? 'float64' : 'float32');
  if (height === outHeight && width === outWidth) return image.dtype === dtype ? image.detach() : image.detach().to(dtype);
  const planes = image.numel / (height * width);
  let values: Float64Array<ArrayBufferLike> = Float64Array.from(image.data);
  let currentWidth = width;
  if (outWidth !== width) {
    values = resizeAxis(values, planes, height, width, outWidth, true, mode, integer);
    currentWidth = outWidth;
  }
  if (outHeight !== height) values = resizeAxis(values, planes, height, currentWidth, outHeight, false, mode, integer);
  return tensor(values, { shape: [...shape.slice(0, -2), outHeight, outWidth], dtype });
}

// ---------------------------------------------------------------------------
// ViTImageProcessor.
// ---------------------------------------------------------------------------

/** ``ViTImageProcessor`` class defaults (transformers 5.17). */
export const VIT_PROCESSOR_DEFAULTS: Readonly<JsonObject> = Object.freeze({
  do_normalize: true,
  do_rescale: true,
  do_resize: true,
  image_mean: [0.5, 0.5, 0.5],
  image_processor_type: 'ViTImageProcessor',
  image_std: [0.5, 0.5, 0.5],
  resample: 2,
  rescale_factor: 1 / 255,
  size: { height: 224, width: 224 },
});

/** Keys transformers drops when constructing an image processor. */
const DROPPED_KEYS = new Set(['feature_extractor_type', 'processor_class', '_processor_class']);

function sizeDict(value: JsonValue | undefined): JsonValue | undefined {
  if (typeof value === 'number') return { height: value, width: value };
  if (Array.isArray(value) && value.length === 2) return { height: value[0]!, width: value[1]! };
  return value;
}

/**
 * ``json.loads(ViTImageProcessor(**config).to_json_string())``: class
 * defaults overlaid with the supplied values, integer sizes expanded to
 * ``{height, width}`` and legacy feature-extractor keys dropped.
 */
export function vitProcessorConfig(config: unknown): JsonObject {
  if (!isPlainObject(config)) throw new ValueError('processor configuration must be a JSON object');
  const result: JsonObject = deepCopy(VIT_PROCESSOR_DEFAULTS as JsonObject);
  for (const [key, value] of Object.entries(deepCopy(config as JsonObject))) {
    if (DROPPED_KEYS.has(key)) continue;
    result[key] = value;
  }
  result.image_processor_type = 'ViTImageProcessor';
  for (const key of ['size', 'crop_size']) {
    if (key in result) result[key] = sizeDict(result[key]) as JsonValue;
  }
  return result;
}

function numberList(value: JsonValue | undefined, name: string): number[] {
  if (typeof value === 'number') return [value];
  if (!Array.isArray(value) || !value.every((item) => typeof item === 'number')) throw new ValueError(`${name} must be a number or list of numbers`);
  return value as number[];
}

/**
 * Tensor-only ``ViTImageProcessor``: resize, optional center crop, rescale and
 * normalize. Settings are live (mutable) and serialize with {@link toJson}.
 */
export class ImageProcessor {
  /** Live processor settings in Python spelling (``image_mean``, ``resample`` ...). */
  values: JsonObject;

  constructor(config: unknown = {}) {
    this.values = vitProcessorConfig(config);
  }

  get doResize(): boolean { return this.values.do_resize === true; }
  set doResize(value: boolean) { this.values.do_resize = value; }
  get doRescale(): boolean { return this.values.do_rescale === true; }
  set doRescale(value: boolean) { this.values.do_rescale = value; }
  get doNormalize(): boolean { return this.values.do_normalize === true; }
  set doNormalize(value: boolean) { this.values.do_normalize = value; }
  get rescaleFactor(): number { return this.values.rescale_factor as number; }
  set rescaleFactor(value: number) { this.values.rescale_factor = value; }
  get imageMean(): number[] { return numberList(this.values.image_mean, 'image_mean'); }
  set imageMean(value: number[]) { this.values.image_mean = [...value]; }
  get imageStd(): number[] { return numberList(this.values.image_std, 'image_std'); }
  set imageStd(value: number[]) { this.values.image_std = [...value]; }
  get resample(): number { return this.values.resample as number; }
  set resample(value: number) { this.values.resample = value; }
  get size(): JsonObject { return this.values.size as JsonObject; }
  set size(value: JsonObject) { this.values.size = deepCopy(value); }

  /** ``json.loads(processor.to_json_string())``. */
  toJson(): JsonObject {
    return vitProcessorConfig(this.values);
  }

  equals(other: ImageProcessor): boolean {
    return jsonEqual(this.toJson(), other.toJson());
  }

  private interpolation(): InterpolationMode {
    const code = this.values.resample ?? 2;
    const mode = typeof code === 'number' ? PIL_RESAMPLING[code] : undefined;
    if (mode === undefined) throw new ValueError(`unsupported resample ${JSON.stringify(code)}`);
    if (mode === null) throw new NotImplementedError(`resample ${code} (BOX/HAMMING) is not available in the TypeScript port`);
    return mode;
  }

  private targetSize(height: number, width: number): [number, number] {
    const size = this.values.size;
    if (!isPlainObject(size)) throw new ValueError('processor size must be a mapping');
    if (typeof size.height === 'number' && typeof size.width === 'number') return [size.height, size.width];
    if (typeof size.shortest_edge === 'number') {
      const short = size.shortest_edge;
      const [small, large] = height <= width ? [height, width] : [width, height];
      let newShort = short;
      let newLong = Math.trunc((short * large) / small);
      if (typeof size.longest_edge === 'number' && newLong > size.longest_edge) {
        newShort = Math.trunc((size.longest_edge * newShort) / newLong);
        newLong = size.longest_edge;
      }
      return height <= width ? [newShort, newLong] : [newLong, newShort];
    }
    throw new NotImplementedError("processor size must contain 'height' and 'width' or 'shortest_edge'");
  }

  private centerCrop(image: Tensor): Tensor {
    const crop = this.values.crop_size;
    if (!isPlainObject(crop) || typeof crop.height !== 'number' || typeof crop.width !== 'number') {
      throw new ValueError("The size dictionary must have keys 'height' and 'width'");
    }
    const height = image.shape[image.ndim - 2]!;
    const width = image.shape[image.ndim - 1]!;
    if (crop.height > height || crop.width > width) throw new NotImplementedError('center crop padding is not available in the TypeScript port');
    const top = Math.trunc((height - crop.height) / 2);
    const left = Math.trunc((width - crop.width) / 2);
    return image.slice(image.ndim - 2, top, top + crop.height).slice(image.ndim - 1, left, left + crop.width);
  }

  /**
   * Process decoded images (``CHW``/``BCHW`` tensors or an array of ``CHW``
   * tensors, ``uint8`` 0-255 or float 0-1). Returns ``{pixel_values}`` as a
   * float32 ``BCHW`` tensor in processed-image coordinates.
   */
  preprocess(images: Tensor | readonly Tensor[]): { pixel_values: Tensor } {
    const list: Tensor[] = [];
    if (images instanceof Tensor) {
      if (images.ndim === 3) list.push(images);
      else if (images.ndim === 4) for (let index = 0; index < images.shape[0]!; index += 1) list.push(images.select(0, index));
      else throw new ValueError('images must be CHW or BCHW tensors');
    } else if (Array.isArray(images) && images.length) {
      for (const image of images) {
        if (!(image instanceof Tensor) || image.ndim !== 3) throw new ValueError('images must be CHW tensors');
        list.push(image);
      }
    } else {
      throw new TypeError('images must be decoded tensors; image file decoding is not available in the TypeScript port');
    }
    const processed = list.map((image) => this.processOne(image.detach()));
    const first = processed[0]!.shape;
    if (processed.some((item) => item.shape.some((size, index) => size !== first[index]))) {
      throw new ValueError('processed images must share one size to be batched');
    }
    const flat = new Float32Array(processed.length * processed[0]!.numel);
    processed.forEach((item, index) => flat.set(item.data as ArrayLike<number>, index * item.numel));
    return { pixel_values: tensor(flat, { shape: [processed.length, ...first], dtype: 'float32' }) };
  }

  private processOne(image: Tensor): Tensor {
    const integer = image.dtype === 'uint8';
    if (!integer && !image.isFloatingPoint) throw new ValueError('images must be uint8 (0-255) or floating point (0-1) tensors');
    let current = image;
    if (this.doResize) {
      const [height, width] = this.targetSize(current.shape[1]!, current.shape[2]!);
      current = resizeImage(current, [height, width], this.interpolation());
    }
    if (this.values.do_center_crop === true) current = this.centerCrop(current);
    const channels = current.shape[0]!;
    const plane = current.shape[1]! * current.shape[2]!;
    const source = current.data;
    const out = new Float32Array(current.numel);
    const mean = this.doNormalize ? this.imageMean : [];
    const std = this.doNormalize ? this.imageStd : [];
    if (this.doNormalize && ((mean.length !== 1 && mean.length !== channels) || (std.length !== 1 && std.length !== channels))) {
      throw new ValueError('processor normalization must match image channels');
    }
    const factor = this.rescaleFactor;
    for (let channel = 0; channel < channels; channel += 1) {
      let m = 0;
      let s = 1;
      if (this.doNormalize) {
        m = Math.fround(mean.length === 1 ? mean[0]! : mean[channel]!);
        s = Math.fround(std.length === 1 ? std[0]! : std[channel]!);
        if (integer && this.doRescale) {
          // transformers fuses rescale and normalize: (x - mean / f) / (std / f).
          m = Math.fround(m * Math.fround(1 / factor));
          s = Math.fround(s * Math.fround(1 / factor));
        }
      }
      for (let index = 0; index < plane; index += 1) {
        const position = channel * plane + index;
        let value = source[position]!;
        if (this.doNormalize) value = Math.fround(Math.fround(value - m) / s);
        else if (integer && this.doRescale) value = Math.fround(value * factor);
        out[position] = value;
      }
    }
    return tensor(out, { shape: [...current.shape], dtype: 'float32' });
  }
}
