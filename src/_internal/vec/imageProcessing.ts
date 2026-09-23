/**
 * Image preprocessing: the ``ViTImageProcessor`` (transformers 5 torchvision
 * backend) equivalent used by ``ImageEncoder``, and the antialiased
 * ``F.interpolate`` resize used by ``Scene`` and the Idefics3 processor.
 *
 * The processor accepts what transformers' torchvision backend accepts:
 * decoded tensors (``CHW``, ``HWC`` or batched, any dtype), decoded
 * {@link RasterImage}s (the counterpart of ``PIL.Image.Image``), and string
 * sources (file paths, base64 text, ``data:image/...`` URIs and, through
 * {@link ImageProcessor.apreprocess}, ``http(s)://`` URLs) decoded exactly
 * like ``torchvision.io.decode_image(source, mode=RGB)``, in any nesting of
 * lists. Encoded bytes are rejected like Python rejects ``bytes``: decode them
 * first with ``decodeImage`` or ``openImage``.
 *
 * Resizing reproduces torchvision's ``resize`` over ``F.interpolate``:
 * separable antialiased filters (width first, then height), PyTorch's
 * fixed-point int16 weights for ``uint8`` images, float32 arithmetic for other
 * dtypes followed by torchvision's rounding and casting, ``nearest-exact``
 * copies, and PyTorch's ``NotImplementedError`` for ``BOX``/``HAMMING``.
 */
import { Tensor, tensor } from '../../nn/tensor.js';
import { roundToDType, type DType } from '../../nn/dtype.js';
import { IndexError, NotImplementedError, ValueError } from '../../errors.js';

export { IndexError };
import { deepCopy, isPlainObject, jsonEqual, type JsonObject, type JsonValue } from '../json.js';
import { RasterImage } from '../image/raster.js';
import { fma, fmaf, sin as glibcSin, sinf as glibcSinf } from '../../nn/randomMath.js';
import { torchvisionDecode, type DecodedArray } from '../image/torchvision.js';
import { fetchSourceBytes, sourceBytes } from '../image/load.js';

export type InterpolationMode = 'nearest-exact' | 'bilinear' | 'bicubic' | 'lanczos' | 'box' | 'hamming';

/** PIL resampling codes accepted by processor configurations (transformers' ``pil_torch_interpolation_mapping``). */
export const PIL_RESAMPLING: Readonly<Record<number, InterpolationMode | null>> = Object.freeze({
  0: 'nearest-exact', 1: 'lanczos', 2: 'bilinear', 3: 'bicubic', 4: 'box', 5: 'hamming',
});

/** Inputs the processor accepts (``ImageInput`` plus string sources, nested in arrays). */
export type ImageSource = Tensor | RasterImage | string;
export type ImageInputs = ImageSource | readonly ImageInputs[];

// ---------------------------------------------------------------------------
// Filters and weights.
// ---------------------------------------------------------------------------

type FilterMode = 'bilinear' | 'bicubic' | 'lanczos';

function filterDouble(mode: FilterMode, x: number): number {
  const value = Math.abs(x);
  if (mode === 'bilinear') return value < 1 ? 1 - value : 0;
  if (mode === 'bicubic') {
    const a = -0.5;
    if (value < 1) return fma(fma(a + 2, value, -(a + 3)) * value, value, 1);
    if (value < 2) return fma(fma(fma(a, value, -5 * a), value, 8 * a), value, -4 * a);
    return 0;
  }
  const sinc = (v: number): number => {
    if (v === 0) return 1;
    const scaled = v * Math.PI;
    return glibcSin(scaled) / scaled;
  };
  return value < 3 ? sinc(value) * sinc(value / 3) : 0;
}

const f32 = Math.fround;

const PI_F32 = f32(Math.PI);

/** PyTorch's ``aa_filter<float>``: float32 arithmetic (``std::sin`` on floats). */
function filterFloat(mode: FilterMode, x: number): number {
  const value = Math.abs(x);
  if (mode === 'bilinear') return value < 1 ? f32(1 - value) : 0;
  if (mode === 'bicubic') {
    const a = f32(-0.5);
    if (value < 1) return fmaf(f32(fmaf(f32(a + 2), value, -f32(a + 3)) * value), value, 1);
    if (value < 2) return fmaf(fmaf(fmaf(a, value, -f32(5 * a)), value, f32(8 * a)), value, -f32(4 * a));
    return 0;
  }
  // ``sinc_filter(x)`` runs on floats; ``sinc_filter(x / 3.0)`` deduces double.
  const sincFloat = (v: number): number => {
    if (v === 0) return 1;
    const scaled = f32(v * PI_F32);
    return f32(glibcSinf(scaled) / scaled);
  };
  const sincDouble = (v: number): number => {
    if (v === 0) return 1;
    const scaled = v * Math.PI;
    return glibcSin(scaled) / scaled;
  };
  return value < 3 ? f32(sincFloat(value) * sincDouble(value / 3)) : 0;
}

interface AxisWeights {
  starts: Int32Array;
  sizes: Int32Array;
  /** ``[output, maxInterp]`` normalized weights. */
  weights: Float64Array;
  maxInterp: number;
  maxWeight: number;
}

const INTERP_SIZE: Record<FilterMode, number> = { bilinear: 2, bicubic: 4, lanczos: 6 };

/**
 * PyTorch ``HelperInterpBase::_compute_index_ranges_weights`` (antialias,
 * ``align_corners=False``) with ``scalar_t`` = double (``float32 = false``)
 * or float (``float32 = true``).
 */
function axisWeights(inputSize: number, outputSize: number, mode: FilterMode, float32 = false): AxisWeights {
  const interpSize = INTERP_SIZE[mode];
  const cast = float32 ? f32 : (value: number): number => value;
  const scale = cast(inputSize / outputSize);
  const support = cast(scale >= 1 ? interpSize * 0.5 * scale : interpSize * 0.5);
  const maxInterp = Math.ceil(support) * 2 + 1;
  const invscale = cast(scale >= 1 ? 1 / scale : 1);
  const starts = new Int32Array(outputSize);
  const sizes = new Int32Array(outputSize);
  const weights = new Float64Array(outputSize * maxInterp);
  let maxWeight = 0;
  for (let i = 0; i < outputSize; i += 1) {
    const center = cast(scale * (i + 0.5));
    const xmin = Math.max(Math.trunc(cast(center - support) + 0.5), 0);
    let xsize = Math.min(Math.trunc(cast(center + support) + 0.5), inputSize) - xmin;
    xsize = Math.min(Math.max(xsize, 0), maxInterp);
    let total = 0;
    for (let j = 0; j < xsize; j += 1) {
      const w = float32
        ? filterFloat(mode, f32((f32(j + xmin - center) + 0.5) * invscale))
        : filterDouble(mode, (j + xmin - center + 0.5) * invscale);
      weights[i * maxInterp + j] = w;
      total = cast(total + w);
    }
    if (total !== 0) {
      for (let j = 0; j < xsize; j += 1) {
        weights[i * maxInterp + j] = cast(weights[i * maxInterp + j]! / total);
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

type Kernel = 'uint8' | 'float32' | 'float64';

/** Resize ``values`` laid out as ``[planes, height, width]`` along one axis. */
function resizeAxis(
  values: Float64Array, planes: number, height: number, width: number, output: number, horizontal: boolean,
  mode: InterpolationMode, kernel: Kernel,
): Float64Array {
  const inputSize = horizontal ? width : height;
  const outHeight = horizontal ? height : output;
  const outWidth = horizontal ? output : width;
  const result = new Float64Array(planes * outHeight * outWidth);
  const read = (plane: number, row: number, column: number): number => values[(plane * height + row) * width + column]!;
  if (mode === 'nearest-exact') {
    const scale = f32(inputSize / output);
    for (let plane = 0; plane < planes; plane += 1) {
      for (let row = 0; row < outHeight; row += 1) {
        for (let column = 0; column < outWidth; column += 1) {
          const target = horizontal ? column : row;
          const source = Math.min(Math.floor(f32((target + 0.5) * scale)), inputSize - 1);
          result[(plane * outHeight + row) * outWidth + column] = horizontal ? read(plane, row, source) : read(plane, source, column);
        }
      }
    }
    return result;
  }
  const filter = mode as FilterMode;
  const axis = axisWeights(inputSize, output, filter, kernel === 'float32');
  const fixed = kernel === 'uint8' ? int16Weights(axis) : null;
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
        } else if (kernel === 'float32') {
          // PyTorch's aarch64 build runs ``output += t * w`` as a 4x unrolled
          // multiply-then-add loop with a fused multiply-add remainder.
          const main = Math.floor((size - 1) / 4) * 4;
          const first = horizontal ? read(plane, row, start) : read(plane, start, column);
          value = f32(first * axis.weights[offset]!);
          for (let j = 1; j < size; j += 1) {
            const sample = horizontal ? read(plane, row, start + j) : read(plane, start + j, column);
            const weight = axis.weights[offset + j]!;
            value = j <= main ? f32(value + f32(sample * weight)) : fmaf(sample, weight, value);
          }
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

/** ``F.interpolate`` rejects BOX and HAMMING with this ``NotImplementedError``. */
function unsupportedMode(mode: 'box' | 'hamming'): NotImplementedError {
  return new NotImplementedError(
    'Input Error: Only 3D, 4D and 5D input Tensors supported (got 4D) for the modes: nearest | linear | bilinear | bicubic | '
    + `trilinear | lanczos | area | nearest-exact (got ${mode})`,
  );
}

// ---------------------------------------------------------------------------
// Dtype-tagged image arrays (tensors, including torch.uint16).
// ---------------------------------------------------------------------------

/** Tensor dtypes (16-bit PNGs decode to ``uint16``, as ``torch.uint16``). */
type ArrayDType = DType;

interface ImageArray {
  dtype: ArrayDType;
  shape: number[];
  data: Float64Array;
}

const INTEGER_DTYPES = new Set<ArrayDType>(['uint8', 'int8', 'int16', 'int32', 'int64', 'uint16']);

function isFloating(dtype: ArrayDType): boolean {
  return dtype === 'float32' || dtype === 'float64' || dtype === 'float16' || dtype === 'bfloat16';
}

/** ``torch.round`` (half to even). */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Cast a computed float to ``dtype`` the way ``Tensor.to`` does. */
function castTo(dtype: ArrayDType, value: number): number {
  switch (dtype) {
    case 'uint8': return (Math.trunc(value) & 0xff) >>> 0;
    case 'int8': return (Math.trunc(value) << 24) >> 24;
    case 'int16': return (Math.trunc(value) << 16) >> 16;
    case 'int32': return Math.trunc(value) | 0;
    case 'int64': return Math.trunc(value);
    case 'uint16': return Math.min(Math.max(Math.trunc(value), 0), 65535);
    case 'bool': return value !== 0 && !Number.isNaN(value) ? 1 : 0;
    case 'float32': return f32(value);
    case 'float16': case 'bfloat16': return roundToDType(dtype, value);
    default: return value;
  }
}

/**
 * torchvision ``resize_image`` for a ``[C, H, W]`` array: ``uint8`` uses the
 * native fixed-point kernel for bilinear/bicubic/lanczos; other dtypes are
 * resized in float32 (float64 stays float64), rounded for signed integers and
 * ``uint8`` (not ``uint16``/``bool``) and cast back.
 */
function resizeArray(image: ImageArray, size: readonly [number, number], mode: InterpolationMode): ImageArray {
  const [outHeight, outWidth] = size;
  const height = image.shape[image.shape.length - 2]!;
  const width = image.shape[image.shape.length - 1]!;
  if (mode === 'box' || mode === 'hamming') throw unsupportedMode(mode);
  if (height === outHeight && width === outWidth) return image;
  const planes = image.data.length / (height * width);
  const dtype = image.dtype;
  const native = dtype === 'uint8' || isFloating(dtype) && dtype !== 'float16' && dtype !== 'bfloat16';
  let kernel: Kernel = dtype === 'uint8' ? 'uint8' : dtype === 'float64' ? 'float64' : 'float32';
  if (mode === 'nearest-exact' && dtype === 'uint8') kernel = 'uint8';
  let values: Float64Array = kernel === 'float32' ? Float64Array.from(image.data, (v) => f32(v)) : image.data;
  let currentWidth = width;
  if (outWidth !== width) {
    values = resizeAxis(values, planes, height, width, outWidth, true, mode, kernel);
    currentWidth = outWidth;
  }
  if (outHeight !== height) values = resizeAxis(values, planes, height, currentWidth, outHeight, false, mode, kernel);
  if (!native) {
    const round = INTEGER_DTYPES.has(dtype) && dtype !== 'uint16';
    for (let i = 0; i < values.length; i += 1) values[i] = castTo(dtype, round ? roundHalfEven(values[i]!) : values[i]!);
  } else if (dtype === 'float32') {
    for (let i = 0; i < values.length; i += 1) values[i] = f32(values[i]!);
  }
  return { dtype, shape: [...image.shape.slice(0, -2), outHeight, outWidth], data: values };
}

function tensorDType(dtype: ArrayDType): DType {
  return dtype;
}

function arrayToTensor(array: ImageArray): Tensor {
  const dtype = tensorDType(array.dtype);
  if (dtype === 'float32' || dtype === 'float16' || dtype === 'bfloat16') return tensor(Float32Array.from(array.data), { shape: array.shape, dtype });
  return new Tensor(Float64Array.from(array.data), array.shape, dtype);
}

function tensorToArray(value: Tensor): ImageArray {
  return { dtype: value.dtype, shape: [...value.shape], data: Float64Array.from(value.data) };
}

function decodedToArray(decoded: DecodedArray): ImageArray {
  return { dtype: decoded.dtype, shape: [...decoded.shape], data: Float64Array.from(decoded.data) };
}

/**
 * Antialiased resize of a ``CHW``/``BCHW`` tensor to ``[height, width]``
 * (``torchvision.transforms.v2.functional.resize(..., antialias=True)``):
 * ``uint8`` inputs use PyTorch's fixed-point path, floating inputs are
 * computed in their precision (float32 arithmetic for float32), other integer
 * dtypes are computed in float32 and rounded back. ``box``/``hamming`` raise
 * PyTorch's ``NotImplementedError``. Not differentiable.
 */
export function resizeImage(image: Tensor, size: readonly [number, number], mode: InterpolationMode = 'bilinear'): Tensor {
  if (!(image instanceof Tensor) || (image.ndim !== 3 && image.ndim !== 4)) throw new ValueError('resize expects a CHW or BCHW tensor');
  const [outHeight, outWidth] = size;
  if (!Number.isInteger(outHeight) || !Number.isInteger(outWidth) || outHeight < 1 || outWidth < 1) {
    throw new ValueError('resize size must contain positive integers');
  }
  if (mode === 'box' || mode === 'hamming') throw unsupportedMode(mode);
  const height = image.shape[image.ndim - 2]!;
  const width = image.shape[image.ndim - 1]!;
  if (height === outHeight && width === outWidth) return image.detach();
  return arrayToTensor(resizeArray(tensorToArray(image.detach()), [outHeight, outWidth], mode));
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
  for (const key of ['size', 'crop_size', 'pad_size']) {
    if (key in result) result[key] = sizeDict(result[key]) as JsonValue;
  }
  return result;
}

function numberList(value: JsonValue | undefined, name: string): number[] {
  if (typeof value === 'number') return [value];
  if (!Array.isArray(value) || !value.every((item) => typeof item === 'number')) throw new ValueError(`${name} must be a number or list of numbers`);
  return value as number[];
}

function positive(value: JsonValue | undefined): number | null {
  return typeof value === 'number' && value ? value : null;
}

/** Python ``round`` (half to even) for sizes. */
function pythonRound(value: number): number {
  return roundHalfEven(value);
}

/** Decoded processor input before channel handling. */
type Prepared = ImageArray;

/**
 * ``ViTImageProcessor`` (torchvision backend): resize, optional center crop
 * (with zero padding), rescale, normalize and optional padding. Settings are
 * live (mutable) and serialize with {@link toJson}.
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

  /** ``_validate_preprocess_kwargs`` (``do_pad`` is not validated: padding falls back to the batch maximum). */
  private validate(): void {
    const v = this.values;
    if (v.do_rescale && (v.rescale_factor === undefined || v.rescale_factor === null)) {
      throw new ValueError('`rescale_factor` must be specified if `do_rescale` is `True`.');
    }
    if (v.do_normalize && (v.image_mean === undefined || v.image_mean === null || v.image_std === undefined || v.image_std === null)) {
      throw new ValueError('`image_mean` and `image_std` must both be specified if `do_normalize` is `True`.');
    }
    if (v.do_center_crop && (v.crop_size === undefined || v.crop_size === null)) {
      throw new ValueError('`crop_size` must be specified if `do_center_crop` is `True`.');
    }
    if (v.do_resize && (v.size === undefined || v.size === null || v.resample === undefined || v.resample === null)) {
      throw new ValueError('`size` and `resample` must be specified if `do_resize` is `True`.');
    }
  }

  private interpolation(): InterpolationMode {
    const code = this.values.resample ?? 2;
    const mode = typeof code === 'number' ? PIL_RESAMPLING[code] : undefined;
    if (!mode) throw new ValueError(`unsupported resample ${JSON.stringify(code)}`);
    return mode;
  }

  /** ``TorchvisionBackend.resize`` output size for an image of ``height`` x ``width``. */
  private targetSize(height: number, width: number): [number, number] {
    const size = this.values.size;
    if (!isPlainObject(size)) throw new ValueError('processor size must be a mapping');
    const shortest = positive(size.shortest_edge);
    const longest = positive(size.longest_edge);
    if (shortest && longest) {
      // get_size_with_aspect_ratio((height, width), shortest_edge, longest_edge).
      let target = shortest;
      let raw: number | null = null;
      const small = Math.min(height, width);
      const large = Math.max(height, width);
      if ((large / small) * target > longest) {
        raw = (longest * small) / large;
        target = pythonRound(raw);
      }
      if ((height <= width && height === target) || (width <= height && width === target)) return [height, width];
      if (width < height) return [Math.trunc(raw !== null ? (raw * height) / width : (target * height) / width), target];
      return [target, Math.trunc(raw !== null ? (raw * width) / height : (target * width) / height)];
    }
    if (shortest) {
      // get_resize_output_image_size(default_to_square=False).
      const [short, long] = width <= height ? [width, height] : [height, width];
      const newShort = shortest;
      const newLong = Math.trunc((newShort * long) / short);
      return width <= height ? [newLong, newShort] : [newShort, newLong];
    }
    const maxHeight = positive(size.max_height);
    const maxWidth = positive(size.max_width);
    if (maxHeight && maxWidth) {
      const scale = Math.min(maxHeight / height, maxWidth / width);
      return [Math.trunc(height * scale), Math.trunc(width * scale)];
    }
    const h = positive(size.height);
    const w = positive(size.width);
    if (h && w) return [h, w];
    throw new ValueError(`Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got ${JSON.stringify(size)}.`);
  }

  /** ``TorchvisionBackend.center_crop`` (pads with zeros when the crop exceeds the image). */
  private centerCrop(image: ImageArray): ImageArray {
    const crop = this.values.crop_size;
    const cropHeight = isPlainObject(crop) ? positive(crop.height) : null;
    const cropWidth = isPlainObject(crop) ? positive(crop.width) : null;
    if (!cropHeight || !cropWidth) {
      throw new ValueError(`The size dictionary must have keys 'height' and 'width'. Got ${JSON.stringify(isPlainObject(crop) ? Object.keys(crop) : crop)}`);
    }
    let current = image;
    let height = current.shape[current.shape.length - 2]!;
    let width = current.shape[current.shape.length - 1]!;
    if (cropWidth > width || cropHeight > height) {
      const left = cropWidth > width ? Math.floor((cropWidth - width) / 2) : 0;
      const top = cropHeight > height ? Math.floor((cropHeight - height) / 2) : 0;
      const right = cropWidth > width ? Math.floor((cropWidth - width + 1) / 2) : 0;
      const bottom = cropHeight > height ? Math.floor((cropHeight - height + 1) / 2) : 0;
      current = padArray(current, top, left, height + top + bottom, width + left + right);
      height = current.shape[current.shape.length - 2]!;
      width = current.shape[current.shape.length - 1]!;
      if (cropWidth === width && cropHeight === height) return current;
    }
    const top = Math.trunc((height - cropHeight) / 2);
    const left = Math.trunc((width - cropWidth) / 2);
    return cropArray(current, top, left, cropHeight, cropWidth);
  }

  /**
   * Process images (see the module description for accepted inputs). Returns
   * ``{pixel_values}`` as a ``BCHW`` tensor in processed-image coordinates
   * (float32 after normalization or rescaling of non-float64 inputs).
   * ``http(s)://`` URLs need {@link apreprocess}.
   */
  preprocess(images: ImageInputs): { pixel_values: Tensor } {
    return this.run(this.fetch(images, (source) => torchvisionDecode(sourceBytes(source), 'RGB')));
  }

  /** {@link preprocess} that also fetches ``http(s)://`` URLs (like transformers' ``httpx.get``). */
  async apreprocess(images: ImageInputs): Promise<{ pixel_values: Tensor }> {
    const sources: string[] = [];
    const collect = (value: unknown): void => {
      if (typeof value === 'string') sources.push(value);
      else if (Array.isArray(value)) value.forEach(collect);
    };
    collect(images);
    const decoded = new Map<string, DecodedArray>();
    for (const source of sources) {
      if (!decoded.has(source)) decoded.set(source, torchvisionDecode(await fetchSourceBytes(source), 'RGB'));
    }
    return this.run(this.fetch(images, (source) => decoded.get(source)!));
  }

  /** ``fetch_images``: decode string sources, keep valid images, reject anything else. */
  private fetch(images: unknown, decode: (source: string) => DecodedArray): Fetched {
    if (Array.isArray(images)) return images.map((item) => this.fetch(item, decode));
    if (typeof images === 'string') return decodedToArray(decode(images));
    if (images instanceof Tensor) return tensorToArray(images.detach());
    if (images instanceof RasterImage) return images;
    throw new TypeError(`only a single or a list of entries is supported but got type=<class '${pythonTypeName(images)}'>`);
  }

  private run(fetched: Fetched): { pixel_values: Tensor } {
    this.validate();
    // make_flat_list_of_images / _preprocess index ``images[0]`` of an empty batch.
    if (Array.isArray(fetched) && fetched.every((item) => Array.isArray(item) && !item.length)) throw new IndexError('list index out of range');
    const list = makeFlatList(fetched).map((image) => this.prepareImage(image));
    let processed = list.map((image) => this.processOne(image));
    if (this.values.do_pad === true) processed = this.pad(processed);
    const first = processed[0]!;
    if (processed.some((item) => item.dtype !== first.dtype || item.shape.length !== first.shape.length || item.shape.some((size, index) => size !== first.shape[index]))) {
      throw new ValueError('Unable to create tensor, you should probably activate padding with \'padding=True\' to have batched tensors with the same length.');
    }
    const plane = first.data.length;
    const data = new Float64Array(processed.length * plane);
    processed.forEach((item, index) => data.set(item.data, index * plane));
    return { pixel_values: arrayToTensor({ dtype: first.dtype, shape: [processed.length, ...first.shape], data }) };
  }

  /** ``TorchvisionBackend.process_image``: RGB conversion (PIL only), pil_to_tensor, channels first. */
  private prepareImage(image: ImageArray | RasterImage): Prepared {
    let array: ImageArray;
    if (image instanceof RasterImage) {
      const converted = this.values.do_convert_rgb === true && image.mode !== 'RGB' ? image.convert('RGB') : image;
      const bands = converted.bands;
      const plane = converted.width * converted.height;
      const data = new Float64Array(converted.data.length);
      for (let i = 0; i < plane; i += 1) {
        for (let c = 0; c < bands; c += 1) {
          const v = converted.data[i * bands + c]!;
          data[c * plane + i] = converted.mode === '1' ? (v ? 1 : 0) : v;
        }
      }
      const dtype: ArrayDType = converted.mode === '1' ? 'bool' : converted.mode === 'I;16' ? 'uint16'
        : converted.mode === 'I' ? 'int32' : converted.mode === 'F' ? 'float32' : 'uint8';
      array = { dtype, shape: [bands, converted.height, converted.width], data };
    } else {
      array = image;
    }
    if (array.shape.length === 2) array = { ...array, shape: [1, ...array.shape] };
    const channelsLast = inferChannelsLast(array.shape);
    if (channelsLast) array = moveChannelsFirst(array);
    return array;
  }

  private processOne(image: Prepared): ImageArray {
    let current = image;
    if (this.doResize) {
      const height = current.shape[current.shape.length - 2]!;
      const width = current.shape[current.shape.length - 1]!;
      current = resizeArray(current, this.targetSize(height, width), this.interpolation());
    }
    if (this.values.do_center_crop === true) current = this.centerCrop(current);
    return this.rescaleAndNormalize(current);
  }

  /** ``rescale_and_normalize`` (fused into float32 ``normalize`` when both are enabled). */
  private rescaleAndNormalize(image: ImageArray): ImageArray {
    const doRescale = this.doRescale;
    const doNormalize = this.doNormalize;
    if (!doRescale && !doNormalize) return image;
    const factor = this.rescaleFactor;
    if (!doNormalize) {
      // images * scale: result_type(dtype, float) keeps float64/float16/bfloat16, otherwise float32.
      const dtype: ArrayDType = image.dtype === 'float64' || image.dtype === 'float16' || image.dtype === 'bfloat16' ? image.dtype : 'float32';
      const data = Float64Array.from(image.data, (v) => castTo(dtype, dtype === 'float64' ? v * factor : f32(f32(v) * f32(factor))));
      return { dtype, shape: image.shape, data };
    }
    let mean = this.imageMean.map((value) => f32(value));
    let std = this.imageStd.map((value) => f32(value));
    if (doRescale) {
      const inverse = f32(1 / factor);
      mean = mean.map((value) => f32(value * inverse));
      std = std.map((value) => f32(value * inverse));
    }
    const channels = image.shape[image.shape.length - 3]!;
    const height = image.shape[image.shape.length - 2]!;
    const width = image.shape[image.shape.length - 1]!;
    const plane = height * width;
    const outChannels = broadcast(channels, mean.length, std.length);
    const data = new Float64Array(outChannels * plane);
    for (let c = 0; c < outChannels; c += 1) {
      const source = channels === 1 ? 0 : c;
      const m = mean.length === 1 ? mean[0]! : mean[c]!;
      const s = std.length === 1 ? std[0]! : std[c]!;
      for (let i = 0; i < plane; i += 1) data[c * plane + i] = f32(f32(f32(image.data[source * plane + i]!) - m) / s);
    }
    return { dtype: 'float32', shape: [outChannels, height, width], data };
  }

  /** ``TorchvisionBackend.pad``: zero-pad bottom/right to ``pad_size`` (or the batch maximum). */
  private pad(images: ImageArray[]): ImageArray[] {
    const padSize = this.values.pad_size;
    let target: [number, number];
    if (isPlainObject(padSize)) {
      const height = positive(padSize.height);
      const width = positive(padSize.width);
      if (!height || !width) throw new ValueError(`Pad size must contain 'height' and 'width' keys only. Got pad_size=${JSON.stringify(padSize)}.`);
      target = [height, width];
    } else {
      target = [Math.max(...images.map((image) => image.shape[1]!)), Math.max(...images.map((image) => image.shape[2]!))];
    }
    return images.map((image) => {
      const height = image.shape[1]!;
      const width = image.shape[2]!;
      if (target[0] < height || target[1] < width) {
        throw new ValueError(`Padding dimensions are negative. Please make sure that the \`pad_size\` is larger than the image size. Got pad_size=(${target[0]}, ${target[1]}), image_size=(${height}, ${width}).`);
      }
      return height === target[0] && width === target[1] ? image : padArray(image, 0, 0, target[0], target[1]);
    });
  }
}

type Fetched = ImageArray | RasterImage | Fetched[];

/** ``type(value).__name__`` for the JavaScript values a caller might pass. */
function pythonTypeName(value: unknown): string {
  if (value === null || value === undefined) return 'NoneType';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'number') return Number.isInteger(value) ? 'int' : 'float';
  if (typeof value === 'bigint') return 'int';
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return 'bytes';
  if (typeof value === 'function') return 'function';
  if (value instanceof Map || typeof value === 'object') return 'dict';
  return typeof value;
}

/** ``make_flat_list_of_images(images, expected_ndims=3)``. */
function makeFlatList(images: Fetched): (ImageArray | RasterImage)[] {
  const ndim = (item: ImageArray | RasterImage): number => (item instanceof RasterImage ? 3 : item.shape.length);
  const isImage = (item: Fetched): item is ImageArray | RasterImage => !Array.isArray(item);
  const frames = (item: ImageArray): ImageArray[] => {
    const [count, ...rest] = item.shape;
    const size = rest.reduce((a, b) => a * b, 1);
    return Array.from({ length: count! }, (_, index) => ({ dtype: item.dtype, shape: rest, data: item.data.slice(index * size, (index + 1) * size) }));
  };
  if (Array.isArray(images) && images.length && images.every((item) => Array.isArray(item))
    && images.every((item) => (item as Fetched[]).every(isImage))) {
    return (images as Fetched[][]).flat() as (ImageArray | RasterImage)[];
  }
  if (Array.isArray(images) && images.length && images.every(isImage)) {
    const first = images[0] as ImageArray | RasterImage;
    if (first instanceof RasterImage || ndim(first) === 3) return images as (ImageArray | RasterImage)[];
    if (ndim(first) === 4) return (images as ImageArray[]).flatMap(frames);
  }
  if (!Array.isArray(images)) {
    if (images instanceof RasterImage || ndim(images) === 3) return [images];
    if (ndim(images) === 4) return frames(images);
  }
  throw new ValueError('Could not make a flat list of images from the input');
}

/** ``infer_channel_dimension_format`` for ``(1, 3)`` channels: ``true`` for channels last. */
function inferChannelsLast(shape: readonly number[]): boolean {
  const [first, last] = shape.length === 3 ? [0, 2] : shape.length === 4 ? [1, 3] : shape.length === 5 ? [2, 4] : [-1, -1];
  if (first < 0) throw new ValueError(`Unsupported number of image dimensions: ${shape.length}`);
  const channels = (size: number): boolean => size === 1 || size === 3;
  if (channels(shape[first]!)) return false;
  if (channels(shape[last]!)) return true;
  throw new ValueError('Unable to infer channel dimension format');
}

function moveChannelsFirst(array: ImageArray): ImageArray {
  const [height, width, channels] = array.shape.slice(-3) as [number, number, number];
  const data = new Float64Array(array.data.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      for (let c = 0; c < channels; c += 1) data[(c * height + y) * width + x] = array.data[(y * width + x) * channels + c]!;
    }
  }
  return { dtype: array.dtype, shape: [channels, height, width], data };
}

function padArray(array: ImageArray, top: number, left: number, height: number, width: number): ImageArray {
  const [channels, inHeight, inWidth] = array.shape.slice(-3) as [number, number, number];
  const data = new Float64Array(channels * height * width);
  for (let c = 0; c < channels; c += 1) {
    for (let y = 0; y < inHeight; y += 1) {
      for (let x = 0; x < inWidth; x += 1) data[(c * height + y + top) * width + x + left] = array.data[(c * inHeight + y) * inWidth + x]!;
    }
  }
  return { dtype: array.dtype, shape: [channels, height, width], data };
}

function cropArray(array: ImageArray, top: number, left: number, height: number, width: number): ImageArray {
  const [channels, inHeight, inWidth] = array.shape.slice(-3) as [number, number, number];
  const data = new Float64Array(channels * height * width);
  for (let c = 0; c < channels; c += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const sy = y + top;
        const sx = x + left;
        data[(c * height + y) * width + x] = sy >= 0 && sy < inHeight && sx >= 0 && sx < inWidth ? array.data[(c * inHeight + sy) * inWidth + sx]! : 0;
      }
    }
  }
  return { dtype: array.dtype, shape: [channels, height, width], data };
}

/** Broadcast channel count of ``image - mean`` then ``/ std`` (``[C,H,W]`` against ``[N,1,1]``). */
function broadcast(channels: number, meanLength: number, stdLength: number): number {
  let out = channels;
  for (const length of [meanLength, stdLength]) {
    if (length === out || length === 1) continue;
    if (out === 1) {
      out = length;
      continue;
    }
    throw new ValueError(`The size of tensor a (${out}) must match the size of tensor b (${length}) at non-singleton dimension 0`);
  }
  return out;
}
