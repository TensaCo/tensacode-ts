/**
 * Public image decoding entry points: the TypeScript counterparts of the
 * decoders Python TensorCode reaches through its dependencies.
 *
 * - {@link decodeImage}: ``torchvision.io.decode_image`` (what transformers'
 *   torchvision image processors use for path, base64 and URL inputs).
 * - {@link loadImage} / {@link loadImageAsync}: transformers'
 *   ``load_image_as_tensor`` (URL, file path or base64 text, ``RGB``).
 * - {@link openImage}: ``PIL.Image.open(fp)`` followed by ``load()``,
 *   returning a {@link RasterImage} with Pillow's mode, ``convert``,
 *   ``resize`` and ``exif_transpose``.
 *
 * Everything is pure TypeScript: PNG (every colour type, bit depth and
 * interlacing), JPEG (baseline, extended and progressive Huffman, any
 * sampling factors, restart markers, CMYK/YCCK), GIF (all frames), WebP
 * (lossy, lossless, alpha, animation) and BMP. Pixel values match
 * libjpeg-turbo, libpng, libwebp, giflib and Pillow exactly.
 */
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { torchvisionDecode, type DecodedArray, type ImageReadMode } from './torchvision.js';
import { openImageBytes, RasterImage } from './raster.js';
import { fetchSourceBytes, readFileBytes, sourceBytes } from './load.js';

export { RasterImage, type RasterMode, type RasterInfo } from './raster.js';
export { Resampling } from './resample.js';
export type { ImageReadMode } from './torchvision.js';

/** Encoded image input: bytes (``Uint8Array``/``Buffer``/``ArrayBuffer``) or a file path. */
export type EncodedImage = Uint8Array | ArrayBuffer | string;

function toBytes(input: EncodedImage): Uint8Array {
  if (typeof input === 'string') return readFileBytes(input);
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  throw new TypeError('expected encoded image bytes (Uint8Array, Buffer or ArrayBuffer) or a file path');
}

function toTensor(decoded: DecodedArray): Tensor {
  return new Tensor(Float64Array.from(decoded.data), decoded.shape, decoded.dtype === 'uint16' ? 'int32' : 'uint8');
}

/**
 * ``torchvision.io.decode_image(input, mode, apply_exif_orientation)``:
 * a ``uint8`` ``[C, H, W]`` tensor (``[N, 3, H, W]`` for animated GIFs).
 * 16-bit PNGs decode to exact 0-65535 values in an ``int32`` tensor
 * (torchvision returns ``uint16``). ``mode`` defaults to ``'UNCHANGED'``.
 */
export function decodeImage(input: EncodedImage, options: { mode?: ImageReadMode; applyExifOrientation?: boolean } = {}): Tensor {
  const bytes = toBytes(input);
  return toTensor(torchvisionDecode(bytes, options.mode ?? 'UNCHANGED', options.applyExifOrientation ?? false));
}

/**
 * transformers ``load_image_as_tensor(source)`` for a file path, base64 text
 * or ``data:image/...`` URI: an ``RGB`` ``[3, H, W]`` tensor. ``http(s)://``
 * URLs need {@link loadImageAsync}.
 */
export function loadImage(source: string): Tensor {
  if (typeof source !== 'string') {
    throw new TypeError('Incorrect format used for image. Should be a URL, a local path, a base64 string, or a PIL image.');
  }
  return toTensor(torchvisionDecode(sourceBytes(source), 'RGB'));
}

/** {@link loadImage} that also fetches ``http(s)://`` URLs (redirects followed). */
export async function loadImageAsync(source: string, options: { timeout?: number | null } = {}): Promise<Tensor> {
  if (typeof source !== 'string') {
    throw new TypeError('Incorrect format used for image. Should be a URL, a local path, a base64 string, or a PIL image.');
  }
  return toTensor(torchvisionDecode(await fetchSourceBytes(source, options), 'RGB'));
}

/**
 * ``PIL.Image.open(fp)`` + ``load()`` for JPEG, PNG, GIF, WebP and BMP data
 * (the first frame of animations). Pillow's lazy formats beyond these raise
 * ``ValueError('cannot identify image file')``.
 */
export function openImage(input: EncodedImage): RasterImage {
  const bytes = toBytes(input);
  if (!bytes.length) throw new ValueError('cannot identify image file');
  return openImageBytes(bytes);
}
