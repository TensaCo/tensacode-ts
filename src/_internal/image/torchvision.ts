/**
 * ``torchvision.io.decode_image`` semantics (torchvision 0.29): format
 * sniffing, libpng/libjpeg-turbo/libwebp/giflib colour conversions for each
 * ``ImageReadMode`` and ``apply_exif_orientation``. transformers' torchvision
 * image processors decode path, base64 and URL inputs with
 * ``decode_image(..., mode=RGB)``.
 */
import { ValueError } from '../../errors.js';
import { decodePng, isPng, type PngData } from './png.js';
import { decodeJpeg, isJpeg } from './jpeg.js';
import { decodeGif, isGif } from './gif.js';
import { decodeWebp, isWebp } from './webp.js';
import { orientPixels, torchvisionOrientation } from './exif.js';

export type ImageReadMode = 'UNCHANGED' | 'GRAY' | 'GRAY_ALPHA' | 'RGB' | 'RGB_ALPHA';

/** A decoded ``uint8``/``uint16`` array in ``CHW`` (or ``NCHW`` for animated GIFs) layout. */
export interface DecodedArray {
  shape: number[];
  dtype: 'uint8' | 'uint16';
  data: Uint8Array | Uint16Array;
}

/** torchvision raises ``RuntimeError``; JavaScript reports a plain ``Error``. */
function runtimeError(message: string): Error {
  return new Error(message);
}

function toChw(pixels: Uint8Array | Uint16Array, width: number, height: number, channels: number): DecodedArray {
  const out = pixels instanceof Uint16Array ? new Uint16Array(pixels.length) : new Uint8Array(pixels.length);
  const plane = width * height;
  for (let i = 0; i < plane; i += 1) for (let c = 0; c < channels; c += 1) out[c * plane + i] = pixels[i * channels + c]!;
  return { shape: [channels, height, width], dtype: pixels instanceof Uint16Array ? 'uint16' : 'uint8', data: out };
}

const UNSUPPORTED = 'Unsupported image file. Only jpeg, png, webp and gif are currently supported. For avif and heic format, please rely on `decode_avif` and `decode_heic` directly.';

function startsWith(bytes: Uint8Array, signature: readonly number[], offset = 0): boolean {
  return signature.every((value, index) => bytes[offset + index] === value);
}

/** Decode encoded image bytes like ``torchvision.io.decode_image`` (signature sniffing included). */
export function torchvisionDecode(bytes: Uint8Array, mode: ImageReadMode = 'UNCHANGED', applyExifOrientation = false): DecodedArray {
  if (!bytes.length) throw runtimeError('Expected a non empty 1-dimensional tensor');
  if (bytes.length < 3) throw runtimeError(UNSUPPORTED);
  if (startsWith(bytes, [0xff, 0xd8, 0xff])) return decodeJpegTv(bytes, mode, applyExifOrientation);
  if (bytes.length < 4) throw runtimeError(UNSUPPORTED);
  if (startsWith(bytes, [0x89, 0x50, 0x4e, 0x47])) return decodePngTv(bytes, mode, applyExifOrientation);
  if (bytes.length < 6) throw runtimeError(UNSUPPORTED);
  if (isGif(bytes)) return decodeGifTv(bytes);
  if (bytes.length < 15) throw runtimeError(UNSUPPORTED);
  if (startsWith(bytes, [0x52, 0x49, 0x46, 0x46]) && startsWith(bytes, [0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38], 8)) {
    return decodeWebpTv(bytes, mode);
  }
  throw runtimeError(UNSUPPORTED);
}

function decodeJpegTv(bytes: Uint8Array, mode: ImageReadMode, exif: boolean): DecodedArray {
  let jpeg;
  try {
    jpeg = decodeJpeg(bytes);
  } catch (error) {
    throw runtimeError(`Error while decoding JPEG image: ${(error as Error).message}`);
  }
  const cmyk = jpeg.colorSpace === 'CMYK' || jpeg.colorSpace === 'YCCK';
  let pixels: Uint8Array;
  let channels: number;
  if (mode === 'UNCHANGED') {
    channels = jpeg.components;
    pixels = channels === 1 ? jpeg.output('GRAYSCALE') : channels === 4 ? jpeg.output('CMYK') : jpeg.output('RGB');
  } else if (mode === 'GRAY' || mode === 'RGB') {
    const pixelsCount = jpeg.width * jpeg.height;
    if (cmyk) {
      const source = jpeg.output('CMYK');
      const rgb = new Uint8Array(pixelsCount * 3);
      for (let i = 0; i < pixelsCount; i += 1) {
        const k = source[i * 4 + 3]!;
        for (let c = 0; c < 3; c += 1) {
          // clamped_cmyk_rgb_convert(k, 255 - c): k - MULDIV255(k, 255 - c).
          const v = k * (255 - source[i * 4 + c]!) + 128;
          rgb[i * 3 + c] = Math.min(Math.max(k - (((v >> 8) + v) >> 8), 0), 255);
        }
      }
      if (mode === 'RGB') {
        pixels = rgb;
      } else {
        pixels = new Uint8Array(pixelsCount);
        for (let i = 0; i < pixelsCount; i += 1) {
          pixels[i] = (19595 * rgb[i * 3]! + 38470 * rgb[i * 3 + 1]! + 7471 * rgb[i * 3 + 2]! + 32768) >> 16;
        }
      }
    } else {
      pixels = jpeg.output(mode === 'RGB' ? 'RGB' : 'GRAYSCALE');
    }
    channels = mode === 'RGB' ? 3 : 1;
  } else {
    throw runtimeError('The provided mode is not supported for JPEG files');
  }
  let width = jpeg.width;
  let height = jpeg.height;
  if (exif) {
    const orientation = jpeg.firstApp1 && jpeg.firstApp1.length > 6 ? torchvisionOrientation(jpeg.firstApp1.subarray(6)) : -1;
    ({ pixels, width, height } = orientPixels(pixels, width, height, channels, orientation));
  }
  return toChw(pixels, width, height, channels);
}

/**
 * libpng's transformations for one ``ImageReadMode`` (torchvision
 * ``decode_png``): expansion, alpha stripping, RGB to gray, gray to RGB and
 * the alpha filler, written into rows of the channel count torchvision
 * declares. Where libpng produces fewer bytes per row than declared
 * (sub-byte palettes in ``UNCHANGED`` mode, palettes without ``tRNS`` in the
 * alpha modes) the rows keep libpng's layout and the rest of the buffer is
 * zero.
 */
function decodePngTv(bytes: Uint8Array, mode: ImageReadMode, exif: boolean): DecodedArray {
  let png: PngData;
  try {
    png = decodePng(bytes);
  } catch (error) {
    throw runtimeError(`Internal error decoding PNG: ${(error as Error).message}`);
  }
  const { width, height, bitDepth, colorType } = png;
  const count = width * height;
  const sixteen = bitDepth === 16;
  const samples = png.samples;
  const inChannels = png.channels;
  const palette = png.palette ?? new Uint8Array(0);
  const trns = png.transparency;
  const isPalette = colorType === 3;
  const hasColor = (colorType & 2) !== 0;
  let hasAlpha = (colorType & 4) !== 0;
  // Gray samples below 8 bits are scaled to 8 bits (png_set_expand_gray_1_2_4_to_8).
  const scale = colorType === 0 && bitDepth < 8 ? 255 / ((1 << bitDepth) - 1) : 1;
  const filler = (1 << bitDepth) - 1;
  let declared: number;
  // Per-pixel pipeline state: channel values after each libpng step.
  let expandPalette = false;
  let stripAlpha = false;
  let toGray = false;
  let grayToRgb = false;
  let addAlpha = false;
  switch (mode) {
    case 'UNCHANGED':
      declared = inChannels;
      break;
    case 'GRAY':
      declared = 1;
      if (colorType !== 0) {
        if (isPalette) { expandPalette = true; hasAlpha = true; }
        if (hasAlpha) stripAlpha = true;
        if (hasColor) toGray = true;
      }
      break;
    case 'GRAY_ALPHA':
      declared = 2;
      if (colorType !== 4) {
        if (isPalette) { expandPalette = true; hasAlpha = true; }
        if (!hasAlpha) addAlpha = true;
        if (hasColor) toGray = true;
      }
      break;
    case 'RGB':
      declared = 3;
      if (colorType !== 2) {
        if (isPalette) { expandPalette = true; hasAlpha = true; } else if (!hasColor) grayToRgb = true;
        if (hasAlpha) stripAlpha = true;
      }
      break;
    default:
      declared = 4;
      if (colorType !== 6) {
        if (isPalette) { expandPalette = true; hasAlpha = true; } else if (!hasColor) grayToRgb = true;
        if (!hasAlpha) addAlpha = true;
      }
      break;
  }
  const grayGamma = toGray ? rgbToGrayGamma(png) : null;
  const out = sixteen ? new Uint16Array(count * declared) : new Uint8Array(count * declared);
  if (mode === 'UNCHANGED' && bitDepth < 8 && colorType === 3) {
    // No packing transform: libpng writes packed bytes at the start of each row.
    const rowBytes = Math.ceil((width * bitDepth) / 8);
    for (let y = 0; y < height; y += 1) {
      for (let b = 0; b < rowBytes; b += 1) {
        let value = 0;
        for (let k = 0; k < 8 / bitDepth; k += 1) {
          const x = b * (8 / bitDepth) + k;
          const sample = x < width ? samples[y * width + x]! : 0;
          value |= sample << (8 - bitDepth * (k + 1));
        }
        out[y * width + b] = value;
      }
    }
    return finishPng(out, width, height, declared, exif, png);
  }
  const row: number[] = [];
  for (let y = 0; y < height; y += 1) {
    let cursor = y * width * declared;
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      row.length = 0;
      if (mode === 'UNCHANGED') {
        for (let c = 0; c < inChannels; c += 1) row.push(colorType === 0 ? samples[i * inChannels + c]! * scale : samples[i * inChannels + c]!);
      } else {
        // Expansion.
        if (expandPalette) {
          const index = samples[i]!;
          for (let c = 0; c < 3; c += 1) row.push(index * 3 + c < palette.length ? palette[index * 3 + c]! : 0);
          if (trns) row.push(index < trns.length ? trns[index]! : 255);
        } else {
          for (let c = 0; c < inChannels; c += 1) row.push(colorType === 0 ? samples[i * inChannels + c]! * scale : samples[i * inChannels + c]!);
        }
        const alphaPresent = row.length === 2 || row.length === 4;
        if (stripAlpha && alphaPresent) row.pop();
        if (toGray && row.length >= 3) {
          const alpha = row.length === 4 ? row[3]! : undefined;
          const gray = grayGamma ? grayGamma(row[0]!, row[1]!, row[2]!) : rgbToGray(row[0]!, row[1]!, row[2]!, sixteen);
          row.length = 0;
          row.push(gray);
          if (alpha !== undefined) row.push(alpha);
        }
        if (grayToRgb && (row.length === 1 || row.length === 2)) {
          const alpha = row.length === 2 ? row[1] : undefined;
          const g = row[0]!;
          row.length = 0;
          row.push(g, g, g);
          if (alpha !== undefined) row.push(alpha);
        }
        if (addAlpha && (row.length === 1 || row.length === 3)) row.push(sixteen ? 65535 : filler & 0xff);
      }
      for (const value of row) out[cursor++] = value;
    }
  }
  return finishPng(out, width, height, declared, exif, png);
}

function finishPng(out: Uint8Array | Uint16Array, width: number, height: number, channels: number, exif: boolean, png: PngData): DecodedArray {
  let pixels = out;
  let outWidth = width;
  let outHeight = height;
  if (exif) {
    const orientation = png.exifBeforeData && png.exifBeforeData.length ? torchvisionOrientation(png.exifBeforeData) : -1;
    ({ pixels, width: outWidth, height: outHeight } = orientPixels(pixels, width, height, channels, orientation));
  }
  return toChw(pixels, outWidth, outHeight, channels);
}

/** libpng ``png_set_rgb_to_gray(png_ptr, 1, 0.2989, 0.587)`` without gamma tables. */
const GRAY_RED = Math.floor((29890 * 32768) / 100000);
const GRAY_GREEN = Math.floor((58700 * 32768) / 100000);
const GRAY_BLUE = 32768 - GRAY_RED - GRAY_GREEN;

/** ``png_reciprocal``. */
function reciprocal(a: number): number {
  const r = Math.floor(1e10 / a + 0.5);
  return r <= 2147483647 && r >= -2147483648 ? r : 0;
}

/** ``png_reciprocal2``. */
function reciprocal2(a: number, b: number): number {
  if (a === 0 || b === 0) return 0;
  const r = Math.floor(1e15 / a / b + 0.5);
  return r <= 2147483647 && r >= -2147483648 ? r : 0;
}

function gammaSignificant(gamma: number): boolean {
  return gamma < 100000 - 5000 || gamma > 100000 + 5000;
}

/**
 * libpng's gamma-corrected ``png_do_rgb_to_gray`` (tables built by
 * ``png_build_gamma_table`` when the file gamma is significant; the screen
 * gamma defaults to its reciprocal), or ``null`` for the plain path.
 */
function rgbToGrayGamma(png: PngData): ((r: number, g: number, b: number) => number) | null {
  const fileGamma = png.chunkGamma;
  if (fileGamma <= 0) return null;
  const screenGamma = reciprocal(fileGamma);
  if (!gammaSignificant(fileGamma) && !gammaSignificant(screenGamma)) return null;
  const correction = reciprocal2(screenGamma, fileGamma);
  const toLinear = reciprocal(fileGamma);
  const fromLinear = reciprocal(screenGamma);
  if (png.bitDepth !== 16) {
    const table = (gamma: number): Uint8Array => Uint8Array.from({ length: 256 }, (_, i) => (
      !gammaSignificant(gamma) || i === 0 || i === 255 ? i : Math.floor(255 * Math.pow(i / 255, gamma * 0.00001) + 0.5)
    ));
    const direct = table(correction);
    const to1 = table(toLinear);
    const from1 = table(fromLinear);
    return (r, g, b) => {
      if (r === g && r === b) return direct[r]!;
      return from1[(GRAY_RED * to1[r]! + GRAY_GREEN * to1[g]! + GRAY_BLUE * to1[b]! + 16384) >> 15]!;
    };
  }
  const bits = png.significantBits;
  const significant = bits ? ((png.colorType & 2) !== 0 ? Math.max(bits.red, bits.green, bits.blue) : bits.gray) : 0;
  const shift = Math.min(significant > 0 && significant < 16 ? 16 - significant : 0, 8);
  const table16 = (gamma: number): (value: number) => number => {
    const max = 2 ** (16 - shift) - 1;
    const maxBy2 = 2 ** (15 - shift);
    const fmax = 1 / max;
    return (value: number): number => {
      const i = (value & 0xff) >> shift;
      const j = value >> 8;
      let ig = (j << (8 - shift)) + i;
      if (gammaSignificant(gamma)) return Math.floor(65535 * Math.pow(ig * fmax, gamma * 0.00001) + 0.5);
      if (shift !== 0) ig = Math.floor((ig * 65535 + maxBy2) / max);
      return ig;
    };
  };
  const direct = table16(correction);
  const to1 = table16(toLinear);
  const from1 = table16(fromLinear);
  return (r, g, b) => {
    if (r === g && r === b) return direct(r);
    const gray = (GRAY_RED * to1(r) + GRAY_GREEN * to1(g) + GRAY_BLUE * to1(b) + 16384) >> 15;
    return from1(gray & 0xffff);
  };
}

function rgbToGray(r: number, g: number, b: number, sixteen: boolean): number {
  if (sixteen) return Math.floor((GRAY_RED * r + GRAY_GREEN * g + GRAY_BLUE * b + 16384) / 32768);
  if (r === g && g === b) return r;
  return Math.floor((GRAY_RED * r + GRAY_GREEN * g + GRAY_BLUE * b) / 32768);
}

function decodeGifTv(bytes: Uint8Array): DecodedArray {
  let gif;
  try {
    gif = decodeGif(bytes);
  } catch (error) {
    throw runtimeError(`DGifSlurp() failed: ${(error as Error).message}`);
  }
  // torchvision decode_gif: frames composited onto the background colour or the previous frame.
  const outWidth = Math.max(gif.width, gif.frames[0]!.width);
  const outHeight = Math.max(gif.height, gif.frames[0]!.height);
  const frames = gif.frames.length;
  const plane = outWidth * outHeight;
  const data = new Uint8Array(frames * 3 * plane);
  const background = gif.globalPalette && gif.background * 3 + 2 < gif.globalPalette.length
    ? [gif.globalPalette[gif.background * 3]!, gif.globalPalette[gif.background * 3 + 1]!, gif.globalPalette[gif.background * 3 + 2]!]
    : gif.globalPalette ? [0, 0, 0] : [0, 0, 0];
  gif.frames.forEach((frame, index) => {
    const base = index * 3 * plane;
    // Unspecified, "do not dispose" and "restore previous" keep the previous frame (torchvision 0.29).
    if (index > 0 && (frame.disposal === 0 || frame.disposal === 1 || frame.disposal === 3)) {
      data.copyWithin(base, base - 3 * plane, base);
    } else {
      for (let y = 0; y < gif.height; y += 1) {
        for (let x = 0; x < gif.width; x += 1) {
          for (let c = 0; c < 3; c += 1) data[base + c * plane + y * outWidth + x] = background[c]!;
        }
      }
    }
    const palette = frame.palette ?? gif.globalPalette;
    if (!palette) throw runtimeError('Global and local color maps are missing. This should never happen!');
    for (let y = 0; y < frame.height; y += 1) {
      const ty = y + frame.top;
      if (ty >= outHeight) continue;
      for (let x = 0; x < frame.width; x += 1) {
        const tx = x + frame.left;
        if (tx >= outWidth) continue;
        const value = frame.indices[y * frame.width + x]!;
        if (value === frame.transparentIndex) continue;
        for (let c = 0; c < 3; c += 1) data[base + c * plane + ty * outWidth + tx] = value * 3 + c < palette.length ? palette[value * 3 + c]! : 0;
      }
    }
  });
  return { shape: frames === 1 ? [3, outHeight, outWidth] : [frames, 3, outHeight, outWidth], dtype: 'uint8', data };
}

function decodeWebpTv(bytes: Uint8Array, mode: ImageReadMode): DecodedArray {
  let webp;
  try {
    webp = decodeWebp(bytes);
  } catch (error) {
    throw runtimeError(`WebPGetFeatures failed: ${(error as Error).message}`);
  }
  if (webp.animated) throw runtimeError('Animated webp files are not supported.');
  const rgb = mode === 'RGB' ? true : mode === 'RGB_ALPHA' ? false : !webp.hasAlpha;
  const rgba = webp.rgba();
  const count = webp.width * webp.height;
  if (!rgb) return toChw(rgba, webp.width, webp.height, 4);
  const out = new Uint8Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    out[i * 3] = rgba[i * 4]!;
    out[i * 3 + 1] = rgba[i * 4 + 1]!;
    out[i * 3 + 2] = rgba[i * 4 + 2]!;
  }
  return toChw(out, webp.width, webp.height, 3);
}

export { ValueError };
