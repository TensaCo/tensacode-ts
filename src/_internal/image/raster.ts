/**
 * ``RasterImage``: the TypeScript counterpart of a decoded ``PIL.Image.Image``.
 *
 * ``openImage`` reproduces ``PIL.Image.open(...)`` followed by ``load()`` for
 * JPEG, PNG, GIF (first frame), WebP (first frame) and BMP files: the Pillow
 * mode each file opens in, its samples, palette and ``transparency``/EXIF
 * info. ``convert``, ``resize`` (all six Pillow filters, see
 * ``resample.ts``), ``transpose``/``exifTranspose`` and ``toTensor``
 * (``torchvision.transforms.functional.pil_to_tensor``) follow Pillow 12.
 */
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { decodePng, isPng, type PngData } from './png.js';
import { decodeJpeg, isJpeg } from './jpeg.js';
import { decodeGif, isGif } from './gif.js';
import { decodeWebp, isWebp } from './webp.js';
import { decodeBmp, isBmp } from './bmp.js';
import { pilOrientation, tiffPayload } from './exif.js';
import { resizeRaster } from './resample.js';

export type RasterMode = '1' | 'L' | 'LA' | 'La' | 'P' | 'PA' | 'RGB' | 'RGBA' | 'RGBa' | 'CMYK' | 'I;16' | 'I' | 'F';

export type RasterData = Uint8Array | Uint16Array | Int32Array | Float32Array;

export interface RasterInfo {
  /** Pillow ``info['transparency']``: a palette index, a gray value, an RGB triple or per-entry alphas. */
  transparency?: number | readonly number[] | Uint8Array;
  /** Raw EXIF payload (``Exif\0\0`` header optional), Pillow ``info['exif']``. */
  exif?: Uint8Array;
  /** XMP packet text (``info['xmp']`` / ``info['XML:com.adobe.xmp']``). */
  xmp?: string;
  /** Pillow ``format`` of the source file. */
  format?: string;
  /** Number of frames in the source file (``n_frames``). */
  frames?: number;
}

const BANDS: Record<RasterMode, number> = {
  1: 1, L: 1, LA: 2, La: 2, P: 1, PA: 2, RGB: 3, RGBA: 4, RGBa: 4, CMYK: 4, 'I;16': 1, I: 1, F: 1,
};

/** Number of bands of a Pillow mode (``len(image.getbands())``). */
export function modeBands(mode: RasterMode): number {
  return BANDS[mode];
}

function allocate(mode: RasterMode, length: number): RasterData {
  if (mode === 'I;16') return new Uint16Array(length);
  if (mode === 'I') return new Int32Array(length);
  if (mode === 'F') return new Float32Array(length);
  return new Uint8Array(length);
}

const muldiv255 = (a: number, b: number): number => {
  const tmp = a * b + 128;
  return ((tmp >> 8) + tmp) >> 8;
};
const clip8 = (value: number): number => (value <= 0 ? 0 : value < 256 ? value : 255);
const l24 = (r: number, g: number, b: number): number => (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;

/** A decoded image with Pillow semantics (mode, size, interleaved samples, palette and info). */
export class RasterImage {
  readonly mode: RasterMode;
  readonly width: number;
  readonly height: number;
  /** Interleaved ``[height, width, bands]`` samples (mode ``1`` stored as 0/255). */
  readonly data: RasterData;
  /** RGB palette (256 entries) for ``P``/``PA`` images. */
  readonly palette: Uint8Array | null;
  /** Per-entry palette alphas (``putpalettealpha``), used by ``P`` to ``RGBA`` conversion. */
  readonly paletteAlpha: Uint8Array | null;
  readonly info: RasterInfo;

  constructor(mode: RasterMode, width: number, height: number, data: RasterData, options: {
    palette?: Uint8Array | null; paletteAlpha?: Uint8Array | null; info?: RasterInfo;
  } = {}) {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 0 || height < 0) throw new ValueError('invalid image size');
    if (data.length !== width * height * BANDS[mode]) throw new ValueError(`image data does not match ${mode} ${width}x${height}`);
    this.mode = mode;
    this.width = width;
    this.height = height;
    this.data = data;
    let palette = options.palette ?? null;
    if (palette && palette.length < 768) {
      const full = new Uint8Array(768);
      full.set(palette);
      palette = full;
    }
    this.palette = mode === 'P' || mode === 'PA' ? (palette ?? new Uint8Array(768)) : null;
    this.paletteAlpha = options.paletteAlpha ?? null;
    this.info = { ...(options.info ?? {}) };
  }

  /** ``(width, height)``. */
  get size(): [number, number] {
    return [this.width, this.height];
  }

  /** Number of bands (``len(getbands())``). */
  get bands(): number {
    return BANDS[this.mode];
  }

  /** ``getexif()[Orientation]`` (EXIF, then Pillow's XMP fallback), or ``null`` when absent. */
  get orientation(): number | null {
    return pilOrientation(this.info.exif, this.info.xmp) ?? null;
  }

  private with(mode: RasterMode, data: RasterData, width = this.width, height = this.height, info: RasterInfo = this.info): RasterImage {
    return new RasterImage(mode, width, height, data, { palette: this.palette, paletteAlpha: this.paletteAlpha, info });
  }

  copy(): RasterImage {
    return this.with(this.mode, this.data.slice());
  }

  /** ``Image.convert(mode)`` for ``1``, ``L``, ``LA``, ``RGB``, ``RGBA`` (plus the premultiplied ``La``/``RGBa``). */
  convert(mode: RasterMode): RasterImage {
    if (mode === this.mode) return this.copy();
    const hasTransparency = this.info.transparency !== undefined;
    const info: RasterInfo = { ...this.info };
    if (hasTransparency) {
      const t = this.info.transparency!;
      if ((['1', 'L', 'I', 'I;16'].includes(this.mode) && (mode === 'LA' || mode === 'RGBA'))
        || (this.mode === 'RGB' && ['La', 'LA', 'RGBa', 'RGBA'].includes(mode))) {
        delete info.transparency;
        return this.convertTransparent(mode, t, info);
      }
      if (this.mode === 'P' && ['LA', 'PA', 'RGBA'].includes(mode)) {
        delete info.transparency;
        const alpha = new Uint8Array(256).fill(255);
        if (this.paletteAlpha) alpha.set(this.paletteAlpha);
        if (t instanceof Uint8Array) alpha.set(t.subarray(0, 256));
        else if (typeof t === 'number') alpha[t] = 0;
        else throw new ValueError('Transparency for P mode should be bytes or int');
        return new RasterImage(this.mode, this.width, this.height, this.data, { palette: this.palette, paletteAlpha: alpha, info })
          .convertPlain(mode, info);
      }
      if (['L', 'RGB', 'P'].includes(this.mode) && ['L', 'RGB', 'P'].includes(mode)) {
        if (t instanceof Uint8Array) delete info.transparency;
        else {
          const probe = new RasterImage(this.mode, 1, 1, allocate(this.mode, BANDS[this.mode]), { palette: this.palette });
          const values = typeof t === 'number' ? [t] : [...t];
          for (let c = 0; c < BANDS[this.mode]; c += 1) probe.data[c] = values[c] ?? 0;
          const converted = probe.convertPlain(mode === 'L' ? 'L' : 'RGB', {});
          info.transparency = mode === 'L' ? converted.data[0]! : [converted.data[0]!, converted.data[1]!, converted.data[2]!];
        }
      }
    }
    return this.convertPlain(mode, info);
  }

  private convertTransparent(mode: RasterMode, t: number | readonly number[] | Uint8Array, info: RasterInfo): RasterImage {
    const values = typeof t === 'number' ? [t, t, t] : [...t];
    const [r, g, b] = [values[0] ?? 0, values[1] ?? values[0] ?? 0, values[2] ?? values[0] ?? 0];
    const rgba = this.convertPlain('RGBA', {});
    const out = rgba.data as Uint8Array;
    for (let i = 0; i < this.width * this.height; i += 1) {
      if (out[i * 4] === r && out[i * 4 + 1] === g && out[i * 4 + 2] === b && out[i * 4 + 3] === 255) out[i * 4 + 3] = 0;
    }
    if (mode === 'RGBA') return new RasterImage('RGBA', this.width, this.height, out, { info });
    if (mode === 'RGBa') return new RasterImage('RGBA', this.width, this.height, out, { info }).convertPlain('RGBa', info);
    const la = new Uint8Array(this.width * this.height * 2);
    for (let i = 0; i < this.width * this.height; i += 1) {
      la[i * 2] = this.mode === 'RGB' ? l24(out[i * 4]!, out[i * 4 + 1]!, out[i * 4 + 2]!) : out[i * 4]!;
      la[i * 2 + 1] = out[i * 4 + 3]!;
    }
    const result = new RasterImage('LA', this.width, this.height, la, { info });
    return mode === 'La' ? result.convertPlain('La', info) : result;
  }

  /** Pixel conversions without transparency handling (``ImagingConvert``). */
  private convertPlain(mode: RasterMode, info: RasterInfo): RasterImage {
    if (mode === this.mode) return new RasterImage(mode, this.width, this.height, this.data.slice(), { palette: this.palette, paletteAlpha: this.paletteAlpha, info });
    const count = this.width * this.height;
    const source = this.data;
    const inBands = this.bands;
    // Normalize to RGBA (Pillow's 4-byte pixel layout) first, then pack.
    const rgba = new Uint8Array(count * 4);
    const from = this.mode;
    const unsupported = (): never => {
      throw new ValueError(`conversion from ${from} to ${mode} not supported`);
    };
    if (mode === 'I' || mode === 'F' || mode === 'I;16' || mode === 'P' || mode === 'PA' || mode === 'CMYK') unsupported();
    const palette = this.palette;
    const alpha = this.paletteAlpha;
    for (let i = 0; i < count; i += 1) {
      const o = i * 4;
      switch (from) {
        case '1':
        case 'L': {
          const v = from === '1' ? (source[i] ? 255 : 0) : source[i]!;
          rgba[o] = rgba[o + 1] = rgba[o + 2] = v;
          rgba[o + 3] = 255;
          break;
        }
        case 'LA':
        case 'La':
          rgba[o] = rgba[o + 1] = rgba[o + 2] = source[i * 2]!;
          rgba[o + 3] = source[i * 2 + 1]!;
          break;
        case 'P':
        case 'PA': {
          const index = source[i * inBands]!;
          rgba[o] = palette![index * 3]!;
          rgba[o + 1] = palette![index * 3 + 1]!;
          rgba[o + 2] = palette![index * 3 + 2]!;
          rgba[o + 3] = from === 'PA' ? source[i * 2 + 1]! : (alpha ? alpha[index]! : 255);
          break;
        }
        case 'RGB':
          rgba[o] = source[i * 3]!;
          rgba[o + 1] = source[i * 3 + 1]!;
          rgba[o + 2] = source[i * 3 + 2]!;
          rgba[o + 3] = 255;
          break;
        case 'RGBA':
        case 'RGBa':
          for (let c = 0; c < 4; c += 1) rgba[o + c] = source[i * 4 + c]!;
          break;
        case 'CMYK': {
          const nk = 255 - source[i * 4 + 3]!;
          for (let c = 0; c < 3; c += 1) rgba[o + c] = clip8(nk - muldiv255(source[i * 4 + c]!, nk));
          rgba[o + 3] = 255;
          break;
        }
        case 'I;16': {
          const value = source[i]!;
          rgba[o] = rgba[o + 1] = rgba[o + 2] = value >> 8 === 0 ? value : 255;
          rgba[o + 3] = 255;
          break;
        }
        case 'I': {
          const value = source[i]!;
          rgba[o] = rgba[o + 1] = rgba[o + 2] = value <= 0 ? 0 : value >= 255 ? 255 : value;
          rgba[o + 3] = 255;
          break;
        }
        case 'F': {
          // No direct F -> RGB converter: Pillow goes through L (f2l truncates).
          const value = source[i]!;
          rgba[o] = rgba[o + 1] = rgba[o + 2] = value <= 0 ? 0 : value >= 255 ? 255 : Math.trunc(value);
          rgba[o + 3] = 255;
          break;
        }
      }
    }
    const out = new Uint8Array(count * BANDS[mode]);
    const colorSource = from === 'RGB' || from === 'RGBA' || from === 'RGBa' || from === 'CMYK' || from === 'P' || from === 'PA';
    for (let i = 0; i < count; i += 1) {
      const o = i * 4;
      const r = rgba[o]!;
      const g = rgba[o + 1]!;
      const b = rgba[o + 2]!;
      const a = rgba[o + 3]!;
      switch (mode) {
        case 'RGB':
          out[i * 3] = r; out[i * 3 + 1] = g; out[i * 3 + 2] = b;
          break;
        case 'RGBA':
          if (from === 'RGBa') {
            if (a === 255 || a === 0) { out[o] = r; out[o + 1] = g; out[o + 2] = b; } else {
              out[o] = clip8(Math.trunc((255 * r) / a));
              out[o + 1] = clip8(Math.trunc((255 * g) / a));
              out[o + 2] = clip8(Math.trunc((255 * b) / a));
            }
            out[o + 3] = a;
          } else {
            out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = a;
          }
          break;
        case 'RGBa':
          if (from === 'RGBA' || from === 'RGB') {
            out[o] = muldiv255(r, a); out[o + 1] = muldiv255(g, a); out[o + 2] = muldiv255(b, a); out[o + 3] = a;
          } else unsupported();
          break;
        case 'L':
          out[i] = colorSource ? l24(r, g, b) : r;
          break;
        case '1': {
          const l = colorSource ? l24(r, g, b) : r;
          out[i] = l >= 128 ? 255 : 0;
          break;
        }
        case 'LA':
          if (from === 'La') {
            out[i * 2] = a === 255 || a === 0 ? r : clip8(Math.trunc((255 * r) / a));
          } else {
            out[i * 2] = colorSource ? l24(r, g, b) : r;
          }
          out[i * 2 + 1] = a;
          break;
        case 'La':
          if (from !== 'LA') unsupported();
          out[i * 2] = muldiv255(r, a);
          out[i * 2 + 1] = a;
          break;
        default:
          unsupported();
      }
    }
    return new RasterImage(mode, this.width, this.height, out, { info });
  }

  /** ``Image.transpose(method)`` with Pillow's ``Transpose`` codes 0-6. */
  transpose(method: number): RasterImage {
    const { width, height } = this;
    const bands = this.bands;
    const swapped = method === 2 || method === 4 || method === 5 || method === 6;
    const outWidth = swapped ? height : width;
    const outHeight = swapped ? width : height;
    const out = allocate(this.mode, this.data.length);
    for (let y = 0; y < outHeight; y += 1) {
      for (let x = 0; x < outWidth; x += 1) {
        let sx: number;
        let sy: number;
        switch (method) {
          case 0: sx = width - 1 - x; sy = y; break; // FLIP_LEFT_RIGHT
          case 1: sx = x; sy = height - 1 - y; break; // FLIP_TOP_BOTTOM
          case 2: sx = width - 1 - y; sy = x; break; // ROTATE_90 (counter-clockwise)
          case 3: sx = width - 1 - x; sy = height - 1 - y; break; // ROTATE_180
          case 4: sx = y; sy = height - 1 - x; break; // ROTATE_270
          case 5: sx = y; sy = x; break; // TRANSPOSE
          case 6: sx = width - 1 - y; sy = height - 1 - x; break; // TRANSVERSE
          default: throw new ValueError('Expected one of Transpose.FLIP_LEFT_RIGHT, ... TRANSVERSE');
        }
        for (let c = 0; c < bands; c += 1) out[(y * outWidth + x) * bands + c] = this.data[(sy * width + sx) * bands + c]!;
      }
    }
    return this.with(this.mode, out, outWidth, outHeight);
  }

  /**
   * ``Image.resize((width, height), resample, box, reducing_gap)`` with
   * Pillow's filters (``Resampling`` codes: NEAREST 0, LANCZOS 1, BILINEAR 2,
   * BICUBIC 3 (default), BOX 4, HAMMING 5).
   */
  resize(
    size: readonly [number, number], resample = 3, box: readonly [number, number, number, number] | null = null,
    reducingGap: number | null = null,
  ): RasterImage {
    return resizeRaster(this, size, resample, box, reducingGap);
  }

  /** ``ImageOps.exif_transpose(image)`` (orientation removed from the result's EXIF). */
  exifTranspose(): RasterImage {
    const orientation = this.orientation;
    const method = orientation ? ({ 2: 0, 3: 3, 4: 1, 5: 5, 6: 4, 7: 6, 8: 2 } as Record<number, number>)[orientation] : undefined;
    if (method === undefined) return this.copy();
    const result = this.transpose(method);
    const info = { ...result.info };
    if (info.exif) info.exif = withoutOrientation(tiffPayload(info.exif));
    if (info.xmp) info.xmp = info.xmp.replace(/tiff:Orientation="([0-9])"/g, '').replace(/<tiff:Orientation>([0-9])<\/tiff:Orientation>/g, '');
    return new RasterImage(result.mode, result.width, result.height, result.data, { palette: result.palette, paletteAlpha: result.paletteAlpha, info });
  }

  /**
   * ``torchvision.transforms.functional.pil_to_tensor``: a ``[bands, H, W]``
   * tensor. Mode ``1`` gives ``bool``, ``I``/``I;16`` give ``int32`` (TypeScript
   * has no ``uint16`` tensors; values are exact), ``F`` gives ``float32`` and
   * the other modes ``uint8`` (``P`` gives palette indices).
   */
  toTensor(): Tensor {
    const bands = this.bands;
    const plane = this.width * this.height;
    const values = new Float64Array(this.data.length);
    for (let i = 0; i < plane; i += 1) {
      for (let c = 0; c < bands; c += 1) {
        const v = this.data[i * bands + c]!;
        values[c * plane + i] = this.mode === '1' ? (v ? 1 : 0) : v;
      }
    }
    const dtype = this.mode === '1' ? 'bool' : this.mode === 'I' || this.mode === 'I;16' ? 'int32' : this.mode === 'F' ? 'float32' : 'uint8';
    if (dtype === 'float32') return new Tensor(Float32Array.from(values), [bands, this.height, this.width], 'float32');
    return new Tensor(values, [bands, this.height, this.width], dtype);
  }
}

function latin1(bytes: Uint8Array): string {
  let out = '';
  for (const byte of bytes) out += String.fromCharCode(byte);
  return out;
}

/** Zero the orientation tag of a TIFF EXIF payload (Pillow deletes it). */
function withoutOrientation(tiff: Uint8Array): Uint8Array {
  const copy = tiff.slice();
  const little = copy[0] === 0x49;
  const u16 = (o: number): number => (little ? copy[o]! | (copy[o + 1]! << 8) : (copy[o]! << 8) | copy[o + 1]!);
  const u32 = (o: number): number => (little
    ? (copy[o]! | (copy[o + 1]! << 8) | (copy[o + 2]! << 16) | (copy[o + 3]! << 24)) >>> 0
    : ((copy[o]! << 24) | (copy[o + 1]! << 16) | (copy[o + 2]! << 8) | copy[o + 3]!) >>> 0);
  if (copy.length < 8) return copy;
  const ifd = u32(4);
  if (ifd + 2 > copy.length) return copy;
  const count = u16(ifd);
  for (let i = 0; i < count; i += 1) {
    const entry = ifd + 2 + i * 12;
    if (entry + 12 > copy.length) break;
    if (u16(entry) === 0x0112) {
      // Mark the tag as orientation 1 (top-left).
      if (little) { copy[entry + 8] = 1; copy[entry + 9] = 0; } else { copy[entry + 8] = 0; copy[entry + 9] = 1; }
    }
  }
  return copy;
}

// ---------------------------------------------------------------------------
// Image.open for each format.
// ---------------------------------------------------------------------------

/** ``PIL.Image.open(fp)`` + ``load()`` over encoded bytes (first frame of animations). */
export function openImageBytes(bytes: Uint8Array): RasterImage {
  if (isPng(bytes)) return openPng(bytes);
  if (isJpeg(bytes)) return openJpeg(bytes);
  if (isGif(bytes)) return openGif(bytes);
  if (isWebp(bytes)) return openWebp(bytes);
  if (isBmp(bytes)) {
    const bmp = decodeBmp(bytes);
    return new RasterImage(bmp.mode, bmp.width, bmp.height, bmp.data, { palette: bmp.palette, info: { format: 'BMP', frames: 1 } });
  }
  throw new ValueError('cannot identify image file');
}

function openPng(bytes: Uint8Array): RasterImage {
  const png: PngData = decodePng(bytes);
  const { width, height, bitDepth, colorType, samples } = png;
  const count = width * height;
  const info: RasterInfo = { format: 'PNG', frames: 1 };
  if (png.exif) info.exif = png.exif;
  else if (png.text['Raw profile type exif'] !== undefined) {
    const hex = png.text['Raw profile type exif'].split('\n').slice(3).join('');
    const bytes = new Uint8Array(hex.length >> 1);
    for (let i = 0; i < bytes.length; i += 1) bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    info.exif = bytes;
  }
  if (png.text['XML:com.adobe.xmp'] !== undefined) info.xmp = png.text['XML:com.adobe.xmp'];
  const trns = png.transparency;
  const be16 = (offset: number): number => (trns![offset]! << 8) | trns![offset + 1]!;
  let image: RasterImage;
  if (colorType === 0) {
    if (bitDepth === 16) {
      image = new RasterImage('I;16', width, height, Uint16Array.from(samples), { info });
    } else if (bitDepth === 1) {
      image = new RasterImage('1', width, height, Uint8Array.from(samples, (v) => (v ? 255 : 0)), { info });
    } else {
      const scale = 255 / ((1 << bitDepth) - 1);
      image = new RasterImage('L', width, height, Uint8Array.from(samples, (v) => v * scale), { info });
    }
    if (trns && trns.length >= 2) image.info.transparency = be16(0);
  } else if (colorType === 2) {
    const data = new Uint8Array(count * 3);
    for (let i = 0; i < data.length; i += 1) data[i] = bitDepth === 16 ? samples[i]! >> 8 : samples[i]!;
    image = new RasterImage('RGB', width, height, data, { info });
    if (trns && trns.length >= 6) image.info.transparency = [be16(0), be16(2), be16(4)];
  } else if (colorType === 3) {
    image = new RasterImage('P', width, height, Uint8Array.from(samples), { palette: png.palette, info });
    if (trns) {
      // _simple_palette: one fully transparent entry, the others opaque.
      let zero = -1;
      let simple = true;
      for (let i = 0; i < trns.length; i += 1) {
        if (trns[i] === 0) {
          if (zero >= 0) simple = false;
          zero = i;
        } else if (trns[i] !== 255) simple = false;
      }
      if (simple && zero >= 0) image.info.transparency = zero;
      else if (!simple) image.info.transparency = trns.slice();
    }
  } else if (colorType === 4) {
    if (bitDepth === 16) {
      const data = new Uint8Array(count * 4);
      for (let i = 0; i < count; i += 1) {
        const l = samples[i * 2]! >> 8;
        data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = l;
        data[i * 4 + 3] = samples[i * 2 + 1]! >> 8;
      }
      image = new RasterImage('RGBA', width, height, data, { info });
    } else {
      image = new RasterImage('LA', width, height, Uint8Array.from(samples), { info });
    }
  } else {
    const data = new Uint8Array(count * 4);
    for (let i = 0; i < data.length; i += 1) data[i] = bitDepth === 16 ? samples[i]! >> 8 : samples[i]!;
    image = new RasterImage('RGBA', width, height, data, { info });
  }
  return image;
}

function openJpeg(bytes: Uint8Array): RasterImage {
  const jpeg = decodeJpeg(bytes);
  // JpegImagePlugin rejects anything but 8-bit samples ("cannot handle N-bit layers").
  if (jpeg.precision !== 8) throw new ValueError('cannot identify image file');
  const info: RasterInfo = { format: 'JPEG', frames: 1 };
  if (jpeg.exif) info.exif = jpeg.exif;
  if (jpeg.xmp) info.xmp = latin1(jpeg.xmp);
  if (jpeg.components === 1) return new RasterImage('L', jpeg.width, jpeg.height, jpeg.output('GRAYSCALE'), { info });
  if (jpeg.components === 3) return new RasterImage('RGB', jpeg.width, jpeg.height, jpeg.output('RGB'), { info });
  if (jpeg.components === 4) {
    // Pillow assumes Adobe conventions: rawmode "CMYK;I" inverts libjpeg's CMYK output.
    const cmyk = jpeg.output('CMYK');
    for (let i = 0; i < cmyk.length; i += 1) cmyk[i] = 255 - cmyk[i]!;
    return new RasterImage('CMYK', jpeg.width, jpeg.height, cmyk, { info });
  }
  throw new ValueError(`cannot handle ${jpeg.components}-layer images`);
}

function openGif(bytes: Uint8Array): RasterImage {
  const gif = decodeGif(bytes);
  const frame = gif.frames[0]!;
  const palette = frame.palette ?? gif.globalPalette;
  // Pillow keeps identity grayscale palettes as mode "L".
  const needsPalette = (table: Uint8Array | null): boolean => {
    if (!table) return false;
    for (let i = 0; i * 3 + 2 < table.length; i += 1) {
      if (!(table[i * 3] === i && table[i * 3 + 1] === i && table[i * 3 + 2] === i)) return true;
    }
    return false;
  };
  const framePalette = frame.palette ? (needsPalette(frame.palette) ? frame.palette : null) : (needsPalette(gif.globalPalette) ? gif.globalPalette : null);
  void palette;
  const width = Math.max(gif.width, frame.left + frame.width);
  const height = Math.max(gif.height, frame.top + frame.height);
  const fill = frame.transparentIndex >= 0 ? frame.transparentIndex : 0;
  const data = new Uint8Array(width * height).fill(fill);
  for (let y = 0; y < frame.height; y += 1) {
    for (let x = 0; x < frame.width; x += 1) data[(frame.top + y) * width + frame.left + x] = frame.indices[y * frame.width + x]!;
  }
  const info: RasterInfo = { format: 'GIF', frames: gif.frames.length };
  if (framePalette) {
    if (frame.transparentIndex >= 0) info.transparency = frame.transparentIndex;
    const full = new Uint8Array(768);
    full.set(framePalette.subarray(0, 768));
    return new RasterImage('P', width, height, data, { palette: full, info });
  }
  if (frame.transparentIndex >= 0) info.transparency = frame.transparentIndex;
  return new RasterImage('L', width, height, data, { info });
}

function openWebp(bytes: Uint8Array): RasterImage {
  const webp = decodeWebp(bytes);
  const canvas = webp.firstCanvas();
  const info: RasterInfo = { format: 'WEBP', frames: webp.frameCount };
  if (webp.exif) info.exif = webp.exif;
  if (webp.xmp) info.xmp = latin1(webp.xmp);
  if (webp.hasAlpha) return new RasterImage('RGBA', webp.width, webp.height, canvas, { info });
  const rgb = new Uint8Array(webp.width * webp.height * 3);
  for (let i = 0; i < webp.width * webp.height; i += 1) {
    rgb[i * 3] = canvas[i * 4]!;
    rgb[i * 3 + 1] = canvas[i * 4 + 1]!;
    rgb[i * 3 + 2] = canvas[i * 4 + 2]!;
  }
  return new RasterImage('RGB', webp.width, webp.height, rgb, { info });
}
