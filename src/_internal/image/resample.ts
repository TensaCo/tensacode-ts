/**
 * Pillow 12 ``Image.resize`` (``libImaging/Resample.c`` and the nearest
 * neighbour ``ImagingScaleAffine`` / ``ImagingGenericTransform`` paths):
 * NEAREST, BOX, BILINEAR, HAMMING, BICUBIC and LANCZOS with Pillow's double
 * precision coefficients, 22-bit fixed-point accumulation for 8-bit bands,
 * double accumulation with ``ROUND_UP`` for ``I``/``I;16``, float output for
 * ``F``, the ``box`` region, ``reducing_gap`` and the premultiplied-alpha
 * handling of ``LA``/``RGBA``.
 */
import { ValueError } from '../../errors.js';
import { RasterImage, type RasterData, type RasterMode } from './raster.js';
import { fma, glibcCos, glibcSin } from './fpmath.js';

/** ``PIL.Image.Resampling`` codes. */
export const Resampling = Object.freeze({ NEAREST: 0, LANCZOS: 1, BILINEAR: 2, BICUBIC: 3, BOX: 4, HAMMING: 5 });

interface Filter {
  filter(x: number): number;
  support: number;
}

const F32_054 = Math.fround(0.54);
const F32_046 = Math.fround(0.46);

const FILTERS: Record<number, Filter> = {
  4: { support: 0.5, filter: (x) => (x > -0.5 && x <= 0.5 ? 1 : 0) },
  2: { support: 1, filter: (x) => {
    const v = Math.abs(x);
    return v < 1 ? 1 - v : 0;
  } },
  5: { support: 1, filter: (x) => {
    let v = Math.abs(x);
    if (v === 0) return 1;
    if (v >= 1) return 0;
    v *= Math.PI;
    return (glibcSin(v) / v) * (F32_054 + F32_046 * glibcCos(v));
  } },
  3: { support: 2, filter: (x) => {
    const a = -0.5;
    const v = Math.abs(x);
    if (v < 1) return ((a + 2) * v - (a + 3)) * v * v + 1;
    if (v < 2) return (((v - 5) * v + 8) * v - 4) * a;
    return 0;
  } },
  1: { support: 3, filter: (x) => {
    const sinc = (value: number): number => {
      if (value === 0) return 1;
      const scaled = value * Math.PI;
      return glibcSin(scaled) / scaled;
    };
    return x >= -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0;
  } },
};

interface Coefficients {
  ksize: number;
  bounds: Int32Array;
  kk: Float64Array;
}

/** ``precompute_coeffs`` (``in0``/``in1`` are float box edges). */
function precomputeCoeffs(inSize: number, in0: number, in1: number, outSize: number, filter: Filter): Coefficients {
  // (double)(in1 - in0) / outSize with float box edges.
  const scale = Math.fround(in1 - in0) / outSize;
  const filterscale = scale < 1 ? 1 : scale;
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Float64Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const inverse = 1 / filterscale;
  for (let xx = 0; xx < outSize; xx += 1) {
    const center = USE_FMA ? fma(xx + 0.5, scale, in0) : in0 + (xx + 0.5) * scale;
    let ww = 0;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const base = xx * ksize;
    for (let x = 0; x < xmax; x += 1) {
      const w = filter.filter((x + xmin - center + 0.5) * inverse);
      kk[base + x] = w;
      ww += w;
    }
    if (ww !== 0) for (let x = 0; x < xmax; x += 1) kk[base + x] /= ww;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

const PRECISION_BITS = 32 - 8 - 2;
// Pillow's aarch64 wheels contract ``ss += a * b`` and ``in0 + (xx + 0.5) * scale`` into fused multiply-adds.
const USE_FMA = true;

/** ``normalize_coeffs_8bpc``. */
function fixedCoefficients(kk: Float64Array): Int32Array {
  const out = new Int32Array(kk.length);
  for (let i = 0; i < kk.length; i += 1) {
    const v = kk[i]! * (1 << PRECISION_BITS);
    out[i] = kk[i]! < 0 ? Math.trunc(-0.5 + v) : Math.trunc(0.5 + v);
  }
  return out;
}

function clip8(value: number): number {
  const index = value >> PRECISION_BITS;
  return index < 0 ? 0 : index > 255 ? 255 : index;
}

/** ``ROUND_UP`` on a double. */
function roundUp(value: number): number {
  return Math.trunc(value >= 0 ? value + 0.5 : value - 0.5);
}

type Kind = '8bpc' | 'I' | 'F' | 'I;16';

function kindOf(mode: RasterMode): Kind {
  if (mode === 'I') return 'I';
  if (mode === 'F') return 'F';
  if (mode === 'I;16') return 'I;16';
  return '8bpc';
}

/** Pillow's 4-byte pixel slots used by multi-band 8-bit images (``LA`` keeps bands 0 and 3). */
function resamplePass(
  data: RasterData, bands: number, kind: Kind, inWidth: number, rows: number, rowOffset: number,
  coefficients: Coefficients, outLength: number, horizontal: boolean, inHeight: number,
): RasterData {
  const { ksize, bounds, kk } = coefficients;
  const out = kind === 'I;16' ? new Uint16Array(outLength) : kind === 'I' ? new Int32Array(outLength) : kind === 'F' ? new Float32Array(outLength) : new Uint8Array(outLength);
  const fixed = kind === '8bpc' ? fixedCoefficients(kk) : null;
  const outWidth = horizontal ? bounds.length / 2 : inWidth;
  const outHeight = horizontal ? rows : bounds.length / 2;
  void inHeight;
  for (let yy = 0; yy < outHeight; yy += 1) {
    for (let xx = 0; xx < outWidth; xx += 1) {
      const target = horizontal ? xx : yy;
      const min = bounds[target * 2]!;
      const max = bounds[target * 2 + 1]!;
      const k = target * ksize;
      for (let b = 0; b < bands; b += 1) {
        const at = (i: number): number => (horizontal
          ? data[((yy + rowOffset) * inWidth + min + i) * bands + b]!
          : data[((min + i) * inWidth + xx) * bands + b]!);
        const o = (yy * outWidth + xx) * bands + b;
        if (fixed) {
          let ss = 1 << (PRECISION_BITS - 1);
          for (let i = 0; i < max; i += 1) ss += at(i) * fixed[k + i]!;
          out[o] = clip8(ss | 0);
        } else {
          let ss = 0;
          for (let i = 0; i < max; i += 1) ss = USE_FMA ? fma(at(i), kk[k + i]!, ss) : ss + at(i) * kk[k + i]!;
          if (kind === 'F') out[o] = ss;
          else if (kind === 'I;16') {
            const value = roundUp(ss);
            out[o] = value <= 0 ? 0 : value < 65536 ? value : 65535;
          } else out[o] = roundUp(ss) | 0;
        }
      }
    }
  }
  return out;
}

/** ``ImagingResample`` for a mode Pillow resamples (not ``1``/``P``). */
function imagingResample(image: RasterImage, width: number, height: number, filter: number, box: readonly [number, number, number, number]): RasterImage {
  const filterp = FILTERS[filter];
  if (!filterp) throw new ValueError('unsupported resampling filter');
  const kind = kindOf(image.mode);
  const bands = image.bands;
  const needHorizontal = width !== image.width || box[0] !== 0 || box[2] !== width;
  const needVertical = height !== image.height || box[1] !== 0 || box[3] !== height;
  const vertical = precomputeCoeffs(image.height, box[1], box[3], height, filterp);
  const yFirst = vertical.bounds[0]!;
  const yLast = vertical.bounds[height * 2 - 2]! + vertical.bounds[height * 2 - 1]!;
  let data: RasterData = image.data;
  let currentWidth = image.width;
  let currentHeight = image.height;
  if (needHorizontal) {
    const horizontal = precomputeCoeffs(image.width, box[0], box[2], width, filterp);
    for (let i = 0; i < height; i += 1) vertical.bounds[i * 2] = vertical.bounds[i * 2]! - yFirst;
    const rows = yLast - yFirst;
    data = resamplePass(data, bands, kind, currentWidth, rows, yFirst, horizontal, width * rows * bands, true, currentHeight);
    currentWidth = width;
    currentHeight = rows;
  }
  if (needVertical) {
    data = resamplePass(data, bands, kind, currentWidth, height, 0, vertical, currentWidth * height * bands, false, currentHeight);
    currentHeight = height;
  }
  if (!needHorizontal && !needVertical) data = data.slice();
  return new RasterImage(image.mode, currentWidth, currentHeight, data, { palette: image.palette, paletteAlpha: image.paletteAlpha, info: image.info });
}

/** ``ImagingScaleAffine`` / generic nearest transform for NEAREST resizes. */
function nearest(image: RasterImage, width: number, height: number, box: readonly [number, number, number, number]): RasterImage {
  const a0 = Math.fround(box[2] - box[0]) / width;
  const a4 = Math.fround(box[3] - box[1]) / height;
  const bands = image.bands;
  const out = image.data instanceof Uint16Array ? new Uint16Array(width * height * bands)
    : image.data instanceof Int32Array ? new Int32Array(width * height * bands)
      : image.data instanceof Float32Array ? new Float32Array(width * height * bands) : new Uint8Array(width * height * bands);
  const coord = (value: number): number => (value < 0 ? -1 : Math.trunc(value));
  if (image.mode === 'I;16') {
    // ImagingGenericTransform with affine_transform: direct evaluation per pixel.
    for (let y = 0; y < height; y += 1) {
      const yin = coord(a4 * (y + 0.5) + box[1]);
      for (let x = 0; x < width; x += 1) {
        const xin = coord(a0 * (x + 0.5) + box[0]);
        if (xin < 0 || xin >= image.width || yin < 0 || yin >= image.height) continue;
        for (let b = 0; b < bands; b += 1) out[(y * width + x) * bands + b] = image.data[(yin * image.width + xin) * bands + b]!;
      }
    }
  } else {
    // ImagingScaleAffine: accumulated source coordinates.
    const xintab = new Int32Array(width);
    let xo = box[0] + a0 * 0.5;
    let xmin = width;
    let xmax = 0;
    for (let x = 0; x < width; x += 1) {
      const xin = coord(xo);
      if (xin >= 0 && xin < image.width) {
        xmax = x + 1;
        if (x < xmin) xmin = x;
        xintab[x] = xin;
      }
      xo += a0;
    }
    let yo = box[1] + a4 * 0.5;
    for (let y = 0; y < height; y += 1) {
      const yi = coord(yo);
      if (yi >= 0 && yi < image.height) {
        for (let x = xmin; x < xmax; x += 1) {
          for (let b = 0; b < bands; b += 1) out[(y * width + x) * bands + b] = image.data[(yi * image.width + xintab[x]!) * bands + b]!;
        }
      }
      yo += a4;
    }
  }
  return new RasterImage(image.mode, width, height, out, { palette: image.palette, paletteAlpha: image.paletteAlpha, info: image.info });
}

/** ``division_UINT32``: ``(UINT32)((float)2**32 / (float)(divider << result_bits))``. */
function divisionUint32(divider: number, resultBits: number): number {
  return Math.trunc(Math.fround(2 ** 32 / Math.fround(divider * 2 ** resultBits)));
}

/**
 * ``Image.reduce(factor, box)`` (``libImaging/Reduce.c``): 8-bit bands average
 * ``(sum + count / 2) * multiplier >> 24``, ``I`` rounds a double average and
 * ``F`` keeps it; partial last columns/rows average what is present.
 */
function reduce(image: RasterImage, factorX: number, factorY: number, box: readonly [number, number, number, number]): RasterImage {
  if (image.mode === 'P' || image.mode === '1' || image.mode === 'I;16') throw new ValueError('image has wrong mode');
  if (factorX < 1 || factorY < 1) throw new ValueError('scale must be > 0');
  if (box[0] < 0 || box[1] < 0) throw new ValueError("box offset can't be negative");
  if (box[2] > image.width || box[3] > image.height) throw new ValueError("box can't exceed original image size");
  if (box[2] <= box[0] || box[3] <= box[1]) throw new ValueError("box can't be empty");
  if (factorX === 1 && factorY === 1) return crop(image, box[0], box[1], box[2] - box[0], box[3] - box[1]);
  const [left, top] = box;
  const boxWidth = box[2] - box[0];
  const boxHeight = box[3] - box[1];
  const width = Math.trunc((boxWidth + factorX - 1) / factorX);
  const height = Math.trunc((boxHeight + factorY - 1) / factorY);
  const bands = image.bands;
  const kind = kindOf(image.mode);
  const out = kind === 'I' ? new Int32Array(width * height * bands) : kind === 'F' ? new Float32Array(width * height * bands) : new Uint8Array(width * height * bands);
  const pixel = (x: number, y: number, b: number): number => image.data[(y * image.width + x) * bands + b]!;
  const fullColumns = Math.trunc(boxWidth / factorX);
  const fullRows = Math.trunc(boxHeight / factorY);
  const store = (x: number, y: number, b: number, value: number): void => {
    out[(y * width + x) * bands + b] = value;
  };
  if (kind === '8bpc') {
    // Integer sums are exact: any summation order matches Reduce.c.
    for (let y = 0; y < height; y += 1) {
      const y0 = top + y * factorY;
      const y1 = Math.min(y0 + factorY, top + boxHeight);
      for (let x = 0; x < width; x += 1) {
        const x0 = left + x * factorX;
        const x1 = Math.min(x0 + factorX, left + boxWidth);
        const count = (x1 - x0) * (y1 - y0);
        const multiplier = divisionUint32(count, 8);
        for (let b = 0; b < bands; b += 1) {
          let sum = Math.trunc(count / 2);
          for (let yy = y0; yy < y1; yy += 1) for (let xx = x0; xx < x1; xx += 1) sum += pixel(xx, yy, b);
          store(x, y, b, Math.floor((sum * multiplier) / 2 ** 24) & 0xff);
        }
      }
    }
  } else {
    // ImagingReduceNxN_32bpc: pairs of rows and columns; float32 inputs add in float first.
    const add = kind === 'F' ? (...values: number[]): number => values.reduce((a, v) => Math.fround(a + v)) : (...values: number[]): number => values.reduce((a, v) => a + v);
    const finish = (sum: number, multiplier: number): number => (kind === 'F' ? sum * multiplier : roundUp(sum * multiplier));
    for (let y = 0; y < fullRows; y += 1) {
      const yFrom = top + y * factorY;
      for (let x = 0; x < fullColumns; x += 1) {
        const xFrom = left + x * factorX;
        for (let b = 0; b < bands; b += 1) {
          let ss = 0;
          let yy = yFrom;
          for (; yy < yFrom + factorY - 1; yy += 2) {
            let xx = xFrom;
            for (; xx < xFrom + factorX - 1; xx += 2) ss += add(pixel(xx, yy, b), pixel(xx + 1, yy, b), pixel(xx, yy + 1, b), pixel(xx + 1, yy + 1, b));
            if (factorX & 1) ss += add(pixel(xx, yy, b), pixel(xx, yy + 1, b));
          }
          if (factorY & 1) {
            let xx = xFrom;
            for (; xx < xFrom + factorX - 1; xx += 2) ss += add(pixel(xx, yy, b), pixel(xx + 1, yy, b));
            if (factorX & 1) ss += pixel(xx, yy, b);
          }
          store(x, y, b, finish(ss, 1 / (factorY * factorX)));
        }
      }
    }
    // ImagingReduceCorners_32bpc.
    const corner = (x: number, y: number, x0: number, x1: number, y0: number, y1: number, count: number): void => {
      for (let b = 0; b < bands; b += 1) {
        let ss = 0;
        for (let yy = y0; yy < y1; yy += 1) for (let xx = x0; xx < x1; xx += 1) ss += pixel(xx, yy, b);
        store(x, y, b, finish(ss, 1 / count));
      }
    };
    const remX = boxWidth % factorX;
    const remY = boxHeight % factorY;
    if (remX) {
      for (let y = 0; y < fullRows; y += 1) {
        const yFrom = top + y * factorY;
        corner(fullColumns, y, left + fullColumns * factorX, left + boxWidth, yFrom, yFrom + factorY, remX * factorY);
      }
    }
    if (remY) {
      for (let x = 0; x < fullColumns; x += 1) {
        const xFrom = left + x * factorX;
        corner(x, fullRows, xFrom, xFrom + factorX, top + fullRows * factorY, top + boxHeight, factorX * remY);
      }
    }
    if (remX && remY) {
      corner(fullColumns, fullRows, left + fullColumns * factorX, left + boxWidth, top + fullRows * factorY, top + boxHeight, remX * remY);
    }
  }
  return new RasterImage(image.mode, width, height, out, { palette: image.palette, paletteAlpha: image.paletteAlpha, info: image.info });
}

/** ``Image._get_safe_box``. */
function safeBox(image: RasterImage, size: readonly [number, number], resample: number, box: readonly [number, number, number, number]): [number, number, number, number] {
  const filterSupport = FILTERS[resample]!.support - 0.5;
  const scaleX = (box[2] - box[0]) / size[0];
  const scaleY = (box[3] - box[1]) / size[1];
  const supportX = filterSupport * scaleX;
  const supportY = filterSupport * scaleY;
  return [
    Math.max(0, Math.trunc(box[0] - supportX)),
    Math.max(0, Math.trunc(box[1] - supportY)),
    Math.min(image.width, Math.ceil(box[2] + supportX)),
    Math.min(image.height, Math.ceil(box[3] + supportY)),
  ];
}

/**
 * ``Image.resize(size, resample, box, reducing_gap)``: ``size`` is
 * ``[width, height]`` and ``box`` ``[left, upper, right, lower]`` in source
 * pixels (floats allowed). Defaults to BICUBIC; ``1`` and ``P`` images always
 * use NEAREST.
 */
export function resizeRaster(
  image: RasterImage, size: readonly [number, number], resample: number = Resampling.BICUBIC,
  box: readonly [number, number, number, number] | null = null, reducingGap: number | null = null,
): RasterImage {
  if (!Object.values(Resampling).includes(resample as 0)) {
    throw new ValueError(`Unknown resampling filter (${resample}). Use Image.Resampling.NEAREST (0), Image.Resampling.LANCZOS (1), Image.Resampling.BILINEAR (2), Image.Resampling.BICUBIC (3), Image.Resampling.BOX (4) or Image.Resampling.HAMMING (5)`);
  }
  if (reducingGap !== null && reducingGap < 1) throw new ValueError('reducing_gap must be 1.0 or greater');
  let region: [number, number, number, number] = box ? [box[0], box[1], box[2], box[3]] : [0, 0, image.width, image.height];
  const [width, height] = size;
  if (image.width === width && image.height === height && region[0] === 0 && region[1] === 0 && region[2] === image.width && region[3] === image.height) {
    return image.copy();
  }
  let filter = resample;
  if (image.mode === '1' || image.mode === 'P') filter = Resampling.NEAREST;
  if ((image.mode === 'LA' || image.mode === 'RGBA') && filter !== Resampling.NEAREST) {
    const premultiplied = image.convert(image.mode === 'LA' ? 'La' : 'RGBa');
    return resizeRaster(premultiplied, size, filter, region).convert(image.mode);
  }
  let source = image;
  if (reducingGap !== null && filter !== Resampling.NEAREST) {
    const factorX = Math.trunc((region[2] - region[0]) / width / reducingGap) || 1;
    const factorY = Math.trunc((region[3] - region[1]) / height / reducingGap) || 1;
    if (factorX > 1 || factorY > 1) {
      const reduceBox = safeBox(source, size, filter, region);
      source = reduce(source, factorX, factorY, reduceBox);
      region = [
        (region[0] - reduceBox[0]) / factorX, (region[1] - reduceBox[1]) / factorY,
        (region[2] - reduceBox[0]) / factorX, (region[3] - reduceBox[1]) / factorY,
      ];
    }
  }
  return coreResize(source, width, height, filter, region);
}

/** ``ImagingCore.resize`` (``_resize`` in ``_imaging.c``). */
function coreResize(image: RasterImage, width: number, height: number, filter: number, box: [number, number, number, number]): RasterImage {
  if (width < 1 || height < 1) throw new ValueError('height and width must be > 0');
  const f = box.map((value) => Math.fround(value)) as [number, number, number, number];
  const fsub = (a: number, b: number): number => Math.fround(a - b);
  if (f[0] < 0 || f[1] < 0) throw new ValueError("box offset can't be negative");
  if (f[2] > image.width || f[3] > image.height) throw new ValueError("box can't exceed original image size");
  if (fsub(f[2], f[0]) < 0 || fsub(f[3], f[1]) < 0) throw new ValueError("box can't be empty");
  if (fsub(f[0], Math.trunc(f[0])) === 0 && fsub(f[2], f[0]) === width && fsub(f[1], Math.trunc(f[1])) === 0 && fsub(f[3], f[1]) === height) {
    return crop(image, f[0], f[1], width, height);
  }
  if (filter === Resampling.NEAREST) return nearest(image, width, height, f);
  if (image.mode === 'P' || image.mode === '1') throw new ValueError('image has wrong mode');
  if (image.height > image.width * 100 && height < image.height) {
    const first = imagingResample(image, image.width, height, filter, [0, f[1], image.width, f[3]]);
    return imagingResample(first, width, height, filter, [f[0], 0, f[2], height]);
  }
  return imagingResample(image, width, height, filter, f);
}

function crop(image: RasterImage, left: number, top: number, width: number, height: number): RasterImage {
  const bands = image.bands;
  const out = image.data instanceof Uint16Array ? new Uint16Array(width * height * bands)
    : image.data instanceof Int32Array ? new Int32Array(width * height * bands)
      : image.data instanceof Float32Array ? new Float32Array(width * height * bands) : new Uint8Array(width * height * bands);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      for (let b = 0; b < bands; b += 1) out[(y * width + x) * bands + b] = image.data[((y + top) * image.width + x + left) * bands + b]!;
    }
  }
  return new RasterImage(image.mode, width, height, out, { palette: image.palette, paletteAlpha: image.paletteAlpha, info: image.info });
}
