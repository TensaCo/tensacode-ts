/**
 * PNG decoding: every colour type and bit depth, Adam7 interlacing, palettes
 * with ``tRNS`` and 16-bit samples. The decoder returns raw samples; the
 * libpng (torchvision) and Pillow conversions live in ``decode.ts``.
 */
import { ValueError } from '../../errors.js';
import { inflateZlib } from './inflate.js';

export interface PngData {
  width: number;
  height: number;
  bitDepth: number;
  colorType: number;
  interlaced: boolean;
  /** Samples per pixel (1 gray/palette, 2 gray+alpha, 3 RGB, 4 RGBA). */
  channels: number;
  /** Unscaled samples, one element per sample, row major ``[H, W, C]``. */
  samples: Uint8Array | Uint16Array;
  /** ``PLTE`` entries as ``[r, g, b, ...]``. */
  palette: Uint8Array | null;
  /** Raw ``tRNS`` chunk. */
  transparency: Uint8Array | null;
  /** Raw ``eXIf`` payload (anywhere in the file, as Pillow reads it). */
  exif: Uint8Array | null;
  /** Raw ``eXIf`` payload seen before the image data (what libpng reports). */
  exifBeforeData: Uint8Array | null;
  /** Latin-1/UTF-8 text chunks (``tEXt``, ``zTXt``, ``iTXt``) by keyword. */
  text: Record<string, string>;
  /** Types of every chunk seen before the image data. */
  chunks: Record<string, true>;
  /**
   * libpng 1.6.58 ``chunk_gamma`` (PNGv3 precedence: ``sRGB`` wins over an
   * earlier ``gAMA``; a later ``gAMA`` never replaces it), 0 when unset.
   */
  chunkGamma: number;
  /** ``sBIT`` significant bits (gray, red, green, blue), or ``null``. */
  significantBits: { gray: number; red: number; green: number; blue: number } | null;
}

export const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
const VALID_DEPTHS: Record<number, number[]> = { 0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16] };
const ADAM7 = [[0, 0, 8, 8], [4, 0, 8, 8], [0, 4, 4, 8], [2, 0, 4, 4], [0, 2, 2, 4], [1, 0, 2, 2], [0, 1, 1, 2]] as const;

let crcTable: Uint32Array | null = null;

export function crc32(bytes: Uint8Array, start = 0, end = bytes.length): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n += 1) {
      let c = n;
      for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (let i = start; i < end; i += 1) crc = crcTable[(crc ^ bytes[i]!) & 0xff]! ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

export function isPng(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && PNG_SIGNATURE.every((value, index) => bytes[index] === value);
}

function u32(bytes: Uint8Array, offset: number): number {
  return ((bytes[offset]! << 24) | (bytes[offset + 1]! << 16) | (bytes[offset + 2]! << 8) | bytes[offset + 3]!) >>> 0;
}

/** Decode a PNG file into raw samples. */
export function decodePng(bytes: Uint8Array): PngData {
  if (!isPng(bytes)) throw new ValueError('not a PNG file');
  let offset = 8;
  let header: { width: number; height: number; bitDepth: number; colorType: number; interlaced: boolean } | null = null;
  let palette: Uint8Array | null = null;
  let transparency: Uint8Array | null = null;
  let exif: Uint8Array | null = null;
  let exifBeforeData: Uint8Array | null = null;
  const text: Record<string, string> = {};
  const chunks: Record<string, true> = {};
  let chunkGamma = 0;
  let significantBits: PngData['significantBits'] = null;
  const idat: Uint8Array[] = [];
  while (offset + 8 <= bytes.length) {
    const length = u32(bytes, offset);
    const type = String.fromCharCode(bytes[offset + 4]!, bytes[offset + 5]!, bytes[offset + 6]!, bytes[offset + 7]!);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > bytes.length) {
      if (type === 'IDAT' && header) {
        idat.push(bytes.subarray(dataStart, Math.min(dataEnd, bytes.length)));
        break;
      }
      throw new ValueError(`truncated PNG chunk ${type}`);
    }
    const critical = (bytes[offset + 4]! & 0x20) === 0;
    const crcOk = crc32(bytes, offset + 4, dataEnd) === u32(bytes, dataEnd);
    offset = dataEnd + 4;
    if (!crcOk) {
      if (critical) throw new ValueError(`broken PNG file (bad checksum in ${type})`);
      continue;
    }
    const data = bytes.subarray(dataStart, dataEnd);
    if (!idat.length) chunks[type] = true;
    if (type === 'IHDR') {
      if (length < 13) throw new ValueError('invalid PNG IHDR');
      header = {
        width: u32(data, 0), height: u32(data, 4), bitDepth: data[8]!, colorType: data[9]!, interlaced: data[12] === 1,
      };
      if (data[10] !== 0 || data[11] !== 0 || data[12]! > 1) throw new ValueError('unsupported PNG compression, filter or interlace method');
      const depths = VALID_DEPTHS[header.colorType];
      if (!depths || !depths.includes(header.bitDepth)) {
        throw new ValueError(`invalid PNG colour type ${header.colorType} with bit depth ${header.bitDepth}`);
      }
      if (header.width === 0 || header.height === 0) throw new ValueError('invalid PNG image size');
    } else if (type === 'PLTE') {
      palette = data.slice(0, length - (length % 3));
    } else if (type === 'tRNS') {
      transparency = data.slice();
    } else if (type === 'gAMA' && !idat.length && length === 4) {
      const gamma = u32(data, 0);
      if (gamma <= 0x7fffffff && chunkGamma === 0) chunkGamma = gamma;
    } else if (type === 'sRGB' && !idat.length && length === 1 && data[0]! <= 3) {
      if (!chunks.cICP || chunkGamma === 0) chunkGamma = 45455;
    } else if (type === 'sBIT' && !idat.length && header) {
      if (header.colorType === 0 || header.colorType === 4) significantBits = { gray: data[0]!, red: 0, green: 0, blue: 0 };
      else significantBits = { gray: 0, red: data[0]!, green: data[1]!, blue: data[2]! };
    } else if (type === 'eXIf') {
      exif = data.slice();
      if (!idat.length) exifBeforeData = exif;
    } else if (type === 'tEXt' || type === 'zTXt' || type === 'iTXt') {
      try {
        const [keyword, value] = readText(type, data);
        if (!(keyword in text)) text[keyword] = value;
      } catch {
        // Pillow ignores undecodable text chunks.
      }
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
  }
  if (!header) throw new ValueError('PNG file has no IHDR chunk');
  if (!idat.length) throw new ValueError('PNG file has no image data');
  if (header.colorType === 3 && !palette) throw new ValueError('PNG palette image has no PLTE chunk');
  const total = idat.reduce((sum, chunk) => sum + chunk.length, 0);
  const compressed = new Uint8Array(total);
  let position = 0;
  for (const chunk of idat) {
    compressed.set(chunk, position);
    position += chunk.length;
  }
  const raw = inflateZlib(compressed, true);
  const { width, height, bitDepth, colorType, interlaced } = header;
  const channels = CHANNELS[colorType]!;
  const samples = bitDepth === 16 ? new Uint16Array(width * height * channels) : new Uint8Array(width * height * channels);
  const bitsPerPixel = channels * bitDepth;
  const bpp = Math.max(1, bitsPerPixel >> 3);
  let cursor = 0;
  const pass = (x0: number, y0: number, dx: number, dy: number): void => {
    const passWidth = Math.ceil((width - x0) / dx);
    const passHeight = Math.ceil((height - y0) / dy);
    if (passWidth <= 0 || passHeight <= 0) return;
    const rowBytes = Math.ceil((passWidth * bitsPerPixel) / 8);
    let previous = new Uint8Array(rowBytes);
    let current = new Uint8Array(rowBytes);
    for (let row = 0; row < passHeight; row += 1) {
      if (cursor + 1 + rowBytes > raw.length) throw new ValueError('PNG image data is truncated');
      const filter = raw[cursor]!;
      current.set(raw.subarray(cursor + 1, cursor + 1 + rowBytes));
      cursor += 1 + rowBytes;
      unfilter(filter, current, previous, bpp);
      const y = y0 + row * dy;
      for (let column = 0; column < passWidth; column += 1) {
        const x = x0 + column * dx;
        const base = (y * width + x) * channels;
        for (let channel = 0; channel < channels; channel += 1) {
          samples[base + channel] = readSample(current, column * channels + channel, bitDepth);
        }
      }
      const swap = previous;
      previous = current;
      current = swap;
    }
  };
  if (interlaced) for (const [x0, y0, dx, dy] of ADAM7) pass(x0, y0, dx, dy);
  else pass(0, 0, 1, 1);
  return {
    width, height, bitDepth, colorType, interlaced, channels, samples, palette, transparency,
    exif, exifBeforeData, text, chunks, chunkGamma, significantBits,
  };
}

function latin1(bytes: Uint8Array): string {
  let out = '';
  for (const byte of bytes) out += String.fromCharCode(byte);
  return out;
}

/** Keyword and text of a ``tEXt``/``zTXt``/``iTXt`` chunk. */
function readText(type: string, data: Uint8Array): [string, string] {
  const nul = data.indexOf(0);
  if (nul < 0) throw new ValueError('bad text chunk');
  const keyword = latin1(data.subarray(0, nul));
  if (type === 'tEXt') return [keyword, latin1(data.subarray(nul + 1))];
  if (type === 'zTXt') return [keyword, latin1(inflateZlib(data.subarray(nul + 2)))];
  const compressed = data[nul + 1] === 1;
  let position = nul + 3;
  const languageEnd = data.indexOf(0, position);
  position = languageEnd + 1;
  const translatedEnd = data.indexOf(0, position);
  const payload = data.subarray(translatedEnd + 1);
  return [keyword, new TextDecoder().decode(compressed ? inflateZlib(payload) : payload)];
}

function readSample(row: Uint8Array, index: number, depth: number): number {
  switch (depth) {
    case 8: return row[index]!;
    case 16: return (row[index * 2]! << 8) | row[index * 2 + 1]!;
    default: {
      const bit = index * depth;
      return (row[bit >> 3]! >> (8 - depth - (bit & 7))) & ((1 << depth) - 1);
    }
  }
}

function unfilter(filter: number, row: Uint8Array, previous: Uint8Array, bpp: number): void {
  const length = row.length;
  switch (filter) {
    case 0: return;
    case 1:
      for (let i = bpp; i < length; i += 1) row[i] = (row[i]! + row[i - bpp]!) & 0xff;
      return;
    case 2:
      for (let i = 0; i < length; i += 1) row[i] = (row[i]! + previous[i]!) & 0xff;
      return;
    case 3:
      for (let i = 0; i < length; i += 1) {
        const left = i >= bpp ? row[i - bpp]! : 0;
        row[i] = (row[i]! + ((left + previous[i]!) >> 1)) & 0xff;
      }
      return;
    case 4:
      for (let i = 0; i < length; i += 1) {
        const a = i >= bpp ? row[i - bpp]! : 0;
        const b = previous[i]!;
        const c = i >= bpp ? previous[i - bpp]! : 0;
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        row[i] = (row[i]! + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c)) & 0xff;
      }
      return;
    default:
      throw new ValueError(`invalid PNG filter type ${filter}`);
  }
}
