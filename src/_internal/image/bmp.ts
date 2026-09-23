/**
 * BMP decoding with Pillow 12's ``BmpImagePlugin`` semantics: OS/2 and
 * Windows headers, bottom-up and top-down rows, 1/4/8-bit palettes (grayscale
 * palettes open as ``1``/``L``), 16/24/32-bit raw pixels, ``BI_BITFIELDS``
 * layouts Pillow accepts and RLE4/RLE8 compression.
 */
import { ValueError } from '../../errors.js';

export interface BmpData {
  width: number;
  height: number;
  /** Pillow mode: ``1``, ``L``, ``P``, ``RGB`` or ``RGBA``. */
  mode: '1' | 'L' | 'P' | 'RGB' | 'RGBA';
  /** Interleaved samples (``1`` stored as 0/255). */
  data: Uint8Array;
  /** RGB palette triplets for ``P``. */
  palette: Uint8Array | null;
}

export function isBmp(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x42 && bytes[1] === 0x4d;
}

const SUPPORTED_32: Record<string, string> = {
  '16711680,65280,255,0': 'BGRX',
  '4278190080,16711680,65280,0': 'XBGR',
  '4278190080,65280,255,0': 'BGXR',
  '4278190080,16711680,65280,255': 'ABGR',
  '255,65280,16711680,4278190080': 'RGBA',
  '16711680,65280,255,4278190080': 'BGRA',
  '4278190080,65280,255,16711680': 'BGAR',
  '0,0,0,0': 'BGRA',
};

export function decodeBmp(bytes: Uint8Array): BmpData {
  if (!isBmp(bytes) || bytes.length < 14) throw new ValueError('Not a BMP file');
  const u16 = (offset: number): number => bytes[offset]! | (bytes[offset + 1]! << 8);
  const u32 = (offset: number): number => (bytes[offset]! | (bytes[offset + 1]! << 8) | (bytes[offset + 2]! << 16) | (bytes[offset + 3]! << 24)) >>> 0;
  let offset = u32(10);
  const headerSize = u32(14);
  let position = 18;
  if (position + headerSize - 4 > bytes.length) throw new ValueError('Truncated BMP header');
  let width: number;
  let height: number;
  let bits: number;
  let compression = 0;
  let colors = 0;
  let paletteEntrySize: number;
  let direction = -1;
  let masks: number[] | null = null;
  if (headerSize === 12) {
    width = u16(18);
    height = u16(20);
    bits = u16(24);
    paletteEntrySize = 3;
  } else if ([40, 52, 56, 64, 108, 124].includes(headerSize)) {
    const yFlip = bytes[18 + 7] === 0xff;
    direction = yFlip ? 1 : -1;
    width = u32(18);
    height = yFlip ? 2 ** 32 - u32(22) : u32(22);
    bits = u16(28);
    compression = u32(30);
    colors = u32(46);
    paletteEntrySize = 4;
    if (compression === 3) {
      if (headerSize - 4 >= 48) {
        masks = [u32(54), u32(58), u32(62), headerSize - 4 >= 52 ? u32(66) : 0];
      } else {
        const extra = 14 + headerSize;
        masks = [u32(extra), u32(extra + 4), u32(extra + 8), 0];
        position = extra + 12 - (18 + headerSize - 4);
      }
    }
  } else {
    throw new ValueError(`Unsupported BMP header type (${headerSize})`);
  }
  if (!colors) colors = 2 ** bits;
  const headerEnd = 14 + headerSize + (masks && headerSize - 4 < 48 ? 12 : 0);
  if (offset === 14 + headerSize && bits <= 8) offset += paletteEntrySize * colors;
  const bitModes: Record<number, ['P' | 'RGB', string]> = {
    1: ['P', 'P;1'], 4: ['P', 'P;4'], 8: ['P', 'P'], 16: ['RGB', 'BGR;15'], 24: ['RGB', 'BGR'], 32: ['RGB', 'BGRX'],
  };
  const entry = bitModes[bits];
  if (!entry) throw new ValueError(`Unsupported BMP pixel depth (${bits})`);
  let mode: BmpData['mode'] = entry[0];
  let rawmode = entry[1];
  let rle = false;
  if (compression === 3) {
    const key = masks!.map((value) => value >>> 0);
    if (bits === 32 && SUPPORTED_32[key.join(',')]) {
      rawmode = SUPPORTED_32[key.join(',')]!;
      if (rawmode.includes('A')) mode = 'RGBA';
    } else if (bits === 24 && key.slice(0, 3).join(',') === '16711680,65280,255') {
      rawmode = 'BGR';
    } else if (bits === 16 && key.slice(0, 3).join(',') === '63488,2016,31') {
      rawmode = 'BGR;16';
    } else if (bits === 16 && key.slice(0, 3).join(',') === '31744,992,31') {
      rawmode = 'BGR;15';
    } else {
      throw new ValueError('Unsupported BMP bitfields layout');
    }
  } else if (compression === 1 || compression === 2) {
    rle = true;
  } else if (compression !== 0) {
    throw new ValueError(`Unsupported BMP compression (${compression})`);
  }
  let palette: Uint8Array | null = null;
  if (mode === 'P') {
    if (!(colors > 0 && colors <= 65536)) throw new ValueError(`Unsupported BMP Palette size (${colors})`);
    const paletteStart = headerEnd;
    const raw = bytes.subarray(paletteStart, paletteStart + paletteEntrySize * colors);
    let grayscale = true;
    const indices = colors === 2 ? [0, 255] : Array.from({ length: colors }, (_, index) => index);
    indices.forEach((value, index) => {
      const base = index * paletteEntrySize;
      if (raw[base] !== value || raw[base + 1] !== value || raw[base + 2] !== value) grayscale = false;
    });
    if (grayscale) {
      mode = colors === 2 ? '1' : 'L';
      rawmode = mode;
    } else {
      palette = new Uint8Array(256 * 3);
      for (let index = 0; index < Math.min(colors, 256); index += 1) {
        const base = index * paletteEntrySize;
        if (base + 2 >= raw.length) break;
        palette[index * 3] = raw[base + 2]!;
        palette[index * 3 + 1] = raw[base + 1]!;
        palette[index * 3 + 2] = raw[base]!;
      }
    }
  }
  void position;
  const channels = mode === 'RGB' ? 3 : mode === 'RGBA' ? 4 : 1;
  const out = new Uint8Array(width * height * channels);
  const rowOrder = (row: number): number => (direction === -1 ? height - 1 - row : row);
  if (rle) {
    const rle4 = compression === 2;
    const data: number[] = [];
    let x = 0;
    let p = offset;
    const total = width * height;
    while (data.length < total) {
      if (p + 1 >= bytes.length) break;
      let count = bytes[p]!;
      const byte = bytes[p + 1]!;
      p += 2;
      if (count) {
        if (x + count > width) count = Math.max(0, width - x);
        for (let i = 0; i < count; i += 1) data.push(rle4 ? (i % 2 === 0 ? byte >> 4 : byte & 0x0f) : byte);
        x += count;
      } else if (byte === 0) {
        while (data.length % width !== 0) data.push(0);
        x = 0;
      } else if (byte === 1) {
        break;
      } else if (byte === 2) {
        if (p + 1 >= bytes.length) break;
        const right = bytes[p]!;
        const up = bytes[p + 1]!;
        p += 2;
        for (let i = 0; i < right + up * width; i += 1) data.push(0);
        x = data.length % width;
      } else {
        const byteCount = rle4 ? byte >> 1 : byte;
        const available = Math.min(byteCount, bytes.length - p);
        for (let i = 0; i < available; i += 1) {
          const value = bytes[p + i]!;
          if (rle4) data.push(value >> 4, value & 0x0f);
          else data.push(value);
        }
        p += available;
        if (available < byteCount) break;
        x += byte;
        if ((p - 0) % 2 !== 0) p += 1;
      }
    }
    // Pillow's set_as_raw: rows in file order, flipped for bottom-up images.
    for (let row = 0; row < height; row += 1) {
      const target = rowOrder(row);
      for (let column = 0; column < width; column += 1) {
        const index = row * width + column;
        out[target * width + column] = index < data.length ? data[index]! & 0xff : 0;
      }
    }
    if (mode === '1') for (let i = 0; i < out.length; i += 1) out[i] = out[i] ? 255 : 0;
    return { width, height, mode, data: out, palette };
  }
  const stride = (((width * bits + 31) >> 3) & ~3);
  for (let row = 0; row < height; row += 1) {
    const source = offset + row * stride;
    const target = rowOrder(row) * width * channels;
    for (let column = 0; column < width; column += 1) {
      const o = target + column * channels;
      const read = (index: number): number => (index < bytes.length ? bytes[index]! : 0);
      switch (rawmode) {
        case 'P;1':
        case '1': {
          const bit = (read(source + (column >> 3)) >> (7 - (column & 7))) & 1;
          out[o] = rawmode === '1' ? (bit ? 255 : 0) : bit;
          break;
        }
        case 'P;4': out[o] = (read(source + (column >> 1)) >> (column & 1 ? 0 : 4)) & 0x0f; break;
        case 'P':
        case 'L': out[o] = read(source + column); break;
        case 'BGR;15':
        case 'BGR;16': {
          const pixel = read(source + column * 2) | (read(source + column * 2 + 1) << 8);
          const five = (value: number): number => Math.trunc((value * 255) / 31);
          out[o + 2] = five(pixel & 31);
          if (rawmode === 'BGR;15') {
            out[o + 1] = five((pixel >> 5) & 31);
            out[o] = five((pixel >> 10) & 31);
          } else {
            out[o + 1] = Math.trunc((((pixel >> 5) & 63) * 255) / 63);
            out[o] = five((pixel >> 11) & 31);
          }
          break;
        }
        case 'BGR': {
          const s = source + column * 3;
          out[o] = read(s + 2); out[o + 1] = read(s + 1); out[o + 2] = read(s);
          break;
        }
        default: {
          // 32-bit layouts: channel order of the four stored bytes.
          const s = source + column * 4;
          const order = rawmode;
          const byteAt = (letter: string): number => read(s + order.indexOf(letter));
          out[o] = byteAt('R');
          out[o + 1] = byteAt('G');
          out[o + 2] = byteAt('B');
          if (channels === 4) out[o + 3] = byteAt('A');
          break;
        }
      }
    }
  }
  return { width, height, mode, data: out, palette };
}
