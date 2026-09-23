/**
 * GIF decoding (GIF87a/GIF89a): logical screen, global and local colour
 * tables, graphic control extensions (disposal, transparency), interlaced
 * frames and LZW image data. Frames are returned uncomposited; torchvision's
 * and Pillow's compositing live with their callers.
 */
import { ValueError } from '../../errors.js';

export interface GifFrame {
  left: number;
  top: number;
  width: number;
  height: number;
  /** Colour indices in display (de-interlaced) order. */
  indices: Uint8Array;
  palette: Uint8Array | null;
  /** Transparent colour index, or -1. */
  transparentIndex: number;
  /** GIF disposal method (0 unspecified, 1 keep, 2 background, 3 previous). */
  disposal: number;
  interlaced: boolean;
}

export interface GifData {
  width: number;
  height: number;
  globalPalette: Uint8Array | null;
  background: number;
  frames: GifFrame[];
}

export function isGif(bytes: Uint8Array): boolean {
  return bytes.length >= 6 && bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x38
    && (bytes[4] === 0x37 || bytes[4] === 0x39) && bytes[5] === 0x61;
}

/** Decode every frame of a GIF file (``limit`` stops after that many frames). */
export function decodeGif(bytes: Uint8Array, limit = Infinity): GifData {
  if (!isGif(bytes)) throw new ValueError('not a GIF file');
  if (bytes.length < 13) throw new ValueError('truncated GIF file');
  const u16 = (offset: number): number => bytes[offset]! | (bytes[offset + 1]! << 8);
  const width = u16(6);
  const height = u16(8);
  const flags = bytes[10]!;
  const background = bytes[11]!;
  let offset = 13;
  let globalPalette: Uint8Array | null = null;
  if (flags & 0x80) {
    const size = 3 << ((flags & 7) + 1);
    globalPalette = bytes.slice(offset, offset + size);
    offset += size;
  }
  const frames: GifFrame[] = [];
  let transparentIndex = -1;
  let disposal = 0;
  const readBlocks = (): Uint8Array => {
    const parts: Uint8Array[] = [];
    let total = 0;
    while (offset < bytes.length) {
      const size = bytes[offset]!;
      offset += 1;
      if (size === 0) break;
      parts.push(bytes.subarray(offset, offset + size));
      total += Math.min(size, bytes.length - offset);
      offset += size;
    }
    const out = new Uint8Array(total);
    let position = 0;
    for (const part of parts) {
      out.set(part, position);
      position += part.length;
    }
    return out;
  };
  while (offset < bytes.length && frames.length < limit) {
    const introducer = bytes[offset]!;
    offset += 1;
    if (introducer === 0x3b) break;
    if (introducer === 0x21) {
      const label = bytes[offset]!;
      offset += 1;
      const data = readBlocks();
      if (label === 0xf9 && data.length >= 4) {
        disposal = (data[0]! >> 2) & 7;
        transparentIndex = data[0]! & 1 ? data[3]! : -1;
      }
      continue;
    }
    if (introducer !== 0x2c) {
      if (frames.length) break;
      throw new ValueError('invalid GIF block');
    }
    if (offset + 9 > bytes.length) throw new ValueError('truncated GIF image descriptor');
    const left = u16(offset);
    const top = u16(offset + 2);
    const frameWidth = u16(offset + 4);
    const frameHeight = u16(offset + 6);
    const frameFlags = bytes[offset + 8]!;
    offset += 9;
    let palette: Uint8Array | null = null;
    if (frameFlags & 0x80) {
      const size = 3 << ((frameFlags & 7) + 1);
      palette = bytes.slice(offset, offset + size);
      offset += size;
    }
    const minCodeSize = bytes[offset]!;
    offset += 1;
    const data = readBlocks();
    const pixels = frameWidth * frameHeight;
    const decoded = lzw(data, minCodeSize, pixels);
    const interlaced = (frameFlags & 0x40) !== 0;
    let indices = decoded;
    if (interlaced) {
      indices = new Uint8Array(pixels);
      let row = 0;
      for (const [start, step] of [[0, 8], [4, 8], [2, 4], [1, 2]] as const) {
        for (let y = start; y < frameHeight; y += step) {
          indices.set(decoded.subarray(row * frameWidth, (row + 1) * frameWidth), y * frameWidth);
          row += 1;
        }
      }
    }
    frames.push({ left, top, width: frameWidth, height: frameHeight, indices, palette, transparentIndex, disposal, interlaced });
    transparentIndex = -1;
    disposal = 0;
  }
  if (!frames.length) throw new ValueError('GIF file contains no image');
  return { width, height, globalPalette, background, frames };
}

/** LZW decompression of GIF image data into ``count`` indices (missing data stays 0). */
function lzw(data: Uint8Array, minCodeSize: number, count: number): Uint8Array {
  const out = new Uint8Array(count);
  if (minCodeSize < 1 || minCodeSize > 11) throw new ValueError('invalid GIF LZW code size');
  const clear = 1 << minCodeSize;
  const end = clear + 1;
  const prefix = new Int32Array(4096);
  const suffix = new Uint8Array(4096);
  const lengths = new Int32Array(4096);
  for (let i = 0; i < clear; i += 1) {
    suffix[i] = i;
    lengths[i] = 1;
  }
  let codeSize = minCodeSize + 1;
  let next = end + 1;
  let previous = -1;
  let bitBuffer = 0;
  let bitCount = 0;
  let position = 0;
  let written = 0;
  const stack = new Uint8Array(4097);
  while (written < count) {
    while (bitCount < codeSize && position < data.length) {
      bitBuffer |= data[position++]! << bitCount;
      bitCount += 8;
    }
    if (bitCount < codeSize) break;
    const code = bitBuffer & ((1 << codeSize) - 1);
    bitBuffer >>>= codeSize;
    bitCount -= codeSize;
    if (code === clear) {
      codeSize = minCodeSize + 1;
      next = end + 1;
      previous = -1;
      continue;
    }
    if (code === end) break;
    let current = code;
    let first: number;
    let length: number;
    if (previous === -1) {
      if (code >= clear) throw new ValueError('invalid GIF LZW code');
      out[written++] = code;
      previous = code;
      continue;
    }
    if (code < next) {
      length = lengths[code]!;
    } else if (code === next) {
      length = lengths[previous]! + 1;
      current = previous;
    } else {
      throw new ValueError('invalid GIF LZW code');
    }
    // Unwind ``current`` into the stack.
    let depth = 0;
    let walk = current;
    while (walk >= clear) {
      stack[depth++] = suffix[walk]!;
      walk = prefix[walk]!;
    }
    stack[depth++] = walk;
    first = walk;
    const emitted: number[] = [];
    for (let i = depth - 1; i >= 0; i -= 1) emitted.push(stack[i]!);
    if (code === next) emitted.push(first);
    void length;
    for (const value of emitted) {
      if (written >= count) break;
      out[written++] = value;
    }
    if (next < 4096) {
      prefix[next] = previous;
      suffix[next] = first;
      lengths[next] = lengths[previous]! + 1;
      next += 1;
      if (next === 1 << codeSize && codeSize < 12) codeSize += 1;
    }
    previous = code;
  }
  return out;
}
