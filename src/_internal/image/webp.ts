/**
 * WebP decoding ported from libwebp 1.6 (the decoder behind Pillow and
 * torchvision): the RIFF container (``VP8``, ``VP8L``, ``VP8X`` with ``ALPH``
 * and ``ANIM``/``ANMF``), the VP8 lossy decoder (boolean entropy decoder,
 * intra prediction, inverse transforms, simple and complex loop filters), the
 * VP8L lossless decoder (canonical Huffman codes, colour cache, backward
 * references and the predictor / cross-colour / subtract-green /
 * colour-indexing transforms), alpha planes (raw or lossless, with
 * horizontal / vertical / gradient unfiltering) and ``WebPDecodeRGB[A]``'s
 * fancy YUV 4:2:0 upsampling with libwebp's fixed-point colour conversion.
 */
import { ValueError } from '../../errors.js';
import { AC_TABLE, BMODES_PROBA, COEFFS_PROBA0, COEFFS_UPDATE_PROBA, DC_TABLE } from './webpTables.js';

export interface WebpFrame {
  /** Frame rectangle on the canvas. */
  x: number;
  y: number;
  width: number;
  height: number;
  /** Decoded RGBA pixels of the frame (``WebPDecodeRGBA``). */
  rgba: Uint8Array;
}

export interface WebpData {
  width: number;
  height: number;
  /** ``WebPGetFeatures`` ``has_alpha``. */
  hasAlpha: boolean;
  /** ``WebPGetFeatures`` ``has_animation``. */
  animated: boolean;
  /** Lossless (``VP8L``) bitstream. */
  lossless: boolean;
  frameCount: number;
  /** ``EXIF`` chunk payload of an extended file (Pillow ``info['exif']``). */
  exif: Uint8Array | null;
  /** ``XMP `` chunk payload of an extended file (Pillow ``info['xmp']``). */
  xmp: Uint8Array | null;
  /** ``WebPDecodeRGBA`` of a still image (throws for animations). */
  rgba(): Uint8Array;
  /** ``WebPAnimDecoder``'s first canvas (RGBA, transparent black outside the frame). */
  firstCanvas(): Uint8Array;
}

function tag(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(bytes[offset]!, bytes[offset + 1]!, bytes[offset + 2]!, bytes[offset + 3]!);
}

function le32(bytes: Uint8Array, offset: number): number {
  return (bytes[offset]! | (bytes[offset + 1]! << 8) | (bytes[offset + 2]! << 16) | (bytes[offset + 3]! << 24)) >>> 0;
}

function le24(bytes: Uint8Array, offset: number): number {
  return bytes[offset]! | (bytes[offset + 1]! << 8) | (bytes[offset + 2]! << 16);
}

export function isWebp(bytes: Uint8Array): boolean {
  return bytes.length >= 12 && tag(bytes, 0) === 'RIFF' && tag(bytes, 8) === 'WEBP';
}

interface Bitstream {
  lossless: boolean;
  data: Uint8Array;
  alpha: Uint8Array | null;
}

/** ``ParseHeadersInternal`` for one still image or frame payload. */
function parseStill(bytes: Uint8Array, start: number, end: number, riffSize: number): Bitstream {
  let offset = start;
  let alpha: Uint8Array | null = null;
  while (offset + 8 <= end) {
    const name = tag(bytes, offset);
    const size = le32(bytes, offset + 4);
    if (name === 'VP8 ' || name === 'VP8L') {
      if (riffSize && size > riffSize) throw new ValueError('Inconsistent size information');
      if (offset + 8 + size > end) throw new ValueError('Truncated bitstream');
      return { lossless: name === 'VP8L', data: bytes.subarray(offset + 8, offset + 8 + size), alpha };
    }
    const disk = (8 + size + 1) & ~1;
    if (offset + disk > end) throw new ValueError('Truncated chunk');
    if (name === 'ALPH') alpha = bytes.subarray(offset + 8, offset + 8 + size);
    offset += disk;
  }
  throw new ValueError('No VP8/VP8L chunk found');
}

/** Parse a WebP file (``WebPGetFeatures`` + demuxing). */
export function decodeWebp(bytes: Uint8Array): WebpData {
  if (!isWebp(bytes)) throw new ValueError('Not a WebP file');
  const riffSize = le32(bytes, 4);
  if (riffSize < 12) throw new ValueError('Bad RIFF size');
  if (riffSize > bytes.length - 8) throw new ValueError('Truncated RIFF file');
  const end = Math.min(bytes.length, riffSize + 8);
  let offset = 12;
  let vp8x = false;
  let flags = 0;
  let canvasWidth = 0;
  let canvasHeight = 0;
  if (tag(bytes, offset) === 'VP8X') {
    if (le32(bytes, offset + 4) !== 10) throw new ValueError('Wrong VP8X chunk size');
    flags = le32(bytes, offset + 8);
    canvasWidth = 1 + le24(bytes, offset + 12);
    canvasHeight = 1 + le24(bytes, offset + 15);
    vp8x = true;
    offset += 18;
  }
  const animated = vp8x && (flags & 0x02) !== 0;
  let exif: Uint8Array | null = null;
  let xmp: Uint8Array | null = null;
  if (vp8x) {
    for (let position = offset; position + 8 <= end;) {
      const name = tag(bytes, position);
      const size = le32(bytes, position + 4);
      if (name === 'EXIF' && !exif) exif = bytes.slice(position + 8, Math.min(position + 8 + size, end));
      if (name === 'XMP ' && !xmp) xmp = bytes.slice(position + 8, Math.min(position + 8 + size, end));
      position += (8 + size + 1) & ~1;
    }
  }
  if (animated) {
    const frames: { x: number; y: number; width: number; height: number; stream: Bitstream }[] = [];
    let position = offset;
    while (position + 8 <= end) {
      const name = tag(bytes, position);
      const size = le32(bytes, position + 4);
      const disk = (8 + size + 1) & ~1;
      if (name === 'ANMF' && size >= 16) {
        const base = position + 8;
        frames.push({
          x: le24(bytes, base) * 2,
          y: le24(bytes, base + 3) * 2,
          width: 1 + le24(bytes, base + 6),
          height: 1 + le24(bytes, base + 9),
          stream: parseStill(bytes, base + 16, Math.min(base + size, end), 0),
        });
      }
      position += disk;
    }
    if (!frames.length) throw new ValueError('Animated WebP has no frames');
    const hasAlpha = (flags & 0x10) !== 0;
    return {
      width: canvasWidth, height: canvasHeight, hasAlpha, animated: true, lossless: frames[0]!.stream.lossless,
      frameCount: frames.length, exif, xmp,
      rgba() {
        throw new ValueError('Animated webp files are not supported.');
      },
      firstCanvas() {
        const frame = frames[0]!;
        const decoded = decodeBitstream(frame.stream);
        const canvas = new Uint8Array(canvasWidth * canvasHeight * 4);
        for (let y = 0; y < decoded.height && frame.y + y < canvasHeight; y += 1) {
          for (let x = 0; x < decoded.width && frame.x + x < canvasWidth; x += 1) {
            const target = ((frame.y + y) * canvasWidth + frame.x + x) * 4;
            const source = (y * decoded.width + x) * 4;
            for (let c = 0; c < 4; c += 1) canvas[target + c] = decoded.rgba[source + c]!;
          }
        }
        return canvas;
      },
    };
  }
  const stream = parseStill(bytes, offset, end, riffSize);
  let width: number;
  let height: number;
  let hasAlpha = vp8x ? (flags & 0x10) !== 0 : false;
  if (stream.lossless) {
    const info = vp8lInfo(stream.data);
    width = info.width;
    height = info.height;
    hasAlpha = hasAlpha || info.hasAlpha;
  } else {
    const info = vp8Info(stream.data);
    width = info.width;
    height = info.height;
  }
  if (stream.alpha) hasAlpha = true;
  if (vp8x && (canvasWidth !== width || canvasHeight !== height)) throw new ValueError('VP8X canvas does not match the image');
  let cached: { rgba: Uint8Array } | null = null;
  const decode = (): Uint8Array => (cached ??= decodeBitstream(stream)).rgba;
  return {
    width, height, hasAlpha, animated: false, lossless: stream.lossless, frameCount: 1, exif, xmp,
    rgba: decode,
    firstCanvas: decode,
  };
}

function decodeBitstream(stream: Bitstream): { width: number; height: number; rgba: Uint8Array } {
  if (stream.lossless) {
    const { width, height, argb } = decodeVp8l(stream.data);
    const rgba = new Uint8Array(width * height * 4);
    for (let i = 0; i < width * height; i += 1) {
      const pixel = argb[i]!;
      rgba[i * 4] = (pixel >>> 16) & 0xff;
      rgba[i * 4 + 1] = (pixel >>> 8) & 0xff;
      rgba[i * 4 + 2] = pixel & 0xff;
      rgba[i * 4 + 3] = pixel >>> 24;
    }
    return { width, height, rgba };
  }
  return decodeVp8(stream.data, stream.alpha);
}

// ---------------------------------------------------------------------------
// VP8L (lossless).
// ---------------------------------------------------------------------------

class LosslessReader {
  private position = 0;
  private bitPosition = 0;
  eos = false;
  constructor(private readonly data: Uint8Array) {}

  readBits(count: number): number {
    let value = 0;
    for (let i = 0; i < count; i += 1) value |= this.readBit() << i;
    return value >>> 0;
  }

  readBit(): number {
    if (this.position >= this.data.length) {
      this.eos = true;
      return 0;
    }
    const bit = (this.data[this.position]! >> this.bitPosition) & 1;
    this.bitPosition += 1;
    if (this.bitPosition === 8) {
      this.bitPosition = 0;
      this.position += 1;
    }
    return bit;
  }

  /** Peek up to 15 bits (zeros past the end). */
  peek(count: number): number {
    let value = 0;
    let position = this.position;
    let bit = this.bitPosition;
    for (let i = 0; i < count; i += 1) {
      const byte = position < this.data.length ? this.data[position]! : 0;
      value |= ((byte >> bit) & 1) << i;
      bit += 1;
      if (bit === 8) {
        bit = 0;
        position += 1;
      }
    }
    return value;
  }

  skip(count: number): void {
    const total = this.bitPosition + count;
    this.position += total >> 3;
    this.bitPosition = total & 7;
    if (this.position > this.data.length || (this.position === this.data.length && this.bitPosition > 0)) this.eos = true;
  }
}

/** A canonical Huffman code over LSB-first bits (``VP8LBuildHuffmanTable``). */
interface LosslessHuffman {
  /** ``(symbol << 4) | length`` indexed by the next ``bits`` bits; ``bits`` 0 for single-symbol codes. */
  table: Int32Array;
  bits: number;
}

function buildLosslessHuffman(lengths: ArrayLike<number>, count: number): LosslessHuffman {
  const histogram = new Int32Array(16);
  let used = 0;
  let last = -1;
  for (let symbol = 0; symbol < count; symbol += 1) {
    const length = lengths[symbol]!;
    if (length > 15) throw new ValueError('invalid VP8L Huffman code');
    histogram[length]! += 1;
    if (length) {
      used += 1;
      last = symbol;
    }
  }
  if (!used) throw new ValueError('invalid VP8L Huffman code');
  if (used === 1) {
    const table = new Int32Array(1);
    table[0] = last << 4;
    return { table, bits: 0 };
  }
  // Completeness (libwebp rejects incomplete and over-subscribed codes).
  let open = 1;
  let maxBits = 0;
  for (let length = 1; length <= 15; length += 1) {
    open = open * 2 - histogram[length]!;
    if (open < 0) throw new ValueError('invalid VP8L Huffman code');
    if (histogram[length]) maxBits = length;
  }
  if (open !== 0) throw new ValueError('invalid VP8L Huffman code');
  const table = new Int32Array(1 << maxBits);
  // Canonical codes (DEFLATE ordering), stored bit-reversed for LSB-first lookup.
  const nextCode = new Int32Array(16);
  let code = 0;
  for (let length = 1; length <= 15; length += 1) {
    code = (code + (length > 1 ? histogram[length - 1]! : 0)) << 1;
    nextCode[length] = code;
  }
  for (let symbol = 0; symbol < count; symbol += 1) {
    const length = lengths[symbol]!;
    if (!length) continue;
    let value = nextCode[length]!;
    nextCode[length] = value + 1;
    let reversed = 0;
    for (let i = 0; i < length; i += 1) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    for (let fill = reversed; fill < table.length; fill += 1 << length) table[fill] = (symbol << 4) | length;
  }
  return { table, bits: maxBits };
}

function readSymbol(huffman: LosslessHuffman, reader: LosslessReader): number {
  if (huffman.bits === 0) return huffman.table[0]! >> 4;
  const entry = huffman.table[reader.peek(huffman.bits)]!;
  reader.skip(entry & 15);
  return entry >> 4;
}

const CODE_LENGTH_ORDER = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const CODE_TO_PLANE = [
  0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a, 0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a,
  0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b, 0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03,
  0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c, 0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
  0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b, 0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
  0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b, 0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41,
  0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f, 0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70,
];

function readHuffmanCode(reader: LosslessReader, alphabet: number): LosslessHuffman {
  const lengths = new Uint8Array(alphabet);
  if (reader.readBits(1)) {
    const symbols = reader.readBits(1) + 1;
    const firstBits = reader.readBits(1) ? 8 : 1;
    lengths[reader.readBits(firstBits)] = 1;
    if (symbols === 2) lengths[reader.readBits(8)] = 1;
  } else {
    const codeLengthLengths = new Uint8Array(19);
    const count = reader.readBits(4) + 4;
    for (let i = 0; i < count; i += 1) codeLengthLengths[CODE_LENGTH_ORDER[i]!] = reader.readBits(3);
    const lengthCode = buildLosslessHuffman(codeLengthLengths, 19);
    let maxSymbol = alphabet;
    if (reader.readBits(1)) {
      const lengthBits = 2 + 2 * reader.readBits(3);
      maxSymbol = 2 + reader.readBits(lengthBits);
      if (maxSymbol > alphabet) throw new ValueError('invalid VP8L code lengths');
    }
    let previous = 8;
    let symbol = 0;
    while (symbol < alphabet) {
      if (maxSymbol-- === 0) break;
      const code = readSymbol(lengthCode, reader);
      if (code < 16) {
        lengths[symbol++] = code;
        if (code) previous = code;
      } else {
        const slot = code - 16;
        const repeat = reader.readBits([2, 3, 7][slot]!) + [3, 3, 11][slot]!;
        if (symbol + repeat > alphabet) throw new ValueError('invalid VP8L code lengths');
        const value = code === 16 ? previous : 0;
        for (let i = 0; i < repeat; i += 1) lengths[symbol++] = value;
      }
    }
  }
  if (reader.eos) throw new ValueError('truncated VP8L bitstream');
  return buildLosslessHuffman(lengths, alphabet);
}

function subSample(size: number, bits: number): number {
  return (size + (1 << bits) - 1) >> bits;
}

interface Transform {
  type: number;
  bits: number;
  xsize: number;
  data: Uint32Array;
}

function vp8lInfo(data: Uint8Array): { width: number; height: number; hasAlpha: boolean } {
  if (data.length < 5 || data[0] !== 0x2f || data[4]! >> 5 !== 0) throw new ValueError('bad VP8L signature');
  const reader = new LosslessReader(data);
  reader.readBits(8);
  const width = reader.readBits(14) + 1;
  const height = reader.readBits(14) + 1;
  const hasAlpha = reader.readBits(1) === 1;
  return { width, height, hasAlpha };
}

/** Decode a VP8L bitstream into ARGB pixels. */
export function decodeVp8l(data: Uint8Array): { width: number; height: number; argb: Uint32Array } {
  const { width, height } = vp8lInfo(data);
  const reader = new LosslessReader(data);
  reader.readBits(8 + 14 + 14 + 1 + 3);
  const argb = decodeLosslessImage(reader, width, height, true);
  return { width, height, argb };
}

/** Decode the lossless image stream of an ``ALPH`` chunk (``VP8LDecodeAlphaHeader``). */
function decodeLosslessAlpha(data: Uint8Array, width: number, height: number): Uint8Array {
  const reader = new LosslessReader(data);
  const argb = decodeLosslessImage(reader, width, height, true);
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i += 1) out[i] = (argb[i]! >>> 8) & 0xff;
  return out;
}

function decodeLosslessImage(reader: LosslessReader, xsize: number, ysize: number, level0: boolean): Uint32Array {
  let transformXsize = xsize;
  const transforms: Transform[] = [];
  if (level0) {
    let seen = 0;
    while (reader.readBits(1)) {
      const type = reader.readBits(2);
      if (seen & (1 << type)) throw new ValueError('repeated VP8L transform');
      seen |= 1 << type;
      const transform: Transform = { type, bits: 0, xsize: transformXsize, data: new Uint32Array(0) };
      if (type === 0 || type === 1) {
        transform.bits = 2 + reader.readBits(3);
        transform.data = decodeLosslessImage(reader, subSample(transformXsize, transform.bits), subSample(ysize, transform.bits), false);
      } else if (type === 3) {
        const colors = reader.readBits(8) + 1;
        const bits = colors > 16 ? 0 : colors > 4 ? 1 : colors > 2 ? 2 : 3;
        transformXsize = subSample(transform.xsize, bits);
        transform.bits = bits;
        const palette = decodeLosslessImage(reader, colors, 1, false);
        const expanded = new Uint32Array(1 << (8 >> bits));
        const bytes = new Uint8Array(expanded.buffer);
        const source = new Uint8Array(palette.buffer, palette.byteOffset, palette.byteLength);
        bytes.set(source.subarray(0, 4));
        for (let i = 4; i < 4 * colors; i += 1) bytes[i] = (source[i]! + bytes[i - 4]!) & 0xff;
        transform.data = expanded;
      }
      transforms.push(transform);
    }
  }
  let cacheBits = 0;
  if (reader.readBits(1)) {
    cacheBits = reader.readBits(4);
    if (cacheBits < 1 || cacheBits > 11) throw new ValueError('invalid VP8L colour cache size');
  }
  // Meta Huffman codes.
  let huffmanBits = 0;
  let huffmanImage: Uint32Array | null = null;
  let groupCount = 1;
  if (level0 && reader.readBits(1)) {
    huffmanBits = 2 + reader.readBits(3);
    const image = decodeLosslessImage(reader, subSample(transformXsize, huffmanBits), subSample(ysize, huffmanBits), false);
    huffmanImage = new Uint32Array(image.length);
    for (let i = 0; i < image.length; i += 1) {
      const group = (image[i]! >>> 8) & 0xffff;
      huffmanImage[i] = group;
      groupCount = Math.max(groupCount, group + 1);
    }
  }
  if (reader.eos) throw new ValueError('truncated VP8L bitstream');
  const cacheSize = cacheBits ? 1 << cacheBits : 0;
  const groups: LosslessHuffman[][] = [];
  for (let g = 0; g < groupCount; g += 1) {
    groups.push([
      readHuffmanCode(reader, 256 + 24 + cacheSize),
      readHuffmanCode(reader, 256),
      readHuffmanCode(reader, 256),
      readHuffmanCode(reader, 256),
      readHuffmanCode(reader, 40),
    ]);
  }
  const width = transformXsize;
  const total = width * ysize;
  const data = new Uint32Array(total);
  const cache = cacheSize ? new Uint32Array(cacheSize) : null;
  const cacheShift = 32 - cacheBits;
  const huffmanXsize = huffmanBits ? subSample(width, huffmanBits) : 0;
  let cached = 0;
  const insert = (): void => {
    if (!cache) return;
    while (cached < position) {
      const pixel = data[cached++]!;
      cache[Math.imul(pixel, 0x1e35a7bd) >>> cacheShift] = pixel;
    }
  };
  let position = 0;
  while (position < total) {
    const x = position % width;
    const y = (position - x) / width;
    const group = groups[huffmanImage ? huffmanImage[(y >> huffmanBits) * huffmanXsize + (x >> huffmanBits)]! : 0]!;
    const code = readSymbol(group[0]!, reader);
    if (code < 256) {
      const red = readSymbol(group[1]!, reader);
      const blue = readSymbol(group[2]!, reader);
      const alpha = readSymbol(group[3]!, reader);
      if (reader.eos) break;
      data[position++] = ((alpha << 24) | (red << 16) | (code << 8) | blue) >>> 0;
    } else if (code < 256 + 24) {
      const length = copyDistance(code - 256, reader);
      const distanceSymbol = readSymbol(group[4]!, reader);
      const distanceCode = copyDistance(distanceSymbol, reader);
      let distance: number;
      if (distanceCode > 120) distance = distanceCode - 120;
      else {
        const planeCode = CODE_TO_PLANE[distanceCode - 1]!;
        distance = Math.max((planeCode >> 4) * width + (8 - (planeCode & 0xf)), 1);
      }
      if (reader.eos) break;
      if (position < distance || total - position < length) throw new ValueError('invalid VP8L backward reference');
      for (let i = 0; i < length; i += 1) {
        data[position] = data[position - distance]!;
        position += 1;
      }
    } else {
      const key = code - 256 - 24;
      if (!cache || key >= cacheSize) throw new ValueError('invalid VP8L colour cache index');
      insert();
      data[position++] = cache[key]!;
    }
    insert();
  }
  if (reader.eos || position < total) throw new ValueError('truncated VP8L bitstream');
  // Inverse transforms, last read first.
  let pixels: Uint32Array = data;
  for (let index = transforms.length - 1; index >= 0; index -= 1) pixels = inverseTransform(transforms[index]!, pixels, ysize);
  return pixels;
}

function copyDistance(symbol: number, reader: LosslessReader): number {
  if (symbol < 4) return symbol + 1;
  const extra = (symbol - 2) >> 1;
  const offset = (2 + (symbol & 1)) << extra;
  return offset + reader.readBits(extra) + 1;
}

function addPixels(a: number, b: number): number {
  const ag = ((a & 0xff00ff00) >>> 0) + ((b & 0xff00ff00) >>> 0);
  const rb = (a & 0x00ff00ff) + (b & 0x00ff00ff);
  return (((ag & 0xff00ff00) >>> 0) | (rb & 0x00ff00ff)) >>> 0;
}

function average2(a: number, b: number): number {
  return ((((a ^ b) & 0xfefefefe) >>> 1) + ((a & b) >>> 0)) >>> 0;
}

function clip255(value: number): number {
  return value < 0 ? 0 : value > 255 ? 255 : value;
}

function channel(value: number, shift: number): number {
  return (value >>> shift) & 0xff;
}

function select(a: number, b: number, c: number): number {
  let paMinusPb = 0;
  for (const shift of [24, 16, 8, 0]) {
    const pa = channel(a, shift) - channel(c, shift);
    const pb = channel(b, shift) - channel(c, shift);
    paMinusPb += Math.abs(pb) - Math.abs(pa);
  }
  return paMinusPb <= 0 ? a : b;
}

function clampedAddSubtractFull(c0: number, c1: number, c2: number): number {
  let out = 0;
  for (const shift of [24, 16, 8, 0]) out = out * 256 + clip255(channel(c0, shift) + channel(c1, shift) - channel(c2, shift));
  return out >>> 0;
}

function clampedAddSubtractHalf(c0: number, c1: number, c2: number): number {
  const average = average2(c0, c1);
  let out = 0;
  for (const shift of [24, 16, 8, 0]) {
    const a = channel(average, shift);
    const b = channel(c2, shift);
    out = out * 256 + clip255(a + Math.trunc((a - b) / 2));
  }
  return out >>> 0;
}

function predict(mode: number, left: number, top: number, topLeft: number, topRight: number): number {
  switch (mode) {
    case 0: return 0xff000000;
    case 1: return left;
    case 2: return top;
    case 3: return topRight;
    case 4: return topLeft;
    case 5: return average2(average2(left, topRight), top);
    case 6: return average2(left, topLeft);
    case 7: return average2(left, top);
    case 8: return average2(topLeft, top);
    case 9: return average2(top, topRight);
    case 10: return average2(average2(left, topLeft), average2(top, topRight));
    case 11: return select(top, left, topLeft);
    case 12: return clampedAddSubtractFull(left, top, topLeft);
    case 13: return clampedAddSubtractHalf(left, top, topLeft);
    default: return 0xff000000; // Modes 14 and 15: libwebp's padding sentinels (predictor 0).
  }
}

function colorTransformDelta(prediction: number, color: number): number {
  const p = (prediction << 24) >> 24;
  const c = (color << 24) >> 24;
  return (p * c) >> 5;
}

function inverseTransform(transform: Transform, input: Uint32Array, ysize: number): Uint32Array {
  const width = transform.xsize;
  switch (transform.type) {
    case 2: { // subtract green
      const out = new Uint32Array(input.length);
      for (let i = 0; i < input.length; i += 1) {
        const argb = input[i]!;
        const green = (argb >>> 8) & 0xff;
        let redBlue = argb & 0x00ff00ff;
        redBlue = (redBlue + ((green << 16) | green)) & 0x00ff00ff;
        out[i] = (((argb & 0xff00ff00) >>> 0) | redBlue) >>> 0;
      }
      return out;
    }
    case 0: { // predictor
      const out = new Uint32Array(width * ysize);
      const tiles = subSample(width, transform.bits);
      for (let y = 0; y < ysize; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const index = y * width + x;
          let prediction: number;
          if (y === 0) prediction = x === 0 ? 0xff000000 : out[index - 1]!;
          else if (x === 0) prediction = out[index - width]!;
          else {
            const mode = (transform.data[(y >> transform.bits) * tiles + (x >> transform.bits)]! >>> 8) & 0xf;
            prediction = predict(mode, out[index - 1]!, out[index - width]!, out[index - width - 1]!, out[index - width + 1]!);
          }
          out[index] = addPixels(input[index]!, prediction);
        }
      }
      return out;
    }
    case 1: { // cross colour
      const out = new Uint32Array(width * ysize);
      const tiles = subSample(width, transform.bits);
      for (let y = 0; y < ysize; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const code = transform.data[(y >> transform.bits) * tiles + (x >> transform.bits)]!;
          const greenToRed = code & 0xff;
          const greenToBlue = (code >>> 8) & 0xff;
          const redToBlue = (code >>> 16) & 0xff;
          const argb = input[y * width + x]!;
          const green = (argb >>> 8) & 0xff;
          let red = (argb >>> 16) & 0xff;
          let blue = argb & 0xff;
          red = (red + colorTransformDelta(greenToRed, green)) & 0xff;
          blue = (blue + colorTransformDelta(greenToBlue, green) + colorTransformDelta(redToBlue, red)) & 0xff;
          out[y * width + x] = (((argb & 0xff00ff00) >>> 0) | (red << 16) | blue) >>> 0;
        }
      }
      return out;
    }
    default: { // colour indexing
      const bitsPerPixel = 8 >> transform.bits;
      const packedWidth = subSample(width, transform.bits);
      const out = new Uint32Array(width * ysize);
      const mask = (1 << bitsPerPixel) - 1;
      const countMask = (1 << transform.bits) - 1;
      for (let y = 0; y < ysize; y += 1) {
        let packed = 0;
        for (let x = 0; x < width; x += 1) {
          if ((x & countMask) === 0) packed = (input[y * packedWidth + (x >> transform.bits)]! >>> 8) & 0xff;
          out[y * width + x] = transform.data[packed & mask]!;
          packed >>= bitsPerPixel;
        }
      }
      return out;
    }
  }
}

// ---------------------------------------------------------------------------
// VP8 (lossy).
// ---------------------------------------------------------------------------

/** libwebp's ``VP8BitReader`` (boolean entropy decoder). */
class BoolReader {
  private value = 0;
  private range = 254;
  private bits = -8;
  eof = false;
  constructor(private readonly data: Uint8Array, private position: number, private readonly end: number) {
    this.load();
  }

  private load(): void {
    if (this.position < this.end) {
      this.bits += 8;
      this.value = this.data[this.position++]! | (this.value * 256);
    } else if (!this.eof) {
      this.value *= 256;
      this.bits += 8;
      this.eof = true;
    } else {
      this.bits = 0;
    }
  }

  getBit(probability: number): number {
    let range = this.range;
    if (this.bits < 0) this.load();
    const position = this.bits;
    const split = (range * probability) >>> 8;
    const value = Math.floor(this.value / 2 ** position);
    const bit = value > split ? 1 : 0;
    if (bit) {
      range -= split;
      this.value -= (split + 1) * 2 ** position;
    } else {
      range = split + 1;
    }
    const shift = 7 ^ (31 - Math.clz32(range));
    range <<= shift;
    this.bits -= shift;
    this.range = range - 1;
    return bit;
  }

  getValue(bits: number): number {
    let value = 0;
    while (bits-- > 0) value |= this.getBit(0x80) << bits;
    return value;
  }

  getSignedValue(bits: number): number {
    const value = this.getValue(bits);
    return this.getBit(0x80) ? -value : value;
  }

  getSigned(v: number): number {
    if (this.bits < 0) this.load();
    const position = this.bits;
    const split = this.range >>> 1;
    const value = Math.floor(this.value / 2 ** position);
    const mask = (split - value) >> 31; // -1 or 0
    this.bits -= 1;
    this.range = ((this.range + mask) | 1) >>> 0;
    this.value -= ((split + 1) & mask) * 2 ** position;
    return (v ^ mask) - mask;
  }
}

function vp8Info(data: Uint8Array): { width: number; height: number } {
  if (data.length < 10) throw new ValueError('Truncated VP8 header');
  const bits = data[0]! | (data[1]! << 8) | (data[2]! << 16);
  const keyFrame = !(bits & 1);
  if (!keyFrame) throw new ValueError('Not a key frame.');
  if (((bits >> 1) & 7) > 3) throw new ValueError('Incorrect keyframe parameters.');
  if (!((bits >> 4) & 1)) throw new ValueError('Frame not displayable.');
  if (data[3] !== 0x9d || data[4] !== 0x01 || data[5] !== 0x2a) throw new ValueError('Bad code word');
  const width = (data[6]! | (data[7]! << 8)) & 0x3fff;
  const height = (data[8]! | (data[9]! << 8)) & 0x3fff;
  if (!width || !height) throw new ValueError('Invalid VP8 dimensions');
  return { width, height };
}

const ZIGZAG = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const BANDS = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];
const CAT3456 = [[173, 148, 140], [176, 155, 140, 135], [180, 157, 141, 134, 130], [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129]];

function clip8(value: number): number {
  return value < 0 ? 0 : value > 255 ? 255 : value;
}

const mul1 = (a: number): number => ((a * 20091) >> 16) + a;
const mul2 = (a: number): number => (a * 35468) >> 16;

interface Macroblock {
  segment: number;
  skip: number;
  isI4x4: boolean;
  imodes: Uint8Array;
  uvmode: number;
  coeffs: Int16Array;
  nonZeroY: number;
  nonZeroUv: number;
}

/** Decode a VP8 keyframe (plus an optional ``ALPH`` payload) to RGBA. */
function decodeVp8(data: Uint8Array, alphaChunk: Uint8Array | null): { width: number; height: number; rgba: Uint8Array } {
  const { width, height } = vp8Info(data);
  const partitionLength = (data[0]! | (data[1]! << 8) | (data[2]! << 16)) >> 5;
  let offset = 10;
  if (partitionLength > data.length - offset) throw new ValueError('bad partition length');
  const br = new BoolReader(data, offset, offset + partitionLength);
  offset += partitionLength;
  const mbW = (width + 15) >> 4;
  const mbH = (height + 15) >> 4;
  br.getValue(1); // colour space
  br.getValue(1); // clamping type
  // Segment header.
  const useSegment = br.getValue(1);
  let updateMap = 0;
  let absoluteDelta = 1;
  const quantizer = [0, 0, 0, 0];
  const filterStrength = [0, 0, 0, 0];
  const segmentProbs = [255, 255, 255];
  if (useSegment) {
    updateMap = br.getValue(1);
    if (br.getValue(1)) {
      absoluteDelta = br.getValue(1);
      for (let s = 0; s < 4; s += 1) quantizer[s] = br.getValue(1) ? br.getSignedValue(7) : 0;
      for (let s = 0; s < 4; s += 1) filterStrength[s] = br.getValue(1) ? br.getSignedValue(6) : 0;
    }
    if (updateMap) for (let s = 0; s < 3; s += 1) segmentProbs[s] = br.getValue(1) ? br.getValue(8) : 255;
  }
  // Filter header.
  const simple = br.getValue(1);
  const level = br.getValue(6);
  const sharpness = br.getValue(3);
  const useLfDelta = br.getValue(1);
  const refLfDelta = [0, 0, 0, 0];
  const modeLfDelta = [0, 0, 0, 0];
  if (useLfDelta && br.getValue(1)) {
    for (let i = 0; i < 4; i += 1) if (br.getValue(1)) refLfDelta[i] = br.getSignedValue(6);
    for (let i = 0; i < 4; i += 1) if (br.getValue(1)) modeLfDelta[i] = br.getSignedValue(6);
  }
  const filterType = level === 0 ? 0 : simple ? 1 : 2;
  // Partitions.
  const partitionsMinusOne = (1 << br.getValue(2)) - 1;
  const partitions: BoolReader[] = [];
  {
    const size = data.length - offset;
    if (size < 3 * partitionsMinusOne) throw new ValueError('cannot parse partitions');
    let partStart = offset + partitionsMinusOne * 3;
    let sizeLeft = size - partitionsMinusOne * 3;
    for (let p = 0; p < partitionsMinusOne; p += 1) {
      const sz = offset + p * 3;
      let psize = data[sz]! | (data[sz + 1]! << 8) | (data[sz + 2]! << 16);
      if (psize > sizeLeft) psize = sizeLeft;
      partitions.push(new BoolReader(data, partStart, partStart + psize));
      partStart += psize;
      sizeLeft -= psize;
    }
    partitions.push(new BoolReader(data, partStart, partStart + sizeLeft));
    if (partStart >= data.length) throw new ValueError('cannot parse partitions');
  }
  // Quantizers.
  const baseQ0 = br.getValue(7);
  const dqy1Dc = br.getValue(1) ? br.getSignedValue(4) : 0;
  const dqy2Dc = br.getValue(1) ? br.getSignedValue(4) : 0;
  const dqy2Ac = br.getValue(1) ? br.getSignedValue(4) : 0;
  const dquvDc = br.getValue(1) ? br.getSignedValue(4) : 0;
  const dquvAc = br.getValue(1) ? br.getSignedValue(4) : 0;
  const clip = (v: number, m: number): number => (v < 0 ? 0 : v > m ? m : v);
  const dqm: { y1: [number, number]; y2: [number, number]; uv: [number, number] }[] = [];
  for (let s = 0; s < 4; s += 1) {
    let q: number;
    if (useSegment) {
      q = quantizer[s]!;
      if (!absoluteDelta) q += baseQ0;
    } else {
      if (s > 0) {
        dqm.push(dqm[0]!);
        continue;
      }
      q = baseQ0;
    }
    const y2ac = Math.max((AC_TABLE[clip(q + dqy2Ac, 127)]! * 101581) >> 16, 8);
    dqm.push({
      y1: [DC_TABLE[clip(q + dqy1Dc, 127)]!, AC_TABLE[clip(q, 127)]!],
      y2: [DC_TABLE[clip(q + dqy2Dc, 127)]! * 2, y2ac],
      uv: [DC_TABLE[clip(q + dquvDc, 117)]!, AC_TABLE[clip(q + dquvAc, 127)]!],
    });
  }
  br.getValue(1); // update_proba (ignored)
  // Coefficient probabilities.
  const proba = new Uint8Array(4 * 8 * 3 * 11);
  for (let i = 0; i < proba.length; i += 1) proba[i] = br.getBit(COEFFS_UPDATE_PROBA[i]!) ? br.getValue(8) : COEFFS_PROBA0[i]!;
  const useSkipProba = br.getValue(1);
  const skipP = useSkipProba ? br.getValue(8) : 0;
  const probaAt = (type: number, position: number, ctx: number, p: number): number => proba[((type * 8 + BANDS[position]!) * 3 + ctx) * 11 + p]!;

  // Filter strengths.
  const fstrengths: { limit: number; ilevel: number; inner: number; hevThresh: number }[][] = [];
  for (let s = 0; s < 4; s += 1) {
    const row: { limit: number; ilevel: number; inner: number; hevThresh: number }[] = [];
    let baseLevel: number;
    if (useSegment) {
      baseLevel = filterStrength[s]!;
      if (!absoluteDelta) baseLevel += level;
    } else baseLevel = level;
    for (let i4x4 = 0; i4x4 <= 1; i4x4 += 1) {
      let lvl = baseLevel;
      if (useLfDelta) {
        lvl += refLfDelta[0]!;
        if (i4x4) lvl += modeLfDelta[0]!;
      }
      lvl = lvl < 0 ? 0 : lvl > 63 ? 63 : lvl;
      if (lvl > 0) {
        let ilevel = lvl;
        if (sharpness > 0) {
          ilevel >>= sharpness > 4 ? 2 : 1;
          if (ilevel > 9 - sharpness) ilevel = 9 - sharpness;
        }
        if (ilevel < 1) ilevel = 1;
        row.push({ limit: 2 * lvl + ilevel, ilevel, inner: i4x4, hevThresh: lvl >= 40 ? 2 : lvl >= 15 ? 1 : 0 });
      } else {
        row.push({ limit: 0, ilevel: 0, inner: i4x4, hevThresh: 0 });
      }
    }
    fstrengths.push(row);
  }

  // Reconstruction buffers (unfiltered) for the whole frame.
  const yStride = mbW * 16;
  const uvStride = mbW * 8;
  const yPlane = new Uint8Array(yStride * mbH * 16);
  const uPlane = new Uint8Array(uvStride * mbH * 8);
  const vPlane = new Uint8Array(uvStride * mbH * 8);
  const filterInfo: { limit: number; ilevel: number; inner: number; hevThresh: number }[] = new Array(mbW * mbH);

  const topSamples: TopSamples = Array.from({ length: mbW }, () => ({ y: new Uint8Array(16), u: new Uint8Array(8), v: new Uint8Array(8) }));
  const intraT = new Uint8Array(4 * mbW);
  const intraL = new Uint8Array(4);
  const nzTop = new Uint32Array(mbW);
  const nzDcTop = new Uint8Array(mbW);
  let nzLeft = 0;
  let nzDcLeft = 0;

  const getLargeValue = (reader: BoolReader, type: number, n: number, ctx: number): number => {
    const p = (i: number): number => probaAt(type, n, ctx, i);
    let v: number;
    if (!reader.getBit(p(3))) {
      v = !reader.getBit(p(4)) ? 2 : 3 + reader.getBit(p(5));
    } else if (!reader.getBit(p(6))) {
      if (!reader.getBit(p(7))) v = 5 + reader.getBit(159);
      else {
        v = 7 + 2 * reader.getBit(165);
        v += reader.getBit(145);
      }
    } else {
      const bit1 = reader.getBit(p(8));
      const bit0 = reader.getBit(p(9 + bit1));
      const cat = 2 * bit1 + bit0;
      v = 0;
      for (const probability of CAT3456[cat]!) v += v + reader.getBit(probability);
      v += 3 + (8 << cat);
    }
    return v;
  };

  /** ``GetCoeffsFast``: returns the index after the last non-zero coefficient. */
  const getCoeffs = (reader: BoolReader, type: number, ctx: number, dq: [number, number], first: number, out: Int16Array, outOffset: number): number => {
    let n = first;
    let pCtx = ctx;
    let pBand = n;
    for (; n < 16; n += 1) {
      if (!reader.getBit(probaAt(type, pBand, pCtx, 0))) return n;
      while (!reader.getBit(probaAt(type, pBand, pCtx, 1))) {
        n += 1;
        pBand = n;
        pCtx = 0;
        if (n === 16) return 16;
      }
      let v: number;
      if (!reader.getBit(probaAt(type, pBand, pCtx, 2))) {
        v = 1;
        pCtx = 1;
      } else {
        v = getLargeValue(reader, type, pBand, pCtx);
        pCtx = 2;
      }
      pBand = n + 1;
      out[outOffset + ZIGZAG[n]!] = reader.getSigned(v) * dq[n > 0 ? 1 : 0];
    }
    return 16;
  };

  const nzCodeBits = (nzCoeffs: number, nz: number, dcNz: number): number => ((nzCoeffs << 2) | (nz > 3 ? 3 : nz > 1 ? 2 : dcNz)) >>> 0;

  for (let mbY = 0; mbY < mbH; mbY += 1) {
    const tokenBr = partitions[mbY & partitionsMinusOne]!;
    // Intra modes for the row.
    const blocks: Macroblock[] = [];
    for (let mbX = 0; mbX < mbW; mbX += 1) {
      const block: Macroblock = {
        segment: 0, skip: 0, isI4x4: false, imodes: new Uint8Array(16), uvmode: 0, coeffs: new Int16Array(384), nonZeroY: 0, nonZeroUv: 0,
      };
      if (updateMap) {
        block.segment = !br.getBit(segmentProbs[0]!) ? br.getBit(segmentProbs[1]!) : br.getBit(segmentProbs[2]!) + 2;
      }
      if (useSkipProba) block.skip = br.getBit(skipP);
      block.isI4x4 = !br.getBit(145);
      if (!block.isI4x4) {
        const ymode = br.getBit(156) ? (br.getBit(128) ? 1 : 3) : (br.getBit(163) ? 2 : 0);
        block.imodes[0] = ymode;
        intraT.fill(ymode, 4 * mbX, 4 * mbX + 4);
        intraL.fill(ymode);
      } else {
        for (let y = 0; y < 4; y += 1) {
          let ymode = intraL[y]!;
          for (let x = 0; x < 4; x += 1) {
            const base = (intraT[4 * mbX + x]! * 10 + ymode) * 9;
            const prob = (i: number): number => BMODES_PROBA[base + i]!;
            ymode = !br.getBit(prob(0)) ? 0
              : !br.getBit(prob(1)) ? 1
                : !br.getBit(prob(2)) ? 2
                  : !br.getBit(prob(3))
                    ? (!br.getBit(prob(4)) ? 3 : (!br.getBit(prob(5)) ? 4 : 5))
                    : (!br.getBit(prob(6)) ? 6 : (!br.getBit(prob(7)) ? 7 : (!br.getBit(prob(8)) ? 8 : 9)));
            intraT[4 * mbX + x] = ymode;
          }
          block.imodes.set(intraT.subarray(4 * mbX, 4 * mbX + 4), y * 4);
          intraL[y] = ymode;
        }
      }
      block.uvmode = !br.getBit(142) ? 0 : !br.getBit(114) ? 2 : br.getBit(183) ? 1 : 3;
      blocks.push(block);
    }
    if (br.eof) throw new ValueError('Premature end-of-partition0 encountered.');
    // Residuals.
    for (let mbX = 0; mbX < mbW; mbX += 1) {
      const block = blocks[mbX]!;
      let skip = useSkipProba ? block.skip : 0;
      if (!skip) {
        const q = dqm[block.segment]!;
        const dst = block.coeffs;
        let first: number;
        let acType: number;
        if (!block.isI4x4) {
          const dc = new Int16Array(16);
          const ctx = nzDcTop[mbX]! + nzDcLeft;
          const nz = getCoeffs(tokenBr, 1, ctx, q.y2, 0, dc, 0);
          nzDcTop[mbX] = nzDcLeft = nz > 0 ? 1 : 0;
          if (nz > 1) transformWht(dc, dst);
          else {
            const dc0 = (dc[0]! + 3) >> 3;
            for (let i = 0; i < 256; i += 16) dst[i] = dc0;
          }
          first = 1;
          acType = 0;
        } else {
          first = 0;
          acType = 3;
        }
        let tnz = nzTop[mbX]! & 0x0f;
        let lnz = nzLeft & 0x0f;
        let nonZeroY = 0;
        let position = 0;
        for (let y = 0; y < 4; y += 1) {
          let l = lnz & 1;
          let nzCoeffs = 0;
          for (let x = 0; x < 4; x += 1) {
            const ctx = l + (tnz & 1);
            const nz = getCoeffs(tokenBr, acType, ctx, q.y1, first, dst, position);
            l = nz > first ? 1 : 0;
            tnz = (tnz >> 1) | (l << 7);
            nzCoeffs = nzCodeBits(nzCoeffs, nz, dst[position] !== 0 ? 1 : 0);
            position += 16;
          }
          tnz >>= 4;
          lnz = (lnz >> 1) | (l << 7);
          nonZeroY = ((nonZeroY << 8) | nzCoeffs) >>> 0;
        }
        let outTnz = tnz;
        let outLnz = lnz >> 4;
        let nonZeroUv = 0;
        for (let ch = 0; ch < 4; ch += 2) {
          let nzCoeffs = 0;
          tnz = nzTop[mbX]! >> (4 + ch);
          lnz = nzLeft >> (4 + ch);
          for (let y = 0; y < 2; y += 1) {
            let l = lnz & 1;
            for (let x = 0; x < 2; x += 1) {
              const ctx = l + (tnz & 1);
              const nz = getCoeffs(tokenBr, 2, ctx, q.uv, 0, dst, position);
              l = nz > 0 ? 1 : 0;
              tnz = (tnz >> 1) | (l << 3);
              nzCoeffs = nzCodeBits(nzCoeffs, nz, dst[position] !== 0 ? 1 : 0);
              position += 16;
            }
            tnz >>= 2;
            lnz = (lnz >> 1) | (l << 5);
          }
          nonZeroUv |= nzCoeffs << (4 * ch);
          outTnz |= (tnz << 4) << ch;
          outLnz |= (lnz & 0xf0) << ch;
        }
        nzTop[mbX] = outTnz;
        nzLeft = outLnz;
        block.nonZeroY = nonZeroY;
        block.nonZeroUv = nonZeroUv >>> 0;
        skip = !(nonZeroY | nonZeroUv) ? 1 : 0;
      } else {
        nzLeft = 0;
        nzTop[mbX] = 0;
        if (!block.isI4x4) {
          nzDcLeft = 0;
          nzDcTop[mbX] = 0;
        }
        block.nonZeroY = 0;
        block.nonZeroUv = 0;
      }
      if (filterType > 0) {
        const info = fstrengths[block.segment]![block.isI4x4 ? 1 : 0]!;
        filterInfo[mbY * mbW + mbX] = { ...info, inner: info.inner | (skip ? 0 : 1) };
      }
      if (tokenBr.eof) throw new ValueError('Premature end-of-file encountered.');
    }
    // InitScanline.
    nzLeft = 0;
    nzDcLeft = 0;
    intraL.fill(0);
    reconstructRow(blocks, mbY, mbW, mbH, yPlane, uPlane, vPlane, yStride, uvStride, topSamples);
  }

  if (filterType > 0) {
    for (let mbY = 0; mbY < mbH; mbY += 1) {
      for (let mbX = 0; mbX < mbW; mbX += 1) {
        const info = filterInfo[mbY * mbW + mbX]!;
        filterMacroblock(filterType, info, mbX, mbY, yPlane, uPlane, vPlane, yStride, uvStride);
      }
    }
  }

  // Alpha plane.
  let alpha: Uint8Array | null = null;
  if (alphaChunk) alpha = decodeAlpha(alphaChunk, width, height);

  // YUV -> RGBA with fancy upsampling (EmitFancyRGB / UpsampleRgbaLinePair).
  const rgba = new Uint8Array(width * height * 4);
  const row = (dst: number, yRow: number, topUvRow: number, curUvRow: number, bottomRow: number | null, bottomYRow: number): void => {
    upsampleLinePair(yPlane, yRow * yStride, bottomRow === null ? -1 : bottomYRow * yStride, uPlane, vPlane,
      topUvRow * uvStride, curUvRow * uvStride, rgba, dst * width * 4, bottomRow === null ? -1 : bottomRow * width * 4, width);
  };
  row(0, 0, 0, 0, null, 0);
  let uvRow = 0;
  let y = 0;
  for (; y + 2 < height; y += 2) {
    const top = uvRow;
    uvRow += 1;
    row(y + 1, y + 1, top, uvRow, y + 2, y + 2);
  }
  if (!(height & 1)) row(height - 1, height - 1, uvRow, uvRow, null, 0);
  if (alpha) {
    for (let i = 0; i < width * height; i += 1) rgba[i * 4 + 3] = alpha[i]!;
  } else {
    for (let i = 0; i < width * height; i += 1) rgba[i * 4 + 3] = 255;
  }
  return { width, height, rgba };
}

function yuvToR(y: number, v: number): number {
  return clipYuv(((y * 19077) >> 8) + ((v * 26149) >> 8) - 14234);
}

function yuvToG(y: number, u: number, v: number): number {
  return clipYuv(((y * 19077) >> 8) - ((u * 6419) >> 8) - ((v * 13320) >> 8) + 8708);
}

function yuvToB(y: number, u: number): number {
  return clipYuv(((y * 19077) >> 8) + ((u * 33050) >> 8) - 17685);
}

function clipYuv(value: number): number {
  return (value & ~16383) === 0 ? value >> 6 : value < 0 ? 0 : 255;
}

/** ``UpsampleRgbaLinePair`` (dsp/upsampling.c) for one or two output rows. */
function upsampleLinePair(
  yPlane: Uint8Array, topY: number, bottomY: number, uPlane: Uint8Array, vPlane: Uint8Array, topUv: number, curUv: number,
  out: Uint8Array, topDst: number, bottomDst: number, len: number,
): void {
  const put = (dst: number, x: number, yValue: number, u: number, v: number): void => {
    const o = dst + x * 4;
    out[o] = yuvToR(yValue, v);
    out[o + 1] = yuvToG(yValue, u, v);
    out[o + 2] = yuvToB(yValue, u);
  };
  const lastPair = (len - 1) >> 1;
  let tlU = uPlane[topUv]!;
  let tlV = vPlane[topUv]!;
  let lU = uPlane[curUv]!;
  let lV = vPlane[curUv]!;
  put(topDst, 0, yPlane[topY]!, (3 * tlU + lU + 2) >> 2, (3 * tlV + lV + 2) >> 2);
  if (bottomY >= 0) put(bottomDst, 0, yPlane[bottomY]!, (3 * lU + tlU + 2) >> 2, (3 * lV + tlV + 2) >> 2);
  for (let x = 1; x <= lastPair; x += 1) {
    const tU = uPlane[topUv + x]!;
    const tV = vPlane[topUv + x]!;
    const cU = uPlane[curUv + x]!;
    const cV = vPlane[curUv + x]!;
    const avgU = tlU + tU + lU + cU + 8;
    const avgV = tlV + tV + lV + cV + 8;
    const diag12U = (avgU + 2 * (tU + lU)) >> 3;
    const diag12V = (avgV + 2 * (tV + lV)) >> 3;
    const diag03U = (avgU + 2 * (tlU + cU)) >> 3;
    const diag03V = (avgV + 2 * (tlV + cV)) >> 3;
    put(topDst, 2 * x - 1, yPlane[topY + 2 * x - 1]!, (diag12U + tlU) >> 1, (diag12V + tlV) >> 1);
    put(topDst, 2 * x, yPlane[topY + 2 * x]!, (diag03U + tU) >> 1, (diag03V + tV) >> 1);
    if (bottomY >= 0) {
      put(bottomDst, 2 * x - 1, yPlane[bottomY + 2 * x - 1]!, (diag03U + lU) >> 1, (diag03V + lV) >> 1);
      put(bottomDst, 2 * x, yPlane[bottomY + 2 * x]!, (diag12U + cU) >> 1, (diag12V + cV) >> 1);
    }
    tlU = tU;
    tlV = tV;
    lU = cU;
    lV = cV;
  }
  if (!(len & 1)) {
    put(topDst, len - 1, yPlane[topY + len - 1]!, (3 * tlU + lU + 2) >> 2, (3 * tlV + lV + 2) >> 2);
    if (bottomY >= 0) put(bottomDst, len - 1, yPlane[bottomY + len - 1]!, (3 * lU + tlU + 2) >> 2, (3 * lV + tlV + 2) >> 2);
  }
}

// Inverse transforms (dsp/dec.c).

function transformWht(input: Int16Array, out: Int16Array): void {
  const tmp = new Int32Array(16);
  for (let i = 0; i < 4; i += 1) {
    const a0 = input[i]! + input[12 + i]!;
    const a1 = input[4 + i]! + input[8 + i]!;
    const a2 = input[4 + i]! - input[8 + i]!;
    const a3 = input[i]! - input[12 + i]!;
    tmp[i] = a0 + a1;
    tmp[8 + i] = a0 - a1;
    tmp[4 + i] = a3 + a2;
    tmp[12 + i] = a3 - a2;
  }
  for (let i = 0; i < 4; i += 1) {
    const dc = tmp[i * 4]! + 3;
    const a0 = dc + tmp[3 + i * 4]!;
    const a1 = tmp[1 + i * 4]! + tmp[2 + i * 4]!;
    const a2 = tmp[1 + i * 4]! - tmp[2 + i * 4]!;
    const a3 = dc - tmp[3 + i * 4]!;
    const base = i * 64;
    out[base] = (a0 + a1) >> 3;
    out[base + 16] = (a3 + a2) >> 3;
    out[base + 32] = (a0 - a1) >> 3;
    out[base + 48] = (a3 - a2) >> 3;
  }
}

function store(dst: Uint8Array, offset: number, value: number): void {
  dst[offset] = clip8(dst[offset]! + (value >> 3));
}

function transformOne(input: Int16Array, inOffset: number, dst: Uint8Array, offset: number, stride: number): void {
  const c = new Int32Array(16);
  for (let i = 0; i < 4; i += 1) {
    const a = input[inOffset + i]! + input[inOffset + 8 + i]!;
    const b = input[inOffset + i]! - input[inOffset + 8 + i]!;
    const cc = mul2(input[inOffset + 4 + i]!) - mul1(input[inOffset + 12 + i]!);
    const d = mul1(input[inOffset + 4 + i]!) + mul2(input[inOffset + 12 + i]!);
    c[i * 4] = a + d;
    c[i * 4 + 1] = b + cc;
    c[i * 4 + 2] = b - cc;
    c[i * 4 + 3] = a - d;
  }
  for (let i = 0; i < 4; i += 1) {
    const dc = c[i]! + 4;
    const a = dc + c[8 + i]!;
    const b = dc - c[8 + i]!;
    const cc = mul2(c[4 + i]!) - mul1(c[12 + i]!);
    const d = mul1(c[4 + i]!) + mul2(c[12 + i]!);
    const o = offset + i * stride;
    store(dst, o, a + d);
    store(dst, o + 1, b + cc);
    store(dst, o + 2, b - cc);
    store(dst, o + 3, a - d);
  }
}

function transformAc3(input: Int16Array, inOffset: number, dst: Uint8Array, offset: number, stride: number): void {
  const a = input[inOffset]! + 4;
  const c4 = mul2(input[inOffset + 4]!);
  const d4 = mul1(input[inOffset + 4]!);
  const c1 = mul2(input[inOffset + 1]!);
  const d1 = mul1(input[inOffset + 1]!);
  const rows = [a + d4, a + c4, a - c4, a - d4];
  for (let y = 0; y < 4; y += 1) {
    const dc = rows[y]!;
    const o = offset + y * stride;
    store(dst, o, dc + d1);
    store(dst, o + 1, dc + c1);
    store(dst, o + 2, dc - c1);
    store(dst, o + 3, dc - d1);
  }
}

function transformDc(input: Int16Array, inOffset: number, dst: Uint8Array, offset: number, stride: number): void {
  const dc = input[inOffset]! + 4;
  for (let j = 0; j < 4; j += 1) for (let i = 0; i < 4; i += 1) store(dst, offset + j * stride + i, dc);
}

function doTransform(bits: number, input: Int16Array, inOffset: number, dst: Uint8Array, offset: number, stride: number): void {
  switch (bits >>> 30) {
    case 3: transformOne(input, inOffset, dst, offset, stride); break;
    case 2: transformAc3(input, inOffset, dst, offset, stride); break;
    case 1: transformDc(input, inOffset, dst, offset, stride); break;
    default: break;
  }
}

function doUvTransform(bits: number, input: Int16Array, inOffset: number, dst: Uint8Array, offset: number, stride: number): void {
  if (!(bits & 0xff)) return;
  if (bits & 0xaa) {
    for (let k = 0; k < 4; k += 1) {
      transformOne(input, inOffset + k * 16, dst, offset + (k >> 1) * 4 * stride + (k & 1) * 4, stride);
    }
  } else {
    for (let k = 0; k < 4; k += 1) {
      if (input[inOffset + k * 16]) transformDc(input, inOffset + k * 16, dst, offset + (k >> 1) * 4 * stride + (k & 1) * 4, stride);
    }
  }
}

/**
 * Intra prediction and reconstruction of one macroblock row, using libwebp's
 * ``yuv_b`` work area layout (BPS = 32) with its border conventions.
 */
const BPS = 32;
const Y_OFF = BPS + 8;
const U_OFF = Y_OFF + BPS * 16 + BPS;
const V_OFF = U_OFF + 16;
const YUV_SIZE = BPS * 17 + BPS * 9;

type TopSamples = { y: Uint8Array; u: Uint8Array; v: Uint8Array }[];

function reconstructRow(
  blocks: Macroblock[], mbY: number, mbW: number, mbH: number, yPlane: Uint8Array, uPlane: Uint8Array, vPlane: Uint8Array,
  yStride: number, uvStride: number, topSamples: TopSamples,
): void {
  const work = new Uint8Array(YUV_SIZE);
  for (let j = 0; j < 16; j += 1) work[Y_OFF + j * BPS - 1] = 129;
  for (let j = 0; j < 8; j += 1) {
    work[U_OFF + j * BPS - 1] = 129;
    work[V_OFF + j * BPS - 1] = 129;
  }
  if (mbY > 0) {
    work[Y_OFF - 1 - BPS] = work[U_OFF - 1 - BPS] = work[V_OFF - 1 - BPS] = 129;
  } else {
    work.fill(127, Y_OFF - BPS - 1, Y_OFF - BPS - 1 + 16 + 4 + 1);
    work.fill(127, U_OFF - BPS - 1, U_OFF - BPS - 1 + 8 + 1);
    work.fill(127, V_OFF - BPS - 1, V_OFF - BPS - 1 + 8 + 1);
  }
  for (let mbX = 0; mbX < mbW; mbX += 1) {
    const block = blocks[mbX]!;
    if (mbX > 0) {
      for (let j = -1; j < 16; j += 1) work.copyWithin(Y_OFF + j * BPS - 4, Y_OFF + j * BPS + 12, Y_OFF + j * BPS + 16);
      for (let j = -1; j < 8; j += 1) {
        work.copyWithin(U_OFF + j * BPS - 4, U_OFF + j * BPS + 4, U_OFF + j * BPS + 8);
        work.copyWithin(V_OFF + j * BPS - 4, V_OFF + j * BPS + 4, V_OFF + j * BPS + 8);
      }
    }
    const top = topSamples[mbX]!;
    const coeffs = block.coeffs;
    let bits = block.nonZeroY;
    if (mbY > 0) {
      work.set(top.y, Y_OFF - BPS);
      work.set(top.u, U_OFF - BPS);
      work.set(top.v, V_OFF - BPS);
    }
    if (block.isI4x4) {
      const topRight = Y_OFF - BPS + 16;
      if (mbY > 0) {
        if (mbX >= mbW - 1) work.fill(top.y[15]!, topRight, topRight + 4);
        else {
          const next = topSamples[mbX + 1]!;
          work.set(next.y.subarray(0, 4), topRight);
        }
      }
      for (const k of [1, 2, 3]) work.copyWithin(topRight + k * 4 * BPS, topRight, topRight + 4);
      for (let n = 0; n < 16; n += 1, bits = (bits << 2) >>> 0) {
        const dst = Y_OFF + (n & 3) * 4 + (n >> 2) * 4 * BPS;
        predictLuma4(block.imodes[n]!, work, dst);
        doTransform(bits, coeffs, n * 16, work, dst, BPS);
      }
    } else {
      predictLuma16(checkMode(mbX, mbY, block.imodes[0]!), work, Y_OFF);
      if (bits !== 0) {
        for (let n = 0; n < 16; n += 1, bits = (bits << 2) >>> 0) {
          doTransform(bits, coeffs, n * 16, work, Y_OFF + (n & 3) * 4 + (n >> 2) * 4 * BPS, BPS);
        }
      }
    }
    const uvMode = checkMode(mbX, mbY, block.uvmode);
    predictChroma8(uvMode, work, U_OFF);
    predictChroma8(uvMode, work, V_OFF);
    doUvTransform(block.nonZeroUv, coeffs, 16 * 16, work, U_OFF, BPS);
    doUvTransform(block.nonZeroUv >>> 8, coeffs, 20 * 16, work, V_OFF, BPS);
    if (mbY < mbH - 1) {
      top.y.set(work.subarray(Y_OFF + 15 * BPS, Y_OFF + 15 * BPS + 16));
      top.u.set(work.subarray(U_OFF + 7 * BPS, U_OFF + 7 * BPS + 8));
      top.v.set(work.subarray(V_OFF + 7 * BPS, V_OFF + 7 * BPS + 8));
    }
    for (let j = 0; j < 16; j += 1) yPlane.set(work.subarray(Y_OFF + j * BPS, Y_OFF + j * BPS + 16), (mbY * 16 + j) * yStride + mbX * 16);
    for (let j = 0; j < 8; j += 1) {
      uPlane.set(work.subarray(U_OFF + j * BPS, U_OFF + j * BPS + 8), (mbY * 8 + j) * uvStride + mbX * 8);
      vPlane.set(work.subarray(V_OFF + j * BPS, V_OFF + j * BPS + 8), (mbY * 8 + j) * uvStride + mbX * 8);
    }
  }
}

function checkMode(mbX: number, mbY: number, mode: number): number {
  if (mode === 0) {
    if (mbX === 0) return mbY === 0 ? 6 : 5;
    return mbY === 0 ? 4 : 0;
  }
  return mode;
}

function trueMotion(work: Uint8Array, dst: number, size: number): void {
  const top = dst - BPS;
  const topLeft = work[top - 1]!;
  for (let y = 0; y < size; y += 1) {
    const left = work[dst + y * BPS - 1]!;
    for (let x = 0; x < size; x += 1) work[dst + y * BPS + x] = clip8(left + work[top + x]! - topLeft);
  }
}

function fillBlock(work: Uint8Array, dst: number, size: number, value: number): void {
  for (let y = 0; y < size; y += 1) work.fill(value, dst + y * BPS, dst + y * BPS + size);
}

function predictLuma16(mode: number, work: Uint8Array, dst: number): void {
  switch (mode) {
    case 0: {
      let dc = 16;
      for (let j = 0; j < 16; j += 1) dc += work[dst - 1 + j * BPS]! + work[dst + j - BPS]!;
      fillBlock(work, dst, 16, dc >> 5);
      break;
    }
    case 1: trueMotion(work, dst, 16); break;
    case 2: for (let j = 0; j < 16; j += 1) work.copyWithin(dst + j * BPS, dst - BPS, dst - BPS + 16); break;
    case 3: for (let j = 0; j < 16; j += 1) work.fill(work[dst + j * BPS - 1]!, dst + j * BPS, dst + j * BPS + 16); break;
    case 4: {
      let dc = 8;
      for (let j = 0; j < 16; j += 1) dc += work[dst - 1 + j * BPS]!;
      fillBlock(work, dst, 16, dc >> 4);
      break;
    }
    case 5: {
      let dc = 8;
      for (let i = 0; i < 16; i += 1) dc += work[dst + i - BPS]!;
      fillBlock(work, dst, 16, dc >> 4);
      break;
    }
    default: fillBlock(work, dst, 16, 0x80); break;
  }
}

function predictChroma8(mode: number, work: Uint8Array, dst: number): void {
  switch (mode) {
    case 0: {
      let dc = 8;
      for (let i = 0; i < 8; i += 1) dc += work[dst + i - BPS]! + work[dst - 1 + i * BPS]!;
      fillBlock(work, dst, 8, dc >> 4);
      break;
    }
    case 1: trueMotion(work, dst, 8); break;
    case 2: for (let j = 0; j < 8; j += 1) work.copyWithin(dst + j * BPS, dst - BPS, dst - BPS + 8); break;
    case 3: for (let j = 0; j < 8; j += 1) work.fill(work[dst + j * BPS - 1]!, dst + j * BPS, dst + j * BPS + 8); break;
    case 4: {
      let dc = 4;
      for (let i = 0; i < 8; i += 1) dc += work[dst - 1 + i * BPS]!;
      fillBlock(work, dst, 8, dc >> 3);
      break;
    }
    case 5: {
      let dc = 4;
      for (let i = 0; i < 8; i += 1) dc += work[dst + i - BPS]!;
      fillBlock(work, dst, 8, dc >> 3);
      break;
    }
    default: fillBlock(work, dst, 8, 0x80); break;
  }
}

const avg3 = (a: number, b: number, c: number): number => (a + 2 * b + c + 2) >> 2;
const avg2 = (a: number, b: number): number => (a + b + 1) >> 1;

function predictLuma4(mode: number, work: Uint8Array, dst: number): void {
  const at = (x: number, y: number): number => work[dst + x + y * BPS]!;
  const set = (x: number, y: number, value: number): void => {
    work[dst + x + y * BPS] = value;
  };
  const I = at(-1, 0);
  const J = at(-1, 1);
  const K = at(-1, 2);
  const L = at(-1, 3);
  const X = at(-1, -1);
  const A = at(0, -1);
  const B = at(1, -1);
  const C = at(2, -1);
  const D = at(3, -1);
  const E = at(4, -1);
  const F = at(5, -1);
  const G = at(6, -1);
  const H = at(7, -1);
  switch (mode) {
    case 0: { // DC
      let dc = 4;
      for (let i = 0; i < 4; i += 1) dc += at(i, -1) + at(-1, i);
      dc >>= 3;
      for (let y = 0; y < 4; y += 1) for (let x = 0; x < 4; x += 1) set(x, y, dc);
      break;
    }
    case 1: trueMotion(work, dst, 4); break;
    case 2: { // VE
      const values = [avg3(X, A, B), avg3(A, B, C), avg3(B, C, D), avg3(C, D, E)];
      for (let y = 0; y < 4; y += 1) for (let x = 0; x < 4; x += 1) set(x, y, values[x]!);
      break;
    }
    case 3: { // HE
      const values = [avg3(X, I, J), avg3(I, J, K), avg3(J, K, L), avg3(K, L, L)];
      for (let y = 0; y < 4; y += 1) for (let x = 0; x < 4; x += 1) set(x, y, values[y]!);
      break;
    }
    case 4: { // RD
      set(0, 3, avg3(J, K, L));
      const v1 = avg3(I, J, K); set(1, 3, v1); set(0, 2, v1);
      const v2 = avg3(X, I, J); set(2, 3, v2); set(1, 2, v2); set(0, 1, v2);
      const v3 = avg3(A, X, I); set(3, 3, v3); set(2, 2, v3); set(1, 1, v3); set(0, 0, v3);
      const v4 = avg3(B, A, X); set(3, 2, v4); set(2, 1, v4); set(1, 0, v4);
      const v5 = avg3(C, B, A); set(3, 1, v5); set(2, 0, v5);
      set(3, 0, avg3(D, C, B));
      break;
    }
    case 5: { // VR
      const a = avg2(X, A); set(0, 0, a); set(1, 2, a);
      const b = avg2(A, B); set(1, 0, b); set(2, 2, b);
      const c = avg2(B, C); set(2, 0, c); set(3, 2, c);
      set(3, 0, avg2(C, D));
      set(0, 3, avg3(K, J, I));
      set(0, 2, avg3(J, I, X));
      const d = avg3(I, X, A); set(0, 1, d); set(1, 3, d);
      const e = avg3(X, A, B); set(1, 1, e); set(2, 3, e);
      const f = avg3(A, B, C); set(2, 1, f); set(3, 3, f);
      set(3, 1, avg3(B, C, D));
      break;
    }
    case 6: { // LD
      set(0, 0, avg3(A, B, C));
      const a = avg3(B, C, D); set(1, 0, a); set(0, 1, a);
      const b = avg3(C, D, E); set(2, 0, b); set(1, 1, b); set(0, 2, b);
      const c = avg3(D, E, F); set(3, 0, c); set(2, 1, c); set(1, 2, c); set(0, 3, c);
      const d = avg3(E, F, G); set(3, 1, d); set(2, 2, d); set(1, 3, d);
      const e = avg3(F, G, H); set(3, 2, e); set(2, 3, e);
      set(3, 3, avg3(G, H, H));
      break;
    }
    case 7: { // VL
      set(0, 0, avg2(A, B));
      const a = avg2(B, C); set(1, 0, a); set(0, 2, a);
      const b = avg2(C, D); set(2, 0, b); set(1, 2, b);
      const c = avg2(D, E); set(3, 0, c); set(2, 2, c);
      set(0, 1, avg3(A, B, C));
      const d = avg3(B, C, D); set(1, 1, d); set(0, 3, d);
      const e = avg3(C, D, E); set(2, 1, e); set(1, 3, e);
      const f = avg3(D, E, F); set(3, 1, f); set(2, 3, f);
      set(3, 2, avg3(E, F, G));
      set(3, 3, avg3(F, G, H));
      break;
    }
    case 8: { // HD
      const a = avg2(I, X); set(0, 0, a); set(2, 1, a);
      const b = avg2(J, I); set(0, 1, b); set(2, 2, b);
      const c = avg2(K, J); set(0, 2, c); set(2, 3, c);
      set(0, 3, avg2(L, K));
      set(3, 0, avg3(A, B, C));
      set(2, 0, avg3(X, A, B));
      const d = avg3(I, X, A); set(1, 0, d); set(3, 1, d);
      const e = avg3(J, I, X); set(1, 1, e); set(3, 2, e);
      const f = avg3(K, J, I); set(1, 2, f); set(3, 3, f);
      set(1, 3, avg3(L, K, J));
      break;
    }
    default: { // HU
      set(0, 0, avg2(I, J));
      const a = avg2(J, K); set(2, 0, a); set(0, 1, a);
      const b = avg2(K, L); set(2, 1, b); set(0, 2, b);
      set(1, 0, avg3(I, J, K));
      const c = avg3(J, K, L); set(3, 0, c); set(1, 1, c);
      const d = avg3(K, L, L); set(3, 1, d); set(1, 2, d);
      set(3, 2, L); set(2, 2, L); set(0, 3, L); set(1, 3, L); set(2, 3, L); set(3, 3, L);
      break;
    }
  }
}

// Loop filters (dsp/dec.c).

const sclip1 = (v: number): number => (v < -128 ? -128 : v > 127 ? 127 : v);
const sclip2 = (v: number): number => (v < -16 ? -16 : v > 15 ? 15 : v);
const uclip = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : v);

function doFilter2(p: Uint8Array, o: number, step: number): void {
  const p1 = p[o - 2 * step]!;
  const p0 = p[o - step]!;
  const q0 = p[o]!;
  const q1 = p[o + step]!;
  const a = 3 * (q0 - p0) + sclip1(p1 - q1);
  const a1 = sclip2((a + 4) >> 3);
  const a2 = sclip2((a + 3) >> 3);
  p[o - step] = uclip(p0 + a2);
  p[o] = uclip(q0 - a1);
}

function doFilter4(p: Uint8Array, o: number, step: number): void {
  const p1 = p[o - 2 * step]!;
  const p0 = p[o - step]!;
  const q0 = p[o]!;
  const q1 = p[o + step]!;
  const a = 3 * (q0 - p0);
  const a1 = sclip2((a + 4) >> 3);
  const a2 = sclip2((a + 3) >> 3);
  const a3 = (a1 + 1) >> 1;
  p[o - 2 * step] = uclip(p1 + a3);
  p[o - step] = uclip(p0 + a2);
  p[o] = uclip(q0 - a1);
  p[o + step] = uclip(q1 - a3);
}

function doFilter6(p: Uint8Array, o: number, step: number): void {
  const p2 = p[o - 3 * step]!;
  const p1 = p[o - 2 * step]!;
  const p0 = p[o - step]!;
  const q0 = p[o]!;
  const q1 = p[o + step]!;
  const q2 = p[o + 2 * step]!;
  const a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
  const a1 = (27 * a + 63) >> 7;
  const a2 = (18 * a + 63) >> 7;
  const a3 = (9 * a + 63) >> 7;
  p[o - 3 * step] = uclip(p2 + a3);
  p[o - 2 * step] = uclip(p1 + a2);
  p[o - step] = uclip(p0 + a1);
  p[o] = uclip(q0 - a1);
  p[o + step] = uclip(q1 - a2);
  p[o + 2 * step] = uclip(q2 - a3);
}

function hev(p: Uint8Array, o: number, step: number, thresh: number): boolean {
  const p1 = p[o - 2 * step]!;
  const p0 = p[o - step]!;
  const q0 = p[o]!;
  const q1 = p[o + step]!;
  return Math.abs(p1 - p0) > thresh || Math.abs(q1 - q0) > thresh;
}

function needsFilter(p: Uint8Array, o: number, step: number, t: number): boolean {
  const p1 = p[o - 2 * step]!;
  const p0 = p[o - step]!;
  const q0 = p[o]!;
  const q1 = p[o + step]!;
  return 4 * Math.abs(p0 - q0) + Math.abs(p1 - q1) <= t;
}

function needsFilter2(p: Uint8Array, o: number, step: number, t: number, it: number): boolean {
  const p3 = p[o - 4 * step]!;
  const p2 = p[o - 3 * step]!;
  const p1 = p[o - 2 * step]!;
  const p0 = p[o - step]!;
  const q0 = p[o]!;
  const q1 = p[o + step]!;
  const q2 = p[o + 2 * step]!;
  const q3 = p[o + 3 * step]!;
  if (4 * Math.abs(p0 - q0) + Math.abs(p1 - q1) > t) return false;
  return Math.abs(p3 - p2) <= it && Math.abs(p2 - p1) <= it && Math.abs(p1 - p0) <= it
    && Math.abs(q3 - q2) <= it && Math.abs(q2 - q1) <= it && Math.abs(q1 - q0) <= it;
}

function simpleFilter(p: Uint8Array, o: number, step: number, along: number, thresh: number): void {
  const thresh2 = 2 * thresh + 1;
  for (let i = 0; i < 16; i += 1) if (needsFilter(p, o + i * along, step, thresh2)) doFilter2(p, o + i * along, step);
}

function filterLoop(p: Uint8Array, o: number, hstride: number, vstride: number, size: number, thresh: number, ithresh: number, hevThresh: number, six: boolean): void {
  const thresh2 = 2 * thresh + 1;
  for (let i = 0; i < size; i += 1) {
    const at = o + i * vstride;
    if (needsFilter2(p, at, hstride, thresh2, ithresh)) {
      if (hev(p, at, hstride, hevThresh)) doFilter2(p, at, hstride);
      else if (six) doFilter6(p, at, hstride);
      else doFilter4(p, at, hstride);
    }
  }
}

function filterMacroblock(
  filterType: number, info: { limit: number; ilevel: number; inner: number; hevThresh: number }, mbX: number, mbY: number,
  yPlane: Uint8Array, uPlane: Uint8Array, vPlane: Uint8Array, yStride: number, uvStride: number,
): void {
  const limit = info.limit;
  if (limit === 0) return;
  const y = mbY * 16 * yStride + mbX * 16;
  if (filterType === 1) {
    if (mbX > 0) simpleFilter(yPlane, y, 1, yStride, limit + 4);
    if (info.inner) for (let k = 1; k <= 3; k += 1) simpleFilter(yPlane, y + 4 * k, 1, yStride, limit);
    if (mbY > 0) simpleFilter(yPlane, y, yStride, 1, limit + 4);
    if (info.inner) for (let k = 1; k <= 3; k += 1) simpleFilter(yPlane, y + 4 * k * yStride, yStride, 1, limit);
    return;
  }
  const uv = mbY * 8 * uvStride + mbX * 8;
  const { ilevel, hevThresh } = info;
  if (mbX > 0) {
    filterLoop(yPlane, y, 1, yStride, 16, limit + 4, ilevel, hevThresh, true);
    filterLoop(uPlane, uv, 1, uvStride, 8, limit + 4, ilevel, hevThresh, true);
    filterLoop(vPlane, uv, 1, uvStride, 8, limit + 4, ilevel, hevThresh, true);
  }
  if (info.inner) {
    for (let k = 1; k <= 3; k += 1) filterLoop(yPlane, y + 4 * k, 1, yStride, 16, limit, ilevel, hevThresh, false);
    filterLoop(uPlane, uv + 4, 1, uvStride, 8, limit, ilevel, hevThresh, false);
    filterLoop(vPlane, uv + 4, 1, uvStride, 8, limit, ilevel, hevThresh, false);
  }
  if (mbY > 0) {
    filterLoop(yPlane, y, yStride, 1, 16, limit + 4, ilevel, hevThresh, true);
    filterLoop(uPlane, uv, uvStride, 1, 8, limit + 4, ilevel, hevThresh, true);
    filterLoop(vPlane, uv, uvStride, 1, 8, limit + 4, ilevel, hevThresh, true);
  }
  if (info.inner) {
    for (let k = 1; k <= 3; k += 1) filterLoop(yPlane, y + 4 * k * yStride, yStride, 1, 16, limit, ilevel, hevThresh, false);
    filterLoop(uPlane, uv + 4 * uvStride, uvStride, 1, 8, limit, ilevel, hevThresh, false);
    filterLoop(vPlane, uv + 4 * uvStride, uvStride, 1, 8, limit, ilevel, hevThresh, false);
  }
}

// ---------------------------------------------------------------------------
// ALPH (alpha_dec.c, filters.c).
// ---------------------------------------------------------------------------

function decodeAlpha(chunk: Uint8Array, width: number, height: number): Uint8Array {
  if (chunk.length <= 1) throw new ValueError('Could not decode alpha data.');
  const method = chunk[0]! & 3;
  const filter = (chunk[0]! >> 2) & 3;
  const preprocessing = (chunk[0]! >> 4) & 3;
  const reserved = (chunk[0]! >> 6) & 3;
  if (method > 1 || filter > 3 || preprocessing > 1 || reserved !== 0) throw new ValueError('Could not decode alpha data.');
  const payload = chunk.subarray(1);
  let plane: Uint8Array;
  if (method === 0) {
    if (payload.length < width * height) throw new ValueError('Could not decode alpha data.');
    plane = payload.slice(0, width * height);
  } else {
    plane = decodeLosslessAlpha(payload, width, height);
  }
  if (filter !== 0) {
    for (let y = 0; y < height; y += 1) {
      const row = y * width;
      const previous = y > 0 ? row - width : -1;
      if (filter === 1 || previous < 0) {
        let pred = previous < 0 ? 0 : plane[previous]!;
        for (let i = 0; i < width; i += 1) {
          plane[row + i] = (pred + plane[row + i]!) & 0xff;
          pred = plane[row + i]!;
        }
      } else if (filter === 2) {
        for (let i = 0; i < width; i += 1) plane[row + i] = (plane[previous + i]! + plane[row + i]!) & 0xff;
      } else {
        let top = plane[previous]!;
        let topLeft = top;
        let left = top;
        for (let i = 0; i < width; i += 1) {
          top = plane[previous + i]!;
          const g = left + top - topLeft;
          left = (plane[row + i]! + ((g & ~0xff) === 0 ? g : g < 0 ? 0 : 255)) & 0xff;
          topLeft = top;
          plane[row + i] = left;
        }
      }
    }
  }
  return plane;
}
