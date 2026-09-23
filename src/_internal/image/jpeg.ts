/**
 * JPEG decoding that reproduces libjpeg-turbo (the decoder behind Pillow and
 * torchvision) sample for sample: baseline, extended and progressive Huffman
 * coding, restart intervals, the ``ISLOW`` integer IDCT (``jidctint.c``),
 * fancy upsampling (``jdsample.c``: ``h2v1``, ``h1v2`` and ``h2v2`` triangle
 * filters, integral box upsampling otherwise), libjpeg's colour-space
 * detection (JFIF/Adobe markers, component ids) and its fixed-point colour
 * conversion tables (``jdcolor.c``).
 */
import { ValueError } from '../../errors.js';

/** libjpeg ``J_COLOR_SPACE`` values the decoder distinguishes. */
export type JpegColorSpace = 'GRAYSCALE' | 'YCbCr' | 'RGB' | 'CMYK' | 'YCCK' | 'UNKNOWN';

export interface JpegData {
  width: number;
  height: number;
  /** Colour space of the encoded components. */
  colorSpace: JpegColorSpace;
  /** Number of encoded components. */
  components: number;
  progressive: boolean;
  /** Lossless (``SOF3``) process. */
  lossless: boolean;
  /** Sample precision in bits (8, or 2-8 for lossless files). */
  precision: number;
  /** Arithmetic entropy coding (``SOF9``/``SOF10``). */
  arithmetic: boolean;
  /** Adobe APP14 transform flag, or ``null`` without an Adobe marker. */
  adobeTransform: number | null;
  /** ``Exif\0\0``-prefixed payload of the first APP1 Exif segment (Pillow ``info['exif']``). */
  exif: Uint8Array | null;
  /** Payload of the first APP1 segment of any kind (what torchvision reads orientation from). */
  firstApp1: Uint8Array | null;
  /** XMP packet of the first APP1 XMP segment (Pillow ``info['xmp']``). */
  xmp: Uint8Array | null;
  /**
   * Convert to libjpeg's output colour space: ``GRAYSCALE`` (1 channel),
   * ``RGB`` (3) or ``CMYK`` (4, with YCCK converted). Interleaved ``HWC`` bytes.
   */
  output(space: 'GRAYSCALE' | 'RGB' | 'CMYK'): Uint8Array;
}

const NATURAL_ORDER = new Int32Array(80);
{
  const order = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
  ];
  NATURAL_ORDER.set(order);
  NATURAL_ORDER.fill(63, 64);
}

interface HuffmanTable {
  /** ``lookup[code bits (16, MSB first)]``: ``(length << 8) | symbol``; 0 marks an invalid code. */
  maxcode: Int32Array;
  valptr: Int32Array;
  mincode: Int32Array;
  values: Uint8Array;
  fast: Int32Array;
}

const FAST_BITS = 9;

function buildHuffman(counts: Uint8Array, values: Uint8Array): HuffmanTable {
  // jdhuff.c jpeg_make_d_derived_tbl.
  const huffsize: number[] = [];
  for (let length = 1; length <= 16; length += 1) for (let i = 0; i < counts[length - 1]!; i += 1) huffsize.push(length);
  const huffcode: number[] = [];
  let code = 0;
  let size = huffsize[0] ?? 0;
  let p = 0;
  while (p < huffsize.length) {
    while (huffsize[p] === size) {
      huffcode.push(code);
      code += 1;
      p += 1;
    }
    if (code >= 2 ** size) throw new ValueError('Bogus Huffman table definition');
    code <<= 1;
    size += 1;
  }
  const maxcode = new Int32Array(18);
  const valptr = new Int32Array(17);
  const mincode = new Int32Array(17);
  p = 0;
  for (let length = 1; length <= 16; length += 1) {
    if (counts[length - 1]) {
      valptr[length] = p - huffcode[p]!;
      mincode[length] = huffcode[p]!;
      p += counts[length - 1]!;
      maxcode[length] = huffcode[p - 1]!;
    } else {
      maxcode[length] = -1;
    }
  }
  maxcode[17] = 0x7fffffff;
  const fast = new Int32Array(1 << FAST_BITS);
  p = 0;
  for (let length = 1; length <= FAST_BITS; length += 1) {
    for (let i = 0; i < counts[length - 1]!; i += 1, p += 1) {
      const lookbits = huffcode[p]! << (FAST_BITS - length);
      for (let ctr = 1 << (FAST_BITS - length); ctr > 0; ctr -= 1) fast[lookbits + ctr - 1] = (length << 8) | values[p]!;
    }
  }
  return { maxcode, valptr, mincode, values, fast };
}

interface Component {
  id: number;
  h: number;
  v: number;
  tq: number;
  blocksPerLine: number;
  blocksPerColumn: number;
  /** Allocated (MCU-padded) block grid. */
  paddedBlocksPerLine: number;
  paddedBlocksPerColumn: number;
  coefficients: Int16Array;
  quant: Int32Array | null;
  dcPredictor: number;
  downsampledWidth: number;
  downsampledHeight: number;
  dcTable: HuffmanTable | null;
  acTable: HuffmanTable | null;
  /** Arithmetic-coding conditioning table indices. */
  dcIndex: number;
  acIndex: number;
  /** Lossless mode: decoded samples (``downsampledWidth`` x ``downsampledHeight``). */
  samples: Uint8Array | null;
}

class BitReader {
  private buffer = 0;
  private bits = 0;
  /** Set when a marker or the end of data was reached (libjpeg then inserts zeros). */
  hitMarker = false;
  constructor(private readonly data: Uint8Array, public position: number) {}

  private fill(): void {
    while (this.bits <= 24) {
      let byte = 0;
      if (!this.hitMarker && this.position < this.data.length) {
        byte = this.data[this.position]!;
        if (byte === 0xff) {
          const next = this.data[this.position + 1];
          if (next === 0x00) {
            this.position += 2;
          } else {
            this.hitMarker = true;
            byte = 0;
          }
        } else {
          this.position += 1;
        }
      } else {
        this.hitMarker = true;
      }
      this.buffer = ((this.buffer << 8) | byte) >>> 0;
      this.bits += 8;
    }
  }

  getBits(count: number): number {
    if (count === 0) return 0;
    if (this.bits < count) this.fill();
    this.bits -= count;
    return (this.buffer >>> this.bits) & ((1 << count) - 1);
  }

  peek(count: number): number {
    if (this.bits < count) this.fill();
    return (this.buffer >>> (this.bits - count)) & ((1 << count) - 1);
  }

  skip(count: number): void {
    this.bits -= count;
  }

  decode(table: HuffmanTable): number {
    const look = this.peek(FAST_BITS);
    const entry = table.fast[look]!;
    if (entry) {
      this.skip(entry >> 8);
      return entry & 0xff;
    }
    // jdhuff.c jpeg_huff_decode (slow path).
    let length = 1;
    let code = this.getBits(1);
    while (code > table.maxcode[length]!) {
      code = (code << 1) | this.getBits(1);
      length += 1;
      if (length > 16) return 0; // JWRN_HUFF_BAD_CODE: libjpeg returns a zero symbol.
    }
    return table.values[table.valptr[length]! + code]!;
  }

  /** Discard buffered bits and resynchronize after a restart marker. */
  restart(): void {
    this.buffer = 0;
    this.bits = 0;
    this.hitMarker = false;
    // Skip to the next RSTn marker (libjpeg's read_restart_marker / resync).
    while (this.position + 1 < this.data.length) {
      if (this.data[this.position] === 0xff) {
        const next = this.data[this.position + 1]!;
        if (next >= 0xd0 && next <= 0xd7) {
          this.position += 2;
          return;
        }
        if (next === 0xff) {
          this.position += 1;
          continue;
        }
        if (next !== 0x00) return; // Another marker: leave it for the parser.
      }
      this.position += 1;
    }
  }
}

function extend(value: number, size: number): number {
  return value < 1 << (size - 1) ? value + (-1 << size) + 1 : value;
}

// ---------------------------------------------------------------------------
// ISLOW IDCT (jidctint.c) and range limiting.
// ---------------------------------------------------------------------------

const FIX_0_298631336 = 2446;
const FIX_0_390180644 = 3196;
const FIX_0_541196100 = 4433;
const FIX_0_765366865 = 6270;
const FIX_0_899976223 = 7373;
const FIX_1_175875602 = 9633;
const FIX_1_501321110 = 12299;
const FIX_1_847759065 = 15137;
const FIX_1_961570560 = 16069;
const FIX_2_053119869 = 16819;
const FIX_2_562915447 = 20995;
const FIX_3_072711026 = 25172;
const CONST_BITS = 13;
const PASS1_BITS = 2;

/** ``IDCT_range_limit`` indexed by ``value & 0x3ff``. */
const IDCT_LIMIT = new Uint8Array(1024);
for (let i = 0; i < 1024; i += 1) IDCT_LIMIT[i] = i < 128 ? i + 128 : i < 512 ? 255 : i < 896 ? 0 : i - 896;

function descale(value: number, shift: number): number {
  return Math.floor((value + 2 ** (shift - 1)) / 2 ** shift);
}

const workspace = new Float64Array(64);

/** ``jpeg_idct_islow``: dequantize, inverse DCT and range-limit one block. */
function idctBlock(coefficients: Int16Array, offset: number, quant: Int32Array, out: Uint8Array, outOffset: number, stride: number): void {
  const ws = workspace;
  for (let column = 0; column < 8; column += 1) {
    const c = (index: number): number => coefficients[offset + index * 8 + column]! * quant[index * 8 + column]!;
    if (!coefficients[offset + 8 + column] && !coefficients[offset + 16 + column] && !coefficients[offset + 24 + column]
      && !coefficients[offset + 32 + column] && !coefficients[offset + 40 + column] && !coefficients[offset + 48 + column]
      && !coefficients[offset + 56 + column]) {
      const dc = c(0) * 4;
      for (let row = 0; row < 8; row += 1) ws[row * 8 + column] = dc;
      continue;
    }
    let z2 = c(2);
    let z3 = c(6);
    let z1 = (z2 + z3) * FIX_0_541196100;
    let tmp2 = z1 + z3 * -FIX_1_847759065;
    let tmp3 = z1 + z2 * FIX_0_765366865;
    z2 = c(0);
    z3 = c(4);
    let tmp0 = (z2 + z3) * 8192;
    let tmp1 = (z2 - z3) * 8192;
    const tmp10 = tmp0 + tmp3;
    const tmp13 = tmp0 - tmp3;
    const tmp11 = tmp1 + tmp2;
    const tmp12 = tmp1 - tmp2;
    tmp0 = c(7);
    tmp1 = c(5);
    tmp2 = c(3);
    tmp3 = c(1);
    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    let z4 = tmp1 + tmp3;
    const z5 = (z3 + z4) * FIX_1_175875602;
    tmp0 *= FIX_0_298631336;
    tmp1 *= FIX_2_053119869;
    tmp2 *= FIX_3_072711026;
    tmp3 *= FIX_1_501321110;
    z1 *= -FIX_0_899976223;
    z2 *= -FIX_2_562915447;
    z3 *= -FIX_1_961570560;
    z4 *= -FIX_0_390180644;
    z3 += z5;
    z4 += z5;
    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;
    const shift = CONST_BITS - PASS1_BITS;
    ws[column] = descale(tmp10 + tmp3, shift);
    ws[56 + column] = descale(tmp10 - tmp3, shift);
    ws[8 + column] = descale(tmp11 + tmp2, shift);
    ws[48 + column] = descale(tmp11 - tmp2, shift);
    ws[16 + column] = descale(tmp12 + tmp1, shift);
    ws[40 + column] = descale(tmp12 - tmp1, shift);
    ws[24 + column] = descale(tmp13 + tmp0, shift);
    ws[32 + column] = descale(tmp13 - tmp0, shift);
  }
  const shift = CONST_BITS + PASS1_BITS + 3;
  for (let row = 0; row < 8; row += 1) {
    const w = row * 8;
    const o = outOffset + row * stride;
    if (!ws[w + 1] && !ws[w + 2] && !ws[w + 3] && !ws[w + 4] && !ws[w + 5] && !ws[w + 6] && !ws[w + 7]) {
      const value = IDCT_LIMIT[descale(ws[w]!, PASS1_BITS + 3) & 0x3ff]!;
      for (let i = 0; i < 8; i += 1) out[o + i] = value;
      continue;
    }
    let z2 = ws[w + 2]!;
    let z3 = ws[w + 6]!;
    let z1 = (z2 + z3) * FIX_0_541196100;
    let tmp2 = z1 + z3 * -FIX_1_847759065;
    let tmp3 = z1 + z2 * FIX_0_765366865;
    let tmp0 = (ws[w]! + ws[w + 4]!) * 8192;
    let tmp1 = (ws[w]! - ws[w + 4]!) * 8192;
    const tmp10 = tmp0 + tmp3;
    const tmp13 = tmp0 - tmp3;
    const tmp11 = tmp1 + tmp2;
    const tmp12 = tmp1 - tmp2;
    tmp0 = ws[w + 7]!;
    tmp1 = ws[w + 5]!;
    tmp2 = ws[w + 3]!;
    tmp3 = ws[w + 1]!;
    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    let z4 = tmp1 + tmp3;
    const z5 = (z3 + z4) * FIX_1_175875602;
    tmp0 *= FIX_0_298631336;
    tmp1 *= FIX_2_053119869;
    tmp2 *= FIX_3_072711026;
    tmp3 *= FIX_1_501321110;
    z1 *= -FIX_0_899976223;
    z2 *= -FIX_2_562915447;
    z3 *= -FIX_1_961570560;
    z4 *= -FIX_0_390180644;
    z3 += z5;
    z4 += z5;
    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;
    out[o] = IDCT_LIMIT[descale(tmp10 + tmp3, shift) & 0x3ff]!;
    out[o + 7] = IDCT_LIMIT[descale(tmp10 - tmp3, shift) & 0x3ff]!;
    out[o + 1] = IDCT_LIMIT[descale(tmp11 + tmp2, shift) & 0x3ff]!;
    out[o + 6] = IDCT_LIMIT[descale(tmp11 - tmp2, shift) & 0x3ff]!;
    out[o + 2] = IDCT_LIMIT[descale(tmp12 + tmp1, shift) & 0x3ff]!;
    out[o + 5] = IDCT_LIMIT[descale(tmp12 - tmp1, shift) & 0x3ff]!;
    out[o + 3] = IDCT_LIMIT[descale(tmp13 + tmp0, shift) & 0x3ff]!;
    out[o + 4] = IDCT_LIMIT[descale(tmp13 - tmp0, shift) & 0x3ff]!;
  }
}

// ---------------------------------------------------------------------------
// Colour conversion tables (jdcolor.c).
// ---------------------------------------------------------------------------

const SCALEBITS = 16;
const ONE_HALF = 1 << (SCALEBITS - 1);
const fix = (value: number): number => Math.floor(value * 65536 + 0.5);
const CR_R = new Int32Array(256);
const CB_B = new Int32Array(256);
const CR_G = new Float64Array(256);
const CB_G = new Float64Array(256);
const RGB_Y_R = new Float64Array(256);
const RGB_Y_G = new Float64Array(256);
const RGB_Y_B = new Float64Array(256);
for (let i = 0; i < 256; i += 1) {
  const x = i - 128;
  CR_R[i] = Math.floor((fix(1.402) * x + ONE_HALF) / 65536);
  CB_B[i] = Math.floor((fix(1.772) * x + ONE_HALF) / 65536);
  CR_G[i] = -fix(0.71414) * x;
  CB_G[i] = -fix(0.34414) * x + ONE_HALF;
  RGB_Y_R[i] = fix(0.299) * i;
  RGB_Y_G[i] = fix(0.587) * i;
  RGB_Y_B[i] = fix(0.114) * i + ONE_HALF;
}

function clamp8(value: number): number {
  return value < 0 ? 0 : value > 255 ? 255 : value;
}

// ---------------------------------------------------------------------------
// Parsing and entropy decoding.
// ---------------------------------------------------------------------------

function u16(bytes: Uint8Array, offset: number): number {
  return (bytes[offset]! << 8) | bytes[offset + 1]!;
}

export function isJpeg(bytes: Uint8Array): boolean {
  return bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
}

/** Decode a JPEG file's entropy-coded data into coefficient blocks. */
export function decodeJpeg(bytes: Uint8Array): JpegData {
  if (bytes.length < 2 || bytes[0] !== 0xff || bytes[1] !== 0xd8) throw new ValueError('Not a JPEG file: starts with 0x' + (bytes[0] ?? 0).toString(16));
  const quantTables: (Int32Array | null)[] = [null, null, null, null];
  const dcTables: (HuffmanTable | null)[] = [null, null, null, null];
  const acTables: (HuffmanTable | null)[] = [null, null, null, null];
  let components: Component[] = [];
  let width = 0;
  let height = 0;
  let maxH = 1;
  let maxV = 1;
  let progressive = false;
  let frameSeen = false;
  let restartInterval = 0;
  let arithmetic = false;
  let lossless = false;
  let precision = 8;
  const arithDcL = new Uint8Array(16);
  const arithDcU = new Uint8Array(16).fill(1);
  const arithAcK = new Uint8Array(16).fill(5);
  let sawJfif = false;
  let adobeTransform: number | null = null;
  let exif: Uint8Array | null = null;
  let firstApp1: Uint8Array | null = null;
  let xmp: Uint8Array | null = null;
  let sawScan = false;
  let offset = 2;

  const nextMarker = (): number => {
    // Skip any garbage and fill bytes before a marker (libjpeg next_marker).
    while (offset < bytes.length && bytes[offset] !== 0xff) offset += 1;
    while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
    if (offset >= bytes.length) return -1;
    return bytes[offset++]!;
  };

  for (;;) {
    const marker = nextMarker();
    if (marker < 0) {
      if (!sawScan) throw new ValueError('Premature end of JPEG file');
      break;
    }
    if (marker === 0xd9) break; // EOI
    if (marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01) continue;
    if (offset + 2 > bytes.length) throw new ValueError('Premature end of JPEG file');
    const length = u16(bytes, offset);
    const start = offset + 2;
    const end = offset + length;
    if (length < 2 || end > bytes.length) {
      if (marker !== 0xda) throw new ValueError('Premature end of JPEG file');
    }
    const segment = bytes.subarray(start, Math.min(end, bytes.length));
    switch (marker) {
      case 0xe0: // APP0
        if (segment.length >= 5 && segment[0] === 0x4a && segment[1] === 0x46 && segment[2] === 0x49 && segment[3] === 0x46 && segment[4] === 0) sawJfif = true;
        break;
      case 0xe1: { // APP1
        firstApp1 ??= segment.slice();
        if (!exif && segment.length >= 6 && segment[0] === 0x45 && segment[1] === 0x78 && segment[2] === 0x69 && segment[3] === 0x66 && segment[4] === 0 && segment[5] === 0) {
          exif = segment.slice();
        }
        const xmpHeader = 'http://ns.adobe.com/xap/1.0/\0';
        if (!xmp && segment.length >= xmpHeader.length && [...xmpHeader].every((ch, i) => segment[i] === ch.charCodeAt(0))) {
          xmp = segment.slice(xmpHeader.length);
        }
        break;
      }
      case 0xee: // APP14
        if (segment.length >= 12 && segment[0] === 0x41 && segment[1] === 0x64 && segment[2] === 0x6f && segment[3] === 0x62 && segment[4] === 0x65) {
          adobeTransform = segment[11]!;
        }
        break;
      case 0xdb: { // DQT
        let p = 0;
        while (p < segment.length) {
          const precision = segment[p]! >> 4;
          const index = segment[p]! & 15;
          if (index > 3) throw new ValueError(`Bogus DQT index ${index}`);
          p += 1;
          const table = new Int32Array(64);
          for (let k = 0; k < 64; k += 1) {
            table[NATURAL_ORDER[k]!] = precision ? u16(segment, p + k * 2) : segment[p + k]!;
          }
          p += precision ? 128 : 64;
          quantTables[index] = table;
        }
        break;
      }
      case 0xc4: { // DHT
        let p = 0;
        while (p + 17 <= segment.length) {
          const info = segment[p]!;
          const counts = segment.subarray(p + 1, p + 17);
          let total = 0;
          for (let i = 0; i < 16; i += 1) total += counts[i]!;
          if (total > 256) throw new ValueError('Bogus Huffman table definition');
          const values = segment.slice(p + 17, p + 17 + total);
          p += 17 + total;
          const table = buildHuffman(counts, values);
          const index = info & 15;
          if (index > 3) throw new ValueError(`Bogus DHT index ${index}`);
          if (info & 0x10) acTables[index] = table;
          else dcTables[index] = table;
        }
        break;
      }
      case 0xdd: // DRI
        restartInterval = u16(segment, 0);
        break;
      case 0xc0: case 0xc1: case 0xc2: case 0xc3: case 0xc9: case 0xca: { // SOF0/1/2/3/9/10
        if (frameSeen) throw new ValueError('Invalid JPEG file structure: two SOF markers');
        frameSeen = true;
        progressive = marker === 0xc2 || marker === 0xca;
        arithmetic = marker === 0xc9 || marker === 0xca;
        lossless = marker === 0xc3;
        precision = segment[0]!;
        // The 8-bit libjpeg-turbo API decodes 8-bit lossy data and 2-8 bit lossless data.
        if (lossless ? precision < 2 || precision > 8 : precision !== 8) throw new ValueError(`Unsupported JPEG data precision ${precision}`);
        height = u16(segment, 1);
        width = u16(segment, 3);
        const count = segment[5]!;
        if (height === 0 || width === 0 || count === 0) throw new ValueError('Empty JPEG image (DNL not supported)');
        components = [];
        for (let i = 0; i < count; i += 1) {
          const base = 6 + i * 3;
          const h = segment[base + 1]! >> 4;
          const v = segment[base + 1]! & 15;
          if (h < 1 || h > 4 || v < 1 || v > 4) throw new ValueError('Bogus sampling factors');
          components.push({
            id: segment[base]!, h, v, tq: segment[base + 2]!, blocksPerLine: 0, blocksPerColumn: 0, paddedBlocksPerLine: 0,
            paddedBlocksPerColumn: 0, coefficients: new Int16Array(0), quant: null, dcPredictor: 0, downsampledWidth: 0,
            downsampledHeight: 0, dcTable: null, acTable: null, dcIndex: 0, acIndex: 0, samples: null,
          });
        }
        maxH = Math.max(...components.map((c) => c.h));
        maxV = Math.max(...components.map((c) => c.v));
        const blockSize = lossless ? 1 : 8;
        const mcusPerLine = Math.ceil(width / (blockSize * maxH));
        const mcusPerColumn = Math.ceil(height / (blockSize * maxV));
        for (const component of components) {
          component.downsampledWidth = Math.ceil((width * component.h) / maxH);
          component.downsampledHeight = Math.ceil((height * component.v) / maxV);
          component.blocksPerLine = Math.ceil(component.downsampledWidth / blockSize);
          component.blocksPerColumn = Math.ceil(component.downsampledHeight / blockSize);
          component.paddedBlocksPerLine = Math.max(mcusPerLine * component.h, component.blocksPerLine);
          component.paddedBlocksPerColumn = Math.max(mcusPerColumn * component.v, component.blocksPerColumn);
          if (lossless) component.samples = new Uint8Array(component.downsampledWidth * component.downsampledHeight);
          else component.coefficients = new Int16Array(component.paddedBlocksPerLine * component.paddedBlocksPerColumn * 64);
        }
        break;
      }
      case 0xcb: // SOF11: lossless arithmetic (libjpeg-turbo: JERR_ARITH_NOTIMPL)
        throw new ValueError('Sorry, arithmetic coding is not implemented');
      case 0xc5: case 0xc6: case 0xc7: case 0xcd: case 0xce: case 0xcf:
        throw new ValueError(`Unsupported JPEG process: SOF type 0x${marker.toString(16)}`);
      case 0xcc: { // DAC
        for (let p = 0; p + 1 < segment.length; p += 2) {
          const index = segment[p]!;
          const value = segment[p + 1]!;
          // NUM_ARITH_TBLS = 16: indices 0-15 condition DC tables, 16-31 AC tables.
          if (index >= 32) throw new ValueError(`Bogus DAC index ${index}`);
          if (index >= 16) arithAcK[index - 16] = value;
          else {
            arithDcL[index] = value & 0x0f;
            arithDcU[index] = value >> 4;
            if (arithDcL[index]! > arithDcU[index]!) throw new ValueError(`Bogus DAC value 0x${value.toString(16)}`);
          }
        }
        break;
      }
      case 0xda: { // SOS
        if (!frameSeen) throw new ValueError('Invalid JPEG file structure: SOS before SOF');
        const count = segment[0]!;
        const scan: Component[] = [];
        for (let i = 0; i < count; i += 1) {
          const id = segment[1 + i * 2]!;
          const tables = segment[2 + i * 2]!;
          const component = components.find((c) => c.id === id);
          if (!component) throw new ValueError(`Invalid component ID ${id} in SOS`);
          component.dcTable = dcTables[tables >> 4] ?? null;
          component.acTable = acTables[tables & 15] ?? null;
          component.dcIndex = tables >> 4;
          component.acIndex = tables & 15;
          // latch_quant_tables: the table in effect at the component's first scan.
          if (!lossless && !component.quant) {
            const table = quantTables[component.tq];
            if (!table) throw new ValueError(`Quantization table 0x${component.tq.toString(16)} was not defined`);
            component.quant = table.slice();
          }
          scan.push(component);
        }
        const p = 1 + count * 2;
        const ss = segment[p]!;
        const se = segment[p + 1]!;
        const ah = segment[p + 2]! >> 4;
        const al = segment[p + 2]! & 15;
        offset = end;
        const params: ScanParameters = { width, height, maxH, maxV, progressive, ss, se, ah, al, restartInterval };
        if (lossless) offset = decodeLosslessScan(bytes, offset, scan, params, precision);
        else if (arithmetic) offset = decodeArithmeticScan(bytes, offset, scan, params, { dcL: arithDcL, dcU: arithDcU, acK: arithAcK });
        else offset = decodeScan(bytes, offset, scan, params);
        sawScan = true;
        continue;
      }
      default:
        break;
    }
    offset = end;
  }
  if (!frameSeen || !sawScan) throw new ValueError('JPEG datastream contains no image');

  let colorSpace: JpegColorSpace;
  if (components.length === 1) colorSpace = 'GRAYSCALE';
  else if (components.length === 3) {
    if (sawJfif) colorSpace = 'YCbCr';
    else if (adobeTransform !== null) colorSpace = adobeTransform === 0 ? 'RGB' : 'YCbCr';
    else {
      const [a, b, c] = components.map((component) => component.id);
      // Without markers, lossless files are assumed RGB (libjpeg-turbo default_decompress_parms).
      colorSpace = lossless || (a === 82 && b === 71 && c === 66) ? 'RGB' : 'YCbCr';
    }
  } else if (components.length === 4) {
    colorSpace = adobeTransform !== null ? (adobeTransform === 0 ? 'CMYK' : 'YCCK') : 'CMYK';
  } else colorSpace = 'UNKNOWN';

  let planes: Uint8Array[] | null = null;
  const upsampled = (): Uint8Array[] => (planes ??= components.map((component) => upsample(component, width, height, maxH, maxV, !lossless)));
  return {
    width, height, colorSpace, components: components.length, progressive, adobeTransform, lossless, arithmetic, precision,
    exif, firstApp1, xmp,
    output(space) {
      // Lossless images allow no colour conversion (jdcolor.c).
      if (lossless && colorSpace !== space) throw new ValueError('Unsupported color conversion request');
      return convertColor(upsampled(), colorSpace, space, width * height);
    },
  };
}


interface ScanParameters {
  width: number;
  height: number;
  maxH: number;
  maxV: number;
  progressive: boolean;
  ss: number;
  se: number;
  ah: number;
  al: number;
  restartInterval: number;
}

function decodeScan(bytes: Uint8Array, offset: number, scan: Component[], params: ScanParameters): number {
  const { progressive, ss, se, ah, al, restartInterval } = params;
  const reader = new BitReader(bytes, offset);
  for (const component of scan) component.dcPredictor = 0;
  let eobrun = 0;
  const p1 = 1 << al;
  const m1 = -1 << al;

  let decodeBlock: (component: Component, block: number) => void;
  if (!progressive) {
    decodeBlock = (component, block) => {
      const coefficients = component.coefficients;
      let s = component.dcTable ? reader.decode(component.dcTable) : 0;
      let diff = 0;
      if (s) diff = extend(reader.getBits(s), s);
      component.dcPredictor += diff;
      coefficients[block] = component.dcPredictor;
      for (let k = 1; k < 64; k += 1) {
        const rs = component.acTable ? reader.decode(component.acTable) : 0;
        const r = rs >> 4;
        s = rs & 15;
        if (s) {
          k += r;
          const value = extend(reader.getBits(s), s);
          coefficients[block + NATURAL_ORDER[k]!] = value;
        } else {
          if (r !== 15) break;
          k += 15;
        }
      }
    };
  } else if (ss === 0) {
    if (ah === 0) {
      decodeBlock = (component, block) => {
        const s = component.dcTable ? reader.decode(component.dcTable) : 0;
        const diff = s ? extend(reader.getBits(s), s) : 0;
        component.dcPredictor += diff;
        component.coefficients[block] = component.dcPredictor * 2 ** al;
      };
    } else {
      decodeBlock = (component, block) => {
        if (reader.getBits(1)) component.coefficients[block] = component.coefficients[block]! | p1;
      };
    }
  } else if (ah === 0) {
    decodeBlock = (component, block) => {
      if (eobrun > 0) {
        eobrun -= 1;
        return;
      }
      const coefficients = component.coefficients;
      for (let k = ss; k <= se; k += 1) {
        const rs = component.acTable ? reader.decode(component.acTable) : 0;
        let r = rs >> 4;
        const s = rs & 15;
        if (s) {
          k += r;
          const value = extend(reader.getBits(s), s);
          coefficients[block + NATURAL_ORDER[k]!] = value * 2 ** al;
        } else if (r === 15) {
          k += 15;
        } else {
          eobrun = 1 << r;
          if (r) {
            r = reader.getBits(r);
            eobrun += r;
          }
          eobrun -= 1;
          break;
        }
      }
    };
  } else {
    decodeBlock = (component, block) => {
      const coefficients = component.coefficients;
      let k = ss;
      if (eobrun === 0) {
        for (; k <= se; k += 1) {
          const rs = component.acTable ? reader.decode(component.acTable) : 0;
          let r = rs >> 4;
          let s = rs & 15;
          if (s) {
            s = reader.getBits(1) ? p1 : m1;
          } else if (r !== 15) {
            eobrun = 1 << r;
            if (r) eobrun += reader.getBits(r);
            break;
          }
          do {
            const position = block + NATURAL_ORDER[k]!;
            const coefficient = coefficients[position]!;
            if (coefficient !== 0) {
              if (reader.getBits(1)) {
                if ((coefficient & p1) === 0) coefficients[position] = coefficient >= 0 ? coefficient + p1 : coefficient + m1;
              }
            } else {
              r -= 1;
              if (r < 0) break;
            }
            k += 1;
          } while (k <= se);
          if (s) coefficients[block + NATURAL_ORDER[k]!] = s;
        }
      }
      if (eobrun > 0) {
        for (; k <= se; k += 1) {
          const position = block + NATURAL_ORDER[k]!;
          const coefficient = coefficients[position]!;
          if (coefficient !== 0 && reader.getBits(1)) {
            if ((coefficient & p1) === 0) coefficients[position] = coefficient >= 0 ? coefficient + p1 : coefficient + m1;
          }
        }
        eobrun -= 1;
      }
    };
  }

  let restartsLeft = restartInterval;
  const handleRestart = (): void => {
    if (!restartInterval) return;
    if (restartsLeft === 0) {
      reader.restart();
      for (const component of scan) component.dcPredictor = 0;
      eobrun = 0;
      restartsLeft = restartInterval;
    }
    restartsLeft -= 1;
  };

  if (scan.length === 1) {
    const component = scan[0]!;
    for (let row = 0; row < component.blocksPerColumn; row += 1) {
      for (let column = 0; column < component.blocksPerLine; column += 1) {
        handleRestart();
        decodeBlock(component, (row * component.paddedBlocksPerLine + column) * 64);
      }
    }
  } else {
    const mcusPerLine = Math.ceil(params.width / (8 * params.maxH));
    const mcusPerColumn = Math.ceil(params.height / (8 * params.maxV));
    for (let mcuRow = 0; mcuRow < mcusPerColumn; mcuRow += 1) {
      for (let mcuColumn = 0; mcuColumn < mcusPerLine; mcuColumn += 1) {
        handleRestart();
        for (const component of scan) {
          for (let v = 0; v < component.v; v += 1) {
            for (let h = 0; h < component.h; h += 1) {
              const row = mcuRow * component.v + v;
              const column = mcuColumn * component.h + h;
              decodeBlock(component, (row * component.paddedBlocksPerLine + column) * 64);
            }
          }
        }
      }
    }
  }
  // Continue after the entropy-coded segment: find the next non-RST marker.
  let position = reader.position;
  while (position + 1 < bytes.length) {
    if (bytes[position] === 0xff) {
      const next = bytes[position + 1]!;
      if (next !== 0 && next !== 0xff && !(next >= 0xd0 && next <= 0xd7)) return position;
    }
    position += 1;
  }
  return bytes.length;
}

/** Position after an entropy-coded segment: the next marker that is not RSTn. */
function segmentEnd(bytes: Uint8Array, position: number): number {
  let p = position;
  while (p + 1 < bytes.length) {
    if (bytes[p] === 0xff) {
      const next = bytes[p + 1]!;
      if (next !== 0 && next !== 0xff && !(next >= 0xd0 && next <= 0xd7)) return p;
    }
    p += 1;
  }
  return bytes.length;
}

// ---------------------------------------------------------------------------
// Arithmetic entropy decoding (jdarith.c, ITU-T T.81 Annex D/F/G).
// ---------------------------------------------------------------------------

/** ``jpeg_aritab``: ``[Qe, next LPS, next MPS, switch MPS]`` rows (T.81 Table D.2). */
const ARITAB: readonly (readonly [number, number, number, number])[] = [
  [0x5a1d, 1, 1, 1], [0x2586, 14, 2, 0], [0x1114, 16, 3, 0], [0x080b, 18, 4, 0], [0x03d8, 20, 5, 0], [0x01da, 23, 6, 0],
  [0x00e5, 25, 7, 0], [0x006f, 28, 8, 0], [0x0036, 30, 9, 0], [0x001a, 33, 10, 0], [0x000d, 35, 11, 0], [0x0006, 9, 12, 0],
  [0x0003, 10, 13, 0], [0x0001, 12, 13, 0], [0x5a7f, 15, 15, 1], [0x3f25, 36, 16, 0], [0x2cf2, 38, 17, 0], [0x207c, 39, 18, 0],
  [0x17b9, 40, 19, 0], [0x1182, 42, 20, 0], [0x0cef, 43, 21, 0], [0x09a1, 45, 22, 0], [0x072f, 46, 23, 0], [0x055c, 48, 24, 0],
  [0x0406, 49, 25, 0], [0x0303, 51, 26, 0], [0x0240, 52, 27, 0], [0x01b1, 54, 28, 0], [0x0144, 56, 29, 0], [0x00f5, 57, 30, 0],
  [0x00b7, 59, 31, 0], [0x008a, 60, 32, 0], [0x0068, 62, 33, 0], [0x004e, 63, 34, 0], [0x003b, 32, 35, 0], [0x002c, 33, 9, 0],
  [0x5ae1, 37, 37, 1], [0x484c, 64, 38, 0], [0x3a0d, 65, 39, 0], [0x2ef1, 67, 40, 0], [0x261f, 68, 41, 0], [0x1f33, 69, 42, 0],
  [0x19a8, 70, 43, 0], [0x1518, 72, 44, 0], [0x1177, 73, 45, 0], [0x0e74, 74, 46, 0], [0x0bfb, 75, 47, 0], [0x09f8, 77, 48, 0],
  [0x0861, 78, 49, 0], [0x0706, 79, 50, 0], [0x05cd, 48, 51, 0], [0x04de, 50, 52, 0], [0x040f, 50, 53, 0], [0x0363, 51, 54, 0],
  [0x02d4, 52, 55, 0], [0x025c, 53, 56, 0], [0x01f8, 54, 57, 0], [0x01a4, 55, 58, 0], [0x0160, 56, 59, 0], [0x0125, 57, 60, 0],
  [0x00f6, 58, 61, 0], [0x00cb, 59, 62, 0], [0x00ab, 61, 63, 0], [0x008f, 61, 32, 0], [0x5b12, 65, 65, 1], [0x4d04, 80, 66, 0],
  [0x412c, 81, 67, 0], [0x37d8, 82, 68, 0], [0x2fe8, 83, 69, 0], [0x293c, 84, 70, 0], [0x2379, 86, 71, 0], [0x1edf, 87, 72, 0],
  [0x1aa9, 87, 73, 0], [0x174e, 72, 74, 0], [0x1424, 72, 75, 0], [0x119c, 74, 76, 0], [0x0f6b, 74, 77, 0], [0x0d51, 75, 78, 0],
  [0x0bb6, 77, 79, 0], [0x0a40, 77, 48, 0], [0x5832, 80, 81, 1], [0x4d1c, 88, 82, 0], [0x438e, 89, 83, 0], [0x3bdd, 90, 84, 0],
  [0x34ee, 91, 85, 0], [0x2eae, 92, 86, 0], [0x299a, 93, 87, 0], [0x2516, 86, 71, 0], [0x5570, 88, 89, 1], [0x4ca9, 95, 90, 0],
  [0x44d9, 96, 91, 0], [0x3e22, 97, 92, 0], [0x3824, 99, 93, 0], [0x32b4, 99, 94, 0], [0x2e17, 93, 86, 0], [0x56a8, 95, 96, 1],
  [0x4f46, 101, 97, 0], [0x47e5, 102, 98, 0], [0x41cf, 103, 99, 0], [0x3c3d, 104, 100, 0], [0x375e, 99, 93, 0], [0x5231, 105, 102, 0],
  [0x4c0f, 106, 103, 0], [0x4639, 107, 104, 0], [0x415e, 103, 99, 0], [0x5627, 105, 106, 1], [0x50e7, 108, 107, 0], [0x4b85, 109, 103, 0],
  [0x5597, 110, 109, 0], [0x504f, 111, 107, 0], [0x5a10, 110, 111, 1], [0x5522, 112, 109, 0], [0x59eb, 112, 111, 1],
  [0x5a1d, 113, 113, 0], // fixed 0.5 estimate (T.851 Table 5)
];

class ArithmeticReader {
  private c = 0;
  private a = 0;
  private ct = -16;
  private marker = false;
  /** Set on a magnitude or spectral overflow (libjpeg's ``ct = -1``): the rest of the scan is skipped. */
  failed = false;

  constructor(private readonly data: Uint8Array, public position: number) {}

  private byte(): number {
    if (this.position >= this.data.length) return 0;
    return this.data[this.position++]!;
  }

  /** ``arith_decode``: one binary decision with the adaptive estimate in ``stats[index]``. */
  decode(stats: Uint8Array, index: number): number {
    while (this.a < 0x8000) {
      this.ct -= 1;
      if (this.ct < 0) {
        let data = 0;
        if (!this.marker) {
          data = this.byte();
          if (data === 0xff) {
            do data = this.byte(); while (data === 0xff);
            if (data === 0) data = 0xff;
            else {
              this.marker = true;
              this.position -= 2;
              data = 0;
            }
          }
        }
        this.c = this.c * 256 + data;
        this.ct += 8;
        if (this.ct < 0) {
          this.ct += 1;
          if (this.ct === 0) this.a = 0x8000;
        }
      }
      this.a *= 2;
    }
    const sv = stats[index]!;
    const [qe, nextLps, nextMps] = ARITAB[sv & 0x7f]!;
    const switchMps = ARITAB[sv & 0x7f]![3];
    const nl = nextLps | (switchMps << 7);
    const nm = nextMps;
    let bit = sv >> 7;
    let temp = this.a - qe;
    this.a = temp;
    temp *= 2 ** this.ct;
    if (this.c >= temp) {
      this.c -= temp;
      if (this.a < qe) {
        this.a = qe;
        stats[index] = (sv & 0x80) ^ nm;
      } else {
        this.a = qe;
        stats[index] = (sv & 0x80) ^ nl;
        bit ^= 1;
      }
    } else if (this.a < 0x8000) {
      if (this.a < qe) {
        stats[index] = (sv & 0x80) ^ nl;
        bit ^= 1;
      } else {
        stats[index] = (sv & 0x80) ^ nm;
      }
    }
    return bit;
  }

  /** ``process_restart``: skip the RSTn marker and reset the decoder. */
  restart(): void {
    let p = this.position;
    while (p + 1 < this.data.length) {
      if (this.data[p] === 0xff) {
        const next = this.data[p + 1]!;
        if (next >= 0xd0 && next <= 0xd7) {
          p += 2;
          break;
        }
        if (next !== 0 && next !== 0xff) break;
      }
      p += 1;
    }
    this.position = p;
    this.c = 0;
    this.a = 0;
    this.ct = -16;
    this.marker = false;
    this.failed = false;
  }
}

interface ArithmeticConditioning {
  dcL: Uint8Array;
  dcU: Uint8Array;
  acK: Uint8Array;
}

function decodeArithmeticScan(bytes: Uint8Array, offset: number, scan: Component[], params: ScanParameters, conditioning: ArithmeticConditioning): number {
  const { progressive, ss, se, ah, al, restartInterval } = params;
  const reader = new ArithmeticReader(bytes, offset);
  const dcStats = new Map<number, Uint8Array>();
  const acStats = new Map<number, Uint8Array>();
  const fixedBin = Uint8Array.of(113);
  const lastDc = new Int32Array(scan.length);
  const dcContext = new Int32Array(scan.length);
  const resetStats = (): void => {
    scan.forEach((component, ci) => {
      if (!progressive || (ss === 0 && ah === 0)) {
        dcStats.set(component.dcIndex, new Uint8Array(64));
        lastDc[ci] = 0;
        dcContext[ci] = 0;
      }
      if (!progressive || ss) acStats.set(component.acIndex, new Uint8Array(256));
    });
  };
  resetStats();
  const p1 = 1 << al;
  const m1 = -1 << al;

  const decodeDc = (ci: number, component: Component): number | null => {
    const tbl = component.dcIndex;
    const stats = dcStats.get(tbl)!;
    let st = dcContext[ci]!;
    if (reader.decode(stats, st) === 0) {
      dcContext[ci] = 0;
    } else {
      const sign = reader.decode(stats, st + 1);
      st += 2 + sign;
      let m = reader.decode(stats, st);
      if (m !== 0) {
        st = 20;
        while (reader.decode(stats, st)) {
          m <<= 1;
          if (m === 0x8000) {
            reader.failed = true;
            return null;
          }
          st += 1;
        }
      }
      if (m < ((1 << conditioning.dcL[tbl]!) >> 1)) dcContext[ci] = 0;
      else if (m > ((1 << conditioning.dcU[tbl]!) >> 1)) dcContext[ci] = 12 + sign * 4;
      else dcContext[ci] = 4 + sign * 4;
      let v = m;
      st += 14;
      while ((m >>= 1)) if (reader.decode(stats, st)) v |= m;
      v += 1;
      if (sign) v = -v;
      lastDc[ci] = (lastDc[ci]! + v) & 0xffff;
    }
    return lastDc[ci]!;
  };

  /** AC coefficients ``k = start..end`` of one block (sequential and progressive first scans). */
  const decodeAc = (component: Component, block: number, start: number, end: number, shift: number): void => {
    const tbl = component.acIndex;
    const stats = acStats.get(tbl)!;
    const coefficients = component.coefficients;
    for (let k = start; k <= end; k += 1) {
      let st = 3 * (k - 1);
      if (reader.decode(stats, st)) break; // EOB
      while (reader.decode(stats, st + 1) === 0) {
        st += 3;
        k += 1;
        if (k > end) {
          reader.failed = true;
          return;
        }
      }
      const sign = reader.decode(fixedBin, 0);
      st += 2;
      let m = reader.decode(stats, st);
      if (m !== 0) {
        if (reader.decode(stats, st)) {
          m <<= 1;
          st = k <= conditioning.acK[tbl]! ? 189 : 217;
          while (reader.decode(stats, st)) {
            m <<= 1;
            if (m === 0x8000) {
              reader.failed = true;
              return;
            }
            st += 1;
          }
        }
      }
      let v = m;
      st += 14;
      while ((m >>= 1)) if (reader.decode(stats, st)) v |= m;
      v += 1;
      if (sign) v = -v;
      coefficients[block + NATURAL_ORDER[k]!] = v * 2 ** shift;
    }
  };

  let decodeBlock: (component: Component, ci: number, block: number) => void;
  if (!progressive) {
    decodeBlock = (component, ci, block) => {
      const dc = decodeDc(ci, component);
      if (dc === null) return;
      component.coefficients[block] = dc;
      decodeAc(component, block, 1, 63, 0);
    };
  } else if (ss === 0 && ah === 0) {
    decodeBlock = (component, ci, block) => {
      const dc = decodeDc(ci, component);
      if (dc !== null) component.coefficients[block] = dc * 2 ** al;
    };
  } else if (ss === 0) {
    decodeBlock = (component, _ci, block) => {
      if (reader.decode(fixedBin, 0)) component.coefficients[block] = component.coefficients[block]! | p1;
    };
  } else if (ah === 0) {
    decodeBlock = (component, _ci, block) => decodeAc(component, block, ss, se, al);
  } else {
    decodeBlock = (component, _ci, block) => {
      const tbl = component.acIndex;
      const stats = acStats.get(tbl)!;
      const coefficients = component.coefficients;
      let kex = se;
      for (; kex > 0; kex -= 1) if (coefficients[block + NATURAL_ORDER[kex]!]) break;
      for (let k = ss; k <= se; k += 1) {
        let st = 3 * (k - 1);
        if (k > kex && reader.decode(stats, st)) break;
        for (;;) {
          const position = block + NATURAL_ORDER[k]!;
          const coefficient = coefficients[position]!;
          if (coefficient) {
            if (reader.decode(stats, st + 2)) coefficients[position] = coefficient < 0 ? coefficient + m1 : coefficient + p1;
            break;
          }
          if (reader.decode(stats, st + 1)) {
            coefficients[position] = reader.decode(fixedBin, 0) ? m1 : p1;
            break;
          }
          st += 3;
          k += 1;
          if (k > se) {
            reader.failed = true;
            return;
          }
        }
      }
    };
  }

  let restartsLeft = restartInterval;
  const beforeMcu = (): boolean => {
    if (restartInterval) {
      if (restartsLeft === 0) {
        reader.restart();
        resetStats();
        restartsLeft = restartInterval;
      }
      restartsLeft -= 1;
    }
    return !reader.failed || (progressive && ss === 0 && ah !== 0);
  };

  if (scan.length === 1) {
    const component = scan[0]!;
    for (let row = 0; row < component.blocksPerColumn; row += 1) {
      for (let column = 0; column < component.blocksPerLine; column += 1) {
        if (!beforeMcu()) continue;
        decodeBlock(component, 0, (row * component.paddedBlocksPerLine + column) * 64);
      }
    }
  } else {
    const mcusPerLine = Math.ceil(params.width / (8 * params.maxH));
    const mcusPerColumn = Math.ceil(params.height / (8 * params.maxV));
    for (let mcuRow = 0; mcuRow < mcusPerColumn; mcuRow += 1) {
      for (let mcuColumn = 0; mcuColumn < mcusPerLine; mcuColumn += 1) {
        if (!beforeMcu()) continue;
        scan.forEach((component, ci) => {
          for (let v = 0; v < component.v; v += 1) {
            for (let h = 0; h < component.h; h += 1) {
              if (reader.failed && !(progressive && ss === 0 && ah !== 0)) return;
              const row = mcuRow * component.v + v;
              const column = mcuColumn * component.h + h;
              decodeBlock(component, ci, (row * component.paddedBlocksPerLine + column) * 64);
            }
          }
        });
      }
    }
  }
  return segmentEnd(bytes, reader.position);
}

// ---------------------------------------------------------------------------
// Lossless decoding (jdlhuff.c, jddiffct.c, jdlossls.c; T.81 Annex H).
// ---------------------------------------------------------------------------

function decodeLosslessScan(bytes: Uint8Array, offset: number, scan: Component[], params: ScanParameters, precision: number): number {
  const { ss: psv, se, ah, al: pt, restartInterval } = params;
  if (psv < 1 || psv > 7 || se !== 0 || ah !== 0 || pt >= precision) {
    throw new ValueError(`Invalid progressive/lossless parameters Ss=${psv} Se=${se} Ah=${ah} Al=${pt}`);
  }
  const reader = new BitReader(bytes, offset);
  const interleaved = scan.length > 1;
  const mcusPerRow = interleaved ? Math.ceil(params.width / params.maxH) : scan[0]!.blocksPerLine;
  const mcuRows = interleaved ? Math.ceil(params.height / params.maxV) : scan[0]!.blocksPerColumn;
  if (restartInterval % mcusPerRow !== 0) {
    throw new ValueError(`Restart interval (${restartInterval}) must be a multiple of the number of MCUs per MCU row (${mcusPerRow})`);
  }
  const rowsPerRestart = restartInterval / mcusPerRow;
  // Differences per component row (with dummy samples at the right edge).
  const diffWidth = scan.map((component) => (interleaved ? mcusPerRow * component.h : component.blocksPerLine));
  const previous = scan.map((component) => new Int32Array(component.downsampledWidth));
  const firstRow = scan.map(() => true);
  const current = scan.map((component) => new Int32Array(component.downsampledWidth));
  const initial = 1 << (precision - pt - 1);
  let restartRows = rowsPerRestart;

  const undifference = (ci: number, diffs: Int32Array, row: number): void => {
    const component = scan[ci]!;
    const width = component.downsampledWidth;
    const prev = previous[ci]!;
    const out = current[ci]!;
    if (firstRow[ci]) {
      let ra = (diffs[0]! + initial) & 0xffff;
      out[0] = ra;
      for (let x = 1; x < width; x += 1) {
        ra = (diffs[x]! + ra) & 0xffff;
        out[x] = ra;
      }
      firstRow[ci] = false;
    } else {
      let rb = prev[0]!;
      let ra = (diffs[0]! + rb) & 0xffff;
      out[0] = ra;
      for (let x = 1; x < width; x += 1) {
        const rc = rb;
        rb = prev[x]!;
        let predictor: number;
        switch (psv) {
          case 1: predictor = ra; break;
          case 2: predictor = rb; break;
          case 3: predictor = rc; break;
          case 4: predictor = ra + rb - rc; break;
          case 5: predictor = ra + ((rb - rc) >> 1); break;
          case 6: predictor = rb + ((ra - rc) >> 1); break;
          default: predictor = (ra + rb) >> 1; break;
        }
        ra = (diffs[x]! + predictor) & 0xffff;
        out[x] = ra;
      }
    }
    const samples = component.samples!;
    for (let x = 0; x < width; x += 1) samples[row * width + x] = (out[x]! << pt) & 0xff;
    prev.set(out);
  };

  const decodeDiff = (component: Component): number => {
    const s = component.dcTable ? reader.decode(component.dcTable) : 0;
    if (s === 0) return 0;
    if (s === 16) return 32768;
    return extend(reader.getBits(s), s);
  };

  // Rows are undifferenced once a whole iMCU row is decoded (jddiffct.c): an iMCU row is one MCU row
  // in interleaved scans and ``v_samp_factor`` MCU rows otherwise, so a restart anywhere in it
  // restarts prediction at its first row.
  const rowsPerIMcu = interleaved ? 1 : scan[0]!.v;
  for (let first = 0; first < mcuRows; first += rowsPerIMcu) {
    const pending: { ci: number; row: number; diffs: Int32Array }[] = [];
    for (let mcuRow = first; mcuRow < Math.min(first + rowsPerIMcu, mcuRows); mcuRow += 1) {
      if (restartInterval && restartRows === 0) {
        reader.restart();
        firstRow.fill(true);
        restartRows = rowsPerRestart;
      }
      const rows = scan.map((component, ci) => Array.from({ length: interleaved ? component.v : 1 }, () => new Int32Array(diffWidth[ci]!)));
      for (let column = 0; column < mcusPerRow; column += 1) {
        scan.forEach((component, ci) => {
          const h = interleaved ? component.h : 1;
          const v = interleaved ? component.v : 1;
          for (let y = 0; y < v; y += 1) for (let x = 0; x < h; x += 1) rows[ci]![y]![column * h + x] = decodeDiff(component);
        });
      }
      if (restartInterval) restartRows -= 1;
      scan.forEach((component, ci) => {
        const v = interleaved ? component.v : 1;
        for (let y = 0; y < v; y += 1) pending.push({ ci, row: mcuRow * v + y, diffs: rows[ci]![y]! });
      });
    }
    for (const ci of scan.keys()) {
      for (const item of pending) {
        if (item.ci !== ci || item.row >= scan[ci]!.downsampledHeight) continue;
        undifference(ci, item.diffs, item.row);
      }
    }
  }
  return segmentEnd(bytes, reader.position);
}

// ---------------------------------------------------------------------------
// IDCT, upsampling (jdsample.c) and colour conversion (jdcolor.c).
// ---------------------------------------------------------------------------

function upsample(component: Component, width: number, height: number, maxH: number, maxV: number, fancy: boolean): Uint8Array {
  let planeWidth: number;
  let plane: Uint8Array;
  if (component.samples) {
    planeWidth = component.downsampledWidth;
    plane = component.samples;
  } else {
    planeWidth = component.paddedBlocksPerLine * 8;
    const planeHeight = component.paddedBlocksPerColumn * 8;
    plane = new Uint8Array(planeWidth * planeHeight);
    const quant = component.quant ?? new Int32Array(64);
    for (let row = 0; row < component.paddedBlocksPerColumn; row += 1) {
      for (let column = 0; column < component.paddedBlocksPerLine; column += 1) {
        idctBlock(component.coefficients, (row * component.paddedBlocksPerLine + column) * 64, quant, plane, row * 8 * planeWidth + column * 8, planeWidth);
      }
    }
  }
  const dw = component.downsampledWidth;
  const dh = component.downsampledHeight;
  const out = new Uint8Array(width * height);
  const hRatio = maxH / component.h;
  const vRatio = maxV / component.v;
  const sample = (x: number, y: number): number => plane[y * planeWidth + x]!;
  const rowAt = (y: number): number => (y < 0 ? 0 : y >= dh ? dh - 1 : y);
  if (!Number.isInteger(hRatio) || !Number.isInteger(vRatio)) throw new ValueError('Fractional sampling not implemented yet');
  if (hRatio === 1 && vRatio === 1) {
    for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) out[y * width + x] = sample(x, y);
    return out;
  }
  const horizontalFancy = (input: (x: number) => number, target: Uint8Array, targetOffset: number): void => {
    // h2v1_fancy_upsample over one input row of ``dw`` samples (output 2*dw, cropped to ``width``).
    const put = (x: number, value: number): void => {
      if (x < width) target[targetOffset + x] = value;
    };
    let invalue = input(0);
    put(0, invalue);
    put(1, (invalue * 3 + input(1) + 2) >> 2);
    for (let i = 1; i < dw - 1; i += 1) {
      invalue = input(i) * 3;
      put(2 * i, (invalue + input(i - 1) + 1) >> 2);
      put(2 * i + 1, (invalue + input(i + 1) + 2) >> 2);
    }
    invalue = input(dw - 1);
    put(2 * dw - 2, (invalue * 3 + input(dw - 2) + 1) >> 2);
    put(2 * dw - 1, invalue);
  };
  if (hRatio === 2 && vRatio === 1) {
    if (fancy && dw > 2) {
      for (let y = 0; y < height; y += 1) horizontalFancy((x) => sample(x, y), out, y * width);
    } else {
      for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) out[y * width + x] = sample(x >> 1, y);
    }
    return out;
  }
  if (fancy && hRatio === 1 && vRatio === 2) {
    // h1v2_fancy_upsample.
    for (let y = 0; y < height; y += 1) {
      const inRow = y >> 1;
      const neighbour = rowAt(y & 1 ? inRow + 1 : inRow - 1);
      const bias = y & 1 ? 2 : 1;
      const near = rowAt(inRow);
      for (let x = 0; x < width; x += 1) out[y * width + x] = (sample(x, near) * 3 + sample(x, neighbour) + bias) >> 2;
    }
    return out;
  }
  if (fancy && hRatio === 2 && vRatio === 2 && dw > 2) {
    // h2v2_fancy_upsample.
    for (let y = 0; y < height; y += 1) {
      const inRow = y >> 1;
      const near = rowAt(inRow);
      const far = rowAt(y & 1 ? inRow + 1 : inRow - 1);
      const colsum = (x: number): number => sample(x, near) * 3 + sample(x, far);
      const o = y * width;
      const put = (x: number, value: number): void => {
        if (x < width) out[o + x] = value;
      };
      let thiscolsum = colsum(0);
      let nextcolsum = colsum(1);
      put(0, (thiscolsum * 4 + 8) >> 4);
      put(1, (thiscolsum * 3 + nextcolsum + 7) >> 4);
      let lastcolsum = thiscolsum;
      thiscolsum = nextcolsum;
      for (let i = 1; i < dw - 1; i += 1) {
        nextcolsum = colsum(i + 1);
        put(2 * i, (thiscolsum * 3 + lastcolsum + 8) >> 4);
        put(2 * i + 1, (thiscolsum * 3 + nextcolsum + 7) >> 4);
        lastcolsum = thiscolsum;
        thiscolsum = nextcolsum;
      }
      put(2 * dw - 2, (thiscolsum * 3 + lastcolsum + 8) >> 4);
      put(2 * dw - 1, (thiscolsum * 4 + 7) >> 4);
    }
    return out;
  }
  // h2v1/h2v2 without room for fancy filtering, and int_upsample: box replication.
  for (let y = 0; y < height; y += 1) {
    const inRow = Math.floor(y / vRatio);
    for (let x = 0; x < width; x += 1) out[y * width + x] = sample(Math.floor(x / hRatio), inRow);
  }
  return out;
}

function convertColor(planes: Uint8Array[], colorSpace: JpegColorSpace, space: 'GRAYSCALE' | 'RGB' | 'CMYK', pixels: number): Uint8Array {
  if (colorSpace === 'UNKNOWN') throw new ValueError('Unsupported color conversion request');
  if (space === 'GRAYSCALE') {
    const out = new Uint8Array(pixels);
    if (colorSpace === 'GRAYSCALE' || colorSpace === 'YCbCr') out.set(planes[0]!.subarray(0, pixels));
    else if (colorSpace === 'RGB') {
      const [r, g, b] = planes as [Uint8Array, Uint8Array, Uint8Array];
      for (let i = 0; i < pixels; i += 1) out[i] = Math.floor((RGB_Y_R[r[i]!]! + RGB_Y_G[g[i]!]! + RGB_Y_B[b[i]!]!) / 65536);
    } else throw new ValueError('Unsupported color conversion request');
    return out;
  }
  if (space === 'RGB') {
    const out = new Uint8Array(pixels * 3);
    if (colorSpace === 'GRAYSCALE') {
      const gray = planes[0]!;
      for (let i = 0; i < pixels; i += 1) out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = gray[i]!;
    } else if (colorSpace === 'RGB') {
      const [r, g, b] = planes as [Uint8Array, Uint8Array, Uint8Array];
      for (let i = 0; i < pixels; i += 1) {
        out[i * 3] = r[i]!;
        out[i * 3 + 1] = g[i]!;
        out[i * 3 + 2] = b[i]!;
      }
    } else if (colorSpace === 'YCbCr') {
      const [yp, cbp, crp] = planes as [Uint8Array, Uint8Array, Uint8Array];
      for (let i = 0; i < pixels; i += 1) {
        const y = yp[i]!;
        const cb = cbp[i]!;
        const cr = crp[i]!;
        out[i * 3] = clamp8(y + CR_R[cr]!);
        out[i * 3 + 1] = clamp8(y + Math.floor((CB_G[cb]! + CR_G[cr]!) / 65536));
        out[i * 3 + 2] = clamp8(y + CB_B[cb]!);
      }
    } else throw new ValueError('Unsupported color conversion request');
    return out;
  }
  const out = new Uint8Array(pixels * 4);
  if (colorSpace === 'CMYK') {
    for (let i = 0; i < pixels; i += 1) for (let c = 0; c < 4; c += 1) out[i * 4 + c] = planes[c]![i]!;
  } else if (colorSpace === 'YCCK') {
    const [yp, cbp, crp, kp] = planes as [Uint8Array, Uint8Array, Uint8Array, Uint8Array];
    for (let i = 0; i < pixels; i += 1) {
      const y = yp[i]!;
      const cb = cbp[i]!;
      const cr = crp[i]!;
      out[i * 4] = clamp8(255 - (y + CR_R[cr]!));
      out[i * 4 + 1] = clamp8(255 - (y + Math.floor((CB_G[cb]! + CR_G[cr]!) / 65536)));
      out[i * 4 + 2] = clamp8(255 - (y + CB_B[cb]!));
      out[i * 4 + 3] = kp[i]!;
    }
  } else throw new ValueError('Unsupported color conversion request');
  return out;
}
