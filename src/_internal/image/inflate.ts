/**
 * DEFLATE decompression (RFC 1950/1951) for PNG and WebP-free image codecs.
 *
 * ``inflateZlib`` uses ``node:zlib`` when the runtime provides it
 * (``process.getBuiltinModule``) and otherwise a bundled pure TypeScript
 * inflater, so decoding works in browsers and other non-Node runtimes.
 */
import { ValueError } from '../../errors.js';

interface ZlibModule {
  inflateSync(data: Uint8Array, options?: { finishFlush?: number }): Uint8Array;
  constants: { Z_SYNC_FLUSH: number };
}

let nodeZlib: ZlibModule | null | undefined;

function zlibModule(): ZlibModule | null {
  if (nodeZlib !== undefined) return nodeZlib;
  nodeZlib = null;
  try {
    const processLike = (globalThis as { process?: { getBuiltinModule?: (name: string) => unknown } }).process;
    const module = processLike?.getBuiltinModule?.('node:zlib') as ZlibModule | undefined;
    if (module && typeof module.inflateSync === 'function') nodeZlib = module;
  } catch {
    nodeZlib = null;
  }
  return nodeZlib;
}

/** Force the bundled inflater (tests); ``null`` restores automatic selection. */
export function setNativeInflate(enabled: boolean | null): void {
  nodeZlib = enabled === false ? null : undefined;
}

/**
 * Inflate a zlib stream. ``partial`` tolerates a truncated stream (returning
 * what was decoded), as libpng does for image data.
 */
export function inflateZlib(data: Uint8Array, partial = false): Uint8Array {
  const zlib = zlibModule();
  if (zlib) {
    try {
      const out = zlib.inflateSync(data, partial ? { finishFlush: zlib.constants.Z_SYNC_FLUSH } : undefined);
      return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
    } catch (error) {
      throw new ValueError(`invalid zlib stream: ${(error as Error).message}`);
    }
  }
  if (data.length < 2) {
    if (partial) return new Uint8Array(0);
    throw new ValueError('invalid zlib stream: missing header');
  }
  const cmf = data[0]!;
  const flg = data[1]!;
  if ((cmf & 0x0f) !== 8 || ((cmf << 8) | flg) % 31 !== 0) throw new ValueError('invalid zlib stream: incorrect header check');
  if (flg & 0x20) throw new ValueError('invalid zlib stream: preset dictionary');
  return inflateRaw(data.subarray(2), partial);
}

// ---------------------------------------------------------------------------
// Pure TypeScript inflate.
// ---------------------------------------------------------------------------

const LENGTH_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LENGTH_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CLEN_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/** Canonical Huffman decoding table: ``(symbol << 4) | length`` indexed by reversed bits. */
interface Huffman {
  table: Int32Array;
  bits: number;
}

function buildHuffman(lengths: ArrayLike<number>, count: number): Huffman {
  let maxBits = 0;
  for (let i = 0; i < count; i += 1) maxBits = Math.max(maxBits, lengths[i]!);
  const bits = Math.max(maxBits, 1);
  const table = new Int32Array(1 << bits).fill(-1);
  const blCount = new Int32Array(16);
  for (let i = 0; i < count; i += 1) blCount[lengths[i]!]! += 1;
  blCount[0] = 0;
  const nextCode = new Int32Array(16);
  let code = 0;
  for (let length = 1; length <= 15; length += 1) {
    code = (code + blCount[length - 1]!) << 1;
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
  return { table, bits };
}

let fixedTables: { literal: Huffman; distance: Huffman } | null = null;

function fixed(): { literal: Huffman; distance: Huffman } {
  if (!fixedTables) {
    const lengths = new Uint8Array(288);
    lengths.fill(8, 0, 144);
    lengths.fill(9, 144, 256);
    lengths.fill(7, 256, 280);
    lengths.fill(8, 280, 288);
    fixedTables = { literal: buildHuffman(lengths, 288), distance: buildHuffman(new Uint8Array(30).fill(5), 30) };
  }
  return fixedTables;
}

class Truncated extends Error {}

/** Inflate a raw DEFLATE stream. */
export function inflateRaw(input: Uint8Array, partial = false): Uint8Array {
  let out = new Uint8Array(Math.max(1024, input.length * 4));
  let outLength = 0;
  let position = 0;
  let bitBuffer = 0;
  let bitCount = 0;

  const need = (count: number): void => {
    while (bitCount < count) {
      if (position >= input.length) throw new Truncated('unexpected end of deflate stream');
      bitBuffer |= input[position++]! << bitCount;
      bitCount += 8;
    }
  };
  const read = (count: number): number => {
    if (count === 0) return 0;
    need(count);
    const value = bitBuffer & ((1 << count) - 1);
    bitBuffer >>>= count;
    bitCount -= count;
    return value;
  };
  const decode = (huffman: Huffman): number => {
    // Fill as many bits as are available, up to the table width.
    while (bitCount < huffman.bits && position < input.length) {
      bitBuffer |= input[position++]! << bitCount;
      bitCount += 8;
    }
    const entry = huffman.table[bitBuffer & ((1 << huffman.bits) - 1)]!;
    const length = entry & 15;
    if (entry < 0 || length > bitCount) {
      if (bitCount < huffman.bits) throw new Truncated('unexpected end of deflate stream');
      throw new ValueError('invalid deflate stream: bad Huffman code');
    }
    bitBuffer >>>= length;
    bitCount -= length;
    return entry >> 4;
  };
  const ensure = (extra: number): void => {
    if (outLength + extra <= out.length) return;
    let size = out.length * 2;
    while (size < outLength + extra) size *= 2;
    const grown = new Uint8Array(size);
    grown.set(out.subarray(0, outLength));
    out = grown;
  };

  try {
    let final = 0;
    while (!final) {
      final = read(1);
      const type = read(2);
      if (type === 0) {
        // Skip to the byte boundary and give back whole buffered bytes.
        position -= (bitCount - (bitCount & 7)) >> 3;
        bitBuffer = 0;
        bitCount = 0;
        if (position + 4 > input.length) throw new Truncated('unexpected end of deflate stream');
        const length = input[position]! | (input[position + 1]! << 8);
        const inverse = input[position + 2]! | (input[position + 3]! << 8);
        position += 4;
        if ((length ^ 0xffff) !== inverse) throw new ValueError('invalid deflate stream: stored block length');
        const available = Math.min(length, input.length - position);
        ensure(available);
        out.set(input.subarray(position, position + available), outLength);
        outLength += available;
        position += available;
        if (available < length) throw new Truncated('unexpected end of deflate stream');
        continue;
      }
      let literal: Huffman;
      let distance: Huffman;
      if (type === 1) {
        ({ literal, distance } = fixed());
      } else if (type === 2) {
        const hlit = read(5) + 257;
        const hdist = read(5) + 1;
        const hclen = read(4) + 4;
        const codeLengths = new Uint8Array(19);
        for (let i = 0; i < hclen; i += 1) codeLengths[CLEN_ORDER[i]!] = read(3);
        const lengthCode = buildHuffman(codeLengths, 19);
        const lengths = new Uint8Array(hlit + hdist);
        for (let i = 0; i < hlit + hdist;) {
          const symbol = decode(lengthCode);
          if (symbol < 16) lengths[i++] = symbol;
          else {
            let repeat: number;
            let value = 0;
            if (symbol === 16) {
              if (i === 0) throw new ValueError('invalid deflate stream: repeat without previous length');
              value = lengths[i - 1]!;
              repeat = 3 + read(2);
            } else if (symbol === 17) repeat = 3 + read(3);
            else repeat = 11 + read(7);
            if (i + repeat > hlit + hdist) throw new ValueError('invalid deflate stream: too many lengths');
            lengths.fill(value, i, i + repeat);
            i += repeat;
          }
        }
        literal = buildHuffman(lengths.subarray(0, hlit), hlit);
        distance = buildHuffman(lengths.subarray(hlit), hdist);
      } else {
        throw new ValueError('invalid deflate stream: invalid block type');
      }
      for (;;) {
        const symbol = decode(literal);
        if (symbol < 256) {
          ensure(1);
          out[outLength++] = symbol;
        } else if (symbol === 256) {
          break;
        } else {
          const index = symbol - 257;
          if (index >= 29) throw new ValueError('invalid deflate stream: invalid literal/length code');
          const length = LENGTH_BASE[index]! + read(LENGTH_EXTRA[index]!);
          const distanceSymbol = decode(distance);
          if (distanceSymbol >= 30) throw new ValueError('invalid deflate stream: invalid distance code');
          const back = DIST_BASE[distanceSymbol]! + read(DIST_EXTRA[distanceSymbol]!);
          if (back > outLength) throw new ValueError('invalid deflate stream: distance too far back');
          ensure(length);
          let from = outLength - back;
          for (let i = 0; i < length; i += 1) out[outLength++] = out[from++]!;
        }
      }
    }
  } catch (error) {
    if (!(error instanceof Truncated)) throw error;
    if (!partial) throw new ValueError(`invalid zlib stream: ${error.message}`);
  }
  return out.subarray(0, outLength);
}
