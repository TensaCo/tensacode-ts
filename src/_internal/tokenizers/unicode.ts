/** Character classes matching the Rust ``tokenizers`` implementation. */

const WHITESPACE = /^\p{White_Space}$/u;
const OTHER = /^\p{C}$/u;
const PUNCTUATION = /^\p{P}$/u;
const MARK_NONSPACING = /^\p{Mn}$/u;
const ASCII_PUNCTUATION = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~');

/** Rust ``char::is_whitespace`` (Unicode White_Space). */
export function isWhitespace(char: string): boolean {
  return WHITESPACE.test(char);
}

/** BERT whitespace: tab/newline/carriage return plus Unicode whitespace. */
export function isBertWhitespace(char: string): boolean {
  return char === '\t' || char === '\n' || char === '\r' || isWhitespace(char);
}

/** BERT control characters: Unicode ``C*`` except tab/newline/carriage return. */
export function isBertControl(char: string): boolean {
  if (char === '\t' || char === '\n' || char === '\r') return false;
  return OTHER.test(char);
}

/** ASCII punctuation or Unicode ``P*`` (``tokenizers`` ``is_punctuation``). */
export function isPunctuation(char: string): boolean {
  return ASCII_PUNCTUATION.has(char) || PUNCTUATION.test(char);
}

export function isNonspacingMark(char: string): boolean {
  return MARK_NONSPACING.test(char);
}

export function isChineseChar(char: string): boolean {
  const code = char.codePointAt(0)!;
  return (code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf)
    || (code >= 0x20000 && code <= 0x2a6df) || (code >= 0x2a700 && code <= 0x2b73f)
    || (code >= 0x2b740 && code <= 0x2b81f) || (code >= 0x2b920 && code <= 0x2ceaf)
    || (code >= 0xf900 && code <= 0xfaff) || (code >= 0x2f800 && code <= 0x2fa1f);
}

export function isDigit(char: string): boolean {
  return /^\p{N}$/u.test(char);
}

/** Split into Unicode code points. */
export function codePoints(text: string): string[] {
  return Array.from(text);
}

/** Per-code-point lowercase (Rust ``char::to_lowercase``, no final-sigma context). */
export function lowercase(text: string): string {
  let result = '';
  for (const char of text) result += char === 'Σ' ? 'σ' : char.toLowerCase();
  return result;
}

let byteEncoder: string[] | null = null;
let byteDecoder: Map<string, number> | null = null;

/** GPT-2 ``bytes_to_unicode`` table: byte → printable character. */
export function bytesToUnicode(): string[] {
  if (byteEncoder) return byteEncoder;
  const bytes: number[] = [];
  for (let b = 0x21; b <= 0x7e; b += 1) bytes.push(b);
  for (let b = 0xa1; b <= 0xac; b += 1) bytes.push(b);
  for (let b = 0xae; b <= 0xff; b += 1) bytes.push(b);
  const chars = [...bytes];
  let extra = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(256 + extra);
      extra += 1;
    }
  }
  const table = new Array<string>(256);
  bytes.forEach((b, index) => { table[b] = String.fromCodePoint(chars[index]!); });
  byteEncoder = table;
  return table;
}

export function unicodeToBytes(): Map<string, number> {
  if (byteDecoder) return byteDecoder;
  byteDecoder = new Map(bytesToUnicode().map((char, byte) => [char, byte]));
  return byteDecoder;
}

const encoder = new TextEncoder();
const lossyDecoder = new TextDecoder('utf-8', { fatal: false });

export function utf8Bytes(text: string): Uint8Array {
  return encoder.encode(text);
}

export function utf8DecodeLossy(bytes: Uint8Array): string {
  return lossyDecoder.decode(bytes);
}

let segmenter: Intl.Segmenter | null = null;

/** Extended grapheme clusters (Rust ``unicode-segmentation`` ``graphemes(true)``). */
export function graphemes(text: string): string[] {
  segmenter ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' });
  return Array.from(segmenter.segment(text), (item) => item.segment);
}

/** Rust-regex → JavaScript: translate the syntax used by tokenizer.json patterns. */
export function compileRegex(pattern: string, flags = 'gu'): RegExp {
  return new RegExp(pattern, flags);
}

export function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\/-]/g, '\\$&');
}
