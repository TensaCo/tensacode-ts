/**
 * SentencePiece ``precompiled_charsmap`` normalization (the ``Precompiled``
 * normalizer used by T5 and other SentencePiece tokenizers), mirroring the
 * ``spm_precompiled`` crate: a darts-clone double-array trie over UTF-8 bytes
 * mapping to NUL-terminated replacement strings, applied per grapheme.
 */
import { graphemes, utf8Bytes } from './unicode.js';

function base64Bytes(text: string): Uint8Array {
  const buffer = (globalThis as { Buffer?: { from(text: string, encoding: string): Uint8Array } }).Buffer;
  if (buffer) {
    const decoded = buffer.from(text, 'base64');
    return new Uint8Array(decoded.buffer, decoded.byteOffset, decoded.byteLength);
  }
  const binary = atob(text);
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

export class PrecompiledCharsmap {
  private readonly units: Uint32Array;
  private readonly normalized: Uint8Array;
  private readonly cache = new Map<string, string | null>();
  private readonly decoder = new TextDecoder('utf-8', { fatal: false });

  constructor(charsmap: string | Uint8Array) {
    const bytes = typeof charsmap === 'string' ? base64Bytes(charsmap) : charsmap;
    if (bytes.byteLength < 4) throw new Error('invalid precompiled charsmap');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const trieSize = view.getUint32(0, true);
    if (trieSize % 4 !== 0 || 4 + trieSize > bytes.byteLength) throw new Error('invalid precompiled charsmap trie size');
    this.units = new Uint32Array(trieSize / 4);
    for (let index = 0; index < this.units.length; index += 1) this.units[index] = view.getUint32(4 + index * 4, true);
    this.normalized = bytes.slice(4 + trieSize);
  }

  private commonPrefixSearch(key: Uint8Array): number[] {
    const units = this.units;
    const results: number[] = [];
    const hasLeaf = (unit: number) => ((unit >>> 8) & 1) === 1;
    const value = (unit: number) => unit & 0x7fffffff;
    const label = (unit: number) => unit & (0x80000000 | 0xff);
    const offset = (unit: number) => (unit >>> 10) << ((unit & (1 << 9)) >>> 6);
    let nodePos = 0;
    let unit = units[nodePos]!;
    nodePos ^= offset(unit);
    for (const byte of key) {
      if (byte === 0) break;
      nodePos ^= byte;
      unit = units[nodePos]!;
      if (unit === undefined || (label(unit) >>> 0) !== byte) return results;
      nodePos ^= offset(unit);
      if (hasLeaf(unit)) results.push(value(units[nodePos]!));
    }
    return results;
  }

  private transform(chunk: string): string | null {
    const cached = this.cache.get(chunk);
    if (cached !== undefined) return cached;
    const results = this.commonPrefixSearch(utf8Bytes(chunk));
    let replacement: string | null = null;
    if (results.length) {
      const start = results[0]!;
      let end = start;
      while (end < this.normalized.length && this.normalized[end] !== 0) end += 1;
      replacement = this.decoder.decode(this.normalized.subarray(start, end));
    }
    if (this.cache.size < 65536) this.cache.set(chunk, replacement);
    return replacement;
  }

  normalize(text: string): string {
    let result = '';
    for (const grapheme of graphemes(text)) {
      if (utf8Bytes(grapheme).length < 6) {
        const replaced = this.transform(grapheme);
        if (replaced !== null) {
          result += replaced;
          continue;
        }
      }
      for (const char of grapheme) {
        const replaced = this.transform(char);
        result += replaced ?? char;
      }
    }
    return result;
  }
}
