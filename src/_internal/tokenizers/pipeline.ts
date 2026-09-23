/**
 * ``tokenizer.json`` pipeline components: normalizers, pre-tokenizers,
 * post-processors and decoders, following the Rust ``tokenizers`` semantics
 * (offsets are not tracked; only token strings and ids are produced).
 */
import { PrecompiledCharsmap } from './precompiled.js';
import {
  bytesToUnicode, compileRegex, escapeRegex, isBertControl, isBertWhitespace, isChineseChar, isDigit,
  isNonspacingMark, isPunctuation, isWhitespace, lowercase, unicodeToBytes, utf8Bytes, utf8DecodeLossy,
} from './unicode.js';

export type Json = Record<string, unknown>;

export class UnsupportedTokenizerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnsupportedTokenizerError';
  }
}

function unsupported(kind: string, type: unknown): never {
  throw new UnsupportedTokenizerError(`unsupported tokenizer ${kind}: ${JSON.stringify(type)}`);
}

function patternOf(pattern: unknown): RegExp {
  const spec = pattern as { String?: string; Regex?: string };
  if (typeof spec?.String === 'string') return new RegExp(escapeRegex(spec.String), 'gu');
  if (typeof spec?.Regex === 'string') return compileRegex(spec.Regex, 'gu');
  throw new UnsupportedTokenizerError(`unsupported pattern ${JSON.stringify(pattern)}`);
}

// ---------------------------------------------------------------------------
// Normalizers.
// ---------------------------------------------------------------------------

export type Normalizer = (text: string) => string;

export function buildNormalizer(spec: Json | null | undefined): Normalizer | null {
  if (!spec) return null;
  switch (spec.type) {
    case 'Sequence': {
      const parts = (spec.normalizers as Json[]).map((item) => buildNormalizer(item)).filter((item): item is Normalizer => item !== null);
      return (text) => parts.reduce((value, part) => part(value), text);
    }
    case 'BertNormalizer': {
      const cleanText = spec.clean_text !== false;
      const chinese = spec.handle_chinese_chars !== false;
      const lower = spec.lowercase !== false;
      const strip = spec.strip_accents === null || spec.strip_accents === undefined ? lower : Boolean(spec.strip_accents);
      return (text) => {
        let value = text;
        if (cleanText) {
          let cleaned = '';
          for (const char of value) {
            if (char === '\0' || char === '\ufffd' || isBertControl(char)) continue;
            cleaned += isBertWhitespace(char) ? ' ' : char;
          }
          value = cleaned;
        }
        if (chinese) {
          let spaced = '';
          for (const char of value) spaced += isChineseChar(char) ? ` ${char} ` : char;
          value = spaced;
        }
        if (strip) value = Array.from(value.normalize('NFD')).filter((char) => !isNonspacingMark(char)).join('');
        if (lower) value = lowercase(value);
        return value;
      };
    }
    case 'Lowercase': return lowercase;
    case 'NFC': return (text) => text.normalize('NFC');
    case 'NFD': return (text) => text.normalize('NFD');
    case 'NFKC': return (text) => text.normalize('NFKC');
    case 'NFKD': return (text) => text.normalize('NFKD');
    case 'StripAccents': return (text) => Array.from(text).filter((char) => !/^\p{M}$/u.test(char)).join('');
    case 'Strip': {
      const left = spec.strip_left !== false;
      const right = spec.strip_right !== false;
      return (text) => {
        let value = text;
        if (left) value = value.replace(/^\p{White_Space}+/u, '');
        if (right) value = value.replace(/\p{White_Space}+$/u, '');
        return value;
      };
    }
    case 'Replace': {
      const regex = patternOf(spec.pattern);
      const content = String(spec.content ?? '');
      return (text) => text.replace(regex, () => content);
    }
    case 'Prepend': {
      const prepend = String(spec.prepend ?? '');
      return (text) => (text ? prepend + text : text);
    }
    case 'Precompiled': {
      if (!spec.precompiled_charsmap) return null;
      const charsmap = new PrecompiledCharsmap(spec.precompiled_charsmap as string);
      return (text) => charsmap.normalize(text);
    }
    case 'ByteLevel': {
      const table = bytesToUnicode();
      return (text) => Array.from(utf8Bytes(text), (byte) => table[byte]!).join('');
    }
    default: return unsupported('normalizer', spec.type);
  }
}

// ---------------------------------------------------------------------------
// Pre-tokenizers.
// ---------------------------------------------------------------------------

/** A pre-tokenized piece; ``first`` marks a piece starting at input offset 0. */
export interface Piece {
  text: string;
  first: boolean;
}

export type PreTokenizer = (pieces: Piece[]) => Piece[];

type Behavior = 'Removed' | 'Isolated' | 'MergedWithPrevious' | 'MergedWithNext' | 'Contiguous';

/** A covering segmentation: ``[start, end, isDelimiter]`` (Rust ``find_matches``). */
type Segments = (text: string) => [number, number, boolean][];

/** Apply a split behavior to a covering segmentation of ``piece``. */
function splitWith(piece: Piece, segmentsOf: Segments, behavior: Behavior): Piece[] {
  const text = piece.text;
  const segments = segmentsOf(text).map(([start, end, delimiter]) => ({ text: text.slice(start, end), delimiter, offset: start }));
  const out: { text: string; offset: number; delimiter: boolean }[] = [];
  switch (behavior) {
    case 'Removed':
      for (const segment of segments) if (!segment.delimiter) out.push(segment);
      break;
    case 'Isolated':
      for (const segment of segments) out.push(segment);
      break;
    case 'Contiguous':
      for (const segment of segments) {
        const last = out[out.length - 1];
        if (segment.delimiter && last?.delimiter) last.text += segment.text;
        else out.push({ ...segment });
      }
      break;
    case 'MergedWithPrevious':
      for (const segment of segments) {
        const last = out[out.length - 1];
        if (segment.delimiter && last && !last.delimiter) {
          last.text += segment.text;
          last.delimiter = true;
        } else out.push({ ...segment });
      }
      break;
    case 'MergedWithNext': {
      let pending: { text: string; offset: number; delimiter: boolean } | null = null;
      for (const segment of segments) {
        if (segment.delimiter) {
          if (pending) out.push(pending);
          pending = { ...segment };
        } else if (pending) {
          pending.text += segment.text;
          out.push(pending);
          pending = null;
        } else out.push({ ...segment });
      }
      if (pending) out.push(pending);
      break;
    }
    default: unsupported('split behavior', behavior);
  }
  return out.filter((item) => item.text.length > 0).map((item) => ({ text: item.text, first: piece.first && item.offset === 0 }));
}

function cover(text: string, matches: [number, number][]): [number, number, boolean][] {
  const result: [number, number, boolean][] = [];
  let cursor = 0;
  for (const [start, end] of matches) {
    if (start > cursor) result.push([cursor, start, false]);
    result.push([start, end, true]);
    cursor = end;
  }
  if (cursor < text.length) result.push([cursor, text.length, false]);
  return result;
}

function regexSegments(regex: RegExp, invert = false): Segments {
  return (text) => {
    const matches: [number, number][] = [];
    regex.lastIndex = 0;
    for (const match of text.matchAll(regex)) {
      if (match[0].length === 0) continue;
      matches.push([match.index!, match.index! + match[0].length]);
    }
    const segments = cover(text, matches);
    return invert ? segments.map(([start, end, delimiter]) => [start, end, !delimiter]) : segments;
  };
}

function charSegments(predicate: (char: string) => boolean): Segments {
  return (text) => {
    const matches: [number, number][] = [];
    let index = 0;
    for (const char of text) {
      if (predicate(char)) matches.push([index, index + char.length]);
      index += char.length;
    }
    return cover(text, matches);
  };
}

const GPT2_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function metaspaceSettings(spec: Json): { replacement: string; scheme: 'always' | 'first' | 'never'; split: boolean } {
  const replacement = String(spec.replacement ?? '▁');
  let scheme: 'always' | 'first' | 'never';
  if (typeof spec.prepend_scheme === 'string') scheme = spec.prepend_scheme as 'always' | 'first' | 'never';
  else scheme = spec.add_prefix_space === false ? 'never' : 'always';
  return { replacement, scheme, split: spec.split !== false };
}

export function buildPreTokenizer(spec: Json | null | undefined): PreTokenizer | null {
  if (!spec) return null;
  const each = (fn: (piece: Piece) => Piece[]): PreTokenizer => (pieces) => pieces.flatMap(fn);
  switch (spec.type) {
    case 'Sequence': {
      const parts = (spec.pretokenizers as Json[]).map((item) => buildPreTokenizer(item)).filter((item): item is PreTokenizer => item !== null);
      return (pieces) => parts.reduce((value, part) => part(value), pieces);
    }
    case 'BertPreTokenizer':
      return each((piece) => splitWith(piece, charSegments(isWhitespace), 'Removed')
        .flatMap((part) => splitWith(part, charSegments(isPunctuation), 'Isolated')));
    case 'Whitespace':
      return each((piece) => splitWith(piece, regexSegments(/\w+|[^\w\s]+/gu, true), 'Removed'));
    case 'WhitespaceSplit':
      return each((piece) => splitWith(piece, charSegments(isWhitespace), 'Removed'));
    case 'Punctuation':
      return each((piece) => splitWith(piece, charSegments(isPunctuation), (spec.behavior as Behavior) ?? 'Isolated'));
    case 'Digits': {
      const individual = spec.individual_digits === true;
      return each((piece) => splitWith(piece, charSegments(isDigit), individual ? 'Isolated' : 'Contiguous'));
    }
    case 'Split': {
      const segments = regexSegments(patternOf(spec.pattern), spec.invert === true);
      return each((piece) => splitWith(piece, segments, spec.behavior as Behavior));
    }
    case 'Metaspace': {
      const { replacement, scheme, split } = metaspaceSettings(spec);
      return each((piece) => {
        let text = piece.text.replace(/ /g, replacement);
        if ((scheme === 'always' || (scheme === 'first' && piece.first)) && !text.startsWith(replacement)) text = replacement + text;
        const replaced = { text, first: piece.first };
        if (!split) return [replaced];
        return splitWith(replaced, regexSegments(new RegExp(escapeRegex(replacement), 'gu')), 'MergedWithNext');
      });
    }
    case 'ByteLevel': {
      const addPrefixSpace = spec.add_prefix_space === true;
      const useRegex = spec.use_regex !== false;
      const table = bytesToUnicode();
      return each((piece) => {
        let text = piece.text;
        if (addPrefixSpace && !text.startsWith(' ')) text = ` ${text}`;
        const parts = useRegex ? splitWith({ text, first: piece.first }, regexSegments(GPT2_PATTERN), 'Isolated') : [{ text, first: piece.first }];
        return parts.map((part) => ({ text: Array.from(utf8Bytes(part.text), (byte) => table[byte]!).join(''), first: part.first }));
      });
    }
    case 'CharDelimiterSplit': {
      const delimiter = String(spec.delimiter);
      return each((piece) => splitWith(piece, charSegments((char) => char === delimiter), 'Removed'));
    }
    default: return unsupported('pre-tokenizer', spec.type);
  }
}

// ---------------------------------------------------------------------------
// Post-processors.
// ---------------------------------------------------------------------------

export interface Sequence {
  ids: number[];
  typeIds: number[];
  special: boolean[];
}

export interface PostProcessor {
  process(first: Sequence, second: Sequence | null): Sequence;
  /** Number of tokens added for a single sequence or a pair. */
  added(pair: boolean): number;
}

function concat(...parts: Sequence[]): Sequence {
  return {
    ids: parts.flatMap((part) => part.ids),
    typeIds: parts.flatMap((part) => part.typeIds),
    special: parts.flatMap((part) => part.special),
  };
}

function specialSequence(ids: number[], typeId: number): Sequence {
  return { ids: [...ids], typeIds: ids.map(() => typeId), special: ids.map(() => true) };
}

function withType(sequence: Sequence, typeId: number): Sequence {
  return { ids: sequence.ids, typeIds: sequence.ids.map(() => typeId), special: sequence.special };
}

export function buildPostProcessor(spec: Json | null | undefined): PostProcessor | null {
  if (!spec) return null;
  switch (spec.type) {
    case 'TemplateProcessing': {
      const specials = spec.special_tokens as Record<string, { ids: number[] }>;
      type Item = { SpecialToken?: { id: string; type_id: number }; Sequence?: { id: 'A' | 'B'; type_id: number } };
      const apply = (template: Item[], first: Sequence, second: Sequence | null): Sequence => concat(...template.map((item) => {
        if (item.SpecialToken) {
          const special = specials[item.SpecialToken.id];
          if (!special) throw new UnsupportedTokenizerError(`missing template special token ${item.SpecialToken.id}`);
          return specialSequence(special.ids, item.SpecialToken.type_id);
        }
        const sequence = item.Sequence!.id === 'A' ? first : second;
        if (!sequence) throw new UnsupportedTokenizerError('template requires a pair');
        return withType(sequence, item.Sequence!.type_id);
      }));
      const single = spec.single as Item[];
      const pair = spec.pair as Item[];
      const count = (template: Item[]) => template.reduce((total, item) => total + (item.SpecialToken ? specials[item.SpecialToken.id]!.ids.length : 0), 0);
      return {
        process: (first, second) => apply(second ? pair : single, first, second),
        added: (isPair) => count(isPair ? pair : single),
      };
    }
    case 'BertProcessing': {
      const [, sep] = spec.sep as [string, number];
      const [, cls] = spec.cls as [string, number];
      return {
        process: (first, second) => second
          ? concat(specialSequence([cls], 0), withType(first, 0), specialSequence([sep], 0), withType(second, 1), specialSequence([sep], 1))
          : concat(specialSequence([cls], 0), withType(first, 0), specialSequence([sep], 0)),
        added: (pair) => (pair ? 3 : 2),
      };
    }
    case 'RobertaProcessing': {
      const [, sep] = spec.sep as [string, number];
      const [, cls] = spec.cls as [string, number];
      return {
        process: (first, second) => second
          ? concat(specialSequence([cls], 0), withType(first, 0), specialSequence([sep], 0), specialSequence([sep], 1), withType(second, 1), specialSequence([sep], 1))
          : concat(specialSequence([cls], 0), withType(first, 0), specialSequence([sep], 0)),
        added: (pair) => (pair ? 4 : 2),
      };
    }
    case 'ByteLevel':
      return { process: (first, second) => (second ? concat(first, withType(second, 1)) : first), added: () => 0 };
    case 'Sequence': {
      const parts = (spec.processors as Json[]).map((item) => buildPostProcessor(item)).filter((item): item is PostProcessor => item !== null);
      return {
        process: (first, second) => {
          // Only one component may add special tokens; byte-level trimming is a no-op here.
          const adding = parts.filter((part) => part.added(false) > 0 || part.added(true) > 0);
          if (adding.length > 1) throw new UnsupportedTokenizerError('multiple special-token post-processors');
          return adding.length ? adding[0]!.process(first, second) : (second ? concat(first, withType(second, 1)) : first);
        },
        added: (pair) => parts.reduce((total, part) => total + part.added(pair), 0),
      };
    }
    default: return unsupported('post-processor', spec.type);
  }
}

// ---------------------------------------------------------------------------
// Decoders (``decode_chain`` semantics, then concatenation).
// ---------------------------------------------------------------------------

export type Decoder = (tokens: string[]) => string[];

function wordpieceCleanup(text: string): string {
  return text.replaceAll(' .', '.').replaceAll(' ?', '?').replaceAll(' !', '!').replaceAll(' ,', ',')
    .replaceAll(" ' ", "'").replaceAll(" n't", "n't").replaceAll(" 'm", "'m").replaceAll(' do not', " don't")
    .replaceAll(" 's", "'s").replaceAll(" 've", "'ve").replaceAll(" 're", "'re");
}

export function buildDecoder(spec: Json | null | undefined): Decoder | null {
  if (!spec) return null;
  switch (spec.type) {
    case 'Sequence': {
      const parts = (spec.decoders as Json[]).map((item) => buildDecoder(item)).filter((item): item is Decoder => item !== null);
      return (tokens) => parts.reduce((value, part) => part(value), tokens);
    }
    case 'WordPiece': {
      const prefix = String(spec.prefix ?? '##');
      const cleanup = spec.cleanup !== false;
      return (tokens) => tokens.map((token, index) => {
        let value = token;
        if (index !== 0) value = value.startsWith(prefix) ? value.replace(prefix, '') : ` ${value}`;
        return cleanup ? wordpieceCleanup(value) : value;
      });
    }
    case 'Metaspace': {
      const { replacement, scheme } = metaspaceSettings(spec);
      return (tokens) => tokens.map((token, index) => {
        let value = '';
        for (const char of token) {
          if (char === replacement) {
            if (!(index === 0 && scheme !== 'never')) value += ' ';
          } else value += char;
        }
        return value;
      });
    }
    case 'ByteLevel': {
      const table = unicodeToBytes();
      return (tokens) => {
        const bytes: number[] = [];
        for (const token of tokens) {
          for (const char of token) {
            const byte = table.get(char);
            if (byte === undefined) bytes.push(...utf8Bytes(char));
            else bytes.push(byte);
          }
        }
        return [utf8DecodeLossy(Uint8Array.from(bytes))];
      };
    }
    case 'BPEDecoder': {
      const suffix = String(spec.suffix ?? '</w>');
      return (tokens) => tokens.map((token, index) => token.replaceAll(suffix, index === tokens.length - 1 ? '' : ' '));
    }
    case 'Replace': {
      const regex = patternOf(spec.pattern);
      const content = String(spec.content ?? '');
      return (tokens) => tokens.map((token) => token.replace(regex, () => content));
    }
    case 'Strip': {
      const content = String(spec.content ?? ' ');
      const start = Number(spec.start ?? 0);
      const stop = Number(spec.stop ?? 0);
      return (tokens) => tokens.map((token) => {
        const chars = Array.from(token);
        let left = 0;
        for (let index = 0; index < start && chars[index] === content; index += 1) left = index + 1;
        let right = chars.length;
        for (let index = 0; index < stop && chars[chars.length - 1 - index] === content; index += 1) right = chars.length - 1 - index;
        return chars.slice(left, Math.max(left, right)).join('');
      });
    }
    case 'Fuse':
      return (tokens) => [tokens.join('')];
    case 'ByteFallback':
      return (tokens) => {
        const out: string[] = [];
        let pending: number[] = [];
        const flush = () => {
          if (pending.length) {
            out.push(utf8DecodeLossy(Uint8Array.from(pending)));
            pending = [];
          }
        };
        for (const token of tokens) {
          const match = /^<0x([0-9A-Fa-f]{2})>$/.exec(token);
          if (match) pending.push(parseInt(match[1]!, 16));
          else {
            flush();
            out.push(token);
          }
        }
        flush();
        return out;
      };
    case 'CTC': {
      const pad = String(spec.pad_token ?? '<pad>');
      const delimiter = String(spec.word_delimiter_token ?? '|');
      const cleanup = spec.cleanup !== false;
      return (tokens) => tokens.filter((token, index) => index === 0 || token !== tokens[index - 1])
        .filter((token) => token !== pad)
        .map((token) => {
          let value = token.replaceAll(delimiter, ' ');
          if (cleanup) value = wordpieceCleanup(value).trim() ? wordpieceCleanup(value) : value;
          return value;
        });
    }
    default: return unsupported('decoder', spec.type);
  }
}
