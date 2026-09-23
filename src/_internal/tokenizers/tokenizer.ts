/**
 * A dependency-free interpreter for Hugging Face ``tokenizer.json`` files and a
 * ``PreTrainedTokenizerFast``-equivalent wrapper ({@link FastTokenizer}).
 *
 * Supported components: normalizers (BertNormalizer, Lowercase, NFC/NFD/NFKC/
 * NFKD, StripAccents, Strip, Replace, Prepend, Precompiled, ByteLevel,
 * Sequence), pre-tokenizers (BertPreTokenizer, Whitespace, WhitespaceSplit,
 * Punctuation, Digits, Split, Metaspace, ByteLevel, CharDelimiterSplit,
 * Sequence), models (WordPiece, BPE, Unigram, WordLevel), post-processors
 * (TemplateProcessing, BertProcessing, RobertaProcessing, ByteLevel, Sequence)
 * and decoders (WordPiece, Metaspace, ByteLevel, BPEDecoder, Replace, Strip,
 * Fuse, ByteFallback, CTC, Sequence). Offsets are not computed.
 */
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Tensor, tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { parseJsonStrict, type JsonObject } from '../json.js';
import { canonicalBackendJson, loadsThroughRust } from './serialization.js';
import { buildModel, type TokenModel } from './models.js';
import {
  buildDecoder, buildNormalizer, buildPostProcessor, buildPreTokenizer,
  type Decoder, type Json, type Normalizer, type Piece, type PostProcessor, type PreTokenizer, type Sequence,
} from './pipeline.js';
import { escapeRegex, isWhitespace } from './unicode.js';

export interface AddedToken {
  id: number;
  content: string;
  singleWord: boolean;
  lstrip: boolean;
  rstrip: boolean;
  normalized: boolean;
  special: boolean;
}

interface Segment {
  text: string;
  added: number | null;
  first: boolean;
}

function isWordChar(char: string | undefined): boolean {
  return char !== undefined && /^[\p{L}\p{N}_]$/u.test(char);
}

/** The ``tokenizers`` backend: raw text ↔ token ids. */
export class Tokenizer {
  readonly spec: Json;
  readonly model: TokenModel;
  readonly normalizer: Normalizer | null;
  readonly preTokenizer: PreTokenizer | null;
  readonly postProcessor: PostProcessor | null;
  readonly decoder: Decoder | null;
  readonly addedTokens: AddedToken[] = [];
  private readonly addedByContent = new Map<string, AddedToken>();
  private readonly addedById = new Map<number, AddedToken>();
  private matchers: { raw: RegExp | null; normalized: RegExp | null; rawSpecialFree: RegExp | null; normalizedSpecialFree: RegExp | null } | null = null;

  constructor(spec: Json) {
    this.spec = spec;
    this.model = buildModel(spec.model as Json);
    this.normalizer = buildNormalizer(spec.normalizer as Json | null);
    this.preTokenizer = buildPreTokenizer(spec.pre_tokenizer as Json | null);
    this.postProcessor = buildPostProcessor(spec.post_processor as Json | null);
    this.decoder = buildDecoder(spec.decoder as Json | null);
    for (const raw of (spec.added_tokens as Json[] | undefined) ?? []) {
      this.registerAdded({
        id: Number(raw.id), content: String(raw.content), singleWord: raw.single_word === true,
        lstrip: raw.lstrip === true, rstrip: raw.rstrip === true, normalized: raw.normalized !== false,
        special: raw.special === true,
      });
    }
  }

  static fromString(text: string): Tokenizer {
    return new Tokenizer(parseJsonStrict(text) as Json);
  }

  private registerAdded(token: AddedToken): void {
    this.addedTokens.push(token);
    this.addedByContent.set(token.content, token);
    this.addedById.set(token.id, token);
    this.matchers = null;
  }

  /** Add a special token (as transformers does for missing special tokens). Returns its id. */
  addSpecialToken(content: string): number {
    const added = this.addedByContent.get(content);
    if (added) {
      added.special = true;
      return added.id;
    }
    // Like ``tokenizers``' AddedVocabulary: an in-vocabulary token becomes a
    // special added token that keeps the model's id (so it is matched whole and
    // removed by ``skipSpecialTokens``).
    const existing = this.model.tokenToId(content);
    if (existing !== undefined) {
      this.registerAdded({ id: existing, content, singleWord: false, lstrip: false, rstrip: false, normalized: false, special: true });
      return existing;
    }
    let id = this.model.vocabSize;
    for (const token of this.addedTokens) id = Math.max(id, token.id + 1);
    this.registerAdded({ id, content, singleWord: false, lstrip: false, rstrip: false, normalized: false, special: true });
    return id;
  }

  get vocabSize(): number {
    let size = this.model.vocabSize;
    for (const token of this.addedTokens) if (this.model.tokenToId(token.content) === undefined || this.model.idToToken(token.id) !== token.content) size = Math.max(size, token.id + 1);
    return size;
  }

  tokenToId(token: string): number | undefined {
    const added = this.addedByContent.get(token);
    if (added) return added.id;
    return this.model.tokenToId(token);
  }

  idToToken(id: number): string | undefined {
    return this.addedById.get(id)?.content ?? this.model.idToToken(id);
  }

  isSpecialId(id: number): boolean {
    return this.addedById.get(id)?.special === true;
  }

  private buildMatchers(): NonNullable<Tokenizer['matchers']> {
    const make = (tokens: AddedToken[]): RegExp | null => {
      if (!tokens.length) return null;
      const sorted = [...tokens].sort((a, b) => b.content.length - a.content.length);
      return new RegExp(sorted.map((token) => `(${escapeRegex(token.content)})`).join('|'), 'gu');
    };
    const raw = this.addedTokens.filter((token) => !token.normalized);
    const normalized = this.addedTokens.filter((token) => token.normalized);
    return {
      raw: make(raw), normalized: make(normalized),
      rawSpecialFree: make(raw.filter((token) => !token.special)),
      normalizedSpecialFree: make(normalized.filter((token) => !token.special)),
    };
  }

  /** Split ``text`` around added tokens (leftmost-longest, honouring strip/single-word flags). */
  private splitAdded(text: string, regex: RegExp | null, first: boolean): Segment[] {
    if (!regex || !text) return text ? [{ text, added: null, first }] : [];
    const segments: Segment[] = [];
    let cursor = 0;
    regex.lastIndex = 0;
    for (const match of text.matchAll(regex)) {
      let start = match.index!;
      let end = start + match[0].length;
      const token = this.addedByContent.get(match[0])!;
      if (start < cursor) continue;
      if (token.singleWord && (isWordChar(text[start - 1]) || isWordChar(text[end]))) continue;
      if (token.lstrip) while (start > cursor && isWhitespace(text[start - 1]!)) start -= 1;
      if (token.rstrip) while (end < text.length && isWhitespace(text[end]!)) end += 1;
      if (start > cursor) segments.push({ text: text.slice(cursor, start), added: null, first: first && cursor === 0 });
      segments.push({ text: match[0], added: token.id, first: first && start === 0 });
      cursor = end;
    }
    if (cursor < text.length) segments.push({ text: text.slice(cursor), added: null, first: first && cursor === 0 });
    return segments;
  }

  /** Raw text → model token ids (no post-processing). */
  encodeText(text: string, options: { splitSpecialTokens?: boolean } = {}): number[] {
    this.matchers ??= this.buildMatchers();
    const splitSpecial = options.splitSpecialTokens === true;
    const rawMatcher = splitSpecial ? this.matchers.rawSpecialFree : this.matchers.raw;
    const normalizedMatcher = splitSpecial ? this.matchers.normalizedSpecialFree : this.matchers.normalized;
    const ids: number[] = [];
    for (const segment of this.splitAdded(text, rawMatcher, true)) {
      if (segment.added !== null) {
        ids.push(segment.added);
        continue;
      }
      const normalized = this.normalizer ? this.normalizer(segment.text) : segment.text;
      for (const part of this.splitAdded(normalized, normalizedMatcher, segment.first)) {
        if (part.added !== null) {
          ids.push(part.added);
          continue;
        }
        let pieces: Piece[] = [{ text: part.text, first: part.first }];
        if (this.preTokenizer) pieces = this.preTokenizer(pieces);
        for (const piece of pieces) {
          for (const token of this.model.tokenize(piece.text)) {
            const id = this.model.tokenToId(token) ?? this.tokenToId(token);
            if (id === undefined) throw new ValueError(`token ${JSON.stringify(token)} has no id`);
            ids.push(id);
          }
        }
      }
    }
    return ids;
  }

  numSpecialTokensToAdd(pair: boolean): number {
    return this.postProcessor ? this.postProcessor.added(pair) : 0;
  }

  /** Apply the post-processor to raw ids (special tokens, type ids). */
  postProcess(first: number[], second: number[] | null, addSpecialTokens = true): Sequence {
    const a: Sequence = { ids: first, typeIds: first.map(() => 0), special: first.map(() => false) };
    const b: Sequence | null = second ? { ids: second, typeIds: second.map(() => 1), special: second.map(() => false) } : null;
    if (!addSpecialTokens || !this.postProcessor) {
      return b ? { ids: [...a.ids, ...b.ids], typeIds: [...a.typeIds, ...b.typeIds], special: [...a.special, ...b.special] } : a;
    }
    return this.postProcessor.process(a, b);
  }

  /** Token ids → text (Rust ``Tokenizer::decode``). */
  decode(ids: readonly number[], options: { skipSpecialTokens?: boolean } = {}): string {
    const tokens: string[] = [];
    for (const id of ids) {
      const token = this.idToToken(id);
      if (token === undefined) continue;
      if (options.skipSpecialTokens && this.isSpecialId(id)) continue;
      tokens.push(token);
    }
    if (!this.decoder) return tokens.join(' ');
    return this.decoder(tokens).join('');
  }
}

// ---------------------------------------------------------------------------
// transformers-compatible wrapper.
// ---------------------------------------------------------------------------

/** ``int(1e30)``: transformers' ``VERY_LARGE_INTEGER`` default ``model_max_length``. */
export const VERY_LARGE_INTEGER = 1e30;

export interface TokenizerOptions {
  clean_up_tokenization_spaces?: boolean;
  model_max_length?: number;
  model_input_names?: string[];
  split_special_tokens?: boolean;
}

/** Python ``_tokenizer_config(tokenizer)``: the complete persisted tokenizer. */
export interface TokenizerConfiguration {
  json: string;
  options: Required<TokenizerOptions>;
  special_tokens: Record<string, string | string[]>;
  padding_side: 'left' | 'right';
  truncation_side: 'left' | 'right';
}

export interface EncodeOptions {
  padding?: boolean | 'longest' | 'max_length';
  truncation?: boolean;
  maxLength?: number | null;
  addSpecialTokens?: boolean;
  /** Second sequences for pair encoding (one per text). */
  textPair?: string | readonly string[] | null;
}

export interface BatchEncoding {
  inputIds: number[][];
  attentionMask: number[][];
  tokenTypeIds?: number[][];
}

export interface TensorBatch {
  input_ids: Tensor;
  attention_mask: Tensor;
  token_type_ids?: Tensor;
}

/** transformers tokenizer classes whose default ``model_input_names`` include token type ids. */
const TOKEN_TYPE_CLASSES = new Set([
  'BertTokenizer', 'BertTokenizerFast', 'ElectraTokenizer', 'ElectraTokenizerFast', 'DebertaV2Tokenizer',
  'DebertaV2TokenizerFast', 'DebertaTokenizer', 'DebertaTokenizerFast', 'AlbertTokenizer', 'AlbertTokenizerFast',
]);

/** transformers 5 tokenizer class chosen from ``config.json`` ``model_type`` when ``tokenizer_class`` is absent. */
const MODEL_TYPE_TOKENIZERS: Record<string, string> = {
  bert: 'BertTokenizer', electra: 'BertTokenizer', distilbert: 'DistilBertTokenizer', roberta: 'RobertaTokenizer',
  t5: 'T5Tokenizer', mt5: 'T5Tokenizer', clip: 'CLIPTokenizer', clip_text_model: 'CLIPTokenizer', vit: 'BertTokenizer',
};

/** Class-default special tokens (transformers 5 ``__init__`` defaults). */
const CLASS_SPECIAL_TOKENS: Record<string, Record<string, string>> = {
  BertTokenizer: { unk_token: '[UNK]', sep_token: '[SEP]', pad_token: '[PAD]', cls_token: '[CLS]', mask_token: '[MASK]' },
  DistilBertTokenizer: { unk_token: '[UNK]', sep_token: '[SEP]', pad_token: '[PAD]', cls_token: '[CLS]', mask_token: '[MASK]' },
  ElectraTokenizer: { unk_token: '[UNK]', sep_token: '[SEP]', pad_token: '[PAD]', cls_token: '[CLS]', mask_token: '[MASK]' },
  RobertaTokenizer: { bos_token: '<s>', eos_token: '</s>', unk_token: '<unk>', sep_token: '</s>', pad_token: '<pad>', cls_token: '<s>', mask_token: '<mask>' },
  T5Tokenizer: { eos_token: '</s>', unk_token: '<unk>', pad_token: '<pad>' },
  CLIPTokenizer: { bos_token: '<|startoftext|>', eos_token: '<|endoftext|>', unk_token: '<|endoftext|>', pad_token: '<|endoftext|>' },
};

function baseTokenizerClass(name: string | null): string | null {
  return name === null ? null : name.replace(/Fast$/, '');
}

const SPECIAL_KEYS = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token'] as const;

function cleanUpTokenization(text: string): string {
  return text.replaceAll(' .', '.').replaceAll(' ?', '?').replaceAll(' !', '!').replaceAll(' ,', ',')
    .replaceAll(" ' ", "'").replaceAll(" n't", "n't").replaceAll(" 'm", "'m").replaceAll(" 's", "'s")
    .replaceAll(" 've", "'ve").replaceAll(" 're", "'re");
}

/** Canonical backend JSON (``json.dumps(json.loads(to_str()), sort_keys, compact)`` with padding/truncation reset). */
export function canonicalTokenizerJson(text: string, tokenizerClass: string | null = null, rustParsed = false): string {
  return canonicalBackendJson(text, { tokenizerClass, rustParsed });
}

export class FastTokenizer {
  readonly backend: Tokenizer;
  readonly jsonText: string;
  readonly options: Required<TokenizerOptions>;
  readonly specialTokens: Record<string, string | string[]>;
  paddingSide: 'left' | 'right';
  truncationSide: 'left' | 'right';
  /** transformers class emulated in-process (not persisted, like Python). */
  readonly tokenizerClass: string | null;

  constructor(
    jsonText: string,
    settings: {
      specialTokens?: Record<string, string | string[]>;
      options?: TokenizerOptions;
      paddingSide?: 'left' | 'right';
      truncationSide?: 'left' | 'right';
      /** ``jsonText`` is already the canonical backend JSON (from a saved configuration). */
      canonical?: boolean;
      /**
       * The backend is built with Rust ``Tokenizer.from_str``/``from_file`` in
       * Python, whose float parsing can move Unigram scores by one ULP.
       */
      rustParsed?: boolean;
      /** transformers tokenizer class whose pipeline construction is emulated. */
      tokenizerClass?: string | null;
    } = {},
  ) {
    if (settings.canonical) {
      this.jsonText = settings.rustParsed ? canonicalBackendJson(jsonText, { rustParsed: true }) : jsonText;
    } else {
      this.jsonText = canonicalTokenizerJson(jsonText, settings.tokenizerClass ?? null, settings.rustParsed ?? false);
    }
    this.backend = Tokenizer.fromString(this.jsonText);
    const options = settings.options ?? {};
    this.options = {
      clean_up_tokenization_spaces: options.clean_up_tokenization_spaces ?? false,
      model_max_length: options.model_max_length ?? VERY_LARGE_INTEGER,
      model_input_names: options.model_input_names ?? ['input_ids', 'attention_mask'],
      split_special_tokens: options.split_special_tokens ?? false,
    };
    this.specialTokens = {};
    const supplied = settings.specialTokens ?? {};
    const ordered = [...SPECIAL_KEYS, ...Object.keys(supplied).filter((key) => !(SPECIAL_KEYS as readonly string[]).includes(key))];
    for (const key of ordered) {
      const value = supplied[key];
      if (value === undefined) continue;
      if (Array.isArray(value)) {
        this.specialTokens[key] = value.map(String);
        for (const token of value) this.backend.addSpecialToken(String(token));
      } else if (typeof value === 'string') {
        this.specialTokens[key] = value;
        this.backend.addSpecialToken(value);
      }
    }
    this.paddingSide = settings.paddingSide ?? 'right';
    this.truncationSide = settings.truncationSide ?? 'right';
    this.tokenizerClass = baseTokenizerClass(settings.tokenizerClass ?? null);
  }

  /** Rebuild from a persisted {@link TokenizerConfiguration} (Python ``_tokenizer(config)``). */
  static fromConfiguration(config: TokenizerConfiguration | JsonObject): FastTokenizer {
    const value = config as unknown as TokenizerConfiguration;
    if (typeof value.json !== 'string') throw new ValueError('tokenizer configuration requires json');
    return new FastTokenizer(value.json, {
      specialTokens: value.special_tokens ?? {}, options: value.options ?? {},
      paddingSide: value.padding_side ?? 'right', truncationSide: value.truncation_side ?? 'right', canonical: true,
      // Python ``_tokenizer(config)`` uses ``Tokenizer.from_str(config['json'])``.
      rustParsed: true,
    });
  }

  /**
   * ``PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json), **special_tokens)``
   * as used by ranking/retrieval tools (``tokenizer_json`` + ``tokenizer_special_tokens``).
   */
  static fromJsonString(json: string, specialTokens: Record<string, string> = {}): FastTokenizer {
    return new FastTokenizer(json, { specialTokens, canonical: false, rustParsed: true });
  }

  /** ``AutoTokenizer.from_pretrained(directory, use_fast=True)`` for a downloaded snapshot. */
  static async fromDirectory(directory: string): Promise<FastTokenizer> {
    const json = await readFile(join(directory, 'tokenizer.json'), 'utf8');
    const readOptional = async (name: string): Promise<Record<string, unknown>> => {
      try {
        return parseJsonStrict(await readFile(join(directory, name), 'utf8')) as Record<string, unknown>;
      } catch {
        return {};
      }
    };
    const config = await readOptional('tokenizer_config.json');
    const map = await readOptional('special_tokens_map.json');
    const modelConfig = await readOptional('config.json');
    const declared = typeof config.tokenizer_class === 'string' ? config.tokenizer_class : null;
    const tokenizerClass = baseTokenizerClass(declared ?? (typeof modelConfig.model_type === 'string' ? MODEL_TYPE_TOKENIZERS[modelConfig.model_type] ?? null : null));
    const text = (value: unknown): string | null => {
      if (typeof value === 'string') return value;
      if (value && typeof value === 'object' && typeof (value as { content?: unknown }).content === 'string') return (value as { content: string }).content;
      return null;
    };
    const defaults = (tokenizerClass && CLASS_SPECIAL_TOKENS[tokenizerClass]) || {};
    const specialTokens: Record<string, string | string[]> = {};
    for (const key of SPECIAL_KEYS) {
      const value = text(config[key]) ?? text(map[key]) ?? defaults[key] ?? null;
      if (value !== null) specialTokens[key] = value;
    }
    // transformers 5 keeps sentinel/extra tokens as special added tokens but
    // omits them from ``special_tokens_map``.
    const options: TokenizerOptions = {};
    if (typeof config.clean_up_tokenization_spaces === 'boolean') options.clean_up_tokenization_spaces = config.clean_up_tokenization_spaces;
    if (typeof config.model_max_length === 'number') options.model_max_length = config.model_max_length;
    if (Array.isArray(config.model_input_names)) options.model_input_names = config.model_input_names as string[];
    else if (tokenizerClass && TOKEN_TYPE_CLASSES.has(tokenizerClass)) {
      options.model_input_names = ['input_ids', 'token_type_ids', 'attention_mask'];
    }
    if (typeof config.split_special_tokens === 'boolean') options.split_special_tokens = config.split_special_tokens;
    const side = (value: unknown): 'left' | 'right' | undefined => (value === 'left' || value === 'right' ? value : undefined);
    const paddingSide = side(config.padding_side);
    const truncationSide = side(config.truncation_side);
    return new FastTokenizer(json, {
      specialTokens, options, tokenizerClass, rustParsed: loadsThroughRust(tokenizerClass),
      ...(paddingSide ? { paddingSide } : {}),
      ...(truncationSide ? { truncationSide } : {}),
    });
  }

  /** Python ``_tokenizer_config(tokenizer)``. */
  configuration(): TokenizerConfiguration {
    return {
      json: this.jsonText,
      options: { ...this.options, model_input_names: [...this.options.model_input_names] },
      special_tokens: Object.fromEntries(Object.entries(this.specialTokens).map(([key, value]) => [key, Array.isArray(value) ? [...value] : value])),
      padding_side: this.paddingSide,
      truncation_side: this.truncationSide,
    };
  }

  /** ``special_tokens_map`` (strings, plus ``additional_special_tokens`` lists). */
  get specialTokensMap(): Record<string, string | string[]> {
    return { ...this.specialTokens };
  }

  private specialId(key: (typeof SPECIAL_KEYS)[number]): number | null {
    const token = this.specialTokens[key];
    if (typeof token !== 'string') return null;
    return this.backend.tokenToId(token) ?? null;
  }

  get padTokenId(): number | null { return this.specialId('pad_token'); }
  get eosTokenId(): number | null { return this.specialId('eos_token'); }
  get bosTokenId(): number | null { return this.specialId('bos_token'); }
  get unkTokenId(): number | null { return this.specialId('unk_token'); }
  get clsTokenId(): number | null { return this.specialId('cls_token'); }
  get sepTokenId(): number | null { return this.specialId('sep_token'); }
  get maskTokenId(): number | null { return this.specialId('mask_token'); }

  /** Number of ids including added tokens (Python ``len(tokenizer)``). */
  get length(): number {
    return this.backend.vocabSize;
  }

  convertTokensToIds(tokens: readonly string[]): (number | null)[] {
    return tokens.map((token) => this.backend.tokenToId(token) ?? this.unkTokenId);
  }

  convertIdsToTokens(ids: readonly number[]): (string | null)[] {
    return ids.map((id) => this.backend.idToToken(id) ?? null);
  }

  /** Tokens of ``text`` without special tokens. */
  tokenize(text: string): string[] {
    return this.backend.encodeText(text, { splitSpecialTokens: this.options.split_special_tokens })
      .map((id) => this.backend.idToToken(id) ?? '');
  }

  /** Rust ``truncate_encodings`` (``LongestFirst``, stride 0). */
  private truncate(first: number[], second: number[] | null, maxLength: number, addSpecial: boolean): [number[], number[] | null] {
    const budget = maxLength - (addSpecial ? this.backend.numSpecialTokensToAdd(second !== null) : 0);
    const total = first.length + (second?.length ?? 0);
    if (total <= budget) return [first, second];
    if (budget < 0) throw new ValueError('max_length is smaller than the number of special tokens');
    const cut = (ids: number[], length: number): number[] => (length >= ids.length ? ids
      : this.truncationSide === 'left' ? ids.slice(ids.length - length) : ids.slice(0, length));
    if (!second) return [cut(first, budget), null];
    let n1 = first.length;
    let n2 = second.length;
    const swap = n1 > n2;
    if (swap) [n1, n2] = [n2, n1];
    n2 = n1 > budget ? n1 : Math.max(n1, budget - n1);
    if (n1 + n2 > budget) {
      n1 = Math.floor(budget / 2);
      n2 = n1 + (budget % 2);
    }
    if (swap) [n1, n2] = [n2, n1];
    return [cut(first, n1), cut(second, n2)];
  }

  /**
   * Encode one text or a batch (Python ``tokenizer(texts, padding=..., truncation=..., max_length=...)``).
   * Returns plain id arrays; use {@link encodeTensors} for int64 tensors.
   */
  encode(texts: string | readonly string[], options: EncodeOptions = {}): BatchEncoding {
    const batch = typeof texts === 'string' ? [texts] : [...texts];
    const pairs = options.textPair === undefined || options.textPair === null ? null
      : typeof options.textPair === 'string' ? [options.textPair] : [...options.textPair];
    if (pairs && pairs.length !== batch.length) throw new ValueError('text_pair must match the batch size');
    const addSpecial = options.addSpecialTokens ?? true;
    let maxLength = options.maxLength ?? null;
    if (options.truncation && maxLength === null) maxLength = this.options.model_max_length;
    const sequences: Sequence[] = batch.map((text, index) => {
      if (typeof text !== 'string') throw new TypeError('tokenizer inputs must be strings');
      let first = this.backend.encodeText(text, { splitSpecialTokens: this.options.split_special_tokens });
      let second = pairs ? this.backend.encodeText(pairs[index]!, { splitSpecialTokens: this.options.split_special_tokens }) : null;
      if (options.truncation && maxLength !== null) [first, second] = this.truncate(first, second, maxLength, addSpecial);
      return this.backend.postProcess(first, second, addSpecial);
    });
    const padding = options.padding === true ? 'longest' : options.padding || false;
    let target = 0;
    if (padding === 'longest') target = Math.max(0, ...sequences.map((sequence) => sequence.ids.length));
    if (padding === 'max_length') {
      if (maxLength === null) throw new ValueError("padding='max_length' requires maxLength");
      target = maxLength;
    }
    const pad = this.padTokenId;
    if (padding && pad === null && sequences.some((sequence) => sequence.ids.length < target)) {
      throw new ValueError('Asking to pad but the tokenizer does not have a padding token');
    }
    const result: BatchEncoding = { inputIds: [], attentionMask: [] };
    const includeTypes = this.options.model_input_names.includes('token_type_ids');
    if (includeTypes) result.tokenTypeIds = [];
    for (const sequence of sequences) {
      const missing = padding ? Math.max(0, target - sequence.ids.length) : 0;
      const pads = new Array<number>(missing).fill(pad ?? 0);
      const zeros = new Array<number>(missing).fill(0);
      const ones = new Array<number>(sequence.ids.length).fill(1);
      if (this.paddingSide === 'left') {
        result.inputIds.push([...pads, ...sequence.ids]);
        result.attentionMask.push([...zeros, ...ones]);
        result.tokenTypeIds?.push([...zeros, ...sequence.typeIds]);
      } else {
        result.inputIds.push([...sequence.ids, ...pads]);
        result.attentionMask.push([...ones, ...zeros]);
        result.tokenTypeIds?.push([...sequence.typeIds, ...zeros]);
      }
    }
    return result;
  }

  /** {@link encode} returning rectangular int64 tensors (requires uniform lengths, e.g. ``padding: true``). */
  encodeTensors(texts: string | readonly string[], options: EncodeOptions = {}): TensorBatch {
    const encoded = this.encode(texts, options);
    const toTensor = (rows: number[][]): Tensor => {
      const width = rows[0]?.length ?? 0;
      if (rows.some((row) => row.length !== width)) throw new ValueError('Unable to create a tensor from ragged sequences; enable padding');
      return tensor(rows.flat(), { dtype: 'int64', shape: [rows.length, width] });
    };
    const result: TensorBatch = { input_ids: toTensor(encoded.inputIds), attention_mask: toTensor(encoded.attentionMask) };
    if (encoded.tokenTypeIds) result.token_type_ids = toTensor(encoded.tokenTypeIds);
    return result;
  }

  decode(ids: readonly number[] | Tensor, options: { skipSpecialTokens?: boolean; cleanUpTokenizationSpaces?: boolean } = {}): string {
    const values = ids instanceof Tensor ? ids.toArray() : [...ids];
    let text = this.backend.decode(values, { skipSpecialTokens: options.skipSpecialTokens ?? false });
    if (this.tokenizerClass === 'CLIPTokenizer') {
      // transformers CLIPTokenizer wraps the backend decode (end-of-word suffix → space, stripped).
      const suffix = (this.backend.spec.model as { end_of_word_suffix?: unknown }).end_of_word_suffix;
      if (typeof suffix === 'string' && suffix) text = text.replaceAll(suffix, ' ').trim();
    }
    if (options.cleanUpTokenizationSpaces ?? this.options.clean_up_tokenization_spaces) text = cleanUpTokenization(text);
    return text;
  }

  batchDecode(sequences: readonly (readonly number[])[] | Tensor, options: { skipSpecialTokens?: boolean; cleanUpTokenizationSpaces?: boolean } = {}): string[] {
    const rows = sequences instanceof Tensor
      ? Array.from({ length: sequences.shape[0]! }, (_, index) => sequences.select(0, index).toArray())
      : sequences;
    return rows.map((row) => this.decode(row, options));
  }
}
