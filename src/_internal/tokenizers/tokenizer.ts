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
import { BLANK_TOKENIZERS } from './blankTokenizers.generated.js';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Tensor, tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { emitJsonRaw, parseJsonRaw, parseJsonStrict, rawFromValue, rawGet, rawSet, type JsonObject, type RawNode } from '../json.js';
import { REBUILT_TOKENIZER_CLASSES, canonicalBackendJson, classPostProcessor, loadsThroughRust, rustTokenizerString } from './serialization.js';
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

  /** Model vocabulary size without added tokens (Rust ``get_vocab_size(with_added_tokens=False)``). */
  baseVocabSize(): number {
    return this.model.vocabSize;
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
            const id = (this.model.tokenizedId ?? this.model.tokenToId).call(this.model, token) ?? this.tokenToId(token);
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
/** Class ``model_input_names`` other than ``['input_ids', 'attention_mask']`` (transformers 5). */
const CLASS_INPUT_NAMES: Record<string, string[]> = {
  BertTokenizer: ['input_ids', 'token_type_ids', 'attention_mask'],
  ElectraTokenizer: ['input_ids', 'token_type_ids', 'attention_mask'],
  DebertaV2Tokenizer: ['input_ids', 'attention_mask', 'token_type_ids'],
  DebertaTokenizer: ['input_ids', 'attention_mask', 'token_type_ids'],
};

/** transformers 5 tokenizer class chosen from ``config.json`` ``model_type`` when ``tokenizer_class`` is absent. */
/** transformers 5.17 ``TOKENIZER_MAPPING_NAMES`` for the native model types (unmapped types load as ``TokenizersBackend``). */
const MODEL_TYPE_TOKENIZERS: Record<string, string> = {
  bert: 'BertTokenizer', electra: 'BertTokenizer', distilbert: 'BertTokenizer', roberta: 'RobertaTokenizer',
  t5: 'T5Tokenizer', mt5: 'T5Tokenizer', clip: 'CLIPTokenizer', albert: 'AlbertTokenizer', 'deberta-v2': 'DebertaV2Tokenizer',
};

/** Class-default special tokens (transformers 5 ``__init__`` defaults). */
const CLASS_SPECIAL_TOKENS: Record<string, Record<string, string>> = {
  BertTokenizer: { unk_token: '[UNK]', sep_token: '[SEP]', pad_token: '[PAD]', cls_token: '[CLS]', mask_token: '[MASK]' },
  DistilBertTokenizer: { unk_token: '[UNK]', sep_token: '[SEP]', pad_token: '[PAD]', cls_token: '[CLS]', mask_token: '[MASK]' },
  ElectraTokenizer: { unk_token: '[UNK]', sep_token: '[SEP]', pad_token: '[PAD]', cls_token: '[CLS]', mask_token: '[MASK]' },
  RobertaTokenizer: { bos_token: '<s>', eos_token: '</s>', unk_token: '<unk>', sep_token: '</s>', pad_token: '<pad>', cls_token: '<s>', mask_token: '<mask>' },
  T5Tokenizer: { eos_token: '</s>', unk_token: '<unk>', pad_token: '<pad>' },
  DebertaV2Tokenizer: { bos_token: '[CLS]', eos_token: '[SEP]', unk_token: '[UNK]', sep_token: '[SEP]', pad_token: '[PAD]', cls_token: '[CLS]', mask_token: '[MASK]' },
  AlbertTokenizer: { bos_token: '[CLS]', eos_token: '[SEP]', unk_token: '<unk>', sep_token: '[SEP]', pad_token: '<pad>', cls_token: '[CLS]', mask_token: '[MASK]' },
  GPT2Tokenizer: { bos_token: '<|endoftext|>', eos_token: '<|endoftext|>', unk_token: '<|endoftext|>' },
  LlamaTokenizer: { bos_token: '<s>', eos_token: '</s>', unk_token: '<unk>' },
  CLIPTokenizer: { bos_token: '<|startoftext|>', eos_token: '<|endoftext|>', unk_token: '<|endoftext|>', pad_token: '<|endoftext|>' },
};

/** transformers 5 tokenizer classes whose class attribute ``padding_side`` is ``"left"``. */
const LEFT_PADDING_CLASSES = new Set(['LlamaTokenizer', 'CodeLlamaTokenizer', 'GemmaTokenizer', 'CohereTokenizer', 'Siglip2Tokenizer', 'XLNetTokenizer']);

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
export function canonicalTokenizerJson(
  text: string, tokenizerClass: string | null = null, rustParsed = false, flags: Record<string, unknown> = {},
): string {
  return canonicalBackendJson(text, { tokenizerClass, rustParsed, flags });
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
      /** ``tokenizer_config.json`` (construction flags of rebuilt tokenizer classes). */
      flags?: Record<string, unknown>;
    } = {},
  ) {
    if (settings.canonical) {
      this.jsonText = settings.rustParsed ? canonicalBackendJson(jsonText, { rustParsed: true }) : jsonText;
    } else {
      this.jsonText = canonicalTokenizerJson(jsonText, settings.tokenizerClass ?? null, settings.rustParsed ?? false, settings.flags ?? {});
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
    const readOptional = async (name: string): Promise<string | null> => {
      try {
        return await readFile(join(directory, name), 'utf8');
      } catch {
        return null;
      }
    };
    return FastTokenizer.fromFiles({
      'tokenizer.json': json,
      'tokenizer_config.json': await readOptional('tokenizer_config.json'),
      'special_tokens_map.json': await readOptional('special_tokens_map.json'),
      'added_tokens.json': await readOptional('added_tokens.json'),
      'config.json': await readOptional('config.json'),
    });
  }

  /**
   * ``AutoTokenizer.from_pretrained`` for a checkpoint without any tokenizer
   * files: the model type's tokenizer class built with its defaults (a
   * vocabulary of special tokens only), or ``null`` when transformers cannot
   * build that class without files either.
   */
  static blankForModel(configJson: string): FastTokenizer | null {
    let modelType: unknown;
    try {
      modelType = (parseJsonStrict(configJson) as Record<string, unknown>).model_type;
    } catch {
      return null;
    }
    const blank = typeof modelType === 'string' ? BLANK_TOKENIZERS[modelType] : undefined;
    if (!blank) return null;
    return FastTokenizer.fromFiles({ 'tokenizer.json': blank.json, 'config.json': configJson });
  }

  /**
   * ``AutoTokenizer.from_pretrained`` over in-memory files (``tokenizer.json``
   * plus optional ``tokenizer_config.json``, ``special_tokens_map.json`` and
   * the model ``config.json``); see {@link FastTokenizer.fromDirectory}.
   */
  static fromFiles(files: Record<string, string | null | undefined>): FastTokenizer {
    const json = files['tokenizer.json'];
    if (typeof json !== 'string') throw new ValueError('tokenizer files require tokenizer.json');
    const readOptional = (name: string): Record<string, unknown> => {
      const text = files[name];
      if (typeof text !== 'string') return {};
      try {
        const value = parseJsonStrict(text);
        return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
      } catch {
        return {};
      }
    };
    const config = readOptional('tokenizer_config.json');
    const map = readOptional('special_tokens_map.json');
    const modelConfig = readOptional('config.json');
    const declared = typeof config.tokenizer_class === 'string' ? config.tokenizer_class : null;
    const tokenizerClass = baseTokenizerClass(declared ?? (typeof modelConfig.model_type === 'string' ? MODEL_TYPE_TOKENIZERS[modelConfig.model_type] ?? null : null));
    const text = (value: unknown): string | null => {
      if (typeof value === 'string') return value;
      if (value && typeof value === 'object' && typeof (value as { content?: unknown }).content === 'string') return (value as { content: string }).content;
      return null;
    };
    const defaults = (tokenizerClass && CLASS_SPECIAL_TOKENS[tokenizerClass]) || {};
    const legacy = !('added_tokens_decoder' in config);
    const specialTokens: Record<string, string | string[]> = {};
    for (const key of SPECIAL_KEYS) {
      // ``_from_pretrained``: without ``added_tokens_decoder`` (the legacy
      // layout), special_tokens_map.json values override tokenizer_config.json.
      const value = (legacy && key in map ? text(map[key]) : null) ?? text(config[key]) ?? defaults[key] ?? null;
      if (value !== null) specialTokens[key] = value;
    }
    // transformers 5 keeps sentinel/extra tokens as special added tokens but
    // omits them from ``special_tokens_map``.
    const options: TokenizerOptions = {};
    if (typeof config.clean_up_tokenization_spaces === 'boolean') options.clean_up_tokenization_spaces = config.clean_up_tokenization_spaces;
    if (typeof config.model_max_length === 'number') options.model_max_length = config.model_max_length;
    if (Array.isArray(config.model_input_names)) options.model_input_names = config.model_input_names as string[];
    else if (tokenizerClass && CLASS_INPUT_NAMES[tokenizerClass]) {
      options.model_input_names = [...CLASS_INPUT_NAMES[tokenizerClass]!];
    }
    if (typeof config.split_special_tokens === 'boolean') options.split_special_tokens = config.split_special_tokens;
    const side = (value: unknown): 'left' | 'right' | undefined => (value === 'left' || value === 'right' ? value : undefined);
    // Class-level ``padding_side = "left"`` (LlamaTokenizer and relatives) applies when the file sets none.
    const paddingSide = side(config.padding_side) ?? (tokenizerClass && LEFT_PADDING_CLASSES.has(tokenizerClass) ? 'left' : undefined);
    const truncationSide = side(config.truncation_side);
    // Construction flags: special tokens from special_tokens_map.json or the
    // class defaults, then tokenizer_config.json.
    const flags: Record<string, unknown> = { ...specialTokens, ...config };
    const constructed = new FastTokenizer(json, {
      specialTokens, options, tokenizerClass, rustParsed: loadsThroughRust(tokenizerClass), flags,
      ...(paddingSide ? { paddingSide } : {}),
      ...(truncationSide ? { truncationSide } : {}),
    });
    return initializedTokenizer(constructed, tokenizerClass, config, legacy ? map : {}, flags, readOptional('added_tokens.json'));
  }

  #rustJsonText: string | null = null;

  /**
   * Python ``backend_tokenizer.to_str()``: the backend JSON in Rust field
   * order, as tools persist it in ``tokenizer_json`` fields.
   */
  get rustJsonText(): string {
    this.#rustJsonText ??= rustTokenizerString(this.jsonText);
    return this.#rustJsonText;
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


// ---------------------------------------------------------------------------
// ``TokenizersBackend.__init__``: added and special token registration.
// ---------------------------------------------------------------------------

/** ``AddedToken`` flags (``tokenizers`` field names). */
export interface AddedTokenSpec {
  content: string;
  lstrip: boolean;
  normalized: boolean;
  rstrip: boolean;
  single_word: boolean;
  special: boolean;
}

/** The ``AddedToken`` transformers creates for a special token string. */
export function specialAddedToken(content: string): AddedTokenSpec {
  return { content, lstrip: false, normalized: false, rstrip: false, single_word: false, special: true };
}

/** The tokenizer with a replaced canonical backend JSON (same special tokens and options). */
export function withBackendJson(tokenizer: FastTokenizer, json: string): FastTokenizer {
  return new FastTokenizer(json, {
    specialTokens: tokenizer.specialTokens, options: tokenizer.options, paddingSide: tokenizer.paddingSide,
    truncationSide: tokenizer.truncationSide, canonical: true, tokenizerClass: tokenizer.tokenizerClass,
  });
}

function rawValue(node: RawNode, key: string): unknown {
  const item = rawGet(node, key);
  if (!item) return undefined;
  if (item.t === 'n') return Number(item.raw);
  if (item.t === 's' || item.t === 'l') return item.v;
  return undefined;
}

/**
 * ``Tokenizer.add_tokens``/``add_special_tokens`` of the ``tokenizers``
 * ``AddedVocabulary``: an identical added token is kept, an existing added or
 * vocabulary token keeps its id (taking the new flags), and new tokens take the
 * next id. Returns the tokenizer unchanged when nothing is added.
 */
export function addTokens(tokenizer: FastTokenizer, tokens: readonly AddedTokenSpec[]): FastTokenizer {
  const root = parseJsonRaw(tokenizer.jsonText);
  const list = rawGet(root, 'added_tokens');
  const items = list?.t === 'a' ? [...list.items] : [];
  const flags = ['lstrip', 'normalized', 'rstrip', 'single_word', 'special'] as const;
  const same = (node: RawNode, token: AddedTokenSpec): boolean => rawValue(node, 'content') === token.content
    && flags.every((key) => rawValue(node, key) === token[key]);
  const modelSize = tokenizer.backend.baseVocabSize();
  let changed = false;
  for (const token of tokens) {
    if (!token.content || items.some((node) => same(node, token))) continue;
    const existing = items.find((node) => rawValue(node, 'content') === token.content);
    let id: number;
    if (existing) id = rawValue(existing, 'id') as number;
    else {
      const found = tokenizer.backend.model.tokenToId(token.content);
      const inVocabulary = found !== undefined && tokenizer.backend.model.idToToken(found) === token.content ? found : undefined;
      if (inVocabulary !== undefined) id = inVocabulary;
      else {
        const ids = items.map((node) => rawValue(node, 'id') as number);
        const max = ids.length ? Math.max(...ids) : null;
        id = max === null ? modelSize : (max >= modelSize || modelSize === 0 ? max + 1 : modelSize);
      }
    }
    const node = rawFromValue({ ...token, id });
    const index = items.findIndex((item) => rawValue(item, 'id') === id);
    if (index >= 0) items[index] = node;
    else items.push(node);
    changed = true;
  }
  if (!changed) return tokenizer;
  items.sort((a, b) => (rawValue(a, 'id') as number) - (rawValue(b, 'id') as number));
  rawSet(root, 'added_tokens', { t: 'a', items });
  return withBackendJson(tokenizer, emitJsonRaw(root, { sortKeys: true, separators: [',', ':'] }));
}

/** An ``AddedToken`` from a special-token value (a string or an ``AddedToken`` dictionary). */
function addedTokenFrom(value: unknown, special: boolean): AddedTokenSpec | null {
  if (typeof value === 'string') return special ? specialAddedToken(value) : null;
  if (value && typeof value === 'object' && typeof (value as { content?: unknown }).content === 'string') {
    const token = value as Record<string, unknown>;
    return {
      content: token.content as string, lstrip: token.lstrip === true, normalized: token.normalized === true,
      rstrip: token.rstrip === true, single_word: token.single_word === true, special: special || token.special === true,
    };
  }
  return null;
}

const NAMED_SPECIAL_KEYS = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token'] as const;

/**
 * The tokenizer ``from_pretrained`` returns after ``TokenizersBackend.__init__``:
 * tokens of ``added_tokens_decoder`` (by id), then special and extra special
 * tokens missing from the added vocabulary are registered, and the class's
 * own post-processor (or, for a backend without one, the plain ``$A``/``$A $B``
 * template) is installed.
 */
function initializedTokenizer(
  tokenizer: FastTokenizer, tokenizerClass: string | null, config: Record<string, unknown>, map: Record<string, unknown>,
  flags: Record<string, unknown>, addedTokensFile: Record<string, unknown>,
): FastTokenizer {
  const rebuilt = tokenizerClass !== null && REBUILT_TOKENIZER_CLASSES.has(tokenizerClass);
  const fileTokens = (JSON.parse(tokenizer.jsonText).added_tokens as (AddedTokenSpec & { id: number })[] | undefined) ?? [];
  // ``added_tokens_decoder``: tokenizer_config.json's, else (legacy) added_tokens.json and tokenizer.json's added tokens.
  const decoder = new Map<number, AddedTokenSpec>();
  const declared = config.added_tokens_decoder;
  if (declared && typeof declared === 'object' && !Array.isArray(declared)) {
    for (const [id, entry] of Object.entries(declared as Record<string, unknown>)) {
      const token = addedTokenFrom(entry, false);
      if (token) decoder.set(Number(id), { ...token, special: (entry as { special?: unknown }).special === true });
    }
  } else {
    const specials = new Set(Object.values(tokenizer.specialTokens).flat().map(String));
    for (const [content, id] of Object.entries(addedTokensFile)) {
      if (typeof id !== 'number') continue;
      const special = specials.has(content);
      decoder.set(id, { content, lstrip: false, normalized: !special, rstrip: false, single_word: false, special });
    }
    for (const { id, ...token } of fileTokens) decoder.set(id, token);
  }
  // Classes with their own ``__init__`` build a fresh backend: only the decoder's tokens return.
  let base = tokenizer;
  if (rebuilt && fileTokens.length) {
    const root = parseJsonRaw(tokenizer.jsonText);
    rawSet(root, 'added_tokens', { t: 'a', items: [] });
    base = withBackendJson(tokenizer, emitJsonRaw(root, { sortKeys: true, separators: [',', ':'] }));
  }
  const existing = new Set(rebuilt ? [] : fileTokens.map((token) => token.content));
  const same = (a: AddedTokenSpec, b: AddedTokenSpec): boolean => a.content === b.content && a.lstrip === b.lstrip
    && a.normalized === b.normalized && a.rstrip === b.rstrip && a.single_word === b.single_word && a.special === b.special;
  const tokens: AddedTokenSpec[] = [...decoder.entries()].sort(([a], [b]) => a - b).map(([, token]) => token)
    .filter((token) => rebuilt || !fileTokens.some((file) => same(file, token)));
  const encoder = new Set([...existing, ...tokens.map((token) => token.content)]);
  const register = (token: AddedTokenSpec | null): void => {
    if (!token || encoder.has(token.content)) return;
    tokens.push(token);
    encoder.add(token.content);
  };
  for (const key of NAMED_SPECIAL_KEYS) {
    const value = key in map ? map[key] : key in config ? config[key] : tokenizer.specialTokens[key];
    if (value !== null && value !== undefined) register(addedTokenFrom(value, true));
  }
  let extras: unknown[] = [];
  const listed = map.additional_special_tokens ?? map.extra_special_tokens ?? config.extra_special_tokens ?? config.additional_special_tokens;
  if (Array.isArray(listed)) extras = listed;
  if (tokenizerClass === 'T5Tokenizer' && !extras.some((token) => String(addedTokenFrom(token, true)?.content ?? '').includes('<extra_id_'))) {
    const count = typeof config.extra_ids === 'number' ? config.extra_ids : 100;
    extras = [...extras, ...Array.from({ length: count }, (_, index) => `<extra_id_${index}>`)];
  }
  for (const value of extras) register(addedTokenFrom(value, true));
  tokenizer = base;
  let result = tokens.length ? addTokens(tokenizer, tokens) : tokenizer;
  const root = parseJsonRaw(result.jsonText);
  let post = classPostProcessor(root, tokenizerClass, flags);
  const current = rawGet(root, 'post_processor');
  if (post === null && (!current || current.t === 'l') && !rebuilt) {
    post = {
      type: 'TemplateProcessing', single: [{ Sequence: { id: 'A', type_id: 0 } }],
      pair: [{ Sequence: { id: 'A', type_id: 0 } }, { Sequence: { id: 'B', type_id: 1 } }], special_tokens: {},
    };
  }
  if (post !== null) {
    rawSet(root, 'post_processor', rawFromValue(post));
    result = withBackendJson(result, emitJsonRaw(root, { sortKeys: true, separators: [',', ':'] }));
  }
  return result;
}
