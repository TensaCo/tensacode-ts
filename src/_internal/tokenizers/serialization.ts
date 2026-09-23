/**
 * Canonical backend JSON matching Python's ``json.dumps(json.loads(
 * tokenizer.backend_tokenizer.to_str()), sort_keys=True, separators=(',', ':'))``.
 *
 * The ``tokenizers`` library (0.23) serializes default fields that files may
 * omit, and transformers 5 rebuilds some pipelines from the tokenizer class
 * (notably ``T5Tokenizer``). This module reproduces both for the supported
 * components so artifacts embed the same tokenizer JSON as Python would.
 */
import {
  emitJsonRaw, parseJsonRaw, pythonFloatRepr, rawDelete, rawFromValue, rawGet, rawSet, rawString, type RawNode,
} from '../json.js';

const U64_MAX = 18446744073709551615n;
const POW10 = Array.from({ length: 309 }, (_, index) => Number(`1e${index}`));

/**
 * The ``f64`` that the Rust ``tokenizers`` library parses from a JSON number.
 * Its ``serde_json`` is built without ``float_roundtrip``: the decimal
 * significand (at most a ``u64``) is converted to ``f64`` and then multiplied
 * or divided by a power of ten, which can be one ULP from the correctly
 * rounded value. Python ``Tokenizer.from_str``/``from_file`` therefore change
 * some Unigram scores, and those changes reach Python's persisted tokenizer
 * JSON and fingerprints.
 */
export function rustJsonF64(text: string): number {
  let index = 0;
  let positive = true;
  if (text[index] === '-') {
    positive = false;
    index += 1;
  }
  const isDigit = (): boolean => index < text.length && text.charCodeAt(index) >= 48 && text.charCodeAt(index) <= 57;
  const digit = (): bigint => BigInt(text.charCodeAt(index) - 48);
  const overflows = (value: bigint, next: bigint): boolean => value >= U64_MAX / 10n && (value > U64_MAX / 10n || next > U64_MAX % 10n);
  let significand = 0n;
  let exponent = 0;
  if (text[index] === '0') index += 1;
  else {
    while (isDigit()) {
      const next = digit();
      if (overflows(significand, next)) {
        // Digits beyond a u64 only scale the exponent.
        while (isDigit()) { index += 1; exponent += 1; }
        break;
      }
      significand = significand * 10n + next;
      index += 1;
    }
  }
  if (text[index] === '.') {
    index += 1;
    while (isDigit()) {
      const next = digit();
      if (overflows(significand, next)) {
        while (isDigit()) index += 1; // further decimals are ignored
        break;
      }
      significand = significand * 10n + next;
      exponent -= 1;
      index += 1;
    }
  }
  if (text[index] === 'e' || text[index] === 'E') {
    index += 1;
    let positiveExponent = true;
    if (text[index] === '+') index += 1;
    else if (text[index] === '-') {
      positiveExponent = false;
      index += 1;
    }
    let value = 0;
    while (isDigit()) {
      value = Math.min(value * 10 + Number(digit()), 2 ** 31 - 1);
      index += 1;
    }
    exponent = positiveExponent ? exponent + value : exponent - value;
  }
  let result = Number(significand);
  for (;;) {
    const power = POW10[Math.abs(exponent)];
    if (power !== undefined) {
      if (exponent >= 0) result *= power;
      else result /= power;
      break;
    }
    if (result === 0 || exponent >= 0) break;
    result /= 1e308;
    exponent += 308;
  }
  return positive ? result : -result;
}

/**
 * Apply Rust ``Tokenizer.from_str`` float parsing to the Unigram scores (the
 * only floats of the supported tokenizer components), as a Rust round trip
 * followed by Python ``json.loads`` would see them.
 */
export function applyRustFloatParsing(root: RawNode): void {
  const model = rawGet(root, 'model');
  if (!model || model.t !== 'o' || rawString(rawGet(model, 'type')) !== 'Unigram') return;
  const vocab = rawGet(model, 'vocab');
  if (vocab?.t !== 'a') return;
  for (const entry of vocab.items) {
    const score = entry.t === 'a' ? entry.items[1] : undefined;
    if (score?.t === 'n') score.raw = pythonFloatRepr(rustJsonF64(score.raw));
  }
}

function normalizeComponent(node: RawNode | undefined): void {
  if (!node || node.t !== 'o') return;
  const type = rawString(rawGet(node, 'type'));
  for (const key of ['normalizers', 'pretokenizers', 'processors', 'decoders']) {
    const children = rawGet(node, key);
    if (children?.t === 'a') children.items.forEach(normalizeComponent);
  }
  if (type === 'ByteLevel' && !rawGet(node, 'use_regex')) rawSet(node, 'use_regex', rawFromValue(true));
  if (type === 'Punctuation' && !rawGet(node, 'behavior')) rawSet(node, 'behavior', rawFromValue('Isolated'));
  if (type === 'Metaspace') {
    const legacy = rawGet(node, 'add_prefix_space');
    if (!rawGet(node, 'prepend_scheme')) {
      const always = legacy?.t === 'l' ? legacy.v !== false : true;
      rawSet(node, 'prepend_scheme', rawFromValue(always ? 'always' : 'never'));
    }
    if (legacy) rawDelete(node, 'add_prefix_space');
    if (!rawGet(node, 'split')) rawSet(node, 'split', rawFromValue(true));
  }
}

function normalizeModel(model: RawNode | undefined): void {
  if (!model || model.t !== 'o') return;
  let type = rawString(rawGet(model, 'type'));
  if (!type && rawGet(model, 'continuing_subword_prefix') && !rawGet(model, 'merges')) {
    type = 'WordPiece';
    model.entries.unshift(['type', rawFromValue('WordPiece')]);
  }
  if (type === 'BPE') {
    const defaults: [string, unknown][] = [
      ['dropout', null], ['unk_token', null], ['continuing_subword_prefix', null], ['end_of_word_suffix', null],
      ['fuse_unk', false], ['byte_fallback', false], ['ignore_merges', false],
    ];
    for (const [key, value] of defaults) if (!rawGet(model, key)) rawSet(model, key, rawFromValue(value));
    const merges = rawGet(model, 'merges');
    if (merges?.t === 'a') {
      merges.items = merges.items.map((item) => {
        if (item.t !== 's') return item;
        const index = item.v.indexOf(' ', 1);
        return rawFromValue([item.v.slice(0, index), item.v.slice(index + 1)]);
      });
    }
  }
  if (!type && rawGet(model, 'vocab')?.t === 'a') {
    // Older files omit the model type; Rust's untagged model enum reads a
    // list vocabulary as Unigram.
    type = 'Unigram';
    model.entries.unshift(['type', rawFromValue('Unigram')]);
  }
  if (type === 'Unigram' && !rawGet(model, 'byte_fallback')) rawSet(model, 'byte_fallback', rawFromValue(false));
}

type TokenizerFlags = Record<string, unknown>;

function flag(flags: TokenizerFlags, key: string, fallback: boolean): boolean {
  return typeof flags[key] === 'boolean' ? flags[key] as boolean : fallback;
}

function metaspace(prependScheme: string): unknown {
  return { type: 'Metaspace', replacement: '▁', prepend_scheme: prependScheme, split: true };
}

/** ``[CLS] $A [SEP]`` / ``[CLS] $A [SEP] $B [SEP]`` template (transformers class post-processors). */
function clsSepTemplate(cls: string, clsId: number, sep: string, sepId: number): unknown {
  return {
    type: 'TemplateProcessing',
    single: [{ SpecialToken: { id: cls, type_id: 0 } }, { Sequence: { id: 'A', type_id: 0 } }, { SpecialToken: { id: sep, type_id: 0 } }],
    pair: [
      { SpecialToken: { id: cls, type_id: 0 } }, { Sequence: { id: 'A', type_id: 0 } }, { SpecialToken: { id: sep, type_id: 0 } },
      { Sequence: { id: 'B', type_id: 1 } }, { SpecialToken: { id: sep, type_id: 1 } },
    ],
    special_tokens: { [cls]: { id: cls, ids: [clsId], tokens: [cls] }, [sep]: { id: sep, ids: [sepId], tokens: [sep] } },
  };
}

function vocabularyId(root: RawNode, token: string): number | null {
  const vocab = rawGet(rawGet(root, 'model'), 'vocab');
  if (vocab?.t === 'o') {
    const id = rawGet(vocab, token);
    return id?.t === 'n' ? Number(id.raw) : null;
  }
  return null;
}

function unigramPieces(root: RawNode): string[] {
  const vocab = rawGet(rawGet(root, 'model'), 'vocab');
  if (vocab?.t !== 'a') return [];
  return vocab.items.map((entry) => (entry.t === 'a' ? rawString(entry.items[0]) ?? '' : ''));
}

function tokenId(root: RawNode, token: string): number {
  const index = unigramPieces(root).indexOf(token);
  if (index >= 0) return index;
  const byVocabulary = vocabularyId(root, token);
  if (byVocabulary !== null) return byVocabulary;
  const added = rawGet(root, 'added_tokens');
  if (added?.t === 'a') {
    for (const entry of added.items) {
      if (rawString(rawGet(entry, 'content')) === token) {
        const id = rawGet(entry, 'id');
        if (id?.t === 'n') return Number(id.raw);
      }
    }
  }
  return 0;
}

function setUnigram(root: RawNode, unkId: number | null): void {
  const model = rawGet(root, 'model');
  if (!model || model.t !== 'o') return;
  rawSet(model, 'type', rawFromValue('Unigram'));
  if (unkId !== null) rawSet(model, 'unk_id', rawFromValue(unkId));
  rawSet(model, 'byte_fallback', rawFromValue(false));
}

function text(flags: TokenizerFlags, key: string, fallback: string): string {
  const value = flags[key];
  if (typeof value === 'string') return value;
  if (value && typeof value === 'object' && typeof (value as { content?: unknown }).content === 'string') return (value as { content: string }).content;
  return fallback;
}

/**
 * transformers 5 ``GPT2Tokenizer.__init__``: the BPE vocabulary and merges are
 * kept; normalization is removed, pre-tokenization and decoding become plain
 * ``ByteLevel`` (for example SmolVLM's ``Digits`` split is dropped), and a
 * missing post-processor is built from ``add_bos_token``/``add_eos_token``.
 */
function rebuildGPT2(root: RawNode, flags: TokenizerFlags): void {
  const model = rawGet(root, 'model');
  if (model?.t === 'o') {
    rawSet(model, 'type', rawFromValue('BPE'));
    rawSet(model, 'dropout', rawFromValue(null));
    rawSet(model, 'unk_token', rawFromValue(null));
    rawSet(model, 'continuing_subword_prefix', rawFromValue(''));
    rawSet(model, 'end_of_word_suffix', rawFromValue(''));
    rawSet(model, 'fuse_unk', rawFromValue(false));
    rawSet(model, 'byte_fallback', rawFromValue(false));
    rawSet(model, 'ignore_merges', rawFromValue(false));
  }
  rawSet(root, 'normalizer', rawFromValue(null));
  rawSet(root, 'pre_tokenizer', rawFromValue({
    type: 'ByteLevel', add_prefix_space: flag(flags, 'add_prefix_space', false), trim_offsets: true, use_regex: true,
  }));
  rawSet(root, 'decoder', rawFromValue({ type: 'ByteLevel', add_prefix_space: true, trim_offsets: true, use_regex: true }));
  const explicit = 'add_bos_token' in flags || 'add_eos_token' in flags;
  const post = rawGet(root, 'post_processor');
  if (explicit || !post || post.t === 'l') {
    const addBos = flag(flags, 'add_bos_token', false);
    const addEos = flag(flags, 'add_eos_token', false);
    const bos = text(flags, 'bos_token', '<|endoftext|>');
    const eos = text(flags, 'eos_token', '<|endoftext|>');
    const token = (id: string, typeId: number): unknown => ({ SpecialToken: { id, type_id: typeId } });
    const sequence = (id: string, typeId: number): unknown => ({ Sequence: { id, type_id: typeId } });
    const single = [...(addBos ? [token(bos, 0)] : []), sequence('A', 0), ...(addEos ? [token(eos, 0)] : [])];
    const pair = [...single, ...(addBos ? [token(bos, 1)] : []), sequence('B', 1), ...(addEos ? [token(eos, 1)] : [])];
    const special: Record<string, unknown> = {};
    if (addBos) special[bos] = { id: bos, ids: [tokenId(root, bos)], tokens: [bos] };
    if (addEos) special[eos] = { id: eos, ids: [tokenId(root, eos)], tokens: [eos] };
    rawSet(root, 'post_processor', rawFromValue({ type: 'TemplateProcessing', single, pair, special_tokens: special }));
  }
}

/**
 * transformers 5 ``DebertaV2Tokenizer.__init__``: the Unigram vocabulary is
 * kept and the pipeline rebuilt from its flags (the SentencePiece charsmap is
 * dropped). Its own ``[CLS]``/``[SEP]`` template replaces the file's.
 */
function rebuildDebertaV2(root: RawNode, flags: TokenizerFlags): void {
  const unk = text(flags, 'unk_token', '[UNK]');
  const pieces = unigramPieces(root);
  const scores = rawGet(rawGet(root, 'model'), 'vocab');
  // ``vocab.index((unk_token, 0.0))`` when present, else the ``unk_id`` argument.
  let unkId = typeof flags.unk_id === 'number' ? flags.unk_id : 1;
  if (scores?.t === 'a') {
    const index = pieces.indexOf(unk);
    const entry = index >= 0 ? scores.items[index] : undefined;
    const score = entry?.t === 'a' ? entry.items[1] : undefined;
    if (score?.t === 'n' && Number(score.raw) === 0) unkId = index;
  }
  setUnigram(root, unkId);
  const normalizers: unknown[] = [];
  if (flag(flags, 'do_lower_case', false)) normalizers.push({ type: 'Lowercase' });
  normalizers.push(
    { type: 'Replace', pattern: { Regex: '\\s{2,}|[\\n\\r\\t]' }, content: ' ' },
    { type: 'NFC' },
    { type: 'Strip', strip_left: false, strip_right: true },
  );
  rawSet(root, 'normalizer', rawFromValue({ type: 'Sequence', normalizers }));
  const scheme = flag(flags, 'add_prefix_space', true) ? 'always' : 'first';
  const pretokenizers: unknown[] = [];
  if (flag(flags, 'split_by_punct', false)) pretokenizers.push({ type: 'Punctuation', behavior: 'Isolated' });
  pretokenizers.push(metaspace(scheme));
  rawSet(root, 'pre_tokenizer', rawFromValue({ type: 'Sequence', pretokenizers }));
  rawSet(root, 'decoder', rawFromValue(metaspace(scheme)));
  const cls = text(flags, 'cls_token', '[CLS]');
  const sep = text(flags, 'sep_token', '[SEP]');
  rawSet(root, 'post_processor', rawFromValue(clsSepTemplate(cls, tokenId(root, cls), sep, tokenId(root, sep))));
}

/**
 * transformers 5 ``AlbertTokenizer.__init__``: the Unigram vocabulary (unk id
 * 1) and the file's SentencePiece charsmap and post-processor are kept; the
 * normalizers, pre-tokenizer and decoder are rebuilt from its flags.
 */
function rebuildAlbert(root: RawNode, flags: TokenizerFlags): void {
  setUnigram(root, 1);
  let charsmap: RawNode | undefined;
  const findCharsmap = (node: RawNode | undefined): void => {
    if (!node || node.t !== 'o' || charsmap) return;
    if (rawString(rawGet(node, 'type')) === 'Precompiled') charsmap = rawGet(node, 'precompiled_charsmap');
    const children = rawGet(node, 'normalizers');
    if (children?.t === 'a') children.items.forEach(findCharsmap);
  };
  findCharsmap(rawGet(root, 'normalizer'));
  const normalizers: RawNode[] = [
    rawFromValue({ type: 'Replace', pattern: { String: '``' }, content: '"' }),
    rawFromValue({ type: 'Replace', pattern: { String: "''" }, content: '"' }),
  ];
  if (!flag(flags, 'keep_accents', false)) normalizers.push(rawFromValue({ type: 'NFKD' }), rawFromValue({ type: 'StripAccents' }));
  if (flag(flags, 'do_lower_case', true)) normalizers.push(rawFromValue({ type: 'Lowercase' }));
  if (charsmap) normalizers.push({ t: 'o', entries: [['type', rawFromValue('Precompiled')], ['precompiled_charsmap', charsmap]] });
  rawSet(root, 'normalizer', { t: 'o', entries: [['type', rawFromValue('Sequence')], ['normalizers', { t: 'a', items: normalizers }]] });
  const scheme = flag(flags, 'add_prefix_space', true) ? 'always' : 'never';
  rawSet(root, 'pre_tokenizer', rawFromValue({ type: 'Sequence', pretokenizers: [{ type: 'WhitespaceSplit' }, metaspace(scheme)] }));
  rawSet(root, 'decoder', rawFromValue(metaspace(scheme)));
  const post = rawGet(root, 'post_processor');
  if (!post || post.t === 'l') {
    const cls = text(flags, 'cls_token', '[CLS]');
    const sep = text(flags, 'sep_token', '[SEP]');
    rawSet(root, 'post_processor', rawFromValue(clsSepTemplate('[CLS]', tokenId(root, cls), '[SEP]', tokenId(root, sep))));
  }
}

/** transformers 5 ``T5Tokenizer``: Precompiled only, WhitespaceSplit + Metaspace. */
function rebuildT5(root: RawNode): void {
  const normalizer = rawGet(root, 'normalizer');
  const findCharsmap = (node: RawNode | undefined): RawNode | undefined => {
    if (!node || node.t !== 'o') return undefined;
    if (rawString(rawGet(node, 'type')) === 'Precompiled') return rawGet(node, 'precompiled_charsmap');
    const children = rawGet(node, 'normalizers');
    if (children?.t === 'a') for (const child of children.items) {
      const found = findCharsmap(child);
      if (found) return found;
    }
    return undefined;
  };
  const charsmap = findCharsmap(normalizer);
  rawSet(root, 'normalizer', charsmap
    ? { t: 'o', entries: [['type', rawFromValue('Precompiled')], ['precompiled_charsmap', charsmap]] }
    : rawFromValue(null));
  rawSet(root, 'pre_tokenizer', rawFromValue({
    type: 'Sequence',
    pretokenizers: [{ type: 'WhitespaceSplit' }, { type: 'Metaspace', replacement: '▁', prepend_scheme: 'always', split: true }],
  }));
  rawSet(root, 'decoder', rawFromValue({ type: 'Metaspace', replacement: '▁', prepend_scheme: 'always', split: true }));
}

/** Tokenizer classes whose transformers 5 construction is emulated. */
export const REBUILT_TOKENIZER_CLASSES = new Set([
  'T5Tokenizer', 'T5TokenizerFast', 'DebertaV2Tokenizer', 'DebertaV2TokenizerFast', 'AlbertTokenizer', 'AlbertTokenizerFast',
  'GPT2Tokenizer', 'GPT2TokenizerFast',
]);

/**
 * transformers classes that rebuild their backend from the tokenizer.json
 * vocabulary in Python (exact floats). Generic fast tokenizers load the file
 * through Rust and see its float parsing.
 */
const GENERIC_TOKENIZER_CLASSES = new Set(['PreTrainedTokenizerFast', 'TokenizersBackend']);

/** Whether ``AutoTokenizer`` for this class parses tokenizer.json with Rust. */
export function loadsThroughRust(tokenizerClass: string | null): boolean {
  return tokenizerClass === null || GENERIC_TOKENIZER_CLASSES.has(tokenizerClass);
}

export function canonicalBackendJson(
  text: string, options: { tokenizerClass?: string | null; rustParsed?: boolean; flags?: TokenizerFlags } = {},
): string {
  const root = parseJsonRaw(text);
  if (root.t !== 'o') throw new TypeError('tokenizer.json must contain an object');
  if (options.rustParsed) applyRustFloatParsing(root);
  rawSet(root, 'padding', rawFromValue(null));
  rawSet(root, 'truncation', rawFromValue(null));
  normalizeModel(rawGet(root, 'model'));
  const base = options.tokenizerClass?.replace(/Fast$/, '') ?? null;
  if (base === 'T5Tokenizer') rebuildT5(root);
  else if (base === 'DebertaV2Tokenizer') rebuildDebertaV2(root, options.flags ?? {});
  else if (base === 'AlbertTokenizer') rebuildAlbert(root, options.flags ?? {});
  else if (base === 'GPT2Tokenizer') rebuildGPT2(root, options.flags ?? {});
  for (const key of ['normalizer', 'pre_tokenizer', 'post_processor', 'decoder']) normalizeComponent(rawGet(root, key));
  return emitJsonRaw(root, { sortKeys: true, separators: [',', ':'] });
}
