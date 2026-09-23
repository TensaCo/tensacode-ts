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
  if (type === 'Unigram' && !rawGet(model, 'byte_fallback')) rawSet(model, 'byte_fallback', rawFromValue(false));
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
export const REBUILT_TOKENIZER_CLASSES = new Set(['T5Tokenizer', 'T5TokenizerFast']);

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
  text: string, options: { tokenizerClass?: string | null; rustParsed?: boolean } = {},
): string {
  const root = parseJsonRaw(text);
  if (root.t !== 'o') throw new TypeError('tokenizer.json must contain an object');
  if (options.rustParsed) applyRustFloatParsing(root);
  rawSet(root, 'padding', rawFromValue(null));
  rawSet(root, 'truncation', rawFromValue(null));
  if (options.tokenizerClass && REBUILT_TOKENIZER_CLASSES.has(options.tokenizerClass)) rebuildT5(root);
  normalizeModel(rawGet(root, 'model'));
  for (const key of ['normalizer', 'pre_tokenizer', 'post_processor', 'decoder']) normalizeComponent(rawGet(root, key));
  return emitJsonRaw(root, { sortKeys: true, separators: [',', ':'] });
}
