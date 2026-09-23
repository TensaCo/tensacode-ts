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
  emitJsonRaw, parseJsonRaw, rawDelete, rawFromValue, rawGet, rawSet, rawString, type RawNode,
} from '../json.js';

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

export function canonicalBackendJson(text: string, options: { tokenizerClass?: string | null } = {}): string {
  const root = parseJsonRaw(text);
  if (root.t !== 'o') throw new TypeError('tokenizer.json must contain an object');
  rawSet(root, 'padding', rawFromValue(null));
  rawSet(root, 'truncation', rawFromValue(null));
  if (options.tokenizerClass && REBUILT_TOKENIZER_CLASSES.has(options.tokenizerClass)) rebuildT5(root);
  normalizeModel(rawGet(root, 'model'));
  for (const key of ['normalizer', 'pre_tokenizer', 'post_processor', 'decoder']) normalizeComponent(rawGet(root, key));
  return emitJsonRaw(root, { sortKeys: true, separators: [',', ':'] });
}
