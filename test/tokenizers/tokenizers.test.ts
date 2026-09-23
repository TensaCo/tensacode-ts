import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { FastTokenizer } from '../../src/_internal/tokenizers/index.js';
import { canonicalizeJsonText, pythonFloatRepr, sha256Hex } from '../../src/_internal/json.js';
import { rustFloatRepr, rustJsonF64, rustTokenizerString } from '../../src/_internal/tokenizers/serialization.js';
import { cachedSnapshot } from '../helpers/hub.js';

interface Encoding { text: string; ids: number[]; tokens: string[]; no_special: number[]; decoded: string; decoded_skip: string }
interface Case {
  config: Record<string, unknown>;
  json_sha256: string;
  encodings: Encoding[];
  batch: Record<string, number[][]>;
  pairs?: Record<string, number[][]>;
  tokenizer_json?: string;
  special_tokens?: Record<string, string>;
  canonical_json?: string;
  snapshot?: string;
  tokenizer_class?: string;
  rust_json_sha256?: string;
}

const fixture = (name: string): Record<string, Case> => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));

function check(tokenizer: FastTokenizer, record: Case): void {
  const config = tokenizer.configuration();
  expect(sha256Hex(config.json)).toBe(record.json_sha256);
  const { json: _json, ...rest } = config;
  expect(rest).toEqual(record.config);
  for (const encoding of record.encodings) {
    const ids = tokenizer.encode(encoding.text).inputIds[0]!;
    expect(ids, encoding.text).toEqual(encoding.ids);
    expect(tokenizer.convertIdsToTokens(ids)).toEqual(encoding.tokens);
    expect(tokenizer.encode(encoding.text, { addSpecialTokens: false }).inputIds[0]).toEqual(encoding.no_special);
    expect(tokenizer.decode(ids), encoding.text).toBe(encoding.decoded);
    expect(tokenizer.decode(ids, { skipSpecialTokens: true }), encoding.text).toBe(encoding.decoded_skip);
  }
  const texts = record.encodings.slice(0, 3).map((encoding) => encoding.text);
  const batch = tokenizer.encode(texts, { padding: true, truncation: true, maxLength: 9 });
  expect(batch.inputIds).toEqual(record.batch.input_ids);
  expect(batch.attentionMask).toEqual(record.batch.attention_mask);
  if (record.batch.token_type_ids) expect(batch.tokenTypeIds).toEqual(record.batch.token_type_ids);
  if (record.pairs) {
    const all = record.encodings.map((encoding) => encoding.text);
    const pairs = tokenizer.encode(all.slice(0, 2), { textPair: all.slice(2, 4), padding: true, truncation: true, maxLength: 12 });
    expect(pairs.inputIds).toEqual(record.pairs.input_ids);
    expect(pairs.attentionMask).toEqual(record.pairs.attention_mask);
    if (record.pairs.token_type_ids) expect(pairs.tokenTypeIds).toEqual(record.pairs.token_type_ids);
  }
}

describe('synthetic tokenizers match the tokenizers/transformers reference', () => {
  const cases = fixture('tokenizers_synthetic.json');
  for (const [name, record] of Object.entries(cases)) {
    it(name, () => {
      const tokenizer = FastTokenizer.fromJsonString(record.tokenizer_json!, record.special_tokens);
      expect(tokenizer.configuration().json).toBe(record.canonical_json);
      check(tokenizer, record);
      const restored = FastTokenizer.fromConfiguration(tokenizer.configuration());
      expect(restored.configuration()).toEqual(tokenizer.configuration());
      check(restored, record);
    });
  }
});

describe('cached Hugging Face tokenizers', () => {
  const cases = fixture('tokenizers_cached.json');
  for (const [repo, record] of Object.entries(cases)) {
    const directory = cachedSnapshot(repo, record.snapshot!);
    it.skipIf(!directory)(`${repo} (${record.tokenizer_class})`, async () => {
      const tokenizer = await FastTokenizer.fromDirectory(directory!);
      check(tokenizer, record);
      expect(sha256Hex(tokenizer.rustJsonText)).toBe(record.rust_json_sha256);
    });
  }
});

describe('Rust Tokenizer.to_str() serialization', () => {
  const sorted = (text: string) => canonicalizeJsonText(text, { sortKeys: true, separators: [',', ':'] });

  it('reproduces every supported component, vocabulary order and float format', () => {
    const cases = JSON.parse(readFileSync(new URL('../fixtures/tokenizers_rust_serialization.json', import.meta.url), 'utf8')) as Record<string, string>;
    expect(Object.keys(cases).length).toBeGreaterThan(5);
    for (const [name, rust] of Object.entries(cases)) expect(rustTokenizerString(sorted(rust)), name).toBe(rust);
  });

  it('reproduces trained tokenizers with padding and truncation', () => {
    for (const [name, record] of Object.entries(fixture('tokenizers_synthetic.json'))) {
      expect(rustTokenizerString(sorted(record.tokenizer_json!)), name).toBe(record.tokenizer_json);
    }
  });

  it('formats floats like serde_json', () => {
    const cases: [number, string][] = [
      [0, '0.0'], [-0, '-0.0'], [-1, '-1.0'], [0.1, '0.1'], [1e-5, '0.00001'], [1e-6, '1e-6'], [1.5e-7, '1.5e-7'],
      [1234567890123456.8, '1234567890123456.8'], [1e16, '1e+16'], [-1.2345678901234566e17, '-1.2345678901234566e+17'], [5e-324, '5e-324'],
    ];
    for (const [value, text] of cases) expect(rustFloatRepr(value), String(value)).toBe(text);
    expect(rustFloatRepr(0.1, true)).toBe('0.1');
    expect(rustFloatRepr(0.123456789, true)).toBe('0.12345679');
  });
});

describe('Rust tokenizers float parsing (serde_json without float_roundtrip)', () => {
  // [JSON number, the value Rust Tokenizer.from_str parses, as Python repr]
  const pairs: [string, string][] = [
    ['-2.0122928619384766', '-2.012292861938477'], ['-11.573076248168945', '-11.573076248168944'],
    ['-27.346018365555402', '-27.3460183655554'], ['-10.887551307678223', '-10.887551307678224'],
    ['-12.243208885192871', '-12.243208885192873'], ['-9.678209132777223', '-9.678209132777225'],
    ['-11.254974365234375', '-11.254974365234377'], ['-11.850688807410034', '-11.850688807410034'],
    ['-2.5162914485619368', '-2.5162914485619368'], ['-1.6489629317261454e-05', '-1.6489629317261454e-05'],
    ['-9.23635192279068e-05', '-9.23635192279068e-05'], ['0.0', '0.0'], ['-3', '-3.0'],
  ];
  it('matches the values Rust produced', () => {
    for (const [input, output] of pairs) expect(pythonFloatRepr(rustJsonF64(input)), input).toBe(output);
  });

  it('applies to configurations and generic tokenizer files, not class-specific rebuilds', async () => {
    const json = JSON.stringify({
      version: '1.0', truncation: null, padding: null, added_tokens: [], normalizer: null,
      pre_tokenizer: { type: 'Metaspace', replacement: '▁', prepend_scheme: 'always', split: true }, post_processor: null,
      decoder: { type: 'Metaspace', replacement: '▁', prepend_scheme: 'always', split: true },
      model: { type: 'Unigram', unk_id: 0, vocab: [['<unk>', 0.0], ['▁', -2.0122928619384766], ['▁hello', -3.5]], byte_fallback: false },
    });
    const score = (tokenizer: FastTokenizer) => JSON.parse(tokenizer.configuration().json).model.vocab[1][1];
    const configured = FastTokenizer.fromConfiguration({ json, options: {}, special_tokens: { unk_token: '<unk>' }, padding_side: 'right', truncation_side: 'right' } as never);
    expect(configured.configuration().json).toContain('-2.012292861938477');
    expect(score(FastTokenizer.fromJsonString(json, { unk_token: '<unk>' }))).toBe(-2.012292861938477);
    const directory = mkdtempSync(join(tmpdir(), 'tensorcode-unigram-'));
    try {
      writeFileSync(join(directory, 'tokenizer.json'), json);
      writeFileSync(join(directory, 'tokenizer_config.json'), JSON.stringify({ unk_token: '<unk>' }));
      expect(score(await FastTokenizer.fromDirectory(directory))).toBe(-2.012292861938477);
      writeFileSync(join(directory, 'tokenizer_config.json'), JSON.stringify({ unk_token: '<unk>', tokenizer_class: 'DebertaV2Tokenizer' }));
      expect(score(await FastTokenizer.fromDirectory(directory))).toBe(-2.0122928619384766);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});

describe('added tokens', () => {
  it('match tokens containing regex-special characters and hyphens', () => {
    const json = JSON.stringify({
      version: '1.0', truncation: null, padding: null, normalizer: null, pre_tokenizer: { type: 'Whitespace' }, post_processor: null, decoder: null,
      added_tokens: [
        { id: 3, content: '<global-img>', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
        { id: 4, content: '<|a.b*c|>', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
      ],
      model: { type: 'WordLevel', vocab: { '[UNK]': 0, hello: 1, world: 2, '<global-img>': 3, '<|a.b*c|>': 4 }, unk_token: '[UNK]' },
    });
    const tokenizer = FastTokenizer.fromJsonString(json, { unk_token: '[UNK]' });
    expect(tokenizer.encode('hello<global-img>world <|a.b*c|>').inputIds[0]).toEqual([1, 3, 2, 4]);
  });
});

