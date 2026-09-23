import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { FastTokenizer } from '../../src/_internal/tokenizers/index.js';
import { sha256Hex } from '../../src/_internal/json.js';
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
    });
  }
});
