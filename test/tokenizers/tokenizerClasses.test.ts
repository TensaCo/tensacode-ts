/**
 * transformers 5 tokenizer classes rebuild their backend from the
 * ``tokenizer.json`` vocabulary and their flags, and ``TokenizersBackend``
 * registers special and ``added_tokens_decoder`` tokens (fixtures:
 * scripts/fixtures/tokenizer_classes_fixtures.py).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { FastTokenizer } from '../../src/_internal/tokenizers/index.js';
import { sha256Hex } from '../../src/_internal/json.js';
import { cachedSnapshot } from '../helpers/hub.js';

const root = new URL('../fixtures/tokenizer_classes/', import.meta.url).pathname;
interface Expected {
  class: string; to_str: string; config: { json: string } & { [key: string]: unknown };
  single: number[]; pair: number[]; decoded: string; repo?: string; snapshot?: string;
}
const records = JSON.parse(readFileSync(join(root, 'records.json'), 'utf8')) as {
  texts: [string, string]; synthetic: Record<string, Expected>; hub: Record<string, Expected>;
};
const [text, pairText] = records.texts;

function check(tokenizer: FastTokenizer, expected: Expected, digest: boolean): void {
  const backend = tokenizer.rustJsonText;
  expect(digest ? sha256Hex(backend) : backend).toBe(expected.to_str);
  const config = JSON.parse(JSON.stringify(tokenizer.configuration()));
  if (digest) config.json = sha256Hex(config.json);
  expect(config).toEqual(expected.config);
  expect(tokenizer.encode(text).inputIds[0]).toEqual(expected.single);
  expect(tokenizer.encode(text, { textPair: pairText }).inputIds[0]).toEqual(expected.pair);
  expect(tokenizer.decode(expected.single, { skipSpecialTokens: true })).toBe(expected.decoded);
}

describe('tokenizer class construction', () => {
  it.each(Object.keys(records.synthetic))('%s matches AutoTokenizer', async (name) => {
    check(await FastTokenizer.fromDirectory(join(root, name)), records.synthetic[name]!, false);
  });

  for (const [name, record] of Object.entries(records.hub)) {
    const snapshot = cachedSnapshot(record.repo!, record.snapshot!);
    it.skipIf(!snapshot)(`${name} (${record.repo}) matches AutoTokenizer`, async () => {
      check(await FastTokenizer.fromDirectory(snapshot!), record, true);
    });
  }
});
