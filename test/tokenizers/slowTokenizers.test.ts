/**
 * Checkpoints with slow vocabulary files only (``vocab.txt``, or
 * ``vocab.json`` and ``merges.txt``) load as ``AutoTokenizer`` loads them
 * (fixtures: scripts/fixtures/slow_tokenizer_fixtures.py).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { FastTokenizer } from '../../src/_internal/tokenizers/index.js';

const root = new URL('../fixtures/slow_tokenizers/', import.meta.url).pathname;
const records = JSON.parse(readFileSync(join(root, 'records.json'), 'utf8')) as {
  texts: [string, string];
  cases: Record<string, { to_str: string; config: Record<string, unknown>; single: number[]; pair: number[] }>;
};

describe('slow vocabulary files', () => {
  it.each(Object.keys(records.cases))('%s matches AutoTokenizer', async (name) => {
    const expected = records.cases[name]!;
    const tokenizer = await FastTokenizer.fromDirectory(join(root, name));
    expect(tokenizer.rustJsonText).toBe(expected.to_str);
    expect(JSON.parse(JSON.stringify(tokenizer.configuration()))).toEqual(expected.config);
    expect(tokenizer.encode(records.texts[0]).inputIds[0]).toEqual(expected.single);
    expect(tokenizer.encode(records.texts[0], { textPair: records.texts[1] }).inputIds[0]).toEqual(expected.pair);
  });
});

describe('foundations with slow vocabulary files', () => {
  it('get the tokenizer AutoTokenizer builds from them', async () => {
    const { cpSync, mkdtempSync, rmSync } = await import('node:fs');
    const { tmpdir } = await import('node:os');
    const { loadNativeFoundation } = await import('../../src/_internal/native/foundation.js');
    const directory = mkdtempSync(join(tmpdir(), 'tensorcode-slow-foundation-'));
    try {
      cpSync(new URL('../fixtures/torch_checkpoint/bert_bin', import.meta.url).pathname, directory, { recursive: true });
      for (const name of ['vocab.txt', 'tokenizer_config.json']) cpSync(join(root, 'bert', name), join(directory, name));
      const loaded = await loadNativeFoundation(directory, { head: 'sequence-classification', localFilesOnly: true });
      expect(loaded.tokenizer!.rustJsonText).toBe(records.cases.bert!.to_str);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
