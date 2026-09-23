/**
 * Checkpoints without tokenizer files get the tokenizer ``AutoTokenizer``
 * builds from the model type's class defaults (fixtures:
 * scripts/fixtures/blank_tokenizer_fixtures.py).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { FastTokenizer } from '../../src/_internal/tokenizers/index.js';

const records = JSON.parse(readFileSync(new URL('../fixtures/blank_tokenizers.json', import.meta.url), 'utf8')) as Record<string, {
  config_json: string; class: string; to_str: string; special_tokens_map: Record<string, string>;
  tokenizer_config: Record<string, unknown>; ids: number[];
}>;

describe('tokenizers of checkpoints without tokenizer files', () => {
  it.each(Object.keys(records))('%s matches AutoTokenizer', (modelType) => {
    const record = records[modelType]!;
    const tokenizer = FastTokenizer.blankForModel(record.config_json)!;
    expect(tokenizer.rustJsonText).toBe(record.to_str);
    const special = Object.fromEntries(Object.entries(tokenizer.specialTokensMap).filter(([, value]) => typeof value === 'string'));
    expect(special).toEqual(record.special_tokens_map);
    expect(JSON.parse(JSON.stringify(tokenizer.configuration()))).toEqual(record.tokenizer_config);
    expect(tokenizer.encode("hello world [UNK] </s>").inputIds[0]).toEqual(record.ids);
  });

  it('are absent where transformers cannot build the class without files', () => {
    expect(FastTokenizer.blankForModel(JSON.stringify({ model_type: 'llama' }))).toBeNull();
  });
});

describe('foundations without tokenizer files', () => {
  it('load the class-default tokenizer, as AutoTokenizer.from_pretrained does', async () => {
    const { loadNativeFoundation } = await import('../../src/_internal/native/foundation.js');
    const directory = new URL('../fixtures/training/deberta_v3_foundation', import.meta.url).pathname;
    const loaded = await loadNativeFoundation(directory, { head: 'sequence-classification', localFilesOnly: true });
    // Python: AutoTokenizer.from_pretrained(directory)('hello world') -> [5, 1, 1, 4].
    expect(loaded.tokenizer!.rustJsonText).toBe(records['deberta-v2']!.to_str);
    expect(loaded.tokenizer!.encode('hello world').inputIds[0]).toEqual([5, 1, 1, 4]);
  });
});
