/**
 * BPE-dropout (Rust ``tokenizers`` ``Word::merge_all`` with ``dropout``): each
 * queued merge is skipped with probability ``dropout`` and skipped merges return
 * to the queue after the next merge. The draws are unseeded in both packages, so
 * the check is on the outcomes and their frequencies. For the word ``abc`` with
 * merges ``a b``, ``b c``, ``ab c`` and dropout 0.5, Python tokenizers 0.22
 * gives each of ``abc``, ``ab c``, ``a bc`` and ``a b c`` a quarter of the time
 * (5125/4970/4940/4965 of 20000 draws).
 */
import { describe, expect, it } from 'vitest';
import { Tokenizer } from '../../src/_internal/tokenizers/tokenizer.js';
import { ValueError } from '../../src/errors.js';

function bpe(dropout: number | null): Tokenizer {
  return Tokenizer.fromString(JSON.stringify({
    version: '1.0', truncation: null, padding: null, added_tokens: [], normalizer: null, pre_tokenizer: null,
    post_processor: null, decoder: null,
    model: {
      type: 'BPE', dropout, unk_token: '<unk>', continuing_subword_prefix: null, end_of_word_suffix: null,
      fuse_unk: false, byte_fallback: false, ignore_merges: false,
      vocab: { a: 0, b: 1, c: 2, ab: 3, bc: 4, abc: 5, '<unk>': 6 }, merges: [['a', 'b'], ['b', 'c'], ['ab', 'c']],
    },
  }));
}

const tokens = (tokenizer: Tokenizer, text: string): string =>
  tokenizer.encodeText(text).map((id) => tokenizer.idToToken(id)).join(' ');

describe('BPE dropout', () => {
  it('merges everything at 0 and nothing at 1, like tokenizers', () => {
    expect(tokens(bpe(null), 'abcabc')).toBe('abc abc');
    expect(tokens(bpe(0), 'abcabc')).toBe('abc abc');
    expect(tokens(bpe(1), 'abcabc')).toBe('a b c a b c');
  });

  it('samples each merge order with the Rust queue frequencies', () => {
    const tokenizer = bpe(0.5);
    const counts = new Map<string, number>();
    const draws = 8000;
    for (let index = 0; index < draws; index += 1) {
      const key = tokens(tokenizer, 'abc');
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    expect([...counts.keys()].sort()).toEqual(['a b c', 'a bc', 'ab c', 'abc']);
    for (const count of counts.values()) expect(Math.abs(count / draws - 0.25)).toBeLessThan(0.03);
  });

  it('rejects a dropout outside [0, 1] with the Rust message', () => {
    expect(() => bpe(1.5)).toThrow(ValueError);
    expect(() => bpe(1.5)).toThrow('Dropout should be between 0 and 1, inclusive');
  });
});
