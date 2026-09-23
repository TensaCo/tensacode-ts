/** Port of ``tests/vec/test_text_encoding.py`` (VocabularyEncoder). */
import { describe, expect, it } from 'vitest';
import { type Tensor } from '../../src/nn/index.js';
import { VocabularyEncoder } from '../../src/ops/vec/index.js';
import { tokenize } from '../../src/_internal/vec/vocabulary.js';

describe('vocabulary encoding (tests/vec/test_text_encoding.py)', () => {
  it('batches strings and trains embedding parameters', () => {
    const encoder = new VocabularyEncoder({ vocabulary: ['hello', 'world'], dimensions: 4 });
    const result = encoder.call(['hello world', 'unknown', '']) as Tensor;
    expect(result.shape).toEqual([3, 4]);
    expect(result.allFinite()).toBe(true);
    result.sum().backward();
    expect(encoder.embedding.weight.grad).not.toBeNull();
    expect((encoder.call('hello') as Tensor).shape).toEqual([4]);
  });

  it('rejects non-text without stringifying it', () => {
    const encoder = new VocabularyEncoder({ vocabulary: ['hello'], dimensions: 4 });
    expect(() => encoder.call([{} as unknown as string])).toThrow(TypeError);
    expect(() => encoder.call([])).toThrow(/Empty batch/);
  });

  it('serialized encoder weights produce identical encodings', () => {
    const a = new VocabularyEncoder({ vocabulary: ['hello', 'world'], dimensions: 4 });
    const b = new VocabularyEncoder({ vocabulary: ['hello', 'world'], dimensions: 4 });
    b.loadStateDict(a.stateDict());
    expect((a.call(['hello', 'world']) as Tensor).equal(b.call(['hello', 'world']) as Tensor)).toBe(true);
  });

  it('validates configuration and tokenizes like Python', () => {
    expect(() => new VocabularyEncoder({ vocabulary: ['a', 'a'] })).toThrow(/unique/);
    expect(() => new VocabularyEncoder({ vocabulary: 'a' as any })).toThrow(/list/);
    expect(() => new VocabularyEncoder({ vocabulary: ['a'], dimensions: 0 })).toThrow(/positive integer/);
    expect(() => new VocabularyEncoder({ vocabulary: ['a'], dimensions: 4, output_space: { name: 'x', dimensions: 3 } })).toThrow(/must match dimensions/);
    expect(() => new VocabularyEncoder({ vocabulary: ['a'], extra: 1 } as any)).toThrow(/Unknown configuration fields/);
    expect(new VocabularyEncoder({ vocabulary: ['a'] }).configuration()).toEqual({ vocabulary: ['a'], dimensions: 64, output_space: null });
    expect(tokenize('Hello, WORLD! café_1')).toEqual(['hello', ',', 'world', '!', 'café_1']);
    expect(() => new VocabularyEncoder({ vocabulary: ['a'] }).call('a', { context: { x: 1 } })).toThrow(/context/);
  });
});
