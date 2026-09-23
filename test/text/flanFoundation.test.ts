/** End-to-end owned operations on the cached ``google/flan-t5-small`` snapshot (skipped when absent). */
import { describe, expect, it } from 'vitest';
import * as text from '../../src/ops/text/index.js';
import { PYTHON_FLOAT_KEYS, pythonJsonDumps, sha256Hex } from '../../src/_internal/json.js';
import { bindingRecords } from '../../src/_internal/fingerprint.js';
import { cachedSnapshot } from '../helpers/hub.js';
import { fixtureJson } from '../helpers/fixtures.js';

const expected = fixtureJson('text/flan.json');
const cached = cachedSnapshot('google/flan-t5-small', expected.snapshot) !== null;

describe('flan-t5-small owned text operations', () => {
  it.skipIf(!cached)('classify (likelihood) and transform match Python', async () => {
    const options = { revision: expected.snapshot, localFilesOnly: true };
    const classify = await text.Classify.fromFoundation('google/flan-t5-small', {
      ...options,
      config: {
        labels: ['billing', 'technical'], descriptions: { billing: 'payments, charges and refunds' },
        instructions: 'Route the support ticket', decoding: 'likelihood',
      },
    });
    const configuration = classify.configuration();
    expect(configuration.foundation).toEqual(expected.classify.foundation);
    // Including the embedded tokenizer JSON: its Unigram scores follow Rust
    // ``Tokenizer.from_str`` float parsing, as in Python.
    const tokenizer = configuration.tokenizer as Record<string, unknown>;
    expect(JSON.parse(tokenizer.json as string).model.vocab.length).toBe(expected.classify.tokenizer_vocab_size);
    const floatKeys = new Set([...PYTHON_FLOAT_KEYS, 'length_penalty']);
    const serialized = pythonJsonDumps(configuration, { sortKeys: true, floatKeys, separators: [',', ':'] });
    expect(sha256Hex(serialized)).toBe(expected.classify.configuration_sha256);
    const fingerprints = Object.fromEntries(Object.entries(bindingRecords(classify.operationBindings())).map(([name, record]) => [name, record.fingerprint]));
    expect(fingerprints).toEqual(expected.classify.fingerprints);
    const value = [new text.Message('user', 'I was charged twice for my subscription.')];
    const scores = classify.nativeModel.scoreAlternatives(classify._scoringRequest(value, null), classify._alternatives());
    scores.forEach((score, index) => expect(score).toBeCloseTo(expected.classify.scores[index], 3));
    const result = classify.call(value);
    expect(result.label).toBe('billing');
    expect(classify.loss(value, { label: 'billing', distribution: null, confidence: null, abstained: false }).item())
      .toBeCloseTo(expected.classify.loss, 3);

    const transform = await text.Transform.fromFoundation('google/flan-t5-small', {
      ...options, config: { instructions: 'Answer the question.', generation: { max_new_tokens: 8 } },
    });
    const question = [new text.Message('user', 'What color is the sky?')];
    expect(transform.call(question).at(-1)!.content).toBe(expected.transform.text);
    expect(transform.loss(question, 'blue').item()).toBeCloseTo(expected.transform.loss, 3);
  }, 180_000);
});
