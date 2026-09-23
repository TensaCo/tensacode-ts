/**
 * DeBERTa-v3 against the real cached ``cross-encoder/nli-deberta-v3-small``
 * checkpoint: tokenizer ids and NLI logits equal transformers'
 * (``scripts/fixtures/deberta_cached_fixtures.py``). Skips when the checkpoint
 * is not in the local Hugging Face cache.
 */
import { describe, expect, it } from 'vitest';
import { loadNativeFoundation } from '../../src/_internal/native/foundation.js';
import type { DebertaV2ForSequenceClassification } from '../../src/_internal/native/debertaV2.js';
import { noGrad } from '../../src/nn/index.js';
import { cachedSnapshot } from '../helpers/hub.js';
import { fixtureJson, ints } from '../helpers/fixtures.js';
import { expectClose } from '../helpers/gradcheck.js';

const record = fixtureJson('deberta_cached.json');

describe('real DeBERTa-v3 checkpoint', () => {
  it.skipIf(!cachedSnapshot(record.repo, record.snapshot))(`${record.repo} matches transformers`, async () => {
    const loaded = await loadNativeFoundation(record.repo, { revision: record.snapshot, localFilesOnly: true, head: 'sequence-classification' });
    const pairs = record.pairs as [string, string][];
    const batch = loaded.tokenizer!.encode(pairs.map(([premise]) => premise), { textPair: pairs.map(([, hypothesis]) => hypothesis), padding: true });
    expect(batch.inputIds).toEqual(record.input_ids);
    expect(batch.attentionMask).toEqual(record.attention_mask);
    const logits = noGrad(() => (loaded.model as DebertaV2ForSequenceClassification).forward({
      inputIds: ints(batch.inputIds), attentionMask: ints(batch.attentionMask), tokenTypeIds: ints(record.token_type_ids),
    }).logits);
    expectClose(logits.data, record.logits.data, 2e-5, 1e-4);
    const labels = loaded.config.get('id2label') as Record<string, string>;
    const predictions = logits.argmax(-1).tolist() as number[];
    expect(predictions.map((index) => labels[String(index)])).toEqual(record.predictions);
  }, 120_000);
});
