/**
 * Interop acceptance: a real Python-published Investigator (Electra foundation,
 * rank-only) loads from the local Hugging Face cache and reproduces the
 * receipts Python recorded for the same inputs. Skipped when not cached.
 */
import { describe, expect, it } from 'vitest';
import { Investigator } from '../../src/tools/investigator.js';
import { cachedSnapshot } from '../helpers/hub.js';
import { expectClose } from '../helpers/gradcheck.js';
import { TOOLS } from './helpers.js';

const REPO = 'jacob-valdez/tensorcode-investigator-hotpot-001';
const REVISION = '1bc225917c3646fcb9702df91ff5e445846c1dc7';
const cached = cachedSnapshot(REPO, REVISION) !== null && TOOLS.hotpot !== null;

describe('published Python Investigator artifact', () => {
  it.skipIf(!cached)('loads offline and reproduces Python receipts', async () => {
    const tool = await Investigator.fromPretrained(REPO, { revision: REVISION, localFilesOnly: true });
    expect(Object.keys(tool.configuration()).sort()).toEqual(TOOLS.hotpot.configuration_keys);
    expect(tool.training).toBe(false);
    TOOLS.hotpot.cases.forEach((input: Record<string, unknown>, index: number) => {
      const expected = TOOLS.hotpot.receipts[index];
      const receipt = tool.call(structuredClone(input));
      expect(receipt.selected_id).toBe(expected.selected_id);
      expect(receipt.attention_source_ids).toEqual(expected.attention_source_ids);
      expect((receipt.candidates as { id: string }[]).map((item) => item.id)).toEqual(expected.candidates.map((item: { id: string }) => item.id));
      expectClose((receipt.candidates as { predicted_score: number }[]).map((item) => item.predicted_score),
        expected.candidates.map((item: { predicted_score: number }) => item.predicted_score), 1e-4, 1e-4);
      expectClose((receipt.candidates as { probability: number }[]).map((item) => item.probability),
        expected.candidates.map((item: { probability: number }) => item.probability), 1e-4, 1e-4);
      expectClose((receipt.attention as number[][]).flat(), expected.attention.flat(), 1e-4, 1e-4);
      expectClose((receipt.relations as number[][]).flat(), expected.relations.flat(), 1e-4, 1e-4);
      expect(receipt.evidence).toEqual(expected.evidence);
    });
  }, 300_000);
});
