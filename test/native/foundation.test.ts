import { copyFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadNativeFoundation } from '../../src/_internal/native/foundation.js';
import { generateSeq2Seq } from '../../src/_internal/native/generation.js';
import { parameterAliases } from '../../src/_internal/native/modules.js';
import type { T5ForConditionalGeneration } from '../../src/_internal/native/t5.js';
import type { CLIPModel } from '../../src/_internal/native/clip.js';
import { noGrad, tensor } from '../../src/nn/index.js';
import { sha256Hex } from '../../src/_internal/json.js';
import { cachedSnapshot } from '../helpers/hub.js';
import { fixtureJson, ints } from '../helpers/fixtures.js';
import { expectClose } from '../helpers/gradcheck.js';

const cases = fixtureJson('foundations_cached.json');

function summaryClose(values: ArrayLike<number>, summary: { head: number[]; sum: number; abs_sum: number }, atol = 1e-3): void {
  expectClose(Array.from(values).slice(0, summary.head.length), summary.head, atol, 1e-3);
  let sum = 0;
  let abs = 0;
  for (const value of Array.from(values)) { sum += value; abs += Math.abs(value); }
  expect(Math.abs(abs - summary.abs_sum) / summary.abs_sum).toBeLessThan(1e-4);
  expect(Math.abs(sum - summary.sum)).toBeLessThan(1e-3 * summary.abs_sum);
}

/** Combine config and weights cached under different snapshots into one directory. */
function assemble(repo: string, configSnapshot: string, weightsSnapshot: string, files: string[]): string | null {
  const configDir = cachedSnapshot(repo, configSnapshot);
  const weightsDir = cachedSnapshot(repo, weightsSnapshot);
  if (!configDir || !weightsDir) return null;
  const directory = mkdtempSync(join(tmpdir(), 'tensorcode-foundation-'));
  for (const file of files) copyFileSync(join(configDir, file), join(directory, file));
  copyFileSync(join(weightsDir, 'model.safetensors'), join(directory, 'model.safetensors'));
  return directory;
}

describe('real foundations load like transformers', () => {
  const t5 = cases['google/flan-t5-small'];
  it.skipIf(!t5 || !cachedSnapshot('google/flan-t5-small', t5.snapshot))('google/flan-t5-small (seq2seq)', async () => {
    const loaded = await loadNativeFoundation('google/flan-t5-small', { revision: t5.snapshot, localFilesOnly: true, head: 'seq2seq', restoreRawTieFlags: true });
    const model = loaded.model as T5ForConditionalGeneration;
    expect(loaded.config.toDiffDict()).toEqual(t5.config);
    expect(loaded.generationConfig).toEqual(t5.generation_config);
    const aliases = Object.fromEntries(Object.entries(parameterAliases(model)).filter(([key, value]) => key !== value));
    expect(aliases).toEqual(t5.aliases);
    expect(sha256Hex(loaded.tokenizer!.configuration().json)).toBe(t5.tokenizer_sha);
    const batch = loaded.tokenizer!.encode(['translate English to German: The house is wonderful.', 'Answer the question: what color is the sky?'], { padding: true });
    expect(batch.inputIds).toEqual(t5.input_ids);
    const inputs = { inputIds: ints(batch.inputIds), attentionMask: ints(batch.attentionMask) };
    noGrad(() => summaryClose(model.encode(inputs).data, t5.encoder));
    const greedy = generateSeq2Seq(model, inputs, { max_new_tokens: 8, do_sample: false }, { generationConfig: loaded.generationConfig });
    expect(greedy.tolist()).toEqual(t5.greedy);
    expect(loaded.tokenizer!.batchDecode(greedy, { skipSpecialTokens: true })).toEqual(t5.decoded);
    const first = { inputIds: ints(batch.inputIds.slice(0, 1)), attentionMask: ints(batch.attentionMask.slice(0, 1)) };
    expect(generateSeq2Seq(model, first, { max_new_tokens: 6, num_beams: 3, num_return_sequences: 3, do_sample: false }, { generationConfig: loaded.generationConfig }).tolist()).toEqual(t5.beam);
  }, 120_000);

  const electra = cases['google/electra-small-discriminator'];
  it.skipIf(!electra)('google/electra-small-discriminator (base, prefixed checkpoint)', async () => {
    const directory = assemble('google/electra-small-discriminator', electra.config_snapshot, electra.weights_snapshot, ['config.json', 'tokenizer.json', 'tokenizer_config.json']);
    if (!directory) return;
    try {
      const loaded = await loadNativeFoundation(directory, { head: 'base' });
      const expected = { ...electra.to_dict, _name_or_path: directory };
      expect(loaded.config.toDict()).toEqual(expected);
      expect(loaded.unexpectedKeys.every((key) => key.startsWith('discriminator_predictions.'))).toBe(true);
      const output = noGrad(() => (loaded.model as any).forward({
        inputIds: ints(electra.input_ids), attentionMask: ints(electra.attention_mask), tokenTypeIds: ints(electra.token_type_ids),
      }));
      summaryClose(output.lastHiddenState.data, electra.hidden);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }, 60_000);

  const clip = cases['openai/clip-vit-base-patch32'];
  it.skipIf(!clip)('openai/clip-vit-base-patch32 (legacy position_ids ignored)', async () => {
    const directory = assemble('openai/clip-vit-base-patch32', clip.config_snapshot, clip.weights_snapshot, ['config.json']);
    if (!directory) return;
    try {
      const loaded = await loadNativeFoundation(directory, { head: 'base', tokenizer: false });
      const model = loaded.model as CLIPModel;
      const count = 3 * 224 * 224;
      const values = new Float32Array(count);
      for (let index = 0; index < count; index += 1) values[index] = (Math.sin(index * 0.37) + 1) / 2;
      const output = noGrad(() => model.forward({ inputIds: ints(clip.input_ids), pixelValues: tensor(values, { shape: [1, 3, 224, 224] }) }));
      expectClose(output.textEmbeds.data, clip.text_embeds, 2e-4, 1e-3);
      expectClose(output.imageEmbeds.data, clip.image_embeds, 2e-4, 1e-3);
      expectClose(output.logitsPerImage.data, clip.logits_per_image, 5e-3, 1e-3);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }, 120_000);
});
