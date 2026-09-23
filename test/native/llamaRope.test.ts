/**
 * Every transformers RoPE type (``default``, ``linear``, ``dynamic``, ``yarn``,
 * ``longrope``, ``llama3``, ``proportional``) in the Llama text model
 * (``scripts/fixtures/llama_rope_fixtures.py``): frequencies, full forwards
 * and cached incremental decoding past ``original_max_position_embeddings``,
 * including the stateful ``dynamic`` growth and reset.
 */
import { describe, expect, it } from 'vitest';
import { nativeConfig } from '../../src/_internal/native/config.js';
import { LlamaModel, LlamaRotaryEmbedding } from '../../src/_internal/native/llama.js';
import { cat, loadModelFromBytes, noGrad } from '../../src/nn/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, ints } from '../helpers/fixtures.js';

const records = fixtureJson('rope/records.json');

describe('Llama RoPE variants match transformers', () => {
  for (const [name, record] of Object.entries<any>(records.variants)) {
    it(name, () => {
      const model = new LlamaModel(nativeConfig(record.config));
      loadModelFromBytes(model, fixtureBytes(`rope/${name}.safetensors`));
      model.eval();
      const rotary = model.rotary_emb as LlamaRotaryEmbedding;
      expectClose(Array.from(rotary.getBuffer('original_inv_freq')!.data), record.inv_freq, 1e-6, 1e-7);
      expect(rotary.attentionScaling).toBeCloseTo(record.attention_scaling, 12);
      const ids: number[][] = records.input_ids;
      noGrad(() => {
        const full = model.forward({ inputIds: ints(ids) });
        expectClose(full.data, record.full.data, 2e-5, 2e-5);
        const cache = model.newCache();
        const steps = [model.forward({ inputIds: ints([ids[0]!.slice(0, 6)]), cache }).select(1, 5)];
        for (let position = 6; position < ids[0]!.length; position += 1) {
          steps.push(model.forward({ inputIds: ints([[ids[0]![position]!]]), cache }).select(1, 0));
        }
        expectClose(cat(steps.map((step) => step.unsqueeze(1)), 1).data, record.steps.data, 2e-5, 2e-5);
        const short = model.forward({ inputIds: ints([ids[0]!.slice(0, 5)]) });
        expectClose(short.data, record.short.data, 2e-5, 2e-5);
      });
    });
  }
});
