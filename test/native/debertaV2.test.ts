/** DeBERTa-v2/v3 parity with transformers 5.17 (fixtures from ``scripts/fixtures/training_fixtures.py``). */
import { describe, expect, it } from 'vitest';
import { NativeConfig } from '../../src/_internal/native/config.js';
import {
  DebertaV2ForSequenceClassification, DebertaV2Model, buildRelativePosition, logBucketPosition,
} from '../../src/_internal/native/debertaV2.js';
import { createNativeModel } from '../../src/_internal/native/registry.js';
import { loadNativeFoundation } from '../../src/_internal/native/foundation.js';
import { pythonJsonDumps } from '../../src/_internal/json.js';
import { loadModelFromBytes, noGrad, tensor } from '../../src/nn/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, ints } from '../helpers/fixtures.js';

const cases = fixtureJson('training/deberta_v2.json');
const names = ['v3', 'v2', 'plain'];
const ours = (value: unknown) => pythonJsonDumps(value, { sortKeys: true, separators: [',', ':'] });

describe('DeBERTa-v2 matches transformers', () => {
  for (const name of names) {
    const record = cases[name];

    it(`${name}: configuration round-trips`, () => {
      const config = NativeConfig.fromDict(record.input);
      expect(ours(config.toDict())).toBe(ours(record.to_dict));
      expect(ours(config.toDiffDict())).toBe(ours(record.diff));
      expect(NativeConfig.fromDict(record.diff).toDiffDict()).toEqual(record.diff);
      expect(NativeConfig.fromDict(record.to_dict).toDict()).toEqual(record.to_dict);
    });

    it(`${name}: base model state and outputs`, () => {
      const model = new DebertaV2Model(NativeConfig.fromDict(record.input)).eval();
      expect([...model.stateDict().keys()]).toEqual(record.state_keys);
      loadModelFromBytes(model, fixtureBytes(`training/deberta_${name}.safetensors`));
      noGrad(() => {
        const output = model.forward({
          inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask),
          tokenTypeIds: record.token_type_ids ? ints(record.token_type_ids) : null,
        });
        expect(output.lastHiddenState.shape).toEqual(record.last_hidden_state.shape);
        expectClose(output.lastHiddenState.data, record.last_hidden_state.data, 2e-5, 1e-4);
        expect(output.poolerOutput).toBeNull();
      });
    });

    it(`${name}: sequence classifier`, () => {
      const config = NativeConfig.fromDict(record.classifier.config);
      const model = createNativeModel(config, 'sequence-classification') as DebertaV2ForSequenceClassification;
      model.eval();
      expect(model).toBeInstanceOf(DebertaV2ForSequenceClassification);
      expect([...model.stateDict().keys()]).toEqual(record.classifier.state_keys);
      loadModelFromBytes(model, fixtureBytes(`training/deberta_${name}_classifier.safetensors`));
      noGrad(() => {
        const output = model.forward({
          inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask),
          tokenTypeIds: record.token_type_ids ? ints(record.token_type_ids) : null,
        });
        expectClose(output.logits.data, record.classifier.logits.data, 2e-5, 1e-4);
      });
    });
  }

  it('log-bucketed relative positions match make_log_bucket_position', () => {
    for (const table of cases.buckets) {
      const values = (table.values as number[]).map((_, index) => logBucketPosition(table.start + index, table.bucket_size, table.max_position));
      expect(values).toEqual(table.values);
    }
    const plain = buildRelativePosition(3, 3);
    expect([...plain]).toEqual([0, -1, -2, 1, 0, -1, 2, 1, 0]);
  });

  it('trains through disentangled attention (gradients reach relative embeddings)', () => {
    const record = cases.v3;
    const model = new DebertaV2ForSequenceClassification(NativeConfig.fromDict(record.classifier.config));
    loadModelFromBytes(model, fixtureBytes('training/deberta_v3_classifier.safetensors'));
    model.eval();
    const output = model.forward({ inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask) });
    output.logits.sum().backward();
    const relative = model.deberta.encoder.rel_embeddings!.weight.grad!;
    expect(relative.abs().sum().item()).toBeGreaterThan(0);
    expect(model.deberta.embeddings.word_embeddings.weight.grad!.abs().sum().item()).toBeGreaterThan(0);
    const embedded = new DebertaV2Model(NativeConfig.fromDict(record.input)).eval();
    loadModelFromBytes(embedded, fixtureBytes('training/deberta_v3.safetensors'));
    noGrad(() => {
      const ids = ints(record.input_ids);
      const viaIds = embedded.forward({ inputIds: ids, attentionMask: ints(record.attention_mask) }).lastHiddenState;
      const embeds = embedded.getInputEmbeddings().forward(ids);
      const viaEmbeds = embedded.forward({ inputsEmbeds: embeds, attentionMask: ints(record.attention_mask) }).lastHiddenState;
      expectClose(viaEmbeds.data, viaIds.data, 1e-6, 1e-6);
    });
    expect(() => embedded.forward({})).toThrow(/input_ids or inputs_embeds/);
    expect(tensor([1]).item()).toBe(1);
  });

  it('loads a pretrained-style DeBERTa-v3 classifier directory', async () => {
    const directory = new URL('../fixtures/training/deberta_v3_foundation', import.meta.url).pathname;
    const loaded = await loadNativeFoundation(directory, { head: 'sequence-classification', localFilesOnly: true });
    expect(loaded.model).toBeInstanceOf(DebertaV2ForSequenceClassification);
    const record = cases.v3;
    noGrad(() => {
      const output = (loaded.model as DebertaV2ForSequenceClassification).forward({
        inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask),
      });
      expectClose(output.logits.data, record.classifier.logits.data, 2e-5, 1e-4);
    });
    expect(loaded.config.get('id2label')).toEqual({ 0: 'entailment', 1: 'neutral', 2: 'contradiction' });
  });
});
