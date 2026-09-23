import { describe, expect, it } from 'vitest';
import { NativeConfig } from '../../src/_internal/native/config.js';
import {
  BertForSequenceClassification, BertModel, DistilBertForSequenceClassification, DistilBertModel,
  ElectraForSequenceClassification, ElectraModel, RobertaForSequenceClassification, RobertaModel,
} from '../../src/_internal/native/bert.js';
import { loadModelFromBytes, noGrad } from '../../src/nn/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, fromJson, ints } from '../helpers/fixtures.js';

const cases = fixtureJson('native_encoders.json');
const bases: Record<string, any> = { bert: BertModel, roberta: RobertaModel, electra: ElectraModel, distilbert: DistilBertModel };
const heads: Record<string, any> = {
  bert: BertForSequenceClassification, roberta: RobertaForSequenceClassification,
  electra: ElectraForSequenceClassification, distilbert: DistilBertForSequenceClassification,
};

describe('BERT-family encoders match transformers', () => {
  for (const [name, record] of Object.entries<any>(cases)) {
    it(`${name} base model`, () => {
      const model = new bases[name](NativeConfig.fromDict(record.config)).eval();
      expect([...model.stateDict().keys()]).toEqual(record.state_keys);
      loadModelFromBytes(model, fixtureBytes(`native_${name}.safetensors`));
      noGrad(() => {
        const output = model.forward({
          inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask),
          tokenTypeIds: record.token_type_ids ? ints(record.token_type_ids) : null,
        });
        expect(output.lastHiddenState.shape).toEqual(record.last_hidden_state.shape);
        expectClose(output.lastHiddenState.data, record.last_hidden_state.data, 2e-5, 1e-4);
        if (record.pooler_output) expectClose(output.poolerOutput.data, record.pooler_output.data, 2e-5, 1e-4);
        const embedded = model.forward({ inputsEmbeds: fromJson(record.inputs_embeds), attentionMask: ints(record.inputs_embeds_mask) });
        expectClose(embedded.lastHiddenState.data, record.inputs_embeds_output.data, 2e-5, 1e-4);
      });
    });

    it(`${name} sequence classifier`, () => {
      const model = new heads[name](NativeConfig.fromDict(record.classifier.config)).eval();
      expect([...model.stateDict().keys()]).toEqual(record.classifier.state_keys);
      loadModelFromBytes(model, fixtureBytes(`native_${name}_classifier.safetensors`));
      noGrad(() => {
        const output = model.forward({
          inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask),
          tokenTypeIds: record.token_type_ids ? ints(record.token_type_ids) : null,
        });
        expectClose(output.logits.data, record.classifier.logits.data, 2e-5, 1e-4);
      });
    });
  }
});
