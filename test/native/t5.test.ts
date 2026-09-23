import { describe, expect, it } from 'vitest';
import { NativeConfig } from '../../src/_internal/native/config.js';
import { T5ForConditionalGeneration } from '../../src/_internal/native/t5.js';
import { generateSeq2Seq } from '../../src/_internal/native/generation.js';
import { loadModelFromBytes, noGrad } from '../../src/nn/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, fromJson, ints } from '../helpers/fixtures.js';

const cases = fixtureJson('native_t5.json');

describe('T5 matches transformers', () => {
  for (const [name, record] of Object.entries<any>(cases)) {
    const build = () => {
      const model = new T5ForConditionalGeneration(NativeConfig.fromDict(record.config)).eval();
      loadModelFromBytes(model, fixtureBytes(`native_${name}.safetensors`));
      return model;
    };

    it(`${name}: state layout, encoder, logits and loss`, () => {
      const model = build();
      expect([...model.stateDict().keys()]).toEqual(record.state_keys);
      noGrad(() => {
        const inputs = { inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask) };
        expectClose(model.encode(inputs).data, record.encoder_last_hidden_state.data, 2e-5, 1e-4);
        const output = model.forward({ ...inputs, labels: ints(record.labels) });
        expectClose(output.logits.data, record.logits.data, 5e-5, 1e-4);
        expect(output.loss!.item()).toBeCloseTo(record.loss, 4);
        const embedded = model.forward({ inputsEmbeds: fromJson(record.inputs_embeds), attentionMask: ints(record.inputs_embeds_mask), labels: ints(record.labels) });
        expect(embedded.loss!.item()).toBeCloseTo(record.embeds_loss, 4);
      });
    });

    it(`${name}: generation`, () => {
      const model = build();
      const inputs = { inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask) };
      const generation = { generationConfig: record.generation_config };
      const run = (settings: Record<string, unknown>, source = inputs) => generateSeq2Seq(model, source, settings, generation).tolist();
      expect(run({ max_new_tokens: 6, do_sample: false })).toEqual(record.greedy);
      expect(run({ max_new_tokens: 6, min_new_tokens: 4, do_sample: false })).toEqual(record.greedy_min);
      expect(run({ max_new_tokens: 6, repetition_penalty: 1.5, do_sample: false })).toEqual(record.repetition);
      const first = { inputIds: ints(record.input_ids.slice(0, 1)), attentionMask: ints(record.attention_mask.slice(0, 1)) };
      expect(run({ max_new_tokens: 5, num_beams: 3, num_return_sequences: 3, do_sample: false }, first)).toEqual(record.beam);
      expect(run({ max_new_tokens: 5, num_beams: 2, do_sample: false, length_penalty: 0.5, early_stopping: true })).toEqual(record.beam_batch);
      const embeds = { inputsEmbeds: fromJson(record.inputs_embeds), attentionMask: ints(record.inputs_embeds_mask) };
      expect(run({ max_new_tokens: 5, do_sample: false }, embeds as any)).toEqual(record.embeds_greedy);
    });
  }

  it('trains through the teacher-forced loss', () => {
    const record = cases.t5_0;
    const model = new T5ForConditionalGeneration(NativeConfig.fromDict(record.config)).train();
    const loss = model.forward({ inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask), labels: ints(record.labels) }).loss!;
    loss.backward();
    // Untied T5 (transformers 5.17) never reads ``shared`` in the forward pass.
    const trainable = model.namedParameters().filter(([name]) => name !== 'shared.weight');
    expect(trainable.every(([, parameter]) => parameter.grad !== null && parameter.grad.allFinite())).toBe(true);
    expect(model.shared.weight.grad).toBeNull();
  });
});
