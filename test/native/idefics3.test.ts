/** Idefics3 (SmolVLM) matches transformers (``scripts/fixtures/idefics3_fixtures.py``). */
import { describe, expect, it } from 'vitest';
import { nativeConfig } from '../../src/_internal/native/config.js';
import { Idefics3ForConditionalGeneration, greedySettings } from '../../src/_internal/native/idefics3.js';
import { createNativeModel } from '../../src/_internal/native/registry.js';
import { loadModelFromBytes, noGrad, tensor } from '../../src/nn/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, fromJson, ints } from '../helpers/fixtures.js';

const cases = fixtureJson('native_idefics3.json');

function load(name: string): Idefics3ForConditionalGeneration {
  const model = createNativeModel(nativeConfig(cases[name].config), 'image-text-to-text') as Idefics3ForConditionalGeneration;
  loadModelFromBytes(model, fixtureBytes(`native_idefics3_${name}.safetensors`));
  return model.eval();
}

describe('Idefics3ForConditionalGeneration', () => {
  for (const name of Object.keys(cases)) {
    const record = cases[name];
    it(`${name}: module topology and state keys`, () => {
      const model = createNativeModel(nativeConfig(record.config), 'image-text-to-text');
      expect(model.namedModules().map(([path]) => path)).toEqual(record.modules);
      expect([...model.stateDict().keys()]).toEqual(record.state_keys);
    });

    it(`${name}: vision tower, image features, logits and loss`, () => {
      const model = load(name);
      const pixels = fromJson(record.pixel_values);
      const pixelMask = tensor(record.pixel_attention_mask.flat(3), { shape: [1, 3, 8, 8], dtype: 'int64' });
      noGrad(() => {
        const vision = model.model.vision_model.forward(pixels.select(0, 0).slice(0, 0, 2));
        expectClose(vision.data, record.vision.data, 1e-5, 1e-5);
        const features = model.getImageFeatures(pixels, pixelMask);
        expect(features.shape).toEqual(record.image_features.shape);
        expectClose(features.data, record.image_features.data, 1e-5, 1e-5);
        const output = model.forward({
          inputIds: ints(record.input_ids), attentionMask: ints(record.input_ids.map((row: number[]) => row.map(() => 1))),
          pixelValues: pixels, pixelAttentionMask: pixelMask, labels: ints(record.labels),
        });
        expectClose(output.logits.data, record.logits.data, 1e-5, 1e-5);
        expect(output.loss!.item()).toBeCloseTo(record.loss, 5);
      });
    });

    it(`${name}: greedy generation with logits processors`, () => {
      const model = load(name);
      const pixels = fromJson(record.pixel_values);
      const pixelMask = tensor(record.pixel_attention_mask.flat(3), { shape: [1, 3, 8, 8], dtype: 'int64' });
      const inputIds = ints(record.input_ids);
      const generated = model.generateGreedy({ inputIds, pixelValues: pixels, pixelAttentionMask: pixelMask }, greedySettings(record.generation_config, 6));
      expect(generated).toEqual(record.generated);
      const features = noGrad(() => model.getImageFeatures(pixels, pixelMask));
      const plain = model.generateGreedy({ inputIds, imageHiddenStates: features }, greedySettings({ bos_token_id: 1, eos_token_id: 2, pad_token_id: 0 }, 5));
      expect(plain).toEqual(record.plain_generated);
    });
  }

  it('gradients reach image hidden states through the inputs merger', () => {
    const model = load('gqa');
    const pixels = fromJson(cases.gqa.pixel_values);
    const features = noGrad(() => model.getImageFeatures(pixels)).detach().requiresGrad_();
    const output = model.forward({ inputIds: ints(cases.gqa.input_ids), imageHiddenStates: features, labels: ints(cases.gqa.labels) });
    output.loss!.backward();
    expect(features.grad!.abs().sum().item()).toBeGreaterThan(0);
  });
});
