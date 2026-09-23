import { describe, expect, it } from 'vitest';
import { NativeConfig } from '../../src/_internal/native/config.js';
import { ViTModel } from '../../src/_internal/native/vit.js';
import { CLIPModel } from '../../src/_internal/native/clip.js';
import { loadModelFromBytes, noGrad } from '../../src/nn/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, fromJson, ints } from '../helpers/fixtures.js';

const cases = fixtureJson('native_vision.json');

describe('vision foundations match transformers', () => {
  it('ViTModel (no pooler)', () => {
    const record = cases.vit;
    const model = new ViTModel(NativeConfig.fromDict(record.config), { addPoolingLayer: false }).eval();
    expect([...model.stateDict().keys()]).toEqual(record.state_keys);
    loadModelFromBytes(model, fixtureBytes('native_vit.safetensors'));
    noGrad(() => {
      const output = model.forward({ pixelValues: fromJson(record.pixel_values) });
      expectClose(output.lastHiddenState.data, record.last_hidden_state.data, 2e-5, 1e-4);
    });
  });

  it('CLIPModel', () => {
    const record = cases.clip;
    const model = new CLIPModel(NativeConfig.fromDict(record.config)).eval();
    expect([...model.stateDict().keys()]).toEqual(record.state_keys);
    loadModelFromBytes(model, fixtureBytes('native_clip.safetensors'));
    noGrad(() => {
      const output = model.forward({ inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask), pixelValues: fromJson(record.pixel_values) });
      expectClose(output.text.lastHiddenState.data, record.text_last_hidden_state.data, 2e-5, 1e-4);
      expectClose(output.vision.lastHiddenState.data, record.vision_last_hidden_state.data, 2e-5, 1e-4);
      expectClose(output.vision.poolerOutput.data, record.vision_pooler_output.data, 2e-5, 1e-4);
      expectClose(output.textEmbeds.data, record.text_embeds.data, 2e-5, 1e-4);
      expectClose(output.imageEmbeds.data, record.image_embeds.data, 2e-5, 1e-4);
      expectClose(output.logitsPerImage.data, record.logits_per_image.data, 1e-4, 1e-4);
    });
  });
});
