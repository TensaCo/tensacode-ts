/**
 * Python-saved text/vision artifacts and tiny native foundations: TypeScript
 * loading, numerics, byte-identical re-save, fingerprints and ``fromFoundation``
 * configuration parity (fixtures: ``scripts/fixtures/vec_fixtures.py``).
 */
import { describe, expect, it } from 'vitest';
import { noGrad, tensor } from '../../src/nn/index.js';
import { ImageEncoder, Latent, Space, TextDecoder, TextEncoder } from '../../src/ops/vec/index.js';
import { ImageProcessor, resizeImage } from '../../src/_internal/vec/imageProcessing.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson, fromJson } from '../helpers/fixtures.js';
import { expectByteIdenticalResave, fingerprints, fixturePath, relocate, scratchDirectory } from './helpers.js';

const text = fixtureJson('vec/text.json');
const vision = fixtureJson('vec/vision.json');

const scratch = scratchDirectory();

describe('text artifacts written by Python', () => {
  it('TextEncoder (T5 encoder, sequence readout)', async () => {
    const encoder = await TextEncoder.fromPretrained(fixturePath('text_encoder_t5'));
    const result = noGrad(() => encoder.call(['hello world', 'hello']));
    expectClose(result.tensor.data, text.encoder_t5.output.data, 1e-5);
    expect(result.mask!.tolist()).toEqual(text.encoder_t5.mask);
    expect(fingerprints(encoder.operationBindings())).toEqual(text.encoder_t5.bindings);
    await expectByteIdenticalResave(encoder, 'text_encoder_t5', scratch);
  });

  it('TextEncoder (BERT, OUTPUT_ENCODING readout with latent prefixes)', async () => {
    const record = text.encoder_bert;
    const encoder = await TextEncoder.fromPretrained(fixturePath('text_encoder_bert'));
    expect([...encoder.stateDict().keys()]).toEqual(record.state_keys);
    const prefix = new Latent(fromJson(record.prefix), encoder.contextSpace!, { mask: tensor(record.prefix_mask, { dtype: 'bool' }) });
    expectClose(noGrad(() => encoder.call(['hello world', 'hello'], { context: { latents: [prefix] } })).tensor.data, record.output.data, 1e-5);
    expectClose(noGrad(() => encoder.call(['hello world', 'hello'])).tensor.data, record.plain.data, 1e-5);
    expect(fingerprints(encoder.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(encoder, 'text_encoder_bert', scratch);
  });

  it('TextDecoder (T5): teacher-forced loss, logits, generation and identity bridge', async () => {
    const record = text.decoder_t5;
    const decoder = await TextDecoder.fromPretrained(fixturePath('text_decoder_t5'));
    expect([...decoder.stateDict().keys()]).toEqual(record.state_keys);
    const value = new Latent(fromJson(record.value), decoder.inputSpace, { mask: tensor(record.mask, { dtype: 'bool' }) });
    expect(decoder.loss(value, ['answer', 'hello']).item()).toBeCloseTo(record.loss, 5);
    const [embeds, mask] = decoder.inputs(value, null);
    const logits = noGrad(() => decoder.model.forward({
      inputsEmbeds: embeds, attentionMask: mask, decoderInputIds: tensor([[0, 2, 3], [0, 5, 4]], { dtype: 'int64' }),
    }).logits);
    expectClose(logits.data, record.logits.data, 1e-5);
    expect(decoder.call(value)).toEqual(record.generated);
    expect(fingerprints(decoder.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(decoder, 'text_decoder_t5', scratch);

    const native = await TextDecoder.fromFoundation(fixturePath('t5_foundation'), {
      inputSpace: new Space(`${fixturePath('t5_foundation')}:encoder:input_embeddings`, 8, { version: 'unversioned', organization: 'sequence' }),
      bridge: 'identity', generation: { max_new_tokens: 3 },
    });
    const embedded = native.embedText(['hello world', 'hello']);
    expectClose(embedded.tensor.data, text.decoder_identity.embeddings.data, 1e-6);
    expect(embedded.mask!.tolist()).toEqual(text.decoder_identity.mask);
    expect(native.call(embedded)).toEqual(text.decoder_identity.generated);
    expect(native.loss(embedded, ['answer', 'hello']).item()).toBeCloseTo(text.decoder_identity.loss, 5);
  });

  it('fromFoundation builds the same configuration as Python', async () => {
    const t5 = fixturePath('t5_foundation');
    const encoder = await TextEncoder.fromFoundation(t5);
    expect(encoder.configuration()).toEqual(relocate(text.encoder_t5_foundation_configuration, text.t5_path, t5));
    const bert = fixturePath('bert_foundation');
    const outputEncoding = await TextEncoder.fromFoundation(bert, {
      readout: 'output_encoding', contextSpace: new Space('context', 8, { organization: 'sequence' }),
    });
    expect(outputEncoding.configuration()).toEqual(relocate(text.encoder_bert_foundation_configuration, text.bert_path, bert));
    const decoder = await TextDecoder.fromFoundation(t5, {
      inputSpace: new Space('arbitrary', 3, { organization: 'sequence' }), generation: { max_new_tokens: 3 },
    });
    expect(decoder.configuration()).toEqual(relocate(text.decoder_foundation_configuration, text.t5_path, t5));
    const pooled = await TextEncoder.fromFoundation(bert, { readout: 'pooled', contextSpace: new Space('context', 8, { organization: 'sequence' }) });
    const loaded = await TextEncoder.fromPretrained(fixturePath('text_encoder_bert'));
    pooled.model.loadStateDict(loaded.model.stateDict());
    const prefix = new Latent(fromJson(text.encoder_bert.prefix), pooled.contextSpace!, { mask: tensor(text.encoder_bert.prefix_mask, { dtype: 'bool' }) });
    expectClose(noGrad(() => pooled.call(['hello world', 'hello'], { context: { latents: [prefix] } })).tensor.data, text.encoder_bert_pooled.output.data, 1e-5);
    expectClose(noGrad(() => pooled.call(['hello world', 'hello'])).tensor.data, text.encoder_bert_pooled.plain.data, 1e-5);
  });
});

describe('vision artifacts written by Python', () => {
  it('ImageEncoder (ViT, OUTPUT_ENCODING with context) and sequence readout', async () => {
    const record = vision.output_encoding;
    const encoder = await ImageEncoder.fromPretrained(fixturePath('image_encoder'));
    expect([...encoder.stateDict().keys()]).toEqual(record.state_keys);
    const pixels = fromJson(record.pixels);
    const prefix = new Latent(fromJson(record.prefix), encoder.contextSpace!, { mask: tensor(record.prefix_mask, { dtype: 'bool' }) });
    expectClose(noGrad(() => encoder.call(pixels, { context: { latents: [prefix] } })).tensor.data, record.output.data, 1e-5);
    expectClose(noGrad(() => encoder.call(pixels)).tensor.data, record.plain.data, 1e-5);
    expect(fingerprints(encoder.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(encoder, 'image_encoder', scratch);

    const sequence = new ImageEncoder({
      model: encoder.model.config.toDiffDict(), processor: encoder.configuration().processor!, readout: 'sequence',
      output_space: new Space('vision', 8, { organization: 'sequence' }).configuration() as any,
    });
    sequence.model.loadStateDict(encoder.model.stateDict());
    sequence.eval();
    const result = noGrad(() => sequence.call(pixels));
    expectClose(result.tensor.data, vision.sequence.output.data, 1e-5);
    expectClose(result.coordinates!.data, vision.sequence.coordinates.data, 0);
  });

  it('ImageEncoder.fromFoundation matches the Python configuration', async () => {
    const path = fixturePath('vit_foundation');
    const encoder = await ImageEncoder.fromFoundation(path, {
      readout: 'output_encoding', outputSpace: new Space('visual-readout', 8), contextSpace: new Space('context', 8, { organization: 'sequence' }),
    });
    expect(encoder.configuration()).toEqual(relocate(vision.foundation_configuration, vision.vit_path, path));
  });

  it('processor resize/rescale/normalize matches transformers for uint8 images', () => {
    const processing = vision.processing;
    const image = tensor(processing.uint8, { dtype: 'uint8' });
    for (const key of ['processor_0', 'processor_2', 'processor_3', 'upscale']) {
      const processor = new ImageProcessor(processing[key].config);
      expect(processor.toJson()).toEqual(processing[key].config);
      const pixels = processor.preprocess(image).pixel_values;
      expect(pixels.shape).toEqual(processing[key].pixel_values.shape);
      expectClose(pixels.data, processing[key].pixel_values.data, 1e-5, 1e-6);
    }
  });

  it('antialiased float interpolation matches F.interpolate', () => {
    const floats = fromJson(vision.processing.float);
    for (const mode of ['bilinear', 'bicubic'] as const) {
      for (const [height, width] of [[8, 8], [20, 24], [9, 30]] as const) {
        const expected = vision.processing[`interpolate_${mode}_${height}x${width}`];
        const actual = resizeImage(floats, [height, width], mode);
        expect(actual.shape).toEqual(expected.shape);
        expectClose(actual.data, expected.data, 2e-6, 1e-5);
      }
    }
  });
});
