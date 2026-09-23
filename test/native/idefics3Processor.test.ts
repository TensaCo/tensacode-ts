/**
 * ``Idefics3Processor`` batches with several images per prompt, its errors,
 * the special tokens it adds to tokenizers that lack them, and Idefics3
 * forwards and generation over multi-image batches
 * (``scripts/fixtures/idefics3_processor_fixtures.py``).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Idefics3ImageProcessor, Idefics3Processor } from '../../src/_internal/native/idefics3Processing.js';
import { FastTokenizer } from '../../src/_internal/tokenizers/index.js';
import { rustTokenizerString } from '../../src/_internal/tokenizers/serialization.js';
import { sha256Hex } from '../../src/_internal/json.js';
import { tensorBytes } from '../../src/nn/safetensors.js';
import { noGrad, tensor, type Tensor } from '../../src/nn/index.js';
import { Scene } from '../../src/tools/scene.js';
import { expectClose } from '../helpers/gradcheck.js';

const root = new URL('../fixtures/scene_language/', import.meta.url).pathname;
const records = JSON.parse(readFileSync(join(root, 'processor.json'), 'utf8'));

function image(values: number[][][]): Tensor {
  return tensor(values.flat(2), { shape: [values.length, values[0]!.length, values[0]![0]!.length], dtype: 'uint8' });
}

function smolProcessor(): Idefics3Processor {
  const folder = join(root, 'smol', 'processor');
  const names = ['tokenizer.json', 'tokenizer_config.json', 'processor_config.json', 'chat_template.jinja'];
  const processor = Idefics3Processor.fromAssets(Object.fromEntries(names.map((name) => [name, readFileSync(join(folder, name), 'utf8')])));
  processor.tokenizer.paddingSide = 'left';
  return processor;
}

function expectBatch(actual: ReturnType<Idefics3Processor['call']>, expected: Record<string, any>): void {
  expect(actual.inputIds.tolist()).toEqual(expected.input_ids);
  expect(actual.attentionMask.tolist()).toEqual(expected.attention_mask);
  if (expected.pixel_shape) {
    expect(actual.pixelValues!.shape).toEqual(expected.pixel_shape);
    expect(sha256Hex(tensorBytes(actual.pixelValues!))).toBe(expected.pixel_sha256);
    expect(actual.pixelAttentionMask!.sum().item()).toBe(expected.mask_sum);
    expect(sha256Hex(tensorBytes(actual.pixelAttentionMask!.to('int64')))).toBe(expected.mask_sha256);
  } else {
    expect(actual.pixelValues).toBeNull();
  }
}

describe('Idefics3Processor batches with several images per prompt (SmolVLM)', () => {
  const smol = records.smol;
  it.skipIf(!smol)('nested and flat image lists, padding, text-only batches and errors', () => {
    const processor = smolProcessor();
    const images: Tensor[] = smol.images.map(image);
    expectBatch(processor.call(smol.prompts, [[images[0]!, images[1]!], [images[2]!]], { padding: true }), smol.nested);
    expectBatch(processor.call(smol.prompts, images, { padding: true }), smol.flat);
    expectBatch(processor.call(['<|im_start|>User: hello', 'hi'], null, { padding: true }), smol.text_only);
    const errors: Record<string, () => unknown> = {
      mismatch: () => processor.call(smol.prompts, [[images[0]!], [images[1]!, images[2]!]], { padding: true }),
      no_images: () => processor.call(smol.prompts, null, { padding: true }),
      extra_image: () => processor.call([smol.prompts[1]], [images[0]!, images[1]!]),
      ragged: () => processor.call(smol.prompts, [[images[0]!, images[1]!], [images[2]!]]),
    };
    for (const [name, call] of Object.entries(errors)) {
      expect(call, name).toThrow(smol.errors[name].message);
    }
  }, 120_000);

  it.skipIf(!smol)('Idefics3 forward and generation over the multi-image batch equal transformers', async () => {
    const processor = smolProcessor();
    const images: Tensor[] = smol.images.map(image);
    const batch = processor.call(smol.prompts, [[images[0]!, images[1]!], [images[2]!]], { padding: true });
    const tool = await Scene.fromPretrained(join(root, 'smol'));
    const model = tool.language!.model;
    const inputs = { inputIds: batch.inputIds, attentionMask: batch.attentionMask, pixelValues: batch.pixelValues, pixelAttentionMask: batch.pixelAttentionMask };
    const logits = noGrad(() => model.forward(inputs).logits);
    const [rows, length, vocab] = logits.shape as [number, number, number];
    const head: number[] = [];
    const argmax: number[] = [];
    for (let row = 0; row < rows; row += 1) {
      const offset = (row * length + length - 1) * vocab;
      const last = logits.data.subarray(offset, offset + vocab);
      head.push(...Array.from(last.subarray(0, 256)));
      let best = 0;
      for (let index = 1; index < vocab; index += 1) if (last[index]! > last[best]!) best = index;
      argmax.push(best);
    }
    expectClose(head, smol.model.last_logits_head.data, 1e-4, 1e-4);
    expect(argmax).toEqual(smol.model.argmax);
    const generated = model.generate(inputs, { generationConfig: tool.language!.generationConfig, settings: { max_new_tokens: 4, do_sample: false } });
    expect(generated.sequences).toEqual(smol.model.generated);
  }, 300_000);
});

describe('Idefics3Processor adds missing special tokens', () => {
  const added = records.added;
  it('tokenizer.json, ids and image token id equal transformers', () => {
    const tokenizer = FastTokenizer.fromFiles({ 'tokenizer.json': added.tokenizer_json, 'tokenizer_config.json': added.tokenizer_config });
    const processor = new Idefics3Processor(new Idefics3ImageProcessor({
      do_resize: false, do_image_splitting: false, max_image_size: { longest_edge: 8 }, size: { longest_edge: 8 },
    }), tokenizer, 2);
    expect(rustTokenizerString(processor.tokenizer.jsonText, { pretty: true })).toBe(added.saved['tokenizer.json']);
    expect(processor.imageTokenId).toBe(added.image_token_id);
    const out = processor.call('describe <image> left <end_of_utterance> object<fake_token_around_image>', [image(added.image)]);
    expect(out.inputIds.tolist()).toEqual(added.input_ids);
    // Adding again is a no-op (the tokens are already special added tokens).
    const again = new Idefics3Processor(processor.imageProcessor, processor.tokenizer, 2);
    expect(again.tokenizer).toBe(processor.tokenizer);
  });
});
