/** Port of ``tests/vec/test_pretrained_vision.py`` (image files are replaced by decoded uint8 tensors). */
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { cat, full, noGrad, ones, rand, randn, tensor, zeros, type Tensor } from '../../src/nn/index.js';
import { ImageEncoder, Latent, Space } from '../../src/ops/vec/index.js';
import { serializeSafetensors } from '../../src/nn/safetensors.js';
import { trace } from '../../src/_internal/tracing.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson } from '../helpers/fixtures.js';
import { fixturePath } from './helpers.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-vision-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const VIT = fixturePath('vit_foundation');
const vision = fixtureJson('vec/vision.json');
const MODEL = vision.foundation_configuration.model;
const PROCESSOR = vision.foundation_configuration.processor;

function config(readout: 'sequence' | 'pooled' | 'output_encoding' = 'sequence'): any {
  return {
    model: MODEL, processor: PROCESSOR, readout,
    output_space: new Space('vision', 8, { organization: readout === 'sequence' ? 'sequence' : 'feature' }).configuration(),
    context_space: new Space('vision-context', 8, { organization: 'sequence' }).configuration(),
  };
}

function close(a: Tensor, b: Tensor, atol = 1e-5): void {
  expect(a.shape).toEqual(b.shape);
  expectClose(a.data, b.data, atol, 1e-5);
}

const normalized = (pixels: Tensor) => pixels.sub(0.5).div(0.5);

describe('pretrained vision (tests/vec/test_pretrained_vision.py)', () => {
  it('native transformer outputs and gradients', () => {
    const model = new ImageEncoder(config()).eval();
    const pixels = rand([2, 3, 8, 8]);
    pixels.requiresGrad = true;
    const result = model.call(pixels);
    const native = noGrad(() => model.model.forward({ pixelValues: normalized(pixels.detach()) }).lastHiddenState.slice(1, 1));
    close(result.tensor.detach(), native);
    expect(result.mask!.shape).toEqual([2, 4]);
    expect(result.coordinates!.select(0, 0).tolist()).toEqual([[2, 2], [2, 6], [6, 2], [6, 6]]);
    result.tensor.select(2, 0).sum().backward();
    expect(pixels.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(model.model.embeddings.patch_embeddings.projection.weight.grad!.abs().sum().item()).toBeGreaterThan(0);
  });

  it('context attention mask, provenance and full reload', async () => {
    const model = new ImageEncoder(config('pooled')).eval();
    const pixels = rand([1, 3, 8, 8]);
    const tokens = randn([1, 2, 8]);
    tokens.requiresGrad = true;
    const space = new Space('vision-context', 8, { organization: 'sequence' });
    const context = new Latent(tokens, space, { mask: tensor([[true, false]]), sources: ['context-source'] });
    close(noGrad(() => model.call(pixels).tensor), noGrad(() => model.model.forward({ pixelValues: normalized(pixels) }).lastHiddenState.select(1, 0)));
    const result = model.call(pixels, { context: { latents: [context] } });
    const shifted = tokens.detach().add(cat([zeros([1, 1, 8]), full([1, 1, 8], 100)], 1));
    close(result.tensor.detach(), noGrad(() => model.call(pixels, { context: { latents: [new Latent(shifted, space, { mask: context.mask })] } }).tensor));
    expect(result.tensor.detach().sub(noGrad(() => model.call(pixels).tensor)).abs().max().item()).toBeGreaterThan(1e-6);
    expect(result.sources).toContain('context-source');
    result.tensor.select(1, 0).sum().backward();
    expect(tokens.grad!.select(1, 0).abs().sum().item()).toBeGreaterThan(0);
    expect(tokens.grad!.select(1, 1).abs().sum().item()).toBe(0);
    await model.savePretrained(join(scratch, 'vision'));
    const loaded = await ImageEncoder.fromPretrained(join(scratch, 'vision'));
    close(result.tensor.detach(), noGrad(() => loaded.call(pixels, { context: { latents: [context] } }).tensor));
    expect(loaded.training).toBe(false);
    loaded.train();
    expect(loaded.model.training).toBe(true);
  });

  it('input contract and local foundation', async () => {
    const model = new ImageEncoder(config());
    expect(() => model.call(full([3, 8, 8], 2))).toThrow();
    expect(() => model.call(rand([3, 9, 8]))).toThrow();
    expect(() => model.call(rand([3, 8, 8]), { context: { surprise: 1 } })).toThrow();
    const loaded = await ImageEncoder.fromFoundation(VIT, { outputSpace: config().output_space });
    const reference = await ImageEncoder.fromPretrained(fixturePath('image_encoder'));
    const pixels = rand([1, 3, 8, 8]);
    close(noGrad(() => loaded.call(pixels).tensor), noGrad(() => reference.model.forward({ pixelValues: normalized(pixels) }).lastHiddenState.slice(1, 1)));
    expect((loaded.configuration().foundation as any).repo).toBe(VIT);
    await expect(ImageEncoder.fromFoundation(VIT, { outputSpace: config().output_space, device: 'cuda' })).rejects.toThrow(/device/);
  });

  it('processor assets and trace replay', async () => {
    const model = new ImageEncoder(config()).eval();
    const image = full([3, 16, 16], 0, { dtype: 'uint8' });
    noGrad(() => image.select(0, 0).fill_(255));
    const pixels = model.preprocess(image);
    expect(pixels.pixel_values.shape).toEqual([1, 3, 8, 8]);
    const session = trace();
    const result = session.run(() => model.call(pixels));
    close((session.replay(session.ref(result)) as Latent).tensor, result.tensor);
    await model.savePretrained(join(scratch, 'owned'));
    const restored = await ImageEncoder.fromPretrained(join(scratch, 'owned'));
    close(restored.call(pixels).tensor, result.tensor);
    writeFileSync(join(scratch, 'owned', 'vision_processor.json'), '{}');
    await expect(ImageEncoder.fromPretrained(join(scratch, 'owned'))).rejects.toThrow(/processor/);
  });

  it('rejects non-ViT foundations before loading weights', async () => {
    const path = join(scratch, 'bert-config');
    cpSync(fixturePath('bert_foundation'), path, { recursive: true });
    await expect(ImageEncoder.fromFoundation(path, { outputSpace: config().output_space })).rejects.toThrow(/ViT/);
  });

  it('rejects incomplete foundation weights', async () => {
    const path = join(scratch, 'vit-incomplete');
    cpSync(VIT, path, { recursive: true });
    writeFileSync(join(path, 'model.safetensors'), serializeSafetensors({ unrelated: ones([1]) }));
    await expect(ImageEncoder.fromFoundation(path, { outputSpace: config().output_space })).rejects.toThrow(/missing/);
  });

  it('processed source provenance and unknown input keys', () => {
    const model = new ImageEncoder(config()).eval();
    expect(model.call({ pixel_values: zeros([1, 3, 8, 8]), sources: ['photo:1'] }).sources).toEqual(['photo:1']);
    expect(() => model.call({ pixel_values: zeros([1, 3, 8, 8]), ignored: true } as any)).toThrow(/pixel_values/);
  });

  it('live processor configuration survives artifact reload', async () => {
    const model = new ImageEncoder(config()).eval();
    const original = model.configuration();
    model.processor.imageMean = [0, 0.1, 0.2];
    model.processor.imageStd = [0.8, 0.9, 1];
    model.processor.resample = 0;
    const image = rand([1, 3, 8, 8]);
    const expected = model.call(image);
    expect(model.configuration()).not.toEqual(original);
    await model.savePretrained(join(scratch, 'updated'));
    const restored = await ImageEncoder.fromPretrained(join(scratch, 'updated'));
    close(restored.call(image).tensor, expected.tensor);
    const raw = tensor(Array.from({ length: 3 * 13 * 17 }, (_, index) => (index * 37) % 256), { shape: [3, 13, 17], dtype: 'uint8' });
    close(restored.preprocess(raw).pixel_values, model.preprocess(raw).pixel_values, 0);
    expect(JSON.parse(readFileSync(join(scratch, 'updated', 'vision_processor.json'), 'utf8')).resample).toBe(0);
  });

  it('uses the output_space and readout contract', () => {
    const settings = config();
    const model = new ImageEncoder(settings);
    expect(model.outputSpace.equals(Space.fromConfig(settings.output_space))).toBe(true);
    expect(model.readout).toBe('sequence');
    const bad = { ...config(), output_space: { ...config().output_space, organization: 'feature' } };
    expect(() => new ImageEncoder(bad)).toThrow(/space/);
  });

  it('context prefix order, native positions and validation', () => {
    const model = new ImageEncoder(config()).eval();
    const pixels = rand([1, 3, 8, 8]);
    const space = model.contextSpace!;
    const first = new Latent(randn([1, 1, 8]), space, { sources: ['first'] });
    const second = new Latent(randn([1, 2, 8]), space, { mask: tensor([[true, false]]), sources: ['second'] });
    const result = noGrad(() => model.call(pixels, { context: { latents: [first, second] } }));
    const embedded = noGrad(() => model.model.embed(normalized(pixels)));
    const hidden = cat([first.tensor, second.tensor.maskedFill(second.mask!.logicalNot().unsqueeze(-1), 0), embedded], 1);
    const bias = tensor([0, 0, -3.4028234663852886e38, 0, 0, 0, 0, 0], { shape: [1, 1, 1, 8] });
    const expected = noGrad(() => model.model.forward({ inputsEmbeds: hidden, attentionBias: bias }).lastHiddenState.slice(1, 3).slice(1, 1));
    close(result.tensor, expected);
    expect(result.tensor.shape).toEqual([1, 4, 8]);
    expect(result.sources).toEqual(['first', 'second']);
    expect(() => model.call(pixels, { context: { latents: [new Latent(randn([2, 1, 8]), space)] } })).toThrow(/batch/);
    expect(() => model.call(pixels, { context: { latents: [new Latent(first.tensor, new Space('wrong', 8, { organization: 'sequence' }))] } })).toThrow(/space/);
    const withoutContext = new ImageEncoder({ ...config(), context_space: null });
    expect(() => withoutContext.call(pixels, { context: { latents: [first] } })).toThrow(/context_space/);
    expect(() => new ImageEncoder({ ...config(), context_space: new Space('bad-width', 3).configuration() })).toThrow(/context_space/);
  });

  it.each([['space', {}], ['output', 'sequence']])('rejects legacy constructor key %s', (key, value) => {
    expect(() => new ImageEncoder({ ...config(), [key]: value })).toThrow(/output_space|readout/);
  });

  it('rejects unsupported architectures and processors', () => {
    expect(() => new ImageEncoder({ ...config(), model: { ...MODEL, model_type: 'bert' } })).toThrow(/ViTModel/);
    expect(() => new ImageEncoder({ ...config(), processor: { ...PROCESSOR, image_processor_type: 'CLIPImageProcessor' } })).toThrow(/ViTImageProcessor/);
    expect(() => new ImageEncoder({ ...config(), processor: { ...PROCESSOR, image_mean: [0.5] } })).toThrow(/normalization/);
    expect(() => new ImageEncoder({ ...config(), readout: 'mean' })).toThrow(/readout/);
  });
});
