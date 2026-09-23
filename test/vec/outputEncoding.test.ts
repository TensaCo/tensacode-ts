/** Port of ``tests/models/test_output_encoding.py`` (ALBERT is not a supported native architecture in TypeScript). */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { cat, noGrad, rand, randn, tensor, type Tensor } from '../../src/nn/index.js';
import { ImageEncoder, Latent, Space, TextEncoder } from '../../src/ops/vec/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixturePath } from './helpers.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-output-encoding-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

function close(a: Tensor, b: Tensor, atol = 1e-5): void {
  expect(a.shape).toEqual(b.shape);
  expectClose(a.data, b.data, atol, 1e-5);
}

async function textEncoder(kind: 'bert' | 't5' | 'roberta'): Promise<TextEncoder> {
  const path = fixturePath(`${kind}_foundation`);
  const encoder = await TextEncoder.fromFoundation(path, { readout: 'output_encoding', contextSpace: new Space('context', 8, { organization: 'sequence' }) });
  expect(encoder.parameters().every((parameter) => parameter.numel > 0)).toBe(true);
  return encoder;
}

describe('OUTPUT_ENCODING readout (tests/models/test_output_encoding.py)', () => {
  it.each(['bert', 't5', 'roberta'] as const)('text native readout, padding, context, gradients and artifact (%s)', async (kind) => {
    const encoder = await textEncoder(kind);
    const native = encoder.model.config.isEncoderDecoder ? encoder.model.getEncoder!() : (encoder.model as any);
    const width = native.getInputEmbeddings().weight.shape[1];
    const prefix = randn([2, 2, width]);
    prefix.requiresGrad = true;
    const context = new Latent(prefix, encoder.contextSpace!, { mask: tensor([[true, false], [true, true]]), sources: ['evidence'] });
    const result = encoder.call(['hello', 'hello world'], { context: { latents: [context] } });
    [[2], [2, 3]].forEach((ids, row) => {
      const valid = prefix.detach().slice(0, row, row + 1).slice(1, 0, row + 1);
      const embedded = native.getInputEmbeddings().forward(tensor([ids], { dtype: 'int64' }));
      const manual = noGrad(() => native.forward({ inputsEmbeds: cat([valid, embedded, encoder.outputEncoding!.detach()], 1) }).lastHiddenState);
      close(result.tensor.detach().slice(0, row, row + 1), manual.select(1, manual.shape[1] - 1));
    });
    expect(result.sources).toEqual(['evidence']);
    expect(result.mask!.tolist()).toEqual([true, true]);
    close(noGrad(() => encoder.call(['hello', 'hello world']).tensor.select(0, 0)), noGrad(() => encoder.call('hello').tensor.select(0, 0)));
    encoder.tokenizer.paddingSide = 'left';
    close(noGrad(() => encoder.call(['hello', 'hello world']).tensor.select(0, 0)), noGrad(() => encoder.call('hello').tensor.select(0, 0)));
    expect(result.tensor.detach().sub(noGrad(() => encoder.call(['hello', 'hello world']).tensor)).abs().max().item()).toBeGreaterThan(1e-6);
    result.tensor.select(1, 0).sum().backward();
    expect(encoder.outputEncoding!.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(native.getInputEmbeddings().weight.grad.abs().sum().item()).toBeGreaterThan(0);
    expect(prefix.grad!.select(0, 0).select(0, 0).abs().sum().item()).toBeGreaterThan(0);
    expect(prefix.grad!.select(0, 0).select(0, 1).abs().sum().item()).toBe(0);
    await encoder.savePretrained(join(scratch, kind));
    const loaded = await TextEncoder.fromPretrained(join(scratch, kind));
    expect(loaded.call(['hello', 'hello world'], { context: { latents: [context] } }).tensor.equal(noGrad(() => encoder.call(['hello', 'hello world'], { context: { latents: [context] } }).tensor))).toBe(true);
    expect(result.metadata.readout_initialization).toBe('untrained');
  });

  it.each(['bert', 't5', 'roberta'] as const)('text position limit and output space (%s)', async (kind) => {
    const encoder = await textEncoder(kind);
    const bad = encoder.configuration();
    (bad.output_space as any).organization = 'sequence';
    expect(() => new TextEncoder(bad)).toThrow(/space/);
    if (kind !== 't5') {
      const maximum = kind === 'roberta' ? 6 : 7;
      encoder.call(Array(maximum).fill('hello').join(' '));
      expect(() => encoder.call(Array(maximum + 1).fill('hello').join(' '))).toThrow(/position/);
      const prefix = new Latent(randn([1, 1, encoder.contextSpace!.dimensions]), encoder.contextSpace!);
      expect(() => encoder.call(Array(maximum).fill('hello').join(' '), { context: { latents: [prefix] } })).toThrow(/position/);
    }
  });

  it('ViT native readout, context, gradients and artifact', async () => {
    const encoder = await ImageEncoder.fromFoundation(fixturePath('vit_foundation'), {
      readout: 'output_encoding', outputSpace: new Space('visual-readout', 8), contextSpace: new Space('context', 8, { organization: 'sequence' }),
    });
    const pixels = rand([2, 3, 8, 8]);
    pixels.requiresGrad = true;
    const prefix = randn([2, 2, 8]);
    prefix.requiresGrad = true;
    const context = new Latent(prefix, encoder.contextSpace!, { mask: tensor([[true, false], [true, true]]) });
    const result = encoder.call(pixels, { context: { latents: [context] } });
    for (let row = 0; row < 2; row += 1) {
      const embedded = encoder.model.embed(pixels.detach().slice(0, row, row + 1).sub(0.5).div(0.5));
      const hidden = cat([prefix.detach().slice(0, row, row + 1).slice(1, 0, row + 1), embedded, encoder.outputEncoding!.detach()], 1);
      const manual = noGrad(() => encoder.model.forward({ inputsEmbeds: hidden }).lastHiddenState);
      close(result.tensor.detach().select(0, row), manual.select(0, 0).select(0, manual.shape[1]! - 1));
    }
    expect(result.coordinates).toBeNull();
    expect(result.tensor.detach().sub(noGrad(() => encoder.call(pixels.detach()).tensor)).abs().max().item()).toBeGreaterThan(1e-6);
    result.tensor.select(1, 0).sum().backward();
    expect(pixels.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(encoder.outputEncoding!.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(prefix.grad!.select(0, 0).select(0, 0).abs().sum().item()).toBeGreaterThan(0);
    expect(prefix.grad!.select(0, 0).select(0, 1).abs().sum().item()).toBe(0);
    await encoder.savePretrained(join(scratch, 'vit'));
    const loaded = await ImageEncoder.fromPretrained(join(scratch, 'vit'));
    expect(noGrad(() => loaded.call(pixels.detach(), { context: { latents: [context] } }).tensor).equal(result.tensor.detach())).toBe(true);
    const bad = encoder.configuration();
    (bad.output_space as any).organization = 'sequence';
    expect(() => new ImageEncoder(bad)).toThrow(/space/);
    expect(result.metadata.readout_initialization).toBe('untrained');
  });
});
