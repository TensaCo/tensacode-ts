/** ALBERT matches transformers (``scripts/fixtures/albert_fixtures.py``). */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { NativeConfig } from '../../src/_internal/native/config.js';
import { AlbertForSequenceClassification, AlbertModel } from '../../src/_internal/native/bert.js';
import { loadModelFromBytes, noGrad, tensor } from '../../src/nn/index.js';
import { Latent, Space, TextEncoder } from '../../src/ops/vec/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, fromJson, ints } from '../helpers/fixtures.js';

const record = fixtureJson('native_albert.json');
const text = fixtureJson('vec/albert.json');
// Repo-relative, as the Python fixture generator loaded it (vitest runs from the package root).
const foundation = 'test/fixtures/vec/albert_foundation';
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-albert-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

describe('ALBERT', () => {
  it('base model with shared layer groups', () => {
    const model = new AlbertModel(NativeConfig.fromDict(record.config)).eval();
    expect([...model.stateDict().keys()]).toEqual(record.state_keys);
    loadModelFromBytes(model, fixtureBytes('native_albert.safetensors'));
    noGrad(() => {
      const output = model.forward({ inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask), tokenTypeIds: ints(record.token_type_ids) });
      expectClose(output.lastHiddenState.data, record.last_hidden_state.data, 2e-5, 1e-4);
      expectClose(output.poolerOutput!.data, record.pooler_output.data, 2e-5, 1e-4);
      const embedded = model.forward({ inputsEmbeds: fromJson(record.inputs_embeds), attentionMask: ints(record.inputs_embeds_mask) });
      expectClose(embedded.lastHiddenState.data, record.inputs_embeds_output.data, 2e-5, 1e-4);
    });
  });

  it('sequence classifier', () => {
    const model = new AlbertForSequenceClassification(NativeConfig.fromDict(record.classifier.config)).eval();
    expect([...model.stateDict().keys()]).toEqual(record.classifier.state_keys);
    loadModelFromBytes(model, fixtureBytes('native_albert_classifier.safetensors'));
    noGrad(() => {
      const output = model.forward({ inputIds: ints(record.input_ids), attentionMask: ints(record.attention_mask), tokenTypeIds: ints(record.token_type_ids) });
      expectClose(output.logits.data, record.classifier.logits.data, 2e-5, 1e-4);
    });
  });

  it('text context uses the input embedding width (tests/vec/test_pretrained_text.py)', async () => {
    const space = new Space('albert-input', 4, { organization: 'sequence' });
    const encoder = await TextEncoder.fromFoundation(foundation, { contextSpace: space });
    expect(encoder.outputSpace.dimensions).toBe(8);
    const { foundation: source, ...configuration } = encoder.configuration();
    const { foundation: expectedSource, ...expected } = text.configuration;
    expect(configuration).toEqual(expected);
    expect((source as any).revision).toBe(expectedSource.revision);
    const raw = fromJson(text.prefix);
    raw.requiresGrad = true;
    const output = encoder.call('hello world', { context: { latents: [new Latent(raw, space, { mask: tensor([[true, false]]) })] } });
    expectClose(output.tensor.data, text.output.data, 2e-5, 1e-4);
    output.tensor.select(-1, 0).sum().backward();
    expect(raw.grad!.select(1, 0).abs().sum().item()).toBeGreaterThan(0);
    expect(raw.grad!.select(1, 1).abs().sum().item()).toBe(0);
    const plain = noGrad(() => encoder.call(['hello world', 'hello']));
    expectClose(plain.tensor.data, text.plain.data, 2e-5, 1e-4);
    expect(plain.mask!.tolist()).toEqual(text.plain_mask);
    await expect(TextEncoder.fromFoundation(foundation, { contextSpace: new Space('wrong-hidden-width', 8, { organization: 'sequence' }) }))
      .rejects.toThrow(/context_space/);
    await encoder.savePretrained(join(scratch, 'owned-albert'));
    const restored = await TextEncoder.fromPretrained(join(scratch, 'owned-albert'));
    const again = noGrad(() => restored.call('hello world', { context: { latents: [new Latent(raw.detach(), space, { mask: tensor([[true, false]]) })] } }));
    expect(again.tensor.equal(output.tensor.detach())).toBe(true);
  });
});
