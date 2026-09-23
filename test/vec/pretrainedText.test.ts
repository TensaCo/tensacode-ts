/**
 * Port of ``tests/vec/test_pretrained_text.py`` over tiny native foundations
 * written by Python (``test/fixtures/vec/*_foundation``). Durable
 * experience/Trainer restarts belong to the training module's tests.
 */
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import {
  Parameter, cat, manualSeed, noGrad, randn, serializeSafetensors, deserializeSafetensors, tensor, type Tensor,
} from '../../src/nn/index.js';
import { Latent, Space, TextDecoder, TextEncoder } from '../../src/ops/vec/index.js';
import { bindingRecords } from '../../src/_internal/fingerprint.js';
import { generateSeq2Seq } from '../../src/_internal/native/generation.js';
import { FastTokenizer } from '../../src/_internal/tokenizers/index.js';
import { trace } from '../../src/_internal/tracing.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson, fromJson } from '../helpers/fixtures.js';
import { fixturePath } from './helpers.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-text-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const T5 = fixturePath('t5_foundation');
const BERT = fixturePath('bert_foundation');
const foundations = fixtureJson('vec/foundations.json');

function copyFoundation(source: string, name: string): string {
  const target = join(scratch, name);
  cpSync(source, target, { recursive: true });
  return target;
}

function close(a: Tensor, b: Tensor, atol = 1e-5): void {
  expect(a.shape).toEqual(b.shape);
  expectClose(a.data, b.data, atol, 1e-5);
}

describe('pretrained text operations (tests/vec/test_pretrained_text.py)', () => {
  it('encoder native context, padding and offline reload', async () => {
    const encoder = await TextEncoder.fromFoundation(T5);
    const value = encoder.call(['hello world', 'hello']);
    const batch = encoder.tokenizer.encodeTensors(['hello world', 'hello'], { padding: true });
    const expected = encoder.model.getEncoder!().forward({ inputIds: batch.input_ids, attentionMask: batch.attention_mask }).lastHiddenState;
    close(value.tensor, expected);
    expect(value.mask!.tolist()).toEqual([[true, true], [true, false]]);
    await encoder.savePretrained(join(scratch, 'owned-encoder'));
    const restored = await TextEncoder.fromPretrained(join(scratch, 'owned-encoder'), { localFilesOnly: true });
    close(restored.call(['hello world', 'hello']).tensor, value.tensor);
  });

  it('decoder runs the native encoder, identity bridge and loss', async () => {
    const decoder = await TextDecoder.fromFoundation(T5, { inputSpace: new Space('arbitrary', 3, { organization: 'sequence' }), generation: { max_new_tokens: 3 } });
    const saved = bindingRecords(decoder.operationBindings());
    expect(saved.objective!.configuration.type).toBe('tensorcode.ops.vec.decode.TextDecoder.objective');
    expect(JSON.stringify(saved)).not.toContain('tensorcode._internal.vec');
    const x = randn([2, 2, 3]);
    x.requiresGrad = true;
    const value = new Latent(x, decoder.inputSpace, { mask: tensor([[true, true], [true, false]]) });
    const output = decoder.call(value);
    expect(output.length).toBe(2);
    decoder.loss(value, ['answer', 'hello']).backward();
    expect(x.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect((decoder.projection as any).weight.grad.abs().sum().item()).toBeGreaterThan(0);
    const q = decoder.model.getSubmodule('encoder.block.0.layer.0.SelfAttention.q') as any;
    expect(q.weight.grad.abs().sum().item()).toBeGreaterThan(0);
    await decoder.savePretrained(join(scratch, 'owned-decoder'));
    const restored = await TextDecoder.fromPretrained(join(scratch, 'owned-decoder'));
    expect(restored.call(value)).toEqual(output);
    expect(() => restored.call(new Latent(x, new Space('wrong', 3, { organization: 'sequence' })))).toThrow(/space/);

    const native = await TextDecoder.fromFoundation(T5, { inputSpace: decoder.nativeInputSpace, bridge: 'identity', generation: { max_new_tokens: 3 } });
    const tokens = native.tokenizer.encodeTensors(['hello world', 'hello'], { padding: true });
    const embeds = native.model.getInputEmbeddings().forward(tokens.input_ids);
    const baseline = generateSeq2Seq(native.model, { inputIds: tokens.input_ids, attentionMask: tokens.attention_mask }, { max_new_tokens: 3 }, { generationConfig: native.generationConfig });
    expect(native.call(new Latent(embeds, native.inputSpace, { mask: tokens.attention_mask.bool() })))
      .toEqual(native.tokenizer.batchDecode(baseline, { skipSpecialTokens: true }));
  });

  it('bert mean pooling masks padding and rejects target context', async () => {
    const encoder = await TextEncoder.fromFoundation(BERT, { readout: 'pooled' });
    const batch = encoder.call(['hello', 'hello world']);
    close(batch.tensor.select(0, 0), encoder.call('hello').tensor.select(0, 0));
    expect(() => encoder.call('hello', { context: { targets: 'answer' } })).toThrow(/context/);
  });

  it('decoder context masks and training modes', async () => {
    const decoder = await TextDecoder.fromFoundation(T5, { inputSpace: new Space('features', 3, { organization: 'sequence' }), generation: { max_new_tokens: 2 } });
    const prefix = new Latent(randn([1, 2, 3]), decoder.inputSpace, { mask: tensor([[true, false]]) });
    const value = new Latent(randn([1, 1, 3]), decoder.inputSpace);
    const concat = new Latent(cat([prefix.tensor, value.tensor], 1), decoder.inputSpace, { mask: tensor([[true, false, true]]) });
    const expected = decoder.loss(concat, 'answer').item();
    expect(decoder.loss(value, 'answer', { context: { latents: [prefix] } }).item()).toBeCloseTo(expected, 6);
    const poisoned = prefix.tensor.clone();
    noGrad(() => poisoned.select(1, 1).fill_(1e5));
    expect(decoder.loss(value, 'answer', { context: { latents: [new Latent(poisoned, prefix.space, { mask: prefix.mask })] } }).item()).toBeCloseTo(expected, 6);
    decoder.train();
    decoder.model.encoder.eval();
    const modes = decoder.model.modules().map((module) => module.training);
    decoder.call(value);
    expect(decoder.model.modules().map((module) => module.training)).toEqual(modes);
    expect(() => decoder.call(value, { context: { targets: 'answer' } })).toThrow(/context/);
    await expect(TextDecoder.fromFoundation(T5, { inputSpace: new Space('fake', 8, { organization: 'sequence' }), bridge: 'identity' })).rejects.toThrow(/identity/);
  });

  it('decoder native embeddings and objective envelopes', async () => {
    const decoder = await TextDecoder.fromFoundation(T5, { inputSpace: new Space('input', 3, { organization: 'sequence' }) });
    const native = await TextDecoder.fromFoundation(T5, { inputSpace: decoder.nativeInputSpace, bridge: 'identity' });
    const latent = native.embedText(['hello world', 'hello']);
    expect(latent.mask!.tolist()).toEqual([[true, true], [true, false]]);
    expect(latent.metadata).toEqual({ representation: 'native_input_embeddings' });
    const value = new Latent(randn([1, 2, 3]), decoder.inputSpace);
    const prefixTensor = randn([1, 1, 3]);
    prefixTensor.requiresGrad = true;
    const envelope = { value, context: { latents: [new Latent(prefixTensor, decoder.inputSpace)] } };
    decoder.trainingOperation.call({ inputs: envelope, targets: 'answer' }).backward();
    expect(prefixTensor.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(() => decoder.trainingOperation.call({ inputs: { ...envelope, targets: 'leaked' }, targets: 'answer' })).toThrow(/envelope/);
    expect(() => decoder.trainingOperation.call({ inputs: { value, context: { targets: 'leaked' } }, targets: 'answer' })).toThrow(/context/);
    expect(decoder.trainingInputsIncludeTargets).toBe(true);
    expect(decoder.operationBindings().objective).toBe(decoder.trainingOperation);
  });

  it('foundation requires complete weights', async () => {
    const path = copyFoundation(T5, 't5-incomplete');
    const file = join(path, 'model.safetensors');
    const bytes = readFileSync(file);
    const contents = deserializeSafetensors(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    contents.tensors.delete('encoder.block.0.layer.0.SelfAttention.q.weight');
    writeFileSync(file, serializeSafetensors(contents.tensors, { format: 'pt' }));
    await expect(TextEncoder.fromFoundation(path)).rejects.toThrow(/missing/);
    await expect(TextEncoder.fromFoundation(T5, { trustRemoteCode: true })).rejects.toThrow(/native code/);
  });

  it('foundation generation defaults are saved but sampling stays explicit', async () => {
    const path = copyFoundation(T5, 't5-sampling');
    const file = join(path, 'generation_config.json');
    const config = JSON.parse(readFileSync(file, 'utf8'));
    writeFileSync(file, JSON.stringify({ ...config, do_sample: true, eos_token_id: 3 }));
    const model = await TextDecoder.fromFoundation(path, { inputSpace: new Space('input', 3) });
    expect(model.generation.do_sample).toBe(false);
    expect(model.replayable).toBe(true);
    await model.savePretrained(join(scratch, 'sampling-saved'));
    const restored = await TextDecoder.fromPretrained(join(scratch, 'sampling-saved'));
    expect(restored.generationConfig.eos_token_id).toBe(3);
    const sampling = new TextDecoder({ ...restored.configuration(), generation: { do_sample: true } });
    expect(sampling.replayable).toBe(false);
    expect(() => new TextDecoder({ ...restored.configuration(), generation: { beams: 2 } })).toThrow(/unsupported generation setting/);
  });

  it('live tokenizer and generation settings survive the artifact', async () => {
    const encoder = await TextEncoder.fromFoundation(T5);
    encoder.tokenizer.paddingSide = 'left';
    encoder.tokenizer.options.clean_up_tokenization_spaces = true;
    const before = encoder.call(['hello world', 'hello']);
    const config = encoder.configuration();
    encoder.call('hello');
    expect(encoder.configuration()).toEqual(config);
    await encoder.savePretrained(join(scratch, 'live-encoder'));
    const restored = await TextEncoder.fromPretrained(join(scratch, 'live-encoder'));
    expect(restored.tokenizer.paddingSide).toBe('left');
    expect(restored.tokenizer.options.clean_up_tokenization_spaces).toBe(true);
    close(before.tensor, restored.call(['hello world', 'hello']).tensor);
    const decoder = await TextDecoder.fromFoundation(T5, { inputSpace: new Space('input', 3) });
    decoder.generation.max_new_tokens = 7;
    decoder.generationConfig.eos_token_id = 3;
    decoder.tokenizer.paddingSide = 'left';
    await decoder.savePretrained(join(scratch, 'live-decoder'));
    const decoded = await TextDecoder.fromPretrained(join(scratch, 'live-decoder'));
    expect(decoded.generation.max_new_tokens).toBe(7);
    expect(decoded.generationConfig.eos_token_id).toBe(3);
    expect(decoded.tokenizer.paddingSide).toBe('left');
  });

  it('tokenizer cleanup follows the native contract', () => {
    const json = JSON.stringify({
      version: '1.0', truncation: null, padding: null, added_tokens: [], normalizer: null, pre_tokenizer: null, post_processor: null, decoder: null,
      model: { type: 'WordLevel', vocab: { '[UNK]': 0, hello: 1, ',': 2 }, unk_token: '[UNK]' },
    });
    const original = new FastTokenizer(json, { specialTokens: { unk_token: '[UNK]' }, options: { clean_up_tokenization_spaces: true } });
    const restored = FastTokenizer.fromConfiguration(original.configuration());
    expect(original.decode([1, 2])).toBe('hello,');
    expect(restored.decode([1, 2])).toBe('hello,');
  });

  it.each([['unscaled', false], ['scaled', true]] as const)('untied foundation keeps a distinct lm_head (%s)', async (name, scale) => {
    const path = fixturePath(`t5_untied_${name}`);
    const decoder = await TextDecoder.fromFoundation(path, { inputSpace: new Space('input', 3) });
    expect(decoder.model.config.get('tie_word_embeddings')).toBe(false);
    await decoder.savePretrained(join(scratch, `untied-${name}`));
    const restored = await TextDecoder.fromPretrained(join(scratch, `untied-${name}`));
    expect(restored.model.config.get('tie_word_embeddings')).toBe(false);
    expect(restored.model.config.get('scale_decoder_outputs')).toBe(scale);
    const logits = noGrad(() => restored.model.forward({ inputIds: tensor([[2, 3]], { dtype: 'int64' }), decoderInputIds: tensor([[0, 2]], { dtype: 'int64' }) }).logits);
    close(logits, fromJson(foundations[`t5_untied_${name}`]));
    expect(restored.model.shared.weight).not.toBe(restored.model.lm_head.weight);
    expect(restored.model.shared.weight.toArray().every((value) => value === 0.25)).toBe(true);
    expect(restored.model.lm_head.weight.toArray().every((value) => value === 0.75)).toBe(true);
  });

  it('text latent prefix readout, mask and gradient', async () => {
    const space = new Space('explicit-native-embeddings', 8, { organization: 'sequence' });
    const encoder = await TextEncoder.fromFoundation(T5, { readout: 'sequence', contextSpace: space });
    const raw = randn([2, 2, 8]);
    raw.requiresGrad = true;
    const prefixMask = tensor([[true, false], [true, true]]);
    const prefix = new Latent(raw, space, { mask: prefixMask, sources: ['prefix:1'] });
    const texts = ['hello world', 'hello'];
    const result = encoder.call(texts, { context: { latents: [prefix] } });
    const tokens = encoder.tokenizer.encodeTensors(texts, { padding: true });
    const embeds = noGrad(() => encoder.model.getInputEmbeddings().forward(tokens.input_ids));
    const native = encoder.model.getEncoder!();
    for (let row = 0; row < 2; row += 1) {
      const validText = tokens.attention_mask.select(0, row).bool();
      const keep = prefixMask.select(0, row);
      const inputs = cat([raw.detach().select(0, row).maskedSelect(keep), embeds.select(0, row).maskedSelect(validText)], 0).unsqueeze(0);
      const hidden = noGrad(() => native.forward({ inputsEmbeds: inputs }).lastHiddenState.select(0, 0));
      const count = keep.to('int64').sum().item();
      const textCount = validText.to('int64').sum().item();
      close(result.tensor.select(0, row).slice(0, 0, textCount), hidden.slice(0, count));
    }
    expect(result.mask!.tolist()).toEqual([[true, true], [true, false]]);
    expect(result.sources).toEqual(['prefix:1']);
    result.tensor.select(2, 0).sum().backward();
    expect(raw.grad!.select(0, 0).select(0, 0).abs().sum().item()).toBeGreaterThan(0);
    expect(raw.grad!.select(0, 0).select(0, 1).abs().sum().item()).toBe(0);
    const pooled = new TextEncoder({ ...encoder.configuration(), readout: 'pooled', output_space: new Space('pooled', 8).configuration() as any }).eval();
    pooled.model.loadStateDict(encoder.model.stateDict());
    const pooledResult = noGrad(() => pooled.call(texts, { context: { latents: [prefix] } }));
    const weights = result.mask!.to('float32').unsqueeze(-1);
    close(pooledResult.tensor, noGrad(() => result.tensor.mul(weights).sum(1).div(weights.sum(1))));
    expect(() => encoder.call('hello', { context: { latents: [prefix] } })).toThrow(/batch/);
    expect(() => encoder.call(texts, { context: { latents: [new Latent(raw, new Space('wrong', 8, { organization: 'sequence' }))] } })).toThrow(/space/);
    const undeclared = await TextEncoder.fromFoundation(T5);
    expect(() => undeclared.call(texts, { context: { latents: [prefix] } })).toThrow(/context_space/);
    expect(() => encoder.call('hello', { context: { texts: ['prefix'] } })).toThrow(/context/);
    expect(() => new TextEncoder({ ...encoder.configuration(), context_space: new Space('wrong-width', 3).configuration() as any })).toThrow(/context_space/);
  });

  it('encoder and embedText use the actual (untied) encoder embeddings', async () => {
    const space = new Space('encoder-input', 8, { organization: 'sequence' });
    const encoder = await TextEncoder.fromFoundation(T5, { contextSpace: space });
    encoder.model.setParameterAt('encoder.embed_tokens.weight', new Parameter(randn([6, 8])));
    const prefix = new Latent(randn([1, 1, 8]), space);
    const tokens = encoder.tokenizer.encodeTensors(['hello world'], { padding: true });
    const native = encoder.model.getEncoder!();
    const embeds = native.getInputEmbeddings().forward(tokens.input_ids);
    const expected = noGrad(() => native.forward({ inputsEmbeds: cat([prefix.tensor, embeds], 1) }).lastHiddenState.slice(1, 1));
    close(noGrad(() => encoder.call('hello world', { context: { latents: [prefix] } }).tensor), expected);

    const decoder = await TextDecoder.fromFoundation(T5, { inputSpace: new Space('input', 3) });
    decoder.model.setParameterAt('encoder.embed_tokens.weight', new Parameter(randn([6, 8])));
    close(decoder.embedText('hello world').tensor, decoder.model.getEncoder().getInputEmbeddings().forward(tokens.input_ids));
  });

  it('decoder padding holes preserve loss and generation', async () => {
    manualSeed(4);
    const decoder = await TextDecoder.fromFoundation(T5, { inputSpace: new Space('padding', 3, { organization: 'sequence' }), generation: { max_new_tokens: 3 } });
    const value = new Latent(randn([1, 2, 3]), decoder.inputSpace);
    const prefix = new Latent(randn([1, 1, 3]), decoder.inputSpace);
    const padded = new Latent(cat([prefix.tensor, randn([1, 20, 3])], 1), decoder.inputSpace, { mask: tensor([[true, ...Array(20).fill(false)]]) });
    expect(decoder.loss(value, 'answer', { context: { latents: [prefix] } }).item())
      .toBeCloseTo(decoder.loss(value, 'answer', { context: { latents: [padded] } }).item(), 6);
    const [a, maskA] = decoder.inputs(value, { latents: [prefix] });
    const [b, maskB] = decoder.inputs(value, { latents: [padded] });
    close(a, b);
    expect(maskA.equal(maskB)).toBe(true);
    expect(decoder.call(value, { context: { latents: [prefix] } })).toEqual(decoder.call(value, { context: { latents: [padded] } }));
  });

  it('decoder packs varied rows and preserves only valid gradients', async () => {
    const decoder = await TextDecoder.fromFoundation(T5, { inputSpace: new Space('padding', 3, { organization: 'sequence' }) });
    const prefixTensor = randn([2, 4, 3]);
    const mainTensor = randn([2, 4, 3]);
    prefixTensor.requiresGrad = true;
    mainTensor.requiresGrad = true;
    const prefixMask = tensor([[false, true, false, true], [true, false, false, false]]);
    const mainMask = tensor([[true, false, true, false], [false, false, true, false]]);
    const [embeddings, mask] = decoder.inputs(new Latent(mainTensor, decoder.inputSpace, { mask: mainMask }), { latents: [new Latent(prefixTensor, decoder.inputSpace, { mask: prefixMask })] });
    expect(mask.tolist()).toEqual([[true, true, true, true], [true, true, false, false]]);
    const labels = tensor([[5], [5]], { dtype: 'int64' });
    const batch = decoder.model.forward({ inputsEmbeds: embeddings, attentionMask: mask, labels });
    for (let row = 0; row < 2; row += 1) {
      const compact = cat([prefixTensor.detach().select(0, row).maskedSelect(prefixMask.select(0, row)), mainTensor.detach().select(0, row).maskedSelect(mainMask.select(0, row))], 0).unsqueeze(0);
      const [single, singleMask] = decoder.inputs(new Latent(compact, decoder.inputSpace), null);
      const logits = noGrad(() => decoder.model.forward({ inputsEmbeds: single, attentionMask: singleMask, labels: labels.slice(0, 0, 1) }).logits);
      close(batch.logits.detach().select(0, row), logits.select(0, 0), 1e-5);
    }
    batch.loss!.backward();
    for (const [source, valid] of [[prefixTensor, prefixMask], [mainTensor, mainMask]] as const) {
      expect(source.grad!.maskedSelect(valid.logicalNot()).abs().sum().item()).toBe(0);
      expect(source.grad!.allFinite()).toBe(true);
      expect(source.grad!.maskedSelect(valid).abs().sum().item()).toBeGreaterThan(0);
    }
  });

  it.each(['sequence', 'pooled'] as const)('encoder prefix padding is position and gradient invariant (%s)', async (readout) => {
    const space = new Space('native-context', 8, { organization: 'sequence' });
    const encoder = await TextEncoder.fromFoundation(T5, { readout, contextSpace: space });
    encoder.tokenizer.paddingSide = 'left';
    const raw = randn([2, 4, 8]);
    raw.requiresGrad = true;
    const keep = tensor([[false, true, false, true], [true, false, false, false]]);
    const batch = encoder.call(['hello world', 'hello'], { context: { latents: [new Latent(raw, space, { mask: keep })] } });
    ['hello world', 'hello'].forEach((text, row) => {
      const compact = new Latent(raw.detach().select(0, row).maskedSelect(keep.select(0, row)).unsqueeze(0), space);
      const single = noGrad(() => encoder.call(text, { context: { latents: [compact] } }));
      const actual = readout === 'sequence' ? batch.tensor.detach().select(0, row).maskedSelect(batch.mask!.select(0, row)) : batch.tensor.detach().select(0, row);
      const expected = readout === 'sequence' ? single.tensor.select(0, 0).maskedSelect(single.mask!.select(0, 0)) : single.tensor.select(0, 0);
      close(actual, expected);
    });
    if (readout === 'sequence') {
      expect(batch.mask!.tolist()).toEqual([[true, true], [false, true]]);
      expect(batch.tensor.maskedSelect(batch.mask!.logicalNot()).abs().sum().item()).toBe(0);
    }
    batch.tensor.square().sum().backward();
    expect(raw.grad!.maskedSelect(keep.logicalNot()).abs().sum().item()).toBe(0);
    expect(raw.grad!.allFinite()).toBe(true);
    expect(raw.grad!.maskedSelect(keep).abs().sum().item()).toBeGreaterThan(0);
  });

  it.each(['sequence', 'pooled'] as const)('masked text conditioning survives artifact and in-memory replay (%s)', async (readout) => {
    const space = new Space('conditioning', 8, { organization: 'sequence' });
    const model = await TextEncoder.fromFoundation(T5, { contextSpace: space, readout });
    const raw = randn([1, 1, 8]);
    const prefix = new Latent(cat([raw, randn([1, 20, 8]).mul(1e30)], 1), space, { mask: tensor([[true, ...Array(20).fill(false)]]) });
    const session = trace();
    const result = session.run(() => model.call('hello world', { context: { latents: [prefix] } }));
    const expected = model.call('hello world', { context: { latents: [new Latent(raw, space)] } }).tensor;
    close(result.tensor, expected);
    await model.savePretrained(join(scratch, `conditioning-${readout}`));
    const restored = await TextEncoder.fromPretrained(join(scratch, `conditioning-${readout}`));
    close(restored.call('hello world', { context: { latents: [prefix] } }).tensor, expected);
    close((session.replay(session.ref(result)) as Latent).tensor, expected);
  });
});

describe('pretrained text conditioning (remaining tests/vec/test_pretrained_text.py cases)', () => {
  it('training capture keeps a context envelope and replays after a fresh restart', async () => {
    const decoder = await TextDecoder.fromFoundation(T5, { inputSpace: new Space('input', 3, { organization: 'sequence' }) });
    const trainer = Trainer.fromTool(decoder);
    const value = new Latent(randn([1, 2, 3]), decoder.inputSpace);
    const raw = randn([1, 1, 3]);
    raw.requiresGrad = true;
    const prefix = new Latent(raw, decoder.inputSpace);
    const envelope = { value, context: { latents: [prefix] } };
    const loss = decoder.trainingOperation.call({ inputs: envelope, targets: 'answer' });
    loss.backward();
    expect(raw.grad!.abs().sum().item()).toBeGreaterThan(0);
    const session = trainer.capture(envelope, 'answer', { source: 'fixture' });
    const codecs = { latent: Latent, space: Space };
    await session.save(join(scratch, 'context.json'), { operations: trainer.operations, codecs });
    await decoder.savePretrained(join(scratch, 'context-model'));
    const restored = await TextDecoder.fromPretrained(join(scratch, 'context-model'));
    const restarted = Trainer.fromTool(restored);
    const experience = await loadExperience(join(scratch, 'context.json'), { operations: restarted.operations, codecs });
    expect(Number.isFinite(restarted.step(experience))).toBe(true);
    expect(() => trainer.capture({ ...envelope, targets: 'leaked' }, 'answer', { source: 'fixture' })).toThrow(/envelope/);
    expect(() => trainer.capture({ value, context: { targets: 'leaked' } }, 'answer', { source: 'fixture' })).toThrow(/context/);
  });

  it.each(['sequence', 'pooled'] as const)('encoder prefix padding is position and gradient invariant (%s)', async (readout) => {
    const space = new Space('native-context', 8, { organization: 'sequence' });
    const encoder = await TextEncoder.fromFoundation(T5, { readout, contextSpace: space });
    encoder.tokenizer.paddingSide = 'left';
    const raw = randn([2, 4, 8]);
    raw.requiresGrad = true;
    const keep = tensor([[false, true, false, true], [true, false, false, false]]);
    const batch = encoder.call(['hello world', 'hello'], { context: { latents: [new Latent(raw, space, { mask: keep })] } });
    ['hello world', 'hello'].forEach((text, row) => {
      const compact = new Latent(noGrad(() => raw.select(0, row).maskedSelect(keep.select(0, row)).unsqueeze(0)), space);
      const single = noGrad(() => encoder.call(text, { context: { latents: [compact] } }));
      if (readout === 'sequence') {
        close(noGrad(() => batch.tensor.select(0, row).maskedSelect(batch.mask!.select(0, row))),
          single.tensor.select(0, 0).maskedSelect(single.mask!.select(0, 0)), 1e-6);
      } else {
        close(noGrad(() => batch.tensor.select(0, row)), single.tensor.select(0, 0), 1e-6);
      }
    });
    if (readout === 'sequence') {
      expect(batch.mask!.tolist()).toEqual([[true, true], [false, true]]);
      expect(noGrad(() => batch.tensor.maskedSelect(batch.mask!.logicalNot()).abs().sum().item())).toBe(0);
    }
    batch.tensor.square().sum().backward();
    const grad = raw.grad!;
    expect(grad.maskedSelect(keep.logicalNot()).abs().sum().item()).toBe(0);
    expect(grad.allFinite()).toBe(true);
    expect(grad.maskedSelect(keep).abs().sum().item()).toBeGreaterThan(0);
  });

  it.each(['decoder', 'sequence', 'pooled'] as const)('masked text conditioning survives the artifact and durable replay (%s)', async (kind) => {
    const space = new Space('conditioning', kind === 'decoder' ? 3 : 8, { organization: 'sequence' });
    const model: TextDecoder | TextEncoder = kind === 'decoder'
      ? await TextDecoder.fromFoundation(T5, { inputSpace: space })
      : await TextEncoder.fromFoundation(T5, { contextSpace: space, readout: kind });
    const value = kind === 'decoder' ? new Latent(randn([1, 2, 3]), space) : 'hello world';
    const raw = randn([1, 1, space.dimensions]);
    const huge = tensor(new Array(20 * space.dimensions).fill(1e30), { shape: [1, 20, space.dimensions] });
    const prefix = new Latent(cat([raw, huge], 1), space, { mask: tensor([[true, ...new Array(20).fill(false)]]) });
    const compact = new Latent(raw, space);
    let session;
    let expected: Tensor;
    if (model instanceof TextDecoder) {
      const trainer = Trainer.fromTool(model);
      session = trainer.capture({ value, context: { latents: [prefix] } }, 'answer', { source: 'padding regression fixture' });
      expected = noGrad(() => model.loss(value as Latent, 'answer', { context: { latents: [compact] } }));
    } else {
      session = trace();
      const result = session.run(() => model.call(value as string, { context: { latents: [prefix] } }));
      expected = noGrad(() => model.call(value as string, { context: { latents: [compact] } }).tensor);
      close(result.tensor, expected, 1e-6);
    }
    const directory = join(scratch, `masked-${kind}`);
    await model.savePretrained(directory);
    const restored = kind === 'decoder' ? await TextDecoder.fromPretrained(directory) : await TextEncoder.fromPretrained(directory);
    const codecs = { latent: Latent, space: Space };
    await session.save(join(directory, 'trace.json'), { operations: model.operationBindings(), codecs });
    const replayed = await loadExperience(join(directory, 'trace.json'), { operations: restored.operationBindings(), codecs });
    const actual = replayed.replay(replayed.calls.at(-1)!.output) as Tensor | Latent;
    close(noGrad(() => (actual instanceof Latent ? actual.tensor : actual)), expected, 1e-6);
    if (kind === 'decoder') expect(Number.isFinite(Trainer.fromTool(restored as TextDecoder).step(replayed))).toBe(true);
  });
});
