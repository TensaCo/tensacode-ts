/** Port of python/tests/models/test_retrieval_encoder.py (tiny local encoders; pooling/ownership, not retrieval quality). */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { eye, noGrad, serializeModel } from '../../src/nn/index.js';
import { RetrievalEncoder } from '../../src/_internal/retrieval.js';
import { LearnedEpisodicMemory, type MemoryOwner } from '../../src/_internal/memory/learned.js';
import { Investigator } from '../../src/tools/investigator.js';
import { Evidence } from '../../src/tools/cognition.js';
import { NativeConfig } from '../../src/_internal/native/config.js';
import { createNativeModel } from '../../src/_internal/native/registry.js';
import { retrievalConfig, scratch } from './helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

const owner = (tool: Investigator) => tool as unknown as MemoryOwner;

describe('RetrievalEncoder', () => {
  it('masked mean has no projection and padding does not change embeddings', () => {
    const model = new RetrievalEncoder(retrievalConfig()).eval();
    const parameters = new Set(model.parameters());
    noGrad(() => {
      const alone = model.call(['alpha']);
      const together = model.call(['alpha', 'beta beta beta']);
      expect(alone.shape).toEqual([1, 8]);
      expect(together.dtype).toBe('float32');
      alone.toArray().forEach((value, index) => expect(Math.abs(value - together.data[index]!)).toBeLessThan(1e-6));
      together.norm(2, -1).toArray().forEach((value) => expect(Math.abs(value - 1)).toBeLessThan(1e-6));
    });
    expect(new Set(model.parameters())).toEqual(parameters);
    expect(new Set(model.parameters())).toEqual(new Set(model.encode.parameters()));
    const receipt = model.receipt(['alpha alpha alpha alpha alpha']);
    expect(receipt.input_truncated).toEqual([true]);
    expect(receipt.pooling).toBe('masked_mean');
    expect(receipt.normalized).toBe(true);
    expect(receipt.dimensions).toBe(8);
    for (const invalid of [{ pooling: 'cls' }, { normalize: false }, { generator: {} }]) {
      expect(() => new RetrievalEncoder({ ...retrievalConfig(), ...invalid })).toThrow();
    }
  });

  it('explicit contrastive targets produce gradients without new parameters', () => {
    const model = new RetrievalEncoder(retrievalConfig());
    const parameters = new Set(model.parameters());
    const loss = model.loss(['alpha', 'beta'], ['alpha', 'beta'], eye(2, { dtype: 'bool' }));
    expect(Number.isFinite(loss.item())).toBe(true);
    loss.backward();
    expect(model.encode.parameters().some((p) => p.grad !== null && p.grad.abs().sum().item() > 0)).toBe(true);
    expect(new Set(model.parameters())).toEqual(parameters);
    expect(() => model.loss(['alpha'], ['beta'], [[false]])).toThrow();
  });

  it('Investigator owns the retrieval artifact and memory fingerprints only that encoder', async () => {
    const tool = new Investigator({ vocabulary: ['alpha', 'beta'], dimensions: 4, slots: 2, steps: 1, retrieval_encoder: retrievalConfig() }).eval();
    expect(tool.episodicEncoder).not.toBeNull();
    expect('episodic_encoder.encode' in tool.operationBindings()).toBe(true);
    const memory = new LearnedEpisodicMemory(owner(tool));
    memory.remember(new Evidence('a', 'alpha', 'document'), { episodeId: 'past' });
    expect(memory.metadata.encoder).toBe('owned_retrieval_encoder');
    const fingerprint = memory.fingerprint;
    noGrad(() => (tool.rank.encode.module as unknown as { weight: { add_(value: number): void } }).weight.add_(1));
    expect(memory.fingerprint).toBe(fingerprint);
    expect(memory.retrieve('alpha')[0]!.score).toBeCloseTo(1, 6);
    const expected = noGrad(() => tool.episodicEncoder!.call(['alpha', 'beta']));
    await tool.savePretrained(join(temp.dir, 'model'));
    const restored = await Investigator.fromPretrained(join(temp.dir, 'model'));
    expect(noGrad(() => restored.episodicEncoder!.call(['alpha', 'beta'])).equal(expected)).toBe(true);
    const rebuilt = LearnedEpisodicMemory.fromSnapshot(memory.snapshot(), { investigator: owner(restored) });
    expect(rebuilt.retrieve('alpha')[0]!.evidence.sourceId).toBe('document');
    noGrad(() => tool.episodicEncoder!.parameters()[0]!.add_(0.1));
    expect(() => memory.retrieve('alpha')).toThrow(/stale/);
    memory.rebuildIndex();
    expect(memory.retrieve('alpha')[0]!.score).toBeCloseTo(1, 6);
  });

  it('explicit local foundation bootstrap copies the encoder and tokenizer', async () => {
    const config = retrievalConfig();
    const root = join(temp.dir, 'foundation');
    mkdirSync(root, { recursive: true });
    const nativeConfig = NativeConfig.fromDict(config.foundation_config);
    const native = createNativeModel(nativeConfig, 'base').eval();
    writeFileSync(join(root, 'config.json'), JSON.stringify(nativeConfig.toDict()));
    writeFileSync(join(root, 'model.safetensors'), serializeModel(native));
    writeFileSync(join(root, 'tokenizer.json'), config.tokenizer_json as string);
    writeFileSync(join(root, 'tokenizer_config.json'), JSON.stringify({ tokenizer_class: 'PreTrainedTokenizerFast', pad_token: '<pad>', unk_token: '<unk>' }));
    const owned = await RetrievalEncoder.fromFoundation(root, { pooling: 'masked_mean', normalize: true, maxTokens: 4, localFilesOnly: true });
    expect(owned.encode.module.model.parameters()[0]!.equal(native.parameters()[0]!)).toBe(true);
    expect((owned.configuration().foundation as { repository: string }).repository).toBe(root);
    const tool = await Investigator.fromRetrievalFoundation(root, {
      pooling: 'masked_mean', normalize: true, maxTokens: 4, localFilesOnly: true,
      options: { vocabulary: ['alpha', 'beta'], dimensions: 4, slots: 2, steps: 1 },
    });
    expect(noGrad(() => tool.episodicEncoder!.call(['alpha'])).equal(noGrad(() => owned.call(['alpha'])))).toBe(true);
    await expect(RetrievalEncoder.fromFoundation(root, { pooling: 'cls', normalize: true })).rejects.toThrow(/masked_mean/);
  });

  it('retrieval loss requires queries/documents inputs and positives as targets', () => {
    const tool = new Investigator({ vocabulary: ['alpha', 'beta'], dimensions: 4, slots: 2, steps: 1, retrieval_encoder: retrievalConfig() }).eval();
    const inputs = { queries: ['alpha', 'beta'], documents: ['alpha', 'beta'] };
    const targets = [[true, false], [false, true]];
    expect(() => tool.retrievalLoss({ ...inputs, targets }, targets)).toThrow();
    expect(Number.isFinite(tool.retrievalLoss(inputs, targets).item())).toBe(true);
    expect(Number.isFinite(tool.episodicEncoder!.contrastiveLoss(inputs.queries, inputs.documents, targets).item())).toBe(true);
    const objective = tool.objective.call({ inputs: { mode: 'retrieval', inputs }, targets });
    expect(Number.isFinite(objective.item())).toBe(true);
    expect(() => new Investigator({ vocabulary: ['alpha'] }).retrievalLoss(inputs, targets)).toThrow(/not configured/);
  });
});
