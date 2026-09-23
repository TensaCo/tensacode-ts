/** Port of ``tests/vec/test_owned_vector_operations.py`` (Trainer restarts live in the training module). */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Linear, noGrad, ones, randn, tensor, type Tensor } from '../../src/nn/index.js';
import { ValueError } from '../../src/errors.js';
import { CandidateSet, Classify, Decode, Latent, Score, Space, Transform } from '../../src/ops/vec/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixturePath } from './helpers.js';
import { DotScore } from './modules.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-owned-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const S = new Space('owned-input', 3);
const O = new Space('owned-output', 2);
const s = S.configuration() as any;
const o = O.configuration() as any;

describe('owned vector operations (tests/vec/test_owned_vector_operations.py)', () => {
  it('owned linear transform roundtrip and loss', async () => {
    const op = new Transform({ architecture: 'linear', input_space: s, output_space: o });
    const value = new Latent(randn([2, 3]), S);
    const target = new Latent(randn([2, 2]), O);
    const loss = op.loss(value, target);
    loss.backward();
    expect(op.parameters().every((parameter) => parameter.grad !== null)).toBe(true);
    await op.savePretrained(join(scratch, 'model'));
    const loaded = await Transform.fromPretrained(join(scratch, 'model'));
    expect(loaded.configuration()).toEqual(op.configuration());
    expect((loaded.call(value) as Latent).tensor.equal((op.call(value) as Latent).tensor)).toBe(true);
    expect(loaded.loss(value, target).item()).toBe(loss.item());
  });

  it.each([Transform, Classify, Score, Decode])('old module constructors are rejected (%o)', (cls) => {
    expect(() => new (cls as any)(new Linear(3, 2))).toThrow();
  });

  it('owned classification and decode', async () => {
    const value = new Latent(randn([2, 3]), S);
    for (const [cls, extra, targets] of [
      [Classify, { labels: ['a', 'b'] }, tensor([0, 1], { dtype: 'int64' })],
      [Decode, { output_dimensions: 2, output: 'regression values' }, randn([2, 2])],
    ] as const) {
      const op = new (cls as any)({ architecture: 'mlp', input_space: s, hidden_dimensions: [4], ...extra });
      const loss = op.loss(value, targets) as Tensor;
      loss.backward();
      expect(op.parameters().every((parameter: any) => parameter.grad !== null)).toBe(true);
      await op.savePretrained(join(scratch, cls.name));
      const loaded = await (cls as any).fromPretrained(join(scratch, cls.name));
      expect(loaded.loss(value, targets).item()).toBe(loss.item());
    }
  });

  it('owned pair score artifact', async () => {
    const value = new CandidateSet(new Latent(randn([3]), S), new Latent(randn([4, 3]), S), ['a', 'b', 'c', 'd']);
    const op = new Score({ architecture: 'mlp', query_space: s, candidate_space: s, hidden_dimensions: [5], meaning: 'authored relevance logits' });
    op.loss(value, randn([4])).backward();
    expect(op.parameters().every((parameter) => parameter.grad !== null)).toBe(true);
    await op.savePretrained(join(scratch, 'score'));
    const loaded = await Score.fromPretrained(join(scratch, 'score'));
    expect(loaded.call(value).values.equal(op.call(value).values)).toBe(true);
  });

  it('expert module artifact is rejected', async () => {
    const op = Transform.fromModule(new Linear(3, 2));
    expect((op.call(ones([3])) as Tensor).shape).toEqual([2]);
    await expect(op.savePretrained(join(scratch, 'unsupported'))).rejects.toThrow(/supplied|reconstruct/);
    const score = Score.fromModule(new DotScore(), { querySpace: S, candidateSpace: S, meaning: 'x' });
    await expect(score.savePretrained(join(scratch, 'unsupported-score'))).rejects.toThrow(/supplied/);
  });

  it('native transformer context mask and artifact', async () => {
    const seq = new Space('tokens', 3, { organization: 'sequence' });
    const out = new Space('states', 2, { organization: 'sequence' });
    const op = new Transform({
      architecture: 'transformer', input_space: seq.configuration() as any, output_space: out.configuration() as any,
      native_config: { model_type: 'bert', hidden_size: 4, num_hidden_layers: 1, num_attention_heads: 2, intermediate_size: 6, hidden_dropout_prob: 0.0, attention_probs_dropout_prob: 0.0, vocab_size: 8 },
    }).eval();
    const x = new Latent(randn([2, 3]), seq);
    const prefix = new Latent(randn([2, 3]), seq, { mask: tensor([true, false]) });
    const result = op.call(x, { context: { latents: [prefix] } }) as Latent;
    const changed = new Latent(prefix.tensor.add(tensor([[0], [999]])), seq, { mask: prefix.mask });
    expectClose(result.tensor.data, (op.call(x, { context: { latents: [changed] } }) as Latent).tensor.data, 1e-5);
    expect(result.tensor.shape).toEqual([2, 2]);
    op.loss(x, new Latent(randn([2, 2]), out), { context: { latents: [prefix] } }).backward();
    await op.savePretrained(join(scratch, 'native'));
    const loaded = await Transform.fromPretrained(join(scratch, 'native'));
    expect((loaded.call(x, { context: { latents: [prefix] } }) as Latent).tensor.equal(result.tensor)).toBe(true);
    expect(() => op.call(x, { context: { targets: x } })).toThrow(/context/);
  });

  it('linear score has query-candidate interaction', () => {
    const space = new Space('scalar', 1);
    const op = new Score({ architecture: 'linear', query_space: space.configuration() as any, candidate_space: space.configuration() as any, meaning: 'pair utility' });
    const first = new CandidateSet(new Latent(tensor([1]), space), new Latent(tensor([[1], [2]]), space), ['a', 'b']);
    const second = new CandidateSet(new Latent(tensor([3]), space), first.candidates, first.identities);
    const diff = (values: Tensor) => values.get(1) - values.get(0);
    expect(Math.abs(diff(op.call(first).values) - diff(op.call(second).values))).toBeGreaterThan(1e-7);
  });

  it('foundation native encoder weights survive the owned artifact', async () => {
    const foundation = fixturePath('bert_foundation');
    const op = await Transform.fromFoundation(foundation, { inputSpace: S, outputSpace: O, localFilesOnly: true });
    expect(op.configuration().foundation).toMatchObject({ input_bridge: 'untrained', output_head: 'untrained', repo: foundation });
    const value = new Latent(randn([2, 3]), S);
    const expected = (noGrad(() => op.call(value)) as Latent).tensor;
    await op.savePretrained(join(scratch, 'owned'));
    const loaded = await Transform.fromPretrained(join(scratch, 'owned'));
    expect((noGrad(() => loaded.call(value)) as Latent).tensor.equal(expected)).toBe(true);
    const scorer = await Score.fromFoundation(foundation, { querySpace: S, candidateSpace: S, meaning: 'pair utility', localFilesOnly: true });
    const candidates = new CandidateSet(new Latent(randn([3]), S), new Latent(randn([2, 3]), S), ['a', 'b']);
    await scorer.savePretrained(join(scratch, 'scorer'));
    const loadedScore = await Score.fromPretrained(join(scratch, 'scorer'));
    expect(noGrad(() => scorer.call(candidates)).values.equal(noGrad(() => loadedScore.call(candidates)).values)).toBe(true);
    const native = (await Transform.fromFoundation(foundation, { inputSpace: S, outputSpace: O })).model!;
    expect(native.getInputEmbeddings().weight.equal(op.model!.getInputEmbeddings().weight)).toBe(true);
    expect((scorer.module as any).model.getInputEmbeddings().weight.equal(op.model!.getInputEmbeddings().weight)).toBe(true);
    await expect(Transform.fromFoundation(foundation, { inputSpace: S, outputSpace: O, useSafetensors: false })).rejects.toThrow(/safetensors/);
  });

  it.each(['linear', 'mlp', 'transformer'])('unknown owned fields are rejected (%s)', (architecture) => {
    expect(() => new Transform({ architecture, input_space: s, output_space: o, module: 'untrusted' })).toThrow(/unknown/);
  });

  it('masked regression targets do not poison gradients', () => {
    const seq = new Space('regression-source', 3, { organization: 'sequence' });
    const out = new Space('regression-output', 2, { organization: 'sequence' });
    const value = new Latent(randn([2, 3]), seq, { mask: tensor([true, false]) });
    const target = new Latent(tensor([[1, 2], [Number.NaN, Number.NaN]]), out);
    const op = new Transform({ input_space: seq.configuration() as any, output_space: out.configuration() as any });
    const loss = op.loss(value, target);
    loss.backward();
    expect(Number.isFinite(loss.item())).toBe(true);
    expect(op.parameters().every((parameter) => parameter.grad!.allFinite())).toBe(true);
  });

  it('sequence tensor decode loss excludes source padding', () => {
    const seq = new Space('decode-source', 3, { organization: 'sequence' });
    const op = new Decode({ input_space: seq.configuration() as any, output_dimensions: 2, output: 'regression values', readout: 'sequence' });
    const value = new Latent(randn([2, 3]), seq, { mask: tensor([true, false]) });
    const expected = noGrad(() => (op.call(value) as Tensor).select(0, 0).sub(1).square().mean()).item();
    expect(noGrad(() => op.loss(value, tensor([[1, 1], [100, 100]]))).item()).toBeCloseTo(expected, 6);
  });

  it('owned objective is registered with the public identity', () => {
    const op = new Transform({ input_space: s, output_space: o });
    expect(op.operationBindings().objective).toBe(op.trainingOperation);
    expect(op.trainingOperation.operationIdentity()).toBe('tensorcode.ops.vec.transform.Transform.objective');
    const value = new Latent(randn([2, 3]), S);
    const target = new Latent(randn([2, 2]), O);
    expect(op.trainingOperation.call({ inputs: value, targets: target }).item()).toBeCloseTo(op.loss(value, target).item(), 6);
    expect(op.trainingInputsIncludeTargets).toBe(true);
    expect(() => op.trainingOperation.call({ inputs: value } as any)).toThrow(/inputs and targets/);
  });

  it('validation errors follow the Python messages', () => {
    expect(() => new Classify({ input_space: s, labels: ['a', 'a'] })).toThrow(/nonempty unique/);
    expect(() => new Classify({ input_space: s, labels: ['a'], readout: 'sequence' })).toThrow(/pooled/);
    expect(() => new Decode({ input_space: s, output_dimensions: 0, output: 'x' })).toThrow(/positive integer/);
    expect(() => new Decode({ input_space: s, output_dimensions: 1, output: '' })).toThrow(/description/);
    expect(() => new Transform({ input_space: s, output_space: o, architecture: 'rnn' })).toThrow(/linear, mlp or transformer/);
    expect(() => new Transform({ input_space: s, output_space: o, architecture: 'mlp' })).toThrow(/nonempty hidden_dimensions/);
    expect(() => new Transform({ input_space: s, output_space: o, hidden_dimensions: [2] })).toThrow(/does not use hidden_dimensions/);
    expect(() => new Transform({ input_space: s, output_space: o, native_config: {} })).toThrow(/require transformer/);
    expect(() => new Transform({ input_space: s, output_space: new Space('x', 2, { organization: 'sequence' }).configuration() as any })).toThrow(/organization/);
    expect(() => new Transform({ architecture: 'transformer', input_space: s, output_space: o, native_config: { model_type: 't5' } })).toThrow(/bert, roberta, distilbert/);
    expect(() => new Transform({ input_space: s, output_space: o }).call(new Latent(ones([3]), S), { context: { latents: [new Latent(ones([3]), S)] } })).toThrow(/transformer/);
    expect(() => new Classify({ input_space: s, labels: ['a', 'b'] }).loss(new Latent(ones([3]), S), 'c')).toThrow(ValueError);
  });
});
