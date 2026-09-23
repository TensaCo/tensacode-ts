/**
 * Python-saved vector artifacts load in TypeScript, evaluate identically,
 * re-save byte-identically and fingerprint identically
 * (fixtures: ``scripts/fixtures/vec_fixtures.py``).
 */
import { describe, expect, it } from 'vitest';
import { noGrad, tensor, type Tensor } from '../../src/nn/index.js';
import {
  CandidateSet, Classify, Decode, Latent, Score, Space, Transform, VocabularyEncoder, PatchEncoder,
} from '../../src/ops/vec/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson, fromJson } from '../helpers/fixtures.js';
import { expectByteIdenticalResave, fingerprints, fixturePath, scratchDirectory } from './helpers.js';

const records = fixtureJson('vec/owned.json');
const scratch = scratchDirectory();

const S = new Space('owned-input', 3);
const O = new Space('owned-output', 2);

describe('owned vector artifacts written by Python', () => {
  it('Transform (linear) evaluates, reports configuration and fingerprints like Python', async () => {
    const record = records.transform_linear;
    const op = await Transform.fromPretrained(fixturePath('transform_linear'));
    expect(op).toBeInstanceOf(Transform);
    expect(op.configuration()).toEqual(record.configuration);
    expect([...op.stateDict().keys()]).toEqual(record.state_keys);
    const value = new Latent(fromJson(record.value), S);
    const output = noGrad(() => op.call(value)) as Latent;
    expectClose(output.tensor.data, record.output.data, 1e-6);
    expect(noGrad(() => op.loss(value, new Latent(fromJson(record.target), O))).item()).toBeCloseTo(record.loss, 5);
    expect(fingerprints(op.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(op, 'transform_linear', scratch);
  });

  it('Classify (mlp) predictions and loss', async () => {
    const record = records.classify_mlp;
    const op = await Classify.fromPretrained(fixturePath('classify_mlp'));
    expect(op.configuration()).toEqual(record.configuration);
    expect([...op.stateDict().keys()]).toEqual(record.state_keys);
    const value = fromJson(record.value);
    const prediction = noGrad(() => op.call(new Latent(value, S)));
    expectClose(prediction.logits.data, record.logits.data, 1e-6);
    expect(prediction.values).toEqual(record.values);
    expectClose(noGrad(() => op.call(new Latent(value.select(0, 0), S))).logits.data, record.single.data, 1e-6);
    expect(op.loss(new Latent(value, S), tensor([0, 1], { dtype: 'int64' })).item()).toBeCloseTo(record.loss, 5);
    expect(op.loss(new Latent(value, S), ['a', 'b']).item()).toBeCloseTo(record.loss, 5);
    expect(fingerprints(op.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(op, 'classify_mlp', scratch);
  });

  it('Decode (sequence readout) masks source padding', async () => {
    const record = records.decode_sequence;
    const op = await Decode.fromPretrained(fixturePath('decode_sequence'));
    expect(op.configuration()).toEqual(record.configuration);
    const space = new Space('decode-source', 3, { organization: 'sequence' });
    const value = new Latent(fromJson(record.value), space, { mask: tensor(record.mask, { dtype: 'bool' }) });
    expectClose((noGrad(() => op.call(value)) as Tensor).data, record.output.data, 1e-6);
    expect(op.loss(value, tensor([[1, 1], [100, 100]])).item()).toBeCloseTo(record.loss, 5);
    expect(fingerprints(op.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(op, 'decode_sequence', scratch);
  });

  it('Score (mlp pair readout) with masked candidates', async () => {
    const record = records.score_mlp;
    const op = await Score.fromPretrained(fixturePath('score_mlp'));
    expect(op.configuration()).toEqual(record.configuration);
    expect([...op.stateDict().keys()]).toEqual(record.state_keys);
    const values = new CandidateSet(
      new Latent(fromJson(record.query), S),
      new Latent(fromJson(record.candidates), S, { mask: tensor(record.mask, { dtype: 'bool' }) }),
      ['a', 'b', 'c', 'd'],
    );
    expectClose(noGrad(() => op.call(values)).values.data, record.scores.data, 1e-6);
    expect(op.loss(values, fromJson(record.target)).item()).toBeCloseTo(record.loss, 5);
    expect(fingerprints(op.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(op, 'score_mlp', scratch);
  });

  it('Transform (native BERT) with latent prefix context', async () => {
    const record = records.transform_bert;
    const op = await Transform.fromPretrained(fixturePath('transform_bert'));
    expect(op.configuration()).toEqual(record.configuration);
    expect([...op.stateDict().keys()]).toEqual(record.state_keys);
    const tokens = new Space('tokens', 3, { organization: 'sequence' });
    const value = new Latent(fromJson(record.value), tokens);
    const prefix = new Latent(fromJson(record.prefix), tokens, { mask: tensor(record.prefix_mask, { dtype: 'bool' }) });
    const output = noGrad(() => op.call(value, { context: { latents: [prefix] } })) as Latent;
    expectClose(output.tensor.data, record.output.data, 1e-5);
    expectClose((noGrad(() => op.call(value)) as Latent).tensor.data, record.plain.data, 1e-5);
    expect(fingerprints(op.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(op, 'transform_bert', scratch);
  });

  it('VocabularyEncoder and PatchEncoder', async () => {
    const vocabulary = await VocabularyEncoder.fromPretrained(fixturePath('vocabulary'));
    expect(vocabulary.configuration()).toEqual(records.vocabulary.configuration);
    expectClose((noGrad(() => vocabulary.call(records.vocabulary.texts)) as Latent).tensor.data, records.vocabulary.output.data, 1e-6);
    expect(fingerprints(vocabulary.operationBindings())).toEqual(records.vocabulary.bindings);
    await expectByteIdenticalResave(vocabulary, 'vocabulary', scratch);

    const patch = await PatchEncoder.fromPretrained(fixturePath('patch'));
    expect(patch.configuration()).toEqual(records.patch.configuration);
    const result = noGrad(() => patch.call(fromJson(records.patch.images)));
    expectClose(result.tensor.data, records.patch.output.data, 1e-6);
    expectClose(result.coordinates!.data, records.patch.coordinates.data, 0);
    expect(fingerprints(patch.operationBindings())).toEqual(records.patch.bindings);
    await expectByteIdenticalResave(patch, 'patch', scratch);
    const fresh = new PatchEncoder({ in_channels: 3, patch_size: 2, output_space: new Space('patches', 4, { organization: 'spatial' }).configuration() as any });
    expect(fresh.configuration()).toEqual(records.patch_default_configuration);
  });
});
