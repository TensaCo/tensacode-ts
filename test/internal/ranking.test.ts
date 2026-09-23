import { describe, expect, it } from 'vitest';
import { Workspace } from '../../src/_internal/workspace.js';
import { RankOperation, normalizeRankingConfig } from '../../src/_internal/ranking.js';
import { TensorAdapter } from '../../src/_internal/vec/adapter.js';
import { fingerprint, operationConfiguration } from '../../src/_internal/fingerprint.js';
import { Dropout, GELU, LayerNorm, Linear, Sequential, loadModelFromBytes, noGrad, tensor } from '../../src/nn/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, fromJson } from '../helpers/fixtures.js';
import { trace } from '../../src/_internal/tracing.js';

const cases = fixtureJson('ranking.json');

describe('shared workspace and ranking match Python', () => {
  it('Workspace forward', () => {
    const record = cases.workspace;
    const workspace = new Workspace(4, 2, 2).eval();
    expect([...workspace.stateDict().keys()]).toEqual(record.state_keys);
    expect(workspace.configuration()).toEqual(record.configuration);
    loadModelFromBytes(workspace, fixtureBytes('workspace.safetensors'));
    const mask = tensor(record.mask, { dtype: 'bool' });
    const output = noGrad(() => workspace.forward(fromJson(record.encoded), mask));
    expectClose(output.conditioning.data, record.conditioning.data, 1e-5);
    expectClose(output.attention.data, record.attention.data, 1e-5);
    expectClose(output.relations.data, record.relations.data, 1e-5);
  });

  it('RankOperation receipt and configuration', () => {
    const record = cases.rank;
    expect(normalizeRankingConfig({ vocabulary: record.config.vocabulary, dimensions: 8, slots: 2, steps: 2 })).toEqual(record.config);
    const rank = new RankOperation(record.config, { taskKey: 'question', candidatesKey: 'hypotheses' }).eval();
    expect([...rank.stateDict().keys()]).toEqual(record.state_keys);
    expect(rank.configuration()).toEqual(record.configuration);
    loadModelFromBytes(rank, fixtureBytes('rank.safetensors'));
    const receipt = noGrad(() => rank.receipt(record.inputs, { probabilities: true })) as any;
    expect(receipt.selected_id).toBe(record.receipt.selected_id);
    expect(receipt.attention_source_ids).toEqual(record.receipt.attention_source_ids);
    expectClose(receipt.candidates.map((c: any) => c.predicted_score), record.receipt.candidates.map((c: any) => c.predicted_score), 1e-5);
    expectClose(receipt.candidates.map((c: any) => c.probability), record.receipt.candidates.map((c: any) => c.probability), 1e-5);
    expectClose(receipt.attention.flat(), record.receipt.attention.flat(), 1e-5);
  });

  it('operation fingerprints equal Python fingerprints', () => {
    const adapter = new TensorAdapter(new Sequential(new Linear(3, 2), new GELU(), new LayerNorm(2), new Dropout(0.1)));
    expect(operationConfiguration(adapter)).toEqual(cases.fingerprints.adapter.configuration);
    expect(fingerprint(operationConfiguration(adapter))).toBe(cases.fingerprints.adapter.fingerprint);
    const workspace = new Workspace(4, 2, 2);
    expect(fingerprint(operationConfiguration(workspace.key))).toBe(cases.fingerprints.workspace_key.fingerprint);
    const rank = new RankOperation(cases.rank.config, { taskKey: 'question', candidatesKey: 'hypotheses' });
    expect(fingerprint(operationConfiguration(rank))).toBe(cases.fingerprints.rank.fingerprint);
  });

  it('ranking gradients reach all parameters and nested calls are traced', () => {
    const rank = new RankOperation(cases.rank.config, { taskKey: 'question', candidatesKey: 'hypotheses' }).train();
    const session = trace();
    const scores = session.run(() => rank.call(cases.rank.inputs));
    expect(session.calls.length).toBeGreaterThan(1);
    scores.logSoftmax(-1).select(0, 0).neg().backward();
    const missing = rank.namedParameters().filter(([, parameter]) => parameter.grad === null).map(([name]) => name);
    expect(missing).toEqual([]);
    const replayed = session.replay(scores) as ReturnType<typeof tensor>;
    expectClose(replayed.data, scores.data, 1e-6);
  });
});
