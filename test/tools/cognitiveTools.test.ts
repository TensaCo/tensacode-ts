/** Port of python/tests/models/test_cognitive_tools.py. */
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Adam, manualSeed, noGrad } from '../../src/nn/index.js';
import { Investigator } from '../../src/tools/investigator.js';
import { Planner } from '../../src/tools/planner.js';
import { Decision } from '../../src/tools/decision.js';
import { RankingSession } from '../../src/_internal/sessions/ranking.js';
import { FoundationEncoding } from '../../src/_internal/ranking.js';
import type { JsonObject } from '../../src/_internal/json.js';
import { bertConfig, scratch, wordLevelTokenizer } from './helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

const CONFIG: JsonObject = { vocabulary: ['find', 'red', 'blue', 'evidence', 'choose'], dimensions: 8, slots: 2, steps: 1 };
const CASE = {
  question: 'find red', evidence: [{ source_id: 'original', text: 'red evidence' }],
  hypotheses: [{ id: 'r', text: 'red' }, { id: 'b', text: 'blue' }],
};
const kase = (): JsonObject => structuredClone(CASE) as JsonObject;

describe('ranking tools', () => {
  it('Investigator gradients and receipt', () => {
    manualSeed(11);
    const tool = new Investigator(CONFIG);
    const parameters = tool.parameters();
    const before = kase();
    const loss = tool.objective.call({ inputs: kase(), targets: 'r' });
    loss.backward();
    expect((tool.rank.encode.module as unknown as { weight: { grad: { abs(): { sum(): { item(): number } } } } }).weight.grad.abs().sum().item()).toBeGreaterThan(0);
    expect(tool.rank.workspace.queries.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(tool.parameters()).toEqual(parameters);
    const receipt = tool.call(kase());
    expect(receipt.evidence).toEqual(CASE.evidence);
    expect(['r', 'b']).toContain(receipt.selected_id);
    expect((receipt.candidates as JsonObject[]).length).toBe(2);
    expect((receipt.candidates as JsonObject[]).reduce((total, item) => total + (item.probability as number), 0)).toBeCloseTo(1, 6);
    expect((receipt.attention as number[][])[0]!.length).toBe((receipt.attention_source_ids as unknown[]).length);
    expect(receipt.attention_source_ids).toContain('original');
    expect(kase()).toEqual(before);
    expect('objective' in tool.operationBindings()).toBe(true);
  });

  it('context changes candidate scores and targets never enter prediction', () => {
    manualSeed(4);
    const tool = new Investigator(CONFIG);
    const other = kase();
    ((other.evidence as JsonObject[])[0]!).text = 'blue evidence';
    expect(noGrad(() => tool.rank.call(kase())).equal(noGrad(() => tool.rank.call(other)))).toBe(false);
    const scores = noGrad(() => tool.rank.call(kase())).detach().clone();
    tool.loss(kase(), 0);
    tool.loss(kase(), 1);
    expect(noGrad(() => tool.rank.call(kase())).equal(scores)).toBe(true);
    expect(scores.data[0]).not.toBe(scores.data[1]);
  });

  for (const [name, cls] of [['Investigator', Investigator], ['Planner', Planner], ['Decision', Decision]] as const) {
    it(`${name} checkpoint round trip`, async () => {
      const tool = new cls(CONFIG);
      const input = cls === Planner ? { goal: CASE.question, evidence: CASE.evidence, plans: CASE.hypotheses } : kase();
      const directory = join(temp.dir, name);
      await tool.savePretrained(directory);
      const restored = await (cls as typeof Investigator).fromPretrained(directory);
      expect(noGrad(() => (restored as Investigator).rank.call(input)).equal(noGrad(() => tool.rank.call(input)))).toBe(true);
      expect(restored.call(structuredClone(input))).toEqual(tool.call(structuredClone(input)));
    });
  }

  it('observed plan loss never fabricates other targets', () => {
    const tool = new Planner(CONFIG);
    const input = { goal: 'choose red', evidence: CASE.evidence, plans: CASE.hypotheses };
    const observed = { candidate_id: 'r', outcome: 0.75 };
    const expected = (noGrad(() => tool.rank.call(input)).data[0]! - 0.75) ** 2;
    expect(tool.loss(input, observed).item()).toBeCloseTo(expected, 5);
    tool.loss(input, observed).backward();
    expect(tool.rank.workspace.queries.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect((tool.call(input).candidates as JsonObject[]).every((candidate) => !('outcome' in candidate))).toBe(true);
    expect(() => tool.loss(input, { candidate_id: 'missing', outcome: 1 })).toThrow();
  });

  it('rejects invalid inputs', () => {
    const tool = new Investigator(CONFIG);
    const bad = kase();
    ((bad.hypotheses as JsonObject[])[1]!).id = 'r';
    expect(() => tool.call(bad)).toThrow();
    expect(() => tool.loss(kase(), true)).toThrow();
    expect(() => new Investigator({ ...CONFIG, vocabulary: ['red', 'red'] })).toThrow();
  });

  it('training reduces explicit supervised loss', () => {
    manualSeed(14);
    const tool = new Investigator(CONFIG);
    const initial = tool.loss(kase(), 'r').item();
    const optimizer = new Adam(tool.parameters(), { lr: 0.02 });
    for (let step = 0; step < 15; step += 1) {
      optimizer.zeroGrad();
      const loss = tool.loss(kase(), 'r');
      loss.backward();
      optimizer.step();
    }
    expect(tool.loss(kase(), 'r').item()).toBeLessThan(initial * 0.25);
    expect(tool.call(kase()).selected_id).toBe('r');
  });

  it('sessions are independent, transactional and restorable', async () => {
    const tool = new Investigator(CONFIG);
    const one = tool.newSession();
    const two = tool.newSession();
    const first = kase();
    first.hypotheses = (first.hypotheses as JsonObject[]).slice(0, 1);
    const receipt = one.call(first);
    expect(receipt.revised).toBe(false);
    ((first.evidence as JsonObject[])[0]!).text = 'mutated';
    (receipt.evidence as unknown[]).length = 0;
    expect(((one.history[0]!.inputs.evidence as JsonObject[])[0]!).text).toBe('red evidence');
    expect((one.history[0]!.receipt.evidence as unknown[]).length).toBeGreaterThan(0);
    expect(two.history).toEqual([]);
    const before = one.history;
    expect(() => one.call({ question: 'find red', hypotheses: [] })).toThrow();
    expect(one.history).toEqual(before);
    const path = join(temp.dir, 'session.json');
    await one.save(path);
    const loaded = await RankingSession.load(path, tool);
    const second = kase();
    second.hypotheses = (second.hypotheses as JsonObject[]).slice(1);
    const updated = loaded.call(second);
    expect(updated.previous_selected_id).toBe('r');
    expect(updated.selected_id).toBe('b');
    expect(updated.revised).toBe(true);
    expect(one.history).toEqual(before);
    await expect(RankingSession.load(path, new Planner(CONFIG))).rejects.toThrow();
  });

  it('workspace ablations are explicit and differentiable', () => {
    const tool = new Investigator(CONFIG);
    const baseline = tool.rank.call(kase());
    const zero = tool.rank.compute(kase(), { workspaceAblation: 'zero' }).scores;
    const bypass = tool.rank.compute(kase(), { workspaceAblation: 'bypass' }).scores;
    expect(zero.shape).toEqual(baseline.shape);
    expect(bypass.shape).toEqual(baseline.shape);
    expect(baseline.equal(zero)).toBe(false);
    expect(baseline.equal(bypass)).toBe(false);
    expect(() => tool.call(kase(), { context: { ignored: true } })).toThrow();
  });

  it('distribution targets supervise all positive candidates', () => {
    const tool = new Investigator(CONFIG);
    const logits = noGrad(() => tool.rank.call(kase()));
    const expected = -logits.logSoftmax(-1).toArray().reduce((total, value) => total + 0.5 * value, 0);
    expect(tool.loss(kase(), [0.5, 0.5]).item()).toBeCloseTo(expected, 5);
    for (const target of [[1, 1], [-1, 2], [Number.NaN, 0], [1]]) expect(() => tool.loss(kase(), target)).toThrow();
  });

  it('owned native foundation reconstructs without download', async () => {
    const config: JsonObject = {
      foundation_config: bertConfig({ vocab_size: 6, hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2, intermediate_size: 16 }),
      tokenizer_json: wordLevelTokenizer({ '[UNK]': 0, '[PAD]': 1, find: 2, red: 3, blue: 4, evidence: 5 }, '[UNK]'),
      tokenizer_special_tokens: { unk_token: '[UNK]', pad_token: '[PAD]' },
      dimensions: 8, slots: 2, steps: 1, max_tokens: 16, cache_records: 2,
    };
    const tool = new Investigator(config);
    tool.train();
    const encoding = tool.rank.encode.module as FoundationEncoding;
    expect(encoding.model.training).toBe(false);
    expect(encoding.model.parameters().some((p) => p.requiresGrad)).toBe(false);
    let calls = 0;
    const forward = encoding.model.forward.bind(encoding.model);
    encoding.model.forward = (inputs) => {
      calls += 1;
      return forward(inputs);
    };
    const first = tool.rank.call(kase()).detach();
    expect(tool.rank.call(kase()).equal(first)).toBe(true);
    expect(calls).toBe(1);
    tool.loadStateDict(tool.stateDict());
    expect(tool.rank.call(kase()).equal(first)).toBe(true);
    expect(calls).toBe(2);
    encoding.model.forward = forward;
    tool.rank.clearEncodingCache();
    noGrad(() => tool.call(kase()));
    tool.loss(kase(), [0.5, 0.5]).backward();
    expect((tool.rank.projection!.module as unknown as { weight: { grad: { abs(): { sum(): { item(): number } } } } }).weight.grad.abs().sum().item()).toBeGreaterThan(0);
    mkdirSync(join(temp.dir, 'native'), { recursive: true });
    await tool.savePretrained(join(temp.dir, 'native'));
    const restored = await Investigator.fromPretrained(join(temp.dir, 'native'));
    expect(noGrad(() => restored.rank.call(kase())).equal(noGrad(() => tool.rank.call(kase())))).toBe(true);
    const unfrozen = new Investigator({ ...config, freeze_foundation: false });
    unfrozen.loss(kase(), 0).backward();
    expect((unfrozen.rank.encode.module as FoundationEncoding).model.parameters().some((p) => p.grad !== null && p.grad.abs().sum().item() > 0)).toBe(true);
  });
});
