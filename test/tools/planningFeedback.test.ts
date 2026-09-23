/** Port of python/tests/models/test_planning_feedback.py (CPU mechanics; supplied policies do not establish planning). */
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Linear, SGD, manualSeed, noGrad, tensor } from '../../src/nn/index.js';
import { crossEntropy as crossEntropyLoss } from '../../src/nn/ops/nn.js';
import { ActionOutcome } from '../../src/tools/actions.js';
import {
  ExecutablePlan, PlanExecutionResult, PlanExecutor, PlanStep, type ReplanRequest,
} from '../../src/_internal/execution/planning.js';
import { Planner } from '../../src/tools/planner.js';
import type { JsonObject } from '../../src/_internal/json.js';
import { scratch, tinyConfig } from './helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

function plan(candidateId: string, ...actions: string[]): ExecutablePlan {
  return new ExecutablePlan(candidateId, actions.map((action) => new PlanStep(action)));
}

describe('plan execution and feedback', () => {
  it('replans from real observations with an injected learned policy', async () => {
    // Fit a tiny policy to explicit fixture labels; the runtime does not supply policy.
    manualSeed(8);
    const classifier = new Linear(1, 2);
    const optimizer = new SGD(classifier.parameters(), { lr: 0.4 });
    for (let step = 0; step < 35; step += 1) {
      optimizer.zeroGrad();
      crossEntropyLoss(classifier.forward(tensor([[-1], [1]])), tensor([0, 1], { dtype: 'int64' })).backward();
      optimizer.step();
    }
    const calls: string[] = [];
    const policyInputs: unknown[] = [];
    const inspect = (state: { visited: string[] }) => {
      state.visited.push('inspect');
      calls.push('inspect');
      return new ActionOutcome(state, { measurement: 1 });
    };
    const finish = (state: unknown) => {
      calls.push('positive');
      return new ActionOutcome(state, { result: 'finished' }, true);
    };
    const policy = (request: ReplanRequest) => {
      policyInputs.push(request.evidence);
      const measurement = (request.experiences[request.experiences.length - 1]!.observation as JsonObject).measurement as number;
      const selected = noGrad(() => classifier.forward(tensor([[measurement]])).argmax().item());
      return plan('revised', ['negative', 'positive'][selected]!);
    };
    const source = { visited: [] as string[] };
    const executor = new PlanExecutor<{ visited: string[] }>({
      actions: { inspect, negative: (state) => new ActionOutcome(state, 'wrong'), positive: finish as (state: { visited: string[] }) => ActionOutcome<{ visited: string[] }> },
      replan: policy, maxSteps: 2,
    });
    const result = await executor.call(source, plan('initial', 'inspect', 'negative'));
    expect(result.stopReason).toBe('completed');
    expect(calls).toEqual(['inspect', 'positive']);
    expect(source).toEqual({ visited: [] });
    expect(result.experiences.map((item) => item.candidateId)).toEqual(['initial', 'revised']);
    expect((policyInputs[0] as { source_id: string }[])[0]!.source_id).toBe(result.experiences[0]!.sourceId);
    expect(result.experiences[1]!.toTarget(1)).toEqual({ candidate_id: 'revised', outcome: 1 });
    const path = join(temp.dir, 'trajectory.json');
    await result.save(path);
    expect(await PlanExecutionResult.load(path)).toEqual(result);
    expect(readFileSync(path, 'utf8')).not.toContain('classifier');
  });

  for (const [label, bad] of [
    ['unknown action', plan('bad', 'allowed', 'unknown')],
    ['unexpected arguments', new ExecutablePlan('bad', [new PlanStep('allowed'), new PlanStep('allowed', { unexpected: 1 })])],
  ] as const) {
    it(`validates the whole plan before any effect (${label})`, async () => {
      const calls: number[] = [];
      const executor = new PlanExecutor({ actions: { allowed: (state: unknown) => { calls.push(1); return new ActionOutcome(state, null); } }, replan: () => null, maxSteps: 2 });
      await expect(executor.call({}, bad)).rejects.toThrow();
      expect(calls).toEqual([]);
    });
  }

  for (const [label, output] of [['text', 'do allowed next'], ['selection', { selected_id: 'allowed' }], ['unknown step', plan('bad', 'unknown')]] as const) {
    it(`invalid policies never default to the first action (${label})`, async () => {
      const calls: number[] = [];
      const allowed = (state: unknown) => {
        calls.push(1);
        return new ActionOutcome(state, { actual: true });
      };
      const result = await new PlanExecutor({ actions: { allowed }, replan: () => output, maxSteps: 3 }).call({}, plan('initial', 'allowed'));
      expect(calls).toEqual([1]);
      expect(result.stopReason).toBe('policy_error');
      expect(result.experiences.length).toBe(1);
      expect(result.policyErrors.length).toBeGreaterThan(0);
    });
  }

  it('records failures as observations and exposes policy errors', async () => {
    const external: string[] = [];
    const effect = (state: { local: string[] }) => {
      external.push('effect');
      state.local.push('mutated');
      throw new Error('after external effect');
    };
    const policy = (request: ReplanRequest) => {
      expect(request.experiences[0]!.status).toBe('error');
      expect(request.state).toEqual({ local: [] });
      throw new TypeError('model callback failed');
    };
    const result = await new PlanExecutor({ actions: { effect }, replan: policy, maxSteps: 2 }).call({ local: [] }, plan('p', 'effect'));
    expect(external).toEqual(['effect']);
    expect((result.experiences[0]!.observation as JsonObject).error_type).toBe('Error');
    expect((result.experiences[0]!.observation as JsonObject).message).toBe('after external effect');
    expect(result.policyErrors).toEqual(['TypeError: model callback failed']);
  });

  it('owns plan generation, checkpoints and observed-only loss', async () => {
    const model = new Planner({ vocabulary: ['goal', 'step', 'hello'], dimensions: 8, generator: tinyConfig() }).eval();
    const inputs = { goal: 'hello', evidence: [{ source_id: 's1', text: 'hello' }] };
    const tokenizer = model.generator!.tokenizer as { batchDecode: unknown };
    tokenizer.batchDecode = () => ['1. Read observation\n2. Revise action', ''];
    const generated = model.propose(inputs, { count: 2 });
    expect(generated.length).toBe(1);
    expect(generated[0]!.text).toBe('1. Read observation\n2. Revise action');
    expect(generated[0]!.origin).toBe('generated');
    expect(generated[0]!.source_ids).toEqual(['s1']);
    const receipt = model.call(inputs);
    expect(receipt.selected_id).toBe(generated[0]!.id);
    const loss = model.loss({ ...inputs, plans: generated }, { candidate_id: generated[0]!.id, outcome: 0.5 });
    loss.backward();
    expect(model.rank.parameters().some((p) => p.grad !== null)).toBe(true);
    await model.savePretrained(join(temp.dir, 'planner'));
    const loaded = await Planner.fromPretrained(join(temp.dir, 'planner'), { localFilesOnly: true });
    const state = loaded.stateDict();
    for (const [name, value] of model.stateDict()) expect(value.equal(state.get(name)!), name).toBe(true);
    expect(loaded.generator!.configuration()).toEqual(model.generator!.configuration());
    tokenizer.batchDecode = () => ['', '', ''];
    expect(model.call(inputs).selected_id).toBeNull();
    expect(inputs).toEqual({ goal: 'hello', evidence: [{ source_id: 's1', text: 'hello' }] });
  });

  it('generation objective reaches the owned generator; nonfinite scores fail', () => {
    const model = new Planner({ vocabulary: ['hello'], dimensions: 8, generator: tinyConfig() });
    model.generationLoss({ goal: 'hello' }, 'answer').backward();
    expect(model.generator!.parameters().some((p) => p.grad !== null && p.grad.abs().sum().item() > 0)).toBe(true);
    const head = model.rank.score.module.modules().filter((module) => (module as { bias?: unknown }).bias).pop() as unknown as { bias: { fill_(value: number): void } };
    noGrad(() => head.bias.fill_(Number.NaN));
    expect(() => model.call({ goal: 'hello', plans: [{ id: 'a', text: 'hello' }] })).toThrow(/nonfinite/);
  });

  async function savedTrajectory(name: string): Promise<[string, JsonObject]> {
    const result = await new PlanExecutor({ actions: { a: (state: unknown) => new ActionOutcome(state, 'ok', true) }, replan: () => null, maxSteps: 1 })
      .call({}, plan('p', 'a'));
    const path = join(temp.dir, name);
    await result.save(path);
    return [path, JSON.parse(readFileSync(path, 'utf8'))];
  }

  for (const [field, value] of [
    ['version', true], ['version', '1.0'], ['stop_reason', 'fabricated'], ['stop_reason', []], ['policy_errors', 'oops'],
    ['policy_errors', [1]], ['policy_errors', ['unexpected']], ['experiences', {}], ['state', 'NaN'], ['extra', 'unknown'],
  ] as const) {
    it(`rejects malformed trajectory topology (${field}=${JSON.stringify(value)})`, async () => {
      const [path, data] = await savedTrajectory(`topology-${field}.json`);
      (data as Record<string, unknown>)[field] = value;
      let text = JSON.stringify(data);
      if (field === 'version' && value === '1.0') text = text.replace('"version":"1.0"', '"version":1.5');
      if (value === 'NaN') text = text.replace('"state":"NaN"', '"state":NaN');
      writeFileSync(path, text);
      await expect(PlanExecutionResult.load(path)).rejects.toThrow();
    });
  }

  for (const [field, value] of [
    ['candidate_id', 123], ['candidate_id', ' '], ['action', { bad: 1 }], ['source_id', []], ['arguments', []],
    ['status', 'invented'], ['observation', 'Infinity'], ['expected_observation', 'NaN'], ['extra', 'unknown'],
  ] as const) {
    it(`rejects malformed experiences (${field}=${JSON.stringify(value)})`, async () => {
      const [path, data] = await savedTrajectory(`experience-${field}.json`);
      ((data.experiences as JsonObject[])[0] as Record<string, unknown>)[field] = value;
      let text = JSON.stringify(data);
      if (value === 'Infinity' || value === 'NaN') text = text.replace(`"${value}"`, value);
      writeFileSync(path, text);
      await expect(PlanExecutionResult.load(path)).rejects.toThrow();
    });
  }

  for (const [label, args] of [['tuple', { value: Object.freeze([1, 2]) }], ['map', { value: new Map([[1, 'coerced']]) }]] as const) {
    it(`non-JSON arguments fail before any effect (${label})`, async () => {
      const calls: unknown[] = [];
      const executor = new PlanExecutor({
        actions: { a: (state: unknown, values: JsonObject) => { calls.push(values); return new ActionOutcome(state, null); } }, replan: () => null, maxSteps: 1,
      });
      await expect(executor.call({}, new ExecutablePlan('p', [new PlanStep('a', args as Record<string, unknown>)]))).rejects.toThrow(/JSON/);
      expect(calls).toEqual([]);
    });
  }

  it('saves trajectories atomically and rejects duplicate fields', async () => {
    const directory = join(temp.dir, 'atomic');
    const path = join(directory, 'trajectory.json');
    const result = new PlanExecutionResult({}, [], 'budget_exhausted');
    await result.save(path);
    const previous = readFileSync(path);
    await expect(new PlanExecutionResult(Number.NaN, [], 'budget_exhausted').save(path)).rejects.toThrow();
    expect(Buffer.compare(readFileSync(path), previous)).toBe(0);
    expect(readdirSync(directory)).toEqual(['trajectory.json']);
    writeFileSync(path, '{"version":1,"version":1}');
    await expect(PlanExecutionResult.load(path)).rejects.toThrow(/duplicate/);
  });

  for (const candidateId of ['', ' ', '\t\n']) {
    it(`rejects invalid candidate IDs before any effect (${JSON.stringify(candidateId)})`, async () => {
      const calls: string[] = [];
      const executor = new PlanExecutor({ actions: { valid: (state: unknown) => { calls.push('effect'); return new ActionOutcome(state, null); } }, replan: () => null, maxSteps: 1 });
      await expect(executor.call({}, plan(candidateId, 'valid'))).rejects.toThrow(/candidate_id/);
      expect(calls).toEqual([]);
    });
  }

  for (const actionId of ['', ' ', '\t\n']) {
    it(`rejects invalid registry IDs (${JSON.stringify(actionId)})`, () => {
      expect(() => new PlanExecutor({ actions: { [actionId]: (state: unknown) => new ActionOutcome(state, null) }, replan: () => null, maxSteps: 1 })).toThrow(/names/);
    });
  }

  it('preserves exact nonempty identities through persistence', async () => {
    const result = await new PlanExecutor({ actions: { ' action ': (state: unknown) => new ActionOutcome(state, { value: 1 }, true) }, replan: () => null, maxSteps: 1 })
      .call({}, plan(' candidate ', ' action '));
    const path = join(temp.dir, 'identity.json');
    await result.save(path);
    const restored = await PlanExecutionResult.load(path);
    expect(restored).toEqual(result);
    expect(restored.experiences[0]!.candidateId).toBe(' candidate ');
    expect(restored.experiences[0]!.action).toBe(' action ');
  });

  it('accepts asynchronous actions and policies', async () => {
    const executor = new PlanExecutor({
      actions: {
        step: async (state: { count: number }) => new ActionOutcome({ count: state.count + 1 }, { count: state.count + 1 }),
        done: async (state: { count: number }, values: JsonObject) => new ActionOutcome({ count: state.count + (values.by as number) }, 'done', true),
      },
      replan: async () => new ExecutablePlan('next', [new PlanStep('done', { by: 10 })]),
      maxSteps: 3,
    });
    const result = await executor.call({ count: 0 }, plan('first', 'step'));
    expect(result.stopReason).toBe('completed');
    expect(result.state).toEqual({ count: 11 });
  });
});
