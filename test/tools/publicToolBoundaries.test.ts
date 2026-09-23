/**
 * Port of python/tests/models/test_public_tool_boundaries.py and the tools/actions
 * part of python/tests/integration/test_text_namespace.py.
 */
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Investigator, InvestigationSession, Evidence as InvestigationEvidence } from '../../src/tools/investigator.js';
import { Planner, ExecutablePlan, PlanStep } from '../../src/tools/planner.js';
import { Evidence } from '../../src/tools/cognition.js';
import { ActionOutcome, ActionRequest, actionLoop } from '../../src/tools/actions.js';
import { investigatorConfig, scratch } from './helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

describe('public tool boundaries', () => {
  it('supports the public cognitive session lifecycle', async () => {
    expect(InvestigationEvidence).toBe(Evidence);
    const tool = new Investigator({ vocabulary: ['alpha', 'beta'], dimensions: 8, slots: 2, steps: 1 });
    const session = tool.newCognitiveSession({ policy: { min_support: 0.8 }, memory: { capacity: 8, top_k: 2 }, maxRecords: 8 });
    expect(session).toBeInstanceOf(InvestigationSession);
    const other = tool.newCognitiveSession({ memory: { capacity: 8 } });
    session.ingest([new Evidence('a', 'alpha', 'source')]);
    session.remember('a');
    session.reviseEvidence('a', 'beta');
    const hits = session.retrieve('beta');
    expect(hits[0]!.evidence.text).toBe('beta');
    expect(hits[0]!.evidence.sourceId).toBe('source');
    expect(other.activeEvidence).toEqual([]);
    expect(other.retrieve('beta')).toEqual([]);
    const path = join(temp.dir, 'session.json');
    await session.save(path);
    const restored = await tool.loadCognitiveSession(path);
    expect(restored.snapshot()).toEqual(session.snapshot());
    expect(restored.investigator).toBe(tool);
    restored.removeEvidence('a');
    expect(session.activeEvidence.length).toBeGreaterThan(0);
    expect(restored.activeEvidence).toEqual([]);
    expect(tool.newSession().history).toEqual([]);
  });

  for (const [label, options] of [
    ['callable policy', { policy: (value: unknown) => value }], ['opaque memory', { memory: new Date() }],
    ['obsolete memory field', { memory: { obsolete: true } }], ['obsolete policy field', { policy: { obsolete: true } }],
    ['zero records', { maxRecords: 0 }],
  ] as const) {
    it(`cognitive factory rejects non-JSON or obsolete options (${label})`, () => {
      const tool = new Investigator({ vocabulary: ['alpha'], dimensions: 8, slots: 2, steps: 1 });
      expect(() => tool.newCognitiveSession(options as never)).toThrow();
    });
  }

  it('plan factory validates before effects and bounds receipts', async () => {
    const tool = new Planner({ vocabulary: ['alpha'], dimensions: 8, slots: 2, steps: 1 });
    const calls: number[] = [];
    const act = (state: number) => {
      calls.push(state);
      return new ActionOutcome(state + 1, { count: state + 1 });
    };
    const plan = new ExecutablePlan('chosen', [new PlanStep('act')]);
    const executor = tool.newExecutor({ actions: { act }, replan: () => plan, maxSteps: 2 });
    expect(calls).toEqual([]);
    await expect(executor.call(0, new ExecutablePlan('invalid', [new PlanStep('act'), new PlanStep('missing')]))).rejects.toThrow();
    expect(calls).toEqual([]);
    const result = await executor.call(0, plan);
    expect(result.state).toBe(2);
    expect(result.experiences.length).toBe(2);
    expect(result.stopReason).toBe('budget_exhausted');
  });

  it('action loop factory constructs without running callbacks', async () => {
    const calls: number[] = [];
    const loop = actionLoop({
      chooser: (request) => {
        expect(request).toBeInstanceOf(ActionRequest);
        return 'increment';
      },
      actions: {
        increment: (state: number) => {
          calls.push(state);
          return new ActionOutcome(state + 1, { previous: state });
        },
      },
      maxSteps: 2,
    });
    expect(calls).toEqual([]);
    const result = await loop.call(0);
    expect(result.state).toBe(2);
    expect(result.receipts.map((receipt) => receipt.effect)).toEqual([{ previous: 0 }, { previous: 1 }]);
  });

  it('public cognitive investigation preserves source evidence', async () => {
    const tool = new Investigator(investigatorConfig()).eval();
    const session = tool.newCognitiveSession({ memory: { capacity: 8 }, maxRecords: 16 });
    session.ingest([new Evidence('original', 'hello', 'document')]);
    const first = session.investigate('hello', { hypotheses: [{ id: 'candidate', text: 'world' }] });
    expect(first.evidence).toEqual([{ id: 'original', text: 'hello', source_id: 'document' }]);
    expect(session.state.hypotheses[0]!.text).toBe('world');
    expect(session.state.observations).toEqual([]);
    session.remember('original');
    session.reviseEvidence('original', 'world');
    const second = session.investigate('hello', { hypotheses: [{ id: 'candidate', text: 'world' }] });
    expect((second.evidence as { text: string; source_id: string }[])[0]!.text).toBe('world');
    expect((second.evidence as { text: string; source_id: string }[])[0]!.source_id).toBe('document');
    const path = join(temp.dir, 'cognitive.json');
    await session.save(path);
    expect((await tool.loadCognitiveSession(path)).snapshot()).toEqual(session.snapshot());
  });
});

describe('tools namespace composition (test_text_namespace, tools/actions part)', () => {
  it('imports cognition and action records without loading model modules', async () => {
    const cognition = await import('../../src/tools/cognition.js');
    const actions = await import('../../src/tools/actions.js');
    const evidence = new cognition.Evidence('e', 'text', 'source');
    expect(evidence.toRecord()).toEqual({ id: 'e', text: 'text', source_id: 'source' });
    const loop = actions.actionLoop({ chooser: () => 'noop', actions: { noop: (state: string) => new actions.ActionOutcome(state, null, true) }, maxSteps: 1 });
    expect((await loop.call('ready')).stopReason).toBe('completed');
  });
});
