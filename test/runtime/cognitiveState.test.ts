/** Port of python/tests/runtime/test_cognitive_state.py. */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import {
  Assessment, CognitiveState, Evidence, Goal, Hypothesis, Observation, Plan,
} from '../../src/_internal/cognition/state.js';
import { ValueError } from '../../src/errors.js';
import { scratch } from '../tools/helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

function base(): CognitiveState {
  return new CognitiveState()
    .addEvidence([new Evidence('e', 'source words', 'document')])
    .addHypotheses([new Hypothesis('h', 'interpretation', { modelProvenance: 'weights-a' })]);
}

describe('CognitiveState', () => {
  it('revision preserves sources and contradictions and invalidates selection', () => {
    const state = base().assess([new Assessment('e', 'h', { entailed: 0.2, contradicted: 0.7, unknown: 0.1 }, 'judge-a')]).select(['h']);
    const changed = state.addHypotheses([new Hypothesis('h', 'revised', { modelProvenance: 'weights-b' })]);
    expect(changed.evidence[0]!.text).toBe('source words');
    expect(changed.assessments[0]!.scores.contradicted).toBe(0.7);
    expect(changed.isStale(changed.assessments[0]!)).toBe(true);
    expect(changed.selectionStale).toBe(true);
    expect(state.hypotheses[0]!.text).toBe('interpretation');
    expect(state.isStale(state.assessments[0]!)).toBe(false);
    expect(changed.observations).toEqual([]);
  });

  it('failed batches are atomic and states independent', () => {
    const state = base();
    expect(() => state.addEvidence([new Evidence('ok', 'new', 'src'), new Evidence('e', 'overwrite', 'src')])).toThrow(ValueError);
    expect(() => state.assess([new Assessment('e', 'h', { yes: 0.5 }, 'judge'), new Assessment('missing', 'h', { yes: 1 }, 'judge')])).toThrow(ValueError);
    expect(() => state.assess([new Assessment('e', 'h', { yes: Number.NaN }, 'judge')])).toThrow(ValueError);
    expect(state.evidence.length).toBe(1);
    expect(state.assessments).toEqual([]);
    expect(new CognitiveState().evidence).toEqual([]);
  });

  it('is bounded and round-trips strictly', async () => {
    const state = base().addGoals([new Goal('g', 'investigate', 'user')]).addPlans([new Plan('p', ['inspect'], ['new evidence'])])
      .observe([new Observation('o', 'actual response', 'receipt-1')]);
    const path = join(temp.dir, 'session.json');
    await state.save(path);
    const loaded = await CognitiveState.load(path);
    expect(loaded.toDict()).toEqual(state.toDict());
    expect(() => new CognitiveState({ maxRecords: 1 }).addEvidence([new Evidence('a', 'a', 's'), new Evidence('b', 'b', 's')])).toThrow(ValueError);
    const data = JSON.parse(readFileSync(path, 'utf8'));
    data.unexpected = true;
    writeFileSync(path, JSON.stringify(data));
    await expect(CognitiveState.load(path)).rejects.toThrow(ValueError);
    writeFileSync(path, '{"schema_version": 1, "schema_version": 1}');
    await expect(CognitiveState.load(path)).rejects.toThrow(/duplicate/);
  });

  it('captured scores cannot be mutated', () => {
    const scores = { unknown: 1.0 };
    const state = base().assess([new Assessment('e', 'h', scores, 'judge')]);
    scores.unknown = 0;
    expect(state.assessments[0]!.scores.unknown).toBe(1);
    expect(() => { (state.assessments[0]!.scores as Record<string, number>).unknown = 0; }).toThrow(TypeError);
  });

  it('round-trips history and rejects future revisions', async () => {
    let state = base().assess([new Assessment('e', 'h', { supports: 1, conflicts: -1 }, 'judge')]).select(['h']);
    state = state.addEvidence([new Evidence('e2', 'contrary source', 'second')]);
    const path = join(temp.dir, 'state.json');
    await state.save(path);
    const loaded = await CognitiveState.load(path);
    expect(loaded.isStale(loaded.assessments[0]!)).toBe(true);
    expect(loaded.selectionStale).toBe(true);
    expect(loaded.evidence[1]!.sourceId).toBe('second');
    const data = loaded.toDict();
    ((data.assessments as Record<string, unknown>[])[0]!).revision = 999;
    expect(() => CognitiveState.fromDict(data)).toThrow(ValueError);
    expect(() => loaded.select(['missing'])).toThrow(ValueError);
  });

  it('capacity rejects an assessment without changing state', () => {
    const state = new CognitiveState({ maxRecords: 2 }).addEvidence([new Evidence('e', 'E', 'source')])
      .addHypotheses([new Hypothesis('h', 'H', { modelProvenance: 'model' })]);
    expect(() => state.assess([new Assessment('e', 'h', { unknown: 1 }, 'judge')])).toThrow(ValueError);
    expect(state.assessments).toEqual([]);
  });

  it('reassessment marks previous model scores stale without content change', () => {
    const state = base().assess([new Assessment('e', 'h', { unknown: 1 }, 'model-v1')]);
    const updated = state.assess([new Assessment('e', 'h', { support: 1 }, 'model-v2')]);
    expect(updated.isStale(updated.assessments[0]!)).toBe(true);
    expect(updated.isStale(updated.assessments[1]!)).toBe(false);
  });

  it('hypotheses require stated provenance', () => {
    expect(() => new Hypothesis('h', 'interpretation', undefined as never)).toThrow(TypeError);
    expect(() => new Hypothesis('h', 'interpretation', {} as never)).toThrow(TypeError);
    expect(() => new Hypothesis('h', 'interpretation', { modelProvenance: '' })).toThrow(/model_provenance/);
    expect(new Hypothesis('h', 'text', { origin: 'supplied', modelProvenance: 'caller-supplied' }).origin).toBe('supplied');
  });
});
