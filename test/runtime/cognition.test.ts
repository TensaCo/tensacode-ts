/** Port of python/tests/runtime/test_cognition.py (random models with authored classifier biases). */
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { noGrad, tensor } from '../../src/nn/index.js';
import { Investigator, type VerifierPair } from '../../src/tools/investigator.js';
import { Evidence } from '../../src/tools/cognition.js';
import { CognitiveSession } from '../../src/_internal/cognition/session.js';
import { CognitiveState } from '../../src/_internal/cognition/state.js';
import { LearnedEpisodicMemory, type MemoryOwner } from '../../src/_internal/memory/learned.js';
import { SelectionPolicy, type Distribution } from '../../src/_internal/cognition/policy.js';
import { cognitionFingerprint } from '../../src/_internal/cognition/locking.js';
import type { JsonObject } from '../../src/_internal/json.js';
import { cognitionConfig, scratch, tinyConfig, wordLevelTokenizer } from '../tools/helpers.js';
import { NativeConfig } from '../../src/_internal/native/config.js';

const temp = scratch();
afterAll(() => temp.cleanup());

type Classifier = { classifier: { weight: { zero_(): void }; bias: { copy_(value: unknown): void; add_(value: number): void } } };

function classifierOf(tool: Investigator): Classifier['classifier'] {
  return (tool.verifier!.model as unknown as Classifier).classifier;
}

function investigator(): Investigator {
  const model = new Investigator(cognitionConfig()).eval();
  noGrad(() => {
    classifierOf(model).weight.zero_();
    classifierOf(model).bias.copy_(tensor([8, 0, 0]));
  });
  return model;
}

const owner = (tool: Investigator) => tool as unknown as MemoryOwner;
const H = (text: string) => ({ hypotheses: [{ id: 'h', text }] });

describe('cognitive sessions', () => {
  it('revisions re-verify sources and abstain without promoting hypotheses', () => {
    const tool = investigator();
    const session = new CognitiveSession(tool);
    session.ingest([new Evidence('a', 'alpha', 'document')]);
    const first = session.investigate('alpha', H('beta'));
    expect(first.selected_id).toBe('h');
    expect(first.abstained).toBe(false);
    const prior = session.state;
    session.reviseEvidence('a', 'beta');
    expect(session.state.selectionStale).toBe(true);
    expect(session.state.evidence[0]!.text).toBe('alpha');
    noGrad(() => classifierOf(tool).bias.copy_(tensor([0, 8, 0])));
    const second = session.investigate('alpha', H('beta'));
    expect(second.abstained).toBe(true);
    expect(second.selected_id).toBeNull();
    expect(((second.evidence as JsonObject[])[0]!).text).toBe('beta');
    expect(((second.evidence as JsonObject[])[0]!).source_id).toBe('document');
    expect(session.state.assessments.length).toBe(2);
    expect(session.state.observations).toEqual([]);
    expect(prior.evidence[0]!.text).toBe('alpha');
  });

  it('snapshots round-trip; forks and failures are transactional', async () => {
    const tool = investigator();
    const session = new CognitiveSession(tool);
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    session.reviseEvidence('a', 'beta');
    const fork = session.fork();
    fork.ingest([new Evidence('b', 'beta', 'doc2')]);
    expect(session.state.evidence.length).toBe(2);
    const before = session.snapshot();
    expect(() => session.investigate('alpha', { hypotheses: [] })).toThrow();
    expect(session.snapshot()).toEqual(before);
    const path = join(temp.dir, 'session.json');
    await session.save(path);
    const restored = await CognitiveSession.load(path, { investigator: tool });
    expect(restored.snapshot()).toEqual(session.snapshot());
    const broken = session.snapshot();
    broken.active_evidence.a = 'absent';
    expect(() => CognitiveSession.fromSnapshot(broken, { investigator: tool })).toThrow();
  });

  it('authored policy uses strongest-source unknown and retains conflict', () => {
    const policy = new SelectionPolicy();
    expect(policy.accepts([{ support: 0.95, contradiction: 0.02, unknown: 0.03 }, { support: 0.01, contradiction: 0.01, unknown: 0.98 }])).toBe(true);
    expect(policy.accepts([{ support: 0.95, contradiction: 0.02, unknown: 0.03 }, { support: 0.01, contradiction: 0.98, unknown: 0.01 }])).toBe(false);
    expect(policy.accepts([])).toBe(false);
  });

  it('owned encoder memory detects weight changes and rebuilds', () => {
    const tool = investigator();
    const memory = new LearnedEpisodicMemory(owner(tool));
    memory.remember(new Evidence('a', 'alpha', 'doc'), { episodeId: 'past' });
    expect(memory.retrieve('alpha')[0]!.evidence.sourceId).toBe('doc');
    noGrad(() => (tool.rank.encode.module as unknown as { weight: { add_(value: number): void } }).weight.add_(0.5));
    expect(() => memory.retrieve('alpha')).toThrow(/stale/);
    memory.rebuildIndex();
    expect(memory.retrieve('alpha')[0]!.score).toBeCloseTo(1, 6);
  });

  it('active retrieval persists and removed sources do not return', async () => {
    const tool = investigator();
    const memory = new LearnedEpisodicMemory(owner(tool));
    memory.remember(new Evidence('past', 'alpha', 'past-document'), { episodeId: 'past', question: 'alpha', outcome: 'external feedback' });
    const session = new CognitiveSession(tool, { memory });
    const receipt = session.investigate('alpha', H('beta'));
    expect(receipt.selected_id).toBe('h');
    expect(((((receipt.retrieval as JsonObject[])[0]!).evidence as JsonObject).source_id)).toBe('past-document');
    const path = join(temp.dir, 'with-memory.json');
    await session.save(path);
    const restored = await CognitiveSession.load(path, { investigator: tool });
    expect(restored.retrieve('alpha')[0]!.outcome).toBe('external feedback');
    restored.ingest([new Evidence('past', 'alpha', 'past-document')]);
    restored.removeEvidence('past');
    expect(restored.state.selectionStale).toBe(true);
    const next = restored.investigate('alpha', H('beta'));
    expect(next.evidence).toEqual([]);
    expect(next.abstained).toBe(true);
    expect(restored.state.evidence[0]!.text).toBe('alpha');
  });

  it('configured memory and revisions exclude archived sources', () => {
    const tool = investigator();
    const session = new CognitiveSession(tool, { memory: { capacity: 2, top_k: 1 } });
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    session.remember('a', { episodeId: 'past' });
    session.reviseEvidence('a', 'beta');
    const receipt = session.investigate('alpha', H('beta'));
    expect((receipt.evidence as JsonObject[]).map((row) => row.text)).toEqual(['beta']);
    expect(receipt.retrieval).toEqual([]);
    expect(CognitiveSession.fromSnapshot(session.snapshot(), { investigator: tool }).snapshot()).toEqual(session.snapshot());
  });

  it('model weight changes invalidate live and restored selections', () => {
    const tool = investigator();
    const session = new CognitiveSession(tool);
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    session.investigate('alpha', H('beta'));
    const snapshot = session.snapshot();
    noGrad(() => classifierOf(tool).bias.add_(0.5));
    expect(session.state.selectionStale).toBe(true);
    expect(session.state.isStale(session.state.assessments[session.state.assessments.length - 1]!)).toBe(true);
    const restored = CognitiveSession.fromSnapshot(snapshot, { investigator: tool });
    expect(restored.state.selectionStale).toBe(true);
    const revision = restored.state.revision;
    expect(restored.state.revision).toBe(revision);
  });

  it('new episodes retrieve past sources; forked memory is independent', () => {
    const tool = investigator();
    const session = new CognitiveSession(tool, { memory: { capacity: 3 } });
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    session.remember('a');
    const fork = session.fork({ copyMemory: true });
    fork.ingest([new Evidence('b', 'beta', 'doc2')]);
    fork.remember('b');
    expect(session.memory!.memory.size).toBe(1);
    session.newEpisode();
    expect(session.episodeId).toBe('episode-1');
    expect(session.activeEvidence).toEqual([]);
    const receipt = session.investigate('alpha', H('beta'));
    expect((((receipt.retrieval as JsonObject[])[0]!).evidence as JsonObject).id).toBe('a');
    session.removeEvidence('a');
    expect(session.investigate('alpha', H('beta')).abstained).toBe(true);
    expect(CognitiveSession.fromSnapshot(session.snapshot(), { investigator: tool }).episodeId).toBe('episode-1');
  });

  it('truncated verification abstains despite authored support', () => {
    const tool = investigator();
    tool.verifier!.maxTokens = 2;
    const session = new CognitiveSession(tool);
    session.ingest([new Evidence('a', 'alpha alpha alpha alpha', 'doc')]);
    const receipt = session.investigate('alpha', H('beta beta beta beta'));
    expect(((((receipt.candidates as JsonObject[])[0]!).verifications as JsonObject[])[0]!).input_truncated).toBe(true);
    expect(receipt.abstained).toBe(true);
  });

  it('owned generator outputs remain hypotheses', () => {
    const base = investigator().configuration();
    base.generator = {
      foundation_config: NativeConfig.fromDict({
        model_type: 't5', vocab_size: 4, d_model: 8, d_ff: 16, num_layers: 1, num_decoder_layers: 1, num_heads: 2, d_kv: 4,
        decoder_start_token_id: 0, pad_token_id: 0, eos_token_id: 1, dropout_rate: 0.0,
      }).toDict(),
      tokenizer_json: base.verifier_tokenizer_json!, tokenizer_special_tokens: base.verifier_tokenizer_special_tokens!,
      workspace: { slots: 2, steps: 1 }, max_new_tokens: 2, max_input_tokens: 32,
    };
    const tool = new Investigator(base).eval();
    (tool.generator!.tokenizer as { batchDecode: unknown }).batchDecode = () => ['beta'];
    const session = new CognitiveSession(tool);
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    const receipt = session.investigate('alpha', { count: 1 });
    expect(session.state.hypotheses[0]!.origin).toBe('generated');
    expect(session.state.hypotheses[0]!.text).toBe('beta');
    expect(session.state.evidence.map((row) => row.text)).toEqual(['alpha']);
    expect(((receipt.candidates as JsonObject[])[0]!).epistemic_status).toBe('hypothesis');
    expect(session.state.observations).toEqual([]);
  });

  it('public retrieval excludes revised and removed sources', () => {
    const tool = investigator();
    const session = new CognitiveSession(tool, { memory: { capacity: 3 } });
    session.ingest([new Evidence('a', 'alpha', 'doc'), new Evidence('b', 'beta', 'doc2')]);
    session.remember('a');
    session.remember('b');
    session.reviseEvidence('a', 'beta');
    session.removeEvidence('b');
    expect(session.retrieve('alpha').map((hit) => hit.evidence.text)).toEqual(['beta']);
    expect(session.memory!.retrieve('alpha').every((hit) => hit.evidence.id !== 'a')).toBe(true);
    expect(session.memory!.retrieve('alpha').length).toBe(2); // Removed b remains explicitly archived.
  });

  it('joint support preserves each source contradiction veto', () => {
    const policy = new SelectionPolicy();
    const unknown = { support: 0.01, contradiction: 0.01, unknown: 0.98 };
    const supported = { support: 0.95, contradiction: 0.02, unknown: 0.03 };
    const conflict = { support: 0.01, contradiction: 0.98, unknown: 0.01 };
    expect(policy.accepts([unknown, unknown])).toBe(false);
    expect(policy.accepts([unknown, unknown], { jointDistribution: supported })).toBe(true);
    expect(policy.accepts([unknown, conflict], { jointDistribution: supported })).toBe(false);
    expect(policy.accepts([supported], { jointDistribution: unknown })).toBe(false);
    expect(policy.accepts([], { jointDistribution: supported })).toBe(false);
  });

  it('joint receipts cover sources, revisions and artifacts', async () => {
    const base = investigator();
    const settings = base.configuration();
    settings.verification_scope = 'joint';
    const tool = new Investigator(settings).eval();
    tool.loadStateDict(base.stateDict());
    const session = new CognitiveSession(tool);
    session.ingest([new Evidence('a', 'alpha', 'doc-a'), new Evidence('b', 'beta', 'doc-b')]);
    const first = session.investigate('alpha', H('beta'));
    expect(first.abstained).toBe(false);
    const joint = ((first.candidates as JsonObject[])[0]!).joint_verification as JsonObject;
    expect(joint.evidence_ids).toEqual(['a', 'b']);
    expect(joint.source_ids).toEqual(['doc-a', 'doc-b']);
    expect(joint.input_truncated).toBe(false);
    session.reviseEvidence('a', 'alpha beta');
    expect(session.state.selectionStale).toBe(true);
    const second = session.investigate('alpha', H('beta'));
    expect(((((second.candidates as JsonObject[])[0]!).joint_verification as JsonObject).evidence_ids as string[])[0]).not.toBe('a');
    expect(joint.evidence_ids).toEqual(['a', 'b']);
    await tool.savePretrained(join(temp.dir, 'joint'));
    const restored = await Investigator.fromPretrained(join(temp.dir, 'joint'));
    expect(restored.configuration().verification_scope).toBe('joint');
    expect(new CognitiveSession(restored).modelIdentity()).not.toBe(new CognitiveSession(base).modelIdentity());
    expect(CognitiveSession.fromSnapshot(session.snapshot(), { investigator: restored }).snapshot()).toEqual(session.snapshot());
  });

  it('joint truncation abstains even when each source fits', () => {
    const base = investigator();
    const settings = base.configuration();
    settings.verification_scope = 'joint';
    const tool = new Investigator(settings).eval();
    tool.loadStateDict(base.stateDict());
    tool.verifier!.maxTokens = 4;
    const session = new CognitiveSession(tool);
    session.ingest([new Evidence('a', 'alpha alpha', 'doc-a'), new Evidence('b', 'beta beta', 'doc-b')]);
    const receipt = session.investigate('alpha', H('beta'));
    const candidate = (receipt.candidates as JsonObject[])[0]!;
    expect((candidate.verifications as JsonObject[]).some((row) => row.input_truncated)).toBe(false);
    expect((candidate.joint_verification as JsonObject).input_truncated).toBe(true);
    expect(receipt.abstained).toBe(true);
  });

  it('rejects unknown verification scopes', () => {
    const settings = investigator().configuration();
    settings.verification_scope = 'automatic';
    expect(() => new Investigator(settings)).toThrow(/verification_scope/);
  });

  it('joint selection uses combined support, not individual support', () => {
    const settings = investigator().configuration();
    settings.verification_scope = 'joint';
    const tool = new Investigator(settings);
    // Authored classifier outputs isolate evidence aggregation, not learned inference.
    tool.verifier!.forward = (pairs: VerifierPair[]) => tensor(pairs.map((row) => (row.premise.includes('\n\n') ? [8, 0, 0] : [0, 0, 8])));
    const session = new CognitiveSession(tool);
    session.ingest([new Evidence('a', 'alpha', 'one'), new Evidence('b', 'beta', 'two')]);
    const result = session.investigate('alpha', H('beta'));
    expect(result.selected_id).toBe('h');
    expect((((result.candidates as JsonObject[])[0]!).verifications as JsonObject[]).every((row) => ((row.distribution as JsonObject).support as number) < 0.01)).toBe(true);
    session.removeEvidence('b');
    expect(session.investigate('alpha', H('beta')).abstained).toBe(true);
  });

  it('joint coverage must match exact order and ids', () => {
    const policy = new SelectionPolicy();
    const good = { support: 0.95, contradiction: 0.02, unknown: 0.03 };
    const checks = [{ source_id: 'a', distribution: good }, { source_id: 'b', distribution: good }];
    for (const ids of [['b', 'a'], ['a'], ['a', 'a']]) {
      expect(() => policy.acceptsVerification({ verifications: checks, joint_verification: { scope: 'joint', source_ids: ids, distribution: good } }, ['a', 'b'], { scope: 'joint' }))
        .toThrow(/joint verification sources/);
    }
  });

  const cases: [string, string, unknown][] = [
    ['joint', 'input_truncated', 'missing'], ['joint', 'input_truncated', null], ['joint', 'input_truncated', 0], ['joint', 'input_truncated', 'false'],
    ['joint', 'token_count', 'missing'], ['joint', 'token_count', null], ['joint', 'token_count', 0], ['joint', 'token_count', true],
    ['joint', 'token_count', -1], ['joint', 'token_count', 1.5], ['joint', 'max_tokens', 'missing'], ['joint', 'max_tokens', null],
    ['joint', 'max_tokens', 0], ['joint', 'max_tokens', true], ['joint', 'max_tokens', -1], ['joint', 'max_tokens', 1.5],
    ['joint', 'max_tokens', 2], ['joint', 'token_count', 3000], ['joint', 'input_truncated', true],
    ['source', 'input_truncated', 'missing'], ['source', 'input_truncated', null], ['source', 'input_truncated', 0], ['source', 'input_truncated', 'false'],
  ];
  for (const [where, field, value] of cases) {
    it(`joint screening requires complete input metadata (${where}.${field}=${JSON.stringify(value)})`, () => {
      const good: Distribution = { support: 0.95, contradiction: 0.02, unknown: 0.03 };
      const source: Record<string, unknown> = { source_id: 'a', distribution: good, input_truncated: false };
      const joint: Record<string, unknown> = { scope: 'joint', source_ids: ['a'], distribution: good, input_truncated: false, token_count: 3, max_tokens: 512 };
      const receipt = { verifications: [source], joint_verification: joint };
      const policy = new SelectionPolicy();
      expect(policy.acceptsVerification(receipt, ['a'], { scope: 'joint' })).toBe(true);
      const target = where === 'joint' ? joint : source;
      if (value === 'missing') delete target[field];
      else target[field] = value;
      expect(() => policy.acceptsVerification(receipt, ['a'], { scope: 'joint' })).toThrow(/coverage metadata/);
    });
  }

  it('fresh sessions reuse the weight hash but invalidate on updates', () => {
    const tool = investigator();
    const sessions = [new CognitiveSession(tool), new CognitiveSession(tool)];
    const cache = cognitionFingerprint(tool);
    const identity = sessions[0]!.modelIdentity();
    const first = cache.computations;
    expect(first).toBeGreaterThan(0);
    expect(sessions[1]!.modelIdentity()).toBe(identity);
    expect(cache.computations).toBe(first);
    noGrad(() => tool.parameters()[0]!.add_(1));
    const changed = sessions[1]!.modelIdentity();
    expect(changed).not.toBe(identity);
    expect(cache.computations).toBeGreaterThan(first);
    const afterUpdate = cache.computations;
    expect(sessions[0]!.modelIdentity()).toBe(changed);
    expect(cache.computations).toBe(afterUpdate);
    sessions[0]!.invalidateFingerprint();
    expect(sessions[1]!.modelIdentity()).toBe(changed);
    expect(cache.computations).toBeGreaterThan(afterUpdate);
  });

  for (const recallFirst of [false, true]) {
    it(`corrects remembered evidence across episodes and reloads (recall first=${recallFirst})`, async () => {
      const tool = investigator();
      const session = new CognitiveSession(tool, { memory: { capacity: 3 } });
      session.ingest([new Evidence('a', 'alpha', 'document')]);
      session.remember('a', { question: 'original question', outcome: 'supplied outcome' });
      session.newEpisode();
      if (recallFirst) session.investigate('alpha', H('beta'));
      session.reviseEvidence('a', 'beta');
      const corrected = session.activeEvidence[0]!;
      expect(corrected.sourceId).toBe('document');
      expect(corrected.text).toBe('beta');
      expect(session.retrieve('beta').map((hit) => hit.evidence)).toEqual([corrected]);
      expect(session.memory!.retrieve('alpha').map((hit) => hit.evidence)).toEqual([corrected]);
      expect(session.memory!.snapshot().records[0]!.question).toBe('original question');
      expect(session.memory!.snapshot().records[0]!.outcome).toBe('');
      expect(session.state.evidence[0]!.equals(new Evidence('a', 'alpha', 'document'))).toBe(true);
      session.newEpisode();
      const path = join(temp.dir, `corrected-${recallFirst}.json`);
      await session.save(path);
      const restored = await CognitiveSession.load(path, { investigator: tool });
      restored.reviseEvidence('a', 'alpha beta');
      expect(restored.activeEvidence[0]!.text).toBe('alpha beta');
      restored.newEpisode();
      const receipt = restored.investigate('alpha', H('beta'));
      expect((receipt.evidence as JsonObject[]).map((row) => row.text)).toEqual(['alpha beta']);
      expect(() => restored.reviseEvidence(corrected.id, 'stale')).toThrow();
      restored.removeEvidence('a');
      expect(() => restored.reviseEvidence('a', 'removed')).toThrow();
    });
  }

  it('a recalled revision embedding failure leaves state and memory unchanged', () => {
    const tool = investigator();
    const session = new CognitiveSession(tool, { memory: { capacity: 2 } });
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    session.remember('a');
    session.newEpisode();
    const before = session.snapshot();
    const original = LearnedEpisodicMemory.prototype.embed;
    LearnedEpisodicMemory.prototype.embed = () => {
      throw new Error('encoder failure');
    };
    try {
      expect(() => session.reviseEvidence('a', 'beta')).toThrow('encoder failure');
    } finally {
      LearnedEpisodicMemory.prototype.embed = original;
    }
    expect(session.snapshot()).toEqual(before);
  });

  it('revises external memory and literal @ ids', () => {
    const tool = investigator();
    const memory = new LearnedEpisodicMemory(owner(tool));
    memory.remember(new Evidence('literal@12', 'alpha', 'doc'), { episodeId: 'external' });
    const session = new CognitiveSession(tool, { memory });
    session.reviseEvidence('literal@12', 'beta');
    expect(session.activeEvidence[0]!.sourceId).toBe('doc');
    expect(session.state.evidence.length).toBe(2);
    expect(session.state.evidence[0]!.id).toBe('literal@12');
    expect(session.retrieve('beta')[0]!.evidence.text).toBe('beta');
  });

  it('revision id collisions and forged lineage fail transactionally', () => {
    const session = new CognitiveSession(investigator());
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    const collision = `a@${session.state.revision + 2}`;
    session.ingest([new Evidence(collision, 'beta', 'other')]);
    const before = session.snapshot();
    expect(() => session.reviseEvidence('a', 'beta')).toThrow(/conflict/);
    expect(session.snapshot()).toEqual(before);
    const corrupted = session.snapshot();
    corrupted.evidence_lineage.a!.push(collision);
    expect(() => CognitiveSession.fromSnapshot(corrupted, { investigator: session.investigator })).toThrow(/lineage/);
  });

  for (const external of [false, true]) {
    it(`snapshots reject conflicting memory evidence (external=${external})`, () => {
      const tool = investigator();
      const session = new CognitiveSession(tool, { memory: { capacity: 2 } });
      session.ingest([new Evidence('a', 'alpha', 'doc')]);
      session.remember('a');
      const snapshot = session.snapshot();
      let memory: LearnedEpisodicMemory | null = null;
      if (external) {
        snapshot.memory = null;
        memory = new LearnedEpisodicMemory(owner(tool));
        memory.remember(new Evidence('a', 'beta', 'doc'), { episodeId: 'external' });
      } else {
        snapshot.memory!.records[0]!.evidence.text = 'beta';
      }
      expect(() => CognitiveSession.fromSnapshot(snapshot, { investigator: tool, memory })).toThrow(/conflict/);
    });
  }

  it('the constructor rejects conflicting external memory', () => {
    const tool = investigator();
    const memory = new LearnedEpisodicMemory(owner(tool));
    memory.remember(new Evidence('a', 'beta', 'doc'), { episodeId: 'past' });
    const state = new CognitiveState().addEvidence([new Evidence('a', 'alpha', 'doc')]);
    expect(() => new CognitiveSession(tool, { state, memory })).toThrow(/conflict/);
  });

  it('ingest rejects conflicting external memory transactionally', () => {
    const tool = investigator();
    const memory = new LearnedEpisodicMemory(owner(tool));
    memory.remember(new Evidence('a', 'alpha', 'doc'), { episodeId: 'past' });
    const session = new CognitiveSession(tool, { memory });
    const before = session.snapshot();
    expect(() => session.ingest([new Evidence('b', 'beta', 'other'), new Evidence('a', 'beta', 'doc')])).toThrow(/conflict/);
    expect(session.snapshot()).toEqual(before);
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    expect(session.activeEvidence.map((row) => row.toRecord())).toEqual([{ id: 'a', text: 'alpha', source_id: 'doc' }]);
  });

  it('dialogue conditions retrieval without becoming evidence', () => {
    const tool = investigator();
    const session = new CognitiveSession(tool, { memory: { capacity: 3 } });
    session.ingest([new Evidence('a', 'alpha', 'doc')]);
    session.remember('a');
    session.newEpisode();
    const queries: string[] = [];
    const memory = session.memory!;
    const retrieve = memory.retrieve.bind(memory);
    memory.retrieve = (question, options) => {
      queries.push(question);
      return retrieve(question, options);
    };
    const dialogue = [{ role: 'user' as const, text: 'alpha earlier' }, { role: 'assistant' as const, text: 'unverified beta' }];
    const receipt = session.investigate('what caused that?', { ...H('alpha'), conversationContext: dialogue });
    expect(queries[0]).toContain('alpha earlier');
    expect(queries[0]).toContain('what caused that?');
    expect(receipt.retrieval_query).toBe(queries[0]);
    expect((receipt.evidence as JsonObject[]).map((row) => row.text)).toEqual(['alpha']);
    expect(session.state.evidence.map((row) => row.text)).toEqual(['alpha']);
    tool.rank.config.max_tokens = 2;
    const before = session.snapshot();
    expect(() => session.investigate('what caused that?', { ...H('alpha'), conversationContext: dialogue })).toThrow(/retrieval token budget/);
    expect(session.snapshot()).toEqual(before);
  });
});

describe('helpers used by the cognition port', () => {
  it('builds tokenizers and tiny configurations', () => {
    expect(JSON.parse(wordLevelTokenizer({ a: 0 }, 'a')).model.type).toBe('WordLevel');
    expect(tinyConfig().max_turns).toBe(2);
  });
});
