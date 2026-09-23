/** Port of python/tests/models/test_cognitive_chatbot.py (authored tiny fixtures isolate cognitive plumbing). */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { noGrad, tensor } from '../../src/nn/index.js';
import { Chatbot } from '../../src/tools/chatbot.js';
import type { Investigator, SourceText } from '../../src/tools/investigator.js';
import { conversationBlock } from '../../src/_internal/conversation.js';
import type { JsonObject } from '../../src/_internal/json.js';
import { investigatorConfig, scratch, tinyConfig } from './helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

const ABSTAIN = 'I do not have enough supported evidence to answer.';

function config(): JsonObject {
  const result = tinyConfig();
  result.max_input_tokens = 256;
  const investigator = investigatorConfig();
  (investigator.generator as JsonObject).max_input_tokens = 256;
  result.cognition = { investigator, proposal_count: 1 };
  return result;
}

type Classifier = { classifier: { weight: { zero_(): void }; bias: { copy_(value: unknown): void } } };

function supportVerifier(investigator: Investigator): void {
  const model = investigator.verifier!.model as unknown as Classifier;
  noGrad(() => {
    model.classifier.weight.zero_();
    model.classifier.bias.copy_(tensor([-5, -5, 5]));
  });
}

function prepared(options: { memory?: boolean } = {}): Chatbot {
  const settings = config();
  if (options.memory) (settings.cognition as JsonObject).memory = { capacity: 8, top_k: 2 };
  const model = new Chatbot(settings).eval();
  model.investigator!.propose = () => [{ id: 'h1', text: 'hello', origin: 'generated', generated_by: 'authored-test-output' }] as never;
  supportVerifier(model.investigator!);
  model.generateBatch = (inputs) => inputs.map(() => 'realized answer');
  return model;
}

const INPUT = { question: 'hello?', evidence: [{ id: 'e1', source_id: 'source-one', text: 'hello world' }] };
const input = (): JsonObject => structuredClone(INPUT) as JsonObject;

function snapshot(model: Chatbot): JsonObject {
  return model.defaultSession.cognition!.snapshot() as unknown as JsonObject;
}

describe('cognitive Chatbot', () => {
  it('owns complete cognitive parameters and round-trips locally', async () => {
    const model = new Chatbot(config()).eval();
    expect(model.investigator!.generator).not.toBeNull();
    expect(model.investigator!.verifier).not.toBeNull();
    const ids = new Set(model.parameters());
    expect(model.investigator!.parameters().every((p) => ids.has(p))).toBe(true);
    expect(model.capabilities.persistent_cognitive_state).toBe(true);
    expect(new Chatbot(tinyConfig()).capabilities.source_verification).toBe(false);
    await model.savePretrained(join(temp.dir, 'model'));
    const restored = await Chatbot.fromPretrained(join(temp.dir, 'model'));
    const state = restored.stateDict();
    for (const [key, value] of model.stateDict()) expect(value.equal(state.get(key)!), key).toBe(true);
    expect(restored.configuration()).toEqual(model.configuration());
    expect(Object.keys(model.operationBindings()).some((key) => key.startsWith('investigator.'))).toBe(true);
    expect(model.investigator!.generator!.investigator).toBeNull();
  });

  it('rejects recursive or incomplete configurations', () => {
    const recursive = config();
    (((recursive.cognition as JsonObject).investigator as JsonObject).generator as JsonObject).cognition = (config().cognition as JsonObject);
    expect(() => new Chatbot(recursive)).toThrow(/Recursive/);
    const incomplete = config();
    delete ((incomplete.cognition as JsonObject).investigator as JsonObject).generator;
    expect(() => new Chatbot(incomplete)).toThrow(/owned proposal/);
  });

  it('keeps sources, state and sessions independent', async () => {
    const model = prepared();
    const first = model.newSession();
    const second = model.newSession();
    expect(first.call(input())).toBe('realized answer');
    const recorded = first.cognition!.snapshot();
    expect(JSON.stringify(recorded)).not.toContain('realized answer');
    expect(recorded.state.evidence as JsonObject[]).not.toContainEqual(expect.objectContaining({ text: 'hello?' }));
    expect(second.cognition!.snapshot()).not.toEqual(recorded);
    const path = join(temp.dir, 'independent.json');
    await first.save(path);
    await second.load(path);
    expect(second.cognition!.snapshot()).toEqual(recorded);
    expect(second.history).toEqual(first.history);
  });

  it('never treats the question as evidence; abstention is authored', () => {
    const model = prepared();
    expect(model.call('hello?')).toBe(ABSTAIN);
    expect(model.lastResult!.abstention_enforced).toBe(true);
    expect(((snapshot(model).state as JsonObject).evidence as unknown[]).length).toBe(0);
  });

  it('a failed decoder commits neither evidence nor interpretations', () => {
    const model = prepared();
    const before = snapshot(model);
    model.generateBatch = () => {
      throw new Error('decoder failed');
    };
    expect(() => model.call(input())).toThrow('decoder failed');
    expect(snapshot(model)).toEqual(before);
    expect(model.history).toEqual([]);
  });

  it('realization receives structured interpretations and source evidence', () => {
    const model = prepared();
    const prompts: string[] = [];
    model.generateBatch = (inputs) => {
      prompts.push(...inputs);
      return ['answer'];
    };
    model.call(input());
    expect(prompts[0]).toContain('source-one');
    expect(prompts[0]).toContain('Selected hypothesis');
    expect(prompts[0]).toContain('hello world');
  });

  it('source revision retains the original and never ingests the assistant', () => {
    const model = prepared();
    model.call(input());
    model.call({ question: 'hello?', revisions: [{ evidence_id: 'e1', text: 'world changed', source_id: 'corrected-source' }] });
    const recorded = snapshot(model);
    const texts = ((recorded.state as JsonObject).evidence as JsonObject[]).map((row) => row.text);
    expect(texts).toContain('hello world');
    expect(texts).toContain('world changed');
    expect(texts).not.toContain('realized answer');
    expect((recorded.active_evidence as JsonObject).e1).not.toBe('e1');
  });

  it('invalid cognitive session loads are transactional', async () => {
    const model = prepared();
    model.call(input());
    const before = snapshot(model);
    const history = model.history;
    const path = join(temp.dir, 'invalid.json');
    await model.saveSession(path);
    const value = JSON.parse(readFileSync(path, 'utf8'));
    value.cognition.active_evidence.e1 = 'missing';
    writeFileSync(path, JSON.stringify(value));
    await expect(model.loadSession(path)).rejects.toThrow();
    expect(snapshot(model)).toEqual(before);
    expect(model.history).toEqual(history);
  });

  it('language and investigator objectives own distinct supervision', () => {
    const model = new Chatbot(config());
    model.lossBatch(['hello'], ['world']).backward();
    expect(model.foundation.parameters().some((p) => p.grad !== null)).toBe(true);
    expect(model.investigator!.parameters().every((p) => p.grad === null)).toBe(true);
    model.investigator!.verificationLoss([{ premise: 'hello', hypothesis: 'world' }], ['unknown']).backward();
    expect(model.investigator!.verifier!.parameters().some((p) => p.grad !== null)).toBe(true);
  });

  it('unsupported realization enforces abstention after a valid selection', () => {
    const model = prepared();
    const verifier = model.investigator!.verifier!;
    const original = verifier.verify.bind(verifier);
    verifier.verify = (text: string, evidence: readonly SourceText[]) => (text === 'realized answer'
      ? evidence.map((row) => ({ source_id: row.source_id, distribution: { support: 0.01, contradiction: 0.98, unknown: 0.01 } }))
      : original(text, evidence));
    const answer = model.call(input());
    expect((model.lastResult!.cognition as JsonObject).abstained).toBe(false);
    expect(model.lastResult!.abstention_enforced).toBe(true);
    expect(answer).toBe(ABSTAIN);
    expect(model.lastResult!.response_proposal).toEqual({ text: 'realized answer', origin: 'model_generation', epistemic_status: 'unverified_proposal' });
    expect(model.history.every((row) => row.text !== 'realized answer')).toBe(true);
    expect(model.cognitiveState!.evidence.every((row) => row.text !== 'realized answer')).toBe(true);
  });

  it('realization budget prioritizes sources and reports omissions', () => {
    const model = prepared();
    model.config.max_input_tokens = 64;
    const receipt: JsonObject = {
      selected_id: 'h1',
      candidates: [{ id: 'h1', text: 'hello', verifications: [{ evidence_id: 'e1', distribution: { support: 0.99 } }] }],
      evidence: [{ id: 'e1', source_id: 'one', text: 'hello '.repeat(1000) }, { id: 'e2', source_id: 'two', text: 'world '.repeat(1000) }],
    };
    const [prompt, visible, omitted] = model.realizationInput('hello?', receipt);
    expect(model.tokenizer.encode(prompt).inputIds[0]!.length).toBeLessThanOrEqual(64);
    expect(visible.length).toBeGreaterThan(0);
    expect(visible[0]!.id).toBe('e1');
    expect(omitted.length).toBe(2);
    expect(prompt).not.toContain('distribution');
    expect(prompt).not.toContain('verifications');
  });

  it('truncated realization verification cannot release an answer', () => {
    const model = prepared();
    const verifier = model.investigator!.verifier!;
    const original = verifier.verify.bind(verifier);
    verifier.verify = (text: string, evidence: readonly SourceText[]) => {
      const result = original(text, evidence);
      if (text === 'realized answer') for (const row of result) row.input_truncated = true;
      return result;
    };
    model.call(input());
    expect(model.lastResult!.abstention_enforced).toBe(true);
    expect((model.lastResult!.realization_verifications as JsonObject[]).some((row) => row.input_truncated)).toBe(true);
  });

  it('retains opaque memory across episodes and session saves', async () => {
    const model = prepared({ memory: true });
    model.call(input());
    expect(model.lastResult!.retained_evidence_ids).toEqual(['e1']);
    expect(JSON.stringify(snapshot(model).memory)).not.toContain('realized answer');
    model.newEpisode();
    expect(model.history).toEqual([]);
    model.call('hello?');
    expect(((model.lastResult!.cognition as JsonObject).retrieval as unknown[]).length).toBeGreaterThan(0);
    expect((((model.lastResult!.cognition as JsonObject).evidence as JsonObject[])[0]!).source_id).toBe('source-one');
    const path = join(temp.dir, 'memory.json');
    await model.saveSession(path);
    const session = model.newSession();
    await session.load(path);
    expect(session.cognition!.snapshot()).toEqual(snapshot(model));
  });

  it('memory is not committed on a failed decode', () => {
    const model = prepared({ memory: true });
    const before = snapshot(model);
    model.generateBatch = () => {
      throw new Error('decoder failed');
    };
    expect(() => model.call(input())).toThrow();
    expect(snapshot(model)).toEqual(before);
  });

  it('memory weight changes require an explicit index rebuild', () => {
    const model = prepared({ memory: true });
    model.call(input());
    model.newEpisode();
    noGrad(() => model.investigator!.rank.encode.parameters()[0]!.add_(0.1));
    expect(() => model.call('hello?')).toThrow(/stale/);
    model.rebuildMemory();
    model.call('hello?');
    expect(((model.lastResult!.cognition as JsonObject).retrieval as unknown[]).length).toBeGreaterThan(0);
  });

  for (const recallBeforeRevision of [false, true]) {
    it(`cross-episode source corrections survive session reloads (recall first=${recallBeforeRevision})`, async () => {
      const model = prepared({ memory: true });
      model.call(input());
      model.newEpisode();
      if (recallBeforeRevision) model.call('hello?');
      model.call({ question: 'hello?', revisions: [{ evidence_id: 'e1', text: 'corrected world', source_id: 'corrected-source' }] });
      const receipt = model.lastResult!.cognition as JsonObject;
      expect((receipt.evidence as JsonObject[]).map((row) => [row.text, row.source_id])).toEqual([['corrected world', 'corrected-source']]);
      const texts = new Set(model.cognitiveState!.evidence.map((row) => row.text));
      expect(texts.has('hello world') && texts.has('corrected world')).toBe(true);
      model.newEpisode();
      const path = join(temp.dir, `corrected-${recallBeforeRevision}.json`);
      await model.saveSession(path);
      const restored = await model.newSession().load(path);
      restored.call('hello?');
      expect(((restored.lastResult!.cognition as JsonObject).evidence as JsonObject[]).map((row) => row.text)).toEqual(['corrected world']);
      expect(restored.cognition!.retrieve('hello?').every((hit) => hit.evidence.text !== 'hello world')).toBe(true);
      // The caller keeps the logical source ID through repeated corrections.
      restored.call({ question: 'hello?', revisions: [{ evidence_id: 'e1', text: 'latest world' }] });
      expect(((restored.lastResult!.cognition as JsonObject).evidence as JsonObject[]).map((row) => row.text)).toEqual(['latest world']);
    });
  }

  it('a failed cross-episode correction commits no memory', () => {
    const model = prepared({ memory: true });
    model.call(input());
    model.newEpisode();
    const before = snapshot(model);
    const history = model.history;
    model.generateBatch = () => {
      throw new Error('decoder failed after revision');
    };
    expect(() => model.call({ question: 'hello?', revisions: [{ evidence_id: 'e1', text: 'corrected world' }] })).toThrow('decoder failed after revision');
    expect(snapshot(model)).toEqual(before);
    expect(model.history).toEqual(history);
  });

  for (const [field, value] of [['text', 'conflicting world'], ['source_id', 'different-source']]) {
    it(`conflicting remembered source loads are transactional (${field})`, async () => {
      const model = prepared({ memory: true });
      model.call(input());
      const path = join(temp.dir, `inconsistent-${field}.json`);
      await model.saveSession(path);
      const payload = JSON.parse(readFileSync(path, 'utf8'));
      payload.cognition.memory.records[0].evidence[field!] = value;
      writeFileSync(path, JSON.stringify(payload));
      const before = snapshot(model);
      const history = model.history;
      await expect(model.loadSession(path)).rejects.toThrow(/conflict/);
      expect(snapshot(model)).toEqual(before);
      expect(model.history).toEqual(history);
    });
  }

  it('repeated sources are idempotent across questions and episodes', () => {
    const model = prepared({ memory: true });
    model.call(input());
    const again = { ...input(), question: 'world?' };
    model.call(again);
    expect(model.lastResult!.retained_evidence_ids).toEqual([]);
    model.newEpisode();
    model.call(structuredClone(again));
    expect(((snapshot(model).memory as JsonObject).records as unknown[]).length).toBe(1);
    expect(model.lastResult!.retained_evidence_ids).toEqual([]);
  });

  it('empty owned generation withdraws selection without a fake hypothesis', () => {
    const model = prepared({ memory: true });
    model.call(input());
    expect(model.cognitiveState!.selection.length).toBeGreaterThan(0);
    // Restore the owned proposal pipeline; only its rendered output is authored.
    delete (model.investigator as unknown as { propose?: unknown }).propose;
    (model.investigator!.generator!.tokenizer as { batchDecode: unknown }).batchDecode = () => [''];
    expect(model.call('world?')).toBe(ABSTAIN);
    expect(model.lastResult!.abstention_enforced).toBe(true);
    expect((model.lastResult!.cognition as JsonObject).candidates).toEqual([]);
    expect(model.cognitiveState!.selection).toEqual([]);
    expect(model.cognitiveState!.evidence.map((row) => row.text)).toEqual(['hello world']);
    expect(model.cognitiveState!.hypotheses.every((row) => row.text.trim())).toBe(true);
    expect(((snapshot(model).memory as JsonObject).records as unknown[]).length).toBe(1);
  });

  it('pretrained default sessions use the loaded memory encoder', async () => {
    const settings = config();
    (settings.cognition as JsonObject).memory = { capacity: 8, top_k: 2 };
    const model = new Chatbot(settings).eval();
    await model.savePretrained(join(temp.dir, 'memory-model'));
    const restored = await Chatbot.fromPretrained(join(temp.dir, 'memory-model'));
    restored.investigator!.propose = () => [{ id: 'h1', text: 'hello', origin: 'generated' }] as never;
    restored.generateBatch = () => ['hello'];
    expect(typeof restored.call(input())).toBe('string');
    expect(restored.lastResult!.retained_evidence_ids).toEqual(['e1']);
    expect(((snapshot(restored).memory as JsonObject).records as unknown[]).length).toBe(1);
    const history = restored.history;
    restored.loadStateDict(new Chatbot(settings).stateDict());
    expect(() => restored.call('world?')).toThrow(/stale/);
    expect(restored.history).toEqual(history);
    expect(((snapshot(restored).memory as JsonObject).records as unknown[]).length).toBe(1);
  });

  it('joint scope screens realization and persists the complete model', async () => {
    const settings = config();
    ((settings.cognition as JsonObject).investigator as JsonObject).verification_scope = 'joint';
    const original = new Chatbot(settings).eval();
    supportVerifier(original.investigator!);
    await original.savePretrained(join(temp.dir, 'joint'));
    const model = await Chatbot.fromPretrained(join(temp.dir, 'joint'));
    expect(((model.configuration().cognition as JsonObject).investigator as JsonObject).verification_scope).toBe('joint');
    model.investigator!.propose = () => [{ id: 'h', text: 'hello' }] as never;
    model.generateBatch = () => ['realized answer'];
    const verifier = model.investigator!.verifier!;
    const joint = verifier.verifyJoint.bind(verifier);
    verifier.verifyJoint = (text: string, evidence: readonly SourceText[]) => {
      const result = joint(text, evidence)!;
      if (text === 'realized answer') result.distribution = { support: 0.01, contradiction: 0.01, unknown: 0.98 };
      return result;
    };
    model.call(input());
    expect((model.lastResult!.cognition as JsonObject).abstained).toBe(false);
    expect(model.lastResult!.abstention_enforced).toBe(true);
    expect((model.lastResult!.realization_joint_verification as JsonObject).source_ids).toEqual(['e1']);
  });

  it('joint realization full-evidence check retains an omitted conflict', () => {
    const settings = config();
    ((settings.cognition as JsonObject).investigator as JsonObject).verification_scope = 'joint';
    const model = new Chatbot(settings).eval();
    supportVerifier(model.investigator!);
    model.investigator!.propose = () => [{ id: 'h', text: 'hello' }] as never;
    model.generateBatch = () => ['realized answer'];
    model.realizationInput = (_question, interpretation) => [
      'hello', (interpretation.evidence as JsonObject[]).slice(0, 1), [{ evidence_id: 'e2', included_characters: 0 }],
    ];
    const verifier = model.investigator!.verifier!;
    const original = verifier.verify.bind(verifier);
    verifier.verify = (text: string, evidence: readonly SourceText[]) => {
      const checks = original(text, evidence);
      if (text === 'realized answer') {
        for (const row of checks) if (row.source_id === 'e2') row.distribution = { support: 0.01, contradiction: 0.98, unknown: 0.01 };
      }
      return checks;
    };
    const value = input();
    (value.evidence as JsonObject[]).push({ id: 'e2', source_id: 'two', text: 'world' });
    model.call(value);
    expect((model.lastResult!.cognition as JsonObject).abstained).toBe(false);
    expect(model.lastResult!.abstention_enforced).toBe(true);
    expect((model.lastResult!.realization_joint_verification as JsonObject).source_ids).toEqual(['e1']);
    expect((model.lastResult!.full_realization_joint_verification as JsonObject).source_ids).toEqual(['e1', 'e2']);
  });

  it('follow-ups use dialogue without promoting it to evidence', () => {
    const model = prepared();
    const proposed: JsonObject[] = [];
    const realized: string[][] = [];
    const ranked: JsonObject[] = [];
    model.investigator!.propose = (inputs) => {
      proposed.push(structuredClone(inputs) as JsonObject);
      return [{ id: 'h1', text: 'hello', origin: 'generated', generated_by: 'fixture' }] as never;
    };
    model.generateBatch = (inputs) => {
      realized.push([...inputs]);
      return ['realized answer'];
    };
    const rank = model.investigator!.rank;
    const receipt = rank.receipt.bind(rank);
    rank.receipt = (value, options) => {
      ranked.push(structuredClone(value) as JsonObject);
      return receipt(value, options);
    };
    const first = model.newSession();
    const second = model.newSession();
    for (const [session, question] of [[first, 'alpha antecedent'], [second, 'beta antecedent']] as const) {
      session.call({ ...input(), question });
      session.call('What caused that?');
    }
    expect(proposed[1]).not.toEqual(proposed[3]);
    expect(ranked[1]).not.toEqual(ranked[3]);
    expect(realized[1]).not.toEqual(realized[3]);
    expect(proposed[1]!.question).toBe('What caused that?');
    expect((proposed[1]!.conversation_context as JsonObject[]).every((row) => row.text !== 'What caused that?')).toBe(true);
    expect(proposed[1]!.evidence).toEqual(proposed[3]!.evidence);
    expect((proposed[1]!.evidence as JsonObject[]).every((row) => row.source_id === 'e1')).toBe(true);
    expect(first.cognition!.state.evidence.length).toBe(1);
    expect(realized[1]![0]).toContain('not source evidence');
  });

  it('bounds dialogue context and round-trips configuration and sessions', async () => {
    const model = prepared({ memory: true });
    model.call(input());
    model.call('hello again');
    const [rows, truncated] = model.conversationContextFor(model.defaultSession.history);
    expect(truncated).toBe(false);
    const pair = rows.slice(-2);
    (model.config.cognition as JsonObject).conversation_context_tokens = Math.max(
      ...[model.tokenizer, model.investigator!.generator!.tokenizer].map((tokenizer) => tokenizer.encode(conversationBlock(pair)).inputIds[0]!.length),
    );
    const [selected, cut] = model.conversationContextFor(model.defaultSession.history);
    expect(selected).toEqual(pair);
    expect(cut).toBe(true);
    await model.savePretrained(join(temp.dir, 'dialogue'));
    await model.saveSession(join(temp.dir, 'dialogue-session.json'));
    const loaded = await Chatbot.fromPretrained(join(temp.dir, 'dialogue'));
    await loaded.loadSession(join(temp.dir, 'dialogue-session.json'));
    expect(loaded.conversationContextFor(loaded.defaultSession.history)).toEqual([selected, cut]);
    expect((loaded.config.cognition as JsonObject).conversation_context_tokens).toBe((model.config.cognition as JsonObject).conversation_context_tokens);
    (model.config.cognition as JsonObject).conversation_context_tokens = 1;
    const before = snapshot(model);
    expect(() => model.call('followup')).toThrow(/too small/);
    expect(snapshot(model)).toEqual(before);
  });

  it('realization context overflow never evicts source text', () => {
    const model = prepared();
    const interpretation: JsonObject = { candidates: [], selected_id: null, evidence: [{ id: 'e1', source_id: 'doc', text: new Array(200).fill('hello').join(' ') }] };
    const [baseline, visible] = model.realizationInput('hello', interpretation);
    model.config.max_input_tokens = model.tokenizer.encode(baseline).inputIds[0]!.length;
    expect(() => model.realizationInput('hello', interpretation, {
      conversationContext: [{ role: 'user', text: 'antecedent' }, { role: 'assistant', text: 'unverified answer' }],
    })).toThrow(/realization token budget/);
    expect(model.realizationInput('hello', interpretation)[1]).toEqual(visible);
  });
});
