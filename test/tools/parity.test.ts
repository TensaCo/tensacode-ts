/**
 * Python-written tool artifacts, receipts, proposals and persisted runtime
 * records load in TypeScript and reproduce (fixtures: scripts/fixtures/tools_fixtures.py).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { noGrad, tensor } from '../../src/nn/index.js';
import { fingerprint, operationConfiguration } from '../../src/_internal/fingerprint.js';
import { Chatbot, boundedMemoryUpdate } from '../../src/tools/chatbot.js';
import { Investigator } from '../../src/tools/investigator.js';
import { Planner } from '../../src/tools/planner.js';
import { Decision } from '../../src/tools/decision.js';
import { proposalPrompt } from '../../src/_internal/proposals.js';
import { RankingSession } from '../../src/_internal/sessions/ranking.js';
import { CognitiveSession } from '../../src/_internal/cognition/session.js';
import { CognitiveState } from '../../src/_internal/cognition/state.js';
import { PlanExecutionResult } from '../../src/_internal/execution/planning.js';
import { JsonMemory } from '../../src/_internal/memory/json.js';
import { RetrievalEncoder } from '../../src/_internal/retrieval.js';
import { Evidence } from '../../src/tools/cognition.js';
import type { OperationLike } from '../../src/ops/base.js';
import { expectClose } from '../helpers/gradcheck.js';
import { TOOLS, expectDeepClose, fixturePath, scratch } from './helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

function bindingFingerprints(bindings: Record<string, OperationLike>): Record<string, string> {
  return Object.fromEntries(Object.entries(bindings).map(([name, operation]) => [name, fingerprint(operationConfiguration(operation))]));
}

function bytes(path: string): Buffer {
  return readFileSync(path);
}

/**
 * The manifest re-saves byte-identically. Safetensors data sections are
 * byte-identical; headers are compared as JSON because Python itself orders
 * the tied-alias ``__metadata__`` entries nondeterministically (Rust HashMap).
 */
async function expectByteIdenticalResave(model: { savePretrained(path: string): Promise<string> }, source: string, name: string): Promise<void> {
  const target = join(temp.dir, name);
  await model.savePretrained(target);
  expect(Buffer.compare(bytes(join(target, 'tensorcode_config.json')), bytes(join(source, 'tensorcode_config.json')))).toBe(0);
  const split = (buffer: Buffer): [unknown, Buffer] => {
    const length = Number(buffer.readBigUInt64LE(0));
    return [JSON.parse(buffer.subarray(8, 8 + length).toString('utf8')), buffer.subarray(8 + length)];
  };
  const [actualHeader, actualData] = split(bytes(join(target, 'model.safetensors')));
  const [expectedHeader, expectedData] = split(bytes(join(source, 'model.safetensors')));
  expect(actualHeader).toEqual(expectedHeader);
  expect(Buffer.compare(actualData, expectedData)).toBe(0);
}

describe('Chatbot parity with Python', () => {
  const record = TOOLS.chatbot;

  it('constructs the same configuration, fingerprint, state and operation identities', () => {
    const model = new Chatbot(record.config);
    expect(model.configuration()).toEqual(record.configuration);
    expect(model.fingerprint).toBe(record.fingerprint);
    expect([...model.stateDict().keys()]).toEqual(record.state_keys);
    expect(model.encoder.configuration()).toEqual(record.encoder_configuration);
    expect(model.decoder.configuration()).toEqual(record.decoder_configuration);
    expect(bindingFingerprints(model.operationBindings())).toEqual(record.bindings);
  });

  it('loads the Python artifact, reproduces losses, encodings and decoding, and re-saves byte-identically', async () => {
    const source = fixturePath('chatbot');
    const model = await Chatbot.fromPretrained(source);
    expect(model.configuration()).toEqual(record.configuration);
    noGrad(() => {
      expectClose([model.lossBatch(['hello world'], ['answer']).item()], [record.losses.single], 1e-5);
      expectClose([model.lossBatch(['hello', 'hello world user'], ['answer', 'world answer']).item()], [record.losses.batch], 1e-5);
      expectClose([model.lossBatch(['hello'], ['answer'], { workspaceAblation: 'bypass' }).item()], [record.losses.bypass], 1e-5);
      expectClose([model.lossBatch(['hello'], ['answer'], { workspaceAblation: 'zero' }).item()], [record.losses.zero], 1e-5);
      expectClose(model.encoder.call(['hello', 'hello world']).encoded.data, record.encoded.data, 1e-5);
      expectClose(model.encodeWorkspace(['hello', 'hello world']).conditioning.data, record.conditioning.data, 1e-5);
      const state = model.encodeWorkspace(['hello world']);
      expect((model.decoder.call(state, { context: { max_new_tokens: 3, do_sample: false } }) as ReturnType<typeof tensor>).tolist()).toEqual(record.greedy_tokens);
      const beams = model.decoder.call(state, { context: { max_new_tokens: 3, do_sample: false, num_beams: 3, num_return_sequences: 3 } });
      expect((beams as ReturnType<typeof tensor>).tolist()).toEqual(record.beam_tokens);
    });
    expect(model.generateBatch(['hello', 'hello world', 'user : hello'])).toEqual(record.generations);
    await expectByteIdenticalResave(model, source, 'chatbot');
  });

  it('loads a Python chat session and writes identical session files', async () => {
    const model = await Chatbot.fromPretrained(fixturePath('chatbot'));
    const loaded = model.newSession();
    await loaded.load(fixturePath('chatbot_session.json'));
    expect(loaded.history).toEqual(record.session_history);
    const replay = model.newSession();
    replay.call('hello');
    replay.call('world answer');
    expect(replay.history).toEqual(record.session_history);
    expect(replay.lastResult).toEqual(record.session_last_result);
    const path = join(temp.dir, 'chat-session.json');
    await replay.save(path);
    expect(readFileSync(path, 'utf8')).toBe(readFileSync(fixturePath('chatbot_session.json'), 'utf8'));
  });
});

describe('Investigator parity with Python', () => {
  const record = TOOLS.investigator;

  it('constructs identical configuration, state names and binding fingerprints', () => {
    const tool = new Investigator(record.config);
    expect(tool.configuration()).toEqual(record.configuration);
    expect([...tool.stateDict().keys()]).toEqual(record.state_keys);
    expect(bindingFingerprints(tool.operationBindings())).toEqual(record.bindings);
    expect(tool.generator!.fingerprint).toBe(record.generator_fingerprint);
  });

  it('builds identical proposal prompts', () => {
    const input = { question: 'hello', evidence: [{ source_id: 'a', text: 'world' }, { source_id: 'b', text: 'hello' }] };
    expect(proposalPrompt(input, 'question')).toBe(record.prompts.plain);
    expect(proposalPrompt({ goal: 'hello', evidence: input.evidence }, 'goal')).toBe(record.prompts.goal);
    expect(proposalPrompt({
      ...input, conversation_context: [{ role: 'user', text: 'which café ☕ incident 𝄞?' }, { role: 'assistant', text: 'unverified "quoted" answer' }],
    }, 'question')).toBe(record.prompts.conversation);
    expect(proposalPrompt({ question: 'naïve café — ☕?', evidence: [{ source_id: 'ü', text: 'ß\n"x"' }] }, 'question')).toBe(record.prompts.unicode);
    expect(proposalPrompt(input, 'question', { templateVersion: 2 })).toBe(record.prompts.v2);
  });

  it('loads the Python artifact and reproduces proposals, verification and receipts', async () => {
    const source = fixturePath('investigator');
    const tool = await Investigator.fromPretrained(source);
    const input = { question: 'hello', evidence: [{ source_id: 'a', text: 'world' }, { source_id: 'b', text: 'hello' }] };
    expect(tool.propose(input, { count: 3 })).toEqual(record.proposals);
    expect(tool.propose(input, { count: 1 })).toEqual(record.single_proposal);
    const pairs = [{ premise: 'world', hypothesis: 'hello' }, { premise: 'hello world', hypothesis: 'world' }];
    expectClose(noGrad(() => tool.verifier!.call(pairs)).data, record.verifier_logits.data, 1e-5);
    expectDeepClose(tool.verifier!.verify('hello', input.evidence), record.verifications);
    expectDeepClose(tool.verifier!.verifyJoint('hello', input.evidence), record.joint_verification);
    expectDeepClose(tool.investigate({ ...input, hypotheses: [{ id: 'h1', text: 'hello' }, { id: 'h2', text: 'world' }] }), record.receipt);
    expectClose([noGrad(() => tool.proposalLoss(input, 'answer')).item()], [record.proposal_loss], 1e-5);
    await expectByteIdenticalResave(tool, source, 'investigator');
  });

  it('validates and replays a Python ranking session file', async () => {
    const tool = await Investigator.fromPretrained(fixturePath('investigator'));
    const session = await RankingSession.load(fixturePath('investigator_session.json'), tool);
    expect(session.history).toEqual(record.session_history);
    const path = join(temp.dir, 'ranking-session.json');
    await session.save(path);
    expect(readFileSync(path, 'utf8')).toBe(readFileSync(fixturePath('investigator_session.json'), 'utf8'));
    const replay = tool.newSession();
    replay.call({ question: 'hello', evidence: [{ source_id: 'a', text: 'world' }, { source_id: 'b', text: 'hello' }], hypotheses: [{ id: 'h1', text: 'hello' }, { id: 'h2', text: 'world' }] });
    replay.call({ question: 'hello', evidence: [{ source_id: 'a', text: 'world' }, { source_id: 'b', text: 'hello' }] });
    expectDeepClose(replay.history, record.session_history);
  });

  it('records calibration against the same verifier weight digest', async () => {
    const tool = await Investigator.fromPretrained(fixturePath('investigator'));
    tool.verifier!.calibration.fit(tensor([[5, 0, 0], [5, 0, 0]]), tensor([0, 1], { dtype: 'int64' }));
    expect(tool.verifier!.calibrationWeightDigest.toArray()).toEqual(record.calibration_digest);
    expectClose([tool.verifier!.calibration.temperature.item()], [record.calibration_temperature], 1e-6);
  });
});

describe('Planner and Decision parity with Python', () => {
  const record = TOOLS.planner;

  it('loads Python planner artifacts and reproduces receipts and plan proposals', async () => {
    const source = fixturePath('planner');
    const planner = await Planner.fromPretrained(source);
    expect(planner.configuration()).toEqual(record.configuration);
    expect([...planner.stateDict().keys()]).toEqual(record.state_keys);
    expect(bindingFingerprints(planner.operationBindings())).toEqual(record.bindings);
    expectDeepClose(planner.call(record.inputs), record.receipt);
    expect(planner.propose({ goal: 'hello' }, { count: 2 })).toEqual(record.proposals);
    await expectByteIdenticalResave(planner, source, 'planner');
  });

  it('keeps the distinct Decision identity', async () => {
    const source = fixturePath('decision');
    const decision = await Decision.fromPretrained(source);
    expect(decision).toBeInstanceOf(Decision);
    expect(decision.configuration()).toEqual(record.decision.configuration);
    expectDeepClose(decision.call(record.decision.case), record.decision.receipt);
    expect(bindingFingerprints(decision.operationBindings())).toEqual(record.decision.bindings);
    await expect(Investigator.fromPretrained(source)).rejects.toThrow(/incompatible model tool/);
    await expectByteIdenticalResave(decision, source, 'decision');
  });
});

describe('Cognitive runtime parity with Python', () => {
  const record = TOOLS.cognition;

  it('restores a Python cognitive session with the same model identity and snapshot', async () => {
    const tool = await Investigator.fromPretrained(fixturePath('cognitive_investigator'));
    const session = await tool.loadCognitiveSession(fixturePath('cognitive_session.json'));
    expect(session.modelIdentity()).toBe(record.identity);
    expect(session.memory!.fingerprint).toBe(record.memory_fingerprint);
    expect(session.snapshot()).toEqual(record.snapshot);
    const path = join(temp.dir, 'cognitive-session.json');
    await session.save(path);
    expect(readFileSync(path, 'utf8')).toBe(readFileSync(fixturePath('cognitive_session.json'), 'utf8'));
  });

  it('replays the same investigation sequence', async () => {
    const tool = await Investigator.fromPretrained(fixturePath('cognitive_investigator'));
    const session = tool.newCognitiveSession({ memory: { capacity: 4, top_k: 2 }, maxRecords: 64 });
    session.ingest([new Evidence('a', 'alpha', 'doc-a'), new Evidence('b', 'beta beta', 'doc-b')]);
    session.remember('a', { question: 'alpha?', outcome: 'supplied outcome' });
    expectDeepClose(session.investigate('alpha', { hypotheses: [{ id: 'h', text: 'beta' }] }), record.first);
    session.reviseEvidence('a', 'alpha beta', 'doc-a2');
    session.newEpisode();
    expectDeepClose(session.investigate('alpha', { hypotheses: [{ id: 'h', text: 'alpha' }] }), record.second);
    expectDeepClose(session.snapshot(), record.snapshot);
  });

  it('round-trips Python cognitive state, trajectories and JSON memory files byte-identically', async () => {
    const state = await CognitiveState.load(fixturePath('cognitive_state.json'));
    const statePath = join(temp.dir, 'state.json');
    await state.save(statePath);
    expect(readFileSync(statePath, 'utf8')).toBe(readFileSync(fixturePath('cognitive_state.json'), 'utf8'));
    const trajectory = await PlanExecutionResult.load(fixturePath('trajectory.json'));
    expect(trajectory.toData()).toEqual(TOOLS.runtime.trajectory);
    const trajectoryPath = join(temp.dir, 'trajectory.json');
    await trajectory.save(trajectoryPath);
    expect(readFileSync(trajectoryPath, 'utf8')).toBe(readFileSync(fixturePath('trajectory.json'), 'utf8'));
    const memoryPath = join(temp.dir, 'memory.json');
    await import('node:fs/promises').then(({ copyFile }) => copyFile(fixturePath('memory.json'), memoryPath));
    const memory = await JsonMemory.open(memoryPath, { retrieve: (search) => search.candidates.slice(0, search.limit) });
    expect(memory.records.map((row) => row.toRecord())).toEqual([
      { source_id: 'memory-00000001', kind: 'observation', value: 'Customer asked about a card fee ☕', metadata: { turn: 1 } },
      { source_id: 'explicit', kind: 'response', value: { nested: [1, 2.5, null] }, metadata: { turn: 2 } },
    ]);
    await memory.transaction(() => undefined);
    expect(readFileSync(memoryPath, 'utf8')).toBe(readFileSync(fixturePath('memory.json'), 'utf8'));
  });

  it('bounds memory updates exactly like Python', () => {
    const bounded = TOOLS.runtime.bounded;
    const residual = boundedMemoryUpdate(
      tensor(bounded.native.data, { shape: bounded.native.shape }), tensor(bounded.update.data, { shape: bounded.update.shape }),
      tensor(bounded.mask.flat().map(Number), { shape: [2, 3], dtype: 'bool' }), tensor(bounded.gate),
    );
    expectClose(residual.data, bounded.residual.data, 1e-5);
  });
});

describe('RetrievalEncoder parity with Python', () => {
  const record = TOOLS.retrieval;

  it('loads the Python artifact, embeds identically and reports the same tensor schema', async () => {
    const source = fixturePath('retrieval');
    const encoder = await RetrievalEncoder.fromPretrained(source);
    expect(encoder.configuration()).toEqual({ ...record.config, freeze_foundation: false });
    expectDeepClose(encoder.receipt(['alpha', 'alpha beta beta', 'beta alpha alpha alpha alpha']), record.receipt);
    expect(encoder.encode.configuration()).toEqual(record.encode_configuration);
    expect(bindingFingerprints(encoder.operationBindings())).toEqual(record.bindings);
    await expectByteIdenticalResave(encoder, source, 'retrieval');
    const fresh = new CognitiveSession(await Investigator.fromPretrained(fixturePath('cognitive_investigator')));
    expect(fresh.modelIdentity()).toBe(TOOLS.cognition.identity);
  });
});
