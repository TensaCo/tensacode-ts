/**
 * Tool training across modules (tools + training), ported from the Python tests
 * that need the Trainer:
 * - models/test_investigation.py::test_training_modes_durable_replay_and_optimizer_checkpoint
 * - models/test_chatbot_model.py::test_tool_trainer_durable_experience_and_resume
 * - models/test_retrieval_encoder.py::test_retrieval_tooltrainer_trace_roundtrip_and_optimizer_invalidates_index
 * - models/test_retrieval_encoder.py::test_rank_trace_roundtrip_normalizes_retrieval_defaults_before_rank_construction
 * - training/test_tool_training.py::test_owned_model_checkpoint_and_experience_round_trip
 * - training/test_experience_persistence.py::test_immutable_message_and_graph_payloads_roundtrip
 * Tiny random models and authored targets test mechanisms, not competence.
 */
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { trace } from '../../src/index.js';
import { Operation, type Context } from '../../src/ops/index.js';
import { Graph, SourceAnchor } from '../../src/ops/graph/index.js';
import { ClassificationResult } from '../../src/ops/text/index.js';
import { Chatbot, Investigator } from '../../src/tools/index.js';
import { Evidence } from '../../src/tools/cognition.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import { LearnedEpisodicMemory, type MemoryOwner } from '../../src/_internal/memory/learned.js';
import { investigatorConfig, retrievalConfig, scratch, tinyConfig } from '../tools/helpers.js';

const temp = scratch('tensorcode-integration-');
afterAll(() => temp.cleanup());

let counter = 0;
const directory = (name: string) => join(temp.dir, `${name}-${counter++}`);

const INPUT = { question: 'hello', evidence: [{ source_id: 'a', text: 'world' }, { source_id: 'b', text: 'hello' }] };

describe('Investigator training modes', () => {
  for (const mode of ['rank', 'proposal', 'verification'] as const) {
    it(`${mode}: durable replay and optimizer checkpoint`, async () => {
      const root = directory(mode);
      const tool = new Investigator(investigatorConfig());
      const trainer = Trainer.fromTool(tool);
      let inputs: unknown;
      let targets: unknown;
      if (mode === 'rank') {
        inputs = { ...INPUT, hypotheses: [{ id: 'a', text: 'hello' }, { id: 'b', text: 'world' }] };
        targets = 'a';
      } else if (mode === 'proposal') {
        inputs = INPUT;
        targets = 'answer';
      } else {
        inputs = [{ premise: 'world', hypothesis: 'hello' }];
        targets = ['support'];
      }
      const experience = trainer.capture({ mode, inputs }, targets, { source: 'authored-mechanism-fixture' });
      expect(trainer.step(experience)).toBeGreaterThanOrEqual(0);
      await experience.save(join(root, 'experience.json'), { operations: trainer.operations });
      await tool.savePretrained(join(root, 'model'));
      await trainer.saveCheckpoint(join(root, 'resume'));
      const restored = Trainer.fromTool(await Investigator.fromPretrained(join(root, 'model')));
      await restored.loadCheckpoint(join(root, 'resume'));
      expect(restored.steps).toBe(1);
      const loaded = await loadExperience(join(root, 'experience.json'), { operations: restored.operations });
      expect(restored.step(loaded)).toBeGreaterThanOrEqual(0);
    });
  }
});

describe('Chatbot training', () => {
  it('durable experience and resume', async () => {
    const root = directory('chatbot');
    const model = new Chatbot(tinyConfig());
    const trainer = Trainer.fromTool(model);
    const session = trainer.capture(['hello'], ['answer'], { source: 'test authored target' });
    await session.save(join(root, 'experience.json'), { operations: trainer.operations });
    const restoredSession = await loadExperience(join(root, 'experience.json'), { operations: trainer.operations });
    expect(Number.isFinite(trainer.step(restoredSession))).toBe(true);
    await trainer.saveCheckpoint(join(root, 'resume'), { progress: { batch: 1 } });
    const fresh = Trainer.fromTool(new Chatbot(tinyConfig()));
    expect(await fresh.loadCheckpoint(join(root, 'resume'))).toEqual({ batch: 1 });
    expect(fresh.steps).toBe(1);
    const freshExperience = await loadExperience(join(root, 'experience.json'), { operations: fresh.operations });
    expect(Number.isFinite(fresh.step(freshExperience))).toBe(true);
  });
});

describe('retrieval training', () => {
  const retrievalInvestigator = () => new Investigator({
    vocabulary: ['alpha', 'beta'], dimensions: 4, slots: 2, steps: 1, retrieval_encoder: retrievalConfig(),
  }).eval();

  it('trace roundtrip; an optimizer step invalidates the episodic index', async () => {
    const root = directory('retrieval');
    const tool = retrievalInvestigator();
    const memory = new LearnedEpisodicMemory(tool as unknown as MemoryOwner);
    memory.remember(new Evidence('a', 'alpha', 'source'), { episodeId: 'past' });
    const inputs = { queries: ['alpha', 'beta'], documents: ['alpha', 'beta'] };
    const targets = [[true, false], [false, true]];
    const trainer = Trainer.fromTool(tool, { lr: 0.01 });
    const experience = trainer.capture({ mode: 'retrieval', inputs }, targets, { source: 'authored-mechanism-fixture' });
    const before = tool.episodicEncoder!.parameters()[0]!.detach().clone();
    expect(trainer.step(experience)).toBeGreaterThanOrEqual(0);
    expect(before.equal(tool.episodicEncoder!.parameters()[0]!)).toBe(false);
    expect(() => memory.retrieve('alpha')).toThrow(/stale/);
    await experience.save(join(root, 'experience.json'), { operations: trainer.operations });
    await tool.savePretrained(join(root, 'model'));
    await trainer.saveCheckpoint(join(root, 'resume'));
    const restored = Trainer.fromTool(await Investigator.fromPretrained(join(root, 'model')), { lr: 0.01 });
    await restored.loadCheckpoint(join(root, 'resume'));
    const loaded = await loadExperience(join(root, 'experience.json'), { operations: restored.operations });
    expect(restored.step(loaded)).toBeGreaterThanOrEqual(0);
    expect(() => tool.retrievalLoss({ ...inputs, targets }, targets)).toThrow();
    expect(Number.isFinite(tool.episodicEncoder!.contrastiveLoss(inputs.queries, inputs.documents, targets).item())).toBe(true);
  });

  it('rank experience roundtrip normalizes retrieval defaults before rank construction', async () => {
    const root = directory('rank');
    const tool = retrievalInvestigator();
    const trainer = Trainer.fromTool(tool);
    const inputs = { question: 'alpha', evidence: [], hypotheses: [{ id: 'a', text: 'alpha' }, { id: 'b', text: 'beta' }] };
    const experience = trainer.capture(inputs, 'a', { source: 'authored-mechanism-fixture' });
    await experience.save(join(root, 'rank-experience.json'), { operations: trainer.operations });
    await tool.savePretrained(join(root, 'model'));
    const restored = Trainer.fromTool(await Investigator.fromPretrained(join(root, 'model')));
    const loaded = await loadExperience(join(root, 'rank-experience.json'), { operations: restored.operations });
    expect(restored.step(loaded)).toBeGreaterThanOrEqual(0);
  });
});

describe('owned tool checkpoints', () => {
  it('checkpoint and experience round trip on a fresh Investigator', async () => {
    const root = directory('owned');
    const config = { vocabulary: ['test', 'yes', 'no'], dimensions: 8, slots: 2, steps: 1 };
    const first = Trainer.fromTool(new Investigator(config));
    const inputs = { question: 'test', evidence: [], hypotheses: [{ id: 'y', text: 'yes' }, { id: 'n', text: 'no' }] };
    const session = first.capture(inputs, 'y', { source: 'test:review' });
    first.step(session);
    await session.save(join(root, 'experience.json'), { operations: first.operations });
    await first.saveCheckpoint(join(root, 'resume'));
    const second = Trainer.fromTool(new Investigator(config));
    await second.loadCheckpoint(join(root, 'resume'));
    expect(second.parameters.length).toBe(first.parameters.length);
    for (const [index, expected] of first.parameters.entries()) expect(expected.equal(second.parameters[index]!)).toBe(true);
    const restored = await loadExperience(join(root, 'experience.json'), { operations: second.operations });
    expect(second.step(restored)).toBeGreaterThan(0);
  });
});

describe('experience payloads', () => {
  it('graph and classification records round trip immutably', async () => {
    class Echo extends Operation<unknown, Record<string, unknown>> {
      override get replayable(): boolean { return true; }
      // Return new containers to retain unambiguous object identity.
      forward(value: unknown, context: Context | null): Record<string, unknown> {
        return { value, context };
      }
    }
    const op = new Echo();
    const graph = new Graph(['a'], {
      sources: ['source:1'], attributes: { nested: { flag: true } },
      sourceAnchors: [new SourceAnchor('source:1', { target: 'a', location: { line: 2 } })],
    });
    const result = new ClassificationResult('yes', { distribution: { yes: 0.75, no: 0.25 } });
    const session = trace();
    const output = session.run(() => op.call(graph, { context: { decision: result } }));
    const codecs = { graph: Graph, anchor: SourceAnchor, classification: ClassificationResult };
    session.supervise(output, result, { loss: 'custom' });
    const path = join(directory('payloads'), 'immutable.json');
    await session.save(path, { operations: { echo: op }, codecs, release: true });
    const loaded = await loadExperience(path, { operations: { echo: op }, codecs });
    const restored = loaded.replay(loaded.supervisions[0]!.output) as { value: Graph; context: { decision: ClassificationResult } };
    expect(restored.value).toBeInstanceOf(Graph);
    expect(restored.value.equals(graph)).toBe(true);
    expect(restored.context.decision).toBeInstanceOf(ClassificationResult);
    expect(restored.context.decision.toRecord()).toEqual(result.toRecord());
    expect(() => { (restored.context.decision.distribution as Record<string, number>).yes = 0; }).toThrow(TypeError);
  });
});
