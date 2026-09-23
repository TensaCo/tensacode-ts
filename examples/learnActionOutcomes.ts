/**
 * Measured feedback in an explicitly authored, tiny service-recovery
 * simulation (Python ``examples/learn_action_outcomes.py``).
 *
 *     npm run build
 *     node examples/learnActionOutcomes.ts --output /tmp/action-outcome-run --epochs 18
 *
 * This is a mechanism demonstration, not real-world competence or causal
 * inference. The environment, actions, telemetry semantics, exploration and
 * reward are fixtures; nothing here adds domain policy to the library core.
 * Train/test scenario IDs and telemetry strings are disjoint; both share the
 * same three authored status classes. Every label comes from an executed
 * transition. The run writes sourced traces, observed trajectories, a model, a
 * separate training checkpoint, session state and a report, then checks that
 * weights, experience, session, trajectory and optimizer state restore exactly.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { isDeepStrictEqual, parseArgs } from 'node:util';
import { Adam, manualSeed } from 'tensorcode/nn';
import { ActionOutcome } from 'tensorcode/tools/actions';
import { ExecutablePlan, PlanExecutionResult, PlanStep, Planner } from 'tensorcode/tools/planner';
import { Trainer, loadExperience } from 'tensorcode/training';

type State = { scenario: string; status: string };
type Json = Record<string, any>;

const CANDIDATES = ['cool', 'reindex', 'serve'].map((name) => ({ id: name, text: name }));
const TRAIN: State[] = ['hot', 'corrupt', 'ready', 'hot', 'corrupt', 'ready'].map((status, i) => ({ scenario: `train-${i}`, status }));
const TEST: State[] = ['hot', 'corrupt', 'hot', 'corrupt', 'hot', 'corrupt'].map((status, i) => ({ scenario: `test-${i}`, status }));

function registry(): Record<string, (state: State) => ActionOutcome<State>> {
  const transition = (name: string) => (current: State): ActionOutcome<State> => {
    const state = structuredClone(current);
    const before = state.status;
    const done = name === 'serve' && before === 'ready';
    const repaired = (name === 'cool' && before === 'hot') || (name === 'reindex' && before === 'corrupt');
    if (repaired) state.status = 'ready';
    if (done) state.status = 'serving';
    const reward = done ? 1 : repaired ? 0.5 : -0.5;
    return new ActionOutcome(state, { before, after: state.status, reward, scenario: state.scenario }, done);
  };
  return Object.fromEntries(['cool', 'reindex', 'serve'].map((name) => [name, transition(name)]));
}

function inputs(state: State): Json {
  return {
    goal: 'restore service', plans: structuredClone(CANDIDATES),
    evidence: [{ source_id: `${state.scenario}:telemetry:${state.status}`, text: `status ${state.status} scenario ${state.scenario}` }],
  };
}

function structured(candidateId: string): ExecutablePlan {
  if (!CANDIDATES.some((item) => item.id === candidateId)) throw new Error('selected candidate is not registered; no fallback');
  return new ExecutablePlan(candidateId, [new PlanStep(candidateId)]);
}

async function runScenario(model: Planner, state: State, baseline = false): Promise<PlanExecutionResult<State>> {
  const choose = (current: State) => structured(baseline ? 'cool' : model.call(inputs(current)).selected_id as string);
  return model.newExecutor<State>({ actions: registry(), replan: (request) => choose(request.state), maxSteps: 2 }).call(state, choose(state));
}

async function evaluate(model: Planner, baseline = false): Promise<Json> {
  const runs = [];
  for (const state of TEST) runs.push(await runScenario(model, state, baseline));
  return {
    success_rate: runs.filter((run) => run.stopReason === 'completed').length / runs.length,
    mean_reward: runs.reduce((total, run) => total + run.experiences.reduce((sum, item) => sum + (item.observation as Json).reward, 0), 0) / runs.length,
    scenarios: runs.length,
  };
}

async function run(outputPath: string, epochs = 18, seed = 12): Promise<Json> {
  if (!Number.isInteger(epochs) || epochs < 1) throw new Error('epochs must be positive');
  manualSeed(seed);
  mkdirSync(outputPath, { recursive: true });
  const model = new Planner({
    vocabulary: ['restore', 'service', 'status', 'hot', 'corrupt', 'ready', 'scenario', 'train', 'test', 'cool', 'reindex', 'serve'],
    dimensions: 12, slots: 2, steps: 1,
  });
  const trainer = Trainer.fromTool(model, { optimizer: (parameters) => new Adam(parameters, { lr: 0.008 }) });
  const before = await evaluate(model);
  const baseline = await evaluate(model, true);
  const sessions = [];
  const observations: Json[] = [];
  // Explicit exploration: every label comes from a fresh actual transition.
  // Even with all candidates in the input, each trace labels only its executed ID.
  for (const state of TRAIN) {
    for (const candidate of CANDIDATES) {
      const trajectory = await model.newExecutor<State>({ actions: registry(), replan: () => null, maxSteps: 1 }).call(state, structured(candidate.id));
      const experience = trajectory.experiences[0]!;
      const target = experience.toTarget((experience.observation as Json).reward);
      const session = trainer.capture(inputs(state), target, { source: experience.sourceId });
      const name = `experience-${String(sessions.length).padStart(2, '0')}.json`;
      await session.save(join(outputPath, name), { operations: trainer.operations });
      await trajectory.save(join(outputPath, name.replace('experience', 'trajectory')));
      sessions.push(session);
      observations.push({ scenario: state.scenario, candidate_id: candidate.id, source_id: experience.sourceId, target, file: name });
    }
  }
  const losses: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    losses.push(sessions.reduce((total, session) => total + trainer.step(session), 0) / sessions.length);
  }
  const after = await evaluate(model);
  await model.savePretrained(join(outputPath, 'model'));
  await trainer.saveCheckpoint(join(outputPath, 'training'), { progress: { epochs } });
  const session = model.newSession();
  const expected = session.call(inputs(TEST[0]!));
  await session.save(join(outputPath, 'session.json'));
  const trajectory = await runScenario(model, TEST[0]!);
  await trajectory.save(join(outputPath, 'evaluation-trajectory.json'));
  const restored = await Planner.fromPretrained(join(outputPath, 'model'), { localFilesOnly: true });
  const resumed = Trainer.fromTool(restored, { optimizer: (parameters) => new Adam(parameters, { lr: 0.008 }) });
  await resumed.loadCheckpoint(join(outputPath, 'training'));
  const loadedTrace = await loadExperience(join(outputPath, observations[0]!.file), { operations: resumed.operations });
  // The session adds revision fields to its receipt; compare stable candidate scores.
  const parity = isDeepStrictEqual(restored.call(inputs(TEST[0]!)).candidates, expected.candidates);
  const sessionClass = session.constructor as typeof session.constructor & { load(path: string, tool: Planner): Promise<typeof session> };
  const sessionParity = isDeepStrictEqual((await sessionClass.load(join(outputPath, 'session.json'), restored)).history, session.history);
  const trajectoryParity = isDeepStrictEqual((await PlanExecutionResult.load(join(outputPath, 'evaluation-trajectory.json'))).toData(), trajectory.toData());
  const expectedLoss = trainer.step(sessions[0]!);
  const actualLoss = resumed.step(loadedTrace);
  const restoredParameters = restored.parameters();
  const continuation = expectedLoss === actualLoss && model.parameters().every((parameter, index) => parameter.equal(restoredParameters[index]!));
  let unknownRejected: boolean;
  try {
    model.loss(inputs(TRAIN[0]!), { candidate_id: 'unseen-action', outcome: 0 });
    unknownRejected = false;
  } catch {
    unknownRejected = true;
  }
  const frozen = [];
  for (const state of TEST) {
    frozen.push(await model.newExecutor<State>({ actions: registry(), replan: (request) => request.previousPlan, maxSteps: 2 })
      .call(state, structured(restored.call(inputs(state)).selected_id as string)));
  }
  const report = {
    scope: 'Authored deterministic simulation; shared status classes; no real-world competence or causal estimate.',
    seed, epochs, updates: epochs * sessions.length,
    train_ids: TRAIN.map((item) => item.scenario), test_ids: TEST.map((item) => item.scenario),
    before, after, fixed_cool_baseline: baseline,
    frozen_feedback_success_rate: frozen.filter((item) => item.stopReason === 'completed').length / TEST.length,
    first_epoch_loss: losses[0], last_epoch_loss: losses[losses.length - 1],
    observations: observations.length, labels_per_observation: 1,
    model_parity: parity, session_parity: sessionParity, trajectory_parity: trajectoryParity,
    optimizer_continuation_parity: continuation, unseen_label_rejected: unknownRejected,
  };
  writeFileSync(join(outputPath, 'observations.json'), `${JSON.stringify(observations, null, 2)}\n`);
  writeFileSync(join(outputPath, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  return report;
}

const { values } = parseArgs({ options: { output: { type: 'string' }, epochs: { type: 'string', default: '18' } } });
if (!values.output) throw new Error('--output is required');
console.log(JSON.stringify(await run(values.output, Number(values.epochs)), null, 2));
