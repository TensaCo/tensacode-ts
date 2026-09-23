/**
 * Learn to rank supplied plans from observed outcomes; never execute a plan
 * (Python ``examples/plan_learning.py``).
 *
 *     npm run build
 *     node examples/planLearning.ts collect --input observed.jsonl --artifacts /tmp/plans
 *     node examples/planLearning.ts train --artifacts /tmp/plans --epochs 100
 *     node examples/planLearning.ts predict --input heldout.jsonl --artifacts /tmp/plans
 *
 * JSONL rows: id, task, evidence=[{id, text}], plans=[{id, text, outcome,
 * source}]. Training outcomes must be finite numbers on a shared scale where
 * higher is better; source identifies the actual observation. Prediction rows
 * omit outcome/source, or include both for held-out MSE. Candidate generation,
 * text pooling and maximizing the predicted outcome are authored policies.
 * This is not autonomous planning, causal inference or calibrated confidence.
 * Artifacts (experience files, checkpoints) are interchangeable with Python's.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { trace, type Trace } from 'tensorcode';
import { Adam, manualSeed, noGrad, tensor, type Tensor } from 'tensorcode/nn';
import { Decode, VocabularyEncoder, latentCodecs, type Latent } from 'tensorcode/ops/vec';
import { Trainer, loadExperience } from 'tensorcode/training';

type Item = { id: string; text: string };
type Plan = Item & { outcome?: number; source?: string };
type Row = { id: string; task: string; evidence: Item[]; plans: Plan[] };
type Manifest = { dimensions: number; vocabulary: string[]; training_ids: string[]; training_texts: string[]; seed: number; experiences: string[]; observations: unknown[] };
type Operations = { interpret: VocabularyEncoder; anticipate: Decode };

/** Authored vocabulary policy matching the mechanical encoder tokenizer. */
function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) ?? [];
}

const nonempty = (value: unknown): value is string => typeof value === 'string' && Boolean(value.trim());

function readRows(path: string, supervised: boolean): Row[] {
  const rows: Row[] = [];
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as Row;
    if (row === null || typeof row !== 'object' || !nonempty(row.id) || !nonempty(row.task)) throw new Error('Each task needs nonempty id and task strings');
    if (!Array.isArray(row.evidence) || !Array.isArray(row.plans) || !row.plans.length) throw new Error('Each task needs evidence list and nonempty plans list');
    for (const items of [row.evidence, row.plans]) {
      if (items.some((item) => item === null || typeof item !== 'object' || !nonempty(item.id) || !nonempty(item.text))) {
        throw new Error('Evidence and plans need nonempty id/text strings');
      }
      if (new Set(items.map((item) => item.id)).size !== items.length) throw new Error('Evidence and plan IDs must be unique within each list');
    }
    for (const plan of row.plans) {
      if (supervised || 'outcome' in plan || 'source' in plan) {
        if (typeof plan.outcome !== 'number' || !Number.isFinite(plan.outcome)) throw new Error('Observed outcomes must be finite numbers');
        if (!nonempty(plan.source)) throw new Error('Observed outcomes require a source ID');
      }
    }
    if (!supervised && row.plans.some((plan) => 'outcome' in plan) && !row.plans.every((plan) => 'outcome' in plan)) {
      throw new Error('Held-out outcomes must cover every candidate or none');
    }
    rows.push(row);
  }
  if (!rows.length || new Set(rows.map((row) => row.id)).size !== rows.length) throw new Error('Input must contain tasks with unique IDs');
  return rows;
}

function texts(row: Row): string[] {
  // Role prefixes distinguish evidence words from candidate-plan words.
  const task = tokenize(row.task).map((word) => `task_${word}`).join(' ');
  const evidence = row.evidence.flatMap((item) => tokenize(item.text).map((word) => `evidence_${word}`)).join(' ');
  return row.plans.map((plan) => `${task} ${evidence} ${tokenize(plan.text).map((word) => `plan_${word}`).join(' ')}`);
}

/** Construct every operation before capture, replay, or inference. */
function bindings(manifest: Manifest): Operations {
  const width = manifest.dimensions;
  const space = { name: 'application.plan-text', dimensions: width };
  return {
    interpret: new VocabularyEncoder({ vocabulary: manifest.vocabulary, dimensions: width, output_space: space }),
    anticipate: new Decode({ architecture: 'mlp', input_space: space, hidden_dimensions: [width], output_dimensions: 1, output: 'outcome' }),
  };
}

function predict(operations: Operations, row: Row): Tensor {
  return operations.anticipate.call(operations.interpret.call(texts(row)) as Latent) as Tensor;
}

/** Python ``json.dumps(text)`` (``ensure_ascii=True``). */
function pyString(text: string): string {
  return JSON.stringify(text).replace(/[\u0080-\uffff]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
}

/** Python ``json.dumps(value, indent=2)`` (non-ASCII escaped). */
function write(path: string, value: unknown): void {
  const text = JSON.stringify(value, null, 2).replace(/[\u0080-\uffff]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
  writeFileSync(path, `${text}\n`, 'utf8');
}

async function collect(input: string, artifacts: string, dimensions = 24, seed = 7): Promise<unknown> {
  if (dimensions < 1) throw new Error('dimensions must be positive');
  const rows = readRows(input, true);
  const manifest: Manifest = {
    dimensions,
    vocabulary: [...new Set(rows.flatMap((row) => texts(row).flatMap(tokenize)))].sort(),
    training_ids: rows.map((row) => row.id),
    training_texts: rows.flatMap(texts),
    seed, experiences: [],
    observations: rows.map((row) => ({
      task_id: row.id, evidence_ids: row.evidence.map((item) => item.id), plans: row.plans.map((plan) => ({ id: plan.id, source: plan.source })),
    })),
  };
  mkdirSync(artifacts, { recursive: false });
  manualSeed(seed);
  const operations = bindings(manifest);
  await Trainer.fromOps(operations).saveCheckpoint(join(artifacts, 'initial-checkpoint'));
  for (const [index, row] of rows.entries()) {
    const session = trace();
    const scores = noGrad(() => session.run(() => predict(operations, row)));
    const targets = tensor(row.plans.map((plan) => [plan.outcome!]), { dtype: 'float32' });
    // Python ``json.dumps({'task': ..., 'outcomes': [...]})`` (default separators, ASCII escapes).
    const source = `{"task": ${pyString(row.id)}, "outcomes": [${row.plans.map((plan) => pyString(plan.source!)).join(', ')}]}`;
    session.supervise(scores, targets, { loss: 'mse', source });
    const name = `experience-${String(index).padStart(4, '0')}.json`;
    await session.save(join(artifacts, name), { operations, codecs: latentCodecs(), release: true });
    manifest.experiences.push(name);
  }
  // Preserve original evidence and feedback alongside normalized replay inputs.
  write(join(artifacts, 'observations.json'), rows);
  write(join(artifacts, 'manifest.json'), manifest);
  return { tasks: rows.length, observed_outcomes: rows.reduce((total, row) => total + row.plans.length, 0) };
}

async function restore(artifacts: string, checkpoint = 'trained-checkpoint'): Promise<[Manifest, Operations]> {
  const manifest = JSON.parse(readFileSync(join(artifacts, 'manifest.json'), 'utf8')) as Manifest;
  const operations = bindings(manifest);
  const optimizer = checkpoint === 'trained-checkpoint' ? (parameters: Tensor[]) => new Adam(parameters) : null;
  await Trainer.fromOps(operations, { optimizer }).loadCheckpoint(join(artifacts, checkpoint));
  return [manifest, operations];
}

async function train(artifacts: string, epochs = 100, lr = 0.01): Promise<unknown> {
  if (epochs < 1 || !Number.isFinite(lr) || lr <= 0) throw new Error('epochs and learning rate must be positive and finite');
  const [manifest, operations] = await restore(artifacts, 'initial-checkpoint');
  const sessions: Trace[] = [];
  for (const name of manifest.experiences) sessions.push(await loadExperience(join(artifacts, name), { operations, codecs: latentCodecs() }));
  const optimizer = new Adam(Object.values(operations).flatMap((operation) => operation.parameters()), { lr });
  const trainer = Trainer.fromOps(operations, { optimizer });
  const losses: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    losses.push(sessions.reduce((total, session) => total + trainer.step(session), 0) / sessions.length);
  }
  await trainer.saveCheckpoint(join(artifacts, 'trained-checkpoint'));
  const report = { first_loss: losses[0], last_loss: losses[losses.length - 1], updates: epochs * sessions.length, mean_loss_by_epoch: losses };
  write(join(artifacts, 'training.json'), report);
  return report;
}

async function evaluate(input: string, artifacts: string, checkpoint = 'trained-checkpoint'): Promise<unknown> {
  const rows = readRows(input, false);
  const [manifest, operations] = await restore(artifacts, checkpoint);
  const evaluated = rows.filter((row) => row.plans.some((plan) => 'outcome' in plan));
  const trainingIds = new Set(manifest.training_ids);
  const trainingTexts = new Set(manifest.training_texts);
  if (evaluated.some((row) => trainingIds.has(row.id) || texts(row).some((text) => trainingTexts.has(text)))) {
    throw new Error('Prediction inputs must be held out from collection (IDs and exact inputs)');
  }
  for (const operation of Object.values(operations)) operation.eval();
  const result = [];
  const errors: number[] = [];
  for (const row of rows) {
    // Anticipate the outcome of EVERY supplied plan before choosing any.
    const values = Array.from(noGrad(() => predict(operations, row)).toArray());
    if (!values.every(Number.isFinite)) throw new Error('Model produced nonfinite predicted outcomes');
    const candidates = row.plans.map((plan, index) => ({ id: plan.id, text: plan.text, predicted_outcome: values[index]! }));
    const selected = candidates.reduce((best, candidate) => (candidate.predicted_outcome > best.predicted_outcome ? candidate : best));
    result.push({
      task_id: row.id, task: row.task, evidence: row.evidence, candidates, selected_plan: selected.id,
      executed: false, uncertainty: 'Uncalibrated regression; scores are not confidence or causal effects.',
    });
    row.plans.forEach((plan, index) => { if ('outcome' in plan) errors.push((values[index]! - plan.outcome!) ** 2); });
  }
  return { tasks: result, heldout_mse: errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : null };
}

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    artifacts: { type: 'string' }, input: { type: 'string' }, dimensions: { type: 'string', default: '24' },
    seed: { type: 'string', default: '7' }, epochs: { type: 'string', default: '100' }, lr: { type: 'string', default: '0.01' },
    checkpoint: { type: 'string', default: 'trained-checkpoint' },
  },
});
const stage = positionals[0];
if (!stage || !['collect', 'train', 'predict'].includes(stage) || !values.artifacts) {
  throw new Error('usage: planLearning.ts {collect,train,predict} --artifacts DIR [--input FILE]');
}
if (stage !== 'train' && !values.input) throw new Error('--input is required for collect/predict');
if (!['initial-checkpoint', 'trained-checkpoint'].includes(values.checkpoint!)) throw new Error('--checkpoint must be initial-checkpoint or trained-checkpoint');
const report = stage === 'collect'
  ? await collect(values.input!, values.artifacts, Number(values.dimensions), Number(values.seed))
  : stage === 'train'
    ? await train(values.artifacts, Number(values.epochs), Number(values.lr))
    : await evaluate(values.input!, values.artifacts, values.checkpoint);
console.log(JSON.stringify(report, null, 2));
