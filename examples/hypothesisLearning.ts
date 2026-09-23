/**
 * Learn revisable interpretations from caller-reviewed evidence sequences
 * (Python ``examples/hypothesis_learning.py``).
 *
 *     npm run build
 *     node examples/hypothesisLearning.ts collect --input reviewed.jsonl --artifacts /tmp/hypotheses \
 *       --hypothesis database --hypothesis network
 *     node examples/hypothesisLearning.ts train --artifacts /tmp/hypotheses --epochs 30
 *     node examples/hypothesisLearning.ts predict --input cases.jsonl --artifacts /tmp/hypotheses
 *
 * JSONL: `{"case_id": "...", "evidence": [{"source_id": "...", "text": "...",
 * "target": "reviewed hypothesis", "reviewer": "..."}, ...]}`. Prediction
 * inputs omit target/reviewer. Hypotheses are explicitly supplied at
 * collection time.
 *
 * The learned component is a small bag-of-words classifier over each growing
 * prefix, not general reasoning. Evidence order and provenance are retained in
 * receipts, but mean pooling cannot model order, negation or source
 * reliability. Probabilities are uncalibrated. Realization is an authored
 * display template. Artifacts are interchangeable with Python's.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { trace, type Trace } from 'tensorcode';
import { Adam, manualSeed, noGrad, type Tensor } from 'tensorcode/nn';
import { Classify, VocabularyEncoder, latentCodecs, type Latent, type Prediction } from 'tensorcode/ops/vec';
import { Trainer, loadExperience } from 'tensorcode/training';

type Step = { source_id: string; text: string; target?: string; reviewer?: string };
type Case = { case_id: string; evidence: Step[] };
type Manifest = { labels: string[]; vocabulary: string[]; dimensions: number; seed: number; experiences: { file: string; case_id: string; step: number }[] };
type Operations = { evidence: VocabularyEncoder; interpretation: Classify };

/** Authored vocabulary policy matching the mechanical encoder tokenizer. */
function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) ?? [];
}

/** Python ``json.dumps(value, indent=2)`` (non-ASCII escaped). */
function writeJson(path: string, value: unknown): void {
  const text = JSON.stringify(value, null, 2).replace(/[\u0080-\uffff]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
  writeFileSync(path, `${text}\n`, 'utf8');
}

/** Validate bounded input, preserving supplied source text and step order. */
function readCases(path: string, labels: string[] | null = null): Case[] {
  const cases: Case[] = [];
  const seen = new Set<string>();
  readFileSync(path, 'utf8').split('\n').forEach((line, index) => {
    if (!line.trim()) return;
    try {
      const value = JSON.parse(line) as Case;
      if (value === null || typeof value !== 'object' || Object.keys(value).sort().join() !== 'case_id,evidence') throw new Error('expected case_id and evidence');
      if (typeof value.case_id !== 'string' || !value.case_id.trim() || seen.has(value.case_id)) throw new Error('case_id must be a unique nonempty string');
      if (!Array.isArray(value.evidence) || value.evidence.length < 1 || value.evidence.length > 100) throw new Error('evidence must contain 1..100 steps');
      const sources = new Set<string>();
      const required = labels !== null ? ['reviewer', 'source_id', 'target', 'text'] : ['source_id', 'text'];
      for (const step of value.evidence) {
        if (step === null || typeof step !== 'object' || Object.keys(step).sort().join() !== required.join()) {
          throw new Error(`each step requires exactly [${required.map((key) => `'${key}'`).join(', ')}]`);
        }
        if (required.some((key) => typeof step[key as keyof Step] !== 'string' || !(step[key as keyof Step] as string).trim())) {
          throw new Error('step fields must be nonempty strings');
        }
        if (sources.has(step.source_id)) throw new Error('source_id must be unique within a case');
        if (step.text.length > 10000) throw new Error('evidence text exceeds 10000 characters');
        sources.add(step.source_id);
        if (labels !== null && !labels.includes(step.target!)) throw new Error('reviewed target is outside the hypothesis vocabulary');
      }
      seen.add(value.case_id);
      cases.push(value);
      if (cases.length > 10000) throw new Error('input exceeds 10000 cases');
    } catch (error) {
      throw new Error(`${path}:${index + 1}: ${(error as Error).message}`);
    }
  });
  if (!cases.length) throw new Error('input contains no cases');
  return cases;
}

/** Construct all public operations before collecting or replaying any input. */
function bindings(manifest: Manifest): Operations {
  const space = { name: 'application.reviewed-text', dimensions: manifest.dimensions };
  return {
    evidence: new VocabularyEncoder({ vocabulary: manifest.vocabulary, dimensions: manifest.dimensions, output_space: space }),
    interpretation: new Classify({ architecture: 'linear', input_space: space, labels: manifest.labels }),
  };
}

function infer(operations: Operations, text: string): Prediction {
  return operations.interpretation.call(operations.evidence.call(text) as Latent) as Prediction;
}

function* prefixes(value: Case): Generator<[number, Step, string]> {
  const texts: string[] = [];
  for (const [index, step] of value.evidence.entries()) {
    texts.push(step.text);
    yield [index, step, texts.join('\n')];
  }
}

async function collect(input: string, artifacts: string, labels: string[], dimensions = 24, seed = 7): Promise<unknown> {
  if (labels.length < 2 || new Set(labels).size !== labels.length || labels.some((label) => !label.trim())) {
    throw new Error('supply at least two unique nonempty hypotheses');
  }
  if (dimensions < 1) throw new Error('dimensions must be positive');
  const cases = readCases(input, labels);
  // Only training evidence builds the vocabulary. Reviewed targets never enter text.
  const vocabulary = [...new Set(cases.flatMap((item) => item.evidence.flatMap((step) => tokenize(step.text))))].sort();
  const manifest: Manifest = { labels, vocabulary, dimensions, seed, experiences: [] };
  manualSeed(seed);
  const operations = bindings(manifest);
  mkdirSync(artifacts, { recursive: false });
  await Trainer.fromOps(operations).saveCheckpoint(join(artifacts, 'initial-checkpoint'));
  for (const item of cases) {
    for (const [index, step, text] of prefixes(item)) {
      const session = trace();
      const output = noGrad(() => session.run(() => infer(operations, text)));
      session.supervise(output, step.target!, { source: step.reviewer! });
      const file = `experience-${String(manifest.experiences.length).padStart(6, '0')}.json`;
      await session.save(join(artifacts, file), { operations, codecs: latentCodecs(), release: true });
      manifest.experiences.push({ file, case_id: item.case_id, step: index });
    }
  }
  // Source IDs and exact original evidence remain available beside portable traces.
  writeJson(join(artifacts, 'evidence.json'), cases);
  writeJson(join(artifacts, 'manifest.json'), manifest);
  return { cases: cases.length, supervised_steps: manifest.experiences.length };
}

async function restore(artifacts: string, checkpoint = 'trained-checkpoint'): Promise<[Manifest, Operations]> {
  const manifest = JSON.parse(readFileSync(join(artifacts, 'manifest.json'), 'utf8')) as Manifest;
  const operations = bindings(manifest);
  const optimizer = checkpoint === 'trained-checkpoint' ? (parameters: Tensor[]) => new Adam(parameters) : null;
  await Trainer.fromOps(operations, { optimizer }).loadCheckpoint(join(artifacts, checkpoint));
  return [manifest, operations];
}

async function train(artifacts: string, epochs = 30, lr = 0.03): Promise<unknown> {
  if (epochs < 1 || !Number.isFinite(lr) || lr <= 0) throw new Error('epochs and finite learning rate must be positive');
  const [manifest, operations] = await restore(artifacts, 'initial-checkpoint');
  const experiences: Trace[] = [];
  for (const row of manifest.experiences) experiences.push(await loadExperience(join(artifacts, row.file), { operations, codecs: latentCodecs() }));
  const optimizer = new Adam(Object.values(operations).flatMap((operation) => operation.parameters()), { lr });
  const trainer = Trainer.fromOps(operations, { optimizer });
  const losses: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    losses.push(experiences.reduce((total, session) => total + trainer.step(session), 0) / experiences.length);
  }
  await trainer.saveCheckpoint(join(artifacts, 'trained-checkpoint'));
  const receipt = { epochs, mean_step_loss_by_epoch: losses, supervision: 'caller-reviewed targets', probabilities_calibrated: false };
  writeJson(join(artifacts, 'training.json'), receipt);
  return receipt;
}

async function predict(input: string, artifacts: string): Promise<unknown> {
  const cases = readCases(input);
  const [, operations] = await restore(artifacts);
  for (const operation of Object.values(operations)) operation.eval();
  const receipts = [];
  for (const item of cases) {
    let previous: string | null = null;
    for (const [index, , text] of prefixes(item)) {
      const prediction = noGrad(() => infer(operations, text));
      // Interpretation is chosen before the authored language rendering.
      const interpretation = prediction.value;
      const probabilities = Array.from(prediction.probabilities.toArray());
      receipts.push({
        case_id: item.case_id, step: index, evidence: item.evidence.slice(0, index + 1),
        distribution: Object.fromEntries(prediction.labels.map((label, position) => [label, probabilities[position]])),
        interpretation, revised: previous !== null && previous !== interpretation,
        realization: `Current interpretation: ${interpretation}.`, probabilities_calibrated: false,
      });
      previous = interpretation;
    }
  }
  return receipts;
}

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    input: { type: 'string' }, artifacts: { type: 'string' }, hypothesis: { type: 'string', multiple: true },
    dimensions: { type: 'string', default: '24' }, seed: { type: 'string', default: '7' },
    epochs: { type: 'string', default: '30' }, lr: { type: 'string', default: '0.03' },
  },
});
const stage = positionals[0];
if (!stage || !['collect', 'train', 'predict'].includes(stage) || !values.artifacts) {
  throw new Error('usage: hypothesisLearning.ts {collect,train,predict} --artifacts DIR [--input FILE] [--hypothesis NAME ...]');
}
let result: unknown;
if (stage === 'collect') {
  if (!values.input || !values.hypothesis) throw new Error('collect requires --input and --hypothesis');
  result = await collect(values.input, values.artifacts, values.hypothesis, Number(values.dimensions), Number(values.seed));
} else if (stage === 'train') {
  result = await train(values.artifacts, Number(values.epochs), Number(values.lr));
} else {
  if (!values.input) throw new Error('predict requires --input');
  result = await predict(values.input, values.artifacts);
}
console.log(JSON.stringify(result, null, 2));
