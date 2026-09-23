/**
 * Fine-tune an owned Chatbot on explicit JSONL input/target/source records
 * (Python ``examples/train_chatbot.py``).
 *
 *     npm run build
 *     node examples/trainChatbot.ts --train reviewed-train.jsonl --test reviewed-test.jsonl --output /tmp/chatbot-run
 *     node examples/pretrainedChatbot.ts /tmp/chatbot-run/model --prompt 'Which evidence should we examine next?'
 *
 * Each row requires nonempty `id`, `input` and `target`; inputs must contain
 * only evidence available at inference time. The prescribed schedule never
 * selects checkpoints using held-out scores. A new workspace is initialized
 * over the explicitly selected seq2seq foundation (downloaded unless cached or
 * `--local-files-only`). Shuffling uses Python's `random` algorithm.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { parseArgs } from 'node:util';
import { AdamW, PythonRandom, manualSeed, noGrad, type Tensor } from 'tensorcode/nn';
import { Chatbot } from 'tensorcode/tools';
import { Trainer } from 'tensorcode/training';

type Record_ = { id: string; input: string; target: string };

function readRecords(path: string): Record_[] {
  const records = readFileSync(path, 'utf8').split('\n').filter((line) => line.trim()).map((line) => JSON.parse(line) as Record_);
  if (!records.length) throw new Error('Dataset must contain records');
  for (const row of records) {
    if (row === null || typeof row !== 'object' || ['id', 'input', 'target'].some((key) => typeof row[key as keyof Record_] !== 'string' || !row[key as keyof Record_].trim())) {
      throw new Error('Every record requires nonempty string id, input, target');
    }
  }
  if (new Set(records.map((row) => row.id)).size !== records.length) throw new Error('Duplicate source IDs');
  return records;
}

/** SQuAD-style normalization (Python ``string.punctuation``, articles, whitespace). */
function normalize(text: string): string {
  const stripped = text.toLowerCase().replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, '');
  return stripped.replace(/\b(a|an|the)\b/g, ' ').split(/\s+/).filter(Boolean).join(' ');
}

function scores(prediction: string, target: string): { exact_match: number; token_f1: number } {
  const predicted = normalize(prediction);
  const expected = normalize(target);
  const a = predicted ? predicted.split(' ') : [];
  const b = expected ? expected.split(' ') : [];
  const counts = new Map<string, number>();
  for (const token of b) counts.set(token, (counts.get(token) ?? 0) + 1);
  let common = 0;
  for (const token of a) {
    const left = counts.get(token) ?? 0;
    if (left > 0) { common += 1; counts.set(token, left - 1); }
  }
  const f1 = a.length && b.length ? (2 * common) / (a.length + b.length) : Number(JSON.stringify(a) === JSON.stringify(b));
  return { exact_match: Number(predicted === expected), token_f1: f1 };
}

function evaluate(model: Chatbot, records: Record_[], batchSize: number, ablation: 'bypass' | 'zero' | null = null) {
  const predictions: Record<string, unknown>[] = [];
  const losses: [number, number][] = [];
  model.eval();
  for (let offset = 0; offset < records.length; offset += batchSize) {
    const batch = records.slice(offset, offset + batchSize);
    const inputs = batch.map((row) => row.input);
    const targets = batch.map((row) => row.target);
    const loss = noGrad(() => model.lossBatch(inputs, targets, { workspaceAblation: ablation }).item());
    losses.push([loss, batch.length]);
    const outputs = noGrad(() => model.generateBatch(inputs, { workspaceAblation: ablation }));
    batch.forEach((row, index) => predictions.push({ id: row.id, target: row.target, prediction: outputs[index], ...scores(outputs[index]!, row.target) }));
  }
  const mean = (key: 'exact_match' | 'token_f1') => predictions.reduce((total, row) => total + (row[key] as number), 0) / records.length;
  return {
    examples: records.length, mean_batch_cross_entropy: losses.reduce((total, [loss, n]) => total + loss * n, 0) / records.length,
    exact_match: mean('exact_match'), token_f1: mean('token_f1'), predictions,
  };
}

/** ``torch.nn.utils.clip_grad_norm_(parameters, maxNorm)`` (2-norm over all gradients). */
function clipGradNorm(parameters: Tensor[], maxNorm: number): number {
  const grads = parameters.map((parameter) => parameter.grad).filter((grad): grad is Tensor => grad !== null);
  let squares = 0;
  for (const grad of grads) {
    let own = 0;
    for (const value of grad.data) own += value * value;
    squares += Math.fround(Math.sqrt(own)) ** 2;
  }
  const total = Math.fround(Math.sqrt(squares));
  const coefficient = Math.min(Math.fround(maxNorm / (total + 1e-6)), 1);
  if (coefficient < 1) for (const grad of grads) grad.data.forEach((value, index) => { grad.data[index] = Math.fround(value * coefficient); });
  return total;
}

const sha256 = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');

const { values } = parseArgs({
  options: {
    train: { type: 'string' }, test: { type: 'string' }, output: { type: 'string' },
    foundation: { type: 'string', default: 'google/flan-t5-small' }, revision: { type: 'string', default: '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab' },
    'local-files-only': { type: 'boolean', default: false }, epochs: { type: 'string', default: '3' }, 'batch-size': { type: 'string', default: '4' },
    lr: { type: 'string', default: '3e-5' }, 'workspace-lr': { type: 'string', default: '1e-3' }, seed: { type: 'string', default: '7' },
    device: { type: 'string', default: 'cpu' }, threads: { type: 'string', default: '4' }, 'max-input-tokens': { type: 'string', default: '512' },
    'max-target-tokens': { type: 'string', default: '64' }, slots: { type: 'string', default: '8' }, 'training-checkpoint': { type: 'boolean', default: false },
  },
});
if (!values.train || !values.test || !values.output) throw new Error('--train, --test and --output are required');
const epochs = Number(values.epochs);
const batchSize = Number(values['batch-size']);
const lr = Number(values.lr);
const workspaceLr = Number(values['workspace-lr']);
const seed = Number(values.seed);
const maxInput = Number(values['max-input-tokens']);
const maxTarget = Number(values['max-target-tokens']);
const slots = Number(values.slots);
if (Math.min(epochs, batchSize, Number(values.threads), maxInput, maxTarget, slots) <= 0 || Math.min(lr, workspaceLr) <= 0) {
  throw new Error('Counts and learning rates must be positive');
}
manualSeed(seed);
const random = new PythonRandom(seed);
const train = readRecords(values.train);
const test = readRecords(values.test);
if (train.some((row) => test.some((other) => other.id === row.id))) throw new Error('Train/test source IDs overlap');
if (train.some((row) => test.some((other) => other.input === row.input))) throw new Error('Train/test inputs overlap');
const output = values.output;
if (existsSync(output) && readdirSync(output).length) throw new Error('Output directory must be new or empty');
mkdirSync(output, { recursive: true });
const started = performance.now();
const model = await Chatbot.fromFoundation(values.foundation!, {
  revision: values.revision, localFilesOnly: values['local-files-only'],
  options: { max_input_tokens: maxInput, max_target_tokens: maxTarget, max_new_tokens: maxTarget, workspace: { slots, steps: 2 } },
});
const initialGate = model.memoryGate.item();
const report: Record<string, unknown> = {
  seed, foundation: values.foundation, revision: values.revision,
  schedule: { epochs, batch_size: batchSize, foundation_lr: lr, workspace_lr: workspaceLr },
  data: Object.fromEntries(([['train', values.train, train], ['test', values.test, test]] as const).map(([name, path, records]) => [name, {
    path, sha256: sha256(path), source_ids: records.map((row) => row.id),
  }])),
};
console.log('Evaluating fixed held-out split before training');
report.before = evaluate(model, test, batchSize);
const manifest = join(dirname(values.train), 'chat-data-manifest.json');
if (existsSync(manifest)) report.data_provenance = JSON.parse(readFileSync(manifest, 'utf8'));
await model.savePretrained(join(output, 'initial'));
const other = model.namedParameters().filter(([name]) => !name.startsWith('foundation.')).map(([, parameter]) => parameter);
const optimizer = new AdamW([{ params: model.foundation.parameters(), lr }, { params: other, lr: workspaceLr }]);
const trainer = Trainer.fromTool(model, { optimizer });
// Capture one explicit, portable feedback batch to demonstrate durable replay.
const sample = train.slice(0, batchSize);
const experience = trainer.capture(sample.map((row) => row.input), sample.map((row) => row.target), {
  source: `train JSONL ids: ${sample.map((row) => row.id).join(',')}`,
});
await experience.save(join(output, 'experience.json'), { operations: trainer.operations });
const epochLosses: number[] = [];
for (let epoch = 0; epoch < epochs; epoch += 1) {
  model.train();
  const order = train.map((_, index) => index);
  random.shuffle(order);
  let total = 0;
  for (let offset = 0; offset < order.length; offset += batchSize) {
    const batch = order.slice(offset, offset + batchSize).map((index) => train[index]!);
    optimizer.zeroGrad();
    const loss = model.lossBatch(batch.map((row) => row.input), batch.map((row) => row.target));
    loss.backward();
    clipGradNorm(model.parameters(), 1);
    optimizer.step();
    trainer.steps += 1;
    total += loss.item() * batch.length;
  }
  epochLosses.push(total / train.length);
  console.log(JSON.stringify({ epoch: epoch + 1, training_loss: epochLosses[epochLosses.length - 1] }));
}
report.epoch_losses = epochLosses;
const after = evaluate(model, test, batchSize);
const bypass = evaluate(model, test, batchSize, 'bypass');
report.after = after;
report.bypass = bypass;
report.zero = evaluate(model, test, batchSize, 'zero');
report.memory_gate = { before: initialGate, after: model.memoryGate.item() };
report.elapsed_seconds = (performance.now() - started) / 1000;
await model.savePretrained(join(output, 'model'));
// Persist optimizer/RNG separately; it is not part of the Hub weight package.
if (values['training-checkpoint']) await trainer.saveCheckpoint(join(output, 'training'), { progress: { epochs } });
const restored = await Chatbot.fromPretrained(join(output, 'model'));
const probe = [test[0]!.input];
report.reload_generation_equal = JSON.stringify(noGrad(() => model.generateBatch(probe))) === JSON.stringify(noGrad(() => restored.generateBatch(probe)));
writeFileSync(join(output, 'evaluation.json'), JSON.stringify(report, null, 2), 'utf8');
writeFileSync(join(output, 'model', 'evaluation.json'), JSON.stringify(report, null, 2), 'utf8');
const card = `---
library_name: tensorcode
pipeline_tag: text2text-generation
base_model: ${values.foundation}
license: apache-2.0
---
# TensorCode evidence-conditioned chatbot

Owned sequence encoder, relational slot workspace, and language decoder. Initialized
from \`${values.foundation}\` at \`${values.revision}\` and fine-tuned using the explicit data
hashes and source IDs in evaluation.json. Schedule: ${epochs} epochs; seed ${seed}.

This checkpoint is a narrow evidence-conditioned QA demonstration, not evidence of
general autonomous cognition. Foundation instruction behavior is inherited; training
updates both foundation and workspace. Data preparation may supply oracle supporting
passages: retrieval competence is not established by this experiment. The model does
not independently verify its generated claims. Workspace bypass and zero-evidence
ablations are reported separately; zero ablation alone does not show workspace utility.

Held-out exact match: ${after.exact_match.toFixed(4)}; token F1:
${after.token_f1.toFixed(4)}. Workspace-bypassed exact match:
${bypass.exact_match.toFixed(4)}. See evaluation.json for every prediction,
before/after metrics, source provenance and limitations. Published weights contain no
runtime conversation; training examples must be licensed and reviewed by their supplier.

\`\`\`python
from tensorcode.tools.chatbot import Chatbot
model = Chatbot.from_pretrained("PATH_OR_HUB_ID")
answer = model.generate_batch(["Question: ...\\nEvidence: ..."])[0]
\`\`\`
`;
writeFileSync(join(output, 'model', 'README.md'), card, 'utf8');
console.log(JSON.stringify({
  epoch_losses: epochLosses, elapsed_seconds: report.elapsed_seconds, memory_gate: report.memory_gate,
  reload_generation_equal: report.reload_generation_equal,
}));
