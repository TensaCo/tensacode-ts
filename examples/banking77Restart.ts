/**
 * Persist supervised Banking77 traces, train after the process exits, then
 * evaluate (Python ``examples/banking77_restart.py``).
 *
 *     npm run build
 *     node examples/banking77Restart.ts --train banking_data/train.csv --test banking_data/test.csv \
 *       --artifacts /tmp/banking77 --output /tmp/banking77-results.json
 *
 * Use the official train/test CSV files
 * (https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).
 * Artifacts stay in `--artifacts`; this script never downloads data. Each
 * stage (capture, before, train, after) runs in a separate, sequentially
 * terminated process. Shuffling uses Python's `random.Random` algorithm, so
 * experience files, checkpoints and results match the Python example.
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { trace, version, type Trace } from 'tensorcode';
import { Adam, F, PythonRandom, manualSeed, noGrad, tensor, type Tensor } from 'tensorcode/nn';
import { Classify, VocabularyEncoder, latentCodecs, type Latent, type Prediction } from 'tensorcode/ops/vec';
import { Trainer, loadExperience } from 'tensorcode/training';

type Manifest = Record<string, any> & { vocabulary: string[]; labels: string[]; dimensions: number; test_sha256: string; experiences: string[] };
type Operations = { encode: VocabularyEncoder; classify: Classify };

/** Authored vocabulary policy matching the mechanical encoder tokenizer. */
function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) ?? [];
}

/** ``csv.DictReader`` rows (RFC 4180 quoting) as ``[text, category]``. */
function read(path: string): [string, string][] {
  const source = readFileSync(path, 'utf8');
  const records: string[][] = [];
  let field = '';
  let record: string[] = [];
  let quoted = false;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index]!;
    if (quoted) {
      if (char === '"' && source[index + 1] === '"') { field += '"'; index += 1; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { record.push(field); field = ''; }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && source[index + 1] === '\n') index += 1;
      record.push(field);
      records.push(record);
      field = '';
      record = [];
    } else field += char;
  }
  if (field || record.length) { record.push(field); records.push(record); }
  const [header, ...rows] = records.filter((row) => row.length > 1 || row[0] !== '');
  const text = header!.indexOf('text');
  const category = header!.indexOf('category');
  return rows.map((row) => [row[text] ?? '', row[category] ?? '']);
}

/** Python ``json.dumps(value, indent=2)`` (non-ASCII escaped). */
function write(path: string, value: unknown): void {
  const text = JSON.stringify(value, null, 2).replace(/[\u0080-\uffff]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
  writeFileSync(path, `${text}\n`);
}

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function bindings(manifest: Manifest): Operations {
  const space = { name: 'application.reviewed-text', dimensions: manifest.dimensions };
  return {
    encode: new VocabularyEncoder({ vocabulary: manifest.vocabulary, dimensions: manifest.dimensions, output_space: space }),
    classify: new Classify({ architecture: 'linear', input_space: space, labels: manifest.labels }),
  };
}

/** Python passes a tuple of texts; tuples are frozen arrays in TypeScript. */
function predict(operations: Operations, texts: string[]): Prediction {
  return operations.classify.call(operations.encode.call(Object.freeze(texts)) as Latent) as Prediction;
}

interface Args { train: string; test: string; artifacts: string; epochs: number; dimensions: number; batchSize: number; seed: number; lr: number }

async function stage(name: string, args: Args): Promise<void> {
  manualSeed(args.seed);
  const root = args.artifacts;
  const started = performance.now();
  let result: Record<string, unknown>;
  if (name === 'capture') {
    let training = read(args.train);
    const testing = read(args.test);
    const heldout = new Set(testing.map(([text]) => text.trim().toLowerCase()));
    const originalCount = training.length;
    training = training.filter(([text]) => !heldout.has(text.trim().toLowerCase()));
    const labels = [...new Set(training.map(([, label]) => label))].sort();
    // Vocabulary uses only official training text; held-out labels never guide fitting.
    const vocabulary = [...new Set(training.flatMap(([text]) => tokenize(text)))].sort();
    const manifest: Manifest = {
      vocabulary, labels, dimensions: args.dimensions, seed: args.seed, batch_size: args.batchSize, train_rows: training.length,
      test_rows: testing.length, excluded_overlap_rows: originalCount - training.length,
      train_sha256: sha256(args.train), test_sha256: sha256(args.test), experiences: [],
    };
    const operations = bindings(manifest);
    await Trainer.fromOps(operations).saveCheckpoint(join(root, 'initial-checkpoint'));
    new PythonRandom(args.seed).shuffle(training);
    for (let start = 0; start < training.length; start += args.batchSize) {
      const batch = training.slice(start, start + args.batchSize);
      const session = trace();
      const output = noGrad(() => session.run(() => predict(operations, batch.map(([text]) => text))));
      session.supervise(output, Object.freeze(batch.map(([, label]) => label)), { source: 'Banking77 official training labels' });
      const filename = `experience-${String(manifest.experiences.length).padStart(4, '0')}.json`;
      await session.save(join(root, filename), { operations, codecs: latentCodecs(), release: true });
      manifest.experiences.push(filename);
    }
    write(join(root, 'manifest.json'), manifest);
    result = { experiences: manifest.experiences.length, corrections: training.length };
  } else {
    const manifest = JSON.parse(readFileSync(join(root, 'manifest.json'), 'utf8')) as Manifest;
    const operations = bindings(manifest);
    if (name === 'train') {
      const optimizer = new Adam(Object.values(operations).flatMap((operation) => operation.parameters()), { lr: args.lr });
      await Trainer.fromOps(operations).loadCheckpoint(join(root, 'initial-checkpoint'));
      const experiences: Trace[] = [];
      for (const file of manifest.experiences) experiences.push(await loadExperience(join(root, file), { operations, codecs: latentCodecs() }));
      const trainer = Trainer.fromOps(operations, { optimizer });
      const rng = new PythonRandom(args.seed);
      const losses: number[] = [];
      for (let epoch = 0; epoch < args.epochs; epoch += 1) {
        rng.shuffle(experiences);
        losses.push(experiences.reduce((total, session) => total + trainer.step(session), 0) / experiences.length);
      }
      await trainer.saveCheckpoint(join(root, 'trained-checkpoint'));
      result = { epochs: args.epochs, mean_batch_loss_by_epoch: losses, loaded_experiences: experiences.length };
    } else {
      const checkpoint = name === 'before' ? 'initial-checkpoint' : 'trained-checkpoint';
      const optimizer = checkpoint === 'trained-checkpoint' ? (parameters: Tensor[]) => new Adam(parameters) : null;
      await Trainer.fromOps(operations, { optimizer }).loadCheckpoint(join(root, checkpoint));
      for (const operation of Object.values(operations)) operation.eval();
      const testing = read(args.test);
      if (sha256(args.test) !== manifest.test_sha256) throw new Error('Held-out data changed between stages');
      const lookup = new Map(manifest.labels.map((label, index) => [label, index]));
      let correct = 0;
      let totalLoss = 0;
      noGrad(() => {
        for (let start = 0; start < testing.length; start += 256) {
          const batch = testing.slice(start, start + 256);
          const output = predict(operations, batch.map(([text]) => text));
          const target = tensor(batch.map(([, label]) => lookup.get(label)!), { dtype: 'int64' });
          const predicted = output.logits.argmax(-1).toArray();
          const expected = target.toArray();
          for (let index = 0; index < predicted.length; index += 1) if (predicted[index] === expected[index]) correct += 1;
          totalLoss += F.crossEntropy(output.logits, target, { reduction: 'sum' }).item();
        }
      });
      result = { accuracy: correct / testing.length, cross_entropy: totalLoss / testing.length, rows: testing.length };
    }
  }
  Object.assign(result, { stage: name, pid: process.pid, seconds: (performance.now() - started) / 1000 });
  write(join(root, `${name}-receipt.json`), result);
}

function sizeOf(path: string): number {
  return readdirSync(path, { withFileTypes: true }).reduce((total, entry) => {
    const child = join(path, entry.name);
    return total + (entry.isDirectory() ? sizeOf(child) : entry.isFile() ? statSync(child).size : 0);
  }, 0);
}

const { values } = parseArgs({
  options: {
    train: { type: 'string' }, test: { type: 'string' }, artifacts: { type: 'string' }, output: { type: 'string' },
    epochs: { type: 'string', default: '20' }, dimensions: { type: 'string', default: '96' }, 'batch-size': { type: 'string', default: '128' },
    seed: { type: 'string', default: '7' }, lr: { type: 'string', default: '0.01' }, stage: { type: 'string' },
  },
});
if (!values.train || !values.test || !values.artifacts) throw new Error('--train, --test and --artifacts are required');
const args: Args = {
  train: resolve(values.train), test: resolve(values.test), artifacts: resolve(values.artifacts), epochs: Number(values.epochs),
  dimensions: Number(values.dimensions), batchSize: Number(values['batch-size']), seed: Number(values.seed), lr: Number(values.lr),
};
if (Math.min(args.epochs, args.dimensions, args.batchSize) < 1 || !(args.lr > 0)) {
  throw new Error('epochs, dimensions, batch size, and learning rate must be positive');
}
if (values.stage) {
  if (!['capture', 'before', 'train', 'after'].includes(values.stage)) throw new Error('unknown stage');
  await stage(values.stage, args);
} else {
  mkdirSync(args.artifacts, { recursive: false });
  const script = fileURLToPath(import.meta.url);
  const command = [process.execPath, script, '--train', args.train, '--test', args.test, '--artifacts', args.artifacts,
    '--epochs', String(args.epochs), '--dimensions', String(args.dimensions), '--batch-size', String(args.batchSize),
    '--seed', String(args.seed), '--lr', String(args.lr)];
  const receipts: Record<string, unknown>[] = [];
  for (const name of ['capture', 'before', 'train', 'after']) {
    const completed = spawnSync(command[0]!, [...command.slice(1), '--stage', name], { stdio: 'inherit' });
    if (completed.status !== 0) throw new Error(`stage ${name} failed with exit status ${completed.status}`);
    const receipt = JSON.parse(readFileSync(join(args.artifacts, `${name}-receipt.json`), 'utf8'));
    receipt.returncode = completed.status;
    receipt.terminated_before_next_stage = true;
    receipts.push(receipt);
    console.log(JSON.stringify(receipt));
  }
  const manifest = JSON.parse(readFileSync(join(args.artifacts, 'manifest.json'), 'utf8')) as Manifest;
  const { vocabulary, labels, experiences, ...rest } = manifest;
  const pick = (receipt: Record<string, unknown>) => ({ accuracy: receipt.accuracy, cross_entropy: receipt.cross_entropy });
  const report = {
    dataset: 'Banking77 official CSV splits',
    source: 'https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data',
    ...rest, vocabulary_size: vocabulary.length, label_count: labels.length,
    epochs: args.epochs, optimizer: 'Adam', learning_rate: args.lr, experience_files: experiences.length, processes: receipts,
    before: pick(receipts[1]!), after: pick(receipts[3]!), artifacts_bytes: sizeOf(args.artifacts),
    command, tensorcode: version, runtime: `node ${process.version}`,
    limitations: 'Supplied ground-truth labels and authored tokenizer/mean pooling. No pretrained model, autonomous discovery, or calibration claim. Fixed settings; held-out split used only for before/after evaluation. Session replay differentiates these local operations, not arbitrary code or remote services.',
  };
  write(join(args.artifacts, 'results.json'), report);
  if (values.output) {
    mkdirSync(dirname(resolve(values.output)), { recursive: true });
    write(values.output, report);
  }
  console.log(JSON.stringify(report, null, 2));
}
