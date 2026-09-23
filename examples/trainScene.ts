/**
 * Train and evaluate an owned scene ranker on supplied image-and-candidate
 * JSONL (Python ``examples/train_scene.py``).
 *
 *     npm run build
 *     node examples/trainScene.ts --train scenes-train.jsonl --test scenes-test.jsonl --model /tmp/scene-model
 *     node examples/trainScene.ts --test scenes-test.jsonl --model /tmp/scene-model     # evaluate a saved model
 *
 * Each record: `image_path`, `source_id`, `question`, `candidates: [{id, text}]`,
 * `target` (a candidate ID). Images must be caller-licensed files; they are
 * decoded and resized exactly as Pillow's `Image.open(...).convert('RGB')
 * .resize((size, size))`. Reported patch coordinates refer to that resized
 * input, not the original photo. `--foundation`/`--revision` start from a
 * CLIP checkpoint instead of a fresh configuration. Shuffling uses Python's
 * `random` algorithm.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, isAbsolute, join } from 'node:path';
import { parseArgs } from 'node:util';
import { Adam, F, PythonRandom, manualSeed, noGrad, tensor, zeros, type Tensor } from 'tensorcode/nn';
import { Scene, type SceneInputs } from 'tensorcode/tools';
import { Trainer } from 'tensorcode/training';
// The TypeScript counterpart of ``PIL.Image.open`` (Pillow-exact decoding and resampling).
import { openImage } from 'tensorcode/ops/vec';

type Row = [SceneInputs, string];

function readRecords(path: string, size: number): Row[] {
  const records: Row[] = [];
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    const imagePath = isAbsolute(row.image_path) ? row.image_path : join(dirname(path), row.image_path);
    const image = openImage(imagePath).convert('RGB').resize([size, size]);
    const pixels = image.toTensor().to('float32').div(255);
    records.push([{ pixels, question: row.question, source_id: row.source_id, candidates: row.candidates }, row.target]);
  }
  if (!records.length) throw new Error('dataset must not be empty');
  return records;
}

function evaluate(tool: Scene, rows: Row[]) {
  const result: Record<string, unknown> = {};
  noGrad(() => {
    for (const mode of ['full', 'blank_image', 'zero_workspace', 'bypass_workspace']) {
      let correct = 0;
      let loss = 0;
      for (const [inputs, target] of rows) {
        const value = mode === 'blank_image' ? { ...inputs, pixels: zeros((inputs.pixels as Tensor).shape) } : inputs;
        const ablation = ({ zero_workspace: 'zero', bypass_workspace: 'bypass' } as Record<string, 'zero' | 'bypass'>)[mode] ?? null;
        const logits = tool.rank.compute(value, { workspaceAblation: ablation }).logits;
        const index = (inputs.candidates as { id: string }[]).map((item) => item.id).indexOf(target);
        const scores = Array.from(logits.toArray());
        correct += Number(scores.indexOf(Math.max(...scores)) === index);
        loss += F.crossEntropy(logits.unsqueeze(0), tensor([index], { dtype: 'int64' })).item();
      }
      result[mode] = { accuracy: correct / rows.length, cross_entropy: loss / rows.length, count: rows.length };
    }
  });
  return result;
}

/** ``torch.nn.utils.clip_grad_norm_(parameters, maxNorm)``. */
function clipGradNorm(parameters: Tensor[], maxNorm: number): void {
  const grads = parameters.map((parameter) => parameter.grad).filter((grad): grad is Tensor => grad !== null);
  let squares = 0;
  for (const grad of grads) {
    let own = 0;
    for (const value of grad.data) own += value * value;
    squares += Math.fround(Math.sqrt(own)) ** 2;
  }
  const coefficient = Math.min(Math.fround(maxNorm / (Math.fround(Math.sqrt(squares)) + 1e-6)), 1);
  if (coefficient < 1) for (const grad of grads) grad.data.forEach((value, index) => { grad.data[index] = Math.fround(value * coefficient); });
}

const { values } = parseArgs({
  options: {
    train: { type: 'string' }, test: { type: 'string' }, model: { type: 'string' }, foundation: { type: 'string' }, revision: { type: 'string' },
    device: { type: 'string', default: 'cpu' }, epochs: { type: 'string', default: '10' }, 'image-size': { type: 'string', default: '64' },
    seed: { type: 'string', default: '17' }, report: { type: 'string' },
  },
});
if (!values.test || !values.model) throw new Error('--test and --model are required');
const seed = Number(values.seed);
const epochs = Number(values.epochs);
const imageSize = Number(values['image-size']);
manualSeed(seed);
const random = new PythonRandom(seed);
const report: Record<string, unknown> = {
  seed, image_size: imageSize, foundation: values.foundation ?? null, foundation_revision: values.revision ?? null, epochs,
  limitation: 'Explicit candidate ranking; attention is not proof and this does not establish general scene understanding.',
};
const test = readRecords(values.test, imageSize);
if (values.train) {
  const rows = readRecords(values.train, imageSize);
  const trainSources = new Set(rows.map(([inputs]) => inputs.source_id as string));
  if (test.some(([inputs]) => trainSources.has(inputs.source_id as string))) throw new Error('training and evaluation image sources must be disjoint');
  const words = (value: string) => value.toLowerCase().match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) ?? [];
  const vocabulary = [...new Set(rows.flatMap(([inputs]) => [inputs.question as string, ...(inputs.candidates as { text: string }[]).map((item) => item.text)].flatMap(words)))].sort();
  let tool: Scene;
  if (values.foundation) {
    if (!values.revision) throw new Error('--foundation requires --revision');
    tool = await Scene.fromFoundation(values.foundation, { revision: values.revision });
  } else {
    tool = new Scene({ vocabulary, dimensions: 32, slots: 4, steps: 2, max_image_size: imageSize, patch_size: 8 });
  }
  // All image/text/workspace/scoring parameters already exist here.
  const trainer = Trainer.fromTool(tool, { optimizer: (parameters) => new Adam(parameters, { lr: 0.001 }) });
  report.before = evaluate(tool, test);
  const output = values.model;
  mkdirSync(output, { recursive: true });
  // Demonstrate durable feedback collection separately from published weights.
  const capture = trainer.capture(rows[0]![0], rows[0]![1], { source: `user-supplied:${values.train}` });
  await capture.save(join(dirname(output), `${basename(output)}-experience.json`), { operations: trainer.operations });
  const losses: number[] = [];
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    random.shuffle(rows);
    const epochLosses: number[] = [];
    for (const [inputs, target] of rows) {
      trainer.optimizer.zeroGrad();
      const loss = tool.loss(inputs, target);
      loss.backward();
      clipGradNorm(tool.parameters(), 1);
      trainer.optimizer.step();
      trainer.steps += 1;
      epochLosses.push(loss.item());
    }
    losses.push(epochLosses.reduce((a, b) => a + b, 0) / epochLosses.length);
  }
  report.losses = losses;
  await tool.savePretrained(output);
  await trainer.saveCheckpoint(join(dirname(output), `${basename(output)}-training`), { progress: { epochs } });
  report.train_count = rows.length;
}
const tool = await Scene.fromPretrained(values.model);
report.model_configuration = tool.configuration();
report.after_reload = evaluate(tool, test);
const rendered = JSON.stringify(report, null, 2);
if (values.report) writeFileSync(values.report, `${rendered}\n`);
console.log(rendered);
