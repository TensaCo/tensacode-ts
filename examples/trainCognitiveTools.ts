/**
 * Train owned Investigator and Planner models on pinned HotpotQA support
 * annotations, not invented outcomes (Python ``examples/train_cognitive_tools.py``).
 *
 *     npm run build
 *     node examples/trainCognitiveTools.ts --output /tmp/cognitive-tools --train-count 512 --validation-count 128
 *
 * Reads the first official distractor training rows and official validation
 * rows of `hotpotqa/hotpot_qa` at the pinned revision (in shard order, through
 * the Hugging Face dataset viewer's row API after checking that it serves that
 * revision) and records the shards' SHA-256 from the Hub. Validation examples
 * are fixed before fitting and never used for vocabulary, gradients or model
 * selection. Planner supervision is document relevance, NOT the observed
 * utility of executed plans. Shuffling uses Python's `random` algorithm.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { AdamW, F, PythonRandom, manualSeed, noGrad, stack, tensor, type Tensor } from 'tensorcode/nn';
import { Investigator, Planner } from 'tensorcode/tools';

const REVISION = '1908d6afbbead072334abe2965f91bd2709910ab';
const DATASET = 'hotpotqa/hotpot_qa';
const VIEWER = 'https://datasets-server.huggingface.co';

interface Candidate { id: string; text: string }
interface Row { id: string; question: string; candidates: Candidate[]; targets: number[]; target: number }
type Kind = 'investigator' | 'planner';
type Tool = Investigator | Planner;

function prepareRecord(row: any): Row {
  const titles = row.context.title as string[];
  const sentences = row.context.sentences as string[][];
  if (titles.length !== sentences.length || !titles.length) throw new Error('context titles and passages must align');
  const support = new Set(row.supporting_facts.title as string[]);
  const targets = titles.map((title) => Number(support.has(title)));
  if (!targets.some(Boolean)) throw new Error('record has no supporting passage');
  const candidates = titles.map((title, index) => ({ id: `doc-${index}`, text: `${title}\n${sentences[index]!.join(' ')}` }));
  return { id: row.id, question: row.question, candidates, targets, target: targets.indexOf(1) };
}

/** Python ``re.findall(r'\w+|[^\w\s]', text.casefold())``. */
function words(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) ?? [];
}

function vocabulary(records: Row[], limit = 20000): string[] {
  const counts = new Map<string, number>();
  for (const record of records) {
    for (const text of [record.question, ...record.candidates.map((candidate) => candidate.text)]) {
      for (const word of words(text)) counts.set(word, (counts.get(word) ?? 0) + 1);
    }
  }
  return [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)).slice(0, limit).map(([word]) => word);
}

function checkSplits(train: Row[], validation: Row[]): void {
  const a = train.map((row) => row.id);
  const b = validation.map((row) => row.id);
  if (new Set(a).size !== a.length || new Set(b).size !== b.length || a.some((id) => b.includes(id))) {
    throw new Error('training and validation must have unique disjoint question IDs');
  }
}

function inputs(record: Row, kind: Kind): Record<string, unknown> {
  // The question is evidence available before reading any candidate document.
  // Candidate passages are supplied options, never added as verified evidence.
  return kind === 'investigator'
    ? { question: record.question, evidence: [], hypotheses: record.candidates }
    : { goal: record.question, evidence: [], plans: record.candidates };
}

async function json(url: string, init?: RequestInit): Promise<any> {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

async function loadRecords(split: 'train' | 'validation', count: number, offset = 0): Promise<[Row[], { file: string; sha256: string }]> {
  const filename = `distractor/${split}-00000-of-0000${split === 'train' ? 2 : 1}.parquet`;
  const info = await json(`https://huggingface.co/api/datasets/${DATASET}/revision/${REVISION}`);
  const main = await json(`https://huggingface.co/api/datasets/${DATASET}`);
  if (info.sha !== REVISION || main.sha !== REVISION) {
    throw new Error(`the dataset viewer serves ${DATASET}@${main.sha}, not the pinned revision ${REVISION}`);
  }
  const [path] = await json(`https://huggingface.co/api/datasets/${DATASET}/paths-info/${REVISION}`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ paths: [filename] }),
  });
  const records: Row[] = [];
  for (let start = 0; start < count + offset; start += 100) {
    const length = Math.min(100, count + offset - start);
    const page = await json(`${VIEWER}/rows?dataset=${encodeURIComponent(DATASET)}&config=distractor&split=${split}&offset=${start}&length=${length}`);
    records.push(...(page.rows as { row: unknown }[]).map((item) => prepareRecord(item.row)));
    if ((page.rows as unknown[]).length < length) break;
  }
  if (records.length < count + offset) throw new Error('requested count exceeds available shard');
  return [records.slice(offset, count + offset), { file: filename, sha256: path.lfs.oid }];
}

/** Authored token-overlap comparison, not a learned cognitive model. */
function lexicalBaseline(records: Row[]) {
  let hit = 0;
  let recall = 0;
  const tokens = (text: string) => new Set(text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? []);
  for (const record of records) {
    const question = tokens(record.question);
    const order = record.candidates.map((_, index) => index).sort((a, b) => {
      const overlap = (index: number) => [...tokens(record.candidates[index]!.text)].filter((token) => question.has(token)).length;
      return overlap(b) - overlap(a) || a - b;
    });
    hit += record.targets[order[0]!]!;
    recall += order.slice(0, 2).reduce((total, index) => total + record.targets[index]!, 0) / record.targets.reduce((a, b) => a + b, 0);
  }
  return {
    method: 'unique casefold alphanumeric token overlap; candidate-order tie break',
    support_hit_at_1: hit / records.length, support_recall_at_2: recall / records.length,
  };
}

function evaluate(tool: Tool, records: Row[], kind: Kind, zeroWorkspace = false) {
  tool.eval();
  let loss = 0;
  let hit1 = 0;
  let recall2 = 0;
  const foundationLoss = Boolean(tool.config.foundation_config) || tool.config.encoder_type === 'foundation';
  noGrad(() => {
    for (const record of records) {
      const scores = tool.rank.compute(inputs(record, kind), { workspaceAblation: zeroWorkspace ? 'zero' : null }).scores;
      const truth = tensor(record.targets, { dtype: scores.dtype });
      const total = record.targets.reduce((a, b) => a + b, 0);
      if (kind === 'investigator') {
        const target = foundationLoss ? truth.div(total).unsqueeze(0) : tensor([record.target], { dtype: 'int64' });
        loss += F.crossEntropy(scores.unsqueeze(0), target).item();
      } else {
        loss += F.mseLoss(scores, truth).item();
      }
      const values = Array.from(scores.toArray());
      hit1 += record.targets[values.indexOf(Math.max(...values))]!;
      const top = scores.topk(Math.min(2, values.length)).indices.toArray();
      recall2 += Math.fround(Array.from(top).reduce((sum, index) => sum + record.targets[Number(index)]!, 0) / total);
    }
  });
  return { loss: loss / records.length, support_hit_at_1: hit1 / records.length, support_recall_at_2: recall2 / records.length };
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

async function run(options: {
  output: string; trainCount: number; validationCount: number; epochs: number; dimensions: number; seed: number; learningRate: number;
  foundation: string | null; foundationRevision: string | null; validationOffset: number; device: string; devCount: number;
}) {
  const { output, trainCount, validationCount, epochs, dimensions, seed, learningRate, foundation, foundationRevision, validationOffset, devCount } = options;
  manualSeed(seed);
  mkdirSync(output, { recursive: true });
  const [train, trainSource] = await loadRecords('train', trainCount);
  const [validation, validationSource] = await loadRecords('validation', validationCount, validationOffset);
  checkSplits(train, validation);
  const development = devCount ? (await loadRecords('train', devCount, trainCount))[0] : [];
  if (development.length) {
    checkSplits(train, development);
    checkSplits(development, validation);
  }
  const config = { vocabulary: vocabulary(train), dimensions, slots: 4, steps: 2, max_tokens: 128 };
  const { vocabulary: _vocabulary, ...rest } = config;
  const manifest: Record<string, any> = {
    dataset: DATASET, revision: REVISION, train_source: trainSource, validation_source: validationSource,
    train_ids: train.map((row) => row.id), validation_ids: validation.map((row) => row.id), development_ids: development.map((row) => row.id),
    seed, epochs, learning_rate: learningRate, batch_size: 8, config: rest, foundation, foundation_revision: foundationRevision,
    validation_offset: validationOffset, device: options.device,
    target_policy: foundation ? 'uniform over all annotated supporting titles' : 'first supporting title',
    vocabulary_size: foundation ? null : config.vocabulary.length,
    vocabulary_source: foundation ? 'inherited pinned foundation tokenizer' : 'training subset only',
    selection: 'first N official rows, fixed before fitting; final epoch, no validation selection',
    limitations: ['Investigator pilot used first supporting title; foundation runs use uniform support distribution. Hit/recall accept any supporting title.',
      'Planner targets are human document support annotations, a relevance proxy, not executed action outcomes.',
      'Candidates are supplied passages; no hypothesis generation, general planning, or general cognition is established.'],
    lexical_baseline: lexicalBaseline(validation), results: {},
  };
  const classes: [Kind, typeof Investigator | typeof Planner][] = [['investigator', Investigator], ['planner', Planner]];
  for (const [kind, Class] of classes) {
    manualSeed(seed);
    const tool: Tool = foundation
      ? await (Class as typeof Investigator).fromFoundation(foundation, {
        revision: foundationRevision, options: { freeze_foundation: true, cache_records: 2048, dimensions, slots: 4, steps: 2, max_tokens: 128 },
      }) as Tool
      : new Class(config);
    const optimizer = new AdamW(tool.parameters(), { lr: learningRate });
    const before = evaluate(tool, validation, kind);
    const history: number[] = [];
    const developmentHistory: unknown[] = [];
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      tool.train();
      const order = train.map((_, index) => index);
      new PythonRandom(seed + epoch).shuffle(order);
      let total = 0;
      for (let start = 0; start < order.length; start += 8) {
        const batch = order.slice(start, start + 8).map((index) => train[index]!);
        optimizer.zeroGrad();
        const losses = batch.map((row) => {
          const sum = row.targets.reduce((a, b) => a + b, 0);
          const target = kind === 'investigator' ? (foundation ? row.targets.map((value) => value / sum) : row.target) : row.targets;
          return tool.loss(inputs(row, kind), target);
        });
        const loss = stack(losses).mean();
        loss.backward();
        clipGradNorm(tool.parameters(), 1);
        optimizer.step();
        total += loss.item() * batch.length;
      }
      history.push(total / train.length);
      if (development.length) developmentHistory.push(evaluate(tool, development, kind));
      console.log(JSON.stringify({ tool: kind, epoch: epoch + 1, training_loss: history[history.length - 1] }));
    }
    const after = evaluate(tool, validation, kind);
    const ablation = evaluate(tool, validation, kind, true);
    const folder = join(output, kind);
    await tool.savePretrained(folder);
    const restored = await (Class as typeof Investigator).fromPretrained(folder) as Tool;
    const reloaded = evaluate(restored, validation, kind);
    if (JSON.stringify(reloaded) !== JSON.stringify(after)) throw new Error('saved model predictions differ after reload');
    const result = { before, after, zero_workspace: ablation, reloaded, training_loss: history, development_metrics: developmentHistory };
    manifest.results[kind] = result;
    writeFileSync(join(folder, 'evaluation.json'), `${JSON.stringify(result, null, 2)}\n`);
    const name = Class === Investigator ? 'Investigator' : 'Planner';
    writeFileSync(join(folder, 'README.md'), `---\nlibrary_name: tensorcode\ndatasets:\n- hotpotqa/hotpot_qa\ntags:\n- tensorcode\n- experimental\n---\n\n# TensorCode ${name} support relevance prototype\n\n`
      + 'Owned encoder, shared differentiable workspace and candidate scoring head; the manifest identifies inherited foundation weights when used. '
      + 'This checkpoint ranks supplied HotpotQA documents using human supporting-fact annotations. It is not a general cognitive agent. '
      + 'Planner scores are relevance proxies, not measured plan utility. Investigator target policy is in the manifest; foundation runs supervise all supporting titles. '
      + 'Newly built vocabulary and gradients use training questions only; foundation tokenizer assets are inherited; held-out official validation IDs are fixed before fitting. '
      + 'The final epoch is saved without validation-based selection. Inherited foundation models, if used, are pinned in the manifest.\n\n'
      + `Load with \`from tensorcode.tools.${kind} import ${name}\` then \`${name}.from_pretrained(path_or_repo)\`. `
      + 'Inputs contain question/hypotheses for Investigator or goal/plans for Planner; each candidate has id/text. See the TensorCode example for the complete schema.\n\n'
      + `## Measured results\n\n\`\`\`json\n${JSON.stringify(result, null, 2)}\n\`\`\`\n\n`
      + 'Data, split IDs, hashes, hyperparameters and limitations are recorded in training-manifest.json. '
      + 'Attention is learned routing, not proof of factual support. Scores are uncalibrated. '
      + 'Compare the zero-workspace ablation and authored lexical baseline before attributing quality to the workspace.\n\n'
      + `## Authored lexical comparison\n\n\`\`\`json\n${JSON.stringify(manifest.lexical_baseline, null, 2)}\n\`\`\`\n`);
  }
  writeFileSync(join(output, 'training-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  for (const kind of Object.keys(manifest.results)) writeFileSync(join(output, kind, 'training-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

const { values } = parseArgs({
  options: {
    output: { type: 'string' }, 'train-count': { type: 'string', default: '512' }, 'validation-count': { type: 'string', default: '128' },
    epochs: { type: 'string', default: '8' }, dimensions: { type: 'string', default: '32' }, seed: { type: 'string', default: '17' },
    'learning-rate': { type: 'string', default: '0.003' }, foundation: { type: 'string' }, 'foundation-revision': { type: 'string' },
    'validation-offset': { type: 'string', default: '0' }, device: { type: 'string', default: 'cpu' }, 'dev-count': { type: 'string', default: '0' },
  },
});
if (!values.output) throw new Error('--output is required');
const numbers = {
  trainCount: Number(values['train-count']), validationCount: Number(values['validation-count']), epochs: Number(values.epochs),
  dimensions: Number(values.dimensions), seed: Number(values.seed), learningRate: Number(values['learning-rate']),
  validationOffset: Number(values['validation-offset']), devCount: Number(values['dev-count']),
};
if (numbers.validationOffset < 0 || numbers.devCount < 0) throw new Error('offset and development count must be nonnegative');
if (values.foundation && !values['foundation-revision']) throw new Error('foundation loading requires an explicit pinned revision');
if (Math.min(numbers.trainCount, numbers.validationCount, numbers.epochs, numbers.dimensions) < 1 || !(numbers.learningRate > 0) || !Number.isFinite(numbers.learningRate)) {
  throw new Error('counts, epochs, dimensions and learning rate must be positive');
}
console.log(JSON.stringify(await run({
  output: values.output, ...numbers, foundation: values.foundation ?? null, foundationRevision: values['foundation-revision'] ?? null, device: values.device!,
}), null, 2));
