/**
 * Trace a small tensor program, supervise it, persist the experience, train
 * with `Trainer.fromOps`, save a complete checkpoint and resume it in a fresh
 * process.
 *
 *     npm run build
 *     node examples/traceAndTrain.ts            # Node >= 22.18 runs .ts directly
 *     npx tsx examples/traceAndTrain.ts         # or with tsx
 *
 * The second phase (`resume`) is started automatically in a child process: it
 * constructs fresh, randomly initialized operations, restores the checkpoint
 * (weights, optimizer, module modes, steps, progress and random generator) and
 * continues training from the saved experiences.
 */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { trace, type Trace } from 'tensorcode';
import { Adam, Linear, manualSeed, noGrad, tensor, type Tensor } from 'tensorcode/nn';
import { ModuleOperation, type Context } from 'tensorcode/ops';
import { Trainer, loadExperience } from 'tensorcode/training';

/** A traced, replayable linear scorer over two features with named labels. */
class Scorer extends ModuleOperation<Tensor, Tensor> {
  // A stable persisted identity: experiences and checkpoints bind by configuration.
  static override readonly qualifiedName: string = 'examples.Scorer';
  readonly linear: Linear;
  readonly labels: readonly string[] = ['negative', 'positive'];

  constructor() {
    super();
    this.linear = this.registerModule('linear', new Linear(2, 2));
  }

  override get replayable(): boolean {
    return true;
  }

  configuration(): Record<string, unknown> {
    return { labels: [...this.labels], features: 2 };
  }

  forward(value: Tensor, context: Context | null): Tensor {
    void context;
    return this.linear.forward(value);
  }
}

const EXAMPLES: [number[], number][] = [[[-2, -1], 0], [[-1, -0.5], 0], [[1, 0.5], 1], [[2, 1.5], 1]];

async function train(directory: string): Promise<void> {
  manualSeed(7);
  const scorer = new Scorer();
  // 1. Trace each prediction and attach an explicit, sourced target.
  const sessions: Trace[] = EXAMPLES.map(([features, label], index) => {
    const session = trace();
    const logits = session.run(() => scorer.call(tensor(features)));
    session.supervise(logits, label, { loss: 'cross_entropy', source: `review:${index}` });
    return session;
  });
  // 2. Persist portable experience files (tensorcode.experience v1 JSON).
  for (const [index, session] of sessions.entries()) {
    await session.save(join(directory, `experience-${index}.json`), { operations: { scorer }, release: true });
  }
  // 3. Load them back bound to the live operation and train.
  const loaded = await Promise.all(EXAMPLES.map((_, index) => loadExperience(join(directory, `experience-${index}.json`), { operations: { scorer } })));
  const trainer = Trainer.fromOps({ scorer }, { optimizer: (params) => new Adam(params, { lr: 0.05 }) });
  const losses = trainer.fit(loaded, { epochs: 10 });
  console.log(`trained ${trainer.steps} steps: loss ${losses[0]!.toFixed(4)} -> ${losses[losses.length - 1]!.toFixed(4)}`);
  // 4. Save complete continuation state and record where we are.
  await trainer.saveCheckpoint(join(directory, 'checkpoint'), { progress: { epoch: 10 } });
  const expected = trainer.fit(loaded, { epochs: 1 });
  console.log(`next epoch in this process: ${expected.map((value) => value.toFixed(6)).join(', ')}`);
  // 5. Resume in a fresh process from the checkpoint alone.
  const script = fileURLToPath(import.meta.url);
  const child = spawnSync(process.execPath, [...process.execArgv, script, 'resume', directory], { encoding: 'utf8' });
  process.stdout.write(child.stdout);
  process.stderr.write(child.stderr);
  if (child.status !== 0) throw new Error('resume process failed');
  const resumed = child.stdout.trim().split('\n').pop()!.replace('resumed epoch: ', '');
  console.log(resumed === expected.map((value) => value.toFixed(6)).join(', ')
    ? 'resumed training matches the uninterrupted run exactly' : 'resumed training differs');
}

async function resume(directory: string): Promise<void> {
  manualSeed(12345); // Different initialization; the checkpoint restores everything.
  const scorer = new Scorer();
  const trainer = Trainer.fromOps({ scorer }, { optimizer: (params) => new Adam(params, { lr: 0.05 }) });
  const progress = await trainer.loadCheckpoint(join(directory, 'checkpoint'));
  const loaded = await Promise.all(EXAMPLES.map((_, index) => loadExperience(join(directory, `experience-${index}.json`), { operations: { scorer } })));
  console.log(`restored progress ${JSON.stringify(progress)} at step ${trainer.steps}`);
  const probabilities = noGrad(() => scorer.call(tensor([1.5, 1])).softmax(-1).toArray());
  console.log(`p(positive | [1.5, 1]) = ${probabilities[1]!.toFixed(4)}`);
  const losses = trainer.fit(loaded, { epochs: 1 });
  console.log(`resumed epoch: ${losses.map((value) => value.toFixed(6)).join(', ')}`);
}

const [mode, target] = process.argv.slice(2);
if (mode === 'resume') {
  await resume(target!);
} else {
  const directory = mkdtempSync(join(tmpdir(), 'tensorcode-example-'));
  try {
    await train(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}
