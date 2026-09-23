/**
 * Offline owned vector-operation lifecycle (Python
 * examples/owned_vector_lifecycle.py): compose a VocabularyEncoder and a
 * Classify head, trace and supervise a prediction, persist the experience,
 * train with `Trainer.fromOps`, save each operation, reload them, and resume
 * training from the checkpoint.
 *
 *     npm run build
 *     node examples/ownedVectorLifecycle.ts [--epochs 5]
 *
 * Two authored cases do not measure quality.
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { trace } from 'tensorcode';
import { manualSeed, noGrad } from 'tensorcode/nn';
import { Classify, VocabularyEncoder, latentCodecs, type Prediction } from 'tensorcode/ops/vec';
import { Trainer, loadExperience } from 'tensorcode/training';

type Operations = { evidence: VocabularyEncoder; interpretation: Classify };

const epochsFlag = process.argv.indexOf('--epochs');
const epochs = epochsFlag >= 0 ? Number(process.argv[epochsFlag + 1]) : 5;
const output = mkdtempSync(join(tmpdir(), 'tensorcode-vector-'));

const texts = ['database refused', 'network loss'];
const targets = ['database', 'network'];

function predict(bound: Operations): Prediction {
  return bound.interpretation.call(bound.evidence.call(texts)) as Prediction;
}

try {
  manualSeed(7);
  const space = { name: 'example.reviewed-text', dimensions: 8 };
  // Every parameter exists before collection and optimizer construction.
  const operations: Operations = {
    evidence: new VocabularyEncoder({
      vocabulary: ['database', 'network', 'refused', 'loss'], dimensions: 8, output_space: space,
    }),
    interpretation: new Classify({ architecture: 'linear', input_space: space, labels: ['database', 'network'] }),
  };

  const session = trace();
  const prediction = session.run(() => predict(operations));
  session.supervise(prediction, targets, { source: 'two authored lifecycle cases' });
  const codecs = latentCodecs();
  await session.save(join(output, 'experience.json'), { operations, codecs, release: true });
  const experience = await loadExperience(join(output, 'experience.json'), { operations, codecs });

  const trainer = Trainer.fromOps(operations, { lr: 0.05 });
  const losses = trainer.fit([experience], { epochs });
  console.log(`loss ${losses[0]!.toFixed(4)} -> ${losses[losses.length - 1]!.toFixed(4)}`);
  await trainer.saveCheckpoint(join(output, 'checkpoint'));
  for (const [name, operation] of Object.entries(operations)) {
    operation.eval();
    await operation.savePretrained(join(output, name));
  }

  const restored: Operations = {
    evidence: await VocabularyEncoder.fromPretrained(join(output, 'evidence')),
    interpretation: await Classify.fromPretrained(join(output, 'interpretation')),
  };
  for (const operation of Object.values(restored)) operation.eval();
  const reloadEqual = noGrad(() => predict(restored).logits.equal(predict(operations).logits));

  const resumed = Trainer.fromOps(restored, { lr: 0.05 });
  await resumed.loadCheckpoint(join(output, 'checkpoint'));
  const replay = await loadExperience(join(output, 'experience.json'), { operations: restored, codecs });
  resumed.step(replay);

  console.log({
    updates: losses.length,
    reload_equal: reloadEqual,
    resume_update: resumed.steps === losses.length + 1,
    predicted: noGrad(() => predict(restored).values),
    limitations: 'Authored lifecycle fixture; no held-out quality evaluation.',
  });
} finally {
  rmSync(output, { recursive: true, force: true });
}
