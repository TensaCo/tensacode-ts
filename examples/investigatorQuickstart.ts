/**
 * The 30-second example (Python README / docs/quickstart.md): train an
 * Investigator to rank two supplied hypotheses from log evidence, persist the
 * reviewed experience, save the model and reload it for prediction.
 *
 *     npm run build
 *     node examples/investigatorQuickstart.ts   # Node >= 22.18 runs .ts directly
 *
 * Runs offline on CPU in about a second. Two authored cases demonstrate the
 * lifecycle; they are not an evaluation of investigation competence.
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AdamW, manualSeed } from 'tensorcode/nn';
import { Investigator } from 'tensorcode/tools';
import { Trainer, loadExperience } from 'tensorcode/training';

const root = mkdtempSync(join(tmpdir(), 'tensorcode-investigator-'));
try {
  // All parameters exist before the optimizer is constructed.
  manualSeed(0);
  const model = new Investigator({
    vocabulary: ['database', 'network', 'connection', 'refused', 'packet', 'loss'],
    dimensions: 16, slots: 2, steps: 1,
  });
  const trainer = Trainer.fromTool(model, { optimizer: (params) => new AdamW(params, { lr: 0.01 }) });

  const hypotheses = [
    { id: 'database', text: 'database connection refused' },
    { id: 'network', text: 'network packet loss' },
  ];
  const incident = (logLine: string, source = 'log:1') => ({
    question: 'which component failed',
    evidence: [{ source_id: source, text: logLine }],
    hypotheses,
  });

  // Reviewed feedback, with explicit provenance, becomes training experience.
  const cases: [string, string][] = [['connection refused', 'database'], ['packet loss', 'network']];
  for (const [index, [line, target]] of cases.entries()) {
    const experience = trainer.capture(incident(line), target, { source: `review:${index + 1}` });
    await experience.save(join(root, `experience-${index}.json`), { operations: trainer.operations, release: true });
  }
  const experiences = await Promise.all(cases.map((_, index) =>
    loadExperience(join(root, `experience-${index}.json`), { operations: trainer.operations })));

  const losses = trainer.fit(experiences, { epochs: 30 });
  console.log(`loss ${losses[0]!.toFixed(4)} -> ${losses[losses.length - 1]!.toFixed(4)} over ${trainer.steps} steps`);

  await model.savePretrained(join(root, 'investigator'));
  await trainer.saveCheckpoint(join(root, 'training'), { progress: { epochs: 30 } });

  const restored = await Investigator.fromPretrained(join(root, 'investigator'));
  const receipt = restored.call(incident('packet loss', 'log:2'));
  console.log('selected:', receipt.selected_id); // network
  for (const candidate of receipt.candidates as { id: string; predicted_score: number }[]) {
    console.log(`  ${candidate.id}: ${candidate.predicted_score.toFixed(3)}`);
  }
} finally {
  rmSync(root, { recursive: true, force: true });
}
