/**
 * End-to-end mirror of the Python quickstart (python/docs/quickstart.md):
 * initialize an Investigator, capture reviewed experience, persist and reload
 * it, train with Trainer.fit, save, reload the model in a fresh module
 * registry and predict. Imports go through the public entry modules.
 */
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AdamW, manualSeed } from '../../src/nn/index.js';
import { Investigator } from '../../src/tools/index.js';
import { Trainer, loadExperience } from '../../src/training/index.js';

const HYPOTHESES = [
  { id: 'database', text: 'database connection refused' },
  { id: 'network', text: 'network packet loss' },
];

function inputs(text: string, source: string): Record<string, unknown> {
  return { question: 'which component failed', evidence: [{ source_id: source, text }], hypotheses: HYPOTHESES };
}

const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

describe('quickstart lifecycle', () => {
  it('captures experience, trains, saves and predicts after a fresh load', async () => {
    const root = mkdtempSync(join(tmpdir(), 'tensorcode-quickstart-'));
    directories.push(root);

    // All parameters exist before the optimizer is constructed.
    manualSeed(7);
    const model = new Investigator({
      vocabulary: ['which', 'component', 'failed', 'database', 'network', 'connection', 'refused', 'packet', 'loss'],
      dimensions: 16,
      slots: 2,
      steps: 1,
    });
    const trainer = Trainer.fromTool(model, { optimizer: (params) => new AdamW(params, { lr: 0.01 }) });
    await model.savePretrained(join(root, 'initial'));

    const cases: [string, string][] = [['database connection refused', 'database'], ['network packet loss', 'network']];
    for (const [index, [text, target]] of cases.entries()) {
      const experience = trainer.capture(inputs(text, `observation:${index}`), target, { source: `authored-example:${index}` });
      await experience.save(join(root, `experience-${index}.json`), { operations: trainer.operations, release: true });
    }

    const files = readdirSync(root).filter((name) => /^experience-\d+\.json$/.test(name)).sort();
    expect(files).toEqual(['experience-0.json', 'experience-1.json']);
    const experiences = await Promise.all(files.map((name) => loadExperience(join(root, name), { operations: trainer.operations })));
    expect(experiences.map((experience) => experience.supervisions[0]!.source))
      .toEqual(['authored-example:0', 'authored-example:1']);

    const losses = trainer.fit(experiences, { epochs: 60 });
    expect(losses).toHaveLength(120);
    expect(losses.every(Number.isFinite)).toBe(true);
    const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
    // Loss decreases: the last epoch is far below the first.
    expect(mean(losses.slice(-2))).toBeLessThan(mean(losses.slice(0, 2)) * 0.5);
    expect(trainer.steps).toBe(120);

    await model.savePretrained(join(root, 'model'));
    await trainer.saveCheckpoint(join(root, 'training'), { progress: { epochs: 60 } });
    expect(existsSync(join(root, 'model', 'tensorcode_config.json'))).toBe(true);
    expect(existsSync(join(root, 'model', 'model.safetensors'))).toBe(true);
    expect(existsSync(join(root, 'training', 'training.json'))).toBe(true);

    model.eval();
    const expected = model.call(inputs('network packet loss', 'observation:new')) as Record<string, any>;
    expect(expected.selected_id).toBe('network');

    // Fresh load: a new module registry, as in a separate process.
    vi.resetModules();
    const fresh = await import('../../src/tools/index.js');
    expect(fresh.Investigator).not.toBe(Investigator);
    const restored = await fresh.Investigator.fromPretrained(join(root, 'model'));
    for (const [text, target] of cases) {
      const result = restored.call(inputs(text, 'observation:new')) as Record<string, any>;
      expect(result.selected_id).toBe(target);
    }
    const result = restored.call(inputs('network packet loss', 'observation:new')) as Record<string, any>;
    expect(result.candidates.map((candidate: any) => candidate.id)).toEqual(expected.candidates.map((candidate: any) => candidate.id));
    for (const [index, candidate] of (result.candidates as any[]).entries()) {
      expect(candidate.predicted_score).toBeCloseTo(expected.candidates[index].predicted_score, 5);
    }

    // The untrained initial artifact also reloads (it is a separate deployment).
    mkdirSync(join(root, 'scratch'), { recursive: true });
    const initial = await fresh.Investigator.fromPretrained(join(root, 'initial'));
    expect(initial.call(inputs('network packet loss', 'observation:new'))).toHaveProperty('candidates');
  });
});
