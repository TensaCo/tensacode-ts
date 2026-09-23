/**
 * Whole-number float hyperparameters (``lr=1.0``, ``weight_decay=0.0``) keep
 * their Python kind through TypeScript checkpoints (fixtures:
 * scripts/fixtures/optimizer_kinds_fixtures.py).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { float, int } from '../../src/index.js';
import { AdamW, Linear, SGD, manualSeed, type Optimizer, type Tensor } from '../../src/nn/index.js';
import { saveCheckpoint } from '../../src/_internal/training/checkpoint.js';
import { Trainer } from '../../src/training/index.js';
import { TensorAdapter as Transform } from '../../src/_internal/vec/adapter.js';
import { scratchDirectory } from './helpers.js';

const fixtures = new URL('../fixtures/training/', import.meta.url).pathname;
const directory = scratchDirectory('tensorcode-optimizer-kinds-');

const OPTIMIZERS: Record<string, (params: Iterable<Tensor>) => Optimizer> = {
  adamw: (params) => new AdamW(params, { lr: float(1), betas: [0.5, 0.75], weightDecay: float(0) }),
  sgd: (params) => new SGD(params, { lr: float(1), momentum: float(0), dampening: int(0), weightDecay: float(0) }),
};

describe('optimizer hyperparameter kinds', () => {
  it.each(Object.keys(OPTIMIZERS))('%s: float() markers write the bytes Python writes', async (name) => {
    manualSeed(5);
    const head = new Transform(new Linear(2, 2));
    const path = join(directory(), 'checkpoint.json');
    await saveCheckpoint(path, { operations: { head }, optimizer: OPTIMIZERS[name]!(head.parameters()) });
    expect(readFileSync(path, 'utf8')).toBe(readFileSync(join(fixtures, `optimizer_kinds_${name}.json`), 'utf8'));
  });

  it.each(Object.keys(OPTIMIZERS))('%s: a Python checkpoint re-saves byte for byte', async (name) => {
    const head = new Transform(new Linear(2, 2));
    const learner = Trainer.fromOps({ head }, { optimizer: (params) => (name === 'sgd' ? new SGD(params, { lr: 0.5 }) : new AdamW(params, { lr: 0.5 })) });
    const source = join(fixtures, `optimizer_kinds_${name}.json`);
    await learner.loadCheckpoint(source);
    const path = join(directory(), 'resaved.json');
    await saveCheckpoint(path, { operations: { head }, optimizer: learner.optimizer });
    expect(readFileSync(path, 'utf8')).toBe(readFileSync(source, 'utf8'));
  });
});
