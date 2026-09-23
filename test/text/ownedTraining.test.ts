/**
 * Cross-module part of ``tests/text/test_owned_operations.py``
 * (``test_owned_loss_artifact_and_training_restart``): owned text operations
 * trained through ``Trainer.fromTool`` with experience files and checkpoints.
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import * as text from '../../src/ops/text/index.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import type { Tensor } from '../../src/nn/index.js';

const FOUNDATION = 'test/fixtures/text/foundation';
const VALUE = Object.freeze([new text.Message('user', 'question')]);
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-text-training-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const CASES: [string, any, Record<string, unknown>, unknown][] = [
  ['Transform', text.Transform, {}, 'answer'],
  ['Classify', text.Classify, { labels: ['yes', 'no'] }, { label: 'yes', distribution: null, confidence: null, abstained: false }],
  ['Score', text.Score, { rubric: ['bad', 'good'] }, { score: 1, distribution: null, confidence: null, abstained: false }],
  ['Decide', text.Decide, { options: ['yes', 'no'], decoding: 'likelihood' }, { choice: 'yes', distribution: null, confidence: null, abstained: false }],
  ['Retrieve', text.Retrieve, { items: { a: 'answer' } }, { keys: ['a'], scores: null, abstained: false }],
];

describe('owned text operations train through Trainer.fromTool', () => {
  for (const [name, cls, options, target] of CASES) {
    it(`${name}: experience, checkpoint and deterministic restart`, async () => {
      const op = await cls.fromFoundation(FOUNDATION, { config: { ...options, generation: { max_new_tokens: 2 } } });
      await op.savePretrained(join(scratch, `${name}-model`));
      const restored = await cls.fromPretrained(join(scratch, `${name}-model`));
      const trainer = Trainer.fromTool(restored, { lr: 0.001 });
      let session = trainer.capture(VALUE, target, { source: 'authored-test' });
      const operations = trainer.operations;
      const path = join(scratch, `${name}-session.json`);
      await session.save(path, { operations, codecs: { message: text.Message } });
      session = await loadExperience(path, { operations, codecs: { message: text.Message } });
      const before = (restored.loss(VALUE, target) as Tensor).item();
      trainer.step(session);
      await trainer.saveCheckpoint(join(scratch, `${name}-checkpoint`), { progress: { batch: 1 } });
      trainer.step(session);
      const expected = new Map([...restored.stateDict()].map(([key, value]: [string, Tensor]) => [key, value.clone()]));
      expect(await trainer.loadCheckpoint(join(scratch, `${name}-checkpoint`))).toEqual({ batch: 1 });
      trainer.step(session);
      for (const [key, value] of expected) expect(restored.stateDict().get(key)!.toArray()).toEqual(value.toArray());
      expect((restored.loss(VALUE, target) as Tensor).item()).toBeLessThan(before);
    });
  }
});
