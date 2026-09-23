/** Port of ``tests/text/test_owned_operations.py`` (training-engine parts: see ``trainingOperation`` checks). */
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it, vi } from 'vitest';
import * as text from '../../src/ops/text/index.js';
import { SGD, type Tensor } from '../../src/nn/index.js';
import { trace } from '../../src/_internal/tracing.js';
import { isTrainableTool } from '../../src/_internal/contracts.js';
import { ValueError } from '../../src/errors.js';

/** Tiny seeded T5 + WordLevel tokenizer saved by transformers (``scripts/fixtures/text_fixtures.py``). */
const FOUNDATION = 'test/fixtures/text/foundation';
const VALUE = Object.freeze([new text.Message('user', 'question')]);
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-text-owned-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const OWNED: [string, any, Record<string, unknown>, unknown][] = [
  ['Transform', text.Transform, {}, 'answer'],
  ['Classify', text.Classify, { labels: ['yes', 'no'] }, { label: 'yes', distribution: null, confidence: null, abstained: false }],
  ['Score', text.Score, { rubric: ['bad', 'good'] }, { score: 1, distribution: null, confidence: null, abstained: false }],
  ['Decide', text.Decide, { options: ['yes', 'no'] }, { choice: 'yes', distribution: null, confidence: null, abstained: false }],
  ['Retrieve', text.Retrieve, { items: { a: 'answer' } }, { keys: ['a'], scores: null, abstained: false }],
];

describe('owned text operations', () => {
  for (const [name, cls] of OWNED) {
    it(`${name} constructor rejects providers`, () => {
      expect(() => new cls((messages: unknown) => 'answer')).toThrow(/config|JSON/);
    });
  }

  it('explicit provider factory', async () => {
    const op = text.Transform.fromModel((_messages: readonly text.Message[]) => 'answer');
    expect(op.call([new text.Message('user', 'question')]).at(-1)!.content).toBe('answer');
    expect(op.replayable).toBe(false);
    await expect(op.savePretrained('unused')).rejects.toThrow(/external/);
  });

  for (const [name, cls, options, target] of OWNED) {
    it(`${name}: loss, artifact and training restart`, async () => {
      const op = await cls.fromFoundation(FOUNDATION, { config: { ...options, generation: { max_new_tokens: 2 } } });
      const seen: Tensor[] = [];
      const encode = op.nativeModel.model.encode.bind(op.nativeModel.model);
      const spy = vi.spyOn(op.nativeModel.model, 'encode').mockImplementation((inputs: any) => {
        seen.push(inputs.inputIds.clone());
        return encode(inputs);
      });
      const loss: Tensor = op.loss(VALUE, target);
      loss.backward();
      expect(op.parameters().some((parameter: Tensor) => parameter.grad !== null && parameter.grad.abs().sum().item() > 0)).toBe(true);
      spy.mockRestore();
      expect(seen[0]!.tolist()).toEqual(op.nativeModel.inputs(op._request(VALUE, null)).inputIds.tolist());

      const path = join(scratch, `owned-${name}`);
      await op.savePretrained(path);
      const manifest = JSON.parse(readFileSync(join(path, 'tensorcode_config.json'), 'utf8'));
      expect(manifest.tool).toBe(cls.qualifiedName);
      expect(manifest.tool).toBe(`tensorcode.ops.text.${name.toLowerCase()}.${name}`);
      const restored = await cls.fromPretrained(path);
      expect(restored.loss(VALUE, target).item()).toBeCloseTo(loss.item(), 5);
      expect(restored.configuration()).toEqual(op.configuration());
      expect(restored.training).toBe(false);

      // Trainer contract (Python ``Trainer.from_tool``): bindings, joint inputs, objective capture.
      expect(isTrainableTool(restored)).toBe(true);
      expect(restored.trainingInputsIncludeTargets).toBe(true);
      const bindings = restored.operationBindings();
      expect(bindings.operation).toBe(restored);
      expect(bindings.objective).toBe(restored.trainingOperation);
      restored.train();
      const session = trace();
      const output = session.run(() => restored.trainingOperation.call({ inputs: VALUE, targets: target }));
      session.supervise(output, target, { loss: 'tool_objective', source: 'authored-test' });
      expect(session.calls.length).toBe(1);
      const replayed = session.replay(output) as Tensor;
      expect(replayed.item()).toBeCloseTo(loss.item(), 5);

      const optimizer = new SGD(restored.trainingOperation.parameters(), { lr: 0.05 });
      const before = restored.loss(VALUE, target).item();
      for (let step = 0; step < 3; step += 1) {
        optimizer.zeroGrad();
        (restored.trainingOperation.call({ inputs: VALUE, targets: target }) as Tensor).backward();
        optimizer.step();
      }
      expect(restored.loss(VALUE, target).item()).toBeLessThan(before);
      const state = new Map([...restored.stateDict()].map(([key, value]) => [key, value.clone()]));
      const reloaded = await cls.fromPretrained(path);
      reloaded.loadStateDict(state);
      expect(reloaded.loss(VALUE, target).item()).toBeCloseTo(restored.loss(VALUE, target).item(), 6);
    });
  }

  it('objective accepts a conditioning envelope', async () => {
    const op = await text.Transform.fromFoundation(FOUNDATION);
    const context = { policy: [new text.Message('system', 'be brief')] };
    const direct = op.loss(VALUE, 'answer', { context }).item();
    expect((op.trainingOperation.call({ inputs: { value: VALUE, context }, targets: 'answer' }) as Tensor).item()).toBeCloseTo(direct, 6);
    expect(() => op.trainingOperation.call({ inputs: VALUE })).toThrow(ValueError);
    expect(op.trainingOperation.operationIdentity()).toBe('tensorcode.ops.text.transform.Transform.objective');
    expect(op.trainingOperation.replayable).toBe(true);
  });

  it('owned structured targets and generation are strict', async () => {
    const op = await text.Classify.fromFoundation(FOUNDATION, { config: { labels: ['yes', 'no'], generation: { max_new_tokens: 1 } } });
    expect(() => op.loss(VALUE, 'yes')).toThrow(/mapping/);
    expect(() => op.loss(VALUE, { label: 'yes', abstained: false })).toThrow(text.InvalidModelOutput);
    expect(() => op.loss(VALUE, { label: 'yes', abstained: false })).toThrow(/fields/);
    expect(() => op.loss(VALUE, { label: 'other', distribution: null, confidence: null, abstained: false })).toThrow(/configured labels/);
    // A tiny random model cannot manufacture a valid configured JSON answer.
    expect(() => op.call(VALUE)).toThrow(text.InvalidModelOutput);
    const transform = await text.Transform.fromFoundation(FOUNDATION);
    expect(() => transform.loss(VALUE, { text: 'x' })).toThrow(/text targets must be a string/);
  });

  it('owned config rejects unknown fields and class mismatch', async () => {
    const op = await text.Transform.fromFoundation(FOUNDATION);
    expect(() => new text.Transform({ ...op.configuration(), old_model: 'discarded' })).toThrow(/Unknown config fields: \['old_model'\]/);
    await op.savePretrained(join(scratch, 'owned'));
    await expect(text.Classify.fromPretrained(join(scratch, 'owned'))).rejects.toThrow(/tool/);
    expect(() => new text.Transform({ ...op.configuration(), generation: { beams: 2 } })).toThrow(/unsupported generation setting/);
    const copy = new text.Transform(op.configuration());
    expect(copy.configuration()).toEqual(op.configuration());
  });

  it('external transform configuration tracks instructions', () => {
    const provider = (_messages: readonly text.Message[]) => 'answer';
    const first = text.Transform.fromModel(provider, { instructions: 'Summarize' });
    const second = text.Transform.fromModel(provider, { instructions: 'Translate' });
    expect(first.configuration()).not.toEqual(second.configuration());
    expect(first.configuration()).toEqual({ type: 'text_transform', instructions: 'Summarize', model: { type: 'js:provider' } });
  });

  it('native generation runs in evaluation mode and restores training mode', async () => {
    const op = (await text.Transform.fromFoundation(FOUNDATION)).train();
    const modes: boolean[] = [];
    const step = op.nativeModel.model.decodeStep.bind(op.nativeModel.model);
    vi.spyOn(op.nativeModel.model, 'decodeStep').mockImplementation((...args: Parameters<typeof step>) => {
      modes.push(op.nativeModel.model.training);
      return step(...args);
    });
    const request = op._request(VALUE, null);
    await Promise.all([op.acall(VALUE), op.acall(VALUE), Promise.resolve().then(() => op.nativeModel.loss(request, 'answer'))]);
    expect(modes.length).toBeGreaterThan(0);
    expect(modes.every((mode) => mode === false)).toBe(true);
    expect(op.nativeModel.model.training).toBe(true);
    expect(op.nativeModel.model.modules().every((module: { training: boolean }) => module.training)).toBe(true);
  });

  it('replayability follows sampling', async () => {
    expect((await text.Transform.fromFoundation(FOUNDATION)).replayable).toBe(true);
    expect((await text.Transform.fromFoundation(FOUNDATION, { config: { generation: { do_sample: true } } })).replayable).toBe(false);
    await expect(text.Transform.fromFoundation(FOUNDATION, { config: { tokenizer: {} } })).rejects.toThrow(/foundation config/);
  });
});
