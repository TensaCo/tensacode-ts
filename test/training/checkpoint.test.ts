/**
 * Cross-language checkpoint and experience interoperability against Python
 * fixtures (``scripts/fixtures/training_fixtures.py``).
 */
import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  Adam, Dropout, F, Linear, Sequential, Tensor, deserializeSafetensors, getRngState, manualSeed, ones, pythonRandom, rand, randn, tensor,
  zeros,
} from '../../src/nn/index.js';
import { Operation, type Context } from '../../src/ops/base.js';
import { trace } from '../../src/_internal/tracing.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import { saveCheckpoint } from '../../src/_internal/training/checkpoint.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson, fromJson, type TensorJson } from '../helpers/fixtures.js';
import { scratchDirectory, transform } from './helpers.js';

const scratch = scratchDirectory('tensorcode-interop-');
const fixtures = new URL('../fixtures/training/', import.meta.url).pathname;
const interop = fixtureJson('training/interop.json');

class External extends Operation<{ x: number; y: number }, Tensor> {
  static override readonly qualifiedName: string = 'training_fixtures.External';
  forward(value: { x: number; y: number }, context: Context | null): Tensor {
    void context;
    return tensor([value.x * 2, value.y]);
  }
}

class Note {
  static readonly recordFields = ['text', 'weight'] as const;
  constructor(readonly text: string, readonly weight: number) {
    Object.freeze(this);
  }
  static fromRecord(fields: Record<string, unknown>): Note {
    return new Note(fields.text as string, fields.weight as number);
  }
  toRecord(): Record<string, unknown> {
    return { text: this.text, weight: this.weight };
  }
}

function linear(state: Record<string, TensorJson>): Linear {
  const module = new Linear(2, 2);
  module.loadStateDict({ weight: fromJson(state.weight!), bias: fromJson(state.bias!) });
  return module;
}

function directory(): string {
  const path = scratch();
  mkdirSync(path, { recursive: true });
  return path;
}

describe('Python interoperability', () => {
  it('writes experiences byte-identical to Python for the same trace', async () => {
    const external = new External();
    const head = transform(linear(interop.experience.weights));
    const session = trace();
    const value = { x: 1.5, y: -2, tags: Object.freeze(['a', 'b']), raw: new Uint8Array([0, 255, 104, 105]) };
    const output = session.run(() => head.call(external.call(value, { context: { note: new Note('hello é', 0.5) } })));
    session.supervise(output, tensor([0.5, 0.25]), { loss: 'mse', source: 'review:é' });
    const path = join(directory(), 'experience.json');
    await session.save(path, { operations: { external, head }, codecs: { note: Note } });
    expect(readFileSync(path, 'utf8')).toBe(readFileSync(join(fixtures, 'experience_python.json'), 'utf8'));
  });

  it('loads, validates, replays and trains a Python-written experience', async () => {
    const external = new External();
    const head = transform(linear(interop.experience.weights));
    const loaded = await loadExperience(join(fixtures, 'experience_python.json'), {
      operations: { external, head }, codecs: { note: Note },
    });
    expect(loaded.inputs.get(5)).toEqual(new Note('hello é', 0.5));
    expect(loaded.inputs.get(4)).toEqual(new Uint8Array([0, 255, 104, 105]));
    expect([loaded.inputs.get(0), loaded.inputs.get(1), loaded.inputs.get(2), loaded.inputs.get(3)]).toEqual([1.5, -2, 'a', 'b']);
    const prediction = loaded.replay(loaded.supervisions[0]!.output, { boundary: 'recorded' }) as Tensor;
    expectClose(prediction.data, interop.experience.prediction.data, 1e-6, 1e-6);
    const loss = Trainer.fromOps({ external, head }, { lr: 0.1 }).step(loaded);
    expect(loss).toBeCloseTo(interop.experience.loss, 5);
    expectClose(head.module.weight.data, interop.experience.trained.weight.data, 1e-5, 1e-5);
    expectClose(head.module.bias!.data, interop.experience.trained.bias.data, 1e-5, 1e-5);
    // A changed binding configuration is rejected.
    await expect(loadExperience(join(fixtures, 'experience_python.json'), {
      operations: { external, head: transform(new Sequential(new Linear(2, 2))) }, codecs: { note: Note },
    })).rejects.toThrow(/Incompatible operation configuration: head/);
    expect(external).toBeInstanceOf(External);
  });

  it('restores a Python standalone Adam checkpoint and re-saves it byte-identically', async () => {
    const expected = interop.checkpoint;
    const head = transform(new Linear(2, 2));
    const learner = Trainer.fromOps({ head }, { optimizer: (params) => new Adam(params, { lr: 0.01 }) });
    const source = join(fixtures, 'checkpoint_python.json');
    await learner.loadCheckpoint(source);
    expect([...head.module.weight.data]).toEqual(expected.saved.weight.data.map(Math.fround));
    expect([...head.module.bias!.data]).toEqual(expected.saved.bias.data.map(Math.fround));
    expect(learner.optimizer.state.get(head.module.weight)!.step!.item()).toBe(1);
    const path = join(directory(), 'resaved.json');
    await saveCheckpoint(path, { operations: { head }, optimizer: learner.optimizer });
    expect(readFileSync(path, 'utf8')).toBe(readFileSync(source, 'utf8'));
    const session = trace();
    const out = session.run(() => head.call(tensor(expected.inputs)));
    session.supervise(out, tensor(expected.target), { loss: 'mse' });
    expect(learner.step(session)).toBeCloseTo(expected.next_loss, 6);
    expectClose(head.module.weight.data, expected.next.weight.data, 1e-6, 1e-6);
    expectClose(head.module.bias!.data, expected.next.bias.data, 1e-6, 1e-6);
  });

  it('restores a Python training directory, including its PyTorch and CPython RNG states', async () => {
    const expected = interop.resume;
    const head = transform(new Sequential(new Linear(2, 2), new Dropout(0.4)));
    const learner = Trainer.fromOps({ head }, { optimizer: (params) => new Adam(params, { lr: 0.01 }) });
    expect(head.training).toBe(true);
    manualSeed(1);
    pythonRandom.seed(1);
    expect(await learner.loadCheckpoint(join(fixtures, 'python_resume'))).toEqual({ cursor: 3, note: 'python' });
    const saved = JSON.parse(readFileSync(join(fixtures, 'python_resume', 'training.json'), 'utf8'));
    const tensors = deserializeSafetensors(readFileSync(join(fixtures, 'python_resume', saved.tensors.file))).tensors;
    const torchState = tensors.get(saved.state.torch_rng.key)!;
    expect(Array.from(getRngState())).toEqual(Array.from(torchState.data));
    const pythonState = saved.state.python_rng.items;
    const [version, internal, gaussNext] = pythonRandom.getstate();
    expect([version, [...internal], gaussNext]).toEqual([pythonState[0], pythonState[1].items, pythonState[2]]);
    // The next draws equal the ones Python makes after restoring the same checkpoint.
    const state = getRngState();
    const pythonSnapshot = pythonRandom.getstate();
    expect(Array.from(rand([4]).data)).toEqual(expected.after_restore.torch_rand);
    expect(Array.from(randn([20]).data)).toEqual(expected.after_restore.torch_randn);
    expect(Array.from(F.dropout(ones([12]), 0.4, true).data)).toEqual(expected.after_restore.dropout);
    expect([pythonRandom.random(), pythonRandom.random(), pythonRandom.random()]).toEqual(expected.after_restore.python_random);
    manualSeed(0);
    pythonRandom.seed(0);
    await learner.loadCheckpoint(join(fixtures, 'python_resume'));
    expect(getRngState()).toEqual(state);
    expect(pythonRandom.getstate()).toEqual(pythonSnapshot);
    expect(learner.steps).toBe(1);
    expect(head.namedModules().map(([, module]) => module.training)).toEqual([false, false, false, false]);
    for (const [key, value] of head.stateDict()) expectClose(value.data, expected.state[key].data, 0, 0);
    const experience = await loadExperience(join(fixtures, 'experience_resume.json'), { operations: { head } });
    expect(learner.step(experience)).toBeCloseTo(expected.next_loss, 6);
    for (const [key, value] of head.stateDict()) expectClose(value.data, expected.next[key].data, 1e-6, 1e-6);
  });

  it('writes directory checkpoints whose model section keeps the Python layout', async () => {
    const head = transform(new Sequential(new Linear(2, 2), new Dropout(0.4)));
    const learner = Trainer.fromOps({ head }, { optimizer: (params) => new Adam(params, { lr: 0.01 }) });
    const session = trace();
    const out = session.run(() => head.call(ones([2])));
    session.supervise(out, zeros([2]), { loss: 'mse' });
    learner.step(session);
    const path = directory();
    await learner.saveCheckpoint(path, { progress: { cursor: 3 } });
    const ours = JSON.parse(readFileSync(join(path, 'training.json'), 'utf8'));
    const python = JSON.parse(readFileSync(join(fixtures, 'python_resume', 'training.json'), 'utf8'));
    expect(Object.keys(ours).sort()).toEqual(Object.keys(python).sort());
    expect(Object.keys(ours.model).sort()).toEqual(Object.keys(python.model).sort());
    expect(ours.model.operations.head.configuration).toEqual(python.model.operations.head.configuration);
    expect(ours.model.aliases).toEqual(python.model.aliases);
    expect(ours.model.optimizer.layout).toEqual(python.model.optimizer.layout);
    expect(ours.state.modes).toEqual({ head: { '': true, module: true, 'module.0': true, 'module.1': true } });
    expect(Object.keys(ours.state).sort()).toEqual(Object.keys(python.state).sort());
    expect(ours.state.torch_rng).toEqual({ type: 'tensor_ref', key: 'tensor_0', shape: [5056], dtype: 'uint8' });
    expect(ours.state.cuda_rng).toEqual({ type: 'list', items: [] });
    expect(ours.state.python_rng.type).toBe('tuple');
    expect(ours.state.python_rng.items[0]).toBe(3);
    expect(ours.state.python_rng.items[1].type).toBe('tuple');
    expect(ours.state.python_rng.items[1].items).toHaveLength(625);
    expect(Array.from(getRngState())).toEqual(Array.from(
      deserializeSafetensors(readFileSync(join(path, ours.tensors.file))).tensors.get('tensor_0')!.data,
    ));
    expect(ours.tensors.file).toMatch(/^tensors-[0-9a-f]{32}\.safetensors$/);
  });
});
