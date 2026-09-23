/** Port of Python ``tests/training/test_tool_training.py``. */
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  Adam, F, Linear, Sequential, Tensor, deserializeSafetensors, getDefaultGenerator, getRngState, noGrad, ones, rand, serializeSafetensors, setRngState,
  tensor, zeros,
} from '../../src/nn/index.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import { TrainingEngine } from '../../src/_internal/training/tool.js';
import { loadCheckpoint, saveCheckpoint } from '../../src/_internal/training/checkpoint.js';
import { digestBytes } from '../../src/_internal/training/tensorStore.js';
import { writeArtifact } from '../../src/_internal/training/persistence.js';
import { TensorAdapter } from '../../src/_internal/vec/adapter.js';
import type { OperationLike } from '../../src/ops/base.js';
import { DropoutTool, Tool, allEqual, clones, scratchDirectory, transform } from './helpers.js';

const scratch = scratchDirectory('tensorcode-tool-training-');

function trainer(): Trainer {
  return Trainer.fromTool(new Tool(), { optimizer: (params) => new Adam(params, { lr: 0.01 }) });
}

function readJson(path: string): any {
  return JSON.parse(readFileSync(path, 'utf8'));
}

async function writeJson(path: string, data: unknown): Promise<void> {
  await writeArtifact(path, data);
}

function dropoutLearner(options: { adam?: boolean } = {}): Trainer {
  return Trainer.fromTool(new DropoutTool({ width: 4 }), options.adam ? { optimizer: (params) => new Adam(params, { lr: 0.01 }) } : {});
}

afterEach(() => {
  TrainingEngine.writeManifest = writeArtifact;
  TrainingEngine.restoreRng = setRngState;
});

describe('tool training', () => {
  it('replays persisted feedback on a fresh tool', async () => {
    const first = trainer();
    const session = first.capture(tensor([1, 2]), tensor([3, 4]), { source: 'reviewed:42' });
    const directory = scratch();
    mkdirSync(directory, { recursive: true });
    const path = join(directory, 'experience.json');
    await session.save(path, { operations: first.operations, release: true });
    const second = trainer();
    const restored = await loadExperience(path, { operations: second.operations });
    expect(restored.supervisions[0]!.source).toBe('reviewed:42');
    expect((restored.supervisions[0]!.target as Tensor).equal(tensor([3, 4]))).toBe(true);
    const before = second.parameters[0]!.detach().clone();
    second.step(restored);
    expect(before.equal(second.parameters[0]!)).toBe(false);
    expect(second.steps).toBe(1);
  });

  it('resume restores the stochastic next step and optimizer', async () => {
    const first = trainer();
    const session = first.capture(tensor([1, 2]), tensor([3, 4]), { source: 'reviewed:42' });
    const directory = scratch();
    mkdirSync(directory, { recursive: true });
    const path = join(directory, 'experience.json');
    await session.save(path, { operations: first.operations });
    first.step(session);
    await first.saveCheckpoint(join(directory, 'resume'), { progress: { next_example: 8 } });
    const expectedLoss = first.step(session);
    const expected = first.parameters[0]!.detach().clone();
    const second = trainer();
    const restored = await loadExperience(path, { operations: second.operations });
    const progress = await second.loadCheckpoint(join(directory, 'resume'));
    expect(progress).toEqual({ next_example: 8 });
    expect(second.steps).toBe(1);
    expect(second.step(restored)).toBe(expectedLoss);
    expect(second.parameters[0]!.equal(expected)).toBe(true);
    expect(second.steps).toBe(2);
  });

  it('rejects an invalid generator state before mutating the model', async () => {
    const first = trainer();
    const directory = scratch();
    await first.saveCheckpoint(directory);
    const path = join(directory, 'training.json');
    const payload = readJson(path);
    const original = JSON.parse(JSON.stringify(payload.state.python_rng));
    payload.state.python_rng = 0;
    writeFileSync(path, JSON.stringify(payload));
    const before = first.parameters[0]!.detach().clone();
    await expect(first.loadCheckpoint(directory)).rejects.toThrow();
    expect(before.equal(first.parameters[0]!)).toBe(true);
    payload.state.python_rng = { ...original, items: [4, original.items[1], null] };
    writeFileSync(path, JSON.stringify(payload));
    await expect(first.loadCheckpoint(directory)).rejects.toThrow(/version 4/);
    payload.state.python_rng = { ...original, items: [3, { type: 'list', items: original.items[1].items }, null] };
    writeFileSync(path, JSON.stringify(payload));
    await expect(first.loadCheckpoint(directory)).rejects.toThrow(/must be a tuple/);
    payload.state.python_rng = original;
    payload.state.cuda_rng = { type: 'list', items: [0] };
    writeFileSync(path, JSON.stringify(payload));
    await expect(first.loadCheckpoint(directory)).rejects.toThrow(/CUDA device topology/);
    payload.state.cuda_rng = { type: 'list', items: [] };
    payload.state.rng = 0;
    writeFileSync(path, JSON.stringify(payload));
    await expect(first.loadCheckpoint(directory)).rejects.toThrow(/Malformed training progress/);
    expect(before.equal(first.parameters[0]!)).toBe(true);
  });

  it('requires a feedback source and supports the trainingLoss protocol', () => {
    class LogitTool {
      readonly trainingOperation = transform(new Linear(2, 2));
      operationBindings(): Record<string, OperationLike> {
        return { prediction: this.trainingOperation };
      }
      trainingLoss(prediction: Tensor, targets: Tensor): Tensor {
        return F.crossEntropy(prediction, targets);
      }
    }
    const learner = Trainer.fromTool(new LogitTool());
    expect(() => learner.capture(ones([2]), tensor(1, { dtype: 'int64' }), { source: ' ' })).toThrow(/source/);
    const session = learner.capture(ones([2]), tensor(1, { dtype: 'int64' }), { source: 'human' });
    expect(learner.step(session)).toBeGreaterThan(0);
    class Undeclared {
      readonly trainingOperation = transform(new Linear(2, 2));
      operationBindings(): Record<string, OperationLike> {
        return { other: transform(new Linear(2, 2)) };
      }
    }
    expect(() => Trainer.fromTool(new Undeclared())).toThrow(/training_operation must appear/);
  });

  it('resumes a pretrained dropout tool with exact mixed modes and RNG', async () => {
    const first = dropoutLearner();
    const tool = first.tool as DropoutTool;
    expect(tool.training).toBe(true);
    expect(tool.frozen.training).toBe(false);
    const session = first.capture(ones([3, 4]), zeros([3, 1]), { source: 'review' });
    first.step(session);
    const directory = scratch();
    await tool.savePretrained(join(directory, 'pretrained'));
    await session.save(join(directory, 'experience.json'), { operations: first.operations });
    // Preserve a deliberately mixed mode beyond the tool's frozen policy.
    tool.prediction.module.at(0).eval();
    await first.saveCheckpoint(join(directory, 'resume'));
    const expectedLoss = first.step(session);
    const expectedParams = clones(first.parameters);
    const expectedRng = rand([5]);
    const expectedUniform = getDefaultGenerator().random();

    const restoredTool = await DropoutTool.fromPretrained(join(directory, 'pretrained'));
    expect(restoredTool.training).toBe(false);
    const second = Trainer.fromTool(restoredTool);
    expect(restoredTool.training).toBe(true);
    expect(restoredTool.prediction.module.at(1).training).toBe(true);
    expect(restoredTool.frozen.training).toBe(false);
    const restoredSession = await loadExperience(join(directory, 'experience.json'), { operations: second.operations });
    await second.loadCheckpoint(join(directory, 'resume'));
    expect(restoredTool.prediction.module.at(0).training).toBe(false);
    expect(restoredTool.prediction.module.at(1).training).toBe(true);
    expect(restoredTool.frozen.training).toBe(false);
    expect(second.step(restoredSession)).toBe(expectedLoss);
    expect(allEqual(expectedParams, second.parameters)).toBe(true);
    expect(rand([5]).equal(expectedRng)).toBe(true);
    expect(getDefaultGenerator().random()).toBe(expectedUniform);
  });

  it('rejects an invalid module mode before mutation', async () => {
    const learner = dropoutLearner();
    const directory = scratch();
    await learner.saveCheckpoint(directory);
    const path = join(directory, 'training.json');
    const payload = readJson(path);
    payload.state.modes.tool['prediction.module.1'] = 'train';
    writeFileSync(path, JSON.stringify(payload));
    const before = clones(learner.parameters);
    await expect(learner.loadCheckpoint(directory)).rejects.toThrow(/boolean/);
    expect(allEqual(before, learner.parameters)).toBe(true);
    expect((learner.tool as DropoutTool).training).toBe(true);
  });

  it('rolls back weights, modes and RNG when resume is interrupted', async () => {
    const learner = dropoutLearner();
    const tool = learner.tool as DropoutTool;
    const directory = scratch();
    await learner.saveCheckpoint(directory);
    noGrad(() => {
      for (const parameter of learner.parameters) parameter.add_(1);
    });
    tool.eval();
    const before = clones(learner.parameters);
    const rng = getRngState();
    let calls = 0;
    TrainingEngine.restoreRng = () => {
      calls += 1;
      throw new Error('interrupted restore');
    };
    await expect(learner.loadCheckpoint(directory)).rejects.toThrow(/interrupted restore/);
    expect(calls).toBe(1);
    expect(allEqual(before, learner.parameters)).toBe(true);
    expect(tool.training).toBe(false);
    expect(tool.prediction.module.at(1).training).toBe(false);
    expect(getRngState()).toEqual(rng);
  });

  it('rolls back an earlier module when a direct checkpoint load is interrupted', async () => {
    const first = transform(new Linear(1, 1));
    const second = transform(new Linear(1, 1));
    const operations = { first, second };
    const directory = scratch();
    mkdirSync(directory, { recursive: true });
    const path = join(directory, 'checkpoint.json');
    await saveCheckpoint(path, { operations });
    noGrad(() => {
      first.module.weight.add_(2);
      second.module.weight.add_(3);
    });
    const expected = Object.fromEntries(Object.entries(operations).map(([name, op]) => [
      name, new Map([...op.stateDict()].map(([key, value]) => [key, value.clone()])),
    ]));
    const original = second.loadStateDict.bind(second);
    let calls = 0;
    (second as { loadStateDict: unknown }).loadStateDict = (...args: Parameters<typeof original>) => {
      calls += 1;
      if (calls === 1) throw new Error('second module interrupted');
      return original(...args);
    };
    await expect(loadCheckpoint(path, { operations })).rejects.toThrow(/interrupted/);
    for (const [name, op] of Object.entries(operations)) {
      for (const [key, value] of op.stateDict()) expect(expected[name]!.get(key)!.equal(value)).toBe(true);
    }
  });

  it('keeps metadata small for large tensors', async () => {
    const tool = new Tool();
    tool.trainingOperation = new TensorAdapter(new Linear(1024, 1024));
    const learner = Trainer.fromTool(tool);
    const original = Tensor.prototype.tolist;
    Tensor.prototype.tolist = () => {
      throw new Error('Tensor converted to a JavaScript list');
    };
    const directory = scratch();
    try {
      await learner.saveCheckpoint(directory);
    } finally {
      Tensor.prototype.tolist = original;
    }
    const manifest = join(directory, 'training.json');
    expect(statSync(manifest).size).toBeLessThan(50_000);
    const payload = readJson(manifest);
    expect(statSync(join(directory, payload.tensors.file)).size).toBeGreaterThan(4_000_000);
    await learner.loadCheckpoint(directory);
  });

  for (const corruption of ['digest', 'dangling', 'shape', 'dtype', 'nonfinite', 'path', 'unreferenced']) {
    it(`rejects binary corruption (${corruption}) before mutation`, async () => {
      const learner = dropoutLearner();
      const directory = scratch();
      await learner.saveCheckpoint(directory);
      const path = join(directory, 'training.json');
      const payload = readJson(path);
      const tensorPath = join(directory, payload.tensors.file);
      if (corruption === 'digest') {
        writeFileSync(tensorPath, Buffer.concat([readFileSync(tensorPath), Buffer.from('bad')]));
      } else if (corruption === 'path') {
        payload.tensors.file = '../elsewhere.safetensors';
      } else if (corruption === 'nonfinite' || corruption === 'unreferenced') {
        const tensors = deserializeSafetensors(new Uint8Array(readFileSync(tensorPath))).tensors;
        if (corruption === 'nonfinite') {
          const floating = [...tensors.values()].find((value) => value.isFloatingPoint)!;
          floating.data[0] = Number.NaN;
        } else {
          tensors.set('tensor_999', zeros([1]));
        }
        const bytes = serializeSafetensors(tensors);
        writeFileSync(tensorPath, bytes);
        payload.tensors.sha256 = digestBytes(bytes);
      } else {
        const reference = payload.model.states.tool.items[0][1];
        expect(reference.type).toBe('tensor_ref');
        if (corruption === 'dangling') reference.key = 'absent';
        else if (corruption === 'shape') reference.shape = [123];
        else reference.dtype = 'float64';
      }
      await writeJson(path, payload);
      const before = clones(learner.parameters);
      await expect(learner.loadCheckpoint(directory)).rejects.toThrow(/./);
      await expect(learner.loadCheckpoint(directory)).rejects.toSatisfy((error: Error) => error.name === 'ValueError');
      expect(allEqual(before, learner.parameters)).toBe(true);
    });
  }

  it('a failed manifest switch preserves the previous checkpoint', async () => {
    const learner = dropoutLearner();
    const directory = scratch();
    await learner.saveCheckpoint(directory);
    const previous = readFileSync(join(directory, 'training.json'));
    const expected = clones(learner.parameters);
    noGrad(() => {
      for (const parameter of learner.parameters) parameter.add_(1);
    });
    TrainingEngine.writeManifest = async (path, data) => {
      if (path.endsWith('training.json')) throw new Error('interrupted manifest write');
      return writeArtifact(path, data);
    };
    await expect(learner.saveCheckpoint(directory)).rejects.toThrow(/interrupted manifest/);
    expect(readFileSync(join(directory, 'training.json')).equals(previous)).toBe(true);
    TrainingEngine.writeManifest = writeArtifact;
    await learner.loadCheckpoint(directory);
    expect(allEqual(expected, learner.parameters)).toBe(true);
  });

  it('an optimizer interruption restores the entire training transaction', async () => {
    const learner = dropoutLearner({ adam: true });
    const tool = learner.tool as DropoutTool;
    const session = learner.capture(ones([3, 4]), zeros([3, 1]), { source: 'review' });
    learner.step(session);
    const directory = scratch();
    await learner.saveCheckpoint(directory, { progress: { cursor: 1 } });
    learner.step(session);
    learner.optimizer.paramGroups[0]!.lr = 0.123;
    tool.eval();
    (tool.prediction.module.at(0)).train();
    learner.progress = { cursor: 2 };
    const weights = new Map([...tool.stateDict()].map(([key, value]) => [key, value.clone()]));
    const optimizer = learner.optimizer.stateDict();
    const modes = Object.fromEntries(tool.namedModules().map(([name, module]) => [name, module.training]));
    const rng = getRngState();
    const original = learner.optimizer.loadStateDict.bind(learner.optimizer);
    let interrupted = false;
    (learner.optimizer as { loadStateDict: unknown }).loadStateDict = (state: Parameters<typeof original>[0]) => {
      const result = original(state);
      if (!interrupted) {
        interrupted = true;
        getDefaultGenerator().random();
        rand([3]);
        tool.train();
        throw new Error('optimizer applied before interruption');
      }
      return result;
    };
    await expect(learner.loadCheckpoint(directory)).rejects.toThrow(/optimizer applied/);
    for (const [key, value] of weights) expect(value.equal(tool.stateDict().get(key)!)).toBe(true);
    const restored = learner.optimizer.stateDict();
    expect(restored.param_groups).toEqual(optimizer.param_groups);
    expect(Object.keys(restored.state)).toEqual(Object.keys(optimizer.state));
    for (const [key, slots] of Object.entries(optimizer.state)) {
      expect(Object.keys(restored.state[key]!)).toEqual(Object.keys(slots));
      for (const [name, value] of Object.entries(slots)) expect(value.equal(restored.state[key]![name]!)).toBe(true);
    }
    expect(Object.fromEntries(tool.namedModules().map(([name, module]) => [name, module.training]))).toEqual(modes);
    expect(getRngState()).toEqual(rng);
    expect(learner.steps).toBe(2);
    expect(learner.progress).toEqual({ cursor: 2 });
  });

  it('checkpoints the whole pretrained tool as the {tool} binding', async () => {
    const learner = dropoutLearner();
    const directory = scratch();
    await learner.saveCheckpoint(directory);
    const payload = readJson(join(directory, 'training.json'));
    expect(Object.keys(payload.model.operations)).toEqual(['tool']);
    expect(payload.model.operations.tool.configuration.type).toBe('tests.training.DropoutTool');
    expect(Object.keys(payload.state).sort()).toEqual(['cuda_rng', 'modes', 'progress', 'python_rng', 'steps', 'torch_rng']);
    expect(payload.state.cuda_rng).toEqual({ type: 'list', items: [] });
    expect(new Sequential(new Linear(1, 1)).length).toBe(1);
  });
});
