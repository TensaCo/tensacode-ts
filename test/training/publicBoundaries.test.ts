/** Port of Python ``tests/training/test_public_training_boundaries.py``. */
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  Adam, Dropout, Linear, Parameter, Sequential, Tensor, getRngState, noGrad, ones, tensor, zeros,
} from '../../src/nn/index.js';
import { Operation, type Context } from '../../src/ops/base.js';
import * as root from '../../src/index.js';
import * as tracing from '../../src/_internal/tracing.js';
import * as training from '../../src/training/index.js';
import { Trainer } from '../../src/training/index.js';
import { saveCheckpoint } from '../../src/_internal/training/checkpoint.js';
import { Tool, allEqual, clones, scratchDirectory, transform } from './helpers.js';


const scratch = scratchDirectory('tensorcode-boundaries-');

describe('public training boundaries', () => {
  it('exposes the public trace types and hides internal paths', () => {
    expect(tracing.trace()).toBeInstanceOf(root.Trace);
    expect([root.Trace, root.InputRef, root.OutputRef]).toEqual([tracing.Trace, tracing.InputRef, tracing.OutputRef]);
    for (const name of ['ToolTrainer', 'load', 'saveCheckpoint', 'loadCheckpoint', 'save_checkpoint', 'load_checkpoint']) {
      expect(name in training).toBe(false);
    }
  });

  it('resumes operation graphs completely and rejects tool capture', async () => {
    const head = transform(new Sequential(new Linear(2, 2), new Dropout(0.4)));
    head.eval();
    const learner = Trainer.fromOps({ head }, { optimizer: (params) => new Adam(params, { lr: 0.01 }) });
    expect(head.training).toBe(false);
    expect(() => learner.capture(ones([2]), zeros([2]), { source: 'review' })).toThrow(/trace.*supervise/);
    head.train();
    const experience = tracing.trace();
    const out = experience.run(() => head.call(ones([2])));
    experience.supervise(out, zeros([2]), { loss: 'mse' });
    learner.step(experience);
    const directory = scratch();
    await learner.saveCheckpoint(directory, { progress: { cursor: 3 } });
    const expectedLoss = learner.step(experience);
    const expected = clones(learner.parameters);
    learner.steps = 50;
    expect(await learner.loadCheckpoint(directory)).toEqual({ cursor: 3 });
    expect(learner.steps).toBe(1);
    expect(learner.step(experience)).toBe(expectedLoss);
    expect(allEqual(expected, learner.parameters)).toBe(true);
  });

  it('checkpoints parameterless external operations and non-module wrappers', async () => {
    class External extends Operation<Tensor, Tensor> {
      forward(value: Tensor): Tensor {
        return value;
      }
    }
    class Wrapper extends Operation<Tensor, Tensor> {
      private readonly inner = new Linear(2, 2);
      override get replayable(): boolean {
        return true;
      }
      configuration(): unknown {
        return { width: 2 };
      }
      parameters(): Parameter[] {
        return this.inner.parameters();
      }
      namedParameters(options: { removeDuplicate?: boolean } = {}): [string, Parameter][] {
        return this.inner.namedParameters(options);
      }
      stateDict(): Map<string, Tensor> {
        return this.inner.stateDict();
      }
      loadStateDict(state: Map<string, Tensor> | Record<string, Tensor>, options: { strict?: boolean } = {}): unknown {
        return this.inner.loadStateDict(state, options);
      }
      forward(value: Tensor, context: Context | null): Tensor {
        void context;
        return this.inner.forward(value);
      }
    }
    const external = new External();
    const wrapper = new Wrapper();
    const learner = Trainer.fromOps({ external, wrapper });
    const experience = tracing.trace();
    const output = experience.run(() => wrapper.call(external.call(ones([2]))));
    experience.supervise(output, zeros([2]), { loss: 'mse' });
    learner.step(experience);
    const directory = scratch();
    await learner.saveCheckpoint(directory);
    const expected = learner.step(experience);
    await learner.loadCheckpoint(directory);
    expect(learner.step(experience)).toBe(expected);
  });

  it('rejects checkpoints of trainables without the restore protocol', async () => {
    class Unrestorable extends Operation<Tensor, Tensor> {
      private readonly weight = new Parameter(ones([2]));
      override get replayable(): boolean {
        return true;
      }
      parameters(): Parameter[] {
        return [this.weight];
      }
      configuration(): unknown {
        return {};
      }
      forward(value: Tensor): Tensor {
        return value.mul(this.weight);
      }
    }
    const learner = Trainer.fromOps({ op: new Unrestorable() });
    await expect(learner.saveCheckpoint(scratch())).rejects.toThrow(/state_dict|restor/);
    await expect(learner.saveCheckpoint(scratch())).rejects.toThrow(TypeError);
  });

  it('standalone checkpoint files leave unavailable training state unchanged', async () => {
    const head = transform(new Linear(2, 2));
    const learner = Trainer.fromOps({ head });
    const directory = scratch();
    mkdirSync(directory, { recursive: true });
    const path = join(directory, 'legacy.json');
    await saveCheckpoint(path, { operations: learner.operations, optimizer: learner.optimizer });
    const expected = clones(learner.parameters);
    noGrad(() => {
      for (const parameter of learner.parameters) parameter.add_(10);
    });
    head.eval();
    learner.steps = 9;
    learner.progress = { cursor: 12 };
    const rng = getRngState();
    expect(await learner.loadCheckpoint(path)).toEqual({ cursor: 12 });
    expect(learner.steps).toBe(9);
    expect(head.training).toBe(false);
    expect(getRngState()).toEqual(rng);
    expect(allEqual(expected, learner.parameters)).toBe(true);
  });

  it('uses explicit factories with Python default learning rates', () => {
    const tool = Trainer.fromTool(new Tool());
    const ops = Trainer.fromOps(tool.operations);
    expect(tool).toBeInstanceOf(Trainer);
    expect(ops).toBeInstanceOf(Trainer);
    expect(tool.optimizer.paramGroups[0]!.lr).toBe(0.001);
    expect(ops.optimizer.paramGroups[0]!.lr).toBe(0.01);
    expect(() => new Trainer()).toThrow(TypeError);
    expect(() => new Trainer()).toThrow(/fromTool.*fromOps/);
    expect(() => { tool.steps = -1; }).toThrow(/nonnegative/);
    expect(() => { tool.progress = [] as never; }).toThrow(TypeError);
  });

  it('owns a copy of the supplied operation bindings', async () => {
    const head = transform(new Linear(2, 2));
    const bindings: Record<string, typeof head> = { head };
    const learner = Trainer.fromOps(bindings);
    delete bindings.head;
    const directory = scratch();
    await learner.saveCheckpoint(directory);
    await learner.loadCheckpoint(directory);
    expect(learner.operations).toEqual({ head });
    expect(learner.operations.head).toBe(head);
    expect(tensor([1]).item()).toBe(1);
  });
});
