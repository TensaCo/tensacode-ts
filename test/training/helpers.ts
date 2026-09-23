/**
 * Stand-ins for the Python ``Transform``/``Classify`` operations used by the
 * training tests (the vector operation module is built separately).
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll } from 'vitest';
import {
  Dropout, Linear, Module, Parameter, Sequential, Tensor, getDefaultGenerator, rand, scalar, tensor,
} from '../../src/nn/index.js';
import { ModuleOperation, type Context, type OperationLike } from '../../src/ops/base.js';
import { PretrainedModule } from '../../src/_internal/pretrained.js';
import type { TrainableTool } from '../../src/_internal/contracts.js';
import { TensorAdapter, type ForwardModule } from '../../src/_internal/vec/adapter.js';
import { moduleConfiguration } from '../../src/_internal/vec/configuration.js';
import type { JsonObject } from '../../src/_internal/json.js';

/** Python ``Transform.from_module(module)``. */
export function transform<M extends ForwardModule>(module: M): TensorAdapter<M> {
  return new TensorAdapter(module);
}

/** Classification output record carrying logits and label names (Python ``ClassificationResult``-like). */
export class Prediction {
  static readonly qualifiedName: string = 'tests.training.Prediction';
  static readonly recordFields = ['logits', 'labels'] as const;
  readonly logits: Tensor;
  readonly labels: readonly string[];

  constructor(logits: Tensor, labels: readonly string[]) {
    this.logits = logits;
    this.labels = labels;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Prediction {
    return new Prediction(fields.logits as Tensor, fields.labels as readonly string[]);
  }

  toRecord(): Record<string, unknown> {
    return { logits: this.logits, labels: this.labels };
  }
}

/** Python ``Classify.from_module(module, labels=...)``: logits plus named labels. */
export class LabelHead extends ModuleOperation<Tensor, Prediction> {
  static override readonly qualifiedName: string = 'tests.training.LabelHead';
  readonly module: Linear;
  readonly labels: readonly string[];

  constructor(module: Linear, labels: readonly string[]) {
    super();
    this.module = this.registerModule('module', module);
    this.labels = Object.freeze([...labels]);
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: Tensor, context: Context | null): Prediction {
    void context;
    return new Prediction(this.module.forward(value), this.labels);
  }

  configuration(): JsonObject {
    return { labels: [...this.labels], module: moduleConfiguration(this.module) };
  }
}

/** A frozen value record (Python ``@dataclass(frozen=True) class Number``). */
export class Num {
  static readonly recordFields = ['value'] as const;
  readonly value: number;

  constructor(value: number) {
    this.value = value;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Num {
    return new Num(fields.value as number);
  }

  toRecord(): Record<string, unknown> {
    return { value: this.value };
  }
}

export function vector(values: number[]): Tensor {
  return tensor(values);
}

export function clones(parameters: readonly Tensor[]): Tensor[] {
  return parameters.map((parameter) => parameter.detach().clone());
}

export function allEqual(left: readonly Tensor[], right: readonly Tensor[]): boolean {
  return left.length === right.length && left.every((value, index) => value.equal(right[index]!));
}

/** A scratch directory removed after the test file. */
export function scratchDirectory(prefix: string): () => string {
  const root = mkdtempSync(join(tmpdir(), prefix));
  let counter = 0;
  afterAll(() => rmSync(root, { recursive: true, force: true }));
  return () => {
    counter += 1;
    return join(root, `case-${counter}`);
  };
}

export function modulesOf(module: Module): Map<string, Module> {
  return new Map(module.namedModules());
}

/** Python ``Objective``: a stochastic scalar objective consuming ``{inputs, targets}``. */
export class Objective extends Module {
  static override readonly qualifiedName: string = 'tests.training.Objective';
  readonly weight: Parameter;

  constructor() {
    super();
    this.weight = this.registerParameter('weight', new Parameter(scalar(0.25)));
  }

  forward(value: { inputs: Tensor; targets: Tensor }): Tensor {
    const noise = rand([]).item() + getDefaultGenerator().random();
    return this.weight.mul(value.inputs).add(noise).sub(value.targets).pow(2).mean();
  }
}

/** Python ``Tool``: joint objective receiving inputs and targets. */
export class Tool implements TrainableTool {
  readonly trainingInputsIncludeTargets = true;
  trainingOperation: TensorAdapter<ForwardModule>;

  constructor() {
    this.trainingOperation = new TensorAdapter(new Objective());
  }

  operationBindings(): Record<string, OperationLike> {
    return { objective: this.trainingOperation };
  }
}

/** Python ``DropoutTool``: a pretrained tool with a frozen-dropout training policy. */
export class DropoutTool extends PretrainedModule<Tensor, Tensor> implements TrainableTool {
  static override readonly qualifiedName: string = 'tests.training.DropoutTool';
  readonly prediction: TensorAdapter<Sequential>;
  readonly frozen: Dropout;

  constructor(config: JsonObject) {
    super(config);
    this.prediction = this.registerModule('prediction', new TensorAdapter(new Sequential(new Linear(4, 4), new Dropout(0.5), new Linear(4, 1))));
    this.frozen = this.registerModule('frozen', new Dropout(0.2));
  }

  override train(mode = true): this {
    super.train(mode);
    this.frozen.eval();
    return this;
  }

  get trainingOperation(): OperationLike {
    return this.prediction;
  }

  override operationBindings(): Record<string, OperationLike> {
    return { prediction: this.prediction };
  }

  trainingLoss(output: Tensor, targets: Tensor): Tensor {
    return output.sub(targets).pow(2).mean();
  }

  forward(value: Tensor): Tensor {
    return this.prediction.call(value) as Tensor;
  }
}
