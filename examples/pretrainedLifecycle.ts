/**
 * The owned-model artifact lifecycle: construct from JSON configuration, train
 * one step through the declared objective, `savePretrained`, `fromPretrained`
 * and (optionally) `pushToHub`.
 *
 *     npm run build
 *     node examples/pretrainedLifecycle.ts
 *     HF_TOKEN=hf_... node examples/pretrainedLifecycle.ts --push your-name/tiny-scorer
 *
 * Artifacts contain `tensorcode_config.json` (format, class identity and JSON
 * configuration), `model.safetensors` and a `README.md` model card. They are
 * interchangeable with the Python implementation when both define the same
 * class identity and architecture. Sessions, experiences and optimizer state are
 * never part of a model artifact.
 */
import { mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Linear, manualSeed, noGrad, tensor, type Tensor } from 'tensorcode/nn';
import { ModuleOperation, type Context, type OperationLike } from 'tensorcode/ops';
// The owned-model base class that TensorCode tools extend (Python ``PretrainedTool``).
import { PretrainedModule } from 'tensorcode/tools';
import { Trainer } from 'tensorcode/training';

type Json = Record<string, unknown>;

/** The declared training objective: `{inputs, targets}` → mean squared error. */
class Objective extends ModuleOperation<{ inputs: Tensor; targets: Tensor }, Tensor> {
  static override readonly qualifiedName: string = 'examples.TinyRegressor.objective';
  readonly owner: TinyRegressor;

  constructor(owner: TinyRegressor) {
    super();
    this.owner = owner;
  }

  override get replayable(): boolean {
    return true;
  }

  override parameters(): ReturnType<ModuleOperation['parameters']> {
    return this.owner.parameters();
  }

  configuration(): Json {
    return { owner: 'examples.TinyRegressor', config: this.owner.configuration() };
  }

  forward(value: { inputs: Tensor; targets: Tensor }, context: Context | null): Tensor {
    void context;
    return this.owner.forward(value.inputs).sub(value.targets).pow(2).mean();
  }
}

/** A tiny owned tool: JSON configuration in, trainable weights inside. */
class TinyRegressor extends PretrainedModule<Tensor, Tensor> {
  static override readonly qualifiedName: string = 'examples.TinyRegressor';
  readonly trainingInputsIncludeTargets = true;
  readonly layer: Linear;
  readonly objective: Objective;

  constructor(config: Json) {
    super({ width: 3, ...config });
    this.layer = this.registerModule('layer', new Linear(this.config.width as number, 1));
    // Keep the objective outside the module tree so its parameters are the owner's.
    this.objective = new Objective(this);
  }

  get trainingOperation(): OperationLike {
    return this.objective;
  }

  override operationBindings(): Record<string, OperationLike> {
    return { objective: this.objective };
  }

  forward(value: Tensor): Tensor {
    return this.layer.forward(value);
  }
}

manualSeed(3);
const directory = mkdtempSync(join(tmpdir(), 'tensorcode-lifecycle-'));
try {
  const model = new TinyRegressor({ width: 3 });
  const trainer = Trainer.fromTool(model, { lr: 0.1 });
  const inputs = tensor([[1, 0, -1], [0.5, 2, 1]]);
  const targets = tensor([[1], [-1]]);
  const experience = trainer.capture(inputs, targets, { source: 'example:reviewed' });
  const losses = trainer.fit([experience], { epochs: 20 });
  console.log(`loss ${losses[0]!.toFixed(4)} -> ${losses[losses.length - 1]!.toFixed(4)}`);

  const saved = await model.savePretrained(join(directory, 'model'));
  console.log(`saved ${readdirSync(saved).sort().join(', ')}`);
  console.log(JSON.parse(readFileSync(join(saved, 'tensorcode_config.json'), 'utf8')));

  const restored = await TinyRegressor.fromPretrained(saved);
  const same = noGrad(() => restored.forward(inputs).equal(model.forward(inputs)));
  console.log(`restored in eval mode: ${!restored.training}; identical predictions: ${same}`);

  const pushIndex = process.argv.indexOf('--push');
  if (pushIndex >= 0) {
    const repoId = process.argv[pushIndex + 1];
    if (!repoId) throw new Error('--push requires a repository id');
    const result = await restored.pushToHub(repoId, { private: true, commitMessage: 'Upload TinyRegressor example' });
    console.log('published', result);
  }
} finally {
  rmSync(directory, { recursive: true, force: true });
}
