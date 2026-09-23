# Quickstart

Install from GitHub (Node.js 20.16 or newer):

```bash
npm install github:TensaCo/tensacode-ts
```

From a checkout, run `npm install` in the repository root instead. It builds
`dist/` through the `prepare` script.

This small offline example sets up an evidence-conditioned model, captures
reviewed feedback, trains from persisted experience and saves weights you can
reload. Its two authored cases demonstrate the lifecycle. They are not an
evaluation of investigation competence. The same flow runs in the test suite as
`test/e2e/quickstart.test.ts`.

## Collect and train

Save this as `train-investigator.mts` and run `node train-investigator.mts` from
a writable directory. Node 22.18 and newer run TypeScript directly; on older
versions use `npx tsx train-investigator.mts`.

```ts
import { mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { AdamW, manualSeed } from 'tensorcode/nn';
import { Investigator } from 'tensorcode/tools';
import { Trainer, loadExperience } from 'tensorcode/training';

// All parameters exist before the optimizer is constructed.
manualSeed(7);
const model = new Investigator({
  vocabulary: ['which', 'component', 'failed', 'database', 'network',
               'connection', 'refused', 'packet', 'loss'],
  dimensions: 16,
  slots: 2,
  steps: 1,
});
const trainer = Trainer.fromTool(model, {
  optimizer: (params) => new AdamW(params, { lr: 0.01 }),
});
const root = 'investigation-run';
mkdirSync(root, { recursive: true });
await model.savePretrained(join(root, 'initial'));

const hypotheses = [
  { id: 'database', text: 'database connection refused' },
  { id: 'network', text: 'network packet loss' },
];
const cases = [
  ['database connection refused', 'database'],
  ['network packet loss', 'network'],
];
for (const [index, [text, target]] of cases.entries()) {
  const inputs = {
    question: 'which component failed',
    evidence: [{ source_id: `observation:${index}`, text }],
    hypotheses,
  };
  const experience = trainer.capture(inputs, target, { source: `authored-example:${index}` });
  await experience.save(join(root, `experience-${index}.json`), {
    operations: trainer.operations, release: true,
  });
}

const files = readdirSync(root).filter((name) => /^experience-\d+\.json$/.test(name)).sort();
const experiences = await Promise.all(files.map((name) =>
  loadExperience(join(root, name), { operations: trainer.operations })));
const losses = trainer.fit(experiences, { epochs: 60 });
console.log('first / last loss:', losses[0], losses.at(-1));
await model.savePretrained(join(root, 'model'));
await trainer.saveCheckpoint(join(root, 'training'), { progress: { epochs: 60 } });
```

The input encoder receives evidence and hypotheses, not the target label. The
label enters the supervised objective. Explicit source strings identify who
provided the feedback. Predicted scores are not observations.

## Load in a fresh process

Run this separately after the training program:

```ts
import { Investigator } from 'tensorcode/tools';

const model = await Investigator.fromPretrained('./investigation-run/model');
const result = model.call({
  question: 'which component failed',
  evidence: [{ source_id: 'observation:new', text: 'network packet loss' }],
  hypotheses: [
    { id: 'database', text: 'database connection refused' },
    { id: 'network', text: 'network packet loss' },
  ],
});
console.log(result.selected_id);
console.log(result.candidates);
```

The model ranks the supplied hypotheses and returns source-linked evidence plus
workspace diagnostics. Probabilities are uncalibrated. A tiny fixed vocabulary
and two training cases do not show that the model generalizes to new incidents.
The receipt uses the same snake_case fields as the Python package
(`selected_id`, `candidates`, `attention_source_ids`, ...).

`savePretrained` saves the model configuration and weights
(`tensorcode_config.json`, `model.safetensors` and a `README.md` model card).
The Python package loads the same directory with
`Investigator.from_pretrained(...)`. `saveCheckpoint` also saves optimizer
state, training progress, module modes and the TensorCode random generator.
Experiences and chat sessions are separate artifacts.

## Next steps

- [Training](training.md): resume and replay contracts.
- [Tools](tools.md): Hub loading and chat.
- [Operations](operations.md): composing your own programs.
- [Examples](../examples/README.md): larger runnable workflows.
- The complete guides: [tensorcode.dev/docs](https://tensorcode.dev/docs/).
