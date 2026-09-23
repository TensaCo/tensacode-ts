# Tracing and training

Tools own their parameters and expose stable operation bindings. There are two
ways to build a trainer:

- `Trainer.fromTool(model)` trains the tool's declared objective.
- `Trainer.fromOps(operations, { losses })` trains an explicitly supervised
  program.

Both replay supported local tensor paths with gradients through the built-in
autograd core. This guide mirrors the Python
[training guide](https://tensorcode.dev/docs/). Experience files and
standalone checkpoint files are interchangeable between the two packages. See
the [quickstart](quickstart.md) for a complete runnable program.

## Train an owned tool

```ts
import { Trainer, loadExperience } from 'tensorcode/training';

// Construct or load the model first; all parameters must already exist.
const trainer = Trainer.fromTool(model, { lr: 0.001 });
const experience = trainer.capture(inputs, targets, { source: 'review:42' });
await experience.save('experience.json', { operations: trainer.operations, release: true });

const loaded = await loadExperience('experience.json', { operations: trainer.operations });
const losses = trainer.fit([loaded], { epochs: 10 });
await model.savePretrained('./model');
await trainer.saveCheckpoint('./training', { progress: { next_example: 43 } });
```

`Trainer.fromTool(tool, { optimizer, lr = 0.001 })` uses the tool's declared
training operation and objective, and it applies the tool's training-mode
policy. It defaults to SGD. To override that, pass an optimizer instance or a
factory `(params) => new AdamW(params, { lr })`. `capture` stores snapshotted
inputs and explicit sourced targets; it performs no optimizer step. `step`
returns one loss, and `fit` returns one loss per update.

Investigator and Decision targets name a supplied hypothesis by ID or index, or
give a finite nonnegative distribution that sums to one. Scene targets name a
supplied description. Planner targets contain `candidate_id` and the observed
numeric `outcome`. Chatbot inputs and targets are equal-length text lists
trained with teacher-forced cross-entropy. Feedback never becomes input
evidence.

## Resume training

```ts
const model = await Investigator.fromPretrained('./model');
const trainer = Trainer.fromTool(model, { lr: 0.001 });
const progress = await trainer.loadCheckpoint('./training');
const experience = await loadExperience('experience.json', { operations: trainer.operations });
trainer.fit([experience], { epochs: 1 });
```

Rebuild the same optimizer type when you use a custom optimizer.
`loadCheckpoint` restores the following and returns your progress mapping:

- model and optimizer state
- module training/eval modes
- `trainer.steps`
- the TensorCode random generator

Record data cursors in `progress`. If a load fails, weights, optimizer state,
modes and the generator are all rolled back.

A directory checkpoint is `training.json` plus a checksummed
`tensors-<id>.safetensors` file. Keep them together. `savePretrained` exports
only the model; experiences and chat sessions are saved separately.

**Cross-language checkpoints.**

| Checkpoint | Python → TypeScript | TypeScript → Python |
|---|---|---|
| Standalone `tensorcode.checkpoint` files (model and optimizer only) | Yes | Yes |
| Directory checkpoints | Model, optimizer, modes, steps and progress restore. The PyTorch/CPython RNG states are validated but ignored, because TypeScript cannot reproduce those streams | Not supported: TypeScript writes its own generator state |

To share work across languages, use `savePretrained` artifacts and experience
files.

## Compose and trace individual operations

Tracing records call boundaries and dependencies, whether or not you use a tool
or a trainer. It keeps tensor gradients, snapshots external inputs and keeps
feedback separate from predictions. Plain JavaScript transformations, remote
services and discrete choices do not become differentiable just because they
were observed.

```ts
import { trace } from 'tensorcode';
import { Trainer, loadExperience } from 'tensorcode/training';

const session = trace();
const prediction = session.run(() => head.call(encoder.call(text)));
session.supervise(prediction, 'target label', { source: 'human:review-42' });
const operations = { encoder, head };
await session.save('experience.json', { operations, release: true });
const experience = await loadExperience('experience.json', { operations });
const trainer = Trainer.fromOps(operations, { lr: 0.01 });
const losses = trainer.fit([experience], { epochs: 10 });
await trainer.saveCheckpoint('./program-training', { progress: { epochs: 10 } });
```

`session.run(fn)` captures every operation call made inside `fn`. For async
code, `await session.run(async () => ...)` keeps capturing across `await`s.
`fromOps` binds an already-built program and does not change its module modes.
An operation trainer does not infer an objective, so use `trace()` and
`supervise()` as shown above. See
[`examples/traceAndTrain.ts`](../examples/traceAndTrain.ts) and
[`examples/ownedVectorLifecycle.ts`](../examples/ownedVectorLifecycle.ts).

## Capture and supervision

`trace()` returns a public `Trace`. `InputRef` and `OutputRef` (both from
`tensorcode`) are the reference types for explicit dependencies.

- `session.ref(output)` returns an `OutputRef`. Equal-valued independent outputs
  are not merged. Scalars need explicit `session.calls[index].output` handles.
- `session.example(target)` extracts the dependency closure and its external
  roots.
- `session.supervise(outputOrRef, target, { loss = 'cross_entropy', source })`
  stores a supervision record with snapshotted target data. `source` is a
  nonempty provenance string, and feedback is never inferred from the output.

Mutating a captured intermediate is rejected before reuse, save or release. This
covers in-place tensor writes, which are tracked by version counters, and
container changes, which are tracked by structural stamps. A transient change
restored before the call completes can evade these checks.

## Portable experience and replay

`await session.save(path, { operations, codecs, release })` atomically writes
versioned JSON (`tensorcode.experience`). Bind stable names to the exact
operation instances that were captured.

`await loadExperience(path, { operations, codecs })` returns a trace bound to
operations you have already built. It validates configuration fingerprints,
bindings, the format version, DAG references and codec tags. It never
constructs classes named by the artifact and never restores callbacks.
Fingerprints are the same SHA-256 values Python computes, so an experience
recorded in Python replays in TypeScript and vice versa.

`session.replay(target, { inputs, boundary: 'error' | 'recorded' })` recomputes
pure operations with the current parameters. External effects are rejected by
default. With `boundary: 'recorded'`, their captured results are reused as
constants. `session.release()` drops live results and autograd graphs after
validating the whole session.

## Configuration and codecs

Operations expose `configuration()`, JSON-safe data describing their stable
semantic choices. Fingerprints also cover the qualified operation type
(`static qualifiedName`), replay capability and tensor-state shape and dtype.
They exclude learned values, so updated weights can still load compatible
experience.

Built-in codecs cover primitives, `Uint8Array` bytes, arrays, frozen arrays
(Python tuples), plain objects and finite dense tensors. Records (the
counterpart of frozen dataclasses: a class with `static recordFields`,
`static fromRecord` and `toRecord()`) need an explicit allowlist when saving and
loading, such as `codecs: { 'record-v1': MyRecord }`. `latentCodecs()` from
`tensorcode/ops/vec` covers `Latent`, `Space` and the vector results. Message
experiences need `codecs: { message: text.Message }`.

## Optimizers, losses and checkpoints

`Trainer.fromOps(operations, { optimizer, lr = 0.01, losses })` defaults to SGD.
The optimizer must own exactly the deduplicated trainable parameters.
`tensorcode/nn` provides `SGD` (momentum, nesterov, weight decay), `Adam`
(amsgrad) and `AdamW`, all with PyTorch-compatible state-dict layouts. Trainers
expose `operations`, `parameters` and `optimizer`, plus settable `steps` and
`progress`; `fromTool` trainers also expose `tool`.

`trainer.step(experience)` averages the experience's explicit losses and
performs one update. Cross-entropy accepts integer indices or prediction label
strings, and MSE requires exactly matching shapes. Register custom in-process
losses with `losses: { name: (output, target) => tensor }`; callbacks are never
serialized. Nondifferentiable, disconnected and nonfinite losses are rejected.

## Calibrate verifier scores

```ts
import { TemperatureCalibration, evaluateCalibration, fitThreshold } from 'tensorcode/training';

// heldOutLogits: [examples, classes]; labels: int64 class indices.
const calibration = new TemperatureCalibration();
const report = calibration.fit(heldOutLogits, labels);
const calibrated = calibration.forward(testLogits);
const metrics = evaluateCalibration(calibrated, testLabels);
const threshold = fitThreshold(confidences, correct, { maxError: 0.05 });
```

For an owned Investigator verifier, fit `model.verifier.calibration` so its
buffers persist with `savePretrained`. The Investigator selects an objective
through a mode envelope:

```ts
const trainer = Trainer.fromTool(model, { lr: 0.0001 });
const experience = trainer.capture({
  mode: 'verification',
  inputs: [{ premise: 'The connection was refused.', hypothesis: 'The connection succeeded.' }],
}, ['contradiction'], { source: 'authored-example:review-17' });
```

The modes are `verification`, `proposal`, `rank` and `retrieval`. Ordinary
ranking inputs without an envelope keep the ranking objective. Changing verifier
weights invalidates its calibration, so refit on a separate held-out set.
`fitThreshold` reports an empirical threshold, coverage and error on the
calibration sample. It does not certify error rates on new inputs.

## Train owned retrieval

An Investigator with a `retrieval_encoder` configuration owns
`model.episodicEncoder`. Positive relationships belong in the targets: a
`[queries, documents]` boolean matrix with at least one positive per query.

```ts
const experience = trainer.capture({
  mode: 'retrieval',
  inputs: {
    queries: ['How was the service restored?'],
    documents: ['Restoring the database connection recovered the service.',
                'The maintenance window starts tomorrow.'],
  },
}, [[true, false]], { source: 'authored-example:retrieval-review-17' });
```

Unmarked documents act as negatives. After changing the encoder's weights,
rebuild existing episodic indexes. Live indexes reject stale fingerprints.
