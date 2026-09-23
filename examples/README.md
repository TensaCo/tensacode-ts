# TensorCode TypeScript examples

These are runnable programs for the TypeScript port of
[TensorCode](https://tensorcode.dev) (documentation:
<https://tensorcode.dev/docs/>). They import the package through its public
entry points (`tensorcode`, `tensorcode/nn`, `tensorcode/ops/*`,
`tensorcode/tools`, `tensorcode/training`, `tensorcode/integrations`). Those
entry points resolve to the local build, so build first:

```bash
npm install            # builds dist/ through the prepare script
node examples/investigatorQuickstart.ts     # Node >= 22.18 runs TypeScript directly
```

On older Node versions, run an example with `npx tsx examples/<name>.ts`. To
typecheck all of them, run `npx tsc -p examples/tsconfig.json`. No example needs
network access. CI runs every example on Node 24.

| Example | Shows |
|---|---|
| [`investigatorQuickstart.ts`](investigatorQuickstart.ts) | The README's 30-second example: an `Investigator` built from configuration, then `Trainer.fromTool` with AdamW, `capture` with explicit provenance, persisted experience files, `fit`, `savePretrained`, `saveCheckpoint`, `fromPretrained` and prediction. |
| [`ownedVectorLifecycle.ts`](ownedVectorLifecycle.ts) | Composed vector operations (`VocabularyEncoder` → `Classify`): tracing and supervision, `latentCodecs()` experience, `Trainer.fromOps`, per-operation `savePretrained` / `fromPretrained` with identical logits, and resuming from the checkpoint. |
| [`supportTriage.ts`](supportTriage.ts) | Structured text operations (`Classify`, `Decide`, `Score`) over an explicit provider, several questions at once with `text.ask`, and a traced message composition. Offline, a deterministic keyword provider stands in for a model. Set `OPENAI_BASE_URL` and `OPENAI_MODEL` to call an OpenAI-compatible endpoint instead. |
| [`traceAndTrain.ts`](traceAndTrain.ts) | A traced `ModuleOperation`, explicit `supervise` targets, `trace.save` / `loadExperience`, `Trainer.fromOps` with an Adam factory, `saveCheckpoint`, and resuming in a **fresh process**. The resumed run restores weights, optimizer, modes, steps, progress and the random generator, then continues exactly. |
| [`pretrainedLifecycle.ts`](pretrainedLifecycle.ts) | Your own owned model (extending `PretrainedModule` from `tensorcode/tools`) with a declared objective: `Trainer.fromTool`, `capture`, `fit`, and `savePretrained` / `fromPretrained` (`tensorcode_config.json` + `model.safetensors` + model card). `--push <repo>` publishes with `pushToHub`, only when explicitly asked (token from `HF_TOKEN` or `hf auth login`). |

Tiny authored cases demonstrate mechanisms. They are not evaluations of model
quality.

## Interoperability with Python

- Experience files and standalone `tensorcode.checkpoint` files are
  byte-compatible with the Python package when the bound operations have the
  same configuration, because the fingerprints are identical.
- Directory checkpoints (`training.json` + `tensors-<id>.safetensors`) written
  by Python load in TypeScript. Model, optimizer, module modes, step count and
  progress are restored. The Python/PyTorch random generator states are
  validated but cannot be reproduced, so they are ignored. TypeScript directory
  checkpoints record the TensorCode generator (`state.rng`,
  `state.runtime = "typescript"`) and do not load in Python. To share models,
  use `savePretrained`; to share training data, use experience files.
- Model artifacts saved by either implementation load in the other, as long as
  both define the same class identity (`static qualifiedName`) and
  architecture. Every built-in tool and owned operation does.
