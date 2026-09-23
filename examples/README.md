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
typecheck all of them, run `npx tsc -p examples/tsconfig.json`.

## Offline tours

These need no network access or model downloads. CI runs them on Node 24.

| Example | Shows |
|---|---|
| [`investigatorQuickstart.ts`](investigatorQuickstart.ts) | The README's 30-second example: an `Investigator` built from configuration, then `Trainer.fromTool` with AdamW, `capture` with explicit provenance, persisted experience files, `fit`, `savePretrained`, `saveCheckpoint`, `fromPretrained` and prediction. |
| [`ownedVectorLifecycle.ts`](ownedVectorLifecycle.ts) | Composed vector operations (`VocabularyEncoder` → `Classify`): tracing and supervision, `latentCodecs()` experience, `Trainer.fromOps`, per-operation `savePretrained` / `fromPretrained` with identical logits, and resuming from the checkpoint. |
| [`supportTriage.ts`](supportTriage.ts) | Structured text operations (`Classify`, `Decide`, `Score`) over an explicit provider, several questions at once with `text.ask`, and a traced message composition. Offline, a deterministic keyword provider stands in for a model. Set `OPENAI_BASE_URL` and `OPENAI_MODEL` to call an OpenAI-compatible endpoint instead. With `--input tickets.jsonl --policy policy.txt --label ...` it is Python's `support_triage.py` CLI. |
| [`traceAndTrain.ts`](traceAndTrain.ts) | A traced `ModuleOperation`, explicit `supervise` targets, `trace.save` / `loadExperience`, `Trainer.fromOps` with an Adam factory, `saveCheckpoint`, and resuming in a **fresh process**. The resumed run restores weights, optimizer, modes, steps, progress and the random generator, then continues exactly. |
| [`pretrainedLifecycle.ts`](pretrainedLifecycle.ts) | Your own owned model (extending `PretrainedModule` from `tensorcode/tools`) with a declared objective: `Trainer.fromTool`, `capture`, `fit`, and `savePretrained` / `fromPretrained` (`tensorcode_config.json` + `model.safetensors` + model card). `--push <repo>` publishes with `pushToHub`, only when explicitly asked (token from `HF_TOKEN` or `hf auth login`). |

Tiny authored cases demonstrate mechanisms. They are not evaluations of model
quality.

## Ports of the Python examples

Each program below is the TypeScript port of the Python example of the same
name in [`python/examples`](../../python/examples), with the same command-line
options (spelled `--kebab-case`), inputs, outputs and artifact files. Seeded
runs reproduce the Python numbers: the port uses PyTorch-compatible random
streams and Python's `random` algorithm (`PythonRandom`) where the Python
example shuffles, and experience files, checkpoints and manifests written by
the two are interchangeable (often byte-identical). Pass `--help`-style flags
as in the Python docstrings; every file's header shows a complete command.
HTTP examples call your own OpenAI-compatible endpoint synchronously, as the
Python examples do; selected file contents are sent to that endpoint.

| Example (Python source) | Input and output |
|---|---|
| [`pretrainedLatentLifecycle.ts`](pretrainedLatentLifecycle.ts) (`pretrained_latent_lifecycle.py`) | Pinned FLAN-T5 vectors + four authored targets → trained linear bridge, durable experience, checkpoint and restored weights. Downloads `google/flan-t5-small` unless cached |
| [`outputEncodingLearning.ts`](outputEncodingLearning.ts) (`output_encoding_learning.py`) | Reviewed `{text, target}` JSONL → trained OUTPUT_ENCODING readout and decoder bridge, reloaded with an exact loss check |
| [`hypothesisLearning.ts`](hypothesisLearning.ts) (`hypothesis_learning.py`) | `collect` / `train` / `predict` stages: reviewed evidence sequences → revisable interpretations and saved weights |
| [`planLearning.ts`](planLearning.ts) (`plan_learning.py`) | `collect` / `train` / `predict` stages: observed plan outcomes → learned candidate rankings (never executes a plan) |
| [`learnActionOutcomes.ts`](learnActionOutcomes.ts) (`learn_action_outcomes.py`) | Executed simulated transitions → sourced outcome feedback, trained Planner, trajectories, session and exact restore checks (offline) |
| [`banking77Restart.ts`](banking77Restart.ts) (`banking77_restart.py`) | Official Banking77 CSVs → persisted traces, training and held-out evaluation, each stage in its own process |
| [`trainChatbot.ts`](trainChatbot.ts) (`train_chatbot.py`) | Reviewed input/target JSONL → trained Chatbot, held-out report, ablations and model card |
| [`pretrainedChatbot.ts`](pretrainedChatbot.ts) (`pretrained_chatbot.py`) | Local or Hub Chatbot → one turn (`--prompt`) or an interactive session, with saved sessions |
| [`trainCognitiveTools.ts`](trainCognitiveTools.ts) (`train_cognitive_tools.py`) | Pinned HotpotQA support annotations → Investigator and Planner document rankers, workspace ablation and lexical baseline |
| [`evaluateCognition.ts`](evaluateCognition.ts) (`evaluate_cognition.py`) | Cognitive Chatbot + evidence cases → answers, abstentions, omission/replacement controls, episodic retrieval; optional component assembly |
| [`evaluateTypedDecisions.ts`](evaluateTypedDecisions.ts) (`evaluate_typed_decisions.py`) | Foundation + Banking77 rows / reviewed candidates → generated-JSON vs likelihood decoding validity, accuracy and calibration |
| [`trainScene.ts`](trainScene.ts) (`train_scene.py`) | Images + candidate descriptions → trained scene ranker with image/workspace ablations |
| [`evaluateSceneLanguage.ts`](evaluateSceneLanguage.ts) (`evaluate_scene_language.py`) | Images + spatial yes/no captions → judgments under real, blank and shuffled images (SmolVLM; minutes per interpretation on a CPU) |
| [`documentSearch.ts`](documentSearch.ts) (`document_search.py`) | Text/Markdown directory + question → answer with validated citations of retrieved excerpts |
| [`researchAssistant.ts`](researchAssistant.ts) (`research_assistant.py`) | Local documents + question → bounded `actionLoop` of search/read/finish actions with receipts |
| [`imageInspection.ts`](imageInspection.ts) (`image_inspection.py`) | One image + question → answer from a Transformers.js `LocalModel` or an OpenAI-compatible endpoint |
| [`localMultimodal.ts`](localMultimodal.ts) (`local_multimodal.py`) | One image + a supplied local vision-language model → recorded answers and structured-output failures |

`trainCognitiveTools.ts` reads the pinned HotpotQA rows through the Hugging
Face dataset viewer (it first checks that the viewer serves the pinned
revision) instead of downloading and parsing the Parquet shards, and records
the shards' SHA-256 from the Hub; the rows and results equal the Python run.

Not ported: the examples that require the CUDA training host
(`compare_cognitive_verifiers.py`, `train_hypotheses.py`,
`train_realization.py`, `train_verifier.py`, `prepare_response_quality.py`,
`train_response_quality.py`), and the optional ViT/Stable Diffusion image path
of `pretrained_latent_lifecycle.py` (`--image-input`).

## Interoperability with Python

- Experience files and standalone `tensorcode.checkpoint` files are
  byte-compatible with the Python package when the bound operations have the
  same configuration, because the fingerprints are identical.
- Directory checkpoints (`training.json` + `tensors-<id>.safetensors`) move
  between the languages in both directions. Model, optimizer, module modes,
  step count, progress and the PyTorch (`torch_rng`) and CPython
  (`python_rng`) random states are restored, so a resumed run draws the same
  random numbers as it would in Python.
- Model artifacts saved by either implementation load in the other, as long as
  both define the same class identity (`static qualifiedName`) and
  architecture. Every built-in tool and owned operation does.
