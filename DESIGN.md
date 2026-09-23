# TensorCode for TypeScript — design

This package ports the Python `tensorcode` library (v0.4.0a3, `../python`) to
TypeScript with **reasonable parity**: the same operations, tools, tracing,
training and artifact formats, expressed idiomatically. Product site and
documentation: <https://tensorcode.dev> (docs at <https://tensorcode.dev/docs/>).

The Python sources and `docs/*.md` remain the behavioral specification. When this
document and the Python behavior disagree on something not listed under
[deliberate differences](#deliberate-differences), Python wins.

## Package

| Item | Decision |
|---|---|
| npm name | `tensorcode` (repository `tensacode-ts`), version `0.4.0-alpha.3` (Python `0.4.0a3`) |
| Modules | ESM only (`"type": "module"`, NodeNext resolution, `.js` import suffixes) |
| Language | TypeScript `strict`, `noImplicitOverride`, `verbatimModuleSyntax`, target ES2022 |
| Build | `npm run build` → `tsc -p tsconfig.build.json` → `dist/` (JS + `.d.ts` + maps) |
| Tests | `npm test` → vitest (`test/**/*.test.ts`), forked pool, 60 s default timeout |
| Runtime | Node ≥ 20.16 (uses `fetch`, `AsyncLocalStorage`, `Intl.Segmenter`, `node:fs`) |
| Dependencies | **none at runtime**. Optional peer `@huggingface/transformers` is only for the explicit external `integrations.LocalModel`, loaded lazily |
| Entry points | `tensorcode` (root: tracing only, mirrors Python `tensorcode`), `tensorcode/nn`, `tensorcode/ops`, `tensorcode/ops/{vec,text,graph}`, `tensorcode/tools`, `tensorcode/tools/*`, `tensorcode/training`, `tensorcode/training/*`, `tensorcode/integrations` |

Importing any entry point performs no I/O and loads no model weights (Python:
"importing the core package does not import PyTorch or access the network").

## Numerical core (`src/nn`)

Python relies on PyTorch. The TypeScript port ships its own small, dependency-free
core so that owned models train, save and load with zero native dependencies:

- **Tensors**: dense, contiguous, row-major CPU tensors over `Float32Array`
  (float32/float16/bfloat16 computed in float32) or `Float64Array` (float64 and
  integer/bool dtypes; int64 exact to 2^53). Reshape-style views share storage and a
  version counter (used by tracing to detect mutation); permute/slice copy.
- **Autograd**: reverse-mode, dynamic graph, `backward()`, `noGrad()`/`enableGrad()`,
  gradient accumulation into leaves, `retainGrad()`. No higher-order gradients.
  Every differentiable op is gradient-checked against finite differences.
- **Modules**: `Module` with explicit `registerParameter/Buffer/Module`, PyTorch
  state-dict names and order, tied parameters, `train()/eval()`, `stateDict()`,
  `loadStateDict()`. Layers: `Linear, Embedding, EmbeddingBag, LayerNorm, Dropout,
  GELU, ReLU, Tanh, Sigmoid, SiLU, Sequential, ModuleList, Conv2d, GRUCell, GRU,
  Identity` with PyTorch parameter names, default initializations, qualified names
  (`torch.nn.modules.linear.Linear`) and `configurationAttributes()` equal to
  PyTorch's `vars(module)` (so fingerprints match Python).
- **Optimizers**: `SGD` (momentum, dampening, nesterov, weight decay), `Adam`
  (amsgrad), `AdamW`, with PyTorch-compatible `stateDict()` layouts.
- **Safetensors**: byte-identical serialization (Rust ordering, header padding,
  tied-tensor metadata of `safetensors.torch.save_model`), all common dtypes.
- **RNG**: seedable xoshiro128** `Generator` with JSON-safe state (checkpointable).
  Streams are deterministic but **do not** reproduce PyTorch random sequences.

PyTorch parity (layers, optimizers, safetensors) is verified against fixtures
generated from the Python stack (`scripts/fixtures`).

## Native transformer architectures (`src/_internal/native`)

Tools and owned operations construct Hugging Face architectures from JSON
configuration and import pretrained safetensors weights. TypeScript implements
them natively with transformers **5.17** parameter names and numerics:

| `model_type` | Heads | Notes |
|---|---|---|
| `bert`, `roberta`, `electra`, `distilbert` | base (`AutoModel`), sequence classification | fixtures match to ~1e-5 |
| `t5` | `seq2seq` (`T5ForConditionalGeneration`), `encoder` | teacher-forced loss, KV-cached decoding, greedy / sampling / beam search matching `generate()` |
| `vit` | base (`ViTModel`, 5.17 layout `layers.N.attention.q_proj`) | legacy checkpoint key renames on load |
| `clip` | `CLIPModel` | 5.17 `get_*_features` return projected pooler outputs |
| `deberta-v2` | base, sequence classification | **extension point** (`debertaV2.ts`, module *training*) — NLI verifiers such as `cross-encoder/nli-deberta-v3-small` |

- `NativeConfig` emulates `AutoConfig.for_model(...)` + `to_dict()` /
  `to_json_string()` from tables generated out of transformers
  (`scripts/codegen/native_defaults.py` → `defaults.generated.ts`). Configurations
  already carrying `transformers_version` are kept **verbatim**, so saved artifacts
  reconstruct exactly. `nativeConfig()` mirrors Python `_native_config`.
- `loadNativeFoundation(repo, {head, revision, localFilesOnly, ...})` mirrors
  `AutoModel*.from_pretrained(use_safetensors=True)`: Hub download, config
  normalization (`_name_or_path`, `dtype`, dropped legacy generation keys, CLIP
  `*_config_dict`), transformers 5 `dtype="auto"` (the dtype in `config.json`,
  else the checkpoint's first floating tensor; the text and vector foundation
  factories also accept an explicit `dtype`), `base_model_prefix` stripping, legacy
  ViT renames, transformers' tie/untie rules (flan-T5: embeddings tied, `lm_head`
  untied), missing/mismatched weight rejection, tokenizer and generation config.
- `generateSeq2Seq(model, inputs, settings, {generationConfig})` takes
  `GenerationConfig`-style snake_case settings (`max_new_tokens`, `num_beams`, ...).
- `parameterAliases()` / `restoreParameterAliases()` mirror Python's
  `native_parameter_aliases` handling.

Verified against real cached checkpoints when present (`google/flan-t5-small`,
`google/electra-small-discriminator`, `openai/clip-vit-base-patch32`): identical
configs, parameter aliases, tokenizer JSON hash, encoder outputs, greedy/beam
generations and decoded text.

## Tokenizers (`src/_internal/tokenizers`)

A dependency-free `tokenizer.json` runtime (WordPiece, BPE, Unigram, WordLevel;
BERT/Metaspace/ByteLevel/Split/... pre-tokenizers; Precompiled SentencePiece
normalization; Template/Bert/Roberta post-processing; decoders) and
`FastTokenizer`, the `PreTrainedTokenizerFast` equivalent:

- `FastTokenizer.fromConfiguration(config)` ↔ `configuration()` round-trips Python
  `_tokenizer_config` exactly (`json`, `options`, `special_tokens`, sides).
- `FastTokenizer.fromJsonString(json, specialTokens)` = Python
  `PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json), **special)`.
- `FastTokenizer.fromDirectory(dir)` = `AutoTokenizer.from_pretrained` for the
  supported classes (emulates transformers 5 class defaults and the rebuilt T5
  pipeline; CLIP decode wrapper).
- `encode(texts, {padding, truncation, maxLength, textPair, addSpecialTokens})`,
  `encodeTensors(...)` (int64 tensors keyed `input_ids`/`attention_mask`/
  `token_type_ids`), `decode`, `batchDecode`.

## Public API mapping

Legend for **Owner**: `core` = foundation (read-only for module builders),
`vec`, `text`, `tools`, `training` = the four parallel modules, `integrator` =
wired after the modules land.

| Python module | TypeScript module | Owner |
|---|---|---|
| `tensorcode/__init__.py` | `src/index.ts` | integrator (core stub exists) |
| `ops/base.py` | `src/ops/base.ts`, `src/ops/index.ts` | core |
| `_internal/tracing.py` | `src/_internal/tracing.ts` | core |
| `_internal/operation_config.py` | `src/_internal/operationConfig.ts` | core |
| `_internal/pretrained.py` | `src/_internal/pretrained.ts` (+ `hub.ts`, `files.ts`) | core |
| `_internal/latent_ops.py` | `src/_internal/latentOps.ts` | core |
| `_internal/workspace.py` | `src/_internal/workspace.ts` | core |
| `_internal/ranking.py` | `src/_internal/ranking.ts` | core |
| `_internal/proposals.py` (`conversation_*`) | `src/_internal/conversation.ts` | core |
| `_internal/proposals.py` (rest) | `src/_internal/proposals.ts` | tools |
| `_internal/vec/adapter.py` | `src/_internal/vec/adapter.ts` | core |
| `ops/vec/_configuration.py` | `src/_internal/vec/configuration.ts` | core |
| `_internal/training/persistence.py` (`configuration`, `fingerprint`, `bindings`, `validate_bindings`) | `src/_internal/fingerprint.ts` | core |
| `ops/vec/latent.py` | `src/ops/vec/latent.ts` | core |
| `ops/text/messages.py`, `ops/text/model.py` (+ `InvalidModelOutput`) | `src/ops/text/messages.ts`, `src/ops/text/model.ts` | core |
| `training/calibration.py` | `src/training/calibration.ts` | core |
| `transformers` models/tokenizers | `src/_internal/native/*`, `src/_internal/tokenizers/*` | core |
| `ops/vec/{transform,classify,candidates,score,decide,retrieve,encode,decode,__init__}.py` | `src/ops/vec/*.ts`, `src/ops/vec/index.ts` | vec |
| `_internal/vec/{owned,text,vision,patch,vocabulary,diffusion}.py` | `src/_internal/vec/*.ts` | vec |
| `ops/graph/*` | `src/ops/graph/*.ts`, `src/ops/graph/index.ts` | vec |
| `tools/scene.py` | `src/tools/scene.ts` | vec |
| `ops/text/*` (except messages/model) | `src/ops/text/*.ts`, `src/ops/text/index.ts` | text |
| `_internal/text/{native,owned}.py` | `src/_internal/text/{native,owned}.ts` | text |
| `integrations/*` | `src/integrations/*.ts`, `src/integrations/index.ts` | text |
| `tools/{chatbot,investigator,planner,cognition,actions}.py`, `tools/decision/` | `src/tools/{chatbot,investigator,planner,cognition,actions,decision}.ts` | tools |
| `_internal/{proposals,retrieval}.py`, `_internal/{cognition,memory,sessions,execution}/*` | `src/_internal/...` (camelCase file names) | tools |
| `_internal/text/realization.py`, `_internal/vec/sequence.py` | `src/_internal/text/realization.ts`, `src/_internal/vec/sequence.ts` | tools |
| `tools/__init__.py` | `src/tools/index.ts` (also exports `PretrainedModule`) | integrator |
| `training/__init__.py` | `src/training/index.ts` | training |
| `_internal/training/{persistence,trainer,tool,checkpoint,_tensor_store}.py` | `src/_internal/training/*.ts` | training |
| `PretrainedTool.push_to_hub` transport | `src/_internal/hubUpload.ts` | training |
| `_internal/response_quality.py` | `src/_internal/responseQuality.ts` | training |
| DeBERTa-v2 (transformers) | `src/_internal/native/debertaV2.ts` | training |
| `examples/`, package `README.md`, `docs/` | `examples/`, `README.md`, `docs/` | training, integrator |
| float summation (`sum`, `math.fsum`) | `src/_internal/numeric.ts` | integrator |

## Conventions (every builder)

**Naming.** Methods and options are camelCase with the same concepts as Python:
`savePretrained`, `fromPretrained`, `pushToHub`, `fromFoundation`,
`fromFoundations`, `fromModule`, `fromModel`, `operationBindings`,
`trainingOperation`, `newSession`, `loadCognitiveSession`, `lossBatch`,
`generateBatch`, `verificationLoss`, ... Keyword arguments become one trailing
options object (`fromPretrained(source, { revision, localFilesOnly, cacheDir,
token, device })`). File names are camelCase versions of the Python module
(`response_quality.py` → `responseQuality.ts`, `_structured.py` → `structured.ts`).
No default exports.

**Persisted and reported data keeps Python spelling.** JSON configurations,
manifests, receipts, reports, experience/checkpoint files, record fields and
`configuration()` results use Python's snake_case keys and values so artifacts are
interchangeable (`{selected_id, candidates, attention_source_ids}`,
`{'max_new_tokens': 32}`). Only the TypeScript API surface is camelCase.

**Identities.** Every public or persisted class declares
`static override readonly qualifiedName: string = '<python module>.<Class>'`
(for example `'tensorcode.tools.investigator.Investigator'`,
`'tensorcode.ops.vec.transform.Transform'`). `qualifiedName(x)` reads only a
class's own static, so subclasses never inherit a library identity silently.
Owned objectives with `_operation_identity` implement `operationIdentity()`.

**Operations.** `op.call(value, { context })` (Python `op(value, context=...)`),
`await op.acall(value, { context })`, implementation in `forward(value, context)`
(`context` is `null` when absent or empty) and optionally `aforward`. Weightless
operations extend `Operation`/`ConfigOperation`; tensor-owning ones extend
`ModuleOperation`/`ConfigModuleOperation` or `LatentOperation` (owned model +
traced call). Pure operations opt into replay with `override get replayable()
{ return true; }`. Tools extend `PretrainedModule` (their `call()` is not a trace
boundary, like `nn.Module.__call__`).

**Configuration.** Public constructors take JSON configuration only, never
executable models. Validate with `validatedConfig(config, allowedKeys, defaults)`
or `rejectUnknownFields`; unknown **and obsolete** fields raise `ValueError`
(`Unknown configuration fields: [...]`). `configuration()` returns fresh JSON.

**Async and I/O.** Anything touching the filesystem or network returns a
`Promise` (`savePretrained`, `fromPretrained`, `fromFoundation`, `trace.save`,
`loadExperience`, checkpoints, sessions). Pure computation is synchronous. HTTP
providers are asynchronous (`acomplete`); operations wrapping them are used with
`acall`.

**Values.** Python tuples are frozen arrays (`Object.freeze([...])`, see
`tuple()`); lists are arrays; dicts are plain objects; `bytes` are `Uint8Array`;
dataclasses are records (`static recordFields`, `static fromRecord`,
`toRecord()` with snake_case field names) — the tracer and experience codecs walk
them. Tensors are `Tensor` (int64 for ids/indices, bool masks).

**Errors.** `TypeError` for wrong kinds, `ValueError` (from `src/errors.ts`) for
invalid values, `NotImplementedError` for reserved/unimplemented interfaces and
out-of-scope components, `RangeError` inside the numerical core. Library
families subclass these (`InvalidModelOutput extends ValueError`,
`ProviderError`, `HubError`, `FileNotFoundError`). Error messages should follow the
Python wording where tests match on them.

**Floats.** JavaScript numbers do not distinguish `1` from `1.0`. Python-compatible
JSON writers spell integral numbers under `PYTHON_FLOAT_KEYS` as floats (see
`src/_internal/json.ts`). If a module persists a new float-typed field whose value
can be integral, list it in the module brief handoff for the integrator to add
statically — never call `registerPythonFloatKeys` at import time (fingerprints
must not depend on import order).

**Devices.** CPU only. `device` options accept `'cpu'` (or omission) and reject
others with `ValueError`.

## Tests and fixtures

- Layout: `test/<area>/<name>.test.ts` (`test/vec`, `test/graph`, `test/text`,
  `test/integrations`, `test/tools`, `test/runtime`, `test/training`, ...). Port the
  corresponding Python tests (`../python/tests/...`) test-for-test where the
  behavior is in scope; add parity tests against Python fixtures.
- Fixtures: Python generators in `scripts/fixtures/<name>_fixtures.py` exposing
  `generate()`; `generate.py` discovers them automatically. Write outputs under
  `test/fixtures/<area>/` (keep fixtures small; never commit Hub weights). Run one
  generator with `npm run fixtures -- <name>_fixtures`.
- The suite must pass **without Python and without network**. Tests that need a
  cached Hub model use `cachedSnapshot()` (`test/helpers/hub.ts`) and `it.skipIf`.
- Helpers: `test/helpers/gradcheck.ts` (`gradcheck`, `expectClose`,
  `randomTensor`), `test/helpers/fixtures.ts`, `test/helpers/hub.ts`.

## Deliberate differences

- **Numerics.** Pure JavaScript CPU compute; no GPU, no mixed precision kernels.
  Real foundations run (flan-T5-small generates in seconds) but large models are
  slow. Random streams differ from PyTorch (initializations, dropout, sampling),
  so freshly initialized models are not numerically identical to Python ones;
  loaded weights are.
- **Weights.** Only safetensors checkpoints (no `pytorch_model.bin`/pickle); the
  Hub safetensors-conversion PR lookup is not emulated — pin a revision that
  contains `model.safetensors`.
- **Out of scope (explicit `NotImplementedError` with a clear message):**
  `ops.vec.ImageDecoder`/`ImageDecode` latent diffusion (diffusers UNet/VAE/DDIM);
  `Scene.fromLanguageFoundation`/`interpret` (Idefics3/SmolVLM); image file
  decoding (callers supply decoded CHW float tensors; resize/normalize helpers are
  provided). Graph operations stay symbolic stubs exactly as in Python.
- **External local models.** `integrations.LocalModel` wraps an explicitly
  supplied `@huggingface/transformers` model/processor (optional peer, dynamic
  import) and is asynchronous.
- **Callbacks.** JavaScript cannot distinguish closures from module functions, so
  persisted callbacks (e.g. `combine`) always need explicit `configuration()`.
- **Checkpoint RNG.** TypeScript checkpoints store the TensorCode generator state;
  PyTorch/Python RNG states cannot be restored in TypeScript (and vice versa).
- **Tokenizer JSON.** Artifacts embed the canonical backend JSON (sorted keys),
  as Python does. Wherever Python builds the backend with Rust
  `Tokenizer.from_str`/`from_file` (tokenizer configurations, tool
  `tokenizer_json`, generic fast tokenizers), TypeScript reproduces the Rust
  `serde_json` float parsing, which can move Unigram scores by one ULP
  (`rustJsonF64`). Class-specific transformers tokenizers (T5, DeBERTa-v2,
  ALBERT, ...) rebuild their vocabulary exactly and are loaded exactly; for
  `T5Tokenizer`, `DebertaV2Tokenizer` and `AlbertTokenizer` the pipeline that
  transformers 5 rebuilds from `tokenizer_config.json` flags is reproduced as
  well (`src/_internal/tokenizers/serialization.ts`). Embedded
  tokenizer JSON, configurations and fingerprints of real foundations (for
  example `google/flan-t5-small`) therefore equal Python's.
- **Python examples/research scripts** are not ported beyond the quickstart,
  lifecycle and triage examples in `examples/`.
- **Integral floats.** JavaScript cannot tell `1` from `1.0`. Configuration keys in
  `PYTHON_FLOAT_KEYS` (and session float fields) are written as floats; other
  integral numbers (for example JSON-dumped structured training targets) are
  written as ints. A Python user who passes an int for a float field
  (`hidden_dropout_prob=0`) gets a different fingerprint than TypeScript's `0.0`.
- **Object key order.** Integer-like keys of plain objects are reordered by
  JavaScript (affects `Retrieve` items keyed by integers and non-string Python dict
  keys, which decode as decimal strings).
- **Async providers.** HTTP (`OpenAICompatibleModel`, `JevModel`) and local
  providers are asynchronous only; use `acall`/`aask`. A redirect raises
  `ProviderHTTPError` with its 3xx `.status`, as in Python.
- **Plan actions** receive `(state, args)`; plan validation cannot bind keyword
  arguments against a JavaScript signature. Error observations record JavaScript
  error names (`Error` where Python records `RuntimeError`).
- **Threads.** Python's `RLock`s become single-threaded execution plus a promise
  queue that serializes session persistence.

## Interoperability guarantees

- Model artifacts (`tensorcode_config.json` + `model.safetensors`) written by
  Python load in TypeScript and re-save byte-identically when the TypeScript class
  reconstructs the same architecture (`test/internal/pretrained.test.ts`). One
  qualification: for models with two or more tied parameter aliases (T5), Python's
  `safetensors` writes the `__metadata__` alias entries in nondeterministic (Rust
  `HashMap`) order, so two Python saves already differ; such files match up to
  metadata order with identical tensor bytes.
- Configuration-only operation artifacts (`tensorcode.operation`) are identical.
- Operation fingerprints (`src/_internal/fingerprint.ts`) equal Python's for the
  same configuration (`test/internal/ranking.test.ts`), so experience files are
  portable once the training module implements the codec.
- Experience files and standalone `tensorcode.checkpoint` files are interchangeable.
  Python directory checkpoints load in TypeScript (their PyTorch/CPython RNG states
  are validated and ignored); TypeScript directory checkpoints do not load in Python.
- Tool session files (chat, ranking, cognitive session/state, trajectories, JSON
  memory) written by Python load in TypeScript and re-save byte-identically.
- The Hugging Face cache layout is shared with `huggingface_hub`.

## Former extension points (now implemented)

| File | Owner | Contract |
|---|---|---|
| `src/_internal/training/persistence.ts` | training | `saveExperience(trace, path, {operations, codecs})`, `loadExperience(path, {operations, codecs})` (called by `Trace.save`) |
| `src/_internal/hubUpload.ts` | training | `uploadFolder({repoId, folderPath, private, revision, token, commitMessage})` (called by `pushToHub`) |
| `src/_internal/native/debertaV2.ts` | training | `createDebertaV2Model(config)`, `createDebertaV2ForSequenceClassification(config)` (routed by `registry.ts`) |

Cross-module contracts live in `src/_internal/contracts.ts` (`TrainableTool`,
`HasParameters`, `parseObjectiveEnvelope`) and `src/ops/text/model.ts` (model
protocols).

## Commands

```bash
npm install
npm run build          # tsc → dist/
npm test               # vitest (no Python, no network required)
npm run typecheck      # tsc --noEmit including tests
npm run fixtures -- <generator>   # regenerate fixtures with ../python/.venv
npm run codegen        # regenerate native config defaults from transformers
```
