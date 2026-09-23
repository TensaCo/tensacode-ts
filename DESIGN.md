# TensorCode for TypeScript — design

This package ports the Python `tensorcode` library (v0.4.0a4, `../python`) to
TypeScript with **full parity** apart from the few
[remaining differences](#remaining-differences): the same operations, tools,
tracing, training and artifact formats, expressed idiomatically. Product site and
documentation: <https://tensorcode.dev> (docs at <https://tensorcode.dev/docs/>).

The Python sources and `docs/*.md` remain the behavioral specification. When this
document and the Python behavior disagree on something not listed under
[remaining differences](#remaining-differences), Python wins.

## Package

| Item | Decision |
|---|---|
| npm name | `tensorcode` (repository `tensacode-ts`), version `0.4.0-alpha.4` (Python `0.4.0a4`) |
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
- **Compute backend** (`src/nn/backend`): float32 matrix products (`linear`,
  batched/broadcast `matmul` and their gradients, `conv2d` as im2col products),
  attention, last-dimension softmax, layer normalization, the GELU / SiLU /
  sigmoid / tanh / exp activations, and large adjacent-axis permutes and
  same-shape arithmetic run on WebAssembly SIMD kernels (`scripts/wasm/kernels.rs`,
  built by `node scripts/wasm/build.mjs` into the checked-in
  `kernels.generated.ts`; relaxed-SIMD fused multiply-add where the runtime
  supports it; no toolchain is needed to install or build the package). On Node
  the kernels share one memory with a `worker_threads` pool sized to the
  available parallelism; the calling thread takes part and blocks until a job
  is done, so the API stays synchronous. Results are copied out of kernel
  memory on the calling thread into ordinary `Float32Array`s (workers never
  hold references to results, which their rarely collected heaps would keep
  alive). `TENSORCODE_THREADS` / `setNumThreads(n)` set the thread count
  (`1`: no workers); `TENSORCODE_BACKEND=js` / `setBackend('js')` use only the
  JavaScript kernels, which remain the fallback for float64 and for runtimes
  without WebAssembly SIMD or workers. Products reduce in float32 (four lanes,
  like PyTorch's vectorized CPU kernels), and every output element is computed
  identically whatever the batch size, tile or thread count, so results are
  deterministic and batch invariant. Without gradients, the vision MLP
  (`fc1`, GELU, `fc2`) and self-attention blocks of Idefics3, CLIP and ViT run
  as one kernel call each (`feedForward`, `selfAttention` in
  `src/_internal/native/modules.ts`), keeping intermediates in kernel memory;
  their results are bit-identical to the separate calls. Activations, layer
  norm, permutes, arithmetic and the fused Adam(W) update are bit-identical to
  the JavaScript formulas. Without gradients, float32 attention is one fused kernel that never
  materializes the attention matrix (`enableGqa` shares key/value heads like
  PyTorch's `enable_gqa`). float16/bfloat16 results of products, convolution,
  softmax and layer norm are computed in float32 and rounded to the dtype, as
  PyTorch's CPU kernels do. Weights of small-row products (token-by-token
  decoding) stay resident in kernel memory (`TENSORCODE_WASM_CACHE_MB`, default
  1536; `clearWeightCache()`), keyed by storage and version.
- **Safetensors**: byte-identical serialization (Rust ordering, header padding,
  tied-tensor metadata of `safetensors.torch.save_model`), all common dtypes.
- **RNG**: `Generator` is PyTorch's CPU generator (`at::CPUGeneratorImpl`:
  mt19937, `torch.manual_seed` seeding, cached normal samples, the 5056-byte
  `torch.get_rng_state()` layout). `uniform_`, `normal_` (the 16-wide Box-Muller
  fill and the scalar path, float32/float64/float16/bfloat16), `bernoulli_`,
  `random_`/`randint`, `randperm`, `exponential_` and `multinomial` follow
  ATen's CPU kernels, with the C library functions they call (`logf`, `sinf`,
  `cosf`, `log`, `log1p`, `sin`, `cos`, `fma`) ported from glibc 2.39 AArch64
  (`nn/randomMath.ts`). `PythonRandom` (`nn/randomPython.ts`) is CPython's
  `random.Random`. `init` mirrors `torch.nn.init`, and native models reproduce
  transformers' `post_init` order (`native/hfInit.ts`), so `manualSeed(n)` plus
  the same construction gives bitwise-identical weights and dropout masks, and
  sampling makes the same draws (`test/nn/random.test.ts`,
  `test/nn/initParity.test.ts`).
  `withoutRandomInit` builds modules without drawing (`torch.device('meta')`),
  as `from_pretrained` does.

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
| `deberta-v2` | base, sequence classification | `debertaV2.ts` — NLI verifiers such as `cross-encoder/nli-deberta-v3-small` |
| `albert` | base, sequence classification | factorized embeddings, shared layers |
| `llama` | causal language model (Idefics3 text model) | grouped-query attention (`enableGqa`), every RoPE type, KV-cached decoding |
| `idefics3` | `Idefics3ForConditionalGeneration` (image-text-to-text) | SmolVLM; see Scene language mode below |

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
- `generateCausal(model, inputs, {generationConfig, settings, tokenizer, ...})`
  (`causalGeneration.ts`) is decoder-only `GenerationMixin.generate` for
  `CausalLanguageModel` adapters (the Idefics3 text model): configuration
  merging and validation, all logits processors, warpers and stopping criteria
  (`logitsProcessors.ts`), greedy/sampling/beam/assisted decoding, and
  transformers' errors for rejected settings.
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
`loadExperience`, checkpoints, sessions). Pure computation is synchronous.
Model providers are blocking like Python's (`complete`, run on a worker thread)
and asynchronous (`acomplete`); operations wrapping them work with `call` and
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

**Python numbers and dicts.** JavaScript numbers do not distinguish `1` from
`1.0`, and plain objects enumerate integer-like keys first. `src/_internal/json.ts`
keeps Python's view losslessly: its parsers (`parseJsonStrict`, `pythonJsonLoads`,
`rawToValue`) record the int/float kind of integral numbers (and exact big ints)
and the insertion order of keys beside the parsed containers, and `validatedJson`/
`deepCopy`/`mergeJson`/`shallowCopy`, fingerprints, `pythonJsonDumps`, raw
re-serialization and the experience codec preserve and honour both. Programs mark
numbers with `float()`/`int()` (exported from `tensorcode`) and ordered dicts with
`Map` or `orderedObject`; decoded Python dicts with non-string keys are `Map`s.
Integral numbers without a kind follow the schema: configuration keys in
`PYTHON_FLOAT_KEYS` (the generated transformers defaults carry their own float
kinds) and record classes' `recordFloatFields`/`recordIntKeyFields`. Caller data
that Python writes as given has no float schema: containers marked with
`markPlainData` (`Retrieve` items) and the JSON memory, trajectory and
structured-target writers. Code that
copies JSON data uses these helpers (or `transferPythonNumberKind`/
`orderedEntries`) rather than spreads or `Object.entries` rebuilds, which fall
back to the schema default. If a module persists a new float-typed field whose
value can be integral, add it to the schema statically — never call
`registerPythonFloatKeys` at import time (fingerprints must not depend on import
order).

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

## Implementation notes

- **Weights.** `loadNativeFoundation` resolves weights like transformers'
  `_get_resolved_checkpoint_files`: safetensors (single or sharded) first;
  otherwise `pytorch_model.bin` (single or sharded) through a weights-only
  unpickler (`native/torchCheckpoint.ts`: both `torch.save` formats; only
  containers, primitives and tensor rebuilders; any other global raises, as
  `torch.load(weights_only=True)` does). Call sites where Python passes
  `use_safetensors=True` (`useSafetensors: true`) instead load a Hub
  repository's open `SFconvertbot` conversion pull request based on `main`,
  asking the conversion Space to open one when there is none (transformers'
  `spawn_conversion`); otherwise transformers' background conversion is started
  after loading the PyTorch weights. Offline mode, a pinned revision and
  `DISABLE_SAFETENSORS_CONVERSION` skip the conversion. transformers' `legacy`
  key renames (`LayerNorm.gamma`/`beta`) and the ViT renames apply to every
  checkpoint.
- **Latent diffusion.** `ops.vec.ImageDecoder` ports diffusers 0.40
  `UNet2DConditionModel`, `AutoencoderKL` and `DDIMScheduler` (float32
  schedules bit-identical to PyTorch) in `src/_internal/native/diffusers.ts`:
  every block family those models can run (plain, ResNet-resampling,
  attention, cross-attention, simple added-KV cross-attention and K-diffusion
  UNet blocks; all three UNet mid blocks; attention and plain VAE blocks),
  positional and Gaussian Fourier time embeddings, `default`/`scale_shift`
  conditioning and `AdaGroupNorm`/`SpatialNorm` conditional norms, and the
  `silu`/`swish`/`mish`/`gelu`/`relu` activations, with diffusers' module
  trees and parameter names. Blocks diffusers constructs but cannot run inside
  these models (skip blocks, encoder blocks in a UNet, UNet blocks in a VAE)
  raise `ValueError`. `context.seed` draws noise from `new Generator(seed)`
  exactly like `torch.Generator().manual_seed(seed)`. `fromFoundation`
  downloads only the files diffusers loads (component configurations and
  non-variant safetensors).
- **Scene language mode.** `Scene.fromLanguageFoundation`/`interpret` port
  `SceneLanguage` over an owned `Idefics3ForConditionalGeneration`
  (`src/_internal/native/idefics3.ts`: SigLIP-style vision tower with
  fractional patch positions, pixel-shuffle connector, Llama text model with
  grouped-query attention and every transformers RoPE type — `default`,
  `linear`, `dynamic`, `yarn`, `longrope`, `llama3`, `proportional`) and the
  `Idefics3Processor` (`idefics3Processing.ts`: longest-edge LANCZOS resizing,
  image splitting, fused normalization, batches of prompts with any number of
  images, `<image>` prompt expansion, the image tokens it adds to tokenizers
  that lack them, Jinja chat templates; the `Fast` and `Pil` class names
  resolve to the default processor, as in transformers). Generation is
  transformers 5.17 `generate` (`causalGeneration.ts`, `logitsProcessors.ts`):
  greedy, sampling, beam search and beam sampling, classifier-free guidance,
  prompt lookup, chunked prefill, token healing, every logits processor and
  warper, watermarking with the shared PyTorch-compatible `Generator`, and the
  stopping criteria including `StopStringCriteria`; `interpret` passes
  `do_sample=False` as Python does. `fromLanguageFoundation` reproduces the
  processor assets `Idefics3Processor.save_pretrained` writes (so
  `processor_hashes` equal Python's).
- **Images.** `src/_internal/image/` decodes PNG, JPEG (Huffman and arithmetic,
  sequential/progressive/lossless, libjpeg-turbo's ISLOW IDCT, fancy upsampling
  and colour tables), GIF, WebP (VP8, VP8L, ALPH, ANMF) and BMP in pure
  TypeScript (inflate via `node:zlib` when present, a bundled inflater
  otherwise). `torchvision.ts` reproduces `decode_image` read modes and EXIF
  orientation; `raster.ts` and `resample.ts` reproduce `PIL.Image.open`,
  `convert`, `resize` and `exif_transpose`. The float filters use the same
  glibc ports as the samplers (`nn/randomMath.ts`), so they match bit for bit.
  `vec/imageProcessing.ts` is the `ViTImageProcessor` over all of them, with
  center-crop padding, `do_pad` and PyTorch's `NotImplementedError` for
  BOX/HAMMING. 16-bit images decode to `uint16` tensors.
- **Blocking providers.** HTTP (`OpenAICompatibleModel`, `JevModel`) and local
  providers have Python's blocking `complete` (and `completeQuestions` /
  `completeBatch`), so `op.call` and `ask` work with them: the request runs on
  a worker thread while the caller blocks on `Atomics.wait`
  (`src/integrations/blocking.ts`). `LocalModel.complete` runs the same
  generation in a worker thread with a model loaded there (`fromPretrained`
  arguments, or an explicit `worker: { module, exportName }` loader for
  supplied models). A redirect raises `ProviderHTTPError` with its 3xx
  `.status`, as in Python.
- **Plan actions** receive `(state, args)`. Plan validation binds step
  arguments like `inspect.signature(action).bind(None, **arguments)`, with
  Python's `TypeError` messages, against the properties the action destructures
  from `args` (or an explicit `withSignature`/`signatures` declaration). Error
  observations and policy errors record Python exception names (`Error` →
  `RuntimeError`, `RangeError` → `ValueError`, system errors by `code`;
  `src/_internal/pythonErrors.ts`).
- **Tool configurations.** Tools reject unknown fields with Python 0.4.0a4's
  message (`rejectUnknownToolFields` in `_internal/pretrained.ts`; field lists
  as static `configFields`, `cognitionFields`, `rankingFields`,
  `languageFields`).
- **Tokenizer JSON.** Tokenizer configurations embed the canonical backend JSON
  (sorted keys), as Python does. Tools built from foundations persist
  `backend_tokenizer.to_str()` in `tokenizer_json`, `verifier_tokenizer_json`
  and Scene's `tokenizer.json`/`tokenizer_sha256`; TypeScript writes the same
  bytes (`rustTokenizerString`: Rust struct field order, vocabularies in id
  order, `serde_json` float formatting, unescaped non-ASCII text). Wherever
  Python builds the backend with Rust `Tokenizer.from_str`/`from_file`,
  TypeScript reproduces the Rust `serde_json` float parsing, which can move
  Unigram scores by one ULP (`rustJsonF64`). For `BertTokenizer`,
  `RobertaTokenizer`, `CLIPTokenizer`, `T5Tokenizer`, `DebertaV2Tokenizer`,
  `AlbertTokenizer`, `GPT2Tokenizer` and `LlamaTokenizer` the pipeline that
  transformers 5 rebuilds from the vocabulary and `tokenizer_config.json`
  flags is reproduced as well (`src/_internal/tokenizers/serialization.ts`),
  followed by `TokenizersBackend.__init__`'s registration of
  `added_tokens_decoder`, special and extra special tokens and the class's
  post-processor (`FastTokenizer.fromFiles`), with `_from_pretrained`'s
  precedence between `tokenizer_config.json` and `special_tokens_map.json`.
- **Threads.** Python's `RLock`s become single-threaded execution plus a promise
  queue that serializes session persistence.

## Remaining differences

[Parity with Python](docs/parity.md#remaining-differences) lists them for
users. In short:

- CPU only (no CUDA/GPU); computed results agree with PyTorch within float
  tolerance; bit-exact random streams and resampling follow the AArch64
  reference platform, as PyTorch's own do.
- Native architectures are the listed ones. SentencePiece-only tokenizers
  (no `tokenizer.json`) are not converted (Python needs optional packages).
- Scene language processor assets cover TokenizersBackend, Llama, GPT2, T5,
  ALBERT, DeBERTa-v2, BERT, RoBERTa and CLIP tokenizer classes; other classes
  with their own construction and other image processor classes raise
  `NotImplementedError`.
- `openImage` reads PNG, JPEG, GIF, WebP and BMP only.
- Chat templates: recursive loops, `call` blocks and a few rarely used Jinja
  filters raise `TemplateError`.
- Directory checkpoints that Python saved on a GPU machine carry CUDA generator
  states (PyTorch initializes CUDA while training) and are rejected.
- Blocking provider calls need a thread that can block; `LocalModel` runs
  Transformers.js, not PyTorch.
- Only `SFconvertbot` conversion pull requests load on public repositories.
- Saved callbacks need an explicit `configuration()`; unmarked whole numbers
  in user code follow the field's Python type.
- The CUDA-host examples and the Stable Diffusion path of
  `pretrained_latent_lifecycle.py` are not ported.

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
- Python number kinds, dict order and dict key types survive loading and saving:
  a Python artifact with an int in a float field (`hidden_dropout_prob=0`) or an
  integral float under another key re-saves and fingerprints identically, and
  experience and JSON memory files with integer-like keys, `int`/`None`/tuple
  dict keys and integral floats round-trip byte for byte
  (`test/internal/jsonValues.test.ts`).
- Experience files, standalone `tensorcode.checkpoint` files and directory
  checkpoints are interchangeable. Directory checkpoints store `python_rng`
  (CPython `random.getstate()`), `torch_rng` (`torch.get_rng_state()`) and
  `cuda_rng` in Python's layout, and loading restores both generators in
  either language. Like a CPU-only Python process, TypeScript rejects a
  checkpoint that carries CUDA generator states.
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
