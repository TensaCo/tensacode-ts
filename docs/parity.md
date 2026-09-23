# Parity with Python

The TypeScript package is a port of the Python `tensorcode` package (version
`0.4.0a4`). It has the same operations, tools, tracing, training and artifact
formats. The Python package is the reference: where the two disagree on
something the [remaining differences](#remaining-differences) do not list,
Python's behavior is the correct one and the difference is a bug.

This page says how parity is checked, what matches, what you can move between
the two languages and the few places where they still differ.

## How parity is checked

- **Fixtures from the Python stack.** Generators in `scripts/fixtures` run the
  Python package (with PyTorch 2.14, transformers 5.17, diffusers 0.40,
  torchvision 0.29 and Pillow 12) and write reference outputs to
  `test/fixtures`. The TypeScript tests compare against them: layer outputs
  and gradients, optimizer steps, random streams and freshly initialized
  weights, safetensors and PyTorch checkpoint bytes, tokenizer output, decoded
  images, native model outputs, generation, tool receipts and error messages,
  training losses, experience files, checkpoints and session files.
- **Ported tests.** The suite in `test/` ports the Python tests area by area.
  It runs with `npm test` and needs neither Python nor network access.
- **Gradient checks.** Every differentiable operation in the numerical core is
  checked against finite differences.
- **Real checkpoints.** Tests that need a cached Hugging Face checkpoint (for
  example `google/flan-t5-small`, `google/electra-small-discriminator`,
  `openai/clip-vit-base-patch32`, `cross-encoder/nli-deberta-v3-small` or
  `HuggingFaceTB/SmolVLM-256M-Instruct`) run when it is in the local cache and
  skip otherwise.
- **Examples.** Every ported example was run beside its Python original with
  the same inputs and seeds; see [examples](../examples/README.md).

## By area

| Area | Status | Notes |
|---|---|---|
| Tracing, supervision, replay, release | Match | Async capture uses `AsyncLocalStorage` |
| Operation contract (`call`, `acall`, `forward`) | Match | `op.call(value, { context })` is Python's `op(value, context=...)` |
| Vector operations (`ops.vec`) | Match | Linear, MLP and native transformer variants; T5 text and ViT image encoders; image inputs decode like torchvision and Pillow |
| `ImageDecoder` (latent diffusion) | Match | Every diffusers `UNet2DConditionModel` and `AutoencoderKL` block these models can run, and DDIM. `context.seed` draws the same noise as `torch.Generator().manual_seed(seed)` |
| Text operations (`ops.text`) | Match | Owned T5 models (generation and likelihood decoding) and external providers |
| Graph operations (`ops.graph`) | Symbolic stubs | Same as Python: every graph operation raises `NotImplementedError` |
| `Chatbot`, `Investigator`, `Decision`, `Planner` | Match | Including cognitive sessions, episodic memory, verifiers, plan execution and configuration-field validation |
| `Scene`, ranking mode | Match | CLIP bootstrap through `Scene.fromFoundation` |
| `Scene`, language mode | Match | `Scene.fromLanguageFoundation` and `interpret` over Idefics3 (SmolVLM) with transformers' `generate`. A SmolVLM-256M interpretation takes seconds on a multi-core CPU |
| Training, experience files, checkpoints | Match | Losses agree to about 1e-7 relative on the tool fixtures. Checkpoints restore PyTorch's and CPython's random states in both directions |
| Random numbers and initialization | Match (bitwise) | `manualSeed(n)` is `torch.manual_seed(n)`: fresh tools, operations and native models, dropout masks and sampled tokens equal Python's. See [random numbers](#random-numbers) |
| Tool configurations, validation, fingerprints | Match | Unknown fields raise Python's message. Operation fingerprints equal Python's for the same configuration, so experience files are portable |
| Foundation-built tools | Match | Configurations, including embedded tokenizer JSON, equal Python's (Investigator from ELECTRA, Flan-T5 and DeBERTa-v3; Chatbot; Scene from CLIP and SmolVLM) |
| Foundation weights | Match | safetensors (single or sharded), `pytorch_model.bin` through a weights-only unpickler, and the Hub's safetensors conversion pull request, each where transformers uses it |
| Real foundations | Match within float tolerance | DeBERTa-v3 NLI logits within 7e-6; hosted decision and planner artifacts within 4e-5 relative |
| Tokenizers | Match | `tokenizer.json` runtime with WordPiece, BPE, Unigram and WordLevel models, plus the T5, DeBERTa-v2, ALBERT, GPT2 and Llama rebuilds that transformers 5 performs |
| Native architectures | Listed models | ALBERT, BERT, RoBERTa, Electra, DistilBERT, DeBERTa-v2, T5, ViT, CLIP, Llama and Idefics3, with transformers 5.17 parameter names. See [native architectures](#native-architectures) |
| Integrations | Match | OpenAI-compatible, Jev, and a Transformers.js `LocalModel`, blocking (`complete`) and asynchronous (`acomplete`). No implicit retries or redirects |
| Plans and actions | Match | Step arguments bind like `inspect.signature(action).bind(...)`; error observations record Python exception names |
| Compute | CPU | WebAssembly SIMD kernels on worker threads; see [compute](#compute) |

## What moves between the languages

These files are interchangeable. A file written by one package loads in the
other:

- **Model artifacts** (`tensorcode_config.json`, `model.safetensors` and the
  model card), saved with `savePretrained` / `save_pretrained`, locally or on
  the Hugging Face Hub. A Python artifact loaded in TypeScript and saved again
  has a byte-identical manifest. Weights are byte-identical too, except for
  models with two or more tied parameter aliases (T5): Python writes that
  metadata in a nondeterministic order, so even two Python saves differ there.
  The tensor bytes are identical.
- **Configuration-only operation artifacts** (`tensorcode.operation`).
- **Experience files** written by `trace.save(...)` in either language.
- **Standalone checkpoint files** (`tensorcode.checkpoint`) and **directory
  checkpoints** (`training.json` plus a tensor file), including their PyTorch
  and CPython random states.
- **Tool session files**: chat, ranking, cognitive session and state,
  trajectories and JSON memory. Python session files load in TypeScript and
  save back byte-identically.
- **The Hugging Face cache.** Both packages use the `huggingface_hub` cache
  layout, so a model downloaded by one is found by the other.

## Numbers and dictionaries

JavaScript numbers do not distinguish `1` from `1.0`, and JavaScript objects
list integer-like keys first. TypeScript keeps Python's view anyway:

- Values read from Python files (artifacts, experience, checkpoints, sessions,
  JSON memory) remember whether each whole number was an `int` or a `float` and
  the order of dictionary keys, so they save and fingerprint exactly as Python
  wrote them. A Python caller who passed an int for a float field
  (`hidden_dropout_prob=0`) gets the same fingerprint in both languages.
- In your own code, `float(0)` and `int(0)` from `tensorcode` say which one you
  mean (`float(0)` is written `0.0`). Unmarked whole numbers follow the field's
  Python type: configuration fields that Python stores as floats are written as
  floats, everything else as ints.
- Use `orderedObject([['2', ...], ['1', ...]])` from `tensorcode` (or a `Map`
  where an option expects a mapping, such as `Retrieve` items) for a dictionary
  whose integer-like keys must keep their order. Python dictionaries with `int`,
  `bool`, `None` or tuple keys decode to a `Map` that keeps those key types.

## Random numbers

The random generator is PyTorch's CPU generator, reproduced exactly:
`manualSeed(n)` is `torch.manual_seed(n)`, `getRngState()`/`setRngState()`
exchange the same 5056 bytes as `torch.get_rng_state()`/`torch.set_rng_state()`,
and `rand`, `randn`, `normal_`, `uniform_`, `randint`, `randperm`, `bernoulli`,
`exponential_` and `multinomial` produce the same bits. `init` mirrors
`torch.nn.init`, and native models initialize in transformers' `post_init`
order, so seeded fresh construction of every tool, operation and native model
gives Python's weights bit for bit; loading a foundation draws only for weights
the checkpoint lacks, as `from_pretrained` does. Dropout masks match bit for
bit. Generation sampling (`do_sample`, nucleus and beam sampling) makes the
same draws as `torch.multinomial`, so sampled tokens match Python's whenever the
float32 token probabilities agree, which they do on the fixtures.
`PythonRandom` reproduces CPython's `random.Random`.

## Naming

The API follows TypeScript conventions. Saved and reported data keeps Python's
spelling, which is what makes the files above interchangeable.

| Python | TypeScript |
|---|---|
| `op(value, context=ctx)` | `op.call(value, { context: ctx })` |
| `await op.acall(value, context=ctx)` | `await op.acall(value, { context: ctx })` |
| `model.save_pretrained(path)` | `await model.savePretrained(path)` |
| `Investigator.from_pretrained(repo, revision=rev)` | `await Investigator.fromPretrained(repo, { revision: rev })` |
| `training.Trainer.from_tool(model, optimizer=opt)` | `Trainer.fromTool(model, { optimizer: (params) => opt })` |
| `trainer.capture(inputs, target, source="review:1")` | `trainer.capture(inputs, target, { source: 'review:1' })` |
| `training.load_experience(path, operations=ops)` | `await loadExperience(path, { operations: ops })` |
| `result["selected_id"]` | `result.selected_id` (saved and reported fields stay snake_case) |
| `torch.manual_seed(0)` | `manualSeed(0)` from `tensorcode/nn` |
| `Planner.config_fields` | `Planner.configFields` |

Keyword arguments become one trailing options object. Anything that reads or
writes files or uses the network returns a `Promise`; pure computation is
synchronous, and so are the blocking provider calls (`complete`, `op.call`,
`ask`), as in Python.

## Remaining differences

Everything not listed here behaves as in Python.

### Compute

TypeScript ships its own dependency-free tensor and autograd core
(`tensorcode/nn`) instead of PyTorch, and runs on the CPU only: `device`
options accept `'cpu'`, and there is no CUDA or other GPU backend, so a
checkpoint that carries CUDA generator states is rejected, as a CPU-only
Python process rejects it. Float32 products, convolutions, attention, softmax,
layer norm and activations run on WebAssembly SIMD kernels on worker threads
(`setNumThreads`, `TENSORCODE_THREADS`); browsers run them on one thread.
Products accumulate in float32 like PyTorch's CPU kernels, so computed results
agree with Python within float tolerance rather than bit for bit, and they are
deterministic and independent of batch size and thread count. float16 results
are computed in float32 and rounded once, where PyTorch's float16 CPU
convolution accumulates in float16, so they agree to a few units in the last
place.

### Reference platform

Bit-exact random streams and image resampling follow PyTorch, glibc 2.39 and
Pillow on AArch64, where the fixtures are generated. PyTorch itself is not
reproducible across platforms: on x86-64 with AVX2 its float32 `normal_` uses
a different vectorized kernel and glibc picks different math routines, so
Python's own samples differ there. Two branches the samplers never reach in
practice are not bit-exact: `trunc_normal_` with less than 30% of the mass
inside its bounds (PyTorch computes that log with a vectorized kernel), and
`sin`/`cos` of arguments beyond 105414350 in magnitude.

### Native architectures

Python loads any transformers `AutoModel` for a text foundation. TypeScript
implements ALBERT, BERT, RoBERTa, Electra, DistilBERT, DeBERTa-v2, T5, ViT,
CLIP, Llama and Idefics3 natively and raises `ValueError` for other
`model_type`s. Foundations need a `tokenizer.json`: a repository with only
slow tokenizer files (`vocab.txt`, `spiece.model`, `vocab.json`/`merges.txt`),
or none, loads in Python but not in TypeScript.

### Scene language mode

`fromLanguageFoundation` writes the processor assets Python writes for
`TokenizersBackend` (including `PreTrainedTokenizerFast` and unknown class
names), `LlamaTokenizer`, `GPT2Tokenizer`, `T5Tokenizer`, `AlbertTokenizer` and
`DebertaV2Tokenizer`, with or without `Fast`. Idefics3 foundations with other
tokenizer classes raise `NotImplementedError`. Image processors other than
`Idefics3ImageProcessor` (also named `...Fast` or `...Pil`) raise
`NotImplementedError` as well.

### Images

`openImage` reads PNG, JPEG, GIF, WebP and BMP; other formats Pillow opens
(TIFF, ICO, PPM and so on) raise `ValueError('cannot identify image file')`.
Rare Pillow modes (`RGBX`, `YCbCr`, `LAB`, `HSV`, `I;16B`) are not supported,
and truncated progressive JPEGs are decoded without libjpeg-turbo's block
smoothing. Where torchvision itself returns uninitialized memory (palettes
below 8 bits in `UNCHANGED` mode, palette PNGs without transparency in the
alpha modes, GIF frames beyond the canvas), the pixels cannot match. Some
decoder error messages are worded differently from the C libraries; the error
types match. The synchronous `preprocess` and `loadImage` cannot fetch
`http(s)` URLs; use `apreprocess` or `loadImageAsync`.

### Providers

`complete`, `op.call` and `ask` block on a worker thread. Runtimes that cannot
block a thread (browser main threads, edge runtimes) raise
`SynchronousCallUnavailable` (a `NotImplementedError`) there, and a provider
built with an injected `fetch` function is asynchronous only, because a
function cannot move to another thread; `acall` and `aask` always work.
`LocalModel` runs Transformers.js (ONNX Runtime) rather than PyTorch, so its
answers can differ from Python's where two tokens score almost equally, and
its blocking `complete` loads a second copy of the model in the worker thread.

### Hub conversion pull requests

When a public repository has an open conversion pull request from an author
other than `SFconvertbot`, transformers asks the conversion Space for a new one
and may then load the first matching pull request whoever opened it.
TypeScript only loads pull requests opened by `SFconvertbot` on public
repositories, so it never loads weights a third party proposed.

### JavaScript values

- JavaScript cannot tell a closure from a module function, so a saved callback
  (for example `combine`) always needs an explicit `configuration()`.
- Unmarked whole numbers in your own code follow the field's Python type (see
  [numbers and dictionaries](#numbers-and-dictionaries)). Optimizer options do
  not take `float()`/`int()` markers, so a whole-number learning rate is saved
  as an int.
- Decoded dictionaries with tuple keys are `Map`s whose tuple keys compare by
  identity.

### Not ported

The Python examples that require the CUDA training host
(`compare_cognitive_verifiers.py`, `train_hypotheses.py`,
`train_realization.py`, `train_verifier.py`, `prepare_response_quality.py`,
`train_response_quality.py`) and the optional Stable Diffusion image path of
`pretrained_latent_lifecycle.py`. Every other Python example has a TypeScript
port in [examples](../examples/README.md) that reproduces its results.

The design notes in [DESIGN.md](../DESIGN.md) map every Python module to its
TypeScript file.
