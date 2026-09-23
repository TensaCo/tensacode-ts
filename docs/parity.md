# Parity with Python

The TypeScript package is a port of the Python `tensorcode` package (version
`0.4.0a3`). It has the same operations, tools, tracing, training and artifact
formats. The Python package is the reference: where the two disagree on
something this page does not list as a deliberate difference, Python's behavior
is the correct one and the difference is a bug.

This page says what matches, what you can move between the two languages, and
where they differ on purpose.

## How parity is checked

- **Fixtures from the Python stack.** Generators in `scripts/fixtures` run the
  Python package (with PyTorch and transformers 5.17) and write small reference
  outputs to `test/fixtures`. The TypeScript tests compare against them: layer
  outputs and gradients, optimizer steps, safetensors bytes, tokenizer output,
  native model outputs, tool receipts, training losses, experience files,
  checkpoints and session files.
- **Ported tests.** The suite in `test/` ports the Python tests area by area.
  It runs with `npm test` and needs neither Python nor network access.
- **Gradient checks.** Every differentiable operation in the numerical core is
  checked against finite differences.
- **Real checkpoints.** Tests that need a cached Hugging Face checkpoint (for
  example `google/flan-t5-small`, `google/electra-small-discriminator`,
  `openai/clip-vit-base-patch32` or `cross-encoder/nli-deberta-v3-small`) run
  when it is in the local cache and skip otherwise.

## By area

| Area | Status | Notes |
|---|---|---|
| Tracing, supervision, replay, release | Match | Async capture uses `AsyncLocalStorage` |
| Operation contract (`call`, `acall`, `forward`) | Match | `op.call(value, { context })` is Python's `op(value, context=...)` |
| Vector operations (`ops.vec`) | Match | Linear, MLP and native transformer variants; T5 text and ViT image encoders |
| `ImageDecoder` (latent diffusion) | Match | Every diffusers UNet2DConditionModel and AutoencoderKL block family it can run, and DDIM. `context.seed` draws the same noise as `torch.Generator().manual_seed(seed)`; `context.noise` supplies it explicitly |
| Text operations (`ops.text`) | Match | Owned T5 models (generation and likelihood decoding) and external providers. HTTP providers are async only |
| Graph operations (`ops.graph`) | Symbolic stubs | Same as Python: every graph operation raises `NotImplementedError` |
| `Chatbot`, `Investigator`, `Decision`, `Planner` | Match | Including cognitive sessions, episodic memory, verifiers and plan execution |
| `Scene`, ranking mode | Match | CLIP bootstrap through `Scene.fromFoundation` |
| `Scene`, language mode | Match | `Scene.fromLanguageFoundation` and `interpret` over Idefics3 (SmolVLM) with transformers' `generate`: beam search, guidance, sequence bias, prompt lookup, watermarking and every logits processor the generation configuration selects. A SmolVLM-256M interpretation (13 vision tiles) takes seconds on a multi-core CPU |
| Training, experience files, checkpoints | Match | Losses agree to about 1e-7 relative on the tool fixtures. Directory checkpoints restore PyTorch's and CPython's random states in both directions |
| Random numbers and initialization | Match (bitwise) | `manualSeed(n)` reproduces `torch.manual_seed(n)`: freshly constructed tools, operations and native models and dropout masks are bit-identical to Python's, and sampling makes the same draws. See [random numbers](#random-numbers) |
| Tool configurations, validation, fingerprints | Match | Operation fingerprints equal Python's for the same configuration, so experience files are portable |
| Foundation-built tools | Match | Configurations, including embedded tokenizer JSON, equal Python's (Investigator from ELECTRA, Flan-T5 and DeBERTa-v3; Chatbot; Scene from CLIP) |
| Real foundations | Match within float tolerance | DeBERTa-v3 NLI logits within 7e-6; hosted decision and planner artifacts within 4e-5 relative |
| Tokenizers | Match | `tokenizer.json` runtime with WordPiece, BPE, Unigram and WordLevel models, plus the T5, DeBERTa-v2, ALBERT, GPT2 and Llama rebuilds that transformers 5 performs |
| Native architectures | Listed models only | ALBERT, BERT, RoBERTa, Electra, DistilBERT, DeBERTa-v2, T5, ViT, CLIP, Llama and Idefics3, with transformers 5.17 parameter names. Python loads any transformers `AutoModel` for text foundations |
| Integrations | Match | OpenAI-compatible, Jev, and a Transformers.js `LocalModel`, blocking (`complete`) and asynchronous (`acomplete`). No implicit retries or redirects |
| Compute | CPU only | WebAssembly SIMD kernels on worker threads for float32 products, attention, softmax, layer norm and activations (`setNumThreads`, `TENSORCODE_THREADS`), JavaScript for the rest. No GPU and no mixed-precision kernels |

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

Keyword arguments become one trailing options object. Anything that reads or
writes files or uses the network returns a `Promise`; pure computation is
synchronous.

## Deliberate differences

### Numerics and random numbers

TypeScript ships its own dependency-free tensor and autograd core (`tensorcode/nn`)
instead of PyTorch. It runs on the CPU in float32 (float16 and bfloat16 are
computed in float32) and float64. Computed results agree with Python within
float tolerance.

### Random numbers

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

The reference is PyTorch on AArch64 with glibc 2.39, where the test fixtures
are generated. PyTorch itself is not reproducible across platforms: on x86-64
with AVX2 its float32 `normal_` uses a different vectorized kernel, so Python's
samples there differ from both.

### Weights

TypeScript loads safetensors weights only, never `pytorch_model.bin` or other
pickle files. When a Hub repository's `main` branch has only PyTorch weights,
TypeScript loads `model.safetensors` from the repository's open automated
conversion pull request, as transformers does, and asks the safetensors
conversion Space to open one when there is none. Offline mode, a pinned
revision or the `DISABLE_SAFETENSORS_CONVERSION` environment variable turn the
lookup off. Python can fall back to the `.bin` weights; TypeScript needs the
converted file.

### Scene language mode

`interpret` runs transformers 5.17 `generate` with `do_sample=False`, as Python
does: greedy or beam search (`num_beams`, `length_penalty`, `early_stopping`,
`num_return_sequences`), classifier-free guidance, `sequence_bias`, prompt-lookup
decoding, watermarking and every other logits processor, with the errors Python
raises for settings it rejects (for example `stop_strings` and `token_healing`,
which need a tokenizer that `interpret` does not pass). The owned Idefics3 model's
`generate` also samples (all transformers warpers) from the PyTorch-compatible
generator, and applies `stop_strings` and token healing when given a tokenizer.
The processor batches prompts with any number of images each and adds its image
tokens to tokenizers that lack them. Processor assets are written for
`TokenizersBackend` (`PreTrainedTokenizerFast` and unknown class names),
`LlamaTokenizer`, `GPT2Tokenizer`, `T5Tokenizer`, `AlbertTokenizer` and
`DebertaV2Tokenizer`, with or without `Fast`; other tokenizer classes raise
`NotImplementedError`. Deprecated Hub generation modes (contrastive and
constrained search, group beam search, DoLa) raise the error Python raises
without `trust_remote_code`.

### Images

Image files decode in pure TypeScript with the pixels Python gets: PNG (every
colour type, bit depth and interlacing), JPEG (baseline, extended, progressive,
arithmetic-coded and lossless; any sampling factors; restart markers; CMYK),
GIF, WebP (lossy, lossless, alpha, animation) and BMP. `decodeImage`
reproduces `torchvision.io.decode_image` (what `ImageEncoder.preprocess` uses
for file paths, base64 text and data URIs), `loadImage` transformers'
`load_image_as_tensor`, and `openImage` `PIL.Image.open`, returning a
`RasterImage` with Pillow's modes, `convert`, `resize` (NEAREST, BOX,
BILINEAR, HAMMING, BICUBIC, LANCZOS) and `exifTranspose`. `preprocess` accepts
tensors (any dtype, `CHW` or `HWC`), `RasterImage`s, strings and nested lists,
and rejects bytes as Python does; use `apreprocess` for `http(s)://` URLs, which
TypeScript fetches asynchronously. 16-bit PNGs decode to `int32` tensors because
TypeScript has no `uint16` dtype; the processor still applies torchvision's
`uint16` rules. Float resizing reproduces PyTorch's aarch64 build (fused
multiply-add remainders, glibc `sin`), and Pillow's `I`/`F` modes its aarch64
FMA contraction; other CPUs can differ from Python in the last bit there.

### Providers and plans

HTTP providers (`OpenAICompatibleModel`, `JevModel`) and `LocalModel` block in
`complete` like Python's (a worker thread does the work), so `op.call` and
`ask` work with them; `acall` and `aask` are the asynchronous forms. Where a
thread cannot block on I/O (browsers, edge runtimes) the blocking methods raise
`NotImplementedError`. A provider constructed with an injected `fetch` function
is asynchronous only, and `LocalModel.complete` loads its own copy of the model
in the worker (from `fromPretrained` arguments or an explicit loader module). A
redirect raises `ProviderHTTPError` with its 3xx status, as in Python.

Plan actions receive `(state, args)`. Validation binds each step's arguments
like Python's `inspect.signature(action).bind(None, **arguments)`, with the
same `TypeError` messages, against the properties the action destructures from
`args` (`(state, { amount, note = 'x' })`), or against an explicit signature
(`withSignature`). Error observations record Python exception names:
`RuntimeError` for a JavaScript `Error`, `ValueError` for `RangeError`, class
names for your own error classes.

### JavaScript values

- JavaScript cannot tell a closure from a module function, so a saved callback
  (for example `combine`) always needs an explicit `configuration()`.

### Not ported

The Python examples that require the CUDA training host
(`compare_cognitive_verifiers.py`, `train_hypotheses.py`,
`train_realization.py`, `train_verifier.py`, `prepare_response_quality.py`,
`train_response_quality.py`) and the optional ViT/Stable Diffusion image path of
`pretrained_latent_lifecycle.py`. Every other Python example has a TypeScript
port in [examples](../examples/README.md) that reproduces its results.

The design notes in [DESIGN.md](../DESIGN.md) map every Python module to its
TypeScript file.
