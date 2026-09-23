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
| `ImageDecoder` (latent diffusion) | Match with supplied noise | diffusers cross-attention UNet, AutoencoderKL and DDIM. Pass `context.noise` for samples identical to Python's; `context.seed` draws from the TensorCode generator, so seeded samples differ |
| Text operations (`ops.text`) | Match | Owned T5 models (generation and likelihood decoding) and external providers. HTTP providers are async only |
| Graph operations (`ops.graph`) | Symbolic stubs | Same as Python: every graph operation raises `NotImplementedError` |
| `Chatbot`, `Investigator`, `Decision`, `Planner` | Match | Including cognitive sessions, episodic memory, verifiers and plan execution |
| `Scene`, ranking mode | Match | CLIP bootstrap through `Scene.fromFoundation` |
| `Scene`, language mode | Match | `Scene.fromLanguageFoundation` and `interpret` over Idefics3 (SmolVLM) with greedy decoding. Pure JavaScript compute makes a full SmolVLM interpretation take minutes on a CPU |
| Training, experience files, checkpoints | Match | Losses agree to about 1e-7 relative on the tool fixtures. See [checkpoints](#checkpoints-and-random-numbers) for the RNG difference |
| Tool configurations, validation, fingerprints | Match | Operation fingerprints equal Python's for the same configuration, so experience files are portable |
| Foundation-built tools | Match | Configurations, including embedded tokenizer JSON, equal Python's (Investigator from ELECTRA, Flan-T5 and DeBERTa-v3; Chatbot; Scene from CLIP) |
| Real foundations | Match within float tolerance | DeBERTa-v3 NLI logits within 7e-6; hosted decision and planner artifacts within 4e-5 relative |
| Tokenizers | Match | `tokenizer.json` runtime with WordPiece, BPE, Unigram and WordLevel models, plus the T5, DeBERTa-v2, ALBERT and GPT2 rebuilds that transformers 5 performs |
| Native architectures | Listed models only | ALBERT, BERT, RoBERTa, Electra, DistilBERT, DeBERTa-v2, T5, ViT, CLIP, Llama and Idefics3, with transformers 5.17 parameter names. Python loads any transformers `AutoModel` for text foundations |
| Integrations | Match | OpenAI-compatible, Jev, and a Transformers.js `LocalModel`. No implicit retries or redirects |
| Compute | CPU only | Pure JavaScript, no GPU and no mixed-precision kernels |

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
- **Standalone checkpoint files** (`tensorcode.checkpoint`).
- **Tool session files**: chat, ranking, cognitive session and state,
  trajectories and JSON memory. Python session files load in TypeScript and
  save back byte-identically.
- **The Hugging Face cache.** Both packages use the `huggingface_hub` cache
  layout, so a model downloaded by one is found by the other.

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
computed in float32) and float64. Its random generator is deterministic and
seedable, but it does not reproduce PyTorch's random sequences. Freshly
initialized models, dropout masks and sampling therefore differ from Python.
Loaded weights are identical, so a model loaded from an artifact gives the same
results in both languages, within float tolerance.

### Checkpoints and random numbers

Python directory checkpoints load in TypeScript. Their PyTorch and CPython
random states are validated and then ignored, because TypeScript cannot restore
them. TypeScript directory checkpoints store the TensorCode generator state and
do not load in Python. Standalone checkpoint files work in both directions.

### Weights

TypeScript loads safetensors weights only, never `pytorch_model.bin` or other
pickle files. When a Hub repository's `main` branch has only PyTorch weights,
TypeScript loads `model.safetensors` from the repository's open automated
conversion pull request, as transformers does. Unlike Python, it never asks the
Hub to create that conversion. Offline mode, a pinned revision or the
`DISABLE_SAFETENSORS_CONVERSION` environment variable turn the lookup off.

### Scene language mode

Generation is greedy, with transformers' logits processors (repetition penalty,
n-gram blocking, bad words, minimum lengths, forced BOS and EOS, suppression).
Beam search, guidance, sequence bias and stop strings raise
`NotImplementedError`. One image per prompt. Processor assets are written only
for `GPT2Tokenizer` (SmolVLM) and `PreTrainedTokenizerFast` tokenizers; other
tokenizer classes raise `NotImplementedError`.

### Images

TypeScript does not decode image files. Pass decoded CHW float tensors; the
package includes resize and normalize helpers.

### Providers and plans

HTTP providers (`OpenAICompatibleModel`, `JevModel`) and `LocalModel` are
asynchronous only, so use `acall` and `aask` with them. A redirect raises
`ProviderHTTPError` with its 3xx status, as in Python. Plan actions receive
`(state, args)`, and plan validation cannot bind keyword arguments against a
JavaScript signature. Error observations record JavaScript error names
(`Error` where Python records `RuntimeError`).

### JavaScript values

- JavaScript cannot tell `1` from `1.0`. Configuration fields that Python
  stores as floats are written as floats. A Python caller who passes an int for
  a float field (`hidden_dropout_prob=0`) gets a different fingerprint than
  TypeScript's `0.0`.
- JavaScript orders integer-like object keys first, which changes the order of
  `Retrieve` items keyed by integers. Non-string Python dictionary keys decode
  as decimal strings.
- JavaScript cannot tell a closure from a module function, so a saved callback
  (for example `combine`) always needs an explicit `configuration()`.

### Not ported

The Python research scripts and most Python examples. The TypeScript
[examples](../examples/README.md) cover the quickstart, owned vector
operations, tracing and training, text operations over a provider, and your own
pretrained module.

The design notes in [DESIGN.md](../DESIGN.md) map every Python module to its
TypeScript file.
