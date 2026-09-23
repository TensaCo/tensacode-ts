# Changelog

All notable changes to the `tensorcode` npm package. Versions follow the
Python package (`0.4.0a4` is `0.4.0-alpha.4` here); alpha releases may break
APIs. The Python package is the reference implementation: where the two
disagree on something [Parity with Python](docs/parity.md) does not list,
Python's behavior is correct.

## 0.4.0-alpha.4 (unreleased)

This release closes most of the differences 0.4.0-alpha.3 listed as
deliberate. See [Parity with Python](docs/parity.md) for what remains.

### Added

- **Compute.** WebAssembly SIMD kernels, shared by a pool of worker threads,
  run float32 matrix products, convolutions, attention (with `enableGqa`),
  softmax, layer norm, activations, permutes and arithmetic. The kernels are
  checked in, so installing needs no toolchain. `setNumThreads`,
  `getNumThreads`, `setBackend`, `backendInfo` and `clearWeightCache` are
  exported from `tensorcode/nn`. `TENSORCODE_THREADS`, `TENSORCODE_BACKEND=js`
  and `TENSORCODE_WASM_CACHE_MB` configure them. A SmolVLM-256M
  interpretation now takes seconds instead of about ten minutes.
- **Random numbers.** `Generator` is PyTorch's CPU generator, and
  `manualSeed(n)` is `torch.manual_seed(n)`. The samplers, `torch.nn.init`
  and transformers' initialization order are reproduced, so seeded
  construction of every tool, operation and native model gives Python's
  weights bit for bit. Dropout masks and generation sampling make Python's
  draws. `getRngState`/`setRngState` exchange PyTorch's 5056-byte state, and
  `PythonRandom` is CPython's `random.Random`.
- **Images.** Pure TypeScript PNG, JPEG, GIF, WebP and BMP decoders.
  `decodeImage` matches `torchvision.io.decode_image`, `loadImage` /
  `loadImageAsync` match transformers' `load_image_as_tensor`, and `openImage`
  returns a `RasterImage` with Pillow's modes, `convert`, `resize` and
  `exifTranspose`. `ImageEncoder.preprocess` accepts every input Python's
  `ViTImageProcessor` accepts, fetching `http(s)` URLs while the caller
  blocks like Python's `httpx.get` (`apreprocess` / `loadImageAsync` fetch
  without blocking).
- **Python numbers and dictionaries.** `float()`, `int()` and
  `orderedObject()` from `tensorcode` mark whole-number floats, ints and
  ordered dictionaries. Values read from Python files remember their number
  kinds and key order, so they re-save and fingerprint exactly as Python wrote
  them. Python dictionaries with non-string keys decode to `Map`s.
- **Scene language mode.** `interpret` runs a port of transformers'
  `generate`: beam search, classifier-free guidance, sequence bias,
  prompt-lookup decoding, watermarking, stop strings, token healing and every
  logits processor. The Llama text model supports every RoPE type. The
  processor batches prompts with any number of images, and processor assets
  are written for the TokenizersBackend, Llama, GPT2, T5, ALBERT, DeBERTa-v2,
  BERT, RoBERTa and CLIP tokenizer classes.
- **Blocking providers.** `OpenAICompatibleModel`, `JevModel` and `LocalModel`
  block in `complete` like Python's, so `op.call` and `ask` work with them.
  Runtimes that cannot block a thread raise `SynchronousCallUnavailable`.
- **Plans.** Plan validation binds step arguments like Python's
  `inspect.signature(...).bind(...)`, with the same `TypeError` messages;
  `withSignature` declares a signature explicitly. Error observations record
  Python exception names.
- **Diffusers.** `ImageDecoder` builds every UNet and VAE block diffusers
  0.40 can run in these models, including simple cross-attention and
  K-diffusion blocks.
- **PyTorch weights.** Foundations with only `pytorch_model.bin` weights load
  where Python's `from_pretrained` loads them, through a weights-only
  unpickler that rejects anything but tensors and containers. Where Python
  passes `use_safetensors=True`, a Hub repository without safetensors loads
  its conversion pull request, and TypeScript asks the conversion Space to
  open one when none exists.
- **`uint16` tensors**, so 16-bit PNGs decode to `uint16` as in torchvision.
- Optimizer options accept `float()`/`int()` markers
  (`new AdamW(params, { weightDecay: float(0) })` is saved as `0.0`, like
  Python's `weight_decay=0.0`), and hyperparameters loaded from a Python
  checkpoint keep their kinds when saved again.
- 17 more ported examples. Every Python example that does not need the CUDA
  training host has a TypeScript port, except the optional Stable Diffusion
  image path of `pretrained_latent_lifecycle.py`.

### Changed

- `Chatbot`, `Investigator`, `Decision`, `Planner` and `Scene` reject unknown
  or obsolete configuration fields, as Python 0.4.0a4 does, with a
  `ValueError` such as
  `Unknown Planner configuration fields: ['colour']; valid fields: [...]`.
  Nested `Chatbot` cognition fields and each `Scene` mode are checked the same
  way, so `fromPretrained` rejects saved artifacts with unknown fields.
  `RetrievalEncoder` and `ResponseQualityAssessor` report the same message.
  The accepted fields are available as `configFields` (and `cognitionFields`,
  `rankingFields`, `languageFields`).
- Foundation-backed ranking encoders (for example `Planner.fromFoundation`
  with BERT or Electra) are described by their native configuration and
  tensor schemas, as in Python, so their operation fingerprints equal
  Python's.
- `getRngState()` returns the 5056-byte `Uint8Array` (use `rngStateTensor()`
  for a `uint8` tensor). `nextUint32` is replaced by `randomUint32`, and the
  `GeneratorState` type is removed.
- Loading a foundation draws no random numbers. Where Python accepts a
  checkpoint with missing weights (Chatbot, the Investigator verifier,
  ranking and retrieval encoders, Scene), the missing weights are initialized
  as transformers initializes them instead of raising.
- Directory checkpoints use Python's `python_rng` / `torch_rng` / `cuda_rng`
  layout. Checkpoints in the old TypeScript-only layout are rejected, as
  Python rejects them.
- Optimizer state decoded from a checkpoint is a `Map` with integer keys, as
  Python writes it.
- Native text and CLIP embeddings register `position_ids` / `token_type_ids`
  as non-persistent buffers, as transformers does.
- `ImageDecoder.fromFoundation` downloads only the files diffusers loads.
- 16-bit PNGs decode to `uint16` tensors instead of `int32`.

### Fixed

- A special token added to a Unigram tokenizer no longer takes the unknown
  token's id.
- An `Idefics3ImageProcessorPil` processor configuration loads the default
  processor, as in transformers, instead of raising `NotImplementedError`.
- transformers' legacy `LayerNorm.gamma` / `LayerNorm.beta` weight names load.
- A foundation without any tokenizer files gets the class-default tokenizer
  `AutoTokenizer` builds, instead of none, and one with only slow vocabulary
  files (`vocab.txt` for BERT-family models, `vocab.json`/`merges.txt` for
  RoBERTa, CLIP and GPT-2) gets the tokenizer transformers builds from them.
- A piece repeated in a Unigram vocabulary maps to its last id, as in the Rust
  `tokenizers` library.
- The tokenizer class chosen from `model_type` follows transformers 5.17
  (DistilBERT checkpoints use `BertTokenizer`'s input names).
- `BertTokenizer` (also for ELECTRA and DistilBERT), `RobertaTokenizer` and
  `CLIPTokenizer` rebuild their pipeline from the vocabulary and their flags,
  as transformers 5 does, and every tokenizer registers its
  `added_tokens_decoder`, special and extra special tokens and installs its
  class post-processor. `special_tokens_map.json` takes precedence over
  `tokenizer_config.json` unless the latter has `added_tokens_decoder`, as in
  `_from_pretrained`. Tokenizers whose files differ from what the class builds
  (for example a `BertProcessing` post-processor or a `do_lower_case` flag the
  normalizer does not follow) now give Python's backend JSON, and
  `Scene.fromFoundation` persists the tokenizer Python persists.
- A `tokenizer.json` with a BPE `dropout` loads and samples merges as the Rust
  `tokenizers` library does, instead of raising.
- Chat templates: `escape` escapes HTML and tuples render as tuples, as in
  jinja2. The `max`, `min`, `sum`, `dictsort`, `format`, `batch`, `truncate`,
  `center` and `forceescape` filters, the `divisibleby` and comparison tests,
  and `filter`, `with` and `raw` blocks are supported.
- Generation settings read from a file keep their Python number kind, so
  `"repetition_penalty": 2` or `"top_k": 5.0` raise transformers' messages.
  Too few image features raise torch's `RuntimeError`.
- Generated model cards import tools from `tensorcode/tools` (and operations
  from their entry points) instead of `tensorcode`, which does not export them.

## 0.4.0-alpha.3

First npm release of the TypeScript port, matching Python `0.4.0a3`.
