# TensorCode for TypeScript

**Website:** [tensorcode.dev](https://tensorcode.dev) ·
**Docs:** [tensorcode.dev/docs](https://tensorcode.dev/docs/) ·
**Source:** [GitHub](https://github.com/TensaCo/tensacode-ts) ·
**Python package:** [tensacode-py](https://github.com/TensaCo/tensacode-py)

TensorCode builds trainable programs from callable operations and small tools
that own their models. This is the TypeScript port of the Python `tensorcode`
package. It has the same operations, tools, tracing, training and artifact
formats, and it runs in Node.js with **no runtime dependencies**.

You can compose encoders, scorers and decoders (`tensorcode/ops/*`) or use a
complete tool such as `Investigator`, `Planner` or `Chatbot`
(`tensorcode/tools`). Collect reviewed feedback with explicit provenance, train
it with the built-in autograd core, and save everything as a data-only artifact.
The artifact reloads in a fresh process, from the Hugging Face Hub, or in the
Python package. Tracing records which operation produced which value, so
supervised local tensor paths can be replayed and trained. Tracing does not make
arbitrary JavaScript or remote model calls differentiable.

> **Status: 0.4.0 alpha** (Python `0.4.0a3`). APIs may change between alphas.
> Importing any entry point performs no I/O and loads no model weights. The
> measured behavior and its limits are documented at
> [tensorcode.dev/docs](https://tensorcode.dev/docs/). Consistent benefits of
> the learned cognitive workspace are not yet established.

## Install

This needs Node.js 20.16 or newer. The package is not on npm yet, so install it
from GitHub. The `prepare` script builds `dist/`:

```bash
npm install github:TensaCo/tensacode-ts
```

The package is ESM-only (`import`, not `require`) and ships its own type
declarations. The optional peer `@huggingface/transformers` is only needed for
`integrations.LocalModel`.

## 30-second example

This trains an `Investigator` to rank two supplied hypotheses from log evidence,
saves the model and reloads it. It runs offline on CPU in well under a second.
Save it as `quickstart.mts` and run `node quickstart.mts` (Node 22.18 or newer
runs TypeScript directly; otherwise use `npx tsx quickstart.mts`).

```ts
import { AdamW, manualSeed } from 'tensorcode/nn';
import { Investigator } from 'tensorcode/tools';
import { Trainer } from 'tensorcode/training';

manualSeed(0);
const model = new Investigator({
  vocabulary: ['database', 'network', 'connection', 'refused', 'packet', 'loss'],
  dimensions: 16, slots: 2, steps: 1,
});
const trainer = Trainer.fromTool(model, { optimizer: (params) => new AdamW(params, { lr: 0.01 }) });

const hypotheses = [
  { id: 'database', text: 'database connection refused' },
  { id: 'network', text: 'network packet loss' },
];
const incident = (logLine: string) => ({
  question: 'which component failed',
  evidence: [{ source_id: 'log:1', text: logLine }],
  hypotheses,
});

// Reviewed feedback, with explicit provenance, becomes training experience.
const experiences = [
  trainer.capture(incident('connection refused'), 'database', { source: 'review:1' }),
  trainer.capture(incident('packet loss'), 'network', { source: 'review:2' }),
];
const losses = trainer.fit(experiences, { epochs: 30 });

await model.savePretrained('./investigator');
const restored = await Investigator.fromPretrained('./investigator');
console.log(restored.call(incident('packet loss')).selected_id); // network
```

Two authored cases show the lifecycle. They do not show that the model can
investigate anything. The result also includes every candidate's score and the
source-linked evidence. Probabilities are uncalibrated. The
[quickstart](docs/quickstart.md) extends this with persisted experience files,
resumable training checkpoints and loading in a fresh process.

## What is inside

| Entry point | Python module | Contents |
|---|---|---|
| `tensorcode` | `tensorcode` | `trace()`, `Trace`, `InputRef`, `OutputRef`, `version`, error classes |
| `tensorcode/ops` | `tensorcode.ops` | The `Operation` contract (`op.call(value, { context })`, `await op.acall(...)`) and `ModuleOperation` |
| `tensorcode/ops/vec` | `tensorcode.ops.vec` | `Space`, `Latent`, `Transform`, `Classify`, `Score`, `Decide`, `Retrieve`, `Decode`, `VocabularyEncoder`, `PatchEncoder`, `TextEncoder`, `ImageEncoder`, `TextDecoder` |
| `tensorcode/ops/text` | `tensorcode.ops.text` | `Message`, message encoders, owned and provider-backed `Transform`, `Classify`, `Decide`, `Score`, `Retrieve`, and `ask` |
| `tensorcode/ops/graph` | `tensorcode.ops.graph` | `Graph` and `SourceAnchor` records; graph operations are symbolic stubs that raise `NotImplementedError` |
| `tensorcode/tools` | `tensorcode.tools` | `Chatbot`, `Investigator`, `Decision`, `Planner` and `Scene`, plus the `PretrainedModule` base for your own tools |
| `tensorcode/tools/cognition`, `tensorcode/tools/actions` | same | Cognitive records (`Evidence`, `Hypothesis`, ...), action loops and plan execution |
| `tensorcode/training` | `tensorcode.training` | `Trainer.fromTool` / `Trainer.fromOps`, `loadExperience`, calibration utilities |
| `tensorcode/integrations` | `tensorcode.integrations` | `OpenAICompatibleModel`, `JevModel`, `LocalModel` and provider errors |
| `tensorcode/nn` | PyTorch (subset) | Tensors, reverse-mode autograd, PyTorch-compatible layers, `SGD`/`Adam`/`AdamW`, safetensors, a seedable RNG |

Generated hypotheses are not evidence, and generated plans are not executable
code. Evidence, policies and actions stay explicit in your code.

The TypeScript API uses camelCase (`savePretrained`, `fromPretrained`,
`Trainer.fromTool`, `loadExperience`). Persisted data keeps Python's snake_case
(`selected_id`, `tensorcode_config.json` fields, experience files), so the two
implementations can share artifacts. Anything that touches the filesystem or
network returns a `Promise`. Pure computation is synchronous.

## Parity with Python

| Area | TypeScript | Notes |
|---|---|---|
| Tracing, supervision, replay, release | Yes | Async capture uses `AsyncLocalStorage` |
| Experience files (`tensorcode.experience`) | Interchangeable | Operation fingerprints equal Python's for the same configuration |
| Model artifacts (`savePretrained` / `fromPretrained`, Hub) | Interchangeable | Python artifacts load in TS and re-save with a byte-identical manifest. Weights re-save byte-identically except that tied-alias metadata order varies, because Python itself writes that order nondeterministically. The Hub cache layout is shared |
| Vector operations (`ops.vec`) | Yes | Linear, MLP and native BERT/RoBERTa/DistilBERT transformer variants; T5 text and ViT image encoders |
| `ImageDecoder` (latent diffusion) | Yes | diffusers cross-attention UNet, AutoencoderKL and DDIM. Pass `context.noise` for samples identical to Python's (seeded noise uses the TensorCode generator) |
| Text operations (`ops.text`) | Yes | Owned T5 models (generation and likelihood decoding) and external providers. HTTP providers are async-only |
| Graph operations | Symbolic stubs | Same as Python |
| `Chatbot`, `Investigator`, `Decision`, `Planner` | Yes | Including cognitive sessions, episodic memory, verifiers and plan execution |
| `Scene` ranking mode | Yes | CLIP bootstrap via `Scene.fromFoundation` |
| `Scene` language mode (Idefics3/SmolVLM) | Yes | `Scene.fromLanguageFoundation` and `interpret` with greedy decoding; the processor (image splitting, LANCZOS, prompt expansion) and saved processor assets match Python byte for byte. Pure-JavaScript compute makes a full SmolVLM interpretation slow (minutes on a CPU) |
| `Trainer`, checkpoints | Yes | SGD, Adam, AdamW. Python directory checkpoints load in TS, but their PyTorch/CPython RNG states are ignored. TS directory checkpoints store the TensorCode RNG and do not load in Python. Standalone checkpoint files are interchangeable |
| Native architectures | ALBERT, BERT, RoBERTa, Electra, DistilBERT, DeBERTa-v2, T5, ViT, CLIP, Llama, Idefics3 | transformers 5.17 parameter names; safetensors weights only. Python loads any transformers `AutoModel` for text foundations; TypeScript implements these |
| Tokenizers | `tokenizer.json` runtime | WordPiece, BPE, Unigram, WordLevel |
| Integrations | OpenAI-compatible, Jev, Transformers.js `LocalModel` | No implicit retries or redirects |
| Compute | CPU, pure JavaScript | Random streams differ from PyTorch, so fresh initializations differ; loaded weights are identical |

[DESIGN.md](DESIGN.md) has the full Python-to-TypeScript mapping, the
conventions and the deliberate differences.

## Guides

- [Quickstart](docs/quickstart.md): the offline training lifecycle, end to end.
- [Operations](docs/operations.md): vector, text and graph operations, providers
  and custom operations.
- [Tools](docs/tools.md): Chatbot, Investigator, Decision, Planner and Scene,
  pretrained artifacts and the Hub.
- [Training](docs/training.md): tracing, experience, replay, checkpoints and
  calibration.
- [Examples](examples/README.md): runnable programs.
- Full documentation, validation results and the Python guides:
  [tensorcode.dev/docs](https://tensorcode.dev/docs/).

## Development

```bash
npm install          # also builds dist/ through the prepare script
npm run typecheck    # tsc --noEmit over src and tests
npm run build        # tsc -> dist/
npm test             # vitest; needs neither Python nor network
node examples/investigatorQuickstart.ts
```

Library code lives in `src/`. Tests in `test/` port the Python suite and check
parity against fixtures generated by the Python package
(`npm run fixtures -- <generator>`, which needs `../python/.venv`). Tests that
need a cached Hugging Face checkpoint skip when it is absent.
[MIT license](LICENSE).
