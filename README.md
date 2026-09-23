# TensorCode for TypeScript

Composable cognitive operations and traceable, trainable programs — the
TypeScript port of [TensorCode](https://tensorcode.dev) (documentation:
<https://tensorcode.dev/docs/>).

> **Status: 0.4.0 alpha, under construction.** The foundation is in place: a
> dependency-free tensor/autograd core with PyTorch-compatible modules,
> optimizers and safetensors; native BERT/RoBERTa/Electra/DistilBERT/T5/ViT/CLIP
> architectures that load Hugging Face checkpoints; a `tokenizer.json` runtime;
> tracing and replay; operation fingerprints; and Python-compatible model
> artifacts. Vector, text, tool and training modules are being ported. See
> [DESIGN.md](DESIGN.md) for the architecture, the Python → TypeScript mapping and
> the conventions.

```ts
import { trace } from 'tensorcode';
import { Operation, type Context } from 'tensorcode/ops';

class Double extends Operation<number[], number[]> {
  override get replayable() { return true; }
  forward(value: number[], context: Context | null) {
    return value.map((item) => item * 2);
  }
}

const session = trace();
const result = session.run(() => new Double().call([1, 2]));
session.replay(session.ref(result)); // → [2, 4]
```

Node.js ≥ 20.16. No runtime dependencies; importing performs no I/O.

## Development

```bash
npm install
npm run build
npm test
```

MIT license.
