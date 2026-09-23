# Operations and providers

Every operation uses one calling convention:

```ts
const output = op.call(value, { context });   // Python: op(value, context=...)
const later = await op.acall(value, { context });
```

Implement `forward(value, context)` (and optionally `aforward`) and invoke the
instance through `call`/`acall`, so tracing keeps the boundary. `context` holds
conditioning data and is `null` when absent. Required operands belong in the
primary value. Pure operations opt into replay by overriding
`get replayable() { return true; }`. Importing the operation modules does not
load model weights or touch the network.

This guide mirrors the Python
[operations guide](https://tensorcode.dev/docs/). The concepts, configuration
keys and persisted formats are identical; only the API spelling is camelCase.

## Vectors (`tensorcode/ops/vec`)

```ts
import { tensor } from 'tensorcode/nn';
import * as vec from 'tensorcode/ops/vec';

const textSpace = new vec.Space('application.text', 64);
const sharedSpace = new vec.Space('application.retrieval', 32);
const encode = new vec.VocabularyEncoder({
  vocabulary: ['refund', 'transfer', 'card'], dimensions: 64,
  output_space: textSpace.configuration(),
});
const project = new vec.Transform({
  architecture: 'linear', input_space: textSpace.configuration(),
  output_space: sharedSpace.configuration(),
});
const query = project.call(encode.call('refund'));   // a Latent in sharedSpace
```

`new Space(name, dimensions, { version, organization, dtype, device })`
identifies a representation. Compatibility compares all of its fields. Equal
dimensions alone do not make two spaces compatible. A shared name is a contract
you author; it is not evidence of trained alignment.

`new Latent(tensor, space, { mask, coordinates, sources, metadata })` keeps the
supplied tensor. The last dimension must match the space. A boolean mask matches
the leading dimensions, and coordinates add a coordinate axis to that leading
shape.

| Operation | Contract |
|---|---|
| `TextEncoder(config)` | Owned T5/BERT-family text transformer: raw text → `output_space`. `readout` is `'sequence'`, `'pooled'` (masked mean) or output encoding |
| `ImageEncoder(config)` | Owned ViT plus a tensor-only image processor: CHW image → `output_space` |
| `TextDecoder(config)` | `input_space` → generated text through an explicit linear or identity bridge |
| `ImageDecoder(config)` | **Not available in TypeScript** (latent diffusion); throws `NotImplementedError` |
| `VocabularyEncoder(config)` | A `vocabulary` list, `dimensions`, optional `output_space`; lowercase regex tokens, mean-pooled trainable embeddings |
| `Transform(config)` | Owned `linear`, `mlp` or native `transformer`; declared `input_space` and `output_space`; returns a `Latent` |
| `Classify(config)` | Owned head with `input_space` and `labels`; returns a `Prediction` with `logits`, softmax `probabilities` and `value`/`values` |
| `PatchEncoder(config)` | Owned convolution; `patch_size`, `in_channels`, `output_space`; CHW/BCHW images → spatial latent patches |
| `Decode(config)` | Owned `linear`/`mlp`/`transformer` readout: `input_space` → a tensor of `output_dimensions` |
| `Score(config)` | Owned candidate scorer with declared `query_space`, `candidate_space` and score `meaning` |
| `Decide(config)` / `Retrieve(config)` | Parameter-free selection: `largest`, and `k` for retrieval |

Encoders and decoders also live in `tensorcode/ops/vec/encode` and
`tensorcode/ops/vec/decode`. Their class identities
(`tensorcode.ops.vec.encode.TextEncoder`, ...) match Python's.

Candidate scoring takes
`new CandidateSet(query, candidates, identities, metadata)`. Candidate tensors
have shape `[..., N, features]`, and the query and candidate batch shapes must
agree. Identities are unique, stable strings.

```ts
const candidates = new vec.CandidateSet(
  new vec.Latent(tensor([[1, 0, 0, 0]]), new vec.Space('shared', 4)),
  new vec.Latent(tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]), new vec.Space('shared', 4)),
  ['a', 'b', 'c'],
);
const score = new vec.Score({
  architecture: 'mlp', hidden_dimensions: [8],
  query_space: { name: 'shared', dimensions: 4 },
  candidate_space: { name: 'shared', dimensions: 4 },
  meaning: 'learned unnormalized relevance',
});
const scored = score.call(candidates);
const decision = new vec.Decide({ largest: true }).call(scored);
const retrieval = new vec.Retrieve({ k: 2, largest: true }).call(scored);
```

Scores have shape `[..., N]` and keep their declared meaning. They are not
probabilities unless you make them so. Selectors skip masked candidates, and
retrieval cannot return more than the valid count.

Constructors take JSON configuration and create every parameter up front, so
you can build the optimizer immediately. Owned operations support
`savePretrained(directory)` and `await X.fromPretrained(directory)`. Supported
`await X.fromFoundation(repo, { revision, ... })` factories import pretrained
BERT, RoBERTa, DistilBERT, T5, ViT or CLIP weights; any newly added projections
start untrained. The advanced `X.fromModule(module, options)` factories wrap a
`tensorcode/nn` module you supply. Arbitrary executable modules cannot be
rebuilt safely from data-only artifacts, so saving one throws instead of
silently dropping its behavior.

See [`examples/ownedVectorLifecycle.ts`](../examples/ownedVectorLifecycle.ts)
for a complete train → save → reload → resume program.

## Messages and model providers (`tensorcode/ops/text`)

`new Message(role, content)` holds plain text or immutable parts
(`new TextPart(text)`, `new ImagePart({ data | url, mediaType, detail })`).
Encoding never downloads URLs.

`TextEncoder`, `ImageEncoder` and `TextDecoder` are pure message serialization
operations. The structured operations `Transform`, `Classify`, `Decide`, `Score`
and `Retrieve` can either own a local seq2seq model or wrap an explicit external
provider:

```ts
import * as text from 'tensorcode/ops/text';

class Rules implements text.Model {
  complete(request: text.ModelRequest): text.ModelOutput {
    return new text.ModelOutput({ structured: { label: 'billing', abstained: false } });
  }
}

const route = text.Classify.fromModel(new Rules(), {
  labels: ['billing', 'technical'],
  descriptions: { billing: 'payments, charges and refunds' },
  instructions: 'Route the support ticket',
});
const result = route.call([new text.Message('user', 'I was charged twice')]);
console.log(result.label); // billing
```

Providers implement `Model` (`complete`), `AsyncModel` (`acomplete`),
`BatchModel` or `QuestionModel` (`completeQuestions`). A function
`(messages) => string` is also accepted by `Transform.fromModel`. Structured
responses are validated strictly. A contract violation raises
`InvalidModelOutput`, and nothing is repaired.

Classifications and decisions select only the configured alternatives. A
returned distribution must contain exactly those alternatives, with finite
values in `[0, 1]` that sum to one within `0.001`. Results are frozen records:
`ClassificationResult(label, { distribution, confidence, abstained })`,
`DecisionResult(choice, ...)`, `ScoreResult(value, ...)` and
`RetrievalResult(keys, items, { scores, abstained })`.

Owned operations take `native_config` and an embedded `tokenizer`
configuration, or bootstrap from a T5 checkpoint:

```ts
const route = await text.Classify.fromFoundation('google/flan-t5-small', {
  revision: '<commit>',
  config: { labels: ['billing', 'technical'], decoding: 'likelihood' },
});
```

`decoding: 'generate'` (the default) generates the JSON response as text.
`decoding: 'likelihood'` encodes the input once and scores every alternative
with the decoder in one batch. It always returns a complete distribution, but
the probabilities stay uncalibrated. Owned text operations declare a
teacher-forced objective, so `Trainer.fromTool(operation)` trains them directly.

`text.ask(messages, { name: operation, ... })` answers several named structured
questions about the same messages, and `await text.aask(...)` is the async form.
When every operation wraps the same `QuestionModel`, all questions travel in one
exchange.

### Integrations (`tensorcode/integrations`)

| Adapter | Behavior |
|---|---|
| `new OpenAICompatibleModel({ baseUrl, model, apiKey, api: 'chat_completions' \| 'responses', timeout })` | Text and images, strict JSON schemas |
| `new JevModel({ apiKey, baseUrl, model })` | The TypeSafe `/v1/systemone` mapping; `completeQuestions` sends several questions in one request |
| `await LocalModel.fromPretrained(modelId, { revision })` or `new LocalModel(model, processor, { modelId })` | An explicitly supplied Transformers.js model and processor (optional peer `@huggingface/transformers`) |

HTTP adapters use `fetch` with one buffered request and no retries or fallback.
They are **asynchronous only**, so use `acall`/`aask` with them. Redirects,
refusals and truncated responses raise `ProviderError` subclasses:
`ProviderHTTPError` (with `.status`), `ProviderTimeout` and
`ProviderProtocolError`. Credentials never appear in configuration or error
messages.

## Graphs (`tensorcode/ops/graph`)

`new Graph(nodes, { edges, sources, identity, attributes, nodeAttributes,
edgeAttributes, sourceAnchors })` preserves the supplied node identities,
relation strings, competing edges and source evidence, deep-frozen.
`new SourceAnchor(source, { target, location, attributes })` targets a graph, a
node or an edge index.

Graph operations (`Encode`, `TextEncode`, `Decode`, `TextDecode`, `Transform`,
`Score`, `Retrieve`, `Decide`, `Classify`) are **symbolic stubs**, as in Python.
Every call throws `NotImplementedError`.

## Write your own operation

```ts
import { readFileSync } from 'node:fs';
import { Operation, type Context } from 'tensorcode/ops';

class ReadText extends Operation<string, string> {
  forward(path: string, context: Context | null): string {
    if (context) throw new Error('ReadText does not consume context');
    return readFileSync(path, 'utf8');
  }
}

const contents = new ReadText().call('README.md');
```

Invoke `call(...)`, not `forward(...)`, to keep the tracing boundary. The base
`replayable = false` is right for I/O. Opt into replay only for operations that
can safely recompute. For tensor code, extend `ModuleOperation` or wrap a module
with `vec.Transform.fromModule`. If an operation will be persisted, give it a
stable `static qualifiedName` and a truthful JSON `configuration()`; see
[training](training.md#configuration-and-codecs).
