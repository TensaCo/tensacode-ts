# Tools and pretrained models

The public tools are `Chatbot`, `Investigator`, `Decision`, `Planner` and
`Scene`, all exported from `tensorcode/tools`. Each tool owns its trainable
components. You can build a fresh architecture from JSON configuration or load
a complete TensorCode artifact:

```ts
import { Investigator } from 'tensorcode/tools';

let model = new Investigator({ vocabulary: ['evidence', 'hypothesis'] });
await model.savePretrained('./model');
model = await Investigator.fromPretrained('./model', { device: 'cpu' });
```

Construction does not download anything, and it does not give the model any
pretrained competence. Tools are `tensorcode/nn` modules, so they support
`parameters()`, `train()`, `eval()`, `stateDict()` and `loadStateDict()`. Call
a tool with `tool.call(inputs)`. This guide mirrors the Python
[tools guide](https://tensorcode.dev/docs/). Receipts, configurations and
artifacts use the same snake_case fields in both languages.

## Pretrained artifacts and the Hugging Face Hub

`await Tool.fromPretrained(repoIdOrPath, { revision, localFilesOnly, cacheDir,
token, device })` accepts a local directory or a Hub model ID. Pin `revision` to
a commit for reproducible loading. Downloads use the same cache layout as
`huggingface_hub` (`$HF_HUB_CACHE`, `$HF_HOME/hub` or
`~/.cache/huggingface/hub`). The token comes from `token`, `$HF_TOKEN`, or the
file written by `hf auth login`. Offline loading (`localFilesOnly: true`) needs
an existing directory or a cached snapshot.

`await tool.savePretrained(directory)` writes `tensorcode_config.json`,
`model.safetensors`, a `README.md` model card and any tool-specific assets. The
manifest names a known concrete class (`tensorcode.tools.investigator.Investigator`)
and a format version. Loading rejects incompatible artifacts. It never runs code
chosen by the artifact. The Python and TypeScript packages read each other's
artifacts.

`await tool.pushToHub(repoId, { private, revision, token, commitMessage,
modelCard })` publishes model artifacts explicitly. It never publishes sessions,
optimizer state or collected experience.

`PretrainedModule` (also exported from `tensorcode/tools`) is the base class of
every tool. Extend it to make your own owned, saveable models; see
[`examples/pretrainedLifecycle.ts`](../examples/pretrainedLifecycle.ts).

## Chatbot

```ts
import { Chatbot } from 'tensorcode/tools';

// Run after saving or downloading a compatible TensorCode chatbot artifact.
const bot = await Chatbot.fromPretrained('./chatbot-model');
console.log(bot.call('Help me investigate the evidence.'));

const other = bot.newSession();
console.log(other.call('Start a separate investigation.'));
await bot.saveSession('conversation.json');
await bot.savePretrained('./chatbot-model');
```

The model owns a tokenizing sequence encoder, a learned workspace and a local T5
language decoder. It conditions generation on the workspace representations and
needs no remote provider.

`new Chatbot(config)` builds a fresh seq2seq architecture. The required keys are
`foundation_config` (a supported transformers seq2seq configuration with
`model_type`) and `tokenizer_json`. Optional limits are `max_input_tokens`
(default 512), `max_target_tokens` (128), `max_new_tokens` (64) and `max_turns`
(16), and `workspace` configures the slots and update steps.

`await Chatbot.fromFoundation(repo, { revision, localFilesOnly })` is an
explicit training bootstrap. It imports pretrained T5 weights and sets up a new,
untrained workspace. Supervise with `lossBatch(inputs, targets)`, or use
`trainer.capture` with equal-length text lists.

`generateBatch(texts)` is stateless. `bot.call(text)` continues the default
conversation, while `newSession()` shares the weights but keeps its own history.
A failed turn does not commit partial history. `lastResult` reports the source
evidence and whether token limits truncated the prompt. `capabilities` shows
which cognitive components a loaded artifact contains.

## Investigator and Decision

`new Investigator(config)` takes a nonempty `vocabulary` of unique strings, and
optionally `dimensions` (32), `slots` (4), `steps` (2) and `max_tokens` (256).

```ts
const result = model.call({
  question: 'Which hypothesis best fits the evidence?',
  evidence: [{ source_id: 'report:1', text: 'Observed evidence' }],
  hypotheses: [
    { id: 'a', text: 'First hypothesis' },
    { id: 'b', text: 'Second hypothesis' },
  ],
});
console.log(result.selected_id, result.candidates, result.attention_source_ids);
```

Feedback is a hypothesis ID, an index, or a finite nonnegative distribution over
the supplied hypotheses that sums to one. Candidates include scores and
uncalibrated probabilities. Receipts keep the source IDs, attention and slot
relations. Attention is a model diagnostic, not a causal explanation.

`await Investigator.fromFoundation(repo, { revision, localFilesOnly, options })`
and `Planner.fromFoundation(...)` bootstrap an owned contextual encoder (BERT,
RoBERTa, Electra, DistilBERT). The encoder is frozen by default; pass
`options: { freeze_foundation: false }` to train it too. `fromFoundations(...)` also adds a T5 hypothesis
generator and an NLI verifier (BERT-family or DeBERTa-v2), and
`fromRetrievalFoundation(...)` adds an owned retrieval encoder. The workspace
and scoring head still start untrained.

`newSession()` keeps an independent history of interpretation receipts. Each
call supplies its complete current evidence, and receipts add
`previous_selected_id` and `revised`.

`Decision` has the same architecture and input contract, with its own class
identity. Configurations with an owned generator and verifier also support
`propose`, `verify`, `investigate` and cognitive sessions. None of these prove a
hypothesis true.

## Planner

`Planner` uses the same configuration and workspace. Its inputs contain `goal`,
sourced `evidence` and `plans` with `id` and `text`. It predicts scalar outcomes
and selects the highest-scoring candidate. It does not execute actions.
Supervise an observed outcome with `{ candidate_id: 'plan-id', outcome: 1.0 }`,
or give one finite outcome per candidate when every candidate was actually
observed.

## Scene

In ranking mode, `Scene` ranks supplied descriptions against image pixels and a
question. It uses learned image patches, spatial position encodings, text
representations and the shared workspace. `await Scene.fromFoundation(repoId,
{ revision, dimensions, slots, steps })` imports pinned CLIP perception weights.

```ts
import { Scene } from 'tensorcode/tools';

const scene = await Scene.fromPretrained('./scene-model');
const receipt = scene.call({
  pixels,                     // a finite floating CHW tensor in [0, 1]
  source_id: 'photo:17',
  question: 'Which description matches the image?',
  candidates: [
    { id: 'left', text: 'The cup is left of the plate.' },
    { id: 'right', text: 'The cup is right of the plate.' },
  ],
});
```

Callers decode and preprocess images themselves; TensorCode does not read image
files. Coordinates in the receipt refer to the input tensor.
In language mode, `Scene` interprets a full image with an owned Idefics3
(SmolVLM) foundation. `await Scene.fromLanguageFoundation(repoId, { revision })`
imports the pinned weights and processor assets. The workspace residual starts
inactive (its gate is zero), so the imported model behaves exactly like the
foundation until you supervise it with reviewer-written descriptions.

```ts
const scene = await Scene.fromLanguageFoundation('HuggingFaceTB/SmolVLM-256M-Instruct', { revision: 'a-commit-sha' });
const result = scene.interpret({ pixels, source_id: 'photo:17', question: 'What is on the table?' }, { maxNewTokens: 32 });
result.interpretation;        // generated text
result.verification;          // always 'unverified'
result.completion_status;     // 'complete' or 'token_limit'
scene.loss({ pixels, source_id: 'photo:17', question: 'What is on the table?' }, 'A cup and a plate.');
```

Interpretations are unverified. They are not extracted facts or scene graphs,
and the receipt carries no boxes or claims. Decoding is greedy. Inference runs
on the CPU (WebAssembly SIMD kernels on worker threads), so a SmolVLM-256M
interpretation takes seconds.

## Sessions and explicit actions

For revisable evidence and episodic memory, call
`investigator.newCognitiveSession({ memory: { capacity: 256, top_k: 5 } })`.
The session supports `ingest`, `reviseEvidence`, `removeEvidence`,
`investigate`, `remember`, `retrieve`, `newEpisode`, `fork`, `snapshot` and
`await save(path)`. Restore it with
`await investigator.loadCognitiveSession(path)`. Cognitive records (`Evidence`,
`Hypothesis`, `Assessment`, `RetrievalHit`, `Goal`, `Observation`, `Plan`) come
from `tensorcode/tools/cognition`.

`planner.newExecutor({ actions, replan, maxSteps })` builds a bounded executor
for explicitly structured plans:

```ts
import { ActionOutcome, ExecutablePlan, PlanStep } from 'tensorcode/tools/actions';

const executor = planner.newExecutor({
  actions: {
    restart: (state: { restarted: boolean }) => new ActionOutcome({ restarted: true }, 'restarted', true),
  },
  replan: () => null,           // abstain instead of replanning
  maxSteps: 4,
});
const result = await executor.call({ restarted: false },
  new ExecutablePlan('plan-a', [new PlanStep('restart')]));
console.log(result.stopReason, result.experiences);
```

Actions receive `(state, args)` and return
`ActionOutcome(state, receipt, done)`. Before any effect, every step's
arguments are bound like Python's `inspect.signature(action).bind(None,
**arguments)`: the keyword parameters are the properties the action
destructures from `args` (`(state, { amount, note = 'none' })` requires
`amount`, `note` is optional, `...rest` or a plain `args` parameter accepts any
keyword), and a mismatch raises Python's `TypeError` message (for example
`missing a required argument: 'amount'`). Declare parameters explicitly with
`withSignature(fn, { parameters: ['amount', 'note?'] })` or the executor's
`signatures` option when the source does not show them (bound or native
functions). A failing action is recorded as an error observation with Python's
exception name (`RuntimeError` for a JavaScript `Error`). Execution returns a `PlanExecutionResult`
with a stop reason of `completed`, `abstained`, `policy_error` or
`budget_exhausted`. Each `OutcomeExperience` converts to training feedback with
`toTarget(outcome)` or to sourced evidence with `asEvidence()`. The lower-level
`actionLoop({ chooser, actions, maxSteps })` runs a supplied chooser without a
model-owned planner. Action loops and executors are asynchronous, and callbacks
may be sync or async. Your application supplies the actions and their authority,
and effects may not be reversible.

For composing operations yourself, see [operations](operations.md). For
training and checkpoints, see [training](training.md).
