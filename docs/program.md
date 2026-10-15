# Compiled numerical programs

`tensorcode/program` compiles a bounded numerical subset of TypeScript into a
portable, data-only scalar SSA circuit. Python uses the same versioned artifact.
Programs can execute without training, own trainable parameters, or compose frozen
regions with learned residuals. This API is separate from the symbolic
`tensorcode/ops/graph` namespace.

## How it works

Write a supported numerical function, or construct one with `Builder`. Compilation
turns it into an **intermediate representation (IR)**: a list of numerical
instructions and their dependencies. Each intermediate value is assigned once
(**SSA**, static single assignment). Branches select values, and bounded loops
expand into instructions. No examples, fitted weights or pretrained model are
needed to reproduce the authored rule.

You can run that circuit, check it against an independent function, and then add
parameters to learn. Training changes only those declared parameters; it does
not automatically rewrite the source or discover a new algorithm. Operation
tracing records calls between existing operations and is a separate mechanism.

The examples describe this source checkout; use its development installation
when trying the compiler APIs.

## Compile and check a rule

```ts
import { compile, int32 } from 'tensorcode/program';

const choose = compile((a: number) => a > 4 ? 10 : 12, {
  inputs: { a: int32({ min: -128, max: 127 }) },
  semantics: 'exact',
  backend: 'wasm',
});
choose.run({ a: 5 }); // { result: 10 }
choose.verify(); // { passed: true, cases: 256, exhaustive: true }
choose.graph(); // defensive copy of the portable IR
const artifact = JSON.stringify(choose);
```

The compiler parses syntax; it does not execute a sample input to trace a branch.
A supplied function is retained only as an independent verification reference.
When compiling source text or loading an artifact, supply that reference to
`verify(inputs => expectedOutput)`. Verification executes the supplied reference.

## Source and numeric contracts

Supported syntax includes scalar parameters, numeric/boolean literals, local
`let`/`const` assignments, `+ - * /`, unary minus/not, strict equality/inequality,
ordered comparisons, ternaries, `if`/`else`, early returns, fixed arrays and
indexing, `relu(x)`, `sigmoid(x)`, and `Math.max(0, x)`. Numeric conditions use
finite-number truthiness. Statically bounded `for (let i = literal; i < literal;
i++)` loops (including descending/inclusive variants) are unrolled. A fixed array
return becomes outputs named `"0"`, `"1"`, etc. The builder's `scan` preserves
left-to-right addition order.

No closure capture, arbitrary property access, function calls, dynamic loops,
exceptions, recursion, `var`, local shadowing, async functions, or implicit host
side effects are compiled. Loop induction variables cannot be modified by their
bodies. Unsupported syntax raises an error; source is never evaluated to implement
an unsupported construct. The defaults bound loops to 1,024 total expanded iterations and programs
to 100,000 nodes. `maxIterations` and `maxNodes` configure these budgets.

Domains are `int32({min,max})`, `boolean()`, `float32({min,max})`, and
`float64({min,max})`. Inputs are finite numbers; boolean wires use 0/1. Integer
inputs are checked, never silently truncated or wrapped. Arithmetic defaults to
binary64, matching finite JavaScript Number arithmetic, with no reassociation or
fused multiply-add. `dtype: 'float32'` explicitly requests rounding at each circuit
node; this is a different arithmetic contract from ordinary JavaScript. Input
domains describe input representation independently of computation dtype.
NaN, infinity, division by zero, invalid indices and domain overflow are errors.
Signed zero is preserved by artifacts serialized as objects in memory, but ordinary
JSON number serialization does not preserve negative zero; do not depend on its
sign across JSON transport. Source division by zero is outside the finite contract.

Exact comparison is a comparison instruction, not a ReLU approximation. Thus
`4.5 > 4` produces 1 with a floating input domain. Exact selection executes only
the chosen branch, including in hybrid forward execution.

| Semantics | Forward | Backward |
| --- | --- | --- |
| `exact` | Checked numerical program | Ordinary piecewise derivatives; comparisons/index decisions have zero gradient |
| `relaxed` | Sigmoid comparisons, Gaussian equality, continuous selection | Derivative of the relaxation |
| `hybrid` | Same hard comparisons and selection as exact | Explicit `surrogate: 'sigmoid'` policy |

`temperature` controls relaxations. A vector–Jacobian product (VJP) carries an output gradient back to its inputs.
Hybrid selection's VJP uses both branch
values. If an inactive branch is undefined, exact forward can succeed while the
requested hybrid derivative fails. ReLU's derivative at zero is defined as zero.
These are first-order policies, not claims of mathematical differentiability at
hard decision boundaries or useful learning signals for every program.

## Builder, composition and training

```ts
import { Builder } from 'tensorcode/program';
import { tensor, SGD } from 'tensorcode/nn';

const exact = new Builder();
const a = exact.input('a');
const algorithm = exact.build(exact.select(a.gt(4), 10, 12));

const b = new Builder();
const x = b.input('x');
const correction = b.parameter('correction', 0);
const region = b.include(algorithm, { a: x }, 'algorithm');
const program = b.build(region.result!.add(correction.mul(x)));
const model = program.toTensorModule();
const optimizer = new SGD(model.parameters(), { lr: 0.01 });
optimizer.zeroGrad();
model.forward({ x: tensor(5, { dtype: 'float64' }) }).result!.sub(11).pow(2).backward();
optimizer.step();
const trainedArtifact = model.artifact();
```

The rule still returns 10 at x=5. Initially the correction is zero, so the
combined output is 10; the target is 11. The optimizer updates only `correction`,
leaving the comparison and the rule's constants fixed. This is a learned residual,
not a proof that the trained output still equals the original rule.

Builder operations include `op`, `select`, `gather`, `affine`, `sparseAffine`, `scan`, `call`,
`oneHot`, `gatherOneHot`, `input`, `parameter`, `constant`, `state`, `update` and `include`. Arrays are
scalarized explicitly; this is not an arbitrary tensor-shape compiler.
`include` preserves region semantics and temperature, renames owned state and
parameters using a prefix, and checks the imported input contracts. Use
`program.freeze(names?)` to convert parameters to fixed constants. Use a trained
module's `artifact()` before freezing if you want its updated weights.

Tensor modules use native parameters, optimizers and state dictionaries. They
accept scalar floating tensors and return scalar tensors by output name. Compose
these tensors with existing encoders, transforms, decoders or other owned models
using their normal tensor APIs. The compiler does not turn a remote text model or
a tool side effect into differentiable arithmetic.

`ProgramOperation` from `tensorcode/program/operation` accepts an artifact as JSON
configuration, owns its module, and participates in `trace()` and
`Trainer.fromOps({operation})`, including normal trainer checkpoints. It rejects
state and external calls so replay remains pure. Use `operation.artifact()` to
export trained weights; `configuration()` identifies the original construction.

Explicit numerical calls use `builder.call(name, ...arguments)` and
`run(inputs, {bindings: {name: {forward, vjp}}})`. The VJP returns one derivative
per scalar argument. Differentiation through a call with arguments requires an
explicit VJP. Call bindings are supplied separately and are never serialized.
They must be pure numerical functions when used for differentiation. Tensor
integration captures derivatives during forward and never re-invokes a callback
in backward. No higher-order autograd is promised.

## State and events

```ts
import { Builder, Runtime, int32 } from 'tensorcode/program';
const b = new Builder();
const event = b.input('event', int32({ min: 0, max: 1 }));
const state = b.state('count', 0, int32({ min: 0, max: 100 }));
b.update(state, state.add(event));
const runtime = new Runtime(b.build(state), 100);
runtime.dispatch({ event: 1 }); // output uses old state
const saved = runtime.snapshot();
const resumed = Runtime.restore(saved);
```

All updates read old state and commit simultaneously. A failing update leaves the
runtime's state and step count unchanged. Snapshots include the program, state,
step count and budget and are portable to Python. `replay(events)` dispatches an
explicit sequence; it does not infer event meaning or execute host effects.
For manual ownership use `program.step(inputs, {state})`, which returns
`{outputs, state}` without mutating the supplied state.

## Backends, inspection and verification

- `Program` uses an iterative, lazy CPU interpreter and reverse-mode tape.
- `toWasm()` emits real WebAssembly f64/f32 scalar circuits with memoized lazy
  branches. It supports exact/hybrid forward execution, arithmetic, comparisons,
  gathers, state and checks. Host calls, sigmoid and relaxed regions are rejected.
  Gradients use the shared reference semantics. WASM does not require a GPU.
- `WebGPUProgram` emits WGSL for stateless float32 programs. Supply a WebGPU device
  to `runBatch(device, cases, {atol, rtol})`. Every GPU result is checked against the
  CPU within those explicit tolerances; this validation currently adds CPU work.
  WebGPU does not promise portable bit-exact arithmetic. Integer domains must fit
  exact float32 integers. No host calls or mutable state are accepted here.
- `optimize()` folds valid constant expressions and eliminates dead circuits. It
  preserves input signatures, parameters and state. It packs multiply/add chains into ordered affine kernels while retaining
  each multiplication/addition rounding point. `sparseAffine` stores only supplied
  coefficient/index pairs, in ascending index order; no dense matrix is required.
  Floating arithmetic is not reassociated. This is scalar kernel fusion, not SIMD
  vectorization or a performance claim without a workload benchmark.
- `schedule()` returns dependency layers; `analyze(program)` reports intervals,
  depth and potential invalid operations. Diagnostics are conservative guidance,
  not a proof that every input is safe; runtime checks remain authoritative.
- `verify(reference)` enumerates finite integer domains up to a configurable
  budget. Continuous domains require explicit `cases`, whose result is marked
  non-exhaustive. Empty cases and over-budget exhaustive checks are errors.
- `verifiedReplacement(reference, candidate)` accepts only stateless, frozen,
  call-free exact programs with matching signatures and exhaustive finite-domain
  agreement. It supplies a gate for externally proposed structural changes; it
  does not itself train a synthesizer or prove behavior on continuous domains.

## Practical limits

`graph()` returns inspectable IR data, not a graphical renderer. Verification
compares executions against a supplied reference; finite cases do not prove a
continuous domain or every possible stateful event sequence. Interval analysis
is diagnostic rather than a proof of equivalence.

Event state belongs to `Runtime`. Tensor modules do not carry event state from
one forward call to the next, and this integration does not provide training
through an event history. ProgramOperation deliberately accepts only stateless,
call-free programs.

| Situation | What to do |
| --- | --- |
| A function cannot be inspected or a closure is rejected | Use self-contained source text or Builder; pass inputs explicitly |
| Verification has no reference | Supply an independent function when loading an artifact or compiling text |
| A fractional value is rejected by int32 | Use a floating domain if fractions are intended |
| Loop expansion exceeds the budget | Reduce the bound or explicitly configure the expansion and node limits |
| Hybrid gradients fail although forward succeeds | Check both branch values; the surrogate derivative may need the inactive branch |
| A tensor has more than one element | Select scalar values explicitly; this API is not a general tensor compiler |

## Artifact version 1

The root has `format: "tensorcode.program"`, `version: 1`, `semantics`, `dtype`,
`surrogate`, `temperature`, `nodes`, `outputs` and `updates`. Each node has a
contiguous topological `id`, opcode `op`, preceding IDs in `args`, scalar
`shape: []`, checked `domain`, `differentiability`, and source location (or null).
Bindings add `name`, literals/initial values add `value`; included nodes can
specify their own `semantics`, `surrogate` and `temperature`. Outputs map names to node IDs;
updates map state-node IDs to next-value IDs. There is no executable source in
an artifact. Loading validates IDs, arity, domains, semantics and state targets.

The CPU interpreter and artifact loader work in a browser. The
`tensorcode/program/runtime` entry exports the core Program, Builder and domain
APIs without the TypeScript parser. Import the event `Runtime` class from the
full entry instead. The complete
`tensorcode/program` entry includes the source compiler. No filesystem API is
needed to run a program. Serialize with `JSON.stringify(program)` and reload
with `Program.fromJSON(json)`; Python uses `Program.from_json(json)`.

Validation: `npm run typecheck`, `npm run build`, and
`npx vitest run test/program test/training test/nn`. A real Chromium check is
`node scripts/program/browser-smoke.mjs` (`CHROME` can override the executable).
The browser test exercises CPU, WASM, state, and WebGPU when an adapter is present.
It uses a software graphics adapter, so it validates execution rather than hardware
performance. It needs Chrome and a Node runtime with global WebSocket (tested on
Node 24); absent WebGPU adapters are reported as unavailable.
