# Algorithms across substrates

`tensorcode/algorithm` executes the same version-1 `tensorcode.algorithm` JSON
artifacts as Python. It includes a typed builder and three CPU execution targets:
ALG.0 reference execution without gradients, ALG.1 native autograd circuits, and
ALG.2 scheduled attention over operand registers.

```ts
import { AlgorithmBuilder, Algorithm, convert } from 'tensorcode/algorithm';

const b = new AlgorithmBuilder({ name: 'policy', dtype: 'float64' });
const q = b.input('position', [20]);
const target = b.input('target', [20]);
const gain = b.parameter('gain', 30);
const policy = b.build({ torque: target.sub(q).mul(gain).clip(-8, 8) });

const attention = policy.compile('alg2');
const result = attention.forward({ position: Array(20).fill(0), target: Array(20).fill(.1) });
console.log(result.torque.tolist());

const symbolic = convert(attention, 'alg0');
const restored = Algorithm.fromJSON(attention.artifact().toJSON());
```

`AlgorithmModule` extends the library's native `Module`: `parameters()` exposes
registered parameters to SGD/Adam and `forward` participates in autograd.
`artifact()` exports current weights. Stateful algorithms use
`step(inputs, {state})`, returning `{outputs, state}`. Updates are simultaneous
and input state is never mutated. Carry the returned tensors forward to retain
recurrent gradients.

Hard ALG.2 compiles instruction arguments into one-source attention masks and
executes declared local tensor operations after the attention read. Its QK context
is zero by default; optional register-aligned context features can be supplied.
This is a scheduled attention machine, not compilation of arbitrary programs into
a vanilla pretrained transformer. Soft routing uses finite penalties and is only
an approximation; converting it back to an exact nominal algorithm is rejected.
`manifest()` exposes the execution contract.

The runtime supports arithmetic, comparisons, conditional selection, matrix
multiplication, reshape/permutation, reductions, slices/concatenation, elementary
nonlinearities and masked softmax. It validates data-only artifacts, shape and
input signatures, finite values and attention masks. Conditional selection is
eager: both branches must be numerically valid.

Python supplies the source frontend, pretrained structural-bias adapters, supported
native-module import, standalone source export and the MuJoCo experiment. Those
features are not claimed as TypeScript APIs. TypeScript tests execute the actual
Python robotics artifact through all three substrates and compare a Python-imported
attention graph with Python reference outputs.

See the [Python algorithm guide](https://tensorcode.dev/docs/python/algorithm/)
for the algorithm format, conversion contracts and the robotics example. These are
current-checkout features, not a claim about the latest published package release.


## Task prerequisite composition

`compileSchedule` from `tensorcode/algorithm` accepts a mapping from task names
to prerequisite name arrays. Indices follow `Object.keys` order. Its artifact
accepts `completed` and `enabled` vectors and outputs `ready`: enabled, unfinished
tasks whose dependencies are complete. Values >= 0.5 mean true. Dispatch and
completion verification remain caller responsibilities. Cycles, unknown or
duplicate dependencies and schedules outside 1–256 tasks are rejected.

```typescript
import { compileSchedule } from "tensorcode/algorithm";
const schedule = compileSchedule({ inspect: [], pick: ["inspect"], release: ["pick"] });
const output = schedule.compile("alg2").forward({
  completed: [1, 0, 0], enabled: [1, 1, 1]
}); // output.ready: [0, 1, 0]
```

The portable format is shared with Python. Regression tests replay the Python
kitting example's real decision trace through all three TypeScript substrates.
