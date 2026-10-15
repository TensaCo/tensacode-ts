/** Portable typed algorithm artifacts and ALG.0/1/2 CPU execution. */
import { Module } from "../nn/module.js";
import {
  Tensor,
  Parameter,
  tensor,
  zeros,
  ones,
  type NestedNumbers,
  type DType,
} from "../nn/tensor.js";
import { cat, stack, where } from "../nn/functional.js";
import { noGrad } from "../nn/autograd.js";

export type Target = "alg0" | "alg1" | "alg2";
export interface Node {
  id: number;
  op: string;
  args: number[];
  shape: number[];
  attrs: Record<string, unknown>;
}
export interface AlgorithmArtifact {
  format: "tensorcode.algorithm";
  version: 1;
  name: string;
  dtype: "float32" | "float64";
  nodes: Node[];
  outputs: Record<string, number>;
  updates: Record<string, number>;
}
export type Inputs = Record<string, Tensor | NestedNumbers>;
export interface CompileOptions {
  routing?: "hard" | "soft";
  strength?: number;
  contextDim?: number;
}
export interface StepOptions {
  state?: Inputs;
  context?: Tensor;
}
const leaves = new Set(["input", "constant", "parameter", "state"]);
const unary = new Set([
  "neg",
  "relu",
  "sigmoid",
  "tanh",
  "sin",
  "cos",
  "exp",
  "sqrt",
  "log",
  "abs",
]);
const binary = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "gt",
  "ge",
  "lt",
  "le",
  "eq",
  "ne",
  "maximum",
  "minimum",
  "matmul",
]);
const attributes: Record<string, string[]> = {
  transpose: ["axes"],
  reshape: ["shape"],
  sum: ["axis", "keepdims"],
  mean: ["axis", "keepdims"],
  softmax: ["axis"],
  masked_softmax: ["axis"],
  slice: ["axis", "start", "stop"],
  concat: ["axis"],
};
const arities: Record<string, number> = {
  where: 3,
  transpose: 1,
  reshape: 1,
  sum: 1,
  mean: 1,
  softmax: 1,
  masked_softmax: 2,
  slice: 1,
};
const name = (x: unknown): string => {
  if (
    typeof x !== "string" ||
    ["__proto__", "constructor", "prototype"].includes(x) ||
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(x)
  )
    throw new Error("invalid name");
  return x;
};
const eq = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((x, i) => x === b[i]);
const keys = (a: object, b: readonly string[]): boolean =>
  Object.keys(a).sort().join(",") === [...b].sort().join(",");
function shape(s: unknown): number[] {
  if (
    !Array.isArray(s) ||
    s.length > 6 ||
    s.some((x) => !Number.isInteger(x) || x < 1) ||
    s.reduce((a, b) => a * b, 1) > 1_000_000
  )
    throw new Error("invalid shape");
  return s as number[];
}
function axis(a: unknown, rank: number): number {
  if (typeof a !== "number" || !Number.isInteger(a) || a < -rank || a >= rank)
    throw new Error("axis out of range");
  return (a + rank) % rank;
}
function broadcast(shapes: readonly (readonly number[])[]): number[] {
  const rank = Math.max(...shapes.map((s) => s.length));
  const out = Array<number>(rank).fill(1);
  for (const s of shapes)
    for (let i = 0; i < s.length; i++) {
      const j = rank - s.length + i;
      if (out[j] !== 1 && s[i] !== 1 && out[j] !== s[i])
        throw new Error("broadcast mismatch");
      out[j] = Math.max(out[j]!, s[i]!);
    }
  return shape(out);
}
function infer(
  op: string,
  ss: number[][],
  a: Record<string, unknown>,
): number[] {
  const arity = unary.has(op) ? 1 : binary.has(op) ? 2 : arities[op];
  if (
    (op !== "concat" && arity === undefined) ||
    (arity !== undefined && ss.length !== arity) ||
    !keys(a, attributes[op] ?? [])
  )
    throw new Error("unsupported operation, arity or attributes");
  const s = ss[0]!;
  if (unary.has(op)) return [...s];
  if ((binary.has(op) && op !== "matmul") || op === "where")
    return broadcast(ss);
  if (op === "matmul") {
    const t = ss[1]!;
    if (s.length < 2 || t.length < 2 || s.at(-1) !== t.at(-2))
      throw new Error("matmul shape mismatch");
    return shape([
      ...broadcast([s.slice(0, -2), t.slice(0, -2)]),
      s.at(-2)!,
      t.at(-1)!,
    ]);
  }
  if (op === "reshape") {
    const target = shape(a.shape);
    if (target.reduce((x, y) => x * y, 1) !== s.reduce((x, y) => x * y, 1))
      throw new Error("reshape mismatch");
    return target;
  }
  if (op === "transpose") {
    const axes = a.axes;
    if (
      !Array.isArray(axes) ||
      !eq(
        [...axes].sort((x, y) => x - y),
        s.map((_, i) => i),
      )
    )
      throw new Error("invalid permutation");
    return axes.map((i) => s[i]!);
  }
  if (op === "sum" || op === "mean") {
    if (typeof a.keepdims !== "boolean") throw new Error("invalid keepdims");
    const dims =
      a.axis === null ? s.map((_, i) => i) : [axis(a.axis, s.length)];
    return s.flatMap((d, i) =>
      dims.includes(i) ? (a.keepdims ? [1] : []) : [d],
    );
  }
  if (op === "softmax" || op === "masked_softmax") {
    axis(a.axis, s.length);
    if (op === "masked_softmax" && !eq(broadcast(ss), s))
      throw new Error("mask shape mismatch");
    return [...s];
  }
  if (op === "slice") {
    const dim = axis(a.axis, s.length);
    const start = a.start as number,
      stop = a.stop as number;
    if (
      !Number.isInteger(start) ||
      !Number.isInteger(stop) ||
      start < 0 ||
      stop <= start ||
      stop > s[dim]!
    )
      throw new Error("invalid slice");
    const out = [...s];
    out[dim] = stop - start;
    return out;
  }
  if (op === "concat") {
    if (!ss.length) throw new Error("empty concat");
    const dim = axis(a.axis, s.length);
    const out = [...s];
    out[dim] = 0;
    for (const t of ss) {
      if (t.length !== s.length || t.some((v, i) => i !== dim && v !== s[i]))
        throw new Error("concat mismatch");
      out[dim]! += t[dim]!;
    }
    return shape(out);
  }
  throw new Error(op);
}
function operation(
  op: string,
  xs: Tensor[],
  a: Record<string, unknown>,
): Tensor {
  const x = xs[0]!,
    y = xs[1]!;
  switch (op) {
    case "add":
      return x.add(y);
    case "sub":
      return x.sub(y);
    case "mul":
      return x.mul(y);
    case "div":
      return x.div(y);
    case "gt":
      return x.gt(y);
    case "ge":
      return x.ge(y);
    case "lt":
      return x.lt(y);
    case "le":
      return x.le(y);
    case "eq":
      return x.eq(y);
    case "ne":
      return x.ne(y);
    case "maximum":
      return x.maximum(y);
    case "minimum":
      return x.minimum(y);
    case "matmul":
      return x.matmul(y);
    case "neg":
      return x.neg();
    case "relu":
      return x.relu();
    case "sigmoid":
      return x.sigmoid();
    case "tanh":
      return x.tanh();
    case "sin":
      return x.sin();
    case "cos":
      return x.cos();
    case "exp":
      return x.exp();
    case "sqrt":
      return x.sqrt();
    case "log":
      return x.log();
    case "abs":
      return x.abs();
    case "where":
      return where(x.ne(0), y, xs[2]!);
    case "reshape":
      return x.reshape(a.shape as number[]);
    case "transpose":
      return x.permute(a.axes as number[]);
    case "sum":
      return x.sum(a.axis as number | null, a.keepdims as boolean);
    case "mean":
      return x.mean(a.axis as number | null, a.keepdims as boolean);
    case "concat":
      return cat(xs, a.axis as number);
    case "slice":
      return x.slice(a.axis as number, a.start as number, a.stop as number);
    case "softmax":
      return x.softmax(a.axis as number);
    case "masked_softmax": {
      if (Array.from(y.data).some((v) => v !== 0 && v !== 1))
        throw new Error("mask must be binary");
      const mask = y.mul(ones(x.shape, { dtype: x.dtype }));
      if (Array.from(mask.sum(a.axis as number).data).some((v) => v === 0))
        throw new Error("empty attention row");
      return x.maskedFill(mask.eq(0), -Infinity).softmax(a.axis as number);
    }
    default:
      throw new Error(op);
  }
}

export class Algorithm {
  private readonly ir: AlgorithmArtifact;
  constructor(artifact: AlgorithmArtifact) {
    const ir = structuredClone(artifact);
    if (
      !ir ||
      !keys(ir, [
        "format",
        "version",
        "name",
        "dtype",
        "nodes",
        "outputs",
        "updates",
      ]) ||
      ir.format !== "tensorcode.algorithm" ||
      ir.version !== 1 ||
      !["float32", "float64"].includes(ir.dtype)
    )
      throw new Error("invalid algorithm artifact");
    name(ir.name);
    if (!Array.isArray(ir.nodes) || !ir.nodes.length || ir.nodes.length > 10000)
      throw new Error("invalid node count");
    const names = new Set<string>(),
      ss: number[][] = [],
      states = new Map<string, number[]>();
    for (const [i, n] of ir.nodes.entries()) {
      if (
        !keys(n, ["id", "op", "args", "shape", "attrs"]) ||
        n.id !== i ||
        !Array.isArray(n.args) ||
        n.args.some((j) => !Number.isInteger(j) || j < 0 || j >= i) ||
        !n.attrs ||
        Array.isArray(n.attrs)
      )
        throw new Error("invalid node");
      const s = shape(n.shape),
        a = n.attrs;
      if (leaves.has(n.op)) {
        if (
          n.args.length ||
          !keys(
            a,
            n.op === "input"
              ? ["name"]
              : n.op === "constant"
                ? ["value"]
                : ["name", "value"],
          )
        )
          throw new Error("invalid leaf");
        if (a.name !== undefined) {
          const key = name(a.name);
          if (names.has(key)) throw new Error("duplicate name");
          names.add(key);
        }
        if (n.op !== "input") {
          const v = tensor(a.value as NestedNumbers, { dtype: ir.dtype });
          if (!eq(v.shape, s) || !v.allFinite())
            throw new Error("invalid leaf value");
        }
        if (n.op === "state") states.set(a.name as string, s);
      } else if (
        !eq(
          infer(
            n.op,
            n.args.map((j) => ss[j]!),
            a,
          ),
          s,
        )
      )
        throw new Error("inferred shape mismatch");
      ss.push(s);
    }
    if (
      ss.reduce((total, s) => total + s.reduce((x, y) => x * y, 1), 0) >
        32_000_000 ||
      ir.nodes.reduce((total, n) => total + n.args.length, 0) > 1_000_000
    )
      throw new Error("algorithm aggregate budget exceeded");
    if (
      !ir.outputs ||
      Array.isArray(ir.outputs) ||
      !Object.keys(ir.outputs).length ||
      !ir.updates ||
      Array.isArray(ir.updates)
    )
      throw new Error("invalid outputs");
    for (const map of [ir.outputs, ir.updates])
      for (const [key, i] of Object.entries(map)) {
        name(key);
        if (!Number.isInteger(i) || i < 0 || i >= ss.length)
          throw new Error("invalid output reference");
      }
    for (const [key, i] of Object.entries(ir.updates))
      if (!states.has(key) || !eq(states.get(key)!, ss[i]!))
        throw new Error("invalid state update");
    this.ir = ir;
  }
  graph(): AlgorithmArtifact {
    return structuredClone(this.ir);
  }
  toJSON(): string {
    return JSON.stringify(this.ir);
  }
  static fromJSON(text: string): Algorithm {
    return new Algorithm(JSON.parse(text) as AlgorithmArtifact);
  }
  get inputNames(): string[] {
    return this.ir.nodes
      .filter((n) => n.op === "input")
      .map((n) => n.attrs.name as string);
  }
  get stateNames(): string[] {
    return this.ir.nodes
      .filter((n) => n.op === "state")
      .map((n) => n.attrs.name as string);
  }
  compile(
    target: Target = "alg1",
    options: CompileOptions = {},
  ): AlgorithmModule {
    return new AlgorithmModule(this.ir, target, options);
  }
  run(inputs: Inputs): Record<string, NestedNumbers> {
    if (this.stateNames.length)
      throw new Error("use step for stateful algorithms");
    return this.step(inputs).outputs;
  }
  step(
    inputs: Inputs,
    options: StepOptions = {},
  ): {
    outputs: Record<string, NestedNumbers>;
    state: Record<string, NestedNumbers>;
  } {
    const result = this.compile("alg0").step(inputs, options);
    return {
      outputs: Object.fromEntries(
        Object.entries(result.outputs).map(([k, v]) => [k, v.tolist()]),
      ),
      state: Object.fromEntries(
        Object.entries(result.state).map(([k, v]) => [k, v.tolist()]),
      ),
    };
  }
}

export class AlgorithmModule extends Module {
  static override readonly qualifiedName: string =
    "tensorcode.algorithm.AlgorithmModule";
  readonly algorithm: Algorithm;
  private runtimeDtype: "float32" | "float64";
  readonly target: Target;
  readonly routing: "hard" | "soft";
  readonly strength: number;
  readonly contextDim: number;
  private readonly weights = new Map<number, Tensor>();
  constructor(
    artifact: AlgorithmArtifact,
    target: Target = "alg1",
    options: CompileOptions = {},
  ) {
    super();
    this.algorithm = new Algorithm(artifact);
    this.runtimeDtype = artifact.dtype;
    this.target = target;
    if (!["alg0", "alg1", "alg2"].includes(target))
      throw new Error("unknown substrate");
    if (
      Object.keys(options).some(
        (k) => !["routing", "strength", "contextDim"].includes(k),
      ) ||
      (target !== "alg2" && Object.keys(options).length)
    )
      throw new Error("invalid compiler options");
    this.routing = options.routing ?? "hard";
    this.strength = options.strength ?? 12;
    this.contextDim = options.contextDim ?? 16;
    if (
      !["hard", "soft"].includes(this.routing) ||
      !Number.isFinite(this.strength) ||
      this.strength < 0 ||
      !Number.isInteger(this.contextDim) ||
      this.contextDim < 1 ||
      this.contextDim > 4096
    )
      throw new Error("invalid routing config");
    for (const n of artifact.nodes)
      if (n.op === "parameter") {
        const p = new Parameter(
          tensor(n.attrs.value as NestedNumbers, { dtype: artifact.dtype }),
        );
        this.weights.set(n.id, p);
        this.registerParameter(`p${n.id}`, p);
      } else if (n.op === "constant" || n.op === "state") {
        this.registerBuffer(
          `c${n.id}`,
          tensor(n.attrs.value as NestedNumbers, { dtype: artifact.dtype }),
        );
      }
  }
  override to(dtype: DType): this {
    if (dtype !== "float32" && dtype !== "float64")
      throw new Error("algorithm modules require float32 or float64");
    super.to(dtype);
    this.runtimeDtype = dtype;
    return this;
  }
  forward(
    inputs: Inputs,
    options: { context?: Tensor } = {},
  ): Record<string, Tensor> {
    if (this.algorithm.stateNames.length)
      throw new Error("use step for stateful algorithms");
    return this.step(inputs, options).outputs;
  }
  step(
    inputs: Inputs,
    options: StepOptions = {},
  ): { outputs: Record<string, Tensor>; state: Record<string, Tensor> } {
    const execute = () => this.execute(inputs, options);
    return this.target === "alg0" ? noGrad(execute) : execute();
  }
  private execute(
    inputs: Inputs,
    options: StepOptions,
  ): { outputs: Record<string, Tensor>; state: Record<string, Tensor> } {
    const ir = this.algorithm.graph();
    ir.dtype = this.runtimeDtype;
    if (
      !keys(inputs, this.algorithm.inputNames) ||
      (options.state && !keys(options.state, this.algorithm.stateNames))
    )
      throw new Error("input/state signature mismatch");
    const val = (x: Tensor | NestedNumbers): Tensor =>
      x instanceof Tensor ? x.to(ir.dtype) : tensor(x, { dtype: ir.dtype });
    const values: Tensor[] = [];
    for (const n of ir.nodes) {
      let v: Tensor;
      const a = n.attrs;
      if (n.op === "input") v = val(inputs[a.name as string]!);
      else if (n.op === "state")
        v =
          options.state?.[a.name as string] !== undefined
            ? val(options.state[a.name as string]!)
            : this.getBuffer(`c${n.id}`)!;
      else if (n.op === "parameter") v = this.weights.get(n.id)!;
      else if (n.op === "constant") v = this.getBuffer(`c${n.id}`)!;
      else {
        let operands = n.args.map((i) => values[i]!);
        if (this.target === "alg2") {
          const width = Math.max(...values.map((x) => x.numel));
          if (width * values.length > 16_000_000)
            throw new Error("attention memory budget exceeded");
          const memory = stack(
            values.map((x) =>
              x.numel === width
                ? x.reshape(width)
                : cat(
                    [
                      x.reshape(x.numel),
                      zeros([width - x.numel], { dtype: ir.dtype }),
                    ],
                    0,
                  ),
            ),
            0,
          );
          let logits: Tensor;
          if (options.context) {
            const c = options.context.to(ir.dtype);
            if (
              !eq(c.shape, [ir.nodes.length, this.contextDim]) ||
              !c.allFinite()
            )
              throw new Error("invalid context");
            const q = c.slice(0, n.id, n.id + 1);
            logits = cat(
              n.args.map(() => q),
              0,
            )
              .matmul(c.slice(0, 0, values.length).transpose(0, 1))
              .div(Math.sqrt(this.contextDim));
          } else
            logits = zeros([n.args.length, values.length], { dtype: ir.dtype });
          const bias = tensor(
            n.args.map((source) =>
              values.map((_, i) =>
                i === source
                  ? 0
                  : this.routing === "hard"
                    ? -Infinity
                    : -this.strength,
              ),
            ),
            { dtype: ir.dtype },
          );
          const reads = logits.add(bias).softmax(-1).matmul(memory);
          operands = n.args.map((source, i) =>
            reads
              .slice(0, i, i + 1)
              .reshape(width)
              .slice(0, 0, values[source]!.numel)
              .reshape(values[source]!.shape),
          );
        }
        v = operation(n.op, operands, a).to(ir.dtype);
      }
      if (!eq(v.shape, n.shape) || !v.allFinite())
        throw new Error(`invalid shape or nonfinite node ${n.id}`);
      values.push(v);
    }
    return {
      outputs: Object.fromEntries(
        Object.entries(ir.outputs).map(([k, i]) => [k, values[i]!.clone()]),
      ),
      state: Object.fromEntries(
        ir.nodes
          .filter((n) => n.op === "state")
          .map((n) => [
            n.attrs.name as string,
            values[ir.updates[n.attrs.name as string] ?? n.id]!.clone(),
          ]),
      ),
    };
  }
  artifact(): Algorithm {
    if (this.target === "alg2" && this.routing !== "hard")
      throw new Error(
        "soft routing requires validated replacement; cannot export as exact algorithm",
      );
    const ir = this.algorithm.graph();
    ir.dtype = this.runtimeDtype;
    for (const n of ir.nodes)
      if (n.op === "parameter")
        n.attrs.value = this.weights.get(n.id)!.tolist();
      else if (n.op === "constant" || n.op === "state")
        n.attrs.value = this.getBuffer(`c${n.id}`)!.tolist();
    return new Algorithm(ir);
  }
  convert(target: Target, options: CompileOptions = {}): AlgorithmModule {
    return this.artifact().compile(target, options);
  }
  manifest(): Record<string, unknown> {
    return {
      target: this.target,
      contract:
        this.target === "alg0"
          ? "reference"
          : this.target === "alg1"
            ? "same_operations"
            : this.routing === "hard"
              ? "exact_operand_routing"
              : "structural_prior",
      local_operations: "supplied tensor instructions",
      vanilla_transformer: false,
    };
  }
}

export class Variable {
  constructor(
    readonly builder: AlgorithmBuilder,
    readonly id: number,
  ) {}
  get shape(): number[] {
    return [...this.builder.nodes[this.id]!.shape];
  }
  add(x: Variable | NestedNumbers): Variable {
    return this.builder.op("add", [this, x]);
  }
  sub(x: Variable | NestedNumbers): Variable {
    return this.builder.op("sub", [this, x]);
  }
  mul(x: Variable | NestedNumbers): Variable {
    return this.builder.op("mul", [this, x]);
  }
  div(x: Variable | NestedNumbers): Variable {
    return this.builder.op("div", [this, x]);
  }
  matmul(x: Variable | NestedNumbers): Variable {
    return this.builder.op("matmul", [this, x]);
  }
  tanh(): Variable {
    return this.builder.op("tanh", [this]);
  }
  relu(): Variable {
    return this.builder.op("relu", [this]);
  }
  reshape(...shape: number[]): Variable {
    return this.builder.op("reshape", [this], { shape });
  }
  transpose(...axes: number[]): Variable {
    return this.builder.op("transpose", [this], { axes });
  }
  sum(axis: number | null = null, keepdims = false): Variable {
    return this.builder.op("sum", [this], { axis, keepdims });
  }
  softmax(axis = -1): Variable {
    return this.builder.op("softmax", [this], { axis });
  }
  clip(lo: number, hi: number): Variable {
    return this.builder.op("minimum", [
      this.builder.op("maximum", [this, lo]),
      hi,
    ]);
  }
}
export class AlgorithmBuilder {
  readonly nodes: Node[] = [];
  private readonly updates: Record<string, number> = {};
  private readonly names = new Set<string>();
  constructor(
    readonly options: { name?: string; dtype?: "float32" | "float64" } = {},
  ) {}
  private leaf(
    op: string,
    key?: string,
    value?: NestedNumbers,
    s: number[] = [],
  ): Variable {
    const attrs: Record<string, unknown> = {};
    if (key !== undefined) {
      name(key);
      if (this.names.has(key)) throw new Error("duplicate name");
      this.names.add(key);
      attrs.name = key;
    }
    if (op !== "input") {
      const v = tensor(value!, { dtype: this.options.dtype ?? "float64" });
      if (!v.allFinite()) throw new Error("nonfinite leaf");
      s = [...v.shape];
      attrs.value = v.tolist();
    }
    const id = this.nodes.length;
    if (id >= 10000) throw new Error("node budget exceeded");
    this.nodes.push({ id, op, args: [], shape: shape(s), attrs });
    return new Variable(this, id);
  }
  input(key: string, s: number[] = []): Variable {
    return this.leaf("input", key, undefined, s);
  }
  constant(value: NestedNumbers): Variable {
    return this.leaf("constant", undefined, value);
  }
  parameter(key: string, value: NestedNumbers): Variable {
    return this.leaf("parameter", key, value);
  }
  state(key: string, value: NestedNumbers): Variable {
    return this.leaf("state", key, value);
  }
  value(v: Variable | NestedNumbers): Variable {
    if (v instanceof Variable) {
      if (v.builder !== this) throw new Error("foreign variable");
      return v;
    }
    return this.constant(v);
  }
  op(
    op: string,
    args: (Variable | NestedNumbers)[],
    attrs: Record<string, unknown> = {},
  ): Variable {
    const vs = args.map((x) => this.value(x));
    const id = this.nodes.length;
    if (id >= 10000) throw new Error("node budget exceeded");
    this.nodes.push({
      id,
      op,
      args: vs.map((v) => v.id),
      shape: infer(
        op,
        vs.map((v) => v.shape),
        attrs,
      ),
      attrs: structuredClone(attrs),
    });
    return new Variable(this, id);
  }
  update(s: Variable, v: Variable | NestedNumbers): void {
    this.value(s);
    const value = this.value(v);
    const n = this.nodes[s.id]!;
    if (n.op !== "state" || !eq(s.shape, value.shape))
      throw new Error("invalid state update");
    this.updates[n.attrs.name as string] = value.id;
  }
  build(outputs: Variable | Record<string, Variable>): Algorithm {
    const out = outputs instanceof Variable ? { result: outputs } : outputs;
    return new Algorithm({
      format: "tensorcode.algorithm",
      version: 1,
      name: this.options.name ?? "algorithm",
      dtype: this.options.dtype ?? "float64",
      nodes: this.nodes,
      outputs: Object.fromEntries(
        Object.entries(out).map(([k, v]) => [name(k), this.value(v).id]),
      ),
      updates: this.updates,
    });
  }
}
export function convert(
  source: Algorithm | AlgorithmModule,
  target: Target,
  options: CompileOptions = {},
): AlgorithmModule {
  return (source instanceof Algorithm ? source : source.artifact()).compile(
    target,
    options,
  );
}
