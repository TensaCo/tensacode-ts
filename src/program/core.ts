import { ProgramModule } from './tensor.js';
import { WasmProgram } from './wasm.js';
/** Data-only scalar SSA programs. All numeric operations have explicit rounding points. */
export type Semantics = 'exact' | 'relaxed' | 'hybrid';
export interface Domain { dtype: 'float64' | 'float32' | 'int32' | 'bool'; min: number; max: number }
export const float64 = (bounds: Partial<Pick<Domain, 'min' | 'max'>> = {}): Domain => ({ dtype: 'float64', min: -Number.MAX_VALUE, max: Number.MAX_VALUE, ...bounds });
export const float32 = (bounds: Partial<Pick<Domain, 'min' | 'max'>> = {}): Domain => ({ dtype: 'float32', min: -3.4028234663852886e38, max: 3.4028234663852886e38, ...bounds });
export const int32 = (bounds: Partial<Pick<Domain, 'min' | 'max'>> = {}): Domain => ({ dtype: 'int32', min: -2147483648, max: 2147483647, ...bounds });
export const boolean = (): Domain => ({ dtype: 'bool', min: 0, max: 1 });
export interface Node { id: number; op: string; args: number[]; shape: number[]; domain: Domain; differentiability: string; semantics?: Semantics; temperature?: number; surrogate?: 'sigmoid' | null; source: { language: string; line: number; column: number } | null; name?: string; value?: number }
export interface Artifact { format: 'tensorcode.program'; version: 1; semantics: Semantics; dtype: 'float64' | 'float32'; surrogate: 'sigmoid' | null; temperature: number; nodes: Node[]; outputs: Record<string, number>; updates: Record<string, number> }
export interface Options { semantics?: Semantics; dtype?: 'float64' | 'float32'; surrogate?: 'sigmoid' | null; temperature?: number; maxNodes?: number }
export interface Binding { forward: (...args: number[]) => number; vjp?: (args: number[], result: number) => number[] }
export type ReferenceResult = number | boolean | Record<string, number | boolean> | readonly (number | boolean)[];
export interface RunOptions { state?: Record<string, number>; parameters?: Record<string, number>; bindings?: Record<string, Binding> }
const arities: Record<string, number> = { add: 2, sub: 2, mul: 2, div: 2, neg: 1, relu: 1, sigmoid: 1, gt: 2, ge: 2, lt: 2, le: 2, eq: 2, ne: 2, select: 3, check: 1 };
const own = (x: object, k: string): boolean => Object.hasOwn(x, k);
export function checkDomain(d: Domain): void {
  if (!d || Object.keys(d).sort().join(',') !== 'dtype,max,min' || !['float64', 'float32', 'int32', 'bool'].includes(d.dtype) || !Number.isFinite(d.min) || !Number.isFinite(d.max) || d.min > d.max) throw new Error('invalid domain');
  if (d.dtype === 'int32' || d.dtype === 'bool') {
    const [lo, hi] = d.dtype === 'bool' ? [0, 1] : [-2147483648, 2147483647];
    if (!Number.isInteger(d.min) || !Number.isInteger(d.max) || d.min < lo! || d.max > hi!) throw new Error('invalid integer domain');
  }
}
export function checked(x: number, d: Domain): number {
  if (typeof x !== 'number' || !Number.isFinite(x) || x < d.min || x > d.max) throw new Error('value outside declared finite domain');
  if ((d.dtype === 'int32' || d.dtype === 'bool') && !Number.isInteger(x)) throw new Error('expected integer');
  const y = d.dtype === 'float32' ? Math.fround(x) : x;
  if (!Number.isFinite(y) || y < d.min || y > d.max) throw new Error('rounding outside domain');
  return y;
}
export class Value {
  constructor(readonly builder: Builder, readonly id: number) {}
  add(x: Operand): Value { return this.builder.op('add', this, x); }
  sub(x: Operand): Value { return this.builder.op('sub', this, x); }
  mul(x: Operand): Value { return this.builder.op('mul', this, x); }
  div(x: Operand): Value { return this.builder.op('div', this, x); }
  gt(x: Operand): Value { return this.builder.op('gt', this, x); }
  ge(x: Operand): Value { return this.builder.op('ge', this, x); }
  lt(x: Operand): Value { return this.builder.op('lt', this, x); }
  le(x: Operand): Value { return this.builder.op('le', this, x); }
  eq(x: Operand): Value { return this.builder.op('eq', this, x); }
  relu(): Value { return this.builder.op('relu', this); }
}
export type Operand = Value | number;
export class Builder {
  readonly ir: Artifact;
  readonly maxNodes: number;
  source: Node['source'] = null;
  constructor(options: Options = {}) {
    const { semantics = 'exact', dtype = 'float64', surrogate = null, temperature = 1, maxNodes = 100000 } = options;
    if (!['exact', 'relaxed', 'hybrid'].includes(semantics) || !['float32', 'float64'].includes(dtype)) throw new Error('invalid semantics/dtype');
    if (semantics === 'hybrid' && surrogate !== 'sigmoid') throw new Error('hybrid requires explicit sigmoid surrogate');
    if ((surrogate !== null && surrogate !== 'sigmoid') || !Number.isFinite(temperature) || temperature <= 0) throw new Error('invalid gradient policy');
    this.maxNodes = maxNodes;
    this.ir = { format: 'tensorcode.program', version: 1, semantics, dtype, surrogate, temperature, nodes: [], outputs: {}, updates: {} };
  }
  private node(op: string, args: number[] = [], attrs: Partial<Node> = {}): Value {
    if (this.ir.nodes.length >= this.maxNodes) throw new Error('program node budget exceeded');
    const id = this.ir.nodes.length;
    this.ir.nodes.push({ id, op, args, shape: [], domain: this.ir.dtype === 'float64' ? float64() : float32(), differentiability: ['gt', 'ge', 'lt', 'le', 'eq', 'ne', 'gather', 'onehot'].includes(op) ? 'discrete' : 'piecewise', source: this.source, ...attrs });
    return new Value(this, id);
  }
  value(x: Operand): Value {
    if (x instanceof Value) { if (x.builder !== this) throw new Error('foreign builder value'); return x; }
    return this.constant(x);
  }
  constant(x: number): Value { const domain = this.ir.dtype === 'float64' ? float64() : float32(); return this.node('const', [], { value: checked(x, domain), domain }); }
  private named(op: string, name: string, domain: Domain, value?: number): Value {
    checkDomain(domain);
    if (!name || this.ir.nodes.some(n => n.name === name)) throw new Error('duplicate/empty binding name');
    return this.node(op, [], { name, domain: structuredClone(domain), ...(value === undefined ? {} : { value: checked(value, domain) }) });
  }
  input(name: string, domain = float64()): Value { return this.named('input', name, domain); }
  parameter(name: string, value: number, domain = float64()): Value {
    if (!['float32', 'float64'].includes(domain.dtype)) throw new Error('parameters must be floating point');
    return this.named('parameter', name, domain, value);
  }
  state(name: string, initial: number, domain = float64()): Value { return this.named('state', name, domain, initial); }
  update(state: Value, value: Operand): void {
    const target = this.value(state);
    if (this.ir.nodes[target.id]!.op !== 'state') throw new Error('update target must be state');
    this.ir.updates[String(target.id)] = this.value(value).id;
  }
  op(op: string, ...args: Operand[]): Value {
    if (!own(arities, op) || arities[op] !== args.length) throw new Error('unsupported operation or arity');
    const attrs = ['gt', 'ge', 'lt', 'le', 'eq', 'ne'].includes(op) && this.ir.semantics !== 'relaxed' ? { domain: boolean() } : {};
    return this.node(op, args.map(x => this.value(x).id), attrs);
  }
  select(condition: Operand, yes: Operand, no: Operand): Value { return this.op('select', condition, yes, no); }
  gather(values: Operand[], index: Operand): Value {
    if (!values.length) throw new Error('empty gather');
    return this.node('gather', [this.value(index).id, ...values.map(x => this.value(x).id)]);
  }
  oneHot(index: Operand, size: number): Value[] {
    if (!Number.isInteger(size) || size < 1 || size > this.maxNodes) throw new Error('invalid one-hot size');
    const checkedIndex = this.node('check', [this.value(index).id], { domain: int32({ min: 0, max: size - 1 }) });
    return Array.from({ length: size }, (_, i) => this.op('eq', checkedIndex, i));
  }
  gatherOneHot(values: Operand[], selectors: Operand[]): Value {
    if (!values.length || values.length !== selectors.length) throw new Error('one-hot shape mismatch');
    const index = this.node('onehot', selectors.map(x => this.value(x).id), { domain: int32({ min: 0, max: values.length - 1 }) });
    return this.gather(values, index);
  }
  call(name: string, ...args: Operand[]): Value {
    if (!name) throw new Error('call requires binding name');
    return this.node('call', args.map(x => this.value(x).id), { name });
  }
  affine(values: Operand[], weights: Operand[], bias: Operand = 0): Value {
    if (values.length !== weights.length) throw new Error('affine shape mismatch');
    return this.node('affine', [this.value(bias).id, ...values.flatMap((x, i) => [this.value(x).id, this.value(weights[i]!).id])]);
  }
  sparseAffine(values: Operand[], coefficients: Record<number, Operand>, bias: Operand = 0): Value {
    const indices = Object.keys(coefficients).map(Number).sort((a, b) => a - b);
    if (indices.some(i => !Number.isInteger(i) || i < 0 || i >= values.length) || new Set(indices).size !== indices.length) throw new Error('invalid sparse affine index');
    return this.affine(indices.map(i => values[i]!), indices.map(i => coefficients[i]!), bias);
  }
  scan(values: Operand[], initial: Operand = 0): Value[] {
    let total = this.value(initial); return values.map(x => { total = total.add(x); return total; });
  }
  include(program: Program, inputs: Record<string, Operand>, prefix = 'region'): Record<string, Value> {
    const ir = program.graph();
    if (ir.dtype !== this.ir.dtype) throw new Error('composition dtype mismatch');
    const expected = ir.nodes.filter(n => n.op === 'input').map(n => n.name!);
    if (Object.keys(inputs).length !== expected.length || expected.some(k => !Object.hasOwn(inputs, k))) throw new Error('composition signature mismatch');
    const mapping = new Map<number, Value>();
    for (const n of ir.nodes) {
      let value: Value;
      if (n.op === 'input') value = this.node('check', [this.value(inputs[n.name!]!).id], { domain: n.domain, source: n.source });
      else {
        const { id: _id, op, args, ...attrs } = n;
        if (op === 'parameter' || op === 'state') { attrs.name = `${prefix}.${n.name}`; if (this.ir.nodes.some(x => x.name === attrs.name)) throw new Error('duplicate composition binding'); }
        attrs.semantics = n.semantics ?? ir.semantics; attrs.temperature = n.temperature ?? ir.temperature; attrs.surrogate = n.surrogate === undefined ? ir.surrogate : n.surrogate;
        value = this.node(op, args.map(a => mapping.get(a)!.id), attrs);
      }
      mapping.set(n.id, value);
    }
    for (const [k, v] of Object.entries(ir.updates)) this.update(mapping.get(Number(k))!, mapping.get(v)!);
    return Object.fromEntries(Object.entries(ir.outputs).map(([k, v]) => [k, mapping.get(v)!]));
  }
  build(outputs: Operand | Record<string, Operand>): Program {
    if (typeof outputs === 'number' || outputs instanceof Value) outputs = { result: outputs };
    this.ir.outputs = Object.fromEntries(Object.entries(outputs).map(([k, v]) => [k, this.value(v).id]));
    return new Program(this.ir);
  }
}
export function validate(ir: Artifact): void {
  if (!ir || ir.format !== 'tensorcode.program' || ir.version !== 1) throw new Error('unsupported program artifact');
  if (Object.keys(ir).sort().join(',') !== 'dtype,format,nodes,outputs,semantics,surrogate,temperature,updates,version') throw new Error('unknown or missing artifact fields');
  new Builder(ir);
  if (!Array.isArray(ir.nodes) || ir.nodes.length > 100000) throw new Error('invalid node budget');
  const names = new Set<string>();
  ir.nodes.forEach((n, i) => {
    if (!n || Object.keys(n).some(k => !['id', 'op', 'args', 'shape', 'domain', 'differentiability', 'source', 'name', 'value', 'semantics', 'temperature', 'surrogate'].includes(k))) throw new Error('unknown node fields');
    if (['id', 'op', 'args', 'shape', 'domain', 'differentiability', 'source'].some(k => !Object.hasOwn(n, k))) throw new Error('missing node fields');
    if (!['discrete', 'piecewise'].includes(n.differentiability)) throw new Error('invalid differentiability metadata');
    if (n.op === 'call' && (typeof n.name !== 'string' || !n.name)) throw new Error('invalid call binding');
    if (n.id !== i || !Array.isArray(n.shape) || n.shape.length || !Array.isArray(n.args) || n.args.some(a => !Number.isInteger(a) || a < 0 || a >= i)) throw new Error('invalid scalar SSA node');
    checkDomain(n.domain);
    if (!['exact', 'relaxed', 'hybrid'].includes(n.semantics ?? ir.semantics)) throw new Error('invalid node semantics');
    const policy = n.surrogate === undefined ? ir.surrogate : n.surrogate;
    if (policy !== null && policy !== 'sigmoid' || (n.semantics ?? ir.semantics) === 'hybrid' && policy !== 'sigmoid') throw new Error('hybrid nodes require explicit sigmoid surrogate');
    if (!Number.isFinite(n.temperature ?? ir.temperature) || (n.temperature ?? ir.temperature) <= 0) throw new Error('invalid node temperature');
    const arity = own(arities, n.op) ? arities[n.op] : ['input', 'state', 'parameter', 'const'].includes(n.op) ? 0 : undefined;
    if (arity !== undefined ? n.args.length !== arity : !(n.op === 'call' || n.op === 'gather' && n.args.length >= 2 || n.op === 'onehot' && n.args.length >= 1 || n.op === 'affine' && n.args.length >= 1 && n.args.length % 2 === 1)) throw new Error('invalid operation/arity');
    if (['input', 'state', 'parameter'].includes(n.op)) {
      if (typeof n.name !== 'string' || !n.name || names.has(n.name)) throw new Error('duplicate binding');
      names.add(n.name);
    }
    if (['const', 'state', 'parameter'].includes(n.op)) checked(n.value!, n.domain);
    if (n.op === 'affine' && !['float32', 'float64'].includes(n.domain.dtype)) throw new Error('affine requires a floating computation dtype');
    if (n.op === 'parameter' && !['float32', 'float64'].includes(n.domain.dtype)) throw new Error('integer parameter');
  });
  for (const m of [ir.outputs, ir.updates]) {
    if (!m || typeof m !== 'object' || Array.isArray(m)) throw new Error('invalid outputs/updates');
    for (const v of Object.values(m)) if (!Number.isInteger(v) || v < 0 || v >= ir.nodes.length) throw new Error('invalid result reference');
  }
  if (!Object.keys(ir.outputs).length) throw new Error('program requires outputs');
  for (const k of Object.keys(ir.updates)) if (!/^\d+$/.test(k) || ir.nodes[Number(k)]?.op !== 'state') throw new Error('invalid state update');
}
const sigmoid = (x: number): number => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
export function validateBindings(ir: Artifact, inputs: Record<string, number>, options: RunOptions): void {
  for (const [mapping, kind, required] of [[inputs, 'input', true], [options.state ?? {}, 'state', false], [options.parameters ?? {}, 'parameter', false]] as const) {
    const allowed = new Map(ir.nodes.filter(n => n.op === kind).map(n => [n.name!, n]));
    if (!mapping || typeof mapping !== 'object' || Array.isArray(mapping) || Object.keys(mapping).some(k => !allowed.has(k)) || required && Object.keys(mapping).length !== allowed.size) throw new Error(`${kind} signature mismatch`);
    for (const [name, value] of Object.entries(mapping)) checked(value, allowed.get(name)!.domain);
  }
}
export class Program {
  private readonly ir: Artifact;
  private backend: 'reference' | 'wasm' = 'reference';
  private wasmEngine?: WasmProgram;
  private reference?: (inputs: Record<string, number>) => ReferenceResult;
  constructor(artifact: Artifact) { validate(artifact); this.ir = structuredClone(artifact); }
  withBackend(backend: 'reference' | 'wasm'): this {
    if (backend !== 'reference' && backend !== 'wasm') throw new Error('unsupported backend');
    this.wasmEngine = backend === 'wasm' ? new WasmProgram(this) : undefined; this.backend = backend; return this;
  }
  withReference(reference: (inputs: Record<string, number>) => ReferenceResult): this { this.reference = reference; return this; }
  freeze(names?: string[]): Program {
    const ir = this.graph(), available = new Set(ir.nodes.filter(n => n.op === 'parameter').map(n => n.name!)), selected = new Set(names ?? available);
    for (const name of selected) if (!available.has(name)) throw new Error('unknown parameter');
    for (const n of ir.nodes) if (n.op === 'parameter' && selected.has(n.name!)) { n.op = 'const'; delete n.name; }
    return new Program(ir);
  }
  toTensorModule(): ProgramModule { return new ProgramModule(this); }
  toWasm(): WasmProgram { return new WasmProgram(this); }
  graph(): Artifact { return structuredClone(this.ir); }
  toJSON(): Artifact { return this.graph(); }
  static fromJSON(value: string | Artifact): Program { return new Program(typeof value === 'string' ? JSON.parse(value) as Artifact : value); }
  private execute(inputs: Record<string, number>, options: RunOptions = {}, differentiation = false) {
    const { state = {}, parameters = {}, bindings = {} } = options;
    const ir = this.ir, cache = new Map<number, number>(), edges = new Map<number, [number, number][]>(), order: number[] = [];
    validateBindings(ir, inputs, options);
    const expected = ir.nodes.filter(n => n.op === 'input');
    if (Object.keys(inputs).length !== expected.length || expected.some(n => !own(inputs, n.name!))) throw new Error('input signature mismatch');
    for (const n of expected) checked(inputs[n.name!]!, n.domain);
    const calculate = (i: number): number => {
      if (cache.has(i)) return cache.get(i)!;
      const n = ir.nodes[i]!, ids = n.args, op = n.op, mode = n.semantics ?? ir.semantics;
      let y: number; let deriv: [number, number][] = [];
      if (op === 'input') y = inputs[n.name!]!;
      else if (op === 'const') y = n.value!;
      else if (op === 'parameter') y = own(parameters, n.name!) ? parameters[n.name!]! : n.value!;
      else if (op === 'state') y = own(state, n.name!) ? state[n.name!]! : n.value!;
      else if (op === 'select') {
        const c = ev(ids[0]!);
        if (mode === 'relaxed') { if (c < 0 || c > 1) throw new Error('relaxed select requires a gate in [0,1]'); const a = ev(ids[1]!), b = ev(ids[2]!); y = c * a + (1 - c) * b; deriv = [[ids[0]!, a - b], [ids[1]!, c], [ids[2]!, 1 - c]]; }
        else {
          if (c !== 0 && c !== 1) throw new Error('exact select requires boolean condition');
          const chosen = ids[c ? 1 : 2]!; y = ev(chosen); deriv = [[chosen, 1]];
          if (mode === 'hybrid' && differentiation) deriv.push([ids[0]!, ev(ids[1]!) - ev(ids[2]!)]);
        }
      } else if (op === 'gather') {
        const idx = ev(ids[0]!);
        if (!Number.isInteger(idx) || idx < 0 || idx >= ids.length - 1) throw new Error('gather index out of range');
        const chosen = ids[idx + 1]!; y = ev(chosen); deriv = [[chosen, 1]];
      } else {
        const a = ids.map(ev), x = a[0]!, z = a[1]!;
        const ds = (values: number[]) => { deriv = ids.map((id, j) => [id, values[j]!]); };
        switch (op) {
          case 'call': { const binding = own(bindings, n.name!) ? bindings[n.name!] : undefined; if (!binding) throw new Error(`unbound explicit call: ${n.name}`); y = binding.forward(...a); if (binding.vjp && differentiation) { const g = binding.vjp(a, y); if (g.length !== a.length) throw new Error('custom VJP arity'); ds(g); } break; }
          case 'affine': { y = x; const g = [1], domain = n.domain.dtype === 'float64' ? float64() : float32(); for (let j = 1; j < a.length; j += 2) { y = checked(y + checked(a[j]! * a[j + 1]!, domain), domain); g.push(a[j + 1]!, a[j]!); } ds(g); break; }
          case 'onehot': if (a.some(x => x !== 0 && x !== 1) || a.reduce((s, x) => s + x, 0) !== 1) throw new Error('expected exactly one active one-hot selector'); y = a.indexOf(1); break;
          case 'add': y = x + z; ds([1, 1]); break;
          case 'sub': y = x - z; ds([1, -1]); break;
          case 'mul': y = x * z; ds([z, x]); break;
          case 'div': if (z === 0) throw new Error('division by zero'); y = x / z; ds([1 / z, -y / z]); break;
          case 'check': y = x; ds([1]); break;
          case 'neg': y = -x; ds([-1]); break;
          case 'relu': y = Math.max(0, x); ds([x > 0 ? 1 : 0]); break;
          case 'sigmoid': y = sigmoid(x); ds([y * (1 - y)]); break;
          default: {
            const hard = op === 'gt' ? x > z : op === 'ge' ? x >= z : op === 'lt' ? x < z : op === 'le' ? x <= z : op === 'eq' ? x === z : x !== z;
            y = Number(hard);
            if (mode !== 'exact') {
              const t = n.temperature ?? ir.temperature, delta = (x - z) / t; let soft: number, g: number;
              if (op === 'eq' || op === 'ne') { soft = Math.exp(-delta * delta); g = Number.isFinite(delta) ? -2 * delta * soft / t : 0; if (op === 'ne') { soft = 1 - soft; g = -g; } }
              else { const sign = op === 'lt' || op === 'le' ? -1 : 1; soft = sigmoid(sign * delta); g = sign * soft * (1 - soft) / t; }
              if (mode === 'relaxed') y = soft;
              ds([g, -g]);
            }
          }
        }
      }
      y = checked(y, n.domain); cache.set(i, y); edges.set(i, deriv); order.push(i); return y;
    };
    const ev = (root: number): number => {
      const pending = [root];
      while (pending.length) {
        const i = pending.at(-1)!;
        if (cache.has(i)) { pending.pop(); continue; }
        const n = ir.nodes[i]!, ids = n.args; let deps = ids;
        if (n.op === 'select' || n.op === 'gather') {
          if (!cache.has(ids[0]!)) { pending.push(ids[0]!); continue; }
          const c = cache.get(ids[0]!)!;
          if (n.op === 'select' && ((n.semantics ?? ir.semantics) === 'exact' || (n.semantics ?? ir.semantics) === 'hybrid' && !differentiation)) {
            if (c !== 0 && c !== 1) throw new Error('exact select requires boolean condition');
            deps = [ids[c ? 1 : 2]!];
          } else if (n.op === 'gather') {
            if (!Number.isInteger(c) || c < 0 || c >= ids.length - 1) throw new Error('gather index out of range');
            deps = [ids[c + 1]!];
          }
        }
        const missing = deps.find(a => !cache.has(a));
        if (missing !== undefined) { pending.push(missing); continue; }
        calculate(i); pending.pop();
      }
      return cache.get(root)!;
    };
    const outputs = Object.fromEntries(Object.entries(ir.outputs).map(([k, v]) => [k, ev(v)]));
    const nextState = Object.fromEntries(ir.nodes.filter(n => n.op === 'state').map(n => [n.name!, checked(own(state, n.name!) ? state[n.name!]! : n.value!, n.domain)]));
    for (const [k, v] of Object.entries(ir.updates)) { const n = ir.nodes[Number(k)]!; nextState[n.name!] = checked(ev(v), n.domain); }
    return { outputs, state: nextState, edges, order };
  }
  run(inputs: Record<string, number>, options: RunOptions = {}): Record<string, number> { return this.wasmEngine ? this.wasmEngine.run(inputs, options) : this.execute(inputs, options).outputs; }
  step(inputs: Record<string, number>, options: RunOptions = {}): { outputs: Record<string, number>; state: Record<string, number> } { if (this.wasmEngine) return this.wasmEngine.step(inputs, options); const { outputs, state } = this.execute(inputs, options); return { outputs, state }; }
  gradients(inputs: Record<string, number>, options: RunOptions & { output?: string } = {}): Record<string, number> {
    const output = options.output ?? 'result';
    if (!own(this.ir.outputs, output)) throw new Error('unknown output');
    return this.linearize(inputs, options).gradients[output]!;
  }
  linearize(inputs: Record<string, number>, options: RunOptions = {}): { outputs: Record<string, number>; gradients: Record<string, Record<string, number>> } {
    const { outputs, edges, order } = this.execute(inputs, options, true);
    const gradients = Object.fromEntries(Object.entries(this.ir.outputs).map(([name, output]) => {
      const adj = new Map<number, number>([[output, 1]]);
      for (const i of order.toReversed()) {
        if (!adj.get(i)) continue;
        const n = this.ir.nodes[i]!;
        if (n.op === 'call' && n.args.length && !edges.get(i)!.length) throw new Error('differentiation through a call requires an explicit VJP');
        for (const [j, d] of edges.get(i)!) { const value = (adj.get(j) ?? 0) + adj.get(i)! * d; if (!Number.isFinite(value)) throw new Error('nonfinite gradient under selected policy'); adj.set(j, value); }
      }
      return [name, Object.fromEntries(this.ir.nodes.filter(n => ['input', 'parameter', 'state'].includes(n.op)).map(n => [n.name!, adj.get(n.id) ?? 0]))];
    }));
    return { outputs, gradients };
  }
  verify(reference: ((inputs: Record<string, number>) => ReferenceResult) | undefined = undefined, options: { cases?: Iterable<Record<string, number>>; maxCases?: number; atol?: number; rtol?: number } = {}) {
    reference ??= this.reference;
    if (!reference) throw new Error('provide an independent reference function');
    const maxCases = options.maxCases ?? 100000; let cases = options.cases;
    const exhaustive = cases === undefined;
    if (!cases) {
      const ins = this.ir.nodes.filter(n => n.op === 'input'); let total = 1;
      for (const n of ins) { if (!['int32', 'bool'].includes(n.domain.dtype)) throw new Error('provide cases for continuous domains'); total *= n.domain.max - n.domain.min + 1; if (total > maxCases) throw new Error('exhaustive verification exceeds budget'); }
      function* product(i: number, row: Record<string, number>): Generator<Record<string, number>> { if (i === ins.length) { yield { ...row }; return; } const n = ins[i]!; for (let x = n.domain.min; x <= n.domain.max; x++) { row[n.name!] = x; yield* product(i + 1, row); } }
      cases = product(0, {});
    }
    let count = 0;
    for (const row of cases) {
      if (++count > maxCases) throw new Error('verification exceeds budget');
      const actual = this.run(row), ref = reference(row);
      const raw = typeof ref === 'number' || typeof ref === 'boolean' ? { result: ref } : ref;
      if (!raw || typeof raw !== 'object') throw new Error('reference must return scalar or named scalar outputs');
      const expected = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, typeof v === 'boolean' ? Number(v) : v]));
      if (Object.keys(expected).length !== Object.keys(actual).length || Object.keys(actual).some(k => !own(expected, k) || !Number.isFinite(expected[k]) || Math.abs(actual[k]! - expected[k]!) > (options.atol ?? 0) + (options.rtol ?? 0) * Math.max(Math.abs(actual[k]!), Math.abs(expected[k]!)))) return { passed: false, cases: count, exhaustive, counterexample: row, actual, expected };
    }
    if (!count) throw new Error('verification requires cases');
    return { passed: true, cases: count, exhaustive };
  }
  schedule(): number[][] {
    const levels: number[] = [], layers: number[][] = [];
    for (const n of this.ir.nodes) { const level = 1 + Math.max(-1, ...n.args.map(a => levels[a]!)); levels.push(level); (layers[level] ??= []).push(n.id); }
    return layers;
  }
  optimize(): Program {
    const ir = this.graph(), effects: boolean[] = [];
    for (const n of ir.nodes) effects.push(n.op === 'call' || n.args.some(a => effects[a]));
    for (const n of ir.nodes) {
      if (n.op === 'add' && !effects[n.id]) {
        const acc = ir.nodes[n.args[0]!]!, product = ir.nodes[n.args[1]!]!, full = n.domain.dtype === 'float64' ? float64() : float32();
        const same = (d: Domain) => d.dtype === full.dtype && d.min === full.min && d.max === full.max;
        if (product.op === 'mul' && same(n.domain) && same(product.domain)) {
          n.op = 'affine'; n.args = [acc.id, ...product.args];
          if (acc.op === 'affine' && same(acc.domain)) n.args = [...acc.args, ...product.args];
        }
      }
      if (['input', 'state', 'parameter', 'const', 'call'].includes(n.op) || !n.args.every(a => ir.nodes[a]!.op === 'const')) continue;
      const constants = n.args.map((a, id) => ({ ...structuredClone(ir.nodes[a]!), id }));
      const target = { ...structuredClone(n), id: constants.length, args: constants.map(c => c.id) };
      try {
        const value = new Program({ ...ir, nodes: [...constants, target], outputs: { result: target.id }, updates: {} }).run({}).result!;
        n.op = 'const'; n.args = []; n.value = value;
      } catch { /* Preserve operations whose execution may fail on an active path. */ }
    }
    const live = new Set([...Object.values(ir.outputs), ...Object.values(ir.updates), ...ir.nodes.filter(n => ['input', 'state', 'parameter'].includes(n.op)).map(n => n.id)]);
    for (const n of ir.nodes.toReversed()) if (live.has(n.id)) for (const a of n.args) live.add(a);
    const mapping = new Map<number, number>(); const nodes: Node[] = [];
    for (const n of ir.nodes) if (live.has(n.id)) { mapping.set(n.id, nodes.length); n.id = nodes.length; n.args = n.args.map(a => mapping.get(a)!); nodes.push(n); }
    ir.nodes = nodes; ir.outputs = Object.fromEntries(Object.entries(ir.outputs).map(([k, v]) => [k, mapping.get(v)!])); ir.updates = Object.fromEntries(Object.entries(ir.updates).map(([k, v]) => [String(mapping.get(Number(k))!), mapping.get(v)!]));
    return new Program(ir);
  }
}
