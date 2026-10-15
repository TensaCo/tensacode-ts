/** Compile scalar circuits directly to WebAssembly f64/f32 instructions.
 * Lazy memoized node functions preserve selection and gather semantics.
 * There is no dynamic JavaScript evaluation or external wasm toolchain.
 */
import { Program, checked, validateBindings, type RunOptions } from './core.js';
interface WasmInstance { exports: Record<string, unknown> }
interface WasmMemory { buffer: ArrayBuffer }
const wasm = (globalThis as unknown as { WebAssembly: { Module: new (bytes: Uint8Array) => unknown; Instance: new (module: unknown) => WasmInstance } }).WebAssembly;
const u = (n: number): number[] => { const out: number[] = []; do { const x = n & 127; n >>>= 7; out.push(x | (n ? 128 : 0)); } while (n); return out; };
const str = (s: string): number[] => { const bytes = [...new TextEncoder().encode(s)]; return [...u(bytes.length), ...bytes]; };
const vec = (rows: number[][]): number[] => [...u(rows.length), ...rows.flat()];
const section = (id: number, bytes: number[]): number[] => [id, ...u(bytes.length), ...bytes];
const f64 = (x: number): number[] => { const b = new ArrayBuffer(8); new DataView(b).setFloat64(0, x, true); return [0x44, ...new Uint8Array(b)]; };
const i32 = (x: number): number[] => { const out: number[] = []; let more = true; while (more) { const byte = x & 127; x >>= 7; more = !((x === 0 && !(byte & 64)) || (x === -1 && !!(byte & 64))); out.push(byte | (more ? 128 : 0)); } return [0x41, ...out]; };
const call = (id: number): number[] => [0x10, ...u(id)];
const load = (offset: number): number[] => [...i32(offset), 0x2b, 3, 0];
const store = (offset: number, value: number[]): number[] => [...i32(offset), ...value, 0x39, 3, 0];
export class WasmProgram {
  readonly bytes: Uint8Array;
  private readonly instance: WasmInstance;
  private readonly memory: WasmMemory;
  constructor(readonly program: Program) {
    const ir = program.graph();
    if (ir.semantics === 'relaxed' || ir.nodes.some(n => n.semantics === 'relaxed')) throw new Error('WasmProgram currently requires exact forward semantics');
    if (ir.nodes.some(n => ['call', 'sigmoid'].includes(n.op))) throw new Error('WasmProgram does not lower host calls or sigmoid');
    const binary: Record<string, number> = { add: 0xa0, sub: 0xa1, mul: 0xa2, div: 0xa3, gt: 0x64, ge: 0x66, lt: 0x63, le: 0x65, eq: 0x61, ne: 0x62 };
    const functions = ir.nodes.map(n => {
      const args = n.args, get = (j: number) => call(args[j]!);
      let body: number[];
      if (['input', 'parameter', 'state', 'const'].includes(n.op)) body = load(n.id * 16);
      else if (n.op === 'select') {
        body = [...get(0), ...f64(0), 0x62, 0x04, 0x7c, ...get(1), 0x05, ...get(2), 0x0b];
      } else if (n.op === 'gather') {
        // Index validity is checked after execution, before returning any result.
        let choose: number[] = [0x00];
        for (let j = args.length - 1; j >= 1; j--) choose = [...get(0), ...f64(j - 1), 0x61, 0x04, 0x7c, ...get(j), 0x05, ...choose, 0x0b];
        body = choose;
      } else if (n.op === 'affine') {
        const rounding = n.domain.dtype === 'float32' ? [0xb6, 0xbb] : [];
        body = get(0);
        for (let j = 1; j < args.length; j += 2) body.push(...get(j), ...get(j + 1), 0xa2, ...rounding, 0xa0, ...rounding);
      } else if (n.op === 'onehot') {
        body = f64(0);
        for (let j = 0; j < args.length; j++) body.push(...get(j), ...f64(j), 0xa2, 0xa0);
      } else if (n.op === 'check') body = get(0);
      else if (n.op === 'neg') body = [...get(0), 0x9a];
      else if (n.op === 'relu') body = [...get(0), ...f64(0), 0xa5];
      else if (Object.hasOwn(binary, n.op)) body = [...get(0), ...get(1), binary[n.op]!, ...(['gt', 'ge', 'lt', 'le', 'eq', 'ne'].includes(n.op) ? [0xb7] : [])];
      else throw new Error(`unsupported wasm operation: ${n.op}`);
      if (n.domain.dtype === 'float32') body.push(0xb6, 0xbb);
      // Cache an evaluated node. Every cached intermediate is validated by run().
      const code = [0, ...i32(n.id * 16 + 8), 0x28, 2, 0, 0x04, 0x7c, ...load(n.id * 16), 0x05,
        ...store(n.id * 16, body), ...i32(n.id * 16 + 8), ...i32(1), 0x36, 2, 0, ...load(n.id * 16), 0x0b, 0x0b];
      return [...u(code.length), ...code];
    });
    const exports = [...ir.nodes.map(n => [...str(`n${n.id}`), 0, ...u(n.id)]), [...str('memory'), 2, 0]];
    this.bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0,
      ...section(1, vec([[0x60, 0, 1, 0x7c]])), ...section(3, vec(ir.nodes.map(() => [0]))),
      ...section(5, vec([[0, ...u(Math.max(1, Math.ceil(ir.nodes.length * 16 / 65536)))]])),
      ...section(7, vec(exports)), ...section(10, vec(functions))]);
    this.instance = new wasm.Instance(new wasm.Module(this.bytes));
    this.memory = this.instance.exports.memory as WasmMemory;
  }
  step(inputs: Record<string, number>, options: RunOptions = {}): { outputs: Record<string, number>; state: Record<string, number> } {
    const ir = this.program.graph(), data = new DataView(this.memory.buffer), state = options.state ?? {}, parameters = options.parameters ?? {};
    validateBindings(ir, inputs, options);
    const ins = ir.nodes.filter(n => n.op === 'input');
    if (Object.keys(inputs).length !== ins.length || ins.some(n => !Object.hasOwn(inputs, n.name!))) throw new Error('input signature mismatch');
    new Uint8Array(this.memory.buffer).fill(0);
    for (const n of ir.nodes) if (['input', 'const', 'parameter', 'state'].includes(n.op)) {
      const source = n.op === 'input' ? inputs : n.op === 'parameter' ? parameters : state;
      const value = n.op === 'const' ? n.value! : Object.hasOwn(source, n.name!) ? source[n.name!]! : n.value!;
      data.setFloat64(n.id * 16, checked(value, n.domain), true); data.setInt32(n.id * 16 + 8, 1, true);
    }
    const ev = (i: number): number => (this.instance.exports[`n${i}`] as () => number)();
    const outputs = Object.fromEntries(Object.entries(ir.outputs).map(([k, id]) => [k, ev(id)]));
    const next = Object.fromEntries(ir.nodes.filter(n => n.op === 'state').map(n => [n.name!, data.getFloat64(n.id * 16, true)]));
    for (const [k, id] of Object.entries(ir.updates)) { const n = ir.nodes[Number(k)]!; next[n.name!] = checked(ev(id), n.domain); }
    for (const n of ir.nodes) if (data.getInt32(n.id * 16 + 8, true)) {
      checked(data.getFloat64(n.id * 16, true), n.domain);
      if (n.op === 'onehot') { const a = n.args.map(i => data.getFloat64(i * 16, true)); if (a.some(x => x !== 0 && x !== 1) || a.reduce((s, x) => s + x, 0) !== 1) throw new Error('expected exactly one active one-hot selector'); }
      if (n.op === 'select') { const c = data.getFloat64(n.args[0]! * 16, true); if (c !== 0 && c !== 1) throw new Error('exact select requires boolean condition'); }
    }
    return { outputs, state: next };
  }
  run(inputs: Record<string, number>, options: RunOptions = {}): Record<string, number> { return this.step(inputs, options).outputs; }
}
