/** Batched WGSL execution with explicit numerical verification against the CPU.
 * WebGPU floating point is not a portable bit-exact substitute for IEEE CPU execution.
 */
import { Program, checked, type Artifact, type Node } from './core.js';
interface Buffer { getMappedRange(): ArrayBuffer; mapAsync(mode: number): Promise<void>; unmap(): void; destroy(): void }
interface Pass { setPipeline(p: unknown): void; setBindGroup(i: number, group: unknown): void; dispatchWorkgroups(n: number): void; end(): void }
interface Encoder { beginComputePass(): Pass; copyBufferToBuffer(a: Buffer, ao: number, b: Buffer, bo: number, size: number): void; finish(): unknown }
export interface GPUDeviceLike {
  createShaderModule(descriptor: { code: string }): unknown;
  createComputePipelineAsync(descriptor: unknown): Promise<{ getBindGroupLayout(i: number): unknown }>;
  createBuffer(descriptor: { size: number; usage: number }): Buffer;
  createBindGroup(descriptor: unknown): unknown;
  createCommandEncoder(): Encoder;
  queue: { writeBuffer(buffer: Buffer, offset: number, data: Float32Array): void; submit(commands: unknown[]): void };
}
function code(ir: Artifact): string {
  const inputs = ir.nodes.filter(n => n.op === 'input');
  const literal = (x: number): string => { const v = Math.fround(x); if (!Number.isFinite(v)) throw new Error('constant is not representable as float32'); return `${v === 0 ? (Object.is(v, -0) ? '-0.0' : '0.0') : String(v).includes('.') || String(v).includes('e') ? String(v) : `${v}.0`}f`; };
  const functions = ir.nodes.map(n => {
    const a = n.args.map(i => `n${i}(row)`), mode = n.semantics ?? ir.semantics;
    let expression: string, prelude = '';
    if (n.op === 'input') expression = `inputs[row * ${inputs.length}u + ${inputs.indexOf(n)}u]`;
    else if (n.op === 'const' || n.op === 'parameter') expression = literal(n.value!);
    else if (n.op === 'affine') { prelude = `var accumulator = ${a[0]};`; for (let j = 1; j < a.length; j += 2) prelude += ` let product${j} = ${a[j]} * ${a[j + 1]}; accumulator = accumulator + product${j};`; expression = 'accumulator'; }
    else if (n.op === 'onehot') { prelude = a.map(x => `if (${x} != 0.0f && ${x} != 1.0f) { invalid = true; }`).join(' ') + ` if ((${a.join(' + ')}) != 1.0f) { invalid = true; }`; expression = a.map((x, i) => `(${x} * ${literal(i)})`).join(' + '); }
    else if (n.op === 'check') expression = a[0]!;
    else if (['add', 'sub', 'mul', 'div'].includes(n.op)) expression = `(${a[0]} ${{ add: '+', sub: '-', mul: '*', div: '/' }[n.op]} ${a[1]})`;
    else if (n.op === 'neg') expression = `(-${a[0]})`;
    else if (n.op === 'relu') expression = `max(0.0f, ${a[0]})`;
    else if (n.op === 'sigmoid') expression = `1.0f / (1.0f + exp(-${a[0]}))`;
    else if (n.op === 'select') {
      if (mode === 'relaxed') expression = `(${a[0]} * ${a[1]} + (1.0f - ${a[0]}) * ${a[2]})`;
      else { prelude = `let condition = ${a[0]}; if (condition != 0.0f && condition != 1.0f) { invalid = true; } var chosen: f32; if (condition == 1.0f) { chosen = ${a[1]}; } else { chosen = ${a[2]}; }`; expression = 'chosen'; }
    } else if (n.op === 'gather') {
      prelude = `let index = ${a[0]}; var chosen: f32 = 0.0f; switch (i32(index)) { ${a.slice(1).map((x, i) => `case ${i}: { chosen = ${x}; }`).join(' ')} default: { invalid = true; } } if (index != floor(index) || index < 0.0f || index >= ${literal(a.length - 1)}) { invalid = true; }`; expression = 'chosen';
    } else {
      const op = { gt: '>', ge: '>=', lt: '<', le: '<=', eq: '==', ne: '!=' }[n.op];
      if (!op) throw new Error(`unsupported GPU operation: ${n.op}`);
      if (mode !== 'relaxed') expression = `select(0.0f, 1.0f, ${a[0]} ${op} ${a[1]})`;
      else {
        const delta = `((${a[0]} - ${a[1]}) / ${literal(n.temperature ?? ir.temperature)})`;
        expression = n.op === 'eq' ? `exp(-${delta} * ${delta})` : n.op === 'ne' ? `(1.0f - exp(-${delta} * ${delta}))` : `(1.0f / (1.0f + exp(${['lt', 'le'].includes(n.op) ? '' : '-'}${delta})))`;
      }
    }
    const lo = Math.max(n.domain.min, -3.4028234663852886e38), hi = Math.min(n.domain.max, 3.4028234663852886e38);
    return `fn n${n.id}(row: u32) -> f32 { if (ready[${n.id}]) { return cache[${n.id}]; } ${prelude} let value = ${expression}; if (!(value >= ${literal(lo)} && value <= ${literal(hi)})${['int32', 'bool'].includes(n.domain.dtype) ? ' || value != floor(value)' : ''}) { invalid = true; } cache[${n.id}] = value; ready[${n.id}] = true; return value; }`;
  }).join('\n');
  const outs = Object.values(ir.outputs);
  return `@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
var<private> cache: array<f32, ${ir.nodes.length}>;
var<private> ready: array<bool, ${ir.nodes.length}>;
var<private> invalid: bool;
${functions}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 let row = id.x; if (row >= arrayLength(&outputs) / ${outs.length + 1}u) { return; }
 ${outs.map((n, i) => `outputs[row * ${outs.length + 1}u + ${i}u] = n${n}(row);`).join('\n')}
 outputs[row * ${outs.length + 1}u + ${outs.length}u] = select(0.0f, 1.0f, invalid);
}`;
}
export class WebGPUProgram {
  readonly wgsl: string;
  private readonly ir: Artifact;
  constructor(readonly program: Program) {
    this.ir = program.graph();
    if (this.ir.dtype !== 'float32' || this.ir.nodes.some(n => n.domain.dtype === 'float64' || n.op === 'call' || n.op === 'state')) throw new Error('WebGPU requires a stateless float32 program without host calls');
    if (this.ir.nodes.some(n => n.domain.dtype === 'int32' && (n.domain.min < -16777216 || n.domain.max > 16777216))) throw new Error('WebGPU integer inputs must be exactly representable as float32');
    this.wgsl = code(this.ir);
  }
  /** Every GPU result is checked against the reference within caller-declared tolerances. */
  async runBatch(device: GPUDeviceLike, cases: Record<string, number>[], tolerance: { atol: number; rtol: number }): Promise<Record<string, number>[]> {
    if (![tolerance.atol, tolerance.rtol].every(x => Number.isFinite(x) && x >= 0)) throw new Error('explicit finite nonnegative GPU tolerances required');
    if (!cases.length) return [];
    const ins = this.ir.nodes.filter(n => n.op === 'input'), names = Object.keys(this.ir.outputs);
    const references = cases.map(row => this.program.run(row));
    const input = new Float32Array(Math.max(1, cases.length * ins.length));
    cases.forEach((row, i) => ins.forEach((n, j) => { input[i * ins.length + j] = checked(row[n.name!]!, n.domain); }));
    const size = cases.length * (names.length + 1) * 4;
    const ib = device.createBuffer({ size: input.byteLength, usage: 128 | 8 });
    const ob = device.createBuffer({ size, usage: 128 | 4 });
    const readback = device.createBuffer({ size, usage: 1 | 8 });
    try {
      device.queue.writeBuffer(ib, 0, input);
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: this.wgsl }), entryPoint: 'main' } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ib } }, { binding: 1, resource: { buffer: ob } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(cases.length / 64)); pass.end(); encoder.copyBufferToBuffer(ob, 0, readback, 0, size); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(1);
      const values = new Float32Array(readback.getMappedRange());
      return cases.map((_row, i) => {
        if (values[i * (names.length + 1) + names.length] !== 0) throw new Error(`GPU domain violation in case ${i}`);
        return Object.fromEntries(names.map((name, j) => {
          const actual = values[i * (names.length + 1) + j]!, expected = references[i]![name]!;
          if (!Number.isFinite(actual) || Math.abs(actual - expected) > tolerance.atol + tolerance.rtol * Math.max(Math.abs(actual), Math.abs(expected))) throw new Error(`GPU numerical verification failed in case ${i}, output ${name}`);
          return [name, actual];
        }));
      });
    } finally { readback.unmap(); ib.destroy(); ob.destroy(); readback.destroy(); }
  }
}
