import { describe, it, expect } from 'vitest';
import { Builder, Program, compile, int32, float64, ProgramModule } from '../../src/program/index.js';
import { tensor } from '../../src/nn/tensor.js';

describe('portable numerical programs', () => {
  it('exhaustively verifies bounded branching and rejects fractional integers', () => {
    const p = compile((a: number) => a > 4 ? 10 : 12, { inputs: { a: int32({ min: -128, max: 127 }) } });
    expect(p.verify(({ a }) => a! > 4 ? 10 : 12)).toEqual({ passed: true, cases: 256, exhaustive: true });
    expect(() => p.run({ a: 4.5 })).toThrow();
    const q = compile((a: number) => a > 4 ? 10 : 12, { inputs: { a: float64() } });
    expect(Program.fromJSON(JSON.stringify(q)).run({ a: 4.5 })).toEqual({ result: 10 });
  });
  it('lowers loops and early-return control flow', () => {
    const p = compile(`(x) => { let s = 0; for (let i = 0; i < 5; i++) { s += x + i; } if (s > 20) return s; return -s; }`, { inputs: { x: int32({ min: -10, max: 10 }) } });
    expect(p.verify(({ x }) => 5 * x! + 10 > 20 ? 5 * x! + 10 : -(5 * x! + 10)).passed).toBe(true);
  });
  it('evaluates only the selected exact branch and gathers', () => {
    const b = new Builder(), x = b.input('x'); const p = b.build(b.select(x.gt(0), b.op('div', 1, x), 7));
    expect(p.run({ x: 0 })).toEqual({ result: 7 });
    const c = new Builder(), i = c.input('i', int32({ min: 0, max: 2 }));
    expect(c.build(c.gather([3, 5, 8], i)).run({ i: 2 })).toEqual({ result: 8 });
  });
  it('updates state simultaneously', () => {
    const b = new Builder(), a = b.state('a', 1), z = b.state('z', 2); b.update(a, z); b.update(z, a);
    const state = { a: 3, z: 8 };
    expect(b.build(a.add(z)).step({}, { state })).toEqual({ outputs: { result: 11 }, state: { a: 8, z: 3 } });
    expect(state).toEqual({ a: 3, z: 8 });
  });
  it('composes exact and surrogate programs with native autograd', () => {
    for (const semantics of ['exact', 'relaxed', 'hybrid'] as const) {
      const b = new Builder({ semantics, surrogate: semantics === 'hybrid' ? 'sigmoid' : null });
      const x = b.input('x'), w = b.parameter('weight', 2), p = b.build(b.select(x.gt(0), w.mul(x), 0));
      const m = new ProgramModule(p), input = tensor(1, { dtype: 'float64', requiresGrad: true });
      m.forward({ x: input }).result!.backward();
      expect(m.parameters()[0]!.grad!.item()).toBeGreaterThan(0);
      expect(input.grad!.item()).toBeGreaterThan(0);
      expect(m.artifact().run({ x: 1 })).toEqual(p.run({ x: 1 }));
    }
  });
  it('rejects malformed artifacts, unbound calls and unbounded source', () => {
    const b = new Builder(), x = b.input('x'), p = b.build(x.add(1));
    const ir = p.graph(); ir.nodes.at(-1)!.args[0] = 999;
    expect(() => new Program(ir)).toThrow();
    expect(() => compile('(x) => { while (x) x--; return x; }', { inputs: { x: float64() } })).toThrow();
    expect(() => b.build(b.call('unknown', x)).run({ x: 1 })).toThrow();
  });
});

it('executes actual WebAssembly with identical branch and state semantics', async () => {
  const { WasmProgram } = await import('../../src/program/wasm.js');
  const p = compile('(a) => a > 4 ? 10 : 12', { inputs: { a: int32({ min: -128, max: 127 }) } });
  const wasm = new WasmProgram(p);
  expect(Array.from(wasm.bytes.slice(0, 4))).toEqual([0, 97, 115, 109]);
  for (let a = -128; a <= 127; a++) expect(wasm.run({ a })).toEqual(p.run({ a }));
  const b = new Builder(), x = b.input('x'), s = b.state('s', 0); b.update(s, s.add(x));
  const q = b.build(b.select(x.gt(0), b.op('div', 1, x), s));
  expect(new WasmProgram(q).step({ x: 0 })).toEqual(q.step({ x: 0 }));
});

it('composes frozen program regions, optimizes, and resumes bounded state', async () => {
  const { Runtime, analyze } = await import('../../src/program/index.js');
  const b = new Builder(), a = b.input('a'), weight = b.parameter('weight', 2);
  const region = b.build(weight.mul(a)).freeze();
  const outer = new Builder({ semantics: 'hybrid', surrogate: 'sigmoid' }), x = outer.input('x'), correction = outer.parameter('correction', 0.5);
  const imported = outer.include(region, { a: x });
  const p = outer.build(imported.result!.add(correction.mul(x)));
  expect(p.run({ x: 4 })).toEqual({ result: 10 });
  expect(p.gradients({ x: 4 })).toEqual({ x: 2.5, correction: 4 });
  expect(p.optimize().run({ x: 4 })).toEqual(p.run({ x: 4 }));
  expect(analyze(p).nodeCount).toBeGreaterThan(0);
  const sb = new Builder(), s = sb.state('s', 0, int32({ min: 0, max: 2 })); sb.update(s, s.add(1));
  const r = new Runtime(sb.build(s), 10); r.dispatch({});
  const restored = Runtime.restore(r.snapshot()); restored.dispatch({});
  expect(() => restored.dispatch({})).toThrow();
  expect(restored.state).toEqual({ s: 2 }); expect(restored.steps).toBe(2);
});

it('keeps hybrid forward exact even when the inactive branch is undefined', () => {
  const b = new Builder({ semantics: 'hybrid', surrogate: 'sigmoid' }), x = b.input('x');
  const p = b.build(b.select(x.gt(0), b.op('div', 1, x), 7));
  expect(p.run({ x: 0 })).toEqual({ result: 7 });
  expect(p.toWasm().run({ x: 0 })).toEqual({ result: 7 });
  expect(() => p.gradients({ x: 0 })).toThrow('division by zero');
});

it('trains a compiled operation through trace replay and the existing Trainer', async () => {
  const { ProgramOperation } = await import('../../src/program/operation.js');
  const { Trainer } = await import('../../src/training/index.js');
  const { trace } = await import('../../src/_internal/tracing.js');
  const b = new Builder(), x = b.input('x'), w = b.parameter('weight', 0);
  const op = new ProgramOperation(b.build(w.mul(x)).graph()), trainer = Trainer.fromOps({ op }, { lr: 0.1 });
  const session = trace();
  const output = session.run(() => op.call({ x: tensor(1, { dtype: 'float64' }) }).result!);
  session.supervise(output, tensor(3, { dtype: 'float64' }), { loss: 'mse' });
  const losses = trainer.fit([session], { epochs: 20 });
  expect(losses.at(-1)!).toBeLessThan(0.01);
  expect(new Program(op.artifact()).run({ x: 1 }).result).toBeGreaterThan(2.9);
});

it('executes shared artifacts in Python and returns Python-compiled artifacts to TypeScript', async () => {
  const { spawnSync } = await import('node:child_process');
  const { resolve } = await import('node:path');
  const p = compile('(a) => a > 4 ? 10 : 12', { inputs: { a: int32({ min: -128, max: 127 }) }, backend: 'wasm' });
  const script = `import sys,json\nsys.path.insert(0,sys.argv[1])\nfrom tensorcode.program import Program,compile,int32\np=Program.from_json(sys.stdin.read())\nassert p.verify(lambda a:10 if a>4 else 12)['passed']\nq=compile('def f(a):\\n s=0\\n for i in range(5):\\n  s+=a+i\\n return s',inputs={'a':int32(min=-10,max=10)})\nprint(q.to_json())`;
  const result = spawnSync('python3', ['-c', script, resolve('../python/src')], { input: JSON.stringify(p), encoding: 'utf8' });
  expect(result.status, result.stderr).toBe(0);
  const q = Program.fromJSON(result.stdout);
  expect(q.verify(({ a }) => 5 * a! + 10).passed).toBe(true);
  expect(q.toWasm().run({ a: 7 })).toEqual({ result: 45 });
});

it('checks one-hot constraints, rejects lexical effects and verifies replacements', async () => {
  const { verifiedReplacement } = await import('../../src/program/index.js');
  const b = new Builder(), i = b.input('i', int32({ min: 0, max: 2 }));
  const p = b.build(b.gatherOneHot([3, 5, 8], b.oneHot(i, 3)));
  expect(p.toWasm().run({ i: 2 })).toEqual({ result: 8 });
  expect(verifiedReplacement(p, p.optimize()).run({ i: 1 })).toEqual({ result: 5 });
  const c = new Builder(); const bad = c.build(c.gatherOneHot([3, 5], [1, 1]));
  expect(() => bad.run({})).toThrow('one-hot'); expect(() => bad.toWasm().run({})).toThrow();
  for (const source of ['(x) => { const y = 1; y = 2; return y; }', '(x) => { for (let i=0;i<4;i++) { i=3; } return x; }', '(x) => { if (x) { let y=2; } return y; }']) expect(() => compile(source, { inputs: { x: float64() } })).toThrow();
  const truth = compile('(x) => x ? 1 : 2', { inputs: { x: int32({ min: -2, max: 2 }) } });
  expect(truth.verify(({ x }) => x ? 1 : 2).passed).toBe(true);
});

it('preserves prefix-scan order and distinguishes forward-only calls from gradients', () => {
  const b = new Builder(), x = b.input('x'); const scanned = b.scan([x, 1, 2, 3]);
  const p = b.build(Object.fromEntries(scanned.map((v, i) => [String(i), v])));
  expect(p.toWasm().run({ x: 4 })).toEqual({ '0': 4, '1': 5, '2': 7, '3': 10 });
  const q = b.build(b.call('double', x));
  expect(q.run({ x: 2 }, { bindings: { double: { forward: x => 2 * x } } })).toEqual({ result: 4 });
  expect(() => q.gradients({ x: 2 }, { bindings: { double: { forward: x => 2 * x } } })).toThrow('VJP');
  expect(q.toTensorModule().forward({ x: tensor(2, { dtype: 'float64' }) }, { bindings: { double: { forward: x => 2 * x } } }).result!.item()).toBe(4);
  const division = new Builder(), a = division.input('a'), z = division.input('z');
  expect(division.build(a.div(z)).run({ a: 1e-300, z: 1e-300 })).toEqual({ result: 1 });
});

it('restores a compiled operation through an existing Trainer checkpoint', async () => {
  const { ProgramOperation } = await import('../../src/program/operation.js');
  const { Trainer } = await import('../../src/training/index.js');
  const { trace } = await import('../../src/_internal/tracing.js');
  const { mkdtemp, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const b = new Builder(), x = b.input('x'), w = b.parameter('weight', 0), artifact = b.build(w.mul(x)).graph();
  const op = new ProgramOperation(artifact), trainer = Trainer.fromOps({ op }, { lr: 0.1 });
  const session = trace(), result = session.run(() => op.call({ x: tensor(1, { dtype: 'float64' }) }).result!);
  session.supervise(result, tensor(3, { dtype: 'float64' }), { loss: 'mse' }); trainer.step(session);
  const dir = await mkdtemp(join(tmpdir(), 'program-checkpoint-'));
  try {
    await trainer.saveCheckpoint(join(dir, 'checkpoint'));
    const restored = new ProgramOperation(artifact), continuation = Trainer.fromOps({ op: restored }, { lr: 0.1 });
    await continuation.loadCheckpoint(join(dir, 'checkpoint'));
    expect(continuation.steps).toBe(1);
    expect(restored.artifact()).toEqual(op.artifact());
  } finally { await rm(dir, { recursive: true, force: true }); }
});

it('packs affine chains without changing IEEE rounding or native gradients', () => {
  for (const dtype of ['float64', 'float32'] as const) {
    const b = new Builder({ dtype }), x = b.input('x'), y = b.input('y'), z = b.input('z');
    const p = b.build(b.constant(1).add(x.mul(y)).add(z.mul(3))), q = p.optimize();
    expect(q.graph().nodes.some(n => n.op === 'affine')).toBe(true);
    expect(q.graph().nodes.length).toBeLessThan(p.graph().nodes.length);
    for (const inputs of [{ x: 1.234567, y: 7.654321, z: -3 }, { x: 1e10, y: 1e-8, z: -1e2 }]) {
      expect(q.run(inputs)).toEqual(p.run(inputs));
      expect(q.toWasm().run(inputs)).toEqual(p.run(inputs));
      expect(q.gradients(inputs)).toEqual(p.gradients(inputs));
    }
  }
});


it('normalizes boolean source results during exhaustive verification', () => {
  const p = compile((x: number) => x > 0, { inputs: { x: int32({ min: -2, max: 2 }) } });
  expect(p.verify().passed).toBe(true);
});

it('retains explicit hybrid policies when imported into an exact program', () => {
  const inner = new Builder({ semantics: 'hybrid', surrogate: 'sigmoid', temperature: 0.5 }), a = inner.input('a');
  const region = inner.build(inner.select(a.gt(0), 1, 0));
  const outer = new Builder(), x = outer.input('x');
  const p = outer.build(outer.include(region, { a: x }).result!);
  expect(p.run({ x: 1 })).toEqual({ result: 1 });
  expect(p.gradients({ x: 1 }).x).toBeCloseTo(region.gradients({ a: 1 }).a!, 12);
});

it('does not move an external call ahead of a failing arithmetic boundary', () => {
  const b = new Builder(), x = b.input('x'), effect = b.call('effect', x);
  const p = b.build(b.constant(0).add(x.mul(x)).add(effect.mul(1))).optimize();
  let calls = 0;
  expect(() => p.run({ x: 1e308 }, { bindings: { effect: { forward: () => { calls++; return 1; } } } })).toThrow();
  expect(calls).toBe(0);
});
