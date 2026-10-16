import { describe, it, expect } from "vitest";
import { fixtureJson } from "../helpers/fixtures.js";
import {
  Algorithm,
  AlgorithmBuilder,
  convert,
  type AlgorithmArtifact,
  type Target,
} from "../../src/algorithm/index.js";
import { tensor, type Tensor } from "../../src/nn/tensor.js";
import { SGD } from "../../src/nn/optim.js";

const targets: Target[] = ["alg0", "alg1", "alg2"];
describe("portable algorithm substrates", () => {
  it("runs and roundtrips all substrates", () => {
    const b = new AlgorithmBuilder();
    const x = b.input("x", [2]);
    const w = b.parameter("w", [2, 3]);
    const a = b.build(x.mul(w).tanh());
    for (const target of targets) {
      const m = a.compile(target);
      const y = m.forward({ x: [1, 2] }).result!;
      expect(y.data[0]).toBeCloseTo(Math.tanh(2), 12);
      for (const next of targets)
        expect(convert(m, next).forward({ x: [1, 2] }).result!.data).toEqual(
          y.data,
        );
    }
    expect(Algorithm.fromJSON(a.toJSON()).run({ x: [1, 2] })).toEqual(
      a.run({ x: [1, 2] }),
    );
  });
  it("trains through attention and exports current weights", () => {
    const b = new AlgorithmBuilder();
    const x = b.input("x");
    const w = b.parameter("w", 0);
    const a = b.build(x.mul(w));
    const m = a.compile("alg2");
    const opt = new SGD(m.parameters(), { lr: 0.1 });
    for (let i = 0; i < 40; i++) {
      opt.zeroGrad();
      const loss = m.forward({ x: 1 }).result!.sub(3).square();
      loss.backward();
      opt.step();
    }
    expect(m.forward({ x: 1 }).result!.item()).toBeCloseTo(3, 3);
    expect(m.convert("alg0").forward({ x: 1 }).result!.item()).toBe(
      m.forward({ x: 1 }).result!.item(),
    );
  });
  it("preserves explicit simultaneous state and gradients", () => {
    const b = new AlgorithmBuilder();
    const x = b.input("x");
    const s = b.state("s", 0);
    const w = b.parameter("w", 0.5);
    b.update(s, s.mul(w).add(x));
    const a = b.build(s.mul(w).add(x));
    const grads: number[] = [];
    for (const t of ["alg1", "alg2"] as Target[]) {
      const m = a.compile(t);
      const x = tensor(1, { dtype: "float64", requiresGrad: true });
      let state: Record<string, Tensor> | undefined;
      let result: Tensor | undefined;
      for (let i = 0; i < 4; i++) {
        const r = m.step({ x }, { state });
        state = r.state;
        result = r.outputs.result;
      }
      result!.backward();
      grads.push(x.grad!.item());
    }
    expect(grads[0]).toBeCloseTo(grads[1]!, 12);
    expect(() => a.run({ x: 1 })).toThrow();
  });
  it("rejects soft routing export and malformed artifacts", () => {
    const b = new AlgorithmBuilder();
    const x = b.input("x");
    const a = b.build(x.sub(3));
    const m = a.compile("alg2", { routing: "soft", strength: 0 });
    expect(m.forward({ x: 10 }).result!.item()).toBe(0);
    expect(() => m.artifact()).toThrow(/soft routing/);
    const bad = a.graph();
    bad.nodes.at(-1)!.args[0] = 999;
    expect(() => new Algorithm(bad)).toThrow();
    expect(() => a.compile("alg1", { routing: "soft" })).toThrow();
    expect(() => a.run({ x: [1, 2] })).toThrow();
  });
  it("runs a Python vector feedback artifact with cross-language parity", () => {
    const payload = fixtureJson<{
      artifact: AlgorithmArtifact;
      inputs: Record<string, number[]>;
      expected: number[];
    }>("algorithm_feedback.json");
    const inputs = payload.inputs;
    const a = new Algorithm(payload.artifact);
    for (const target of targets) {
      const result = a.compile(target).forward(inputs).result!;
      for (let i = 0; i < 20; i++)
        expect(result.data[i]).toBeCloseTo(payload.expected[i]!, 10);
    }
  });
  it("imports a native Python attention graph including masks", () => {
    const payload = fixtureJson<{
      algorithm: AlgorithmArtifact;
      x: number[][];
      expected: number[][];
    }>("algorithm_attention.json");
    const a = new Algorithm(payload.algorithm);
    for (const t of targets) {
      const y = a.compile(t).forward({ x: payload.x }).result!;
      for (let i = 0; i < 12; i++)
        expect(y.data[i]).toBeCloseTo(payload.expected.flat()[i]!, 9);
    }
  });
});

it("preserves dtype casts and module checkpoints across export", () => {
  const b = new AlgorithmBuilder();
  const x = b.input("x");
  const w = b.parameter("weight", 0.123456789);
  const a = b.build(x.mul(w).add(0.234567891));
  const m = a.compile("alg2").float();
  expect(m.forward({ x: 1 }).result!.dtype).toBe("float32");
  expect(m.artifact().graph().dtype).toBe("float32");
  m.double();
  const y = m.forward({ x: 1 }).result!.item();
  expect(y).toBe(Math.fround(0.123456789) + Math.fround(0.234567891));
  const loaded = m.artifact().compile("alg1");
  loaded.loadStateDict(m.stateDict());
  expect(loaded.forward({ x: 1 }).result!.item()).toBe(y);
  expect(() => m.half()).toThrow(/float32 or float64/);
});
