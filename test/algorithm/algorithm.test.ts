import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
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
  it("runs the actual Python robotics policy artifact with cross-language parity", () => {
    const path = fileURLToPath(
      new URL(
        "../../../python/docs/results/robotics/calibration/policy.json",
        import.meta.url,
      ),
    );
    const artifact = JSON.parse(
      readFileSync(path, "utf8"),
    ) as AlgorithmArtifact;
    const a = new Algorithm(artifact);
    const inputs = {
      position: Array.from({ length: 20 }, (_, i) => Math.sin(i) * 0.2),
      velocity: Array.from({ length: 20 }, (_, i) => Math.cos(i) * 0.4),
      target: Array(20).fill(0.1),
      target_velocity: Array(20).fill(0),
      feedforward: Array(20).fill(0.3),
    };
    const python = fileURLToPath(
      new URL("../../../python/.venv/bin/python", import.meta.url),
    );
    const expected = JSON.parse(
      execFileSync(
        python,
        [
          "-c",
          "import json,sys; from tensorcode.algorithm import Algorithm; a=Algorithm.from_json(open(sys.argv[1]).read()); print(json.dumps({k:v.tolist() for k,v in a.run(json.loads(sys.argv[2])).items()}))",
          path,
          JSON.stringify(inputs),
        ],
        { encoding: "utf8" },
      ),
    ) as { torque: number[] };
    for (const t of targets) {
      const result = a.compile(t).forward(inputs).torque!;
      for (let i = 0; i < 20; i++)
        expect(result.data[i]).toBeCloseTo(expected.torque[i]!, 10);
    }
  });
  it("imports a native Python attention graph including masks", () => {
    const python = fileURLToPath(
      new URL("../../../python/.venv/bin/python", import.meta.url),
    );
    const payload = JSON.parse(
      execFileSync(
        python,
        [
          "-c",
          `import json,torch
from tensorcode.algorithm import from_attention
torch.manual_seed(9)
m=torch.nn.MultiheadAttention(4,2,batch_first=True,dropout=0.).double()
a=from_attention(m,tokens=3,causal=True)
x=torch.randn(3,4,dtype=torch.float64)
print(json.dumps({'algorithm':a.graph(),'x':x.tolist(),'expected':a.run({'x':x.numpy()})['result'].tolist()}))`,
        ],
        { encoding: "utf8" },
      ),
    ) as { algorithm: AlgorithmArtifact; x: number[][]; expected: number[][] };
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
