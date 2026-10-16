import { expect, it } from "vitest";
import { fixtureJson } from "../helpers/fixtures.js";
import {
  Algorithm,
  compileSchedule,
  type Inputs,
} from "../../src/algorithm/index.js";
it("compiles dependency scheduling on all substrates", () => {
  const a = compileSchedule({ inspect: [], pick: ["inspect"], pack: ["pick"] });
  for (const target of ["alg0", "alg1", "alg2"] as const) {
    const module = a.compile(target);
    for (let bits = 0; bits < 8; bits++) {
      const completed = [bits & 1, (bits >> 1) & 1, (bits >> 2) & 1];
      const y = module.forward({ completed, enabled: [1, 1, 1] }).ready!;
      expect(Array.from(y.data)).toEqual([
        +!completed[0],
        +(!!completed[0] && !completed[1]),
        +(!!completed[1] && !completed[2]),
      ]);
    }
  }
  expect(() => compileSchedule({ a: ["b"], b: ["a"] })).toThrow(/cyclic/);
  expect(() => compileSchedule({ a: ["missing"] })).toThrow(/unknown/);
  expect(() => compileSchedule({ a: [], b: ["a", "a"] })).toThrow(/duplicate/);
  expect(() => compileSchedule({})).toThrow();
});
it("replays exhaustive Python scheduler decisions in TypeScript", () => {
  const payload = fixtureJson<{
    artifact: string;
    cases: { inputs: Inputs; outputs: Record<string, number[]> }[];
  }>("algorithm_schedule.json");
  const a = Algorithm.fromJSON(payload.artifact);
  for (const target of ["alg0", "alg1", "alg2"] as const) {
    const m = a.compile(target);
    for (const c of payload.cases)
      for (const [key, y] of Object.entries(m.forward(c.inputs)))
        expect(Array.from(y.data)).toEqual(c.outputs[key]);
  }
});
