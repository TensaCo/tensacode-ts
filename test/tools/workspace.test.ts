/**
 * Coverage from python/tests/models/test_workspace.py not already in the core
 * Workspace parity test: masking, permutation equivariance, gradients and validation.
 */
import { describe, expect, it } from 'vitest';
import { full, manualSeed, noGrad, ones, randn, tensor, zeros } from '../../src/nn/index.js';
import { Workspace } from '../../src/_internal/workspace.js';
import { expectClose } from '../helpers/gradcheck.js';

const bool = (rows: number[][]) => tensor(rows.flat(), { shape: [rows.length, rows[0]!.length], dtype: 'bool' });

describe('Workspace', () => {
  it('links sources and fully masks excluded evidence', () => {
    manualSeed(12);
    const model = new Workspace(6, 3, 2);
    const evidence = randn([2, 5, 6]);
    const maskRows = [[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]];
    const mask = bool(maskRows);
    const result = noGrad(() => model.forward(evidence, mask));
    const changed = evidence.clone();
    maskRows.flat().forEach((flag, index) => {
      if (!flag) changed.data.fill(1e20, index * 6, index * 6 + 6);
    });
    const other = noGrad(() => model.forward(changed, mask));
    expect(result.conditioning.shape).toEqual([2, 3, 6]);
    expect(result.mask.all().item()).toBe(1);
    expect(result.conditioning.equal(other.conditioning)).toBe(true);
    const attention = result.attention.toArray();
    for (let b = 0; b < 2; b += 1) for (let s = 0; s < 3; s += 1) for (let t = 0; t < 5; t += 1) {
      if (!maskRows[b]![t]) expect(attention[(b * 3 + s) * 5 + t]).toBe(0);
    }
    expectClose(result.attention.sum(-1).toArray(), new Array(6).fill(1), 1e-6);
    expectClose(result.relations.sum(-1).toArray(), new Array(6).fill(1), 1e-6);
    changed.data[0] = changed.data[0]! + 4;
    expect(noGrad(() => model.forward(changed, mask)).conditioning.equal(result.conditioning)).toBe(false);
  });

  it('is equivariant to evidence permutations', () => {
    manualSeed(11);
    const model = new Workspace(5, 3);
    const evidence = randn([2, 4, 5]);
    const mask = bool([[1, 0, 1, 1], [0, 1, 1, 0]]);
    const permutation = [2, 0, 3, 1];
    const original = noGrad(() => model.forward(evidence, mask));
    const shuffled = noGrad(() => model.forward(evidence.indexSelect(1, permutation), mask.indexSelect(1, permutation)));
    expectClose(shuffled.conditioning.data, original.conditioning.data, 1e-5);
    expectClose(shuffled.relations.data, original.relations.data, 1e-5);
    expectClose(shuffled.attention.data, original.attention.indexSelect(2, permutation).data, 1e-5);
  });

  it('propagates gradients to every parameter but not to masked evidence', () => {
    manualSeed(27);
    const model = new Workspace(6, 3);
    const before = model.namedParameters().map(([name, parameter]) => [name, parameter]);
    const evidence = randn([2, 4, 6]).requiresGrad_();
    const maskRows = [[1, 0, 1, 1], [1, 1, 1, 0]];
    const output = model.forward(evidence, bool(maskRows));
    output.conditioning.mul(randn(output.conditioning.shape)).sum().backward();
    expect(model.namedParameters().map(([name, parameter]) => [name, parameter])).toEqual(before);
    for (const [name, parameter] of model.namedParameters()) {
      expect(parameter.grad, name).not.toBeNull();
      expect(parameter.grad!.allFinite(), name).toBe(true);
      expect(parameter.grad!.abs().sum().item(), name).toBeGreaterThan(0);
    }
    const grad = evidence.grad!.toArray();
    let unmasked = 0;
    maskRows.flat().forEach((flag, index) => {
      const row = grad.slice(index * 6, index * 6 + 6);
      if (flag) unmasked += row.reduce((a, b) => a + Math.abs(b), 0);
      else expect(row.every((value) => value === 0)).toBe(true);
    });
    expect(unmasked).toBeGreaterThan(0);
    const restored = new Workspace(6, 3);
    restored.loadStateDict(model.stateDict());
    expect(restored.configuration()).toEqual(model.configuration());
    expectClose(noGrad(() => restored.forward(evidence.detach(), bool(maskRows))).conditioning.data, output.conditioning.data, 1e-6);
  });

  it('rejects invalid configuration', () => {
    expect(() => new Workspace(0)).toThrow();
    expect(() => new Workspace(true as unknown as number)).toThrow();
    expect(() => new Workspace(3, 8, 0)).toThrow();
    expect(() => new Workspace(3, 1.5)).toThrow();
  });

  it('rejects invalid evidence', () => {
    const model = new Workspace(3);
    const cases: [ReturnType<typeof zeros>, ReturnType<typeof zeros> | null][] = [
      [zeros([0, 2, 3]), null], [zeros([2, 0, 3]), null], [ones([2, 2, 4]), null], [ones([2, 3]), null],
      [ones([2, 2, 3], { dtype: 'int64' }), null], [full([2, 2, 3], Number.NaN), null],
      [ones([2, 2, 3]), zeros([2, 2], { dtype: 'bool' })], [ones([2, 2, 3]), ones([2, 2])], [ones([2, 2, 3]), ones([2, 3], { dtype: 'bool' })],
    ];
    for (const [evidence, mask] of cases) expect(() => model.forward(evidence, mask)).toThrow();
  });
});
