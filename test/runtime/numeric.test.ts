import { describe, expect, it } from 'vitest';
import { fsum, pythonSum } from '../../src/_internal/numeric.js';

describe('Python float summation', () => {
  it('pythonSum matches CPython >= 3.12 builtin sum', () => {
    // Values produced by CPython 3.13 ``sum(...)``.
    expect(pythonSum([1e16, 1.0, -1e16])).toBe(1.0);
    expect(pythonSum([0.1, 0.2, 0.3])).toBe(0.6);
    expect(pythonSum([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])).toBe(1.0);
    expect(pythonSum([])).toBe(0);
    expect(pythonSum([Infinity, 1])).toBe(Infinity);
    expect(Number.isNaN(pythonSum([Infinity, -Infinity]))).toBe(true);
  });

  it('fsum is exactly rounded', () => {
    expect(fsum([1e100, 1, -1e100, 1e-100, 1e50, -1, -1e50])).toBe(1e-100);
  });
});
