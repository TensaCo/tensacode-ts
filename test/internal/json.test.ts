import { describe, expect, it } from 'vitest';
import { canonicalJson, canonicalizeJsonText, parseJsonStrict, pythonFloatRepr, pythonJsonDumps, sha256Hex } from '../../src/_internal/json.js';

describe('Python-compatible JSON', () => {
  it('formats floats like Python repr', () => {
    expect(pythonFloatRepr(0.1)).toBe('0.1');
    expect(pythonFloatRepr(1e-12)).toBe('1e-12');
    expect(pythonFloatRepr(1e-5)).toBe('1e-05');
    expect(pythonFloatRepr(1.5e16)).toBe('1.5e+16');
    expect(pythonFloatRepr(123456789.125)).toBe('123456789.125');
    expect(pythonFloatRepr(0.0001)).toBe('0.0001');
  });

  it('dumps like json.dumps with defaults and sort_keys', () => {
    expect(pythonJsonDumps({ b: 1, a: [true, null, 'é'] }, { sortKeys: true })).toBe('{"a": [true, null, "\\u00e9"], "b": 1}');
    expect(canonicalJson({ z: 0.5, a: { c: 'x', b: 2 } })).toBe('{"a":{"b":2,"c":"x"},"z":0.5}');
    expect(pythonJsonDumps({ a: [1] }, { indent: 2 })).toBe('{\n  "a": [\n    1\n  ]\n}');
  });

  it('re-serializes JSON text preserving the float/int distinction', () => {
    const text = '{"b": 1.0, "a": [2, -0.0, 1e-5, 12345678901234567890], "s": "\\u2581x"}';
    expect(canonicalizeJsonText(text, { sortKeys: true, separators: [',', ':'] }))
      .toBe('{"a":[2,-0.0,1e-05,12345678901234567890],"b":1.0,"s":"\\u2581x"}');
  });

  it('rejects duplicate keys when parsing strictly', () => {
    expect(() => parseJsonStrict('{"a": 1, "a": 2}')).toThrow(/Duplicate/);
    expect(() => canonicalizeJsonText('{"a": 1, "a": 2}')).toThrow(/Duplicate/);
  });

  it('hashes with SHA-256', () => {
    expect(sha256Hex('abc')).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  });
});

describe('Python float spelling', () => {
  it('writes integral values under float keys as floats, including inside arrays', () => {
    expect(canonicalJson({ initializer_factor: 1, coordinate_stride: [8, 8.5], k: 2, p: 0 })).toBe('{"coordinate_stride":[8.0,8.5],"initializer_factor":1.0,"k":2,"p":0.0}');
  });
});
