/** Port of ``tests/vec/test_native_configuration_contracts.py``. */
import { describe, expect, it } from 'vitest';
import { ImageDecode, ImageDecoder, ImageEncoder, TextDecoder, TextEncoder } from '../../src/ops/vec/index.js';
import { qualifiedName } from '../../src/_internal/identity.js';

describe('native operation configuration contracts', () => {
  it.each([TextEncoder, TextDecoder, ImageEncoder, ImageDecoder])('%o rejects unknown keys before model construction', (cls) => {
    expect(() => new (cls as any)({ misspelled_option: true })).toThrow(/Unknown configuration/);
  });

  it.each([TextEncoder, TextDecoder, ImageEncoder, ImageDecoder])('%o rejects non-JSON constructors', (cls) => {
    for (const config of [(value: unknown) => value, [['foundation', 'example']]]) {
      expect(() => new (cls as any)(config)).toThrow(/config|JSON/);
    }
  });

  it.each([
    ['space', {}], ['output', 'sequence'], ['pooling', 'mean'],
  ])('text/vision encoders reject legacy constructor key %s', (key, value) => {
    expect(() => new TextEncoder({ [key]: value })).toThrow(/output_space|readout/);
    if (key !== 'pooling') expect(() => new ImageEncoder({ [key]: value })).toThrow(/output_space|readout/);
  });
});

describe('ImageDecoder public identity', () => {
  it('keeps the public identity and aliases', () => {
    expect(ImageDecode).toBe(ImageDecoder);
    expect(qualifiedName(ImageDecoder)).toBe('tensorcode.ops.vec.decode.ImageDecoder');
  });
});
