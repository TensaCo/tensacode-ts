/** Ports of ``tests/vec/test_native_configuration_contracts.py`` and the structural parts of ``test_image_decode.py``. */
import { describe, expect, it } from 'vitest';
import { NotImplementedError } from '../../src/errors.js';
import { ImageDecode, ImageDecoder, ImageEncoder, Space, TextDecoder, TextEncoder } from '../../src/ops/vec/index.js';
import { qualifiedName } from '../../src/_internal/identity.js';

describe('native operation configuration contracts', () => {
  it.each([TextEncoder, TextDecoder, ImageEncoder])('%o rejects unknown keys before model construction', (cls) => {
    expect(() => new (cls as any)({ misspelled_option: true })).toThrow(/Unknown configuration/);
  });

  it.each([TextEncoder, TextDecoder, ImageEncoder])('%o rejects non-JSON constructors', (cls) => {
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

describe('ImageDecoder (latent diffusion) is unavailable in TypeScript', () => {
  it('keeps the public identity and aliases', () => {
    expect(ImageDecode).toBe(ImageDecoder);
    expect(qualifiedName(ImageDecoder)).toBe('tensorcode.ops.vec.decode.ImageDecoder');
  });

  it('construction, foundation import and artifact loading raise NotImplementedError', async () => {
    const config = { input_space: new Space('image-conditioning', 6, { organization: 'sequence' }).configuration(), num_inference_steps: 2 };
    expect(() => new ImageDecoder(config)).toThrow(NotImplementedError);
    expect(() => new ImageDecoder({ misspelled_option: true })).toThrow(/not available in the TypeScript port/);
    await expect(ImageDecoder.fromFoundation('any/diffusion', { inputSpace: new Space('x', 6) })).rejects.toThrow(NotImplementedError);
    await expect(ImageDecoder.fromPretrained('./anything')).rejects.toThrow(/use the Python package/);
  });
});
