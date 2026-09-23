/** Port of ``tests/vec/test_image_encoding.py`` (PatchEncoder). */
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Conv2d, Identity, Linear, Module, SGD, arange, ones, stack, zeros, type Tensor } from '../../src/nn/index.js';
import { Latent, PatchEncoder, Space, Transform } from '../../src/ops/vec/index.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-patch-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const spatial = (name: string, dimensions: number) => new Space(name, dimensions, { organization: 'spatial' });

/** 2x2 average pooling (``F.avg_pool2d(value, 2)``). */
class AvgPool extends Module {
  forward(value: Tensor): Tensor {
    const [b, c, h, w] = value.shape as [number, number, number, number];
    const rows = Math.floor(h / 2);
    const columns = Math.floor(w / 2);
    return value.slice(2, 0, rows * 2).slice(3, 0, columns * 2).reshape(b, c, rows, 2, columns, 2).mean([3, 5]);
  }
}

describe('patch encoding (tests/vec/test_image_encoding.py)', () => {
  it('returns a spatial patch grid and pixel coordinates', () => {
    const encoder = new PatchEncoder({ in_channels: 1, patch_size: [2, 2], dimensions: 3, output_space: spatial('random-image-patches', 3).configuration() as any });
    const result = encoder.call(arange(0, 24, 1, { dtype: 'float32' }).reshape(1, 4, 6));
    expect(result).toBeInstanceOf(Latent);
    expect(result.tensor.shape).toEqual([2, 3, 3]);
    expect(result.coordinates!.tolist()).toEqual([[[1, 1], [1, 3], [1, 5]], [[3, 1], [3, 3], [3, 5]]]);
  });

  it('default coordinates do not rescale over unused border pixels', () => {
    const encoder = new PatchEncoder({ in_channels: 1, patch_size: 2, dimensions: 1, output_space: spatial('odd-image-patches', 1).configuration() as any });
    const result = encoder.call(ones([1, 5, 5]));
    expect(result.tensor.shape).toEqual([2, 2, 1]);
    expect(result.coordinates!.tolist()).toEqual([[[1, 1], [1, 3]], [[3, 1], [3, 3]]]);
  });

  it('batches images and updates real parameters', () => {
    const encoder = new PatchEncoder({ in_channels: 1, patch_size: 2, dimensions: 2, output_space: spatial('trainable-image-patches', 2).configuration() as any });
    const images = stack([zeros([1, 4, 4]), ones([1, 4, 4])]);
    const conv = encoder.module as Conv2d;
    const before = conv.weight.detach().clone();
    encoder.call(images).tensor.square().mean().backward();
    new SGD(encoder.parameters(), { lr: 0.1 }).step();
    expect(encoder.call(images).tensor.shape).toEqual([2, 2, 2, 2]);
    expect(conv.weight.grad).not.toBeNull();
    expect(conv.weight.equal(before)).toBe(false);
  });

  it('exposes a supplied module without claiming semantics', () => {
    const supplied = new Conv2d(3, 5, 4, { stride: 4, bias: false });
    const encoder = PatchEncoder.fromModule(supplied, { patchSize: 4, outputSpace: spatial('caller-trained/model-x', 5) });
    expect(encoder.module).toBe(supplied);
    const result = encoder.call(ones([3, 8, 8]));
    expect(result.tensor.shape).toEqual([2, 2, 5]);
    expect(result.coordinates!.tolist()).toEqual([[[2, 2], [2, 6]], [[6, 2], [6, 6]]]);
    expect(encoder.initialization).toBe('supplied');
    expect(encoder.configuration().initialization).toBe('supplied');
  });

  it('an arbitrary supplied module does not invent patch coordinates', () => {
    const encoder = PatchEncoder.fromModule(new AvgPool(), { patchSize: 2, outputSpace: spatial('caller-module', 1) });
    expect(encoder.call(ones([1, 4, 4])).coordinates).toBeNull();
  });

  it('an arbitrary module can declare spatial coordinate geometry', () => {
    const encoder = PatchEncoder.fromModule(new AvgPool(), { patchSize: 2, outputSpace: spatial('caller-module', 1), coordinateStride: 2, coordinateOffset: 1 });
    const result = encoder.call(ones([1, 5, 5]));
    expect(result.coordinates!.select(2, 0).tolist()).toEqual([[1, 1], [3, 3]]);
    expect(encoder.configuration().coordinate_stride).toEqual([2, 2]);
  });

  it('a supplied image module must preserve the batch count', () => {
    class DropsBatch extends Module {
      forward(value: Tensor): Tensor { return value.slice(0, 0, 1); }
    }
    const encoder = PatchEncoder.fromModule(new DropsBatch(), { patchSize: 1, outputSpace: spatial('bad-module', 1) });
    expect(() => encoder.call(ones([2, 1, 2, 2]))).toThrow(/batch/);
  });

  it('rejects non-spatial spaces and bad image shapes', () => {
    expect(() => new PatchEncoder({ in_channels: 3, patch_size: 2, dimensions: 4, output_space: new Space('not-spatial', 4).configuration() as any })).toThrow(/spatial/);
    const encoder = new PatchEncoder({ in_channels: 3, patch_size: 2, dimensions: 4, output_space: spatial('patches', 4).configuration() as any });
    expect(() => encoder.call(ones([8, 8]))).toThrow(/CHW or BCHW/);
  });

  it('cross-modal projection requires an explicit adapter transform', () => {
    const imageSpace = spatial('image-model/patches', 2);
    const sharedSpace = spatial('paired-model/shared', 4);
    const image = new PatchEncoder({ in_channels: 1, patch_size: 2, dimensions: 2, output_space: imageSpace.configuration() as any }).call(ones([1, 4, 4]));
    const projected = Transform.fromModule(new Linear(2, 4), { inputSpace: imageSpace, outputSpace: sharedSpace }).call(image) as Latent;
    expect(projected.space.equals(sharedSpace)).toBe(true);
    expect(projected.tensor.shape).toEqual([2, 2, 4]);
    expect(projected.coordinates!.equal(image.coordinates!)).toBe(true);
  });

  it('owned patch artifact restores trained weights and geometry', async () => {
    const config = {
      in_channels: 1, patch_size: [2, 3], output_space: spatial('owned-patches', 2).configuration() as any,
      coordinate_stride: [3, 4], coordinate_offset: [-1, 2],
    };
    const encoder = new PatchEncoder(config);
    const images = arange(0, 48, 1, { dtype: 'float32' }).reshape(2, 1, 4, 6).div(48);
    encoder.call(images).tensor.square().mean().backward();
    new SGD(encoder.parameters(), { lr: 0.1 }).step();
    const expected = encoder.call(images);
    await encoder.savePretrained(join(scratch, 'patches'));
    const restored = await PatchEncoder.fromPretrained(join(scratch, 'patches'));
    const actual = restored.call(images);
    expect(restored.configuration()).toEqual(encoder.configuration());
    expect(new PatchEncoder(restored.configuration()).configuration()).toEqual(restored.configuration());
    expect([...restored.stateDict().keys()]).toEqual(['module.weight', 'module.bias']);
    expect(actual.tensor.equal(expected.tensor)).toBe(true);
    expect(actual.coordinates!.equal(expected.coordinates!)).toBe(true);
    expect(actual.space.equals(expected.space)).toBe(true);
    config.coordinate_offset[0] = 99;
    const returned = restored.configuration();
    (returned.coordinate_offset as number[])[0] = 99;
    expect(restored.configuration().coordinate_offset).toEqual([-1, 2]);
  });

  it('a supplied patch module artifact save is rejected before writing', async () => {
    const encoder = PatchEncoder.fromModule(new Conv2d(1, 2, 2), { patchSize: 2, outputSpace: spatial('supplied', 2) });
    const destination = join(scratch, 'unsupported');
    await expect(encoder.savePretrained(destination)).rejects.toThrow(/supplied/);
    expect(existsSync(destination)).toBe(false);
  });

  const base = { in_channels: 1, patch_size: 2, output_space: spatial('patches', 2).configuration() as any };
  it.each([
    [{ module: new Identity() }], [{ obsolete: true }], [{ in_channels: true }], [{ dimensions: true }],
    [{ coordinate_stride: 2 }], [{ coordinate_stride: 2, coordinate_offset: Number.NaN }],
  ])('owned patch config rejects unsupported or invalid fields (%o)', (changes) => {
    expect(() => new PatchEncoder({ ...base, ...changes })).toThrow();
  });
});
