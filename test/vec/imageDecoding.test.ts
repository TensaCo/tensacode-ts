/**
 * Image decoding, resampling and processor input parity with Python
 * (``scripts/fixtures/image_fixtures.py``): every fixture file decoded like
 * ``torchvision.io.decode_image`` (all read modes, EXIF orientation) and like
 * ``PIL.Image.open`` (mode, ``convert('RGB')``, ``exif_transpose``,
 * ``pil_to_tensor``), Pillow ``Image.resize`` for every mode and filter, and
 * ``ViTImageProcessor(**config)(images=...)`` over paths, base64, data URIs,
 * tensors, decoded images and nested lists.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Tensor, tensor } from '../../src/nn/index.js';
import { NotImplementedError, ValueError } from '../../src/errors.js';
import {
  decodeImage, loadImage, openImage, RasterImage, Resampling, type ImageReadMode, type RasterMode,
} from '../../src/ops/vec/index.js';
import { ImageProcessor, resizeImage } from '../../src/_internal/vec/imageProcessing.js';
import { inflateRaw, inflateZlib, setNativeInflate } from '../../src/_internal/image/inflate.js';
import { decodeBase64 } from '../../src/_internal/image/load.js';
import { cos as glibcCos, fma, sin as glibcSin, sinf as glibcSinf } from '../../src/nn/randomMath.js';
import { fixtureBytes, fixtureJson } from '../helpers/fixtures.js';

const directory = new URL('../fixtures/image/', import.meta.url).pathname;
const expected = fixtureJson('image/expected.json') as Record<string, Record<string, any>>;
const names = Object.keys(expected).sort();
const sha = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

function tensorBytes(value: Tensor, dtype = value.dtype as string): Uint8Array {
  const data = value.data;
  switch (dtype) {
    case 'uint16': return new Uint8Array(Uint16Array.from(data).buffer);
    case 'int32': return new Uint8Array(Int32Array.from(data).buffer);
    case 'float32': return new Uint8Array(Float32Array.from(data).buffer);
    case 'float64': return new Uint8Array(Float64Array.from(data).buffer);
    default: return Uint8Array.from(data);
  }
}

/** Reads whose output depends on torchvision's uninitialized memory (libpng writes short rows). */
function undefinedTorchvisionOutput(name: string, key: string): boolean {
  const palette = /_ct3_d[124]/.test(name) || name === 'pil_PA.png' || name === 'png_gamma_palette.png';
  const paletteWithoutTrns = /_ct3_d\d+_i\.png$/.test(name) || name === 'pil_PA.png';
  if (key === 'tv_unchanged' && palette) return true;
  if ((key === 'tv_rgba' || key === 'tv_gray_alpha') && paletteWithoutTrns) return true;
  return false;
}

describe('torchvision.io.decode_image parity', () => {
  const modes: [string, ImageReadMode, boolean][] = [
    ['tv', 'RGB', false], ['tv_exif', 'RGB', true], ['tv_unchanged', 'UNCHANGED', false], ['tv_gray', 'GRAY', false],
    ['tv_gray_alpha', 'GRAY_ALPHA', false], ['tv_rgba', 'RGB_ALPHA', false],
  ];
  it.each(modes)('%s matches for every fixture file', (key, mode, exif) => {
    let compared = 0;
    for (const name of names) {
      if (undefinedTorchvisionOutput(name, key)) continue;
      const record = expected[name]![key];
      const bytes = fixtureBytes(`image/${name}`);
      if (record.error) {
        expect(() => decodeImage(bytes, { mode, applyExifOrientation: exif }), `${name} should fail`).toThrow();
        continue;
      }
      const decoded = decodeImage(bytes, { mode, applyExifOrientation: exif });
      expect(decoded.shape, name).toEqual(record.shape);
      expect(sha(tensorBytes(decoded, record.dtype)), `${key} ${name}`).toBe(record.sha256);
      compared += 1;
    }
    expect(compared).toBeGreaterThan(100);
  });

  it('decodes file paths and rejects unsupported formats like torchvision', () => {
    const rgb = decodeImage(`${directory}jpg_37x21_q95_s2.jpg`, { mode: 'RGB' });
    expect(rgb.shape).toEqual([3, 21, 37]);
    expect(rgb.dtype).toBe('uint8');
    expect(() => decodeImage(fixtureBytes('image/bmp_RGB.bmp'), { mode: 'RGB' })).toThrow(/Unsupported image file/);
    expect(() => decodeImage(new Uint8Array(0))).toThrow(/non empty/);
    expect(() => decodeImage(fixtureBytes('image/jpg_gray.jpg'), { mode: 'RGB_ALPHA' })).toThrow(/not supported for JPEG/);
    expect(() => decodeImage(fixtureBytes('image/webp_anim.webp'), { mode: 'RGB' })).toThrow(/Animated webp/);
    const sixteen = decodeImage(fixtureBytes('image/png_13x9_ct2_d16.png'), { mode: 'RGB' });
    expect(sixteen.dtype).toBe('uint16'); // torch.uint16
    expect(Math.max(...sixteen.data)).toBeGreaterThan(255);
  });
});

describe('PIL.Image.open parity', () => {
  it('mode, size, frames, RGB, exif_transpose and pil_to_tensor match for every fixture file', () => {
    for (const name of names) {
      const record = expected[name]!.pil;
      const image = openImage(fixtureBytes(`image/${name}`));
      expect(image.mode, name).toBe(record.mode);
      expect(image.size, name).toEqual(record.size);
      expect(image.info.frames, name).toBe(record.frames);
      expect(image.orientation, name).toBe(record.orientation ?? null);
      const rgb = image.convert('RGB').toTensor();
      expect(sha(tensorBytes(rgb)), `${name} rgb`).toBe(record.rgb.sha256);
      const transposed = image.exifTranspose().convert('RGB').toTensor();
      expect(sha(tensorBytes(transposed)), `${name} exif`).toBe(record.exif_rgb.sha256);
      if (record.raw.sha256) {
        const raw = image.toTensor();
        expect(raw.shape, name).toEqual(record.raw.shape);
        expect(sha(tensorBytes(raw, record.raw.dtype)), `${name} raw`).toBe(record.raw.sha256);
      }
    }
  });

  it('opens paths, Buffers and ArrayBuffers and rejects unknown data', () => {
    const buffer = readFileSync(`${directory}png_5x3_ct2_d8.png`);
    const fromPath = openImage(`${directory}png_5x3_ct2_d8.png`);
    const fromBuffer = openImage(buffer);
    const fromArrayBuffer = openImage(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
    expect(fromBuffer.data).toEqual(fromPath.data);
    expect(fromArrayBuffer.data).toEqual(fromPath.data);
    expect(() => openImage(new Uint8Array([1, 2, 3, 4]))).toThrow(/cannot identify image file/);
  });
});

describe('Pillow Image.resize parity', () => {
  const BANDS: Record<string, number> = { L: 1, RGB: 3, RGBA: 4, LA: 2, I: 1, F: 1, 'I;16': 1, P: 1, 1: 1, CMYK: 4 };
  const decode = (raw: { dtype: string; bytes: string }): ArrayLike<number> => {
    const buffer = Buffer.from(raw.bytes, 'base64');
    const bytes = new Uint8Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
    if (raw.dtype === 'int32') return new Int32Array(bytes.buffer);
    if (raw.dtype === 'float32') return new Float32Array(bytes.buffer);
    if (raw.dtype === 'uint16') return new Uint16Array(bytes.buffer);
    return bytes;
  };
  const toImage = (mode: string, width: number, height: number, chw: ArrayLike<number>): RasterImage => {
    const bands = BANDS[mode]!;
    const plane = width * height;
    const data = mode === 'I' ? new Int32Array(plane * bands) : mode === 'F' ? new Float32Array(plane * bands)
      : mode === 'I;16' ? new Uint16Array(plane * bands) : new Uint8Array(plane * bands);
    for (let i = 0; i < plane; i += 1) for (let c = 0; c < bands; c += 1) data[i * bands + c] = mode === '1' ? (chw[c * plane + i] ? 255 : 0) : chw[c * plane + i]!;
    return new RasterImage(mode as RasterMode, width, height, data, mode === 'P' ? { palette: new Uint8Array(768) } : {});
  };
  const cases = fixtureJson('image/resample.json') as any[];

  it(`matches ${cases.length} resize cases (all modes, NEAREST/BOX/BILINEAR/HAMMING/BICUBIC/LANCZOS, boxes, reducing_gap)`, () => {
    for (const c of cases) {
      const image = toImage(c.mode, c.size[0], c.size[1], decode(c.input));
      const label = `${c.mode} ${c.size}->${c.target} resample=${c.resample} box=${c.box} gap=${c.reducing_gap}`;
      if (c.error) {
        expect(() => image.resize(c.target, c.resample, c.box, c.reducing_gap), label).toThrow();
        continue;
      }
      const out = image.resize(c.target, c.resample, c.box, c.reducing_gap);
      expect([out.mode, out.width, out.height], label).toEqual([c.out_mode, ...c.out_size]);
      const raw = out.mode === '1' ? Uint8Array.from(out.data, (v) => (v ? 1 : 0)) : out.data;
      const plane = out.width * out.height;
      const chw = new (raw.constructor as new (n: number) => typeof raw)(raw.length);
      for (let i = 0; i < plane; i += 1) for (let b = 0; b < out.bands; b += 1) chw[b * plane + i] = raw[i * out.bands + b]!;
      expect(sha(new Uint8Array(chw.buffer)), label).toBe(c.sha256);
    }
  });

  it('exposes Pillow resampling codes and validates arguments', () => {
    expect(Resampling).toEqual({ NEAREST: 0, LANCZOS: 1, BILINEAR: 2, BICUBIC: 3, BOX: 4, HAMMING: 5 });
    const image = openImage(fixtureBytes('image/png_5x3_ct2_d8.png'));
    expect(() => image.resize([2, 2], 9)).toThrow(/Unknown resampling filter/);
    expect(() => image.resize([2, 2], Resampling.BOX, null, 0.5)).toThrow(/reducing_gap/);
    expect(image.resize([5, 3]).data).toEqual(image.data);
  });
});

describe('ViTImageProcessor input parity', () => {
  const fixture = fixtureJson('image/processor.json') as { configs: Record<string, object>; inputs: Record<string, any>; cases: any[] };
  const build = (spec: any): unknown => {
    if (spec.path) return `${directory}${spec.path}`;
    if (spec.string !== undefined) return spec.string;
    if (spec.bytes) return new Uint8Array(Buffer.from(spec.bytes, 'base64'));
    if (spec.list) return spec.list.map(build);
    if (spec.pil) {
      const image = openImage(fixtureBytes(`image/${spec.pil}`));
      return spec.convert ? image.convert(spec.convert) : image;
    }
    const { dtype, shape, data } = spec.tensor;
    const buffer = Buffer.from(data, 'base64');
    const values = new Float64Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
    return tensor(dtype === 'float32' ? Float32Array.from(values) : values, { shape, dtype });
  };
  const errorName = (error: unknown): string => {
    if (error instanceof NotImplementedError) return 'NotImplementedError';
    if (error instanceof ValueError) return 'ValueError';
    if (error instanceof TypeError) return 'TypeError';
    return 'RuntimeError';
  };

  it(`matches ${fixture.cases.length} (configuration, input) pairs`, () => {
    for (const c of fixture.cases) {
      const processor = new ImageProcessor(fixture.configs[c.config]);
      const label = `${c.config} / ${c.input}`;
      let result: Tensor;
      try {
        result = processor.preprocess(build(fixture.inputs[c.input]) as any).pixel_values;
      } catch (error) {
        expect(c.error, `${label}: ${(error as Error).message}`).toBeDefined();
        expect(errorName(error), label).toBe(c.error);
        continue;
      }
      expect(c.error, `${label} should raise ${c.error}: ${c.message}`).toBeUndefined();
      expect(result.shape, label).toEqual(c.shape);
      expect(sha(tensorBytes(result, c.dtype)), label).toBe(c.sha256);
    }
  });

  it('rejects empty batches and non-image values with Python\'s errors', () => {
    const processor = new ImageProcessor({ size: { height: 4, width: 4 } });
    expect(() => processor.preprocess([])).toThrow(/list index out of range/);
    expect(() => processor.preprocess([[]])).toThrow(/list index out of range/);
    expect(() => processor.preprocess(null as never)).toThrow(/type=<class 'NoneType'>/);
    expect(() => processor.preprocess(5 as never)).toThrow(/type=<class 'int'>/);
    expect(() => processor.preprocess(new Uint8Array([1, 2]) as never)).toThrow(/type=<class 'bytes'>/);
  });

  it('BOX and HAMMING raise PyTorch\'s NotImplementedError', () => {
    const image = tensor(new Uint8Array(3 * 4 * 4), { shape: [3, 4, 4], dtype: 'uint8' });
    expect(() => new ImageProcessor({ resample: 4, size: { height: 2, width: 2 } }).preprocess(image)).toThrow(NotImplementedError);
    expect(() => resizeImage(image, [2, 2], 'hamming')).toThrow(/got hamming/);
  });

  it('fetches URLs synchronously (blocking, like httpx.get) and asynchronously', async () => {
    const processor = new ImageProcessor({ size: { height: 4, width: 4 } });
    // The server runs on its own thread: the synchronous fetch blocks this one.
    const { Worker } = await import('node:worker_threads');
    const server = new Worker(`
      const { parentPort, workerData } = require('node:worker_threads');
      const { createServer } = require('node:http');
      const server = createServer((request, response) => {
        if (request.url === '/moved') { response.writeHead(302, { location: '/image.jpg' }); response.end(); return; }
        response.writeHead(200, { 'content-type': 'image/jpeg' });
        response.end(Buffer.from(workerData.image, 'base64'));
      });
      server.listen(0, '127.0.0.1', () => parentPort.postMessage(server.address().port));
      parentPort.on('message', () => { server.closeAllConnections(); server.close(() => process.exit(0)); });
    `, { eval: true, workerData: { image: readFileSync(`${directory}jpg_37x21_q95_s2.jpg`).toString('base64') } });
    const port = await new Promise<number>((resolve) => server.once('message', resolve));
    try {
      const fromPath = processor.preprocess(`${directory}jpg_37x21_q95_s2.jpg`).pixel_values;
      expect(processor.preprocess(`http://127.0.0.1:${port}/moved`).pixel_values.equal(fromPath)).toBe(true);
      expect((await processor.apreprocess(`http://127.0.0.1:${port}/image.jpg`)).pixel_values.equal(fromPath)).toBe(true);
      expect(loadImage(`http://127.0.0.1:${port}/image.jpg`).equal(decodeImage(`${directory}jpg_37x21_q95_s2.jpg`, { mode: 'RGB' }))).toBe(true);
    } finally {
      server.postMessage('close');
    }
    const fromPath = processor.preprocess(`${directory}jpg_37x21_q95_s2.jpg`).pixel_values;
    const fromAsync = (await processor.apreprocess(`${directory}jpg_37x21_q95_s2.jpg`)).pixel_values;
    expect(fromAsync.equal(fromPath)).toBe(true);
    expect(loadImage(`${directory}jpg_37x21_q95_s2.jpg`).equal(decodeImage(`${directory}jpg_37x21_q95_s2.jpg`, { mode: 'RGB' }))).toBe(true);
  });
});

describe('codec building blocks', () => {
  it('the bundled inflater matches node:zlib', async () => {
    const { deflateSync, deflateRawSync } = await import('node:zlib');
    const random = new Uint8Array(20000).map((_, i) => (i * 2654435761) >>> 24);
    const text = new TextEncoder().encode('tensorcode '.repeat(3000));
    for (const payload of [random, text, new Uint8Array(0)]) {
      for (const level of [0, 1, 6, 9]) {
        expect(inflateZlib(deflateSync(payload, { level }))).toEqual(payload);
        setNativeInflate(false);
        expect(inflateZlib(deflateSync(payload, { level }))).toEqual(payload);
        expect(inflateRaw(deflateRawSync(payload, { level }))).toEqual(payload);
        setNativeInflate(null);
      }
    }
  });

  it('PNG and every other format decode identically without node:zlib', () => {
    setNativeInflate(false);
    try {
      for (const name of names.filter((n) => n.endsWith('.png'))) {
        const decoded = decodeImage(fixtureBytes(`image/${name}`), { mode: 'RGB' });
        expect(sha(tensorBytes(decoded, expected[name]!.tv.dtype)), name).toBe(expected[name]!.tv.sha256);
      }
    } finally {
      setNativeInflate(null);
    }
  });

  it('decodes base64 like CPython 3.13 base64.decodebytes', () => {
    const text = (bytes: Uint8Array): string => new TextDecoder('latin1').decode(bytes);
    expect(text(decodeBase64('aGVsbG8='))).toBe('hello');
    expect(text(decodeBase64('aGV-sbG8='))).toBe('hello');
    expect(text(decodeBase64('aGVsbA==ZQ=='))).toBe('hell\x06P');
    expect(text(decodeBase64('=aGVs'))).toBe('hel');
    expect(() => decodeBase64('aGVsbG8')).toThrow(/Incorrect padding/);
    expect(() => decodeBase64('a')).toThrow(/number of data characters \(1\)/);
    expect(() => decodeBase64('aGVsbG8=eA')).toThrow(/\(9\)/);
  });

  it('glibc sin/cos/sinf and FMA reproduce the C results', () => {
    // Values computed with CPython's math (glibc 2.39) and ctypes libm.sinf.
    expect(glibcSin(1)).toBe(0.8414709848078965);
    expect(glibcSin(2.5)).toBe(0.5984721441039565);
    expect(glibcCos(7.25)).toBe(0.5679241732886948);
    expect(glibcSinf(Math.fround(1.5))).toBe(0.9974949955940247);
    expect(fma(0.1, 10, -1)).toBe(5.551115123125783e-17);
  });
});
