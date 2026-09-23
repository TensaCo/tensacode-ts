"""Image decoding fixtures: encoded files plus what Python decodes from them.

Writes ``test/fixtures/image/*`` (small encoded PNG/JPEG/GIF/WebP/BMP files)
and ``test/fixtures/image/expected.json``. For every file it records

* ``tv``: ``torchvision.io.decode_image(path, mode=RGB)`` (what transformers'
  torchvision image processors decode path/base64/URL inputs with), also with
  ``apply_exif_orientation=True`` and in ``UNCHANGED``/``GRAY``/``RGB_ALPHA``
  modes;
* ``pil``: ``Image.open(path)`` mode/size/frames, ``convert('RGB')`` and
  ``ImageOps.exif_transpose`` followed by ``convert('RGB')`` (what
  transformers' PIL path and ``LocalModel`` decode).

Pixel buffers are stored as SHA-256 digests of the C-contiguous array bytes
(little endian); ``resample.json`` stores PIL resize outputs in full.

Run ``python scripts/fixtures/image_fixtures.py [--stress DIR]``; ``--stress``
writes a larger uncommitted corpus with the same layout for local checks.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'test' / 'fixtures' / 'image'


# ---------------------------------------------------------------------------
# Synthetic content.
# ---------------------------------------------------------------------------

def photo(width: int, height: int, seed: int, noise: float = 12.0) -> np.ndarray:
    """A photo-like RGB image: gradients, shapes, texture and noise."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 128 + 100 * np.sin(x / max(width, 1) * 3.1 + seed)
    g = 128 + 100 * np.cos(y / max(height, 1) * 2.3 + seed * 0.7)
    b = 128 + 90 * np.sin((x + y) / max(width + height, 1) * 5.0)
    image = np.stack([r, g, b], -1) + rng.normal(0, noise, (height, width, 3))
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(pil)
    for _ in range(3):
        x0, y0 = rng.randint(0, max(width, 1)), rng.randint(0, max(height, 1))
        x1, y1 = x0 + rng.randint(1, max(width // 2, 2)), y0 + rng.randint(1, max(height // 2, 2))
        draw.ellipse([x0, y0, x1, y1], fill=tuple(int(v) for v in rng.randint(0, 256, 3)))
    draw.line([0, 0, width, height], fill=(255, 255, 255), width=1)
    return np.asarray(pil)


def noise(width: int, height: int, channels: int, seed: int, high: int = 256) -> np.ndarray:
    rng = np.random.RandomState(seed)
    shape = (height, width) if channels == 1 else (height, width, channels)
    return rng.randint(0, high, shape).astype(np.uint16 if high > 256 else np.uint8)


# ---------------------------------------------------------------------------
# A small PNG writer covering every colour type, bit depth, filter and interlace.
# ---------------------------------------------------------------------------

ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)


def _pack_row(samples: np.ndarray, depth: int) -> bytes:
    if depth == 16:
        return samples.astype('>u2').tobytes()
    if depth == 8:
        return samples.astype(np.uint8).tobytes()
    per = 8 // depth
    values = list(samples.reshape(-1))
    out = bytearray()
    for start in range(0, len(values), per):
        byte = 0
        group = values[start:start + per]
        for index, value in enumerate(group):
            byte |= int(value) << (8 - depth * (index + 1))
        out.append(byte)
    return bytes(out)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _filter(rows: list[bytes], bpp: int, rng: random.Random, mode: str) -> bytes:
    out = bytearray()
    previous = bytes(len(rows[0])) if rows else b''
    for row in rows:
        kind = rng.randint(0, 4) if mode == 'random' else int(mode)
        filtered = bytearray(len(row))
        for i, value in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b = previous[i]
            c = previous[i - bpp] if i >= bpp else 0
            predictor = [0, a, b, (a + b) // 2, _paeth(a, b, c)][kind]
            filtered[i] = (value - predictor) & 0xFF
        out.append(kind)
        out += filtered
        previous = row
    return bytes(out)


def write_png(path: Path, samples: np.ndarray, color_type: int, depth: int, *, interlace: bool = False,
              palette: list[tuple[int, int, int]] | None = None, trns: bytes | None = None, filters: str = 'random',
              exif_orientation: int | None = None, split: int = 0, seed: int = 0, extra: list[tuple[bytes, bytes]] = ()) -> None:
    """``samples`` is ``[H, W]`` or ``[H, W, C]`` integer samples at ``depth`` bits."""
    height, width = samples.shape[:2]
    channels = CHANNELS[color_type]
    samples = samples.reshape(height, width, channels)
    bpp = max(1, channels * depth // 8)
    rng = random.Random(seed)

    def encode(image: np.ndarray) -> bytes:
        rows = [_pack_row(image[y].reshape(-1), depth) for y in range(image.shape[0])]
        return _filter(rows, bpp, rng, filters) if image.shape[1] else b''

    if interlace:
        raw = b''
        for x0, y0, dx, dy in ADAM7:
            sub = samples[y0::dy, x0::dx]
            if sub.shape[0] and sub.shape[1]:
                raw += encode(sub)
    else:
        raw = encode(samples)
    data = zlib.compress(raw, 9)
    body = b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, depth, color_type, 0, 0, int(interlace)))
    for kind, payload in extra:
        body += _chunk(kind, payload)
    if palette is not None:
        body += _chunk(b'PLTE', bytes(v for rgb in palette for v in rgb))
    if trns is not None:
        body += _chunk(b'tRNS', trns)
    if exif_orientation is not None:
        body += _chunk(b'eXIf', _tiff_orientation(exif_orientation))
    if split:
        for start in range(0, len(data), split):
            body += _chunk(b'IDAT', data[start:start + split])
    else:
        body += _chunk(b'IDAT', data)
    body += _chunk(b'IEND', b'')
    path.write_bytes(body)


def _tiff_orientation(value: int, big_endian: bool = False) -> bytes:
    order = '>' if big_endian else '<'
    header = (b'MM' if big_endian else b'II') + struct.pack(order + 'HI', 42, 8)
    ifd = struct.pack(order + 'H', 1) + struct.pack(order + 'HHIHH', 0x0112, 3, 1, value, 0) + struct.pack(order + 'I', 0)
    return header + ifd


# ---------------------------------------------------------------------------
# Corpus.
# ---------------------------------------------------------------------------

def png_corpus(out: Path, sizes: list[tuple[int, int]]) -> list[str]:
    names = []
    seed = 0
    for width, height in sizes:
        for color_type, depths in [(0, [1, 2, 4, 8, 16]), (2, [8, 16]), (3, [1, 2, 4, 8]), (4, [8, 16]), (6, [8, 16])]:
            for depth in depths:
                for interlace in (False, True):
                    seed += 1
                    channels = CHANNELS[color_type]
                    kwargs = {}
                    if color_type == 3:
                        count = 1 << depth
                        rng = np.random.RandomState(seed)
                        kwargs['palette'] = [tuple(int(v) for v in rng.randint(0, 256, 3)) for _ in range(count)]
                        if seed % 2:
                            kwargs['trns'] = bytes(int(v) for v in rng.randint(0, 256, min(count, 5)))
                        samples = noise(width, height, 1, seed, count)
                    else:
                        samples = noise(width, height, channels, seed, 1 << depth)
                        if color_type in (0, 2) and seed % 3 == 0:
                            first = samples.reshape(height, width, channels)[0, 0]
                            kwargs['trns'] = b''.join(struct.pack('>H', int(v)) for v in np.atleast_1d(first))
                    name = f'png_{width}x{height}_ct{color_type}_d{depth}{"_i" if interlace else ""}.png'
                    write_png(out / name, samples, color_type, depth, interlace=interlace, seed=seed,
                              split=97 if seed % 4 == 0 else 0, **kwargs)
                    names.append(name)
    return names


def png_extra(out: Path) -> list[str]:
    names = []
    base = photo(19, 13, 3)
    for orientation in range(1, 9):
        name = f'png_exif_{orientation}.png'
        write_png(out / name, base, 2, 8, exif_orientation=orientation, seed=orientation)
        names.append(name)
    for kind in range(5):
        name = f'png_filter{kind}.png'
        write_png(out / name, base, 2, 8, filters=str(kind))
        names.append(name)
    name = 'png_chunks.png'
    write_png(out / name, base, 2, 8, extra=[(b'gAMA', struct.pack('>I', 45455)), (b'tEXt', b'Comment\x00hello'),
                                              (b'sRGB', b'\x00'), (b'pHYs', struct.pack('>IIB', 2835, 2835, 1))])
    names.append(name)
    gamma = lambda value: (b'gAMA', struct.pack('>I', value))
    for label, color_type, depth, extra in [
        ('gamma_rgb', 2, 8, [gamma(55556)]), ('gamma_srgb', 2, 8, [(b'sRGB', b'\x00')]),
        ('gamma_both', 6, 8, [gamma(80000), (b'sRGB', b'\x01')]), ('gamma_srgb_first', 2, 8, [(b'sRGB', b'\x00'), gamma(100000)]),
        ('gamma_linear', 2, 8, [gamma(100000)]), ('gamma_rgb16', 2, 16, [gamma(45455)]),
        ('gamma_rgba16_sbit', 6, 16, [(b'sBIT', bytes([12, 11, 12, 16])), gamma(38000)]), ('gamma_rgb16_sbit5', 2, 16, [(b'sBIT', bytes([5, 5, 5])), gamma(45455)]),
    ]:
        samples = noise(17, 11, CHANNELS[color_type], 70 + len(names), 1 << depth)
        samples.reshape(11, 17, -1)[:3, :4, :3] = samples.reshape(11, 17, -1)[:3, :4, :1]  # some gray pixels
        name = f'png_{label}.png'
        write_png(out / name, samples, color_type, depth, extra=extra, seed=len(names))
        names.append(name)
    name = 'png_gamma_palette.png'
    write_png(out / name, noise(9, 7, 1, 77, 16), 3, 4, palette=[tuple(int(v) for v in noise(3, 1, 1, 78 + i).reshape(-1)) for i in range(16)],
              trns=bytes([0, 128, 255]), extra=[gamma(45455)])
    names.append(name)
    for mode in ('RGB', 'RGBA', 'L', 'LA', 'P', '1', 'I;16', 'PA'):
        image = Image.fromarray(photo(21, 17, 5))
        if mode == 'RGBA':
            image = image.convert('RGBA')
            image.putalpha(Image.fromarray(noise(21, 17, 1, 9)))
        elif mode == 'LA':
            image = image.convert('LA')
        elif mode == 'P':
            image = image.convert('P', palette=Image.Palette.ADAPTIVE, colors=37)
            image.info['transparency'] = 3
        elif mode == 'PA':
            image = image.convert('P', palette=Image.Palette.ADAPTIVE, colors=16)
        elif mode == 'I;16':
            image = Image.fromarray((noise(21, 17, 1, 11).astype(np.uint16) * 257).astype(np.uint16))
        elif mode != 'RGB':
            image = image.convert(mode)
        name = f'pil_{mode.replace(";", "")}.png'
        image.save(out / name, optimize=True, transparency=image.info.get('transparency')) if mode == 'P' else image.save(out / name, optimize=True)
        names.append(name)
    return names


def jpeg_corpus(out: Path, sizes: list[tuple[int, int]], qualities: list[int]) -> list[str]:
    names = []
    seed = 100
    for width, height in sizes:
        for quality in qualities:
            for subsampling in (0, 1, 2):
                for progressive in (False, True):
                    seed += 1
                    image = Image.fromarray(photo(width, height, seed))
                    name = f'jpg_{width}x{height}_q{quality}_s{subsampling}{"_p" if progressive else ""}.jpg'
                    image.save(out / name, quality=quality, subsampling=subsampling, progressive=progressive,
                               optimize=seed % 3 == 0)
                    names.append(name)
    return names


def jpeg_extra(out: Path) -> list[str]:
    names = []
    base = Image.fromarray(photo(37, 29, 7))
    for orientation in range(1, 9):
        exif = Image.Exif()
        exif[0x0112] = orientation
        name = f'jpg_exif_{orientation}.jpg'
        base.save(out / name, quality=85, exif=exif.tobytes())
        names.append(name)
    variants = {
        'jpg_gray.jpg': (base.convert('L'), {'quality': 80}),
        'jpg_gray_p.jpg': (base.convert('L'), {'quality': 80, 'progressive': True}),
        'jpg_cmyk.jpg': (base.convert('CMYK'), {'quality': 90}),
        'jpg_cmyk_p.jpg': (base.convert('CMYK'), {'quality': 90, 'progressive': True}),
        'jpg_keep_rgb.jpg': (base, {'quality': 90, 'keep_rgb': True, 'subsampling': 0}),
        'jpg_restart_blocks.jpg': (base, {'quality': 75, 'restart_marker_blocks': 3}),
        'jpg_restart_rows.jpg': (base, {'quality': 75, 'restart_marker_rows': 1, 'subsampling': 0}),
        'jpg_restart_p.jpg': (base, {'quality': 75, 'restart_marker_blocks': 2, 'progressive': True}),
        'jpg_1x1.jpg': (Image.fromarray(photo(1, 1, 3)), {'quality': 90}),
        'jpg_2x2_s2.jpg': (Image.fromarray(photo(2, 2, 3)), {'quality': 90, 'subsampling': 2}),
        'jpg_3x1_s2.jpg': (Image.fromarray(photo(3, 1, 3)), {'quality': 90, 'subsampling': 2}),
        'jpg_noise_q100.jpg': (Image.fromarray(noise(24, 16, 3, 5)), {'quality': 100, 'subsampling': 0}),
        'jpg_noise_q100_s2.jpg': (Image.fromarray(noise(24, 16, 3, 6)), {'quality': 100, 'subsampling': 2}),
        'jpg_noise_q5.jpg': (Image.fromarray(noise(24, 16, 3, 7)), {'quality': 5}),
        'jpg_smooth.jpg': (base, {'quality': 60, 'smooth': 50}),
        'jpg_icc.jpg': (base, {'quality': 70, 'icc_profile': b'\x00' * 200, 'comment': b'hello'}),
    }
    for name, (image, options) in variants.items():
        image.save(out / name, **options)
        names.append(name)
    # ImageMagick: sampling factors PIL cannot write, and arithmetic coding.
    source = out / '_im_source.png'
    base.save(source)
    for label, factor in [('411', '4x1'), ('440', '1x2'), ('h2v2', '2x2'), ('h1v4', '1x4'), ('h4v2', '4x2')]:
        for progressive in (False, True):
            name = f'jpg_im_{label}{"_p" if progressive else ""}.jpg'
            command = ['convert', str(source), '-quality', '82', '-sampling-factor', factor]
            if progressive:
                command += ['-interlace', 'JPEG']
            subprocess.run(command + [str(out / name)], check=True)
            names.append(name)
    for name, extra in [('jpg_im_arith.jpg', ['-define', 'jpeg:arithmetic-coding=true']),
                        ('jpg_im_gray_411.jpg', ['-colorspace', 'Gray'])]:
        try:
            subprocess.run(['convert', str(source), '-quality', '82', *extra, str(out / name)], check=True, capture_output=True)
            names.append(name)
        except subprocess.CalledProcessError:
            pass
    source.unlink()
    return names


def gif_corpus(out: Path) -> list[str]:
    names = []
    base = Image.fromarray(photo(23, 17, 4))
    variants = {
        'gif_p.gif': (base.convert('P', palette=Image.Palette.ADAPTIVE, colors=64), {}),
        'gif_p_nointerlace.gif': (base.convert('P', palette=Image.Palette.ADAPTIVE, colors=64), {'interlace': False}),
        'gif_l.gif': (base.convert('L'), {}),
        'gif_trns.gif': (base.convert('P', palette=Image.Palette.ADAPTIVE, colors=16), {'transparency': 2}),
        'gif_bw.gif': (base.convert('1'), {}),
        'gif_rgb.gif': (base, {}),
    }
    for name, (image, options) in variants.items():
        image.save(out / name, **options)
        names.append(name)
    frames = [Image.fromarray(photo(23, 17, seed)).convert('P', palette=Image.Palette.ADAPTIVE, colors=32) for seed in range(4)]
    for disposal in (0, 1, 2, 3):
        name = f'gif_anim_d{disposal}.gif'
        frames[0].save(out / name, save_all=True, append_images=frames[1:], duration=50, loop=0, disposal=disposal,
                       transparency=0, optimize=False)
        names.append(name)
    name = 'gif_anim_optimized.gif'
    frames[0].save(out / name, save_all=True, append_images=frames[1:], duration=[30, 40, 50, 60], loop=0, optimize=True)
    names.append(name)
    # Frames smaller than the canvas and a missing global palette, written by hand.
    name = 'gif_offset.gif'
    (out / name).write_bytes(_hand_gif())
    names.append(name)
    return names


def _lzw(indices: list[int], min_code: int) -> bytes:
    clear, end = 1 << min_code, (1 << min_code) + 1
    size = min_code + 1
    table = {bytes([i]): i for i in range(clear)}
    next_code = end + 1
    bits = []

    def emit(code: int) -> None:
        bits.extend((code >> i) & 1 for i in range(size))

    emit(clear)
    current = b''
    for value in indices:
        candidate = current + bytes([value])
        if candidate in table:
            current = candidate
            continue
        emit(table[current])
        if next_code < 4096:
            table[candidate] = next_code
            next_code += 1
            if next_code > (1 << size) and size < 12:
                size += 1
        else:
            emit(clear)
            table = {bytes([i]): i for i in range(clear)}
            next_code = end + 1
            size = min_code + 1
        current = bytes([value])
    if current:
        emit(table[current])
    emit(end)
    data = bytearray()
    for start in range(0, len(bits), 8):
        data.append(sum(bit << i for i, bit in enumerate(bits[start:start + 8])))
    blocks = bytearray([min_code])
    for start in range(0, len(data), 255):
        chunk = data[start:start + 255]
        blocks += bytes([len(chunk)]) + chunk
    return bytes(blocks + b'\x00')


def _hand_gif() -> bytes:
    rng = np.random.RandomState(3)
    width, height = 20, 14
    gct = rng.randint(0, 256, (8, 3)).astype(np.uint8)
    data = b'GIF89a' + struct.pack('<HHBBB', width, height, 0x80 | 0x70 | 2, 5, 0) + gct.tobytes()
    data += b'!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00'
    frames = [(2, 3, 9, 6, 1, None, 1), (5, 1, 12, 10, 2, 4, 0), (0, 0, 20, 14, 3, None, 2), (7, 7, 6, 5, 0, 2, 0)]
    for index, (left, top, fw, fh, disposal, transparent, local) in enumerate(frames):
        packed = (disposal << 2) | (1 if transparent is not None else 0)
        data += b'!\xf9\x04' + struct.pack('<BHB', packed, 10, transparent or 0) + b'\x00'
        pixels = rng.randint(0, 8, fh * fw).tolist()
        flags = 0
        table = b''
        if local:
            flags = 0x80 | 2
            table = rng.randint(0, 256, (8, 3)).astype(np.uint8).tobytes()
            if local == 2:
                flags |= 0x40  # interlaced
                rows = [pixels[r * fw:(r + 1) * fw] for r in range(fh)]
                order = list(range(0, fh, 8)) + list(range(4, fh, 8)) + list(range(2, fh, 4)) + list(range(1, fh, 2))
                pixels = [value for r in order for value in rows[r]]
        data += b',' + struct.pack('<HHHHB', left, top, fw, fh, flags) + table + _lzw(pixels, 3)
    return data + b';'


def webp_corpus(out: Path) -> list[str]:
    names = []
    for width, height in [(1, 1), (7, 5), (23, 17), (40, 33)]:
        base = Image.fromarray(photo(width, height, width))
        for quality in (5, 50, 90, 100):
            name = f'webp_{width}x{height}_q{quality}.webp'
            base.save(out / name, quality=quality, method=4)
            names.append(name)
        name = f'webp_{width}x{height}_lossless.webp'
        base.save(out / name, lossless=True)
        names.append(name)
        rgba = base.convert('RGBA')
        rgba.putalpha(Image.fromarray(noise(width, height, 1, 4)))
        for suffix, options in [('alpha_lossy', {'quality': 70}), ('alpha_lossless', {'lossless': True}),
                                ('alpha_exact', {'lossless': True, 'exact': True})]:
            name = f'webp_{width}x{height}_{suffix}.webp'
            rgba.save(out / name, **options)
            names.append(name)
    noisy = Image.fromarray(noise(29, 21, 3, 8))
    for method in (0, 6):
        name = f'webp_noise_m{method}.webp'
        noisy.save(out / name, quality=75, method=method)
        names.append(name)
        name = f'webp_noise_lossless_m{method}.webp'
        noisy.save(out / name, lossless=True, method=method, quality=100 if method else 0)
        names.append(name)
    palette = Image.fromarray(photo(33, 25, 9)).convert('P', palette=Image.Palette.ADAPTIVE, colors=11).convert('RGB')
    name = 'webp_palette_lossless.webp'
    palette.save(out / name, lossless=True)
    names.append(name)
    frames = [Image.fromarray(photo(18, 14, seed)) for seed in range(3)]
    name = 'webp_anim.webp'
    frames[0].save(out / name, save_all=True, append_images=frames[1:], duration=40, loop=0, quality=80)
    names.append(name)
    return names


def bmp_corpus(out: Path) -> list[str]:
    names = []
    base = Image.fromarray(photo(19, 13, 6))
    for mode in ('RGB', 'L', 'P', '1', 'RGBA'):
        image = base.convert(mode) if mode != 'P' else base.convert('P', palette=Image.Palette.ADAPTIVE, colors=20)
        name = f'bmp_{mode}.bmp'
        image.save(out / name)
        names.append(name)
    source = out / '_im_source.png'
    base.save(source)
    for name, options in [('bmp_im_rle8.bmp', ['-type', 'Palette', '-compress', 'RLE']),
                          ('bmp_im_565.bmp', ['-define', 'bmp:subtype=RGB565']),
                          ('bmp_im_555.bmp', ['-define', 'bmp:subtype=RGB555']),
                          ('bmp_im_bmp3.bmp', []),
                          ('bmp_im_argb.bmp', ['-alpha', 'set', '-define', 'bmp:format=bmp4'])]:
        target = ('BMP3:' if name == 'bmp_im_bmp3.bmp' else '') + str(out / name)
        try:
            subprocess.run(['convert', str(source), *options, target], check=True, capture_output=True)
            names.append(name)
        except subprocess.CalledProcessError:
            pass
    source.unlink()
    return names


# ---------------------------------------------------------------------------
# Expectations.
# ---------------------------------------------------------------------------

def digest(array) -> dict:
    array = np.ascontiguousarray(np.asarray(array))
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)  # PIL "1" arrays may hold 255 bytes; digest truth values.
    if array.dtype.byteorder == '>':
        array = array.astype(array.dtype.newbyteorder('<'))
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def chw(array) -> np.ndarray:
    """PIL ``HWC`` arrays in ``pil_to_tensor`` layout (``CHW``; 2-D stays ``[1, H, W]``)."""
    array = np.asarray(array)
    return array[None] if array.ndim == 2 else array.transpose(2, 0, 1)


def outcome(function):
    try:
        return function()
    except Exception as error:  # noqa: BLE001 - the error is the expectation
        return {'error': type(error).__name__, 'message': str(error)[:300]}


def tv_record(path: Path, mode, exif: bool = False):
    import torch
    from torchvision.io import decode_image

    def run():
        value = decode_image(str(path), mode=mode, apply_exif_orientation=exif)
        if value.dtype == torch.uint16:
            return digest(value.to(torch.int32).numpy().astype(np.uint16))
        return digest(value.numpy())
    return outcome(run)


def pil_record(path: Path) -> dict:
    def run():
        with Image.open(path) as image:
            record = {'mode': image.mode, 'size': list(image.size), 'frames': getattr(image, 'n_frames', 1),
                      'format': image.format}
            image.load()
            record['raw'] = outcome(lambda: digest(chw(image)))
            record['rgb'] = digest(chw(image.convert('RGB')))
            transposed = ImageOps.exif_transpose(image)
            record['exif_rgb'] = digest(chw(transposed.convert('RGB')))
            record['orientation'] = image.getexif().get(0x0112)
            return record
    return outcome(run)


def expectations(out: Path, names: list[str]) -> dict:
    from torchvision.io import ImageReadMode
    records = {}
    for name in names:
        path = out / name
        records[name] = {
            'tv': tv_record(path, ImageReadMode.RGB),
            'tv_exif': tv_record(path, ImageReadMode.RGB, True),
            'tv_unchanged': tv_record(path, ImageReadMode.UNCHANGED),
            'tv_gray': tv_record(path, ImageReadMode.GRAY),
            'tv_gray_alpha': tv_record(path, ImageReadMode.GRAY_ALPHA),
            'tv_rgba': tv_record(path, ImageReadMode.RGB_ALPHA),
            'pil': pil_record(path),
        }
    return records


def _raw(image: Image.Image) -> dict:
    array = chw(image)
    dtype = {'1': 'uint8', 'I': 'int32', 'F': 'float32', 'I;16': 'uint16'}.get(image.mode, 'uint8')
    array = np.ascontiguousarray(array.astype('<' + np.dtype(dtype).str[1:] if dtype != 'uint8' else np.uint8))
    return {'dtype': dtype, 'bytes': base64.b64encode(array.tobytes()).decode()}


def resample_cases(stress: bool = False) -> list[dict]:
    """Pillow ``Image.resize`` over every mode, filter, boxes and ``reducing_gap``.

    Inputs are stored as base64 ``CHW`` bytes; outputs as SHA-256 digests of
    the ``CHW`` bytes (full data too in the stress corpus).
    """
    rng = np.random.RandomState(42)
    cases = []
    sizes = [((9, 7), (4, 3)), ((9, 7), (13, 11)), ((16, 5), (5, 16))]
    boxes = [None, (1.5, 0.25, 7.75, 6.0)]
    if stress:
        sizes += [((6, 5), (6, 9)), ((64, 48), (17, 13)), ((31, 29), (97, 5)), ((5, 600), (3, 7)), ((200, 3), (7, 2)), ((40, 40), (40, 40))]
        boxes += [(1, 1, 5, 4), (0.5, 0.5, 3.25, 2.0)]
    modes = ['L', 'RGB', 'RGBA', 'LA', 'I', 'F', 'I;16', 'P', '1', 'CMYK']
    for mode in modes:
        for (source, target) in sizes:
            base = rng.randint(0, 256, (source[1], source[0], 4)).astype(np.uint8)
            if mode in ('I', 'I;16'):
                image = Image.fromarray(rng.randint(0, 60000, (source[1], source[0])).astype(np.int32 if mode == 'I' else np.uint16))
                if mode == 'I':
                    image = image.convert('I')
            elif mode == 'F':
                image = Image.fromarray((rng.rand(source[1], source[0]) * 300 - 20).astype(np.float32), 'F')
            else:
                image = Image.fromarray(base, 'RGBA').convert(mode)
            for resample in range(6):
                for box in boxes:
                    if box is not None and (box[2] > source[0] or box[3] > source[1]):
                        continue
                    for gap in ((None, 2.0, 1.5) if stress else (None, 2.0)) if box is None else (None,):
                        case = {'mode': mode, 'size': list(image.size), 'input': _raw(image), 'target': list(target),
                                'resample': resample, 'box': list(box) if box else None, 'reducing_gap': gap}
                        try:
                            out = image.resize(target, resample, box, gap)
                            raw = _raw(out)
                            case.update({'out_mode': out.mode, 'out_size': list(out.size), 'out_dtype': raw['dtype'],
                                         'sha256': hashlib.sha256(base64.b64decode(raw['bytes'])).hexdigest()})
                            if stress:
                                case['out_bytes'] = raw['bytes']
                        except Exception as error:  # noqa: BLE001 - the error is the expectation
                            case.update({'error': type(error).__name__, 'message': str(error)})
                        cases.append(case)
    return cases


PROCESSOR_CONFIGS = {
    'small': {'size': {'height': 8, 'width': 8}},
    'crop_pad': {'size': {'shortest_edge': 7}, 'do_center_crop': True, 'crop_size': {'height': 9, 'width': 10}},
    'crop': {'size': {'shortest_edge': 12}, 'do_center_crop': True, 'crop_size': {'height': 9, 'width': 10}},
    'aspect': {'size': {'shortest_edge': 5, 'longest_edge': 7}},
    'max_hw': {'size': {'max_height': 6, 'max_width': 9}},
    'rescale_only': {'size': {'height': 6, 'width': 5}, 'do_normalize': False},
    'normalize_only': {'size': {'height': 6, 'width': 5}, 'do_rescale': False, 'image_mean': [0.2, 0.4, 0.6], 'image_std': [0.3, 0.5, 0.7]},
    'raw': {'do_resize': False, 'do_rescale': False, 'do_normalize': False},
    'pad': {'size': {'shortest_edge': 6}, 'do_pad': True, 'pad_size': {'height': 12, 'width': 13}},
    'pad_missing': {'size': {'height': 4, 'width': 4}, 'do_pad': True},
    'box': {'size': {'height': 5, 'width': 5}, 'resample': 4},
    'hamming': {'size': {'height': 5, 'width': 5}, 'resample': 5},
    'lanczos': {'size': {'height': 7, 'width': 9}, 'resample': 1},
    'bicubic': {'size': {'height': 11, 'width': 6}, 'resample': 3},
    'nearest': {'size': {'height': 5, 'width': 7}, 'resample': 0},
    'convert_rgb': {'size': {'height': 6, 'width': 6}, 'do_convert_rgb': True},
    'gray_mean': {'size': {'height': 6, 'width': 6}, 'image_mean': [0.5], 'image_std': [0.25]},
}


def processor_cases(out: Path) -> dict:
    """``ViTImageProcessor(**config)(images=...)`` over every input kind TypeScript accepts."""
    import torch
    from transformers import ViTImageProcessor

    rng = np.random.RandomState(9)
    jpg = out / 'jpg_37x21_q95_s2.jpg'
    inputs: dict[str, tuple[dict, object]] = {}

    def tensor_spec(value):
        array = value.numpy()
        dtype = str(value.dtype).removeprefix('torch.')
        return {'tensor': {'dtype': dtype, 'shape': list(array.shape),
                           'data': base64.b64encode(np.ascontiguousarray(array.astype(np.float64)).tobytes()).decode()}}

    for name in ['jpg_37x21_q95_s2.jpg', 'png_13x9_ct2_d16.png', 'png_13x9_ct0_d2.png', 'gif_anim_d1.gif',
                 'webp_23x17_alpha_lossy.webp', 'png_exif_6.png', 'jpg_cmyk.jpg', 'bmp_RGB.bmp']:
        inputs[f'path:{name}'] = ({'path': name}, str(out / name))
    raw = (out / 'png_5x3_ct6_d8.png').read_bytes()
    inputs['base64'] = ({'string': base64.b64encode(raw).decode()}, base64.b64encode(raw).decode())
    inputs['data_uri'] = ({'string': 'data:image/jpeg;base64,' + base64.b64encode(jpg.read_bytes()).decode()},
                          'data:image/jpeg;base64,' + base64.b64encode(jpg.read_bytes()).decode())
    inputs['bad_string'] = ({'string': 'not_a_file.png'}, 'not_a_file.png')
    uint8 = torch.from_numpy(rng.randint(0, 256, (3, 9, 11)).astype(np.uint8))
    tensors = {
        'uint8_chw': uint8,
        'uint8_hwc': uint8.permute(1, 2, 0).contiguous(),
        'float32_chw': torch.from_numpy(rng.rand(3, 9, 11).astype(np.float32)),
        'float64_chw': torch.from_numpy(rng.rand(3, 7, 5)),
        'int32_chw': torch.from_numpy(rng.randint(0, 300, (3, 9, 11)).astype(np.int32)),
        'one_channel': uint8[:1].contiguous(),
        'batch': torch.stack([uint8, uint8.flip(-1)]),
        'two_d': uint8[0].contiguous(),
        'four_channel': torch.cat([uint8, uint8[:1]]),
    }
    for name, value in tensors.items():
        inputs[f'tensor:{name}'] = (tensor_spec(value), value)
    pil_sources = {'rgb_jpg': ('jpg_37x21_q95_s2.jpg', None), 'l_png': ('png_13x9_ct0_d8.png', None), 'p_gif': ('gif_p.gif', None),
                   'rgba_webp': ('webp_23x17_alpha_lossy.webp', None), 'i16_png': ('png_13x9_ct0_d16.png', None),
                   'cmyk_jpg': ('jpg_cmyk.jpg', None), 'converted_l': ('jpg_37x21_q95_s2.jpg', 'L'), 'bit_png': ('png_13x9_ct0_d1.png', None)}
    for name, (file, mode) in pil_sources.items():
        image = Image.open(out / file)
        image.load()
        if mode:
            image = image.convert(mode)
        inputs[f'pil:{name}'] = ({'pil': file, 'convert': mode}, image)
    inputs['list:paths'] = ({'list': [{'path': 'jpg_37x21_q95_s2.jpg'}, {'path': 'png_exif_6.png'}]},
                            [str(out / 'jpg_37x21_q95_s2.jpg'), str(out / 'png_exif_6.png')])
    inputs['list:mixed'] = ({'list': [tensor_spec(uint8), {'path': 'jpg_37x21_q95_s2.jpg'}]}, [uint8, str(jpg)])
    inputs['list:nested'] = ({'list': [{'list': [tensor_spec(uint8)]}, {'list': [{'path': 'png_exif_6.png'}]}]},
                             [[uint8], [str(out / 'png_exif_6.png')]])
    inputs['bytes'] = ({'bytes': base64.b64encode(raw).decode()}, raw)

    cases = []
    for config_name, config in PROCESSOR_CONFIGS.items():
        for input_name, (spec, value) in inputs.items():
            case = {'config': config_name, 'input': input_name}
            try:
                processor = ViTImageProcessor(**config)
                result = processor(images=value, return_tensors='pt')['pixel_values']
                array = result.numpy()
                dtype = str(result.dtype).removeprefix('torch.')
                if array.dtype == np.bool_:
                    array = array.astype(np.uint8)  # PIL "1" data may hold 255 bytes; digest truth values.
                case.update({'shape': list(array.shape), 'dtype': dtype,
                             'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()})
            except Exception as error:  # noqa: BLE001 - the error is the expectation
                case.update({'error': type(error).__name__, 'message': str(error)[:300]})
            cases.append(case)
    return {'configs': PROCESSOR_CONFIGS, 'inputs': {name: spec for name, (spec, _) in inputs.items()}, 'cases': cases}


def libjpeg_corpus(out: Path, tools: Path, stress: bool) -> list[str]:
    """Arithmetic-coded and lossless JPEGs written by libjpeg-turbo's ``cjpeg``/``jpegtran``.

    Pillow cannot write these processes; build libjpeg-turbo 3.x with
    ``-DWITH_ARITH_ENC=1`` and point ``LIBJPEG_TURBO_BIN`` at the directory
    holding ``cjpeg``/``jpegtran`` (``-static`` suffixes are accepted).
    """
    def tool(name):
        for candidate in (tools / name, tools / f'{name}-static'):
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(name)
    cjpeg, jpegtran = tool('cjpeg'), tool('jpegtran')
    names = []
    sizes = [(37, 29), (16, 16)] + ([(65, 47), (9, 7), (120, 80)] if stress else [])
    for index, (width, height) in enumerate(sizes):
        rgb = out / '_src.ppm'
        gray = out / '_src.pgm'
        Image.fromarray(photo(width, height, 40 + index)).save(rgb)
        Image.fromarray(photo(width, height, 50 + index)).convert('L').save(gray)
        variants = [
            ('arith', ['-arithmetic', '-quality', '80']),
            ('arith_p', ['-arithmetic', '-progressive', '-quality', '75']),
            ('arith_444', ['-arithmetic', '-sample', '1x1', '-quality', '90']),
            ('arith_422_p', ['-arithmetic', '-progressive', '-sample', '2x1', '-quality', '60']),
            ('arith_440', ['-arithmetic', '-sample', '1x2', '-quality', '85']),
            ('arith_restart', ['-arithmetic', '-restart', '2B', '-quality', '70']),
            ('arith_restart_p', ['-arithmetic', '-progressive', '-restart', '3B', '-quality', '70']),
            ('lossless1', ['-lossless', '1']),
            ('lossless4', ['-lossless', '4']),
            ('lossless5_pt2', ['-lossless', '5,2']),
            ('lossless7_pt1', ['-lossless', '7,1']),
            ('lossless_restart', ['-lossless', '6', '-restart', '1']),
            ('lossless_sub', ['-lossless', '2', '-sample', '2x2,1x1,1x1']),
        ]
        if stress:
            variants += [(f'arith_q{q}', ['-arithmetic', '-quality', str(q)]) for q in (10, 50, 95, 100)]
            variants += [(f'lossless{p}', ['-lossless', str(p)]) for p in (2, 3, 6)]
            variants += [('lossless_p6', ['-lossless', '1', '-precision', '6']), ('lossless_sub_restart', ['-lossless', '4', '-sample', '2x2,1x1,1x1', '-restart', '1'])]
        for label, options in variants:
            for source, suffix in ((rgb, ''), (gray, '_gray')):
                name = f'lj_{width}x{height}_{label}{suffix}.jpg'
                result = subprocess.run([cjpeg, *options, str(source)], capture_output=True)
                if result.returncode == 0 and result.stdout:
                    (out / name).write_bytes(result.stdout)
                    names.append(name)
        rgb.unlink()
        gray.unlink()
    # Transcode Pillow-written JPEGs (including CMYK and progressive files) to arithmetic coding.
    for name in sorted(p.name for p in out.iterdir() if p.suffix == '.jpg' and not p.name.startswith('lj_')):
        if not stress and name not in ('jpg_cmyk.jpg', 'jpg_cmyk_p.jpg', 'jpg_gray_p.jpg', 'jpg_exif_6.jpg', 'jpg_im_411.jpg', 'jpg_restart_p.jpg'):
            continue
        for options, suffix in ((['-arithmetic'], 'arith'), (['-arithmetic', '-progressive'], 'arith_p')):
            result = subprocess.run([jpegtran, '-copy', 'all', *options, str(out / name)], capture_output=True)
            if result.returncode == 0 and result.stdout:
                target = f'lj_{name[:-4]}_{suffix}.jpg'
                (out / target).write_bytes(result.stdout)
                names.append(target)
    return names


def build(out: Path, stress: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tools = os.environ.get('LIBJPEG_TURBO_BIN')
    keep = set() if tools else {p.name for p in out.iterdir() if p.name.startswith('lj_')}
    for old in out.iterdir():
        if old.name not in keep:
            old.unlink()
    if stress:
        png_sizes = [(1, 1), (3, 2), (9, 7), (31, 17), (64, 5)]
        jpeg_sizes = [(1, 1), (8, 8), (16, 16), (17, 9), (33, 27), (65, 47), (120, 80)]
        qualities = [10, 30, 50, 75, 90, 95, 100]
    else:
        png_sizes = [(5, 3), (13, 9)]
        jpeg_sizes = [(9, 7), (37, 21)]
        qualities = [40, 95, 100]
    names = png_corpus(out, png_sizes) + png_extra(out) + jpeg_corpus(out, jpeg_sizes, qualities) + jpeg_extra(out)
    names += gif_corpus(out) + webp_corpus(out) + bmp_corpus(out)
    names += libjpeg_corpus(out, Path(tools), stress) if tools else sorted(keep)
    (out / 'expected.json').write_text(json.dumps(expectations(out, names), indent=1, sort_keys=True) + '\n')
    (out / 'resample.json').write_text(json.dumps(resample_cases(stress), separators=(',', ':')) + '\n')
    if not stress:
        (out / 'processor.json').write_text(json.dumps(processor_cases(out), indent=1) + '\n')


def generate() -> None:
    build(OUT)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--stress':
        build(Path(sys.argv[2]), stress=True)
    else:
        generate()
