/**
 * EXIF orientation: reading the ``Orientation`` tag (0x0112) the way Pillow
 * (``Image.getexif``, including its XMP ``tiff:Orientation`` fallback) and
 * torchvision (``exif.h``) do, and applying it to interleaved ``HWC`` pixels
 * like ``ImageOps.exif_transpose`` / ``apply_exif_orientation``.
 */

/** Strip an optional ``Exif\0\0`` header (``Image.Exif.load``). */
export function tiffPayload(exif: Uint8Array): Uint8Array {
  if (exif.length >= 6 && exif[0] === 0x45 && exif[1] === 0x78 && exif[2] === 0x69 && exif[3] === 0x66 && exif[4] === 0 && exif[5] === 0) {
    return exif.subarray(6);
  }
  return exif;
}

/**
 * The raw ``Orientation`` value in IFD0 of an EXIF payload, ``undefined`` when
 * the tag is absent (Pillow's typed read: BYTE, SHORT or LONG).
 */
export function exifOrientationTag(exif: Uint8Array): number | undefined {
  const tiff = tiffPayload(exif);
  if (tiff.length < 8) return undefined;
  const little = tiff[0] === 0x49 && tiff[1] === 0x49;
  const big = tiff[0] === 0x4d && tiff[1] === 0x4d;
  if (!little && !big) return undefined;
  const u16 = (offset: number): number => (little ? tiff[offset]! | (tiff[offset + 1]! << 8) : (tiff[offset]! << 8) | tiff[offset + 1]!);
  const u32 = (offset: number): number => (little
    ? (tiff[offset]! | (tiff[offset + 1]! << 8) | (tiff[offset + 2]! << 16) | (tiff[offset + 3]! << 24)) >>> 0
    : ((tiff[offset]! << 24) | (tiff[offset + 1]! << 16) | (tiff[offset + 2]! << 8) | tiff[offset + 3]!) >>> 0);
  if (u16(2) !== 42) return undefined;
  const ifd = u32(4);
  if (ifd + 2 > tiff.length) return undefined;
  const count = u16(ifd);
  for (let index = 0; index < count; index += 1) {
    const entry = ifd + 2 + index * 12;
    if (entry + 12 > tiff.length) return undefined;
    if (u16(entry) !== 0x0112) continue;
    const type = u16(entry + 2);
    return type === 4 || type === 9 ? u32(entry + 8) : type === 1 || type === 6 || type === 7 ? tiff[entry + 8]! : u16(entry + 8);
  }
  return undefined;
}

/** Orientation (1-8) of an EXIF payload, or ``null``. */
export function exifOrientation(exif: Uint8Array): number | null {
  const value = exifOrientationTag(exif);
  return value !== undefined && value >= 1 && value <= 8 ? value : null;
}

/** ``Image.getexif()[Orientation]`` with Pillow's XMP ``tiff:Orientation`` fallback. */
export function pilOrientation(exif: Uint8Array | undefined, xmp: string | undefined): number | undefined {
  const tag = exif ? exifOrientationTag(exif) : undefined;
  if (tag !== undefined) return tag;
  if (xmp) {
    const match = /tiff:Orientation(="|>)([0-9])/.exec(xmp);
    if (match) return Number(match[2]);
  }
  return undefined;
}

/**
 * torchvision's ``fetch_exif_orientation`` over a TIFF payload (no header
 * checks beyond the byte order mark and the 0x2a tag mark; -1 when absent).
 */
export function torchvisionOrientation(data: Uint8Array): number {
  const size = data.length;
  const at = (index: number): number => {
    if (index < 0 || index >= size) throw new Error('Expected index < size');
    return data[index]!;
  };
  let endianness = 0;
  if (size >= 1 && !(size > 1 && data[0] !== data[1])) {
    if (data[0] === 0x49) endianness = 0x49;
    else if (data[0] === 0x4d) endianness = 0x4d;
  }
  const u16 = (offset: number): number => {
    if (offset + 1 >= size) return 0xffff;
    return endianness === 0x49 ? at(offset) + (at(offset + 1) << 8) : (at(offset) << 8) + at(offset + 1);
  };
  const u32 = (offset: number): number => {
    if (offset + 3 >= size) return 0xffff;
    return endianness === 0x49
      ? (at(offset) + (at(offset + 1) << 8) + (at(offset + 2) << 16) + (at(offset + 3) << 24)) >>> 0
      : ((at(offset) << 24) + (at(offset + 1) << 16) + (at(offset + 2) << 8) + at(offset + 3)) >>> 0;
  };
  if (u16(2) !== 0x2a) return -1;
  let offset = u32(4);
  const entries = u16(offset);
  offset += 2;
  for (let entry = 0; entry < entries; entry += 1) {
    const tag = u16(offset);
    if (tag === 0xffff) break;
    if (tag === 0x0112) return u16(offset + 8);
    offset += 12;
  }
  return -1;
}

/**
 * Reorient interleaved ``[height, width, channels]`` pixels for an EXIF
 * orientation. Returns the new pixels and size.
 */
export function orientPixels<T extends Uint8Array | Uint16Array | Int32Array | Float32Array>(
  pixels: T, width: number, height: number, channels: number, orientation: number | null,
): { pixels: T; width: number; height: number } {
  if (!orientation || orientation < 2 || orientation > 8) return { pixels, width, height };
  const swap = orientation >= 5;
  const outWidth = swap ? height : width;
  const outHeight = swap ? width : height;
  const out = new (pixels.constructor as new (length: number) => T)(pixels.length);
  for (let y = 0; y < outHeight; y += 1) {
    for (let x = 0; x < outWidth; x += 1) {
      let sx: number;
      let sy: number;
      switch (orientation) {
        case 2: sx = width - 1 - x; sy = y; break; // mirror horizontal
        case 3: sx = width - 1 - x; sy = height - 1 - y; break; // rotate 180
        case 4: sx = x; sy = height - 1 - y; break; // mirror vertical
        case 5: sx = y; sy = x; break; // transpose
        case 6: sx = y; sy = height - 1 - x; break; // rotate 90 clockwise
        case 7: sx = width - 1 - y; sy = height - 1 - x; break; // transverse
        default: sx = width - 1 - y; sy = x; break; // 8: rotate 90 counter-clockwise
      }
      const source = (sy * width + sx) * channels;
      const target = (y * outWidth + x) * channels;
      for (let c = 0; c < channels; c += 1) out[target + c] = pixels[source + c]!;
    }
  }
  return { pixels: out, width: outWidth, height: outHeight };
}
