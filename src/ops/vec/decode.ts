/**
 * Owned tensor readouts and concrete pretrained modality decoders (Python
 * ``tensorcode/ops/vec/decode.py``).
 */
import type { Tensor } from '../../nn/tensor.js';
import type { OwnedKind } from '../../_internal/vec/owned.js';
import { TextDecoder as TextDecoderImpl } from '../../_internal/vec/text.js';
import { ImageDecoder as ImageDecoderImpl } from '../../_internal/vec/diffusion.js';
import { Transform } from './transform.js';

/** Decode to a tensor with an explicit output width and description. */
export class Decode extends Transform<Tensor> {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.decode.Decode';
  static override readonly kind: OwnedKind = 'decode';
}

export const Decoder = Decode;
export type Decoder = Decode;

/** Generate text from vectors through an owned pretrained transformer. */
export class TextDecoder extends TextDecoderImpl {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.decode.TextDecoder';
}

/**
 * Generate RGB images from vectors through owned latent diffusion.
 * Unavailable in the TypeScript port: construction and loading raise
 * ``NotImplementedError``.
 */
export class ImageDecoder extends ImageDecoderImpl {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.decode.ImageDecoder';
}

export const TextDecode = TextDecoder;
export type TextDecode = TextDecoder;
export const ImageDecode = ImageDecoder;
export type ImageDecode = ImageDecoder;
