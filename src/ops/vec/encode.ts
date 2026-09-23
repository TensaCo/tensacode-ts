/**
 * Encode external inputs into vector representations (Python
 * ``tensorcode/ops/vec/encode.py``).
 *
 * Public classes have stable operation identities. Native model
 * implementations are private and construct their architectures only when a
 * model is built; importing this module performs no I/O.
 */
import { TextEncoder as TextEncoderImpl } from '../../_internal/vec/text.js';
import { ImageEncoder as ImageEncoderImpl } from '../../_internal/vec/vision.js';
import { VocabularyEncoder as VocabularyEncoderImpl } from '../../_internal/vec/vocabulary.js';
import { PatchEncoder as PatchEncoderImpl } from '../../_internal/vec/patch.js';

/** Owned text transformer with sequence, pooled or OUTPUT_ENCODING vector readout. */
export class TextEncoder extends TextEncoderImpl {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.encode.TextEncoder';
}

/** Owned vision transformer (ViT) with sequence, pooled or OUTPUT_ENCODING vector readout. */
export class ImageEncoder extends ImageEncoderImpl {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.encode.ImageEncoder';
}

/**
 * Learn word embeddings from an explicit vocabulary using mean pooling. This
 * specialized encoding operation starts from random weights and uses
 * lowercase regex tokenization; it does not provide pretrained semantics.
 */
export class VocabularyEncoder extends VocabularyEncoderImpl {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.encode.VocabularyEncoder';
}

/**
 * Project image patches into a spatial output Space with source coordinates.
 * The default convolution starts from random weights; supplied modules must
 * satisfy the explicit geometry contract.
 */
export class PatchEncoder extends PatchEncoderImpl {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.encode.PatchEncoder';
}

export const TextEncode = TextEncoder;
export type TextEncode = TextEncoder;
export const ImageEncode = ImageEncoder;
export type ImageEncode = ImageEncoder;
