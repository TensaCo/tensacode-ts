/**
 * Vector operations (Python ``tensorcode.ops.vec``).
 *
 * Import concrete classes from ``encode`` or ``decode`` modules for canonical
 * identities; everything is also re-exported here. Importing performs no I/O
 * and constructs no models.
 */
export { Latent, Space, requireCompatible, latentCodecs, type LatentOptions, type SpaceConfiguration, type SpaceOptions } from './latent.js';
export { Transform } from './transform.js';
export { Classify, Prediction } from './classify.js';
export { CandidateSet, Scores, type CandidateMetadata } from './candidates.js';
export { Score, type PairScoreModule, type ScoreFoundationOptions } from './score.js';
export { Decide, Decision } from './decide.js';
export { Retrieve, Retrieval } from './retrieve.js';
export { TextEncoder, TextEncode, ImageEncoder, ImageEncode, VocabularyEncoder, PatchEncoder } from './encode.js';
export { Decode, Decoder, TextDecoder, TextDecode, ImageDecoder, ImageDecode } from './decode.js';
export type { FromModuleOptions, OwnedFoundationOptions, Readout } from '../../_internal/vec/owned.js';
export type { TextReadout, TextEncoderFoundationOptions, TextDecoderFoundationOptions, Bridge } from '../../_internal/vec/text.js';
export type { ImageReadout, ImageFoundationOptions, ProcessedImages } from '../../_internal/vec/vision.js';
export type { PatchFromModuleOptions, PatchModule } from '../../_internal/vec/patch.js';
