/**
 * Owned vector transformations with explicit architecture and space contracts
 * (Python ``tensorcode/ops/vec/transform.py``).
 */
import type { Tensor } from '../../nn/tensor.js';
import { OwnedMap, type OwnedKind } from '../../_internal/vec/owned.js';
import type { Latent } from './latent.js';

/**
 * Construct a linear, MLP, or supported native transformer (bert, roberta,
 * distilbert) from JSON configuration. Returns a {@link Latent} in the
 * configured ``output_space``; ``fromModule`` operations return the supplied
 * module's result.
 */
export class Transform<O = Latent | Tensor> extends OwnedMap<O> {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.transform.Transform';
  static override readonly kind: OwnedKind = 'transform';
}
