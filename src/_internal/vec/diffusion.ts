/**
 * Owned latent diffusion image decoder (Python
 * ``tensorcode/_internal/vec/diffusion.py``) — OUT OF SCOPE in TypeScript.
 *
 * The Python implementation owns a diffusers ``UNet2DConditionModel`` +
 * ``AutoencoderKL`` with DDIM sampling. The TypeScript port keeps the class,
 * its public identity and exports so compositions and artifact manifests stay
 * nameable, but construction, foundation import and artifact loading raise
 * {@link NotImplementedError}.
 */
import { NotImplementedError } from '../../errors.js';
import type { Tensor } from '../../nn/tensor.js';
import type { Context } from '../../ops/base.js';
import type { Latent } from '../../ops/vec/latent.js';
import { LatentOperation } from '../latentOps.js';

export const IMAGE_DECODER_UNAVAILABLE = 'ImageDecoder (latent diffusion) is not available in the TypeScript port; use the Python package';

/** Generate RGB images from vectors through owned latent diffusion (unavailable in TypeScript). */
export class ImageDecoder extends LatentOperation<Latent, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.diffusion.ImageDecoder';

  constructor(config?: unknown) {
    void config;
    throw new NotImplementedError(IMAGE_DECODER_UNAVAILABLE);
    // eslint-disable-next-line no-unreachable
    super({});
  }

  /** Unavailable: latent diffusion is not implemented in TypeScript. */
  static async fromFoundation(...args: unknown[]): Promise<never> {
    void args;
    throw new NotImplementedError(IMAGE_DECODER_UNAVAILABLE);
  }

  /** Unavailable: latent diffusion artifacts cannot be loaded in TypeScript. */
  static override async fromPretrained(...args: unknown[]): Promise<never> {
    void args;
    throw new NotImplementedError(IMAGE_DECODER_UNAVAILABLE);
  }

  forward(value: Latent, context: Context | null): Tensor {
    void value;
    void context;
    throw new NotImplementedError(IMAGE_DECODER_UNAVAILABLE);
  }
}
