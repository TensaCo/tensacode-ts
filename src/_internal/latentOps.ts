/**
 * Shared ownership and sequence contracts for owned latent operations
 * (Python ``tensorcode/_internal/latent_ops.py``). FOUNDATION-OWNED.
 */
import { Tensor, ones } from '../nn/tensor.js';
import { ValueError } from '../errors.js';
import { invoke, invokeAsync } from './tracing.js';
import { PretrainedModule } from './pretrained.js';
import { OPERATION_BRAND, type CallOptions, type Context, type OperationLike } from '../ops/base.js';
import { Latent, Space, requireCompatible } from '../ops/vec/latent.js';

/**
 * A complete owned model whose public call is an operation boundary. Replay
 * opts out of external effects; it does not promise identical stochastic
 * training samples. Configurations describe architecture; weights remain
 * registered tensors in the complete pretrained artifact.
 */
export abstract class LatentOperation<I = unknown, O = unknown> extends PretrainedModule<I, O> implements OperationLike<I, O> {
  readonly [OPERATION_BRAND] = true as const;

  override get replayable(): boolean {
    return true;
  }

  override call(value: I, options?: CallOptions): O {
    return invoke(this, value, options?.context ?? null, (v, c) => this.forward(v, c));
  }

  acall(value: I, options?: CallOptions): Promise<O> {
    return invokeAsync(this, value, options?.context ?? null, (v, c) => this.aforward(v, c));
  }

  async aforward(value: I, context: Context | null): Promise<O> {
    return this.forward(value, context);
  }

  override operationBindings(): Record<string, OperationLike> {
    return { operation: this, ...super.operationBindings() };
  }
}

/** A {@link Space} or its configuration mapping. */
export function spaceFromConfig(value: unknown): Space {
  if (value instanceof Space) return value;
  return Space.fromConfig(value);
}

/**
 * Normalize explicit feature/sequence/spatial batches without detaching.
 * Feature shapes are ``D``/``B,D``; sequence ``L,D``/``B,L,D``; spatial is
 * explicitly batched ``B,H,W,D``. Invalid tokens are zeroed and remain masked.
 * Returns ``[x (B,L,D), mask (B,L) bool]``.
 */
export function asSequence(value: unknown, expectedSpace: Space | unknown): [Tensor, Tensor] {
  if (!(value instanceof Latent)) throw new TypeError('Expected a space-tagged Latent');
  requireCompatible(spaceFromConfig(expectedSpace), value.space);
  let x = value.tensor;
  if (!x.isFloatingPoint || !x.allFinite()) throw new ValueError('Latent conditioning must contain finite floating tensors');
  let mask = value.mask;
  if (mask !== null && mask.dtype !== 'bool') throw new ValueError('Latent conditioning mask must be boolean');
  if (mask === null) mask = ones(x.shape.slice(0, -1), { dtype: 'bool' });
  const organization = value.space.organization;
  if (organization === 'feature' && (x.ndim === 1 || x.ndim === 2)) {
    if (x.ndim === 1) {
      x = x.reshape(1, 1, x.shape[0]!);
      mask = mask.reshape(1, 1);
    } else {
      x = x.unsqueeze(1);
      mask = mask.unsqueeze(1);
    }
  } else if (organization === 'sequence' && (x.ndim === 2 || x.ndim === 3)) {
    if (x.ndim === 2) {
      x = x.unsqueeze(0);
      mask = mask.unsqueeze(0);
    }
  } else if (organization === 'spatial' && x.ndim === 4) {
    x = x.flatten(1, 2);
    mask = mask.flatten(1, 2);
  } else {
    throw new ValueError('Conditioning organization/shape must be feature D/B,D, sequence L,D/B,L,D or spatial B,H,W,D');
  }
  if (!x.shape[0] || !x.shape[1] || !mask.any(1).all().item()) {
    throw new ValueError('Every conditioning sample requires at least one valid token');
  }
  return [x.maskedFill(mask.logicalNot().unsqueeze(-1), 0), mask];
}
