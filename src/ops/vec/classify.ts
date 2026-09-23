/**
 * Owned trainable label heads over vector inputs (Python
 * ``tensorcode/ops/vec/classify.py``).
 */
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import type { OwnedKind } from '../../_internal/vec/owned.js';
import type { Context } from '../base.js';
import { Transform } from './transform.js';

/** Label logits for one item ``(labels,)`` or a batch ``(batch, labels)``. */
export class Prediction {
  static readonly qualifiedName: string = 'tensorcode.ops.vec.classify.Prediction';
  static readonly recordFields = ['logits', 'labels'] as const;
  readonly logits: Tensor;
  readonly labels: readonly string[];

  constructor(logits: Tensor, labels: Iterable<string>) {
    this.logits = logits;
    this.labels = Object.freeze([...labels]);
    Object.freeze(this);
  }

  /** Softmax over labels; uncalibrated unless separately calibrated. */
  get probabilities(): Tensor {
    return this.logits.softmax(-1);
  }

  /** Top label for a single (unbatched) prediction. */
  get value(): string {
    if (this.logits.ndim !== 1) throw new ValueError('Use values for batched predictions');
    return this.labels[this.logits.argmax().item()]!;
  }

  /** Top label per row for a batched prediction. */
  get values(): readonly string[] {
    if (this.logits.ndim !== 2) throw new ValueError('values requires a batch of predictions');
    return Object.freeze(this.logits.argmax(-1).toArray().map((index) => this.labels[index]!));
  }

  static fromRecord(fields: Record<string, unknown>): Prediction {
    return new Prediction(fields.logits as Tensor, fields.labels as readonly string[]);
  }

  toRecord(): Record<string, unknown> {
    return { logits: this.logits, labels: this.labels };
  }
}

/** Owned trainable label head; native transformer bridges start untrained. */
export class Classify extends Transform<Prediction> {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.classify.Classify';
  static override readonly kind: OwnedKind = 'classify';

  /** Return a {@link Prediction} over the configured labels. */
  override forward(value: unknown, context: Context | null): Prediction {
    const logits = this.tensorOutput(value, context);
    if (!(logits instanceof Tensor) || (logits.ndim !== 1 && logits.ndim !== 2) || logits.shape[logits.ndim - 1] !== this.labels.length) {
      throw new ValueError('Model logits must match the labels (single item or batch)');
    }
    return new Prediction(logits, this.labels);
  }
}
