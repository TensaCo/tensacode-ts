/**
 * Select one existing option from explicit vector scores without executing it
 * (Python ``tensorcode/ops/vec/decide.py``).
 */
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { ConfigModuleOperation } from '../../_internal/operationConfig.js';
import type { JsonObject } from '../../_internal/json.js';
import type { Context } from '../base.js';
import { Scores, gatherLatent, selectPython } from './candidates.js';
import type { Latent } from './latent.js';

/** Selected candidate indices, their scores and gathered items; nothing is executed. */
export class Decision {
  static readonly qualifiedName: string = 'tensorcode.ops.vec.decide.Decision';
  static readonly recordFields = ['indices', 'scores', 'items', 'scored'] as const;
  readonly indices: Tensor;
  readonly scores: Tensor;
  readonly items: Latent;
  readonly scored: Scores;

  constructor(indices: Tensor, scores: Tensor, items: Latent, scored: Scores) {
    this.indices = indices;
    this.scores = scores;
    this.items = items;
    this.scored = scored;
    Object.freeze(this);
  }

  /** Selected candidate identity for an unbatched decision. */
  get identity(): string {
    if (this.indices.ndim !== 0) throw new ValueError('identity is unavailable for a batched decision; use identities');
    return selectPython(this.scored.candidates.identities, this.indices) as string;
  }

  /** Selected candidate identities for a batched decision (nested frozen arrays). */
  get identities(): readonly unknown[] {
    if (this.indices.ndim === 0) throw new ValueError('identities requires a batched decision; use identity');
    return selectPython(this.scored.candidates.identities, this.indices) as readonly unknown[];
  }

  static fromRecord(fields: Record<string, unknown>): Decision {
    return new Decision(fields.indices as Tensor, fields.scores as Tensor, fields.items as Latent, fields.scored as Scores);
  }

  toRecord(): Record<string, unknown> {
    return { indices: this.indices, scores: this.scores, items: this.items, scored: this.scored };
  }
}

/** Select the highest (or, with ``largest: false``, lowest) masked score. */
export class Decide extends ConfigModuleOperation<Scores, Decision> {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.decide.Decide';
  readonly largest: boolean;

  constructor(config: JsonObject | null = null) {
    super(config, ['largest'], { largest: true });
    if (typeof this.config.largest !== 'boolean') throw new ValueError('Decide largest must be a boolean');
    this.largest = this.config.largest;
  }

  override get replayable(): boolean {
    return true;
  }

  /** Select one candidate per query from {@link Scores}; returns a {@link Decision}. */
  forward(value: Scores, context: Context | null): Decision {
    if (context) throw new ValueError('Decide does not consume context');
    if (!(value instanceof Scores)) throw new TypeError('Decide expects Scores');
    let selectable = value.values;
    const mask = value.candidates.candidates.mask;
    if (mask !== null) selectable = selectable.maskedFill(mask.logicalNot(), this.largest ? -Infinity : Infinity);
    const indices = this.largest ? selectable.argmax(-1) : selectable.argmin(-1);
    const selected = value.values.gather(-1, indices.unsqueeze(-1)).squeeze(-1);
    const items = gatherLatent(value.candidates.candidates, indices.unsqueeze(-1));
    const squeezed = items.withTensor(items.tensor.squeeze(-2), {
      mask: items.mask === null ? null : items.mask.squeeze(-1),
      coordinates: items.coordinates === null ? null : items.coordinates.squeeze(-2),
    });
    return new Decision(indices, selected, squeezed, value);
  }

  override configuration(): JsonObject {
    return { largest: this.largest };
  }
}
