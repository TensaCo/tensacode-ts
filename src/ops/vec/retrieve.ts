/**
 * Return top-ranked existing vector candidates without generating new items
 * (Python ``tensorcode/ops/vec/retrieve.py``).
 */
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { ConfigModuleOperation } from '../../_internal/operationConfig.js';
import type { JsonObject } from '../../_internal/json.js';
import type { Context } from '../base.js';
import { Scores, gatherLatent, selectPython } from './candidates.js';
import type { Latent } from './latent.js';

/** Top-``k`` candidate indices, scores and gathered items, best first. */
export class Retrieval {
  static readonly qualifiedName: string = 'tensorcode.ops.vec.retrieve.Retrieval';
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

  /** Identities of the retrieved candidates (frozen arrays, nested per batch). */
  get identities(): readonly unknown[] {
    return selectPython(this.scored.candidates.identities, this.indices) as readonly unknown[];
  }

  /** Metadata mappings of the retrieved candidates. */
  get metadata(): readonly unknown[] {
    return selectPython(this.scored.candidates.metadata, this.indices) as readonly unknown[];
  }

  static fromRecord(fields: Record<string, unknown>): Retrieval {
    return new Retrieval(fields.indices as Tensor, fields.scores as Tensor, fields.items as Latent, fields.scored as Scores);
  }

  toRecord(): Record<string, unknown> {
    return { indices: this.indices, scores: this.scores, items: this.items, scored: this.scored };
  }
}

/** Select the top ``k`` masked scores; scores are not probabilities. */
export class Retrieve extends ConfigModuleOperation<Scores, Retrieval> {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.retrieve.Retrieve';
  readonly k: number;
  readonly largest: boolean;

  constructor(config: JsonObject | null = null) {
    super(config, ['k', 'largest'], { largest: true });
    const k = this.config.k;
    if (typeof k !== 'number' || !Number.isInteger(k) || k <= 0) throw new ValueError('Retrieve k must be a positive integer');
    if (typeof this.config.largest !== 'boolean') throw new ValueError('Retrieve largest must be a boolean');
    this.k = k;
    this.largest = this.config.largest;
  }

  override get replayable(): boolean {
    return true;
  }

  /** Select the top ``k`` candidates per query from {@link Scores}; returns a {@link Retrieval}. */
  forward(value: Scores, context: Context | null): Retrieval {
    if (context) throw new ValueError('Retrieve does not consume context');
    if (!(value instanceof Scores)) throw new TypeError('Retrieve expects Scores');
    const count = value.candidates.count;
    if (this.k > count) throw new ValueError(`Cannot retrieve ${this.k} items from only ${count} candidates`);
    let selectable = value.values;
    const mask = value.candidates.candidates.mask;
    if (mask !== null) {
      if (mask.to('int64').sum(-1).lt(this.k).any().item()) {
        throw new ValueError(`Retrieve k=${this.k} exceeds the valid candidates for a query`);
      }
      selectable = selectable.maskedFill(mask.logicalNot(), this.largest ? -Infinity : Infinity);
    }
    const { indices } = selectable.topk(this.k, -1, this.largest, true);
    const scores = value.values.gather(-1, indices);
    const items = gatherLatent(value.candidates.candidates, indices);
    return new Retrieval(indices, scores, items, value);
  }

  override configuration(): JsonObject {
    return { k: this.k, largest: this.largest };
  }
}
