/** Select authored item keys (Python ``tensorcode/ops/text/retrieve.py``). */
import { ValueError } from '../../errors.js';
import { isPlainObject, markPlainData, orderedEntries, orderedKeys, orderedObject, type JsonObject } from '../../_internal/json.js';
import type { Alternative } from '../../_internal/text/native.js';
import {
  InvalidModelOutput, StructuredOperation, finiteScores, frozenMapping, modelConfiguration, optionalBool, type ReadonlyMapping,
} from './structured.js';

export interface RetrievalFields {
  scores?: Readonly<Record<string, number>> | null;
  abstained?: boolean;
}

/** Selected item ``keys`` and ``items`` in rank order, with optional non-probability scores. */
export class RetrievalResult {
  static readonly qualifiedName: string = 'tensorcode.ops.text.retrieve.RetrievalResult';
  static readonly recordFields = ['keys', 'items', 'scores', 'abstained'] as const;
  /** Python ``float`` annotations (integral values persist as ``1.0``). */
  static readonly recordFloatFields = ['scores'] as const;
  readonly keys: readonly string[];
  readonly items: readonly unknown[];
  readonly scores: ReadonlyMapping<number> | null;
  readonly abstained: boolean;

  constructor(keys: Iterable<string>, items: Iterable<unknown>, fields: RetrievalFields = {}) {
    this.keys = Object.freeze([...keys]);
    this.items = Object.freeze([...items]);
    this.scores = fields.scores === null || fields.scores === undefined ? null : frozenMapping(fields.scores);
    this.abstained = fields.abstained ?? false;
    Object.freeze(this);
  }

  /** Retrieval scores have no probability interpretation. */
  get distribution(): null {
    return null;
  }

  static fromRecord(fields: Record<string, unknown>): RetrievalResult {
    return new RetrievalResult(fields.keys as string[], fields.items as unknown[], fields as RetrievalFields);
  }

  toRecord(): Record<string, unknown> {
    return { keys: this.keys, items: this.items, scores: this.scores, abstained: this.abstained };
  }
}

/** A ``Map`` with string keys as an ordered object (Python ``TypeError`` for other keys); other values unchanged. */
function mappingOf(value: unknown): unknown {
  if (!(value instanceof Map)) return value;
  if (![...value.keys()].every((key) => typeof key === 'string')) throw new TypeError('item keys must be strings');
  return orderedObject(value as Map<string, unknown>);
}

/**
 * Select authored item keys using an owned seq2seq model.
 *
 * ``decoding: 'likelihood'`` scores every item description and returns the
 * ``limit`` highest; its scores are log-likelihoods, not probabilities.
 * ``fromModel`` explicitly wraps an external provider without owned artifacts.
 */
export class Retrieve extends StructuredOperation<RetrievalResult> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text.retrieve.Retrieve';
  static override readonly schemaName: string = 'tensorcode.retrieve';
  static override readonly semanticFields: ReadonlySet<string> = new Set(['instructions', 'limit', 'descriptions', 'items']);
  declare items: ReadonlyMapping<unknown>;
  declare descriptions: ReadonlyMapping<string>;
  declare limit: number;

  /** Item keys in Python insertion order (integer-like keys included). */
  private get itemKeys(): string[] {
    return orderedKeys(this.items);
  }

  _alternatives(): Alternative[] {
    return this.itemKeys.map((key) => [`${key}: ${this.descriptions[key]}`, this.descriptions[key]!] as const);
  }

  _fromScores(scores: readonly number[]): RetrievalResult {
    const keys = this.itemKeys;
    const keyed = orderedObject(keys.map((key, index) => [key, scores[index]!] as const));
    // Stable descending order (Python ``sorted(..., reverse=True)``).
    const ranked = keys.map((key, index) => ({ key, index })).sort((a, b) => (keyed[b.key]! - keyed[a.key]!) || (a.index - b.index));
    const selected = ranked.slice(0, this.limit).map(({ key }) => key);
    return new RetrievalResult(selected, selected.map((key) => this.items[key]), { scores: keyed });
  }

  _targetWeights(result: RetrievalResult): number[] {
    if (result.abstained) throw new ValueError('likelihood decoding has no abstention alternative');
    return this.itemKeys.map((key) => (result.keys.includes(key) ? 1 / result.keys.length : 0));
  }

  protected override _configureSemantics(config: Record<string, unknown>): void {
    super._configureSemantics(config);
    // Items are an ordered Python mapping: a ``Map`` or an object (whose
    // parsed or ``orderedObject`` key order is kept, integer-like keys included).
    const items = mappingOf(config.items);
    let descriptions = mappingOf(config.descriptions ?? null);
    const limit = config.limit ?? 1;
    if (!isPlainObject(items) || !Object.keys(items).length) throw new ValueError('items must be a nonempty mapping of stable string keys');
    if (typeof limit !== 'number' || !Number.isInteger(limit) || limit < 1 || limit > Object.keys(items).length) {
      throw new ValueError('limit must be between 1 and the item count');
    }
    // Item values are caller data, written as Python writes them (no float schema).
    if (config.items !== null && typeof config.items === 'object') markPlainData(config.items as object);
    this.items = markPlainData(frozenMapping(items));
    if (descriptions === null) {
      descriptions = orderedObject(orderedEntries(this.items).filter(([, item]) => typeof item === 'string'));
    }
    const keys = Object.keys(this.items);
    if (!isPlainObject(descriptions) || Object.keys(descriptions).length !== keys.length
      || !keys.every((key) => Object.hasOwn(descriptions as object, key))
      || !Object.values(descriptions).every((description) => typeof description === 'string' && description.length > 0)) {
      throw new ValueError('descriptions must provide nonempty text for every item; arbitrary item values are not stringified');
    }
    this.descriptions = frozenMapping(descriptions as Record<string, string>);
    this.limit = limit;
  }

  responseSchema(): Record<string, unknown> {
    const keys = this.itemKeys;
    const scoreProperties = orderedObject(keys.map((key) => [key, { type: 'number' }] as const));
    return {
      type: 'object',
      properties: {
        keys: {
          type: 'array',
          items: {
            type: 'string',
            enum: [...keys],
            description: `Candidate meanings: ${keys.map((key) => `${key}: ${this.descriptions[key]}`).join('; ')}`,
          },
          maxItems: this.limit,
          uniqueItems: true,
        },
        scores: { type: ['object', 'null'], properties: scoreProperties, required: [...keys], additionalProperties: false },
        abstained: { type: 'boolean' },
      },
      required: ['keys', 'scores', 'abstained'],
      additionalProperties: false,
    };
  }

  _parse(value: Readonly<Record<string, unknown>>): RetrievalResult {
    const abstained = optionalBool(value, 'abstained');
    const raw = value.keys;
    if (!Array.isArray(raw)) throw new InvalidModelOutput('keys must be a sequence');
    const keys = [...raw];
    if (!keys.every((key) => typeof key === 'string')) throw new InvalidModelOutput('Retrieved keys must be strings');
    if (keys.length > this.limit || new Set(keys).size !== keys.length || !keys.every((key) => Object.hasOwn(this.items, key))) {
      throw new InvalidModelOutput('Retrieved keys must be unique configured items within the limit');
    }
    if (abstained && keys.length) throw new InvalidModelOutput('An abstained retrieval must return no keys');
    if (!abstained && !keys.length) throw new InvalidModelOutput('A non-abstained retrieval must return at least one key');
    return new RetrievalResult(keys as string[], (keys as string[]).map((key) => this.items[key]), {
      scores: finiteScores(value.scores, this.itemKeys),
      abstained,
    });
  }

  override configuration(): JsonObject {
    if (this._owned) return super.configuration();
    return {
      type: 'text_retrieve',
      item_keys: this.itemKeys,
      descriptions: orderedObject(orderedEntries(this.descriptions)),
      limit: this.limit,
      instructions: this.instructions,
      model: modelConfiguration(this.model) as JsonObject,
    };
  }
}
