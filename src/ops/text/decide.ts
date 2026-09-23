/** Choose an authored option (Python ``tensorcode/ops/text/decide.py``). */
import {
  InvalidModelOutput, SelectionOperation, alternativeDescriptions, alternativeList, frozenMapping, modelConfiguration,
  optionalBool, optionalConfidence, probabilityDistribution, selectionSchema, type ReadonlyMapping,
} from './structured.js';
import type { SelectionFields } from './classify.js';
import type { JsonObject } from '../../_internal/json.js';

/** Selected ``choice`` (``null`` when abstained) with optional distribution and confidence. */
export class DecisionResult {
  static readonly qualifiedName: string = 'tensorcode.ops.text.decide.DecisionResult';
  static readonly recordFields = ['choice', 'distribution', 'confidence', 'abstained'] as const;
  readonly choice: string | null;
  readonly distribution: ReadonlyMapping<number> | null;
  readonly confidence: number | null;
  readonly abstained: boolean;

  constructor(choice: string | null, fields: SelectionFields = {}) {
    this.choice = choice;
    this.distribution = fields.distribution === null || fields.distribution === undefined ? null : frozenMapping({ ...fields.distribution });
    this.confidence = fields.confidence ?? null;
    this.abstained = fields.abstained ?? false;
    Object.freeze(this);
  }

  /** The selected choice (alias used by generic choosers). */
  get value(): string | null {
    return this.choice;
  }

  static fromRecord(fields: Record<string, unknown>): DecisionResult {
    return new DecisionResult(fields.choice as string | null, fields as SelectionFields);
  }

  toRecord(): Record<string, unknown> {
    return { choice: this.choice, distribution: this.distribution, confidence: this.confidence, abstained: this.abstained };
  }
}

/**
 * Choose an authored option using an owned seq2seq model.
 *
 * ``decoding: 'likelihood'`` scores every option in one encoder pass; the
 * default generates a JSON response. ``fromModel`` explicitly wraps an
 * external provider without owned artifacts.
 */
export class Decide extends SelectionOperation<DecisionResult> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text.decide.Decide';
  static override readonly schemaName: string = 'tensorcode.decide';
  static override readonly semanticFields: ReadonlySet<string> = new Set(['instructions', 'options', 'descriptions']);
  declare options: readonly string[];

  _choices(): readonly string[] {
    return this.options;
  }

  protected override _configureSemantics(config: Record<string, unknown>): void {
    super._configureSemantics(config);
    this.options = alternativeList(config.options ?? [], 'options');
    this.descriptions = alternativeDescriptions(config.descriptions, this.options, 'options');
  }

  protected _result(choice: string | null, fields: SelectionFields): DecisionResult {
    return new DecisionResult(choice, fields);
  }

  responseSchema(): Record<string, unknown> {
    return selectionSchema('choice', this.options, this.descriptions);
  }

  _parse(value: Readonly<Record<string, unknown>>): DecisionResult {
    const abstained = optionalBool(value, 'abstained');
    const choice = value.choice ?? null;
    if (abstained) {
      if (choice !== null) throw new InvalidModelOutput('An abstained decision must have choice null');
    } else if (typeof choice !== 'string' || !this.options.includes(choice)) {
      throw new InvalidModelOutput('Decision choice is not one of the configured options');
    }
    return new DecisionResult(choice as string | null, {
      distribution: probabilityDistribution(value.distribution, this.options),
      confidence: optionalConfidence(value),
      abstained,
    });
  }

  override configuration(): JsonObject {
    if (this._owned) return super.configuration();
    return {
      type: 'text_decide',
      options: [...this.options],
      ...(Object.keys(this.descriptions).length ? { descriptions: { ...this.descriptions } } : {}),
      instructions: this.instructions,
      model: modelConfiguration(this.model) as JsonObject,
    };
  }
}
