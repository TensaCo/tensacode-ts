/** Classification with explicit labels (Python ``tensorcode/ops/text/classify.py``). */
import {
  InvalidModelOutput, SelectionOperation, alternativeDescriptions, alternativeList, frozenMapping, modelConfiguration,
  optionalBool, optionalConfidence, probabilityDistribution, selectionSchema, type ReadonlyMapping,
} from './structured.js';
import type { JsonObject } from '../../_internal/json.js';

export interface SelectionFields {
  distribution?: Readonly<Record<string, number>> | null;
  confidence?: number | null;
  abstained?: boolean;
}

/** Selected ``label`` (``null`` when abstained) with optional distribution and confidence. */
export class ClassificationResult {
  static readonly qualifiedName: string = 'tensorcode.ops.text.classify.ClassificationResult';
  static readonly recordFields = ['label', 'distribution', 'confidence', 'abstained'] as const;
  readonly label: string | null;
  readonly distribution: ReadonlyMapping<number> | null;
  readonly confidence: number | null;
  readonly abstained: boolean;

  constructor(label: string | null, fields: SelectionFields = {}) {
    this.label = label;
    this.distribution = fields.distribution === null || fields.distribution === undefined ? null : frozenMapping({ ...fields.distribution });
    this.confidence = fields.confidence ?? null;
    this.abstained = fields.abstained ?? false;
    Object.freeze(this);
  }

  /** The selected label (alias used by generic choosers). */
  get value(): string | null {
    return this.label;
  }

  static fromRecord(fields: Record<string, unknown>): ClassificationResult {
    return new ClassificationResult(fields.label as string | null, fields as SelectionFields);
  }

  toRecord(): Record<string, unknown> {
    return { label: this.label, distribution: this.distribution, confidence: this.confidence, abstained: this.abstained };
  }
}

/**
 * Classify messages with an owned seq2seq model and explicit labels.
 *
 * ``decoding: 'likelihood'`` scores every label in one encoder pass and always
 * returns a full distribution; the default generates a JSON response.
 * ``fromModel`` explicitly wraps an external provider without owned artifacts.
 */
export class Classify extends SelectionOperation<ClassificationResult> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text.classify.Classify';
  static override readonly schemaName: string = 'tensorcode.classify';
  static override readonly semanticFields: ReadonlySet<string> = new Set(['labels', 'instructions', 'descriptions']);
  declare labels: readonly string[];

  _choices(): readonly string[] {
    return this.labels;
  }

  protected override _configureSemantics(config: Record<string, unknown>): void {
    super._configureSemantics(config);
    this.labels = alternativeList(config.labels ?? [], 'labels');
    this.descriptions = alternativeDescriptions(config.descriptions, this.labels, 'labels');
  }

  protected _result(label: string | null, fields: SelectionFields): ClassificationResult {
    return new ClassificationResult(label, fields);
  }

  responseSchema(): Record<string, unknown> {
    return selectionSchema('label', this.labels, this.descriptions);
  }

  _parse(value: Readonly<Record<string, unknown>>): ClassificationResult {
    const abstained = optionalBool(value, 'abstained');
    const label = value.label ?? null;
    if (abstained) {
      if (label !== null) throw new InvalidModelOutput('An abstained classification must have label null');
    } else if (typeof label !== 'string' || !this.labels.includes(label)) {
      throw new InvalidModelOutput('Classification label is not one of the configured labels');
    }
    return new ClassificationResult(label as string | null, {
      distribution: probabilityDistribution(value.distribution, this.labels),
      confidence: optionalConfidence(value),
      abstained,
    });
  }

  override configuration(): JsonObject {
    if (this._owned) return super.configuration();
    return {
      type: 'text_classify',
      labels: [...this.labels],
      ...(Object.keys(this.descriptions).length ? { descriptions: { ...this.descriptions } } : {}),
      instructions: this.instructions,
      model: modelConfiguration(this.model) as JsonObject,
    };
  }
}
