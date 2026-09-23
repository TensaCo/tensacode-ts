/** Rubric scoring (Python ``tensorcode/ops/text/score.py``). */
import { ValueError } from '../../errors.js';
import { isPlainObject, type JsonObject } from '../../_internal/json.js';
import type { Alternative } from '../../_internal/text/native.js';
import {
  InvalidModelOutput, StructuredOperation, frozenMapping, modelConfiguration, optionalBool, optionalConfidence,
  probabilityDistribution, softmax, type ReadonlyMapping,
} from './structured.js';

export interface ScoreFields {
  /** Per-level probabilities keyed by the rubric index (``'0'``, ``'1'``, ...). */
  distribution?: Readonly<Record<string | number, number>> | null;
  confidence?: number | null;
  abstained?: boolean;
}

/** Rubric level ``value`` with optional per-level distribution (keyed by level index) and confidence. */
export class ScoreResult {
  static readonly qualifiedName: string = 'tensorcode.ops.text.score.ScoreResult';
  static readonly recordFields = ['value', 'distribution', 'confidence', 'abstained'] as const;
  readonly value: number | null;
  readonly distribution: ReadonlyMapping<number> | null;
  readonly confidence: number | null;
  readonly abstained: boolean;

  constructor(value: number | null, fields: ScoreFields = {}) {
    this.value = value;
    this.distribution = fields.distribution === null || fields.distribution === undefined ? null : frozenMapping({ ...fields.distribution });
    this.confidence = fields.confidence ?? null;
    this.abstained = fields.abstained ?? false;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): ScoreResult {
    return new ScoreResult(fields.value as number | null, fields as ScoreFields);
  }

  toRecord(): Record<string, unknown> {
    return { value: this.value, distribution: this.distribution, confidence: this.confidence, abstained: this.abstained };
  }
}

/**
 * Score messages against an authored rubric with an owned seq2seq model.
 *
 * ``decoding: 'likelihood'`` scores every rubric level in one encoder pass and
 * returns the probability-weighted level as ``value``. ``fromModel``
 * explicitly wraps an external provider without owned artifacts.
 */
export class Score extends StructuredOperation<ScoreResult> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text.score.Score';
  static override readonly schemaName: string = 'tensorcode.score';
  static override readonly semanticFields: ReadonlySet<string> = new Set(['instructions', 'rubric']);
  declare rubric: readonly string[];

  _alternatives(): Alternative[] {
    return this.rubric.map((level, index) => [`${index}: ${level}`, level] as const);
  }

  _fromScores(scores: readonly number[]): ScoreResult {
    const probabilities = softmax(scores);
    let value = 0;
    probabilities.forEach((probability, index) => { value += index * probability; });
    const distribution: Record<string, number> = {};
    probabilities.forEach((probability, index) => { distribution[String(index)] = probability; });
    return new ScoreResult(value, { distribution, confidence: Math.max(...probabilities), abstained: false });
  }

  _targetWeights(result: ScoreResult): number[] {
    if (result.abstained) throw new ValueError('likelihood decoding has no abstention alternative');
    const levels = this.rubric.map((_, index) => index);
    if (result.distribution !== null) return levels.map((index) => result.distribution![String(index)]!);
    const value = result.value!;
    if (value !== Math.trunc(value)) throw new ValueError('likelihood score targets need a distribution or an integer level');
    return levels.map((index) => (index === Math.trunc(value) ? 1 : 0));
  }

  protected override _configureSemantics(config: Record<string, unknown>): void {
    super._configureSemantics(config);
    const rubric = config.rubric ?? [];
    if (!Array.isArray(rubric)) throw new ValueError('rubric must be a sequence of strings');
    this.rubric = Object.freeze([...rubric]);
    if (!this.rubric.length || !this.rubric.every((level) => typeof level === 'string')) {
      throw new ValueError('rubric must contain one or more string levels');
    }
  }

  responseSchema(): Record<string, unknown> {
    const keys = this.rubric.map((_, index) => String(index));
    const properties: Record<string, unknown> = {};
    for (const key of keys) properties[key] = { type: 'number', minimum: 0, maximum: 1, description: this.rubric[Number(key)] };
    return {
      type: 'object',
      properties: {
        score: { type: ['number', 'null'], minimum: 0, maximum: this.rubric.length - 1 },
        distribution: { type: ['object', 'null'], properties, required: [...keys], additionalProperties: false },
        confidence: { type: ['number', 'null'], minimum: 0, maximum: 1 },
        abstained: { type: 'boolean' },
      },
      required: ['score', 'distribution', 'confidence', 'abstained'],
      additionalProperties: false,
    };
  }

  _parse(value: Readonly<Record<string, unknown>>): ScoreResult {
    const abstained = optionalBool(value, 'abstained');
    const rawScore = value.score ?? null;
    let score: number | null;
    if (abstained) {
      if (rawScore !== null) throw new InvalidModelOutput('An abstained score must have score null');
      score = null;
    } else {
      if (typeof rawScore !== 'number') throw new InvalidModelOutput('score must be a number');
      score = rawScore;
      if (!(score >= 0 && score <= this.rubric.length - 1)) throw new InvalidModelOutput('score is outside the configured rubric');
    }
    const keys = this.rubric.map((_, index) => String(index));
    const rawDistribution = value.distribution ?? null;
    if (rawDistribution !== null && (!isPlainObject(rawDistribution) || Object.keys(rawDistribution).length !== keys.length
      || !keys.every((key) => Object.hasOwn(rawDistribution, key)))) {
      throw new InvalidModelOutput('distribution keys must be canonical configured rubric indices');
    }
    return new ScoreResult(score, {
      distribution: probabilityDistribution(rawDistribution, keys),
      confidence: optionalConfidence(value),
      abstained,
    });
  }

  override configuration(): JsonObject {
    if (this._owned) return super.configuration();
    return {
      type: 'text_score',
      rubric: [...this.rubric],
      instructions: this.instructions,
      model: modelConfiguration(this.model) as JsonObject,
    };
  }
}
