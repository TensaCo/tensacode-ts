/**
 * Bounded, immutable session data (Python ``tensorcode/tools/cognition.py``).
 * Records do not infer facts or execute policies.
 *
 * Every record is a frozen value object with Python snake_case persisted fields
 * (``static recordFields`` / ``toRecord()`` / ``static fromRecord()``), so the
 * tracer and session codecs walk them like Python dataclasses. Use
 * {@link recordEquals} for Python dataclass ``==`` semantics.
 */
import { ValueError } from '../errors.js';
import { canonicalJson } from '../_internal/json.js';

/** Validate a nonempty string (Python ``_text``). */
export function requireText(value: unknown, name: string): asserts value is string {
  if (typeof value !== 'string' || !value) throw new ValueError(`${name} must be a nonempty string`);
}

/** Python ``dataclasses.asdict`` of a record (nested records and tuples become plain data). */
export function asdict(record: { toRecord(): Record<string, unknown> }): Record<string, unknown> {
  const convert = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(convert);
    if (value !== null && typeof value === 'object' && typeof (value as { toRecord?: unknown }).toRecord === 'function') {
      return asdict(value as { toRecord(): Record<string, unknown> });
    }
    if (value !== null && typeof value === 'object') {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, convert(item)]));
    }
    return value;
  };
  return Object.fromEntries(Object.entries(record.toRecord()).map(([key, value]) => [key, convert(value)]));
}

/** Python dataclass equality: same class and equal fields. */
export function recordEquals(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return false;
  if (a.constructor !== b.constructor || typeof (a as { toRecord?: unknown }).toRecord !== 'function') return false;
  try {
    return canonicalJson(asdict(a as { toRecord(): Record<string, unknown> }))
      === canonicalJson(asdict(b as { toRecord(): Record<string, unknown> }));
  } catch {
    return false;
  }
}

/** Source-identified text supplied by the caller; the unit of sourced evidence. */
export class Evidence {
  static readonly qualifiedName: string = 'tensorcode.tools.cognition.Evidence';
  static readonly recordFields = ['id', 'text', 'source_id'] as const;
  readonly id: string;
  readonly text: string;
  readonly sourceId: string;

  constructor(id: string, text: string, sourceId: string) {
    requireText(id, 'id');
    requireText(text, 'text');
    requireText(sourceId, 'source_id');
    this.id = id;
    this.text = text;
    this.sourceId = sourceId;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Evidence {
    return new Evidence(fields.id as string, fields.text as string, fields.source_id as string);
  }

  toRecord(): { id: string; text: string; source_id: string } {
    return { id: this.id, text: this.text, source_id: this.sourceId };
  }

  equals(other: unknown): boolean {
    return recordEquals(this, other);
  }
}

export type HypothesisOrigin = 'generated' | 'supplied';

/** A candidate interpretation, marked ``generated`` or ``supplied``; never evidence. */
export class Hypothesis {
  static readonly qualifiedName: string = 'tensorcode.tools.cognition.Hypothesis';
  static readonly recordFields = ['id', 'text', 'origin', 'model_provenance'] as const;
  readonly id: string;
  readonly text: string;
  readonly origin: HypothesisOrigin;
  readonly modelProvenance: string;

  /** ``modelProvenance`` is required so provenance is always stated, including for supplied text. */
  constructor(id: string, text: string, options: { origin?: HypothesisOrigin; modelProvenance: string }) {
    if (options === undefined || options === null || !('modelProvenance' in options)) {
      throw new TypeError("Hypothesis missing required keyword-only argument: 'model_provenance'");
    }
    requireText(id, 'id');
    requireText(text, 'text');
    const origin = options.origin ?? 'generated';
    if (origin !== 'generated' && origin !== 'supplied') throw new ValueError('origin must be generated or supplied');
    requireText(options.modelProvenance, 'model_provenance');
    this.id = id;
    this.text = text;
    this.origin = origin;
    this.modelProvenance = options.modelProvenance;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Hypothesis {
    return new Hypothesis(fields.id as string, fields.text as string, {
      origin: fields.origin as HypothesisOrigin, modelProvenance: fields.model_provenance as string,
    });
  }

  toRecord(): { id: string; text: string; origin: HypothesisOrigin; model_provenance: string } {
    return { id: this.id, text: this.text, origin: this.origin, model_provenance: this.modelProvenance };
  }

  equals(other: unknown): boolean {
    return recordEquals(this, other);
  }
}

/** Model scores for one evidence/hypothesis pair, with model provenance and revision. */
export class Assessment {
  static readonly qualifiedName: string = 'tensorcode.tools.cognition.Assessment';
  static readonly recordFields = ['evidence_id', 'hypothesis_id', 'scores', 'model_provenance', 'revision'] as const;
  /** Python ``Mapping[str, float]`` (integral scores persist as ``1.0``). */
  static readonly recordFloatFields = ['scores'] as const;
  readonly evidenceId: string;
  readonly hypothesisId: string;
  /** A frozen copy (Python ``MappingProxyType``); assignment throws ``TypeError``. */
  readonly scores: Readonly<Record<string, number>>;
  readonly modelProvenance: string;
  readonly revision: number;

  constructor(evidenceId: string, hypothesisId: string, scores: Record<string, number>, modelProvenance: string, revision = 0) {
    requireText(evidenceId, 'evidence_id');
    requireText(hypothesisId, 'hypothesis_id');
    requireText(modelProvenance, 'model_provenance');
    if (scores === null || typeof scores !== 'object' || Array.isArray(scores) || !Object.keys(scores).length) {
      throw new ValueError('scores must be a nonempty mapping');
    }
    for (const [label, score] of Object.entries(scores)) {
      requireText(label, 'score label');
      if (typeof score !== 'number' || !Number.isFinite(score)) throw new ValueError('scores must be finite numbers');
    }
    if (typeof revision !== 'number' || !Number.isInteger(revision) || revision < 0) throw new ValueError('invalid assessment revision');
    this.evidenceId = evidenceId;
    this.hypothesisId = hypothesisId;
    this.scores = Object.freeze({ ...scores });
    this.modelProvenance = modelProvenance;
    this.revision = revision;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Assessment {
    return new Assessment(fields.evidence_id as string, fields.hypothesis_id as string, fields.scores as Record<string, number>,
      fields.model_provenance as string, fields.revision === undefined ? 0 : fields.revision as number);
  }

  toRecord(): { evidence_id: string; hypothesis_id: string; scores: Record<string, number>; model_provenance: string; revision: number } {
    return {
      evidence_id: this.evidenceId, hypothesis_id: this.hypothesisId, scores: { ...this.scores },
      model_provenance: this.modelProvenance, revision: this.revision,
    };
  }

  /** Python ``dataclasses.replace``. */
  replace(changes: { revision?: number }): Assessment {
    return new Assessment(this.evidenceId, this.hypothesisId, { ...this.scores }, this.modelProvenance, changes.revision ?? this.revision);
  }

  equals(other: unknown): boolean {
    return recordEquals(this, other);
  }
}

/** A caller-supplied objective with an explicit source. */
export class Goal {
  static readonly qualifiedName: string = 'tensorcode.tools.cognition.Goal';
  static readonly recordFields = ['id', 'text', 'source_id'] as const;
  readonly id: string;
  readonly text: string;
  readonly sourceId: string;

  constructor(id: string, text: string, sourceId: string) {
    requireText(id, 'id');
    requireText(text, 'text');
    requireText(sourceId, 'source_id');
    this.id = id;
    this.text = text;
    this.sourceId = sourceId;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Goal {
    return new Goal(fields.id as string, fields.text as string, fields.source_id as string);
  }

  toRecord(): { id: string; text: string; source_id: string } {
    return { id: this.id, text: this.text, source_id: this.sourceId };
  }

  equals(other: unknown): boolean {
    return recordEquals(this, other);
  }
}

/** Inert textual plan steps and their predicted outcomes; never executed directly. */
export class Plan {
  static readonly qualifiedName: string = 'tensorcode.tools.cognition.Plan';
  static readonly recordFields = ['id', 'steps', 'predicted_outcomes'] as const;
  readonly id: string;
  readonly steps: readonly string[];
  readonly predictedOutcomes: readonly string[];

  constructor(id: string, steps: readonly string[], predictedOutcomes: readonly string[]) {
    requireText(id, 'id');
    const validated = (values: unknown, name: string): readonly string[] => {
      if (!Array.isArray(values)) throw new ValueError(`${name} must be a sequence`);
      for (const value of values) requireText(value, name);
      return Object.freeze([...values] as string[]);
    };
    this.id = id;
    this.steps = validated(steps, 'steps');
    this.predictedOutcomes = validated(predictedOutcomes, 'predicted_outcomes');
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Plan {
    return new Plan(fields.id as string, fields.steps as string[], fields.predicted_outcomes as string[]);
  }

  toRecord(): { id: string; steps: readonly string[]; predicted_outcomes: readonly string[] } {
    return { id: this.id, steps: this.steps, predicted_outcomes: this.predictedOutcomes };
  }

  equals(other: unknown): boolean {
    return recordEquals(this, other);
  }
}

/** Externally supplied actual feedback, never generated from hypotheses. */
export class Observation {
  static readonly qualifiedName: string = 'tensorcode.tools.cognition.Observation';
  static readonly recordFields = ['id', 'text', 'source_id'] as const;
  readonly id: string;
  readonly text: string;
  readonly sourceId: string;

  constructor(id: string, text: string, sourceId: string) {
    requireText(id, 'id');
    requireText(text, 'text');
    requireText(sourceId, 'source_id');
    this.id = id;
    this.text = text;
    this.sourceId = sourceId;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Observation {
    return new Observation(fields.id as string, fields.text as string, fields.source_id as string);
  }

  toRecord(): { id: string; text: string; source_id: string } {
    return { id: this.id, text: this.text, source_id: this.sourceId };
  }

  equals(other: unknown): boolean {
    return recordEquals(this, other);
  }
}

/** Evidence recalled from episodic memory with its score and originating episode. */
export class RetrievalHit {
  static readonly qualifiedName: string = 'tensorcode.tools.cognition.RetrievalHit';
  static readonly recordFields = ['evidence', 'score', 'episode_id', 'question', 'outcome'] as const;
  /** Python ``float`` annotations (integral values persist as ``1.0``). */
  static readonly recordFloatFields = ['score'] as const;
  readonly evidence: Evidence;
  readonly score: number;
  readonly episodeId: string;
  readonly question: string;
  readonly outcome: string;

  constructor(evidence: Evidence, score: number, episodeId: string, question = '', outcome = '') {
    this.evidence = evidence;
    this.score = score;
    this.episodeId = episodeId;
    this.question = question;
    this.outcome = outcome;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): RetrievalHit {
    const evidence = fields.evidence instanceof Evidence ? fields.evidence : Evidence.fromRecord(fields.evidence as Record<string, unknown>);
    return new RetrievalHit(evidence, fields.score as number, fields.episode_id as string,
      (fields.question as string | undefined) ?? '', (fields.outcome as string | undefined) ?? '');
  }

  toRecord(): { evidence: Evidence; score: number; episode_id: string; question: string; outcome: string } {
    return { evidence: this.evidence, score: this.score, episode_id: this.episodeId, question: this.question, outcome: this.outcome };
  }

  equals(other: unknown): boolean {
    return recordEquals(this, other);
  }
}
