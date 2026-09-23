/** Explicit authored selection thresholds (Python ``tensorcode/_internal/cognition/policy.py``). */
import { ValueError } from '../../errors.js';
import { isPlainObject } from '../json.js';

export interface Distribution {
  support: number;
  contradiction: number;
  unknown: number;
}

export interface PolicyThresholds {
  min_support?: number;
  max_contradiction?: number;
  max_unknown?: number;
}

const LABELS = ['support', 'contradiction', 'unknown'];

function isInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value);
}

/**
 * Authored thresholds on model scores, not a learned truth criterion.
 *
 * Require support from at least one source, inspect unknown on that strongest
 * supporting source, and veto contradiction from ANY current source. Explicit
 * joint verification instead requires support/unknown on the combined premise,
 * retaining every source contradiction veto. Source trust is caller supplied.
 */
export class SelectionPolicy {
  static readonly qualifiedName: string = 'tensorcode._internal.cognition.policy.SelectionPolicy';
  static readonly recordFields = ['min_support', 'max_contradiction', 'max_unknown'] as const;
  readonly minSupport: number;
  readonly maxContradiction: number;
  readonly maxUnknown: number;

  constructor(thresholds: PolicyThresholds = {}) {
    this.minSupport = thresholds.min_support ?? 0.7;
    this.maxContradiction = thresholds.max_contradiction ?? 0.2;
    this.maxUnknown = thresholds.max_unknown ?? 0.3;
    for (const value of [this.minSupport, this.maxContradiction, this.maxUnknown]) {
      if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
        throw new ValueError('policy thresholds must be finite values in [0, 1]');
      }
    }
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): SelectionPolicy {
    return new SelectionPolicy(fields as PolicyThresholds);
  }

  /** Python ``asdict(policy)``. */
  toRecord(): { min_support: number; max_contradiction: number; max_unknown: number } {
    return { min_support: this.minSupport, max_contradiction: this.maxContradiction, max_unknown: this.maxUnknown };
  }

  accepts(distributions: readonly Distribution[], options: { jointDistribution?: Distribution | null } = {}): boolean {
    if (!distributions.length) return false;
    const joint = options.jointDistribution ?? null;
    const checked = joint !== null ? [...distributions, joint] : [...distributions];
    for (const row of checked) {
      if (!isPlainObject(row) || Object.keys(row).length !== 3 || !LABELS.every((label) => label in row)
        || Object.values(row).some((value) => typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)) {
        throw new ValueError('expected finite support/contradiction/unknown distributions');
      }
    }
    let strongest = joint;
    if (strongest === null) {
      strongest = distributions[0]!;
      for (const row of distributions) if (row.support > strongest.support) strongest = row;
    }
    return strongest.support >= this.minSupport
      && strongest.unknown <= this.maxUnknown
      && Math.max(...checked.map((row) => row.contradiction)) <= this.maxContradiction;
  }

  /** Require complete identified evidence coverage before screening scores. */
  acceptsVerification(verification: Record<string, unknown>, sourceIds: readonly string[], options: { scope: string }): boolean {
    const scope = options.scope;
    const checks = verification.verifications as Record<string, unknown>[];
    const ids = new Set(sourceIds);
    const checkIds = new Set(checks.map((row) => row.source_id));
    if (checks.length !== sourceIds.length || ids.size !== sourceIds.length
      || checkIds.size !== ids.size || [...checkIds].some((id) => !ids.has(id as string))) {
      throw new ValueError('verification sources must match active evidence exactly');
    }
    const joint = verification.joint_verification as Record<string, unknown> | null | undefined;
    if (scope !== 'source' && scope !== 'joint') throw new ValueError('verification_scope must be source or joint');
    if (scope === 'joint') {
      if (!sourceIds.length) return false;
      const jointIds = isPlainObject(joint) ? joint.source_ids : undefined;
      if (!isPlainObject(joint) || !Array.isArray(jointIds) || jointIds.length !== sourceIds.length
        || jointIds.some((id, index) => id !== sourceIds[index]) || joint.scope !== 'joint') {
        throw new ValueError('joint verification sources must match active evidence exactly');
      }
      if (typeof joint.input_truncated !== 'boolean'
        || !isInteger(joint.token_count) || joint.token_count < 1
        || !isInteger(joint.max_tokens) || joint.max_tokens < 1
        || joint.input_truncated !== (joint.token_count > joint.max_tokens)
        || checks.some((row) => typeof row.input_truncated !== 'boolean')) {
        throw new ValueError('joint verification requires explicit input coverage metadata');
      }
    }
    if (checks.some((row) => Boolean(row.input_truncated ?? false))) return false;
    if (scope === 'joint' && Boolean(joint!.input_truncated ?? false)) return false;
    return this.accepts(checks.map((row) => row.distribution as Distribution), {
      jointDistribution: scope === 'joint' ? joint!.distribution as Distribution : null,
    });
  }

  receipt(options: { scope?: string } = {}): Record<string, unknown> {
    const scope = options.scope ?? 'source';
    return {
      ...this.toRecord(), origin: 'authored_policy',
      aggregation: scope === 'joint'
        ? 'joint-premise support/unknown; maximum contradiction across joint and individual sources'
        : 'strongest-source support/unknown; maximum contradiction across current sources',
      semantics: 'model-score screening, not established truth; source trust supplied by caller',
    };
  }
}
