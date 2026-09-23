/**
 * Explicit operands and tensor-valued results for vector candidate operations
 * (Python ``tensorcode/ops/vec/candidates.py``).
 */
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { isPlainObject } from '../../_internal/json.js';
import { Latent } from './latent.js';

export type CandidateMetadata = Readonly<Record<string, unknown>>;

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((size, index) => size === b[index]);
}

/** A query and existing candidates shaped ``(..., N, features)``. */
export class CandidateSet {
  static readonly qualifiedName: string = 'tensorcode.ops.vec.candidates.CandidateSet';
  static readonly recordFields = ['query', 'candidates', 'identities', 'metadata'] as const;
  readonly query: Latent;
  readonly candidates: Latent;
  readonly identities: readonly string[];
  readonly metadata: readonly CandidateMetadata[];

  constructor(query: Latent, candidates: Latent, identities: Iterable<string>, metadata: Iterable<Record<string, unknown>> = []) {
    if (!(query instanceof Latent) || !(candidates instanceof Latent)) {
      throw new TypeError('CandidateSet query and candidates must be Latent objects');
    }
    if (candidates.tensor.ndim < 2) throw new ValueError('Candidate tensor needs a candidate and feature dimension');
    const count = candidates.tensor.shape[candidates.tensor.ndim - 2]!;
    if (count === 0) throw new ValueError('CandidateSet requires at least one candidate');
    if (!sameShape(query.tensor.shape.slice(0, -1), candidates.tensor.shape.slice(0, -2))) {
      throw new ValueError('Query and candidates must have the same batch shape');
    }
    if (candidates.mask !== null) {
      if (candidates.mask.dtype !== 'bool') throw new ValueError('Candidate availability mask must be boolean');
      if (!candidates.mask.any(-1).all().item()) throw new ValueError('Every query requires at least one valid candidate');
    }
    const ids = [...identities];
    if (ids.length !== count || !ids.every((value) => typeof value === 'string')) {
      throw new ValueError('Candidate identities must be strings matching the candidate count');
    }
    if (ids.some((value) => !value) || new Set(ids).size !== ids.length) {
      throw new ValueError('Candidate identities must be unique nonempty strings');
    }
    const supplied = [...metadata];
    const rows = supplied.length ? supplied : ids.map(() => ({}));
    if (rows.length !== count || !rows.every((row) => isPlainObject(row))) {
      throw new ValueError('Candidate metadata must match the candidate count');
    }
    this.query = query;
    this.candidates = candidates;
    this.identities = Object.freeze(ids);
    this.metadata = Object.freeze(rows.map((row) => Object.freeze({ ...row })));
    Object.freeze(this);
  }

  /** Number of candidates along the candidate axis. */
  get count(): number {
    return this.candidates.tensor.shape[this.candidates.tensor.ndim - 2]!;
  }

  static fromRecord(fields: Record<string, unknown>): CandidateSet {
    return new CandidateSet(
      fields.query as Latent, fields.candidates as Latent, fields.identities as readonly string[],
      (fields.metadata as readonly Record<string, unknown>[] | undefined) ?? [],
    );
  }

  toRecord(): Record<string, unknown> {
    return { query: this.query, candidates: this.candidates, identities: this.identities, metadata: this.metadata };
  }
}

/** Floating-point scores per candidate with an authored ``meaning`` string. */
export class Scores {
  static readonly qualifiedName: string = 'tensorcode.ops.vec.candidates.Scores';
  static readonly recordFields = ['values', 'meaning', 'candidates'] as const;
  readonly values: Tensor;
  readonly meaning: string;
  readonly candidates: CandidateSet;

  constructor(values: Tensor, meaning: string, candidates: CandidateSet) {
    if (!(candidates instanceof CandidateSet)) throw new TypeError('Scores candidates must be a CandidateSet');
    const expected = candidates.candidates.tensor.shape.slice(0, -1);
    if (!(values instanceof Tensor) || !sameShape(values.shape, expected)) {
      throw new ValueError(`Scores must have shape (${expected.join(', ')}${expected.length === 1 ? ',' : ''})`);
    }
    if (!values.isFloatingPoint) throw new ValueError('Scores must be floating-point tensors');
    if (typeof meaning !== 'string' || !meaning.trim()) throw new ValueError('Score meaning must be a nonempty string');
    this.values = values;
    this.meaning = meaning;
    this.candidates = candidates;
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Scores {
    return new Scores(fields.values as Tensor, fields.meaning as string, fields.candidates as CandidateSet);
  }

  toRecord(): Record<string, unknown> {
    return { values: this.values, meaning: this.meaning, candidates: this.candidates };
  }
}

/** Gather candidate-axis items while retaining gradient and provenance. */
export function gatherLatent(candidates: Latent, indices: Tensor): Latent {
  const features = candidates.tensor.shape[candidates.tensor.ndim - 1]!;
  const tensorIndex = indices.unsqueeze(-1).expand([...indices.shape, features]);
  const gathered = candidates.tensor.gather(-2, tensorIndex);
  const mask = candidates.mask === null ? null : candidates.mask.gather(-1, indices);
  let coordinates: Tensor | null = null;
  if (candidates.coordinates !== null) {
    const width = candidates.coordinates.shape[candidates.coordinates.ndim - 1]!;
    coordinates = candidates.coordinates.gather(-2, indices.unsqueeze(-1).expand([...indices.shape, width]));
  }
  return candidates.withTensor(gathered, { mask, coordinates });
}

/** Explicit tensor-to-JavaScript conversion used by result convenience accessors (tuples are frozen arrays). */
export function selectPython<T>(values: readonly T[], indices: Tensor): unknown {
  const convert = (item: unknown): unknown => (Array.isArray(item) ? Object.freeze(item.map(convert)) : values[item as number]);
  return convert(indices.detach().tolist());
}
