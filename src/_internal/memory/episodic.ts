/**
 * Bounded source memory with caller-supplied embeddings, not a truth store
 * (Python ``tensorcode/_internal/memory/episodic.py``).
 */
import { ValueError } from '../../errors.js';
import { Evidence, RetrievalHit, recordEquals, requireText } from '../../tools/cognition.js';
import { fsum } from '../numeric.js';

export { fsum } from '../numeric.js';

/** Python ``math.hypot`` of a vector. */
function hypot(values: readonly number[]): number {
  let scale = 0;
  for (const value of values) scale = Math.max(scale, Math.abs(value));
  if (!Number.isFinite(scale)) return scale;
  if (scale === 0) return 0;
  return scale * Math.sqrt(fsum(values.map((value) => (value / scale) ** 2)));
}

/** Validate and L2-normalize an embedding (Python ``_vector``). */
export function normalizedVector(values: unknown): readonly number[] {
  let vector: number[];
  try {
    if (values === null || values === undefined || typeof (values as Iterable<unknown>)[Symbol.iterator] !== 'function' || typeof values === 'string') {
      throw new TypeError('not iterable');
    }
    vector = [...(values as Iterable<unknown>)].map((value) => {
      if (typeof value !== 'number' && typeof value !== 'boolean') throw new TypeError('not numeric');
      return Number(value);
    });
  } catch (error) {
    throw new ValueError('embedding must be a numeric vector', { cause: error });
  }
  if (!vector.length || !vector.every(Number.isFinite)) throw new ValueError('embedding must be nonempty and finite');
  const norm = hypot(vector);
  if (!norm || !Number.isFinite(norm)) throw new ValueError('embedding must have finite nonzero norm');
  return Object.freeze(vector.map((value) => value / norm));
}

function sameVector(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

/**
 * FIFO capacity; cosine scores indicate embedding proximity only.
 *
 * The owning tool must supply a fingerprint covering its encoder configuration
 * AND weights. Changed fingerprints fail query until all vectors are rebuilt.
 */
export class EpisodicMemory {
  static readonly qualifiedName: string = 'tensorcode._internal.memory.episodic.EpisodicMemory';
  readonly capacity: number;
  private fingerprintValue: string;
  /** Stored entries by evidence id, in insertion order (score is always 0). */
  entries = new Map<string, RetrievalHit>();
  vectors = new Map<string, readonly number[]>();

  constructor(options: { capacity?: number; modelFingerprint: string }) {
    const capacity = options.capacity ?? 256;
    if (typeof capacity !== 'number' || !Number.isInteger(capacity) || capacity < 1) throw new ValueError('capacity must be positive');
    requireText(options.modelFingerprint, 'model_fingerprint');
    this.capacity = capacity;
    this.fingerprintValue = options.modelFingerprint;
  }

  get modelFingerprint(): string {
    return this.fingerprintValue;
  }

  get size(): number {
    return this.entries.size;
  }

  insert(evidence: Evidence, vector: unknown, options: {
    episodeId: string; question?: string; outcome?: string; modelFingerprint?: string | null;
  }): void {
    if (!(evidence instanceof Evidence) || evidence.constructor !== Evidence) throw new ValueError('memory stores source Evidence only');
    const question = options.question ?? '';
    const outcome = options.outcome ?? '';
    requireText(options.episodeId, 'episode_id');
    if (typeof question !== 'string' || typeof outcome !== 'string') throw new ValueError('metadata must be text');
    const fingerprint = options.modelFingerprint ?? null;
    if (fingerprint !== null && fingerprint !== this.modelFingerprint) throw new ValueError('stale embedding index; rebuild required');
    const normalized = normalizedVector(vector);
    const first = this.vectors.values().next();
    if (!first.done && normalized.length !== first.value.length) throw new ValueError('embedding dimension mismatch');
    const entry = new RetrievalHit(evidence, 0, options.episodeId, question, outcome);
    const existing = this.entries.get(evidence.id);
    if (existing !== undefined) {
      if (!recordEquals(existing, entry) || !sameVector(this.vectors.get(evidence.id)!, normalized)) {
        throw new ValueError('evidence id conflicts with stored content or embedding');
      }
      return;
    }
    const entries = new Map(this.entries);
    const vectors = new Map(this.vectors);
    entries.set(evidence.id, entry);
    vectors.set(evidence.id, normalized);
    while (entries.size > this.capacity) {
      const oldest = entries.keys().next().value as string;
      entries.delete(oldest);
      vectors.delete(oldest);
    }
    this.entries = entries;
    this.vectors = vectors;
  }

  query(vector: unknown, options: { modelFingerprint: string; k?: number; excludeEpisodeId?: string | null }): readonly RetrievalHit[] {
    if (options.modelFingerprint !== this.modelFingerprint) throw new ValueError('stale embedding index; rebuild required');
    const k = options.k ?? 5;
    if (typeof k !== 'number' || !Number.isInteger(k) || k < 0) throw new ValueError('k must be nonnegative');
    const normalized = normalizedVector(vector);
    const first = this.vectors.values().next();
    if (!first.done && normalized.length !== first.value.length) throw new ValueError('embedding dimension mismatch');
    const exclude = options.excludeEpisodeId ?? null;
    const hits: RetrievalHit[] = [];
    for (const [key, entry] of this.entries) {
      if (exclude !== null && entry.episodeId === exclude) continue;
      const stored = this.vectors.get(key)!;
      const score = Math.max(-1, Math.min(1, fsum(normalized.map((value, index) => value * stored[index]!))));
      hits.push(new RetrievalHit(entry.evidence, score, entry.episodeId, entry.question, entry.outcome));
    }
    // Stable descending sort, like Python ``sorted(key=-score)``.
    return Object.freeze(hits.map((hit, index) => ({ hit, index }))
      .sort((a, b) => (b.hit.score - a.hit.score) || (a.index - b.index))
      .slice(0, k).map(({ hit }) => hit));
  }

  remove(evidenceId: string): void {
    if (!this.entries.has(evidenceId)) throw new ValueError(`unknown evidence id ${JSON.stringify(evidenceId)}`);
    const entries = new Map(this.entries);
    const vectors = new Map(this.vectors);
    entries.delete(evidenceId);
    vectors.delete(evidenceId);
    this.entries = entries;
    this.vectors = vectors;
  }

  rebuildIndex(vectorsByEvidenceId: Record<string, unknown> | Map<string, unknown>, options: { modelFingerprint: string }): void {
    requireText(options.modelFingerprint, 'model_fingerprint');
    const supplied = vectorsByEvidenceId instanceof Map ? vectorsByEvidenceId : new Map(Object.entries(vectorsByEvidenceId));
    if (supplied.size !== this.entries.size || [...supplied.keys()].some((key) => !this.entries.has(key))) {
      throw new ValueError('rebuild must provide exactly all stored evidence ids');
    }
    const vectors = new Map<string, readonly number[]>();
    for (const [key, value] of supplied) vectors.set(key, normalizedVector(value));
    if (new Set([...vectors.values()].map((value) => value.length)).size > 1) throw new ValueError('embedding dimension mismatch');
    // Keep insertion (FIFO) order of the stored entries.
    this.vectors = new Map([...this.entries.keys()].map((key) => [key, vectors.get(key)!]));
    this.fingerprintValue = options.modelFingerprint;
  }
}
