/**
 * Private model-owned episodic retrieval and persistence (Python
 * ``tensorcode/_internal/memory/learned.py``).
 */
import { noGrad } from '../../nn/autograd.js';
import type { Module } from '../../nn/module.js';
import type { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { deepCopy, isPlainObject, type JsonObject } from '../json.js';
import type { RankOperation } from '../ranking.js';
import type { RetrievalEncoder } from '../retrieval.js';
import { Evidence, asdict, requireText, type RetrievalHit } from '../../tools/cognition.js';
import { EpisodicMemory } from './episodic.js';
import { ModelFingerprint } from '../cognition/locking.js';

/** The investigator surface episodic memory needs (structural, avoids import cycles). */
export interface MemoryOwner extends Module {
  readonly rank: RankOperation;
  readonly episodicEncoder: RetrievalEncoder | null;
  configuration(): JsonObject;
}

/** Run ``fn`` with every module of ``root`` in eval mode, then restore each module's mode. */
export function withEvalModes<T>(root: Module, fn: () => T): T {
  const modes = root.modules().map((module) => [module, module.training] as const);
  try {
    root.eval();
    return fn();
  } finally {
    for (const [module, training] of modes) module.training = training;
  }
}

export interface EpisodicSnapshot {
  schema_version: 1;
  capacity: number;
  records: { evidence: { id: string; text: string; source_id: string }; episode_id: string; question: string; outcome: string }[];
}

/**
 * Owned retrieval embeddings, or explicit rank-pooling artifact fallback.
 *
 * Dedicated sentence encoders preserve their trained embedding geometry. Rank
 * encoder pooling is retained for existing artifacts, without claiming that
 * ranking supervision learned a useful cosine retrieval space.
 */
export class LearnedEpisodicMemory {
  static readonly qualifiedName: string = 'tensorcode._internal.memory.learned.LearnedEpisodicMemory';
  readonly investigator: MemoryOwner;
  private fingerprintCache: ModelFingerprint;
  memory: EpisodicMemory;
  private evidence = new Map<string, Evidence>();

  constructor(investigator: MemoryOwner, options: { capacity?: number } = {}) {
    this.investigator = investigator;
    this.fingerprintCache = new ModelFingerprint();
    this.memory = new EpisodicMemory({ capacity: options.capacity ?? 256, modelFingerprint: this.fingerprint });
  }

  /** Content fingerprint of the embedding model (configuration and weights). */
  get fingerprint(): string {
    const dedicated = this.investigator.episodicEncoder;
    if (dedicated) return this.fingerprintCache.compute([['episodic_encoder', dedicated]], dedicated.configuration());
    const rank = this.investigator.rank;
    const modules: [string, Module][] = [['encode', rank.encode]];
    if (rank.tokenizer !== null) modules.push(['projection', rank.projection!]);
    return this.fingerprintCache.compute(modules, rank.configuration());
  }

  get metadata(): JsonObject {
    const dedicated = this.investigator.episodicEncoder;
    if (dedicated) return dedicated.metadata;
    const rank = this.investigator.rank;
    return {
      encoder: 'rank_encoder_pooling', pooling: 'masked_mean', normalized: true,
      max_tokens: rank.config.max_tokens!, dimensions: rank.config.dimensions!,
      foundation: deepCopy(rank.config.foundation ?? { initialization: 'configured_weights' }),
      semantics: 'legacy rank-space cosine proximity; retrieval quality is not established by rank training',
    };
  }

  /** Required after direct ``tensor.data`` writes to model tensors. */
  invalidateFingerprint(): void {
    this.fingerprintCache.invalidate();
  }

  embed(text: string): number[] {
    requireText(text, 'text');
    const dedicated = this.investigator.episodicEncoder;
    if (dedicated) return (dedicated.receipt([text]).embeddings as number[][])[0]!;
    const rank = this.investigator.rank;
    return withEvalModes(rank, () => noGrad(() => {
      let encoded: Tensor;
      if (rank.tokenizer === null) {
        encoded = rank.encode.call(rank.tokens(text)) as Tensor;
      } else {
        const tokens = rank.tokenizer.encodeTensors([text], { padding: true, truncation: true, maxLength: rank.config.max_tokens as number });
        const projected = rank.projection!.call(rank.encode.call(tokens)) as Tensor;
        encoded = projected.select(0, 0).maskedSelect(tokens.attention_mask.select(0, 0).bool());
      }
      return encoded.mean(0).detach().toArray();
    }));
  }

  remember(evidence: Evidence, options: { episodeId: string; question?: string; outcome?: string }): void {
    const fingerprint = this.fingerprint;
    if (fingerprint !== this.memory.modelFingerprint) throw new ValueError('stale embedding index; rebuild required');
    this.memory.insert(evidence, this.embed(evidence.text), {
      episodeId: options.episodeId, question: options.question ?? '', outcome: options.outcome ?? '', modelFingerprint: fingerprint,
    });
    const ids = new Set(this.memory.entries.keys());
    this.evidence = new Map([...this.evidence].filter(([key]) => ids.has(key)));
    this.evidence.set(evidence.id, evidence);
  }

  retrieve(question: string, options: { k?: number; excludeEpisodeId?: string | null } = {}): readonly RetrievalHit[] {
    const fingerprint = this.fingerprint;
    if (fingerprint !== this.memory.modelFingerprint) throw new ValueError('stale embedding index; rebuild required');
    return this.memory.query(this.embed(question), {
      modelFingerprint: fingerprint, k: options.k ?? 5, excludeEpisodeId: options.excludeEpisodeId ?? null,
    });
  }

  /** Re-encode every stored source after weights change. */
  rebuildIndex(): void {
    const fingerprint = this.fingerprint;
    const vectors = new Map([...this.evidence].map(([key, value]) => [key, this.embed(value.text)]));
    this.memory.rebuildIndex(vectors, { modelFingerprint: fingerprint });
  }

  /** An independent copy sharing the model and fingerprint cache. */
  fork(): LearnedEpisodicMemory {
    const result = Object.create(LearnedEpisodicMemory.prototype) as LearnedEpisodicMemory;
    const writable = result as unknown as {
      investigator: MemoryOwner; fingerprintCache: ModelFingerprint; memory: EpisodicMemory; evidence: Map<string, Evidence>;
    };
    writable.investigator = this.investigator;
    writable.fingerprintCache = this.fingerprintCache;
    writable.memory = new EpisodicMemory({ capacity: this.memory.capacity, modelFingerprint: this.memory.modelFingerprint });
    writable.memory.entries = new Map(this.memory.entries);
    writable.memory.vectors = new Map(this.memory.vectors);
    writable.evidence = new Map(this.evidence);
    return result;
  }

  remove(evidenceId: string): void {
    this.memory.remove(evidenceId);
    this.evidence.delete(evidenceId);
  }

  snapshot(): EpisodicSnapshot {
    return {
      schema_version: 1, capacity: this.memory.capacity,
      records: [...this.memory.entries.values()].map((entry) => ({
        evidence: asdict(entry.evidence) as EpisodicSnapshot['records'][number]['evidence'],
        episode_id: entry.episodeId, question: entry.question, outcome: entry.outcome,
      })),
    };
  }

  static fromSnapshot(snapshot: unknown, options: { investigator: MemoryOwner }): LearnedEpisodicMemory {
    if (!isPlainObject(snapshot) || Object.keys(snapshot).length !== 3 || !['schema_version', 'capacity', 'records'].every((key) => key in snapshot)
      || typeof snapshot.schema_version !== 'number' || !Number.isInteger(snapshot.schema_version) || snapshot.schema_version !== 1) {
      throw new ValueError('invalid episodic snapshot schema');
    }
    if (!Array.isArray(snapshot.records)) throw new ValueError('episodic records must be an array');
    const result = new LearnedEpisodicMemory(options.investigator, { capacity: snapshot.capacity as number });
    if (snapshot.records.length > result.memory.capacity) throw new ValueError('episodic snapshot exceeds capacity');
    const seen = new Set<string>();
    for (const row of snapshot.records) {
      const keys = ['evidence', 'episode_id', 'question', 'outcome'];
      if (!isPlainObject(row) || Object.keys(row).length !== 4 || !keys.every((key) => key in row) || !isPlainObject(row.evidence)
        || Object.keys(row.evidence).length !== 3 || !['id', 'text', 'source_id'].every((key) => key in (row.evidence as object))) {
        throw new ValueError('invalid episodic record schema');
      }
      const evidence = Evidence.fromRecord(row.evidence);
      if (seen.has(evidence.id)) throw new ValueError('duplicate episodic evidence id');
      seen.add(evidence.id);
      result.remember(evidence, { episodeId: row.episode_id as string, question: row.question as string, outcome: row.outcome as string });
    }
    return result;
  }
}
