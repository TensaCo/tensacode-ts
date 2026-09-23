/**
 * Explicit source-grounded session composition; generated claims remain
 * hypotheses (Python ``tensorcode/_internal/cognition/session.py``).
 */
import { readFile } from 'node:fs/promises';
import { noGrad } from '../../nn/autograd.js';
import { ValueError } from '../../errors.js';
import {
  deepCopy, isPlainObject, parseJsonStrict, pythonJsonDumps, type JsonObject, type JsonValue,
} from '../json.js';
import { atomicWriteFile } from '../files.js';
import { conversationBlock, conversationContext, type ConversationRow } from '../conversation.js';
import { casefold, wordTokens } from '../text/casefold.js';
import { Assessment, Evidence, Hypothesis, asdict, recordEquals, requireText, type RetrievalHit } from '../../tools/cognition.js';
import { CognitiveState } from './state.js';
import { SelectionPolicy, type PolicyThresholds } from './policy.js';
import { cognitionFingerprint, type ModelFingerprint } from './locking.js';
import { sessionFloatKeys } from '../sessions/floats.js';
import { LearnedEpisodicMemory, withEvalModes, type EpisodicSnapshot, type MemoryOwner } from '../memory/learned.js';
import type { Investigator } from '../../tools/investigator.js';

export interface MemoryOptions {
  capacity?: number;
  top_k?: number;
}

export interface CognitiveSessionOptions {
  state?: CognitiveState | null;
  policy?: SelectionPolicy | PolicyThresholds | null;
  memory?: LearnedEpisodicMemory | MemoryOptions | null;
}

export interface CognitiveSnapshot {
  schema_version: 2;
  state: JsonObject;
  evidence_lineage: Record<string, string[]>;
  active_evidence: Record<string, string>;
  policy: { min_support: number; max_contradiction: number; max_unknown: number };
  inactive_evidence: string[];
  retrieval_k: number;
  memory: EpisodicSnapshot | null;
  model_provenance: string | null;
  episode: number;
}

const SNAPSHOT_FIELDS = [
  'schema_version', 'state', 'active_evidence', 'evidence_lineage', 'policy', 'inactive_evidence', 'retrieval_k', 'memory',
  'model_provenance', 'episode',
];

function isInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value);
}

function tokenLength(encoder: { tokenizer: { encode(text: string): { inputIds: number[][] } } | null }, text: string): number {
  return encoder.tokenizer !== null ? encoder.tokenizer.encode(text).inputIds[0]!.length : wordTokens(casefold(text)).length;
}

/**
 * Persistent source evidence, revisable hypotheses and explicit abstention.
 *
 * Session snapshots contain data and policy only. Pass the Investigator model
 * separately on restore. No generated hypothesis is inserted as evidence or an
 * actual observation. Methods commit state only after successful validation.
 */
export class CognitiveSession {
  static readonly qualifiedName: string = 'tensorcode._internal.cognition.session.CognitiveSession';
  readonly investigator: Investigator;
  readonly policy: SelectionPolicy;
  memory: LearnedEpisodicMemory | null;
  retrievalK = 5;
  private currentState: CognitiveState;
  private inactiveEvidence = new Set<string>();
  private episode = 0;
  private activeEvidenceIds = new Map<string, string>();
  private evidenceLineage = new Map<string, readonly string[]>();
  private fingerprint: ModelFingerprint;
  private observedModel: string | null;

  constructor(investigator: Investigator, options: CognitiveSessionOptions = {}) {
    this.investigator = investigator;
    const state = options.state ?? new CognitiveState();
    if (!(state instanceof CognitiveState)) throw new ValueError('state must be CognitiveState');
    this.currentState = state;
    let policy = options.policy ?? null;
    if (isPlainObject(policy)) {
      if (Object.keys(policy).some((key) => !['min_support', 'max_contradiction', 'max_unknown'].includes(key))) {
        throw new ValueError('invalid policy configuration');
      }
      policy = new SelectionPolicy(policy as PolicyThresholds);
    }
    this.policy = (policy ?? new SelectionPolicy()) as SelectionPolicy;
    if (!(this.policy instanceof SelectionPolicy)) throw new ValueError('policy must be SelectionPolicy');
    let memory = options.memory ?? null;
    if (isPlainObject(memory)) {
      if (Object.keys(memory).some((key) => key !== 'capacity' && key !== 'top_k')) throw new ValueError('invalid episodic memory configuration');
      const k = (memory as MemoryOptions).top_k ?? 5;
      if (!isInteger(k) || k < 0) throw new ValueError('top_k must be nonnegative');
      this.retrievalK = k;
      memory = new LearnedEpisodicMemory(investigator as unknown as MemoryOwner, { capacity: (memory as MemoryOptions).capacity ?? 256 });
    }
    if (memory !== null && (!(memory instanceof LearnedEpisodicMemory) || memory.investigator !== (investigator as unknown as MemoryOwner))) {
      throw new ValueError('memory must use this session Investigator');
    }
    this.memory = memory as LearnedEpisodicMemory | null;
    this.validateMemoryEvidence(this.currentState.evidence);
    this.activeEvidenceIds = new Map(this.currentState.evidence.map((record) => [record.id, record.id]));
    this.evidenceLineage = new Map(this.currentState.evidence.map((record) => [record.id, Object.freeze([record.id])]));
    // Every session owns separate evidence but shares this model content cache;
    // tensor versions and configuration still invalidate it.
    this.fingerprint = cognitionFingerprint(investigator);
    const assessments = this.currentState.assessments;
    this.observedModel = assessments.length ? assessments[assessments.length - 1]!.modelProvenance : null;
  }

  private validateMemoryEvidence(evidence: readonly Evidence[]): void {
    if (this.memory === null) return;
    for (const record of evidence) {
      const entry = this.memory.memory.entries.get(record.id);
      if (entry !== undefined && !recordEquals(entry.evidence, record)) {
        throw new ValueError('remembered evidence conflicts with immutable session history');
      }
    }
  }

  /** Content fingerprint of the investigator configuration and weights (Python ``_model_identity``). */
  modelIdentity(): string {
    return this.fingerprint.compute([['investigator', this.investigator]], this.investigator.configuration());
  }

  /** Required after direct ``tensor.data`` writes to model tensors. */
  invalidateFingerprint(): void {
    this.fingerprint.invalidate();
    this.memory?.invalidateFingerprint();
  }

  /** Current records; a changed model identity bumps the revision (stale assessments). */
  get state(): CognitiveState {
    if (this.observedModel !== null) {
      const current = this.modelIdentity();
      if (current !== this.observedModel) {
        this.currentState = this.currentState.replace({ revision: this.currentState.revision + 1 });
        this.observedModel = current;
      }
    }
    return this.currentState;
  }

  set state(value: CognitiveState) {
    if (!(value instanceof CognitiveState)) throw new ValueError('state must be CognitiveState');
    this.currentState = value;
  }

  fork(options: { copyMemory?: boolean } = {}): CognitiveSession {
    const copyMemory = options.copyMemory ?? true;
    const memory = copyMemory && this.memory !== null ? this.memory.fork() : this.memory;
    const result = new CognitiveSession(this.investigator, { state: this.state, policy: this.policy, memory });
    result.activeEvidenceIds = new Map(this.activeEvidenceIds);
    result.evidenceLineage = new Map(this.evidenceLineage);
    result.inactiveEvidence = new Set(this.inactiveEvidence);
    result.retrievalK = this.retrievalK;
    result.fingerprint = this.fingerprint;
    result.observedModel = this.observedModel;
    result.episode = this.episode;
    return result;
  }

  get episodeId(): string {
    return `episode-${this.episode}`;
  }

  newEpisode(): CognitiveState {
    this.state = this.state.replace({
      revision: this.state.revision + 1, hypotheses: [], assessments: [], selection: [], selectionRevision: null, goals: [], plans: [],
    });
    this.activeEvidenceIds = new Map();
    this.episode += 1;
    return this.state;
  }

  ingest(evidence: Iterable<Evidence>): CognitiveState {
    const records = [...evidence];
    const updated = this.state.addEvidence(records);
    this.validateMemoryEvidence(updated.evidence);
    const active = new Map(this.activeEvidenceIds);
    const lineage = new Map(this.evidenceLineage);
    const versions = new Set([...lineage.values()].flat());
    for (const record of records) {
      if (this.inactiveEvidence.has(record.id)) throw new ValueError('archived evidence cannot be reactivated implicitly');
      if (!active.has(record.id)) {
        if (versions.has(record.id) && !lineage.has(record.id)) throw new ValueError('revision record cannot become a separate logical source');
        active.set(record.id, record.id);
        if (!lineage.has(record.id)) lineage.set(record.id, Object.freeze([record.id]));
      }
    }
    this.state = updated;
    this.activeEvidenceIds = active;
    this.evidenceLineage = lineage;
    return this.state;
  }

  /** Resolve explicit lineage; never infer relationships from user ID syntax. */
  private resolveLogicalEvidence(evidenceId: string): [Evidence, readonly string[]] {
    requireText(evidenceId, 'evidence_id');
    const lineage = this.evidenceLineage.get(evidenceId);
    if (lineage !== undefined) {
      const latest = lineage[lineage.length - 1]!;
      if (this.inactiveEvidence.has(latest)) throw new ValueError('logical evidence has been removed');
      const source = this.state.evidence.find((record) => record.id === latest)!;
      return [source, lineage];
    }
    if ([...this.evidenceLineage.values()].some((history) => history.includes(evidenceId))) {
      throw new ValueError('revision requires the logical evidence id, not a version');
    }
    if (this.inactiveEvidence.has(evidenceId)) throw new ValueError('archived evidence cannot be revised');
    let source = this.state.evidence.find((record) => record.id === evidenceId) ?? null;
    const entry = this.memory !== null ? this.memory.memory.entries.get(evidenceId) ?? null : null;
    if (source !== null && entry !== null && !recordEquals(source, entry.evidence)) throw new ValueError('evidence id conflicts with remembered content');
    if (source === null && entry !== null) source = entry.evidence;
    if (source === null) throw new ValueError('unknown logical evidence id');
    return [source, Object.freeze([source.id])];
  }

  reviseEvidence(evidenceId: string, text: string, sourceId: string | null = null): CognitiveState {
    const [old, history] = this.resolveLogicalEvidence(evidenceId);
    const current = this.state;
    const newId = `${evidenceId}@${current.revision + 1}`;
    if (current.evidence.some((record) => record.id === newId) || (this.memory !== null && this.memory.memory.entries.has(newId))) {
      throw new ValueError('revision id conflict');
    }
    const replacement = new Evidence(newId, text, sourceId === null || sourceId === undefined ? old.sourceId : sourceId);
    const updated = current.addEvidence([old, replacement]);
    let memory = this.memory;
    if (memory !== null && memory.memory.entries.has(old.id)) {
      const entry = memory.memory.entries.get(old.id)!;
      if (!recordEquals(entry.evidence, old)) throw new ValueError('evidence id conflicts with remembered content');
      memory = memory.fork();
      memory.remove(old.id);
      // Prior outcome feedback described old content, not this correction.
      memory.remember(replacement, { episodeId: entry.episodeId, question: entry.question, outcome: '' });
    }
    const active = new Map(this.activeEvidenceIds);
    active.set(evidenceId, replacement.id);
    const lineage = new Map(this.evidenceLineage);
    lineage.set(evidenceId, Object.freeze([...history, replacement.id]));
    this.state = updated;
    this.activeEvidenceIds = active;
    this.evidenceLineage = lineage;
    this.inactiveEvidence = new Set([...this.inactiveEvidence, old.id]);
    this.memory = memory;
    return this.state;
  }

  removeEvidence(evidenceId: string): CognitiveState {
    const [old, history] = this.resolveLogicalEvidence(evidenceId);
    let updated = this.state.addEvidence([old]);
    updated = updated.replace({ revision: updated.revision + 1 });
    const active = new Map(this.activeEvidenceIds);
    active.delete(evidenceId);
    const lineage = new Map(this.evidenceLineage);
    lineage.set(evidenceId, history);
    this.state = updated;
    this.activeEvidenceIds = active;
    this.evidenceLineage = lineage;
    this.inactiveEvidence = new Set([...this.inactiveEvidence, old.id]);
    return this.state;
  }

  get activeEvidence(): readonly Evidence[] {
    const byId = new Map(this.state.evidence.map((record) => [record.id, record]));
    return Object.freeze([...this.activeEvidenceIds.values()].map((id) => byId.get(id)!));
  }

  investigate(question: string, options: {
    hypotheses?: JsonObject[] | null; count?: number; episodeId?: string | null; conversationContext?: ConversationRow[] | null;
  } = {}): JsonObject {
    requireText(question, 'question');
    const count = options.count ?? 3;
    const supplied = options.hypotheses ?? null;
    const dialogue = conversationContext({ conversation_context: options.conversationContext ?? [] });
    let retrievalQuery = question;
    if (this.memory !== null && dialogue.length) {
      retrievalQuery = `${conversationBlock(dialogue)}Current question: ${question}`;
      const encoder = this.investigator.episodicEncoder ?? this.investigator.rank;
      if (tokenLength(encoder, retrievalQuery) > (encoder.config.max_tokens as number)) {
        throw new ValueError('Conversation and current question exceed retrieval token budget');
      }
    }
    let evidence = [...this.activeEvidence];
    let hits: readonly RetrievalHit[] = [];
    let working = this.state;
    if (this.memory !== null) {
      // Request the bounded index before filtering archived/active sources, so
      // obsolete nearest neighbors cannot crowd out eligible sources.
      const retrieved = this.memory.retrieve(retrievalQuery, {
        k: this.memory.memory.capacity, excludeEpisodeId: options.episodeId ?? this.episodeId,
      });
      const activeIds = new Set(evidence.map((record) => record.id));
      hits = retrieved.filter((hit) => !this.inactiveEvidence.has(hit.evidence.id) && !activeIds.has(hit.evidence.id)).slice(0, this.retrievalK);
      working = working.addEvidence(hits.map((hit) => hit.evidence));
      evidence = [...evidence, ...hits.map((hit) => hit.evidence)];
    }
    // Investigator requires unique source IDs. Immutable evidence IDs provide
    // that identity even when several passages cite the same document.
    const inputs: JsonObject = { question, evidence: evidence.map((record) => ({ source_id: record.id, text: record.text })) };
    if (options.conversationContext !== undefined && options.conversationContext !== null) {
      inputs.conversation_context = dialogue as unknown as JsonValue;
    }
    if (supplied !== null) inputs.hypotheses = deepCopy(supplied);
    const rank = this.investigator.rank;
    const result = withEvalModes(rank, () => {
      rank.clearEncodingCache();
      return noGrad(() => this.investigator.investigate(inputs, { count }));
    });
    // Verification may invalidate stale calibration buffers during inference.
    const provenance = this.modelIdentity();
    const candidates = deepCopy(result.candidates as JsonObject[]);
    const proposals = (result.candidates as JsonObject[]).map((candidate) => new Hypothesis(candidate.id as string, candidate.text as string, {
      origin: supplied === null ? 'generated' : 'supplied', modelProvenance: supplied === null ? provenance : 'caller-supplied',
    }));
    let updated = working.addHypotheses(proposals);
    const assessments: Assessment[] = [];
    const accepted: JsonObject[] = [];
    const byId = new Map(evidence.map((record) => [record.id, record]));
    const scope = this.investigator.config.verification_scope as string;
    for (const candidate of candidates) {
      const checks = candidate.verifications as JsonObject[];
      const checkIds = new Set(checks.map((check) => check.source_id as string));
      if (checks.length !== evidence.length || checkIds.size !== byId.size || [...checkIds].some((id) => !byId.has(id))) {
        throw new ValueError('verification sources must match active evidence exactly');
      }
      candidate.accepted_by_policy = this.policy.acceptsVerification(candidate, [...byId.keys()], { scope });
      for (const check of checks) {
        const evidenceId = check.source_id as string;
        assessments.push(new Assessment(evidenceId, candidate.id as string, check.distribution as Record<string, number>, provenance));
        check.evidence_id = evidenceId;
        check.source_id = byId.get(evidenceId)!.sourceId;
      }
      let truncated = checks.some((check) => Boolean(check.input_truncated ?? false));
      const joint = (candidate.joint_verification ?? null) as JsonObject | null;
      if (joint !== null) {
        truncated = truncated || Boolean(joint.input_truncated ?? false);
        joint.evidence_ids = [...(joint.source_ids as string[])];
        joint.source_ids = (joint.evidence_ids as string[]).map((id) => byId.get(id)!.sourceId);
      }
      candidate.verification_coverage = truncated ? 'truncated; abstention required'
        : joint !== null ? 'complete joint premise and supplied pairs' : 'complete supplied pairs';
      candidate.epistemic_status = 'hypothesis';
      const score = candidate.predicted_score;
      if (typeof score !== 'number' || !Number.isFinite(score)) throw new ValueError('candidate rank scores must be finite');
      if (candidate.accepted_by_policy) accepted.push(candidate);
    }
    updated = updated.assess(assessments);
    let selected: string | null = null;
    if (accepted.length) {
      let best = accepted[0]!;
      for (const candidate of accepted) if ((candidate.predicted_score as number) > (best.predicted_score as number)) best = candidate;
      selected = best.id as string;
    }
    updated = updated.select(selected !== null ? [selected] : []);
    const receipt: JsonObject = {
      question, selected_id: selected, abstained: selected === null, candidates,
      evidence: evidence.map((record) => asdict(record) as JsonObject),
      retrieval: hits.map((hit) => asdict(hit) as JsonObject),
      retrieval_encoder: this.memory !== null ? this.memory.metadata : null,
      state_revision: updated.revision,
      policy: this.policy.receipt({ scope }) as JsonObject,
      verification_scope: scope,
      model_provenance: provenance,
      semantics: 'Generated and supplied candidates remain hypotheses; selection is authored screening of model scores, not truth.',
    };
    if (dialogue.length) {
      receipt.conversation_context = dialogue as unknown as JsonValue;
      receipt.retrieval_query = this.memory !== null ? retrievalQuery : null;
    }
    const lineage = new Map(this.evidenceLineage);
    const known = new Set([...lineage.values()].flat());
    for (const hit of hits) if (!known.has(hit.evidence.id)) lineage.set(hit.evidence.id, Object.freeze([hit.evidence.id]));
    this.state = updated;
    this.evidenceLineage = lineage;
    this.observedModel = provenance;
    return receipt;
  }

  remember(evidenceId: string, options: { episodeId?: string | null; question?: string; outcome?: string } = {}): void {
    if (this.memory === null) throw new ValueError('episodic memory is not configured');
    const lineage = this.evidenceLineage.get(evidenceId);
    const resolved = lineage !== undefined ? lineage[lineage.length - 1]! : evidenceId;
    if (this.inactiveEvidence.has(resolved)) throw new ValueError('archived evidence cannot be remembered');
    const source = this.state.evidence.find((record) => record.id === resolved);
    if (source === undefined) throw new ValueError('unknown evidence id');
    this.memory.remember(source, {
      episodeId: options.episodeId ?? this.episodeId, question: options.question ?? '', outcome: options.outcome ?? '',
    });
  }

  retrieve(question: string, options: { k?: number; excludeEpisodeId?: string | null } = {}): readonly RetrievalHit[] {
    if (this.memory === null) throw new ValueError('episodic memory is not configured');
    const k = options.k ?? 5;
    if (!isInteger(k) || k < 0) throw new ValueError('k must be nonnegative');
    const hits = this.memory.retrieve(question, { k: this.memory.memory.capacity, excludeEpisodeId: options.excludeEpisodeId ?? null });
    return Object.freeze(hits.filter((hit) => !this.inactiveEvidence.has(hit.evidence.id)).slice(0, k));
  }

  snapshot(): CognitiveSnapshot {
    const state = this.state;
    return {
      schema_version: 2, state: state.toDict(),
      evidence_lineage: Object.fromEntries([...this.evidenceLineage].map(([key, history]) => [key, [...history]])),
      active_evidence: Object.fromEntries(this.activeEvidenceIds),
      policy: this.policy.toRecord(),
      inactive_evidence: [...this.inactiveEvidence].sort(comparePython),
      retrieval_k: this.retrievalK,
      memory: this.memory !== null ? this.memory.snapshot() : null,
      model_provenance: this.observedModel, episode: this.episode,
    };
  }

  static fromSnapshot(snapshot: unknown, options: { investigator: Investigator; memory?: LearnedEpisodicMemory | null }): CognitiveSession {
    const { investigator } = options;
    let memory = options.memory ?? null;
    if (!isPlainObject(snapshot) || Object.keys(snapshot).length !== SNAPSHOT_FIELDS.length || !SNAPSHOT_FIELDS.every((key) => key in snapshot)
      || !isInteger(snapshot.schema_version) || snapshot.schema_version !== 2) {
      throw new ValueError('unsupported cognitive session schema');
    }
    const policy = snapshot.policy;
    if (!isPlainObject(policy) || Object.keys(policy).length !== 3 || !['min_support', 'max_contradiction', 'max_unknown'].every((key) => key in policy)) {
      throw new ValueError('invalid policy schema');
    }
    const state = CognitiveState.fromDict(snapshot.state);
    const ids = new Set(state.evidence.map((record) => record.id));
    const active = snapshot.active_evidence;
    if (!isPlainObject(active) || Object.entries(active).some(([key, value]) => !key || !ids.has(key) || typeof value !== 'string' || !ids.has(value))
      || new Set(Object.values(active)).size !== Object.keys(active).length) {
      throw new ValueError('invalid active evidence references');
    }
    const inactive = snapshot.inactive_evidence;
    const activeValues = new Set(Object.values(active) as string[]);
    if (!Array.isArray(inactive) || inactive.some((item) => typeof item !== 'string' || !ids.has(item))
      || new Set(inactive).size !== inactive.length || inactive.some((item) => activeValues.has(item as string))) {
      throw new ValueError('invalid archived evidence references');
    }
    const lineage = snapshot.evidence_lineage;
    if (!isPlainObject(lineage)) throw new ValueError('invalid evidence lineage');
    const seen = new Set<string>();
    const inactiveSet = new Set(inactive as string[]);
    for (const [root, history] of Object.entries(lineage)) {
      if (!ids.has(root) || !Array.isArray(history) || !history.length || history[0] !== root
        || history.some((version) => typeof version !== 'string' || !ids.has(version))
        || new Set(history).size !== history.length || history.some((version) => seen.has(version as string))
        || history.slice(0, -1).some((version) => !inactiveSet.has(version as string))) {
        throw new ValueError('invalid evidence lineage');
      }
      for (const version of history as string[]) seen.add(version);
    }
    if (seen.size !== ids.size || [...ids].some((id) => !seen.has(id))) throw new ValueError('evidence lineage must cover every historical record');
    for (const [root, latest] of Object.entries(active)) {
      const history = lineage[root] as string[] | undefined;
      if (history === undefined || history[history.length - 1] !== latest) throw new ValueError('active evidence conflicts with lineage');
    }
    const retrievalK = snapshot.retrieval_k;
    if (!isInteger(retrievalK) || retrievalK < 0) throw new ValueError('invalid retrieval limit');
    if (snapshot.memory !== null) {
      if (memory !== null) throw new ValueError('snapshot already supplies episodic memory');
      memory = LearnedEpisodicMemory.fromSnapshot(snapshot.memory, { investigator: investigator as unknown as MemoryOwner });
    }
    const result = new CognitiveSession(investigator, { state, policy: new SelectionPolicy(policy as PolicyThresholds), memory });
    result.activeEvidenceIds = new Map(Object.entries(active as Record<string, string>));
    result.evidenceLineage = new Map(Object.entries(lineage as Record<string, string[]>).map(([root, history]) => [root, Object.freeze([...history])]));
    result.inactiveEvidence = new Set(inactive as string[]);
    result.retrievalK = retrievalK;
    const provenance = snapshot.model_provenance;
    if (provenance !== null && (typeof provenance !== 'string' || !provenance)) throw new ValueError('invalid model provenance');
    if (state.assessments.length && provenance === null) throw new ValueError('assessed session requires model provenance');
    if (!isInteger(snapshot.episode) || snapshot.episode < 0) throw new ValueError('invalid episode counter');
    result.episode = snapshot.episode;
    result.observedModel = provenance as string | null;
    // Trigger weight compatibility invalidation before exposing restored state.
    void result.state;
    return result;
  }

  async save(path: string): Promise<void> {
    await atomicWriteFile(path, pythonJsonDumps(this.snapshot(), { allowNan: false, indent: 2, floatKeys: sessionFloatKeys() }));
  }

  static async load(path: string, options: { investigator: Investigator; memory?: LearnedEpisodicMemory | null }): Promise<CognitiveSession> {
    let data: unknown;
    try {
      data = parseJsonStrict(await readFile(path, 'utf8'));
    } catch (error) {
      if (error instanceof SyntaxError && /Duplicate/.test(error.message)) throw new ValueError('duplicate JSON key', { cause: error });
      throw error;
    }
    return CognitiveSession.fromSnapshot(data, options);
  }
}

/** Python string ordering (code points). */
function comparePython(a: string, b: string): number {
  const left = Array.from(a);
  const right = Array.from(b);
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const difference = left[index]!.codePointAt(0)! - right[index]!.codePointAt(0)!;
    if (difference) return difference;
  }
  return left.length - right.length;
}

export { comparePython };
