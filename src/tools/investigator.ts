/**
 * Trainable hypothesis generation, ranking and source-evidence verification
 * (Python ``tensorcode/tools/investigator.py``).
 */
import { Tensor, tensor, zeros } from '../nn/tensor.js';
import { noGrad } from '../nn/autograd.js';
import { crossEntropy } from '../nn/ops/nn.js';
import { ValueError } from '../errors.js';
import { ModuleOperation, type Context, type OperationLike } from '../ops/base.js';
import { PretrainedModule, pythonClassName, rejectUnknownToolFields } from '../_internal/pretrained.js';
import { deepCopy, isPlainObject, sha256Bytes, type JsonObject, type JsonValue } from '../_internal/json.js';
import {
  RANKING_FIELDS, RankOperation, RankingObjective, normalizeRankingConfig, rankingFromFoundation, replayableBindings,
} from '../_internal/ranking.js';
import { FastTokenizer } from '../_internal/tokenizers/index.js';
import { NativeConfig } from '../_internal/native/config.js';
import { createNativeModel } from '../_internal/native/registry.js';
import { loadNativeFoundation, type FoundationOptions } from '../_internal/native/foundation.js';
import type { NativeSequenceClassifier } from '../_internal/native/bert.js';
import { TemperatureCalibration } from '../training/calibration.js';
import { RetrievalEncoder, isDirectory, type RetrievalFoundationOptions } from '../_internal/retrieval.js';
import { generateProposals, proposalLoss, proposalPrompt, type ProposalRecord } from '../_internal/proposals.js';
import { RankingSession } from '../_internal/sessions/ranking.js';
import { CognitiveSession } from '../_internal/cognition/session.js';
import { CognitiveState } from '../_internal/cognition/state.js';
import { Sha256Accumulator, updateTensorDigest } from '../_internal/cognition/locking.js';
import { withEvalModes } from '../_internal/memory/learned.js';
import { Chatbot, type ChatbotFoundationOptions } from './chatbot.js';

/** Investigator configuration fields (Python ``Investigator.config_fields``); Decision inherits them. */
const INVESTIGATOR_FIELDS: readonly string[] = Object.freeze([
  ...RANKING_FIELDS,
  'generator', 'retrieval_encoder', 'verification_scope', 'max_proposals',
  'proposal_template_version', 'verifier_config', 'verifier_tokenizer_json',
  'verifier_tokenizer_special_tokens', 'verifier_labels', 'verifier_foundation',
  'verifier_max_tokens', 'verifier_calibration',
]);

export { Evidence } from './cognition.js';
/** The session type returned by {@link Investigator.newCognitiveSession}. */
export { CognitiveSession as InvestigationSession } from '../_internal/cognition/session.js';

export interface VerifierPair {
  premise: string;
  hypothesis: string;
}

export interface SourceText {
  source_id: string;
  text: string;
}

/** Temperature calibration that records the verifier weights it was fitted against. */
export class VerifierCalibration extends TemperatureCalibration {
  static override readonly qualifiedName: string = 'tensorcode.tools.investigator._VerifierCalibration';
  private readonly owner: EvidenceVerifier;

  constructor(owner: EvidenceVerifier, options: { minTemperature?: number; maxTemperature?: number; iterations?: number } = {}) {
    super(options);
    this.owner = owner;
  }

  override fit(logits: Tensor, labels: Tensor): ReturnType<TemperatureCalibration['fit']> {
    const result = super.fit(logits, labels);
    this.owner.recordCalibrationWeights();
    return result;
  }
}

interface VersionKey {
  name: string;
  tensor: Tensor;
  version: number;
  dtype: string;
  shape: string;
}

/** Owned classifier with explicit semantic label mapping and source identity. */
export class EvidenceVerifier extends ModuleOperation<VerifierPair[], Tensor> {
  static override readonly qualifiedName: string = 'tensorcode.tools.investigator.EvidenceVerifier';
  readonly config: JsonObject;
  readonly model: NativeSequenceClassifier;
  readonly calibration: VerifierCalibration;
  calibrationWeightDigest: Tensor;
  readonly labels: Record<string, number>;
  maxTokens: number;
  readonly tokenizer: FastTokenizer;
  private calibrationVersions: VersionKey[] | null = null;

  constructor(config: JsonObject) {
    super();
    this.config = Object.fromEntries(Object.entries(config).filter(([key]) => key.startsWith('verifier_')).map(([key, value]) => [key, deepCopy(value)]));
    const native = NativeConfig.fromDict(config.verifier_config);
    this.model = this.registerModule('model', createNativeModel(native, 'sequence-classification') as unknown as NativeSequenceClassifier);
    const options = (config.verifier_calibration ?? {}) as JsonObject;
    if (!isPlainObject(options) || Object.keys(options).some((key) => !['min_temperature', 'max_temperature', 'iterations'].includes(key))) {
      throw new TypeError('verifier_calibration accepts only min_temperature, max_temperature and iterations');
    }
    this.calibration = this.registerModule('calibration', new VerifierCalibration(this, {
      ...(options.min_temperature !== undefined ? { minTemperature: options.min_temperature as number } : {}),
      ...(options.max_temperature !== undefined ? { maxTemperature: options.max_temperature as number } : {}),
      ...(options.iterations !== undefined ? { iterations: options.iterations as number } : {}),
    }));
    this.calibrationWeightDigest = this.registerBuffer('calibration_weight_digest', zeros([32], { dtype: 'uint8' }));
    const labels = config.verifier_labels;
    const count = native.numLabels;
    const values = isPlainObject(labels) ? Object.values(labels) : [];
    if (!isPlainObject(labels) || Object.keys(labels).length !== 3 || !['support', 'contradiction', 'unknown'].every((key) => key in labels)
      || values.some((index) => typeof index !== 'number' || !Number.isInteger(index))
      || new Set(values).size !== count || values.some((index) => (index as number) < 0 || (index as number) >= count)) {
      throw new ValueError('verifier_labels must explicitly map support, contradiction, unknown to the three classifier indices');
    }
    this.labels = { ...(labels as Record<string, number>) };
    const maxTokens = config.verifier_max_tokens ?? 512;
    if (typeof maxTokens !== 'number' || !Number.isInteger(maxTokens) || maxTokens < 1) {
      throw new ValueError('verifier_max_tokens must be a positive integer');
    }
    this.maxTokens = maxTokens;
    if (typeof config.verifier_tokenizer_json !== 'string') throw new ValueError('verifier requires verifier_tokenizer_json');
    this.tokenizer = FastTokenizer.fromJsonString(config.verifier_tokenizer_json,
      (config.verifier_tokenizer_special_tokens ?? {}) as Record<string, string>);
    if (this.tokenizer.padTokenId === null) throw new ValueError('verifier tokenizer requires padding token');
  }

  override get replayable(): boolean {
    return true;
  }

  protected override onRegistryChange(): void {
    this.calibrationWeightDigest = this.getBuffer('calibration_weight_digest') ?? this.calibrationWeightDigest;
  }

  configuration(): JsonObject {
    return deepCopy(this.config);
  }

  forward(inputs: VerifierPair[], context: Context | null): Tensor {
    if (context && Object.keys(context).length) throw new ValueError('verifier does not consume context');
    if (!Array.isArray(inputs) || !inputs.length || inputs.some((item) => !isPlainObject(item)
      || ['premise', 'hypothesis'].some((key) => typeof item[key as keyof VerifierPair] !== 'string' || !(item[key as keyof VerifierPair] as string).trim()))) {
      throw new ValueError('verifier inputs must be nonempty premise/hypothesis pairs');
    }
    const tokens = this.tokenizer.encodeTensors(inputs.map((item) => item.premise), {
      textPair: inputs.map((item) => item.hypothesis), padding: true, truncation: true, maxLength: this.maxTokens,
    });
    return this.model.forward({
      inputIds: tokens.input_ids, attentionMask: tokens.attention_mask, tokenTypeIds: tokens.token_type_ids ?? null,
    }).logits;
  }

  /** Supervised NLI cross entropy; resets any calibration fitted to the old weights. */
  loss(inputs: VerifierPair[], targets: unknown): Tensor {
    if (!Array.isArray(targets) || targets.length !== (Array.isArray(inputs) ? inputs.length : -1)
      || targets.some((target) => typeof target !== 'string' || !(target in this.labels))) {
      throw new ValueError('targets must provide one named NLI label per pair');
    }
    this.resetCalibration();
    const logits = this.call(inputs);
    return crossEntropy(logits, tensor((targets as string[]).map((target) => this.labels[target]!), { dtype: 'int64' }));
  }

  private resetCalibration(): void {
    noGrad(() => {
      this.calibration.calibrated.fill_(0);
      this.calibration.sampleCount.zero_();
      this.calibration.temperature.fill_(1);
    });
  }

  private weightDigest(): Tensor {
    const digest = new Sha256Accumulator();
    for (const [name, value] of this.model.stateDict()) updateTensorDigest(digest, name, value);
    return tensor(Array.from(sha256Bytes(digest.bytes())), { dtype: 'uint8' });
  }

  private versions(): VersionKey[] {
    return [...this.model.namedParameters(), ...this.model.namedBuffers()].map(([name, value]) => ({
      name, tensor: value, version: value.version, dtype: value.dtype, shape: value.shape.join(','),
    }));
  }

  private sameVersions(a: VersionKey[] | null, b: VersionKey[]): boolean {
    return a !== null && a.length === b.length && a.every((entry, index) => {
      const other = b[index]!;
      return entry.name === other.name && entry.tensor === other.tensor && entry.version === other.version
        && entry.dtype === other.dtype && entry.shape === other.shape;
    });
  }

  /** @internal record the weights a calibration was fitted against. */
  recordCalibrationWeights(): void {
    noGrad(() => this.calibrationWeightDigest.copy_(this.weightDigest()));
    this.calibrationVersions = this.versions();
  }

  private validateCalibrationWeights(): void {
    if (this.calibration.isCalibrated) {
      const current = this.versions();
      if (!this.sameVersions(this.calibrationVersions, current)) {
        if (!this.calibrationWeightDigest.equal(this.weightDigest())) this.resetCalibration();
        this.calibrationVersions = this.versions();
      }
    }
  }

  /** Model distributions for each source against ``hypothesis`` (never facts). */
  verify(hypothesis: string, evidence: readonly SourceText[]): JsonObject[] {
    return this.verifySources(hypothesis, evidence);
  }

  /** Score all supplied text as one premise, without generated intermediates. */
  verifyJoint(hypothesis: string, evidence: readonly SourceText[]): JsonObject | null {
    if (!evidence.length) return null;
    const ids = evidence.map((row) => row.source_id);
    if (new Set(ids).size !== ids.length) throw new ValueError('joint verification requires unique source IDs');
    const premise = evidence.map((row) => row.text).join('\n\n');
    const result = this.verifySources(hypothesis, [{ source_id: 'joint', text: premise }])[0]!;
    delete result.source_id;
    result.source_ids = ids;
    result.scope = 'joint';
    result.max_tokens = this.maxTokens;
    result.calibration_application = 'same pair-classifier temperature applied to combined premise; joint-domain calibration not established';
    result.token_count = this.tokenizer.encode(premise, { textPair: hypothesis }).inputIds[0]!.length;
    return result;
  }

  private verifySources(hypothesis: string, evidence: readonly SourceText[]): JsonObject[] {
    this.validateCalibrationWeights();
    if (!evidence.length) return [];
    const probabilities = withEvalModes(this, () => noGrad(() => {
      const logits = this.call(evidence.map((row) => ({ premise: row.text, hypothesis })));
      return this.calibration.forward(logits).softmax(-1).tolist() as number[][];
    }));
    const calibrated = this.calibration.isCalibrated;
    const sampleCount = this.calibration.sampleCount.item();
    const model = (this.config.verifier_foundation ?? { initialization: 'configured_weights' }) as JsonValue;
    return evidence.map((source, index) => {
      const row = probabilities[index]!;
      const distribution: JsonObject = {};
      for (const [label, position] of Object.entries(this.labels)) distribution[label] = row[position]!;
      return {
        source_id: source.source_id, distribution, origin: 'model_inference', calibrated,
        calibration_sample_count: sampleCount, model: deepCopy(model),
        input_truncated: this.tokenizer.encode(source.text, { textPair: hypothesis }).inputIds[0]!.length > this.maxTokens,
      };
    });
  }
}

export interface InvestigatorFoundationOptions extends Omit<FoundationOptions, 'head'> {
  /** Additional tool configuration fields. */
  options?: JsonObject;
}

export interface InvestigatorFoundationsOptions extends Omit<FoundationOptions, 'head' | 'revision'> {
  encoderRevision?: string | null;
  generatorRevision?: string | null;
  verifierRevision?: string | null;
  verifierLabels: Record<string, number>;
  /** Configuration fields for the generator Chatbot. */
  generatorOptions?: JsonObject | null;
  retrievalRepo?: string | null;
  retrievalRevision?: string | null;
  retrievalOptions?: Omit<RetrievalFoundationOptions, keyof FoundationOptions> | null;
  /** Additional tool configuration fields. */
  options?: JsonObject;
}

export interface RetrievalBootstrapOptions extends Omit<RetrievalFoundationOptions, 'head'> {
  options?: JsonObject;
}

function hasContext(context: Context | null | undefined): boolean {
  return Boolean(context && Object.keys(context).length);
}

/**
 * Own an encoder, shared workspace, hypothesis ranking head and optional
 * proposal generator, evidence verifier and episodic retrieval encoder.
 *
 * Hypotheses are supplied or generated by the owned generator; generated
 * hypotheses are unverified proposals, not evidence. {@link investigate} ranks
 * and verifies them against supplied sources; {@link newCognitiveSession} adds
 * revisable evidence and episodic memory. Ranking probabilities are
 * uncalibrated model scores. Construction uses random weights; use
 * ``fromPretrained`` to load learned weights.
 */
export class Investigator extends PretrainedModule<Record<string, unknown>, JsonObject> {
  static override readonly qualifiedName: string = 'tensorcode.tools.investigator.Investigator';
  /** Accepted configuration fields; unknown ones raise ``ValueError`` (Python ``config_fields``). */
  static readonly configFields: readonly string[] = INVESTIGATOR_FIELDS;
  declare readonly rank: RankOperation;
  declare readonly objective: RankingObjective;
  declare readonly generator: Chatbot | null;
  declare readonly verifier: EvidenceVerifier | null;
  declare readonly episodicEncoder: RetrievalEncoder | null;

  constructor(config: unknown) {
    if (!isPlainObject(config)) throw new ValueError('model config must be a JSON object');
    const value = deepCopy(config as JsonObject);
    rejectUnknownToolFields(value, INVESTIGATOR_FIELDS, pythonClassName(new.target));
    const generator = 'generator' in value ? new Chatbot(value.generator) : null;
    if (generator !== null) value.generator = generator.configuration();
    const episodic = 'retrieval_encoder' in value ? new RetrievalEncoder(value.retrieval_encoder) : null;
    if (episodic !== null) value.retrieval_encoder = episodic.configuration();
    if (!('verification_scope' in value)) value.verification_scope = 'source';
    if (value.verification_scope !== 'source' && value.verification_scope !== 'joint') throw new ValueError('verification_scope must be source or joint');
    if (!('max_proposals' in value)) value.max_proposals = 16;
    if (!('proposal_template_version' in value)) value.proposal_template_version = 1;
    if (value.proposal_template_version !== 1 && value.proposal_template_version !== 2) {
      throw new ValueError('proposal_template_version must be 1 or 2');
    }
    const maxProposals = value.max_proposals;
    if (typeof maxProposals !== 'number' || !Number.isInteger(maxProposals) || maxProposals < 1) {
      throw new ValueError('max_proposals must be a positive integer');
    }
    super(normalizeRankingConfig(value));
    const writable = this as unknown as Record<string, unknown>;
    writable.rank = this.registerModule('rank', new RankOperation(this.config, { taskKey: 'question', candidatesKey: 'hypotheses' }));
    writable.objective = this.registerModule('objective', new RankingObjective(this));
    writable.generator = generator !== null ? this.registerModule('generator', generator) : null;
    writable.verifier = 'verifier_config' in this.config ? this.registerModule('verifier', new EvidenceVerifier(this.config)) : null;
    writable.episodicEncoder = episodic !== null ? this.registerModule('episodic_encoder', episodic) : null;
  }

  /** Ranking objective used by ``Trainer.fromTool``. */
  get trainingOperation(): RankingObjective {
    return this.objective;
  }

  get trainingInputsIncludeTargets(): boolean {
    return true;
  }

  /**
   * Rank supplied ``hypotheses`` (a receipt with probabilities), else {@link investigate}.
   * Inputs: ``question``, ``evidence`` ``[{source_id, text}]``, optional ``hypotheses`` ``[{id, text}]``.
   */
  forward(inputs: Record<string, unknown>, context: Context | null): JsonObject {
    if (hasContext(context)) throw new ValueError('This tool does not accept context');
    return isPlainObject(inputs) && 'hypotheses' in inputs ? this.rank.receipt(inputs, { probabilities: true }) : this.investigate(inputs);
  }

  /** Alias of {@link forward} without context. */
  predict(inputs: Record<string, unknown>): JsonObject {
    return this.call(inputs);
  }

  /** Generate up to ``count`` unverified hypotheses with the owned generator. */
  propose(inputs: Record<string, unknown>, options: { count?: number } = {}): ProposalRecord[] {
    return generateProposals(this.generator, inputs, {
      taskKey: 'question', count: options.count ?? 3, maxCount: this.config.max_proposals as number,
      templateVersion: this.config.proposal_template_version as number,
    });
  }

  /** Teacher-forced loss for the owned hypothesis generator. */
  proposalLoss(inputs: Record<string, unknown>, targets: unknown): Tensor {
    return proposalLoss(this.generator, inputs, targets, { taskKey: 'question', templateVersion: this.config.proposal_template_version as number });
  }

  /**
   * Rank supplied or generated hypotheses and verify each against evidence.
   *
   * Abstains when generation yields no hypotheses. Verification results are
   * model distributions over the configured scope, not established facts.
   */
  investigate(inputs: Record<string, unknown>, options: { count?: number } = {}): JsonObject {
    if (this.verifier === null) throw new ValueError('evidence verification capability is not configured');
    const value = deepCopy(inputs) as JsonObject;
    const generated = !('hypotheses' in value);
    if (generated) value.hypotheses = this.propose(value, { count: options.count ?? 3 });
    if (generated && !(value.hypotheses as JsonValue[]).length) {
      proposalPrompt(value, 'question');
      return {
        selected_id: null, candidates: [], evidence: deepCopy((value.evidence ?? []) as JsonValue),
        abstained: true, reason: 'no_hypotheses_generated',
      };
    }
    this.rank.validate(value);
    const result = this.rank.receipt(value, { probabilities: true });
    for (const candidate of result.candidates as JsonObject[]) {
      Object.assign(candidate, this.verify(candidate.text as string, (value.evidence ?? []) as unknown as SourceText[]));
    }
    result.verification_scope = this.config.verification_scope!;
    result.verification_semantics = 'NLI model distributions over configured evidence scope; source contradictions retained; calibration fit does not establish facts';
    return result;
  }

  /** Retain each source check and an explicitly configured joint evidence check. */
  verify(hypothesis: string, evidence: readonly SourceText[]): JsonObject {
    if (this.verifier === null) throw new ValueError('evidence verification capability is not configured');
    const result: JsonObject = { verifications: this.verifier.verify(hypothesis, evidence) };
    if (this.config.verification_scope === 'joint') result.joint_verification = this.verifier.verifyJoint(hypothesis, evidence);
    return result;
  }

  /** Supervised NLI loss for the owned evidence verifier. */
  verificationLoss(inputs: unknown, targets: unknown): Tensor {
    if (this.verifier === null) throw new ValueError('evidence verification capability is not configured');
    return this.verifier.loss(inputs as VerifierPair[], targets);
  }

  /** Contrastive loss for the episodic retrieval encoder. */
  retrievalLoss(inputs: unknown, targets: unknown): Tensor {
    if (this.episodicEncoder === null) throw new ValueError('retrieval training capability is not configured');
    if (!isPlainObject(inputs) || Object.keys(inputs).length !== 2 || !('queries' in inputs) || !('documents' in inputs)) {
      throw new ValueError('retrieval inputs require only queries and documents; positives belong in targets');
    }
    return this.episodicEncoder.contrastiveLoss(inputs.queries as string[], inputs.documents as string[], targets);
  }

  /** The complete JSON configuration, including nested components. */
  override configuration(): JsonObject {
    const config = super.configuration();
    if (this.generator !== null) config.generator = this.generator.configuration();
    if (this.episodicEncoder !== null) config.retrieval_encoder = this.episodicEncoder.configuration();
    return config;
  }

  /** Ranking loss for a target hypothesis index, ID or distribution. */
  loss(inputs: unknown, targets: unknown): Tensor {
    const logits = this.rank.call(inputs as Record<string, unknown>);
    if (Array.isArray(targets) || (targets instanceof Tensor && targets.ndim === 1)) {
      const values = targets instanceof Tensor ? targets.toArray() : (targets as unknown[]);
      const numeric = values.every((item) => typeof item === 'number');
      const sum = numeric ? (values as number[]).reduce((a, b) => a + b, 0) : Number.NaN;
      if (!numeric || values.length !== logits.numel || (values as number[]).some((item) => !Number.isFinite(item) || item < 0)
        || !(Math.abs(sum - 1) <= 1e-5 + 1e-5)) {
        throw new ValueError('target distribution must be finite, nonnegative and sum to one');
      }
      const target = tensor(values as number[], { dtype: logits.dtype });
      return target.mul(logits.logSoftmax(-1)).sum().neg();
    }
    let index = targets;
    if (typeof index === 'string') {
      const ids = ((inputs as Record<string, unknown>).hypotheses as JsonObject[]).map((item) => item.id);
      if (!ids.includes(index)) throw new ValueError('target must identify a supplied hypothesis');
      index = ids.indexOf(index);
    }
    if (typeof index !== 'number' || !Number.isInteger(index) || index < 0 || index >= logits.numel) {
      throw new ValueError('target must be a valid hypothesis index or ID');
    }
    return crossEntropy(logits.unsqueeze(0), tensor([index], { dtype: 'int64' }));
  }

  /** Named operations for tracing, experience and checkpoints. */
  override operationBindings(): Record<string, OperationLike> {
    const result = replayableBindings(this);
    if (this.generator !== null) {
      for (const [key, value] of Object.entries(this.generator.operationBindings())) result[`generator.${key}`] = value;
    }
    if (this.episodicEncoder !== null) {
      for (const [key, value] of Object.entries(this.episodicEncoder.operationBindings())) result[`episodic_encoder.${key}`] = value;
    }
    return result;
  }

  /** Create a ranking-history session sharing this tool's weights. */
  newSession(): RankingSession {
    return new RankingSession(this);
  }

  /**
   * Create independent source evidence and optional episodic memory.
   *
   * ``policy`` is a JSON object of authored score thresholds. ``memory`` is
   * ``null`` to disable retrieval, or a JSON object with ``capacity`` and
   * ``top_k``. Model weights are shared; session records are independent.
   */
  newCognitiveSession(options: { policy?: JsonObject | null; memory?: JsonObject | null; maxRecords?: number } = {}): CognitiveSession {
    const policy = options.policy ?? null;
    const memory = options.memory ?? null;
    if (policy !== null && !isPlainObject(policy)) throw new TypeError('policy must be a JSON configuration object or None');
    if (memory !== null && !isPlainObject(memory)) throw new TypeError('memory must be a JSON configuration object or None');
    return new CognitiveSession(this, { state: new CognitiveState({ maxRecords: options.maxRecords ?? 256 }), policy, memory });
  }

  /** Restore independent session data using this tool's owned models. */
  async loadCognitiveSession(path: string): Promise<CognitiveSession> {
    return CognitiveSession.load(path, { investigator: this });
  }

  /** Explicitly load a pretrained encoder; workspace and ranking head start random. */
  static async fromFoundation<T extends Investigator>(
    this: new (config: JsonObject) => T, repo: string, options: InvestigatorFoundationOptions = {},
  ): Promise<T> {
    const { options: extra, ...load } = options;
    const construct = (config: JsonObject) => new this(config);
    const result = await rankingFromFoundation(construct, repo, { ...load, options: extra ?? {} });
    if (!(await isDirectory(repo)) && !(result.config.foundation as JsonObject).revision) {
      throw new ValueError('foundation provenance requires a resolved revision');
    }
    return result;
  }

  /**
   * Explicitly bootstrap encoder, generator, verifier and optional retrieval.
   * Inherited foundation competence is not learned workspace behavior; the
   * ranking head and workspace start untrained.
   */
  static async fromFoundations<T extends Investigator>(
    this: new (config: JsonObject) => T, encoderRepo: string, generatorRepo: string, verifierRepo: string,
    options: InvestigatorFoundationsOptions,
  ): Promise<T> {
    const {
      encoderRevision = null, generatorRevision = null, verifierRevision = null, verifierLabels, generatorOptions = null,
      retrievalRepo = null, retrievalRevision = null, retrievalOptions = null, options: extra = {}, ...load
    } = options;
    const generator = await Chatbot.fromFoundation(generatorRepo, {
      ...load, revision: generatorRevision, options: (generatorOptions ?? {}) as ChatbotFoundationOptions['options'],
    });
    const verifier = await loadNativeFoundation(verifierRepo, { ...load, revision: verifierRevision, head: 'sequence-classification', initializeMissing: true });
    const resolved = verifier.commitHash ?? verifierRevision;
    if (!(await isDirectory(verifierRepo)) && !resolved) throw new ValueError('verifier provenance requires a resolved Hub revision');
    if (!verifier.tokenizer) throw new ValueError('verifier requires a serializable fast tokenizer');
    const config: JsonObject = { ...deepCopy(extra) };
    let retrieval: RetrievalEncoder | null = null;
    if (retrievalRepo !== null) {
      if (retrievalOptions === null) throw new ValueError('retrieval_repo requires retrieval options (pooling, normalize)');
      retrieval = await RetrievalEncoder.fromFoundation(retrievalRepo, { ...load, ...retrievalOptions, revision: retrievalRevision });
      if ('retrieval_encoder' in config) throw new ValueError('supply retrieval_repo or retrieval_encoder configuration, not both');
      config.retrieval_encoder = retrieval.configuration();
    } else if (retrievalRevision !== null || retrievalOptions !== null) {
      throw new ValueError('retrieval options require retrieval_repo');
    }
    const special: JsonObject = {};
    for (const [key, value] of Object.entries(verifier.tokenizer.specialTokensMap)) if (typeof value === 'string') special[key] = value;
    Object.assign(config, {
      generator: generator.configuration(), verifier_config: verifier.config.toDict(),
      verifier_tokenizer_json: verifier.tokenizer.rustJsonText, verifier_tokenizer_special_tokens: special,
      verifier_labels: verifierLabels as unknown as JsonObject, verifier_foundation: { repository: verifierRepo, revision: resolved },
    });
    const result = await (Investigator.fromFoundation as (this: new (config: JsonObject) => T, repo: string, options: InvestigatorFoundationOptions) => Promise<T>)
      .call(this, encoderRepo, { ...load, revision: encoderRevision, options: config });
    result.generator!.loadStateDict(generator.stateDict());
    result.verifier!.model.loadStateDict(verifier.model.stateDict());
    if (retrieval !== null) result.episodicEncoder!.loadStateDict(retrieval.stateDict());
    return result;
  }

  /**
   * Bootstrap only the retrieval encoder; other configured components
   * initialize randomly (supply rank/generator/verifier configuration through
   * ``options``). This does not imply those components are pretrained.
   */
  static async fromRetrievalFoundation<T extends Investigator>(
    this: new (config: JsonObject) => T, repo: string, options: RetrievalBootstrapOptions,
  ): Promise<T> {
    const { options: extra = {}, ...retrievalOptions } = options;
    if ('retrieval_encoder' in extra) throw new ValueError('retrieval_encoder configuration conflicts with foundation bootstrap');
    const retrieval = await RetrievalEncoder.fromFoundation(repo, retrievalOptions);
    const result = new this({ ...deepCopy(extra), retrieval_encoder: retrieval.configuration() });
    result.episodicEncoder!.loadStateDict(retrieval.stateDict());
    result.eval();
    return result;
  }
}
