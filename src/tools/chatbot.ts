/**
 * A local trainable encoder/workspace/decoder model with independent sessions
 * (Python ``tensorcode/tools/chatbot.py``).
 */
import { Parameter, Tensor, onesLike, scalar, zerosLike } from '../nn/tensor.js';
import { Linear } from '../nn/layers.js';
import { noGrad } from '../nn/autograd.js';
import { FLOAT32_EPSILON } from '../nn/dtype.js';
import { div, maximum, where } from '../nn/ops/elementwise.js';
import { ValueError } from '../errors.js';
import { Operation, type Context, type OperationLike } from '../ops/base.js';
import { PretrainedModule, pythonClassName, rejectUnknownToolFields, type FromPretrainedOptions, type PretrainedClass } from '../_internal/pretrained.js';
import { Workspace, type WorkspaceOutput } from '../_internal/workspace.js';
import { deepCopy, isPlainObject, pythonJsonDumps, sha256Hex, type JsonObject, type JsonValue } from '../_internal/json.js';
import { conversationBlock, type ConversationRow } from '../_internal/conversation.js';
import { casefold, wordTokens } from '../_internal/text/casefold.js';
import { FastTokenizer } from '../_internal/tokenizers/index.js';
import { NativeConfig, generationConfigFromFile, generationConfigFromModel } from '../_internal/native/config.js';
import { createNativeModel } from '../_internal/native/registry.js';
import { loadNativeFoundation, type FoundationOptions } from '../_internal/native/foundation.js';
import type { T5ForConditionalGeneration } from '../_internal/native/t5.js';
import { SequenceEncoder } from '../_internal/vec/sequence.js';
import { SequenceDecoder, type DecoderLoss } from '../_internal/text/realization.js';
import { ChatSession, type ChatTurn } from '../_internal/sessions/chat.js';
import { CognitiveSession } from '../_internal/cognition/session.js';
import { CognitiveState } from '../_internal/cognition/state.js';
import { isDirectory } from '../_internal/retrieval.js';
import { withEvalModes } from '../_internal/memory/learned.js';
import { Evidence } from './cognition.js';
import { Investigator, type InvestigatorFoundationsOptions } from './investigator.js';

const ABSTENTION = 'I do not have enough supported evidence to answer.';

/** Configuration fields a Chatbot accepts (Python ``Chatbot.config_fields``). */
const CHATBOT_FIELDS: readonly string[] = Object.freeze([
  'foundation_config', 'untied_lm_head', 'generation_config', 'tokenizer_json',
  'tokenizer_special_tokens', 'foundation', 'max_new_tokens', 'max_input_tokens',
  'max_target_tokens', 'max_turns', 'workspace', 'memory_mode', 'memory_update',
  'cognition',
]);

/** Fields of the nested ``cognition`` configuration (Python ``Chatbot.cognition_fields``). */
const CHATBOT_COGNITION_FIELDS: readonly string[] = Object.freeze([
  'investigator', 'conversation_context_tokens', 'proposal_count', 'abstention_text',
  'max_records', 'policy', 'memory',
]);

/**
 * Scale each example's valid-token update to its native encoder RMS
 * (Python ``_bounded_memory_update``).
 *
 * Norms use float32 and max rescaling, avoiding overflow from squaring large
 * finite projected values. Padding contributes neither energy nor count. The
 * bound holds up to destination-dtype rounding and limits magnitude, not
 * semantic quality. An absolute float32-epsilon update-RMS floor keeps zero and
 * subnormal projected updates from producing unbounded normalization
 * derivatives.
 */
export function boundedMemoryUpdate(native: Tensor, update: Tensor, mask: Tensor, gate: Tensor): Tensor {
  const same = native.ndim === 3 && update.ndim === 3 && native.shape.every((size, index) => size === update.shape[index]);
  if (!same || mask.ndim !== 2 || mask.shape[0] !== native.shape[0] || mask.shape[1] !== native.shape[1]) {
    throw new ValueError('Memory update requires matching [batch, tokens, dimensions] tensors and mask');
  }
  const valid = mask.bool().unsqueeze(-1);
  const original = where(valid, native.float(), 0);
  const projected = where(valid, update.float(), 0);
  if (!original.allFinite() || !projected.allFinite() || !gate.allFinite()) {
    throw new ValueError('Memory update requires finite valid-token values and gate');
  }
  const count = valid.to('int64').sum([1, 2], true).mul(native.shape[2]!).clampMin(1).float();
  const epsilon = FLOAT32_EPSILON;
  const nativeScale = original.detach().abs().amax([1, 2], true);
  const divisor = where(nativeScale.gt(0), nativeScale, onesLike(nativeScale));
  const nativeScaled = original.div(divisor);
  const nativeRmsScaled = nativeScaled.square().sum([1, 2], true).div(count).clampMin(epsilon ** 2).sqrt();
  // An absolute projected-RMS floor also bounds derivatives near zero; a
  // relative-only floor leaves 1 / subnormal-scale gradients infinite.
  const projectedScale = projected.detach().abs().amax([1, 2], true).clampMin(epsilon);
  const projectedScaled = projected.div(projectedScale);
  const absoluteFloorScaled = div(epsilon, projectedScale);
  const projectedRmsScaled = maximum(projectedScaled.square().sum([1, 2], true).div(count), absoluteFloorScaled.square()).sqrt();
  const nativeRms = nativeScale.mul(nativeRmsScaled);
  let residual = projectedScaled.div(projectedRmsScaled).mul(nativeRms).mul(gate.float().tanh());
  residual = where(valid, residual, 0).to(native.dtype);
  if (!residual.allFinite()) throw new ValueError('Bounded memory update exceeds the destination dtype range');
  return residual;
}

export interface ObjectiveInput {
  inputs: string[];
  targets: string[];
}

/** The owned teacher-forced language objective (Python ``_Objective``). */
export class ChatbotObjective extends Operation<ObjectiveInput, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode.tools.chatbot._Objective';
  private readonly owner: Chatbot;

  constructor(owner: Chatbot) {
    super();
    this.owner = owner;
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: ObjectiveInput, context: Context | null): Tensor {
    if (context && Object.keys(context).length) throw new ValueError('Objective does not consume context');
    return this.owner.lossBatch(value.inputs, value.targets).clone();
  }

  parameters(): Parameter[] {
    return this.owner.parameters();
  }

  configuration(): JsonObject {
    return { operation: 'tensorcode.tools.chatbot.Objective', model: this.owner.configuration() };
  }
}

export type WorkspaceAblation = 'bypass' | 'zero' | null;

export interface ChatbotFoundationOptions extends Omit<FoundationOptions, 'head' | 'tokenizer'> {
  /** Extra configuration fields merged into the constructed tool configuration. */
  options?: JsonObject;
}

export interface CognitiveFoundationsOptions {
  languageRevision?: string | null;
  encoderRevision?: string | null;
  generatorRevision?: string | null;
  verifierRevision?: string | null;
  verifierLabels: Record<string, number>;
  localFilesOnly?: boolean;
  cacheDir?: string;
  token?: string | null;
  investigatorOptions?: Partial<InvestigatorFoundationsOptions>;
  cognitivePolicy?: JsonObject | null;
  /** Extra language model configuration fields. */
  options?: JsonObject;
}

type ChatStatics = typeof Chatbot;

/**
 * Complete owned model. Construction initializes; loading supplies weights.
 *
 * No provider, callback, implicit download, or semantic seed is required.
 * {@link Chatbot.fromFoundation} is an explicit training bootstrap: its new
 * workspace has random weights and must be trained before claiming useful
 * behavior. With a ``cognition`` configuration the model owns a complete
 * {@link Investigator} (proposal generator and evidence verifier) and screens
 * its realizations with an authored abstention policy.
 */
export class Chatbot extends PretrainedModule<unknown, string> {
  static override readonly qualifiedName: string = 'tensorcode.tools.chatbot.Chatbot';
  /** Accepted configuration fields; unknown ones raise ``ValueError`` (Python ``config_fields``). */
  static readonly configFields: readonly string[] = CHATBOT_FIELDS;
  /** Accepted ``cognition`` fields (Python ``cognition_fields``). */
  static readonly cognitionFields: readonly string[] = CHATBOT_COGNITION_FIELDS;
  declare readonly foundation: T5ForConditionalGeneration;
  declare readonly tokenizer: FastTokenizer;
  declare readonly workspace: Workspace;
  declare readonly encoder: SequenceEncoder;
  declare readonly decoder: SequenceDecoder;
  declare readonly memoryProjection: Linear;
  declare readonly memoryGate: Parameter;
  declare readonly investigator: Investigator | null;
  declare readonly trainingOperation: ChatbotObjective;
  declare readonly objective: ChatbotObjective;
  /** Serialized generation configuration (``generation_config.to_json_string()`` form). */
  declare generationConfig: JsonObject;
  declare private session: ChatSession;

  constructor(config: unknown) {
    if (!isPlainObject(config)) throw new ValueError('model config must be a JSON object');
    const value: JsonObject = { ...(config as JsonObject) };
    const owner = pythonClassName(new.target);
    rejectUnknownToolFields(value, CHATBOT_FIELDS, owner);
    let cognitive = (value.cognition ?? null) as JsonObject | null;
    if (cognitive !== null) {
      if (!isPlainObject(cognitive) || !isPlainObject(cognitive.investigator)) {
        throw new ValueError('cognition requires a complete investigator configuration');
      }
      rejectUnknownToolFields(cognitive, CHATBOT_COGNITION_FIELDS, `${owner} cognition`);
      const nested = cognitive.investigator as JsonObject;
      if (!isPlainObject(nested.generator) || !('verifier_config' in nested)) {
        throw new ValueError('cognition requires owned proposal generator and verifier');
      }
      if ((nested.generator as JsonObject).cognition !== undefined && (nested.generator as JsonObject).cognition !== null) {
        throw new ValueError('Recursive cognitive generator configurations are not supported');
      }
      cognitive = { ...cognitive };
      if (!('conversation_context_tokens' in cognitive)) cognitive.conversation_context_tokens = 128;
      const tokens = cognitive.conversation_context_tokens;
      if (typeof tokens !== 'number' || !Number.isInteger(tokens) || tokens < 1) {
        throw new ValueError('conversation_context_tokens must be a positive integer');
      }
      const count = cognitive.proposal_count ?? 3;
      if (typeof count !== 'number' || !Number.isInteger(count) || count < 1) throw new ValueError('proposal_count must be positive');
      const abstention = cognitive.abstention_text ?? ABSTENTION;
      if (typeof abstention !== 'string' || !abstention.trim()) throw new ValueError('abstention_text must be nonempty');
    }
    value.foundation_config = deepCopy(value.foundation_config ?? null);
    if (cognitive !== null) value.cognition = deepCopy(cognitive);
    for (const [key, fallback] of [['max_new_tokens', 64], ['max_input_tokens', 512], ['max_target_tokens', 128], ['max_turns', 16]] as const) {
      if (!(key in value)) value[key] = fallback;
      const item = value[key];
      if (typeof item !== 'number' || !Number.isInteger(item) || item <= 0) throw new ValueError(`${key} must be a positive integer`);
    }
    if (!('workspace' in value)) value.workspace = { slots: 8, steps: 2 };
    if (!('tokenizer_special_tokens' in value)) value.tokenizer_special_tokens = {};
    if (!('memory_mode' in value)) value.memory_mode = 'contextualized_evidence';
    if (!('memory_update' in value)) value.memory_update = 'relative_rms_bounded';
    if (value.memory_update !== 'relative_rms_bounded') throw new ValueError('Unsupported memory_update; expected relative_rms_bounded');
    if (value.memory_mode !== 'contextualized_evidence' && value.memory_mode !== 'slots') throw new ValueError('Unknown memory_mode');
    super(value);
    if (!isPlainObject(value.foundation_config)) throw new ValueError('foundation_config must be a native configuration object');
    // Python: ``AutoConfig.for_model(model_type, **foundation_config)``.
    const nativeConfig = NativeConfig.forModel(value.foundation_config);
    const foundation = this.registerModule('foundation', createNativeModel(nativeConfig, 'seq2seq') as T5ForConditionalGeneration);
    if (value.untied_lm_head === true) {
      foundation.setParameterAt('lm_head.weight', new Parameter(foundation.lm_head.weight.detach().clone()));
    }
    const generation = 'generation_config' in value
      ? generationConfigFromFile(value.generation_config)
      : generationConfigFromModel(nativeConfig);
    this.generationConfig = generation;
    this.config.generation_config = deepCopy(generation);
    if (typeof value.tokenizer_json !== 'string') throw new ValueError('tokenizer_json must be a serialized fast tokenizer');
    const tokenizer = FastTokenizer.fromJsonString(value.tokenizer_json, value.tokenizer_special_tokens as Record<string, string>);
    if (tokenizer.padTokenId === null) throw new ValueError('Tokenizer requires an explicit padding token');
    const workspaceConfig = value.workspace;
    if (!isPlainObject(workspaceConfig) || Object.keys(workspaceConfig).some((key) => key !== 'slots' && key !== 'steps')) {
      throw new TypeError('workspace configuration accepts only slots and steps');
    }
    const dModel = nativeConfig.number('d_model');
    const writable = this as unknown as Record<string, unknown>;
    writable.tokenizer = tokenizer;
    writable.workspace = this.registerModule('workspace', new Workspace(dModel, (workspaceConfig.slots as number | undefined) ?? 8, (workspaceConfig.steps as number | undefined) ?? 2));
    writable.encoder = new SequenceEncoder(foundation, tokenizer, { maxTokens: value.max_input_tokens as number });
    writable.decoder = new SequenceDecoder(foundation, () => this.generationConfig);
    writable.memoryProjection = this.registerModule('memory_projection', new Linear(dModel, dModel));
    writable.memoryGate = this.registerParameter('memory_gate', new Parameter(scalar(0.01)));
    writable.foundation = foundation;
    // Investigator's proposal generator is a rankless Chatbot; recursion is rejected above.
    writable.investigator = cognitive !== null ? this.registerModule('investigator', new Investigator(cognitive.investigator as JsonObject)) : null;
    writable.session = new ChatSession(this);
    const objective = new ChatbotObjective(this);
    writable.trainingOperation = objective;
    writable.objective = objective;
  }

  /** Built-in tools receive ``{inputs, targets}`` envelopes from trainers. */
  get trainingInputsIncludeTargets(): boolean {
    return true;
  }

  /** The complete JSON configuration, including generation settings. */
  override configuration(): JsonObject {
    const config = super.configuration();
    config.generation_config = deepCopy(this.generationConfig);
    if (this.investigator !== null) (config.cognition as JsonObject).investigator = this.investigator.configuration();
    return config;
  }

  /** Which cognitive features this configuration provides. */
  get capabilities(): JsonObject {
    const cognitive = this.investigator !== null;
    return {
      language_realization: true, persistent_cognitive_state: cognitive, hypothesis_generation: cognitive,
      source_verification: cognitive, abstention_policy: cognitive ? 'authored' : null,
    };
  }

  /** @internal a fresh cognitive session for a chat session (``null`` without cognition). */
  newCognitiveSessionForChat(): CognitiveSession | null {
    if (this.investigator === null) return null;
    const cognition = this.config.cognition as JsonObject;
    const state = new CognitiveState({ maxRecords: (cognition.max_records as number | undefined) ?? 256 });
    return new CognitiveSession(this.investigator, {
      state, policy: (cognition.policy ?? null) as JsonObject | null, memory: (cognition.memory ?? null) as JsonObject | null,
    });
  }

  /** SHA-256 of the configuration; identifies the architecture, not the weights. */
  get fingerprint(): string {
    return sha256Hex(pythonJsonDumps(this.configuration(), { sortKeys: true }));
  }

  /** Receipt of the default session's most recent response (sources, evidence). */
  get lastResult(): JsonObject | null {
    return this.session.lastResult;
  }

  /** Default session's cognitive records, or ``null`` without cognition. */
  get cognitiveState(): CognitiveState | null {
    return this.session.cognition !== null ? this.session.cognition.state : null;
  }

  /** @internal the default conversation session. */
  get defaultSession(): ChatSession {
    return this.session;
  }

  /** Copy of the default session's retained conversation turns. */
  get history(): readonly ChatTurn[] {
    return Object.freeze(this.session.history.map((item) => ({ ...item })));
  }

  /** Create an independent conversation sharing this model's weights. */
  newSession(): ChatSession {
    return new ChatSession(this);
  }

  /** Clear the default session's conversation and cognitive records. */
  resetSession(): void {
    this.session.reset();
  }

  /** Start a new episode in the default session; memory is retained. */
  newEpisode(): CognitiveState {
    return this.session.newEpisode();
  }

  /** Re-encode episodic memory after weights change. */
  rebuildMemory(): void {
    this.session.rebuildMemory();
  }

  /** Save the default session's data; model weights are saved separately. */
  async saveSession(path: string): Promise<void> {
    await this.session.save(path);
  }

  /** Restore default session data using this model's weights. */
  async loadSession(path: string): Promise<this> {
    await this.session.load(path);
    return this;
  }

  /** Named operations for tracing, experience and checkpoints. */
  override operationBindings(): Record<string, OperationLike> {
    const result = super.operationBindings();
    result.encoder = this.encoder;
    result.decoder = this.decoder;
    result.objective = this.objective;
    if (this.investigator !== null) {
      for (const [key, operation] of Object.entries(this.investigator.operationBindings())) result[`investigator.${key}`] = operation;
    }
    return result;
  }

  /** Load a complete artifact and bind a fresh session to its weights. */
  static override async fromPretrained<T extends PretrainedModule>(
    this: PretrainedClass<T>, source: string, options: FromPretrainedOptions = {},
  ): Promise<T> {
    const model = await PretrainedModule.fromPretrained.call(this as never, source, options) as unknown as Chatbot;
    // The constructor's empty session saw initialization weights; bind the
    // first usable session only after checkpoint weights are final.
    model.resetSession();
    return model as unknown as T;
  }

  /** Initialize from external seq2seq weights plus a new, untrained workspace. */
  static async fromFoundation<T extends Chatbot>(
    this: new (config: JsonObject) => T, repo: string, options: ChatbotFoundationOptions = {},
  ): Promise<T> {
    const { options: extra, ...load } = options;
    const loaded = await loadNativeFoundation(repo, { ...load, head: 'seq2seq', initializeMissing: true });
    if (!loaded.tokenizer) throw new ValueError('Foundation requires a serializable fast tokenizer');
    const resolved = loaded.commitHash ?? load.revision ?? null;
    if (!(await isDirectory(repo)) && !resolved) throw new ValueError('Foundation provenance requires a resolved Hub revision');
    const model = loaded.model as T5ForConditionalGeneration;
    // Preserve both forward semantics and actual parameter aliases.
    const untied = model.shared.weight !== model.lm_head.weight;
    const special: JsonObject = {};
    for (const [key, item] of Object.entries(loaded.tokenizer.specialTokensMap)) if (typeof item === 'string') special[key] = item;
    const config: JsonObject = {
      foundation_config: loaded.config.toDict(), untied_lm_head: untied,
      generation_config: deepCopy(loaded.generationConfig ?? generationConfigFromModel(loaded.config)),
      tokenizer_json: loaded.tokenizer.rustJsonText, tokenizer_special_tokens: special,
      foundation: { repository: repo, revision: resolved, workspace_initialization: 'random' },
      ...(extra ?? {}),
    };
    const result = new this(config);
    result.foundation.loadStateDict(model.stateDict());
    return result;
  }

  /** Explicit bootstrap of all owned components; never a hidden download. */
  static async fromCognitiveFoundations(
    this: ChatStatics, languageRepo: string, encoderRepo: string, generatorRepo: string, verifierRepo: string,
    options: CognitiveFoundationsOptions,
  ): Promise<Chatbot> {
    const extra = options.options ?? {};
    if ('cognition' in extra) throw new ValueError('Bootstrap cognition is supplied through its component arguments');
    const common = { localFilesOnly: options.localFilesOnly ?? false, cacheDir: options.cacheDir, token: options.token ?? null };
    const language = await this.fromFoundation(languageRepo, { ...common, revision: options.languageRevision ?? null, options: extra });
    const investigator = await Investigator.fromFoundations(encoderRepo, generatorRepo, verifierRepo, {
      ...common, ...(options.investigatorOptions ?? {}),
      encoderRevision: options.encoderRevision ?? null, generatorRevision: options.generatorRevision ?? null,
      verifierRevision: options.verifierRevision ?? null, verifierLabels: options.verifierLabels,
    });
    const config = language.configuration();
    config.cognition = { investigator: investigator.configuration() };
    if (options.cognitivePolicy !== undefined && options.cognitivePolicy !== null) (config.cognition as JsonObject).policy = options.cognitivePolicy;
    const result = new this(config);
    const { missingKeys, unexpectedKeys } = result.loadStateDict(language.stateDict(), { strict: false });
    if (unexpectedKeys.length || missingKeys.some((key) => !key.startsWith('investigator.'))) {
      throw new Error('Language bootstrap state does not match the complete architecture');
    }
    result.investigator!.loadStateDict(investigator.stateDict());
    result.resetSession();
    return result;
  }

  /** Encode inputs through the workspace; ``'bypass'`` ablates it for comparison. */
  encodeWorkspace(inputs: readonly string[], options: { workspaceAblation?: WorkspaceAblation | string | null } = {}): WorkspaceOutput {
    const ablation = options.workspaceAblation ?? null;
    const encoded = this.encoder.call([...inputs]);
    let state: WorkspaceOutput = this.workspace.forward(encoded.encoded, encoded.mask.bool());
    if (this.config.memory_mode === 'contextualized_evidence' && ablation !== 'bypass') {
      // Evidence remains source-aligned, but slot relations can revise each
      // token representation before it becomes the decoder memory.
      const tokens = encoded.encoded;
      const slots = state.conditioning;
      const assignment = tokens.matmul(slots.transpose(-1, -2)).div(tokens.shape[tokens.ndim - 1]! ** 0.5).softmax(-1);
      const update = this.memoryProjection.forward(assignment.matmul(slots));
      const residual = boundedMemoryUpdate(tokens, update, encoded.mask, this.memoryGate);
      state = { ...state, conditioning: tokens.add(residual), mask: encoded.mask };
    }
    if (ablation === 'bypass') state = { ...state, conditioning: encoded.encoded, mask: encoded.mask };
    else if (ablation === 'zero') state = { ...state, conditioning: zerosLike(state.conditioning) };
    else if (ablation !== null) throw new ValueError('Unknown workspace ablation');
    return state;
  }

  /** Teacher-forced cross entropy; targets never enter the input encoder. */
  lossBatch(inputs: readonly string[], targets: readonly string[], options: { workspaceAblation?: WorkspaceAblation | null } = {}): Tensor {
    if (!Array.isArray(inputs) || !inputs.length || !Array.isArray(targets) || inputs.length !== targets.length
      || [...inputs, ...targets].some((item) => typeof item !== 'string')) {
      throw new ValueError('Expected equally sized nonempty text input and target batches');
    }
    const state = this.encodeWorkspace([...inputs], options);
    const ids = this.tokenizer.encodeTensors([...targets], { padding: true, truncation: true, maxLength: this.config.max_target_tokens as number }).input_ids;
    const labels = ids.maskedFill(ids.eq(this.tokenizer.padTokenId!), -100);
    if (!labels.ne(-100).any().item()) throw new ValueError('Targets contain no supervised tokens');
    return (this.decoder.call({ ...state, labels }) as DecoderLoss).loss;
  }

  /** Greedy-decode one reply per input text (no session state); ablations: ``'bypass'``, ``'zero'``. */
  generateBatch(inputs: readonly string[], options: { workspaceAblation?: WorkspaceAblation | null } = {}): string[] {
    if (!Array.isArray(inputs) || !inputs.length || inputs.some((item) => typeof item !== 'string')) {
      throw new ValueError('Expected nonempty text batch');
    }
    return withEvalModes(this, () => noGrad(() => {
      const state = this.encodeWorkspace([...inputs], options);
      const tokens = this.decoder.call(state, { context: { max_new_tokens: this.config.max_new_tokens, do_sample: false } }) as Tensor;
      return this.tokenizer.batchDecode(tokens, { skipSpecialTokens: true });
    }));
  }

  /** Reply to ``value`` in the default session; use {@link newSession} for independent chats. */
  forward(value: unknown, context: Context | null): string {
    if (context && Object.keys(context).length) throw new ValueError('Use an independent new_session() for conversation state');
    return this.session.call(value);
  }

  private tokenCount(text: string): number {
    return this.tokenizer.encode(text).inputIds[0]!.length;
  }

  /** @internal one conversation turn for ``session`` (commits only on success). */
  respond(value: unknown, session: ChatSession): string {
    if (session.cognition !== null) return this.respondCognitive(value, session);
    if (typeof value !== 'string' || !value.trim()) throw new ValueError('Chatbot expects nonempty text');
    // Source IDs increase even when old turns are evicted.
    const nextId = nextTurnId(session.history);
    const pending: ChatTurn = { source_id: `turn-${nextId}`, role: 'user', text: value };
    const evidence = [...session.history, pending];
    const prompt = evidence.map((item) => `${item.role}: ${item.text}`).join('\n');
    const answer = this.generateBatch([prompt])[0]!;
    const output: ChatTurn = { source_id: `turn-${nextId + 1}`, role: 'assistant', text: answer };
    // Commit only once encoding, workspace computation, and decoding succeed.
    const receipt: JsonObject = {
      text: answer, source_ids: evidence.map((item) => item.source_id),
      evidence: evidence.map((item) => ({ ...item })),
      input_truncated: this.tokenCount(prompt) > (this.config.max_input_tokens as number),
    };
    session.history = [...evidence, output].slice(-2 * (this.config.max_turns as number));
    session.lastResult = receipt;
    return answer;
  }

  private respondCognitive(input: unknown, session: ChatSession): string {
    const investigator = this.investigator!;
    const cognition = this.config.cognition as JsonObject;
    const value = (typeof input === 'string' ? { question: input } : input) as Record<string, unknown>;
    if (!isPlainObject(value) || typeof value.question !== 'string' || !value.question.trim()) {
      throw new ValueError('Cognitive Chatbot expects a question and optional explicitly sourced evidence');
    }
    if (Object.keys(value).some((key) => !['question', 'evidence', 'revisions', 'remove_evidence'].includes(key))) {
      throw new ValueError('Unknown cognitive input fields');
    }
    const question = value.question;
    const [dialogue, dialogueTruncated] = this.conversationContextFor(session.history);
    const proposed = session.cognition!.fork({ copyMemory: true });
    const rows = value.evidence ?? [];
    if (!Array.isArray(rows)) throw new ValueError('evidence must be a list of explicitly sourced records');
    const evidence: Evidence[] = rows.map((row) => {
      if (!isPlainObject(row) || Object.keys(row).some((key) => !['id', 'source_id', 'text'].includes(key))) {
        throw new ValueError('Malformed evidence record');
      }
      return new Evidence((row.id ?? row.source_id) as string, row.text as string, row.source_id as string);
    });
    if (evidence.length) proposed.ingest(evidence);
    const revisions = value.revisions ?? [];
    if (!Array.isArray(revisions)) throw new ValueError('revisions must be a list');
    for (const row of revisions) {
      if (!isPlainObject(row) || Object.keys(row).some((key) => !['evidence_id', 'text', 'source_id'].includes(key))) {
        throw new ValueError('Malformed evidence revision');
      }
      proposed.reviseEvidence(row.evidence_id as string, row.text as string, (row.source_id as string | undefined) ?? null);
    }
    const removed = value.remove_evidence ?? [];
    if (!Array.isArray(removed) || removed.some((item) => typeof item !== 'string')) {
      throw new ValueError('remove_evidence must contain logical evidence IDs');
    }
    for (const evidenceId of removed as string[]) proposed.removeEvidence(evidenceId);
    const { answer, receipt, pending, output } = withEvalModes(this, () => {
      const interpretation = noGrad(() => proposed.investigate(question, {
        count: (cognition.proposal_count as number | undefined) ?? 3, conversationContext: dialogue,
      }));
      const [prompt, visibleEvidence, truncation] = this.realizationInput(question, interpretation, { conversationContext: dialogue });
      const decoded = this.generateBatch([prompt])[0]!;
      let checks: JsonObject[] = [];
      let jointCheck: JsonValue = null;
      let fullJointCheck: JsonValue = null;
      let supported = false;
      const scope = investigator.config.verification_scope as string;
      const interpretationEvidence = interpretation.evidence as JsonObject[];
      if (!interpretation.abstained && decoded.trim() && visibleEvidence.length) {
        noGrad(() => {
          const verification = investigator.verify(decoded, visibleEvidence.map((row) => ({ source_id: row.id as string, text: row.text as string })));
          checks = verification.verifications as JsonObject[];
          jointCheck = (verification.joint_verification ?? null) as JsonValue;
          supported = proposed.policy.acceptsVerification(verification, visibleEvidence.map((row) => row.id as string), { scope });
          if (supported && truncation.length) {
            const full = investigator.verify(decoded, interpretationEvidence.map((row) => ({ source_id: row.id as string, text: row.text as string })));
            fullJointCheck = (full.joint_verification ?? null) as JsonValue;
            supported = proposed.policy.acceptsVerification(full, interpretationEvidence.map((row) => row.id as string), { scope });
            checks = [...checks, ...(full.verifications as JsonObject[]).map((row) => ({ ...row, scope: 'full-active-evidence' }))];
          }
        });
      }
      const abstained = Boolean(interpretation.abstained) || !supported;
      const text = abstained ? ((cognition.abstention_text as string | undefined) ?? ABSTENTION) : decoded;
      const nextId = nextTurnId(session.history);
      const pendingTurn: ChatTurn = { source_id: `turn-${nextId}`, role: 'user', text: question };
      const outputTurn: ChatTurn = { source_id: `turn-${nextId + 1}`, role: 'assistant', text };
      const retained: string[] = [];
      if (proposed.memory !== null) {
        // Explicit authored retention: supplied source evidence, never
        // questions, generated hypotheses, or assistant utterances.
        const logicalIds = new Set([...evidence.map((row) => row.id), ...(revisions as JsonObject[]).map((row) => row.evidence_id as string)]);
        const active = proposed.snapshot().active_evidence;
        const remembered = new Set(proposed.memory.snapshot().records.map((row) => row.evidence.id));
        for (const logicalId of [...logicalIds].sort()) {
          if (logicalId in active) {
            const actualId = active[logicalId]!;
            if (!remembered.has(actualId)) {
              proposed.remember(actualId, { question });
              retained.push(actualId);
            }
          }
        }
      }
      const result: JsonObject = {
        text, cognition: interpretation,
        conversation_context: dialogue as unknown as JsonValue,
        conversation_context_truncated: dialogueTruncated,
        response_proposal: { text: decoded, origin: 'model_generation', epistemic_status: 'unverified_proposal' },
        retained_evidence_ids: retained,
        retention_policy: 'authored: retain explicitly supplied active sources after successful turn',
        abstention_enforced: abstained,
        realization_verifications: checks,
        verification_scope: scope,
        realization_joint_verification: jointCheck,
        full_realization_joint_verification: fullJointCheck,
        realization_sources: visibleEvidence,
        source_truncation: truncation,
        realization_semantics: 'Authored screening of model NLI scores; not a factual guarantee',
        capabilities: this.capabilities,
        input_truncated: this.tokenCount(prompt) > (this.config.max_input_tokens as number),
      };
      return { answer: text, receipt: result, pending: pendingTurn, output: outputTurn };
    });
    // A failed investigation or decoder never commits evidence or dialogue.
    session.cognition = proposed;
    session.history = [...session.history, pending, output].slice(-2 * (this.config.max_turns as number));
    session.lastResult = receipt;
    return answer;
  }

  /** Retain whole recent turn pairs within a separate contextual token budget. */
  conversationContextFor(history: readonly ChatTurn[]): [ConversationRow[], boolean] {
    const investigator = this.investigator!;
    const rows: ConversationRow[] = history.map((item) => ({ role: item.role, text: item.text }));
    let selected: ConversationRow[] = [];
    const budget = (this.config.cognition as JsonObject).conversation_context_tokens as number;
    const tokenizers = [this.tokenizer, investigator.generator!.tokenizer];
    const fits = (context: ConversationRow[]): boolean => {
      const block = conversationBlock(context);
      const rank = investigator.rank;
      const rankLength = rank.tokenizer !== null ? rank.tokenizer.encode(block).inputIds[0]!.length : wordTokens(casefold(block)).length;
      return rankLength <= (rank.config.max_tokens as number)
        && tokenizers.every((tokenizer) => tokenizer.encode(block).inputIds[0]!.length <= budget);
    };
    for (let index = rows.length - 2; index >= 0; index -= 2) {
      const pair = rows.slice(index, index + 2);
      if (!fits([...pair, ...selected])) break;
      selected = [...pair, ...selected];
    }
    if (rows.length && !selected.length) throw new ValueError('conversation_context_tokens is too small for prior dialogue');
    const truncated = selected.length !== rows.length || selected.some((row, index) => row.role !== rows[index]!.role || row.text !== rows[index]!.text);
    return [selected, truncated];
  }

  /** Budget source text before audit metadata, retaining exact text prefixes. */
  realizationInput(question: string, interpretation: JsonObject, options: { conversationContext?: ConversationRow[] | null } = {}): [string, JsonObject[], JsonObject[]] {
    const candidates = (interpretation.candidates ?? []) as JsonObject[];
    const selected = candidates.find((row) => row.id === interpretation.selected_id) ?? null;
    const hypothesis = selected !== null ? selected.text as string : 'No supported selection';
    let prompt = `Answer using only the evidence below. Preserve uncertainty.\nQuestion: ${question}\n`
      + `Selected hypothesis (not an observation): ${hypothesis}\nEvidence:\n`;
    const visible: JsonObject[] = [];
    const truncated: JsonObject[] = [];
    const sources = [...(interpretation.evidence as JsonObject[])];
    if (selected !== null) {
      const support = new Map<unknown, number>();
      for (const row of (selected.verifications ?? []) as JsonObject[]) {
        support.set(row.evidence_id ?? null, (row.distribution as JsonObject).support as number);
      }
      sources.sort((a, b) => (support.get(b.id) ?? 0) - (support.get(a.id) ?? 0));
    }
    const budget = this.config.max_input_tokens as number;
    for (const row of sources) {
      const prefix = `[${row.source_id}] `;
      const characters = Array.from(row.text as string);
      let lo = 0;
      let hi = characters.length;
      while (lo < hi) {
        const middle = Math.floor((lo + hi + 1) / 2);
        const size = this.tokenCount(`${prompt}${prefix}${characters.slice(0, middle).join('')}\n`);
        if (size <= budget) lo = middle;
        else hi = middle - 1;
      }
      const text = characters.slice(0, lo).join('');
      if (lo && text.trim()) {
        prompt += `${prefix}${text}\n`;
        visible.push({ ...row, text });
      }
      if (lo !== characters.length) {
        truncated.push({ evidence_id: row.id!, source_id: row.source_id!, included_characters: lo, original_characters: characters.length });
      }
    }
    const context = options.conversationContext ?? null;
    if (context !== null && context.length) {
      const contextual = conversationBlock(context) + prompt;
      if (this.tokenCount(contextual) > budget) {
        throw new ValueError('Conversation and source evidence exceed realization token budget; increase max_input_tokens or reduce conversation context');
      }
      prompt = contextual;
    }
    return [prompt, visible, truncated];
  }
}

function nextTurnId(history: readonly ChatTurn[]): number {
  if (!history.length) return 0;
  const last = history[history.length - 1]!.source_id;
  return Number(last.split('-').pop()) + 1;
}
