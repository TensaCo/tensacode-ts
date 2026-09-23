/**
 * Owned, supervised response-quality judgments over complete supplied inputs
 * (Python ``tensorcode/_internal/response_quality.py``).
 *
 * Three independent binary heads assess support, completeness and constraints.
 * Foundation relevance training does not train these newly initialized
 * semantics. Scores are model judgments; neither source truth nor answer
 * correctness is assured.
 */
import { createHash } from 'node:crypto';
import { Module } from '../nn/module.js';
import { Tensor, stack, tensor, zeros } from '../nn/index.js';
import { Linear } from '../nn/layers.js';
import { noGrad } from '../nn/autograd.js';
import { binaryCrossEntropyWithLogits } from '../nn/ops/nn.js';
import { tensorBytes } from '../nn/safetensors.js';
import { ValueError } from '../errors.js';
import { invoke, invokeAsync } from './tracing.js';
import { LatentOperation } from './latentOps.js';
import { isPlainObject, pythonJsonDumps, validatedJson, type JsonObject, type JsonValue } from './json.js';
import { pythonClassName, rejectUnknownToolFields, validatedModelConfig } from './pretrained.js';
import { createNativeModel, loadNativeFoundation, nativeConfig, type NativeEncoder } from './native/index.js';
import { FastTokenizer, type TensorBatch } from './tokenizers/index.js';
import { TemperatureCalibration, type CalibrationReport } from '../training/calibration.js';
import { OPERATION_BRAND, type CallOptions, type Context, type OperationLike } from '../ops/base.js';

export const AXES = Object.freeze(['support', 'completeness', 'constraints'] as const);
export type Axis = (typeof AXES)[number];

export interface EvidenceItem {
  source_id: string;
  text: string;
}

export interface QualityInputs {
  question: string;
  evidence: EvidenceItem[];
  candidate: string;
}

export type QualityTargets = Record<Axis, boolean | null>;

export interface InputMetadata {
  source_ids: string[];
  input_truncated: boolean;
  input_token_count: number;
  max_tokens: number;
}

export interface QualityReceipt extends InputMetadata {
  scores: Record<Axis, number>;
  logits: Record<Axis, number>;
  calibrated: Record<Axis, boolean>;
  calibration_sample_count: Record<Axis, number>;
  origin: 'model_inference';
  model: JsonValue;
  head_initialization: string;
  semantics: string;
}

const ALLOWED = new Set(['foundation_config', 'tokenizer_json', 'tokenizer_special_tokens', 'tokenizer_options', 'max_tokens',
  'foundation', 'calibration', 'input_format']);

function normalizedConfig(input: unknown, owner: string): JsonObject {
  const config = validatedModelConfig(input);
  rejectUnknownToolFields(config, ALLOWED, owner);
  const format = 'input_format' in config ? config.input_format : 'json';
  if (format !== 'json' && format !== 'paired') throw new ValueError('input_format must be json or paired');
  if (!isPlainObject(config.foundation_config) || typeof config.tokenizer_json !== 'string') {
    throw new ValueError('complete native encoder and tokenizer configuration required');
  }
  const native = config.foundation_config as JsonObject;
  if ((native.model_type !== 'bert' && native.model_type !== 'electra') || native.is_decoder || native.is_encoder_decoder) {
    throw new ValueError('response quality supports native BERT or Electra encoders');
  }
  config.max_tokens ??= 512;
  config.tokenizer_special_tokens ??= {};
  config.tokenizer_options ??= {};
  config.calibration ??= {};
  const capacity = typeof native.max_position_embeddings === 'number' ? native.max_position_embeddings : 512;
  const maxTokens = config.max_tokens;
  if (typeof maxTokens !== 'number' || !Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > capacity) {
    throw new ValueError('max_tokens must fit the native position capacity and be at least two');
  }
  for (const key of ['tokenizer_special_tokens', 'tokenizer_options', 'calibration']) {
    if (!isPlainObject(config[key])) throw new ValueError('tokenizer and calibration options must be objects');
  }
  const options = config.tokenizer_options as JsonObject;
  if (Object.keys(options).some((key) => !['options', 'padding_side', 'truncation_side'].includes(key))
    || ['padding_side', 'truncation_side'].some((side) => (options[side] ?? 'right') !== 'right')) {
    throw new ValueError('tokenizer options require right padding/truncation and known asset fields');
  }
  if ('foundation' in config && !isPlainObject(config.foundation)) throw new ValueError('foundation provenance must be an object');
  return config;
}

function calibrationOptions(config: JsonObject): { minTemperature?: number; maxTemperature?: number; iterations?: number } {
  const names: Record<string, 'minTemperature' | 'maxTemperature' | 'iterations'> = {
    min_temperature: 'minTemperature', max_temperature: 'maxTemperature', iterations: 'iterations',
  };
  const result: { minTemperature?: number; maxTemperature?: number; iterations?: number } = {};
  for (const [key, value] of Object.entries(config)) {
    const name = names[key];
    if (!name) throw new TypeError(`TemperatureCalibration got an unexpected keyword argument '${key}'`);
    if (typeof value !== 'number') throw new ValueError(`calibration ${key} must be a number`);
    result[name] = value;
  }
  return result;
}

/** Python ``str(tuple(shape))``. */
function shapeRepr(shape: readonly number[]): string {
  return shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
}

function parseTargets(targets: unknown): [number[], boolean[]] {
  if (!isPlainObject(targets) || Object.keys(targets).length !== AXES.length || AXES.some((axis) => !(axis in targets))
    || AXES.some((axis) => targets[axis] !== null && typeof targets[axis] !== 'boolean')) {
    throw new ValueError('targets must map each response quality axis to bool or None');
  }
  return [AXES.map((axis) => (targets[axis] ? 1 : 0)), AXES.map((axis) => targets[axis] !== null)];
}

/** The owned training objective (``{inputs, targets}`` → scalar loss); its parameters are the owner's. */
class QualityObjective extends Module implements OperationLike<unknown, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.response_quality._QualityObjective';
  readonly [OPERATION_BRAND] = true as const;
  readonly #owner: ResponseQualityAssessor;

  constructor(owner: ResponseQualityAssessor) {
    super();
    this.#owner = owner;
  }

  get replayable(): boolean {
    return true;
  }

  override parameters(recurse = true): ReturnType<Module['parameters']> {
    return this.#owner.parameters(recurse);
  }

  configuration(): JsonObject {
    const owner = this.#owner;
    return {
      operation: 'tensorcode._internal.response_quality._QualityObjective',
      owner: (owner.constructor as typeof ResponseQualityAssessor).toolIdentity(),
      config: owner.configuration(),
    };
  }

  call(value: unknown, options?: CallOptions): Tensor {
    return invoke(this, value, options?.context ?? null, (v, c) => this.forward(v, c));
  }

  acall(value: unknown, options?: CallOptions): Promise<Tensor> {
    return invokeAsync(this, value, options?.context ?? null, (v, c) => this.aforward(v, c));
  }

  async aforward(value: unknown, context: Context | null): Promise<Tensor> {
    return this.forward(value, context);
  }

  forward(value: unknown, context: Context | null): Tensor {
    if (context && Object.keys(context).length) throw new ValueError('response quality objective does not consume context');
    if (!isPlainObject(value) || Object.keys(value).length !== 2 || !('inputs' in value) || !('targets' in value)) {
      throw new ValueError('objective requires inputs and targets');
    }
    return this.#owner.loss(value.inputs as QualityInputs | QualityInputs[], value.targets as QualityTargets | QualityTargets[]);
  }
}

export interface FromFoundationOptions {
  revision?: string | null;
  maxTokens?: number;
  localFilesOnly?: boolean;
  inputFormat?: 'json' | 'paired';
  cacheDir?: string | null;
  token?: string | null;
  endpoint?: string | null;
  fetch?: typeof fetch;
}

/**
 * JSON-configured native encoder and three owned binary classifier heads.
 *
 * ``forward`` returns raw logits in {@link AXES} order, with shape ``[3]`` for
 * one input or ``[batch, 3]`` for a nonempty list. Only {@link loss} consumes
 * targets. {@link receipt} reports one input's scores and full-input
 * truncation status. All tokenizer assets are embedded in the data-only
 * configuration. ``input_format`` defaults to ``json`` for the single
 * serialized input; ``paired`` encodes question/candidate against the complete
 * evidence list using the tokenizer's native pair separators and segment ids.
 */
export class ResponseQualityAssessor extends LatentOperation<QualityInputs | QualityInputs[], Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.response_quality.ResponseQualityAssessor';
  readonly trainingInputsIncludeTargets = true;
  readonly encoder: NativeEncoder;
  readonly head: Linear;
  readonly tokenizer: FastTokenizer;
  readonly calibrations: Module;
  readonly objective: QualityObjective;
  private readonly pairTokenizer: FastTokenizer;
  private calibrationVersions: string | null = null;

  constructor(config: unknown) {
    const normalized = normalizedConfig(config, pythonClassName(new.target));
    super(normalized);
    const native = normalized.foundation_config as JsonObject;
    const encoder = createNativeModel(nativeConfig(native), 'base', native.model_type === 'bert' ? { addPoolingLayer: false } : {});
    this.encoder = this.registerModule('encoder', encoder) as unknown as NativeEncoder;
    this.head = this.registerModule('head', new Linear(encoder.config.hiddenSize, AXES.length));
    const assets = normalized.tokenizer_options as JsonObject;
    const tokenizerConfig = {
      json: normalized.tokenizer_json as string, special_tokens: normalized.tokenizer_special_tokens as JsonObject,
      options: (assets.options ?? {}) as JsonObject, padding_side: 'right', truncation_side: 'right',
    };
    this.tokenizer = FastTokenizer.fromConfiguration(tokenizerConfig);
    // Pair encodings request segment ids (``return_token_type_ids=True``).
    const names = this.tokenizer.options.model_input_names;
    this.pairTokenizer = names.includes('token_type_ids') ? this.tokenizer : FastTokenizer.fromConfiguration({
      ...tokenizerConfig, options: { ...this.tokenizer.options, model_input_names: ['input_ids', 'token_type_ids', 'attention_mask'] },
    });
    if (this.tokenizer.padTokenId === null) throw new ValueError('response quality tokenizer requires a padding token');
    if ((normalized.max_tokens as number) < this.tokenizer.backend.numSpecialTokensToAdd(this.inputFormat === 'paired')) {
      throw new ValueError('max_tokens must accommodate the native tokenizer special tokens');
    }
    if (this.tokenizer.length > encoder.config.number('vocab_size')) {
      throw new ValueError('tokenizer vocabulary exceeds native embedding capacity');
    }
    const calibrations = new Module();
    const options = calibrationOptions(normalized.calibration as JsonObject);
    for (const axis of AXES) calibrations.registerModule(axis, new TemperatureCalibration(options));
    this.calibrations = this.registerModule('calibrations', calibrations);
    this.registerBuffer('calibration_weight_digest', zeros([32], { dtype: 'uint8' }));
    this.objective = this.registerModule('objective', new QualityObjective(this));
  }

  get trainingOperation(): OperationLike {
    return this.objective;
  }

  get inputFormat(): 'json' | 'paired' {
    return (this.config.input_format ?? 'json') as 'json' | 'paired';
  }

  get maxTokens(): number {
    return this.config.max_tokens as number;
  }

  calibration(axis: Axis): TemperatureCalibration {
    return this.calibrations.getModule(axis) as TemperatureCalibration;
  }

  private get weightDigestBuffer(): Tensor {
    return this.getBuffer('calibration_weight_digest')!;
  }

  /** Reject anything except ``{question, evidence, candidate}``; labels belong in targets. */
  static validate(inputs: unknown): asserts inputs is QualityInputs {
    if (!isPlainObject(inputs) || Object.keys(inputs).length !== 3 || !['question', 'evidence', 'candidate'].every((key) => key in inputs)) {
      throw new ValueError('inputs require only question, evidence, candidate; labels belong in targets');
    }
    for (const key of ['question', 'candidate']) {
      const value = inputs[key];
      if (typeof value !== 'string' || !value.trim()) throw new ValueError('question and candidate must be nonempty strings');
    }
    const evidence = inputs.evidence;
    if (!Array.isArray(evidence) || evidence.some((item) => !isPlainObject(item) || Object.keys(item).length !== 2
      || !('source_id' in item) || !('text' in item)
      || ['source_id', 'text'].some((key) => typeof item[key] !== 'string' || !(item[key] as string).trim()))) {
      throw new ValueError('evidence must contain only nonempty source_id/text objects');
    }
    if (new Set(evidence.map((item) => (item as EvidenceItem).source_id)).size !== evidence.length) {
      throw new ValueError('source IDs must be unique');
    }
  }

  private static text(inputs: unknown): string {
    ResponseQualityAssessor.validate(inputs);
    // JSON escaping preserves field boundaries and all supplied text; no gold,
    // rationale, or authored semantic policy enters the encoded representation.
    return pythonJsonDumps({ question: inputs.question, evidence: inputs.evidence, candidate: inputs.candidate }, { ensureAscii: false });
  }

  private encodeBatch(batch: unknown[], options: { padding?: boolean; truncation?: boolean }): TensorBatch | number[][] {
    const settings = { padding: options.padding ?? false, truncation: options.truncation ?? false, maxLength: this.maxTokens };
    if (this.inputFormat === 'json') {
      const texts = batch.map((item) => ResponseQualityAssessor.text(item));
      return options.padding ? this.tokenizer.encodeTensors(texts, settings) : this.tokenizer.encode(texts, settings).inputIds;
    }
    for (const item of batch) ResponseQualityAssessor.validate(item);
    const items = batch as QualityInputs[];
    const first = items.map((item) => `Question: ${item.question}\nCandidate: ${item.candidate}`);
    const evidence = items.map((item) => pythonJsonDumps(item.evidence, { ensureAscii: false }));
    const pairSettings = { ...settings, textPair: evidence };
    return options.padding ? this.pairTokenizer.encodeTensors(first, pairSettings) : this.pairTokenizer.encode(first, pairSettings).inputIds;
  }

  /** Tokenized encoder inputs exactly as ``forward`` passes them (for inspection). */
  encoderInputs(inputs: QualityInputs | QualityInputs[]): TensorBatch {
    const batch = Array.isArray(inputs) ? inputs : [inputs];
    return this.encodeBatch(batch, { padding: true, truncation: true }) as TensorBatch;
  }

  forward(inputs: QualityInputs | QualityInputs[], context: Context | null): Tensor {
    if (context && Object.keys(context).length) throw new ValueError('response quality does not consume context');
    const single = isPlainObject(inputs);
    const batch = single ? [inputs] : inputs;
    if (!Array.isArray(batch) || !batch.length) throw new ValueError('inputs must be one input object or a nonempty list');
    const tokens = this.encodeBatch(batch, { padding: true, truncation: true }) as TensorBatch;
    const hidden = this.encoder.forward({
      inputIds: tokens.input_ids, attentionMask: tokens.attention_mask, tokenTypeIds: tokens.token_type_ids ?? null,
    }).lastHiddenState;
    const logits = this.head.forward(hidden.select(1, 0));
    if (!logits.allFinite()) throw new ValueError('response quality produced nonfinite logits');
    return single ? logits.select(0, 0) : logits;
  }

  /** Masked binary cross entropy averaged over reviewed axes, then over inputs. */
  loss(inputs: QualityInputs | QualityInputs[], targets: QualityTargets | QualityTargets[]): Tensor {
    const single = isPlainObject(inputs);
    const rows = single ? [targets] : targets;
    if (!Array.isArray(rows) || !rows.length || (!single && rows.length !== (inputs as unknown[]).length)) {
      throw new ValueError('one target object is required per input');
    }
    const parsed = rows.map(parseTargets);
    if (!parsed.every(([, mask]) => mask.some(Boolean))) throw new ValueError('at least one reviewed axis is required per input');
    for (const item of single ? [inputs] : (inputs as QualityInputs[])) {
      if (this.inputMetadata(item as QualityInputs).input_truncated) {
        throw new ValueError('supervised response quality inputs cannot be truncated');
      }
    }
    this.invalidateCalibration();
    const logits = this.call(inputs).reshape(-1, AXES.length);
    const values = tensor(parsed.map(([value]) => value), { dtype: logits.dtype });
    const mask = tensor(parsed.map(([, flags]) => flags.map((flag) => (flag ? 1 : 0))), { dtype: logits.dtype });
    const elementwise = binaryCrossEntropyWithLogits(logits, values, 'none');
    return elementwise.mul(mask).sum(-1).div(mask.sum(-1)).mean();
  }

  private weightDigest(): Tensor {
    const digest = createHash('sha256');
    for (const [prefix, module] of [['encoder', this.encoder], ['head', this.head]] as const) {
      for (const [name, value] of module.stateDict()) {
        digest.update(Buffer.from(`${prefix}.${name}torch.${value.dtype}${shapeRepr(value.shape)}`, 'utf8'));
        digest.update(tensorBytes(value));
      }
    }
    return tensor([...digest.digest()], { dtype: 'uint8' });
  }

  private versions(): string {
    const parts: string[] = [];
    for (const module of [this.encoder, this.head]) {
      for (const value of [...module.parameters(), ...module.buffers()]) {
        parts.push(`${value.id}:${value.version}:${value.dtype}:${value.shape.join('x')}`);
      }
    }
    return parts.join('|');
  }

  private invalidateCalibration(): void {
    noGrad(() => {
      for (const axis of AXES) {
        const calibration = this.calibration(axis);
        calibration.calibrated.fill_(0);
        calibration.sampleCount.zero_();
        calibration.temperature.fill_(1);
      }
    });
    this.calibrationVersions = null;
  }

  private validateCalibrationWeights(): void {
    const versions = this.versions();
    if (AXES.some((axis) => this.calibration(axis).isCalibrated) && this.calibrationVersions !== versions) {
      if (!this.weightDigestBuffer.equal(this.weightDigest())) this.invalidateCalibration();
      this.calibrationVersions = versions;
    }
  }

  /**
   * Fit separate temperatures on explicit held-out logits and masked labels.
   * Callers must supply held-out logits from this exact model's current
   * weights; the API cannot establish dataset independence or score origin.
   */
  fitCalibration(logits: Tensor, targets: QualityTargets[]): Record<Axis, CalibrationReport | null> {
    if (!(logits instanceof Tensor) || !logits.isFloatingPoint || logits.ndim !== 2 || logits.shape[1] !== AXES.length
      || !logits.shape[0] || !logits.allFinite()) {
      throw new ValueError('calibration logits must be finite [samples, 3]');
    }
    if (!Array.isArray(targets) || targets.length !== logits.shape[0]) {
      throw new ValueError('one calibration target object is required per sample');
    }
    const parsed = targets.map(parseTargets);
    return noGrad(() => {
      this.invalidateCalibration();
      const results = {} as Record<Axis, CalibrationReport | null>;
      AXES.forEach((axis, index) => {
        const selected = parsed.flatMap(([, mask], row) => (mask[index] ? [row] : []));
        if (!selected.length) {
          results[axis] = null;
          return;
        }
        const values = logits.detach().select(1, index).indexSelect(0, selected);
        const binary = stack([values.mul(0).detach(), values], -1);
        const labels = tensor(selected.map((row) => parsed[row]![0][index]!), { dtype: 'int64' });
        results[axis] = this.calibration(axis).fit(binary, labels);
      });
      this.weightDigestBuffer.copy_(this.weightDigest());
      this.calibrationVersions = this.versions();
      return results;
    });
  }

  /** Inspect full-input token coverage without inference or truncation. */
  inputMetadata(inputs: QualityInputs): InputMetadata {
    const count = (this.encodeBatch([inputs], { truncation: false }) as number[][])[0]!.length;
    return {
      source_ids: inputs.evidence.map((item) => item.source_id),
      input_truncated: count > this.maxTokens, input_token_count: count, max_tokens: this.maxTokens,
    };
  }

  /** Scores for one input with calibration and full-input coverage metadata. */
  receipt(inputs: QualityInputs): QualityReceipt {
    const metadata = this.inputMetadata(inputs);
    this.validateCalibrationWeights();
    const modes = this.modules().map((module) => [module, module.training] as const);
    let logits: Tensor;
    let scores: Record<Axis, number>;
    try {
      this.eval();
      [logits, scores] = noGrad(() => {
        const raw = this.call(inputs);
        const result = {} as Record<Axis, number>;
        AXES.forEach((axis, index) => {
          const value = raw.select(0, index);
          const calibrated = this.calibration(axis).forward(stack([value.mul(0), value])).softmax(-1);
          result[axis] = calibrated.select(0, 1).item();
        });
        if (Object.values(result).some((value) => !Number.isFinite(value))) {
          throw new ValueError('response quality produced nonfinite scores');
        }
        return [raw, result] as const;
      });
    } finally {
      for (const [module, mode] of modes) module.training = mode;
    }
    const values = logits.toArray();
    return {
      scores,
      logits: Object.fromEntries(AXES.map((axis, index) => [axis, values[index]!])) as Record<Axis, number>,
      calibrated: Object.fromEntries(AXES.map((axis) => [axis, this.calibration(axis).isCalibrated])) as Record<Axis, boolean>,
      calibration_sample_count: Object.fromEntries(AXES.map((axis) => [axis, this.calibration(axis).sampleCount.item()])) as Record<Axis, number>,
      ...metadata,
      origin: 'model_inference',
      model: validatedJson(this.config.foundation ?? { initialization: 'random_native_encoder' }),
      head_initialization: 'new_random_binary_heads; foundation task does not train these semantics',
      semantics: 'supervised model judgments, not truth guarantees',
    };
  }

  /** Apply a caller-owned threshold; incomplete input coverage cannot pass. */
  static accepts(receipt: unknown, threshold = 0.5): boolean {
    if (typeof threshold !== 'number' || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
      throw new ValueError('threshold must be finite in [0, 1]');
    }
    if (!isPlainObject(receipt) || typeof receipt.input_truncated !== 'boolean') {
      throw new ValueError('receipt requires explicit input truncation status');
    }
    const count = receipt.input_token_count;
    const limit = receipt.max_tokens;
    if (typeof count !== 'number' || !Number.isInteger(count) || count < 1 || typeof limit !== 'number' || !Number.isInteger(limit)
      || limit < 2 || receipt.input_truncated !== count > limit) {
      throw new ValueError('receipt requires consistent token coverage metadata');
    }
    const scores = receipt.scores;
    if (!isPlainObject(scores) || Object.keys(scores).length !== AXES.length || AXES.some((axis) => !(axis in scores))
      || Object.values(scores).some((value) => typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)) {
      throw new ValueError('receipt requires finite scores for all axes');
    }
    return !receipt.input_truncated && Object.values(scores).every((value) => (value as number) >= threshold);
  }

  /** Explicit native safetensors bootstrap; response-quality heads start random. */
  static async fromFoundation(repo: string, options: FromFoundationOptions = {}): Promise<ResponseQualityAssessor> {
    const { maxTokens = 512, inputFormat = 'json', revision = null, localFilesOnly = false, ...hub } = options;
    const { stat } = await import('node:fs/promises');
    let local = false;
    try {
      local = (await stat(repo)).isDirectory();
    } catch {
      local = false;
    }
    const probe = await loadNativeFoundation(repo, { ...hub, revision, localFilesOnly, head: 'base', tokenizer: true, addPoolingLayer: false, useSafetensors: true });
    const native = probe.config;
    if ((native.modelType !== 'bert' && native.modelType !== 'electra') || native.get('is_decoder') === true || native.isEncoderDecoder) {
      throw new ValueError('response quality supports native BERT or Electra encoders');
    }
    if (!probe.tokenizer) throw new ValueError('foundation requires a fast tokenizer (tokenizer.json)');
    const assets = probe.tokenizer.configuration();
    const resolved = probe.commitHash ?? revision;
    if (!local && !resolved) throw new ValueError('foundation provenance requires a resolved revision');
    const result = new ResponseQualityAssessor({
      foundation_config: native.toDiffDict(), tokenizer_json: assets.json,
      tokenizer_special_tokens: assets.special_tokens as unknown as JsonObject,
      tokenizer_options: { options: assets.options as unknown as JsonObject, padding_side: assets.padding_side, truncation_side: assets.truncation_side },
      max_tokens: maxTokens, input_format: inputFormat,
      foundation: {
        repository: String(repo), revision: resolved, initialization: 'pretrained_encoder_only', response_quality_heads_pretrained: false,
      },
    });
    result.encoder.loadStateDict(probe.model.stateDict(), { strict: true });
    result.eval();
    return result;
  }
}
