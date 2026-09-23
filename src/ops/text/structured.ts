/**
 * Shared structured-operation behavior (Python ``tensorcode/ops/text/_structured.py``):
 * request construction, provider calls, strict response validation, batching
 * and likelihood decoding for owned models.
 */
import { tensor, type Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { isPlainObject, pythonJsonDumps } from '../../_internal/json.js';
import { pythonSum } from '../../_internal/numeric.js';
import { qualifiedName } from '../../_internal/identity.js';
import { activeSession, invoke } from '../../_internal/tracing.js';
import { OwnedTextOperation } from '../../_internal/text/owned.js';
import type { Alternative, NativeModel } from '../../_internal/text/native.js';
import type { Context } from '../base.js';
import { Message } from './messages.js';
import {
  InvalidModelOutput, ModelOutput, ModelRequest, isAsyncBatchModel, isAsyncModel, isBatchModel, isModel,
} from './model.js';

export { InvalidModelOutput };

/** A read-only mapping (Python ``MappingProxyType``). */
export type ReadonlyMapping<V> = Readonly<Record<string, V>>;

/** Freeze a shallow copy of a mapping. */
export function frozenMapping<V>(value: Record<string, V>): ReadonlyMapping<V> {
  return Object.freeze({ ...value });
}

/** JSON description of an external model (its ``configuration()`` or its class identity). */
export function modelConfiguration(model: unknown): unknown {
  const configure = (model as { configuration?: unknown } | null)?.configuration;
  const configured = typeof configure === 'function'
    ? (configure as () => unknown).call(model)
    : { type: qualifiedName(model) };
  try {
    pythonJsonDumps(configured, { allowNan: true });
  } catch (error) {
    throw new TypeError('model.configuration() must return JSON-safe data', { cause: error });
  }
  if (!isJsonSafe(configured)) throw new TypeError('model.configuration() must return JSON-safe data');
  return configured;
}

function isJsonSafe(value: unknown): boolean {
  if (value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return true;
  if (Array.isArray(value)) return value.every(isJsonSafe);
  if (isPlainObject(value)) return Object.values(value).every(isJsonSafe);
  return false;
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return value !== null && value !== undefined && typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === 'function'
    && typeof value !== 'string';
}

/** Context messages (in context order) followed by the primary messages. */
export function messageSequence(value: unknown, context: Context | null = null): readonly Message[] {
  if (!isIterable(value)) throw new TypeError('Expected a Message sequence');
  const messages = [...value];
  if (!messages.length || !messages.every((message) => message instanceof Message)) {
    throw new TypeError('Expected a nonempty Message sequence');
  }
  const conditioning: Message[] = [];
  for (const group of Object.values(context ?? {})) {
    if (!isIterable(group)) throw new TypeError('Context must contain Message sequences');
    const groupMessages = [...group];
    if (!groupMessages.every((message) => message instanceof Message)) throw new TypeError('Context must contain Message sequences');
    conditioning.push(...(groupMessages as Message[]));
  }
  return Object.freeze([...conditioning, ...(messages as Message[])]);
}

function checkedOutput(output: unknown, method: string): ModelOutput {
  if (!(output instanceof ModelOutput)) {
    if (output !== null && typeof output === 'object' && typeof (output as { then?: unknown }).then === 'function') {
      throw new TypeError(`model.${method} returned a promise; implement acomplete and use await operation.acall(...)`);
    }
    throw new TypeError(`model.${method} must return ModelOutput`);
  }
  return output;
}

/** Call ``model.complete(request)`` synchronously. */
export function callModel(model: unknown, request: ModelRequest): ModelOutput {
  if (isModel(model)) return checkedOutput(model.complete(request), 'complete');
  if (isAsyncModel(model)) {
    throw new TypeError('This model is asynchronous (acomplete only); use await operation.acall(...) or abatch(...)');
  }
  throw new TypeError('Structured operations require a model.complete(ModelRequest) method');
}

export function requireStructured(output: ModelOutput): Readonly<Record<string, unknown>> {
  if (output.structured === null) throw new InvalidModelOutput('Model did not supply structured output');
  return output.structured;
}

export function optionalBool(value: Readonly<Record<string, unknown>>, key: string): boolean {
  if (!Object.hasOwn(value, key)) throw new InvalidModelOutput(`${key} must be supplied explicitly`);
  const raw = value[key];
  if (typeof raw !== 'boolean') throw new InvalidModelOutput(`${key} must be a boolean`);
  return raw;
}

export function optionalConfidence(value: Readonly<Record<string, unknown>>): number | null {
  if (!Object.hasOwn(value, 'confidence') || value.confidence === null || value.confidence === undefined) return null;
  const confidence = value.confidence;
  if (typeof confidence !== 'number' || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new InvalidModelOutput('confidence must be a number from 0 to 1');
  }
  return confidence;
}

/** Validate a probability distribution over exactly ``expectedKeys`` summing to one (±0.001). */
export function probabilityDistribution(
  raw: unknown, expectedKeys: readonly string[], keyTransform: (key: string) => string = (key) => key,
): ReadonlyMapping<number> | null {
  if (raw === null || raw === undefined) return null;
  if (!isPlainObject(raw)) throw new InvalidModelOutput('distribution must be a mapping');
  const distribution: Record<string, unknown> = {};
  try {
    for (const [key, value] of Object.entries(raw)) distribution[keyTransform(key)] = value;
  } catch (error) {
    throw new InvalidModelOutput('distribution contains invalid keys', { cause: error });
  }
  const keys = Object.keys(distribution);
  const expected = new Set(expectedKeys);
  if (keys.length !== expected.size || !keys.every((key) => expected.has(key))) {
    throw new InvalidModelOutput('distribution keys must match the configured alternatives');
  }
  const normalized: Record<string, number> = {};
  for (const [key, probability] of Object.entries(distribution)) {
    if (typeof probability !== 'number') throw new InvalidModelOutput('distribution probabilities must be numbers');
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new InvalidModelOutput('distribution probabilities must be between 0 and 1');
    }
    normalized[key] = probability;
  }
  const total = pythonSum(Object.values(normalized));
  if (!(Math.abs(total - 1) <= 1e-3)) throw new InvalidModelOutput('distribution probabilities must sum to 1');
  return frozenMapping(normalized);
}

/** Validate finite (non-probability) scores keyed by exactly the configured items. */
export function finiteScores(raw: unknown, expectedKeys: readonly string[]): ReadonlyMapping<number> | null {
  if (raw === null || raw === undefined) return null;
  const expected = new Set(expectedKeys);
  if (!isPlainObject(raw) || Object.keys(raw).length !== expected.size || !Object.keys(raw).every((key) => expected.has(key))) {
    throw new InvalidModelOutput('scores must contain exactly the configured item keys');
  }
  const result: Record<string, number> = {};
  for (const [key, score] of Object.entries(raw)) {
    if (typeof score !== 'number' || !Number.isFinite(score)) throw new InvalidModelOutput('scores must be finite numbers');
    result[key] = score;
  }
  return frozenMapping(result);
}

export function softmax(scores: readonly number[]): number[] {
  const top = Math.max(...scores);
  const weights = scores.map((score) => Math.exp(score - top));
  const total = pythonSum(weights);
  return weights.map((weight) => weight / total);
}

/** Validate optional nonempty descriptions keyed by configured alternatives. */
export function alternativeDescriptions(raw: unknown, alternatives: readonly string[], name: string): ReadonlyMapping<string> {
  if (raw === null || raw === undefined) return frozenMapping({});
  const allowed = new Set(alternatives);
  if (!isPlainObject(raw) || !Object.keys(raw).every((key) => allowed.has(key))) {
    throw new ValueError(`descriptions must map configured ${name} to text`);
  }
  if (!Object.values(raw).every((text) => typeof text === 'string' && text.length > 0)) {
    throw new ValueError('descriptions must be nonempty strings');
  }
  return frozenMapping(raw as Record<string, string>);
}

/** Per-item contexts for ``batch``/``abatch``. */
function batchContexts(values: readonly unknown[], contexts: readonly (Context | null | undefined)[] | null | undefined): (Context | null)[] {
  if (contexts === null || contexts === undefined) return values.map(() => null);
  const items = [...contexts];
  if (items.length !== values.length) throw new ValueError('contexts must match the number of values');
  return items.map((context) => context ?? null);
}

export interface BatchOptions {
  contexts?: readonly (Context | null | undefined)[] | null;
}

/**
 * Owned or external structured operation (``Classify``, ``Decide``, ``Score``,
 * ``Retrieve``) returning a validated frozen result.
 */
export abstract class StructuredOperation<R> extends OwnedTextOperation<R> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text._structured.StructuredOperation';
  static override readonly decodingFields: ReadonlySet<string> = new Set(['decoding', 'likelihood_normalization']);
  /** Stable response schema name sent to providers. */
  static readonly schemaName: string = 'tensorcode.structured';

  declare decoding: 'generate' | 'likelihood';
  declare likelihoodNormalization: 'sum' | 'mean';

  get schemaName(): string {
    return (this.constructor as typeof StructuredOperation).schemaName;
  }

  /** @internal */
  protected override _configureDecoding(config: Record<string, unknown>): void {
    const decoding = config.decoding ?? 'generate';
    if (decoding !== 'generate' && decoding !== 'likelihood') throw new ValueError("decoding must be 'generate' or 'likelihood'");
    this.decoding = decoding;
    const normalization = config.likelihood_normalization ?? 'sum';
    if (normalization !== 'sum' && normalization !== 'mean') throw new ValueError("likelihood_normalization must be 'sum' or 'mean'");
    this.likelihoodNormalization = normalization;
    if ('likelihood_normalization' in config && this.decoding !== 'likelihood') {
      throw new ValueError("likelihood_normalization requires decoding='likelihood'");
    }
  }

  /** JSON schema the model's structured output must satisfy. */
  abstract responseSchema(): Record<string, unknown>;

  /** Validate a structured response (or target) into the frozen result. */
  abstract override _parse(value: Readonly<Record<string, unknown>>): R;

  /** ``(display, target)`` text for each configured alternative. */
  abstract _alternatives(): Alternative[];

  /** Result of likelihood decoding from per-alternative log-likelihoods. */
  abstract _fromScores(scores: readonly number[]): R;

  /** Probability vector over alternatives for a parsed target result. */
  abstract _targetWeights(result: R): number[];

  /** @internal */
  _scoringRequest(value: readonly Message[], context: Context | null): ModelRequest {
    return new ModelRequest(messageSequence(value, context), { instructions: this.instructions });
  }

  _request(value: readonly Message[], context: Context | null): ModelRequest {
    return new ModelRequest(messageSequence(value, context), {
      instructions: this.instructions, responseSchema: this.responseSchema(), schemaName: this.schemaName,
    });
  }

  /** @internal */
  protected _likelihoodForward(value: readonly Message[], context: Context | null): R {
    const model = this._requireOwned();
    const scores = model.scoreAlternatives(this._scoringRequest(value, context), this._alternatives(), {
      normalization: this.likelihoodNormalization,
    });
    return this._fromScores(scores);
  }

  /** @internal */
  protected override _likelihoodLoss(value: readonly Message[], targets: unknown, context: Context | null): Tensor {
    if (!isPlainObject(targets)) throw new ValueError('structured targets must be an explicit JSON mapping');
    const weights = tensor(this._targetWeights(this._parse(targets)), { dtype: 'float32' });
    const model = this.model as NativeModel;
    const scores = model.alternativeLogLikelihoods(this._scoringRequest(value, context), this._alternatives(), {
      normalization: this.likelihoodNormalization,
    });
    return weights.mul(scores.logSoftmax(-1)).sum().neg();
  }

  forward(value: readonly Message[], context: Context | null): R {
    if (this.decoding === 'likelihood') return this._likelihoodForward(value, context);
    return this._parse(requireStructured(callModel(this.model, this._request(value, context))));
  }

  override async aforward(value: readonly Message[], context: Context | null): Promise<R> {
    if (this.decoding === 'likelihood') return this.forward(value, context);
    const request = this._request(value, context);
    if (isAsyncModel(this.model)) {
      const output = await this.model.acomplete(request);
      if (!(output instanceof ModelOutput)) throw new TypeError('model.acomplete must return ModelOutput');
      return this._parse(requireStructured(output));
    }
    return this.forward(value, context);
  }

  /**
   * Call the operation on every value in order. Outside tracing, a backend
   * ``completeBatch`` fuses the transport; each result stays an ordinary
   * operation boundary. Under tracing each value is called normally.
   */
  batch(values: Iterable<readonly Message[]>, options: BatchOptions = {}): readonly R[] {
    const items = [...values];
    const contexts = batchContexts(items, options.contexts);
    const perItem = (): readonly R[] => Object.freeze(items.map((value, index) => this.call(value, { context: contexts[index] })));
    // A fused provider call cannot reserve and complete ordinary trace calls
    // atomically; keep reference unwrapping and failed-call capture per item.
    if (activeSession() !== null || !isBatchModel(this.model)) return perItem();
    const requests = items.map((value, index) => this._request(value, contexts[index]!));
    const outputs = [...this.model.completeBatch(requests)];
    if (outputs.length !== requests.length) throw new InvalidModelOutput('Model batch result count does not match request count');
    if (!outputs.every((output) => output instanceof ModelOutput)) throw new TypeError('model.completeBatch must return ModelOutput values');
    const parsed = outputs.map((output) => this._parse(requireStructured(output)));
    return Object.freeze(items.map((value, index) => invoke(this, value, contexts[index]!, () => parsed[index]!)));
  }

  /**
   * Asynchronous {@link batch}: explicit ``acall`` per value, or one fused
   * ``acompleteBatch`` exchange outside tracing when the model provides it.
   */
  async abatch(values: Iterable<readonly Message[]>, options: BatchOptions = {}): Promise<readonly R[]> {
    const items = [...values];
    const contexts = batchContexts(items, options.contexts);
    if (activeSession() === null && this.decoding !== 'likelihood' && isAsyncBatchModel(this.model)) {
      const requests = items.map((value, index) => this._request(value, contexts[index]!));
      const outputs = [...await this.model.acompleteBatch(requests)];
      if (outputs.length !== requests.length) throw new InvalidModelOutput('Model batch result count does not match request count');
      if (!outputs.every((output) => output instanceof ModelOutput)) throw new TypeError('model.acompleteBatch must return ModelOutput values');
      const parsed = outputs.map((output) => this._parse(requireStructured(output)));
      return Object.freeze(items.map((value, index) => invoke(this, value, contexts[index]!, () => parsed[index]!)));
    }
    return Object.freeze(await Promise.all(items.map((value, index) => this.acall(value, { context: contexts[index] }))));
  }
}

/** Shared likelihood behavior for ``Classify`` and ``Decide`` alternatives. */
export interface SelectionResult {
  readonly value: string | null;
  readonly distribution: ReadonlyMapping<number> | null;
  readonly abstained: boolean;
}

export abstract class SelectionOperation<R extends SelectionResult> extends StructuredOperation<R> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text._structured.SelectionOperation';
  declare descriptions: ReadonlyMapping<string>;

  /** The configured alternatives (labels or options). */
  abstract _choices(): readonly string[];

  /** Build the result record. */
  protected abstract _result(
    value: string | null, fields: { distribution: Record<string, number> | null; confidence: number | null; abstained: boolean },
  ): R;

  _alternatives(): Alternative[] {
    return this._choices().map((alternative) => [
      Object.hasOwn(this.descriptions, alternative) ? `${alternative}: ${this.descriptions[alternative]}` : alternative, alternative,
    ] as const);
  }

  _fromScores(scores: readonly number[]): R {
    const choices = this._choices();
    const probabilities = softmax(scores);
    let best = 0;
    for (let index = 1; index < probabilities.length; index += 1) if (probabilities[index]! > probabilities[best]!) best = index;
    const distribution: Record<string, number> = {};
    choices.forEach((choice, index) => { distribution[choice] = probabilities[index]!; });
    return this._result(choices[best]!, { distribution, confidence: probabilities[best]!, abstained: false });
  }

  _targetWeights(result: R): number[] {
    if (result.abstained) throw new ValueError('likelihood decoding has no abstention alternative');
    const choices = this._choices();
    if (result.distribution !== null) return choices.map((choice) => result.distribution![choice]!);
    return choices.map((choice) => (choice === result.value ? 1 : 0));
  }
}

/** Selection schema shared by ``Classify`` (``label``) and ``Decide`` (``choice``). */
export function selectionSchema(field: string, alternatives: readonly string[], descriptions: ReadonlyMapping<string> = {}): Record<string, unknown> {
  const properties: Record<string, unknown> = {};
  for (const label of alternatives) {
    properties[label] = {
      type: 'number', minimum: 0, maximum: 1, ...(Object.hasOwn(descriptions, label) ? { description: descriptions[label] } : {}),
    };
  }
  return {
    type: 'object',
    properties: {
      [field]: { type: ['string', 'null'], enum: [...alternatives, null] },
      distribution: { type: ['object', 'null'], properties, required: [...alternatives], additionalProperties: false },
      confidence: { type: ['number', 'null'], minimum: 0, maximum: 1 },
      abstained: { type: 'boolean' },
    },
    required: [field, 'distribution', 'confidence', 'abstained'],
    additionalProperties: false,
  };
}

/** Validate that a configured alternative list is a nonempty array of unique nonempty strings. */
export function alternativeList(raw: unknown, name: string): readonly string[] {
  if (!Array.isArray(raw)) throw new ValueError(`${name} must be a sequence of strings`);
  const items = Object.freeze([...raw]);
  if (!items.length || !items.every((item) => typeof item === 'string' && item.length > 0)) {
    throw new ValueError(`${name} must be nonempty strings`);
  }
  if (new Set(items).size !== items.length) throw new ValueError(`${name} must be unique`);
  return items as readonly string[];
}
