/**
 * Public-operation facade for owned native seq2seq models (Python
 * ``tensorcode/_internal/text/owned.py``).
 *
 * An owned text operation is an ordinary {@link Operation} (not an
 * ``nn.Module``, so its fingerprint has no state topology) that owns a
 * {@link NativeModel}. Constructors take JSON configuration only;
 * ``fromModel`` explicitly wraps an external provider without owned training
 * or artifacts, and ``fromFoundation`` imports pretrained native weights.
 */
import { Operation, type Context } from '../../ops/base.js';
import type { Module, StateDict, LoadStateDictResult } from '../../nn/module.js';
import type { Parameter, Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { deepCopy, isPlainObject, jsonEqual, validatedJson, type JsonObject } from '../json.js';
import { qualifiedName } from '../identity.js';
import { parseObjectiveEnvelope, type TrainableTool } from '../contracts.js';
import {
  defaultModelCard, publishArtifact, readPretrainedArtifact, writePretrainedArtifact,
  type FromPretrainedOptions, type PushToHubOptions,
} from '../pretrained.js';
import { loadNativeFoundation } from '../native/foundation.js';
import { parameterAliases } from '../native/modules.js';
import type { T5ForConditionalGeneration } from '../native/t5.js';
import type { HubOptions } from '../hub.js';
import type { Message } from '../../ops/text/messages.js';
import type { ModelRequest } from '../../ops/text/model.js';
import { NativeModel, sortedJsonDumps, validateStructured } from './native.js';

/** An explicitly supplied external model: a ``Model``/``AsyncModel`` object or a ``messages → text`` callable. */
export type ExternalModel = object | ((messages: readonly Message[]) => unknown);

/** Options of ``fromFoundation`` (Hub options plus the semantic/decoding/generation ``config``). */
export interface FromFoundationOptions extends Omit<HubOptions, 'allowPatterns' | 'fetch' | 'endpoint'> {
  config?: JsonObject | null;
}

/** Private construction routes (``from_model``/``from_foundation``); not constructible outside this module. */
class Construction {
  constructor(
    readonly external: { model: ExternalModel; options: Record<string, unknown> } | null,
    readonly foundation: T5ForConditionalGeneration | null,
  ) {}
}

/** Python ``repr`` of a sorted list of names (``['a', 'b']``). */
function pythonNames(names: string[]): string {
  return `[${names.map((name) => `'${name.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`).join(', ')}]`;
}

/** Python ``OwnedTextOperation._validated_config``. */
export function validatedOwnedConfig(config: unknown): JsonObject {
  if (!isPlainObject(config)) throw new ValueError('config must be a JSON object');
  let result: JsonObject;
  try {
    result = validatedJson<JsonObject>(config, 'config');
  } catch (error) {
    throw new ValueError('config must contain finite JSON data', { cause: error });
  }
  if (!jsonEqual(result, config)) throw new ValueError('config must use JSON types');
  return result;
}

/** Static side shared by the concrete owned text operation classes. */
export interface OwnedTextOperationClass<T> {
  new (config: unknown): T;
  readonly semanticFields: ReadonlySet<string>;
  readonly decodingFields: ReadonlySet<string>;
  readonly artifactFormat: string;
  readonly artifactVersion: number;
  readonly name: string;
}

/**
 * The declared objective of an owned text operation: receives
 * ``{inputs, targets}`` (``inputs`` may be a ``{value, context}`` conditioning
 * envelope) and returns the teacher-forced loss.
 */
export class TextObjective extends Operation<unknown, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.text.owned.TextObjective';
  readonly #owner: OwnedTextOperation<unknown>;

  constructor(owner: OwnedTextOperation<unknown>) {
    super();
    this.#owner = owner;
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: unknown, context: Context | null): Tensor {
    const envelope = parseObjectiveEnvelope(value, context);
    return this.#owner.loss(envelope.inputs, envelope.targets, { context: envelope.context }).clone();
  }

  parameters(): Parameter[] {
    return this.#owner.parameters();
  }

  /** Owned objectives identify their public owner and role. */
  operationIdentity(): string {
    return `${(this.#owner.constructor as unknown as { toolIdentity(): string }).toolIdentity()}.objective`;
  }

  configuration(): JsonObject {
    return { operation: this.operationIdentity(), model: this.#owner.configuration() };
  }
}

/**
 * Base of owned text operations (``Transform``, ``Classify``, ``Decide``,
 * ``Score``, ``Retrieve``).
 */
export abstract class OwnedTextOperation<O> extends Operation<readonly Message[], O> implements TrainableTool {
  static override readonly qualifiedName: string = 'tensorcode._internal.text.owned.OwnedTextOperation';
  /** Semantic configuration fields (also the ``fromModel`` options). */
  static readonly semanticFields: ReadonlySet<string> = new Set(['instructions']);
  /** Owned-model decoding settings that external ``fromModel`` providers do not accept. */
  static readonly decodingFields: ReadonlySet<string> = new Set();
  static readonly artifactFormat: string = 'tensorcode.pretrained';
  static readonly artifactVersion: number = 1;
  readonly trainingInputsIncludeTargets = true;

  declare model: NativeModel | ExternalModel;
  declare config: JsonObject | null;
  declare instructions: string | null;
  /** @internal */
  declare _owned: boolean;
  declare private _objective: TextObjective | null;

  constructor(config: unknown, construction?: unknown) {
    super();
    const cls = this.constructor as unknown as OwnedTextOperationClass<this>;
    if (construction instanceof Construction && construction.external) {
      const { model, options } = construction.external;
      this._configureSemantics(options);
      this._configureDecoding({});
      this.model = model;
      this.config = null;
      this._owned = false;
      this._objective = null;
      return;
    }
    const validated = validatedOwnedConfig(config);
    const allowed = new Set([...cls.semanticFields, ...cls.decodingFields, 'native_config', 'tokenizer',
      'native_generation_config', 'native_parameter_aliases', 'generation', 'foundation']);
    const unknown = Object.keys(validated).filter((key) => !allowed.has(key)).sort();
    if (unknown.length) throw new ValueError(`Unknown config fields: ${pythonNames(unknown)}`);
    this._configureSemantics(validated);
    this._configureDecoding(validated);
    const foundation = construction instanceof Construction ? construction.foundation : null;
    this.model = new NativeModel(validated, foundation ? { model: foundation } : {});
    this.config = validated;
    this._owned = true;
    this._objective = new TextObjective(this as OwnedTextOperation<unknown>);
  }

  /** @internal */
  protected _configureSemantics(config: Record<string, unknown>): void {
    const instructions = config.instructions ?? null;
    if (instructions !== null && typeof instructions !== 'string') throw new TypeError('instructions must be a string or None');
    this.instructions = instructions;
  }

  /** @internal */
  protected _configureDecoding(config: Record<string, unknown>): void {
    void config;
  }

  /** The provider request for ``value`` (subclasses add schemas). */
  abstract _request(value: readonly Message[], context: Context | null): ModelRequest;

  /** Parse a structured response or target (structured subclasses). */
  _parse(value: Record<string, unknown>): unknown {
    void value;
    throw new TypeError('This operation has no structured response');
  }

  /** @internal Likelihood-decoding loss (structured subclasses). */
  protected _likelihoodLoss(value: readonly Message[], targets: unknown, context: Context | null): Tensor {
    void value; void targets; void context;
    throw new ValueError('likelihood decoding is unsupported by this operation');
  }

  /** Explicitly wrap an external provider; it supports no owned training or artifacts. */
  static fromModel<T>(this: OwnedTextOperationClass<T>, model: ExternalModel, options: Record<string, unknown> = {}): T {
    if (model === null || (typeof model !== 'object' && typeof model !== 'function')) {
      throw new TypeError('model must be a Model or AsyncModel object or a messages → text function');
    }
    const supplied = Object.fromEntries(Object.entries(options ?? {}).filter(([, value]) => value !== undefined));
    if (Object.keys(supplied).some((key) => !this.semanticFields.has(key))) throw new ValueError('Unknown external model options');
    return new (this as unknown as new (config: unknown, construction: unknown) => T)(null, new Construction({ model, options: supplied }, null));
  }

  /**
   * Explicitly import native weights and tokenizer assets from a local
   * directory or a pinned Hub repository. ``config`` may set semantic,
   * decoding and ``generation`` fields only.
   */
  static async fromFoundation<T>(
    this: OwnedTextOperationClass<T>, repo: string, options: FromFoundationOptions = {},
  ): Promise<T> {
    const { config, revision, ...hub } = options;
    const settings = validatedOwnedConfig(config ?? {});
    const allowed = new Set([...this.semanticFields, ...this.decodingFields, 'generation']);
    if (Object.keys(settings).some((key) => !allowed.has(key))) {
      throw new ValueError('foundation config supports semantic, decoding and generation fields only');
    }
    const loaded = await loadNativeFoundation(repo, {
      ...hub, revision: revision ?? null, head: 'seq2seq', restoreRawTieFlags: true,
    });
    if (loaded.tokenizer === null) throw new ValueError('foundation has no tokenizer.json; a fast tokenizer is required');
    settings.native_config = loaded.config.toDiffDict();
    settings.tokenizer = loaded.tokenizer.configuration() as unknown as JsonObject;
    settings.native_parameter_aliases = parameterAliases(loaded.model);
    settings.native_generation_config = loaded.generationConfig ?? {};
    settings.foundation = { repo: String(repo), revision: revision ?? null };
    const result = new (this as unknown as new (config: unknown, construction: unknown) => T)(
      settings, new Construction(null, loaded.model as T5ForConditionalGeneration),
    );
    (result as unknown as OwnedTextOperation<unknown>).eval();
    return result;
  }

  /** Persisted identity of this concrete class (Python qualified name). */
  static toolIdentity(): string {
    return qualifiedName(this);
  }

  /** Owned generation is replayable unless it samples. */
  override get replayable(): boolean {
    return this._owned && !((this.model as NativeModel).generation.do_sample ?? false);
  }

  /** @internal */
  _requireOwned(): NativeModel {
    if (!this._owned) throw new ValueError('external models do not support owned training or artifacts');
    return this.model as NativeModel;
  }

  /** The owned native model (``op.model`` narrowed; throws for external models). */
  get nativeModel(): NativeModel {
    return this._requireOwned();
  }

  /** The declared training objective (owned operations only). */
  get trainingOperation(): TextObjective {
    this._requireOwned();
    return this._objective!;
  }

  configuration(): JsonObject {
    const model = this._requireOwned();
    return validatedOwnedConfig({ ...deepCopy(this.config!), ...model.configuration() });
  }

  /**
   * Teacher-forced loss of explicit ``targets``: the complete structured
   * response mapping for structured operations, a string for ``Transform``.
   */
  loss(value: unknown, targets: unknown, options: { context?: Context | null } = {}): Tensor {
    const model = this._requireOwned();
    const context = options.context ?? null;
    if ((this as { decoding?: string }).decoding === 'likelihood') {
      return this._likelihoodLoss(value as readonly Message[], targets, context);
    }
    const request = this._request(value as readonly Message[], context);
    let target: string;
    if (request.responseSchema !== null) {
      if (!isPlainObject(targets)) throw new ValueError('structured targets must be an explicit JSON mapping');
      validateStructured(targets, request.responseSchema);
      this._parse(targets);
      target = sortedJsonDumps(targets, { allowNan: false });
    } else {
      if (typeof targets !== 'string') throw new ValueError('text targets must be a string');
      target = targets;
    }
    return model.loss(request, target);
  }

  operationBindings(): Record<string, Operation> {
    this._requireOwned();
    return { operation: this as Operation, objective: this._objective! };
  }

  parameters(): Parameter[] {
    return this._requireOwned().parameters();
  }

  namedParameters(options: { prefix?: string; removeDuplicate?: boolean } = {}): [string, Parameter][] {
    return this._requireOwned().namedParameters(options);
  }

  namedModules(options: { prefix?: string; removeDuplicate?: boolean } = {}): [string, Module][] {
    return this._requireOwned().namedModules(options);
  }

  stateDict(): StateDict {
    return this._requireOwned().stateDict();
  }

  loadStateDict(state: StateDict | Record<string, Tensor>, options: { strict?: boolean } = {}): LoadStateDictResult {
    return this._requireOwned().loadStateDict(state, options);
  }

  get training(): boolean {
    return this._owned ? (this.model as NativeModel).training : false;
  }

  train(mode = true): this {
    this._requireOwned().train(mode);
    return this;
  }

  eval(): this {
    return this.train(false);
  }

  /** Only ``'cpu'`` is supported. */
  to(device: string): this {
    this._requireOwned();
    if (device !== 'cpu') throw new ValueError(`device ${JSON.stringify(device)} is unsupported; the TypeScript core computes on the CPU`);
    return this;
  }

  defaultModelCard(): string {
    const cls = this.constructor as unknown as OwnedTextOperationClass<this> & { toolIdentity(): string };
    return defaultModelCard(cls.toolIdentity(), cls.name, 'tensorcode/ops/text');
  }

  /** Stage a complete artifact (configuration with the embedded tokenizer, and weights). */
  async savePretrained(directory: string): Promise<string> {
    const model = this._requireOwned();
    const cls = this.constructor as unknown as OwnedTextOperationClass<this> & { toolIdentity(): string };
    return writePretrainedArtifact({
      stateModule: model,
      manifest: {
        format: cls.artifactFormat, version: cls.artifactVersion, identityKey: 'tool',
        identity: cls.toolIdentity(), config: this.configuration(),
      },
      modelCard: () => this.defaultModelCard(),
    }, directory);
  }

  /** Load this known class from a local directory or a pinned Hub snapshot. */
  static async fromPretrained<T extends OwnedTextOperation<unknown>>(
    this: OwnedTextOperationClass<T>, source: string, options: FromPretrainedOptions = {},
  ): Promise<T> {
    const cls = this;
    const { model } = await readPretrainedArtifact<T>({
      format: cls.artifactFormat,
      version: cls.artifactVersion,
      identityKey: 'tool',
      identity: qualifiedName(cls),
      construct: (config) => new cls(config),
      configuration: (operation) => operation.configuration(),
      stateModule: (operation) => operation._requireOwned(),
    }, source, options);
    return model.eval();
  }

  /** Explicitly publish the owned artifact. */
  async pushToHub(repoId: string, options: PushToHubOptions = {}): Promise<unknown> {
    this._requireOwned();
    return publishArtifact((directory) => this.savePretrained(directory), repoId, options);
  }
}
