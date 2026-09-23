/**
 * Data-configured tensor readouts and native transformer latent bridges
 * (Python ``tensorcode/_internal/vec/owned.py``).
 *
 * ``OwnedMap`` is the shared architecture of the public ``Transform``,
 * ``Classify`` and ``Decode`` operations (and ``Score``'s pair readout). The
 * public subclasses define the result contracts; this module owns
 * configuration validation, the linear/MLP/native-transformer networks,
 * latent-prefix context, losses and the supervised ``Objective``.
 */
import { Tensor, tensor } from '../../nn/tensor.js';
import { GELU, Linear, Sequential, type TensorModule } from '../../nn/layers.js';
import { cat } from '../../nn/ops/shape.js';
import { crossEntropy } from '../../nn/ops/nn.js';
import { ValueError } from '../../errors.js';
import { Operation, type Context, type OperationLike } from '../../ops/base.js';
import { Latent, Space } from '../../ops/vec/latent.js';
import { LatentOperation, asSequence } from '../latentOps.js';
import { parseObjectiveEnvelope } from '../contracts.js';
import { qualifiedName } from '../identity.js';
import { isPlainObject, type JsonObject, type JsonValue } from '../json.js';
import { nativeConfig } from '../native/config.js';
import { createNativeModel } from '../native/registry.js';
import { loadNativeFoundation } from '../native/foundation.js';
import type { EncoderOutput, NativeEncoder } from '../native/modules.js';
import type { NativeModel } from '../native/modules.js';
import type { Parameter } from '../../nn/tensor.js';
import { TensorAdapter, type Combine, type ForwardModule } from './adapter.js';
import { callableIdentity, moduleConfiguration, spaceConfiguration } from './configuration.js';

/** Python ``repr`` of a sorted list of strings (``['a', 'b']``). */
export function pythonList(values: readonly string[]): string {
  return `[${values.map((value) => `'${value}'`).join(', ')}]`;
}

/** Keys of ``config`` outside ``allowed``, sorted. */
export function unknownKeys(config: Record<string, unknown>, allowed: Iterable<string>): string[] {
  const known = new Set(allowed);
  return Object.keys(config).filter((key) => !known.has(key)).sort();
}

/** Python ``positive``: a positive integer (booleans rejected). */
export function positive(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
    throw new ValueError(`${name} must be a positive integer`);
  }
  return value;
}

/** A module whose ``forward`` maps one tensor to one tensor. */
export type MapModule = TensorModule;

/** Python ``network``: a single ``Linear`` or ``Linear``/``GELU`` stack. */
export function network(inputDim: number, outputDim: number, architecture: string, config: JsonObject): MapModule {
  const hidden = config.hidden_dimensions ?? [];
  if (!Array.isArray(hidden)) throw new ValueError('hidden_dimensions must be a list');
  if (architecture === 'linear' && hidden.length) throw new ValueError('linear architecture does not use hidden_dimensions');
  if (architecture === 'mlp' && !hidden.length) throw new ValueError('mlp requires nonempty hidden_dimensions');
  const widths = [inputDim, ...hidden.map((value) => positive(value, 'hidden dimension')), outputDim];
  const layers: MapModule[] = [];
  for (let index = 0; index < widths.length - 1; index += 1) {
    layers.push(new Linear(widths[index]!, widths[index + 1]!));
    if (index < widths.length - 2) layers.push(new GELU());
  }
  return layers.length === 1 ? layers[0]! : new Sequential(...layers);
}

/** Owners trained through {@link Objective}. */
export interface ObjectiveOwner {
  loss(value: unknown, targets: unknown, options?: { context?: Context | null }): Tensor;
  configuration(): JsonObject;
  parameters(): Parameter[];
}

/**
 * Supervised objective of an owned operation: ``{inputs, targets}`` (with an
 * optional ``{value, context}`` conditioning envelope) → scalar loss.
 */
export class Objective extends Operation<Record<string, unknown>, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.owned.Objective';
  readonly #owner: ObjectiveOwner;

  constructor(owner: ObjectiveOwner) {
    super();
    this.#owner = owner;
  }

  override get replayable(): boolean {
    return true;
  }

  get owner(): ObjectiveOwner {
    return this.#owner;
  }

  /** Persisted identity: the owner's public identity plus ``.objective``. */
  operationIdentity(): string {
    return `${qualifiedName(this.#owner)}.objective`;
  }

  parameters(): Parameter[] {
    return this.#owner.parameters();
  }

  configuration(): JsonObject {
    return { operation: this.operationIdentity(), model: this.#owner.configuration() };
  }

  forward(value: Record<string, unknown>, context: Context | null): Tensor {
    const envelope = parseObjectiveEnvelope(value, context);
    return this.#owner.loss(envelope.inputs, envelope.targets, { context: envelope.context }).clone();
  }
}

export type OwnedKind = 'transform' | 'classify' | 'decode';
export type Readout = 'sequence' | 'pooled';

/** Internal construction channel (Python ``cls.__new__`` factories). */
const INTERNAL: unique symbol = Symbol('tensorcode.vec.owned.internal');

export interface OwnedInternals {
  readonly [INTERNAL]: true;
  /** ``fromModule``: supplied module mode. */
  readonly supplied?: {
    adapter: TensorAdapter;
    labels?: readonly string[];
    output?: string;
  };
  /** ``fromFoundation``: the loaded backbone replaces the randomly initialized one. */
  readonly nativeModel?: NativeModel;
}

export function ownedInternals(values: Omit<OwnedInternals, typeof INTERNAL>): OwnedInternals {
  return { [INTERNAL]: true, ...values };
}

export interface FromModuleOptions {
  combine?: Combine | null;
  inputSpace?: Space | null;
  outputSpace?: Space | null;
  labels?: readonly string[];
  output?: string;
}

export interface OwnedFoundationOptions {
  inputSpace: Space | JsonObject;
  revision?: string | null;
  outputSpace?: Space | JsonObject;
  labels?: readonly string[];
  outputDimensions?: number;
  output?: string;
  readout?: Readout;
  localFilesOnly?: boolean;
  cacheDir?: string | null;
  token?: string | null;
  endpoint?: string | null;
  trustRemoteCode?: boolean;
  useSafetensors?: boolean;
}

type OwnedClass<T> = (new (config: unknown, internals?: OwnedInternals) => T) & { readonly kind: OwnedKind };

/**
 * Shared owned architecture: ``linear``, ``mlp`` or a native ``transformer``
 * (bert/roberta/distilbert over ``inputs_embeds``). Public subclasses define
 * their result contracts.
 */
export abstract class OwnedMap<O = unknown> extends LatentOperation<unknown, O> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.owned.OwnedMap';
  static readonly kind: OwnedKind = 'transform';

  inputSpace: Space | null = null;
  outputSpace: Space | null = null;
  labels: readonly string[] = Object.freeze([]);
  output: string | null = null;
  readout: Readout = 'sequence';
  /** Native transformer backbone (``architecture: 'transformer'``). */
  model: (NativeModel & NativeEncoder) | null = null;
  inputProjection: Linear | null = null;
  /** Readout network, or the supplied module in ``fromModule`` mode. */
  module!: ForwardModule;
  readonly #supplied: boolean;
  readonly #adapter: TensorAdapter | null;
  readonly #objective: Objective;

  constructor(config: unknown, internals?: OwnedInternals) {
    const supplied = internals?.[INTERNAL] ? internals.supplied : undefined;
    super(supplied ? {} : config);
    const kind = this.kind;
    this.#objective = new Objective(this);
    if (supplied) {
      this.#supplied = true;
      this.#adapter = supplied.adapter;
      this.module = this.registerModule('module', supplied.adapter.module);
      this.inputSpace = supplied.adapter.inputSpace;
      this.outputSpace = supplied.adapter.outputSpace;
      if (kind === 'classify') this.labels = Object.freeze([...(supplied.labels ?? [])]);
      if (kind === 'decode') this.output = supplied.output ?? null;
      return;
    }
    this.#supplied = false;
    this.#adapter = null;
    const cfg = this.config;
    const common = ['architecture', 'input_space', 'hidden_dimensions', 'native_config', 'readout', 'foundation'];
    const extra = { transform: ['output_space'], classify: ['labels'], decode: ['output_dimensions', 'output'] }[kind];
    const unknown = unknownKeys(cfg, [...common, ...extra]);
    if (unknown.length) throw new ValueError(`unknown configuration fields: ${pythonList(unknown)}`);
    this.inputSpace = Space.fromConfig(cfg.input_space);
    cfg.input_space = this.inputSpace.configuration() as unknown as JsonObject;
    let width: number;
    if (kind === 'transform') {
      this.outputSpace = Space.fromConfig(cfg.output_space);
      cfg.output_space = this.outputSpace.configuration() as unknown as JsonObject;
      width = this.outputSpace.dimensions;
    } else if (kind === 'classify') {
      const labels = cfg.labels;
      if (!Array.isArray(labels) || !labels.length || !labels.every((item) => typeof item === 'string' && item)
        || new Set(labels).size !== labels.length) {
        throw new ValueError('labels must be nonempty unique strings');
      }
      this.labels = Object.freeze([...(labels as string[])]);
      width = labels.length;
    } else {
      width = positive(cfg.output_dimensions, 'output_dimensions');
      const output = cfg.output;
      if (typeof output !== 'string' || !output.trim()) throw new ValueError('output must be a nonempty description');
      this.output = output;
    }
    const architecture = cfg.architecture ?? 'linear';
    cfg.architecture = architecture;
    const readout = cfg.readout ?? (kind === 'transform' ? 'sequence' : 'pooled');
    if (readout !== 'sequence' && readout !== 'pooled') throw new ValueError('readout must be sequence or pooled');
    if (kind === 'classify' && readout !== 'pooled') throw new ValueError('classification requires pooled readout');
    this.readout = readout;
    cfg.readout = readout;
    if (kind === 'transform') {
      const organization = readout === 'sequence' ? this.inputSpace.organization : 'feature';
      if (this.outputSpace!.organization !== organization) throw new ValueError('output_space organization must match readout');
    }
    if (architecture === 'linear' || architecture === 'mlp') {
      if ('native_config' in cfg || 'foundation' in cfg) {
        throw new ValueError('native_config and foundation require transformer architecture');
      }
      this.module = this.registerModule('module', network(this.inputSpace.dimensions, width, architecture, cfg));
    } else if (architecture === 'transformer') {
      if ('hidden_dimensions' in cfg) throw new ValueError('transformer uses native_config, not hidden_dimensions');
      const native = nativeConfig(cfg.native_config);
      // Encoder-only models expose stable inputs_embeds/last_hidden_state.
      if (!['bert', 'roberta', 'distilbert'].includes(native.modelType)) {
        throw new ValueError('supported native transformer architectures: bert, roberta, distilbert');
      }
      const model = internals?.[INTERNAL] && internals.nativeModel
        ? internals.nativeModel
        : createNativeModel(native, 'base');
      this.model = this.registerModule('model', model as NativeModel & NativeEncoder);
      cfg.native_config = native.toDiffDict();
      this.inputProjection = this.registerModule('input_projection', new Linear(this.inputSpace.dimensions, native.hiddenSize));
      this.module = this.registerModule('module', new Linear(native.hiddenSize, width));
    } else {
      throw new ValueError('architecture must be linear, mlp or transformer');
    }
  }

  /** ``transform``, ``classify`` or ``decode`` (the class's result contract). */
  get kind(): OwnedKind {
    return (this.constructor as typeof OwnedMap).kind;
  }

  /** True for operations built by {@link OwnedMap.fromModule}. */
  get supplied(): boolean {
    return this.#supplied;
  }

  get trainingInputsIncludeTargets(): boolean {
    return true;
  }

  /** Objective operation used for supervised training. */
  get objective(): Objective {
    return this.#objective;
  }

  get trainingOperation(): Objective {
    return this.#objective;
  }

  /**
   * Advanced: wrap a supplied module (Python ``from_module``). Supplied modules
   * have no declarative reconstruction, so ``savePretrained`` is rejected.
   */
  static fromModule<T extends OwnedMap>(this: OwnedClass<T>, module: ForwardModule, options: FromModuleOptions = {}): T {
    const known = ['combine', 'inputSpace', 'outputSpace', ...(this.kind === 'classify' ? ['labels'] : []), ...(this.kind === 'decode' ? ['output'] : [])];
    const extra = Object.keys(options).filter((key) => !known.includes(key)).sort();
    if (extra.length) throw new TypeError(`unsupported fromModule arguments: ${pythonList(extra)}`);
    const adapter = new TensorAdapter(module, {
      combine: options.combine ?? null, inputSpace: options.inputSpace ?? null, outputSpace: options.outputSpace ?? null,
    });
    let labels: readonly string[] | undefined;
    let output: string | undefined;
    if (this.kind === 'classify') {
      const supplied = options.labels;
      if (!supplied || !supplied.length || !supplied.every((item) => typeof item === 'string' && item) || new Set(supplied).size !== supplied.length) {
        throw new ValueError('labels must be nonempty and unique');
      }
      labels = [...supplied];
    }
    if (this.kind === 'decode') {
      if (typeof options.output !== 'string' || !options.output.trim()) throw new ValueError('output description must be nonempty');
      output = options.output;
    }
    return new this({}, ownedInternals({ supplied: { adapter, ...(labels ? { labels } : {}), ...(output !== undefined ? { output } : {}) } }));
  }

  /**
   * Load a bert/roberta/distilbert backbone (Python ``from_foundation``); the
   * input bridge and output head start untrained.
   */
  static async fromFoundation<T extends OwnedMap>(this: OwnedClass<T>, repo: string, options: OwnedFoundationOptions): Promise<T> {
    const { inputSpace, revision = null, outputSpace, labels, outputDimensions, output, readout, trustRemoteCode, useSafetensors, ...hub } = options;
    if (trustRemoteCode || (useSafetensors !== undefined && useSafetensors !== true)) {
      throw new ValueError('foundation requires native code and safetensors');
    }
    const loaded = await loadNativeFoundation(repo, { ...hub, revision, head: 'base', tokenizer: false });
    const config: JsonObject = {};
    if (outputSpace !== undefined) config.output_space = outputSpace instanceof Space ? outputSpace.configuration() as unknown as JsonObject : outputSpace;
    if (labels !== undefined) config.labels = [...labels];
    if (outputDimensions !== undefined) config.output_dimensions = outputDimensions;
    if (output !== undefined) config.output = output;
    if (readout !== undefined) config.readout = readout;
    Object.assign(config, {
      architecture: 'transformer',
      input_space: inputSpace instanceof Space ? inputSpace.configuration() as unknown as JsonObject : inputSpace,
      native_config: loaded.config.toDiffDict(),
      foundation: { repo: String(repo), revision, input_bridge: 'untrained', output_head: 'untrained' },
    });
    return new this(config, ownedInternals({ nativeModel: loaded.model })).eval();
  }

  override configuration(): JsonObject {
    if (this.#supplied) {
      const adapter = this.#adapter!;
      const config: JsonObject = {
        operation: qualifiedName(this),
        module: moduleConfiguration(this.module),
        combine: callableIdentity(adapter.combine),
        input_space: spaceConfiguration(adapter.inputSpace),
        output_space: spaceConfiguration(adapter.outputSpace),
      };
      if (this.kind === 'classify') config.labels = [...this.labels];
      if (this.kind === 'decode') config.output = this.output;
      return config;
    }
    return super.configuration();
  }

  override operationBindings(): Record<string, OperationLike> {
    return { ...super.operationBindings(), objective: this.#objective };
  }

  /** Save configuration and weights; rejected for ``fromModule`` operations. */
  override async savePretrained(directory: string): Promise<string> {
    if (this.#supplied) throw new ValueError('supplied modules have no declarative reconstruction; cannot save_pretrained');
    return super.savePretrained(directory);
  }

  /** Readout tensor (logits/values/features) before the public result contract. */
  protected tensorOutput(value: unknown, context: Context | null): unknown {
    if (this.#supplied) return this.#adapter!.forward(value, context);
    const ctx = context ?? {};
    if (!isPlainObject(ctx) || Object.keys(ctx).some((key) => key !== 'latents')) throw new ValueError('context supports only latents');
    const prefixes = ctx.latents ?? [];
    if (!Array.isArray(prefixes)) throw new ValueError('context latents must be an ordered list');
    const [x, mask] = asSequence(value, this.inputSpace!);
    const originalShape = (value as Latent).tensor.shape;
    let hidden: Tensor;
    if (this.config.architecture === 'transformer') {
      const pairs = prefixes.map((item) => asSequence(item, this.inputSpace!));
      if (pairs.some(([t]) => t.shape[0] !== x.shape[0] || t.dtype !== x.dtype)) {
        throw new ValueError('context latents must have matching batch, device and dtype');
      }
      const count = pairs.reduce((total, [t]) => total + t.shape[1]!, 0);
      const inputs = cat([...pairs.map(([t]) => t), x], 1);
      const attention = cat([...pairs.map(([, m]) => m), mask], 1);
      const output: EncoderOutput = this.model!.forward({ inputsEmbeds: this.inputProjection!.forward(inputs), attentionMask: attention });
      hidden = output.lastHiddenState.slice(1, count);
    } else {
      if (prefixes.length) throw new ValueError('latent context requires transformer architecture');
      hidden = x;
    }
    const invalid = mask.logicalNot().unsqueeze(-1);
    if (this.readout === 'pooled') {
      const pooled = hidden.maskedFill(invalid, 0).sum(1).div(mask.to(hidden.dtype).sum(1, true));
      const result = (this.module as MapModule).forward(pooled);
      const organization = this.inputSpace!.organization;
      const ndim = (value as Latent).tensor.ndim;
      const single = (organization === 'feature' && ndim === 1) || (organization === 'sequence' && ndim === 2);
      return single ? result.select(0, 0) : result;
    }
    const result = (this.module as MapModule).forward(hidden).maskedFill(invalid, 0);
    return result.reshape(...originalShape.slice(0, -1), result.shape[result.ndim - 1]!);
  }

  /** Map a {@link Latent} (optionally with ``context: {latents: [...]}``). */
  forward(value: unknown, context: Context | null): O {
    const result = this.tensorOutput(value, context);
    if (this.#supplied) return result as O;
    if (this.kind === 'transform') {
      const source = value as Latent;
      const sequence = this.readout === 'sequence';
      return source.withTensor(result as Tensor, {
        space: this.outputSpace!, mask: sequence ? source.mask : null, coordinates: sequence ? source.coordinates : null,
      }) as O;
    }
    return result as O;
  }

  /** Cross-entropy for labels, otherwise masked MSE against target tensors. */
  loss(value: unknown, targets: unknown, options: { context?: Context | null } = {}): Tensor {
    const context = options.context ?? null;
    const result = this.forward(value, context) as unknown;
    if (this.kind === 'classify') {
      const logits = (result as { logits: Tensor }).logits;
      let target = targets;
      if (typeof target === 'string') target = [target];
      if (Array.isArray(target) && target.every((item) => typeof item === 'string')) {
        const indices = (target as string[]).map((label) => {
          const index = this.labels.indexOf(label);
          if (index < 0) throw new ValueError(`${JSON.stringify(label)} is not a configured label`);
          return index;
        });
        target = tensor(indices, { dtype: 'int64' });
        if (logits.ndim === 1) target = (target as Tensor).squeeze(0);
      }
      if (!(target instanceof Tensor) || target.dtype !== 'int64'
        || target.shape.length !== logits.ndim - 1 || target.shape.some((size, index) => size !== logits.shape[index])) {
        throw new ValueError('classification targets must be long indices matching the batch');
      }
      return crossEntropy(logits, target);
    }
    let output = result instanceof Latent ? result.tensor : result as Tensor;
    let target: unknown;
    if (targets instanceof Latent) {
      if (!(result instanceof Latent) || !targets.space.equals(result.space)) throw new ValueError('target space must match output space');
      target = targets.tensor;
    } else {
      target = targets;
    }
    if (!(target instanceof Tensor) || !(output instanceof Tensor) || target.shape.length !== output.shape.length
      || target.shape.some((size, index) => size !== output.shape[index])) {
      throw new ValueError('targets must match output shape');
    }
    let mask: Tensor | null = result instanceof Latent ? result.mask
      : (!this.#supplied && this.readout === 'sequence' && value instanceof Latent ? value.mask : null);
    if (targets instanceof Latent && targets.mask !== null) {
      if (targets.mask.dtype !== 'bool') throw new ValueError('target mask must be boolean');
      mask = mask === null ? targets.mask : mask.logicalAnd(targets.mask);
    }
    let expected = target.dtype === output.dtype ? target : target.to(output.dtype);
    if (mask !== null) {
      if (!mask.any().item()) throw new ValueError('loss requires valid target positions');
      output = output.maskedSelect(mask);
      expected = expected.maskedSelect(mask);
    }
    if (!expected.allFinite()) throw new ValueError('valid targets must be finite');
    return output.sub(expected).square().mean();
  }
}

/** JSON configuration of a space-like option. */
export function spaceJson(value: Space | JsonObject | null | undefined): JsonValue {
  if (value === null || value === undefined) return null;
  return value instanceof Space ? value.configuration() as unknown as JsonObject : value;
}

