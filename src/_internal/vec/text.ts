/**
 * Owned native text transformers with explicit latent-space bridges (Python
 * ``tensorcode/_internal/vec/text.py``).
 *
 * Encoder and decoder context is an ordered list of latent prefixes; targets
 * enter only ``loss``. Configuration embeds the complete fast tokenizer and
 * native architecture, so artifacts reconstruct offline.
 */
import type { DType } from '../../nn/dtype.js';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Module } from '../../nn/module.js';
import { Parameter, Tensor, ones, tensor, zeros } from '../../nn/tensor.js';
import { normal_ } from '../../nn/init.js';
import { noGrad } from '../../nn/autograd.js';
import { Identity, Linear } from '../../nn/layers.js';
import { cat, padSequence } from '../../nn/ops/shape.js';
import { ValueError } from '../../errors.js';
import { Operation, type Context, type OperationLike } from '../../ops/base.js';
import { Latent, Space } from '../../ops/vec/latent.js';
import { LatentOperation, asSequence } from '../latentOps.js';
import { parseObjectiveEnvelope } from '../contracts.js';
import { qualifiedName } from '../identity.js';
import { resolveArtifactDirectory } from '../hub.js';
import { deepCopy, isPlainObject, mergeJson, parseJsonStrict, type JsonObject, type JsonValue } from '../json.js';
import {
  NativeConfig, nativeConfig, generationConfigFromFile, generationConfigFromModel,
} from '../native/config.js';
import { createNativeModel } from '../native/registry.js';
import { loadNativeFoundation, type LoadedFoundation } from '../native/foundation.js';
import { parameterAliases, restoreParameterAliases, type NativeEncoder, type NativeModel } from '../native/modules.js';
import { GENERATION_KEYS, generateSeq2Seq, type GenerationSettings } from '../native/generation.js';
import type { T5ForConditionalGeneration } from '../native/t5.js';
import { FastTokenizer } from '../tokenizers/index.js';
import type { Parameter as ParameterType } from '../../nn/tensor.js';
import { pythonList, spaceJson, unknownKeys } from './owned.js';

/** Hidden width of a native configuration (Python ``_width``). */
export function nativeWidth(config: NativeConfig): number {
  return config.hiddenSize;
}

/** Python ``_context``: the ordered list stored under ``key`` (``[]`` when absent). */
export function contextList(context: Context | null, key: string): unknown[] {
  if (context === null || context === undefined) return [];
  if (!isPlainObject(context) || Object.keys(context).some((name) => name !== key)) {
    throw new ValueError(`context supports only '${key}'`);
  }
  const values = context[key] ?? [];
  if (!Array.isArray(values)) throw new ValueError(`context['${key}'] must be an ordered list`);
  return values;
}

function textBatch(value: unknown): string[] {
  const texts = typeof value === 'string' ? [value] : value;
  if (!Array.isArray(texts) || !texts.length || !texts.every((text) => typeof text === 'string')) {
    throw new ValueError('text input must be a string or nonempty list of strings');
  }
  return [...texts];
}

export interface TextFoundationHubOptions {
  revision?: string | null;
  localFilesOnly?: boolean;
  cacheDir?: string | null;
  token?: string | null;
  endpoint?: string | null;
  trustRemoteCode?: boolean;
  useSafetensors?: boolean;
  /** Parameter dtype (transformers ``dtype``; default ``'auto'``: the checkpoint's dtype). */
  dtype?: DType | 'auto';
}

function nativeDtype(model: NativeModel): DType {
  return model.parameters()[0]?.dtype ?? 'float32';
}

/** Python ``_load_foundation``: native weights, raw tie flags and the fast tokenizer. */
async function loadFoundation(repo: string, options: TextFoundationHubOptions, decoder: boolean): Promise<LoadedFoundation & { tokenizer: FastTokenizer }> {
  const { trustRemoteCode, useSafetensors, revision = null, ...hub } = options;
  if (trustRemoteCode || (useSafetensors !== undefined && useSafetensors !== true)) {
    throw new ValueError('foundation loading requires native code and safetensors');
  }
  let seq2seq = decoder;
  if (!seq2seq) {
    const { path } = await resolveArtifactDirectory(repo, { ...hub, revision, allowPatterns: ['config.json'] });
    const raw = parseJsonStrict(await readFile(join(path, 'config.json'), 'utf8'));
    seq2seq = NativeConfig.fromPretrainedDict(raw, repo).isEncoderDecoder;
  }
  const loaded = await loadNativeFoundation(repo, { ...hub, revision, head: seq2seq ? 'seq2seq' : 'base', restoreRawTieFlags: true });
  if (!loaded.tokenizer) throw new ValueError('a fast tokenizer is required for complete offline artifacts');
  return loaded as LoadedFoundation & { tokenizer: FastTokenizer };
}

type NativeTextModel = NativeModel & { getEncoder?(): NativeEncoder };

function encoderOf(model: NativeTextModel): NativeEncoder {
  return model.config.isEncoderDecoder ? (model as unknown as T5ForConditionalGeneration).getEncoder() : model as unknown as NativeEncoder;
}

const TEXT_INTERNAL: unique symbol = Symbol('tensorcode.vec.text.internal');

interface TextInternals {
  readonly [TEXT_INTERNAL]: true;
  readonly model: NativeModel;
}

export type TextReadout = 'sequence' | 'pooled' | 'output_encoding';

export interface TextEncoderFoundationOptions extends TextFoundationHubOptions {
  readout?: TextReadout;
  outputSpace?: Space | JsonObject | null;
  contextSpace?: Space | JsonObject | null;
}

/**
 * Native states, masked mean, or an owned appended OUTPUT_ENCODING token.
 * The new token starts untrained even with inherited foundation weights;
 * matching widths do not imply a shared semantic space.
 */
export class TextEncoder extends LatentOperation<string | readonly string[], Latent> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.text.TextEncoder';
  readonly model: NativeTextModel;
  readonly tokenizer: FastTokenizer;
  readonly readout: TextReadout;
  readonly outputEncoding: Parameter | null = null;
  readonly outputSpace: Space;
  readonly contextSpace: Space | null;

  constructor(config: unknown, internals?: TextInternals) {
    super(config);
    const cfg = this.config;
    const unknown = unknownKeys(cfg, ['native_config', 'native_parameter_aliases', 'tokenizer', 'readout', 'output_space', 'context_space', 'foundation']);
    if (unknown.length) throw new ValueError(`Unknown configuration fields: ${pythonList(unknown)}; use output_space and readout`);
    const native = nativeConfig(cfg.native_config);
    const model = internals?.[TEXT_INTERNAL] ? internals.model : createNativeModel(native, native.isEncoderDecoder ? 'seq2seq' : 'base');
    this.model = this.registerModule('model', model as NativeTextModel);
    if (!internals?.[TEXT_INTERNAL]) restoreParameterAliases(this.model, cfg.native_parameter_aliases as Record<string, string> | undefined);
    if (!isPlainObject(cfg.tokenizer)) throw new ValueError('tokenizer configuration must be an object');
    this.tokenizer = FastTokenizer.fromConfiguration(cfg.tokenizer);
    const readout = cfg.readout ?? 'sequence';
    if (readout !== 'sequence' && readout !== 'pooled' && readout !== 'output_encoding') {
      throw new ValueError('readout must be sequence, pooled or output_encoding');
    }
    this.readout = readout;
    if (readout === 'output_encoding') {
      const width = this.model.getInputEmbeddings().weight.shape[1]!;
      // ``nn.init.normal_(torch.empty(1, 1, width), std=0.02)``
      this.outputEncoding = this.registerParameter('output_encoding', new Parameter(normal_(zeros([1, 1, width]), 0, 0.02)));
    }
    this.outputSpace = Space.fromConfig(cfg.output_space);
    if (this.outputSpace.dimensions !== nativeWidth(native) || this.outputSpace.organization !== (readout === 'sequence' ? 'sequence' : 'feature')) {
      throw new ValueError('output space must match native width and readout organization');
    }
    this.contextSpace = cfg.context_space ? Space.fromConfig(cfg.context_space) : null;
    const encoder = encoderOf(this.model);
    if (this.contextSpace && this.contextSpace.dimensions !== encoder.getInputEmbeddings().weight.shape[1]) {
      throw new ValueError('context_space must match native input embedding width');
    }
  }

  static async fromFoundation<T extends TextEncoder>(
    this: new (config: unknown, internals?: TextInternals) => T, repo: string, options: TextEncoderFoundationOptions = {},
  ): Promise<T> {
    const { readout = 'sequence', outputSpace = null, contextSpace = null, ...hub } = options;
    const revision = hub.revision ?? null;
    const loaded = await loadFoundation(repo, hub, false);
    const width = nativeWidth(loaded.config);
    const space = outputSpace ?? new Space(`${repo}:encoder:${readout}`, width, {
      version: revision ?? 'unversioned', organization: readout === 'sequence' ? 'sequence' : 'feature',
    });
    const config: JsonObject = {
      native_config: loaded.config.toDiffDict(),
      native_parameter_aliases: parameterAliases(loaded.model),
      tokenizer: loaded.tokenizer.configuration() as unknown as JsonObject,
      readout,
      context_space: spaceJson(contextSpace),
      output_space: spaceJson(space),
      foundation: { repo: String(repo), revision },
    };
    // New owned bridges start in the native parameter dtype (Python creates them with it).
    return new this(config, { [TEXT_INTERNAL]: true, model: loaded.model }).to(nativeDtype(loaded.model)).eval();
  }

  override configuration(): JsonObject {
    const config = super.configuration();
    config.tokenizer = this.tokenizer.configuration() as unknown as JsonObject;
    return deepCopy(config);
  }

  private prefixPieces(prefixes: unknown[], embeds: Tensor, sources: string[]): { pieces: Tensor[]; masks: Tensor[] } {
    const pieces: Tensor[] = [];
    const masks: Tensor[] = [];
    for (const prefix of prefixes) {
      const [sequence, mask] = asSequence(prefix, this.contextSpace!);
      if (sequence.shape[0] !== embeds.shape[0]) throw new ValueError('context batch must match text batch');
      pieces.push(sequence.dtype === embeds.dtype ? sequence : sequence.to(embeds.dtype));
      masks.push(mask);
      sources.push(...(prefix as Latent).sources);
    }
    return { pieces, masks };
  }

  forward(value: string | readonly string[], context: Context | null): Latent {
    const texts = textBatch(value);
    const prefixes = contextList(context, 'latents');
    if (prefixes.length && this.contextSpace === null) throw new ValueError('context requires an explicit context_space');
    const tokens = this.tokenizer.encodeTensors(texts, { padding: true });
    const encoder = encoderOf(this.model);
    const inputIds = tokens.input_ids;
    let mask = tokens.attention_mask.bool();
    const sources: string[] = [];
    let states: Tensor;
    if (this.readout === 'output_encoding') {
      const embeds = encoder.getInputEmbeddings().forward(inputIds);
      const { pieces, masks } = this.prefixPieces(prefixes, embeds, sources);
      const combined = cat([...pieces, embeds], 1);
      const valid = cat([...masks, mask], 1);
      // Absolute readout positions must not depend on batch padding.
      const token = this.outputEncoding!.select(0, 0);
      const rows = Array.from({ length: combined.shape[0]! }, (_, row) => cat([combined.select(0, row).maskedSelect(valid.select(0, row)), token], 0));
      const lengths = rows.map((row) => row.shape[0]!);
      let limit = encoder.config.optionalNumber('max_position_embeddings');
      const positions = (encoder as unknown as { embeddings?: { position_embeddings?: { paddingIdx?: number | null } } }).embeddings?.position_embeddings;
      // RoBERTa-family positions begin after the reserved pad position.
      if (positions && typeof positions.paddingIdx === 'number' && limit !== null) limit -= positions.paddingIdx + 1;
      if (limit !== null && Math.max(...lengths) > limit) {
        throw new ValueError('input, context and OUTPUT_ENCODING exceed native position capacity');
      }
      const packed = padSequence(rows, true);
      const packedMask = lengthMask(lengths, packed.shape[1]!);
      const hidden = encoder.forward({ inputsEmbeds: packed, attentionMask: packedMask }).lastHiddenState;
      states = cat(lengths.map((length, row) => hidden.select(0, row).select(0, length - 1).unsqueeze(0)), 0);
      mask = ones([states.shape[0]!], { dtype: 'bool' });
    } else if (prefixes.length) {
      const embeds = encoder.getInputEmbeddings().forward(inputIds);
      const { pieces, masks } = this.prefixPieces(prefixes, embeds, sources);
      const combined = cat([...pieces, embeds], 1);
      const valid = cat([...masks, mask], 1);
      const rows = Array.from({ length: combined.shape[0]! }, (_, row) => combined.select(0, row).maskedSelect(valid.select(0, row)));
      const lengths = rows.map((row) => row.shape[0]!);
      if (lengths.some((length) => length === 0)) throw new ValueError('every input sequence must contain an unmasked position');
      const packed = padSequence(rows, true);
      const hidden = encoder.forward({ inputsEmbeds: packed, attentionMask: lengthMask(lengths, packed.shape[1]!) }).lastHiddenState;
      // Restore the primary text layout, retaining its original mask.
      const prefixLengths = cat(masks, 1).to('int64').sum(1);
      const positions = prefixLengths.unsqueeze(1).add(mask.to('int64').cumsum(1)).sub(1).maskedFill(mask.logicalNot(), 0);
      const width = hidden.shape[2]!;
      states = hidden.gather(1, positions.unsqueeze(-1).expand(positions.shape[0]!, positions.shape[1]!, width))
        .maskedFill(mask.logicalNot().unsqueeze(-1), 0);
    } else {
      states = encoder.forward({ inputIds, attentionMask: tokens.attention_mask }).lastHiddenState;
    }
    if (this.readout === 'pooled') {
      const weights = mask.to(states.dtype);
      states = states.mul(weights.unsqueeze(-1)).sum(1).div(weights.sum(1, true).clampMin(1));
      mask = mask.any(1);
    }
    return new Latent(states, this.outputSpace, {
      mask, sources,
      metadata: {
        representation: 'native_encoder_states', readout: this.readout,
        foundation: this.config.foundation ?? null,
        readout_initialization: this.readout === 'output_encoding' ? 'untrained' : 'native',
      },
    });
  }
}

/** ``arange(width) < lengths[:, None]`` as a boolean ``[batch, width]`` mask. */
function lengthMask(lengths: readonly number[], width: number): Tensor {
  const values: boolean[] = [];
  for (const length of lengths) for (let index = 0; index < width; index += 1) values.push(index < length);
  return tensor(values, { shape: [lengths.length, width], dtype: 'bool' });
}

/** Owners trained through {@link TextObjective}. */
interface TextObjectiveOwner {
  loss(value: unknown, targets: unknown, options?: { context?: Context | null }): Tensor;
  configuration(): JsonObject;
  parameters(): ParameterType[];
}

/** Supervised teacher-forced objective of a {@link TextDecoder} (Python ``_TextObjective``). */
export class TextObjective extends Operation<Record<string, unknown>, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.text._TextObjective';
  readonly #owner: TextObjectiveOwner;

  constructor(owner: TextObjectiveOwner) {
    super();
    this.#owner = owner;
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: Record<string, unknown>, context: Context | null): Tensor {
    const envelope = parseObjectiveEnvelope(value, context);
    return this.#owner.loss(envelope.inputs, envelope.targets, { context: envelope.context }).clone();
  }

  parameters(): ParameterType[] {
    return this.#owner.parameters();
  }

  operationIdentity(): string {
    return `${qualifiedName(this.#owner)}.objective`;
  }

  configuration(): JsonObject {
    return { operation: qualifiedName(this.#owner), role: 'objective', model: this.#owner.configuration() };
  }
}

export type Bridge = 'linear' | 'identity';

export interface TextDecoderFoundationOptions extends TextFoundationHubOptions {
  inputSpace: Space | JsonObject;
  bridge?: Bridge;
  generation?: GenerationSettings | null;
}

/**
 * Project latent sequences into native encoder *input embeddings*; the
 * foundation encoder processes that sequence before decoding. A learned bridge
 * starts untrained. Identity bridging requires the exact explicitly declared
 * native input-embedding Space, not merely equal width.
 */
export class TextDecoder extends LatentOperation<Latent, string | string[]> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.text.TextDecoder';
  readonly model: T5ForConditionalGeneration;
  readonly tokenizer: FastTokenizer;
  readonly inputSpace: Space;
  readonly nativeInputSpace: Space;
  readonly projection: Linear | Identity;
  /** Live generation settings (``max_new_tokens``, ``num_beams`` ...). */
  readonly generation: GenerationSettings;
  /** Live native ``GenerationConfig`` (``to_json_string`` form). */
  generationConfig: JsonObject;
  readonly #objective: TextObjective;

  constructor(config: unknown, internals?: TextInternals) {
    super(config);
    const cfg = this.config;
    const unknown = unknownKeys(cfg, [
      'native_config', 'native_parameter_aliases', 'native_generation_config', 'tokenizer', 'input_space',
      'native_input_space', 'bridge', 'bridge_training', 'generation', 'foundation',
    ]);
    if (unknown.length) throw new ValueError(`Unknown configuration fields: ${pythonList(unknown)}`);
    const native = nativeConfig(cfg.native_config);
    const model = internals?.[TEXT_INTERNAL] ? internals.model : createNativeModel(native, 'seq2seq');
    this.model = this.registerModule('model', model as T5ForConditionalGeneration);
    if (!internals?.[TEXT_INTERNAL]) restoreParameterAliases(this.model, cfg.native_parameter_aliases as Record<string, string> | undefined);
    this.generationConfig = 'native_generation_config' in cfg
      ? generationConfigFromFile(cfg.native_generation_config)
      : generationConfigFromModel(native);
    if (!isPlainObject(cfg.tokenizer)) throw new ValueError('tokenizer configuration must be an object');
    this.tokenizer = FastTokenizer.fromConfiguration(cfg.tokenizer);
    this.inputSpace = Space.fromConfig(cfg.input_space);
    this.nativeInputSpace = Space.fromConfig(cfg.native_input_space);
    const bridge = cfg.bridge ?? 'linear';
    if (bridge === 'identity') {
      if (!this.inputSpace.equals(this.nativeInputSpace)) throw new ValueError('identity bridge requires the explicit native input embedding space');
      this.projection = this.registerModule('projection', new Identity());
    } else if (bridge === 'linear') {
      this.projection = this.registerModule('projection', new Linear(this.inputSpace.dimensions, nativeWidth(native)));
    } else {
      throw new ValueError('bridge must be linear or identity');
    }
    const supplied = cfg.generation ?? {};
    if (!isPlainObject(supplied)) throw new ValueError('generation must be a mapping');
    this.generation = mergeJson({ max_new_tokens: 32, do_sample: false }, supplied as JsonObject) as GenerationSettings;
    if (Object.keys(this.generation).some((key) => !(GENERATION_KEYS as readonly string[]).includes(key))) {
      throw new ValueError('unsupported generation setting');
    }
    this.#objective = new TextObjective(this);
  }

  static async fromFoundation<T extends TextDecoder>(
    this: new (config: unknown, internals?: TextInternals) => T, repo: string, options: TextDecoderFoundationOptions,
  ): Promise<T> {
    const { inputSpace, bridge = 'linear', generation = null, ...hub } = options;
    const revision = hub.revision ?? null;
    const loaded = await loadFoundation(repo, hub, true);
    const nativeSpace = new Space(`${repo}:encoder:input_embeddings`, nativeWidth(loaded.config), {
      version: revision ?? 'unversioned', organization: 'sequence',
    });
    const config: JsonObject = {
      native_config: loaded.config.toDiffDict(),
      native_parameter_aliases: parameterAliases(loaded.model),
      native_generation_config: loaded.generationConfig ?? generationConfigFromModel(loaded.config),
      tokenizer: loaded.tokenizer.configuration() as unknown as JsonObject,
      input_space: spaceJson(inputSpace),
      native_input_space: nativeSpace.configuration() as unknown as JsonObject,
      bridge,
      bridge_training: bridge === 'identity' ? 'native_identity' : 'untrained',
      generation: (generation ?? { max_new_tokens: 32 }) as JsonValue,
      foundation: { repo: String(repo), revision },
    };
    // New owned bridges start in the native parameter dtype (Python creates them with it).
    return new this(config, { [TEXT_INTERNAL]: true, model: loaded.model }).to(nativeDtype(loaded.model)).eval();
  }

  /** Sampling generations are not replayable. */
  override get replayable(): boolean {
    return !this.generation.do_sample;
  }

  get trainingInputsIncludeTargets(): boolean {
    return true;
  }

  get trainingOperation(): TextObjective {
    return this.#objective;
  }

  override configuration(): JsonObject {
    const config = super.configuration();
    config.tokenizer = this.tokenizer.configuration() as unknown as JsonObject;
    config.generation = deepCopy(this.generation as JsonObject);
    config.native_generation_config = generationConfigFromFile(this.generationConfig);
    return deepCopy(config);
  }

  override operationBindings(): Record<string, OperationLike> {
    return { ...super.operationBindings(), objective: this.#objective };
  }

  /** Expose actual native token embeddings with their exact input Space. */
  embedText(value: string | readonly string[]): Latent {
    const texts = textBatch(value);
    const tokens = this.tokenizer.encodeTensors(texts, { padding: true });
    const embeddings = this.model.getEncoder().getInputEmbeddings().forward(tokens.input_ids);
    return new Latent(embeddings, this.nativeInputSpace, {
      mask: tokens.attention_mask.bool(), metadata: { representation: 'native_input_embeddings' },
    });
  }

  /** Packed, projected encoder inputs ``[embeds, mask]`` (Python ``_inputs``). */
  inputs(value: unknown, context: Context | null): [Tensor, Tensor] {
    const values = [...contextList(context, 'latents'), value];
    const pairs = values.map((item) => asSequence(item, this.inputSpace));
    if (new Set(pairs.map(([t]) => t.shape[0])).size !== 1) throw new ValueError('context latents must have the same batch size');
    const combined = cat(pairs.map(([t]) => t), 1);
    const mask = cat(pairs.map(([, m]) => m), 1);
    if (!mask.any(1).all().item()) throw new ValueError('every input sequence must contain an unmasked position');
    // Masking attention alone leaves positional gaps between valid inputs.
    // Pack before projection so masked values cannot affect its gradients.
    const rows = Array.from({ length: combined.shape[0]! }, (_, row) => combined.select(0, row).maskedSelect(mask.select(0, row)));
    const lengths = rows.map((row) => row.shape[0]!);
    let packed = padSequence(rows, true);
    const weight = this.model.shared.weight;
    if (packed.dtype !== weight.dtype) packed = packed.to(weight.dtype);
    return [(this.projection as Linear).forward(packed), lengthMask(lengths, packed.shape[1]!)];
  }

  forward(value: Latent, context: Context | null): string | string[] {
    const [embeds, mask] = this.inputs(value, context);
    const modes = new Map<Module, boolean>(this.model.modules().map((module) => [module, module.training]));
    let ids: Tensor;
    try {
      this.model.eval();
      ids = generateSeq2Seq(this.model, { inputsEmbeds: embeds, attentionMask: mask }, this.generation, { generationConfig: this.generationConfig });
    } finally {
      for (const [module, mode] of modes) module.training = mode;
    }
    const texts = this.tokenizer.batchDecode(ids, { skipSpecialTokens: true });
    return texts.length === 1 ? texts[0]! : texts;
  }

  /** Differentiable teacher forcing; target text is never encoder context. */
  loss(value: unknown, targets: unknown, options: { context?: Context | null } = {}): Tensor {
    const [embeds, mask] = this.inputs(value, options.context ?? null);
    const texts = typeof targets === 'string' ? [targets] : targets;
    if (!Array.isArray(texts) || texts.length !== embeds.shape[0] || !texts.every((text) => typeof text === 'string')) {
      throw new ValueError('targets must contain one string per input batch row');
    }
    const tokens = this.tokenizer.encodeTensors(texts as string[], { padding: true });
    const labels = tokens.input_ids.maskedFill(tokens.attention_mask.bool().logicalNot(), -100);
    return this.model.forward({ inputsEmbeds: embeds, attentionMask: mask, labels }).loss!;
  }
}

