/**
 * Native local sequence model for owned text operations (Python
 * ``tensorcode/_internal/text/native.py``). Targets only enter teacher-forced
 * labels.
 *
 * The model is synchronous. Python serializes generation and teacher-forced
 * calls with a lock because async fallbacks run them in threads; JavaScript
 * runs every call here to completion without interleaving, so temporary
 * evaluation modes are always restored before another call starts.
 */
import { Module } from '../../nn/module.js';
import { Tensor, tensor } from '../../nn/tensor.js';
import { noGrad } from '../../nn/autograd.js';
import { ValueError } from '../../errors.js';
import { isPlainObject, pythonJsonDumps, deepCopy, type JsonObject, type JsonValue } from '../json.js';
import { nativeConfig, generationConfigFromFile, generationConfigFromModel } from '../native/config.js';
import { createNativeModel } from '../native/registry.js';
import { parameterAliases, restoreParameterAliases } from '../native/modules.js';
import { T5ForConditionalGeneration } from '../native/t5.js';
import { generateSeq2Seq, GENERATION_KEYS, type GenerationSettings } from '../native/generation.js';
import { FastTokenizer } from '../tokenizers/index.js';
import { TextPart, type Message } from '../../ops/text/messages.js';
import { InvalidModelOutput, ModelOutput, type ModelRequest } from '../../ops/text/model.js';

/** Plain ``json.dumps`` spelling: no float-key rewriting inside prompts and targets. */
const NO_FLOAT_KEYS: ReadonlySet<string> = new Set();

/** ``json.dumps(value, sort_keys=True)`` exactly as Python spells it. */
export function sortedJsonDumps(value: unknown, options: { allowNan?: boolean } = {}): string {
  return pythonJsonDumps(value, { sortKeys: true, floatKeys: NO_FLOAT_KEYS, allowNan: options.allowNan ?? true });
}

/** A ``(display, target)`` pair scored by likelihood decoding. */
export type Alternative = readonly [display: string, target: string];

/** Reject anything but exactly the schema's required fields (semantic parsers validate the values). */
export function validateStructured(value: unknown, schema: Readonly<Record<string, unknown>>): void {
  const required = schema.required;
  const keys = isPlainObject(value) ? Object.keys(value) : null;
  const expected = Array.isArray(required) ? new Set(required as unknown[]) : new Set<unknown>();
  if (keys === null || keys.length !== expected.size || !keys.every((key) => expected.has(key))) {
    throw new InvalidModelOutput('Structured output must contain exactly the required fields');
  }
}

function textContent(message: Message): string {
  const content = message.content;
  if (typeof content === 'string') return content;
  if (!content.every((part) => part instanceof TextPart)) {
    throw new ValueError('native text models only support text message content');
  }
  return content.map((part) => (part as TextPart).text).join('');
}

/** Plain-text serialization shared by likelihood scoring and its training. */
export function alternativePrompt(request: ModelRequest, alternatives: readonly Alternative[]): string {
  const lines: string[] = request.instructions ? [request.instructions] : [];
  for (const message of request.messages) lines.push(`${message.role}: ${textContent(message)}`);
  lines.push('Options:');
  for (const [display] of alternatives) lines.push(`- ${display}`);
  lines.push('Answer:');
  return lines.join('\n');
}

/** Python ``GenerationConfig.from_dict(data).to_json_string()`` (keeps a saved ``transformers_version``). */
function generationConfigFromDict(data: unknown): JsonObject {
  const result = generationConfigFromFile(data);
  const version = (data as JsonObject).transformers_version;
  if (typeof version === 'string') result.transformers_version = version;
  return result;
}

export interface NativeModelOptions {
  /** A loaded foundation replacing the freshly constructed architecture (``from_foundation``). */
  model?: T5ForConditionalGeneration;
}

/**
 * An owned native seq2seq model with its embedded tokenizer and generation
 * settings. ``configuration()`` reports the complete persisted native state.
 */
export class NativeModel extends Module {
  static override readonly qualifiedName: string = 'tensorcode._internal.text.native.NativeModel';
  readonly model: T5ForConditionalGeneration;
  readonly tokenizer: FastTokenizer;
  readonly generation: JsonObject;
  /** Serialized ``model.generation_config`` (``to_json_string`` form). */
  generationConfig: JsonObject;

  constructor(config: JsonObject, options: NativeModelOptions = {}) {
    super();
    if (!('native_config' in config) || !('tokenizer' in config)) throw new ValueError('config requires native_config and tokenizer');
    const native = nativeConfig(config.native_config);
    if (!native.isEncoderDecoder) throw new ValueError('owned text operations require a native seq2seq architecture');
    let model = options.model;
    if (model === undefined) {
      const created = createNativeModel(native, 'seq2seq');
      if (!(created instanceof T5ForConditionalGeneration)) throw new ValueError('owned text operations require a native seq2seq architecture');
      model = created;
      restoreParameterAliases(model, config.native_parameter_aliases as Record<string, string> | null | undefined);
    }
    this.model = this.registerModule('model', model);
    this.generationConfig = 'native_generation_config' in config
      ? generationConfigFromDict(config.native_generation_config)
      : generationConfigFromModel(model.config);
    if (!isPlainObject(config.tokenizer)) throw new ValueError('tokenizer configuration must be a JSON object');
    this.tokenizer = FastTokenizer.fromConfiguration(config.tokenizer as JsonObject);
    const generation = config.generation ?? {};
    if (!isPlainObject(generation)) throw new ValueError('unsupported generation setting');
    this.generation = { max_new_tokens: 32, do_sample: false, ...deepCopy(generation as JsonObject) };
    const allowed = new Set<string>(GENERATION_KEYS);
    if (Object.keys(this.generation).some((key) => !allowed.has(key))) throw new ValueError('unsupported generation setting');
  }

  /** The persisted native state (merged over the operation's semantic configuration). */
  configuration(): JsonObject {
    return {
      native_config: this.model.config.toDiffDict(),
      native_parameter_aliases: parameterAliases(this.model),
      tokenizer: this.tokenizer.configuration() as unknown as JsonValue,
      generation: deepCopy(this.generation),
      native_generation_config: deepCopy(this.generationConfig),
    };
  }

  /** ``json.dumps({...}, sort_keys=True)`` serialization of a request. */
  prompt(request: ModelRequest): string {
    const messages = request.messages.map((message) => ({ role: message.role, content: textContent(message) }));
    return sortedJsonDumps({
      messages, instructions: request.instructions, response_schema: request.responseSchema, schema_name: request.schemaName,
    });
  }

  /** Tokenized prompt (``input_ids``/``attention_mask``). */
  inputs(request: ModelRequest): { inputIds: Tensor; attentionMask: Tensor } {
    const tokens = this.tokenizer.encodeTensors([this.prompt(request)], { padding: true });
    return { inputIds: tokens.input_ids, attentionMask: tokens.attention_mask };
  }

  /** Run ``fn`` in evaluation mode, restoring every module's mode afterwards. */
  private withEvaluation<R>(fn: () => R): R {
    const modes = this.model.modules().map((module) => [module, module.training] as const);
    try {
      this.model.eval();
      return fn();
    } finally {
      for (const [module, mode] of modes) module.training = mode;
    }
  }

  /** Generated token ids ``[1, length]`` (including the decoder start token). */
  generateIds(request: ModelRequest): Tensor {
    const inputs = this.inputs(request);
    return this.withEvaluation(() => generateSeq2Seq(this.model, inputs, this.generation as GenerationSettings, {
      generationConfig: this.generationConfig,
    }));
  }

  /** Decoded generated text (special tokens skipped). */
  generateText(request: ModelRequest): string {
    const ids = this.generateIds(request);
    return this.tokenizer.decode(ids.select(0, 0), { skipSpecialTokens: true });
  }

  /** Generate a reply; structured requests must produce exactly the required JSON fields. */
  complete(request: ModelRequest): ModelOutput {
    const text = this.generateText(request);
    if (request.responseSchema === null) return new ModelOutput({ text });
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch (error) {
      throw new InvalidModelOutput('Native model did not generate valid JSON', { cause: error });
    }
    validateStructured(value, request.responseSchema);
    return new ModelOutput({ text, structured: value as Record<string, unknown> });
  }

  /** Teacher-forced loss of ``target`` text (current module mode and autograd state). */
  loss(request: ModelRequest, target: string): Tensor {
    const inputs = this.inputs(request);
    const tokens = this.tokenizer.encode([target], { padding: true });
    const labels = tokens.inputIds.map((row, r) => row.map((id, c) => (tokens.attentionMask[r]![c] ? id : -100)));
    const labelTensor = tensor(labels.flat(), { shape: [labels.length, labels[0]!.length], dtype: 'int64' });
    return this.model.forward({ inputIds: inputs.inputIds, attentionMask: inputs.attentionMask, labels: labelTensor }).loss!;
  }

  /**
   * One log-likelihood per ``(display, target)`` alternative. The prompt is
   * encoded once; every target is scored by the decoder in a single batch.
   * Gradients follow the current autograd/module mode.
   */
  alternativeLogLikelihoods(
    request: ModelRequest, alternatives: readonly Alternative[], options: { normalization?: string } = {},
  ): Tensor {
    const normalization = options.normalization ?? 'sum';
    if (normalization !== 'sum' && normalization !== 'mean') throw new ValueError('normalization must be sum or mean');
    const tokens = this.tokenizer.encodeTensors([alternativePrompt(request, alternatives)]);
    const encoded = this.model.encode({ inputIds: tokens.input_ids, attentionMask: tokens.attention_mask });
    const count = alternatives.length;
    const targets = this.tokenizer.encode(alternatives.map(([, target]) => target), { padding: true });
    const width = targets.inputIds[0]?.length ?? 0;
    const rows = new Set(targets.inputIds.map((row, r) => JSON.stringify(row.filter((_, c) => targets.attentionMask[r]![c]))));
    if (rows.size !== count) throw new ValueError('likelihood alternatives must tokenize to distinct sequences');
    const valid = targets.attentionMask.flat();
    const labels = tensor(targets.inputIds.flat().map((id, index) => (valid[index] ? id : -100)), { shape: [count, width], dtype: 'int64' });
    const index = tensor(targets.inputIds.flat().map((id, position) => (valid[position] ? id : 0)), { shape: [count, width, 1], dtype: 'int64' });
    const mask = tensor(valid, { shape: [count, width], dtype: 'float32' });
    const [, length, hidden] = encoded.shape as [number, number, number];
    const promptLength = tokens.attention_mask.shape[1]!;
    const logits = this.model.forward({
      encoderHiddenStates: encoded.expand(count, length, hidden),
      attentionMask: tokens.attention_mask.expand(count, promptLength),
      decoderInputIds: this.model.shiftRight(labels),
    }).logits;
    const tokenScores = logits.logSoftmax(-1).gather(-1, index).squeeze(-1);
    let scores = tokenScores.mul(mask).sum(-1);
    if (normalization === 'mean') scores = scores.div(mask.sum(-1));
    return scores;
  }

  /** Evaluation-mode, gradient-free alternative scores as numbers. */
  scoreAlternatives(request: ModelRequest, alternatives: readonly Alternative[], options: { normalization?: string } = {}): number[] {
    const scores = this.withEvaluation(() => noGrad(() => this.alternativeLogLikelihoods(request, alternatives, options)));
    return Array.from(scores.data, Number);
  }
}
