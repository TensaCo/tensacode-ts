/**
 * Native (Hugging Face transformers) model configuration.
 *
 * {@link NativeConfig.fromDict} mirrors ``AutoConfig.for_model(model_type, **data)``
 * for the supported architectures, and {@link NativeConfig.toDict} /
 * {@link NativeConfig.toDiffDict} mirror ``to_dict()`` and
 * ``json.loads(to_json_string())`` of transformers ``5.17``.
 *
 * A configuration that already carries ``transformers_version`` has been
 * normalized by Python or TypeScript; it is retained verbatim and returned
 * unchanged by both serializers, so saved artifacts reconstruct exactly.
 * Partial configurations are normalized with defaults generated from
 * transformers (``defaults.generated.ts``).
 */
import { ValueError } from '../../errors.js';
import {
  deepCopy, float, isPlainObject, jsonEqual, mergeJson, orderedEntries, orderedKeys, orderedObject, transferPythonNumberKind,
  type JsonObject, type JsonValue,
} from '../json.js';
import {
  ATTRIBUTE_MAPS, BASE_CONFIG_DEFAULTS, CLASS_CONFIG_DEFAULTS, GENERATION_CONFIG_DEFAULTS, TRANSFORMERS_VERSION,
} from './defaults.generated.js';

export { TRANSFORMERS_VERSION } from './defaults.generated.js';



/** Model types whose architectures are implemented natively in TypeScript. */
export const SUPPORTED_MODEL_TYPES = ['albert', 'bert', 'roberta', 'electra', 'distilbert', 't5', 'vit', 'clip', 'deberta-v2', 'llama', 'idefics3'] as const;
export type SupportedModelType = (typeof SUPPORTED_MODEL_TYPES)[number];

const NESTED: Record<string, Record<string, string>> = {
  clip: { text_config: 'clip_text_model', vision_config: 'clip_vision_model' },
  idefics3: { text_config: 'llama', vision_config: 'idefics3_vision' },
};

/** ``LlamaConfig.default_theta`` (a Python float). */
const LLAMA_DEFAULT_THETA = float(10000.0);

/**
 * ``target[targetKey] = copy(source[sourceKey])``, keeping the Python number
 * kind the source recorded (a Python int in a float field stays an int).
 */
function put(target: JsonObject, targetKey: string, source: JsonObject, sourceKey: string = targetKey): void {
  target[targetKey] = deepCopy(source[sourceKey] as JsonValue);
  transferPythonNumberKind(target, targetKey, source, sourceKey);
}

/**
 * ``RotaryEmbeddingConfigMixin.convert_rope_params_to_dict`` (single global
 * RoPE dictionary): legacy ``rope_theta``/``rope_scaling`` become
 * ``rope_parameters``.
 */
function standardizeRope(values: JsonObject, input: JsonObject): void {
  const scaling = input.rope_scaling;
  const supplied = input.rope_parameters;
  const parameters: JsonObject = isPlainObject(scaling) && Object.keys(scaling).length
    ? deepCopy(scaling as JsonObject)
    : isPlainObject(supplied) ? deepCopy(supplied as JsonObject) : {};
  delete values.rope_scaling;
  delete values.rope_theta;
  if (!('rope_theta' in parameters)) {
    if (input.rope_theta !== undefined) put(parameters, 'rope_theta', input);
    else put(parameters, 'rope_theta', deepCopy<JsonObject>({ rope_theta: LLAMA_DEFAULT_THETA } as unknown as JsonObject));
  }
  if (input.partial_rotary_factor !== undefined && input.partial_rotary_factor !== null && !('partial_rotary_factor' in parameters)) {
    put(parameters, 'partial_rotary_factor', input);
  }
  if (!('rope_type' in parameters)) parameters.rope_type = parameters.type !== undefined ? deepCopy(parameters.type as JsonValue) : 'default';
  if (['llama3', 'yarn', 'longrope'].includes(String(parameters.rope_type)) && !('original_max_position_embeddings' in parameters)) {
    put(parameters, 'original_max_position_embeddings', values, 'max_position_embeddings');
  }
  values.rope_parameters = parameters;
}

function labelMaps(values: JsonObject, input: JsonObject): void {
  if (input.id2label !== undefined && input.id2label !== null) {
    if (!isPlainObject(input.id2label)) throw new ValueError('id2label must be a mapping');
    const entries = orderedEntries(input.id2label as JsonObject).map(([key, label]) => {
      const index = Number(key);
      if (!Number.isInteger(index)) throw new ValueError('id2label keys must be integers');
      return [String(index), deepCopy(label)] as const;
    });
    const id2label = orderedObject(entries);
    values.id2label = id2label;
    if (input.label2id === undefined || input.label2id === null) {
      values.label2id = orderedObject(entries.map(([index, label]) => [String(label), Number(index)] as const));
    }
  } else if (typeof input.num_labels === 'number') {
    const count = input.num_labels;
    values.id2label = Object.fromEntries(Array.from({ length: count }, (_, index) => [String(index), `LABEL_${index}`]));
    values.label2id = Object.fromEntries(Array.from({ length: count }, (_, index) => [`LABEL_${index}`, index]));
  }
  delete values.num_labels;
}

/**
 * Legacy generation parameters that transformers 5.17 model configurations
 * drop on construction (they belong to ``GenerationConfig``).
 */
const LEGACY_GENERATION_KEYS = new Set([
  'assistant_confidence_threshold', 'assistant_lookbehind', 'bad_words_ids', 'begin_suppress_tokens', 'diversity_penalty',
  'do_sample', 'early_stopping', 'encoder_no_repeat_ngram_size', 'encoder_repetition_penalty', 'epsilon_cutoff', 'eta_cutoff',
  'exponential_decay_length_penalty', 'forced_bos_token_id', 'forced_eos_token_id', 'length_penalty', 'max_length',
  'min_length', 'no_repeat_ngram_size', 'num_assistant_tokens', 'num_assistant_tokens_schedule', 'num_beam_groups',
  'num_beams', 'num_return_sequences', 'output_scores', 'remove_invalid_values', 'repetition_penalty',
  'return_dict_in_generate', 'suppress_tokens', 'target_lookbehind', 'temperature', 'top_k', 'top_p', 'typical_p',
]);

function normalize(modelType: string, input: JsonObject): JsonObject {
  const defaults = CLASS_CONFIG_DEFAULTS[modelType];
  if (!defaults) throw new ValueError(`unsupported native model_type ${JSON.stringify(modelType)}`);
  const aliases = ATTRIBUTE_MAPS[modelType] ?? {};
  const values = deepCopy(defaults);
  const nested = NESTED[modelType] ?? {};
  for (const key of orderedKeys(input)) {
    if (key === 'model_type' || key === 'transformers_version' || key in nested || key === 'torch_dtype') continue;
    if (LEGACY_GENERATION_KEYS.has(key)) continue;
    if (key.endsWith('_config_dict') && key.slice(0, -'_dict'.length) in nested) continue;
    put(values, aliases[key] ?? key, input, key);
  }
  // ``torch_dtype`` is the legacy spelling of ``dtype``.
  if (input.dtype === undefined && input.torch_dtype !== undefined) values.dtype = deepCopy(input.torch_dtype as JsonValue);
  labelMaps(values, input);
  if (modelType === 'llama') {
    if (input.head_dim === undefined || input.head_dim === null) values.head_dim = Math.floor((values.hidden_size as number) / (values.num_attention_heads as number));
    if (input.num_key_value_heads === undefined || input.num_key_value_heads === null) values.num_key_value_heads = values.num_attention_heads!;
    standardizeRope(values, input);
  }
  for (const [key, nestedType] of Object.entries(nested)) {
    const supplied = input[key];
    if (modelType === 'idefics3' && key === 'text_config') {
      if (!isPlainObject(supplied)) {
        // ``Idefics3Config`` default text model: Llama with ``rms_norm_eps=1e-5``.
        values[key] = normalize('llama', { rms_norm_eps: 1e-5, pad_token_id: values.pad_token_id! });
        delete (values[key] as JsonObject).transformers_version;
        continue;
      }
      const textType = (supplied as JsonObject).model_type ?? 'llama';
      if (textType !== 'llama') throw new ValueError(`unsupported Idefics3 text model_type ${JSON.stringify(textType)} (supported: llama)`);
    }
    let childInput = isPlainObject(supplied) ? supplied as JsonObject : {};
    // CLIP ``text_config_dict``/``vision_config_dict`` (legacy): a complete
    // configuration built from the dict overrides ``text_config``/``vision_config``.
    const legacy = input[`${key}_dict`];
    if (isPlainObject(legacy)) {
      const complete = normalize(nestedType, legacy as JsonObject);
      delete complete.transformers_version;
      childInput = mergeJson(childInput, complete);
    }
    const child = normalize(nestedType, childInput);
    delete child.transformers_version;
    values[key] = child;
  }
  if (modelType === 'vit') {
    if (input.pooler_output_size) put(values, 'pooler_output_size', input);
    else put(values, 'pooler_output_size', values, 'hidden_size');
  }
  if (modelType === 'deberta-v2') {
    if (typeof values.pos_att_type === 'string') values.pos_att_type = values.pos_att_type.toLowerCase().split('|').map((item) => item.trim());
    if (input.pooler_hidden_size !== undefined) put(values, 'pooler_hidden_size', input);
    else put(values, 'pooler_hidden_size', values, 'hidden_size');
  }
  if (modelType === 't5') {
    const decoderLayers = input.num_decoder_layers;
    if (decoderLayers === undefined || decoderLayers === null) put(values, 'num_decoder_layers', values, 'num_layers');
    else put(values, 'num_decoder_layers', input);
    const projection = String(values.feed_forward_proj);
    const parts = projection.split('-');
    if ((parts.length > 1 && parts[0] !== 'gated') || parts.length > 2) {
      throw new ValueError(`feed_forward_proj ${projection} is not a valid activation of the form gated-{ACT} or {ACT}`);
    }
    values.dense_act_fn = projection === 'gated-gelu' ? 'gelu_new' : parts[parts.length - 1]!;
    values.is_gated_act = parts[0] === 'gated';
    values.scale_decoder_outputs = input.tie_word_embeddings !== false;
    values.tie_word_embeddings = true;
    if (input.scale_decoder_outputs !== undefined) values.scale_decoder_outputs = input.scale_decoder_outputs as JsonValue;
  }
  values.model_type = modelType;
  values.transformers_version = TRANSFORMERS_VERSION;
  return values;
}

function diff(values: JsonObject, modelType: string): JsonObject {
  const classDefaults = CLASS_CONFIG_DEFAULTS[modelType] ?? {};
  const nested = NESTED[modelType] ?? {};
  const result: JsonObject = {};
  for (const [key, value] of Object.entries(values)) {
    if (key in nested && isPlainObject(value)) {
      const childType = nested[key]!;
      const childDefaults = CLASS_CONFIG_DEFAULTS[childType] ?? {};
      const child: JsonObject = {};
      for (const [childKey, childValue] of Object.entries(value)) {
        if (!(childKey in BASE_CONFIG_DEFAULTS) || !jsonEqual(childValue, childDefaults[childKey])) put(child, childKey, value as JsonObject);
      }
      if ('model_type' in value) child.model_type = (value as JsonObject).model_type!;
      result[key] = child;
      continue;
    }
    const inBase = key in BASE_CONFIG_DEFAULTS;
    if (!inBase || key === 'transformers_version' || key === 'vocab_file' || !jsonEqual(value, BASE_CONFIG_DEFAULTS[key])
      || (key in classDefaults && !jsonEqual(value, classDefaults[key]))) {
      put(result, key, values);
    }
  }
  delete result._name_or_path;
  return result;
}

/** A supported native architecture configuration. */
export class NativeConfig {
  readonly modelType: string;
  /** Complete attribute values (class defaults merged with the supplied data). */
  readonly values: Readonly<JsonObject>;
  private readonly verbatim: JsonObject | null;

  private constructor(modelType: string, values: JsonObject, verbatim: JsonObject | null) {
    this.modelType = modelType;
    this.values = values;
    this.verbatim = verbatim;
  }

  /** ``AutoConfig.for_model(data.model_type, **data)``. */
  static fromDict(data: unknown): NativeConfig {
    if (!isPlainObject(data)) throw new ValueError('native configuration must be a JSON object');
    const input = deepCopy(data as JsonObject);
    const modelType = input.model_type;
    if (typeof modelType !== 'string' || !modelType) throw new ValueError('native configuration requires model_type');
    if (!CLASS_CONFIG_DEFAULTS[modelType]) throw new ValueError(`unsupported native model_type ${JSON.stringify(modelType)}`);
    if (typeof input.transformers_version === 'string') {
      const values = normalize(modelType, input);
      // Normalized input: retain every supplied value (including derived fields).
      for (const key of Object.keys(input)) if (!(key in (NESTED[modelType] ?? {}))) put(values, ATTRIBUTE_MAPS[modelType]?.[key] ?? key, input, key);
      return new NativeConfig(modelType, values, input);
    }
    return new NativeConfig(modelType, normalize(modelType, input), null);
  }

  /**
   * ``AutoConfig.for_model(model_type, **data)`` exactly as Python constructs
   * a model from a saved configuration dictionary (for example
   * ``Chatbot``'s ``foundation_config``): always normalized, a supplied
   * ``transformers_version`` is re-stamped and derived fields are recomputed
   * (T5 rewrites the legacy tie flag), so ``toDiffDict()`` equals
   * ``json.loads(model.config.to_json_string())``. {@link fromDict} instead
   * keeps an already-normalized configuration verbatim.
   */
  static forModel(data: unknown): NativeConfig {
    if (!isPlainObject(data)) throw new ValueError('native configuration must be a JSON object');
    const input = deepCopy(data as JsonObject);
    const modelType = input.model_type;
    if (typeof modelType !== 'string' || !modelType) throw new ValueError('native configuration requires model_type');
    if (!CLASS_CONFIG_DEFAULTS[modelType]) throw new ValueError(`unsupported native model_type ${JSON.stringify(modelType)}`);
    return new NativeConfig(modelType, normalize(modelType, input), null);
  }

  /**
   * ``AutoConfig.from_pretrained`` for a downloaded ``config.json``: always
   * normalized (re-stamped with this transformers version), ``_name_or_path``
   * recorded and ``dtype`` (also on sub-configurations) set to the dtype the
   * weights are loaded in.
   */
  static fromPretrainedDict(data: unknown, nameOrPath: string, dtype: string = 'float32'): NativeConfig {
    if (!isPlainObject(data)) throw new ValueError('config.json must contain a JSON object');
    const input = deepCopy(data as JsonObject);
    const modelType = input.model_type;
    if (typeof modelType !== 'string' || !CLASS_CONFIG_DEFAULTS[modelType]) {
      throw new ValueError(`unsupported native model_type ${JSON.stringify(modelType)}`);
    }
    for (const key of ['transformers_version', 'torch_dtype', '_commit_hash', '_name_or_path']) delete input[key];
    const values = normalize(modelType, input);
    values._name_or_path = nameOrPath;
    values.dtype = dtype;
    for (const key of Object.keys(NESTED[modelType] ?? {})) {
      if (isPlainObject(values[key])) (values[key] as JsonObject).dtype = dtype;
    }
    return new NativeConfig(modelType, values, null);
  }

  /** Whether the configuration was supplied already normalized (kept verbatim). */
  get isVerbatim(): boolean {
    return this.verbatim !== null;
  }

  /** ``config.to_dict()`` (verbatim when supplied normalized). */
  toDict(): JsonObject {
    return deepCopy(this.verbatim ?? (this.values as JsonObject));
  }

  /** ``json.loads(config.to_json_string())`` (verbatim when supplied normalized). */
  toDiffDict(): JsonObject {
    return deepCopy(this.verbatim ?? diff(this.values as JsonObject, this.modelType));
  }

  /** Attribute lookup honouring ``attribute_map`` aliases (``hidden_size`` → ``d_model`` for T5). */
  get(key: string): JsonValue | undefined {
    const canonical = ATTRIBUTE_MAPS[this.modelType]?.[key] ?? key;
    return this.values[canonical];
  }

  has(key: string): boolean {
    return this.get(key) !== undefined;
  }

  number(key: string): number {
    const value = this.get(key);
    if (typeof value !== 'number') throw new ValueError(`native configuration ${key} must be a number`);
    return value;
  }

  optionalNumber(key: string): number | null {
    const value = this.get(key);
    return typeof value === 'number' ? value : null;
  }

  string(key: string): string {
    const value = this.get(key);
    if (typeof value !== 'string') throw new ValueError(`native configuration ${key} must be a string`);
    return value;
  }

  boolean(key: string): boolean {
    return this.get(key) === true;
  }

  /** Number of classification labels (``len(id2label)``). */
  get numLabels(): number {
    const labels = this.get('id2label');
    return isPlainObject(labels) ? Object.keys(labels).length : 2;
  }

  get isEncoderDecoder(): boolean {
    return this.get('is_encoder_decoder') === true;
  }

  /** Hidden width (``hidden_size``/``d_model``/``dim``). */
  get hiddenSize(): number {
    return this.number('hidden_size');
  }

  /** A derived runtime configuration (for example a T5 decoder stack); never serialized. */
  derive(overrides: JsonObject): NativeConfig {
    return new NativeConfig(this.modelType, mergeJson(this.values, overrides), null);
  }

  /** Nested sub-configuration (CLIP ``text_config``/``vision_config``). */
  sub(key: string): NativeConfig {
    const nestedType = NESTED[this.modelType]?.[key];
    const value = this.values[key];
    if (!nestedType || !isPlainObject(value)) throw new ValueError(`native configuration has no nested ${key}`);
    return new NativeConfig(nestedType, mergeJson(normalize(nestedType, value as JsonObject), value as JsonObject), null);
  }
}

/**
 * Python ``tensorcode._internal.vec.text._native_config``: ``AutoConfig.for_model``
 * then restore the authored ``tie_word_embeddings``/``scale_decoder_outputs``.
 */
export function nativeConfig(data: unknown): NativeConfig {
  const config = NativeConfig.fromDict(data);
  if (config.isVerbatim) return config;
  const overrides: JsonObject = {};
  for (const name of ['tie_word_embeddings', 'scale_decoder_outputs']) {
    if (name in (data as JsonObject)) put(overrides, name, data as JsonObject);
  }
  return Object.keys(overrides).length ? config.derive(overrides) : config;
}

// ---------------------------------------------------------------------------
// Generation configuration.
// ---------------------------------------------------------------------------

/** ``GenerationConfig.from_model_config(config)`` serialized with ``to_json_string()``. */
export function generationConfigFromModel(config: NativeConfig): JsonObject {
  const result: JsonObject = { _from_model_config: true };
  for (const key of ['bos_token_id', 'decoder_start_token_id', 'eos_token_id', 'pad_token_id']) {
    const value = config.get(key);
    if (value !== undefined && value !== null) put(result, key, config.values as JsonObject, ATTRIBUTE_MAPS[config.modelType]?.[key] ?? key);
  }
  result.output_attentions = config.get('output_attentions') === true;
  result.output_hidden_states = config.get('output_hidden_states') === true;
  result.use_cache = config.get('use_cache') !== false;
  result.transformers_version = TRANSFORMERS_VERSION;
  return result;
}

/** ``GenerationConfig.from_pretrained(...).to_json_string()`` for a ``generation_config.json``. */
export function generationConfigFromFile(data: unknown): JsonObject {
  if (!isPlainObject(data)) throw new ValueError('generation_config.json must contain a JSON object');
  const result: JsonObject = {};
  for (const [key, value] of Object.entries(data as JsonObject)) {
    if (key === 'transformers_version') continue;
    if (!(key in GENERATION_CONFIG_DEFAULTS) || !jsonEqual(value, GENERATION_CONFIG_DEFAULTS[key])) put(result, key, data as JsonObject);
  }
  result.transformers_version = TRANSFORMERS_VERSION;
  return result;
}

/** Effective generation settings: ``GenerationConfig`` defaults overlaid with a serialized config. */
export function generationDefaults(serialized: JsonObject | null | undefined): JsonObject {
  return mergeJson(GENERATION_CONFIG_DEFAULTS, serialized ?? {});
}
