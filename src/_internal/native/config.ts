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
import { deepCopy, isPlainObject, jsonEqual, type JsonObject, type JsonValue } from '../json.js';
import {
  ATTRIBUTE_MAPS, BASE_CONFIG_DEFAULTS, CLASS_CONFIG_DEFAULTS, GENERATION_CONFIG_DEFAULTS, TRANSFORMERS_VERSION,
} from './defaults.generated.js';

export { TRANSFORMERS_VERSION } from './defaults.generated.js';



/** Model types whose architectures are implemented natively in TypeScript. */
export const SUPPORTED_MODEL_TYPES = ['bert', 'roberta', 'electra', 'distilbert', 't5', 'vit', 'clip', 'deberta-v2'] as const;
export type SupportedModelType = (typeof SUPPORTED_MODEL_TYPES)[number];

const NESTED: Record<string, Record<string, string>> = {
  clip: { text_config: 'clip_text_model', vision_config: 'clip_vision_model' },
};

function labelMaps(values: JsonObject, input: JsonObject): void {
  if (input.id2label !== undefined && input.id2label !== null) {
    if (!isPlainObject(input.id2label)) throw new ValueError('id2label must be a mapping');
    const id2label: JsonObject = {};
    for (const [key, label] of Object.entries(input.id2label)) {
      const index = Number(key);
      if (!Number.isInteger(index)) throw new ValueError('id2label keys must be integers');
      id2label[String(index)] = label as JsonValue;
    }
    values.id2label = id2label;
    if (input.label2id === undefined || input.label2id === null) {
      values.label2id = Object.fromEntries(Object.entries(id2label).map(([index, label]) => [String(label), Number(index)]));
    }
  } else if (typeof input.num_labels === 'number') {
    const count = input.num_labels;
    values.id2label = Object.fromEntries(Array.from({ length: count }, (_, index) => [String(index), `LABEL_${index}`]));
    values.label2id = Object.fromEntries(Array.from({ length: count }, (_, index) => [`LABEL_${index}`, index]));
  }
  delete values.num_labels;
}

function normalize(modelType: string, input: JsonObject): JsonObject {
  const defaults = CLASS_CONFIG_DEFAULTS[modelType];
  if (!defaults) throw new ValueError(`unsupported native model_type ${JSON.stringify(modelType)}`);
  const aliases = ATTRIBUTE_MAPS[modelType] ?? {};
  const values = deepCopy(defaults);
  const nested = NESTED[modelType] ?? {};
  for (const [key, value] of Object.entries(input)) {
    if (key === 'model_type' || key === 'transformers_version' || key in nested) continue;
    values[aliases[key] ?? key] = deepCopy(value as JsonValue);
  }
  labelMaps(values, input);
  for (const [key, nestedType] of Object.entries(nested)) {
    const supplied = input[key];
    const child = normalize(nestedType, isPlainObject(supplied) ? supplied as JsonObject : {});
    delete child.transformers_version;
    values[key] = child;
  }
  if (modelType === 'vit') {
    const size = input.pooler_output_size;
    values.pooler_output_size = size ? size as JsonValue : values.hidden_size!;
  }
  if (modelType === 'deberta-v2') {
    if (typeof values.pos_att_type === 'string') values.pos_att_type = values.pos_att_type.toLowerCase().split('|').map((item) => item.trim());
    values.pooler_hidden_size = input.pooler_hidden_size !== undefined ? input.pooler_hidden_size as JsonValue : values.hidden_size!;
  }
  if (modelType === 't5') {
    const decoderLayers = input.num_decoder_layers;
    values.num_decoder_layers = decoderLayers === undefined || decoderLayers === null ? values.num_layers! : decoderLayers as JsonValue;
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
        if (!(childKey in BASE_CONFIG_DEFAULTS) || !jsonEqual(childValue, childDefaults[childKey])) child[childKey] = childValue as JsonValue;
      }
      if ('model_type' in value) child.model_type = (value as JsonObject).model_type!;
      result[key] = child;
      continue;
    }
    const inBase = key in BASE_CONFIG_DEFAULTS;
    if (!inBase || key === 'transformers_version' || key === 'vocab_file' || !jsonEqual(value, BASE_CONFIG_DEFAULTS[key])
      || (key in classDefaults && !jsonEqual(value, classDefaults[key]))) {
      result[key] = value as JsonValue;
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
      for (const [key, value] of Object.entries(input)) if (!(key in (NESTED[modelType] ?? {}))) values[ATTRIBUTE_MAPS[modelType]?.[key] ?? key] = deepCopy(value as JsonValue);
      return new NativeConfig(modelType, values, input);
    }
    return new NativeConfig(modelType, normalize(modelType, input), null);
  }

  /**
   * ``AutoConfig.from_pretrained`` for a downloaded ``config.json``: always
   * normalized (re-stamped with this transformers version), ``_name_or_path``
   * recorded and ``dtype`` set to the float32 weights TypeScript loads.
   */
  static fromPretrainedDict(data: unknown, nameOrPath: string): NativeConfig {
    if (!isPlainObject(data)) throw new ValueError('config.json must contain a JSON object');
    const input = deepCopy(data as JsonObject);
    const modelType = input.model_type;
    if (typeof modelType !== 'string' || !CLASS_CONFIG_DEFAULTS[modelType]) {
      throw new ValueError(`unsupported native model_type ${JSON.stringify(modelType)}`);
    }
    for (const key of ['transformers_version', 'torch_dtype', '_commit_hash', '_name_or_path']) delete input[key];
    const values = normalize(modelType, input);
    values._name_or_path = nameOrPath;
    values.dtype = 'float32';
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
    return new NativeConfig(this.modelType, { ...deepCopy(this.values as JsonObject), ...deepCopy(overrides) }, null);
  }

  /** Nested sub-configuration (CLIP ``text_config``/``vision_config``). */
  sub(key: string): NativeConfig {
    const nestedType = NESTED[this.modelType]?.[key];
    const value = this.values[key];
    if (!nestedType || !isPlainObject(value)) throw new ValueError(`native configuration has no nested ${key}`);
    return new NativeConfig(nestedType, { ...normalize(nestedType, value as JsonObject), ...deepCopy(value as JsonObject) }, null);
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
    if (name in (data as JsonObject)) overrides[name] = deepCopy((data as JsonObject)[name]!);
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
    if (value !== undefined && value !== null) result[key] = deepCopy(value);
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
    if (!(key in GENERATION_CONFIG_DEFAULTS) || !jsonEqual(value, GENERATION_CONFIG_DEFAULTS[key])) result[key] = deepCopy(value as JsonValue);
  }
  result.transformers_version = TRANSFORMERS_VERSION;
  return result;
}

/** Effective generation settings: ``GenerationConfig`` defaults overlaid with a serialized config. */
export function generationDefaults(serialized: JsonObject | null | undefined): JsonObject {
  return { ...deepCopy(GENERATION_CONFIG_DEFAULTS), ...deepCopy(serialized ?? {}) };
}
