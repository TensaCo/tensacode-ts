/**
 * ``Scene.from_language_foundation`` inputs: the owned Idefics3 weights and
 * configuration, the serialized ``GenerationConfig``, and the processor
 * assets exactly as Python's ``Idefics3Processor.from_pretrained(...)
 * .save_pretrained(...)`` writes them (their SHA-256 hashes are persisted in
 * the tool configuration).
 */
import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import type { Tensor } from '../nn/tensor.js';
import { NotImplementedError, ValueError } from '../errors.js';
import { deepCopy, isPlainObject, parseJsonStrict, pythonJsonDumps, sha256Hex, type JsonObject, type JsonValue } from '../_internal/json.js';
import { hubOffline, resolveArtifactDirectory, type HubOptions } from '../_internal/hub.js';
import { pathExists } from '../_internal/files.js';
import { loadNativeFoundation } from '../_internal/native/foundation.js';
import { TRANSFORMERS_VERSION, generationConfigFromModel } from '../_internal/native/config.js';
import { GENERATION_CONFIG_DEFAULTS } from '../_internal/native/defaults.generated.js';
import { IDEFICS3_IMAGE_PROCESSOR_DEFAULTS } from '../_internal/native/idefics3Processing.js';
import { FastTokenizer, VERY_LARGE_INTEGER } from '../_internal/tokenizers/index.js';
import { rustTokenizerString } from '../_internal/tokenizers/serialization.js';

const PROCESSOR_FILES = [
  'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'preprocessor_config.json',
  'processor_config.json', 'chat_template.json', 'chat_template.jinja', 'additional_chat_templates/*.jinja',
];

async function readJson(directory: string, name: string): Promise<JsonObject | null> {
  const path = join(directory, name);
  if (!(await pathExists(path))) return null;
  const value = parseJsonStrict(await readFile(path, 'utf8'));
  if (!isPlainObject(value)) throw new ValueError(`${name} must contain a JSON object`);
  return value as JsonObject;
}

/** ``model.generation_config.to_dict()`` for a loaded foundation. */
function generationToDict(file: JsonObject | null, fromModel: JsonObject): JsonObject {
  const result: JsonObject = deepCopy(GENERATION_CONFIG_DEFAULTS);
  const source = file ?? fromModel;
  for (const [key, value] of Object.entries(source)) if (key !== 'transformers_version') result[key] = deepCopy(value as JsonValue);
  result.transformers_version = TRANSFORMERS_VERSION;
  return result;
}

/** ``json.dumps(value, indent=2, sort_keys=True) + "\n"`` (``to_json_string``). */
function jsonFile(value: JsonObject, ensureAscii: boolean): string {
  const placeholder = '__TENSORCODE_VERY_LARGE_INTEGER__';
  const text = pythonJsonDumps(JSON.parse(JSON.stringify(value, (_, item) => (item === VERY_LARGE_INTEGER ? placeholder : item))),
    { indent: 2, sortKeys: true, ensureAscii });
  // ``int(1e30)``, transformers' ``VERY_LARGE_INTEGER``.
  return `${text.split(`"${placeholder}"`).join('1000000000000000019884624838656')}\n`;
}

const SPECIAL_KEYS = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token'];
const IDEFICS3_EXTRA_SPECIAL = ['<fake_token_around_image>', '<image>', '<end_of_utterance>'];

function tokenText(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (isPlainObject(value) && typeof value.content === 'string') return value.content;
  return null;
}

/**
 * ``tokenizer_config.json`` written by transformers 5.17 ``save_pretrained``
 * for a ``GPT2Tokenizer`` or ``PreTrainedTokenizerFast`` loaded by
 * ``Idefics3Processor.from_pretrained`` (which replaces the extra special
 * tokens with the Idefics3 image tokens).
 */
function savedTokenizerConfig(
  source: JsonObject, specialMap: JsonObject, tokenizer: FastTokenizer, options: { isLocal: boolean; localFilesOnly: boolean },
): JsonObject {
  const declared = typeof source.tokenizer_class === 'string' ? source.tokenizer_class.replace(/Fast$/, '') : 'PreTrainedTokenizer';
  let tokenizerClass: string;
  const config: JsonObject = {};
  if (declared === 'GPT2Tokenizer') {
    tokenizerClass = 'GPT2Tokenizer';
    config.errors = 'replace';
    config.add_prefix_space = false;
  } else if (declared === 'PreTrainedTokenizer' || declared === 'TokenizersBackend') {
    tokenizerClass = 'TokenizersBackend';
  } else {
    throw new NotImplementedError(`saving ${String(source.tokenizer_class)} processor assets is not emulated (GPT2Tokenizer and PreTrainedTokenizerFast are)`);
  }
  for (const [key, value] of Object.entries(source)) {
    if (['added_tokens_decoder', 'additional_special_tokens', 'chat_template', 'tokenizer_class', 'tokenizer_file', 'vocab_file',
      'merges_file', 'name_or_path', 'special_tokens_map_file', 'device_map', 'slow_tokenizer_class', 'add_bos_token', 'add_eos_token',
      'extra_special_tokens', 'auto_map'].includes(key)) continue;
    config[key] = deepCopy(value as JsonValue);
  }
  config.is_local = options.isLocal;
  config.local_files_only = options.localFilesOnly;
  config.backend = 'tokenizers';
  // ``getattr(tokenizer, key)`` for attributes; ``vocab_size`` excludes added tokens.
  if ('vocab_size' in config) config.vocab_size = tokenizer.backend.baseVocabSize();
  config.model_max_length = tokenizer.options.model_max_length;
  for (const key of SPECIAL_KEYS) {
    const value = tokenText(source[key]) ?? tokenText(specialMap[key]);
    if (value !== null) config[key] = value;
    else delete config[key];
  }
  config.extra_special_tokens = [...IDEFICS3_EXTRA_SPECIAL];
  config.tokenizer_class = tokenizerClass;
  config.processor_class = 'Idefics3Processor';
  return config;
}

/** The files ``Idefics3Processor.from_pretrained(directory).save_pretrained(...)`` writes. */
export async function idefics3ProcessorAssets(directory: string, options: { isLocal: boolean; localFilesOnly: boolean }): Promise<Record<string, string>> {
  const tokenizer = await FastTokenizer.fromDirectory(directory);
  const tokenizerConfig = await readJson(directory, 'tokenizer_config.json') ?? {};
  const specialMap = await readJson(directory, 'special_tokens_map.json') ?? {};
  const processorConfig = await readJson(directory, 'processor_config.json') ?? {};
  const imageFile = isPlainObject(processorConfig.image_processor)
    ? processorConfig.image_processor as JsonObject : await readJson(directory, 'preprocessor_config.json') ?? {};
  const image: JsonObject = deepCopy(IDEFICS3_IMAGE_PROCESSOR_DEFAULTS as JsonObject);
  for (const [key, value] of Object.entries(imageFile)) {
    if (key === 'processor_class' || key === '_processor_class' || key === 'image_processor_type') continue;
    image[key] = deepCopy(value as JsonValue);
  }
  image.image_processor_type = 'Idefics3ImageProcessor';
  let template: string | null = null;
  if (await pathExists(join(directory, 'chat_template.jinja'))) template = await readFile(join(directory, 'chat_template.jinja'), 'utf8');
  else {
    const legacy = await readJson(directory, 'chat_template.json');
    if (legacy && typeof legacy.chat_template === 'string') template = legacy.chat_template;
    else if (typeof processorConfig.chat_template === 'string') template = processorConfig.chat_template;
    else if (typeof tokenizerConfig.chat_template === 'string') template = tokenizerConfig.chat_template;
  }
  const assets: Record<string, string> = {
    'processor_config.json': jsonFile({
      image_processor: image, image_seq_len: typeof processorConfig.image_seq_len === 'number' ? processorConfig.image_seq_len : 169,
      processor_class: 'Idefics3Processor',
    }, true),
    'tokenizer.json': rustTokenizerString(tokenizer.jsonText, { pretty: true }),
    'tokenizer_config.json': jsonFile(savedTokenizerConfig(tokenizerConfig, specialMap, tokenizer, options), false),
  };
  if (template !== null) assets['chat_template.jinja'] = template;
  // Named templates (``additional_chat_templates/<name>.jinja``) are loaded and re-saved verbatim.
  const named = join(directory, 'additional_chat_templates');
  if (await pathExists(named)) {
    for (const entry of (await readdir(named)).sort()) {
      if (entry.endsWith('.jinja')) assets[`additional_chat_templates/${entry}`] = await readFile(join(named, entry), 'utf8');
    }
  }
  return assets;
}

export interface LanguageFoundation {
  config: JsonObject;
  weights: Map<string, Tensor>;
}

/** Tool configuration (with private processor assets) and float32 weights of an Idefics3 foundation. */
export async function languageFoundationConfig(
  repoId: string,
  options: { revision: string | null; localFilesOnly?: boolean; cacheDir?: string | null; token?: string | null; endpoint?: string | null; freezeFoundation?: boolean },
): Promise<LanguageFoundation> {
  const { revision, freezeFoundation = true, ...rest } = options;
  const hub: HubOptions = { ...rest, revision };
  const loaded = await loadNativeFoundation(repoId, { ...hub, head: 'image-text-to-text', dtype: 'float32', tokenizer: false });
  const { path, remote } = await resolveArtifactDirectory(repoId, { ...hub, allowPatterns: PROCESSOR_FILES });
  const generation = generationToDict(await readJson(path, 'generation_config.json'), generationConfigFromModel(loaded.config));
  // transformers records ``local_files_only``, forced on in offline mode (``HF_HUB_OFFLINE``).
  const assets = await idefics3ProcessorAssets(path, { isLocal: !remote, localFilesOnly: (options.localFilesOnly ?? false) || hubOffline() });
  const config: JsonObject = {
    mode: 'language', language_config: loaded.config.toDict(), generation_config: generation,
    foundation_source: { repo_id: repoId, revision }, freeze_foundation: freezeFoundation,
    processor_hashes: Object.fromEntries(Object.entries(assets).map(([name, value]) => [name, sha256Hex(value)])),
    _language_assets: assets,
  };
  return { config, weights: loaded.model.stateDict() };
}
