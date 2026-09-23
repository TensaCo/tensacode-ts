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
import {
  deepCopy, isPlainObject, parseJsonStrict, pythonJsonDumps, sha256Hex, type JsonObject, type JsonValue,
} from '../_internal/json.js';
import { hubOffline, resolveArtifactDirectory, type HubOptions } from '../_internal/hub.js';
import { pathExists } from '../_internal/files.js';
import { loadNativeFoundation } from '../_internal/native/foundation.js';
import { TRANSFORMERS_VERSION, generationConfigFromModel } from '../_internal/native/config.js';
import { GENERATION_CONFIG_DEFAULTS } from '../_internal/native/defaults.generated.js';
import {
  IDEFICS3_IMAGE_PROCESSOR_DEFAULTS, addSpecialTokens,
} from '../_internal/native/idefics3Processing.js';
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
 * How a transformers 5 tokenizer class turns ``from_pretrained`` keyword
 * arguments into ``init_kwargs``: parameters its ``__init__`` forwards to
 * ``TokenizersBackend`` (with their defaults) and parameters it consumes.
 */
interface TokenizerClassSpec {
  forwarded: Record<string, JsonValue | ((config: JsonObject, tokenizer: FastTokenizer) => JsonValue)>;
  consumed: readonly string[];
  /** ``vocab_files_names`` keys removed at save. */
  files: readonly string[];
}

const BACKEND_FILES = ['tokenizer_file', 'vocab_file'];

const TOKENIZER_CLASS_SPECS: Record<string, TokenizerClassSpec> = {
  TokenizersBackend: { forwarded: {}, consumed: [], files: BACKEND_FILES },
  GPT2Tokenizer: {
    forwarded: { errors: 'replace', unk_token: '<|endoftext|>', bos_token: '<|endoftext|>', eos_token: '<|endoftext|>', pad_token: null, add_prefix_space: false },
    consumed: ['vocab', 'merges'], files: ['vocab_file', 'merges_file'],
  },
  LlamaTokenizer: {
    forwarded: { clean_up_tokenization_spaces: false, unk_token: '<unk>', bos_token: '<s>', eos_token: '</s>', use_default_system_prompt: false, add_prefix_space: null },
    consumed: ['vocab', 'merges', 'legacy'], files: ['vocab_file', 'tokenizer_file'],
  },
  T5Tokenizer: {
    forwarded: { eos_token: '</s>', unk_token: '<unk>', pad_token: '<pad>', extra_ids: 100 },
    consumed: ['vocab', '_spm_precompiled_charsmap'], files: ['vocab_file', 'tokenizer_file'],
  },
  AlbertTokenizer: {
    forwarded: {
      do_lower_case: true, keep_accents: false, bos_token: '[CLS]', eos_token: '[SEP]', sep_token: '[SEP]', cls_token: '[CLS]',
      unk_token: '<unk>', pad_token: '<pad>', mask_token: '[MASK]', add_prefix_space: true, trim_offsets: true,
    },
    consumed: ['vocab', '_spm_precompiled_charsmap'], files: ['vocab_file', 'tokenizer_file'],
  },
  BertTokenizer: {
    forwarded: {
      do_lower_case: true, unk_token: '[UNK]', sep_token: '[SEP]', pad_token: '[PAD]', cls_token: '[CLS]', mask_token: '[MASK]',
      tokenize_chinese_chars: true, strip_accents: null,
    },
    consumed: ['vocab'], files: ['vocab_file', 'tokenizer_file'],
  },
  RobertaTokenizer: {
    forwarded: {
      errors: 'replace', bos_token: '<s>', eos_token: '</s>', sep_token: '</s>', cls_token: '<s>', unk_token: '<unk>',
      pad_token: '<pad>', mask_token: '<mask>', add_prefix_space: false, trim_offsets: true,
    },
    consumed: ['vocab', 'merges'], files: ['vocab_file', 'merges_file', 'tokenizer_file'],
  },
  CLIPTokenizer: {
    forwarded: { unk_token: '<|endoftext|>', bos_token: '<|startoftext|>', eos_token: '<|endoftext|>', pad_token: '<|endoftext|>' },
    consumed: ['vocab', 'merges'], files: ['vocab_file', 'merges_file', 'tokenizer_file'],
  },
  DebertaV2Tokenizer: {
    forwarded: {
      bos_token: '[CLS]', eos_token: '[SEP]', unk_token: '[UNK]', sep_token: '[SEP]', cls_token: '[CLS]', pad_token: '[PAD]',
      mask_token: '[MASK]', unk_id: (config, tokenizer) => debertaUnkId(config, tokenizer), do_lower_case: false,
      split_by_punct: false, add_prefix_space: true,
    },
    consumed: ['vocab'], files: ['vocab_file', 'tokenizer_file'],
  },
};

/** ``DebertaV2Tokenizer``'s ``unk_id``: the index of ``(unk_token, 0.0)`` in the vocabulary, else the argument. */
function debertaUnkId(config: JsonObject, tokenizer: FastTokenizer): number {
  const unk = tokenText(config.unk_token) ?? '[UNK]';
  const model = (JSON.parse(tokenizer.jsonText) as { model?: { vocab?: unknown } }).model;
  if (Array.isArray(model?.vocab)) {
    const index = (model!.vocab as [string, number][]).findIndex(([piece, score]) => piece === unk && score === 0);
    if (index >= 0) return index;
  }
  return typeof config.unk_id === 'number' ? config.unk_id : 1;
}

/** Tokenizer classes transformers 5.17 registers (unknown names load as ``TokenizersBackend``). */
const TRANSFORMERS_TOKENIZER_CLASSES = new Set([
  'AlbertTokenizer', 'BartTokenizer', 'BarthezTokenizer', 'BartphoTokenizer', 'BertJapaneseTokenizer', 'BertTokenizer', 'BertweetTokenizer',
  'BigBirdTokenizer', 'BioGptTokenizer', 'BlenderbotSmallTokenizer', 'BlenderbotTokenizer', 'ByT5Tokenizer', 'CLIPTokenizer', 'CTRLTokenizer',
  'CamembertTokenizer', 'CanineTokenizer', 'ClvpTokenizer', 'CodeLlamaTokenizer', 'CohereTokenizer', 'CpmAntTokenizer', 'CpmTokenizer',
  'DPRQuestionEncoderTokenizer', 'DebertaTokenizer', 'DebertaV2Tokenizer', 'DiaTokenizer', 'EsmTokenizer', 'EsmcTokenizer', 'FNetTokenizer',
  'FSMTTokenizer', 'FlaubertTokenizer', 'FunnelTokenizer', 'GPT2Tokenizer', 'GPTNeoXJapaneseTokenizer', 'GPTNeoXTokenizer', 'GemmaTokenizer',
  'HerbertTokenizer', 'LEDTokenizer', 'LasrTokenizer', 'LayoutLMv2Tokenizer', 'LayoutLMv3Tokenizer', 'LayoutXLMTokenizer', 'LlamaTokenizer',
  'LukeTokenizer', 'LxmertTokenizer', 'MBart50Tokenizer', 'MBartTokenizer', 'MPNetTokenizer', 'MarkupLMTokenizer', 'MgpstrTokenizer',
  'MobileBertTokenizer', 'MvpTokenizer', 'MyT5Tokenizer', 'NllbTokenizer', 'NougatTokenizer', 'OpenAIGPTTokenizer', 'PLBartTokenizer',
  'ParakeetTokenizer', 'PegasusTokenizer', 'PerceiverTokenizer', 'PhobertTokenizer', 'ProphetNetTokenizer', 'Qwen2Tokenizer',
  'Qwen3_5Tokenizer', 'RagTokenizer', 'ReformerTokenizer', 'RemBertTokenizer', 'RoCBertTokenizer', 'RoFormerTokenizer', 'RobertaTokenizer',
  'SeamlessM4TTokenizer', 'Siglip2Tokenizer', 'SplinterTokenizer', 'T5Tokenizer', 'TapasTokenizer', 'TokenizersBackend', 'UdopTokenizer',
  'VideoPrismTokenizer', 'VitsTokenizer', 'Wav2Vec2CTCTokenizer', 'Wav2Vec2PhonemeCTCTokenizer', 'WhisperTokenizer', 'XGLMTokenizer',
  'XLMRobertaTokenizer', 'XLMTokenizer', 'XLNetTokenizer',
]);

/** The transformers class ``AutoTokenizer`` instantiates for a declared (or model-type) class name. */
export function resolvedTokenizerClass(declared: string | null): string {
  if (declared === null || declared === 'PreTrainedTokenizerFast' || declared === 'PreTrainedTokenizer') return 'TokenizersBackend';
  const base = declared.replace(/Fast$/, '');
  if (TRANSFORMERS_TOKENIZER_CLASSES.has(base) || TRANSFORMERS_TOKENIZER_CLASSES.has(declared)) return base;
  return 'TokenizersBackend';
}

/**
 * ``tokenizer_config.json`` written by transformers 5.17 ``save_pretrained``
 * for the tokenizer ``Idefics3Processor.from_pretrained`` loads (which replaces
 * the extra special tokens with the Idefics3 image tokens): the class's
 * ``init_kwargs`` with attribute values, the special tokens and the class name.
 */
function savedTokenizerConfig(
  source: JsonObject, specialMap: JsonObject, tokenizer: FastTokenizer, className: string,
  options: { isLocal: boolean; localFilesOnly: boolean },
): JsonObject {
  const spec = TOKENIZER_CLASS_SPECS[className];
  if (!spec) {
    throw new NotImplementedError(`saving ${className} processor assets is not emulated (TokenizersBackend, ${Object.keys(TOKENIZER_CLASS_SPECS).slice(1).join(', ')} are)`);
  }
  // ``from_pretrained`` keyword arguments: tokenizer_config.json, then special_tokens_map.json.
  const kwargs: JsonObject = deepCopy(source);
  for (const [key, value] of Object.entries(specialMap)) if (!(key in kwargs)) kwargs[key] = deepCopy(value as JsonValue);
  for (const key of spec.consumed) delete kwargs[key];
  for (const [key, fallback] of Object.entries(spec.forwarded)) {
    if (!(key in kwargs)) kwargs[key] = typeof fallback === 'function' ? fallback(source, tokenizer) : deepCopy(fallback);
    else if (typeof fallback === 'function') kwargs[key] = fallback(source, tokenizer);
  }
  for (const key of ['tokenizer_object', 'gguf_file', 'tokenizer_file', '_json_truncation', '_json_padding', '_spm_precompiled_charsmap',
    'tokenizer_truncation', 'tokenizer_padding', 'post_processor']) delete kwargs[key];
  if (!('backend' in kwargs)) kwargs.backend = 'tokenizers';
  kwargs.is_local = options.isLocal;
  kwargs.local_files_only = options.localFilesOnly;
  // ``save_pretrained``.
  const config: JsonObject = {};
  for (const [key, value] of Object.entries(kwargs)) {
    if (key === 'add_bos_token' || key === 'add_eos_token') continue;
    config[key] = deepCopy(value as JsonValue);
  }
  if ('vocab_size' in config) config.vocab_size = tokenizer.backend.baseVocabSize();
  config.model_max_length = tokenizer.options.model_max_length;
  for (const key of SPECIAL_KEYS) if (key in config) config[key] = tokenText(config[key]);
  for (const key of ['added_tokens_decoder', 'additional_special_tokens', 'chat_template', 'name_or_path', 'special_tokens_map_file',
    'device_map', 'slow_tokenizer_class', 'auto_map', ...spec.files]) delete config[key];
  config.extra_special_tokens = [...IDEFICS3_EXTRA_SPECIAL];
  config.tokenizer_class = className;
  config.processor_class = 'Idefics3Processor';
  return config;
}

/** The files ``Idefics3Processor.from_pretrained(directory).save_pretrained(...)`` writes. */
export async function idefics3ProcessorAssets(directory: string, options: { isLocal: boolean; localFilesOnly: boolean }): Promise<Record<string, string>> {
  const loaded = await FastTokenizer.fromDirectory(directory);
  const tokenizerConfig = await readJson(directory, 'tokenizer_config.json') ?? {};
  const modelConfig = await readJson(directory, 'config.json') ?? {};
  const specialMap = await readJson(directory, 'special_tokens_map.json') ?? {};
  const declared = typeof tokenizerConfig.tokenizer_class === 'string' ? tokenizerConfig.tokenizer_class
    : typeof modelConfig.model_type === 'string' ? loaded.tokenizerClass : null;
  const className = resolvedTokenizerClass(declared);
  // ``Idefics3Processor.__init__`` adds its image tokens to the tokenizer.
  const tokenizer = addSpecialTokens(loaded, IDEFICS3_EXTRA_SPECIAL);
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
    'tokenizer_config.json': jsonFile(savedTokenizerConfig(tokenizerConfig, specialMap, tokenizer, className, options), false),
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
  const loaded = await loadNativeFoundation(repoId, { ...hub, head: 'image-text-to-text', dtype: 'float32', tokenizer: false, initializeMissing: true });
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
