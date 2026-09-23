/**
 * Explicit pretrained foundation import (Python ``AutoModel*.from_pretrained``
 * with ``use_safetensors=True``, ``trust_remote_code=False``).
 *
 * Downloads (or resolves from the shared Hub cache) a model repository, reads
 * ``config.json`` and safetensors weights (single file or sharded index),
 * applies transformers' checkpoint conventions (``base_model_prefix``, legacy
 * ViT key names, tied T5 embeddings with an untied ``lm_head`` when the
 * checkpoint stores distinct values) and rejects missing or mismatched
 * weights. Only safetensors weights are accepted; no remote code runs.
 */
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Parameter, Tensor } from '../../nn/tensor.js';
import { noGrad } from '../../nn/autograd.js';
import { deserializeSafetensors } from '../../nn/safetensors.js';
import { ValueError } from '../../errors.js';
import { parseJsonStrict, type JsonObject } from '../json.js';
import { resolveArtifactDirectory, type HubOptions } from '../hub.js';
import { pathExists } from '../files.js';
import { FastTokenizer } from '../tokenizers/index.js';
import { NativeConfig, generationConfigFromFile, generationConfigFromModel } from './config.js';
import { BASE_MODEL_PREFIX, createNativeModel, type NativeHead } from './registry.js';
import type { NativeModel } from './modules.js';

export const FOUNDATION_FILES = [
  'config.json', 'generation_config.json', '*.safetensors', 'model.safetensors.index.json', 'tokenizer.json',
  'tokenizer_config.json', 'special_tokens_map.json', 'preprocessor_config.json',
];

export interface FoundationOptions extends Omit<HubOptions, 'allowPatterns'> {
  head?: NativeHead;
  addPoolingLayer?: boolean;
  /** Also load ``tokenizer.json`` as a {@link FastTokenizer} (default true when present). */
  tokenizer?: boolean;
  /**
   * Restore the raw ``tie_word_embeddings``/``scale_decoder_outputs`` flags from
   * ``config.json`` onto the loaded configuration (Python ``_load_foundation``).
   */
  restoreRawTieFlags?: boolean;
  /** Override configuration fields before construction (for example ``num_labels``). */
  configOverrides?: JsonObject;
}

export interface LoadedFoundation<T extends NativeModel = NativeModel> {
  model: T;
  config: NativeConfig;
  /** Serialized generation configuration (``generation_config.to_json_string()``) for seq2seq heads. */
  generationConfig: JsonObject | null;
  tokenizer: FastTokenizer | null;
  directory: string;
  /** The raw ``config.json`` object. */
  rawConfig: JsonObject;
  /** Checkpoint tensors not used by the constructed architecture (heads, legacy buffers). */
  unexpectedKeys: string[];
  /** Resolved Hub commit (``config._commit_hash``) when loaded from a Hub snapshot. */
  commitHash: string | null;
}

/** transformers 5.17 renames for legacy ``ViTModel`` checkpoints. */
function legacyVitKey(key: string): string {
  return key.replace(/encoder\.layer\./g, 'layers.')
    .replace('attention.attention.query', 'attention.q_proj')
    .replace('attention.attention.key', 'attention.k_proj')
    .replace('attention.attention.value', 'attention.v_proj')
    .replace('attention.output.dense', 'attention.o_proj')
    .replace('intermediate.dense', 'mlp.fc1')
    .replace(/(^|\.)output\.dense/, '$1mlp.fc2');
}

async function readWeights(directory: string): Promise<Map<string, Tensor>> {
  const single = join(directory, 'model.safetensors');
  const index = join(directory, 'model.safetensors.index.json');
  const files: string[] = [];
  if (await pathExists(single)) files.push(single);
  else if (await pathExists(index)) {
    const map = parseJsonStrict(await readFile(index, 'utf8')) as { weight_map?: Record<string, string> };
    files.push(...new Set(Object.values(map.weight_map ?? {}).map((file) => join(directory, file))));
  } else {
    throw new ValueError(`no safetensors weights in ${directory}; TensorCode accepts only safetensors foundation weights`);
  }
  const tensors = new Map<string, Tensor>();
  for (const file of files) {
    const bytes = await readFile(file);
    const contents = deserializeSafetensors(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    for (const [name, value] of contents.tensors) tensors.set(name, value);
  }
  return tensors;
}

/** Map checkpoint names onto the constructed model's state names. */
function mapCheckpoint(model: NativeModel, tensors: Map<string, Tensor>): { mapped: Map<string, Tensor>; unexpected: string[] } {
  const type = model.config.modelType;
  const own = new Set(model.stateDict().keys());
  const prefix = BASE_MODEL_PREFIX[type];
  const ownPrefixed = prefix !== undefined && [...own].some((key) => key.startsWith(`${prefix}.`));
  const mapped = new Map<string, Tensor>();
  const unexpected: string[] = [];
  for (const [original, value] of tensors) {
    let key = original;
    if (type === 'vit') key = legacyVitKey(key);
    if (prefix && !own.has(key)) {
      if (!ownPrefixed && key.startsWith(`${prefix}.`)) key = key.slice(prefix.length + 1);
      else if (ownPrefixed && own.has(`${prefix}.${key}`)) key = `${prefix}.${key}`;
    }
    if (type === 'vit' && !own.has(key)) key = legacyVitKey(key);
    if (own.has(key)) mapped.set(key, value);
    else unexpected.push(original);
  }
  return { mapped, unexpected };
}

/**
 * Load checkpoint tensors: tied groups present in the checkpoint with distinct
 * values are untied (transformers "will NOT tie them"); absent aliases share
 * the stored tensor; missing or mismatched weights raise.
 */
export function loadCheckpointState(model: NativeModel, tensors: Map<string, Tensor>): { unexpected: string[] } {
  const { mapped, unexpected } = mapCheckpoint(model, tensors);
  const groups = new Map<Parameter, string[]>();
  for (const [name, parameter] of model.namedParameters({ removeDuplicate: false })) {
    const group = groups.get(parameter);
    if (group) group.push(name);
    else groups.set(parameter, [name]);
  }
  for (const names of groups.values()) {
    const present = names.filter((name) => mapped.has(name));
    if (names.length < 2 || present.length < 2) continue;
    // Cluster stored aliases by value; the first cluster keeps the shared tensor.
    const clusters: string[][] = [];
    for (const name of present) {
      const cluster = clusters.find((items) => mapped.get(items[0]!)!.equal(mapped.get(name)!));
      if (cluster) cluster.push(name);
      else clusters.push([name]);
    }
    const byName = new Map(model.namedParameters({ removeDuplicate: false }));
    for (const cluster of clusters.slice(1)) {
      const current = byName.get(cluster[0]!)!;
      const untied = new Parameter(current.detach(), current.requiresGrad);
      for (const name of cluster) model.setParameterAt(name, untied);
    }
  }
  const state = model.stateDict();
  const missing: string[] = [];
  const mismatched: string[] = [];
  const covered = new Set<object>();
  for (const [name, value] of state) if (mapped.has(name)) covered.add(value._storage);
  for (const [name, value] of state) {
    const source = mapped.get(name);
    if (!source) {
      if (!covered.has(value._storage)) missing.push(name);
      continue;
    }
    if (source.shape.length !== value.shape.length || source.shape.some((size, index) => size !== value.shape[index])) mismatched.push(name);
  }
  if (missing.length || mismatched.length) {
    throw new ValueError(`foundation has missing or incompatible weights: ${JSON.stringify({ missing_keys: missing, mismatched_keys: mismatched })}`);
  }
  noGrad(() => {
    for (const [name, value] of state) {
      const source = mapped.get(name);
      if (!source) continue;
      const target = value.data;
      const data = source.data;
      for (let index = 0; index < target.length; index += 1) target[index] = data[index]!;
      value._storage.version += 1;
    }
  });
  return { unexpected };
}

/** ``AutoModel*.from_pretrained(source)`` for supported native architectures. */
export async function loadNativeFoundation(source: string, options: FoundationOptions = {}): Promise<LoadedFoundation> {
  const { head = 'base', addPoolingLayer, tokenizer: wantTokenizer, restoreRawTieFlags, configOverrides, ...hub } = options;
  const { path } = await resolveArtifactDirectory(source, { ...hub, allowPatterns: FOUNDATION_FILES });
  const rawConfig = parseJsonStrict(await readFile(join(path, 'config.json'), 'utf8')) as JsonObject;
  let config = NativeConfig.fromPretrainedDict({ ...rawConfig, ...(configOverrides ?? {}) }, source);
  const model = createNativeModel(config, head, addPoolingLayer === undefined ? {} : { addPoolingLayer });
  const { unexpected } = loadCheckpointState(model, await readWeights(path));
  model.eval();
  if (restoreRawTieFlags) {
    const overrides: JsonObject = {};
    for (const name of ['tie_word_embeddings', 'scale_decoder_outputs']) if (name in rawConfig) overrides[name] = rawConfig[name]!;
    if (Object.keys(overrides).length) {
      config = config.derive(overrides);
      model.config = config;
    }
  }
  let generationConfig: JsonObject | null = null;
  if (head === 'seq2seq') {
    const file = join(path, 'generation_config.json');
    generationConfig = await pathExists(file)
      ? generationConfigFromFile(parseJsonStrict(await readFile(file, 'utf8')))
      : generationConfigFromModel(config);
  }
  const tokenizer = wantTokenizer !== false && (await pathExists(join(path, 'tokenizer.json'))) ? await FastTokenizer.fromDirectory(path) : null;
  const commitHash = /[\\/]snapshots[\\/]([0-9a-f]{40})[\\/]?$/.exec(path)?.[1] ?? null;
  return { model, config, generationConfig, tokenizer, directory: path, rawConfig, unexpectedKeys: unexpected, commitHash };
}
