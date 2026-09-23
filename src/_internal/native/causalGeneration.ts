/**
 * Decoder-only ``GenerationMixin.generate`` of transformers 5.17 for owned
 * causal language models (the Idefics3/SmolVLM text model): generation
 * configuration merging and validation, the complete logits-processor and
 * stopping-criteria pipeline, greedy search, multinomial sampling, beam search
 * and beam sampling (``num_beams``, ``length_penalty``, ``early_stopping``,
 * ``num_return_sequences``), classifier-free guidance, prompt-lookup assisted
 * decoding, chunked prefill and token healing. Deprecated Hub-only modes
 * (contrastive, group/constrained beam search, DoLa) raise transformers'
 * errors. Sampling draws ``torch.multinomial`` samples from the
 * PyTorch-compatible CPU generator (``nn/random.ts``), so a seeded generator
 * reproduces Python's samples wherever the float32 probabilities agree.
 */
import { Tensor } from '../../nn/tensor.js';
import { noGrad } from '../../nn/autograd.js';
import { getDefaultGenerator, multinomialValues, type Generator } from '../../nn/random.js';
import { ValueError } from '../../errors.js';
import { deepCopy, isPlainObject, PYTHON_FLOAT_KEYS, pythonFloatRepr, pythonKindOf, setPythonNumberKind, transferPythonNumberKind, type JsonObject, type JsonValue } from '../json.js';
import { GENERATION_CONFIG_DEFAULTS } from './defaults.generated.js';
import type { LlamaLayerCache } from './llama.js';
import * as P from './logitsProcessors.js';
import type { FastTokenizer } from '../tokenizers/index.js';

const f32 = Math.fround;

// ---------------------------------------------------------------------------
// Model adapter.
// ---------------------------------------------------------------------------

/** One forward pass of a causal language model during generation. */
export interface CausalForwardInputs {
  /** Token ids fed in this pass (already sliced to the uncached suffix). */
  inputIds: number[][];
  /** 2D padding mask over cached and new positions (``null`` for none). */
  attentionMask: number[][] | null;
  /** Position ids of the fed tokens (``null`` lets the model count from its cache). */
  positionIds: number[][] | null;
  /** Per-layer key/value cache updated in place (``null`` without caching). */
  cache: LlamaLayerCache[] | null;
  /** Additional model keyword arguments (for example ``imageHiddenStates``). */
  extras: Record<string, Tensor>;
  /** Number of trailing positions whose logits are returned. */
  keep: number;
}

/** A causal language model driven by {@link generateCausal}. */
export interface CausalLanguageModel {
  /** ``config.get_text_config().vocab_size``. */
  readonly generationVocabSize: number;
  /** ``getattr(config, 'max_position_embeddings', None)`` of the top-level configuration. */
  readonly generationMaxPositions: number | null;
  /** Python class name of the configuration (for transformers' error messages). */
  readonly generationConfigClass: string;
  /** Model keyword arguments dropped after prefill when caching (transformers' multimodal inputs). */
  readonly prefillOnlyInputs: readonly string[];
  /** Model keyword arguments the forward pass accepts. */
  readonly acceptedInputs: readonly string[];
  newCache(): LlamaLayerCache[];
  /** Float32 logits ``[row][keep]`` of the last ``keep`` positions. */
  forwardLogits(inputs: CausalForwardInputs): Float32Array[][];
}

// ---------------------------------------------------------------------------
// Generation configuration.
// ---------------------------------------------------------------------------

/** ``GenerationConfig._get_default_generation_params``. */
export const GLOBAL_GENERATION_DEFAULTS: Readonly<JsonObject> = Object.freeze({
  max_length: 20, min_length: 0, do_sample: false, use_cache: true, early_stopping: false, num_beams: 1, temperature: 1.0,
  top_k: 50, top_p: 1.0, typical_p: 1.0, repetition_penalty: 1.0, length_penalty: 1.0, no_repeat_ngram_size: 0,
  encoder_no_repeat_ngram_size: 0, bad_words_ids: null, num_return_sequences: 1, output_scores: false,
  return_dict_in_generate: false, forced_bos_token_id: null, forced_eos_token_id: null, remove_invalid_values: false,
  exponential_decay_length_penalty: null, suppress_tokens: null, begin_suppress_tokens: null, epsilon_cutoff: 0.0,
  eta_cutoff: 0.0, encoder_repetition_penalty: 1.0, num_assistant_tokens: 20, num_assistant_tokens_schedule: 'constant',
  assistant_confidence_threshold: 0.4, assistant_lookbehind: 10, target_lookbehind: 10, num_beam_groups: 1,
  diversity_penalty: 0.0,
});

const CONFIG_ATTRIBUTES = new Set([...Object.keys(GENERATION_CONFIG_DEFAULTS), 'transformers_version', '_commit_hash']);

const CACHE_IMPLEMENTATIONS = [
  'static', 'offloaded_static', 'sliding_window', 'hybrid', 'hybrid_chunked', 'offloaded_hybrid', 'offloaded_hybrid_chunked',
  'dynamic', 'offloaded', 'quantized',
];

const GENERATE_ARGUMENTS = [
  'logits_processor', 'stopping_criteria', 'prefix_allowed_tokens_fn', 'synced_gpus', 'assistant_model', 'streamer',
  'negative_prompt_ids', 'negative_prompt_attention_mask',
];

/** Mutable ``GenerationConfig`` attributes (snake_case, JSON values). */
export type GenerationValues = Record<string, unknown>;

function isNone(value: unknown): boolean {
  return value === null || value === undefined;
}

/** ``GenerationConfig.validate`` (the checks that raise). */
export function validateGenerationConfig(config: GenerationValues): void {
  const early = config.early_stopping;
  if (!(isNone(early) || early === true || early === false || early === 'never')) {
    throw new ValueError(`\`early_stopping\` must be a boolean or 'never', but is ${String(early)}.`);
  }
  const maxNew = config.max_new_tokens;
  if (typeof maxNew === 'number' && maxNew <= 0) throw new ValueError(`\`max_new_tokens\` must be greater than 0, but is ${maxNew}.`);
  const weight = config.assistant_ensemble_weight;
  if (typeof weight === 'number' && !(weight > 0 && weight < 1)) {
    throw new ValueError(`\`assistant_ensemble_weight\` must be in the open interval \`(0.0, 1.0)\`, but is ${P.pyFloat(weight)}. Use \`None\` for standard (lossless) speculative decoding.`);
  }
  const cache = config.cache_implementation;
  const valid = [...CACHE_IMPLEMENTATIONS, 'paged'];
  if (!isNone(cache) && !valid.includes(cache as string)) {
    throw new ValueError(`Invalid \`cache_implementation\` (${String(cache)}). Choose one of: (${valid.map((item) => `'${item}'`).join(', ')})`);
  }
  if (!isNone(config.compile_config)) {
    const kind = Array.isArray(config.compile_config) ? 'list' : typeof config.compile_config === 'object' ? 'dict' : typeof config.compile_config;
    throw new ValueError(`You provided \`compile_config\` as an instance of <class '${kind}'>, but it must be an instance of \`CompileConfig\`.`);
  }
  const returns = config.num_return_sequences;
  if (typeof returns === 'number' && returns > 1) {
    const beams = config.num_beams;
    if (isNone(beams) || beams === 1) {
      if (!config.do_sample) {
        throw new ValueError(`Greedy methods (do_sample != True) without beam search do not support \`num_return_sequences\` different than 1 (got ${returns}).`);
      }
    } else if (typeof beams === 'number' && returns > beams) {
      throw new ValueError(`\`num_return_sequences\` (${returns}) has to be smaller or equal to \`num_beams\` (${beams}).`);
    }
  }
  if (Array.isArray(config.suppress_tokens)) {
    const suppressed = new Set(config.suppress_tokens as number[]);
    for (const attribute of ['forced_bos_token_id', 'forced_eos_token_id']) {
      const forced = config[attribute];
      if (isNone(forced)) continue;
      const tokens = new Set(Array.isArray(forced) ? forced as number[] : [forced as number]);
      if (tokens.size && [...tokens].every((token) => suppressed.has(token))) {
        throw new ValueError(`Every token in \`${attribute}\` (${P.pyRepr([...tokens].sort((a, b) => a - b))}) is also in \`suppress_tokens\`. Forcing a token while suppressing it sets all logits to \`-inf\` at the forcing step, which produces \`nan\` probabilities and crashes generation. Remove the overlapping token(s) from either \`${attribute}\` or \`suppress_tokens\` (if you meant to prevent an early EOS token, use \`min_new_tokens\` instead).`);
      }
    }
  }
  for (const argument of GENERATE_ARGUMENTS) {
    if (argument in config) {
      throw new ValueError(`Argument \`${argument}\` is not a valid argument of \`GenerationConfig\`. It should be passed to \`generate()\` (or a pipeline) directly.`);
    }
  }
}

function normalizeWatermark(config: GenerationValues): void {
  const value = config.watermarking_config;
  if (isPlainObject(value)) config.watermarking_config = P.watermarkingFromDict(value) as unknown as JsonObject;
}

/**
 * ``GenerationConfig.from_dict(dict)``: the serialized configuration with its
 * attributes, custom entries kept, watermarking parsed and validated (raises
 * transformers' errors for invalid settings).
 */
export function generationConfigFromDict(dict: Record<string, unknown> | null | undefined): GenerationValues {
  const config: GenerationValues = {};
  for (const key of CONFIG_ATTRIBUTES) config[key] = null;
  for (const [key, value] of Object.entries(dict ?? {})) {
    config[key] = deepCopy(value as JsonValue);
    // A whole number keeps the Python kind it was read with (``2`` vs ``2.0``).
    transferPythonNumberKind(config, key, dict!, key);
  }
  normalizeWatermark(config);
  validateGenerationConfig(config);
  return config;
}

/**
 * ``GenerationMixin._prepare_generation_config(None, **kwargs)``: a fresh
 * configuration completed from the model's generation configuration, then the
 * global defaults, then overridden by the call's keyword arguments. Returns the
 * configuration and the keyword arguments that are not generation settings.
 */
export function prepareGenerationConfig(
  model: Record<string, unknown> | null | undefined, kwargs: Record<string, unknown>,
): { config: GenerationValues; modelKwargs: Record<string, unknown> } {
  const config: GenerationValues = {};
  for (const key of CONFIG_ATTRIBUTES) config[key] = null;
  const update = (values: Record<string, unknown>, defaultsOnly: boolean, allowCustom: boolean): Record<string, unknown> => {
    const unused: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(values)) {
      if (allowCustom && !(key in config)) {
        config[key] = deepCopy(value as JsonValue);
        transferPythonNumberKind(config, key, values, key);
      } else if (key in config) {
        if (!defaultsOnly || isNone(config[key])) {
          if (key === 'watermarking_config' && isPlainObject(value)) config[key] = P.watermarkingFromDict(value);
          else config[key] = value instanceof Map ? new Map(value) : deepCopy(value as JsonValue);
          transferPythonNumberKind(config, key, values, key);
        }
      } else unused[key] = value;
    }
    validateGenerationConfig(config);
    return unused;
  };
  update(model ?? {}, true, true);
  update(GLOBAL_GENERATION_DEFAULTS as Record<string, unknown>, true, false);
  const modelKwargs = update(kwargs, false, false);
  if (config.output_attentions) modelKwargs.output_attentions = true;
  if (config.output_hidden_states) modelKwargs.output_hidden_states = true;
  return { config, modelKwargs };
}

type Mode = 'greedy' | 'sample' | 'beam' | 'beam_sample' | 'assisted' | 'contrastive' | 'group_beam' | 'constrained' | 'dola';

/** ``GenerationConfig.get_generation_mode`` (no assistant model). */
function generationMode(config: GenerationValues): Mode {
  let mode: Mode;
  const beams = config.num_beams as number | null;
  if (!isNone(config.constraints) || !isNone(config.force_words_ids)) mode = 'constrained';
  else if (isNone(beams) || beams === 1) {
    if (config.do_sample !== true) {
      const topK = config.top_k as number | null;
      const alpha = config.penalty_alpha as number | null;
      mode = topK !== null && topK > 1 && alpha !== null && alpha > 0 ? 'contrastive' : 'greedy';
    } else mode = 'sample';
  } else if (typeof config.num_beam_groups === 'number' && config.num_beam_groups > 1) mode = 'group_beam';
  else mode = config.do_sample === true ? 'beam_sample' : 'beam';
  if (config.use_mtp || !isNone(config.prompt_lookup_num_tokens) || !isNone(config.assistant_early_exit)) {
    if (mode === 'greedy' || mode === 'sample') mode = 'assisted';
  }
  if (!isNone(config.dola_layers) && (mode === 'greedy' || mode === 'sample')) mode = 'dola';
  return mode;
}

const DEPRECATED_MODES: Partial<Record<Mode, [string, string]>> = {
  contrastive: ['Contrastive Search', 'transformers-community/contrastive-search'],
  group_beam: ['Group Beam Search', 'transformers-community/group-beam-search'],
  constrained: ['Constrained Beam Search', 'transformers-community/constrained-beam-search'],
  dola: ['Dola Generation', 'transformers-community/dola'],
};

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

export interface CausalGenerateInputs {
  /** ``input_ids`` ``[batch, length]``. */
  inputIds: Tensor | readonly (readonly number[])[];
  attentionMask?: Tensor | readonly (readonly number[])[] | null;
  /** Extra model keyword arguments (camelCase names of the adapter). */
  extras?: Record<string, Tensor | null | undefined>;
}

export interface GenerationStreamer {
  put(ids: number[][]): void;
  end(): void;
}

export interface CausalGenerateOptions {
  /** The model's ``generation_config.to_dict()``. */
  generationConfig?: Record<string, unknown> | null;
  /** Generation keyword arguments of the call (``max_new_tokens``, ``num_beams``, ...). */
  settings?: Record<string, unknown>;
  /** Tokenizer for ``stop_strings`` and ``token_healing``. */
  tokenizer?: FastTokenizer | null;
  negativePromptIds?: Tensor | readonly (readonly number[])[] | null;
  negativePromptAttentionMask?: Tensor | readonly (readonly number[])[] | null;
  /** Custom processors appended after the configured ones (``logits_processor``). */
  logitsProcessor?: P.LogitsProcessorFn[];
  /** Custom stopping criteria (``stopping_criteria``). */
  stoppingCriteria?: P.StoppingCriterionFn[];
  prefixAllowedTokensFn?: ((batchId: number, ids: readonly number[]) => readonly number[]) | null;
  streamer?: GenerationStreamer | null;
  /** Generator for sampling (default: the global generator, like ``torch.manual_seed``). */
  generator?: Generator;
  /** Wall clock in seconds for ``max_time`` (tests may inject one). */
  clock?: () => number;
}

export interface CausalGenerateOutput {
  /** Prompt followed by generated ids, ``[batch * num_return_sequences][length]``. */
  sequences: number[][];
  /** Processed scores per generated step (``output_scores``). */
  scores: Float32Array[][] | null;
  /** Raw logits per generated step (``output_logits``). */
  logits: Float32Array[][] | null;
  /** Beam search: final hypothesis scores (``output_scores``). */
  sequencesScores: number[] | null;
  /** Beam search: beam index of every generated token (``output_logits``). */
  beamIndices: number[][] | null;
  /** The resolved generation configuration. */
  config: GenerationValues;
}

function rows(value: Tensor | readonly (readonly number[])[] | null | undefined): number[][] | null {
  if (value === null || value === undefined) return null;
  if (value instanceof Tensor) {
    if (value.ndim !== 2) throw new ValueError('`attention_mask` passed to `generate` must be 2D.');
    const [batch, length] = value.shape as [number, number];
    return Array.from({ length: batch }, (_, b) => Array.from(value.data.subarray(b * length, (b + 1) * length), Number));
  }
  return value.map((row) => [...row]);
}

function repeatInterleave<T>(values: T[], times: number): T[] {
  if (times === 1) return values;
  return values.flatMap((value) => Array.from({ length: times }, () => (Array.isArray(value) ? [...value] as T : value)));
}

function repeatTensor(value: Tensor, times: number): Tensor {
  if (times === 1) return value;
  const indices: number[] = [];
  for (let row = 0; row < value.shape[0]!; row += 1) for (let copy = 0; copy < times; copy += 1) indices.push(row);
  return value.indexSelect(0, indices);
}

function cacheLength(cache: LlamaLayerCache[] | null): number {
  return cache?.[0]?.key?.shape[2] ?? 0;
}

function reorderCache(cache: LlamaLayerCache[], indices: readonly number[]): void {
  for (const layer of cache) {
    if (layer.key) layer.key = layer.key.indexSelect(0, indices);
    if (layer.value) layer.value = layer.value.indexSelect(0, indices);
  }
}

function cropCache(cache: LlamaLayerCache[], remove: number): void {
  if (remove <= 0) return;
  for (const layer of cache) {
    if (layer.key) layer.key = layer.key.slice(2, 0, layer.key.shape[2]! - remove);
    if (layer.value) layer.value = layer.value.slice(2, 0, layer.value.shape[2]! - remove);
  }
}

/**
 * ``torch.multinomial(probabilities, samples, replacement)`` over float32 rows,
 * drawing from the PyTorch-compatible CPU generator.
 */
function multinomial(probabilities: readonly Float32Array[], samples: number, generator: Generator): number[][] {
  const categories = probabilities[0]?.length ?? 0;
  const flat = new Float32Array(probabilities.length * categories);
  probabilities.forEach((row, index) => flat.set(row, index * categories));
  let values: Float64Array;
  try {
    values = multinomialValues(flat, 'float32', probabilities.length, categories, samples, false, generator);
  } catch (error) {
    if (error instanceof RangeError) throw new P.RuntimeError(error.message);
    throw error;
  }
  return probabilities.map((_, row) => Array.from(values.subarray(row * samples, (row + 1) * samples)));
}

function argmax(row: Float32Array): number {
  let best = 0;
  for (let index = 1; index < row.length; index += 1) {
    const value = row[index]!;
    if (value > row[best]! || (Number.isNaN(value) && !Number.isNaN(row[best]!))) best = index;
  }
  return best;
}

interface Special {
  eos: number[] | null;
  pad: number | null;
  bos: number | null;
}

function specialTokens(config: GenerationValues): Special {
  const list = (value: unknown): number[] | null => (isNone(value) ? null : Array.isArray(value) ? value.map(Number) : [Number(value)]);
  const eos = list(config.eos_token_id);
  let pad = isNone(config.pad_token_id) ? null : Number(config.pad_token_id);
  if (pad === null && eos !== null) pad = eos[0]!;
  return { eos, pad, bos: isNone(config.bos_token_id) ? null : Number(config.bos_token_id) };
}

/** Mutable model keyword arguments of a generation loop. */
interface LoopState {
  mask: number[][];
  positions: number[][];
  cache: LlamaLayerCache[] | null;
  extras: Record<string, Tensor>;
}

class Runner {
  readonly model: CausalLanguageModel;
  readonly config: GenerationValues;
  readonly useCache: boolean;
  readonly special: Special;
  readonly generator: Generator;
  readonly options: CausalGenerateOptions;
  processors: P.LogitsProcessorFn[] = [];
  criteria: P.StoppingCriterionFn[] = [];
  readonly scores: Float32Array[][] | null;
  readonly rawLogits: Float32Array[][] | null;

  constructor(model: CausalLanguageModel, config: GenerationValues, special: Special, options: CausalGenerateOptions) {
    this.model = model;
    this.config = config;
    this.useCache = config.use_cache !== false;
    this.special = special;
    this.generator = options.generator ?? getDefaultGenerator();
    this.options = options;
    const dict = config.return_dict_in_generate === true;
    this.scores = dict && config.output_scores === true ? [] : null;
    this.rawLogits = dict && config.output_logits === true ? [] : null;
  }

  process(ids: readonly (readonly number[])[], scores: Float32Array[]): Float32Array[] {
    let result = scores;
    for (const processor of this.processors) result = processor(ids, result);
    return result;
  }

  stopped(ids: readonly (readonly number[])[]): boolean[] {
    const done = ids.map(() => false);
    for (const criterion of this.criteria) criterion(ids).forEach((flag, index) => { done[index] = done[index]! || flag; });
    return done;
  }

  /** ``prepare_inputs_for_generation`` + forward for the uncached suffix (or everything). */
  forward(ids: number[][], state: LoopState, first: boolean, fed: number | null, keep = 1): Float32Array[][] {
    const length = ids[0]!.length;
    const count = fed ?? length;
    const inputIds = ids.map((row) => row.slice(length - count));
    const positions = state.positions.map((row) => row.slice(row.length - count));
    const extras: Record<string, Tensor> = {};
    for (const [key, value] of Object.entries(state.extras)) {
      if (this.model.prefillOnlyInputs.includes(key) && !first && this.useCache) continue;
      extras[key] = value;
    }
    return this.model.forwardLogits({
      inputIds, attentionMask: state.mask, positionIds: positions, cache: state.cache, extras, keep,
    });
  }

  /** ``_prefill``, including chunked prefill (``prefill_chunk_size``). */
  prefill(ids: number[][], state: LoopState, first: boolean, keep = 1): Float32Array[][] {
    const chunk = this.config.prefill_chunk_size;
    if (isNone(chunk)) return this.forward(ids, state, first, null, keep);
    if (!state.cache) throw new ValueError('Cannot use prefill chunking without a cache');
    const size = chunk as number;
    const length = ids[0]!.length;
    const fullMask = state.mask;
    const fullPositions = state.positions;
    let result: Float32Array[][] = [];
    for (let start = 0; start < length; start += size) {
      const end = Math.min(length, start + size);
      const chunkIds = ids.map((row) => row.slice(start, end));
      const extras: Record<string, Tensor> = {};
      // ``prepare_inputs_for_generation`` runs without ``is_first_iteration`` for each chunk.
      for (const [key, value] of Object.entries(state.extras)) if (!this.model.prefillOnlyInputs.includes(key)) extras[key] = value;
      result = this.model.forwardLogits({
        inputIds: chunkIds, attentionMask: fullMask.map((row) => row.slice(0, end)),
        positionIds: fullPositions.map((row) => row.slice(start, end)), cache: state.cache, extras, keep,
      });
    }
    return result;
  }

  static advance(state: LoopState, count = 1): void {
    for (const row of state.mask) for (let index = 0; index < count; index += 1) row.push(1);
    for (const row of state.positions) {
      const last = row[row.length - 1]!;
      for (let index = 0; index < count; index += 1) row.push(last + 1 + index);
    }
  }
}

/** ``_sample``: greedy search or multinomial sampling. */
function sampleLoop(runner: Runner, ids: number[][], state: LoopState): number[][] {
  const { special, config } = runner;
  const doSample = config.do_sample === true;
  const hasEos = special.eos !== null;
  const unfinished = ids.map(() => true);
  const streamer = runner.options.streamer ?? null;
  let logits = runner.prefill(ids, state, config.is_assistant !== true);
  for (;;) {
    Runner.advance(state);
    const last = logits.map((row) => Float32Array.from(row[row.length - 1]!));
    const scores = runner.process(ids, last.map((row) => Float32Array.from(row)));
    runner.scores?.push(scores);
    runner.rawLogits?.push(last);
    const next = doSample
      ? multinomial(scores.map(P.softmaxRow), 1, runner.generator).map((row) => row[0]!)
      : scores.map(argmax);
    const tokens = next.map((token, index) => (hasEos && !unfinished[index] ? special.pad! : token));
    tokens.forEach((token, index) => ids[index]!.push(token));
    const done = runner.stopped(ids);
    done.forEach((flag, index) => { if (flag) unfinished[index] = false; });
    streamer?.put(tokens.map((token) => [token]));
    if (!unfinished.some(Boolean)) break;
    logits = runner.forward(ids, state, false, runner.useCache ? 1 : null);
  }
  return ids;
}

interface BeamResult {
  sequences: number[][];
  sequencesScores: number[];
  beamIndices: number[][];
}

/** ``_beam_search`` (beam search and beam sampling). */
function beamLoop(runner: Runner, flatIds: number[][], state: LoopState): BeamResult {
  const { config, special } = runner;
  const beams = config.num_beams as number;
  const returns = config.num_return_sequences as number;
  const doSample = config.do_sample === true;
  const early = config.early_stopping as boolean | 'never';
  const lp = config.length_penalty as number;
  const maxLength = config.max_length as number;
  const batch = flatIds.length / beams;
  const vocab = runner.model.generationVocabSize;
  const promptLength = flatIds[0]!.length;
  let curLen = promptLength;
  const eosCount = special.eos?.length ?? 0;
  const keep = Math.max(2, 1 + eosCount) * beams;
  if (config.low_memory === true) {
    throw new ValueError('`low_memory=True` is not supported after the beam search refactor. Please check the discussion in #35802 *after the PR got merged*, and add a comment there if your questions are not yet answered.');
  }
  const fill = special.eos !== null ? (special.pad || special.eos[0]!) : -1;
  let running: number[][][] = Array.from({ length: batch }, (_, b) => Array.from({ length: beams }, (_, k) => [...flatIds[b * beams + k]!]));
  let sequences: number[][][] = running.map((group) => group.map((row) => [...row]));
  let runningScores: number[][] = Array.from({ length: batch }, () => Array.from({ length: beams }, (_, k) => (k === 0 ? 0 : -1e9)));
  let beamScores: number[][] = Array.from({ length: batch }, () => new Array<number>(beams).fill(-1e9));
  let finished: boolean[][] = Array.from({ length: batch }, () => new Array<boolean>(beams).fill(false));
  const unsatisfied: boolean[] = new Array<boolean>(batch).fill(true);
  let hits: boolean[][] = Array.from({ length: batch }, () => new Array<boolean>(keep).fill(false));
  let runningIndices: number[][][] = Array.from({ length: batch }, () => Array.from({ length: beams }, () => [] as number[]));
  let beamIndices: number[][][] = runningIndices.map((group) => group.map(() => [] as number[]));
  const flatRunning = (): number[][] => running.flat().map((row) => [...row]);
  let logits = runner.prefill(flatIds, state, config.is_assistant !== true);
  let consumed = false;
  for (;;) {
    if (consumed) {
      const ids = flatRunning();
      logits = runner.forward(ids, state, false, runner.useCache ? 1 : null);
    }
    consumed = true;
    Runner.advance(state);
    const raw = logits.map((row) => Float32Array.from(row[row.length - 1]!));
    const flat = flatRunning();
    const logProbs = runner.process(flat, raw.map(P.logSoftmaxRow));
    runner.rawLogits?.push(raw);
    runner.scores?.push(logProbs);
    const nextRunning: number[][][] = [];
    const nextScores: number[][] = [];
    const nextIndices: number[][][] = [];
    const reorder: number[] = [];
    const accumulatedRows = Array.from({ length: batch }, (_, b) => {
      const accumulated = new Float32Array(beams * vocab);
      for (let k = 0; k < beams; k += 1) {
        const row = logProbs[b * beams + k]!;
        const base = runningScores[b]![k]!;
        for (let v = 0; v < vocab; v += 1) accumulated[k * vocab + v] = f32(row[v]! + base);
      }
      return accumulated;
    });
    // ``torch.multinomial(softmax(accumulated), beams_to_keep)`` draws for the whole batch at once.
    const sampledPerBatch = doSample ? multinomial(accumulatedRows.map(P.softmaxRow), keep, runner.generator) : null;
    for (let b = 0; b < batch; b += 1) {
      const accumulated = accumulatedRows[b]!;
      let chosen: number[];
      if (doSample) {
        chosen = sampledPerBatch![b]!;
      } else {
        const order = Array.from(accumulated.keys());
        order.sort((x, y) => (accumulated[y]! - accumulated[x]!) || (x - y));
        chosen = order.slice(0, keep);
      }
      const topScores = chosen.map((index) => accumulated[index]!);
      const topSequences = chosen.map((index) => [...running[b]![Math.floor(index / vocab)]!, index % vocab]);
      const topIndices = chosen.map((index) => {
        const beam = Math.floor(index / vocab);
        const indices = [...runningIndices[b]![beam]!];
        indices[curLen - promptLength] = beam + b * beams;
        return indices;
      });
      const stops = runner.stopped(topSequences);
      hits[b] = stops;
      // Running beams for the next iteration.
      const runningLogProbs = topScores.map((score, j) => f32(score + (stops[j] ? -1e9 : 0)));
      const nextOrder = Array.from(runningLogProbs.keys()).sort((x, y) => (runningLogProbs[y]! - runningLogProbs[x]!) || (x - y)).slice(0, beams);
      nextRunning.push(nextOrder.map((j) => topSequences[j]!));
      nextScores.push(nextOrder.map((j) => runningLogProbs[j]!));
      nextIndices.push(nextOrder.map((j) => topIndices[j]!));
      // Completed hypotheses.
      const full = finished[b]!.every(Boolean) && early === true;
      const denominator = f32((curLen + 1 - promptLength) ** lp);
      const merged = [
        ...sequences[b]!.map((sequence, j) => ({ sequence, score: beamScores[b]![j]!, indices: beamIndices[b]![j]!, flag: finished[b]![j]! })),
        ...topSequences.map((sequence, j) => {
          const just = stops[j]! && j < beams;
          let score = f32(topScores[j]! / denominator);
          score = f32(score + (full ? -1e9 : 0));
          score = f32(score + (unsatisfied[b] ? 0 : -1e9));
          score = f32(score + (just ? 0 : -1e9));
          return { sequence, score, indices: topIndices[j]!, flag: just };
        }),
      ];
      const best = Array.from(merged.keys()).sort((x, y) => (merged[y]!.score - merged[x]!.score) || (x - y)).slice(0, beams);
      sequences[b] = best.map((j) => merged[j]!.sequence);
      beamScores[b] = best.map((j) => merged[j]!.score);
      beamIndices[b] = best.map((j) => merged[j]!.indices);
      finished[b] = best.map((j) => merged[j]!.flag);
    }
    running = nextRunning;
    runningScores = nextScores;
    runningIndices = nextIndices;
    for (let b = 0; b < batch; b += 1) for (const indices of running[b]!.map((_, k) => runningIndices[b]![k]!)) reorder.push(indices[curLen - promptLength]!);
    if (state.cache) reorderCache(state.cache, reorder);
    curLen += 1;
    // ``_check_early_stop_heuristic``.
    for (let b = 0; b < batch; b += 1) {
      const bestLength = early === 'never' && lp > 0 ? maxLength - promptLength : curLen - promptLength;
      const bestPossible = f32(runningScores[b]![0]! / f32(bestLength ** lp));
      const worst = Math.min(...beamScores[b]!);
      const improvable = finished[b]!.some((flag) => bestPossible > (flag ? worst : -1e9));
      unsatisfied[b] = unsatisfied[b]! && improvable;
    }
    const improvementPossible = unsatisfied.some(Boolean);
    const openBeam = !(finished.every((group) => group.every(Boolean)) && early === true);
    const validContinuations = !hits.every((group) => group.every(Boolean));
    if (!(improvementPossible && openBeam && validContinuations)) break;
  }
  const outSequences: number[][] = [];
  const outScores: number[] = [];
  const outIndices: number[][] = [];
  for (let b = 0; b < batch; b += 1) {
    for (let j = 0; j < returns; j += 1) {
      outSequences.push(sequences[b]![j]!);
      outScores.push(beamScores[b]![j]!);
      outIndices.push(beamIndices[b]![j]!);
    }
  }
  const generated = Math.max(...outIndices.map((indices) => indices.filter((value) => value !== undefined).length));
  const width = promptLength + generated;
  return {
    sequences: outSequences.map((sequence) => {
      const row = sequence.slice(0, width);
      while (row.length < width) row.push(fill);
      return row;
    }),
    sequencesScores: outScores,
    beamIndices: outIndices.map((indices) => {
      const row = Array.from({ length: generated }, (_, index) => indices[index] ?? -1);
      return row;
    }),
  };
}

/** ``PromptLookupCandidateGenerator.get_candidates``. */
function promptLookupCandidates(runner: Runner, ids: number[]): number[] {
  const { config } = runner;
  const outputTokens = config.prompt_lookup_num_tokens as number;
  const maxNgram = (config.max_matching_ngram_size as number | null) || 2;
  const maxLength = config.max_length as number;
  if (maxNgram <= 0 || outputTokens <= 0) throw new ValueError('Invalid max_matching_ngram_size or num_output_tokens');
  const length = ids.length;
  if (maxLength === length + 1) return ids;
  let chosen: number[] | null = null;
  let found = false;
  for (let size = Math.min(maxNgram, length - 1); size > 0; size -= 1) {
    const ngram = ids.slice(length - size);
    for (let index = 0; index + size <= length; index += 1) {
      let matches = true;
      for (let offset = 0; offset < size; offset += 1) if (ids[index + offset] !== ngram[offset]) { matches = false; break; }
      if (!matches) continue;
      const start = index + size;
      const end = Math.min(start + outputTokens, length, maxLength);
      if (start >= end) continue;
      chosen = ids.slice(start, end);
      if (runner.processors.length) {
        let sequence = ids;
        const fake = [new Float32Array(runner.model.generationVocabSize).fill(1)];
        for (let position = 0; position < chosen.length; position += 1) {
          const value = runner.process([sequence], fake.map((row) => Float32Array.from(row)))[0]![chosen[position]!]!;
          if (value === -Infinity || value === -3.4028234663852886e38) {
            chosen = chosen.slice(0, position);
            break;
          }
          sequence = [...ids, ...chosen.slice(0, position + 1)];
        }
        if (chosen.length === 0) continue;
      }
      found = true;
      if (runner.special.eos !== null) {
        const eos = runner.special.eos;
        const first = chosen.findIndex((token) => eos.includes(token));
        if (first >= 0) chosen = chosen.slice(0, first);
      }
      break;
    }
    if (found) break;
  }
  if (!found || chosen === null || chosen.length === 0) return ids;
  return [...ids, ...chosen];
}

/** ``_assisted_decoding`` with prompt-lookup candidates. */
function assistedLoop(runner: Runner, ids: number[][], state: LoopState): number[][] {
  const { config, model } = runner;
  if (!runner.useCache) throw new ValueError('assisted generate requires `use_cache=True`');
  if (['static', 'hybrid', 'sliding_window'].includes(config.cache_implementation as string)) {
    throw new ValueError('assisted generate is not supported with Static cache classes`');
  }
  if (!isNone(config.assistant_early_exit)) {
    throw new P.AttributeError(`'${model.generationConfigClass}' object has no attribute 'num_hidden_layers'`);
  }
  if (isNone(config.prompt_lookup_num_tokens)) {
    throw new ValueError('Could not find `num_mtp_layers` in the model config. This model probably has no associated mtp weights.');
  }
  if (!isNone(config.assistant_ensemble_weight)) {
    throw new ValueError('Setting `assistant_ensemble_weight` requires candidate logits from the assistant model. It is not supported with prompt lookup decoding.');
  }
  if (ids.length > 1) throw new ValueError('assisted generate is only supported for batch_size = 1');
  const doSample = config.do_sample === true;
  const maxLength = config.max_length as number;
  const sequence = ids[0]!;
  const streamer = runner.options.streamer ?? null;
  let first = true;
  let unfinished = true;
  while (unfinished) {
    const curLen = sequence.length;
    const candidate = promptLookupCandidates(runner, sequence);
    const candidateLength = candidate.length - curLen;
    const doneCandidate = runner.stopped([candidate])[0]!;
    const candidateState: LoopState = {
      mask: state.mask.map((row) => [...row]), positions: state.positions.map((row) => [...row]), cache: state.cache, extras: state.extras,
    };
    Runner.advance(candidateState, candidateLength);
    const logits = runner.forward([candidate], candidateState, first, first ? null : candidateLength + 1, candidateLength + 1);
    const newLogits: Float32Array[] = [];
    for (let position = 0; position <= candidateLength; position += 1) {
      const raw = Float32Array.from(logits[0]![position]!);
      newLogits.push(runner.process([candidate.slice(0, curLen + position)], [raw])[0]!);
    }
    const selected = doSample ? multinomial(newLogits.map(P.softmaxRow), 1, runner.generator).map((row) => row[0]!) : newLogits.map(argmax);
    let matches = 0;
    while (matches < candidateLength && candidate[curLen + matches] === selected[matches]) matches += 1;
    if (doneCandidate && matches === candidateLength) matches -= 1;
    let valid = selected.slice(0, matches + 1);
    const budget = maxLength - curLen;
    if (valid.length > budget) {
      valid = valid.slice(0, budget);
      matches = valid.length - 1;
    }
    sequence.push(...valid);
    streamer?.put([valid]);
    if (state.cache) cropCache(state.cache, candidateLength - matches);
    Runner.advance(state, matches + 1);
    for (let index = 0; index <= matches; index += 1) {
      runner.scores?.push([newLogits[index]!]);
      runner.rawLogits?.push([Float32Array.from(logits[0]![index]!)]);
    }
    unfinished = !runner.stopped([sequence])[0];
    first = false;
  }
  return [sequence];
}

/** Unbatched classifier-free guidance (``UnbatchedClassifierFreeGuidanceLogitsProcessor``). */
function guidanceProcessor(runner: Runner, scale: number, negativeIds: number[][] | null, negativeMask: number[][] | null): P.LogitsProcessorFn {
  const { model } = runner;
  const context = {
    ids: negativeIds, mask: negativeMask, cache: runner.useCache ? model.newCache() : null as LlamaLayerCache[] | null, first: true,
  };
  const unconditional = (ids: readonly (readonly number[])[]): Float32Array[] => {
    let input: number[][];
    let mask: number[][];
    if (context.first) {
      context.ids ??= ids.map((row) => [row[row.length - 1]!]);
      context.mask ??= context.ids.map((row) => row.map(() => 1));
      input = context.ids;
      mask = context.mask;
      context.first = false;
    } else {
      mask = context.mask!.map((row) => [...row, 1]);
      input = runner.useCache
        ? ids.map((row) => [row[row.length - 1]!])
        : context.ids!.map((row, index) => [...row, ids[index]![ids[index]!.length - 1]!]);
      context.ids = input;
      context.mask = mask;
    }
    const logits = model.forwardLogits({ inputIds: input, attentionMask: mask, positionIds: null, cache: context.cache, extras: {}, keep: 1 });
    return logits.map((row) => row[row.length - 1]!);
  };
  const g = f32(scale);
  return (ids, scores) => {
    const conditional = scores.map(P.logSoftmaxRow);
    const other = unconditional(ids).map(P.logSoftmaxRow);
    return conditional.map((row, index) => {
      const u = other[index]!;
      return row.map((value, token) => f32(f32(g * f32(value - u[token]!)) + u[token]!));
    });
  };
}

function buildProcessors(
  runner: Runner, promptLength: number, encoderIds: number[][], negative: { ids: number[][] | null; mask: number[][] | null },
): P.LogitsProcessorFn[] {
  const { config, special, options } = runner;
  const list: P.LogitsProcessorFn[] = [];
  const number = (key: string): number | null => (isNone(config[key]) ? null : Number(config[key]));
  // transformers' processors check isinstance(value, float) / isinstance(value, int): a whole
  // number recorded with the other Python kind (read from a file, or float()/int()) fails there.
  const kind = (key: string): 'int' | 'float' | null => {
    const value = config[key];
    if (typeof value !== 'number') return null;
    if (!Number.isInteger(value)) return 'float';
    return pythonKindOf(value, config, key) ?? (PYTHON_FLOAT_KEYS.has(key) ? 'float' : 'int');
  };
  const requireFloat = (key: string, message: (shown: string) => string): void => {
    if (kind(key) === 'int') throw new ValueError(message(String(number(key))));
  };
  const requireInt = (key: string, message: (shown: string) => string): void => {
    if (kind(key) === 'float') throw new ValueError(message(pythonFloatRepr(number(key)!)));
  };
  const guidance = number('guidance_scale');
  if (guidance !== null && guidance !== 1) list.push(guidanceProcessor(runner, guidance, negative.ids, negative.mask));
  if (!isNone(config.sequence_bias)) list.push(P.sequenceBiasProcessor(P.normalizeSequenceBias(config.sequence_bias)));
  const encoderPenalty = number('encoder_repetition_penalty');
  if (encoderPenalty !== null && encoderPenalty !== 1) requireFloat('encoder_repetition_penalty', (shown) => `\`penalty\` has to be a strictly positive float, but is ${shown}`);
  if (encoderPenalty !== null && encoderPenalty !== 1) list.push(P.encoderRepetitionPenaltyProcessor(encoderPenalty, encoderIds));
  const penalty = number('repetition_penalty');
  if (penalty !== null && penalty !== 1) {
    requireFloat('repetition_penalty', (shown) => `\`penalty\` has to be a strictly positive float, but is ${shown}`);
    list.push(P.repetitionPenaltyProcessor(penalty));
  }
  const ngram = number('no_repeat_ngram_size');
  if (ngram !== null && ngram > 0) requireInt('no_repeat_ngram_size', (shown) => `\`ngram_size\` has to be a strictly positive integer, but is ${shown}`);
  if (ngram !== null && ngram > 0) list.push(P.noRepeatNGramProcessor(ngram));
  const encoderNgram = number('encoder_no_repeat_ngram_size');
  if (encoderNgram !== null && encoderNgram > 0) requireInt('encoder_no_repeat_ngram_size', (shown) => `\`encoder_ngram_size\` has to be a strictly positive integer, but is ${shown}`);
  if (encoderNgram !== null && encoderNgram > 0) list.push(P.encoderNoRepeatNGramProcessor(encoderNgram, encoderIds));
  if (!isNone(config.bad_words_ids)) list.push(P.noBadWordsProcessor(config.bad_words_ids, special.eos));
  const minLength = number('min_length');
  if (minLength !== null && special.eos !== null && minLength > 0) requireInt('min_length', (shown) => `\`min_length\` has to be a non-negative integer, but is ${shown}`);
  if (minLength !== null && special.eos !== null && minLength > 0) list.push(P.minLengthProcessor(minLength, special.eos));
  const minNew = number('min_new_tokens');
  if (minNew !== null && special.eos !== null && minNew > 0) requireInt('min_new_tokens', (shown) => `\`min_new_tokens\` has to be a positive integer, but is ${shown}`);
  if (minNew !== null && special.eos !== null && minNew > 0) list.push(P.minNewTokensProcessor(promptLength, minNew, special.eos));
  if (options.prefixAllowedTokensFn) list.push(P.prefixConstrainedProcessor(options.prefixAllowedTokensFn, config.num_beams as number));
  const forcedBos = number('forced_bos_token_id');
  if (forcedBos !== null) list.push(P.forcedBosProcessor(forcedBos));
  if (!isNone(config.forced_eos_token_id)) {
    const forced = config.forced_eos_token_id;
    list.push(P.forcedEosProcessor(config.max_length as number, Array.isArray(forced) ? forced.map(Number) : [Number(forced)]));
  }
  if (config.remove_invalid_values === true) list.push(P.infNanRemoveProcessor());
  if (!isNone(config.exponential_decay_length_penalty)) {
    list.push(P.exponentialDecayLengthPenalty(config.exponential_decay_length_penalty, special.eos, promptLength));
  }
  if (!isNone(config.suppress_tokens)) list.push(P.suppressTokensProcessor((config.suppress_tokens as number[]).map(Number)));
  if (!isNone(config.begin_suppress_tokens)) {
    const begin = promptLength > 1 || forcedBos === null ? promptLength : promptLength + 1;
    list.push(P.suppressTokensAtBeginProcessor((config.begin_suppress_tokens as number[]).map(Number), begin));
  }
  list.push(...(options.logitsProcessor ?? []));
  if (config.do_sample === true) {
    const beams = config.num_beams as number | null;
    const minKeep = beams !== null && beams > 1 ? (special.eos?.length ?? 1) + 1 : 1;
    const temperature = number('temperature');
    if (temperature !== null && temperature !== 1) {
      requireFloat('temperature', (shown) => `\`temperature\` (=${shown}) has to be a strictly positive float, otherwise your next token scores will be invalid.`);
      list.push(P.temperatureWarper(temperature));
    }
    if (!isNone(config.top_h)) list.push(P.topHWarper(Number(config.top_h)));
    const topK = number('top_k');
    if (topK !== null && topK !== 0) {
      requireInt('top_k', (shown) => `\`top_k\` has to be a strictly positive integer, but is ${shown}`);
      list.push(P.topKWarper(topK, minKeep));
    }
    const topP = number('top_p');
    if (topP !== null && topP < 1) list.push(P.topPWarper(topP, minKeep));
    if (!isNone(config.min_p)) list.push(P.minPWarper(Number(config.min_p), minKeep));
    const typical = number('typical_p');
    if (typical !== null && typical < 1) list.push(P.typicalWarper(typical, minKeep));
    const epsilon = number('epsilon_cutoff');
    if (epsilon !== null && epsilon > 0 && epsilon < 1) list.push(P.epsilonWarper(epsilon, minKeep));
    const eta = number('eta_cutoff');
    if (eta !== null && eta > 0 && eta < 1) list.push(P.etaWarper(eta, minKeep));
  }
  if (!isNone(config.watermarking_config)) {
    list.push(P.watermarkProcessor(runner.model.generationVocabSize, config.watermarking_config as unknown as P.WatermarkSettings));
  }
  if (config.renormalize_logits === true) list.push(P.logitNormalization());
  return list;
}

/** ``StopStringCriteria`` tokenizer surface of a {@link FastTokenizer}. */
export function stopStringTokenizer(tokenizer: FastTokenizer): P.StopStringTokenizer {
  const backend = tokenizer.backend;
  return {
    vocab: () => {
      const map = new Map<string, number>();
      for (let id = 0; id < backend.vocabSize; id += 1) {
        const token = backend.idToToken(id);
        if (token !== undefined) map.set(token, id);
      }
      return map;
    },
    encodePlain: (text) => tokenizer.encode(text, { addSpecialTokens: false }).inputIds[0]!,
    idToToken: (id) => backend.idToToken(id) ?? null,
    tokensToString: (tokens) => (backend.decoder ? backend.decoder(tokens).join('') : tokens.join(' ')),
    decoderConfig: () => (backend.spec as { decoder?: unknown }).decoder ?? null,
  };
}

function buildCriteria(runner: Runner): P.StoppingCriterionFn[] {
  const { config, options, special } = runner;
  const list: P.StoppingCriterionFn[] = [];
  if (!isNone(config.max_length)) list.push(P.maxLengthCriteria(config.max_length as number));
  if (!isNone(config.max_time)) list.push(P.maxTimeCriteria(Number(config.max_time), options.clock));
  if (!isNone(config.stop_strings)) {
    if (!options.tokenizer) {
      throw new ValueError("There are one or more stop strings, either in the arguments to `generate` or in the model's generation config, but we could not locate a tokenizer. When generating with stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`.");
    }
    list.push(P.stopStringCriteria(stopStringTokenizer(options.tokenizer), config.stop_strings as string | string[]));
  }
  if (special.eos !== null) list.push(P.eosTokenCriteria(special.eos));
  list.push(...(options.stoppingCriteria ?? []));
  return list;
}

/** ``ExtensionsTrie(vocab).extensions(prefix)`` (partial prefixes stop at the deepest matching node). */
function trieExtensions(vocab: readonly string[], prefix: string): string[] {
  type Node = Map<string, Node> & { end?: boolean };
  const root: Node = new Map();
  for (const word of vocab) {
    if (!word) continue;
    let node = root;
    for (const char of word) {
      let child = node.get(char);
      if (!child) {
        child = new Map();
        node.set(char, child);
      }
      node = child;
    }
    node.end = true;
  }
  let node = root;
  for (const char of prefix) {
    const child = node.get(char);
    if (!child) break;
    node = child;
  }
  const collect = (current: Node): string[] => {
    const out: string[] = current.end ? [''] : [];
    for (const [char, child] of current) for (const tail of collect(child)) out.push(char + tail);
    return out;
  };
  return collect(node).map((tail) => prefix + tail);
}

/** ``GenerationMixin.heal_tokens``. */
function healTokens(model: CausalLanguageModel, ids: number[][], tokenizer: FastTokenizer | null | undefined, options: CausalGenerateOptions): number[][] {
  if (!tokenizer) {
    throw new ValueError(" When generating with token healing, you must pass the model's tokenizer to the `tokenizer` argument of `generate`.");
  }
  const bos = tokenizer.bosTokenId;
  const pad = tokenizer.padTokenId;
  const vocab = stopStringTokenizer(tokenizer).vocab();
  const prompts = ids.map((row) => tokenizer.decode(row, { skipSpecialTokens: true }).trim());
  const encoded = tokenizer.encode(prompts, { padding: true }).inputIds.map((row) => row.map((id) => (id === bos ? pad! : id)));
  if (!encoded.length || !encoded[0]!.length) return encoded;
  // ``convert_tokens_to_ids(' ')`` falls back to the unknown token id; its token's first character replaces spaces.
  const spaceId = vocab.get(' ') ?? tokenizer.unkTokenId;
  const tails = encoded.map((row) => row[row.length - 1]!);
  const tailTokens = tails.map((id) => {
    const text = tokenizer.decode([id]);
    return spaceId !== null ? text.replaceAll(' ', (tokenizer.backend.idToToken(spaceId) ?? '')[0] ?? '') : text;
  });
  const words = [...vocab.keys()];
  encoded.forEach((row, index) => {
    if (row.every((id) => id === pad)) return;
    const bias = new Map<number, number>();
    for (const alternative of trieExtensions(words, tailTokens[index]!)) {
      const id = vocab.get(alternative) ?? tokenizer.unkTokenId;
      if (id !== null && id !== undefined) bias.set(id, 10.0);
    }
    if (bias.size === 1) return;
    const tail = tails[index]!;
    if (!bias.has(tail)) throw new P.KeyError(`(${tail},)`);
    bias.set(tail, bias.get(tail)! + 1.0);
    const trimmed = row.slice(0, -1);
    if (!trimmed.length) return;
    if (row.filter((id) => id !== pad).length === 1) trimmed[trimmed.length - 1] = bos!;
    // A Python dict keyed by token tuples (``normalizeSequenceBias`` dict form).
    const sequenceBias = new Map([...bias].map(([id, value]) => [[id], value] as [number[], number]));
    const healed = generateCausal(model, { inputIds: [trimmed] }, {
      generationConfig: options.generationConfig ?? null, tokenizer: null, generator: options.generator,
      settings: { max_new_tokens: 1, pad_token_id: pad, sequence_bias: sequenceBias },
    });
    encoded[index] = healed.sequences[0]!;
  });
  return encoded;
}

/**
 * ``model.generate(input_ids, attention_mask=..., **model_kwargs, **settings)``
 * for a decoder-only model.
 */
export function generateCausal(model: CausalLanguageModel, inputs: CausalGenerateInputs, options: CausalGenerateOptions = {}): CausalGenerateOutput {
  return noGrad(() => generateImpl(model, inputs, options));
}

function generateImpl(model: CausalLanguageModel, inputs: CausalGenerateInputs, options: CausalGenerateOptions): CausalGenerateOutput {
  const settings = { ...(options.settings ?? {}) };
  const modelDict = options.generationConfig ?? {};
  const hasDefaultMaxLength = isNone(settings.max_length) && isNone(modelDict.max_length);
  const { config, modelKwargs } = prepareGenerationConfig(modelDict, settings);
  const mode = generationMode(config);
  const deprecated = DEPRECATED_MODES[mode];
  if (deprecated) {
    throw new ValueError(`${deprecated[0]} requires \`trust_remote_code=True\` in your \`generate\` call, since it loads https://hf.co/${deprecated[1]}.`);
  }
  const unused = Object.keys(modelKwargs).filter((key) => !['output_attentions', 'output_hidden_states'].includes(key) && modelKwargs[key] !== null && modelKwargs[key] !== undefined);
  if (unused.length) {
    throw new ValueError(`The following \`model_kwargs\` are not used by the model: ${P.pyRepr(unused)} (note: typos in the generate arguments will also show up in this list)`);
  }
  if (mode === 'beam' && options.streamer) throw new ValueError('`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1.');
  if (mode === 'assisted' && (config.num_return_sequences as number) > 1) {
    throw new ValueError(`num_return_sequences has to be 1 when doing assisted generate, but is ${config.num_return_sequences}.`);
  }
  const extras: Record<string, Tensor> = {};
  for (const [key, value] of Object.entries(inputs.extras ?? {})) {
    if (value === null || value === undefined) continue;
    if (!model.acceptedInputs.includes(key)) {
      throw new ValueError(`The following \`model_kwargs\` are not used by the model: ['${key}'] (note: typos in the generate arguments will also show up in this list)`);
    }
    extras[key] = value;
  }
  let ids = rows(inputs.inputIds)!;
  if (!ids.length) throw new ValueError('generation requires at least one sequence');
  const special = specialTokens(config);
  // Decoder-only attention mask (``_prepare_attention_mask_for_generation`` when absent).
  let mask = rows(inputs.attentionMask ?? null);
  if (mask === null) {
    const pad = special.pad;
    const padInInputs = pad !== null && ids.some((row) => row.includes(pad));
    const padIsEos = pad !== null && special.eos !== null && special.eos.includes(pad);
    mask = ids.map((row) => row.map((id) => (padInInputs && !padIsEos ? (id !== pad ? 1 : 0) : 1)));
  }
  if (mask.some((row, index) => row.length !== ids[index]!.length)) throw new ValueError('attention_mask must match input_ids');
  const encoderIds = ids.map((row) => [...row]);
  const positions = mask.map((row) => {
    let total = 0;
    return row.map((value) => {
      total += value ? 1 : 0;
      return value ? total - 1 : 0;
    });
  });
  // Expand for beams / returned sequences.
  const expand = Math.max(config.num_beams as number, config.num_return_sequences as number);
  ids = repeatInterleave(ids, expand);
  const state: LoopState = {
    mask: repeatInterleave(mask, expand), positions: repeatInterleave(positions, expand),
    cache: null, extras: Object.fromEntries(Object.entries(extras).map(([key, value]) => [key, repeatTensor(value, expand)])),
  };
  if (config.token_healing === true) {
    ids = healTokens(model, ids, options.tokenizer, options);
  }
  const promptLength = ids[0]!.length;
  // ``_prepare_generated_length``.
  if (!isNone(config.max_new_tokens)) config.max_length = (config.max_new_tokens as number) + promptLength;
  else if (hasDefaultMaxLength) {
    config.max_length = (config.max_length as number) + promptLength;
    if (model.generationMaxPositions !== null) config.max_length = Math.min(config.max_length as number, model.generationMaxPositions);
  }
  if (!isNone(config.min_new_tokens)) {
    config.min_length = (config.min_new_tokens as number) + promptLength;
    // Python adds an int to min_new_tokens, so a float min_new_tokens makes min_length a float.
    if (pythonKindOf(config.min_new_tokens, config, 'min_new_tokens') === 'float') setPythonNumberKind(config, 'min_length', 'float');
  }
  if (promptLength >= (config.max_length as number)) {
    throw new ValueError(`Input length of input_ids is ${promptLength}, but \`max_length\` is set to ${config.max_length}. This can lead to unexpected behavior. You should consider increasing \`max_length\` or, better yet, setting \`max_new_tokens\`.`);
  }
  // ``_prepare_cache_for_generation``.
  const runner = new Runner(model, config, special, options);
  if (runner.useCache) {
    const implementation = config.cache_implementation as string | null;
    if (implementation && implementation.startsWith('offloaded')) throw new ValueError('Expected a cuda device, but got: cpu');
    if (implementation === 'quantized') {
      throw new P.ImportError('You need to install optimum-quanto in order to use KV cache quantization with optimum-quanto backend. Please install it via  with `pip install optimum-quanto`');
    }
    state.cache = model.newCache();
  }
  const negative = { ids: rows(options.negativePromptIds ?? null), mask: rows(options.negativePromptAttentionMask ?? null) };
  runner.processors = buildProcessors(runner, promptLength, encoderIds, negative);
  runner.criteria = buildCriteria(runner);
  if (config.is_assistant === true) throw new TypeError("'NoneType' object is not subscriptable");
  options.streamer?.put(ids.map((row) => [...row]));
  let sequences: number[][];
  let sequencesScores: number[] | null = null;
  let beamIndices: number[][] | null = null;
  if (mode === 'beam' || mode === 'beam_sample') {
    const result = beamLoop(runner, ids, state);
    sequences = result.sequences;
    if (config.return_dict_in_generate === true) {
      if (config.output_scores === true) sequencesScores = result.sequencesScores;
      if (config.output_logits === true) beamIndices = result.beamIndices;
    }
  } else if (mode === 'assisted') {
    sequences = assistedLoop(runner, ids, state);
  } else {
    sequences = sampleLoop(runner, ids, state);
  }
  options.streamer?.end();
  return { sequences, scores: runner.scores, logits: runner.rawLogits, sequencesScores, beamIndices, config };
}
