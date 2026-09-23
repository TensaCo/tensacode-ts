/**
 * Encoder-decoder generation following transformers 5.17 semantics: greedy
 * decoding, multinomial sampling (temperature/top-k/top-p warpers, drawn with
 * ``torch.multinomial``'s sampler on the PyTorch-compatible generator) and
 * vectorized beam search (beam sampling included)
 * with length penalty and early stopping. Logits processors: repetition
 * penalty and minimum new tokens. Settings use GenerationConfig's snake_case
 * keys so persisted ``generation`` configurations apply directly.
 */
import { Tensor, tensor } from '../../nn/tensor.js';
import { noGrad } from '../../nn/autograd.js';
import { getDefaultGenerator, multinomialValues, type Generator } from '../../nn/random.js';
import { ValueError } from '../../errors.js';
import type { JsonObject } from '../json.js';
import { generationDefaults } from './config.js';
import type { LayerCache, T5ForConditionalGeneration } from './t5.js';

/** Python ``GenerationConfig`` fields honoured by TypeScript generation. */
export interface GenerationSettings {
  max_new_tokens?: number | null;
  max_length?: number | null;
  min_new_tokens?: number | null;
  num_beams?: number;
  num_return_sequences?: number;
  do_sample?: boolean;
  temperature?: number;
  top_k?: number | null;
  top_p?: number | null;
  repetition_penalty?: number;
  length_penalty?: number;
  early_stopping?: boolean | 'never';
  decoder_start_token_id?: number | null;
  eos_token_id?: number | number[] | null;
  pad_token_id?: number | null;
  return_dict_in_generate?: boolean;
  [key: string]: unknown;
}

/** Generation keys accepted by owned text operations (Python allow-list). */
export const GENERATION_KEYS = [
  'max_new_tokens', 'min_new_tokens', 'num_beams', 'do_sample', 'temperature', 'top_k', 'top_p',
  'repetition_penalty', 'length_penalty', 'early_stopping',
] as const;

export interface GenerationInputs {
  inputIds?: Tensor | null;
  inputsEmbeds?: Tensor | null;
  attentionMask?: Tensor | null;
  /** Precomputed encoder states. */
  encoderHiddenStates?: Tensor | null;
}

interface Resolved {
  maxLength: number;
  minNewTokens: number;
  numBeams: number;
  numReturn: number;
  doSample: boolean;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  lengthPenalty: number;
  earlyStopping: boolean | 'never';
  start: number;
  eos: number[];
  pad: number | null;
}

function resolve(model: T5ForConditionalGeneration, modelGeneration: JsonObject | null, call: GenerationSettings): Resolved {
  const settings = { ...generationDefaults(modelGeneration), ...call } as GenerationSettings;
  const config = model.config;
  const start = settings.decoder_start_token_id ?? config.optionalNumber('decoder_start_token_id');
  if (start === null || start === undefined) throw new ValueError('decoder_start_token_id must be defined for encoder-decoder generation');
  const eosRaw = settings.eos_token_id ?? config.get('eos_token_id') ?? null;
  const eos = eosRaw === null ? [] : Array.isArray(eosRaw) ? eosRaw.map(Number) : [Number(eosRaw)];
  const pad = settings.pad_token_id ?? config.optionalNumber('pad_token_id');
  const maxNew = settings.max_new_tokens;
  const maxLength = maxNew !== undefined && maxNew !== null ? 1 + maxNew : (settings.max_length ?? 20);
  const numBeams = settings.num_beams ?? 1;
  const numReturn = settings.num_return_sequences ?? 1;
  if (!Number.isInteger(numBeams) || numBeams < 1) throw new ValueError('num_beams must be a positive integer');
  if (numReturn > numBeams && !(settings.do_sample && numBeams === 1)) {
    throw new ValueError('num_return_sequences has to be smaller or equal to num_beams');
  }
  return {
    maxLength, minNewTokens: settings.min_new_tokens ?? 0, numBeams, numReturn,
    doSample: settings.do_sample === true, temperature: settings.temperature ?? 1, topK: settings.top_k ?? 0,
    topP: settings.top_p ?? 1, repetitionPenalty: settings.repetition_penalty ?? 1, lengthPenalty: settings.length_penalty ?? 1,
    earlyStopping: settings.early_stopping ?? false, start, eos, pad: pad ?? null,
  };
}

const f32 = Math.fround;

function logSoftmax(row: Float32Array): Float32Array {
  let max = -Infinity;
  for (const value of row) if (value > max) max = value;
  let total = 0;
  for (const value of row) total += Math.exp(value - max);
  const logTotal = Math.log(total) + max;
  const result = new Float32Array(row.length);
  for (let index = 0; index < row.length; index += 1) result[index] = row[index]! - logTotal;
  return result;
}

/** Logits processors shared by all strategies (in transformers' order). */
function processScores(scores: Float32Array, sequence: readonly number[], resolved: Resolved, newTokens: number): void {
  if (resolved.repetitionPenalty !== 1) {
    const seen = new Set(sequence);
    for (const id of seen) {
      if (id < 0 || id >= scores.length) continue;
      const score = scores[id]!;
      scores[id] = score < 0 ? f32(score * resolved.repetitionPenalty) : f32(score / resolved.repetitionPenalty);
    }
  }
  if (newTokens < resolved.minNewTokens) for (const id of resolved.eos) if (id < scores.length) scores[id] = -Infinity;
}

/** Temperature, top-k and top-p warpers for sampling. */
function warp(scores: Float32Array, resolved: Resolved, minKeep = 1): void {
  if (resolved.temperature !== 1) {
    if (!(resolved.temperature > 0)) throw new ValueError('temperature must be strictly positive');
    for (let index = 0; index < scores.length; index += 1) scores[index] = f32(scores[index]! / resolved.temperature);
  }
  if (resolved.topK && resolved.topK > 0) {
    const k = Math.min(Math.max(resolved.topK, minKeep), scores.length);
    const sorted = Array.from(scores).sort((a, b) => b - a);
    const threshold = sorted[k - 1]!;
    for (let index = 0; index < scores.length; index += 1) if (scores[index]! < threshold) scores[index] = -Infinity;
  }
  if (resolved.topP < 1) {
    const order = Array.from(scores.keys()).sort((a, b) => scores[a]! - scores[b]!);
    let max = -Infinity;
    for (const value of scores) if (value > max) max = value;
    let total = 0;
    for (const value of scores) total += Math.exp(value - max);
    let cumulative = 0;
    const remove: number[] = [];
    order.forEach((id, position) => {
      cumulative += Math.exp(scores[id]! - max) / total;
      if (cumulative <= 1 - resolved.topP && position < order.length - minKeep) remove.push(id);
    });
    for (const id of remove) scores[id] = -Infinity;
  }
}

/** ``nn.functional.softmax(scores, dim=-1)`` in float32. */
function softmaxF32(scores: Float32Array): Float32Array {
  let max = -Infinity;
  for (const value of scores) if (value > max) max = value;
  const probabilities = new Float32Array(scores.length);
  let total = 0;
  for (let index = 0; index < scores.length; index += 1) {
    probabilities[index] = Math.exp(Math.fround(scores[index]! - max));
    total += probabilities[index]!;
  }
  const sum = Math.fround(total);
  for (let index = 0; index < probabilities.length; index += 1) probabilities[index] = probabilities[index]! / sum;
  return probabilities;
}

/** ``torch.multinomial(softmax(scores), 1)``: PyTorch's exponential-race sampler on the shared generator. */
function sample(scores: Float32Array, generator: Generator): number {
  return multinomialValues(softmaxF32(scores), 'float32', 1, scores.length, 1, false, generator)[0]!;
}

function argmax(scores: Float32Array): number {
  let best = 0;
  for (let index = 1; index < scores.length; index += 1) if (scores[index]! > scores[best]!) best = index;
  return best;
}

function rowsOf(logits: Tensor): Float32Array[] {
  const [batch, steps, vocab] = logits.shape as [number, number, number];
  const rows: Float32Array[] = [];
  for (let b = 0; b < batch; b += 1) {
    const offset = (b * steps + steps - 1) * vocab;
    rows.push(Float32Array.from(logits.data.subarray(offset, offset + vocab)));
  }
  return rows;
}

function column(tokens: readonly number[]): Tensor {
  return tensor(tokens, { shape: [tokens.length, 1], dtype: 'int64' });
}

function repeatRows(value: Tensor, times: number): Tensor {
  if (times === 1) return value;
  const indices: number[] = [];
  for (let row = 0; row < value.shape[0]!; row += 1) for (let copy = 0; copy < times; copy += 1) indices.push(row);
  return value.indexSelect(0, indices);
}

function reorderCache(cache: LayerCache[], indices: readonly number[]): void {
  for (const layer of cache) {
    if (layer.self.key) layer.self.key = layer.self.key.indexSelect(0, indices);
    if (layer.self.value) layer.self.value = layer.self.value.indexSelect(0, indices);
  }
}

function toTensor(rows: number[][]): Tensor {
  const width = rows[0]?.length ?? 0;
  return tensor(rows.flat(), { shape: [rows.length, width], dtype: 'int64' });
}

/**
 * ``model.generate(...)`` for a T5 model. Returns int64 ids
 * ``[batch * num_return_sequences, length]`` including the decoder start token.
 */
export function generateSeq2Seq(
  model: T5ForConditionalGeneration, inputs: GenerationInputs,
  settings: GenerationSettings = {}, options: { generationConfig?: JsonObject | null; generator?: Generator } = {},
): Tensor {
  const resolved = resolve(model, options.generationConfig ?? null, settings);
  const generator = options.generator ?? getDefaultGenerator();
  return noGrad(() => {
    const encoderHidden = inputs.encoderHiddenStates ?? model.encode({
      inputIds: inputs.inputIds ?? null, inputsEmbeds: inputs.inputsEmbeds ?? null, attentionMask: inputs.attentionMask ?? null,
    });
    const mask = inputs.attentionMask ?? null;
    if (resolved.numBeams > 1) return beamSearch(model, encoderHidden, mask, resolved, generator);
    return sampleOrGreedy(model, encoderHidden, mask, resolved, generator);
  });
}

function sampleOrGreedy(model: T5ForConditionalGeneration, encoder: Tensor, mask: Tensor | null, resolved: Resolved, generator: Generator): Tensor {
  const copies = resolved.doSample ? resolved.numReturn : 1;
  const hidden = repeatRows(encoder, copies);
  const encoderMask = mask ? repeatRows(mask, copies) : null;
  const batch = hidden.shape[0]!;
  const sequences = Array.from({ length: batch }, () => [resolved.start]);
  const unfinished = new Array<boolean>(batch).fill(true);
  const cache = model.newCache();
  const hasEos = resolved.eos.length > 0;
  if (hasEos && resolved.pad === null) throw new ValueError('pad_token_id must be defined when eos_token_id is set');
  let past = 0;
  let current = sequences.map((row) => row[row.length - 1]!);
  while (sequences[0]!.length < resolved.maxLength) {
    const logits = model.decodeStep(column(current), hidden, encoderMask, cache, past);
    past += 1;
    const rows = rowsOf(logits);
    const next = rows.map((scores, b) => {
      processScores(scores, sequences[b]!, resolved, sequences[b]!.length - 1);
      if (!resolved.doSample) return argmax(scores);
      warp(scores, resolved);
      return sample(scores, generator);
    });
    for (let b = 0; b < batch; b += 1) {
      const token = hasEos && !unfinished[b] ? resolved.pad! : next[b]!;
      sequences[b]!.push(token);
      if (resolved.eos.includes(token)) unfinished[b] = false;
    }
    current = sequences.map((row) => row[row.length - 1]!);
    if (!unfinished.some(Boolean)) break;
  }
  return toTensor(sequences);
}

interface Candidate {
  score: number;
  beam: number;
  token: number;
  index: number;
}

function topK<T extends { score: number; index: number }>(items: T[], k: number): T[] {
  return [...items].sort((a, b) => (b.score - a.score) || (a.index - b.index)).slice(0, k);
}

function beamSearch(model: T5ForConditionalGeneration, encoder: Tensor, mask: Tensor | null, resolved: Resolved, generator: Generator): Tensor {
  const beams = resolved.numBeams;
  const batch = encoder.shape[0]!;
  const hidden = repeatRows(encoder, beams);
  const encoderMask = mask ? repeatRows(mask, beams) : null;
  const promptLength = 1;
  const eosCount = resolved.eos.length;
  const keep = Math.max(2, 1 + eosCount) * beams;
  const fill = resolved.pad ? resolved.pad : (eosCount ? resolved.eos[0]! : -1);
  const lp = resolved.lengthPenalty;
  const early = resolved.earlyStopping;
  const running: number[][][] = Array.from({ length: batch }, () => Array.from({ length: beams }, () => [resolved.start]));
  const runningScores: number[][] = Array.from({ length: batch }, () => Array.from({ length: beams }, (_, beam) => (beam === 0 ? 0 : -1e9)));
  const finished: number[][][] = Array.from({ length: batch }, () => Array.from({ length: beams }, () => [resolved.start]));
  const finishedScores: number[][] = Array.from({ length: batch }, () => new Array<number>(beams).fill(-1e9));
  const finishedFlags: boolean[][] = Array.from({ length: batch }, () => new Array<boolean>(beams).fill(false));
  const heuristic = new Array<boolean>(batch).fill(true);
  const cache = model.newCache();
  let curLen = 1;
  let past = 0;
  for (;;) {
    const last = running.flatMap((items) => items.map((sequence) => sequence[sequence.length - 1]!));
    const logits = model.decodeStep(column(last), hidden, encoderMask, cache, past);
    past += 1;
    const rows = rowsOf(logits);
    const vocab = rows[0]!.length;
    const reorder: number[] = [];
    let allHit = true;
    for (let b = 0; b < batch; b += 1) {
      const candidates: Candidate[] = [];
      for (let beam = 0; beam < beams; beam += 1) {
        const scores = logSoftmax(rows[b * beams + beam]!);
        processScores(scores, running[b]![beam]!, resolved, curLen - promptLength);
        if (resolved.doSample) warp(scores, resolved, 2);
        for (let token = 0; token < vocab; token += 1) {
          candidates.push({ score: f32(scores[token]! + runningScores[b]![beam]!), beam, token, index: beam * vocab + token });
        }
      }
      let top: Candidate[];
      if (resolved.doSample) {
        // ``torch.multinomial(softmax(accumulated_log_probs), num_samples=beams_to_keep)``.
        const probabilities = softmaxF32(Float32Array.from(candidates, (item) => item.score));
        top = Array.from(multinomialValues(probabilities, 'float32', 1, candidates.length, keep, false, generator), (index) => candidates[index]!);
      } else top = topK(candidates, keep);
      const hits = top.map((item) => resolved.eos.includes(item.token) || curLen + 1 >= resolved.maxLength);
      if (!hits.every(Boolean)) allHit = false;
      const sequences = top.map((item) => [...running[b]![item.beam]!, item.token]);
      // Next running beams: best non-finished continuations.
      const runningCandidates = top.map((item, index) => ({ score: f32(item.score + (hits[index] ? -1e9 : 0)), index, item, sequence: sequences[index]! }));
      const nextRunning = topK(runningCandidates, beams);
      running[b] = nextRunning.map((entry) => entry.sequence);
      runningScores[b] = nextRunning.map((entry) => entry.score);
      for (const entry of nextRunning) reorder.push(b * beams + entry.item.beam);
      // Finished hypotheses.
      const full = finishedFlags[b]!.every(Boolean) && early === true;
      const merged = [
        ...finished[b]!.map((sequence, index) => ({ score: finishedScores[b]![index]!, index, sequence, flag: finishedFlags[b]![index]! })),
        ...top.map((item, index) => {
          const justFinished = hits[index]! && index < beams;
          let score = f32(item.score / (curLen + 1 - promptLength) ** lp);
          if (full) score = f32(score - 1e9);
          if (!heuristic[b]) score = f32(score - 1e9);
          if (!justFinished) score = f32(score - 1e9);
          return { score, index: beams + index, sequence: sequences[index]!, flag: justFinished };
        }),
      ];
      const best = topK(merged, beams);
      finished[b] = best.map((entry) => entry.sequence);
      finishedScores[b] = best.map((entry) => entry.score);
      finishedFlags[b] = best.map((entry) => entry.flag);
    }
    reorderCache(cache, reorder);
    curLen += 1;
    for (let b = 0; b < batch; b += 1) {
      const length = early === 'never' && lp > 0 ? resolved.maxLength - promptLength : curLen - promptLength;
      const bestRunning = runningScores[b]![0]! / length ** lp;
      const worst = Math.min(...finishedScores[b]!);
      const improvable = finishedFlags[b]!.some((flag) => bestRunning > (flag ? worst : -1e9));
      heuristic[b] = heuristic[b]! && improvable;
    }
    const improvementPossible = heuristic.some(Boolean);
    const openBeam = !(finishedFlags.every((flags) => flags.every(Boolean)) && early === true);
    if (!(improvementPossible && openBeam && !allHit)) break;
  }
  const outputs: number[][] = [];
  for (let b = 0; b < batch; b += 1) for (let index = 0; index < resolved.numReturn; index += 1) outputs.push(finished[b]![index]!);
  const width = Math.max(...outputs.map((sequence) => sequence.length));
  return toTensor(outputs.map((sequence) => [...sequence, ...new Array<number>(width - sequence.length).fill(fill)]));
}
