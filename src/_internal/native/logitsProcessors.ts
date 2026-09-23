/**
 * transformers 5.17 logits processors, warpers and stopping criteria
 * (``transformers.generation.logits_process`` / ``stopping_criteria``) over
 * plain rows: ``ids`` are the token ids of every sequence so far and
 * ``scores`` one float32 row of next-token scores per sequence. Arithmetic is
 * rounded to float32 where PyTorch computes in float32 (Python scalars are
 * cast to the tensor dtype first).
 */
import { AttributeError, ImportError, IndexError, KeyError, RuntimeError, ValueError } from '../../errors.js';

export { AttributeError, ImportError, IndexError, KeyError, RuntimeError };
import { Generator, randpermValues } from '../../nn/random.js';
import { pythonFloatRepr } from '../json.js';

const f32 = Math.fround;
const FLOAT32_MAX = 3.4028234663852886e38;

/** Python ``repr`` of a JSON-like value (floats need an explicit ``float`` hint). */
export function pyRepr(value: unknown): string {
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : pythonFloatRepr(value);
  if (typeof value === 'string') return `'${value.replaceAll('\\', '\\\\').replaceAll("'", "\\'")}'`;
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(', ')}]`;
  if (typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>).map(([k, v]) => `${pyRepr(k)}: ${pyRepr(v)}`).join(', ')}}`;
  return String(value);
}

/** Python ``repr`` of a number known to be a ``float``. */
export function pyFloat(value: number): string {
  if (Number.isNaN(value)) return 'nan';
  if (value === Infinity) return 'inf';
  if (value === -Infinity) return '-inf';
  return pythonFloatRepr(value);
}

/** A logits processor: ``processor(input_ids, scores)`` returning the processed rows. */
export type LogitsProcessorFn = (ids: readonly (readonly number[])[], scores: Float32Array[]) => Float32Array[];

/** ``torch.softmax`` of one float32 row. */
export function softmaxRow(row: Float32Array): Float32Array {
  let max = -Infinity;
  for (const value of row) if (value > max) max = value;
  const out = new Float32Array(row.length);
  if (max === -Infinity) {
    out.fill(Number.NaN);
    return out;
  }
  let total = 0;
  for (let index = 0; index < row.length; index += 1) {
    const value = f32(Math.exp(f32(row[index]! - max)));
    out[index] = value;
    total += value;
  }
  const sum = f32(total);
  for (let index = 0; index < row.length; index += 1) out[index] = f32(out[index]! / sum);
  return out;
}

/** ``torch.log_softmax`` of one float32 row. */
export function logSoftmaxRow(row: Float32Array): Float32Array {
  let max = -Infinity;
  for (const value of row) if (value > max) max = value;
  const out = new Float32Array(row.length);
  if (max === -Infinity || Number.isNaN(max)) {
    out.fill(max === -Infinity ? Number.NaN : max);
    return out;
  }
  let total = 0;
  for (const value of row) total += Math.exp(value - max);
  const logTotal = f32(Math.log(total));
  for (let index = 0; index < row.length; index += 1) out[index] = f32(f32(row[index]! - max) - logTotal);
  return out;
}

/** Indices of ``row`` sorted by value (``torch.sort``; ties keep index order). */
function sortedIndices(row: Float32Array, descending: boolean): number[] {
  const order = Array.from(row.keys());
  order.sort((a, b) => {
    const x = row[a]!;
    const y = row[b]!;
    if (x === y) return a - b;
    return descending ? (x > y ? -1 : 1) : (x < y ? -1 : 1);
  });
  return order;
}

/** ``torch.topk(row, k)[0][-1]``: the k-th largest value. */
function kthLargest(row: Float32Array, k: number): number {
  const sorted = Float32Array.from(row).sort();
  return sorted[sorted.length - k]!;
}

/** ``torch.cumsum`` of float32 values (CPU accumulates in double, stores float32). */
function cumsum(values: ArrayLike<number>): Float32Array {
  const out = new Float32Array(values.length);
  let total = 0;
  for (let index = 0; index < values.length; index += 1) {
    total += values[index]!;
    out[index] = total;
  }
  return out;
}

function eosList(eos: readonly number[] | null, what = 'eos_token_id'): number[] {
  if (eos === null) throw new RuntimeError('Could not infer dtype of NoneType');
  if (eos.some((id) => id < 0)) throw new ValueError(`\`${what}\` has to be a list of positive integers, but is tensor(${pyRepr(eos)})`);
  return [...eos];
}

function mapRows(scores: Float32Array[], fn: (row: Float32Array, index: number) => Float32Array): Float32Array[] {
  return scores.map((row, index) => fn(row, index));
}

/** ``MinLengthLogitsProcessor``. */
export function minLengthProcessor(minLength: number, eos: readonly number[]): LogitsProcessorFn {
  if (!Number.isInteger(minLength) || minLength < 0) throw new ValueError(`\`min_length\` has to be a non-negative integer, but is ${pyRepr(minLength)}`);
  return (ids, scores) => {
    if ((ids[0]?.length ?? 0) >= minLength) return scores;
    return mapRows(scores, (row) => {
      const out = Float32Array.from(row);
      for (const id of eos) if (id >= 0 && id < out.length) out[id] = -Infinity;
      return out;
    });
  };
}

/** ``MinNewTokensLengthLogitsProcessor``. */
export function minNewTokensProcessor(promptLength: number, minNewTokens: number, eos: readonly number[]): LogitsProcessorFn {
  for (const [name, value] of [['prompt_length_to_skip', promptLength], ['min_new_tokens', minNewTokens]] as const) {
    if (!Number.isInteger(value) || value < 0) throw new ValueError(`\`${name}\` has to be a positive integer, but is ${pyRepr(value)}`);
  }
  return (ids, scores) => {
    if ((ids[0]?.length ?? 0) - promptLength >= minNewTokens) return scores;
    return mapRows(scores, (row) => {
      const out = Float32Array.from(row);
      for (const id of eos) if (id >= 0 && id < out.length) out[id] = -Infinity;
      return out;
    });
  };
}

/** ``TemperatureLogitsWarper``. */
export function temperatureWarper(temperature: number): LogitsProcessorFn {
  if (typeof temperature !== 'number' || !(temperature > 0)) {
    let message = `\`temperature\` (=${pyFloat(Number(temperature))}) has to be a strictly positive float, otherwise your next token scores will be invalid.`;
    if (temperature === 0) message += " If you're looking for greedy decoding strategies, set `do_sample=False`.";
    throw new ValueError(message);
  }
  const t = f32(temperature);
  return (_, scores) => mapRows(scores, (row) => row.map((value) => f32(value / t)));
}

/** ``RepetitionPenaltyLogitsProcessor``. */
export function repetitionPenaltyProcessor(penalty: number, promptIgnoreLength: number | null = null): LogitsProcessorFn {
  if (typeof penalty !== 'number' || !(penalty > 0)) throw new ValueError(`\`penalty\` has to be a strictly positive float, but is ${pyRepr(penalty)}`);
  const p = f32(penalty);
  return (ids, scores) => mapRows(scores, (row, index) => {
    const out = Float32Array.from(row);
    const tokens = promptIgnoreLength ? ids[index]!.slice(promptIgnoreLength) : ids[index]!;
    for (const id of new Set(tokens)) {
      if (id < 0 || id >= out.length) throw new RuntimeError(`index ${id} is out of bounds for dimension 1 with size ${out.length}`);
      const score = row[id]!;
      out[id] = score < 0 ? f32(score * p) : f32(score / p);
    }
    return out;
  });
}

/**
 * ``EncoderRepetitionPenaltyLogitsProcessor``. As in PyTorch's gather/scatter,
 * only the first ``encoder_input_ids`` rows are penalized.
 */
export function encoderRepetitionPenaltyProcessor(penalty: number, encoderIds: readonly (readonly number[])[]): LogitsProcessorFn {
  if (typeof penalty !== 'number' || !(penalty > 0)) throw new ValueError(`\`penalty\` has to be a strictly positive float, but is ${pyRepr(penalty)}`);
  const inverse = f32(1 / penalty);
  return (_, scores) => mapRows(scores, (row, index) => {
    if (index >= encoderIds.length) return row;
    const out = Float32Array.from(row);
    for (const id of encoderIds[index]!) {
      const score = row[id]!;
      out[id] = score < 0 ? f32(score * inverse) : f32(score / inverse);
    }
    return out;
  });
}

/** ``TopPLogitsWarper``. */
export function topPWarper(topP: number, minTokensToKeep = 1): LogitsProcessorFn {
  if (topP < 0 || topP > 1) throw new ValueError(`\`top_p\` has to be a float > 0 and < 1, but is ${pyFloat(topP)}`);
  if (!Number.isInteger(minTokensToKeep) || minTokensToKeep < 1) throw new ValueError(`\`min_tokens_to_keep\` has to be a positive integer, but is ${minTokensToKeep}`);
  const threshold = f32(1 - topP);
  return (_, scores) => mapRows(scores, (row) => {
    const order = sortedIndices(row, false);
    const probabilities = softmaxRow(Float32Array.from(order, (id) => row[id]!));
    const cumulative = cumsum(probabilities);
    const out = Float32Array.from(row);
    for (let position = 0; position < order.length - minTokensToKeep; position += 1) {
      if (cumulative[position]! <= threshold) out[order[position]!] = -Infinity;
    }
    return out;
  });
}

/** ``TopKLogitsWarper``. */
export function topKWarper(topK: number, minTokensToKeep = 1): LogitsProcessorFn {
  if (!Number.isInteger(topK) || topK <= 0) throw new ValueError(`\`top_k\` has to be a strictly positive integer, but is ${pyRepr(topK)}`);
  const keep = Math.max(topK, minTokensToKeep);
  return (_, scores) => mapRows(scores, (row) => {
    const threshold = kthLargest(row, Math.min(keep, row.length));
    return row.map((value) => (value < threshold ? -Infinity : value));
  });
}

/** ``TopHLogitsWarper`` (entropy-bounded nucleus over the top 100 tokens). */
export function topHWarper(topH: number): LogitsProcessorFn {
  if (!(topH > 0 && topH <= 1)) throw new ValueError('`top_h` must be in the range (0, 1].');
  return (_, scores) => mapRows(scores, (row) => {
    const n = Math.min(100, row.length);
    const top = sortedIndices(row, true).slice(0, n);
    const logits = Float32Array.from(top, (id) => row[id]!);
    const probabilities = softmaxRow(logits);
    const logs = probabilities.map((p) => f32(Math.log(p)));
    let entropy = 0;
    for (let index = 0; index < n; index += 1) if (probabilities[index]! > 0) entropy -= probabilities[index]! * logs[index]!;
    const tau = f32(f32(entropy) * f32(topH));
    const terms = Float32Array.from(probabilities, (p, index) => f32(-p * logs[index]!));
    const cumulative = cumsum(terms);
    const out = new Float32Array(row.length).fill(-Infinity);
    for (let index = 0; index < n; index += 1) {
      if (index === 0 || cumulative[index]! <= tau) out[top[index]!] = row[top[index]!]!;
    }
    return out;
  });
}

/** ``MinPLogitsWarper``. */
export function minPWarper(minP: number, minTokensToKeep = 1): LogitsProcessorFn {
  if (!(minP >= 0 && minP <= 1)) throw new ValueError(`\`min_p\` has to be a float in the [0, 1] interval, but is ${pyFloat(minP)}`);
  if (!Number.isInteger(minTokensToKeep) || minTokensToKeep < 1) throw new ValueError(`\`min_tokens_to_keep\` has to be a positive integer, but is ${minTokensToKeep}`);
  return (_, scores) => mapRows(scores, (row) => {
    const probabilities = softmaxRow(row);
    let top = -Infinity;
    for (const p of probabilities) if (p > top) top = p;
    const scaled = f32(f32(minP) * top);
    const keep = new Set(sortedIndices(probabilities, true).slice(0, Math.min(minTokensToKeep, row.length)));
    return row.map((value, index) => (probabilities[index]! < scaled && !keep.has(index) ? -Infinity : value));
  });
}

/** ``TypicalLogitsWarper``. */
export function typicalWarper(mass: number, minTokensToKeep = 1): LogitsProcessorFn {
  if (!(mass > 0 && mass < 1)) throw new ValueError(`\`typical_p\` has to be a float > 0 and < 1, but is ${pyFloat(mass)}`);
  if (!Number.isInteger(minTokensToKeep) || minTokensToKeep < 1) throw new ValueError(`\`min_tokens_to_keep\` has to be a positive integer, but is ${minTokensToKeep}`);
  const massF = f32(mass);
  return (_, scores) => mapRows(scores, (row) => {
    const normalized = logSoftmaxRow(row);
    let entropy = 0;
    for (const value of normalized) {
      const product = f32(value * f32(Math.exp(value)));
      if (!Number.isNaN(product)) entropy += product;
    }
    const ent = f32(-entropy);
    const shifted = normalized.map((value) => f32(Math.abs(f32(-value - ent))));
    const order = sortedIndices(shifted, false);
    const sortedScores = Float32Array.from(order, (id) => shifted[id]!);
    const cumulative = cumsum(softmaxRow(Float32Array.from(order, (id) => row[id]!)));
    let last = 0;
    for (const value of cumulative) if (value < massF) last += 1;
    last = Math.min(last, row.length - 1);
    const limit = sortedScores[last]!;
    const out = Float32Array.from(row);
    for (let position = minTokensToKeep; position < order.length; position += 1) {
      if (sortedScores[position]! > limit) out[order[position]!] = -Infinity;
    }
    return out;
  });
}

/** ``EpsilonLogitsWarper``. */
export function epsilonWarper(epsilon: number, minTokensToKeep = 1): LogitsProcessorFn {
  if (epsilon <= 0 || epsilon >= 1) throw new ValueError(`\`epsilon_cutoff\` has to be a float > 0 and < 1, but is ${pyFloat(epsilon)}`);
  if (minTokensToKeep < 1) throw new ValueError(`\`min_tokens_to_keep\` has to be a strictly positive integer, but is ${minTokensToKeep}`);
  const eps = f32(epsilon);
  return (_, scores) => mapRows(scores, (row) => {
    const probabilities = softmaxRow(row);
    const threshold = kthLargest(row, Math.min(minTokensToKeep, row.length));
    return row.map((value, index) => (probabilities[index]! < eps && value < threshold ? -Infinity : value));
  });
}

/** ``EtaLogitsWarper``. */
export function etaWarper(epsilon: number, minTokensToKeep = 1): LogitsProcessorFn {
  if (epsilon <= 0 || epsilon >= 1) throw new ValueError(`\`eta_cutoff\` has to be a float > 0 and < 1, but is ${pyFloat(epsilon)}`);
  if (minTokensToKeep < 1) throw new ValueError(`\`min_tokens_to_keep\` has to be a strictly positive integer, but is ${minTokensToKeep}`);
  const eps = f32(epsilon);
  return (_, scores) => mapRows(scores, (row) => {
    const probabilities = softmaxRow(row);
    const logs = logSoftmaxRow(row);
    let entropy = 0;
    for (let index = 0; index < row.length; index += 1) if (probabilities[index]! > 0) entropy -= probabilities[index]! * logs[index]!;
    const eta = Math.min(eps, f32(f32(Math.sqrt(eps)) * f32(Math.exp(-f32(entropy)))));
    const threshold = kthLargest(row, Math.min(minTokensToKeep, row.length));
    return row.map((value, index) => (probabilities[index]! < eta && value < threshold ? -Infinity : value));
  });
}

/** ``NoRepeatNGramLogitsProcessor``. */
export function noRepeatNGramProcessor(ngramSize: number): LogitsProcessorFn {
  if (!Number.isInteger(ngramSize) || ngramSize <= 0) throw new ValueError(`\`ngram_size\` has to be a strictly positive integer, but is ${pyRepr(ngramSize)}`);
  return (ids, scores) => {
    const length = ids[0]?.length ?? 0;
    if (length < ngramSize) return scores;
    return mapRows(scores, (row, index) => {
      const sequence = ids[index]!;
      const out = Float32Array.from(row);
      const prefix = sequence.slice(length + 1 - ngramSize);
      for (let start = 0; start + ngramSize <= length; start += 1) {
        let matches = true;
        for (let offset = 0; offset < ngramSize - 1; offset += 1) {
          if (sequence[start + offset] !== prefix[offset]) { matches = false; break; }
        }
        if (matches) {
          const banned = sequence[start + ngramSize - 1]!;
          if (banned >= 0 && banned < out.length) out[banned] = -Infinity;
        }
      }
      return out;
    });
  };
}

/** Python slice ``values[start:stop]`` (negative indices count from the end). */
function pySlice<T>(values: readonly T[], start: number, stop: number): T[] {
  const n = values.length;
  const s = start < 0 ? Math.max(0, n + start) : Math.min(start, n);
  const e = stop < 0 ? Math.max(0, n + stop) : Math.min(stop, n);
  return values.slice(s, Math.max(s, e));
}

/** ``EncoderNoRepeatNGramLogitsProcessor`` over the prompt (``encoder_input_ids``). */
export function encoderNoRepeatNGramProcessor(ngramSize: number, encoderIds: readonly (readonly number[])[]): LogitsProcessorFn {
  if (!Number.isInteger(ngramSize) || ngramSize <= 0) throw new ValueError(`\`encoder_ngram_size\` has to be a strictly positive integer, but is ${pyRepr(ngramSize)}`);
  const generated = encoderIds.map((tokens) => {
    const map = new Map<string, number[]>();
    for (let start = 0; start + ngramSize <= tokens.length; start += 1) {
      const key = tokens.slice(start, start + ngramSize - 1).join(',');
      const list = map.get(key) ?? [];
      list.push(tokens[start + ngramSize - 1]!);
      map.set(key, list);
    }
    return map;
  });
  return (ids, scores) => {
    const hypotheses = scores.length;
    const beams = Math.floor(hypotheses / generated.length);
    return mapRows(scores, (row, index) => {
      const length = ids[index]!.length;
      const key = pySlice(ids[index]!, length + 1 - ngramSize, length).join(',');
      const banned = generated[Math.floor(index / beams)]!.get(key) ?? [];
      if (!banned.length) return row;
      const out = Float32Array.from(row);
      for (const id of banned) out[id] = -Infinity;
      return out;
    });
  };
}

/**
 * Validated ``sequence_bias`` as ``[token sequence, bias]`` pairs (Python dict
 * order): the JSON list form ``[[ids, bias], ...]``, or a ``Map`` from token
 * id arrays to biases for Python's tuple-keyed dict form.
 */
export function normalizeSequenceBias(value: unknown): [number[], number][] {
  const isInt = (token: unknown): token is number => typeof token === 'number' && Number.isInteger(token);
  const empty = value instanceof Map ? value.size === 0 : !Array.isArray(value) || value.length === 0;
  if (empty) {
    throw new ValueError(`\`sequence_bias\` has to be a non-empty dictionary, or non-empty list of lists but is ${pyRepr(value instanceof Map ? {} : value)}.`);
  }
  if (value instanceof Map) {
    const entries = [...value.entries()] as [unknown, unknown][];
    const shown = (): string => `{${entries.map(([key, bias]) => `(${(key as number[]).join(', ')}${(key as number[]).length === 1 ? ',' : ''}): ${pyFloat(Number(bias))}`).join(', ')}}`;
    if (entries.some(([key]) => !Array.isArray(key))) throw new ValueError(`\`sequence_bias\` has to be a dict with tuples as keys, but is ${shown()}.`);
    if (entries.some(([key]) => (key as unknown[]).length === 0 || (key as unknown[]).some((token) => !isInt(token) || token < 0))) {
      throw new ValueError(`Each key in \`sequence_bias\` has to be a non-empty tuple of positive integers, but is ${shown()}.`);
    }
    if (entries.some(([, bias]) => typeof bias !== 'number')) throw new ValueError(`\`sequence_bias\` has to be a dict with floats as values, but is ${shown()}.`);
    const map = new Map<string, [number[], number]>();
    for (const [key, bias] of entries) map.set((key as number[]).join(','), [[...(key as number[])], bias as number]);
    return [...map.values()];
  }
  // Python prints the biases, which are floats, with a decimal point.
  const shown = (): string => `[${(value as unknown[]).map((entry) => (Array.isArray(entry) && entry.length === 2 && typeof entry[1] === 'number'
    ? `[${pyRepr(entry[0])}, ${pyFloat(entry[1])}]` : pyRepr(entry))).join(', ')}]`;
  for (const entry of value as unknown[]) {
    if (!Array.isArray(entry)) throw new ValueError(`Each element in \`sequence_bias\` has to be a non-empty list of lists of positive integers and float, but is ${shown()}.`);
    if (entry.length === 0) throw new IndexError('list index out of range');
    const valid = Array.isArray(entry[0]) && (entry[0] as unknown[]).every((token) => isInt(token) && token > 0) && typeof entry[1] === 'number';
    if (!valid) throw new ValueError(`Each element in \`sequence_bias\` has to be a non-empty list of lists of positive integers and float, but is ${shown()}.`);
  }
  const map = new Map<string, [number[], number]>();
  for (const entry of value as [number[], number][]) {
    const key = entry[0].join(',');
    const existing = map.get(key);
    if (existing) existing[1] = entry[1];
    else map.set(key, [[...entry[0]], entry[1]]);
  }
  return [...map.values()];
}

/** ``SequenceBiasLogitsProcessor`` over validated ``[sequence, bias]`` pairs. */
export function sequenceBiasProcessor(pairs: readonly (readonly [readonly number[], number])[]): LogitsProcessorFn {
  let prepared: Float32Array | null = null;
  return (ids, scores) => {
    const vocab = scores[0]?.length ?? 0;
    if (!prepared) {
      const invalid: number[] = [];
      for (const [sequence] of pairs) for (const token of sequence) if (token >= vocab) invalid.push(token);
      if (invalid.length) throw new ValueError(`The model vocabulary size is ${vocab}, but the following tokens were being biased: ${pyRepr(invalid)}`);
      prepared = new Float32Array(vocab);
      for (const [sequence, bias] of pairs) if (sequence.length === 1) prepared[sequence[0]!] = f32(bias);
    }
    const single = prepared;
    const length = ids[0]?.length ?? 0;
    return mapRows(scores, (row, index) => {
      const bias = Float32Array.from(single);
      const sequence = ids[index]!;
      for (const [tokens, value] of pairs) {
        if (tokens.length === 1 || tokens.length > length) continue;
        const prefix = tokens.length - 1;
        let matches = true;
        for (let offset = 0; offset < prefix; offset += 1) {
          if (sequence[length - prefix + offset] !== tokens[offset]) { matches = false; break; }
        }
        const last = tokens[tokens.length - 1]!;
        bias[last] = f32(bias[last]! + (matches ? f32(value) : 0));
      }
      return row.map((score, token) => f32(score + bias[token]!));
    });
  };
}

/** ``NoBadWordsLogitsProcessor``. */
export function noBadWordsProcessor(badWords: unknown, eos: readonly number[] | null): LogitsProcessorFn {
  if (!Array.isArray(badWords) || badWords.length === 0) throw new ValueError(`\`bad_words_ids\` has to be a non-empty list, but is ${pyRepr(badWords)}.`);
  if (badWords.some((word) => !Array.isArray(word))) throw new ValueError(`\`bad_words_ids\` has to be a list of lists, but is ${pyRepr(badWords)}.`);
  if ((badWords as unknown[][]).some((word) => word.some((token) => typeof token !== 'number' || !Number.isInteger(token) || token < 0))) {
    throw new ValueError(`Each list in \`bad_words_ids\` has to be a list of positive integers, but is ${pyRepr(badWords)}.`);
  }
  let words = badWords as number[][];
  if (eos !== null) words = words.filter((word) => eos.every((id) => !(word.length === 1 && word[0] === id)));
  const map = new Map<string, [number[], number]>();
  for (const word of words) map.set(word.join(','), [[...word], -Infinity]);
  return sequenceBiasProcessor([...map.values()]);
}

/** ``PrefixConstrainedLogitsProcessor``. */
export function prefixConstrainedProcessor(fn: (batchId: number, ids: readonly number[]) => readonly number[], numBeams: number): LogitsProcessorFn {
  return (ids, scores) => mapRows(scores, (row, index) => {
    const batchId = Math.floor(index / numBeams);
    const allowed = fn(batchId, ids[index]!);
    if (allowed.length === 0) {
      throw new ValueError(`\`prefix_allowed_tokens_fn\` returned an empty list for batch ID ${batchId}.This means that the constraint is unsatisfiable. Please check your implementationof \`prefix_allowed_tokens_fn\` `);
    }
    const mask = new Float32Array(row.length).fill(-Infinity);
    for (const id of allowed) mask[id] = 0;
    return row.map((value, token) => f32(value + mask[token]!));
  });
}

/** ``ForcedBOSTokenLogitsProcessor``. */
export function forcedBosProcessor(bos: number): LogitsProcessorFn {
  return (ids, scores) => {
    if ((ids[0]?.length ?? 0) !== 1) return scores;
    return mapRows(scores, (row) => {
      const out = new Float32Array(row.length).fill(-Infinity);
      out[bos] = 0;
      return out;
    });
  };
}

/** ``ForcedEOSTokenLogitsProcessor``. */
export function forcedEosProcessor(maxLength: number, eos: readonly number[]): LogitsProcessorFn {
  const tokens = eosList(eos);
  return (ids, scores) => {
    if ((ids[0]?.length ?? 0) !== maxLength - 1) return scores;
    return mapRows(scores, (row) => {
      const out = new Float32Array(row.length).fill(-Infinity);
      for (const id of tokens) out[id] = 0;
      return out;
    });
  };
}

/** ``InfNanRemoveLogitsProcessor``. */
export function infNanRemoveProcessor(): LogitsProcessorFn {
  return (_, scores) => mapRows(scores, (row) => row.map((value) => {
    if (Number.isNaN(value)) return 0;
    if (value === Infinity) return FLOAT32_MAX;
    if (value === -Infinity) return -FLOAT32_MAX;
    return value;
  }));
}

/** ``ExponentialDecayLengthPenalty``. */
export function exponentialDecayLengthPenalty(setting: unknown, eos: readonly number[] | null, promptLength: number): LogitsProcessorFn {
  const pair = setting as [number, number];
  const start = pair[0]! + promptLength;
  const factor = pair[1]!;
  const tokens = eosList(eos);
  return (ids, scores) => {
    const length = ids[0]?.length ?? 0;
    if (length <= start) return scores;
    const multiplier = f32(factor ** (length - start) - 1);
    return mapRows(scores, (row) => {
      const penalties = new Float32Array(row.length);
      for (const id of tokens) penalties[id] = f32(Math.abs(row[id]!) * multiplier);
      return row.map((value, token) => f32(value + penalties[token]!));
    });
  };
}

/** ``LogitNormalization``. */
export function logitNormalization(): LogitsProcessorFn {
  return (_, scores) => scores.map(logSoftmaxRow);
}

/** ``SuppressTokensLogitsProcessor``. */
export function suppressTokensProcessor(tokens: readonly number[]): LogitsProcessorFn {
  return (_, scores) => mapRows(scores, (row) => {
    const out = Float32Array.from(row);
    for (const id of tokens) if (id >= 0 && id < out.length) out[id] = -Infinity;
    return out;
  });
}

/** ``SuppressTokensAtBeginLogitsProcessor``. */
export function suppressTokensAtBeginProcessor(tokens: readonly number[], beginIndex: number): LogitsProcessorFn {
  const suppress = suppressTokensProcessor(tokens);
  return (ids, scores) => ((ids[0]?.length ?? 0) === beginIndex ? suppress(ids, scores) : scores);
}

// ---------------------------------------------------------------------------
// Watermarking (``WatermarkLogitsProcessor``) with PyTorch's CPU generator.
// ---------------------------------------------------------------------------

/** ``torch.randperm(n, generator=generator)`` as int32 token ids. */
function randpermIds(n: number, generator: Generator): Int32Array {
  return Int32Array.from(randpermValues(n, generator));
}

/** ``WatermarkingConfig`` fields. */
export interface WatermarkSettings {
  greenlist_ratio: number;
  bias: number;
  hashing_key: number;
  seeding_scheme: string;
  context_width: number;
}

export const WATERMARK_DEFAULTS: WatermarkSettings = {
  greenlist_ratio: 0.25, bias: 2.0, hashing_key: 15485863, seeding_scheme: 'lefthash', context_width: 1,
};

/** ``WatermarkingConfig.from_dict`` then ``validate`` (a JSON object of its fields). */
export function watermarkingFromDict(value: Record<string, unknown>): WatermarkSettings {
  const known = new Set(Object.keys(WATERMARK_DEFAULTS));
  for (const key of Object.keys(value)) {
    if (!known.has(key)) throw new TypeError(`WatermarkingConfig.__init__() got an unexpected keyword argument '${key}'`);
  }
  const config = { ...WATERMARK_DEFAULTS, ...value } as WatermarkSettings;
  const message = (key: string, correct: string, found: unknown): string =>
    `Some of the keys in \`watermarking_config\` are defined incorrectly. \`${key}\` should be ${correct}\` but found ${String(found)}`;
  if (!['selfhash', 'lefthash'].includes(config.seeding_scheme)) {
    throw new ValueError(message('seeding_scheme', '[`selfhash`, `lefthash`]', config.seeding_scheme));
  }
  if (!(config.greenlist_ratio >= 0 && config.greenlist_ratio <= 1)) {
    throw new ValueError(message('greenlist_ratio', 'in range between 0.0 and 1.0', config.seeding_scheme));
  }
  if (!(config.context_width >= 1)) throw new ValueError(message('context_width', 'a positive integer', config.context_width));
  return config;
}

/** ``WatermarkLogitsProcessor``. */
export function watermarkProcessor(vocabSize: number, config: WatermarkSettings): LogitsProcessorFn {
  if (!['selfhash', 'lefthash'].includes(config.seeding_scheme)) {
    throw new ValueError(`seeding_scheme has to be one of [\`selfhash\`, \`lefthash\`], but found ${config.seeding_scheme}`);
  }
  if (config.greenlist_ratio >= 1 || config.greenlist_ratio <= 0) {
    throw new ValueError(`greenlist_ratio has be in range between 0.0 and 1.0, exclusively. but found ${pyFloat(config.greenlist_ratio)}`);
  }
  const greenlistSize = Math.trunc(vocabSize * config.greenlist_ratio);
  const rng = new Generator();
  const hashKey = BigInt(config.hashing_key);
  const tableSize = 1_000_003;
  let table: Int32Array | null = null;
  const fixedTable = (): Int32Array => {
    table ??= randpermIds(tableSize, new Generator(hashKey));
    return table;
  };
  const modulus = 2n ** 64n - 1n;
  const setSeed = (sequence: readonly number[]): void => {
    const window = sequence.slice(-config.context_width);
    let seed: bigint;
    if (config.seeding_scheme === 'selfhash') {
      const t = fixedTable();
      const last = BigInt(t[window[window.length - 1]! % tableSize]! + 1);
      let best: bigint | null = null;
      for (const token of window) {
        const a = BigInt(t[token % tableSize]! + 1);
        const product = BigInt.asIntN(64, BigInt.asIntN(64, hashKey * a) * last);
        if (best === null || product < best) best = product;
      }
      seed = best!;
    } else {
      seed = hashKey * BigInt(window[window.length - 1]!);
    }
    rng.manualSeed(((seed % modulus) + modulus) % modulus);
  };
  const greenlist = (sequence: readonly number[]): Int32Array => {
    setSeed(sequence);
    return randpermIds(vocabSize, rng).subarray(0, greenlistSize);
  };
  const bias = f32(config.bias);
  return (ids, scores) => {
    if ((ids[0]?.length ?? 0) < config.context_width) return scores;
    return mapRows(scores, (row, index) => {
      const sequence = ids[index]!;
      let green: Iterable<number>;
      if (config.seeding_scheme === 'selfhash') {
        const predictions = sortedIndices(row, true);
        const chosen: number[] = [];
        for (let rank = 0; rank < 40; rank += 1) {
          if (rank >= predictions.length) throw new IndexError(`index ${rank} is out of bounds for dimension 0 with size ${predictions.length}`);
          const candidate = predictions[rank]!;
          if (greenlist([...sequence, candidate]).includes(candidate)) chosen.push(candidate);
        }
        // ``torch.tensor([])`` is a float tensor, which cannot index.
        if (!chosen.length) throw new IndexError('tensors used as indices must be long, int, byte or bool tensors');
        green = chosen;
      } else {
        green = greenlist(sequence);
      }
      const out = Float32Array.from(row);
      // ``scores[b, ids] = scores[b, ids] + bias``: repeated ids are written once.
      for (const id of new Set(green)) out[id] = f32(row[id]! + bias);
      return out;
    });
  };
}

// ---------------------------------------------------------------------------
// Stopping criteria.
// ---------------------------------------------------------------------------

/** A stopping criterion: per-sequence ``is_done`` flags. */
export type StoppingCriterionFn = (ids: readonly (readonly number[])[], newTokenLength?: number) => boolean[];

/** ``MaxLengthCriteria``. */
export function maxLengthCriteria(maxLength: number): StoppingCriterionFn {
  return (ids) => ids.map((row) => row.length >= maxLength);
}

/** ``time.time()`` with sub-millisecond resolution. */
function wallClock(): number {
  return (performance.timeOrigin + performance.now()) / 1000;
}

/** ``MaxTimeCriteria`` (wall clock, seconds). */
export function maxTimeCriteria(maxTime: number, now: () => number = wallClock): StoppingCriterionFn {
  const start = now();
  return (ids) => {
    const done = now() - start > maxTime;
    return ids.map(() => done);
  };
}

/** ``EosTokenCriteria``. */
export function eosTokenCriteria(eos: readonly number[]): StoppingCriterionFn {
  return (ids, newTokenLength = 1) => ids.map((row) => row.slice(row.length - newTokenLength).some((id) => eos.includes(id)));
}

/** Tokenizer surface ``StopStringCriteria`` needs (``get_vocab``, ``convert_tokens_to_string``, ...). */
export interface StopStringTokenizer {
  /** ``get_vocab()`` (token to id, including added tokens). */
  vocab(): Map<string, number>;
  /** ``tokenizer(text, add_special_tokens=False)['input_ids']``. */
  encodePlain(text: string): number[];
  /** ``_convert_id_to_token``. */
  idToToken(id: number): string | null;
  /** ``convert_tokens_to_string``. */
  tokensToString(tokens: string[]): string;
  /** The backend ``decoder`` JSON (``None`` when absent). */
  decoderConfig(): unknown;
}

/** GPT-2 ``bytes_to_unicode`` inverse: character to byte. */
function byteLevelDecoder(): Map<string, number> {
  const bytes: number[] = [];
  for (let b = 33; b <= 126; b += 1) bytes.push(b);
  for (let b = 161; b <= 172; b += 1) bytes.push(b);
  for (let b = 174; b <= 255; b += 1) bytes.push(b);
  const chars = [...bytes];
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(256 + n);
      n += 1;
    }
  }
  const map = new Map<string, number>();
  bytes.forEach((b, index) => map.set(String.fromCodePoint(chars[index]!), b));
  return map;
}

function decoderHasType(config: unknown, type: string): boolean {
  if (Array.isArray(config)) return config.some((value) => decoderHasType(value, type));
  if (config && typeof config === 'object') {
    const record = config as Record<string, unknown>;
    if (record.type === type) return true;
    return Object.values(record).some((value) => decoderHasType(value, type));
  }
  return false;
}

type Units = number[];

const utf8 = new TextEncoder();

function units(text: string, mode: string | null): Units {
  return mode === null ? Array.from(text, (char) => char.codePointAt(0)!) : [...utf8.encode(text)];
}

function startsWith(values: Units, prefix: Units): boolean {
  if (prefix.length > values.length) return false;
  for (let index = 0; index < prefix.length; index += 1) if (values[index] !== prefix[index]) return false;
  return true;
}

/**
 * ``StopStringCriteria``: a sequence stops when its trailing tokens spell one
 * of ``stopStrings`` (token-string matching, byte-level for byte-level and
 * byte-fallback tokenizers), with transformers' vectorized algorithm.
 */
export function stopStringCriteria(tokenizer: StopStringTokenizer, stopStrings: string | readonly string[]): StoppingCriterionFn {
  const strings = typeof stopStrings === 'string' ? [stopStrings] : [...stopStrings];
  const decoder = tokenizer.decoderConfig();
  let mode: string | null = null;
  if (decoder && typeof decoder === 'object') {
    if ((decoder as { type?: unknown }).type === 'ByteLevel') mode = 'byte_level';
    else if (decoderHasType(decoder, 'ByteFallback')) mode = 'byte_fallback';
    else if (decoderHasType(decoder, 'ByteLevel')) mode = 'byte_level';
  }
  const targets = strings.map((text) => units(text, mode));
  const vocab = tokenizer.vocab();
  // ``clean_tokenizer_vocab``.
  const byteDecoder = mode === 'byte_level' ? byteLevelDecoder() : null;
  const staticPrefix = 'abcdef';
  const base = tokenizer.encodePlain(staticPrefix).map((id) => tokenizer.idToToken(id) ?? '');
  const tokenList: Units[] = [];
  const tokenIndices: number[] = [];
  for (const [token, id] of vocab) {
    let clean: Units | null = null;
    if (mode === 'byte_level' && byteDecoder) {
      const chars = Array.from(token);
      if (chars.every((char) => byteDecoder.has(char))) clean = chars.map((char) => byteDecoder.get(char)!);
    } else if (mode === 'byte_fallback') {
      if (token.length === 6 && token.startsWith('<0x') && token.endsWith('>') && /^[0-9a-fA-F]{2}$/.test(token.slice(3, 5))) {
        clean = [parseInt(token.slice(3, 5), 16)];
      }
    }
    if (clean === null) {
      const text = tokenizer.tokensToString([...base, token]);
      const position = text.indexOf(staticPrefix);
      if (position < 0) throw new ValueError('substring not found');
      clean = units(text.slice(position + staticPrefix.length), mode);
    }
    tokenList.push(clean);
    tokenIndices.push(id);
  }
  // ``_stop_string_get_matching_positions``.
  const validPositions: Map<number, number[]>[] = [];
  const endOverlaps: Map<number, number[]>[] = [];
  for (const stop of targets) {
    const reversedStop = [...stop].reverse();
    const valid = new Map<number, number[]>();
    const ends = new Map<number, number[]>();
    tokenList.forEach((token, position) => {
      const reversedToken = [...token].reverse();
      const matching: number[] = [];
      const possible: number[] = [];
      for (let i = 1 - token.length; i < stop.length; i += 1) {
        let tok: Units;
        let offset = i;
        if (i < 0) {
          tok = reversedToken.slice(-i);
          offset = 0;
        } else tok = reversedToken;
        const piece = reversedStop.slice(offset, offset + tok.length);
        if (startsWith(tok, piece)) {
          if (offset === 0) possible.push(Math.min(tok.length, piece.length));
          else matching.push(offset);
        }
      }
      const id = tokenIndices[position]!;
      if (matching.length) valid.set(id, matching);
      if (possible.length) ends.set(id, possible);
    });
    validPositions.push(valid);
    endOverlaps.push(ends);
  }
  // ``_stop_string_create_embedding_vec``.
  const allValid = validPositions.flatMap((map) => [...map.values()].map((list) => list.length));
  const maxValid = allValid.length ? Math.max(...allValid) : 1;
  const allEnds = endOverlaps.flatMap((map) => [...map.values()].map((list) => list.length));
  if (!allEnds.length) {
    throw new ValueError('Stop string preprocessing was unable to identify tokens matching one or more of the supplied stop string(s). This is most often caused by the stop strings containing unusual characters that are not in the tokenizer vocabulary.');
  }
  const maxEnds = Math.max(...allEnds);
  const count = targets.length;
  const width = count * (maxValid + maxEnds) + 1;
  const rows = Math.max(...tokenIndices) + 2;
  const table = new Int32Array(rows * width).fill(-1);
  for (let s = 0; s < count; s += 1) {
    for (const [id, list] of validPositions[s]!) list.forEach((value, k) => { table[id * width + maxValid * s + k] = value; });
    for (const [id, list] of endOverlaps[s]!) list.forEach((value, k) => { table[id * width + maxValid * count + maxEnds * s + k] = value; });
  }
  tokenList.forEach((token, position) => { table[tokenIndices[position]! * width + width - 1] = token.length; });
  const maximumTokenLength = Math.max(...targets.map((target) => target.length));
  const targetLengths = targets.map((target) => target.length);
  return (ids) => ids.map((sequence) => {
    const window = sequence.slice(Math.max(0, sequence.length - maximumTokenLength)).reverse().map((id) => Math.min(id, rows - 1));
    const length = window.length;
    if (!length) return false;
    const at = (position: number, column: number): number => table[window[position]! * width + column]!;
    for (let s = 0; s < count; s += 1) {
      let best = -Infinity;
      for (let k = 0; k < maxEnds; k += 1) {
        const endLength = at(0, maxValid * count + maxEnds * s + k);
        let cumulative = endLength;
        let alive = endLength > 0;
        let value = alive ? cumulative : 0;
        if (value > best) best = value;
        for (let p = 1; p < length; p += 1) {
          const previous = cumulative;
          cumulative += at(p, width - 1);
          if (alive) {
            let matched = false;
            for (let v = 0; v < maxValid; v += 1) if (at(p, maxValid * s + v) === previous) { matched = true; break; }
            alive = matched;
          }
          value = alive ? cumulative : 0;
          if (value > best) best = value;
        }
      }
      if (best >= targetLengths[s]!) return true;
    }
    return false;
  });
}
