/**
 * The model-facing half of ``LocalModel.complete`` (Python
 * ``tensorcode/integrations/local.py``): chat template, image decoding,
 * processor call, greedy generation and decoding.
 *
 * ``runLocalEngine`` is deliberately self-contained (no imports, no closures,
 * no TypeScript-only runtime helpers): the synchronous ``LocalModel.complete``
 * runs the same function source inside its worker thread.
 */

/** A plain, structured-clonable copy of an ``ImagePart``. */
export interface LocalImageInput {
  readonly data: Uint8Array;
  readonly mediaType: string | null;
  readonly url: string | null;
  readonly detail: string | null;
  readonly sourceRef: string | null;
}

/** Chat messages with ``{type: 'image'}`` placeholders for ``images`` (in order). */
export interface LocalEngineInput {
  readonly messages: Record<string, unknown>[];
  readonly images: readonly unknown[];
  readonly maxNewTokens: number;
}

export interface LocalEngineResult {
  /** Number of generated tokens (after the prompt). */
  readonly count: number;
  /** Whether the last generated token is an end-of-sequence token. */
  readonly finished: boolean;
  /** The decoded, stripped answer (``null`` when generation was cut off). */
  readonly answer: string | null;
}

/**
 * Generate for one prepared request. ``loadImage`` decodes each image input
 * for the processor. Throws ``TypeError`` for incompatible model/processor
 * shapes. Does not decode an answer that reached the token limit unfinished.
 */
export async function runLocalEngine(
  model: { generate(inputs: Record<string, unknown>): unknown; generation_config?: { eos_token_id?: unknown } | null },
  processor: {
    apply_chat_template(messages: unknown[], options: Record<string, unknown>): unknown;
    batch_decode(sequences: number[][], options: Record<string, unknown>): unknown;
    tokenizer?: unknown;
  } & ((text: string, images?: unknown[] | null, options?: Record<string, unknown>) => unknown),
  loadImage: (image: unknown) => unknown,
  input: LocalEngineInput,
): Promise<LocalEngineResult> {
  const rows = (value: unknown): number[][] => {
    const list = value !== null && typeof value === 'object' && typeof (value as { tolist?: unknown }).tolist === 'function'
      ? (value as { tolist(): unknown }).tolist()
      : value;
    if (!Array.isArray(list)) throw new TypeError('model.generate must return token ids');
    const matrix = Array.isArray(list[0]) ? list : [list];
    return (matrix as unknown[][]).map((row) => row.map((id) => Number(id)));
  };
  const promptLength = (inputs: Record<string, unknown>): number => {
    const ids = inputs.input_ids as { dims?: number[]; shape?: number[]; tolist?: () => unknown } | unknown[] | undefined;
    if (Array.isArray(ids)) return Array.isArray(ids[0]) ? (ids[0] as unknown[]).length : ids.length;
    const dims = ids?.dims ?? ids?.shape;
    if (Array.isArray(dims) && dims.length) return dims[dims.length - 1]!;
    if (ids && typeof ids.tolist === 'function') return rows(ids)[0]!.length;
    throw new TypeError('processor inputs must include input_ids');
  };
  const images: unknown[] = [];
  for (const image of input.images) images.push(await loadImage(image));
  const prompt = await processor.apply_chat_template(input.messages, { add_generation_prompt: true, tokenize: false });
  if (typeof prompt !== 'string') throw new TypeError('processor.apply_chat_template must return the prompt text');
  // Text-only prompts go through the processor's tokenizer, which is what a
  // Python processor does without images. Transformers.js multimodal
  // processors (for example Idefics3/SmolVLM) fail when called without images.
  const tokenizer = processor.tokenizer;
  const inputs = await (images.length || typeof tokenizer !== 'function'
    ? processor(prompt, images.length ? images : null)
    : (tokenizer as (text: string) => unknown)(prompt)) as Record<string, unknown>;
  if (inputs === null || typeof inputs !== 'object') throw new TypeError('processor must return model inputs');
  const length = promptLength(inputs);
  const output = await model.generate({ ...inputs, max_new_tokens: input.maxNewTokens, do_sample: false });
  const generated = rows(output).map((row) => row.slice(length));
  const eos = model.generation_config?.eos_token_id ?? null;
  const stopTokens = eos === null ? [] : Array.isArray(eos) ? eos.map(Number) : [Number(eos)];
  const count = generated[0]?.length ?? 0;
  const finished = count > 0 && stopTokens.includes(generated[0]![count - 1]!);
  if (count >= input.maxNewTokens && !finished) return { count, finished, answer: null };
  const decoded = await processor.batch_decode(generated, { skip_special_tokens: true }) as unknown[];
  return { count, finished, answer: String(decoded[0] ?? '').trim() };
}
