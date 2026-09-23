/**
 * Explicitly supplied local Transformers.js models, with no implicit model
 * selection (Python ``tensorcode/integrations/local.py``).
 *
 * ``@huggingface/transformers`` is an optional peer dependency. It is never
 * imported at module load: only ``LocalModel.fromPretrained`` and the default
 * image decoder import it, and they raise {@link MissingDependencyError} when
 * it is absent.
 */
import { MissingDependencyError, ValueError } from '../errors.js';
import { isPlainObject, pythonJsonDumps } from '../_internal/json.js';
import { ImagePart, TextPart } from '../ops/text/messages.js';
import { ModelOutput, type ModelRequest } from '../ops/text/model.js';

/** A supplied image/text generation model (Transformers.js ``PreTrainedModel`` shape). */
export interface LocalGenerationModel {
  generate(inputs: Record<string, unknown>): unknown;
  eval?(): unknown;
  generation_config?: { eos_token_id?: number | number[] | null } | null;
}

/** A supplied processor (Transformers.js ``Processor`` shape). */
export interface LocalProcessor {
  apply_chat_template(messages: unknown[], options: Record<string, unknown>): unknown;
  batch_decode(sequences: number[][], options: Record<string, unknown>): string[] | Promise<string[]>;
  (text: string, images?: unknown[] | null, options?: Record<string, unknown>): unknown;
}

export interface LocalModelOptions {
  /** Identifies the supplied model (reported in configuration and metadata). */
  modelId: string;
  revision?: string | null;
  /** Generation budget (default 128). */
  maxNewTokens?: number;
  /** Decode supplied image bytes for the processor (default: Transformers.js ``RawImage``). */
  loadImage?: (part: ImagePart) => unknown;
}

const TRANSFORMERS = '@huggingface/transformers';

async function importTransformers(): Promise<Record<string, unknown>> {
  try {
    return await import(TRANSFORMERS) as Record<string, unknown>;
  } catch (error) {
    throw new MissingDependencyError(
      "LocalModel requires the optional peer dependency '@huggingface/transformers'; install it explicitly", { cause: error },
    );
  }
}

/** Decode image bytes with Transformers.js ``RawImage`` (RGB). */
async function rawImage(part: ImagePart): Promise<unknown> {
  const transformers = await importTransformers();
  const RawImage = transformers.RawImage as { fromBlob(blob: Blob): Promise<{ rgb(): unknown }> };
  const image = await RawImage.fromBlob(new Blob([part.data!], part.mediaType ? { type: part.mediaType } : {}));
  return image.rgb();
}

function rows(value: unknown): number[][] {
  const list = typeof (value as { tolist?: unknown })?.tolist === 'function' ? (value as { tolist(): unknown }).tolist() : value;
  if (!Array.isArray(list)) throw new TypeError('model.generate must return token ids');
  const matrix = Array.isArray(list[0]) ? list : [list];
  return (matrix as unknown[][]).map((row) => row.map((id) => Number(id)));
}

function promptLength(inputs: Record<string, unknown>): number {
  const ids = inputs.input_ids as { dims?: number[]; shape?: number[]; tolist?: () => unknown } | unknown[] | undefined;
  if (Array.isArray(ids)) return Array.isArray(ids[0]) ? (ids[0] as unknown[]).length : ids.length;
  const dims = ids?.dims ?? ids?.shape;
  if (Array.isArray(dims) && dims.length) return dims[dims.length - 1]!;
  if (ids && typeof ids.tolist === 'function') return rows(ids)[0]!.length;
  throw new TypeError('processor inputs must include input_ids');
}

/**
 * Adapt a supplied image/text generation model and processor.
 *
 * ``fromPretrained`` explicitly loads the caller's model ID or local
 * directory. Image URLs are not fetched: supply image bytes. Structured
 * answers are generated JSON, never repaired or assigned invented confidence;
 * operation-level validators still validate their schemas. Inference is
 * serialized for shared model safety. The adapter is asynchronous.
 */
export class LocalModel {
  static readonly qualifiedName: string = 'tensorcode.integrations.local.LocalModel';
  readonly model: LocalGenerationModel;
  readonly processor: LocalProcessor;
  readonly modelId: string;
  readonly revision: string | null;
  readonly maxNewTokens: number;
  readonly #loadImage: (part: ImagePart) => unknown;
  #queue: Promise<unknown> = Promise.resolve();

  constructor(model: LocalGenerationModel, processor: LocalProcessor, options: LocalModelOptions) {
    const { modelId, revision = null, maxNewTokens = 128 } = options;
    if (typeof modelId !== 'string' || !modelId) throw new ValueError('model_id must identify the supplied model');
    if (typeof maxNewTokens !== 'number' || !Number.isInteger(maxNewTokens) || maxNewTokens < 1) {
      throw new ValueError('max_new_tokens must be a positive integer');
    }
    if (typeof model?.generate !== 'function') throw new TypeError('model must provide generate()');
    if (typeof model.eval === 'function') model.eval();
    this.model = model;
    this.processor = processor;
    this.modelId = modelId;
    this.revision = revision;
    this.maxNewTokens = maxNewTokens;
    this.#loadImage = options.loadImage ?? rawImage;
  }

  /**
   * Load an explicit model; downloads require ``localFilesOnly: false``.
   * Requires the optional peer dependency ``@huggingface/transformers``.
   */
  static async fromPretrained(modelId: string, options: {
    revision?: string | null; localFilesOnly?: boolean; device?: string; maxNewTokens?: number;
  } = {}): Promise<LocalModel> {
    const transformers = await importTransformers();
    const settings: Record<string, unknown> = { local_files_only: options.localFilesOnly ?? true, device: options.device ?? 'cpu' };
    if (options.revision) settings.revision = options.revision;
    const processorClass = transformers.AutoProcessor as { from_pretrained(id: string, o: unknown): Promise<LocalProcessor> };
    const modelClass = (transformers.AutoModelForImageTextToText ?? transformers.AutoModelForVision2Seq) as
      { from_pretrained(id: string, o: unknown): Promise<LocalGenerationModel> } | undefined;
    if (!modelClass) throw new MissingDependencyError('the installed @huggingface/transformers has no image-text-to-text models');
    const processor = await processorClass.from_pretrained(modelId, settings);
    const model = await modelClass.from_pretrained(modelId, settings);
    return new LocalModel(model, processor, {
      modelId, revision: options.revision ?? null, ...(options.maxNewTokens === undefined ? {} : { maxNewTokens: options.maxNewTokens }),
    });
  }

  /** JSON description of this adapter; never includes credentials. */
  configuration(): Record<string, unknown> {
    return { model_id: this.modelId, revision: this.revision, max_new_tokens: this.maxNewTokens };
  }

  /** Send one request and return a ``ModelOutput`` (inference is serialized). */
  acomplete(request: ModelRequest): Promise<ModelOutput> {
    const result = this.#queue.then(() => this.run(request));
    this.#queue = result.catch(() => undefined);
    return result;
  }

  /** Explicit sequential fallback; no claim of native batched generation. */
  async acompleteBatch(requests: readonly ModelRequest[]): Promise<readonly ModelOutput[]> {
    const outputs: ModelOutput[] = [];
    for (const request of requests) outputs.push(await this.acomplete(request));
    return Object.freeze(outputs);
  }

  private async run(request: ModelRequest): Promise<ModelOutput> {
    const messages: Record<string, unknown>[] = [];
    const images: unknown[] = [];
    let instructions = request.instructions ?? '';
    if (request.responseSchema !== null) {
      instructions += '\nReturn only a JSON object, without Markdown fences or explanation, matching this schema:\n'
        + pythonJsonDumps({ ...request.responseSchema }, { sortKeys: true, floatKeys: new Set() });
    }
    if (instructions) messages.push({ role: 'system', content: [{ type: 'text', text: instructions }] });
    for (const message of request.messages) {
      const parts = typeof message.content === 'string' ? [new TextPart(message.content)] : message.content;
      const content: Record<string, unknown>[] = [];
      for (const part of parts) {
        if (part instanceof TextPart) content.push({ type: 'text', text: part.text });
        else if (part instanceof ImagePart) {
          if (part.data === null) throw new ValueError('LocalModel requires image bytes; fetch URLs explicitly');
          images.push(await this.#loadImage(part));
          content.push({ type: 'image' });
        } else throw new TypeError('Unsupported message part');
      }
      messages.push({ role: message.role, content });
    }
    const prompt = await this.processor.apply_chat_template(messages, { add_generation_prompt: true, tokenize: false });
    if (typeof prompt !== 'string') throw new TypeError('processor.apply_chat_template must return the prompt text');
    const inputs = await this.processor(prompt, images.length ? images : null) as Record<string, unknown>;
    if (inputs === null || typeof inputs !== 'object') throw new TypeError('processor must return model inputs');
    const length = promptLength(inputs);
    const output = await this.model.generate({ ...inputs, max_new_tokens: this.maxNewTokens, do_sample: false });
    const generated = rows(output).map((row) => row.slice(length));
    const eos = this.model.generation_config?.eos_token_id ?? null;
    const stopTokens = eos === null ? [] : Array.isArray(eos) ? eos.map(Number) : [Number(eos)];
    const count = generated[0]?.length ?? 0;
    const finished = count > 0 && stopTokens.includes(generated[0]![count - 1]!);
    if (count >= this.maxNewTokens && !finished) throw new ValueError('Local model reached the token limit before finishing its answer');
    const decoded = await this.processor.batch_decode(generated, { skip_special_tokens: true });
    const answer = String(decoded[0] ?? '').trim();
    const providerMetadata = {
      model_id: this.modelId, revision: this.revision, backend: 'transformers.js', source: 'supplied_pretrained_model',
      generated_tokens: count, finish_reason: 'stop',
    };
    if (request.responseSchema !== null) {
      let structured: unknown;
      try {
        structured = JSON.parse(answer);
      } catch (error) {
        throw new ValueError('Local model did not return valid JSON', { cause: error });
      }
      if (!isPlainObject(structured)) throw new ValueError('Local model JSON must be an object');
      return new ModelOutput({ text: answer, structured, providerMetadata });
    }
    return new ModelOutput({ text: answer, providerMetadata });
  }
}
