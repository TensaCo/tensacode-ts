/**
 * Explicitly supplied local Transformers.js models, with no implicit model
 * selection (Python ``tensorcode/integrations/local.py``).
 *
 * ``@huggingface/transformers`` is an optional peer dependency. It is never
 * imported at module load: only ``LocalModel.fromPretrained``, the default
 * image decoder and the synchronous ``complete`` import it, and they raise
 * {@link MissingDependencyError} when it is absent.
 */
import { MissingDependencyError, ValueError } from '../errors.js';
import { isPlainObject, pythonJsonDumps, pythonJsonLoads, type JsonValue } from '../_internal/json.js';
import { ImagePart, TextPart } from '../ops/text/messages.js';
import { openImage, type RasterImage } from '../_internal/image/index.js';
import { ModelOutput, type ModelRequest } from '../ops/text/model.js';
import { blockingCall, moduleUrl, postWithoutWaiting, resolveModuleUrl } from './blocking.js';
import { runLocalEngine, type LocalEngineResult, type LocalImageInput } from './localEngine.js';

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
  /** The processor's text tokenizer; text-only requests are tokenized with it when present. */
  tokenizer?: ((text: string, options?: Record<string, unknown>) => unknown) | null;
}

/**
 * How the synchronous ``complete`` obtains the model in its worker thread.
 *
 * A model object lives on one thread, so a blocking call cannot use the
 * supplied instance. ``module`` (a path or URL) is imported in the worker and
 * its ``exportName`` export (default ``default``) is called with ``args``; it
 * returns ``{ model, processor, loadImage? }`` like the constructor's
 * arguments. ``LocalModel.fromPretrained`` fills this in automatically.
 */
export interface LocalWorkerLoader {
  readonly module: string | URL;
  readonly exportName?: string;
  /** Structured-clonable argument for the loader. */
  readonly args?: unknown;
}

export interface LocalModelOptions {
  /** Identifies the supplied model (reported in configuration and metadata). */
  modelId: string;
  revision?: string | null;
  /** Generation budget (default 128). */
  maxNewTokens?: number;
  /** Decode supplied image bytes for the processor (default: Transformers.js ``RawImage``). */
  loadImage?: (part: ImagePart) => unknown;
  /** Worker-thread loader used by the synchronous ``complete`` (see {@link LocalWorkerLoader}). */
  worker?: LocalWorkerLoader | null;
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

/**
 * Decode image bytes like Python's ``Image.open(BytesIO(data)).convert('RGB')``
 * (TensorCode's Pillow-exact decoder) into a Transformers.js ``RawImage``;
 * formats that decoder does not read fall back to ``RawImage.fromBlob``.
 */
async function rawImage(part: ImagePart): Promise<unknown> {
  const transformers = await importTransformers();
  const RawImage = transformers.RawImage as {
    new (data: Uint8Array, width: number, height: number, channels: number): unknown;
    fromBlob(blob: Blob): Promise<{ rgb(): unknown }>;
  };
  let decoded: RasterImage | null = null;
  try {
    decoded = openImage(part.data!).convert('RGB');
  } catch {
    decoded = null;
  }
  if (decoded) return new RawImage(decoded.data as Uint8Array, decoded.width, decoded.height, 3);
  const image = await RawImage.fromBlob(new Blob([part.data!], part.mediaType ? { type: part.mediaType } : {}));
  return image.rgb();
}

interface PreparedRequest {
  readonly messages: Record<string, unknown>[];
  readonly images: ImagePart[];
}

/** ``fromPretrained`` arguments, replayed in the worker for ``complete``. */
interface PretrainedSource {
  readonly modelId: string;
  readonly settings: Record<string, unknown>;
}

let nextWorkerHandle = 1;

/** Releases a collected adapter's worker-thread model copy. */
const workerModels = typeof FinalizationRegistry === 'function'
  ? new FinalizationRegistry<number>((handle) => postWithoutWaiting({ kind: 'local-dispose', handle }))
  : null;

/**
 * Adapt a supplied image/text generation model and processor.
 *
 * ``fromPretrained`` explicitly loads the caller's model ID or local
 * directory. Image URLs are not fetched: supply image bytes. Structured
 * answers are generated JSON, never repaired or assigned invented confidence;
 * operation-level validators still validate their schemas. Inference is
 * serialized for shared model safety.
 *
 * ``complete``/``completeBatch`` block like Python's: they run the same
 * generation in a worker thread (with its own copy of the model, loaded on
 * first use) while the calling thread waits. ``acomplete``/``acompleteBatch``
 * use the supplied model directly.
 */
export class LocalModel {
  static readonly qualifiedName: string = 'tensorcode.integrations.local.LocalModel';
  readonly model: LocalGenerationModel;
  readonly processor: LocalProcessor;
  readonly modelId: string;
  readonly revision: string | null;
  readonly maxNewTokens: number;
  readonly #loadImage: (part: ImagePart) => unknown;
  readonly #worker: LocalWorkerLoader | null;
  #pretrained: PretrainedSource | null = null;
  #workerHandle: number | null = null;
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
    const worker = options.worker ?? null;
    if (worker !== null && (typeof worker !== 'object' || !(typeof worker.module === 'string' || worker.module instanceof URL))) {
      throw new TypeError('worker must be { module, exportName?, args? }');
    }
    this.#worker = worker;
  }

  /**
   * Load an explicit model; downloads require ``localFilesOnly: false``.
   * Requires the optional peer dependency ``@huggingface/transformers``.
   */
  static async fromPretrained(modelId: string, options: {
    revision?: string | null; localFilesOnly?: boolean; device?: string; maxNewTokens?: number;
    /** Transformers.js weight precision (for example ``fp32`` or ``q8``); default: the library's choice. */
    dtype?: string | Record<string, string> | null;
  } = {}): Promise<LocalModel> {
    const transformers = await importTransformers();
    const settings: Record<string, unknown> = { local_files_only: options.localFilesOnly ?? true, device: options.device ?? 'cpu' };
    if (options.revision) settings.revision = options.revision;
    if (options.dtype) settings.dtype = options.dtype;
    const processorClass = transformers.AutoProcessor as { from_pretrained(id: string, o: unknown): Promise<LocalProcessor> };
    const modelClass = (transformers.AutoModelForImageTextToText ?? transformers.AutoModelForVision2Seq) as
      { from_pretrained(id: string, o: unknown): Promise<LocalGenerationModel> } | undefined;
    if (!modelClass) throw new MissingDependencyError('the installed @huggingface/transformers has no image-text-to-text models');
    const processor = await processorClass.from_pretrained(modelId, settings);
    const model = await modelClass.from_pretrained(modelId, settings);
    const local = new LocalModel(model, processor, {
      modelId, revision: options.revision ?? null, ...(options.maxNewTokens === undefined ? {} : { maxNewTokens: options.maxNewTokens }),
    });
    local.#pretrained = { modelId, settings };
    return local;
  }

  /** JSON description of this adapter; never includes credentials. */
  configuration(): Record<string, unknown> {
    return { model_id: this.modelId, revision: this.revision, max_new_tokens: this.maxNewTokens };
  }

  /** Send one request and return a ``ModelOutput`` (blocks until generation finishes). */
  complete(request: ModelRequest): ModelOutput {
    const prepared = this.prepare(request);
    const handle = this.workerModel();
    const images: LocalImageInput[] = prepared.images.map((part) => ({
      data: part.data!, mediaType: part.mediaType, url: part.url, detail: part.detail, sourceRef: part.sourceRef,
    }));
    const result = blockingCall<LocalEngineResult>({
      kind: 'local-run', handle, input: { messages: prepared.messages, images, maxNewTokens: this.maxNewTokens },
    });
    return this.finish(request, result);
  }

  /** Explicit sequential fallback; no claim of native batched generation. */
  completeBatch(requests: readonly ModelRequest[]): readonly ModelOutput[] {
    return Object.freeze(requests.map((request) => this.complete(request)));
  }

  /** Asynchronous {@link complete} with the supplied model (inference is serialized). */
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

  /** Load (once) this model in the blocking worker; returns its handle. */
  private workerModel(): number {
    if (this.#workerHandle !== null) return this.#workerHandle;
    const transformersUrl = resolveModuleUrl(TRANSFORMERS, import.meta.url);
    const handle = nextWorkerHandle;
    nextWorkerHandle += 1;
    if (this.#worker !== null) {
      blockingCall({
        kind: 'local-load', handle, loaderUrl: moduleUrl(this.#worker.module), exportName: this.#worker.exportName ?? 'default',
        args: (this.#worker.args ?? null) as JsonValue, transformersUrl,
      });
    } else if (this.#pretrained !== null) {
      if (transformersUrl === null) {
        throw new MissingDependencyError("LocalModel requires the optional peer dependency '@huggingface/transformers'; install it explicitly");
      }
      blockingCall({ kind: 'local-load', handle, transformersUrl, modelId: this.#pretrained.modelId, settings: this.#pretrained.settings });
    } else {
      throw new TypeError(
        'LocalModel.complete runs the model in a worker thread and cannot use a supplied model object; '
        + 'create the adapter with LocalModel.fromPretrained, pass options.worker = { module, exportName } that loads it, '
        + 'or use await acomplete(...)',
      );
    }
    this.#workerHandle = handle;
    workerModels?.register(this, handle);
    return handle;
  }

  /** Messages with image placeholders, and the image parts in order (Python's message loop). */
  private prepare(request: ModelRequest): PreparedRequest {
    const messages: Record<string, unknown>[] = [];
    const images: ImagePart[] = [];
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
          images.push(part);
          content.push({ type: 'image' });
        } else throw new TypeError('Unsupported message part');
      }
      messages.push({ role: message.role, content });
    }
    return { messages, images };
  }

  private async run(request: ModelRequest): Promise<ModelOutput> {
    const prepared = this.prepare(request);
    const result = await runLocalEngine(this.model, this.processor as never, (image) => this.#loadImage(image as ImagePart), {
      messages: prepared.messages, images: prepared.images, maxNewTokens: this.maxNewTokens,
    });
    return this.finish(request, result);
  }

  private finish(request: ModelRequest, result: LocalEngineResult): ModelOutput {
    if (result.answer === null) throw new ValueError('Local model reached the token limit before finishing its answer');
    const answer = result.answer;
    const providerMetadata = {
      model_id: this.modelId, revision: this.revision, backend: 'transformers.js', source: 'supplied_pretrained_model',
      generated_tokens: result.count, finish_reason: 'stop',
    };
    if (request.responseSchema !== null) {
      let structured: unknown;
      try {
        structured = pythonJsonLoads(answer);
      } catch (error) {
        throw new ValueError('Local model did not return valid JSON', { cause: error });
      }
      if (!isPlainObject(structured)) throw new ValueError('Local model JSON must be an object');
      return new ModelOutput({ text: answer, structured, providerMetadata });
    }
    return new ModelOutput({ text: answer, providerMetadata });
  }
}
