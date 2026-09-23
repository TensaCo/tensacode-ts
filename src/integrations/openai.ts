/** OpenAI-compatible Chat Completions and Responses HTTP model adapter (Python ``tensorcode/integrations/openai.py``). */
import { ValueError } from '../errors.js';
import { isPlainObject, pythonJsonLoads } from '../_internal/json.js';
import { ImagePart, TextPart, type Message, type MessagePart } from '../ops/text/messages.js';
import { ModelOutput, ModelRequest } from '../ops/text/model.js';
import { ProviderProtocolError, endpoint, postJson, postJsonSync, pythonRepr, validateTimeout, type FetchLike } from './http.js';

export type OpenAIApi = 'chat_completions' | 'responses';

export interface OpenAICompatibleModelOptions {
  /** Absolute HTTP(S) base URL, for example ``https://api.openai.com/v1``. */
  baseUrl: string;
  model: string;
  apiKey?: string | null;
  /** Request timeout in seconds (default 30). */
  timeout?: number;
  api?: OpenAIApi;
  /** Injected ``fetch`` (tests, proxies); defaults to the global ``fetch``. */
  fetch?: FetchLike | null;
}

type Json = Record<string, unknown>;

function parts(content: Message['content']): readonly MessagePart[] {
  return typeof content === 'string' ? [new TextPart(content)] : content;
}

/** A provider-safe response schema name (``[A-Za-z0-9_-]``, at most 64 characters). */
export function schemaName(value: string | null | undefined): string {
  const normalized = (value || 'tensorcode_response').replace(/[^A-Za-z0-9_-]/gu, '_');
  return Array.from(normalized).slice(0, 64).join('') || 'tensorcode_response';
}

/** Image bytes as a media-typed base64 data URL; URLs stay URLs. */
export function imageUrl(part: ImagePart): string {
  if (part.url !== null) return part.url;
  const mediaType = part.mediaType || 'application/octet-stream';
  return `data:${mediaType};base64,${Buffer.from(part.data!).toString('base64')}`;
}

export function chatMessage(message: Message): Json {
  if (typeof message.content === 'string') return { role: message.role, content: message.content };
  const content: Json[] = [];
  for (const part of parts(message.content)) {
    if (part instanceof TextPart) content.push({ type: 'text', text: part.text });
    else if (part instanceof ImagePart) {
      const image: Json = { url: imageUrl(part) };
      if (part.detail !== null) image.detail = part.detail;
      content.push({ type: 'image_url', image_url: image });
    } else throw new TypeError('Unsupported message part');
  }
  return { role: message.role, content };
}

export function responsesMessage(message: Message): Json {
  const content: Json[] = [];
  for (const part of parts(message.content)) {
    if (part instanceof TextPart) content.push({ type: 'input_text', text: part.text });
    else if (part instanceof ImagePart) {
      const item: Json = { type: 'input_image', image_url: imageUrl(part) };
      if (part.detail !== null) item.detail = part.detail;
      content.push(item);
    } else throw new TypeError('Unsupported message part');
  }
  return { role: message.role, content };
}

function chatText(response: Json): string {
  const choices = response.choices;
  const choice = Array.isArray(choices) ? choices[0] : undefined;
  if (!isPlainObject(choice) || !Object.hasOwn(choice, 'finish_reason') || !isPlainObject(choice.message)
    || !Object.hasOwn(choice.message, 'content')) {
    throw new ProviderProtocolError('Chat completion response has no assistant content');
  }
  const finishReason = choice.finish_reason;
  const message = choice.message;
  if (finishReason !== 'stop') throw new ProviderProtocolError(`Chat completion finish reason is ${pythonRepr(finishReason)}, not 'stop'`);
  if (message.refusal) throw new ProviderProtocolError('Chat completion returned a refusal');
  if (typeof message.content !== 'string') throw new ProviderProtocolError('Chat completion assistant content must be text');
  return message.content;
}

function responsesText(response: Json): string {
  if (response.status !== 'completed') {
    throw new ProviderProtocolError(`Responses status is ${pythonRepr(response.status)}, not 'completed'`);
  }
  if (response.error !== undefined && response.error !== null) throw new ProviderProtocolError('Responses response contains an error');
  if (typeof response.output_text === 'string') return response.output_text;
  const texts: string[] = [];
  const outputs = response.output;
  if (!Array.isArray(outputs)) throw new ProviderProtocolError('Responses response has invalid output content');
  for (const output of outputs) {
    if (!isPlainObject(output)) throw new ProviderProtocolError('Responses response has invalid output content');
    if (output.type !== 'message') continue;
    const content = output.content ?? [];
    if (!Array.isArray(content)) throw new ProviderProtocolError('Responses response has invalid output content');
    for (const part of content) {
      if (!isPlainObject(part)) throw new ProviderProtocolError('Responses response has invalid output content');
      if (part.type === 'refusal') throw new ProviderProtocolError('Responses response contains a refusal');
      if (part.type === 'output_text' && typeof part.text === 'string') texts.push(part.text);
    }
  }
  if (texts.length) return texts.join('');
  throw new ProviderProtocolError('Responses response has no output text');
}

/**
 * Call an explicitly configured OpenAI-compatible JSON endpoint.
 *
 * ``api: 'chat_completions'`` targets the broadly implemented
 * ``/chat/completions`` contract; ``api: 'responses'`` targets ``/responses``.
 * Requests are never retried or silently routed to another provider.
 * ``complete`` blocks until the response arrives (``op.call``); ``acomplete``
 * is its asynchronous form (``op.acall``).
 */
export class OpenAICompatibleModel {
  static readonly qualifiedName: string = 'tensorcode.integrations.openai.OpenAICompatibleModel';
  readonly baseUrl: string;
  readonly model: string;
  readonly timeout: number;
  readonly api: OpenAIApi;
  readonly #apiKey: string | null;
  readonly #fetch: FetchLike | null;

  constructor(options: OpenAICompatibleModelOptions) {
    const { baseUrl, model, apiKey = null, timeout = 30, api = 'chat_completions' } = options;
    if (typeof model !== 'string' || !model) throw new ValueError('model must be a nonempty string');
    if (api !== 'chat_completions' && api !== 'responses') throw new ValueError("api must be 'chat_completions' or 'responses'");
    if (typeof baseUrl !== 'string') throw new TypeError('base_url must be a string');
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.model = model;
    this.#apiKey = apiKey;
    this.timeout = validateTimeout(timeout);
    this.api = api;
    this.#fetch = options.fetch ?? null;
    // Validate before a request is attempted.
    endpoint(this.baseUrl, api === 'chat_completions' ? 'chat/completions' : 'responses');
    if (apiKey !== null && (typeof apiKey !== 'string' || !apiKey)) throw new ValueError('api_key must be a nonempty string or None');
  }

  toString(): string {
    return `OpenAICompatibleModel(base_url=${pythonRepr(this.baseUrl)}, model=${pythonRepr(this.model)}, api=${pythonRepr(this.api)}, timeout=${this.timeout})`;
  }

  /** JSON description of this adapter; never includes credentials. */
  configuration(): Json {
    return { type: 'openai_compatible', base_url: this.baseUrl, model: this.model, api: this.api, timeout: this.timeout };
  }

  /** Send one request and return a ``ModelOutput`` (blocks; see {@link postJsonSync}). */
  complete(request: ModelRequest): ModelOutput {
    if (!(request instanceof ModelRequest)) throw new TypeError('complete expects ModelRequest');
    const [url, payload] = this.request(request);
    return this.output(request, postJsonSync(url, payload, { apiKey: this.#apiKey, timeout: this.timeout, fetch: this.#fetch }));
  }

  /** Asynchronous {@link complete}. */
  async acomplete(request: ModelRequest): Promise<ModelOutput> {
    if (!(request instanceof ModelRequest)) throw new TypeError('acomplete expects ModelRequest');
    const [url, payload] = this.request(request);
    return this.output(request, await postJson(url, payload, { apiKey: this.#apiKey, timeout: this.timeout, fetch: this.#fetch }));
  }

  private request(request: ModelRequest): [string, Json] {
    const chat = this.api === 'chat_completions';
    const payload = chat ? this.chatPayload(request) : this.responsesPayload(request);
    return [endpoint(this.baseUrl, chat ? 'chat/completions' : 'responses'), payload];
  }

  private output(request: ModelRequest, response: Json): ModelOutput {
    const text = this.api === 'chat_completions' ? chatText(response) : responsesText(response);
    const metadata: Json = {};
    for (const key of ['id', 'model', 'usage', 'status']) if (Object.hasOwn(response, key)) metadata[key] = response[key];
    const providerMetadata = Object.keys(metadata).length ? metadata : null;
    if (request.responseSchema === null) return new ModelOutput({ text, providerMetadata });
    let structured: unknown;
    try {
      structured = pythonJsonLoads(text);
    } catch (error) {
      throw new ProviderProtocolError('Structured provider response is not valid JSON', { cause: error });
    }
    if (!isPlainObject(structured)) throw new ProviderProtocolError('Structured provider response must be a JSON object');
    return new ModelOutput({ structured, providerMetadata });
  }

  /** The ``/chat/completions`` request body. */
  chatPayload(request: ModelRequest): Json {
    const messages: Json[] = [];
    if (request.instructions) messages.push({ role: 'system', content: request.instructions });
    messages.push(...request.messages.map(chatMessage));
    const payload: Json = { model: this.model, messages, stream: false };
    if (request.responseSchema !== null) {
      payload.response_format = {
        type: 'json_schema',
        json_schema: { name: schemaName(request.schemaName), strict: true, schema: { ...request.responseSchema } },
      };
    }
    return payload;
  }

  /** The ``/responses`` request body. */
  responsesPayload(request: ModelRequest): Json {
    const payload: Json = { model: this.model, input: request.messages.map(responsesMessage), store: false };
    if (request.instructions) payload.instructions = request.instructions;
    if (request.responseSchema !== null) {
      payload.text = {
        format: { type: 'json_schema', name: schemaName(request.schemaName), strict: true, schema: { ...request.responseSchema } },
      };
    }
    return payload;
  }
}
