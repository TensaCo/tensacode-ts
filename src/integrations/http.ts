/**
 * Small dependency-free JSON HTTP transport for explicit provider adapters
 * (Python ``tensorcode/integrations/_http.py``): one buffered request, no
 * retries, no redirects, API keys never echoed.
 */
import { ValueError } from '../errors.js';
import { pythonJsonDumps, pythonJsonLoads } from '../_internal/json.js';
import { BlockingTimeout, blockingCall } from './blocking.js';

/** Base class for provider transport and protocol failures. */
export class ProviderError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** The provider did not respond within the configured timeout. */
export class ProviderTimeout extends ProviderError {}

/** A request or response does not fit the provider's documented contract. */
export class ProviderProtocolError extends ProviderError {}

/** The provider returned a non-success HTTP status, kept as ``status``. */
export class ProviderHTTPError extends ProviderError {
  readonly status: number;

  constructor(status: number, message: string, options?: { cause?: unknown }) {
    super(`Provider returned HTTP ${status}: ${message}`, options);
    this.status = status;
  }
}

/** A ``fetch``-compatible transport (injectable for tests). */
export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

/** ``base_url`` joined with ``path``; the base must be an absolute HTTP(S) URL without credentials, query or fragment. */
export function endpoint(baseUrl: unknown, path: string): string {
  if (typeof baseUrl !== 'string') throw new TypeError('base_url must be a string');
  const match = /^([A-Za-z][A-Za-z0-9+.-]*):\/\/([^/?#]*)([^?#]*)(\?[^#]*)?(#.*)?$/.exec(baseUrl);
  const scheme = match?.[1]?.toLowerCase();
  if (!match || (scheme !== 'http' && scheme !== 'https') || !match[2]) throw new ValueError('base_url must be an absolute HTTP(S) URL');
  try {
    new URL(baseUrl);
  } catch (error) {
    throw new ValueError('base_url must be an absolute HTTP(S) URL', { cause: error });
  }
  if (match[2].includes('@') || match[4] !== undefined || match[5] !== undefined) {
    throw new ValueError('base_url cannot contain credentials, query parameters or a fragment');
  }
  // Python ``urlsplit`` lowercases the scheme and keeps the network location as written.
  const joined = `${(match[3] ?? '').replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
  return `${scheme}://${match[2]}${joined}`;
}

/** Python ``f'{timeout:g}'`` for the timeout message. */
function formatSeconds(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
}

export function validateTimeout(timeout: unknown): number {
  if (typeof timeout !== 'number' || !(timeout > 0) || !Number.isFinite(timeout)) throw new ValueError('timeout must be a positive number');
  return timeout;
}

export interface PostJsonOptions {
  apiKey?: string | null;
  /** Seconds (default 30). */
  timeout?: number;
  fetch?: FetchLike | null;
}

function redact(text: string, apiKey: string | null): string {
  return apiKey ? text.split(apiKey).join('[REDACTED]') : text;
}

interface PreparedRequest {
  readonly body: string;
  readonly headers: Record<string, string>;
  readonly apiKey: string | null;
  readonly timeout: number;
}

function prepare(payload: unknown, options: PostJsonOptions): PreparedRequest {
  const timeout = validateTimeout(options.timeout ?? 30);
  const body = pythonJsonDumps(payload, { separators: [',', ':'], ensureAscii: false, floatKeys: new Set() });
  const headers: Record<string, string> = { 'Content-Type': 'application/json', Accept: 'application/json' };
  const apiKey = options.apiKey ?? null;
  if (apiKey !== null) {
    if (typeof apiKey !== 'string' || !apiKey) throw new ValueError('api_key must be a nonempty string or None');
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return { body, headers, apiKey, timeout };
}

function timeoutError(timeout: number, cause?: unknown): ProviderTimeout {
  return new ProviderTimeout(`Provider request timed out after ${formatSeconds(timeout)} seconds`, cause === undefined ? undefined : { cause });
}

/** Status check and JSON decoding shared by the asynchronous and blocking transports. */
function decode(status: number, bytes: Uint8Array, apiKey: string | null): Record<string, unknown> {
  if (status < 200 || status >= 300) {
    // Redirects are never followed, so credentials are never forwarded.
    const message = redact(new globalThis.TextDecoder('utf-8').decode(bytes).slice(0, 1000), apiKey);
    throw new ProviderHTTPError(status, message);
  }
  let decoded: unknown;
  try {
    decoded = pythonJsonLoads(new globalThis.TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch (error) {
    throw new ProviderProtocolError('Provider response is not valid JSON', { cause: error });
  }
  if (decoded === null || typeof decoded !== 'object' || Array.isArray(decoded)) {
    throw new ProviderProtocolError('Provider response must be a JSON object');
  }
  return decoded as Record<string, unknown>;
}

/** POST ``payload`` as compact UTF-8 JSON and return the decoded JSON object. */
export async function postJson(url: string, payload: unknown, options: PostJsonOptions = {}): Promise<Record<string, unknown>> {
  const { body, headers, apiKey, timeout } = prepare(payload, options);
  const transport = options.fetch ?? (globalThis.fetch as FetchLike | undefined);
  if (typeof transport !== 'function') throw new ProviderError('Provider request failed: no fetch implementation is available');
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeout * 1000);
  let status: number;
  let bytes: Uint8Array;
  try {
    let response: Response;
    try {
      response = await transport(url, { method: 'POST', headers, body, redirect: 'manual', signal: controller.signal });
    } catch (error) {
      if (timedOut) throw timeoutError(timeout, error);
      const reason = error instanceof Error ? ((error.cause as Error | undefined)?.message ?? error.message) : String(error);
      throw new ProviderError(`Provider request failed: ${redact(reason, apiKey)}`);
    }
    try {
      bytes = new Uint8Array(await response.arrayBuffer());
    } catch (error) {
      if (timedOut) throw timeoutError(timeout, error);
      throw new ProviderError('Provider request failed: the response body could not be read');
    }
    status = response.status;
    if (response.type === 'opaqueredirect' && !status) status = 302;
  } finally {
    clearTimeout(timer);
  }
  return decode(status, bytes, apiKey);
}

type BlockingReply =
  | { outcome: 'response'; status: number; bytes: Uint8Array }
  | { outcome: 'timeout' }
  | { outcome: 'failed'; reason: string }
  | { outcome: 'unreadable' };

/**
 * Blocking {@link postJson} (Python ``post_json``): the request runs on the
 * global ``fetch`` of a worker thread while this thread waits. An injected
 * ``fetch`` function cannot cross threads, so it is only usable
 * asynchronously.
 */
export function postJsonSync(url: string, payload: unknown, options: PostJsonOptions = {}): Record<string, unknown> {
  const { body, headers, apiKey, timeout } = prepare(payload, options);
  if (options.fetch) {
    throw new TypeError('A provider with an injected fetch function cannot block: use await operation.acall(...) or model.acomplete(...), or omit fetch for synchronous calls');
  }
  let reply: BlockingReply;
  try {
    // The worker enforces ``timeout``; the extra wait only guards a stuck worker.
    reply = blockingCall<BlockingReply>({ kind: 'http', url, headers, body, timeoutMs: timeout * 1000 }, { waitSeconds: timeout + 30 });
  } catch (error) {
    if (error instanceof BlockingTimeout) throw timeoutError(timeout, error);
    throw error;
  }
  if (reply.outcome === 'timeout') throw timeoutError(timeout);
  if (reply.outcome === 'failed') throw new ProviderError(`Provider request failed: ${redact(reply.reason, apiKey)}`);
  if (reply.outcome === 'unreadable') throw new ProviderError('Provider request failed: the response body could not be read');
  return decode(reply.status, reply.bytes, apiKey);
}

/** Python ``repr`` of a JSON scalar in protocol messages. */
export function pythonRepr(value: unknown): string {
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'string') {
    const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
    const escaped = value.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/\t/g, '\\t');
    return quote + (quote === "'" ? escaped.replace(/'/g, "\\'") : escaped) + quote;
  }
  if (typeof value === 'number') return String(value);
  return pythonJsonDumps(value);
}
