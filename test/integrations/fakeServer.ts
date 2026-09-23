/** An injected ``fetch`` standing in for the Python tests' local HTTP server. */
export interface RecordedRequest {
  url: string;
  path: string;
  headers: Record<string, string>;
  json: any;
}

export type Responder = (request: RecordedRequest) => [status: number, payload: unknown, delaySeconds: number, headers?: Record<string, string>];

export interface FakeServer {
  url: string;
  requests: RecordedRequest[];
  fetch: (input: string, init: RequestInit) => Promise<Response>;
}

let port = 40000;

export function fakeServer(responder: Responder): FakeServer {
  port += 1;
  const url = `http://127.0.0.1:${port}`;
  const requests: RecordedRequest[] = [];
  const fetch = async (input: string, init: RequestInit): Promise<Response> => {
    const target = new URL(input);
    if (target.origin !== url) throw new TypeError('fetch failed', { cause: new Error(`connect ECONNREFUSED ${target.host}`) });
    const headers: Record<string, string> = {};
    new Headers(init.headers).forEach((value, key) => { headers[key.toLowerCase()] = value; });
    const request = { url: input, path: target.pathname, headers, json: JSON.parse(String(init.body)) };
    requests.push(request);
    const [status, payload, delay, extra = {}] = responder(request);
    if (delay) {
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, delay * 1000);
        init.signal?.addEventListener('abort', () => {
          clearTimeout(timer);
          reject(new DOMException('This operation was aborted', 'AbortError'));
        });
      });
    }
    if (init.redirect !== 'manual' && status >= 300 && status < 400) throw new Error('fake server expects redirect: manual');
    return new Response(JSON.stringify(payload), { status, headers: { 'content-type': 'application/json', ...extra } });
  };
  return { url, requests, fetch };
}

/** Route one injected ``fetch`` to several fake servers by origin. */
export function network(...servers: FakeServer[]): (input: string, init: RequestInit) => Promise<Response> {
  return (input, init) => {
    const server = servers.find((candidate) => input.startsWith(candidate.url));
    if (!server) return Promise.reject(new TypeError('fetch failed'));
    return server.fetch(input, init);
  };
}
