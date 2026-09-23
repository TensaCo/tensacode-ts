/** The global ``fetch`` transport against a loopback HTTP server (no external network). */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { OpenAICompatibleModel, ProviderHTTPError, ProviderTimeout } from '../../src/integrations/index.js';
import { Message, ModelRequest } from '../../src/ops/text/index.js';

const seen: { path: string; authorization?: string; body: any }[] = [];
let server: Server;
let base = '';

function handler(request: IncomingMessage, response: ServerResponse): void {
  const chunks: Buffer[] = [];
  request.on('data', (chunk: Buffer) => chunks.push(chunk));
  request.on('end', () => {
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : null;
    seen.push({ path: request.url!, authorization: request.headers.authorization, body });
    const reply = (status: number, payload: unknown, headers: Record<string, string> = {}) => {
      response.writeHead(status, { 'content-type': 'application/json', ...headers });
      response.end(JSON.stringify(payload));
    };
    if (request.url === '/redirect/chat/completions') reply(302, {}, { Location: `${base}/stolen/chat/completions` });
    else if (request.url === '/slow/chat/completions') setTimeout(() => reply(200, { choices: [] }), 300);
    else reply(200, { id: 'x', choices: [{ finish_reason: 'stop', message: { content: 'héllo' } }] });
  });
}

beforeAll(async () => {
  server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
afterAll(async () => {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
});

const request = () => new ModelRequest([new Message('user', 'hello ✓')]);

describe('global fetch transport', () => {
  it('posts compact UTF-8 JSON and decodes the reply', async () => {
    const output = await new OpenAICompatibleModel({ baseUrl: `${base}/ok`, model: 'm', apiKey: 'k' }).acomplete(request());
    expect(output.text).toBe('héllo');
    expect(output.providerMetadata).toEqual({ id: 'x' });
    const last = seen.at(-1)!;
    expect(last.authorization).toBe('Bearer k');
    expect(last.body.messages[0].content).toBe('hello ✓');
  });

  it('never follows redirects (the status is reported)', async () => {
    const before = seen.length;
    const caught = await new OpenAICompatibleModel({ baseUrl: `${base}/redirect`, model: 'm', apiKey: 'secret' })
      .acomplete(request()).catch((error: unknown) => error);
    expect(caught).toBeInstanceOf(ProviderHTTPError);
    expect((caught as ProviderHTTPError).status).toBe(302);
    expect(seen.slice(before).map((entry) => entry.path)).toEqual(['/redirect/chat/completions']);
  });

  it('times out', async () => {
    await expect(new OpenAICompatibleModel({ baseUrl: `${base}/slow`, model: 'm', timeout: 0.05 }).acomplete(request()))
      .rejects.toThrow(ProviderTimeout);
  });
});
