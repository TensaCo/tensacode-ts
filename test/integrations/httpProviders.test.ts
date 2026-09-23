/** Port of ``tests/text/test_http_providers.py`` with an injected fake ``fetch``. */
import { inspect } from 'node:util';
import { describe, expect, it } from 'vitest';
import {
  JevModel, OpenAICompatibleModel, ProviderError, ProviderHTTPError, ProviderProtocolError, ProviderTimeout,
} from '../../src/integrations/index.js';
import * as text from '../../src/ops/text/index.js';
import { ValueError } from '../../src/errors.js';
import { fakeServer, network } from './fakeServer.js';

const bytes = (value: string) => new Uint8Array(Buffer.from(value));
const hello = () => new text.ModelRequest([new text.Message('user', 'hello')]);

describe('OpenAI-compatible provider', () => {
  it('chat wire preserves multimodal parts', async () => {
    const server = fakeServer(() => {
      const content = JSON.stringify({ label: 'cat', distribution: { cat: 0.75, dog: 0.25 }, confidence: null, abstained: false });
      return [200, { choices: [{ finish_reason: 'stop', message: { role: 'assistant', content } }] }, 0];
    });
    const model = new OpenAICompatibleModel({ baseUrl: `${server.url}/v1`, model: 'vision-test', apiKey: 'secret-key', fetch: server.fetch });
    const operation = text.Classify.fromModel(model, { labels: ['cat', 'dog'], instructions: 'Identify it' });
    const message = new text.Message('user', [
      new text.TextPart('What animal?', { sourceRef: 'prompt:1' }),
      new text.ImagePart({ data: bytes('image-bytes'), mediaType: 'image/png', sourceRef: 'upload:1' }),
      new text.ImagePart({ url: 'https://example.test/cat.jpg', detail: 'low' }),
    ]);

    const result = await operation.acall([message]);

    expect(result.label).toBe('cat');
    const request = server.requests[0]!;
    expect(request.path).toBe('/v1/chat/completions');
    expect(request.headers.authorization).toBe('Bearer secret-key');
    expect(request.headers['content-type']).toBe('application/json');
    expect(request.json.model).toBe('vision-test');
    expect(request.json.messages[0]).toEqual({ role: 'system', content: 'Identify it' });
    const content = request.json.messages[1].content;
    expect(content[0]).toEqual({ type: 'text', text: 'What animal?' });
    expect(content[1].image_url.url).toBe(`data:image/png;base64,${Buffer.from('image-bytes').toString('base64')}`);
    expect(content[2]).toEqual({ type: 'image_url', image_url: { url: 'https://example.test/cat.jpg', detail: 'low' } });
    expect(request.json.response_format.json_schema.strict).toBe(true);
    expect(request.json.response_format.json_schema.name).toBe('tensorcode_classify');
    const configuration = operation.configuration();
    expect((configuration.model as Record<string, unknown>).type).toBe('openai_compatible');
    expect(JSON.stringify(configuration)).not.toContain('api_key');
    expect(JSON.stringify(configuration)).not.toContain('secret-key');
    expect(() => operation.call([message])).toThrow(/acall/);
  });

  it('plain text and responses api', async () => {
    const server = fakeServer((request) => {
      expect(request.json.store).toBe(false);
      return [200, {
        status: 'completed',
        output: [{ type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'ans' }, { type: 'output_text', text: 'wer' }] }],
      }, 0];
    });
    const model = new OpenAICompatibleModel({ baseUrl: `${server.url}/v1`, model: 'text-test', api: 'responses', fetch: server.fetch });
    const output = await model.acomplete(hello());
    expect(output.text).toBe('answer');
    expect(server.requests[0]!.path).toBe('/v1/responses');
    expect(server.requests[0]!.json.input).toEqual([{ role: 'user', content: [{ type: 'input_text', text: 'hello' }] }]);
    expect(server.requests[0]!.headers.authorization).toBeUndefined();
    const transform = text.Transform.fromModel(model, { instructions: 'Be brief' });
    const reply = await transform.acall([new text.Message('user', 'hello')]);
    expect(reply.at(-1)).toEqual(new text.Message('assistant', 'answer'));
    expect(server.requests[1]!.json.instructions).toBe('Be brief');
  });

  it('http provider errors, timeout and secret safe configuration', async () => {
    const errorServer = fakeServer(() => [503, { error: 'key=secret-key' }, 0]);
    const model = new OpenAICompatibleModel({ baseUrl: `${errorServer.url}/v1`, model: 'test', apiKey: 'secret-key', timeout: 0.2, fetch: errorServer.fetch });
    const caught = await model.acomplete(hello()).catch((error: unknown) => error);
    expect(caught).toBeInstanceOf(ProviderHTTPError);
    expect(caught).toBeInstanceOf(ProviderError);
    expect((caught as ProviderHTTPError).status).toBe(503);
    expect(String(caught)).not.toContain('secret-key');
    expect(String(caught)).toContain('[REDACTED]');
    expect(String(model)).not.toContain('secret-key');
    expect(inspect(model, { showHidden: true, depth: 5 })).not.toContain('secret-key');
    expect(JSON.stringify(model)).not.toContain('secret-key');
    expect('api_key' in model.configuration()).toBe(false);
    JSON.stringify(model.configuration());

    const slowServer = fakeServer(() => [200, { choices: [] }, 0.2]);
    const slow = new OpenAICompatibleModel({ baseUrl: `${slowServer.url}/v1`, model: 'test', timeout: 0.03, fetch: slowServer.fetch });
    await expect(slow.acomplete(hello())).rejects.toThrow(ProviderTimeout);
    await expect(slow.acomplete(hello())).rejects.toThrow('Provider request timed out after 0.03 seconds');

    const offline = new OpenAICompatibleModel({ baseUrl: 'http://127.0.0.1:1/v1', model: 'test', fetch: errorServer.fetch });
    await expect(offline.acomplete(hello())).rejects.toThrow(/Provider request failed/);
  });

  const MALFORMED = [
    { choices: [] },
    { choices: [{ finish_reason: 'stop', message: { content: 7 } }] },
    { choices: [{ finish_reason: 'stop', message: { content: 'not json' } }] },
    { choices: [{ finish_reason: 'stop', message: { content: '[1, 2]' } }] },
  ];
  for (const payload of MALFORMED) {
    it(`rejects malformed structured responses ${JSON.stringify(payload)}`, async () => {
      const server = fakeServer(() => [200, payload, 0]);
      const classify = text.Classify.fromModel(new OpenAICompatibleModel({ baseUrl: `${server.url}/v1`, model: 'test', fetch: server.fetch }), { labels: ['yes', 'no'] });
      await expect(classify.acall([new text.Message('user', 'question')])).rejects.toThrow(ProviderProtocolError);
    });
  }

  it('rejects non-JSON and non-object bodies', async () => {
    const raw = (body: string) => async () => new Response(body, { status: 200 });
    await expect(new OpenAICompatibleModel({ baseUrl: 'https://x.test/v1', model: 'm', fetch: raw('not json') }).acomplete(hello()))
      .rejects.toThrow('Provider response is not valid JSON');
    await expect(new OpenAICompatibleModel({ baseUrl: 'https://x.test/v1', model: 'm', fetch: raw('[1]') }).acomplete(hello()))
      .rejects.toThrow('Provider response must be a JSON object');
  });

  it('rejects redirect without forwarding authorization', async () => {
    const target = fakeServer(() => [200, { choices: [{ finish_reason: 'stop', message: { content: 'unexpected' } }] }, 0]);
    const redirect = fakeServer(() => [302, {}, 0, { Location: `${target.url}/stolen` }]);
    const model = new OpenAICompatibleModel({ baseUrl: `${redirect.url}/v1`, model: 'test', apiKey: 'do-not-forward', fetch: network(target, redirect) });
    const caught = await model.acomplete(hello()).catch((error: unknown) => error);
    expect(caught).toBeInstanceOf(ProviderHTTPError);
    expect((caught as ProviderHTTPError).status).toBe(302);
    expect(target.requests).toEqual([]);
  });

  it('rejects truncation, refusal and incomplete responses', async () => {
    const chat = fakeServer(() => [200, { choices: [{ finish_reason: 'length', message: { content: 'partial' } }] }, 0]);
    await expect(new OpenAICompatibleModel({ baseUrl: `${chat.url}/v1`, model: 'test', fetch: chat.fetch }).acomplete(hello()))
      .rejects.toThrow("Chat completion finish reason is 'length', not 'stop'");
    const refusal = fakeServer(() => [200, { choices: [{ finish_reason: 'stop', message: { content: 'x', refusal: 'no' } }] }, 0]);
    await expect(new OpenAICompatibleModel({ baseUrl: `${refusal.url}/v1`, model: 'test', fetch: refusal.fetch }).acomplete(hello()))
      .rejects.toThrow(/refusal/);
    const responses = fakeServer(() => [200, { status: 'incomplete', output: [{ type: 'message', content: [{ type: 'refusal', refusal: 'no' }] }] }, 0]);
    await expect(new OpenAICompatibleModel({ baseUrl: `${responses.url}/v1`, model: 'test', api: 'responses', fetch: responses.fetch }).acomplete(hello()))
      .rejects.toThrow(/status/);
    const refused = fakeServer(() => [200, { status: 'completed', output: [{ type: 'message', content: [{ type: 'refusal', refusal: 'no' }] }] }, 0]);
    await expect(new OpenAICompatibleModel({ baseUrl: `${refused.url}/v1`, model: 'test', api: 'responses', fetch: refused.fetch }).acomplete(hello()))
      .rejects.toThrow(/refusal/);
  });

  it('validates configuration before any request', () => {
    expect(() => new OpenAICompatibleModel({ baseUrl: 'ftp://x.test', model: 'm' })).toThrow(ValueError);
    expect(() => new OpenAICompatibleModel({ baseUrl: 'https://user:pw@x.test', model: 'm' })).toThrow(/credentials/);
    expect(() => new OpenAICompatibleModel({ baseUrl: 'https://x.test/v1?key=1', model: 'm' })).toThrow(/query/);
    expect(() => new OpenAICompatibleModel({ baseUrl: 'https://x.test/v1', model: '' })).toThrow(/model/);
    expect(() => new OpenAICompatibleModel({ baseUrl: 'https://x.test/v1', model: 'm', api: 'other' as never })).toThrow(/api/);
    expect(() => new OpenAICompatibleModel({ baseUrl: 'https://x.test/v1', model: 'm', timeout: 0 })).toThrow(/timeout/);
    const model = new OpenAICompatibleModel({ baseUrl: 'https://x.test/v1/', model: 'm' });
    expect(model.configuration()).toEqual({ type: 'openai_compatible', base_url: 'https://x.test/v1', model: 'm', api: 'chat_completions', timeout: 30 });
  });
});

describe('Jev provider', () => {
  it('wire maps documented choice and score answers', async () => {
    const answers = [
      {
        model: 'jev-latest',
        answers: { result: { type: 'choice', choice: 'billing', confidence: 0.8, probabilities: { billing: 0.8, technical: 0.2 } } },
        usage: { input_tokens: 10, output_tokens: 2 },
      },
      {
        model: 'jev-latest',
        answers: {
          result: {
            type: 'score', score: 1.7, confidence: 0.9, legend: { 0: 'low', 1: 'medium', 2: 'high' }, probabilities: { 0: 0.1, 1: 0.1, 2: 0.8 },
          },
        },
        usage: { input_tokens: 10, output_tokens: 2 },
      },
    ];
    const server = fakeServer(() => [200, answers.shift(), 0]);
    const model = new JevModel({ baseUrl: server.url, apiKey: 'jev-secret', fetch: server.fetch });
    const classification = await text.Classify.fromModel(model, { labels: ['billing', 'technical'], instructions: 'Route ticket' })
      .acall([new text.Message('user', 'charged twice')]);
    const score = await text.Score.fromModel(model, { rubric: ['low', 'medium', 'high'], instructions: 'Urgency' })
      .acall([new text.Message('user', 'help now')]);
    expect(classification.distribution).toEqual({ billing: 0.8, technical: 0.2 });
    expect(classification.confidence).toBe(0.8);
    expect(score.value).toBe(1.7);
    const first = server.requests[0]!;
    expect(first.path).toBe('/v1/systemone');
    expect(first.headers.authorization).toBe('Bearer jev-secret');
    expect(first.json.questions).toEqual({ result: { type: 'choice', instructions: 'Route ticket', criteria: { billing: null, technical: null } } });
    expect(server.requests[1]!.json.questions.result.criteria).toEqual(['low', 'medium', 'high']);
    expect(String(model)).not.toContain('jev-secret');
    expect(model.configuration()).toEqual({ type: 'jev', base_url: server.url, model: 'jev-latest', timeout: 30 });
  });

  it('rejects unsupported multimodal and retrieval requests', async () => {
    const server = fakeServer(() => [200, {}, 0]);
    const model = new JevModel({ baseUrl: server.url, apiKey: 'key', fetch: server.fetch });
    await expect(text.Classify.fromModel(model, { labels: ['a', 'b'] })
      .acall([new text.Message('user', [new text.ImagePart({ data: bytes('x'), mediaType: 'image/png' })])])).rejects.toThrow(/image/);
    const retrieve = text.Retrieve.fromModel(model, { items: { a: 'first', b: 'second' }, descriptions: { a: 'first', b: 'second' } });
    await expect(retrieve.acall([new text.Message('user', 'which')])).rejects.toThrow(/retrieve/);
    await expect(text.Transform.fromModel(model).acall([new text.Message('user', 'hi')])).rejects.toThrow(ProviderProtocolError);
    expect(server.requests).toEqual([]);
    expect(() => new JevModel({ apiKey: '' })).toThrow(/api_key/);
  });
});
