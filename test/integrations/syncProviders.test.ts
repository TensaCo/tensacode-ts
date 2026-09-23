/**
 * Synchronous provider calls (Python's blocking ``complete``): ``op.call`` over
 * HTTP and local providers blocks on a worker thread. Mirrors the Python
 * ``tests/text/test_http_providers.py`` and ``test_local_model.py`` flows,
 * which call the providers synchronously against a threaded server.
 */
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  JevModel, LocalModel, OpenAICompatibleModel, ProviderError, ProviderHTTPError, ProviderProtocolError, ProviderTimeout, type LocalProcessor,
} from '../../src/integrations/index.js';
import { SynchronousCallUnavailable } from '../../src/integrations/blocking.js';
import { NotImplementedError } from '../../src/errors.js';
import * as text from '../../src/ops/text/index.js';
import { ValueError } from '../../src/errors.js';
import { threadedServer, type ThreadedServer } from './threadedServer.js';

let server: ThreadedServer;
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-sync-'));

beforeAll(async () => {
  server = await threadedServer((request) => {
    const path = request.path;
    if (path === '/slow/chat/completions') return [200, { choices: [] }, 1];
    if (path === '/redirect/chat/completions') return [302, {}, 0, { location: 'http://127.0.0.1:9/stolen' }];
    if (path === '/error/chat/completions') return [503, { error: `key=${request.headers.authorization}` }, 0];
    if (path === '/notjson/chat/completions') return [200, 'not an object', 0];
    if (path === '/v1/responses') {
      return [200, { status: 'completed', output: [{ type: 'message', content: [{ type: 'output_text', text: 'ans' }, { type: 'output_text', text: 'wer' }] }] }, 0];
    }
    if (path === '/v1/systemone') {
      const questions = request.json.questions;
      const answers: Record<string, unknown> = {};
      for (const name of Object.keys(questions)) {
        answers[name] = questions[name].type === 'noul'
          ? { type: 'noul', noul: 0.75 }
          : { type: 'choice', choice: 'billing', confidence: 0.8, probabilities: { billing: 0.8, technical: 0.2 } };
      }
      return [200, { model: 'jev-latest', answers, usage: { input_tokens: 3 } }, 0];
    }
    if (request.json.response_format) {
      const content = JSON.stringify({ label: 'cat', distribution: { cat: 0.75, dog: 0.25 }, confidence: null, abstained: false });
      return [200, { id: 'c1', choices: [{ finish_reason: 'stop', message: { role: 'assistant', content } }] }, 0];
    }
    return [200, { id: 'c2', choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'héllo' } }] }, 0];
  });
});

afterAll(async () => {
  await server.close();
  rmSync(scratch, { recursive: true, force: true });
});

const hello = () => new text.ModelRequest([new text.Message('user', 'hello ✓')]);

describe('synchronous HTTP providers', () => {
  it('raises a clear error where a thread cannot block', () => {
    // Runs first: the worker bridge has not started in this file yet.
    const processLike = process as unknown as { getBuiltinModule?: unknown };
    const original = processLike.getBuiltinModule;
    processLike.getBuiltinModule = undefined;
    try {
      const model = new OpenAICompatibleModel({ baseUrl: 'http://127.0.0.1:9/v1', model: 'm' });
      expect(() => model.complete(hello())).toThrow(SynchronousCallUnavailable);
      expect(() => model.complete(hello())).toThrow(NotImplementedError);
      expect(() => model.complete(hello())).toThrow(/acomplete/);
    } finally {
      processLike.getBuiltinModule = original;
    }
  });

  it('complete and op.call block on the request (chat completions)', async () => {
    const model = new OpenAICompatibleModel({ baseUrl: `${server.url}/v1`, model: 'vision-test', apiKey: 'secret-key' });
    const output = model.complete(hello());
    expect(output.text).toBe('héllo');
    expect(output.providerMetadata).toEqual({ id: 'c2' });
    expect(await model.acomplete(hello())).toEqual(output);
    const classify = text.Classify.fromModel(model, { labels: ['cat', 'dog'], instructions: 'Identify it' });
    const result = classify.call([new text.Message('user', [
      new text.TextPart('What animal?'), new text.ImagePart({ data: new Uint8Array([1, 2]), mediaType: 'image/png' }),
    ])]);
    expect(result.label).toBe('cat');
    expect(result.distribution).toEqual({ cat: 0.75, dog: 0.25 });
    const requests = await server.requests();
    const last = requests.at(-1)!;
    expect(last.path).toBe('/v1/chat/completions');
    expect(last.headers.authorization).toBe('Bearer secret-key');
    expect(last.json.messages[0]).toEqual({ role: 'system', content: 'Identify it' });
    expect(last.json.messages[1].content[1].image_url.url).toBe('data:image/png;base64,AQI=');
    expect(requests[0]!.json.messages[0].content).toBe('hello ✓');
  });

  it('responses api and Transform.call', () => {
    const model = new OpenAICompatibleModel({ baseUrl: `${server.url}/v1`, model: 'text-test', api: 'responses' });
    expect(model.complete(hello()).text).toBe('answer');
    const reply = text.Transform.fromModel(model, { instructions: 'Be brief' }).call([new text.Message('user', 'hello')]);
    expect(reply.at(-1)).toEqual(new text.Message('assistant', 'answer'));
  });

  it('reports HTTP errors, redirects, protocol errors, timeouts and connection failures like the async path', async () => {
    const failing = new OpenAICompatibleModel({ baseUrl: `${server.url}/error`, model: 'm', apiKey: 'secret-key' });
    const caught = (() => {
      try {
        failing.complete(hello());
        return null;
      } catch (error) {
        return error;
      }
    })();
    expect(caught).toBeInstanceOf(ProviderHTTPError);
    expect((caught as ProviderHTTPError).status).toBe(503);
    expect(String(caught)).toContain('[REDACTED]');
    expect(String(caught)).not.toContain('secret-key');
    const asyncCaught = await failing.acomplete(hello()).catch((error: unknown) => error);
    expect(String(asyncCaught)).toBe(String(caught));

    const redirect = new OpenAICompatibleModel({ baseUrl: `${server.url}/redirect`, model: 'm', apiKey: 'k' });
    expect(() => redirect.complete(hello())).toThrow(expect.objectContaining({ status: 302 }));
    expect(() => new OpenAICompatibleModel({ baseUrl: `${server.url}/notjson`, model: 'm' }).complete(hello()))
      .toThrow(new ProviderProtocolError('Provider response must be a JSON object'));
    expect(() => new OpenAICompatibleModel({ baseUrl: `${server.url}/slow`, model: 'm', timeout: 0.1 }).complete(hello()))
      .toThrow(new ProviderTimeout('Provider request timed out after 0.1 seconds'));
    const refused = new OpenAICompatibleModel({ baseUrl: 'http://127.0.0.1:9/v1', model: 'm' });
    const syncError = (() => {
      try {
        refused.complete(hello());
        return null;
      } catch (error) {
        return error as Error;
      }
    })();
    expect(syncError).toBeInstanceOf(ProviderError);
    expect(syncError!.message).toBe(((await refused.acomplete(hello()).catch((error: Error) => error)) as Error).message);
  });

  it('an injected fetch cannot block', () => {
    const model = new OpenAICompatibleModel({ baseUrl: `${server.url}/v1`, model: 'm', fetch: async () => new Response('{}') });
    expect(() => text.Classify.fromModel(model, { labels: ['a', 'b'] }).call([new text.Message('user', 'x')])).toThrow(/acall/);
  });

  it('Jev complete, completeQuestions and ask', async () => {
    const model = new JevModel({ baseUrl: server.url, apiKey: 'jev-secret' });
    const route = text.Classify.fromModel(model, { labels: ['billing', 'technical'], instructions: 'Route ticket' });
    const urgent = text.Classify.fromModel(model, { labels: ['true', 'false'], instructions: 'Urgent?' });
    const routed = route.call([new text.Message('user', 'charged twice')]);
    expect(routed.label).toBe('billing');
    expect(routed.confidence).toBe(0.8);
    const before = (await server.requests()).length;
    const answers = text.ask([new text.Message('user', 'charged twice')], { route, urgent });
    expect(answers.route.label).toBe('billing');
    expect(answers.urgent.label).toBe('true');
    expect(answers.urgent.distribution).toEqual({ true: 0.75, false: 0.25 });
    const requests = (await server.requests()).slice(before);
    expect(requests.length).toBe(1); // all questions in one request, as in Python
    expect(Object.keys(requests[0]!.json.questions)).toEqual(['route', 'urgent']);
    expect(requests[0]!.headers.authorization).toBe('Bearer jev-secret');
  });
});

describe('synchronous LocalModel', () => {
  const loader = join(scratch, 'fakeLocalModel.mjs');
  writeFileSync(loader, `
export async function load(args) {
  const tensor = (rows) => ({ dims: [rows.length, rows[0].length], tolist: () => rows.map((row) => row.map(BigInt)) });
  const processor = Object.assign(async (text, images) => ({ input_ids: tensor([[1, 2]]), images: images ? images.length : 0 }), {
    apply_chat_template: (messages) => JSON.stringify(messages),
    batch_decode: (rows) => [' ' + args.answer + ' '],
  });
  const model = {
    generation_config: { eos_token_id: 4 },
    eval() { this.evaluated = true; return this; },
    async generate(inputs) {
      if (inputs.do_sample !== false) throw new Error('sampling');
      return tensor([[1, 2, 3, args.finish ? 4 : 5]]);
    },
  };
  return { model, processor, loadImage: (image) => ({ bytes: Array.from(image.data), mediaType: image.mediaType }) };
}
`);
  const fakeAsync = () => {
    const processor = Object.assign(async () => ({ input_ids: [[1, 2]] }), {
      apply_chat_template: () => 'prompt', batch_decode: () => [' A cat. '],
    }) as unknown as LocalProcessor;
    return { model: { generation_config: { eos_token_id: 4 }, generate: async () => [[1, 2, 3, 4]] }, processor };
  };

  it('runs complete and completeBatch in a worker loaded from options.worker', () => {
    const { model, processor } = fakeAsync();
    const local = new LocalModel(model, processor, {
      modelId: 'supplied-test-model', maxNewTokens: 2, worker: { module: loader, exportName: 'load', args: { answer: 'A cat.', finish: true } },
    });
    const output = local.complete(new text.ModelRequest([new text.Message('user', [
      new text.TextPart('What is it?'), new text.ImagePart({ data: new Uint8Array([7, 8]), mediaType: 'image/png' }),
    ])]));
    expect(output.text).toBe('A cat.');
    expect(output.providerMetadata).toEqual({
      model_id: 'supplied-test-model', revision: null, backend: 'transformers.js', source: 'supplied_pretrained_model',
      generated_tokens: 2, finish_reason: 'stop',
    });
    const outputs = local.completeBatch([new text.ModelRequest([new text.Message('user', 'a')]), new text.ModelRequest([new text.Message('user', 'b')])]);
    expect(outputs.map((item) => item.text)).toEqual(['A cat.', 'A cat.']);
    expect(text.Transform.fromModel(local).call([new text.Message('user', 'hi')]).at(-1)!.content).toBe('A cat.');
  });

  it('structured answers and the token limit behave as in the async path', async () => {
    const structured = new LocalModel(fakeAsync().model, fakeAsync().processor, {
      modelId: 'm', maxNewTokens: 2, worker: { module: pathToFileURL(loader), exportName: 'load', args: { answer: '{"label": "b"}', finish: true } },
    });
    expect(structured.complete(new text.ModelRequest([new text.Message('user', 'x')], { responseSchema: { type: 'object' } })).structured)
      .toEqual({ label: 'b' });
    const cut = new LocalModel(fakeAsync().model, fakeAsync().processor, {
      modelId: 'm', maxNewTokens: 2, worker: { module: loader, exportName: 'load', args: { answer: 'unfinished', finish: false } },
    });
    expect(() => cut.complete(new text.ModelRequest([new text.Message('user', 'x')])))
      .toThrow(new ValueError('Local model reached the token limit before finishing its answer'));
    expect(() => cut.complete(new text.ModelRequest([new text.Message('user', [new text.ImagePart({ url: 'https://example.test/x.png' })])])))
      .toThrow(new ValueError('LocalModel requires image bytes; fetch URLs explicitly'));
    const missing = new LocalModel(fakeAsync().model, fakeAsync().processor, { modelId: 'm', worker: { module: loader, exportName: 'absent' } });
    expect(() => missing.complete(new text.ModelRequest([new text.Message('user', 'x')]))).toThrow(TypeError);
    // The supplied object itself cannot cross threads.
    const supplied = new LocalModel(fakeAsync().model, fakeAsync().processor, { modelId: 'm' });
    expect(() => supplied.complete(new text.ModelRequest([new text.Message('user', 'x')]))).toThrow(/fromPretrained/);
    expect((await supplied.acomplete(new text.ModelRequest([new text.Message('user', 'x')]))).text).toBe('A cat.');
  });
});
