/**
 * Text-only parts of ``tests/integration/test_public_operations.py``,
 * ``test_regressions.py`` and ``test_text_namespace.py``.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as text from '../../src/ops/text/index.js';
import { trace } from '../../src/_internal/tracing.js';

describe('public text operations', () => {
  it('messages preserve context roles and caller state', () => {
    const observed: (readonly text.Message[])[] = [];
    const model = (messages: readonly text.Message[]) => {
      observed.push(messages);
      return 'answer';
    };
    const encode = new text.TextEncoder();
    const respond = text.Transform.fromModel(model);
    const original = encode.call('hello');
    const result = respond.call(original, { context: { policy: encode.call('be brief') } });
    expect(original).toEqual([new text.Message('user', 'hello')]);
    expect(result.at(-1)).toEqual(new text.Message('assistant', 'answer'));
    expect(observed[0]!.some((message) => (message.content as string).includes('be brief'))).toBe(true);
    expect(observed[0]!.at(-1)!.content).toBe('hello');
    expect(new text.TextDecoder().call(result)).toBe('answer');
  });

  it('invalid model response is not silently converted to text', () => {
    expect(() => text.Transform.fromModel(() => null).call(new text.TextEncoder().call('hello'))).toThrow(TypeError);
    expect(() => text.Transform.fromModel(async () => 'x').call(new text.TextEncoder().call('hello'))).toThrow(/acall/);
  });

  it('async function providers run through acall', async () => {
    const respond = text.Transform.fromModel(async (messages: readonly text.Message[]) => `${messages.length}`);
    expect((await respond.acall(new text.TextEncoder().call('hello'))).at(-1)!.content).toBe('1');
  });

  it('two message compositions in one trace preserve history', () => {
    const encode = new text.TextEncoder();
    const respond = text.Transform.fromModel(() => 'reply');
    const decode = new text.TextDecoder();
    let history: readonly text.Message[] = [];
    const episode = trace();
    episode.run(() => {
      for (const value of ['one', 'two']) {
        history = respond.call([...history, ...encode.call(value)]);
        expect(decode.call(history)).toBe('reply');
      }
    });
    expect(history.length).toBe(4);
    expect(episode.calls.length).toBe(6);
  });

  it('text namespace composes without loading optional dependencies', async () => {
    // ``@huggingface/transformers`` is an optional peer that is not installed for the test suite:
    // importing the namespaces must not require it.
    const namespace = await import('../../src/ops/text/index.js');
    const integrations = await import('../../src/integrations/index.js');
    const messages = new namespace.TextEncoder().call('hello');
    const response = namespace.Transform.fromModel(() => 'answer').call(messages);
    expect(new namespace.TextDecoder().call(response)).toBe('answer');
    expect('Decode' in namespace).toBe(false);
    expect(typeof integrations.LocalModel).toBe('function');
    for (const file of ['local.ts', 'index.ts', 'openai.ts', 'jev.ts', 'http.ts']) {
      const source = readFileSync(new URL(`../../src/integrations/${file}`, import.meta.url), 'utf8');
      expect(source).not.toMatch(/^import[^;]*@huggingface\/transformers/m);
    }
  });
});
