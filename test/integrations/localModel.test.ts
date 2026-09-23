/** Port of ``tests/text/test_local_model.py``: adapter tests use supplied fakes (transport, not model ability). */
import { describe, expect, it } from 'vitest';
import { LocalModel, type LocalProcessor } from '../../src/integrations/index.js';
import { Classify, ImagePart, Message, ModelRequest } from '../../src/ops/text/index.js';
import { MissingDependencyError } from '../../src/errors.js';

class FakeTensor {
  constructor(private readonly rows: bigint[][]) {}
  get dims(): number[] {
    return [this.rows.length, this.rows[0]!.length];
  }
  tolist(): bigint[][] {
    return this.rows;
  }
}

function processor(answer = 'A small cat.') {
  const state: { messages: any[] | null; images: unknown[] | null; decoded: number[][] | null } = { messages: null, images: null, decoded: null };
  const fake = Object.assign(
    async (_text: string, images?: unknown[] | null) => {
      state.images = images ?? null;
      return { input_ids: new FakeTensor([[1n, 2n]]) };
    },
    {
      apply_chat_template(messages: any[]) {
        state.messages = messages;
        return 'prompt';
      },
      batch_decode(tokens: number[][]) {
        state.decoded = tokens;
        expect(tokens).toEqual([[3, 4]]);
        return [answer];
      },
    },
  ) as unknown as LocalProcessor;
  return { fake, state };
}

const model = () => ({
  evaluated: false,
  eval() {
    this.evaluated = true;
    return this;
  },
  async generate(inputs: Record<string, unknown>) {
    expect(inputs.do_sample).toBe(false);
    return new FakeTensor([[1n, 2n, 3n, 4n]]);
  },
});

function adapter(answer = 'A small cat.') {
  const { fake, state } = processor(answer);
  const loaded: ImagePart[] = [];
  const local = new LocalModel(model(), fake, {
    modelId: 'supplied-test-model',
    loadImage: (part) => {
      loaded.push(part);
      return { width: 3, height: 2, bytes: part.data };
    },
  });
  return { local, state, loaded };
}

describe('LocalModel', () => {
  it('text and image preserve order without prompt echo', async () => {
    const { local, state, loaded } = adapter();
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    const output = await local.acomplete(new ModelRequest([new Message('user', [new ImagePart({ data: png, mediaType: 'image/png', sourceRef: 'photo-1' })])]));
    expect(output.text).toBe('A small cat.');
    expect(state.images).toEqual([{ width: 3, height: 2, bytes: png }]);
    expect(loaded[0]!.sourceRef).toBe('photo-1');
    expect(state.messages![0].content[0].type).toBe('image');
    expect(output.providerMetadata!.model_id).toBe('supplied-test-model');
    expect(output.providerMetadata!.generated_tokens).toBe(2);
    expect(local.model.eval).toBeTypeOf('function');
  });

  it('url never downloaded implicitly', async () => {
    const { local } = adapter();
    await expect(local.acomplete(new ModelRequest([new Message('user', [new ImagePart({ url: 'https://example.invalid/photo.jpg' })])])))
      .rejects.toThrow(/bytes/);
  });

  it('structured output is actual json or an error', async () => {
    const { local, state } = adapter('{"label": "cat"}');
    const result = await local.acomplete(new ModelRequest([new Message('user', 'Classify.')], { responseSchema: { type: 'object' }, schemaName: 'choice' }));
    expect(result.structured).toEqual({ label: 'cat' });
    expect(state.messages![0]).toEqual({
      role: 'system',
      content: [{ type: 'text', text: '\nReturn only a JSON object, without Markdown fences or explanation, matching this schema:\n{"type": "object"}' }],
    });
    const bad = adapter('probably cat').local;
    await expect(bad.acomplete(new ModelRequest([new Message('user', 'Classify.')], { responseSchema: { type: 'object' } }))).rejects.toThrow(/JSON/);
    const array = adapter('[1]').local;
    await expect(array.acomplete(new ModelRequest([new Message('user', 'x')], { responseSchema: { type: 'object' } }))).rejects.toThrow(/object/);
  });

  it('invalid token budget', () => {
    expect(() => new LocalModel(model(), processor().fake, { modelId: 'test', maxNewTokens: 0 })).toThrow(/max_new_tokens/);
    expect(() => new LocalModel(model(), processor().fake, { modelId: '' })).toThrow(/model_id/);
  });

  it('token limit does not return partial answer as complete', async () => {
    const local = new LocalModel(model(), processor().fake, { modelId: 'test', maxNewTokens: 2 });
    await expect(local.acomplete(new ModelRequest([new Message('user', 'Explain.')]))).rejects.toThrow(/token limit/);
    const finished = Object.assign(model(), { generation_config: { eos_token_id: [4] } });
    const done = new LocalModel(finished, processor().fake, { modelId: 'test', maxNewTokens: 2 });
    expect((await done.acomplete(new ModelRequest([new Message('user', 'Explain.')]))).text).toBe('A small cat.');
  });

  it('serves structured operations through acall and abatch (serialized)', async () => {
    let active = 0;
    let peak = 0;
    const slow = {
      async generate() {
        active += 1;
        peak = Math.max(peak, active);
        await new Promise((resolve) => setTimeout(resolve, 5));
        active -= 1;
        return new FakeTensor([[1n, 2n, 3n, 4n]]);
      },
    };
    const local = new LocalModel(slow, processor('{"label": "a", "distribution": null, "confidence": null, "abstained": false}').fake, { modelId: 'm' });
    const classify = Classify.fromModel(local, { labels: ['a', 'b'] });
    const results = await classify.abatch([[new Message('user', 'x')], [new Message('user', 'y')]]);
    expect(results.map((result) => result.label)).toEqual(['a', 'a']);
    await Promise.all([classify.acall([new Message('user', 'x')]), classify.acall([new Message('user', 'y')])]);
    expect(peak).toBe(1);
    expect(local.configuration()).toEqual({ model_id: 'm', revision: null, max_new_tokens: 128 });
  });

  it('fromPretrained requires the optional peer dependency', async () => {
    let installed = true;
    try {
      await import('@huggingface/transformers' as string);
    } catch {
      installed = false;
    }
    if (!installed) await expect(LocalModel.fromPretrained('any/model')).rejects.toThrow(MissingDependencyError);
  });
});
