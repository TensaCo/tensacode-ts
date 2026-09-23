/** Port of ``tests/text/test_likelihood_and_questions.py``. */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it, vi } from 'vitest';
import * as text from '../../src/ops/text/index.js';
import { JevModel, ProviderProtocolError } from '../../src/integrations/index.js';
import { SGD, type Tensor } from '../../src/nn/index.js';
import { trace } from '../../src/_internal/tracing.js';
import { fakeServer } from '../integrations/fakeServer.js';

const FOUNDATION = 'test/fixtures/text/foundation';
const VALUE = Object.freeze([new text.Message('user', 'question')]);
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-text-likelihood-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const LIKELIHOOD_CASES: [string, any, Record<string, unknown>, Record<string, unknown>][] = [
  ['Classify', text.Classify, { labels: ['question', 'answer'], descriptions: { answer: 'answer' } },
    { label: 'answer', distribution: null, confidence: null, abstained: false }],
  ['Decide', text.Decide, { options: ['question', 'answer'] },
    { choice: 'answer', distribution: { question: 0.25, answer: 0.75 }, confidence: null, abstained: false }],
  ['Score', text.Score, { rubric: ['question', 'answer'] }, { score: 1, distribution: null, confidence: null, abstained: false }],
  ['Retrieve', text.Retrieve, { items: { a: 'answer', b: 'question' }, limit: 1 }, { keys: ['b'], scores: null, abstained: false }],
];

const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);

describe('likelihood decoding', () => {
  for (const [name, cls, options, target] of LIKELIHOOD_CASES) {
    it(`${name} scores alternatives in one encoder pass`, async () => {
      const op = await cls.fromFoundation(FOUNDATION, { config: { ...options, decoding: 'likelihood' } });
      const encoder = vi.spyOn(op.model.model.encoder, 'run');
      const result = op.call(VALUE);
      expect(encoder).toHaveBeenCalledTimes(1);
      encoder.mockRestore();
      if (cls === text.Retrieve) {
        expect(Object.keys(result.scores).sort()).toEqual(Object.keys(options.items as object).sort());
        expect(result.keys.length).toBe(1);
        const best = Object.entries<number>(result.scores).sort((a, b) => b[1] - a[1])[0]![0];
        expect(result.keys[0]).toBe(best);
        expect(result.distribution).toBeNull();
      } else {
        const distribution = result.distribution as Record<string, number>;
        expect(Math.abs(sum(Object.values(distribution)) - 1)).toBeLessThan(1e-6);
        expect(result.confidence).toBe(Math.max(...Object.values(distribution)));
        expect(result.abstained).toBe(false);
        if (cls === text.Score) {
          expect(result.value).toBeCloseTo(sum(Object.entries(distribution).map(([key, p]) => Number(key) * p)), 12);
        } else {
          const best = Object.entries(distribution).sort((a, b) => b[1] - a[1])[0]![0];
          expect(result.value).toBe(best);
        }
      }

      const loss: Tensor = op.loss(VALUE, target);
      loss.backward();
      expect(op.parameters().some((parameter: Tensor) => parameter.grad !== null && parameter.grad.abs().sum().item() > 0)).toBe(true);

      await op.savePretrained(join(scratch, name));
      const restored = await cls.fromPretrained(join(scratch, name));
      expect(restored.decoding).toBe('likelihood');
      expect(restored.call(VALUE)).toEqual(result);
      expect(restored.loss(VALUE, target).item()).toBeCloseTo(loss.item(), 5);

      restored.train();
      // The fixture's weights are sharpened (x4), so a smaller step than Python's lr=.5 keeps SGD stable.
      const optimizer = new SGD(restored.parameters(), { lr: 0.05 });
      const before = restored.loss(VALUE, target).item();
      for (let step = 0; step < 5; step += 1) {
        optimizer.zeroGrad();
        (restored.trainingOperation.call({ inputs: VALUE, targets: target }) as Tensor).backward();
        optimizer.step();
      }
      expect(restored.loss(VALUE, target).item()).toBeLessThan(before);
    });
  }

  it('likelihood configuration is strict', async () => {
    await expect(text.Classify.fromFoundation(FOUNDATION, { config: { labels: ['a'], decoding: 'beam' } })).rejects.toThrow(/decoding/);
    await expect(text.Classify.fromFoundation(FOUNDATION, { config: { labels: ['a'], likelihood_normalization: 'mean' } })).rejects.toThrow(/requires/);
    await expect(text.Classify.fromFoundation(FOUNDATION, { config: { labels: ['a'], descriptions: { b: 'x' } } })).rejects.toThrow(/descriptions/);
    expect(() => text.Classify.fromModel((_request: unknown) => null, { labels: ['a'], decoding: 'likelihood' })).toThrow(/Unknown/);
    await expect(text.Transform.fromFoundation(FOUNDATION, { config: { decoding: 'likelihood' } })).rejects.toThrow(/foundation config/);
    // Python's 4-token fixture maps yes/no to [UNK]; this fixture knows them, so use two unknown words.
    const same = await text.Classify.fromFoundation(FOUNDATION, { config: { labels: ['zebra', 'yak'], decoding: 'likelihood' } });
    expect(() => same.call(VALUE)).toThrow(/distinct/);
    const op = await text.Classify.fromFoundation(FOUNDATION, {
      config: { labels: ['yes', 'no'], decoding: 'likelihood', likelihood_normalization: 'mean' },
    });
    expect(() => op.loss(VALUE, { label: null, distribution: null, confidence: null, abstained: true })).toThrow(/abstention/);
    expect(op.configuration().likelihood_normalization).toBe('mean');
    expect(op.likelihoodNormalization).toBe('mean');
  });

  it('default generation mode is unchanged', async () => {
    const op = await text.Classify.fromFoundation(FOUNDATION, { config: { labels: ['yes', 'no'] } });
    expect(op.decoding).toBe('generate');
    expect('decoding' in op.configuration()).toBe(false);
  });

  it('score targets need a distribution or an integer level', async () => {
    const op = await text.Score.fromFoundation(FOUNDATION, { config: { rubric: ['bad', 'good'], decoding: 'likelihood' } });
    expect(() => op.loss(VALUE, { score: 0.5, distribution: null, confidence: null, abstained: false })).toThrow(/integer level/);
    expect(op.loss(VALUE, { score: 0.5, distribution: { 0: 0.5, 1: 0.5 }, confidence: null, abstained: false }).item()).toBeGreaterThan(0);
  });
});

class QuestionProvider {
  readonly calls: [string, unknown][] = [];

  answer(request: text.ModelRequest): Record<string, unknown> {
    if (request.schemaName === 'tensorcode.score') return { score: 1, distribution: null, confidence: null, abstained: false };
    return { label: 'b', distribution: null, confidence: 0.5, abstained: false };
  }

  complete(request: text.ModelRequest): text.ModelOutput {
    this.calls.push(['complete', request.instructions]);
    return new text.ModelOutput({ structured: this.answer(request) });
  }

  completeQuestions(requests: Record<string, text.ModelRequest>): Record<string, text.ModelOutput> {
    this.calls.push(['questions', Object.keys(requests)]);
    return Object.fromEntries(Object.entries(requests).map(([name, request]) => [name, new text.ModelOutput({ structured: this.answer(request) })]));
  }

  async acompleteQuestions(requests: Record<string, text.ModelRequest>): Promise<Record<string, text.ModelOutput>> {
    return this.completeQuestions(requests);
  }
}

describe('ask', () => {
  it('fuses questions for one shared provider', async () => {
    const provider = new QuestionProvider();
    const questions = {
      topic: text.Classify.fromModel(provider, { labels: ['a', 'b'], instructions: 'topic' }),
      urgency: text.Score.fromModel(provider, { rubric: ['low', 'high'], instructions: 'urgency' }),
    };
    const answers = text.ask(VALUE, questions);
    expect(provider.calls).toEqual([['questions', ['topic', 'urgency']]]);
    expect(answers.topic.label).toBe('b');
    expect(answers.urgency.value).toBe(1);
    expect(() => { (answers as Record<string, unknown>).topic = null; }).toThrow(TypeError);
    expect((await text.aask(VALUE, questions)).urgency.value).toBe(1);
  });

  it('calls each operation when fusion is unavailable', async () => {
    const provider = new QuestionProvider();
    const other = new QuestionProvider();
    const questions = {
      first: text.Classify.fromModel(provider, { labels: ['a', 'b'], instructions: 'first' }),
      second: text.Classify.fromModel(other, { labels: ['a', 'b'], instructions: 'second' }),
    };
    expect(text.ask(VALUE, questions).second.label).toBe('b');
    expect(provider.calls).toEqual([['complete', 'first']]);
    expect(other.calls).toEqual([['complete', 'second']]);

    const shared = {
      x: text.Classify.fromModel(provider, { labels: ['a', 'b'] }),
      y: text.Classify.fromModel(provider, { labels: ['a', 'b'] }),
    };
    provider.calls.length = 0;
    const session = trace();
    session.run(() => text.ask(VALUE, shared));
    expect(provider.calls.map(([kind]) => kind)).toEqual(['complete', 'complete']);
    expect(session.calls.length).toBe(2);

    const owned = await text.Classify.fromFoundation(FOUNDATION, { config: { labels: ['yes', 'no'], decoding: 'likelihood' } });
    const mixed = await text.aask(VALUE, { owned, external: text.Classify.fromModel(provider, { labels: ['a', 'b'] }) });
    expect(mixed.owned.distribution).not.toBeNull();
    expect(mixed.external.label).toBe('b');
  });

  it('validates questions and answers', () => {
    expect(() => text.ask(VALUE, {})).toThrow(/nonempty/);
    expect(() => text.ask(VALUE, { t: text.Transform.fromModel(() => 'x') as never })).toThrow(/Classify/);

    class Wrong extends QuestionProvider {
      override completeQuestions(): Record<string, text.ModelOutput> {
        return { other: new text.ModelOutput({ structured: {} }) };
      }
    }
    expect(() => text.ask(VALUE, { a: text.Classify.fromModel(new Wrong(), { labels: ['a', 'b'] }) })).toThrow(text.InvalidModelOutput);
    expect(() => text.ask(VALUE, { a: text.Classify.fromModel(new Wrong(), { labels: ['a', 'b'] }) })).toThrow(/names/);
  });

  it('jev fuses questions, maps noul and descriptions', async () => {
    const response = {
      model: 'jev-1.13.0',
      answers: {
        spam: { type: 'noul', noul: 0.2 },
        route: { type: 'choice', choice: 'billing', confidence: 0.6, probabilities: { billing: 0.8, technical: 0.2 } },
      },
      usage: { input_tokens: 12, output_tokens: 0 },
    };
    const server = fakeServer(() => [200, response, 0]);
    const model = new JevModel({ baseUrl: server.url, apiKey: 'key', fetch: server.fetch });
    const questions = {
      spam: text.Classify.fromModel(model, { labels: ['true', 'false'], instructions: 'Is this spam?', descriptions: { true: 'unsolicited advertising' } }),
      route: text.Decide.fromModel(model, { options: ['billing', 'technical'], instructions: 'Route', descriptions: { billing: 'payments and charges' } }),
    };
    const answers = await text.aask(VALUE, questions);
    expect(server.requests.length).toBe(1);
    const sent = server.requests[0]!.json;
    expect(sent.state).toEqual([{ role: 'user', content: 'question' }]);
    expect(sent.questions).toEqual({
      spam: { type: 'noul', instructions: 'Is this spam?', criteria: { true: 'unsolicited advertising', false: null } },
      route: { type: 'choice', instructions: 'Route', criteria: { billing: 'payments and charges', technical: null } },
    });
    expect(answers.spam.label).toBe('false');
    expect(answers.spam.distribution).toEqual({ true: 0.2, false: 0.8 });
    expect(answers.spam.confidence).toBeNull();
    expect(answers.route.choice).toBe('billing');
    expect(answers.route.confidence).toBe(0.6);
    // A synchronous ask cannot fuse an asynchronous provider; each question needs acall.
    expect(() => text.ask(VALUE, questions)).toThrow(/acall/);

    const other = [new text.Message('user', 'different')];
    const requests = Object.fromEntries(Object.entries(questions).map(([name, op]) => [name, op._request(name === 'spam' ? VALUE : other, null)]));
    await expect(model.acompleteQuestions(requests)).rejects.toThrow(/same messages/);
    await expect(model.acompleteQuestions(requests)).rejects.toThrow(ProviderProtocolError);
    expect(server.requests.length).toBe(1);

    const single = fakeServer(() => [200, { answers: { result: { type: 'noul', noul: 1.5 } } }, 0]);
    await expect(text.Classify.fromModel(new JevModel({ baseUrl: single.url, apiKey: 'key', fetch: single.fetch }), { labels: ['true', 'false'] })
      .acall(VALUE)).rejects.toThrow(/probability/);
  });
});
