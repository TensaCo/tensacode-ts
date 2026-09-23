/** Port of ``tests/text/test_structured_operations.py``. */
import { describe, expect, it } from 'vitest';
import * as text from '../../src/ops/text/index.js';
import { trace } from '../../src/_internal/tracing.js';

class ScriptedModel {
  readonly outputs: text.ModelOutput[];
  readonly requests: text.ModelRequest[] = [];

  constructor(...outputs: text.ModelOutput[]) {
    this.outputs = outputs;
  }

  complete(request: text.ModelRequest): text.ModelOutput {
    this.requests.push(request);
    return this.outputs.shift()!;
  }
}

const structured = (value: Record<string, unknown>) => new text.ModelOutput({ structured: value });
const messages = (content = 'example') => Object.freeze([new text.Message('user', content)]);

describe('structured operations', () => {
  it('classify returns only provider supplied distribution', () => {
    const model = new ScriptedModel(
      structured({ label: 'urgent', distribution: { routine: 0.1, urgent: 0.9 }, abstained: false }),
      structured({ label: 'routine', abstained: false }),
    );
    const classify = text.Classify.fromModel(model, { labels: ['routine', 'urgent'], instructions: 'Assess urgency' });
    const supplied = classify.call(messages());
    const missing = classify.call(messages('second'));
    expect(supplied).toEqual(new text.ClassificationResult('urgent', { distribution: { routine: 0.1, urgent: 0.9 } }));
    expect(missing.label).toBe('routine');
    expect(missing.distribution).toBeNull();
    expect(missing.confidence).toBeNull();
    expect(model.requests[0]!.schemaName).toBe('tensorcode.classify');
    expect(Object.isFrozen(supplied) && Object.isFrozen(supplied.distribution)).toBe(true);
  });

  it('classify validates labels and probability distribution', () => {
    let classify = text.Classify.fromModel(
      new ScriptedModel(structured({ label: 'invented', distribution: { routine: 0.4, urgent: 0.6 }, abstained: false })),
      { labels: ['routine', 'urgent'] },
    );
    expect(() => classify.call(messages())).toThrow(expect.objectContaining({
      name: 'InvalidModelOutput', message: expect.stringMatching(/configured labels/),
    }));
    classify = text.Classify.fromModel(
      new ScriptedModel(structured({ label: 'urgent', distribution: { routine: 0.4, urgent: 0.4 }, abstained: false })),
      { labels: ['routine', 'urgent'] },
    );
    expect(() => classify.call(messages())).toThrow(/sum to 1/);
  });

  it('distribution values must be finite numbers in [0, 1] over exactly the labels', () => {
    const run = (distribution: unknown) => text.Classify.fromModel(
      new ScriptedModel(structured({ label: 'a', distribution, abstained: false })), { labels: ['a', 'b'] },
    ).call(messages());
    expect(() => run({ a: 1 })).toThrow(/keys must match/);
    expect(() => run({ a: 1.5, b: -0.5 })).toThrow(/between 0 and 1/);
    expect(() => run({ a: true, b: 0 })).toThrow(/must be numbers/);
    expect(() => run([0.5, 0.5])).toThrow(/mapping/);
    expect(run({ a: 0.9995, b: 0.0 }).distribution).toEqual({ a: 0.9995, b: 0 });
    const confidence = (value: unknown) => text.Classify.fromModel(
      new ScriptedModel(structured({ label: 'a', confidence: value, abstained: false })), { labels: ['a', 'b'] },
    ).call(messages());
    expect(() => confidence(1.2)).toThrow(/confidence/);
    expect(() => confidence(true)).toThrow(/confidence/);
    expect(confidence(null).confidence).toBeNull();
  });

  it('classify represents explicit abstention without distribution', () => {
    const result = text.Classify.fromModel(new ScriptedModel(structured({ label: null, abstained: true })), { labels: ['yes', 'no'] })
      .call(messages());
    expect(result).toEqual(new text.ClassificationResult(null, { abstained: true }));
    expect(result.value).toBeNull();
  });

  it('structured result requires explicit abstention state', () => {
    const classify = text.Classify.fromModel(new ScriptedModel(structured({ label: 'yes' })), { labels: ['yes', 'no'] });
    expect(() => classify.call(messages())).toThrow(/abstained/);
  });

  it('score validates rubric distribution and preserves provider confidence', () => {
    const score = text.Score.fromModel(
      new ScriptedModel(structured({ score: 1.7, distribution: { 0: 0.1, 1: 0.1, 2: 0.8 }, confidence: 0.81, abstained: false })),
      { rubric: ['can wait', 'this week', 'today'], instructions: 'Assess urgency' },
    );
    expect(score.call(messages())).toEqual(new text.ScoreResult(1.7, { distribution: { 0: 0.1, 1: 0.1, 2: 0.8 }, confidence: 0.81 }));
  });

  it('score rejects noncanonical or colliding distribution keys', () => {
    const score = text.Score.fromModel(
      new ScriptedModel(structured({ score: 1.0, distribution: { 0: 0.25, '00': 0.25, 1: 0.5 }, abstained: false })),
      { rubric: ['low', 'high'] },
    );
    expect(() => score.call(messages())).toThrow(/keys/);
    const outside = text.Score.fromModel(new ScriptedModel(structured({ score: 2, abstained: false })), { rubric: ['low', 'high'] });
    expect(() => outside.call(messages())).toThrow(/outside the configured rubric/);
  });

  it('decide rejects unconfigured choice and allows abstention', () => {
    const decide = text.Decide.fromModel(new ScriptedModel(structured({ choice: 'delete', abstained: false })), { options: ['archive', 'reply'] });
    expect(() => decide.call(messages())).toThrow(/configured options/);
    const abstained = text.Decide.fromModel(new ScriptedModel(structured({ choice: null, abstained: true })), { options: ['archive', 'reply'] })
      .call(messages());
    expect(abstained).toEqual(new text.DecisionResult(null, { abstained: true }));
  });

  it('retrieve returns only configured items and keeps scores semantically distinct', () => {
    const retrieve = text.Retrieve.fromModel(
      new ScriptedModel(structured({ keys: ['policy'], scores: { policy: 2.4, faq: -1.0 }, abstained: false })),
      { items: { policy: { text: 'refund policy' }, faq: 'general' }, descriptions: { policy: 'refund policy', faq: 'general questions' }, limit: 1 },
    );
    const result = retrieve.call(messages('refund'));
    expect(result.keys).toEqual(['policy']);
    expect(result.items).toEqual([{ text: 'refund policy' }]);
    expect(result.scores).toEqual({ policy: 2.4, faq: -1.0 });
    expect(result.distribution).toBeNull();
  });

  it('retrieve reports non string model keys as invalid output', () => {
    const retrieve = text.Retrieve.fromModel(new ScriptedModel(structured({ keys: [{}], scores: null, abstained: false })), { items: { a: 'first' } });
    expect(() => retrieve.call(messages())).toThrow(text.InvalidModelOutput);
    expect(() => text.Retrieve.fromModel(new ScriptedModel(structured({ keys: [{}], scores: null, abstained: false })), { items: { a: 'first' } })
      .call(messages())).toThrow(/keys/);
  });

  it('retrieve configuration requires descriptions for non-text items', () => {
    const model = new ScriptedModel();
    expect(() => text.Retrieve.fromModel(model, { items: { a: { x: 1 } } })).toThrow(/descriptions/);
    expect(() => text.Retrieve.fromModel(model, { items: {} })).toThrow(/nonempty mapping/);
    expect(() => text.Retrieve.fromModel(model, { items: { a: 'x' }, limit: 2 })).toThrow(/limit/);
    expect(() => text.Classify.fromModel(model, { labels: ['a', 'a'] })).toThrow(/unique/);
    expect(() => text.Classify.fromModel(model, { labels: [] })).toThrow(/nonempty/);
    expect(() => text.Classify.fromModel(model, { labels: ['a'], descriptions: { b: 'x' } })).toThrow(/descriptions/);
    expect(() => text.Score.fromModel(model, { rubric: [] })).toThrow(/rubric/);
    expect(() => text.Classify.fromModel(model, { labels: ['a'], other: 1 })).toThrow(/Unknown external model options/);
  });

  it('structured operation rejects text only output instead of guessing json', () => {
    const classify = text.Classify.fromModel(new ScriptedModel(new text.ModelOutput({ text: '{"label": "yes"}' })), { labels: ['yes', 'no'] });
    expect(() => classify.call(messages())).toThrow(/structured/);
  });

  it('acall is explicit and sync call never returns awaitable', async () => {
    const classify = text.Classify.fromModel(
      new ScriptedModel(structured({ label: 'a', abstained: false }), structured({ label: 'b', abstained: false })), { labels: ['a', 'b'] },
    );
    const syncResult = classify.call(messages('sync'));
    expect(typeof (syncResult as unknown as { then?: unknown }).then).toBe('undefined');
    const asyncResult = await classify.acall(messages('async'));
    expect(syncResult.label).toBe('a');
    expect(asyncResult.label).toBe('b');
  });

  it('batch preserves input order and requires exact result count', () => {
    class BatchModel extends ScriptedModel {
      completeBatch(requests: readonly text.ModelRequest[]): readonly text.ModelOutput[] {
        this.requests.push(...requests);
        return [structured({ label: 'a', abstained: false }), structured({ label: 'b', abstained: false })];
      }
    }
    const classify = text.Classify.fromModel(new BatchModel(), { labels: ['a', 'b'] });
    const results = classify.batch([messages('one'), messages('two')]);
    expect(results.map((result) => result.label)).toEqual(['a', 'b']);

    class BadBatchModel extends BatchModel {
      override completeBatch(): readonly text.ModelOutput[] {
        return [];
      }
    }
    expect(() => text.Classify.fromModel(new BadBatchModel(), { labels: ['a', 'b'] }).batch([messages('one')])).toThrow(/batch result count/);
    expect(() => classify.batch([messages('one')], { contexts: [] })).toThrow(/contexts must match/);
  });

  it('batch falls back to per item calls inside a trace', () => {
    class TraceSafeModel extends ScriptedModel {
      completeBatch(): never {
        throw new Error('backend batch must be disabled while tracing');
      }
    }
    const classify = text.Classify.fromModel(new TraceSafeModel(structured({ label: 'a', abstained: false })), { labels: ['a', 'b'] });
    const session = trace();
    const results = session.run(() => {
      new text.TextEncoder().call('one');
      const encodedRef = session.calls[session.calls.length - 1]!.output;
      return classify.batch([encodedRef as never]);
    });
    expect(results[0]!.label).toBe('a');
    expect(session.calls.length).toBe(2);
  });

  it('batch records provider failures inside a trace', () => {
    const failing = {
      complete(): never {
        throw new Error('offline');
      },
      completeBatch(): never {
        throw new Error('backend batch must be disabled while tracing');
      },
    };
    const classify = text.Classify.fromModel(failing, { labels: ['a', 'b'] });
    const session = trace();
    expect(() => session.run(() => classify.batch([messages()]))).toThrow(/offline/);
    expect(session.calls.length).toBe(1);
    expect(session.calls[0]!.error).toBe('Error: offline');
  });

  it('structured results with distributions are traceable', () => {
    const classify = text.Classify.fromModel(
      new ScriptedModel(structured({ label: 'a', distribution: { a: 0.75, b: 0.25 }, abstained: false })), { labels: ['a', 'b'] },
    );
    const session = trace();
    const result = session.run(() => classify.call(messages()));
    expect(result.distribution).toEqual({ a: 0.75, b: 0.25 });
    expect(session.calls.length).toBe(1);
  });

  it('abatch supports async only models', async () => {
    const asyncOnly = {
      async acomplete(request: text.ModelRequest): Promise<text.ModelOutput> {
        const label = request.messages[request.messages.length - 1]!.content;
        return structured({ label, abstained: false });
      },
    };
    const classify = text.Classify.fromModel(asyncOnly, { labels: ['a', 'b'] });
    const results = await classify.abatch([messages('a'), messages('b')]);
    expect(results.map((result) => result.label)).toEqual(['a', 'b']);
    expect(() => classify.call(messages('a'))).toThrow(/acall/);
  });

  it('abatch fuses acompleteBatch outside tracing and calls per item inside a trace', async () => {
    const calls: string[] = [];
    const model = {
      async acomplete(request: text.ModelRequest) {
        calls.push('acomplete');
        return structured({ label: request.messages[0]!.content, abstained: false });
      },
      async acompleteBatch(requests: readonly text.ModelRequest[]) {
        calls.push('batch');
        return requests.map((request) => structured({ label: request.messages[0]!.content, abstained: false }));
      },
    };
    const classify = text.Classify.fromModel(model, { labels: ['a', 'b'] });
    expect((await classify.abatch([messages('b'), messages('a')])).map((result) => result.label)).toEqual(['b', 'a']);
    expect(calls).toEqual(['batch']);
    const session = trace();
    await session.run(() => classify.abatch([messages('a')]));
    expect(calls).toEqual(['batch', 'acomplete']);
    expect(session.calls.length).toBe(1);
  });

  it('external operations are not replayable and have no owned artifacts', async () => {
    const classify = text.Classify.fromModel(new ScriptedModel(), { labels: ['a', 'b'] });
    expect(classify.replayable).toBe(false);
    expect(() => classify.trainingOperation).toThrow(/external/);
    expect(() => classify.loss(messages(), {})).toThrow(/external/);
    await expect(classify.savePretrained('unused')).rejects.toThrow(/external/);
    expect(classify.configuration()).toEqual({ type: 'text_classify', labels: ['a', 'b'], instructions: null, model: { type: 'js:ScriptedModel' } });
  });
});
