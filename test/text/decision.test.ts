/** Port of ``tests/runtime/test_decision.py``: ordinary composition of public operations and authored policies. */
import { describe, expect, it } from 'vitest';
import * as text from '../../src/ops/text/index.js';

class Model {
  readonly requests: text.ModelRequest[] = [];
  constructor(private readonly structured: Record<string, unknown>) {}
  complete(request: text.ModelRequest): text.ModelOutput {
    this.requests.push(request);
    return new text.ModelOutput({ structured: this.structured });
  }
}

describe('decision compositions', () => {
  it('keeps distribution and explicit instructions', () => {
    const model = new Model({ label: 'billing', distribution: { billing: 0.75, technical: 0.25 }, abstained: false });
    const encode = new text.TextEncoder();
    const decide = text.Classify.fromModel(model, { labels: ['billing', 'technical'], instructions: 'Route this support request' });
    const result = decide.call(encode.call('I was charged twice'));
    expect(result.value).toBe('billing');
    expect(result.distribution).toEqual({ billing: 0.75, technical: 0.25 });
    expect(model.requests[0]!.instructions).toBe('Route this support request');
  });

  it('applies an explicit authored selection policy', () => {
    const model = new Model({ label: 'billing', distribution: { billing: 0.51, technical: 0.49 }, abstained: false });
    const seen: text.ClassificationResult[] = [];
    const requireMargin = (result: text.ClassificationResult) => {
      seen.push(result);
      return new text.ClassificationResult(null, { distribution: result.distribution, abstained: true });
    };
    const encode = new text.TextEncoder();
    const decide = text.Classify.fromModel(model, { labels: ['billing', 'technical'] });
    const result = requireMargin(decide.call(encode.call('ambiguous')));
    expect(result.abstained && result.value === null).toBe(true);
    expect(result.distribution).toEqual({ billing: 0.51, technical: 0.49 });
    expect(seen[0]!.value).toBe('billing');
  });

  it('passes explicit context to a decision operation', () => {
    const calls: unknown[] = [];
    const encode = (value: string) => {
      calls.push(['encode', value]);
      return value.toUpperCase();
    };
    const decide = (value: string, { context }: { context?: Record<string, unknown> } = {}) => {
      calls.push(['decide', value, context]);
      return 'chosen';
    };
    expect(decide(encode('input'), { context: { x: 1 } })).toBe('chosen');
    expect(calls).toEqual([['encode', 'input'], ['decide', 'INPUT', { x: 1 }]]);
  });
});
