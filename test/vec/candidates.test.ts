/** Port of ``tests/vec/test_candidates.py``. */
import { describe, expect, it } from 'vitest';
import { Linear, empty, ones, tensor, zeros } from '../../src/nn/index.js';
import { ValueError } from '../../src/errors.js';
import { CandidateSet, Decide, Decode, Latent, Retrieve, Score, Space } from '../../src/ops/vec/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { DotScore } from './modules.js';
import { trace } from '../../src/_internal/tracing.js';
import type { Decision, Retrieval } from '../../src/ops/vec/index.js';

function singleCandidates(): CandidateSet {
  const space = new Space('retrieval/shared', 2);
  return new CandidateSet(
    new Latent(tensor([1, 0], { requiresGrad: true }), space),
    new Latent(tensor([[0.2, 0], [0.9, 0], [-0.5, 0]], { requiresGrad: true }), space, { sources: ['memory:index'], metadata: { snapshot: 4 } }),
    ['low', 'high', 'negative'],
    [{ row: 1 }, { row: 2 }, { row: 3 }],
  );
}

function scorer(space: Space, meaning: string): Score {
  return Score.fromModule(new DotScore(), { querySpace: space, candidateSpace: space, meaning });
}

describe('vector candidates (tests/vec/test_candidates.py)', () => {
  it('score returns named tensor scores and preserves gradients', () => {
    const space = new Space('retrieval/shared', 2);
    const score = scorer(space, 'unnormalized dot-product similarity');
    const result = score.call(singleCandidates());
    expect(result.meaning).toBe('unnormalized dot-product similarity');
    expectClose(result.values.data, [0.2, 0.9, -0.5], 1e-6);
    result.values.sum().backward();
    expect((score.module as DotScore).scale.grad).not.toBeNull();
    expect(result.candidates.query.tensor.grad).not.toBeNull();
    expect(result.candidates.candidates.tensor.grad).not.toBeNull();
  });

  it('score rejects equal dimensions from an incompatible space', () => {
    const expected = new Space('model-a', 2);
    const values = new CandidateSet(new Latent(ones([2]), expected), new Latent(ones([2, 2]), new Space('model-b', 2)), ['a', 'b']);
    expect(() => scorer(expected, 'similarity').call(values)).toThrow(/incompatible.*space/);
  });

  it('candidate set rejects empty or misaligned candidates', () => {
    const space = new Space('candidates', 2);
    expect(() => new CandidateSet(new Latent(ones([2]), space), new Latent(empty([0, 2]), space), [])).toThrow(/at least one/);
    expect(() => new CandidateSet(new Latent(ones([2, 2]), space), new Latent(ones([3, 4, 2]), space), ['a', 'b', 'c', 'd'])).toThrow(/batch shape/);
    expect(() => new CandidateSet(new Latent(ones([2]), space), new Latent(ones([2, 2]), space), ['only-one'])).toThrow(/identities/);
    for (const identities of [['', 'b'], ['same', 'same']]) {
      expect(() => new CandidateSet(new Latent(ones([2]), space), new Latent(ones([2, 2]), space), identities)).toThrow(/unique nonempty/);
    }
  });

  it('decide handles batches while identity conversion is explicit', () => {
    const space = new Space('decision/shared', 2);
    const candidates = new CandidateSet(
      new Latent(tensor([[1, 0], [0, 1]]), space),
      new Latent(tensor([[[0.1, 0], [0.9, 0]], [[0, 0.8], [0, 0.2]]]), space),
      ['first', 'second'],
    );
    const decision = new Decide().call(scorer(space, 'utility logit').call(candidates));
    expect(decision.indices.toArray()).toEqual([1, 0]);
    expect(decision.indices.dtype).toBe('int64');
    expectClose(decision.scores.data, [0.9, 0.8], 1e-6);
    expect(decision.identities).toEqual(['second', 'first']);
    expect(Object.isFrozen(decision.identities)).toBe(true);
    expect(() => decision.identity).toThrow(/batched/);
    expect(decision.items.tensor.shape).toEqual([2, 2]);
  });

  it('retrieve ranks existing candidates and preserves candidate metadata', () => {
    const candidates = singleCandidates();
    const scored = scorer(candidates.query.space, 'relevance score').call(candidates);
    const retrieval = new Retrieve({ k: 2 }).call(scored);
    expect(retrieval.indices.toArray()).toEqual([1, 0]);
    expectClose(retrieval.scores.data, [0.9, 0.2], 1e-6);
    expect(retrieval.identities).toEqual(['high', 'low']);
    expect(retrieval.metadata).toEqual([{ row: 2 }, { row: 1 }]);
    expect(retrieval.items.sources).toEqual(['memory:index']);
    expect(retrieval.items.metadata).toEqual({ snapshot: 4 });
    expectClose(retrieval.items.tensor.data, [0.9, 0, 0.2, 0], 1e-6);
  });

  it('retrieve rejects k beyond the candidate bound', () => {
    const candidates = singleCandidates();
    const scored = scorer(candidates.query.space, 'relevance').call(candidates);
    expect(() => new Retrieve({ k: 4 }).call(scored)).toThrow(/only 3 candidates/);
    expect(() => new Retrieve({ k: 0 })).toThrow(/positive/);
  });

  it('decide and retrieve exclude masked candidates and bound the valid count', () => {
    const space = new Space('masked/shared', 1);
    const mask = tensor([true, false, true]);
    const candidates = new CandidateSet(new Latent(tensor([1]), space), new Latent(tensor([[1], [100], [2]]), space, { mask }), ['one', 'masked', 'two']);
    const scored = scorer(space, 'similarity').call(candidates);
    expect(new Decide().call(scored).identity).toBe('two');
    expect(new Retrieve({ k: 2 }).call(scored).identities).toEqual(['two', 'one']);
    expect(() => new Retrieve({ k: 3 }).call(scored)).toThrow(/valid candidates/);
    const descending = scorer(space, 'cost').call(new CandidateSet(
      new Latent(tensor([1]), space), new Latent(tensor([[1], [-100], [2]]), space, { mask }), ['one', 'masked', 'two'],
    ));
    expect(new Decide({ largest: false }).call(descending).identity).toBe('one');
    expect(new Retrieve({ k: 2, largest: false }).call(descending).identities).toEqual(['one', 'two']);
  });

  it('candidate availability mask must be boolean and nonempty per batch', () => {
    const space = new Space('masked/shared', 1);
    expect(() => new CandidateSet(new Latent(tensor([1]), space), new Latent(ones([2, 1]), space, { mask: ones([2]) }), ['a', 'b'])).toThrow(/boolean/);
    expect(() => new CandidateSet(new Latent(tensor([1]), space), new Latent(ones([2, 1]), space, { mask: zeros([2], { dtype: 'bool' }) }), ['a', 'b'])).toThrow(/valid candidate/);
  });

  it('decode uses the supplied module and retains autograd', () => {
    const space = new Space('decoder/input', 3);
    const module = new Linear(3, 2, { bias: false });
    const decode = Decode.fromModule(module, { inputSpace: space, output: 'two regression values' });
    const source = ones([3]);
    source.requiresGrad = true;
    const result = decode.call(new Latent(source, space));
    expect(result.shape).toEqual([2]);
    result.sum().backward();
    expect(source.grad).not.toBeNull();
    expect(module.weight.grad).not.toBeNull();
    expect(() => decode.call(new Latent(ones([3]), new Space('other', 3)))).toThrow(/incompatible.*space/);
  });

  it('scores and candidate records validate shapes and are frozen records', () => {
    const candidates = singleCandidates();
    expect(Object.isFrozen(candidates)).toBe(true);
    expect(candidates.count).toBe(3);
    expect(candidates.metadata).toEqual([{ row: 1 }, { row: 2 }, { row: 3 }]);
    expect(CandidateSet.fromRecord(candidates.toRecord()).identities).toEqual(candidates.identities);
    const scored = scorer(candidates.query.space, 'relevance').call(candidates);
    expect(() => new (scored.constructor as any)(ones([2]), 'relevance', candidates)).toThrow(ValueError);
    expect(() => new (scored.constructor as any)(ones([3], { dtype: 'int64' }), 'relevance', candidates)).toThrow(/floating-point/);
    expect(() => new (scored.constructor as any)(ones([3]), '  ', candidates)).toThrow(/meaning/);
  });

  it('score, decide and retrieve records trace and replay as one composition', () => {
    const candidates = singleCandidates();
    const score = scorer(candidates.query.space, 'relevance');
    const session = trace();
    const [decision, retrieval] = session.run(() => {
      const scored = score.call(candidates);
      return [new Decide().call(scored), new Retrieve({ k: 2 }).call(scored)] as const;
    });
    expect(session.calls.length).toBe(3);
    const replayed = session.replay(session.ref(decision)) as Decision;
    expect(replayed.identity).toBe('high');
    expect(replayed.scores.item()).toBeCloseTo(decision.scores.item(), 6);
    expect((session.replay(session.ref(retrieval)) as Retrieval).identities).toEqual(['high', 'low']);
    expect(session.example(session.ref(decision)).inputs.size).toBe(1);
  });
});
