/** Port of python/tests/runtime/test_episodic.py. */
import { describe, expect, it } from 'vitest';
import { EpisodicMemory, fsum } from '../../src/_internal/memory/episodic.js';
import { Evidence } from '../../src/tools/cognition.js';
import { ValueError } from '../../src/errors.js';

describe('EpisodicMemory', () => {
  it('retrieves by cosine, excludes episodes, removes and bounds capacity', () => {
    const memory = new EpisodicMemory({ capacity: 2, modelFingerprint: 'v1' });
    memory.insert(new Evidence('a', 'A', 'source-a'), [1, 0], { episodeId: 'past' });
    memory.insert(new Evidence('b', 'B', 'source-b'), [0, 1], { episodeId: 'now' });
    const hits = memory.query([1, 0], { modelFingerprint: 'v1' });
    expect(hits[0]!.evidence.sourceId).toBe('source-a');
    expect(hits[0]!.score).toBe(1);
    expect(memory.query([0, 1], { modelFingerprint: 'v1', excludeEpisodeId: 'now' }).length).toBe(1);
    memory.insert(new Evidence('c', 'C', 'source-c'), [1, 1], { episodeId: 'later' });
    expect(new Set(memory.query([1, 0], { modelFingerprint: 'v1' }).map((hit) => hit.evidence.id))).toEqual(new Set(['b', 'c']));
    memory.remove('c');
    expect(memory.query([1, 0], { modelFingerprint: 'v1' }).length).toBe(1);
  });

  it('keeps conflicts, bad vectors and stale rebuilds atomic', () => {
    const memory = new EpisodicMemory({ modelFingerprint: 'v1' });
    const evidence = new Evidence('a', 'A', 's');
    memory.insert(evidence, [1, 0], { episodeId: 'past' });
    memory.insert(evidence, [1, 0], { episodeId: 'past' });
    expect(memory.size).toBe(1);
    expect(() => memory.insert(new Evidence('a', 'different', 's'), [1, 0], { episodeId: 'past' })).toThrow(ValueError);
    expect(() => memory.insert(new Evidence('b', 'B', 's'), [Number.NaN, 0], { episodeId: 'past' })).toThrow(ValueError);
    expect(() => memory.query([1, 0], { modelFingerprint: 'v2' })).toThrow(ValueError);
    expect(() => memory.rebuildIndex({}, { modelFingerprint: 'v2' })).toThrow(ValueError);
    expect(memory.query([1, 0], { modelFingerprint: 'v1' })[0]!.evidence.equals(evidence)).toBe(true);
    memory.rebuildIndex({ a: [0, 1] }, { modelFingerprint: 'v2' });
    expect(memory.query([0, 1], { modelFingerprint: 'v2' })[0]!.score).toBe(1);
  });

  it('invalid queries and insert fingerprints do not corrupt memory', () => {
    const memory = new EpisodicMemory({ modelFingerprint: 'v1' });
    memory.insert(new Evidence('a', 'A', 's'), [1, 0], { episodeId: 'episode' });
    for (const vector of [[0, 0], [1], [Number.POSITIVE_INFINITY, 0]]) {
      expect(() => memory.query(vector, { modelFingerprint: 'v1' })).toThrow(ValueError);
    }
    expect(() => memory.insert(new Evidence('b', 'B', 's'), [0, 1], { episodeId: 'episode', modelFingerprint: 'v2' })).toThrow(ValueError);
    expect(memory.size).toBe(1);
    expect(memory.query([1, 0], { modelFingerprint: 'v1', k: 0 })).toEqual([]);
  });

  it('matches Python math.fsum rounding', () => {
    expect(fsum([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])).toBe(1);
    expect(fsum([1e100, 1, -1e100, 1e-100, 1e50, -1, -1e50])).toBe(1e-100);
    expect(fsum([])).toBe(0);
  });
});
