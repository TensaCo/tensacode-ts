/** Port of ``tests/graph/test_representation.py``. */
import { describe, expect, it } from 'vitest';
import { FrozenMap, Graph, SourceAnchor, thawJson } from '../../src/ops/graph/index.js';
import { ValueError } from '../../src/errors.js';

describe('graph representation', () => {
  it('freezes extensible attributes away from caller mutation', () => {
    const attributes = { tags: ['review'], owner: { name: 'Ada' } };
    const graph = new Graph(['case'], { attributes });
    attributes.tags.push('mutated');
    attributes.owner.name = 'Grace';
    expect(graph.attributes.tags).toEqual(['review']);
    expect((graph.attributes.owner as FrozenMap).name).toBe('Ada');
    expect(() => { (graph.attributes as Record<string, unknown>).new = true; }).toThrow(TypeError);
    expect(() => { ((graph.attributes.tags as unknown) as string[]).push('x'); }).toThrow(TypeError);
  });

  it('equal attribute maps compare equal regardless of input order', () => {
    const first = new Graph(['case'], { attributes: { a: 1, b: 2 } });
    const second = new Graph(['case'], { attributes: { b: 2, a: 1 } });
    expect(first.equals(second)).toBe(true);
    expect(first.equals(new Graph(['case'], { attributes: { a: 2 } }))).toBe(false);
  });

  it('a source anchor contributes its explicit source reference', () => {
    const graph = new Graph(['case'], { sourceAnchors: [new SourceAnchor('document:anchor', { target: 'case' })] });
    expect(graph.sources).toEqual(['document:anchor']);
  });

  it('preserves competing edges and source anchors', () => {
    const graph = new Graph(['ticket', 'open', 'closed'], {
      edges: [['ticket', 'state', 'open'], ['ticket', 'state', 'closed']],
      sourceAnchors: [new SourceAnchor('document:a', { target: 0 })],
    });
    expect(graph.neighbors('ticket', { relation: 'state' })).toEqual(['open', 'closed']);
    expect(graph.sources).toEqual(['document:a']);
    expect(graph.sourceAnchors[0]!.target).toBe(0);
  });

  it('rejects unknown referents', () => {
    expect(() => new Graph(['known'], { edges: [['known', 'mentions', 'missing']] })).toThrow(/existing nodes/);
    expect(() => new Graph(['a'], { sourceAnchors: [new SourceAnchor('doc', { target: 'b' })] })).toThrow(/unknown node/);
    expect(() => new Graph(['a'], { sourceAnchors: [new SourceAnchor('doc', { target: 3 })] })).toThrow(/unknown edge/);
    expect(() => new Graph(['a', 'a'])).toThrow(/Duplicate node/);
    expect(() => new Graph(['a'], { nodeAttributes: { b: {} } })).toThrow(/unknown identities/);
    expect(() => new Graph(['a'], { attributes: { x: Number.NaN } })).toThrow(ValueError);
    expect(() => new Graph(['a'], { attributes: { x: new Date() } })).toThrow(TypeError);
  });

  it.each([-1, 1.5, true])('edge attribute lookup rejects noncanonical index %o', (index) => {
    const graph = new Graph(['a', 'b'], { edges: [['a', 'link', 'b']] });
    expect(() => graph.attributesForEdge(index as number)).toThrow(/edge index/);
  });

  it('aligned node/edge attributes and thawing', () => {
    const graph = new Graph(['a', 'b'], { edges: [['a', 'link', 'b']], nodeAttributes: { b: { weight: 2 } }, edgeAttributes: [{ kind: ['x'] }] });
    expect(graph.attributesForNode('a')).toEqual({});
    expect(graph.attributesForNode('b')).toEqual({ weight: 2 });
    expect(thawJson(graph.attributesForEdge(0))).toEqual({ kind: ['x'] });
    expect(() => graph.attributesForNode('c')).toThrow(/Unknown node/);
    expect(Graph.fromRecord(graph.toRecord()).equals(graph)).toBe(true);
    expect(FrozenMap({ a: [1] })).toEqual({ a: [1] });
    expect(Object.isFrozen(FrozenMap({ a: [1] }).a)).toBe(true);
  });
});
