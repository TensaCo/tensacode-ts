/**
 * Immutable symbolic graph values with JSON-compatible extension data (Python
 * ``tensorcode/ops/graph/representation.py``).
 *
 * Graph attribute maps are recursively frozen plain objects (the TypeScript
 * spelling of Python's ``FrozenMap``); lists become frozen arrays (tuples).
 * Assigning to a frozen value throws ``TypeError`` in strict-mode code.
 */
import { ValueError } from '../../errors.js';
import { canonicalJson, isPlainObject, type JsonValue } from '../../_internal/json.js';

/** A recursively immutable JSON value. */
export type FrozenJson = null | string | number | boolean | readonly FrozenJson[] | FrozenMap;

/** A recursively immutable mapping with string keys (Python ``FrozenMap``). */
export type FrozenMap = { readonly [key: string]: FrozenJson };

/** Freeze a mapping with string keys into a recursively immutable {@link FrozenMap}. */
export function FrozenMap(values: Readonly<Record<string, unknown>> | null | undefined = {}): FrozenMap {
  const source = values ?? {};
  if (!isPlainObject(source)) throw new TypeError('Graph attribute maps require string keys');
  const result: Record<string, FrozenJson> = {};
  for (const [key, value] of Object.entries(source)) result[key] = freezeJson(value);
  return Object.freeze(result);
}

function describe(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (typeof value === 'object' && value !== null) return value.constructor?.name ?? 'object';
  return typeof value;
}

/** Copy a JSON value into immutable containers, rejecting opaque objects. */
export function freezeJson(value: unknown): FrozenJson {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new ValueError('Graph attributes require finite JSON numbers');
    return value;
  }
  if (isPlainObject(value)) return FrozenMap(value);
  if (Array.isArray(value)) return Object.freeze(value.map(freezeJson));
  throw new TypeError(`Graph attributes require JSON values, got ${describe(value)}`);
}

/** Return ordinary (mutable) JSON containers for an immutable graph value. */
export function thawJson(value: unknown): JsonValue {
  if (Array.isArray(value)) return value.map(thawJson);
  if (isPlainObject(value)) {
    const result: Record<string, JsonValue> = {};
    for (const [key, item] of Object.entries(value)) result[key] = thawJson(item);
    return result;
  }
  return value as JsonValue;
}

export interface SourceAnchorOptions {
  target?: string | number | null;
  location?: unknown;
  attributes?: Readonly<Record<string, unknown>>;
}

/** A source reference attached to a graph, node ID, or edge index. */
export class SourceAnchor {
  static readonly qualifiedName: string = 'tensorcode.ops.graph.representation.SourceAnchor';
  static readonly recordFields = ['source', 'target', 'location', 'attributes'] as const;
  readonly source: string;
  readonly target: string | number | null;
  readonly location: FrozenJson;
  readonly attributes: FrozenMap;

  constructor(source: string, options: SourceAnchorOptions = {}) {
    if (typeof source !== 'string' || !source) throw new ValueError('A source anchor requires a nonempty source reference');
    const target = options.target ?? null;
    if (target !== null && typeof target !== 'string' && !(typeof target === 'number' && Number.isInteger(target))) {
      throw new TypeError('An anchor target must be a node ID, edge index, or None');
    }
    this.source = source;
    this.target = target;
    this.location = freezeJson(options.location ?? null);
    this.attributes = FrozenMap(options.attributes ?? {});
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): SourceAnchor {
    return new SourceAnchor(fields.source as string, {
      target: fields.target as string | number | null, location: fields.location,
      attributes: fields.attributes as Record<string, unknown>,
    });
  }

  toRecord(): Record<string, unknown> {
    return { source: this.source, target: this.target, location: this.location, attributes: this.attributes };
  }
}

type AlignedAttributes = Readonly<Record<string, Readonly<Record<string, unknown>>>> | readonly Readonly<Record<string, unknown>>[];

function alignedAttributes(values: AlignedAttributes, identities: readonly (string | number)[], name: string): readonly FrozenMap[] {
  if (!Array.isArray(values)) {
    if (!isPlainObject(values)) throw new TypeError(`${name} must be a mapping or a list`);
    const keys = identities.map(String);
    const unknown = Object.keys(values).filter((key) => !keys.includes(key)).sort();
    if (unknown.length) throw new ValueError(`${name} refer to unknown identities: ${JSON.stringify(unknown)}`);
    return Object.freeze(keys.map((key) => FrozenMap((values as Record<string, Record<string, unknown>>)[key] ?? {})));
  }
  if (!values.length) return Object.freeze(identities.map(() => FrozenMap()));
  if (values.length !== identities.length) throw new ValueError(`${name} must align one-to-one with graph entries`);
  return Object.freeze(values.map((item) => FrozenMap(item)));
}

export type Edge = readonly [string, string, string];

export interface GraphOptions {
  edges?: Iterable<readonly [string, string, string]>;
  sources?: Iterable<string>;
  identity?: string | null;
  attributes?: Readonly<Record<string, unknown>>;
  /** Per node, by identity (mapping) or aligned with ``nodes`` (list). */
  nodeAttributes?: AlignedAttributes;
  /** Per edge, by index (mapping with integer keys) or aligned with ``edges`` (list). */
  edgeAttributes?: AlignedAttributes;
  sourceAnchors?: Iterable<SourceAnchor>;
}

/** Immutable labeled graph: nodes, ``[source, relation, target]`` edges, attributes and source anchors. */
export class Graph {
  static readonly qualifiedName: string = 'tensorcode.ops.graph.representation.Graph';
  static readonly recordFields = [
    'nodes', 'edges', 'sources', 'identity', 'attributes', 'node_attributes', 'edge_attributes', 'source_anchors',
  ] as const;
  readonly nodes: readonly string[];
  readonly edges: readonly Edge[];
  readonly sources: readonly string[];
  readonly identity: string | null;
  readonly attributes: FrozenMap;
  readonly nodeAttributes: readonly FrozenMap[];
  readonly edgeAttributes: readonly FrozenMap[];
  readonly sourceAnchors: readonly SourceAnchor[];

  constructor(nodes: Iterable<string>, options: GraphOptions = {}) {
    const nodeList = [...nodes];
    const edges = [...(options.edges ?? [])].map((edge) => [...edge]);
    const sources = [...(options.sources ?? [])];
    const anchors = [...(options.sourceAnchors ?? [])];
    if (!anchors.every((anchor) => anchor instanceof SourceAnchor)) throw new TypeError('source_anchors must contain SourceAnchor values');
    if (![...nodeList, ...sources].every((value) => typeof value === 'string')) {
      throw new TypeError('Node identities and source references must be strings');
    }
    if (nodeList.some((node) => !node)) throw new ValueError('Node identities must be nonempty');
    if (new Set(nodeList).size !== nodeList.length) throw new ValueError('Duplicate node identity');
    if (new Set(sources).size !== sources.length) throw new ValueError('Duplicate source reference');
    const allSources = [...new Set([...sources, ...anchors.map((anchor) => anchor.source)])];
    const identity = options.identity ?? null;
    if (identity !== null && (typeof identity !== 'string' || !identity)) throw new ValueError('Graph identity must be a nonempty string or None');
    for (const edge of edges) {
      if (edge.length !== 3 || !edge.every((value) => typeof value === 'string') || !nodeList.includes(edge[0]!) || !nodeList.includes(edge[2]!)) {
        throw new ValueError('Edges must connect existing nodes');
      }
    }
    for (const anchor of anchors) {
      if (typeof anchor.target === 'string' && !nodeList.includes(anchor.target)) throw new ValueError('Source anchor refers to an unknown node');
      if (typeof anchor.target === 'number' && !(anchor.target >= 0 && anchor.target < edges.length)) {
        throw new ValueError('Source anchor refers to an unknown edge');
      }
    }
    this.nodes = Object.freeze(nodeList);
    this.edges = Object.freeze(edges.map((edge) => Object.freeze(edge) as unknown as Edge));
    this.sources = Object.freeze(allSources);
    this.identity = identity;
    this.attributes = FrozenMap(options.attributes ?? {});
    this.nodeAttributes = alignedAttributes(options.nodeAttributes ?? [], nodeList, 'Node attributes');
    this.edgeAttributes = alignedAttributes(options.edgeAttributes ?? [], edges.map((_, index) => index), 'Edge attributes');
    this.sourceAnchors = Object.freeze(anchors);
    Object.freeze(this);
  }

  /** Targets of edges leaving ``node``, optionally filtered by ``relation``. */
  neighbors(node: string, options: { relation?: string | null } = {}): readonly string[] {
    if (!this.nodes.includes(node)) throw new ValueError('Unknown node identity');
    const relation = options.relation ?? null;
    return Object.freeze(this.edges.filter(([source, label]) => source === node && (relation === null || relation === label)).map((edge) => edge[2]));
  }

  /** Attributes of one node by identity. */
  attributesForNode(node: string): FrozenMap {
    const index = this.nodes.indexOf(node);
    if (index < 0) throw new ValueError('Unknown node identity');
    return this.nodeAttributes[index]!;
  }

  /** Attributes of one edge by index. */
  attributesForEdge(index: number): FrozenMap {
    if (typeof index !== 'number' || !Number.isInteger(index) || index < 0 || index >= this.edgeAttributes.length) {
      throw new ValueError('Unknown edge index');
    }
    return this.edgeAttributes[index]!;
  }

  /** Structural equality (Python dataclass ``==``). */
  equals(other: unknown): boolean {
    return other instanceof Graph && canonicalJson(graphJson(this)) === canonicalJson(graphJson(other));
  }

  static fromRecord(fields: Record<string, unknown>): Graph {
    return new Graph(fields.nodes as readonly string[], {
      edges: fields.edges as readonly Edge[], sources: fields.sources as readonly string[],
      identity: fields.identity as string | null, attributes: fields.attributes as Record<string, unknown>,
      nodeAttributes: fields.node_attributes as readonly Record<string, unknown>[],
      edgeAttributes: fields.edge_attributes as readonly Record<string, unknown>[],
      sourceAnchors: fields.source_anchors as readonly SourceAnchor[],
    });
  }

  toRecord(): Record<string, unknown> {
    return {
      nodes: this.nodes, edges: this.edges, sources: this.sources, identity: this.identity, attributes: this.attributes,
      node_attributes: this.nodeAttributes, edge_attributes: this.edgeAttributes, source_anchors: this.sourceAnchors,
    };
  }
}

function graphJson(graph: Graph): JsonValue {
  return {
    nodes: [...graph.nodes], edges: graph.edges.map((edge) => [...edge]), sources: [...graph.sources], identity: graph.identity,
    attributes: thawJson(graph.attributes), node_attributes: graph.nodeAttributes.map(thawJson),
    edge_attributes: graph.edgeAttributes.map(thawJson),
    source_anchors: graph.sourceAnchors.map((anchor) => ({
      source: anchor.source, target: anchor.target, location: thawJson(anchor.location), attributes: thawJson(anchor.attributes),
    })),
  };
}
