/**
 * Persistent memory with caller-defined retrieval semantics (Python
 * ``tensorcode/_internal/memory/json.py``).
 *
 * Differences from Python: loading and persisting are asynchronous
 * (``JsonMemory.open(path, ...)``, ``await memory.append(...)``) and
 * transactions take a callback (``await memory.transaction(async (tx) => ...)``)
 * instead of a ``with`` block.
 */
import { readFile } from 'node:fs/promises';
import { ValueError } from '../../errors.js';
import { isPlainObject, jsonEqual, parseJsonStrict, pythonJsonDumps, validatedJson, type JsonValue } from '../json.js';
import { atomicWriteFile, pathExists } from '../files.js';

/** One stored value and its stable source identity. */
export class MemoryRecord<V = unknown> {
  static readonly qualifiedName: string = 'tensorcode._internal.memory.json.MemoryRecord';
  static readonly recordFields = ['source_id', 'kind', 'value', 'metadata'] as const;
  readonly sourceId: string;
  readonly kind: string;
  readonly value: V;
  readonly metadata: Readonly<Record<string, unknown>>;

  constructor(sourceId: string, kind: string, value: V, metadata: Record<string, unknown> = {}) {
    if (typeof sourceId !== 'string' || !sourceId) throw new ValueError('source_id must be a non-empty string');
    if (typeof kind !== 'string' || !kind) throw new ValueError('kind must be a non-empty string');
    if (metadata === null || typeof metadata !== 'object' || Array.isArray(metadata)) throw new TypeError('metadata must be a mapping');
    this.sourceId = sourceId;
    this.kind = kind;
    this.value = value;
    this.metadata = Object.freeze({ ...metadata });
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): MemoryRecord {
    return new MemoryRecord(fields.source_id as string, fields.kind as string, fields.value, (fields.metadata as Record<string, unknown>) ?? {});
  }

  toRecord(): Record<string, unknown> {
    return { source_id: this.sourceId, kind: this.kind, value: this.value, metadata: { ...this.metadata } };
  }

  /** Python dataclass equality. */
  equals(other: unknown): boolean {
    return other instanceof MemoryRecord && other.sourceId === this.sourceId && other.kind === this.kind
      && structurallyEqual(other.value, this.value) && jsonEqual({ ...other.metadata }, { ...this.metadata });
  }
}

function structurallyEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a !== null && typeof a === 'object' && typeof (a as { equals?: unknown }).equals === 'function') {
    return (a as { equals(other: unknown): boolean }).equals(b);
  }
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((item, index) => structurallyEqual(item, b[index]));
  if (a instanceof Uint8Array) return b instanceof Uint8Array && a.length === b.length && a.every((item, index) => item === b[index]);
  if (isPlainObject(a) || (a !== null && typeof a === 'object' && b !== null && typeof b === 'object' && a.constructor === b.constructor)) {
    if (b === null || typeof b !== 'object') return false;
    const keys = Object.keys(a as object);
    return keys.length === Object.keys(b).length
      && keys.every((key) => structurallyEqual((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]));
  }
  return false;
}

/** Explicit operand passed to a caller-supplied retrieval policy. */
export class MemorySearch<V = unknown> {
  static readonly qualifiedName: string = 'tensorcode._internal.memory.json.MemorySearch';
  readonly query: unknown;
  readonly candidates: readonly MemoryRecord<V>[];
  readonly limit: number;

  constructor(query: unknown, candidates: readonly MemoryRecord<V>[], limit: number) {
    this.query = query;
    this.candidates = Object.freeze([...candidates]);
    this.limit = limit;
    Object.freeze(this);
  }
}

export type Retrieve<V = unknown> = (search: MemorySearch<V>) => Iterable<MemoryRecord<V>>;

export interface JsonMemoryOptions<V = unknown> {
  retrieve: Retrieve<V>;
  encodeValue?: ((value: V) => unknown) | null;
  decodeValue?: ((value: unknown) => V) | null;
}

/** A staged set of appends committed atomically by {@link JsonMemory.transaction}. */
export class MemoryTransaction<V = unknown> {
  private readonly memory: JsonMemory<V>;
  /** Staged record snapshots (independent of the committed store). */
  readonly records: MemoryRecord<V>[];
  nextId: number;
  closed = false;

  constructor(memory: JsonMemory<V>) {
    this.memory = memory;
    this.records = memory.committed.map((record) => memory.snapshotRecord(record));
    this.nextId = memory.nextId;
  }

  append(value: V, options: { kind: string; sourceId?: string | null; metadata?: Record<string, unknown> | null }): MemoryRecord<V> {
    if (this.closed) throw new Error('memory transaction is closed');
    let sourceId = options.sourceId ?? null;
    if (sourceId === null) {
      const occupied = new Set(this.records.map((record) => record.sourceId));
      for (;;) {
        sourceId = `memory-${String(this.nextId).padStart(8, '0')}`;
        this.nextId += 1;
        if (!occupied.has(sourceId)) break;
      }
    }
    if (this.records.some((record) => record.sourceId === sourceId)) throw new ValueError(`duplicate memory source_id: ${sourceId}`);
    const record = this.memory.snapshotRecord(new MemoryRecord(sourceId, options.kind, value, options.metadata ?? {}));
    // Validate before exposing a staged record; persistence never executes code.
    this.memory.encodeRecord(record);
    this.records.push(record);
    return this.memory.snapshotRecord(record);
  }
}

/**
 * A deterministic JSON store; relevance is supplied by ``retrieve``.
 *
 * Values must be JSON-compatible unless ``encodeValue`` and ``decodeValue`` are
 * supplied. The codecs are runtime configuration and are never loaded from the
 * data file.
 */
export class JsonMemory<V = unknown> {
  static readonly qualifiedName: string = 'tensorcode._internal.memory.json.JsonMemory';
  static readonly FORMAT = 'tensorcode-memory';
  static readonly VERSION = 1;
  readonly path: string | null = null;
  readonly retrieve: Retrieve<V>;
  readonly encodeValue: (value: V) => unknown;
  readonly decodeValue: (value: unknown) => V;
  private transactionActive = false;
  /** @internal committed records */
  committed: readonly MemoryRecord<V>[] = [];
  /** @internal next automatic id */
  nextId = 1;

  /** An in-memory store; use {@link JsonMemory.open} for a persistent file. */
  constructor(options: JsonMemoryOptions<V>) {
    if (typeof options?.retrieve !== 'function') throw new TypeError('retrieve must be callable');
    const encode = options.encodeValue ?? null;
    const decode = options.decodeValue ?? null;
    if ((encode === null) !== (decode === null)) throw new ValueError('encode_value and decode_value must be supplied together');
    this.retrieve = options.retrieve;
    this.encodeValue = encode ?? ((value: V) => value);
    this.decodeValue = decode ?? ((value: unknown) => value as V);
  }

  /** Open (and load, when it exists) a persistent store at ``path``. */
  static async open<V = unknown>(path: string | null, options: JsonMemoryOptions<V>): Promise<JsonMemory<V>> {
    const memory = new JsonMemory<V>(options);
    if (path !== null && path !== undefined) {
      (memory as { path: string | null }).path = path;
      if (await pathExists(path)) await memory.load();
    }
    return memory;
  }

  /** A store using the public message-sequence JSON codec. */
  static async forMessages(path: string | null, options: { retrieve: Retrieve<readonly import('../../ops/text/messages.js').Message[]> }) {
    const { decodeMessageSequence, encodeMessageSequence } = await import('./messages.js');
    return JsonMemory.open(path, {
      retrieve: options.retrieve,
      encodeValue: encodeMessageSequence as (value: readonly import('../../ops/text/messages.js').Message[]) => unknown,
      decodeValue: decodeMessageSequence,
    });
  }

  get records(): readonly MemoryRecord<V>[] {
    return Object.freeze(this.committed.map((record) => this.snapshotRecord(record)));
  }

  async append(value: V, options: { kind: string; sourceId?: string | null; metadata?: Record<string, unknown> | null }): Promise<MemoryRecord<V>> {
    return this.transaction((transaction) => transaction.append(value, options));
  }

  /** Stage records and commit them atomically when ``fn`` succeeds. */
  async transaction<T>(fn: (transaction: MemoryTransaction<V>) => T | Promise<T>): Promise<T> {
    if (this.transactionActive) throw new Error('nested memory transactions are not supported');
    this.transactionActive = true;
    try {
      const transaction = new MemoryTransaction(this);
      let result: T;
      try {
        result = await fn(transaction);
      } catch (error) {
        transaction.closed = true;
        throw error;
      }
      await this.persist(transaction.records, transaction.nextId);
      this.committed = Object.freeze(transaction.records.map((record) => this.snapshotRecord(record)));
      this.nextId = transaction.nextId;
      transaction.closed = true;
      return result;
    } finally {
      this.transactionActive = false;
    }
  }

  search(query: unknown, options: { limit?: number | null; kinds?: Iterable<string> | null } = {}): readonly MemoryRecord<V>[] {
    const limit = options.limit ?? null;
    if (limit !== null && (typeof limit !== 'number' || !Number.isInteger(limit) || limit < 0)) {
      throw new ValueError('limit must be a non-negative integer');
    }
    const kinds = options.kinds ? new Set(options.kinds) : null;
    const candidates = this.committed.filter((record) => kinds === null || kinds.has(record.kind)).map((record) => this.snapshotRecord(record));
    const resolved = limit === null ? candidates.length : limit;
    const byId = new Map(candidates.map((record) => [record.sourceId, this.snapshotRecord(record)]));
    const selected = [...this.retrieve(new MemorySearch(query, candidates, resolved))];
    for (const record of selected) {
      if (!(record instanceof MemoryRecord)) throw new TypeError('retrieve must return MemoryRecord candidates');
      const expected = byId.get(record.sourceId);
      if (!expected || !expected.equals(record)) throw new ValueError('retrieve returned an item outside the supplied candidates');
    }
    return Object.freeze(selected.slice(0, resolved));
  }

  /** @internal an independent copy through the JSON codec. */
  snapshotRecord(record: MemoryRecord<V>): MemoryRecord<V> {
    let value: JsonValue;
    let metadata: JsonValue;
    try {
      value = validatedJson(this.encodeValue(record.value), 'memory value');
      metadata = validatedJson({ ...record.metadata }, 'memory metadata');
    } catch (error) {
      if (error instanceof TypeError || error instanceof ValueError) throw error;
      throw new TypeError('memory value and metadata must be JSON serializable', { cause: error });
    }
    return new MemoryRecord(record.sourceId, record.kind, this.decodeValue(value), metadata as Record<string, unknown>);
  }

  /** @internal persisted record payload. */
  encodeRecord(record: MemoryRecord<V>): Record<string, JsonValue> {
    try {
      return validatedJson({
        source_id: record.sourceId, kind: record.kind, value: this.encodeValue(record.value), metadata: { ...record.metadata },
      }, 'memory record');
    } catch (error) {
      throw new TypeError('memory value and metadata must be JSON serializable', { cause: error });
    }
  }

  private async persist(records: readonly MemoryRecord<V>[], nextId: number): Promise<void> {
    if (this.path === null) return;
    const payload = {
      format: JsonMemory.FORMAT, version: JsonMemory.VERSION, next_id: nextId,
      records: records.map((record) => this.encodeRecord(record)),
    };
    await atomicWriteFile(this.path, pythonJsonDumps(payload, { allowNan: false, separators: [',', ':'] }));
  }

  private async load(): Promise<void> {
    const payload = parseJsonStrict(await readFile(this.path!, 'utf8'));
    if (!isPlainObject(payload)) throw new ValueError('memory file must contain an object');
    if (payload.format !== JsonMemory.FORMAT || payload.version !== JsonMemory.VERSION) throw new ValueError('unsupported memory file format or version');
    const raw = payload.records;
    const nextId = payload.next_id;
    if (!Array.isArray(raw) || typeof nextId !== 'number' || !Number.isInteger(nextId) || nextId < 1) throw new ValueError('malformed memory file');
    const records = raw.map((item) => {
      if (!isPlainObject(item) || Object.keys(item).length !== 4 || !['source_id', 'kind', 'value', 'metadata'].every((key) => key in item)) {
        throw new ValueError('malformed memory record');
      }
      if (!isPlainObject(item.metadata)) throw new TypeError('metadata must be a mapping');
      return new MemoryRecord(item.source_id as string, item.kind as string, this.decodeValue(item.value), item.metadata as Record<string, unknown>);
    });
    if (new Set(records.map((record) => record.sourceId)).size !== records.length) throw new ValueError('duplicate memory source_id in file');
    this.committed = Object.freeze(records);
    this.nextId = nextId;
  }
}
