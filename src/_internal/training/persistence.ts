/**
 * Versioned JSON artifacts (Python ``tensorcode/_internal/training/persistence.py``).
 * Loading never imports or constructs types named by an artifact.
 *
 * Record codecs are an explicit mapping of stable names to trusted record
 * classes. Tensor payloads use ``dtype``/``shape``/``data``, never executable
 * serialization. Operation bindings are supplied by the caller, and
 * configuration identity is separate from mutable model weights.
 *
 * Files are byte-compatible with Python: the same fields and codec payloads,
 * written like ``json.dumps(data, sort_keys=True, allow_nan=False)``. Float
 * tensors spell integral values as Python floats (``1.0``). JavaScript numbers
 * outside tensors cannot carry the int/float distinction, so integral plain
 * numbers are written as integers.
 */
import { readFile } from 'node:fs/promises';
import { Tensor, tensor as makeTensor, type NestedNumbers } from '../../nn/tensor.js';
import { isFloatingDType, isDType, type DType } from '../../nn/dtype.js';
import { ValueError } from '../../errors.js';
import { atomicWriteFile } from '../files.js';
import { bindingRecords, validateBindings } from '../fingerprint.js';
import { qualifiedName } from '../identity.js';
import {
  PythonFloat, comparePythonStrings, isPlainObject, isPythonNumber, mapKeyKind, orderedEntries, orderedObject, parseJsonStrict,
  pythonJsonDumps, pythonNumber, pythonNumberKind, pythonNumberText, setMapKeyKind, transferPythonNumberKind, unboxNumber,
  type JsonValue,
} from '../json.js';
import { isRecordClass, recordClassOf, recordFields, type RecordClass } from '../records.js';
import { Call, InputRef, OutputRef, Trace, Tree, type Bound, type PathKey } from '../tracing.js';
import type { OperationLike } from '../../ops/base.js';

export interface ExperienceOptions {
  /** Stable names bound to already-constructed operation instances. */
  operations: Record<string, OperationLike>;
  /** Explicit allowlist of trusted record classes keyed by codec name. */
  codecs?: Record<string, RecordClass> | null;
}

// ---------------------------------------------------------------------------
// Python-compatible JSON emission.
// ---------------------------------------------------------------------------

/** A number Python stores as ``float`` (written ``1.0`` when integral); the public ``float()`` marker. */
export { PythonFloat as PyFloat };

/** Pre-rendered JSON text inserted verbatim (for example fingerprinted configurations). */
export class RawJson {
  constructor(readonly text: string) {}
}

const compareCodePoints = comparePythonStrings;

/**
 * ``json.dumps(data, sort_keys=True, allow_nan=False)`` with default
 * separators, honouring {@link float}/{@link int} markers, recorded Python
 * number kinds and {@link RawJson} text.
 */
export function pythonDumps(value: unknown): string {
  const encode = (item: unknown, container: object | null, key: unknown): string => {
    if (item === null) return 'null';
    if (item === true) return 'true';
    if (item === false) return 'false';
    if (typeof item === 'number' || isPythonNumber(item)) {
      if (!Number.isFinite(unboxNumber(item))) throw new ValueError('Out of range float values are not JSON compliant');
      return pythonNumberText(item, container, key);
    }
    if (typeof item === 'string') return pythonJsonDumps(item);
    if (item instanceof RawJson) return item.text;
    if (Array.isArray(item)) return `[${item.map((child, index) => encode(child, item, index)).join(', ')}]`;
    if (isPlainObject(item)) {
      const keys = Object.keys(item).filter((name) => item[name] !== undefined).sort(compareCodePoints);
      return `{${keys.map((name) => `${pythonJsonDumps(name)}: ${encode(item[name], item, name)}`).join(', ')}}`;
    }
    throw new TypeError(`Object of type ${qualifiedName(item)} is not JSON serializable`);
  };
  return encode(value, null, null);
}

/** Python ``sorted()`` order for nested lists of strings and numbers. */
export function pythonCompare(a: unknown, b: unknown): number {
  if (Array.isArray(a) && Array.isArray(b)) {
    const length = Math.min(a.length, b.length);
    for (let index = 0; index < length; index += 1) {
      const order = pythonCompare(a[index], b[index]);
      if (order) return order;
    }
    return a.length - b.length;
  }
  if (typeof a === 'string' && typeof b === 'string') return compareCodePoints(a, b);
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  throw new TypeError('unorderable values');
}

// ---------------------------------------------------------------------------
// Safe value codec.
// ---------------------------------------------------------------------------

const TENSOR_DTYPES: readonly DType[] = ['float16', 'bfloat16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'bool'];

function base64Encode(bytes: Uint8Array): string {
  return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
}

function base64Decode(text: unknown): Uint8Array {
  // Python ``base64.b64decode(..., validate=True)``.
  if (typeof text !== 'string' || text.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(text)) {
    throw new ValueError('Invalid base64-encoded bytes');
  }
  const buffer = Buffer.from(text, 'base64');
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength).slice();
}

function sameKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const own = Object.keys(value);
  return own.length === keys.length && keys.every((key) => Object.prototype.hasOwnProperty.call(value, key));
}

/**
 * Encode and decode trace values: JSON scalars, bytes, dense tensors, tuples
 * (frozen arrays), lists, mappings and allowlisted records.
 */
export class Codec {
  readonly types: Record<string, RecordClass>;

  constructor(codecs: Record<string, RecordClass> | null | undefined = null) {
    this.types = { ...(codecs ?? {}) };
    if (codecs !== null && codecs !== undefined && !isPlainObject(codecs)) {
      throw new TypeError('codecs must map nonempty names to trusted dataclass types');
    }
    for (const [name, cls] of Object.entries(this.types)) {
      if (!name || !isRecordClass(cls)) throw new TypeError('codecs must map nonempty names to trusted dataclass types');
    }
    if (new Set(Object.values(this.types)).size !== Object.keys(this.types).length) {
      throw new ValueError('A dataclass must have exactly one codec name');
    }
  }

  name(cls: RecordClass): string {
    for (const [name, candidate] of Object.entries(this.types)) if (candidate === cls) return name;
    throw new TypeError(`No allowlisted codec for ${qualifiedName(cls)}`);
  }

  encode(value: unknown): unknown {
    return this.encodeAt(value, null, null, false);
  }

  /**
   * Encode ``value`` held at ``container[key]``: numbers keep their explicit
   * or recorded Python kind (``asFloat`` inside float-annotated record fields).
   */
  encodeAt(value: unknown, container: object | null, key: unknown, asFloat = false): unknown {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
    if (typeof value === 'number' || isPythonNumber(value)) {
      if (!Number.isFinite(unboxNumber(value))) throw new ValueError('Nonfinite values cannot be persisted');
      return pythonNumber(value, container, key, asFloat);
    }
    if (value instanceof Uint8Array) return { type: 'bytes', data: base64Encode(value) };
    // Nested tensors dispatch through ``encode`` so subclasses (checkpoint tensor stores) apply.
    if (value instanceof Tensor) return container === null ? this.encodeTensor(value) : this.encode(value);
    if (Array.isArray(value)) {
      return { type: Object.isFrozen(value) ? 'tuple' : 'list', items: value.map((item, index) => this.encodeAt(item, value, index, asFloat)) };
    }
    if (value instanceof Map) {
      const items = [...value.entries()].map(([name, item]) => [
        typeof name === 'number' && mapKeyKind(value, name) === 'float' ? new PythonFloat(name) : this.encodeAt(name, null, null, false),
        this.encodeAt(item, value, name, asFloat),
      ]);
      return { type: 'dict', items };
    }
    const record = recordClassOf(value);
    if (record) {
      const floats = new Set(record.recordFloatFields ?? []);
      const intKeys = new Set(record.recordIntKeyFields ?? []);
      const fields: Record<string, unknown> = {};
      for (const [name, item] of Object.entries(recordFields(value))) {
        const keyed = intKeys.has(name) && isPlainObject(item) ? intKeyedMap(item) : item;
        fields[name] = this.encodeAt(keyed, value as object, name, floats.has(name));
      }
      return { type: 'dataclass', codec: this.name(record), fields };
    }
    if (isPlainObject(value)) {
      return { type: 'dict', items: orderedEntries(value).map(([name, item]) => [name, this.encodeAt(item, value, name, asFloat)]) };
    }
    throw new TypeError(`No safe codec for ${qualifiedName(value)}`);
  }

  protected encodeTensor(value: Tensor): unknown {
    const floating = isFloatingDType(value.dtype);
    const encodeData = (item: NestedNumbers): unknown => {
      if (Array.isArray(item)) return { type: 'list', items: item.map(encodeData) };
      if (typeof item === 'number') {
        if (!Number.isFinite(item)) throw new ValueError('Nonfinite values cannot be persisted');
        return floating ? new PythonFloat(item) : item;
      }
      return item;
    };
    return { type: 'tensor', dtype: value.dtype, shape: [...value.shape], data: encodeData(value.detach().tolist()) };
  }

  decode(value: unknown): unknown {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (!isPlainObject(value)) throw new ValueError('Malformed encoded value');
    const kind = value.type;
    if (kind === 'bytes' && sameKeys(value, ['type', 'data'])) return base64Decode(value.data);
    if ((kind === 'list' || kind === 'tuple') && sameKeys(value, ['type', 'items']) && Array.isArray(value.items)) {
      const source = value.items;
      const items = source.map((item) => this.decode(item));
      source.forEach((_, index) => transferPythonNumberKind(items, index, source, index));
      return kind === 'tuple' ? Object.freeze(items) : items;
    }
    if (kind === 'dict' && sameKeys(value, ['type', 'items'])) {
      const items = value.items;
      if (!Array.isArray(items) || items.some((pair) => !Array.isArray(pair) || pair.length !== 2)) {
        throw new ValueError('Malformed mapping');
      }
      // Python ``dict``: insertion order and key types (``int``, ``bool``,
      // ``None``, tuples, ...) are kept. String-keyed mappings decode to ordered
      // plain objects, any other key type to a ``Map``.
      const pairs = (items as unknown[][]).map((pair) => {
        const key = this.decode(pair[0]);
        return { key, pair, value: this.decode(pair[1]) };
      });
      const seen = new Set<string>();
      for (const { key } of pairs) {
        const identity = pythonKeyIdentity(key);
        if (seen.has(identity)) throw new ValueError('Duplicate mapping key');
        seen.add(identity);
      }
      if (pairs.every(({ key }) => typeof key === 'string')) {
        const result = orderedObject(pairs.map(({ key, value: item }) => [key as string, item] as const));
        for (const { key, pair } of pairs) transferPythonNumberKind(result, key, pair, 1);
        return result;
      }
      const result = new Map<unknown, unknown>(pairs.map(({ key, value: item }) => [key, item]));
      for (const { key, pair } of pairs) {
        transferPythonNumberKind(result, key, pair, 1);
        if (typeof key === 'number' && pythonNumberKind(pair, 0) === 'float') setMapKeyKind(result, key, 'float');
      }
      return result;
    }
    if (kind === 'dataclass' && sameKeys(value, ['type', 'codec', 'fields'])) {
      const codec = value.codec;
      const cls = typeof codec === 'string' && Object.prototype.hasOwnProperty.call(this.types, codec) ? this.types[codec] : undefined;
      if (cls === undefined) throw new ValueError(`Unknown allowlisted codec: ${String(codec)}`);
      const fields = value.fields;
      if (!isPlainObject(fields) || !sameKeys(fields, cls.recordFields)) throw new ValueError('Dataclass fields differ from codec');
      const intKeys = new Set(cls.recordIntKeyFields ?? []);
      const decoded: Record<string, unknown> = {};
      for (const [key, item] of Object.entries(fields)) {
        const field = this.decode(item);
        decoded[key] = intKeys.has(key) && field instanceof Map ? decimalKeyedObject(field) : field;
      }
      const record = cls.fromRecord(decoded);
      // Remember the Python kind of number fields (a record rebuilt from a
      // Python artifact re-encodes ``1.0`` as ``1.0``).
      if (record !== null && typeof record === 'object') {
        for (const key of Object.keys(fields)) if (typeof decoded[key] === 'number') transferPythonNumberKind(record, key, fields, key);
      }
      return record;
    }
    if (kind === 'tensor' && sameKeys(value, ['type', 'dtype', 'shape', 'data'])) {
      const { dtype, shape } = value;
      if (typeof dtype !== 'string' || !isDType(dtype) || !TENSOR_DTYPES.includes(dtype) || !Array.isArray(shape)
        || shape.some((size) => typeof size !== 'number' || !Number.isInteger(size) || size < 0)) {
        throw new ValueError('Unsupported tensor dtype or shape');
      }
      const data = this.decode(value.data);
      return tensorFromPayload(data, dtype, shape as number[]);
    }
    throw new ValueError('Unknown or malformed safe codec payload');
  }
}

/** A Python ``dict`` key's hash identity (``1 == 1.0 == True``); unhashable keys raise like Python. */
function pythonKeyIdentity(key: unknown): string {
  if (typeof key === 'string') return `s${key}`;
  if (typeof key === 'number' || typeof key === 'boolean') return `n${Number(key)}`;
  if (key === null) return 'z';
  if (key instanceof Uint8Array) return `b${base64Encode(key)}`;
  if (Array.isArray(key) && Object.isFrozen(key)) return `t[${key.map(pythonKeyIdentity).join(',')}]`;
  throw new TypeError(`unhashable type: '${Array.isArray(key) ? 'list' : qualifiedName(key)}'`);
}

/** ``Mapping[int, ...]`` held as an object keyed by decimal strings → a ``Map`` with Python int keys. */
function intKeyedMap(value: Record<string, unknown>): Map<unknown, unknown> | Record<string, unknown> {
  const keys = orderedEntries(value);
  if (!keys.every(([key]) => /^-?(?:0|[1-9]\d*)$/.test(key) && Number.isSafeInteger(Number(key)))) return value;
  const result = new Map<unknown, unknown>();
  for (const [key, item] of keys) {
    result.set(Number(key), item);
    transferPythonNumberKind(result, Number(key), value, key);
  }
  return result;
}

/** A decoded ``Map`` with int keys → the decimal-string-keyed object a record field holds. */
function decimalKeyedObject(value: Map<unknown, unknown>): Record<string, unknown> | Map<unknown, unknown> {
  if (![...value.keys()].every((key) => typeof key === 'number' && Number.isInteger(key))) return value;
  const result = orderedObject([...value.entries()].map(([key, item]) => [String(key), item] as const));
  for (const key of value.keys()) transferPythonNumberKind(result, String(key), value, key);
  return result;
}

function tensorFromPayload(data: unknown, dtype: DType, shape: number[]): Tensor {
  const check = (item: unknown): void => {
    if (Array.isArray(item)) item.forEach(check);
    else if (typeof item !== 'number' && typeof item !== 'boolean') throw new ValueError('Tensor data must contain only numbers');
  };
  check(data);
  let result: Tensor;
  try {
    result = makeTensor(data as NestedNumbers, { dtype });
  } catch (error) {
    throw new ValueError('Malformed tensor data', { cause: error });
  }
  const count = shape.reduce((total, size) => total * size, 1);
  // Empty multidimensional tensors need their shape restored explicitly.
  if (result.numel === 0 && count === 0) return result.reshape(shape);
  if (result.shape.length !== shape.length || result.shape.some((size, index) => size !== shape[index])) {
    throw new ValueError('Tensor shape does not match payload');
  }
  return result;
}

// ---------------------------------------------------------------------------
// Artifact files.
// ---------------------------------------------------------------------------

/** Build the entire JSON before touching the destination, then atomically replace it. */
export async function writeArtifact(path: string, data: unknown): Promise<void> {
  const content = pythonDumps(data);
  await atomicWriteFile(path, content);
}

/** Parse an artifact rejecting duplicate keys and nonfinite constants, then check format/version. */
export function parseArtifact(text: string, artifact: string): Record<string, unknown> {
  let data: JsonValue;
  try {
    data = parseJsonStrict(text);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new ValueError(message.startsWith('Duplicate JSON key') ? `Duplicate JSON key (${message})` : `Invalid JSON artifact: ${message}`, { cause: error });
  }
  if (!isPlainObject(data) || data.format !== artifact || data.version !== 1) {
    throw new ValueError('Unknown artifact format or version');
  }
  return data;
}

export async function readArtifact(path: string, artifact: string): Promise<Record<string, unknown>> {
  return parseArtifact(await readFile(path, 'utf8'), artifact);
}

/** Configured bindings as verbatim Python-compatible JSON (float keys honoured). */
export function bindingsJson(operations: Record<string, OperationLike>): RawJson {
  return new RawJson(pythonJsonDumps(bindingRecords(operations), { sortKeys: true, allowNan: false }));
}

// ---------------------------------------------------------------------------
// Experience save/load.
// ---------------------------------------------------------------------------

/** Atomically write ``trace`` as a ``tensorcode.experience`` v1 JSON artifact. */
export async function saveExperience(trace: Trace, path: string, options: ExperienceOptions): Promise<void> {
  if (trace._entered) throw new Error('Close the trace before saving');
  const codec = new Codec(options.codecs ?? null);
  const operations = options.operations;
  if (!isPlainObject(operations)) throw new ValueError('operations must provide nonempty named bindings');
  const names = new Map<OperationLike, string>();
  for (const [name, operation] of Object.entries(operations)) {
    if (names.has(operation)) throw new ValueError('Each operation instance needs one unambiguous binding name');
    names.set(operation, name);
  }
  const bound = (value: Bound): unknown => {
    if (value instanceof InputRef) return { kind: 'input', key: value.key };
    if (value instanceof OutputRef) return { kind: 'output', call: value.call, path: [...value.path] };
    const kind = value.kind === 'dict' || value.kind === 'tuple' || value.kind === 'list' ? value.kind : 'dataclass';
    const children = Array.isArray(value.children)
      ? (value.children as readonly Bound[]).map(bound)
      : Object.fromEntries(Object.entries(value.children as Record<string, Bound>).map(([key, child]) => [key, bound(child)]));
    const result: Record<string, unknown> = { kind, children };
    if (kind === 'dataclass') result.codec = codec.name(value.kind as RecordClass);
    return result;
  };
  const calls: unknown[] = [];
  const used: Record<string, OperationLike> = {};
  trace.calls.forEach((call, index) => {
    if (call.error) throw new ValueError('Cannot persist failed calls');
    if (!trace._released) trace._get(call.output); // Reject silently mutated captured outputs.
    const name = names.get(call.operation);
    if (name === undefined) throw new ValueError('Missing named operation binding for traced call');
    used[name] = call.operation;
    const record: Record<string, unknown> = { operation: name, value: bound(call.value), context: bound(call.context) };
    if (!call.operation.replayable) {
      // Recorded boundaries (for example loaded from Python) keep their Python number kinds.
      record.boundary = trace._boundaries.has(index)
        ? codec.encodeAt(trace._boundaries.get(index), trace._boundaries, index) : codec.encode(call.result);
    }
    calls.push(record);
  });
  const inputs: Record<string, unknown> = {};
  for (const [key, value] of trace.inputs) inputs[String(key)] = codec.encodeAt(value, trace.inputs, key);
  const data = {
    format: 'tensorcode.experience', version: 1, operations: bindingsJson(used), inputs, calls,
    supervisions: trace.supervisions.map((supervision) => ({
      output: bound(supervision.output), target: codec.encodeAt(supervision.target, supervision, 'target'), loss: supervision.loss, source: supervision.source,
    })),
  };
  await writeArtifact(path, data);
}

/**
 * Load an experience bound to explicitly supplied operations and allowlisted
 * record codecs. Weights may differ from capture: only stable configuration is
 * compared. Replay intentionally uses the supplied current parameters, which
 * enables training.
 */
export async function loadExperience(path: string, options: ExperienceOptions): Promise<Trace> {
  const text = await readFile(path, 'utf8');
  try {
    return decodeExperience(text, options);
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) {
      throw new ValueError(`Malformed experience artifact: ${error.message}`, { cause: error });
    }
    throw error;
  }
}

const EXPERIENCE_FIELDS = ['format', 'version', 'operations', 'inputs', 'calls', 'supervisions'];

function decodeExperience(text: string, options: ExperienceOptions): Trace {
  const data = parseArtifact(text, 'tensorcode.experience');
  if (!sameKeys(data, EXPERIENCE_FIELDS)) throw new ValueError('Malformed experience fields');
  const operations = options.operations;
  validateBindings(data.operations, operations);
  const saved = data.operations as Record<string, unknown>;
  const codec = new Codec(options.codecs ?? null);
  const session = new Trace();
  session._closed = true;
  session._released = true;
  if (!isPlainObject(data.inputs)) throw new ValueError('Malformed input keys');
  const inputs = new Map<number, unknown>();
  for (const [key, value] of Object.entries(data.inputs)) {
    if (!/^-?(0|[1-9]\d*)$/.test(key)) throw new ValueError('Malformed input keys');
    const index = Number(key);
    if (index < 0) throw new ValueError('Malformed input keys');
    inputs.set(index, codec.decode(value));
    transferPythonNumberKind(inputs, index, data.inputs, key);
  }
  session.inputs = inputs;
  const bound = (value: unknown, before: number): Bound => {
    if (!isPlainObject(value)) throw new ValueError('Unknown dependency node');
    const kind = value.kind;
    if (kind === 'input' && sameKeys(value, ['kind', 'key'])) {
      const key = value.key;
      if (typeof key !== 'number' || !Number.isInteger(key) || !inputs.has(key)) throw new ValueError('Unknown input root');
      return new InputRef(key);
    }
    if (kind === 'output' && sameKeys(value, ['kind', 'call', 'path'])) {
      const call = value.call;
      const path = value.path;
      if (typeof call !== 'number' || !Number.isInteger(call) || call < 0 || call >= before || !Array.isArray(path)
        || path.some((key) => typeof key !== 'string' && !(typeof key === 'number' && Number.isInteger(key)))) {
        throw new ValueError('Invalid or forward output dependency');
      }
      return new OutputRef(session.id, call, path as PathKey[]);
    }
    if (kind === 'dict' || kind === 'dataclass') {
      const expected = kind === 'dataclass' ? ['kind', 'children', 'codec'] : ['kind', 'children'];
      if (!sameKeys(value, expected) || !isPlainObject(value.children)) throw new ValueError('Malformed tree');
      let cls: RecordClass | 'dict' | undefined = 'dict';
      if (kind === 'dataclass') {
        const name = value.codec;
        cls = typeof name === 'string' && Object.prototype.hasOwnProperty.call(codec.types, name) ? codec.types[name] : undefined;
        if (cls === undefined) throw new ValueError('Unknown allowlisted dataclass codec');
        if (!sameKeys(value.children, cls.recordFields)) throw new ValueError('Malformed dataclass tree fields');
      }
      const children = orderedObject(orderedEntries(value.children).map(([key, child]) => [key, bound(child, before)] as const));
      return new Tree(cls, children);
    }
    if ((kind === 'tuple' || kind === 'list') && sameKeys(value, ['kind', 'children']) && Array.isArray(value.children)) {
      return new Tree(kind, Object.freeze(value.children.map((child) => bound(child, before))));
    }
    throw new ValueError('Unknown dependency node');
  };
  if (!Array.isArray(data.calls) || !Array.isArray(data.supervisions)) throw new ValueError('Malformed calls or supervision');
  data.calls.forEach((record, index) => {
    if (!isPlainObject(record)) throw new ValueError('Malformed call or missing external boundary');
    const name = record.operation;
    if (typeof name !== 'string' || !Object.prototype.hasOwnProperty.call(saved, name)) throw new ValueError('Unvalidated operation binding');
    const operation = operations[name]!;
    const expected = ['operation', 'value', 'context', ...(operation.replayable ? [] : ['boundary'])];
    if (!sameKeys(record, expected)) throw new ValueError('Malformed call or missing external boundary');
    session.calls.push(new Call(operation, bound(record.value, index), bound(record.context, index), new OutputRef(session.id, index)));
    if ('boundary' in record) {
      session._boundaries.set(index, codec.decode(record.boundary));
      transferPythonNumberKind(session._boundaries, index, record, 'boundary');
    }
  });
  for (const record of data.supervisions) {
    if (!isPlainObject(record) || !sameKeys(record, ['output', 'target', 'loss', 'source'])) throw new ValueError('Malformed supervision');
    const ref = bound(record.output, session.calls.length);
    if (!(ref instanceof OutputRef)) throw new ValueError('Supervision must target an output');
    if (typeof record.loss !== 'string' || !record.loss) throw new ValueError('Loss must be a nonempty name; supply custom callbacks to Trainer');
    if (typeof record.source !== 'string') throw new ValueError('Supervision source must be a nonempty explicit provenance string');
    session.supervise(ref, codec.decode(record.target), { loss: record.loss, source: record.source });
    transferPythonNumberKind(session.supervisions[session.supervisions.length - 1]!, 'target', record, 'target');
  }
  return session;
}
