/**
 * In-memory operation provenance and opt-in replay, separate from autograd.
 *
 * Only captured operation boundaries are observable. Plain scalar dependencies
 * need explicit {@link OutputRef} handles. Inputs are snapshotted; outputs stay
 * live so tensor gradients survive. Portable persistence is opt-in through
 * {@link Trace.save}.
 *
 * Value model (TypeScript equivalents of the Python types the tracer walks):
 *
 * | Python              | TypeScript                                        |
 * |---------------------|---------------------------------------------------|
 * | ``str/int/float/bool/None`` | ``string``/``number``/``boolean``/``null`` |
 * | ``bytes``           | ``Uint8Array`` (content-stamped, copied)          |
 * | ``torch.Tensor``    | ``Tensor`` (version-stamped, cloned snapshots)    |
 * | ``list``            | mutable ``Array``                                 |
 * | ``tuple``           | frozen ``Array`` (``Object.freeze([...])``)       |
 * | ``dict``/Mapping    | plain object (string keys)                        |
 * | dataclass           | record instance (see ``records.ts``)              |
 */
import { Tensor } from '../nn/tensor.js';
import { ValueError } from '../errors.js';
import { ContextVariable } from './context.js';
import {
  PythonFloat, PythonInt, isPlainObject, isPythonNumber, mapKeyKind, orderedEntries, orderedObject, setMapKeyKind,
  transferPythonNumberKind,
} from './json.js';
import { recordClassOf, recordFields, type RecordClass } from './records.js';
import type { Context, OperationLike } from '../ops/base.js';

export type PathKey = string | number;

/** A handle on a captured call output (or a path inside it). */
export class OutputRef {
  static readonly qualifiedName: string = 'tensorcode._internal.tracing.OutputRef';
  readonly session: string;
  readonly call: number;
  readonly path: readonly PathKey[];

  constructor(session: string, call: number, path: readonly PathKey[] = []) {
    this.session = session;
    this.call = call;
    this.path = Object.freeze([...path]);
    Object.freeze(this);
  }

  /** Value-equality key (Python frozen dataclass hashing). */
  get key(): string {
    return `${this.session}:${this.call}:${JSON.stringify(this.path)}`;
  }

  equals(other: unknown): boolean {
    return other instanceof OutputRef && other.key === this.key;
  }

  child(key: PathKey): OutputRef {
    return new OutputRef(this.session, this.call, [...this.path, key]);
  }
}

/** A handle on a snapshotted external root input. */
export class InputRef {
  static readonly qualifiedName: string = 'tensorcode._internal.tracing.InputRef';
  readonly key: number;

  constructor(key: number) {
    this.key = key;
    Object.freeze(this);
  }
}

export type TreeKind = 'dict' | 'list' | 'tuple' | RecordClass;

/** A container whose children carry dependencies. */
export class Tree {
  readonly kind: TreeKind;
  /** Keyed children for dict/record kinds; ordered children for list/tuple. */
  readonly children: Record<string, Bound> | readonly Bound[];

  constructor(kind: TreeKind, children: Record<string, Bound> | readonly Bound[]) {
    this.kind = kind;
    this.children = children;
    Object.freeze(this);
  }
}

export type Bound = OutputRef | InputRef | Tree;

export class Call {
  operation: OperationLike;
  value: Bound;
  context: Bound;
  output: OutputRef;
  result: unknown = null;
  error: string | null = null;
  pending = false;

  constructor(operation: OperationLike, value: Bound, context: Bound, output: OutputRef, options: { pending?: boolean } = {}) {
    this.operation = operation;
    this.value = value;
    this.context = context;
    this.output = output;
    this.pending = options.pending ?? false;
  }
}

/** A dependency closure: external roots, call indices and the target port. */
export class Example {
  readonly inputs: Map<number, unknown>;
  readonly calls: readonly number[];
  readonly target: OutputRef;

  constructor(inputs: Map<number, unknown>, calls: readonly number[], target: OutputRef) {
    this.inputs = inputs;
    this.calls = Object.freeze([...calls]);
    this.target = target;
    Object.freeze(this);
  }
}

/** An explicitly supplied target, never inferred from a model prediction. */
export class Supervision {
  readonly output: OutputRef;
  readonly target: unknown;
  readonly loss: string;
  readonly source: string;

  constructor(output: OutputRef, target: unknown, loss: string, source: string) {
    this.output = output;
    this.target = target;
    this.loss = loss;
    this.source = source;
    Object.freeze(this);
  }
}

// ---------------------------------------------------------------------------
// Value helpers.
// ---------------------------------------------------------------------------

export function isScalar(value: unknown): value is string | number | boolean | null | undefined | PythonFloat | PythonInt {
  return value === null || value === undefined || typeof value === 'string' || typeof value === 'number'
    || typeof value === 'boolean' || typeof value === 'bigint' || isPythonNumber(value);
}

function isBytes(value: unknown): value is Uint8Array {
  return value instanceof Uint8Array;
}

/** True for a frozen array, the TypeScript spelling of a Python tuple. */
export function isTuple(value: unknown): value is readonly unknown[] {
  return Array.isArray(value) && Object.isFrozen(value);
}

export function tuple<T>(items: Iterable<T>): readonly T[] {
  return Object.freeze([...items]);
}

/** Independent copy of an input: tensors are detached clones, containers copied. */
export function snapshot<T>(value: T): T {
  return snapshotValue(value) as T;
}

function snapshotValue(value: unknown): unknown {
  if (value instanceof Tensor) return value.detach().clone();
  if (isBytes(value)) return value.slice();
  if (Array.isArray(value)) {
    const items = value.map(snapshotValue);
    value.forEach((_, index) => transferPythonNumberKind(items, index, value, index));
    return Object.isFrozen(value) ? Object.freeze(items) : items;
  }
  const record = recordClassOf(value);
  if (record) {
    const fields = recordFields(value);
    const copied: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(fields)) copied[key] = snapshotValue(item);
    const result = record.fromRecord(copied);
    if (result !== null && typeof result === 'object') {
      for (const [key, item] of Object.entries(fields)) if (typeof item === 'number') transferPythonNumberKind(result, key, value as object, key, item);
    }
    return result;
  }
  if (isPlainObject(value)) {
    const result = orderedObject(orderedEntries(value).map(([key, item]) => [key, snapshotValue(item)] as const));
    for (const key of Object.keys(value)) transferPythonNumberKind(result, key, value, key);
    return result;
  }
  if (value instanceof Map) {
    // A Python ``dict`` with non-string keys (for example decoded from Python).
    const result = new Map([...value].map(([key, item]) => [key, snapshotValue(item)]));
    for (const key of value.keys()) {
      transferPythonNumberKind(result, key, value, key);
      if (typeof key === 'number') setMapKeyKind(result, key, mapKeyKind(value, key));
    }
    return result;
  }
  return value;
}

const classIds = new WeakMap<object, number>();
let nextClassId = 1;

function classId(cls: object): number {
  let id = classIds.get(cls);
  if (id === undefined) {
    id = nextClassId++;
    classIds.set(cls, id);
  }
  return id;
}

/**
 * A structural mutation stamp. Tensors contribute identity and version counter;
 * containers contribute their structure. Unsupported opaque objects are rejected.
 */
export function stamp(value: unknown): string {
  if (value instanceof Tensor) return `T${value.id}v${value.version}`;
  if (value === undefined) return 'u';
  if (value === null) return 'n';
  if (typeof value === 'string') return `s${JSON.stringify(value)}`;
  if (typeof value === 'number') return `f${Object.is(value, -0) ? '-0' : String(value)}`;
  if (typeof value === 'boolean') return value ? 'b1' : 'b0';
  if (value instanceof PythonFloat) return `pf${Object.is(value.value, -0) ? '-0' : String(value.value)}`;
  if (value instanceof PythonInt) return `pi${value.text}`;
  if (typeof value === 'bigint') return `i${value}`;
  if (isBytes(value)) return `y${Array.from(value, (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
  if (Array.isArray(value)) return `${Object.isFrozen(value) ? '(' : '['}${value.map(stamp).join(',')}${Object.isFrozen(value) ? ')' : ']'}`;
  const record = recordClassOf(value);
  if (record) {
    const fields = recordFields(value);
    return `R${classId(record)}{${Object.entries(fields).map(([key, item]) => `${JSON.stringify(key)}:${stamp(item)}`).join(',')}}`;
  }
  if (isPlainObject(value)) {
    return `{${Object.entries(value).map(([key, item]) => `${JSON.stringify(key)}:${stamp(item)}`).join(',')}}`;
  }
  if (value instanceof Map) return `M{${[...value].map(([key, item]) => `${stamp(key)}:${stamp(item)}`).join(',')}}`;
  const name = (value as { constructor?: { name?: string } })?.constructor?.name ?? typeof value;
  throw new TypeError(`Unsupported trace value ${name}; use tensors, records, arrays or plain objects`);
}

function childEntries(value: unknown): [PathKey, unknown][] | null {
  if (value instanceof Tensor || isBytes(value) || isScalar(value)) return null;
  const record = recordClassOf(value);
  if (record) return Object.entries(recordFields(value));
  if (Array.isArray(value)) return value.map((item, index) => [index, item]);
  if (isPlainObject(value)) return orderedEntries(value);
  // A Python dict with non-string keys: children are addressed by their keys.
  if (value instanceof Map) return [...value].map(([key, item]) => [key as PathKey, item]);
  return null;
}

function emptyToNull(context: unknown): Context | null {
  if (context === null || context === undefined) return null;
  if (isPlainObject(context) && Object.keys(context).length === 0) return null;
  return context as Context;
}

export function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return (typeof value === 'object' || typeof value === 'function') && value !== null
    && typeof (value as { then?: unknown }).then === 'function';
}

function randomId(): string {
  const bytes = new Uint8Array(16);
  const crypto = (globalThis as { crypto?: { getRandomValues?: (array: Uint8Array) => Uint8Array } }).crypto;
  if (crypto?.getRandomValues) crypto.getRandomValues(bytes);
  else for (let index = 0; index < bytes.length; index += 1) bytes[index] = Math.floor(Math.random() * 256);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

// ---------------------------------------------------------------------------
// The trace session.
// ---------------------------------------------------------------------------

const activeTrace = new ContextVariable<Trace>();

/** The trace session active in the current (async) execution context. */
export function activeSession(): Trace | null {
  return activeTrace.get();
}

/** Run ``fn`` with no active trace (replay and fused-transport helpers). */
export function withoutTrace<R>(fn: () => R): R {
  return activeTrace.run(null, fn);
}

interface ObjectRecord {
  value: unknown;
  ref: OutputRef;
  stamp: string;
}

export interface ReplayOptions {
  /** Replacement root inputs keyed by input index. */
  inputs?: Map<number, unknown> | Record<number, unknown> | null;
  /** ``'error'`` (default) rejects external boundaries; ``'recorded'`` reuses their captured results. */
  boundary?: 'error' | 'recorded';
}

export interface SuperviseOptions {
  loss?: string;
  source?: string;
}

export interface SaveOptions {
  /** Stable names bound to the exact captured operation instances. */
  operations: Record<string, OperationLike>;
  /** Explicit allowlist of record classes by stable codec name. */
  codecs?: Record<string, RecordClass> | null;
  release?: boolean;
}

/**
 * In-memory record of operation calls, inputs and explicit supervision.
 *
 * Create with {@link trace} and enter once with {@link Trace.run}. It records
 * local calls; it does not make remote or discrete calls differentiable.
 */
export class Trace {
  static readonly qualifiedName: string = 'tensorcode._internal.tracing.Trace';
  readonly id: string = randomId();
  readonly calls: Call[] = [];
  /** Snapshotted external roots by input index. */
  inputs: Map<number, unknown> = new Map();
  readonly supervisions: Supervision[] = [];
  /** @internal live object lookup (identity → producing refs). */
  readonly _objects = new Map<unknown, ObjectRecord[]>();
  /** @internal mutation stamps by ``OutputRef.key``. */
  readonly _stamps = new Map<string, string>();
  /** @internal detached external-boundary results by call index. */
  _boundaries = new Map<number, unknown>();
  /** @internal */
  _entered = false;
  /** @internal */
  _closed = false;
  /** @internal */
  _released = false;

  get closed(): boolean {
    return this._closed;
  }

  get released(): boolean {
    return this._released;
  }

  /**
   * Enter the session for the duration of ``fn`` (Python ``with trace():``).
   * Async callbacks keep the session active across ``await``; the session
   * closes when the returned promise settles. A session can be entered once.
   */
  run<R>(fn: () => R): R {
    if (this._entered || this._closed) throw new Error('A trace session can be entered only once');
    this._entered = true;
    const finish = (): void => {
      this._entered = false;
      this._closed = true;
    };
    let result: R;
    try {
      result = activeTrace.run(this, fn);
    } catch (error) {
      finish();
      throw error;
    }
    if (isPromiseLike(result)) {
      return Promise.resolve(result).then(
        (value) => { finish(); return value; },
        (error: unknown) => { finish(); throw error; },
      ) as R;
    }
    finish();
    return result;
  }

  /** @internal */
  _check(ref: unknown): asserts ref is OutputRef {
    if (!(ref instanceof OutputRef) || ref.session !== this.id) throw new ValueError('Output reference belongs to another session');
    if (!Number.isInteger(ref.call) || ref.call < 0 || ref.call >= this.calls.length) throw new ValueError('Unknown output reference');
    const call = this.calls[ref.call]!;
    if (call.pending) throw new Error('Call is still pending; await it before using or persisting its output');
    if (call.error) throw new ValueError('Failed call has no usable output');
  }

  /** @internal */
  _get(ref: OutputRef, results?: Map<number, unknown>): unknown {
    this._check(ref);
    let value = results === undefined ? this.calls[ref.call]!.result : results.get(ref.call);
    for (const key of ref.path) {
      const record = recordClassOf(value);
      if (record) {
        if (typeof key !== 'string' || !record.recordFields.includes(key)) throw new ValueError('Output path must name a record field');
        value = recordFields(value)[key];
      } else if (value instanceof Map) {
        value = value.get(key);
      } else if (value !== null && typeof value === 'object') {
        value = (value as Record<PathKey, unknown>)[key];
      } else {
        throw new ValueError('Output path does not address a container');
      }
    }
    if (results === undefined) {
      const saved = this._stamps.get(ref.key);
      if (saved !== undefined && stamp(value) !== saved) {
        throw new ValueError('A traced intermediate was mutated; represent state changes explicitly');
      }
    }
    return value;
  }

  /** The {@link OutputRef} that produced ``value`` (or validate an existing ref). */
  ref(value: unknown): OutputRef {
    if (value instanceof OutputRef) {
      if (this._released) this._check(value);
      else this._get(value);
      return value;
    }
    if (isScalar(value)) throw new ValueError('Scalar lineage requires an explicit call.output reference');
    const matches = this._objects.get(value) ?? [];
    if (matches.length !== 1) throw new ValueError('Unknown or aliased value: use an explicit call.output reference');
    return matches[0]!.ref;
  }

  /** @internal */
  _bind(value: unknown): Bound {
    if (value instanceof OutputRef) {
      this._check(value);
      return value;
    }
    const matches = isScalar(value) ? [] : (this._objects.get(value) ?? []);
    if (matches.length) {
      if (matches.length !== 1) throw new ValueError('Aliased value requires an explicit output reference');
      const match = matches[0]!;
      if (stamp(match.value) !== match.stamp) {
        throw new ValueError('A traced intermediate was mutated; represent state changes explicitly');
      }
      return match.ref;
    }
    if (isPlainObject(value)) {
      // Python binds dict children in insertion order (input roots are numbered in that order).
      const children = orderedObject(orderedEntries(value).map(([key, item]) => [key, this._bind(item)] as const));
      return new Tree('dict', children);
    }
    if (value instanceof Map) {
      // A ``Map`` is a Python dict: traced like one, so its keys must be strings.
      if (![...value.keys()].every((key) => typeof key === 'string')) throw new TypeError('Traced mapping keys must be strings');
      const children = orderedObject([...value].map(([key, item]) => [key as string, this._bind(item)] as const));
      return new Tree('dict', children);
    }
    if (Array.isArray(value)) {
      return new Tree(Object.isFrozen(value) ? 'tuple' : 'list', Object.freeze(value.map((item) => this._bind(item))));
    }
    const record = recordClassOf(value);
    if (record && this._hasProducer(value)) {
      const children: Record<string, Bound> = {};
      for (const [key, item] of Object.entries(recordFields(value))) children[key] = this._bind(item);
      return new Tree(record, children);
    }
    stamp(value); // reject unsupported opaque/mutable objects
    const key = this.inputs.size;
    this.inputs.set(key, snapshot(value));
    return new InputRef(key);
  }

  /** @internal */
  _hasProducer(value: unknown): boolean {
    if (value instanceof OutputRef) return true;
    if (!isScalar(value) && this._objects.has(value)) return true;
    const children = childEntries(value);
    return children !== null && children.some(([, item]) => this._hasProducer(item));
  }

  /** @internal */
  _resolve(bound: Bound, results?: Map<number, unknown>, inputs?: Map<number, unknown>): unknown {
    if (bound instanceof OutputRef) return this._get(bound, results);
    if (bound instanceof InputRef) {
      const roots = inputs ?? this.inputs;
      if (!roots.has(bound.key)) throw new ValueError('Unknown input root');
      return snapshot(roots.get(bound.key));
    }
    if (bound.kind === 'list' || bound.kind === 'tuple') {
      const items = (bound.children as readonly Bound[]).map((child) => this._resolve(child, results, inputs));
      return bound.kind === 'tuple' ? Object.freeze(items) : items;
    }
    const resolved = orderedObject(orderedEntries(bound.children as Record<string, Bound>)
      .map(([key, child]) => [key, this._resolve(child, results, inputs)] as const));
    return bound.kind === 'dict' ? resolved : bound.kind.fromRecord(resolved);
  }

  /** @internal */
  _register(value: unknown, ref: OutputRef): void {
    const valueStamp = stamp(value);
    this._stamps.set(ref.key, valueStamp);
    if (!isScalar(value)) {
      let existing = this._objects.get(value);
      if (!existing) {
        existing = [];
        this._objects.set(value, existing);
      }
      // A child carried through another container retains its original
      // producer. Returning the same object as a new root stays ambiguous.
      if (!ref.path.length || !existing.length) existing.push({ value, ref, stamp: valueStamp });
    }
    const children = childEntries(value);
    if (children === null) return;
    for (const [key, child] of children) this._register(child, ref.child(key));
  }

  /** @internal */
  _capture<I, O>(operation: OperationLike<I, O>, value: I, context: Context | null, forward: (value: I, context: Context | null) => O): O {
    if (this._closed) throw new Error('Cannot capture into a closed trace session');
    const boundValue = this._bind(value);
    const boundContext = this._bind({ ...(context ?? {}) });
    const ref = new OutputRef(this.id, this.calls.length);
    const call = new Call(operation as OperationLike, boundValue, boundContext, ref, { pending: true });
    this.calls.push(call);
    try {
      // Preserve the original tensors/objects and gradient graph on live execution.
      const result = forward(unwrap(value, this) as I, emptyToNull(unwrap(context ?? {}, this)));
      if (isPromiseLike(result)) {
        throw new TypeError('forward returned a promise; implement aforward and use acall for asynchronous work');
      }
      call.result = result;
      this._register(result, ref);
      return result;
    } catch (error) {
      call.error = describeError(error);
      throw error;
    } finally {
      call.pending = false;
    }
  }

  /** @internal */
  async _captureAsync<I, O>(
    operation: OperationLike<I, O>, value: I, context: Context | null,
    forward: (value: I, context: Context | null) => Promise<O>,
  ): Promise<O> {
    if (this._closed) throw new Error('Cannot capture into a closed trace session');
    const boundValue = this._bind(value);
    const boundContext = this._bind({ ...(context ?? {}) });
    const ref = new OutputRef(this.id, this.calls.length);
    const call = new Call(operation as OperationLike, boundValue, boundContext, ref, { pending: true });
    this.calls.push(call);
    try {
      const liveValue = unwrap(value, this) as I;
      const liveContext = unwrap(context ?? {}, this);
      // Awaited execution still consumes the original objects so gradients
      // survive. Their root snapshots must describe those same inputs.
      const beforeValue = stamp(liveValue);
      const beforeContext = stamp(liveContext);
      const result = await forward(liveValue, emptyToNull(liveContext));
      if (stamp(liveValue) !== beforeValue || stamp(liveContext) !== beforeContext) {
        throw new ValueError('Inputs or context mutated during async capture; represent state changes explicitly');
      }
      call.result = result;
      this._register(result, ref);
      return result;
    } catch (error) {
      call.error = describeError(error);
      throw error;
    } finally {
      call.pending = false;
    }
  }

  /** Dependency closure (external roots and required calls) of ``target``. */
  example(target: unknown): Example {
    const ref = this.ref(target);
    const required = new Set<number>();
    const roots = new Set<number>();
    const visiting = new Set<number>();
    const visit = (bound: Bound): void => {
      if (bound instanceof InputRef) {
        roots.add(bound.key);
      } else if (bound instanceof OutputRef) {
        this._check(bound);
        if (visiting.has(bound.call)) throw new ValueError('Cyclic or composite call graph is not replayable');
        if (!required.has(bound.call)) {
          visiting.add(bound.call);
          const call = this.calls[bound.call]!;
          visit(call.value);
          visit(call.context);
          visiting.delete(bound.call);
          required.add(bound.call);
        }
      } else {
        const children = Array.isArray(bound.children) ? bound.children : Object.values(bound.children);
        for (const child of children as Bound[]) visit(child);
      }
    };
    visit(ref);
    const inputs = new Map<number, unknown>();
    for (const key of [...roots].sort((a, b) => a - b)) inputs.set(key, snapshot(this.inputs.get(key)));
    return new Example(inputs, [...required].sort((a, b) => a - b), ref);
  }

  /** Recompute pure operations; recorded external outputs require opt-in. */
  replay(target: unknown, options: ReplayOptions = {}): unknown {
    const boundary = options.boundary ?? 'error';
    if (boundary !== 'error' && boundary !== 'recorded') throw new ValueError("boundary must be 'error' or 'recorded'");
    const example = this.example(target);
    const replacements = toInputMap(options.inputs);
    if (replacements && [...replacements.keys()].some((key) => !example.inputs.has(key))) {
      throw new ValueError('Replacement inputs must name roots of this example');
    }
    for (const index of example.calls) {
      if (!this.calls[index]!.operation.replayable && boundary === 'error') {
        throw new ValueError('Operation has not opted into effect-free replay');
      }
    }
    const roots = new Map(example.inputs);
    if (replacements) for (const [key, value] of replacements) roots.set(key, value);
    const results = new Map<number, unknown>();
    return withoutTrace(() => {
      for (const index of example.calls) {
        const call = this.calls[index]!;
        if (!call.operation.replayable) {
          if (replacements && replacements.size) throw new ValueError('Cannot replace inputs across a recorded external boundary');
          results.set(index, snapshot(this._boundaries.has(index) ? this._boundaries.get(index) : call.result));
          continue;
        }
        const value = this._resolve(call.value, results, roots);
        const context = this._resolve(call.context, results, roots) as Context;
        results.set(index, call.operation.call(value, { context }));
      }
      return this._get(example.target, results);
    });
  }

  /** Record an explicit target for a captured output. */
  supervise(outputOrRef: unknown, target: unknown, options: SuperviseOptions = {}): Supervision {
    const loss = options.loss ?? 'cross_entropy';
    const source = options.source ?? 'human';
    if (typeof source !== 'string' || !source.trim()) {
      throw new ValueError('Supervision source must be a nonempty explicit provenance string');
    }
    if (typeof loss !== 'string' || !loss) {
      throw new ValueError('Loss must be a nonempty name; supply custom callbacks to Trainer');
    }
    const output = this.ref(outputOrRef);
    this.example(output);
    const supervision = new Supervision(output, snapshot(target), loss, source);
    this.supervisions.push(supervision);
    return supervision;
  }

  /**
   * Atomically write a versioned experience JSON artifact (Python
   * ``Trace.save``), then optionally {@link release}.
   */
  async save(path: string, options: SaveOptions): Promise<void> {
    const { saveExperience } = await import('./training/persistence.js');
    await saveExperience(this, path, { operations: options.operations, codecs: options.codecs ?? null });
    if (options.release) this.release();
  }

  /**
   * Drop live outputs/autograd graphs; retain roots, DAG and external
   * boundaries. Keep {@link OutputRef} handles first; object lookup is
   * unavailable afterwards.
   */
  release(): void {
    if (this._entered) throw new Error('Cannot release an active trace session');
    if (!this._released) {
      for (const call of this.calls) if (!call.error) this._get(call.output);
    }
    const boundaries = new Map(this._boundaries);
    this.calls.forEach((call, index) => {
      if (!call.operation.replayable && !call.error && !boundaries.has(index)) boundaries.set(index, snapshot(call.result));
    });
    this._boundaries = boundaries;
    for (const call of this.calls) call.result = null;
    this._objects.clear();
    this._stamps.clear();
    this._released = true;
  }
}

function toInputMap(inputs: ReplayOptions['inputs']): Map<number, unknown> | null {
  if (inputs === null || inputs === undefined) return null;
  if (inputs instanceof Map) return inputs;
  return new Map(Object.entries(inputs).map(([key, value]) => [Number(key), value]));
}

function describeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return `Error: ${String(error)}`;
}

/** Replace output references (and containers holding them) with live values. */
export function unwrap(value: unknown, session: Trace): unknown {
  if (value instanceof OutputRef) return session._get(value);
  if (isPlainObject(value)) {
    let changed = false;
    const resolved = orderedObject(orderedEntries(value).map(([key, item]) => {
      const result = unwrap(item, session);
      if (result !== item) changed = true;
      return [key, result] as const;
    }));
    if (changed) for (const key of Object.keys(value)) transferPythonNumberKind(resolved, key, value, key);
    return changed ? resolved : value;
  }
  if (Array.isArray(value)) {
    const resolved = value.map((item) => unwrap(item, session));
    if (resolved.every((item, index) => item === value[index])) return value;
    return Object.isFrozen(value) ? Object.freeze(resolved) : resolved;
  }
  const record = recordClassOf(value);
  if (record && session._hasProducer(value)) {
    const fields = recordFields(value);
    let changed = false;
    const resolved: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(fields)) {
      resolved[key] = unwrap(item, session);
      if (resolved[key] !== item) changed = true;
    }
    return changed ? record.fromRecord(resolved) : value;
  }
  return value;
}

/** Invoke ``forward`` through the active trace boundary (synchronous). */
export function invoke<I, O>(
  operation: OperationLike<I, O>, value: I, context: Context | null,
  forward: (value: I, context: Context | null) => O,
): O {
  const session = activeTrace.get();
  if (session === null) {
    if (value instanceof OutputRef) throw new ValueError('Output references require their active trace session');
    return forward(value, emptyToNull(context));
  }
  return session._capture(operation, value, context, forward);
}

/** Invoke ``forward`` through the active trace boundary (asynchronous). */
export async function invokeAsync<I, O>(
  operation: OperationLike<I, O>, value: I, context: Context | null,
  forward: (value: I, context: Context | null) => Promise<O>,
): Promise<O> {
  const session = activeTrace.get();
  if (session === null) {
    if (value instanceof OutputRef) throw new ValueError('Output references require their active trace session');
    return forward(value, emptyToNull(context));
  }
  return session._captureAsync(operation, value, context, forward);
}

/** Create a fresh in-memory trace; no persistence or model calls on creation. */
export function trace(): Trace {
  return new Trace();
}
