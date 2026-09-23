/**
 * JSON utilities shared by configuration, persistence and fingerprints.
 *
 * TensorCode data is interchangeable with Python, whose JSON distinguishes
 * ``1`` from ``1.0`` and whose dictionaries keep insertion order. JavaScript
 * numbers carry neither the int/float distinction nor (for integer-like
 * keys) the key order of plain objects. This module keeps both *losslessly*:
 *
 * - {@link parseJsonStrict} and {@link pythonJsonLoads} record, on each parsed
 *   object and array, the Python number kind of integral values
 *   (``1.0`` versus ``1``, and exact big integers) and the original key order
 *   when JavaScript would reorder it. The metadata lives beside the data (in
 *   a ``WeakMap``), so parsed values are ordinary JavaScript objects.
 * - {@link validatedJson}/{@link deepCopy} and every Python-compatible writer
 *   ({@link pythonJsonDumps}, {@link canonicalJson}, raw re-serialization)
 *   preserve and honour it, so values read from Python artifacts, experience
 *   files and configurations fingerprint and save byte-identically.
 * - Programs mark numbers explicitly with {@link float}/{@link int}
 *   (``float(0)`` is written ``0.0``), and ordered dictionaries with
 *   ``Map`` or {@link orderedObject}.
 * - Integral numbers without recorded or explicit kind follow the schema:
 *   configuration keys that Python stores as floats ({@link PYTHON_FLOAT_KEYS})
 *   are written as floats, everything else as ints.
 */

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

// ---------------------------------------------------------------------------
// Python numbers.
// ---------------------------------------------------------------------------

/** The Python type of a JSON number: ``int`` (``1``) or ``float`` (``1.0``). */
export type PythonNumberKind = 'int' | 'float';

/**
 * A number Python stores as ``float``: ``float(1)`` is written ``1.0`` by every
 * Python-compatible writer and fingerprint. Accepted anywhere JSON data is
 * (configurations, trace values, targets); validated copies store the plain
 * number and remember its kind.
 */
export class PythonFloat {
  static readonly qualifiedName: string = 'builtins.float';
  readonly value: number;

  constructor(value: number) {
    if (typeof value !== 'number') throw new TypeError('float() requires a number');
    this.value = value;
    Object.freeze(this);
  }

  valueOf(): number {
    return this.value;
  }

  toJSON(): number {
    return this.value;
  }

  toString(): string {
    return pythonFloatRepr(this.value);
  }
}

/**
 * A number Python stores as ``int``: written ``1`` even under a float
 * configuration key (a Python caller passing ``hidden_dropout_prob=0``).
 * ``text`` holds the exact decimal digits, so integers beyond 2^53 survive.
 */
export class PythonInt {
  static readonly qualifiedName: string = 'builtins.int';
  readonly value: number;
  readonly text: string;

  constructor(value: number | bigint | string) {
    let big: bigint;
    if (typeof value === 'bigint') big = value;
    else if (typeof value === 'string') {
      if (!/^-?\d+$/.test(value)) throw new TypeError(`int() requires an integer, got ${JSON.stringify(value)}`);
      big = BigInt(value);
    } else {
      if (typeof value !== 'number' || !Number.isInteger(value)) throw new TypeError(`int() requires an integer, got ${String(value)}`);
      big = BigInt(value);
    }
    this.text = big.toString();
    this.value = Number(big);
    Object.freeze(this);
  }

  valueOf(): number {
    return this.value;
  }

  toJSON(): number {
    return this.value;
  }

  toString(): string {
    return this.text;
  }
}

/** Python ``float(value)``: mark a number as a float (``float(0)`` is written ``0.0``). */
export function float(value: number | PythonFloat | PythonInt): PythonFloat {
  return new PythonFloat(typeof value === 'number' ? value : value.value);
}

/** Python ``int(value)``: mark an integral number (or exact ``bigint``/digits) as an int. */
export function int(value: number | bigint | string | PythonFloat | PythonInt): PythonInt {
  if (value instanceof PythonInt) return value;
  return new PythonInt(value instanceof PythonFloat ? value.value : value);
}

/** Whether ``value`` is an explicit {@link PythonFloat}/{@link PythonInt} marker. */
export function isPythonNumber(value: unknown): value is PythonFloat | PythonInt {
  return value instanceof PythonFloat || value instanceof PythonInt;
}

/** The plain JavaScript number of a number or {@link float}/{@link int} marker (other values unchanged). */
export function unboxNumber<T>(value: T): T extends PythonFloat | PythonInt ? number : T {
  return (isPythonNumber(value) ? value.value : value) as T extends PythonFloat | PythonInt ? number : T;
}

// ---------------------------------------------------------------------------
// Number kinds and key order recorded beside JSON containers.
// ---------------------------------------------------------------------------

interface NumberNote {
  readonly value: number;
  readonly kind: PythonNumberKind;
  /** Exact decimal digits of an int beyond 2^53. */
  readonly text?: string;
}

interface ContainerNotes {
  /** Kinds of integral numbers: object key, array index or ``Map`` key → note. */
  numbers?: Map<unknown, NumberNote>;
  /** Kinds of integral number keys of a ``Map``. */
  keys?: Map<number, NumberNote>;
  /** Python insertion order of an object whose JavaScript key order differs. */
  order?: readonly string[];
  /** Caller data (for example ``Retrieve`` items): no configuration float schema inside. */
  plain?: boolean;
}

const NOTES = new WeakMap<object, ContainerNotes>();

function notesOf(container: object, create: true): ContainerNotes;
function notesOf(container: object, create?: false): ContainerNotes | undefined;
function notesOf(container: object, create = false): ContainerNotes | undefined {
  let notes = NOTES.get(container);
  if (!notes && create) {
    notes = {};
    NOTES.set(container, notes);
  }
  return notes;
}

function noteFor(value: number, kind: PythonNumberKind, text?: string): NumberNote {
  return text === undefined ? { value, kind } : { value, kind, text };
}

function setNote(container: object, key: unknown, note: NumberNote | null): void {
  if (note === null) {
    NOTES.get(container)?.numbers?.delete(key);
    return;
  }
  const notes = notesOf(container, true);
  (notes.numbers ??= new Map()).set(key, note);
}

function currentNote(container: object, key: unknown, value: unknown): NumberNote | null {
  if (typeof value !== 'number') return null;
  const note = NOTES.get(container)?.numbers?.get(key);
  return note !== undefined && note.value === value ? note : null;
}

/** The note a child value contributes: an explicit marker, else the source container's record. */
function childNote(container: object | null, key: unknown, value: unknown): NumberNote | null {
  if (value instanceof PythonFloat) return noteFor(value.value, 'float');
  if (value instanceof PythonInt) return noteFor(value.value, 'int', Number.isSafeInteger(value.value) ? undefined : value.text);
  return container === null ? null : currentNote(container, key, value);
}

/**
 * The Python number kind recorded for ``container[key]`` (an object key, array
 * index or ``Map`` key) while it still holds the recorded value, else ``null``
 * (writers then apply the schema default).
 */
export function pythonNumberKind(container: object, key: unknown): PythonNumberKind | null {
  const value = container instanceof Map ? container.get(key) : (container as Record<string, unknown>)[key as string];
  return childNote(container, key, value)?.kind ?? null;
}

/**
 * The Python kind of ``value`` (a number or marker) held at ``container[key]``:
 * explicit for markers, recorded for parsed numbers, else ``null``.
 */
export function pythonKindOf(value: unknown, container: object | null = null, key: unknown = null): PythonNumberKind | null {
  return childNote(container, key, value)?.kind ?? null;
}

/** ``value`` with every {@link float}/{@link int} marker in nested arrays replaced by its number. */
export function unboxNumbers(value: unknown): unknown {
  if (isPythonNumber(value)) return value.value;
  if (Array.isArray(value)) return value.map(unboxNumbers);
  return value;
}

/**
 * Record (or with ``null`` clear) the Python number kind of the number at
 * ``container[key]``. Writers honour it while that value is unchanged.
 */
export function setPythonNumberKind(container: object, key: unknown, kind: PythonNumberKind | null): void {
  const value = container instanceof Map ? container.get(key) : (container as Record<string, unknown>)[key as string];
  const number = unboxNumber(value);
  if (kind === null || typeof number !== 'number') {
    setNote(container, key, null);
    return;
  }
  setNote(container, key, noteFor(number, kind));
}

/**
 * Copy the recorded kind of ``source[sourceKey]`` to ``target[targetKey]``
 * (clearing it when there is none). ``value`` overrides the value read from
 * ``source`` (record instances expose fields through ``toRecord()``).
 */
export function transferPythonNumberKind(target: object, targetKey: unknown, source: object, sourceKey: unknown, ...value: [unknown?]): void {
  const current = value.length ? value[0]
    : source instanceof Map ? source.get(sourceKey) : (source as Record<string, unknown>)[sourceKey as string];
  setNote(target, targetKey, childNote(source, sourceKey, current));
}

/** The Python kind recorded for an integral number key of a ``Map`` (``{1.0: ...}``), else ``null``. */
export function mapKeyKind(map: ReadonlyMap<unknown, unknown>, key: number): PythonNumberKind | null {
  return NOTES.get(map)?.keys?.get(key)?.kind ?? null;
}

/** Record the Python kind of an integral number key of a ``Map`` (``float`` for ``{1.0: ...}``). */
export function setMapKeyKind(map: ReadonlyMap<unknown, unknown>, key: number, kind: PythonNumberKind | null): void {
  if (kind === null) {
    NOTES.get(map)?.keys?.delete(key);
    return;
  }
  (notesOf(map, true).keys ??= new Map()).set(key, noteFor(key, kind));
}

/**
 * Mark ``value`` (an object, array or ``Map``) as caller data that Python
 * writes as given: the configuration float schema ({@link PYTHON_FLOAT_KEYS})
 * does not apply anywhere inside it; recorded and explicit kinds still do.
 * Copies made by {@link validatedJson} keep the mark.
 */
export function markPlainData<T extends object>(value: T): T {
  notesOf(value, true).plain = true;
  return value;
}

function isPlainData(value: object): boolean {
  return NOTES.get(value)?.plain === true;
}

/** Canonical array-index keys, which JavaScript enumerates before other keys in ascending order. */
function isIndexKey(key: string): boolean {
  return /^(?:0|[1-9]\d{0,9})$/.test(key) && Number(key) < 4294967295;
}

/**
 * Keys of ``value`` in Python insertion order: the order recorded when the
 * object was parsed or built with {@link orderedObject}, followed by keys added
 * since (JavaScript itself enumerates integer-like keys first).
 */
export function orderedKeys(value: object): string[] {
  const own = Object.keys(value);
  const order = NOTES.get(value)?.order;
  if (!order) return own;
  const present = new Set(own);
  const result = order.filter((key) => present.has(key));
  if (result.length === own.length) return result;
  const seen = new Set(result);
  for (const key of own) if (!seen.has(key)) result.push(key);
  return result;
}

/** ``[key, value]`` pairs of ``value`` in Python insertion order (see {@link orderedKeys}). */
export function orderedEntries<T>(value: Readonly<Record<string, T>>): [string, T][] {
  return orderedKeys(value).map((key) => [key, value[key]!]);
}

/** Record the Python insertion order of an object's keys (only kept when JavaScript's order differs). */
export function setKeyOrder(value: object, keys: readonly string[]): void {
  const own = Object.keys(value);
  const differs = keys.length !== own.length || keys.some((key, index) => key !== own[index]);
  const notes = NOTES.get(value);
  if (!differs) {
    if (notes) delete notes.order;
    return;
  }
  notesOf(value, true).order = Object.freeze([...keys]);
}

/**
 * A plain object with ``entries`` in Python insertion order, including
 * integer-like keys (``orderedObject([['2', 'b'], ['1', 'a']])`` enumerates
 * ``'2'`` first through {@link orderedKeys} and every Python-compatible writer).
 * A ``Map`` with string keys is accepted as well.
 */
export function orderedObject<T>(entries: Iterable<readonly [string, T]> | ReadonlyMap<string, T>): Record<string, T> {
  const result: Record<string, T> = {};
  const keys: string[] = [];
  const pairs = entries instanceof Map ? entries.entries() : entries;
  for (const [key, value] of pairs as Iterable<readonly [string, T]>) {
    if (typeof key !== 'string') throw new TypeError('orderedObject keys must be strings');
    if (!Object.prototype.hasOwnProperty.call(result, key)) keys.push(key);
    Object.defineProperty(result, key, { value, enumerable: true, writable: true, configurable: true });
  }
  setKeyOrder(result, keys);
  return result;
}

/**
 * A shallow copy of an object (Python ``dict(value)``) keeping its key order
 * and the recorded Python kinds of its number values; values are not copied.
 */
export function shallowCopy<T>(value: Readonly<Record<string, T>>): Record<string, T> {
  const result = orderedObject(orderedEntries(value));
  for (const key of Object.keys(value)) transferPythonNumberKind(result, key, value, key);
  if (isPlainData(value)) markPlainData(result);
  return result;
}

/**
 * Copy recorded number kinds and key order from ``template`` onto the
 * structurally corresponding parts of ``target`` wherever the values are equal
 * (for example a configuration rebuilt from a parsed Python one).
 */
export function graftPythonNumbers(target: unknown, template: unknown): void {
  const visit = (into: unknown, from: unknown): void => {
    if (from instanceof Map) from = stringKeyed(from as Map<unknown, unknown>) === null ? null : validatedJson(from);
    if (Array.isArray(into) && Array.isArray(from)) {
      into.forEach((item, index) => {
        if (index >= from.length) return;
        const note = childNote(from, index, from[index]);
        if (typeof item === 'number' && Number.isInteger(item) && note && note.value === item) setNote(into, index, note);
        else visit(item, from[index]);
      });
      return;
    }
    if (isPlainObject(into) && isPlainObject(from)) {
      for (const key of Object.keys(into)) {
        if (!Object.prototype.hasOwnProperty.call(from, key)) continue;
        const item = into[key];
        const note = childNote(from, key, from[key]);
        if (typeof item === 'number' && Number.isInteger(item) && note && note.value === item) setNote(into, key, note);
        else visit(item, from[key]);
      }
      if (!NOTES.get(into)?.order && NOTES.get(from)?.order) {
        const order = orderedKeys(from).filter((key) => Object.prototype.hasOwnProperty.call(into, key));
        if (order.length === Object.keys(into).length) setKeyOrder(into, order);
      }
    }
  };
  visit(target, template);
}

// ---------------------------------------------------------------------------
// Validation, copies and equality.
// ---------------------------------------------------------------------------

/**
 * Validate that ``value`` is finite JSON data (plain objects, arrays, strings,
 * finite numbers, booleans, null) and return an independent deep copy. A
 * ``Map`` with string keys becomes an ordered plain object and
 * {@link float}/{@link int} markers become numbers; recorded number kinds and
 * key order are carried over to the copy.
 */
export function validatedJson<T = JsonValue>(value: unknown, what = 'value'): T {
  const finite = (number: number, path: string): number => {
    if (!Number.isFinite(number)) throw new Error(`${what} must contain finite JSON data (${path})`);
    return number;
  };
  const visit = (item: unknown, path: string): JsonValue => {
    if (item === null || typeof item === 'string' || typeof item === 'boolean') return item;
    if (typeof item === 'number') return finite(item, path);
    if (isPythonNumber(item)) return finite(item.value, path);
    if (Array.isArray(item)) {
      const result: JsonValue[] = [];
      item.forEach((child, index) => {
        result.push(visit(child, `${path}[${index}]`));
        const note = childNote(item, index, child);
        if (note) setNote(result, index, note);
      });
      if (isPlainData(item)) markPlainData(result);
      return result;
    }
    const map = item instanceof Map;
    if (map || isPlainObject(item)) {
      const keys = map ? [...(item as Map<unknown, unknown>).keys()] : orderedKeys(item as object);
      const result: JsonObject = {};
      for (const key of keys) {
        if (typeof key !== 'string') throw new Error(`${what} must use JSON types (${path} has a ${describe(key)} key)`);
        const child = map ? (item as Map<string, unknown>).get(key) : (item as Record<string, unknown>)[key];
        if (child === undefined) throw new Error(`${what} must use JSON types (${path}.${key} is undefined)`);
        Object.defineProperty(result, key, { value: visit(child, `${path}.${key}`), enumerable: true, writable: true, configurable: true });
        const note = childNote(item as object, key, child);
        if (note) setNote(result, key, note);
      }
      setKeyOrder(result, keys as string[]);
      if (isPlainData(item as object)) markPlainData(result);
      return result;
    }
    throw new Error(`${what} must use JSON types (${path} is ${describe(item)})`);
  };
  return visit(value, '$') as T;
}

export function isJsonValue(value: unknown): value is JsonValue {
  try {
    validatedJson(value);
    return true;
  } catch {
    return false;
  }
}

function describe(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (typeof value === 'object') return value?.constructor?.name ?? 'object';
  return typeof value;
}

/** Deep copy of JSON data, keeping recorded number kinds and key order (see {@link validatedJson}). */
export function deepCopy<T>(value: T): T {
  return validatedJson<T>(value);
}

/**
 * Merge JSON objects left to right like ``{**a, **b}`` (later keys win and
 * keep their first position), keeping recorded number kinds and key order.
 */
export function mergeJson(...objects: readonly Readonly<Record<string, unknown>>[]): JsonObject {
  const result: JsonObject = {};
  const keys: string[] = [];
  for (const object of objects) {
    const copy = validatedJson<JsonObject>(object);
    for (const key of orderedKeys(copy)) {
      if (!Object.prototype.hasOwnProperty.call(result, key)) keys.push(key);
      Object.defineProperty(result, key, { value: copy[key], enumerable: true, writable: true, configurable: true });
      transferPythonNumberKind(result, key, copy, key);
    }
  }
  setKeyOrder(result, keys);
  return result;
}

/** Structural equality of JSON data like Python ``==`` (``1 == 1.0``; object key order ignored). */
export function jsonEqual(a: unknown, b: unknown): boolean {
  a = unboxNumber(a);
  b = unboxNumber(b);
  if (a === b) return true;
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((item, index) => jsonEqual(item, b[index]));
  const left = a instanceof Map ? stringKeyed(a) : a;
  const right = b instanceof Map ? stringKeyed(b) : b;
  if (isPlainObject(left)) {
    if (!isPlainObject(right)) return false;
    const keys = Object.keys(left);
    if (keys.length !== Object.keys(right).length) return false;
    return keys.every((key) => Object.prototype.hasOwnProperty.call(right, key) && jsonEqual(left[key], right[key]));
  }
  return false;
}

function stringKeyed(map: Map<unknown, unknown>): Record<string, unknown> | null {
  const result: Record<string, unknown> = {};
  for (const [key, value] of map) {
    if (typeof key !== 'string') return null;
    Object.defineProperty(result, key, { value, enumerable: true, writable: true, configurable: true });
  }
  return result;
}

// ---------------------------------------------------------------------------
// Python-compatible serialization.
// ---------------------------------------------------------------------------

/**
 * Configuration keys whose values Python stores as ``float`` even when
 * integral (for example ``initializer_factor: 1.0`` or ``coordinate_stride:
 * [8.0, 8.0]``): the schema default for integral numbers that carry no
 * recorded or explicit kind (see the module comment). Python-compatible
 * writers render them (including inside arrays) as ``1.0``; Python's strict
 * configuration dataclasses reject ``1`` for some of them, and fingerprints
 * hash the float spelling. A recorded kind always wins, so a Python caller's
 * ``hidden_dropout_prob=0`` stays ``0``.
 *
 * The set is static so fingerprints never depend on module import order.
 * Extend it only through {@link registerPythonFloatKeys} at application
 * start-up, before computing any fingerprint.
 */
export const PYTHON_FLOAT_KEYS = new Set<string>([
  // transformers strict float fields
  'initializer_factor', 'initializer_range', 'layer_norm_eps', 'layer_norm_epsilon', 'rope_theta',
  // transformers ``float | int`` fields whose defaults are floats
  'attention_dropout', 'attention_probs_dropout_prob', 'classifier_dropout', 'dropout', 'dropout_rate',
  'hidden_dropout_prob', 'logit_scale_init_value', 'pooler_dropout', 'qa_dropout', 'seq_classif_dropout',
  'summary_last_dropout',
  // PyTorch module attributes (``vars(module)``)
  'norm_type', 'p', 'eps',
  // transformers generation settings (``GenerationConfig``)
  'temperature', 'top_p', 'repetition_penalty', 'length_penalty',
  // transformers image processor settings
  'image_mean', 'image_std', 'rescale_factor',
  // provider settings (``timeout=30.0``)
  'timeout',
  // TensorCode configurations
  'min_temperature', 'max_temperature', 'min_support', 'max_contradiction', 'max_unknown',
  'coordinate_stride', 'coordinate_offset',
  // diffusers configurations (UNet2DConditionModel, AutoencoderKL, DDIMScheduler)
  'beta_start', 'beta_end', 'clip_sample_range', 'sample_max_value', 'dynamic_thresholding_ratio',
  'resnet_out_scale_factor', 'scaling_factor', 'shift_factor', 'norm_eps',
  // Idefics3/SmolVLM configurations (Llama text model attributes, legacy generation fields)
  'neftune_noise_alpha', 'diversity_penalty', 'typical_p',
]);

/**
 * Float-typed fields of persisted tool sessions (probabilities, scores,
 * attention weights, NLI distributions), which session writers add to
 * {@link PYTHON_FLOAT_KEYS}.
 */
export const SESSION_FLOAT_FIELDS: readonly string[] = Object.freeze([
  'predicted_score', 'probability', 'score', 'attention', 'relations', 'embeddings',
  'support', 'contradiction', 'unknown',
]);

/** Declare additional float keys (application start-up only; see {@link PYTHON_FLOAT_KEYS}). */
export function registerPythonFloatKeys(...keys: string[]): void {
  for (const key of keys) PYTHON_FLOAT_KEYS.add(key);
}

/** Keys under which parsers record Python ints (so a schema float default cannot respell them). */
function floatContext(key: string | null): boolean {
  return key !== null && (PYTHON_FLOAT_KEYS.has(key) || SESSION_FLOAT_FIELDS.includes(key));
}

export interface DumpsOptions {
  /** Keys whose unmarked integral numbers are written as Python floats (default {@link PYTHON_FLOAT_KEYS}). */
  floatKeys?: ReadonlySet<string>;
  sortKeys?: boolean;
  /** ``[itemSeparator, keySeparator]``; Python default ``[', ', ': ']`` (``[',', ': ']`` with indent). */
  separators?: readonly [string, string];
  ensureAscii?: boolean;
  allowNan?: boolean;
  indent?: number | null;
}

/** Python ``repr(float)`` for a finite, non-integral double. */
export function pythonFloatRepr(value: number): string {
  if (Number.isNaN(value)) return 'NaN';
  if (value === Number.POSITIVE_INFINITY) return 'Infinity';
  if (value === Number.NEGATIVE_INFINITY) return '-Infinity';
  if (value === 0) return Object.is(value, -0) ? '-0.0' : '0.0';
  const exponential = value.toExponential();
  const match = /^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(exponential);
  if (!match) return String(value);
  const [, sign, lead, rest = '', exponentText] = match;
  const digits = (lead! + rest).replace(/0+$/, '') || '0';
  const exponent = Number(exponentText);
  if (exponent >= -4 && exponent < 16) {
    let text: string;
    if (exponent >= 0) {
      const integerDigits = exponent + 1;
      text = digits.length <= integerDigits
        ? `${digits}${'0'.repeat(integerDigits - digits.length)}.0`
        : `${digits.slice(0, integerDigits)}.${digits.slice(integerDigits)}`;
    } else {
      text = `0.${'0'.repeat(-exponent - 1)}${digits}`;
    }
    return sign + text;
  }
  const mantissa = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  const exponentSign = exponent < 0 ? '-' : '+';
  const magnitude = String(Math.abs(exponent)).padStart(2, '0');
  return `${sign}${mantissa}e${exponentSign}${magnitude}`;
}

/** Python ``str(int)`` of an integral double. */
function pythonIntText(value: number): string {
  if (Object.is(value, -0)) return '0';
  return Math.abs(value) >= 1e21 ? BigInt(value).toString() : String(value);
}

/**
 * JSON text of a number the way Python writes it: a recorded or explicit kind
 * first, else ``float`` when ``schemaFloat``, else ``int`` for integral values.
 */
function numberText(value: number, note: NumberNote | null, schemaFloat: boolean, allowNan: boolean): string {
  if (!Number.isFinite(value)) {
    if (!allowNan) throw new Error('Out of range float values are not JSON compliant');
    return pythonFloatRepr(value);
  }
  if (!Number.isInteger(value)) return pythonFloatRepr(value);
  const kind = note?.kind ?? (schemaFloat ? 'float' : 'int');
  if (kind === 'float') return pythonFloatRepr(value);
  return note?.text ?? pythonIntText(value);
}

/**
 * Python JSON text of a number or {@link float}/{@link int} marker held at
 * ``container[key]``: its explicit or recorded kind, else ``float`` when
 * ``schemaFloat``, else ``int`` for integral values.
 */
export function pythonNumberText(
  value: number | PythonFloat | PythonInt, container: object | null = null, key: unknown = null,
  options: { schemaFloat?: boolean; allowNan?: boolean } = {},
): string {
  return numberText(unboxNumber(value), childNote(container, key, value), options.schemaFloat ?? false, options.allowNan ?? true);
}

/**
 * A number ready for a Python-compatible writer: integral values become a
 * {@link PythonFloat} or {@link PythonInt} marker when their explicit or
 * recorded kind (else ``schemaFloat``) says so, so the kind survives moving
 * the value into a new container. Other values are returned as plain numbers.
 */
export function pythonNumber(
  value: number | PythonFloat | PythonInt, container: object | null = null, key: unknown = null, schemaFloat = false,
): number | PythonFloat | PythonInt {
  const number = unboxNumber(value);
  if (!Number.isFinite(number) || !Number.isInteger(number)) return number;
  const note = childNote(container, key, value);
  const kind = note?.kind ?? (schemaFloat ? 'float' : 'int');
  if (kind === 'float') return value instanceof PythonFloat ? value : new PythonFloat(number);
  if (note?.text !== undefined) return new PythonInt(note.text);
  return schemaFloat ? new PythonInt(number) : number;
}

function escapeString(text: string, ensureAscii: boolean): string {
  let result = '"';
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    const char = text[index]!;
    switch (char) {
      case '"': result += '\\"'; continue;
      case '\\': result += '\\\\'; continue;
      case '\n': result += '\\n'; continue;
      case '\r': result += '\\r'; continue;
      case '\t': result += '\\t'; continue;
      case '\b': result += '\\b'; continue;
      case '\f': result += '\\f'; continue;
      default: break;
    }
    if (code < 0x20 || (ensureAscii && code > 0x7e)) {
      result += `\\u${code.toString(16).padStart(4, '0')}`;
    } else {
      result += char;
    }
  }
  return `${result}"`;
}

/** Python ``str`` ordering: by code point (UTF-16 surrogates compared as code points). */
export function comparePythonStrings(a: string, b: string): number {
  const left = Array.from(a);
  const right = Array.from(b);
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const difference = left[index]!.codePointAt(0)! - right[index]!.codePointAt(0)!;
    if (difference) return difference;
  }
  return left.length - right.length;
}

/** Python ``sorted()`` of dictionary keys (strings, or numbers and booleans; mixed kinds are unorderable). */
function comparePythonKeys(a: unknown, b: unknown): number {
  if (typeof a === 'string' && typeof b === 'string') return comparePythonStrings(a, b);
  const numeric = (value: unknown): number | null => (typeof value === 'number' ? value : typeof value === 'boolean' ? Number(value) : null);
  const left = numeric(a);
  const right = numeric(b);
  if (left !== null && right !== null) return left - right;
  throw new TypeError(`'<' not supported between instances of '${pythonTypeName(a)}' and '${pythonTypeName(b)}'`);
}

function pythonTypeName(value: unknown): string {
  if (value === null || value === undefined) return 'NoneType';
  if (typeof value === 'string') return 'str';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'number') return Number.isInteger(value) ? 'int' : 'float';
  if (Array.isArray(value)) return Object.isFrozen(value) ? 'tuple' : 'list';
  return describe(value);
}

/** Python ``json.dumps`` conversion of a ``dict`` key to its JSON object key. */
function mapKeyText(key: unknown, note: NumberNote | null): string {
  if (typeof key === 'string') return key;
  if (key === true) return 'true';
  if (key === false) return 'false';
  if (key === null || key === undefined) return 'null';
  if (typeof key === 'number') return numberText(key, note, false, true);
  if (isPythonNumber(key)) return numberText(key.value, childNote(null, null, key), false, true);
  throw new TypeError(`keys must be str, int, float, bool or None, not ${pythonTypeName(key)}`);
}

/**
 * Serialize like Python's ``json.dumps``. Objects are written in Python
 * insertion order ({@link orderedKeys}); a ``Map`` is a Python ``dict`` (its
 * keys converted as Python does); integral numbers follow their recorded or
 * explicit kind, else {@link DumpsOptions.floatKeys}.
 */
export function pythonJsonDumps(value: unknown, options: DumpsOptions = {}): string {
  const indent = options.indent ?? null;
  const [itemSeparator, keySeparator] = options.separators ?? (indent === null ? [', ', ': '] : [',', ': ']);
  const ensureAscii = options.ensureAscii ?? true;
  const allowNan = options.allowNan ?? true;
  const sortKeys = options.sortKeys ?? false;
  const floatKeys = options.floatKeys ?? PYTHON_FLOAT_KEYS;
  const encode = (item: unknown, depth: number, asFloat: boolean, note: NumberNote | null, plain = false): string => {
    if (item === null || item === undefined) return 'null';
    if (item === true) return 'true';
    if (item === false) return 'false';
    if (typeof item === 'number') return numberText(item, note, asFloat, allowNan);
    if (isPythonNumber(item)) return numberText(item.value, childNote(null, null, item), asFloat, allowNan);
    if (typeof item === 'string') return escapeString(item, ensureAscii);
    const newline = indent === null ? '' : `\n${' '.repeat(indent * (depth + 1))}`;
    const closing = indent === null ? '' : `\n${' '.repeat(indent * depth)}`;
    // Inside caller data (``markPlainData``) the configuration float schema does not apply.
    const inPlain = plain || (typeof item === 'object' && isPlainData(item));
    if (inPlain) asFloat = false;
    if (Array.isArray(item)) {
      if (!item.length) return '[]';
      const parts = item.map((child, index) => encode(child, depth + 1, asFloat, childNote(item, index, child), inPlain));
      return `[${newline}${parts.join(itemSeparator + newline)}${closing}]`;
    }
    let pairs: [string, unknown, NumberNote | null, boolean][];
    if (item instanceof Map) {
      const keyNotes = NOTES.get(item)?.keys;
      let keys = [...item.keys()].filter((key) => item.get(key) !== undefined);
      if (sortKeys) keys = keys.sort(comparePythonKeys);
      pairs = keys.map((key) => {
        const keyNote = typeof key === 'number' ? keyNotes?.get(key) ?? null : null;
        const text = mapKeyText(key, keyNote);
        const child = item.get(key);
        return [text, child, childNote(item, key, child), !inPlain && typeof key === 'string' && floatKeys.has(key)];
      });
    } else if (isPlainObject(item)) {
      let keys = (sortKeys ? Object.keys(item) : orderedKeys(item)).filter((key) => item[key] !== undefined);
      if (sortKeys) keys = keys.sort(comparePythonStrings);
      pairs = keys.map((key) => [key, item[key], childNote(item, key, item[key]), !inPlain && floatKeys.has(key)]);
    } else {
      throw new TypeError(`Object of type ${describe(item)} is not JSON serializable`);
    }
    if (!pairs.length) return '{}';
    const parts = pairs.map(([key, child, childKind, schemaFloat]) => `${escapeString(key, ensureAscii)}${keySeparator}${encode(child, depth + 1, schemaFloat, childKind, inPlain)}`);
    return `{${newline}${parts.join(itemSeparator + newline)}${closing}}`;
  };
  return encode(value, 0, false, null);
}

/** ``json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)``. */
export function canonicalJson(value: unknown): string {
  return pythonJsonDumps(value, { sortKeys: true, separators: [',', ':'], allowNan: false });
}

// ---------------------------------------------------------------------------
// Lossless parsing.
// ---------------------------------------------------------------------------

export interface LoadsOptions {
  /**
   * ``'reject'`` (default for {@link parseJsonStrict}) raises on duplicate
   * object keys; ``'last'`` keeps the last value at the first key's position
   * like Python ``json.loads``.
   */
  duplicates?: 'reject' | 'last';
  /** Accept Python's ``NaN``/``Infinity``/``-Infinity`` constants (``json.loads`` does). */
  constants?: boolean;
}

/** Record, on a parsed container, the Python kind a number lexeme carries (see the module comment). */
function recordLexeme(container: object, key: unknown, value: number, lexeme: string, context: string | null): void {
  if (!Number.isInteger(value)) return;
  if (/[.eE]/.test(lexeme)) {
    setNote(container, key, noteFor(value, 'float'));
  } else if (!Number.isSafeInteger(value)) {
    setNote(container, key, noteFor(value, 'int', BigInt(lexeme).toString()));
  } else if (floatContext(context)) {
    setNote(container, key, noteFor(value, 'int'));
  }
}

function parseJsonText(text: string, options: LoadsOptions): JsonValue {
  const duplicates = options.duplicates ?? 'reject';
  let position = 0;
  let lexeme: string | null = null;
  const fail = (message: string): never => {
    throw new SyntaxError(`${message} at position ${position}`);
  };
  const whitespace = (): void => {
    while (position < text.length && ' \t\n\r'.includes(text[position]!)) position += 1;
  };
  const parseValue = (context: string | null): JsonValue => {
    whitespace();
    lexeme = null;
    const char = text[position];
    if (char === '{') return parseObject();
    if (char === '[') return parseArray(context);
    if (char === '"') return parseString();
    if (char === 't' && text.startsWith('true', position)) { position += 4; return true; }
    if (char === 'f' && text.startsWith('false', position)) { position += 5; return false; }
    if (char === 'n' && text.startsWith('null', position)) { position += 4; return null; }
    if (options.constants) {
      for (const [word, value] of [['NaN', Number.NaN], ['Infinity', Number.POSITIVE_INFINITY], ['-Infinity', Number.NEGATIVE_INFINITY]] as const) {
        if (text.startsWith(word, position)) {
          position += word.length;
          return value;
        }
      }
    }
    return parseNumber();
  };
  const parseNumber = (): number => {
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(position, position + 400));
    if (!match) return fail('Invalid JSON value');
    position += match[0].length;
    const value = Number(match[0]);
    if (!Number.isFinite(value)) {
      if (!options.constants) fail('Nonfinite JSON number');
    }
    lexeme = match[0];
    return value;
  };
  const parseString = (): string => {
    position += 1;
    let result = '';
    for (;;) {
      if (position >= text.length) fail('Unterminated string');
      const char = text[position]!;
      if (char === '"') {
        position += 1;
        return result;
      }
      if (char === '\\') {
        const next = text[position + 1];
        const simple: Record<string, string> = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
        if (next !== undefined && next in simple) {
          result += simple[next];
          position += 2;
        } else if (next === 'u') {
          const hex = text.slice(position + 2, position + 6);
          if (!/^[0-9a-fA-F]{4}$/.test(hex)) fail('Invalid unicode escape');
          result += String.fromCharCode(parseInt(hex, 16));
          position += 6;
        } else {
          fail('Invalid escape');
        }
      } else {
        if (char.charCodeAt(0) < 0x20) fail('Control character in string');
        result += char;
        position += 1;
      }
    }
  };
  const parseArray = (context: string | null): JsonValue[] => {
    position += 1;
    const result: JsonValue[] = [];
    whitespace();
    if (text[position] === ']') {
      position += 1;
      return result;
    }
    for (;;) {
      const value = parseValue(context);
      if (typeof value === 'number' && lexeme !== null) recordLexeme(result, result.length, value, lexeme, context);
      result.push(value);
      whitespace();
      if (text[position] === ',') {
        position += 1;
        continue;
      }
      if (text[position] === ']') {
        position += 1;
        return result;
      }
      fail('Expected , or ]');
    }
  };
  const parseObject = (): JsonObject => {
    position += 1;
    const result: JsonObject = {};
    const keys: string[] = [];
    // JavaScript enumerates index keys first (ascending): Python's order differs
    // once an index key follows a non-index key or a larger index key.
    let reorder = false;
    let sawNonIndex = false;
    let lastIndex = -1;
    whitespace();
    if (text[position] === '}') {
      position += 1;
      return result;
    }
    for (;;) {
      whitespace();
      if (text[position] !== '"') fail('Expected string key');
      const key = parseString();
      const duplicate = Object.prototype.hasOwnProperty.call(result, key);
      if (duplicate && duplicates === 'reject') throw new SyntaxError(`Duplicate JSON key: ${key}`);
      whitespace();
      if (text[position] !== ':') fail('Expected :');
      position += 1;
      const value = parseValue(key);
      Object.defineProperty(result, key, { value, enumerable: true, writable: true, configurable: true });
      if (typeof value === 'number' && lexeme !== null) recordLexeme(result, key, value, lexeme, key);
      else if (duplicate) setNote(result, key, null);
      if (!duplicate) {
        if (isIndexKey(key)) {
          const index = Number(key);
          if (sawNonIndex || index < lastIndex) reorder = true;
          lastIndex = Math.max(lastIndex, index);
        } else {
          sawNonIndex = true;
        }
        keys.push(key);
      }
      whitespace();
      if (text[position] === ',') {
        position += 1;
        continue;
      }
      if (text[position] === '}') {
        position += 1;
        if (reorder) setKeyOrder(result, keys);
        return result;
      }
      fail('Expected , or }');
    }
  };
  const value = parseValue(null);
  whitespace();
  if (position !== text.length) fail('Unexpected trailing data');
  return value;
}

/**
 * Parse JSON rejecting duplicate object keys and non-standard constants. Plain
 * ``JSON.parse`` silently keeps the last duplicate; persisted artifacts must not.
 * Parsing is lossless: Python number kinds and key order are recorded (see the
 * module comment), so re-serializing reproduces Python's bytes.
 */
export function parseJsonStrict(text: string): JsonValue {
  return parseJsonText(text, { duplicates: 'reject', constants: false });
}

/**
 * Python ``json.loads``: last duplicate key wins (at the first key's position),
 * ``NaN``/``Infinity`` accepted; lossless like {@link parseJsonStrict}.
 */
export function pythonJsonLoads(text: string, options: LoadsOptions = {}): JsonValue {
  return parseJsonText(text, { duplicates: options.duplicates ?? 'last', constants: options.constants ?? true });
}

// ---------------------------------------------------------------------------
// SHA-256 (dependency-free, synchronous).
// ---------------------------------------------------------------------------

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** SHA-256 digest bytes of ``data`` (strings are UTF-8 encoded). */
export function sha256Bytes(data: Uint8Array | string): Uint8Array {
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  const length = bytes.length;
  const blocks = Math.ceil((length + 9) / 64);
  const padded = new Uint8Array(blocks * 64);
  padded.set(bytes);
  padded[length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(length / 0x20000000), false);
  view.setUint32(padded.length - 4, (length * 8) >>> 0, false);
  const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const w = new Uint32Array(64);
  for (let block = 0; block < blocks; block += 1) {
    for (let index = 0; index < 16; index += 1) w[index] = view.getUint32(block * 64 + index * 4, false);
    for (let index = 16; index < 64; index += 1) {
      const a = w[index - 15]!;
      const b = w[index - 2]!;
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      w[index] = (w[index - 16]! + s0 + w[index - 7]! + s1) >>> 0;
    }
    let a = h[0]!; let b = h[1]!; let c = h[2]!; let d = h[3]!;
    let e = h[4]!; let f = h[5]!; let g = h[6]!; let k = h[7]!;
    for (let index = 0; index < 64; index += 1) {
      const s1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const choice = (e & f) ^ (~e & g);
      const t1 = (k + s1 + choice + K[index]! + w[index]!) >>> 0;
      const s0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (s0 + majority) >>> 0;
      k = g; g = f; f = e; e = (d + t1) >>> 0;
      d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    h[0] = (h[0]! + a) >>> 0; h[1] = (h[1]! + b) >>> 0; h[2] = (h[2]! + c) >>> 0; h[3] = (h[3]! + d) >>> 0;
    h[4] = (h[4]! + e) >>> 0; h[5] = (h[5]! + f) >>> 0; h[6] = (h[6]! + g) >>> 0; h[7] = (h[7]! + k) >>> 0;
  }
  const out = new Uint8Array(32);
  const outView = new DataView(out.buffer);
  for (let index = 0; index < 8; index += 1) outView.setUint32(index * 4, h[index]!, false);
  return out;
}

export function sha256Hex(data: Uint8Array | string): string {
  return Array.from(sha256Bytes(data), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

// ---------------------------------------------------------------------------
// Lossless re-serialization (Python float/int distinction preserved).
// ---------------------------------------------------------------------------

/** Lossless JSON syntax tree: numbers keep their lexeme (Python int/float distinction). */
export type RawNode =
  | { t: 'o'; entries: [string, RawNode][] }
  | { t: 'a'; items: RawNode[] }
  | { t: 's'; v: string }
  | { t: 'n'; raw: string }
  | { t: 'l'; v: true | false | null };

export function parseJsonRaw(text: string): RawNode {
  let position = 0;
  const fail = (message: string): never => {
    throw new SyntaxError(`${message} at position ${position}`);
  };
  const whitespace = (): void => {
    while (position < text.length && ' \t\n\r'.includes(text[position]!)) position += 1;
  };
  const string = (): string => {
    const start = position;
    position += 1;
    let simple = true;
    while (position < text.length) {
      const char = text[position]!;
      if (char === '"') break;
      if (char === '\\') {
        simple = false;
        position += 2;
        continue;
      }
      position += 1;
    }
    if (position >= text.length) fail('Unterminated string');
    position += 1;
    const slice = text.slice(start, position);
    return simple ? slice.slice(1, -1) : (JSON.parse(slice) as string);
  };
  const value = (): RawNode => {
    whitespace();
    const char = text[position];
    if (char === '{') {
      position += 1;
      const entries: [string, RawNode][] = [];
      const seen = new Set<string>();
      whitespace();
      if (text[position] === '}') { position += 1; return { t: 'o', entries }; }
      for (;;) {
        whitespace();
        if (text[position] !== '"') fail('Expected string key');
        const key = string();
        if (seen.has(key)) throw new SyntaxError(`Duplicate JSON key: ${key}`);
        seen.add(key);
        whitespace();
        if (text[position] !== ':') fail('Expected :');
        position += 1;
        entries.push([key, value()]);
        whitespace();
        if (text[position] === ',') { position += 1; continue; }
        if (text[position] === '}') { position += 1; return { t: 'o', entries }; }
        fail('Expected , or }');
      }
    }
    if (char === '[') {
      position += 1;
      const items: RawNode[] = [];
      whitespace();
      if (text[position] === ']') { position += 1; return { t: 'a', items }; }
      for (;;) {
        items.push(value());
        whitespace();
        if (text[position] === ',') { position += 1; continue; }
        if (text[position] === ']') { position += 1; return { t: 'a', items }; }
        fail('Expected , or ]');
      }
    }
    if (char === '"') return { t: 's', v: string() };
    if (text.startsWith('true', position)) { position += 4; return { t: 'l', v: true }; }
    if (text.startsWith('false', position)) { position += 5; return { t: 'l', v: false }; }
    if (text.startsWith('null', position)) { position += 4; return { t: 'l', v: null }; }
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(position, position + 400));
    if (!match) return fail('Invalid JSON value');
    position += match[0].length;
    return { t: 'n', raw: match[0] };
  };
  const result = value();
  whitespace();
  if (position !== text.length) fail('Unexpected trailing data');
  return result;
}

function rawNumber(raw: string): string {
  if (/[.eE]/.test(raw)) {
    const value = Number(raw);
    if (!Number.isFinite(value)) return value > 0 ? 'Infinity' : '-Infinity';
    return pythonFloatRepr(value);
  }
  return BigInt(raw).toString();
}

/**
 * Re-serialize JSON text exactly as Python's ``json.dumps(json.loads(text), ...)``
 * would: integers stay integers (arbitrary size), floats use Python ``repr``.
 * Used for byte-compatible embedded tokenizer and native configuration JSON.
 */
export function canonicalizeJsonText(
  text: string, options: DumpsOptions & { topLevelOverrides?: Record<string, string> } = {},
): string {
  const root = parseJsonRaw(text);
  if (options.topLevelOverrides && root.t === 'o') {
    for (const [key, replacement] of Object.entries(options.topLevelOverrides)) rawSet(root, key, parseJsonRaw(replacement));
  }
  return emitJsonRaw(root, options);
}

/** Emit a {@link RawNode} like Python ``json.dumps`` of the equivalent parsed data. */
export function emitJsonRaw(root: RawNode, options: DumpsOptions = {}): string {
  const [itemSeparator, keySeparator] = options.separators ?? (options.indent == null ? [', ', ': '] : [',', ': ']);
  const ensureAscii = options.ensureAscii ?? true;
  const indent = options.indent ?? null;
  const encode = (node: RawNode, depth: number): string => {
    switch (node.t) {
      case 'l': return node.v === null ? 'null' : node.v ? 'true' : 'false';
      case 'n': return rawNumber(node.raw);
      case 's': return pythonJsonDumps(node.v, { ensureAscii });
      default: break;
    }
    const newline = indent === null ? '' : `\n${' '.repeat(indent * (depth + 1))}`;
    const closing = indent === null ? '' : `\n${' '.repeat(indent * depth)}`;
    if (node.t === 'a') {
      if (!node.items.length) return '[]';
      return `[${newline}${node.items.map((item) => encode(item, depth + 1)).join(itemSeparator + newline)}${closing}]`;
    }
    if (!node.entries.length) return '{}';
    let entries = node.entries;
    if (options.sortKeys) entries = [...entries].sort(([a], [b]) => comparePythonStrings(a, b));
    return `{${newline}${entries.map(([key, item]) => `${pythonJsonDumps(key, { ensureAscii })}${keySeparator}${encode(item, depth + 1)}`)
      .join(itemSeparator + newline)}${closing}}`;
  };
  return encode(root, 0);
}

/**
 * Build a raw node from JSON data. Integral numbers follow their recorded or
 * explicit Python kind ({@link float}/{@link int}), else they are ints.
 */
export function rawFromValue(value: unknown): RawNode {
  const node = (item: unknown, note: NumberNote | null): RawNode => {
    if (item === null || item === true || item === false) return { t: 'l', v: item };
    if (typeof item === 'string') return { t: 's', v: item };
    if (typeof item === 'number' || isPythonNumber(item)) {
      const number = unboxNumber(item);
      if (!Number.isFinite(number)) throw new TypeError('rawFromValue requires finite numbers');
      return { t: 'n', raw: numberText(number, childNote(null, null, item) ?? note, false, false) };
    }
    if (Array.isArray(item)) return { t: 'a', items: item.map((child, index) => node(child, childNote(item, index, child))) };
    if (isPlainObject(item)) return { t: 'o', entries: orderedKeys(item).map((key) => [key, node(item[key], childNote(item, key, item[key]))]) };
    throw new TypeError('rawFromValue requires JSON data');
  };
  return node(value, null);
}

/** Convert a raw node to plain JavaScript data, recording Python number kinds and key order. */
export function rawToValue(node: RawNode): JsonValue {
  const convert = (item: RawNode, context: string | null): JsonValue => {
    switch (item.t) {
      case 'l': return item.v;
      case 's': return item.v;
      case 'n': return Number(item.raw);
      case 'a': {
        const result: JsonValue[] = [];
        item.items.forEach((child, index) => {
          const value = convert(child, context);
          result.push(value);
          if (child.t === 'n') recordLexeme(result, index, value as number, child.raw, context);
        });
        return result;
      }
      default: {
        const result: JsonObject = {};
        for (const [key, child] of item.entries) {
          const value = convert(child, key);
          Object.defineProperty(result, key, { value, enumerable: true, writable: true, configurable: true });
          if (child.t === 'n') recordLexeme(result, key, value as number, child.raw, key);
        }
        setKeyOrder(result, item.entries.map(([key]) => key));
        return result;
      }
    }
  };
  return convert(node, null);
}

export function rawGet(node: RawNode | undefined, key: string): RawNode | undefined {
  return node?.t === 'o' ? node.entries.find(([name]) => name === key)?.[1] : undefined;
}

export function rawSet(node: RawNode, key: string, value: RawNode): void {
  if (node.t !== 'o') throw new TypeError('rawSet requires an object node');
  const entry = node.entries.find(([name]) => name === key);
  if (entry) entry[1] = value;
  else node.entries.push([key, value]);
}

export function rawDelete(node: RawNode, key: string): void {
  if (node.t !== 'o') throw new TypeError('rawDelete requires an object node');
  const index = node.entries.findIndex(([name]) => name === key);
  if (index >= 0) node.entries.splice(index, 1);
}

export function rawString(node: RawNode | undefined): string | undefined {
  return node?.t === 's' ? node.v : undefined;
}
