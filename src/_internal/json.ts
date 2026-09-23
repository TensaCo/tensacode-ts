/**
 * JSON utilities shared by configuration, persistence and fingerprints.
 *
 * ``pythonJsonDumps`` reproduces Python's ``json.dumps`` text (separators,
 * ``sort_keys``, ``ensure_ascii`` and float ``repr``) so fingerprints and
 * embedded JSON strings match the Python implementation where JavaScript can
 * represent the data. JavaScript numbers do not distinguish ``1`` from ``1.0``;
 * integral numbers are always written as integers.
 */

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };

export function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/**
 * Validate that ``value`` is finite JSON data (plain objects, arrays, strings,
 * finite numbers, booleans, null) and return an independent deep copy.
 */
export function validatedJson<T = JsonValue>(value: unknown, what = 'value'): T {
  const visit = (item: unknown, path: string): JsonValue => {
    if (item === null || typeof item === 'string' || typeof item === 'boolean') return item;
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) throw new Error(`${what} must contain finite JSON data (${path})`);
      return item;
    }
    if (Array.isArray(item)) return item.map((child, index) => visit(child, `${path}[${index}]`));
    if (isPlainObject(item)) {
      const result: JsonObject = {};
      for (const [key, child] of Object.entries(item)) {
        if (child === undefined) throw new Error(`${what} must use JSON types (${path}.${key} is undefined)`);
        result[key] = visit(child, `${path}.${key}`);
      }
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

export function deepCopy<T>(value: T): T {
  return validatedJson<T>(value);
}

/** Structural equality of JSON data (object key order ignored). */
export function jsonEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((item, index) => jsonEqual(item, b[index]));
  if (isPlainObject(a)) {
    if (!isPlainObject(b)) return false;
    const keys = Object.keys(a);
    if (keys.length !== Object.keys(b).length) return false;
    return keys.every((key) => Object.prototype.hasOwnProperty.call(b, key) && jsonEqual(a[key], b[key]));
  }
  return false;
}

// ---------------------------------------------------------------------------
// Python-compatible serialization.
// ---------------------------------------------------------------------------

/**
 * Object keys whose values Python stores as ``float`` even when integral (for
 * example ``initializer_factor: 1.0`` or ``coordinate_stride: [8.0, 8.0]``).
 * JavaScript numbers cannot carry that distinction, so Python-compatible
 * writers render integral numbers under these keys (including inside arrays)
 * as ``1.0``. Python's strict configuration dataclasses reject ``1`` for some
 * of them, and fingerprints hash the float spelling.
 *
 * The set is static so fingerprints never depend on module import order.
 * Extend it only through {@link registerPythonFloatKeys} at application
 * start-up, before computing any fingerprint.
 */
export const PYTHON_FLOAT_KEYS = new Set<string>([
  // transformers strict float fields
  'initializer_factor', 'initializer_range', 'layer_norm_eps', 'layer_norm_epsilon',
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
]);

/** Declare additional float keys (application start-up only; see {@link PYTHON_FLOAT_KEYS}). */
export function registerPythonFloatKeys(...keys: string[]): void {
  for (const key of keys) PYTHON_FLOAT_KEYS.add(key);
}

export interface DumpsOptions {
  /** Keys whose integral numbers are written as Python floats (default {@link PYTHON_FLOAT_KEYS}). */
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

function pythonNumber(value: number, allowNan: boolean): string {
  if (!Number.isFinite(value)) {
    if (!allowNan) throw new Error('Out of range float values are not JSON compliant');
    return pythonFloatRepr(value);
  }
  if (Number.isInteger(value)) {
    if (Object.is(value, -0)) return '0';
    return Math.abs(value) >= 1e21 ? BigInt(value).toString() : String(value);
  }
  return pythonFloatRepr(value);
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

function compareKeys(a: string, b: string): number {
  // Python sorts by code point; compare UTF-16 while treating surrogates correctly.
  const left = Array.from(a);
  const right = Array.from(b);
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const difference = left[index]!.codePointAt(0)! - right[index]!.codePointAt(0)!;
    if (difference) return difference;
  }
  return left.length - right.length;
}

/** Serialize like Python's ``json.dumps``. */
export function pythonJsonDumps(value: unknown, options: DumpsOptions = {}): string {
  const indent = options.indent ?? null;
  const [itemSeparator, keySeparator] = options.separators ?? (indent === null ? [', ', ': '] : [',', ': ']);
  const ensureAscii = options.ensureAscii ?? true;
  const allowNan = options.allowNan ?? true;
  const sortKeys = options.sortKeys ?? false;
  const floatKeys = options.floatKeys ?? PYTHON_FLOAT_KEYS;
  const encode = (item: unknown, depth: number, asFloat = false): string => {
    if (item === null || item === undefined) return 'null';
    if (item === true) return 'true';
    if (item === false) return 'false';
    if (typeof item === 'number') {
      if (asFloat && Number.isInteger(item)) return pythonFloatRepr(item);
      return pythonNumber(item, allowNan);
    }
    if (typeof item === 'string') return escapeString(item, ensureAscii);
    const newline = indent === null ? '' : `\n${' '.repeat(indent * (depth + 1))}`;
    const closing = indent === null ? '' : `\n${' '.repeat(indent * depth)}`;
    if (Array.isArray(item)) {
      if (!item.length) return '[]';
      return `[${newline}${item.map((child) => encode(child, depth + 1, asFloat)).join(itemSeparator + newline)}${closing}]`;
    }
    if (isPlainObject(item)) {
      let keys = Object.keys(item).filter((key) => item[key] !== undefined);
      if (!keys.length) return '{}';
      if (sortKeys) keys = keys.sort(compareKeys);
      const parts = keys.map((key) => `${escapeString(key, ensureAscii)}${keySeparator}${encode(item[key], depth + 1, floatKeys.has(key))}`);
      return `{${newline}${parts.join(itemSeparator + newline)}${closing}}`;
    }
    throw new TypeError(`Object of type ${describe(item)} is not JSON serializable`);
  };
  return encode(value, 0);
}

/** ``json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)``. */
export function canonicalJson(value: unknown): string {
  return pythonJsonDumps(value, { sortKeys: true, separators: [',', ':'], allowNan: false });
}

// ---------------------------------------------------------------------------
// Strict parsing.
// ---------------------------------------------------------------------------

/**
 * Parse JSON rejecting duplicate object keys and non-standard constants. Plain
 * ``JSON.parse`` silently keeps the last duplicate; persisted artifacts must not.
 */
export function parseJsonStrict(text: string): JsonValue {
  let position = 0;
  const fail = (message: string): never => {
    throw new SyntaxError(`${message} at position ${position}`);
  };
  const whitespace = (): void => {
    while (position < text.length && ' \t\n\r'.includes(text[position]!)) position += 1;
  };
  const parseValue = (): JsonValue => {
    whitespace();
    const char = text[position];
    if (char === '{') return parseObject();
    if (char === '[') return parseArray();
    if (char === '"') return parseString();
    if (char === 't' && text.startsWith('true', position)) { position += 4; return true; }
    if (char === 'f' && text.startsWith('false', position)) { position += 5; return false; }
    if (char === 'n' && text.startsWith('null', position)) { position += 4; return null; }
    return parseNumber();
  };
  const parseNumber = (): number => {
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(position, position + 400));
    if (!match) return fail('Invalid JSON value');
    position += match[0].length;
    const value = Number(match[0]);
    if (!Number.isFinite(value)) fail('Nonfinite JSON number');
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
  const parseArray = (): JsonValue[] => {
    position += 1;
    const result: JsonValue[] = [];
    whitespace();
    if (text[position] === ']') {
      position += 1;
      return result;
    }
    for (;;) {
      result.push(parseValue());
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
    whitespace();
    if (text[position] === '}') {
      position += 1;
      return result;
    }
    for (;;) {
      whitespace();
      if (text[position] !== '"') fail('Expected string key');
      const key = parseString();
      if (Object.prototype.hasOwnProperty.call(result, key)) throw new SyntaxError(`Duplicate JSON key: ${key}`);
      whitespace();
      if (text[position] !== ':') fail('Expected :');
      position += 1;
      const value = parseValue();
      Object.defineProperty(result, key, { value, enumerable: true, writable: true, configurable: true });
      whitespace();
      if (text[position] === ',') {
        position += 1;
        continue;
      }
      if (text[position] === '}') {
        position += 1;
        return result;
      }
      fail('Expected , or }');
    }
  };
  const value = parseValue();
  whitespace();
  if (position !== text.length) fail('Unexpected trailing data');
  return value;
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
    if (options.sortKeys) entries = [...entries].sort(([a], [b]) => comparePythonKeys(a, b));
    return `{${newline}${entries.map(([key, item]) => `${pythonJsonDumps(key, { ensureAscii })}${keySeparator}${encode(item, depth + 1)}`)
      .join(itemSeparator + newline)}${closing}}`;
  };
  return encode(root, 0);
}

function comparePythonKeys(a: string, b: string): number {
  const left = Array.from(a);
  const right = Array.from(b);
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const difference = left[index]!.codePointAt(0)! - right[index]!.codePointAt(0)!;
    if (difference) return difference;
  }
  return left.length - right.length;
}

/** Build a raw node from plain JSON data (numbers are emitted as given: integers as ints). */
export function rawFromValue(value: unknown): RawNode {
  if (value === null || value === true || value === false) return { t: 'l', v: value };
  if (typeof value === 'string') return { t: 's', v: value };
  if (typeof value === 'number') return { t: 'n', raw: Number.isInteger(value) ? String(value) : String(value) };
  if (Array.isArray(value)) return { t: 'a', items: value.map(rawFromValue) };
  if (isPlainObject(value)) return { t: 'o', entries: Object.entries(value).map(([key, item]) => [key, rawFromValue(item)]) };
  throw new TypeError('rawFromValue requires JSON data');
}

/** Convert a raw node to plain JavaScript data (floats become numbers). */
export function rawToValue(node: RawNode): JsonValue {
  switch (node.t) {
    case 'l': return node.v;
    case 's': return node.v;
    case 'n': return Number(node.raw);
    case 'a': return node.items.map(rawToValue);
    default: {
      const result: JsonObject = {};
      for (const [key, value] of node.entries) Object.defineProperty(result, key, { value: rawToValue(value), enumerable: true, writable: true, configurable: true });
      return result;
    }
  }
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
