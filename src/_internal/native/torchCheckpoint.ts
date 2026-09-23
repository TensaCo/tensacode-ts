/**
 * PyTorch checkpoint files (``pytorch_model.bin``), read the way transformers
 * reads them: ``torch.load(path, map_location='cpu', weights_only=True)``.
 *
 * Both of ``torch.save``'s formats are supported: the zip archive written
 * since PyTorch 1.6 (``<archive>/data.pkl`` plus one raw file per storage)
 * and the legacy stream format (magic number, protocol, system info, the
 * pickled object, storage keys, then the storages). The pickle is interpreted
 * by a restricted machine that, like PyTorch's weights-only unpickler, only
 * builds containers, primitive values and tensors: any other global or opcode
 * is rejected, so a checkpoint can never run code.
 */
import { decodeTensorBytes } from '../../nn/safetensors.js';
import { allocate, itemSize, type DType } from '../../nn/dtype.js';
import { Tensor, fromStorage } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { inflateRaw } from '../image/inflate.js';

/** ``torch.<Name>Storage`` classes and their element types. */
const STORAGE_TYPES: Readonly<Record<string, DType>> = Object.freeze({
  DoubleStorage: 'float64', FloatStorage: 'float32', HalfStorage: 'float16', BFloat16Storage: 'bfloat16',
  LongStorage: 'int64', IntStorage: 'int32', ShortStorage: 'int16', CharStorage: 'int8', ByteStorage: 'uint8',
  BoolStorage: 'bool', UntypedStorage: 'uint8',
});

/** ``torch`` dtype globals (``torch.float32`` ...) that newer pickles may reference. */
const TORCH_DTYPES: Readonly<Record<string, DType>> = Object.freeze({
  float64: 'float64', double: 'float64', float32: 'float32', float: 'float32', float16: 'float16', half: 'float16',
  bfloat16: 'bfloat16', int64: 'int64', long: 'int64', int32: 'int32', int: 'int32', int16: 'int16', short: 'int16',
  int8: 'int8', uint8: 'uint8', bool: 'bool',
});

const LEGACY_MAGIC = 0x1950a86a20f9469cfc6cn;
const LEGACY_PROTOCOL = 1001;

class WeightsOnlyError extends ValueError {
  constructor(detail: string) {
    super(`Weights only load failed. ${detail}`);
  }
}

class Global {
  constructor(readonly module: string, readonly name: string) {}
  get qualified(): string {
    return `${this.module}.${this.name}`;
  }
}

/** A Python tuple (frozen array). */
type Tuple = readonly unknown[];

/** A storage referenced by a persistent id; bytes are read on first use. */
class Storage {
  private decoded: Tensor | null = null;
  constructor(readonly dtype: DType, readonly numel: number, private readonly read: () => Uint8Array) {}

  /** All elements as a flat tensor. */
  values(): Tensor {
    if (this.decoded === null) {
      const bytes = this.read();
      if (bytes.byteLength < this.numel * itemSize(this.dtype)) throw new ValueError('PyTorch checkpoint storage is truncated');
      this.decoded = decodeTensorBytes(this.dtype, bytes.subarray(0, this.numel * itemSize(this.dtype)), [this.numel]);
    }
    return this.decoded;
  }
}

/**
 * A tensor rebuilt once its storage bytes are available (the legacy format
 * stores them after the pickle).
 */
class LazyTensor {
  constructor(readonly build: () => Tensor) {}
}

/** ``torch._utils._rebuild_tensor_v2``: a contiguous copy of a strided view of ``storage``. */
function rebuildTensor(storage: unknown, offset: unknown, size: unknown, stride: unknown): LazyTensor {
  if (!(storage instanceof Storage)) throw new WeightsOnlyError('tensor data must come from a storage');
  const shape = toInts(size, 'size');
  const strides = toInts(stride, 'stride');
  const start = toInt(offset, 'storage_offset');
  if (shape.length !== strides.length) throw new WeightsOnlyError('tensor size and stride lengths differ');
  return new LazyTensor(() => materialize(storage, start, shape, strides));
}

function materialize(storage: Storage, start: number, shape: number[], strides: number[]): Tensor {
  const source = storage.values().data;
  const count = shape.reduce((product, value) => product * value, 1);
  const out = allocate(storage.dtype, count);
  if (count > 0) {
    const index = new Array<number>(shape.length).fill(0);
    let position = start;
    const last = source.length;
    for (let flat = 0; flat < count; flat += 1) {
      if (position < 0 || position >= last) throw new WeightsOnlyError('tensor view exceeds its storage');
      out[flat] = source[position]!;
      for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
        index[axis] += 1;
        position += strides[axis]!;
        if (index[axis]! < shape[axis]!) break;
        position -= strides[axis]! * shape[axis]!;
        index[axis] = 0;
      }
    }
  }
  return fromStorage(out, shape, storage.dtype);
}

function toInt(value: unknown, what: string): number {
  if (typeof value === 'bigint') value = Number(value);
  if (typeof value !== 'number' || !Number.isSafeInteger(value)) throw new WeightsOnlyError(`${what} must be an integer`);
  return value;
}

function toInts(value: unknown, what: string): number[] {
  if (!Array.isArray(value)) throw new WeightsOnlyError(`${what} must be a tuple of integers`);
  return value.map((item) => toInt(item, what));
}

/** Calls a weights-only pickle may make (PyTorch's ``_get_allowed_globals`` subset for state dicts). */
function callGlobal(target: Global, args: Tuple): unknown {
  switch (target.qualified) {
    case 'collections.OrderedDict':
      return new Map<unknown, unknown>(args.length ? toPairs(args[0]) : []);
    case 'torch._utils._rebuild_tensor_v2':
    case 'torch._utils._rebuild_tensor':
      return rebuildTensor(args[0], args[1], args[2], args[3]);
    case 'torch._utils._rebuild_parameter':
    case 'torch._utils._rebuild_parameter_with_state':
      if (!(args[0] instanceof LazyTensor)) throw new WeightsOnlyError('parameter data must be a tensor');
      return args[0];
    case 'torch._tensor._rebuild_from_type_v2': {
      const [fn, , inner] = args as [unknown, unknown, unknown];
      if (!(fn instanceof Global) || !Array.isArray(inner)) throw new WeightsOnlyError('invalid tensor subclass rebuild');
      return callGlobal(fn, inner);
    }
    case '_codecs.encode': {
      const [text, encoding] = args as [unknown, unknown];
      if (typeof text !== 'string' || (encoding !== 'latin1' && encoding !== 'latin-1')) throw new WeightsOnlyError('unsupported _codecs.encode call');
      return Uint8Array.from(text, (char) => char.charCodeAt(0) & 0xff);
    }
    case 'builtins.set':
    case 'builtins.frozenset':
      return new Set(args.length ? (args[0] as unknown[]) : []);
    case 'torch.Size':
      return Object.freeze(args.length ? [...(args[0] as unknown[])] : []);
    default:
      throw new WeightsOnlyError(`Unsupported global: GLOBAL ${target.qualified} was not an allowed global by default.`);
  }
}

function toPairs(value: unknown): [unknown, unknown][] {
  if (value instanceof Map) return [...value];
  if (!Array.isArray(value)) throw new WeightsOnlyError('OrderedDict items must be pairs');
  return value.map((pair) => {
    if (!Array.isArray(pair) || pair.length !== 2) throw new WeightsOnlyError('OrderedDict items must be pairs');
    return [pair[0], pair[1]] as [unknown, unknown];
  });
}

const ALLOWED_GLOBALS = new Set([
  'collections.OrderedDict', 'torch._utils._rebuild_tensor_v2', 'torch._utils._rebuild_tensor', 'torch._utils._rebuild_parameter',
  'torch._utils._rebuild_parameter_with_state', 'torch._tensor._rebuild_from_type_v2', '_codecs.encode', 'builtins.set',
  'builtins.frozenset', 'torch.Size', 'torch.Tensor', 'torch.nn.parameter.Parameter',
]);

function resolveGlobal(module: string, name: string): Global {
  const target = new Global(module, name);
  if (module === 'torch' && (name in STORAGE_TYPES || name in TORCH_DTYPES)) return target;
  if (!ALLOWED_GLOBALS.has(target.qualified)) {
    throw new WeightsOnlyError(`Unsupported global: GLOBAL ${target.qualified} was not an allowed global by default.`);
  }
  return target;
}

type PersistentLoad = (pid: unknown) => unknown;

const decoder = new TextDecoder('utf-8', { fatal: true });

/**
 * Interpret one pickle starting at ``start``; returns the value and the offset
 * after its ``STOP`` opcode.
 */
function unpickle(bytes: Uint8Array, start: number, persistentLoad: PersistentLoad): { value: unknown; end: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = start;
  const stack: unknown[] = [];
  const marks: number[] = [];
  const memo = new Map<number, unknown>();
  const need = (count: number): void => {
    if (offset + count > bytes.length) throw new WeightsOnlyError('pickle data was truncated');
  };
  const u8 = (): number => {
    need(1);
    return bytes[offset++]!;
  };
  const u16 = (): number => {
    need(2);
    const value = view.getUint16(offset, true);
    offset += 2;
    return value;
  };
  const u32 = (): number => {
    need(4);
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const u64 = (): number => {
    need(8);
    const value = view.getBigUint64(offset, true);
    offset += 8;
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new WeightsOnlyError('pickle length is too large');
    return Number(value);
  };
  const take = (count: number): Uint8Array => {
    need(count);
    const out = bytes.subarray(offset, offset + count);
    offset += count;
    return out;
  };
  const line = (): string => {
    const end = bytes.indexOf(0x0a, offset);
    if (end < 0) throw new WeightsOnlyError('pickle data was truncated');
    const text = decoder.decode(bytes.subarray(offset, end));
    offset = end + 1;
    return text;
  };
  const pop = (): unknown => {
    if (!stack.length) throw new WeightsOnlyError('pickle stack underflow');
    return stack.pop();
  };
  const popMark = (): unknown[] => {
    const mark = marks.pop();
    if (mark === undefined) throw new WeightsOnlyError('pickle MARK not found');
    return stack.splice(mark);
  };
  const top = (): unknown => {
    if (!stack.length) throw new WeightsOnlyError('pickle stack underflow');
    return stack[stack.length - 1];
  };
  const long = (data: Uint8Array): number | bigint => {
    if (!data.length) return 0;
    let value = 0n;
    for (let index = data.length - 1; index >= 0; index -= 1) value = (value << 8n) | BigInt(data[index]!);
    if (data[data.length - 1]! & 0x80) value -= 1n << BigInt(8 * data.length);
    return value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value;
  };
  const setItems = (target: unknown, items: unknown[]): void => {
    if (!(target instanceof Map)) throw new WeightsOnlyError('SETITEM target is not a dict');
    for (let index = 0; index + 1 < items.length; index += 2) target.set(items[index], items[index + 1]);
  };
  const append = (target: unknown, items: unknown[]): void => {
    if (target instanceof Set) {
      for (const item of items) target.add(item);
      return;
    }
    if (!Array.isArray(target) || Object.isFrozen(target)) throw new WeightsOnlyError('APPEND target is not a list');
    target.push(...items);
  };
  for (;;) {
    const opcode = u8();
    switch (opcode) {
      case 0x80: u8(); break; // PROTO
      case 0x95: u64(); break; // FRAME
      case 0x28: marks.push(stack.length); break; // MARK
      case 0x2e: // STOP
        return { value: pop(), end: offset };
      case 0x30: pop(); break; // POP
      case 0x31: popMark(); break; // POP_MARK
      case 0x32: stack.push(top()); break; // DUP
      case 0x4e: stack.push(null); break; // NONE
      case 0x88: stack.push(true); break; // NEWTRUE
      case 0x89: stack.push(false); break; // NEWFALSE
      case 0x4a: { need(4); stack.push(view.getInt32(offset, true)); offset += 4; break; } // BININT
      case 0x4b: stack.push(u8()); break; // BININT1
      case 0x4d: stack.push(u16()); break; // BININT2
      case 0x8a: stack.push(long(take(u8()))); break; // LONG1
      case 0x8b: { need(4); const size = view.getInt32(offset, true); offset += 4; if (size < 0) throw new WeightsOnlyError('negative LONG4 size'); stack.push(long(take(size))); break; }
      case 0x49: { // INT
        const text = line();
        stack.push(text === '01' ? true : text === '00' ? false : Number.parseInt(text, 10));
        break;
      }
      case 0x4c: { const text = line().replace(/L$/, ''); const value = BigInt(text); stack.push(value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value); break; } // LONG
      case 0x47: { need(8); stack.push(view.getFloat64(offset, false)); offset += 8; break; } // BINFLOAT
      case 0x46: stack.push(Number.parseFloat(line())); break; // FLOAT
      case 0x58: stack.push(decoder.decode(take(u32()))); break; // BINUNICODE
      case 0x8c: stack.push(decoder.decode(take(u8()))); break; // SHORT_BINUNICODE
      case 0x8d: stack.push(decoder.decode(take(u64()))); break; // BINUNICODE8
      case 0x55: stack.push(String.fromCharCode(...take(u8()))); break; // SHORT_BINSTRING (latin-1)
      case 0x54: { need(4); const size = view.getInt32(offset, true); offset += 4; stack.push(String.fromCharCode(...take(size))); break; } // BINSTRING
      case 0x42: stack.push(take(u32()).slice()); break; // BINBYTES
      case 0x43: stack.push(take(u8()).slice()); break; // SHORT_BINBYTES
      case 0x8e: stack.push(take(u64()).slice()); break; // BINBYTES8
      case 0x96: stack.push(take(u64()).slice()); break; // BYTEARRAY8
      case 0x5d: stack.push([]); break; // EMPTY_LIST
      case 0x7d: stack.push(new Map()); break; // EMPTY_DICT
      case 0x29: stack.push(Object.freeze([])); break; // EMPTY_TUPLE
      case 0x8f: stack.push(new Set()); break; // EMPTY_SET
      case 0x6c: stack.push(popMark()); break; // LIST
      case 0x74: stack.push(Object.freeze(popMark())); break; // TUPLE
      case 0x85: stack.push(Object.freeze([pop()])); break; // TUPLE1
      case 0x86: { const b = pop(); const a = pop(); stack.push(Object.freeze([a, b])); break; } // TUPLE2
      case 0x87: { const c = pop(); const b = pop(); const a = pop(); stack.push(Object.freeze([a, b, c])); break; } // TUPLE3
      case 0x64: { const items = popMark(); const dict = new Map(); setItems(dict, items); stack.push(dict); break; } // DICT
      case 0x91: stack.push(new Set(popMark())); break; // FROZENSET
      case 0x61: { const item = pop(); append(top(), [item]); break; } // APPEND
      case 0x65: { const items = popMark(); append(top(), items); break; } // APPENDS
      case 0x90: { const items = popMark(); append(top(), items); break; } // ADDITEMS
      case 0x73: { const value = pop(); const key = pop(); setItems(top(), [key, value]); break; } // SETITEM
      case 0x75: { const items = popMark(); setItems(top(), items); break; } // SETITEMS
      case 0x71: memo.set(u8(), top()); break; // BINPUT
      case 0x72: memo.set(u32(), top()); break; // LONG_BINPUT
      case 0x70: memo.set(Number.parseInt(line(), 10), top()); break; // PUT
      case 0x94: memo.set(memo.size, top()); break; // MEMOIZE
      case 0x68: case 0x6a: case 0x67: { // BINGET, LONG_BINGET, GET
        const key = opcode === 0x68 ? u8() : opcode === 0x6a ? u32() : Number.parseInt(line(), 10);
        if (!memo.has(key)) throw new WeightsOnlyError(`memo key ${key} is missing`);
        stack.push(memo.get(key));
        break;
      }
      case 0x63: { const module = line(); const name = line(); stack.push(resolveGlobal(module, name)); break; } // GLOBAL
      case 0x93: { // STACK_GLOBAL
        const name = pop();
        const module = pop();
        if (typeof module !== 'string' || typeof name !== 'string') throw new WeightsOnlyError('STACK_GLOBAL requires strings');
        stack.push(resolveGlobal(module, name));
        break;
      }
      case 0x52: { // REDUCE
        const args = pop();
        const target = pop();
        if (!(target instanceof Global) || !Array.isArray(args)) throw new WeightsOnlyError('REDUCE requires an allowed global and a tuple');
        stack.push(callGlobal(target, args));
        break;
      }
      case 0x81: { // NEWOBJ
        const args = pop();
        const target = pop();
        if (!(target instanceof Global) || !Array.isArray(args)) throw new WeightsOnlyError('NEWOBJ requires an allowed class and a tuple');
        stack.push(callGlobal(target, args));
        break;
      }
      case 0x62: { // BUILD
        const state = pop();
        const target = top();
        // OrderedDict and tensor states carry nothing a state dict needs
        // (``_metadata`` versions, empty backward hooks).
        if (!(target instanceof Map) && !(target instanceof LazyTensor)) throw new WeightsOnlyError('BUILD target is not an allowed object');
        void state;
        break;
      }
      case 0x51: stack.push(persistentLoad(pop())); break; // BINPERSID
      case 0x50: stack.push(persistentLoad(line())); break; // PERSID
      default:
        throw new WeightsOnlyError(`Unsupported operand ${opcode}`);
    }
  }
}

function storageDType(type: unknown): DType {
  if (!(type instanceof Global) || type.module !== 'torch' || !(type.name in STORAGE_TYPES)) {
    if (type instanceof Global && type.module === 'torch' && type.name in TORCH_DTYPES) return TORCH_DTYPES[type.name]!;
    throw new WeightsOnlyError(`unsupported storage type ${type instanceof Global ? type.qualified : String(type)}`);
  }
  return STORAGE_TYPES[type.name]!;
}

// ---------------------------------------------------------------------------
// Zip archives (``torch.save`` since PyTorch 1.6)
// ---------------------------------------------------------------------------

interface ZipEntry {
  name: string;
  method: number;
  compressedSize: number;
  size: number;
  localOffset: number;
}

function zipEntries(bytes: Uint8Array): Map<string, ZipEntry> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let index = bytes.length - 22; index >= Math.max(0, bytes.length - 22 - 65535); index -= 1) {
    if (view.getUint32(index, true) === 0x06054b50) {
      eocd = index;
      break;
    }
  }
  if (eocd < 0) throw new ValueError('PyTorch checkpoint zip archive has no end of central directory');
  let count = view.getUint16(eocd + 10, true);
  let directory = view.getUint32(eocd + 16, true);
  if (count === 0xffff || directory === 0xffffffff) {
    const locator = eocd - 20;
    if (locator < 0 || view.getUint32(locator, true) !== 0x07064b50) throw new ValueError('PyTorch checkpoint zip64 locator is missing');
    const record = Number(view.getBigUint64(locator + 8, true));
    if (view.getUint32(record, true) !== 0x06064b50) throw new ValueError('PyTorch checkpoint zip64 directory is missing');
    count = Number(view.getBigUint64(record + 32, true));
    directory = Number(view.getBigUint64(record + 48, true));
  }
  const entries = new Map<string, ZipEntry>();
  let offset = directory;
  for (let index = 0; index < count; index += 1) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new ValueError('PyTorch checkpoint zip central directory is corrupt');
    const method = view.getUint16(offset + 10, true);
    let compressedSize = view.getUint32(offset + 20, true);
    let size = view.getUint32(offset + 24, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    let localOffset = view.getUint32(offset + 42, true);
    const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameLength));
    // Zip64 extended information: present fields replace the 0xFFFFFFFF placeholders in order.
    let extra = offset + 46 + nameLength;
    const extraEnd = extra + extraLength;
    while (extra + 4 <= extraEnd) {
      const id = view.getUint16(extra, true);
      const length = view.getUint16(extra + 2, true);
      if (id === 0x0001) {
        let field = extra + 4;
        if (size === 0xffffffff) { size = Number(view.getBigUint64(field, true)); field += 8; }
        if (compressedSize === 0xffffffff) { compressedSize = Number(view.getBigUint64(field, true)); field += 8; }
        if (localOffset === 0xffffffff) localOffset = Number(view.getBigUint64(field, true));
      }
      extra += 4 + length;
    }
    entries.set(name, { name, method, compressedSize, size, localOffset });
    offset = extraEnd + commentLength;
  }
  return entries;
}

function zipRead(bytes: Uint8Array, entry: ZipEntry): Uint8Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(entry.localOffset, true) !== 0x04034b50) throw new ValueError(`PyTorch checkpoint zip entry ${entry.name} is corrupt`);
  const start = entry.localOffset + 30 + view.getUint16(entry.localOffset + 26, true) + view.getUint16(entry.localOffset + 28, true);
  const data = bytes.subarray(start, start + entry.compressedSize);
  if (entry.method === 0) return data;
  if (entry.method === 8) return inflateRaw(data);
  throw new ValueError(`PyTorch checkpoint zip entry ${entry.name} uses unsupported compression ${entry.method}`);
}

function loadZip(bytes: Uint8Array): unknown {
  const entries = zipEntries(bytes);
  const pickle = [...entries.keys()].find((name) => name.endsWith('data.pkl') && name.split('/').length === 2);
  if (!pickle) throw new ValueError('PyTorch checkpoint archive has no data.pkl');
  const root = pickle.slice(0, -'data.pkl'.length);
  const order = entries.get(`${root}byteorder`);
  if (order && decoder.decode(zipRead(bytes, order)).trim() !== 'little') {
    throw new ValueError('big-endian PyTorch checkpoints are not supported');
  }
  const storages = new Map<string, Storage>();
  const load: PersistentLoad = (pid) => {
    if (!Array.isArray(pid) || pid[0] !== 'storage' || pid.length < 5) throw new WeightsOnlyError('unsupported persistent id');
    const [, type, key, , numel] = pid as [unknown, unknown, unknown, unknown, unknown];
    if (typeof key !== 'string') throw new WeightsOnlyError('storage key must be a string');
    let storage = storages.get(key);
    if (!storage) {
      const dtype = storageDType(type);
      const entry = entries.get(`${root}data/${key}`);
      if (!entry) throw new ValueError(`PyTorch checkpoint archive is missing storage ${key}`);
      storage = new Storage(dtype, toInt(numel, 'storage size'), () => zipRead(bytes, entry));
      storages.set(key, storage);
    }
    return storage;
  };
  return unpickle(zipRead(bytes, entries.get(pickle)!), 0, load).value;
}

// ---------------------------------------------------------------------------
// Legacy stream format (``torch.save(..., _use_new_zipfile_serialization=False)``)
// ---------------------------------------------------------------------------

function loadLegacy(bytes: Uint8Array): unknown {
  const none: PersistentLoad = () => {
    throw new WeightsOnlyError('unexpected persistent id');
  };
  let position = 0;
  const next = (load: PersistentLoad = none): unknown => {
    const { value, end } = unpickle(bytes, position, load);
    position = end;
    return value;
  };
  const magic = next();
  if (magic !== LEGACY_MAGIC) throw new ValueError('Invalid magic number; corrupt file?');
  const protocol = next();
  if (protocol !== LEGACY_PROTOCOL) throw new ValueError(`Invalid protocol version: ${String(protocol)}`);
  const info = next();
  if (info instanceof Map && info.get('little_endian') === false) throw new ValueError('big-endian PyTorch checkpoints are not supported');
  const storages = new Map<string, Storage>();
  const pending = new Map<string, { dtype: DType; numel: number; data: Uint8Array | null }>();
  const load: PersistentLoad = (pid) => {
    if (!Array.isArray(pid) || pid[0] !== 'storage' || pid.length < 5) throw new WeightsOnlyError('unsupported persistent id');
    const [, type, key, , numel, view] = pid as [unknown, unknown, unknown, unknown, unknown, unknown];
    if (typeof key !== 'string') throw new WeightsOnlyError('storage key must be a string');
    const dtype = storageDType(type);
    if (!pending.has(key)) pending.set(key, { dtype, numel: toInt(numel, 'storage size'), data: null });
    const record = pending.get(key)!;
    let storage = storages.get(key);
    if (!storage) {
      storage = new Storage(record.dtype, record.numel, () => {
        if (record.data === null) throw new ValueError(`PyTorch checkpoint is missing storage ${key}`);
        return record.data;
      });
      storages.set(key, storage);
    }
    if (view !== null && view !== undefined) {
      if (!Array.isArray(view) || view.length !== 3) throw new WeightsOnlyError('invalid storage view');
      const [, viewOffset, viewSize] = view as [unknown, unknown, unknown];
      const start = toInt(viewOffset, 'view offset');
      const size = toInt(viewSize, 'view size');
      const width = itemSize(record.dtype);
      return new Storage(record.dtype, size, () => {
        if (record.data === null) throw new ValueError(`PyTorch checkpoint is missing storage ${key}`);
        return record.data.subarray(start * width, (start + size) * width);
      });
    }
    return storage;
  };
  const result = next(load);
  const keys = next();
  if (!Array.isArray(keys)) throw new ValueError('PyTorch checkpoint storage keys are missing');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (const key of keys) {
    const record = typeof key === 'string' ? pending.get(key) : undefined;
    if (!record) throw new ValueError(`PyTorch checkpoint lists unknown storage ${String(key)}`);
    if (position + 8 > bytes.length) throw new ValueError('PyTorch checkpoint storage is truncated');
    const numel = Number(view.getBigInt64(position, true));
    position += 8;
    const length = numel * itemSize(record.dtype);
    if (position + length > bytes.length) throw new ValueError('PyTorch checkpoint storage is truncated');
    record.data = bytes.subarray(position, position + length);
    record.numel = numel;
    position += length;
  }
  return result;
}

/**
 * The tensors of a PyTorch state-dict checkpoint (``torch.load(...,
 * weights_only=True)``), in file order. Non-tensor entries are ignored, as
 * transformers ignores them.
 */
export function loadTorchStateDict(bytes: Uint8Array): Map<string, Tensor> {
  const isZip = bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b && bytes[2] === 0x03 && bytes[3] === 0x04;
  const value = isZip ? loadZip(bytes) : loadLegacy(bytes);
  if (!(value instanceof Map)) throw new ValueError('PyTorch checkpoint does not contain a state dict');
  const tensors = new Map<string, Tensor>();
  for (const [key, item] of value) {
    if (typeof key === 'string' && item instanceof LazyTensor) tensors.set(key, item.build());
  }
  return tensors;
}
