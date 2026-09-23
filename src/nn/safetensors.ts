/**
 * Safetensors serialization compatible with the reference implementation.
 *
 * ``serializeSafetensors`` orders tensors like the Rust library (dtype
 * descending, then name), writes compact JSON with ``__metadata__`` first and
 * pads the header with spaces to 8 bytes, so identical inputs produce identical
 * bytes. ``saveModel``/``loadModel`` mirror ``safetensors.torch``: tied tensors
 * are stored once and their removed names recorded in metadata.
 */
import {
  allocate, bfloat16BitsToFloat32, float16BitsToFloat32, float32ToBFloat16Bits, float32ToFloat16Bits,
  itemSize, type DType,
} from './dtype.js';
import { numelOf } from './shape.js';
import { Tensor, fromStorage } from './tensor.js';
import type { Module, StateDict } from './module.js';

type SafeDType = 'BOOL' | 'U8' | 'I8' | 'I16' | 'F16' | 'BF16' | 'I32' | 'F32' | 'F64' | 'I64';

const TO_SAFE: Record<DType, SafeDType> = {
  bool: 'BOOL', uint8: 'U8', int8: 'I8', int16: 'I16', float16: 'F16', bfloat16: 'BF16',
  int32: 'I32', float32: 'F32', float64: 'F64', int64: 'I64',
};

const FROM_SAFE: Record<string, DType> = Object.fromEntries(
  Object.entries(TO_SAFE).map(([dtype, safe]) => [safe, dtype as DType]),
);

/** Declaration order of the Rust ``Dtype`` enum (sorting key). */
const RUST_ORDER: SafeDType[] = ['BOOL', 'U8', 'I8', 'I16', 'F16', 'BF16', 'I32', 'F32', 'F64', 'I64'];

export interface SafetensorsContents {
  tensors: Map<string, Tensor>;
  metadata: Record<string, string> | null;
}

/** Raw little-endian bytes of a tensor (``tensor.contiguous().view(torch.uint8)``). */
export function tensorBytes(tensor: Tensor): Uint8Array {
  const size = itemSize(tensor.dtype);
  const bytes = new Uint8Array(tensor.numel * size);
  const data = tensor.data;
  if (tensor.dtype === 'float32' && data instanceof Float32Array) {
    bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    return bytes;
  }
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < data.length; index += 1) {
    const value = data[index]!;
    const offset = index * size;
    switch (tensor.dtype) {
      case 'float32': view.setFloat32(offset, value, true); break;
      case 'float64': view.setFloat64(offset, value, true); break;
      case 'float16': view.setUint16(offset, float32ToFloat16Bits(value), true); break;
      case 'bfloat16': view.setUint16(offset, float32ToBFloat16Bits(value), true); break;
      case 'int64': view.setBigInt64(offset, BigInt(Math.trunc(value)), true); break;
      case 'int32': view.setInt32(offset, value, true); break;
      case 'int16': view.setInt16(offset, value, true); break;
      case 'int8': view.setInt8(offset, value); break;
      case 'uint8': view.setUint8(offset, value); break;
      case 'bool': view.setUint8(offset, value ? 1 : 0); break;
    }
  }
  return bytes;
}

/** A tensor of ``shape`` from little-endian ``dtype`` bytes (safetensors and PyTorch storages). */
export function decodeTensorBytes(dtype: DType, bytes: Uint8Array, shape: number[]): Tensor {
  const count = numelOf(shape);
  const size = itemSize(dtype);
  if (bytes.byteLength !== count * size) throw new Error('tensor byte length does not match its shape');
  const out = allocate(dtype, count);
  if (dtype === 'float32' && bytes.byteOffset % 4 === 0) {
    out.set(new Float32Array(bytes.buffer, bytes.byteOffset, count));
    return fromStorage(out, shape, dtype);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < count; index += 1) {
    const offset = index * size;
    let value: number;
    switch (dtype) {
      case 'float32': value = view.getFloat32(offset, true); break;
      case 'float64': value = view.getFloat64(offset, true); break;
      case 'float16': value = float16BitsToFloat32(view.getUint16(offset, true)); break;
      case 'bfloat16': value = bfloat16BitsToFloat32(view.getUint16(offset, true)); break;
      case 'int64': value = Number(view.getBigInt64(offset, true)); break;
      case 'int32': value = view.getInt32(offset, true); break;
      case 'int16': value = view.getInt16(offset, true); break;
      case 'int8': value = view.getInt8(offset); break;
      case 'uint8': value = view.getUint8(offset); break;
      case 'bool': value = view.getUint8(offset) ? 1 : 0; break;
    }
    out[index] = value;
  }
  return fromStorage(out, shape, dtype);
}

/** Serialize named tensors (and optional string metadata) to safetensors bytes. */
export function serializeSafetensors(
  tensors: Map<string, Tensor> | Record<string, Tensor>,
  metadata: Record<string, string> | null = null,
): Uint8Array {
  const entries = [...(tensors instanceof Map ? tensors.entries() : Object.entries(tensors))];
  for (const [name, value] of entries) {
    if (!(value instanceof Tensor)) throw new TypeError(`${name} is not a tensor`);
  }
  entries.sort(([leftName, left], [rightName, right]) => {
    const order = RUST_ORDER.indexOf(TO_SAFE[right.dtype]) - RUST_ORDER.indexOf(TO_SAFE[left.dtype]);
    if (order !== 0) return order;
    return leftName < rightName ? -1 : leftName > rightName ? 1 : 0;
  });
  const buffers: Uint8Array[] = [];
  let offset = 0;
  const parts: string[] = [];
  if (metadata !== null) {
    for (const [key, value] of Object.entries(metadata)) {
      if (typeof value !== 'string') throw new TypeError(`metadata value for ${key} must be a string`);
    }
    parts.push(`"__metadata__":${JSON.stringify(metadata)}`);
  }
  for (const [name, value] of entries) {
    const bytes = tensorBytes(value);
    buffers.push(bytes);
    parts.push(`${JSON.stringify(name)}:${JSON.stringify({
      dtype: TO_SAFE[value.dtype], shape: [...value.shape], data_offsets: [offset, offset + bytes.byteLength],
    })}`);
    offset += bytes.byteLength;
  }
  let header = `{${parts.join(',')}}`;
  let headerBytes = new TextEncoder().encode(header);
  const padding = (8 - (headerBytes.byteLength % 8)) % 8;
  if (padding) {
    header += ' '.repeat(padding);
    headerBytes = new TextEncoder().encode(header);
  }
  const output = new Uint8Array(8 + headerBytes.byteLength + offset);
  new DataView(output.buffer).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  output.set(headerBytes, 8);
  let position = 8 + headerBytes.byteLength;
  for (const bytes of buffers) {
    output.set(bytes, position);
    position += bytes.byteLength;
  }
  return output;
}

/** Parse safetensors bytes. Tensors are copied into fresh storage. */
export function deserializeSafetensors(bytes: Uint8Array): SafetensorsContents {
  if (bytes.byteLength < 8) throw new Error('safetensors data is too short');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerSize = Number(view.getBigUint64(0, true));
  if (headerSize > bytes.byteLength - 8 || headerSize > 100_000_000) throw new Error('invalid safetensors header size');
  const headerText = new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(8, 8 + headerSize));
  const header = JSON.parse(headerText) as Record<string, unknown>;
  if (!header || typeof header !== 'object' || Array.isArray(header)) throw new Error('invalid safetensors header');
  const dataStart = 8 + headerSize;
  const tensors = new Map<string, Tensor>();
  let metadata: Record<string, string> | null = null;
  const ranges: [number, number][] = [];
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') {
      if (info !== null && (typeof info !== 'object' || Array.isArray(info))) throw new Error('invalid safetensors metadata');
      metadata = info as Record<string, string> | null;
      continue;
    }
    const record = info as { dtype?: unknown; shape?: unknown; data_offsets?: unknown };
    const dtype = FROM_SAFE[String(record.dtype)];
    if (!dtype) throw new Error(`unsupported safetensors dtype ${String(record.dtype)} for ${name}`);
    if (!Array.isArray(record.shape) || record.shape.some((size) => !Number.isInteger(size) || size < 0)) {
      throw new Error(`invalid safetensors shape for ${name}`);
    }
    const offsets = record.data_offsets;
    if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every((value) => Number.isInteger(value))) {
      throw new Error(`invalid safetensors offsets for ${name}`);
    }
    const [begin, end] = offsets as [number, number];
    if (begin < 0 || end < begin || dataStart + end > bytes.byteLength) throw new Error(`safetensors offsets out of range for ${name}`);
    ranges.push([begin, end]);
    tensors.set(name, decodeTensorBytes(dtype, bytes.subarray(dataStart + begin, dataStart + end), record.shape as number[]));
  }
  ranges.sort((a, b) => a[0] - b[0]);
  let cursor = 0;
  for (const [begin, end] of ranges) {
    if (begin !== cursor) throw new Error('safetensors tensor data is not contiguous');
    cursor = end;
  }
  if (dataStart + cursor !== bytes.byteLength) throw new Error('safetensors file has trailing or missing data');
  return { tensors, metadata };
}

/** Names that alias the same storage, grouped. */
export function findSharedTensors(state: StateDict): string[][] {
  const groups = new Map<object, string[]>();
  for (const [name, value] of state) {
    const key = value._storage;
    const group = groups.get(key);
    if (group) group.push(name);
    else groups.set(key, [name]);
  }
  return [...groups.values()].filter((group) => group.length > 1);
}

/** ``safetensors.torch._remove_duplicate_names``: kept name → removed names. */
export function removeDuplicateNames(
  state: StateDict, options: { preferredNames?: Iterable<string>; discardNames?: Iterable<string> } = {},
): Map<string, string[]> {
  const preferred = new Set(options.preferredNames ?? []);
  const discard = new Set(options.discardNames ?? []);
  const result = new Map<string, string[]>();
  for (const shared of findSharedTensors(state)) {
    // Every alias covers the whole storage in this library (no partial views).
    const complete = [...shared].sort();
    let keep = complete[0]!;
    const notDiscarded = complete.filter((name) => !discard.has(name));
    if (notDiscarded.length) keep = notDiscarded[0]!;
    const preferredMatches = complete.filter((name) => preferred.has(name));
    if (preferredMatches.length) keep = preferredMatches[0]!;
    const removed = [...shared].sort().filter((name) => name !== keep);
    result.set(keep, removed);
  }
  return result;
}

async function fs(): Promise<typeof import('node:fs/promises')> {
  return import('node:fs/promises');
}

export async function readSafetensorsFile(path: string): Promise<SafetensorsContents> {
  const bytes = await (await fs()).readFile(path);
  return deserializeSafetensors(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
}

export async function writeSafetensorsFile(
  path: string, tensors: Map<string, Tensor> | Record<string, Tensor>, metadata: Record<string, string> | null = null,
): Promise<void> {
  await (await fs()).writeFile(path, serializeSafetensors(tensors, metadata));
}

/** Serialize a module state like ``safetensors.torch.save_model``. */
export function serializeModel(module: Module, metadata: Record<string, string> | null = null): Uint8Array {
  const state = new Map(module.stateDict());
  let meta = metadata ? { ...metadata } : null;
  for (const [kept, removed] of removeDuplicateNames(state)) {
    for (const name of removed) {
      meta ??= {};
      if (!(name in meta)) meta[name] = kept;
      state.delete(name);
    }
  }
  return serializeSafetensors(state, meta);
}

export async function saveModel(module: Module, path: string, metadata: Record<string, string> | null = null): Promise<void> {
  await (await fs()).writeFile(path, serializeModel(module, metadata));
}

/** Load bytes into a module like ``safetensors.torch.load_model``. */
export function loadModelFromBytes(module: Module, bytes: Uint8Array, options: { strict?: boolean } = {}): { missingKeys: string[]; unexpectedKeys: string[] } {
  const strict = options.strict ?? true;
  const { tensors } = deserializeSafetensors(bytes);
  const modelState = module.stateDict();
  const toRemove = removeDuplicateNames(modelState, { preferredNames: tensors.keys() });
  const missing = new Set([...modelState.keys()].filter((key) => !tensors.has(key)));
  const unexpected = [...tensors.keys()].filter((key) => !modelState.has(key));
  for (const group of toRemove.values()) {
    for (const name of group) {
      if (!missing.has(name)) unexpected.push(name);
      else missing.delete(name);
    }
  }
  if (strict && (missing.size || unexpected.length)) {
    throw new Error(`Error(s) in loading state_dict: missing keys ${JSON.stringify([...missing])}, unexpected keys ${JSON.stringify(unexpected)}`);
  }
  module.loadStateDict(tensors, { strict: false });
  return { missingKeys: [...missing], unexpectedKeys: unexpected };
}

export async function loadModel(module: Module, path: string, options: { strict?: boolean } = {}): Promise<{ missingKeys: string[]; unexpectedKeys: string[] }> {
  const bytes = await (await fs()).readFile(path);
  return loadModelFromBytes(module, new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), options);
}
