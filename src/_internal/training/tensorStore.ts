/**
 * Private safetensors codec keeping training checkpoint metadata small (Python
 * ``tensorcode/_internal/training/_tensor_store.py``). Tensors are replaced by
 * ``{type: 'tensor_ref', key, shape, dtype}`` references into one immutable,
 * checksummed ``tensors-<uuid>.safetensors`` file.
 */
import { createHash, randomUUID } from 'node:crypto';
import { lstat, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Tensor } from '../../nn/tensor.js';
import { noGrad } from '../../nn/autograd.js';
import { deserializeSafetensors, serializeSafetensors } from '../../nn/safetensors.js';
import { ValueError } from '../../errors.js';
import { isPlainObject } from '../json.js';
import { Codec } from './persistence.js';

/** SHA-256 hex digest of a file's bytes. */
export async function digest(path: string): Promise<string> {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

export function digestBytes(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

export interface TensorFileRecord {
  file: string;
  sha256: string;
}

interface AliasEntry {
  shape: readonly number[];
  dtype: string;
  key: string;
}

export class TensorStore extends Codec {
  readonly tensors: Map<string, Tensor>;
  private readonly aliases = new Map<object, AliasEntry[]>();

  constructor(tensors: Map<string, Tensor> | null = null) {
    super();
    this.tensors = tensors ?? new Map();
  }

  override encode(value: unknown): unknown {
    if (!(value instanceof Tensor)) return super.encode(value);
    // Views of the same storage with the same geometry are stored once.
    const entries = this.aliases.get(value._storage) ?? [];
    let entry = entries.find((item) => item.dtype === value.dtype && item.shape.length === value.shape.length
      && item.shape.every((size, index) => size === value.shape[index]));
    if (!entry) {
      const key = `tensor_${this.tensors.size}`;
      const copy = noGrad(() => value.detach().clone());
      if (!copy.allFinite()) throw new ValueError('Nonfinite tensor in checkpoint');
      this.tensors.set(key, copy);
      entry = { shape: [...value.shape], dtype: value.dtype, key };
      entries.push(entry);
      this.aliases.set(value._storage, entries);
    }
    return { type: 'tensor_ref', key: entry.key, shape: [...value.shape], dtype: value.dtype };
  }

  override decode(value: unknown): unknown {
    if (isPlainObject(value) && value.type === 'tensor_ref') {
      const keys = Object.keys(value);
      const { key, shape, dtype } = value;
      if (keys.length !== 4 || !['type', 'key', 'shape', 'dtype'].every((name) => keys.includes(name))
        || typeof key !== 'string' || !Array.isArray(shape)
        || shape.some((size) => typeof size !== 'number' || !Number.isInteger(size) || size < 0)
        || typeof dtype !== 'string') {
        throw new ValueError('Malformed checkpoint tensor reference');
      }
      const tensor = this.tensors.get(key);
      if (tensor === undefined) throw new ValueError('Dangling checkpoint tensor reference');
      if (shape.length !== tensor.shape.length || shape.some((size, index) => size !== tensor.shape[index]) || dtype !== tensor.dtype) {
        throw new ValueError('Checkpoint tensor reference shape or dtype differs');
      }
      return tensor;
    }
    return super.decode(value);
  }

  /** Write the collected tensors to a fresh uniquely named file. */
  async write(directory: string): Promise<TensorFileRecord> {
    const file = `tensors-${randomUUID().replace(/-/g, '')}.safetensors`;
    const bytes = serializeSafetensors(this.tensors, null);
    await writeFile(join(directory, file), bytes);
    return { file, sha256: digestBytes(bytes) };
  }

  /** Verify and read a tensor file; every stored tensor must be referenced by ``payload``. */
  static async read(directory: string, record: unknown, payload: unknown): Promise<TensorStore> {
    if (!isPlainObject(record) || Object.keys(record).length !== 2 || typeof record.file !== 'string'
      || !/^tensors-[0-9a-f]{32}\.safetensors$/.test(record.file)
      || typeof record.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(record.sha256)) {
      throw new ValueError('Invalid checkpoint tensor file reference');
    }
    const path = join(directory, record.file);
    if ((await lstat(path)).isSymbolicLink()) throw new ValueError('Checkpoint tensor file must not be a symlink');
    const buffer = await readFile(path);
    const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    if (digestBytes(bytes) !== record.sha256) throw new ValueError('Checkpoint tensor file digest mismatch');
    let tensors: Map<string, Tensor>;
    try {
      tensors = deserializeSafetensors(bytes).tensors;
    } catch (error) {
      throw new ValueError('Invalid checkpoint tensor file', { cause: error });
    }
    for (const tensor of tensors.values()) if (!tensor.allFinite()) throw new ValueError('Nonfinite tensor in checkpoint');
    const codec = new TensorStore(tensors);
    const used = new Set<string>();
    const visit = (value: unknown): void => {
      if (Array.isArray(value)) {
        value.forEach(visit);
      } else if (isPlainObject(value)) {
        if (value.type === 'tensor_ref') {
          codec.decode(value);
          used.add(value.key as string);
        } else {
          Object.values(value).forEach(visit);
        }
      }
    };
    visit(payload);
    if (used.size !== tensors.size || [...tensors.keys()].some((key) => !used.has(key))) {
      throw new ValueError('Unreferenced checkpoint tensors');
    }
    return codec;
  }
}
