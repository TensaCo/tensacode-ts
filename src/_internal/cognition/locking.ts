/**
 * Model-content fingerprints shared by cognitive sessions (Python
 * ``tensorcode/_internal/cognition/locking.py``).
 *
 * Python guards cognitive state with ``threading.RLock``. JavaScript runs model
 * computation on one thread, so synchronous methods need no lock; asynchronous
 * persistence uses {@link SerialQueue} where exclusive access matters.
 */
import type { Module } from '../../nn/module.js';
import { Tensor, arange, zeros } from '../../nn/tensor.js';
import { torchDTypeName } from '../../nn/dtype.js';
import { tensorBytes } from '../../nn/safetensors.js';
import { canonicalJson, sha256Hex } from '../json.js';

export { tensorBytes };

/** Python ``str(tuple(shape))``: ``()``, ``(3,)``, ``(2, 4)``. */
export function pythonShape(shape: readonly number[]): string {
  if (!shape.length) return '()';
  if (shape.length === 1) return `(${shape[0]},)`;
  return `(${shape.join(', ')})`;
}

/** Incremental SHA-256 over concatenated byte chunks (Python ``hashlib.sha256().update``). */
export class Sha256Accumulator {
  private readonly chunks: Uint8Array[] = [];
  private length = 0;

  update(data: Uint8Array | string): this {
    const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
    this.chunks.push(bytes);
    this.length += bytes.length;
    return this;
  }

  bytes(): Uint8Array {
    const joined = new Uint8Array(this.length);
    let offset = 0;
    for (const chunk of this.chunks) {
      joined.set(chunk, offset);
      offset += chunk.length;
    }
    return joined;
  }

  hexdigest(): string {
    return sha256Hex(this.bytes());
  }
}

/** Hash every named tensor as Python does: name, ``torch.<dtype>``, shape tuple, raw bytes. */
export function updateTensorDigest(digest: Sha256Accumulator, name: string, tensor: Tensor): void {
  digest.update(name);
  digest.update(torchDTypeName(tensor.dtype));
  digest.update(pythonShape(tensor.shape));
  digest.update(tensorBytes(tensor));
}

/**
 * Python registers non-persistent ``position_ids``/``token_type_ids`` buffers
 * on BERT-family embeddings; they appear in ``named_buffers()`` (and therefore
 * in content fingerprints and tensor schemas) but never in state dicts. When
 * the TypeScript module does not register them, equivalent virtual buffers
 * keep fingerprints identical to Python's.
 */
const VIRTUAL_BUFFERS: Record<string, readonly ('position_ids' | 'token_type_ids')[]> = {
  AlbertEmbeddings: ['position_ids', 'token_type_ids'],
  BertEmbeddings: ['position_ids', 'token_type_ids'],
  RobertaEmbeddings: ['position_ids', 'token_type_ids'],
  DistilEmbeddings: ['position_ids'],
};
const virtualCache = new WeakMap<Module, Map<string, Tensor>>();

function virtualBuffers(module: Module): [string, Tensor][] {
  const names = VIRTUAL_BUFFERS[module.constructor.name];
  const positions = module.getModule('position_embeddings') as (Module & { numEmbeddings?: number; weight?: Tensor }) | null;
  if (!names || !positions) return [];
  const length = positions.weight?.shape[0];
  if (length === undefined) return [];
  let cache = virtualCache.get(module);
  if (!cache) {
    cache = new Map();
    virtualCache.set(module, cache);
  }
  const result: [string, Tensor][] = [];
  for (const name of names) {
    if (module.getBuffer(name) !== null) continue;
    let value = cache.get(name);
    if (!value || value.shape[1] !== length) {
      value = name === 'position_ids'
        ? arange(0, length, 1, { dtype: 'int64' }).reshape(1, length)
        : zeros([1, length], { dtype: 'int64' });
      cache.set(name, value);
    }
    result.push([name, value]);
  }
  return result;
}

/** Python ``module.named_buffers()`` (including non-persistent buffers Python registers). */
export function pythonNamedBuffers(root: Module): [string, Tensor][] {
  const result: [string, Tensor][] = [];
  const seen = new Set<Tensor>();
  for (const [path, module] of root.namedModules()) {
    for (const [name, buffer] of [...module.namedBuffers({ recurse: false }), ...virtualBuffers(module)]) {
      if (seen.has(buffer)) continue;
      seen.add(buffer);
      result.push([path ? `${path}.${name}` : name, buffer]);
    }
  }
  return result;
}

interface TensorKey {
  name: string;
  tensor: Tensor;
  version: number;
  dtype: string;
  shape: string;
}

/**
 * Cache content hashes by tensor identity/version and configuration.
 *
 * Optimizer and in-place tensor updates bump version counters and invalidate
 * the hash. Writing to ``tensor.data`` directly bypasses the counters; call
 * {@link invalidate} after any such external mutation.
 */
export class ModelFingerprint {
  private key: { serialized: string; tensors: TensorKey[] } | null = null;
  private digest: string | null = null;
  /** Number of full content hashes computed (diagnostics). */
  computations = 0;

  invalidate(): void {
    this.key = null;
  }

  compute(modules: readonly [string, Module][], config: unknown): string {
    const serialized = canonicalJson(config);
    const tensors: TensorKey[] = [];
    for (const [prefix, module] of modules) {
      for (const [name, tensor] of [...module.namedParameters(), ...pythonNamedBuffers(module)] as [string, Tensor][]) {
        tensors.push({ name: `${prefix}.${name}`, tensor, version: tensor.version, dtype: tensor.dtype, shape: pythonShape(tensor.shape) });
      }
    }
    if (!this.matches(serialized, tensors)) {
      const digest = new Sha256Accumulator().update(serialized);
      for (const entry of tensors) updateTensorDigest(digest, entry.name, entry.tensor);
      this.digest = digest.hexdigest();
      this.key = { serialized, tensors };
      this.computations += 1;
    }
    return this.digest!;
  }

  private matches(serialized: string, tensors: TensorKey[]): boolean {
    const key = this.key;
    if (key === null || key.serialized !== serialized || key.tensors.length !== tensors.length) return false;
    return key.tensors.every((entry, index) => {
      const other = tensors[index]!;
      return entry.name === other.name && entry.tensor === other.tensor && entry.version === other.version
        && entry.dtype === other.dtype && entry.shape === other.shape;
    });
  }
}

const fingerprints = new WeakMap<object, ModelFingerprint>();

/** The content-fingerprint cache shared by every cognitive session of one investigator. */
export function cognitionFingerprint(investigator: object): ModelFingerprint {
  let result = fingerprints.get(investigator);
  if (!result) {
    result = new ModelFingerprint();
    fingerprints.set(investigator, result);
  }
  return result;
}

/** Serialize asynchronous critical sections (a minimal promise mutex). */
export class SerialQueue {
  private tail: Promise<unknown> = Promise.resolve();

  run<T>(task: () => Promise<T> | T): Promise<T> {
    const result = this.tail.then(task, task);
    this.tail = result.catch(() => undefined);
    return result;
  }
}
