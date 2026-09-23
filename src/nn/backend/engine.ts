/**
 * WebAssembly SIMD compute engine with an optional worker-thread pool.
 *
 * One linear memory holds kernel operands. On Node it is a shared
 * ``WebAssembly.Memory``, and worker threads (``node:worker_threads``)
 * instantiate the same kernel module over it; the calling thread publishes a
 * job (kernel, argument block, task count), takes part in it, and blocks with
 * ``Atomics.wait`` until every task is done, so the public tensor API stays
 * synchronous. Without WebAssembly SIMD the engine reports itself unavailable
 * and callers use their JavaScript kernels; without workers or shared memory it
 * runs single threaded.
 *
 * Environment:
 * - ``TENSORCODE_THREADS``: total compute threads (default: available
 *   parallelism). ``1`` disables the worker pool.
 * - ``TENSORCODE_BACKEND=js``: disable WebAssembly kernels entirely.
 * - ``TENSORCODE_WASM_CACHE_MB``: budget for weights kept resident in kernel
 *   memory between calls (default 1536; ``0`` disables the cache).
 */
import { RELAXED_MEMORY_FLAGS, RELAXED_WASM, SIMD_MEMORY_FLAGS, SIMD_WASM } from './kernels.generated.js';

const PAGE = 65536;
const MAX_PAGES = 65536;
/** Argument block for the running job (the kernels use no shadow stack or data). */
export const ARGS = PAGE;
const ARGS_WORDS = 64;
const HEAP_START = PAGE + 4096;
/** Results at least this long (floats) are copied out by every thread into shared memory. */
const COPY_PARALLEL = 1 << 20;
const COPY_CHUNK = 1 << 18;
const ALIGN = 64;

export const enum Kernel {
  GemmNT = 0,
  Transpose = 1,
  Softmax = 2,
  Attention = 3,
  Unary = 4,
  LayerNorm = 5,
  /** Copy kernel memory into a shared output buffer (JavaScript, see ``copyOut``). */
  CopyOut = 6,
  SwapAxes = 7,
  Binary = 8,
}

type TaskFunction = (args: number, task: number, thread: number) => void;

// Minimal WebAssembly typings (the package compiles without DOM libraries).
interface WasmModule { readonly __wasmModule?: never }
interface WasmMemory {
  readonly buffer: ArrayBufferLike;
  grow(pages: number): number;
}
interface WasmApi {
  Module: new (bytes: Uint8Array) => WasmModule;
  Instance: new (module: WasmModule, imports: object) => { readonly exports: Record<string, unknown> };
  Memory: new (descriptor: { initial: number; maximum?: number; shared?: boolean }) => WasmMemory;
  validate(bytes: Uint8Array): boolean;
}

function webAssembly(): WasmApi | null {
  const api = (globalThis as { WebAssembly?: WasmApi }).WebAssembly;
  return typeof api === 'object' && api !== null ? api : null;
}

interface NodeBuiltins {
  workerThreads: typeof import('node:worker_threads') | null;
  os: typeof import('node:os') | null;
}

function nodeBuiltins(): NodeBuiltins {
  const get = (globalThis as { process?: { getBuiltinModule?: (name: string) => unknown } }).process?.getBuiltinModule;
  if (typeof get !== 'function') return { workerThreads: null, os: null };
  try {
    return {
      workerThreads: get('node:worker_threads') as typeof import('node:worker_threads'),
      os: get('node:os') as typeof import('node:os'),
    };
  } catch {
    return { workerThreads: null, os: null };
  }
}

function environment(name: string): string | undefined {
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
  return env?.[name];
}

function decodeBase64(text: string): Uint8Array {
  const buffer = (globalThis as { Buffer?: { from(text: string, encoding: string): Uint8Array } }).Buffer;
  if (buffer) return new Uint8Array(buffer.from(text, 'base64'));
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

// ------------------------------------------------------------------ worker pool

// Control words (Int32 indices), each on its own cache line.
const WAKE = 0;
const DONE = 32;
const KIND = 16;
const ARGS_POINTER = 17;
const FAILED = 18;
const NEXT_OFFSET = 192;
const CONTROL_BYTES = 256;
const INDEX_BITS = 20n;
const INDEX_MASK = (1n << INDEX_BITS) - 1n;
const GENERATION_SHIFT = 2n * INDEX_BITS;
const GENERATION_MASK = (1n << 23n) - 1n;
/** Maximum number of tasks in one job. */
export const MAX_TASKS = Number(INDEX_MASK);

/**
 * Worker source (CommonJS, evaluated). The ``next`` word packs the job
 * generation, its task count and the next task index; a worker claims a task
 * with a compare-and-swap, which only succeeds while that job is current, so a
 * stale worker can never run a task with another job's arguments.
 */
const WORKER_SOURCE = `
const { workerData, receiveMessageOnPort } = require('node:worker_threads');
const { module, memory, control, thread, spin, port } = workerData;
const instance = new WebAssembly.Instance(module, { env: { memory } });
const e = instance.exports;
// Output buffers of copy jobs arrive on the port; only the newest is kept
// (so a worker may hold the latest output buffer until the next copy job).
let target = null;
let targetId = -1;
function latest() {
  for (let message = receiveMessageOnPort(port); message !== undefined; message = receiveMessageOnPort(port)) {
    targetId = message.message.id;
    target = message.message.buffer;
  }
}
function copyOut(args, index) {
  const header = new Uint32Array(memory.buffer, args, 4);
  const source = header[0], length = header[1], chunk = header[2], id = header[3];
  while (targetId !== id) latest();
  const start = index * chunk;
  const count = Math.min(length, start + chunk) - start;
  new Float32Array(target, start * 4, count).set(new Float32Array(memory.buffer, source + start * 4, count));
}
const kernels = [e.task_gemm_nt, e.task_transpose, e.task_softmax, e.task_attention, e.task_unary, e.task_layernorm, copyOut, e.task_swap_axes, e.task_binary];
const words = new Int32Array(control);
const next = new BigInt64Array(control, ${NEXT_OFFSET}, 1);
for (;;) {
  const wake = Atomics.load(words, ${WAKE});
  for (;;) {
    const value = Atomics.load(next, 0);
    const index = Number(value & ${INDEX_MASK}n);
    const total = Number((value >> ${INDEX_BITS}n) & ${INDEX_MASK}n);
    if (index >= total) break;
    const kind = Atomics.load(words, ${KIND});
    const args = Atomics.load(words, ${ARGS_POINTER});
    if (Atomics.compareExchange(next, 0, value, value + 1n) !== value) continue;
    try { kernels[kind](args, index, thread); } catch { Atomics.store(words, ${FAILED}, 1); }
    if (Atomics.add(words, ${DONE}, 1) + 1 === total) Atomics.notify(words, ${DONE});
  }
  latest();
  let count = 0;
  while (Atomics.load(words, ${WAKE}) === wake && count < spin) count += 1;
  if (Atomics.load(words, ${WAKE}) === wake) Atomics.wait(words, ${WAKE}, wake);
}
`;

class Pool {
  readonly words: Int32Array;
  readonly next: BigInt64Array;
  private generation = 0n;
  private readonly workers: { terminate(): unknown }[] = [];
  private readonly ports: { postMessage(value: unknown): void; close(): void }[] = [];
  private copyId = 0;

  constructor(module: WasmModule, memory: WasmMemory, threads: number, workerThreads: typeof import('node:worker_threads')) {
    const control = new SharedArrayBuffer(CONTROL_BYTES);
    this.words = new Int32Array(control);
    this.next = new BigInt64Array(control, NEXT_OFFSET, 1);
    for (let thread = 1; thread < threads; thread += 1) {
      const channel = new workerThreads.MessageChannel();
      const worker = new workerThreads.Worker(WORKER_SOURCE, {
        eval: true,
        workerData: { module, memory, control, thread, spin: 20_000, port: channel.port2 },
        transferList: [channel.port2],
        stdout: false,
        stderr: false,
      });
      worker.unref();
      worker.on('error', () => {});
      channel.port1.unref();
      this.workers.push(worker);
      this.ports.push(channel.port1);
    }
  }

  /**
   * Hand ``buffer`` to every worker for the next copy job; returns its id.
   * Workers keep only the newest buffer they received.
   */
  shareOutput(buffer: SharedArrayBuffer): number {
    this.copyId = (this.copyId + 1) | 0;
    for (const port of this.ports) port.postMessage({ id: this.copyId, buffer });
    return this.copyId;
  }

  get size(): number {
    return this.workers.length + 1;
  }

  /** Run ``tasks`` work items of ``kernel`` on every thread; returns once all are done. */
  run(kernel: TaskFunction, kind: Kernel, args: number, tasks: number): void {
    const { words, next } = this;
    Atomics.store(words, DONE, 0);
    Atomics.store(words, FAILED, 0);
    Atomics.store(words, KIND, kind);
    Atomics.store(words, ARGS_POINTER, args);
    this.generation = (this.generation + 1n) & GENERATION_MASK;
    Atomics.store(next, 0, (this.generation << GENERATION_SHIFT) | (BigInt(tasks) << INDEX_BITS));
    Atomics.add(words, WAKE, 1);
    // Copy jobs wake everyone so every worker drains its port (queued
    // messages would keep their output buffers alive).
    Atomics.notify(words, WAKE, kind === Kernel.CopyOut ? this.workers.length : Math.min(tasks - 1, this.workers.length));
    let failure: unknown = null;
    for (;;) {
      const value = Atomics.load(next, 0);
      const index = Number(value & INDEX_MASK);
      if (index >= tasks) break;
      if (Atomics.compareExchange(next, 0, value, value + 1n) !== value) continue;
      try {
        kernel(args, index, 0);
      } catch (error) {
        failure = error;
        Atomics.store(words, FAILED, 1);
      }
      Atomics.add(words, DONE, 1);
    }
    for (let done = Atomics.load(words, DONE); done < tasks; done = Atomics.load(words, DONE)) {
      Atomics.wait(words, DONE, done, 1000);
    }
    if (Atomics.load(words, FAILED)) throw failure instanceof Error ? failure : new Error('WebAssembly kernel failed in a worker thread');
  }

  terminate(): void {
    for (const worker of this.workers) void worker.terminate();
    for (const port of this.ports) port.close();
    this.workers.length = 0;
    this.ports.length = 0;
  }
}

// ------------------------------------------------------------------ memory

interface FreeBlock {
  start: number;
  end: number;
}

/** Resident copy of a tensor's storage in kernel memory. */
interface CacheEntry {
  pointer: number;
  bytes: number;
  version: number;
  data: Float32Array;
  layout: string;
  lastUse: number;
  /** Call during which the entry is in use (never evicted then). */
  pinned: number;
  key: WeakRef<object>;
}

export class Engine {
  readonly memory: WasmMemory;
  readonly relaxed: boolean;
  readonly shared: boolean;
  private readonly module: WasmModule;
  private readonly kernels: TaskFunction[];
  private f32: Float32Array;
  private u32: Uint32Array;
  private top = HEAP_START;
  private free: FreeBlock[] = [];
  private pool: Pool | null = null;
  private poolFailed = false;
  private threads: number;
  private readonly workerThreads: typeof import('node:worker_threads') | null;
  // Weight cache.
  private readonly cache = new WeakMap<object, CacheEntry>();
  private readonly seen = new WeakMap<object, number>();
  private readonly live = new Set<CacheEntry>();
  private cachedBytes = 0;
  private clock = 0;
  private epoch = 0;
  readonly cacheBudget: number;
  private readonly registry = new FinalizationRegistry<CacheEntry>((entry) => this.drop(entry));

  constructor() {
    const builtins = nodeBuiltins();
    this.workerThreads = builtins.workerThreads;
    let shared = typeof SharedArrayBuffer === 'function' && builtins.workerThreads !== null;
    const api = webAssembly();
    if (!api) throw new Error('WebAssembly is not available');
    let memory: WasmMemory;
    try {
      memory = new api.Memory({ initial: 2, maximum: MAX_PAGES, shared });
    } catch {
      shared = false;
      memory = new api.Memory({ initial: 2, maximum: MAX_PAGES });
    }
    this.memory = memory;
    this.shared = shared;
    const candidates: [string, number, boolean][] = [[RELAXED_WASM, RELAXED_MEMORY_FLAGS, true], [SIMD_WASM, SIMD_MEMORY_FLAGS, false]];
    let module: WasmModule | null = null;
    let relaxed = false;
    for (const [text, flagsOffset, isRelaxed] of candidates) {
      const bytes = decodeBase64(text);
      bytes[flagsOffset] = shared ? 0x03 : 0x01;
      if (!api.validate(bytes)) continue;
      module = new api.Module(bytes);
      relaxed = isRelaxed;
      break;
    }
    if (!module) throw new Error('WebAssembly SIMD is not supported');
    this.module = module;
    this.relaxed = relaxed;
    const instance = new api.Instance(module, { env: { memory } });
    const exports = instance.exports as Record<string, TaskFunction>;
    this.kernels = [
      exports.task_gemm_nt!, exports.task_transpose!, exports.task_softmax!, exports.task_attention!, exports.task_unary!,
      exports.task_layernorm!, (args, task) => this.copyTask(args, task), exports.task_swap_axes!, exports.task_binary!,
    ];
    this.f32 = new Float32Array(memory.buffer);
    this.u32 = new Uint32Array(memory.buffer);
    this.threads = defaultThreads(builtins.os);
    const budget = Number(environment('TENSORCODE_WASM_CACHE_MB') ?? 1536);
    this.cacheBudget = Number.isFinite(budget) && budget > 0 ? budget * 1024 * 1024 : 0;
  }

  /** Current float32 view of kernel memory (refreshed after growth). */
  get heap(): Float32Array {
    if (this.f32.buffer.byteLength !== this.memory.buffer.byteLength) this.refresh();
    return this.f32;
  }

  private refresh(): void {
    this.f32 = new Float32Array(this.memory.buffer);
    this.u32 = new Uint32Array(this.memory.buffer);
  }

  get threadCount(): number {
    return this.threads;
  }

  setThreads(count: number): void {
    const threads = Math.max(1, Math.floor(count));
    if (threads === this.threads) return;
    this.threads = threads;
    this.pool?.terminate();
    this.pool = null;
    this.poolFailed = false;
  }

  /** Threads that can take part in a job right now (creates the pool lazily). */
  private ensurePool(): Pool | null {
    if (this.pool || this.poolFailed) return this.pool;
    if (this.threads <= 1 || !this.shared || !this.workerThreads) return null;
    try {
      this.pool = new Pool(this.module, this.memory, this.threads, this.workerThreads);
    } catch {
      this.poolFailed = true;
    }
    return this.pool;
  }

  /** Threads a job would use (for sizing per-thread scratch). */
  parallelism(): number {
    if (this.threads <= 1 || !this.shared || !this.workerThreads || this.poolFailed) return 1;
    return this.threads;
  }

  /**
   * Run ``tasks`` items of ``kind`` with ``args`` (u32 words written to the
   * argument block). ``parallel`` false keeps the job on the calling thread.
   */
  run(kind: Kernel, args: readonly number[], tasks: number, parallel: boolean): void {
    if (tasks <= 0) return;
    if (args.length > ARGS_WORDS) throw new RangeError('too many kernel arguments');
    if (tasks > MAX_TASKS) throw new RangeError('too many kernel tasks');
    const u32 = this.heapU32;
    for (let index = 0; index < args.length; index += 1) u32[ARGS / 4 + index] = args[index]!;
    const kernel = this.kernels[kind]!;
    const pool = parallel && tasks > 1 ? this.ensurePool() : null;
    if (pool) {
      pool.run(kernel, kind, ARGS, tasks);
      return;
    }
    for (let task = 0; task < tasks; task += 1) kernel(ARGS, task, 0);
  }

  /**
   * Copy ``length`` floats at ``pointer`` into a new array. Large results go
   * to a ``SharedArrayBuffer`` that every thread fills (and page-faults) in
   * parallel; small ones are sliced on the calling thread.
   */
  copyOut(pointer: number, length: number): Float32Array {
    const base = pointer >>> 2;
    const pool = length >= COPY_PARALLEL ? this.ensurePool() : null;
    if (!pool) return this.heap.slice(base, base + length);
    const buffer = new SharedArrayBuffer(length * 4);
    this.copyTarget = buffer;
    try {
      const id = pool.shareOutput(buffer);
      this.run(Kernel.CopyOut, [pointer, length, COPY_CHUNK, id], Math.ceil(length / COPY_CHUNK), true);
    } finally {
      this.copyTarget = null;
    }
    return new Float32Array(buffer);
  }

  private copyTarget: SharedArrayBuffer | null = null;

  private copyTask(args: number, task: number): void {
    const u32 = this.heapU32;
    const source = u32[args >>> 2]!;
    const length = u32[(args >>> 2) + 1]!;
    const chunk = u32[(args >>> 2) + 2]!;
    const start = task * chunk;
    const count = Math.min(length, start + chunk) - start;
    new Float32Array(this.copyTarget!, start * 4, count).set(this.heap.subarray((source >>> 2) + start, (source >>> 2) + start + count));
  }

  get heapU32(): Uint32Array {
    if (this.u32.buffer.byteLength !== this.memory.buffer.byteLength) this.refresh();
    return this.u32;
  }

  // -------------------------------------------------------------- allocation

  /** Allocate ``bytes`` (64-byte aligned); returns 0 when memory is exhausted. */
  alloc(bytes: number): number {
    const size = Math.max(ALIGN, Math.ceil(bytes / ALIGN) * ALIGN);
    let pointer = this.take(size);
    while (pointer === 0 && this.evictOne()) pointer = this.take(size);
    return pointer;
  }

  private take(size: number): number {
    for (let index = 0; index < this.free.length; index += 1) {
      const block = this.free[index]!;
      if (block.end - block.start < size) continue;
      const pointer = block.start;
      block.start += size;
      if (block.start === block.end) this.free.splice(index, 1);
      return pointer;
    }
    const end = this.top + size;
    if (end > MAX_PAGES * PAGE) return 0;
    const current = this.memory.buffer.byteLength;
    if (end > current) {
      const pages = Math.ceil((end - current) / PAGE);
      try {
        this.memory.grow(Math.max(pages, Math.min(MAX_PAGES - current / PAGE, Math.ceil(current / PAGE / 4))));
      } catch {
        try {
          this.memory.grow(pages);
        } catch {
          return 0;
        }
      }
      this.refresh();
    }
    const pointer = this.top;
    this.top = end;
    return pointer;
  }

  release(pointer: number, bytes: number): void {
    if (!pointer) return;
    const size = Math.max(ALIGN, Math.ceil(bytes / ALIGN) * ALIGN);
    let start = pointer;
    let end = pointer + size;
    // Insert sorted and coalesce with neighbours.
    let index = 0;
    while (index < this.free.length && this.free[index]!.start < start) index += 1;
    const previous = this.free[index - 1];
    const following = this.free[index];
    if (previous && previous.end === start) {
      start = previous.start;
      this.free.splice(index - 1, 1);
      index -= 1;
    }
    if (following && following.start === end) {
      end = following.end;
      this.free.splice(index, 1);
    }
    if (end === this.top) {
      this.top = start;
      return;
    }
    this.free.splice(index, 0, { start, end });
  }

  // -------------------------------------------------------------- weight cache

  /**
   * Pointer to a resident copy of ``data`` (the storage record ``key`` at
   * ``version``) in ``layout``, or 0 when it is not cached. A storage is cached
   * on its second use at the same version, so tensors used once are never kept.
   */
  cached(key: object, version: number, data: Float32Array, layout: string): number {
    const entry = this.cache.get(key);
    if (entry && entry.version === version && entry.data === data && entry.layout === layout) {
      entry.lastUse = ++this.clock;
      entry.pinned = this.epoch;
      return entry.pointer;
    }
    return 0;
  }

  /** Whether ``cache`` should keep a copy of ``key`` (seen before at this version). */
  shouldCache(key: object, version: number, bytes: number): boolean {
    if (this.cacheBudget <= 0 || bytes > this.cacheBudget / 2) return false;
    const previous = this.seen.get(key);
    this.seen.set(key, version);
    return previous === version;
  }

  /** Keep ``pointer`` (``bytes``, filled with ``data`` in ``layout``) resident for ``key``. */
  remember(key: object, version: number, data: Float32Array, layout: string, pointer: number, bytes: number): void {
    const old = this.cache.get(key);
    if (old) this.drop(old);
    const entry: CacheEntry = {
      pointer, bytes, version, data, layout, lastUse: ++this.clock, pinned: this.epoch, key: new WeakRef(key),
    };
    this.cache.set(key, entry);
    this.live.add(entry);
    this.cachedBytes += bytes;
    this.registry.register(key, entry, entry);
    while (this.cachedBytes > this.cacheBudget && this.evictOne()) { /* evict least recently used */ }
  }

  private drop(entry: CacheEntry): void {
    if (!this.live.delete(entry)) return;
    this.registry.unregister(entry);
    const key = entry.key.deref();
    if (key && this.cache.get(key) === entry) this.cache.delete(key);
    this.cachedBytes -= entry.bytes;
    this.release(entry.pointer, entry.bytes);
  }

  /** Start a kernel call: weights it uses stay resident until the next call. */
  beginCall(): void {
    this.epoch += 1;
  }

  private evictOne(): boolean {
    let oldest: CacheEntry | null = null;
    for (const entry of this.live) if (entry.pinned !== this.epoch && (!oldest || entry.lastUse < oldest.lastUse)) oldest = entry;
    if (!oldest) return false;
    this.drop(oldest);
    return true;
  }

  /** Drop every cached weight. */
  clearCache(): void {
    for (const entry of [...this.live]) this.drop(entry);
  }

  get residentBytes(): number {
    return this.cachedBytes;
  }
}

function defaultThreads(os: typeof import('node:os') | null): number {
  const configured = environment('TENSORCODE_THREADS');
  if (configured !== undefined && configured.trim() !== '') {
    const value = Number(configured);
    if (Number.isInteger(value) && value >= 1) return value;
  }
  if (!os) return 1;
  const available = typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length;
  return Math.max(1, available);
}

let engine: Engine | null | undefined;

/** The shared engine, or ``null`` when WebAssembly SIMD kernels are unavailable or disabled. */
export function getEngine(): Engine | null {
  if (engine !== undefined) return engine;
  engine = null;
  if (environment('TENSORCODE_BACKEND')?.toLowerCase() === 'js') return engine;
  if (!webAssembly()) return engine;
  try {
    engine = new Engine();
  } catch {
    engine = null;
  }
  return engine;
}

/** Enable or disable the WebAssembly kernels at runtime (tests and benchmarks). */
export function setBackend(name: 'wasm' | 'js'): void {
  if (name === 'js') {
    engineDisabled = true;
    return;
  }
  engineDisabled = false;
}

let engineDisabled = false;

/** The engine if enabled. */
export function activeEngine(): Engine | null {
  return engineDisabled ? null : getEngine();
}
