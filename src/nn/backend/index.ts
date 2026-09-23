/**
 * Compute backend controls (``torch.set_num_threads`` /
 * ``torch.get_num_threads`` equivalents and backend selection).
 *
 * Float32 matrix products, attention, softmax, layer normalization and the
 * common activations run on WebAssembly SIMD kernels, in parallel on
 * ``node:worker_threads`` over one shared memory when available. Everything
 * else, float64 tensors, and environments without WebAssembly SIMD use the
 * JavaScript kernels.
 */
import { activeEngine, getEngine, setBackend as selectBackend } from './engine.js';

export interface BackendInfo {
  /** ``'wasm-relaxed-simd'`` (fused multiply-add), ``'wasm-simd'`` or ``'js'``. */
  backend: 'wasm-relaxed-simd' | 'wasm-simd' | 'js';
  /** Compute threads, including the calling thread. */
  threads: number;
  /** Whether kernel memory is shared with worker threads. */
  sharedMemory: boolean;
  /** Bytes of weights currently resident in kernel memory. */
  residentWeightBytes: number;
}

/** Current backend, thread count and resident weight bytes. */
export function backendInfo(): BackendInfo {
  const engine = activeEngine();
  if (!engine) return { backend: 'js', threads: 1, sharedMemory: false, residentWeightBytes: 0 };
  return {
    backend: engine.relaxed ? 'wasm-relaxed-simd' : 'wasm-simd',
    threads: engine.parallelism(),
    sharedMemory: engine.shared,
    residentWeightBytes: engine.residentBytes,
  };
}

/**
 * Set the number of compute threads (``torch.set_num_threads``), including
 * the calling thread; ``1`` runs every kernel on the calling thread. The
 * default is the machine's available parallelism, or ``TENSORCODE_THREADS``.
 */
export function setNumThreads(count: number): void {
  if (!Number.isInteger(count) || count < 1) throw new RangeError('setNumThreads expects a positive integer');
  getEngine()?.setThreads(count);
}

/** Number of compute threads (``torch.get_num_threads``). */
export function getNumThreads(): number {
  return getEngine()?.threadCount ?? 1;
}

/**
 * Use the WebAssembly kernels (``'wasm'``, the default when available) or
 * only the JavaScript kernels (``'js'``; also ``TENSORCODE_BACKEND=js``).
 */
export function setBackend(name: 'wasm' | 'js'): void {
  if (name !== 'wasm' && name !== 'js') throw new RangeError("backend must be 'wasm' or 'js'");
  selectBackend(name);
}

/** Release weights kept resident in kernel memory (they are re-staged on next use). */
export function clearWeightCache(): void {
  getEngine()?.clearCache();
}
