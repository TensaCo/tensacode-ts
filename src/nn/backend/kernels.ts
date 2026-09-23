/**
 * Float32 kernels over the WebAssembly engine: matrix products, softmax and
 * fused attention. Every function returns ``null`` when the engine is
 * unavailable (or kernel memory is exhausted) so callers fall back to their
 * JavaScript implementation.
 *
 * Products reduce over a dimension zero padded to a multiple of four, in four
 * float32 lanes combined as ``(l0 + l2) + (l1 + l3)``; every output element is
 * computed the same way whatever the batch size, tile or thread count, so
 * results are deterministic and batch invariant.
 */
import { activeEngine, Kernel, MAX_TASKS, type Engine } from './engine.js';

/**
 * Work (multiply-adds, or element operations) per parallel task. Small jobs
 * get few tasks, so only as many threads take part as can pay for waking up;
 * below two tasks a job stays on the calling thread.
 */
const TASK_WORK = 1 << 20;

/** Number of tasks worth running ``work`` in parallel (1: stay on the calling thread). */
function parallelTasks(engine: Engine, work: number): number {
  const threads = engine.parallelism();
  if (threads <= 1) return 1;
  return Math.max(1, Math.min(threads * 4, Math.floor(work / TASK_WORK)));
}

const pad4 = (value: number): number => (value + 3) & ~3;

/** Temporary kernel-memory allocations released together. */
class Scratch {
  private readonly blocks: [number, number][] = [];
  failed = false;

  constructor(private readonly engine: Engine) {}

  alloc(bytes: number): number {
    if (this.failed) return 0;
    const pointer = this.engine.alloc(bytes);
    if (!pointer) {
      this.failed = true;
      return 0;
    }
    this.blocks.push([pointer, bytes]);
    return pointer;
  }

  /** Keep ``pointer`` (hand it to the weight cache) instead of releasing it. */
  keep(pointer: number): void {
    const index = this.blocks.findIndex(([candidate]) => candidate === pointer);
    if (index >= 0) this.blocks.splice(index, 1);
  }

  release(): void {
    for (let index = this.blocks.length - 1; index >= 0; index -= 1) this.engine.release(...this.blocks[index]!);
    this.blocks.length = 0;
  }
}

/** Copy ``count`` matrices ``[rows, cols]`` from ``source`` into kernel memory with row stride ``ld``. */
function stage(engine: Engine, source: Float32Array, offset: number, count: number, rows: number, cols: number, ld: number, pointer: number): void {
  const heap = engine.heap;
  const base = pointer >>> 2;
  const total = count * rows;
  if (ld === cols) {
    heap.set(source.subarray(offset, offset + total * cols), base);
    return;
  }
  for (let row = 0; row < total; row += 1) {
    const target = base + row * ld;
    heap.set(source.subarray(offset + row * cols, offset + (row + 1) * cols), target);
    heap.fill(0, target + cols, target + ld);
  }
}

/** Transpose ``count`` matrices ``[rows, cols]`` (row stride ``srcLd``) into ``[cols, dstLd]``, zero padding. */
function transpose(engine: Engine, src: number, dst: number, count: number, rows: number, cols: number, srcLd: number, dstLd: number): void {
  const width = 16;
  const blocks = Math.ceil(cols / width);
  engine.run(Kernel.Transpose, [src, dst, rows, cols, srcLd, dstLd, rows * srcLd, cols * dstLd, blocks, width],
    count * blocks, parallelTasks(engine, count * rows * cols / 4) > 1);
}

interface GemmOperands {
  a: number;
  lda: number;
  b: number;
  ldb: number;
  c: number;
  ldc: number;
  bias: number;
  m: number;
  n: number;
  kp: number;
  batch: number;
  aOffsets: number;
  aStride: number;
  bOffsets: number;
  bStride: number;
  cStride: number;
}

/** ``C[batch] = A[batch] @ B[batch]^T (+ bias)`` over K-contiguous, padded operands. */
function gemm(engine: Engine, g: GemmOperands): void {
  const target = parallelTasks(engine, g.batch * g.m * g.n * g.kp);
  // A block of B rows stays in cache while A rows stream past it.
  let nb = Math.max(8, Math.min(256, Math.floor(16384 / g.kp) & ~7));
  let mb = 64;
  const count = () => g.batch * Math.ceil(g.m / mb) * Math.ceil(g.n / nb);
  if (target > 1) {
    while (count() < target && nb > 8) nb = Math.max(8, (nb >> 1) & ~7);
    while (count() < target && mb > 4) mb = Math.max(4, (mb >> 1) & ~3);
  }
  while (count() > MAX_TASKS) {
    mb *= 2;
    nb *= 2;
  }
  const tilesM = Math.ceil(g.m / mb);
  const tilesN = Math.ceil(g.n / nb);
  engine.run(Kernel.GemmNT, [
    g.a, g.lda, g.b, g.ldb, g.c, g.ldc, g.bias, g.m, g.n, g.kp, mb, nb, tilesM, tilesN,
    g.aOffsets, g.aStride, g.bOffsets, g.bStride, g.cStride,
  ], g.batch * tilesM * tilesN, target > 1);
}

function read(engine: Engine, pointer: number, length: number): Float32Array {
  return engine.copyOut(pointer, length);
}

/** A weight operand that may stay resident between calls. */
export interface WeightSource {
  data: Float32Array;
  /** Storage identity (``tensor._storage``) and its mutation counter. */
  key: object | null;
  version: number;
}

/**
 * Products with at most this many input rows keep their weight resident in
 * kernel memory: staging the weight would cost as much as the product itself
 * (token-by-token decoding). Larger products stage it per call.
 */
const RESIDENT_ROWS = 64;

/** Stage ``[rows, cols]`` ``weight`` as ``[rows, pad4(cols)]``, from the cache when resident. */
function stageWeight(engine: Engine, scratch: Scratch, weight: WeightSource, rows: number, cols: number, inputRows: number): number {
  const ld = pad4(cols);
  const layout = `rows:${ld}`;
  if (weight.key) {
    const pointer = engine.cached(weight.key, weight.version, weight.data, layout);
    if (pointer) return pointer;
  }
  const bytes = rows * ld * 4;
  const pointer = scratch.alloc(bytes);
  if (!pointer) return 0;
  stage(engine, weight.data, 0, 1, rows, cols, ld, pointer);
  if (weight.key && inputRows <= RESIDENT_ROWS && engine.shouldCache(weight.key, weight.version, bytes)) {
    scratch.keep(pointer);
    engine.remember(weight.key, weight.version, weight.data, layout, pointer, bytes);
  }
  return pointer;
}

/** ``x [rows, k] @ weight[n, k]^T (+ bias[n])``. */
export function linearForward(
  x: Float32Array, rows: number, k: number, weight: WeightSource, n: number, bias: Float32Array | null,
): Float32Array | null {
  const engine = activeEngine();
  if (!engine || rows === 0 || n === 0 || k === 0) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const kp = pad4(k);
    const w = stageWeight(engine, scratch, weight, n, k, rows);
    const a = scratch.alloc(rows * kp * 4);
    const b = bias ? scratch.alloc(n * 4) : 0;
    const c = scratch.alloc(rows * n * 4);
    if (!w || scratch.failed) return null;
    stage(engine, x, 0, 1, rows, k, kp, a);
    if (bias) engine.heap.set(bias.subarray(0, n), b >>> 2);
    gemm(engine, { a, lda: kp, b: w, ldb: kp, c, ldc: n, bias: b, m: rows, n, kp, batch: 1, aOffsets: 0, aStride: 0, bOffsets: 0, bStride: 0, cStride: 0 });
    return read(engine, c, rows * n);
  } finally {
    scratch.release();
  }
}

/** ``grad [rows, n] @ weight [n, k]`` (the input gradient of ``linear``). */
export function linearBackwardInput(grad: Float32Array, rows: number, n: number, weight: Float32Array, k: number): Float32Array | null {
  const engine = activeEngine();
  if (!engine || rows === 0 || n === 0 || k === 0) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const np = pad4(n);
    const raw = scratch.alloc(n * k * 4);
    const wt = scratch.alloc(k * np * 4);
    const a = scratch.alloc(rows * np * 4);
    const c = scratch.alloc(rows * k * 4);
    if (scratch.failed) return null;
    stage(engine, weight, 0, 1, n, k, k, raw);
    transpose(engine, raw, wt, 1, n, k, k, np);
    stage(engine, grad, 0, 1, rows, n, np, a);
    gemm(engine, { a, lda: np, b: wt, ldb: np, c, ldc: k, bias: 0, m: rows, n: k, kp: np, batch: 1, aOffsets: 0, aStride: 0, bOffsets: 0, bStride: 0, cStride: 0 });
    return read(engine, c, rows * k);
  } finally {
    scratch.release();
  }
}

/** ``grad [rows, n]^T @ x [rows, k]`` (the weight gradient of ``linear``). */
export function linearBackwardWeight(grad: Float32Array, rows: number, n: number, x: Float32Array, k: number): Float32Array | null {
  const engine = activeEngine();
  if (!engine || rows === 0 || n === 0 || k === 0) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const rp = pad4(rows);
    const rawG = scratch.alloc(rows * n * 4);
    const rawX = scratch.alloc(rows * k * 4);
    const gt = scratch.alloc(n * rp * 4);
    const xt = scratch.alloc(k * rp * 4);
    const c = scratch.alloc(n * k * 4);
    if (scratch.failed) return null;
    stage(engine, grad, 0, 1, rows, n, n, rawG);
    stage(engine, x, 0, 1, rows, k, k, rawX);
    transpose(engine, rawG, gt, 1, rows, n, n, rp);
    transpose(engine, rawX, xt, 1, rows, k, k, rp);
    gemm(engine, { a: gt, lda: rp, b: xt, ldb: rp, c, ldc: k, bias: 0, m: n, n: k, kp: rp, batch: 1, aOffsets: 0, aStride: 0, bOffsets: 0, bStride: 0, cStride: 0 });
    return read(engine, c, n * k);
  } finally {
    scratch.release();
  }
}

/** One operand of {@link batchedMatmul}: ``count`` stored matrices of ``[rows, cols]``. */
export interface MatrixBatch {
  data: Float32Array;
  count: number;
  rows: number;
  cols: number;
  /** Use the transpose of each stored matrix. */
  transposed: boolean;
  /** Stored matrix used by each output batch entry. */
  index: ArrayLike<number>;
}

/** ``op(A)[index] @ op(B)[index]`` for every output batch entry: ``[batch, n, m]``. */
export function batchedMatmul(left: MatrixBatch, right: MatrixBatch, batch: number): Float32Array | null {
  const engine = activeEngine();
  if (!engine || batch === 0) return null;
  const n = left.transposed ? left.cols : left.rows;
  const k = left.transposed ? left.rows : left.cols;
  const m = right.transposed ? right.rows : right.cols;
  if ((right.transposed ? right.cols : right.rows) !== k) throw new RangeError('batched matmul inner dimensions differ');
  if (n === 0 || m === 0 || k === 0) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const kp = pad4(k);
    // A' = op(A) as [count, n, kp]; B' = op(B)^T as [count, m, kp].
    const a = scratch.alloc(left.count * n * kp * 4);
    const b = scratch.alloc(right.count * m * kp * 4);
    const c = scratch.alloc(batch * n * m * 4);
    const identityA = isIdentity(left.index, batch, left.count);
    const identityB = isIdentity(right.index, batch, right.count);
    const aOffsets = identityA ? 0 : scratch.alloc(batch * 4);
    const bOffsets = identityB ? 0 : scratch.alloc(batch * 4);
    const rawA = left.transposed ? scratch.alloc(left.count * left.rows * left.cols * 4) : 0;
    const rawB = right.transposed ? 0 : scratch.alloc(right.count * right.rows * right.cols * 4);
    if (scratch.failed) return null;
    if (left.transposed) {
      stage(engine, left.data, 0, left.count, left.rows, left.cols, left.cols, rawA);
      transpose(engine, rawA, a, left.count, left.rows, left.cols, left.cols, kp);
    } else {
      stage(engine, left.data, 0, left.count, n, k, kp, a);
    }
    if (right.transposed) {
      stage(engine, right.data, 0, right.count, m, k, kp, b);
    } else {
      stage(engine, right.data, 0, right.count, right.rows, right.cols, right.cols, rawB);
      transpose(engine, rawB, b, right.count, right.rows, right.cols, right.cols, kp);
    }
    const u32 = engine.heapU32;
    if (aOffsets) for (let index = 0; index < batch; index += 1) u32[(aOffsets >>> 2) + index] = left.index[index]! * n * kp;
    if (bOffsets) for (let index = 0; index < batch; index += 1) u32[(bOffsets >>> 2) + index] = right.index[index]! * m * kp;
    gemm(engine, {
      a, lda: kp, b, ldb: kp, c, ldc: m, bias: 0, m: n, n: m, kp, batch,
      aOffsets, aStride: n * kp, bOffsets, bStride: m * kp, cStride: n * m,
    });
    return read(engine, c, batch * n * m);
  } finally {
    scratch.release();
  }
}

function isIdentity(index: ArrayLike<number>, batch: number, count: number): boolean {
  if (count !== batch) return false;
  for (let position = 0; position < batch; position += 1) if (index[position] !== position) return false;
  return true;
}

/** Softmax over the last dimension of ``[rows, n]``. */
export function softmaxRows(x: Float32Array, rows: number, n: number): Float32Array | null {
  const engine = activeEngine();
  if (!engine || rows === 0 || n === 0) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const pointer = scratch.alloc(rows * n * 4);
    if (!pointer) return null;
    engine.heap.set(x.subarray(0, rows * n), pointer >>> 2);
    const tasks = parallelTasks(engine, rows * n * 8);
    const per = Math.max(1, Math.ceil(rows / Math.max(tasks, 1)));
    engine.run(Kernel.Softmax, [pointer, rows, n, n, per], Math.ceil(rows / per), tasks > 1);
    return read(engine, pointer, rows * n);
  } finally {
    scratch.release();
  }
}

/** Additive attention bias broadcast over ``[B, H, Lq, Lk]`` with keys contiguous. */
export interface AttentionBias {
  data: Float32Array;
  batchStride: number;
  headStride: number;
  queryStride: number;
}

/**
 * ``softmax(q k^T * scale + bias) v`` for ``q [B, H, Lq, D]``, ``k``
 * ``[B, Hkv, Lk, D]`` and ``v [B, Hkv, Lk, Dv]`` (``H`` a multiple of ``Hkv``):
 * ``[B, H, Lq, Dv]``.
 */
export function attentionForward(
  q: Float32Array, k: Float32Array, v: Float32Array,
  batch: number, heads: number, kvHeads: number, queries: number, keys: number, dim: number, valueDim: number,
  scale: number, bias: AttentionBias | null,
): Float32Array | null {
  const engine = activeEngine();
  if (!engine || batch * heads * queries === 0 || keys === 0 || dim === 0 || valueDim === 0) return null;
  if (heads % kvHeads !== 0) throw new RangeError('attention heads must be a multiple of key/value heads');
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const dp = pad4(dim);
    const kp = pad4(keys);
    const parallel = parallelTasks(engine, batch * heads * queries * keys * (dim + valueDim)) > 1;
    const threads = parallel ? engine.parallelism() : 1;
    const qb = Math.max(1, Math.min(queries, 32));
    const blocks = Math.ceil(queries / qb);
    const qPointer = scratch.alloc(batch * heads * queries * dp * 4);
    const kPointer = scratch.alloc(batch * kvHeads * keys * dp * 4);
    const vRaw = scratch.alloc(batch * kvHeads * keys * valueDim * 4);
    const vt = scratch.alloc(batch * kvHeads * valueDim * kp * 4);
    const out = scratch.alloc(batch * heads * queries * valueDim * 4);
    const biasPointer = bias ? scratch.alloc(bias.data.length * 4) : 0;
    const scoreBytes = qb * kp * 4;
    const scores = scratch.alloc(threads * scoreBytes);
    if (scratch.failed) return null;
    stage(engine, q, 0, 1, batch * heads * queries, dim, dp, qPointer);
    stage(engine, k, 0, 1, batch * kvHeads * keys, dim, dp, kPointer);
    stage(engine, v, 0, 1, batch * kvHeads * keys, valueDim, valueDim, vRaw);
    transpose(engine, vRaw, vt, batch * kvHeads, keys, valueDim, valueDim, kp);
    if (bias) engine.heap.set(bias.data, biasPointer >>> 2);
    const scaleBits = new Uint32Array(new Float32Array([scale]).buffer)[0]!;
    engine.run(Kernel.Attention, [
      qPointer, kPointer, vt, out, biasPointer,
      bias?.batchStride ?? 0, bias?.headStride ?? 0, bias?.queryStride ?? 0,
      batch, heads, kvHeads, queries, keys, dp, kp, valueDim, scaleBits, qb, scores, scoreBytes, blocks,
    ], batch * heads * blocks, parallel);
    return read(engine, out, batch * heads * queries * valueDim);
  } finally {
    scratch.release();
  }
}

/** Elementwise activations computed exactly as their JavaScript counterparts (float64, then rounded). */
export const enum UnaryOp {
  GeluTanh = 0,
  GeluErf = 1,
  Sigmoid = 2,
  Silu = 3,
  Tanh = 4,
  Exp = 5,
  GeluTanhDerivative = 6,
  GeluErfDerivative = 7,
}

/** Relative cost of one element of each {@link UnaryOp} in multiply-adds (for task sizing). */
const UNARY_COST = [48, 160, 32, 32, 40, 24, 48, 180];

/** ``op`` applied to every element of ``x``. */
export function unary(op: UnaryOp, x: Float32Array): Float32Array | null {
  const engine = activeEngine();
  const n = x.length;
  if (!engine || n === 0) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const src = scratch.alloc(n * 4);
    const dst = scratch.alloc(n * 4);
    if (scratch.failed) return null;
    engine.heap.set(x, src >>> 2);
    const tasks = parallelTasks(engine, n * UNARY_COST[op]!);
    const per = Math.max(1024, Math.ceil(n / tasks));
    engine.run(Kernel.Unary, [src, dst, n, per, op], Math.ceil(n / per), tasks > 1);
    return read(engine, dst, n);
  } finally {
    scratch.release();
  }
}

/**
 * Layer normalization of ``[rows, width]`` (float64 statistics, bit-identical
 * to the JavaScript ``layerNorm``).
 */
export function layerNormRows(
  x: Float32Array, rows: number, width: number, weight: Float32Array | null, bias: Float32Array | null, eps: number,
): Float32Array | null {
  const engine = activeEngine();
  if (!engine || rows === 0 || width === 0) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const src = scratch.alloc(rows * width * 4);
    const dst = scratch.alloc(rows * width * 4);
    const w = weight ? scratch.alloc(width * 4) : 0;
    const b = bias ? scratch.alloc(width * 4) : 0;
    if (scratch.failed) return null;
    const heap = engine.heap;
    heap.set(x.subarray(0, rows * width), src >>> 2);
    if (weight) heap.set(weight.subarray(0, width), w >>> 2);
    if (bias) heap.set(bias.subarray(0, width), b >>> 2);
    const bits = new Uint32Array(new Float64Array([eps]).buffer);
    const tasks = parallelTasks(engine, rows * width * 8);
    const per = Math.max(1, Math.ceil(rows / tasks));
    engine.run(Kernel.LayerNorm, [src, dst, rows, width, w, b, bits[0]!, bits[1]!, per], Math.ceil(rows / per), tasks > 1);
    return read(engine, dst, rows * width);
  } finally {
    scratch.release();
  }
}

/** Tensors smaller than this (elements) keep data-movement ops in JavaScript. */
const MOVE_MIN = 1 << 20;

/**
 * Swap two adjacent axes of a contiguous ``[count, rows, cols, block]``
 * float32 array (``[count, cols, rows, block]``); a pure copy.
 */
export function swapAxes(x: Float32Array, count: number, rows: number, cols: number, block: number): Float32Array | null {
  const engine = activeEngine();
  const n = count * rows * cols * block;
  if (!engine || n < MOVE_MIN || engine.parallelism() <= 1) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const src = scratch.alloc(n * 4);
    const dst = scratch.alloc(n * 4);
    if (scratch.failed) return null;
    engine.heap.set(x.subarray(0, n), src >>> 2);
    const perBatch = Math.max(1, Math.ceil(cols / Math.max(1, Math.ceil(parallelTasks(engine, n) / count))));
    const tasks = count * Math.ceil(cols / perBatch);
    engine.run(Kernel.SwapAxes, [src, dst, count, rows, cols, block, perBatch], tasks, tasks > 1);
    return read(engine, dst, n);
  } finally {
    scratch.release();
  }
}

export const enum BinaryOp {
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
}

/** Float32 ``a op b`` for equal lengths, or a one-element ``b``. */
export function binary(op: BinaryOp, a: Float32Array, b: Float32Array): Float32Array | null {
  const engine = activeEngine();
  const n = a.length;
  if (!engine || n < MOVE_MIN || engine.parallelism() <= 1 || (b.length !== n && b.length !== 1)) return null;
  engine.beginCall();
  const scratch = new Scratch(engine);
  try {
    const pa = scratch.alloc(n * 4);
    const pb = scratch.alloc(b.length * 4);
    const out = scratch.alloc(n * 4);
    if (scratch.failed) return null;
    const heap = engine.heap;
    heap.set(a, pa >>> 2);
    heap.set(b, pb >>> 2);
    const tasks = parallelTasks(engine, n);
    const per = Math.max(4096, Math.ceil(n / tasks / 4) * 4);
    engine.run(Kernel.Binary, [pa, pb, out, n, per, op, b.length === n ? 1 : 0], Math.ceil(n / per), true);
    return read(engine, out, n);
  } finally {
    scratch.release();
  }
}
