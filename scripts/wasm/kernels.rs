//! TensorCode WebAssembly SIMD kernels (float32).
//!
//! Built by `scripts/wasm/build.mjs` into `src/nn/backend/kernels.generated.ts`
//! (base64). Two variants are produced from this source: baseline SIMD128 and
//! relaxed SIMD (`--cfg relaxed`, fused multiply-add where the CPU has one).
//!
//! Every entry point has the signature `task_*(args, task, thread)`: `args`
//! points at a little-endian `u32` argument block in linear memory, `task` is
//! the index of the work item and `thread` the calling thread's index. Work
//! items are independent, so any number of threads can run them in any order.
//!
//! Numerics: every output element of a product is computed by the same
//! arithmetic regardless of the operand shapes, tile position or thread count
//! (four lane-wise float32 partial sums over the padded reduction dimension,
//! combined as `(l0 + l2) + (l1 + l3)`), so results are batch invariant and
//! deterministic. Reduction dimensions are zero padded to a multiple of four
//! by the caller.
#![no_std]

use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unreachable()
}

#[inline(always)]
fn madd(a: v128, b: v128, c: v128) -> v128 {
    #[cfg(relaxed)]
    {
        f32x4_relaxed_madd(a, b, c)
    }
    #[cfg(not(relaxed))]
    {
        f32x4_add(f32x4_mul(a, b), c)
    }
}

#[inline(always)]
fn hsum(v: v128) -> f32 {
    let s = f32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    f32x4_extract_lane::<0>(s) + f32x4_extract_lane::<1>(s)
}

#[inline(always)]
unsafe fn arg(args: *const u32, index: usize) -> usize {
    *args.add(index) as usize
}

#[inline(always)]
unsafe fn load(p: *const f32) -> v128 {
    v128_load(p as *const v128)
}

#[inline(always)]
unsafe fn store(p: *mut f32, v: v128) {
    v128_store(p as *mut v128, v)
}

/// `MR x NR` dot products of `a` rows against `b` rows over `kp` (multiple of 4).
#[inline(always)]
unsafe fn dots<const MR: usize, const NR: usize>(
    a: *const f32, lda: usize, b: *const f32, ldb: usize, kp: usize,
) -> [[f32; NR]; MR] {
    let mut acc = [[f32x4_splat(0.0); NR]; MR];
    let mut p = 0;
    while p < kp {
        let mut bv = [f32x4_splat(0.0); NR];
        let mut j = 0;
        while j < NR {
            bv[j] = load(b.add(j * ldb + p));
            j += 1;
        }
        let mut i = 0;
        while i < MR {
            let av = load(a.add(i * lda + p));
            let mut j = 0;
            while j < NR {
                acc[i][j] = madd(av, bv[j], acc[i][j]);
                j += 1;
            }
            i += 1;
        }
        p += 4;
    }
    let mut out = [[0.0f32; NR]; MR];
    let mut i = 0;
    while i < MR {
        let mut j = 0;
        while j < NR {
            out[i][j] = hsum(acc[i][j]);
            j += 1;
        }
        i += 1;
    }
    out
}

/// `value + bias[j]`, or `value` when `bias` is null.
#[inline(always)]
unsafe fn finish(value: f32, j: usize, bias: *const f32) -> f32 {
    if bias.is_null() { value } else { value + *bias.add(j) }
}

#[inline(always)]
unsafe fn block<const MR: usize, const NR: usize>(
    a: *const f32, lda: usize, b: *const f32, ldb: usize, c: *mut f32, ldc: usize, kp: usize,
    i: usize, j: usize, bias: *const f32,
) {
    let out = dots::<MR, NR>(a.add(i * lda), lda, b.add(j * ldb), ldb, kp);
    let mut r = 0;
    while r < MR {
        let mut s = 0;
        while s < NR {
            *c.add((i + r) * ldc + j + s) = finish(out[r][s], j + s, bias);
            s += 1;
        }
        r += 1;
    }
}

/// `C[i, j] = dot(A[i, :kp], B[j, :kp]) (+ bias[j])` for rows `m0..m1`, columns `n0..n1`.
#[inline(always)]
unsafe fn gemm_nt_tile(
    a: *const f32, lda: usize, b: *const f32, ldb: usize, c: *mut f32, ldc: usize, kp: usize,
    m0: usize, m1: usize, n0: usize, n1: usize, bias: *const f32,
) {
    let mut i = m0;
    while i + 4 <= m1 {
        let mut j = n0;
        while j + 4 <= n1 {
            block::<4, 4>(a, lda, b, ldb, c, ldc, kp, i, j, bias);
            j += 4;
        }
        while j < n1 {
            block::<4, 1>(a, lda, b, ldb, c, ldc, kp, i, j, bias);
            j += 1;
        }
        i += 4;
    }
    while i < m1 {
        let mut j = n0;
        while j + 8 <= n1 {
            block::<1, 8>(a, lda, b, ldb, c, ldc, kp, i, j, bias);
            j += 8;
        }
        while j < n1 {
            block::<1, 1>(a, lda, b, ldb, c, ldc, kp, i, j, bias);
            j += 1;
        }
        i += 1;
    }
}

/// Batched `C = A @ B^T (+ bias)`.
///
/// args: 0 a, 1 lda, 2 b, 3 ldb, 4 c, 5 ldc, 6 bias (0: none), 7 M, 8 N, 9 Kp,
/// 10 tile rows, 11 tile columns, 12 tiles along M, 13 tiles along N,
/// 14 A batch offsets (u32 element offsets, 0: use 15), 15 A batch stride,
/// 16 B batch offsets (0: use 17), 17 B batch stride, 18 C batch stride.
#[no_mangle]
pub unsafe extern "C" fn task_gemm_nt(args: *const u32, task: u32, _thread: u32) {
    let task = task as usize;
    let tiles_m = arg(args, 12);
    let tiles_n = arg(args, 13);
    let per_batch = tiles_m * tiles_n;
    if tiles_n == 0 || per_batch == 0 {
        return;
    }
    let batch = task / per_batch;
    let rest = task % per_batch;
    let ti = rest / tiles_n;
    let tj = rest % tiles_n;
    let m = arg(args, 7);
    let n = arg(args, 8);
    let mb = arg(args, 10);
    let nb = arg(args, 11);
    let a_offsets = arg(args, 14) as *const u32;
    let b_offsets = arg(args, 16) as *const u32;
    let a_off = if a_offsets.is_null() { batch * arg(args, 15) } else { *a_offsets.add(batch) as usize };
    let b_off = if b_offsets.is_null() { batch * arg(args, 17) } else { *b_offsets.add(batch) as usize };
    let a = (arg(args, 0) as *const f32).add(a_off);
    let b = (arg(args, 2) as *const f32).add(b_off);
    let c = (arg(args, 4) as *mut f32).add(batch * arg(args, 18));
    let m0 = ti * mb;
    let n0 = tj * nb;
    let m1 = if m0 + mb < m { m0 + mb } else { m };
    let n1 = if n0 + nb < n { n0 + nb } else { n };
    gemm_nt_tile(a, arg(args, 1), b, arg(args, 3), c, arg(args, 5), arg(args, 9), m0, m1, n0, n1,
                 arg(args, 6) as *const f32);
}

/// Batched transpose with zero padding: `dst[c, r] = src[r, c]` for
/// `r < rows`, and `dst[c, r] = 0` for `rows <= r < dst_ld`.
///
/// args: 0 src, 1 dst, 2 rows, 3 columns, 4 src_ld, 5 dst_ld, 6 src batch
/// stride, 7 dst batch stride, 8 column blocks per batch, 9 column block size.
#[no_mangle]
pub unsafe extern "C" fn task_transpose(args: *const u32, task: u32, _thread: u32) {
    let task = task as usize;
    let blocks = arg(args, 8);
    let width = arg(args, 9);
    if blocks == 0 {
        return;
    }
    let batch = task / blocks;
    let block = task % blocks;
    let rows = arg(args, 2);
    let columns = arg(args, 3);
    let src_ld = arg(args, 4);
    let dst_ld = arg(args, 5);
    let src = (arg(args, 0) as *const f32).add(batch * arg(args, 6));
    let dst = (arg(args, 1) as *mut f32).add(batch * arg(args, 7));
    let c0 = block * width;
    let c1 = if c0 + width < columns { c0 + width } else { columns };
    let mut r = 0;
    while r < rows {
        let row = src.add(r * src_ld);
        let mut c = c0;
        while c < c1 {
            *dst.add(c * dst_ld + r) = *row.add(c);
            c += 1;
        }
        r += 1;
    }
    let mut c = c0;
    while c < c1 {
        let mut r = rows;
        while r < dst_ld {
            *dst.add(c * dst_ld + r) = 0.0;
            r += 1;
        }
        c += 1;
    }
}

/// Vectorized `exp` (Cephes `expf`, about 1 ulp); inputs below -87.34 give 0,
/// above 88 give +inf, and NaN propagates.
#[inline(always)]
fn exp4(x: v128) -> v128 {
    let lo = f32x4_splat(-87.336_548);
    let hi = f32x4_splat(88.0);
    let xc = f32x4_min(f32x4_max(x, lo), hi);
    let fx = f32x4_nearest(f32x4_mul(xc, f32x4_splat(1.442_695_04)));
    let r = f32x4_sub(f32x4_sub(xc, f32x4_mul(fx, f32x4_splat(0.693_359_375))), f32x4_mul(fx, f32x4_splat(-2.121_944_4e-4)));
    let r2 = f32x4_mul(r, r);
    let mut y = f32x4_splat(1.987_569_15e-4);
    y = f32x4_add(f32x4_mul(y, r), f32x4_splat(1.398_199_950_7e-3));
    y = f32x4_add(f32x4_mul(y, r), f32x4_splat(8.333_451_907_3e-3));
    y = f32x4_add(f32x4_mul(y, r), f32x4_splat(4.166_579_589_4e-2));
    y = f32x4_add(f32x4_mul(y, r), f32x4_splat(1.666_666_545_9e-1));
    y = f32x4_add(f32x4_mul(y, r), f32x4_splat(5.000_000_120_1e-1));
    y = f32x4_add(f32x4_add(f32x4_mul(y, r2), r), f32x4_splat(1.0));
    let n = i32x4_trunc_sat_f32x4(fx);
    let scale = i32x4_shl(i32x4_add(n, i32x4_splat(127)), 23);
    let mut result = f32x4_mul(y, scale);
    result = v128_andnot(result, f32x4_lt(x, lo));
    v128_bitselect(f32x4_splat(f32::INFINITY), result, f32x4_gt(x, hi))
}

#[inline(always)]
fn exp1(x: f32) -> f32 {
    f32x4_extract_lane::<0>(exp4(f32x4_splat(x)))
}

/// In-place softmax of `row[..n]`, zeroing `row[n..padded]`. A row whose
/// maximum is -inf becomes NaN (PyTorch); NaN propagates.
#[inline(always)]
unsafe fn softmax_row(row: *mut f32, n: usize, padded: usize) {
    let mut vmax = f32x4_splat(f32::NEG_INFINITY);
    let mut j = 0;
    while j + 4 <= n {
        vmax = f32x4_max(vmax, load(row.add(j)));
        j += 4;
    }
    let mut maximum = f32x4_extract_lane::<0>(vmax);
    maximum = fmax(maximum, f32x4_extract_lane::<1>(vmax));
    maximum = fmax(maximum, f32x4_extract_lane::<2>(vmax));
    maximum = fmax(maximum, f32x4_extract_lane::<3>(vmax));
    while j < n {
        maximum = fmax(maximum, *row.add(j));
        j += 1;
    }
    if maximum == f32::NEG_INFINITY {
        let mut j = 0;
        while j < n {
            *row.add(j) = f32::NAN;
            j += 1;
        }
    } else {
        let vm = f32x4_splat(maximum);
        let mut vsum = f32x4_splat(0.0);
        let mut j = 0;
        while j + 4 <= n {
            let e = exp4(f32x4_sub(load(row.add(j)), vm));
            store(row.add(j), e);
            vsum = f32x4_add(vsum, e);
            j += 4;
        }
        let mut total = hsum(vsum);
        while j < n {
            let e = exp1(*row.add(j) - maximum);
            *row.add(j) = e;
            total += e;
            j += 1;
        }
        let vt = f32x4_splat(total);
        let mut j = 0;
        while j + 4 <= n {
            store(row.add(j), f32x4_div(load(row.add(j)), vt));
            j += 4;
        }
        while j < n {
            *row.add(j) /= total;
            j += 1;
        }
    }
    let mut j = n;
    while j < padded {
        *row.add(j) = 0.0;
        j += 1;
    }
}

/// NaN-propagating maximum (wasm `f32.max`).
#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    f32x4_extract_lane::<0>(f32x4_max(f32x4_splat(a), f32x4_splat(b)))
}

/// Softmax over contiguous rows. args: 0 data, 1 rows, 2 n, 3 row stride,
/// 4 rows per task.
#[no_mangle]
pub unsafe extern "C" fn task_softmax(args: *const u32, task: u32, _thread: u32) {
    let rows = arg(args, 1);
    let n = arg(args, 2);
    let ld = arg(args, 3);
    let per = arg(args, 4);
    let r0 = task as usize * per;
    let r1 = if r0 + per < rows { r0 + per } else { rows };
    let data = arg(args, 0) as *mut f32;
    let mut r = r0;
    while r < r1 {
        softmax_row(data.add(r * ld), n, n);
        r += 1;
    }
}

/// Fused attention `softmax(Q K^T * scale + bias) V` for one query block of one head.
///
/// args: 0 q [B, H, Lq, Dp], 1 k [B, Hkv, Lk, Dp], 2 v^T [B, Hkv, D, Lkp],
/// 3 out [B, H, Lq, D], 4 bias (0: none), 5-7 bias strides for batch, head and
/// query (elements, keys contiguous), 8 B, 9 H, 10 Hkv, 11 Lq, 12 Lk, 13 Dp,
/// 14 Lkp, 15 D, 16 scale (f32 bits), 17 query block, 18 scratch, 19 scratch
/// bytes per thread, 20 query blocks per head.
#[no_mangle]
pub unsafe extern "C" fn task_attention(args: *const u32, task: u32, thread: u32) {
    let task = task as usize;
    let heads = arg(args, 9);
    let kv_heads = arg(args, 10);
    let lq = arg(args, 11);
    let lk = arg(args, 12);
    let dp = arg(args, 13);
    let lkp = arg(args, 14);
    let d = arg(args, 15);
    let scale = f32::from_bits(*args.add(16));
    let qb = arg(args, 17);
    let blocks = arg(args, 20);
    if heads == 0 || kv_heads == 0 || blocks == 0 || heads / kv_heads == 0 {
        return;
    }
    let bh = task / blocks;
    let block = task % blocks;
    let b = bh / heads;
    let h = bh % heads;
    let hk = h / (heads / kv_heads);
    let q0 = block * qb;
    let q1 = if q0 + qb < lq { q0 + qb } else { lq };
    let rows = q1 - q0;
    let q = (arg(args, 0) as *const f32).add((bh * lq + q0) * dp);
    let k = (arg(args, 1) as *const f32).add((b * kv_heads + hk) * lk * dp);
    let vt = (arg(args, 2) as *const f32).add((b * kv_heads + hk) * d * lkp);
    let out = (arg(args, 3) as *mut f32).add((bh * lq + q0) * d);
    let scores = (arg(args, 18) + thread as usize * arg(args, 19)) as *mut f32;
    let bias_base = arg(args, 4) as *const f32;
    let (sb, sh, sq) = (arg(args, 5), arg(args, 6), arg(args, 7));
    gemm_nt_tile(q, dp, k, dp, scores, lkp, dp, 0, rows, 0, lk, core::ptr::null());
    let vscale = f32x4_splat(scale);
    let mut r = 0;
    while r < rows {
        let row = scores.add(r * lkp);
        let mut j = 0;
        if bias_base.is_null() {
            while j + 4 <= lk {
                store(row.add(j), f32x4_mul(load(row.add(j)), vscale));
                j += 4;
            }
            while j < lk {
                *row.add(j) *= scale;
                j += 1;
            }
        } else {
            let bias = bias_base.add(b * sb + h * sh + (q0 + r) * sq);
            while j + 4 <= lk {
                store(row.add(j), f32x4_add(f32x4_mul(load(row.add(j)), vscale), v128_load(bias.add(j) as *const v128)));
                j += 4;
            }
            while j < lk {
                *row.add(j) = *row.add(j) * scale + *bias.add(j);
                j += 1;
            }
        }
        softmax_row(row, lk, lkp);
        r += 1;
    }
    gemm_nt_tile(scores, lkp, vt, lkp, out, d, lkp, 0, rows, 0, d, core::ptr::null());
}

// ---------------------------------------------------------------------------
// Elementwise activations in float64, bit-identical to the JavaScript
// implementations: V8's `Math.exp`, `Math.expm1` and `Math.tanh` are the
// fdlibm algorithms ported below, and `erf` follows `erfValue` in
// `src/nn/ops/elementwise.ts` operation for operation.

#[inline(always)]
fn high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline(always)]
fn low_word(x: f64) -> u32 {
    x.to_bits() as u32
}

#[inline(always)]
fn from_words(high: u32, low: u32) -> f64 {
    f64::from_bits(((high as u64) << 32) | low as u64)
}

#[inline(always)]
fn with_high_word(x: f64, high: u32) -> f64 {
    f64::from_bits(((high as u64) << 32) | (x.to_bits() & 0xffff_ffff))
}

/// fdlibm `__ieee754_exp`.
fn exp_f64(mut x: f64) -> f64 {
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const U_THRESHOLD: f64 = -7.45133219101941108420e+02;
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const INVLN2: f64 = 1.44269504088896338700e+00;
    const P1: f64 = 1.66666666666666019037e-01;
    const P2: f64 = -2.77777777770155933842e-03;
    const P3: f64 = 6.61375632143793436117e-05;
    const P4: f64 = -1.65339022054652515390e-06;
    const P5: f64 = 4.13813679705723846039e-08;
    const TWOM1000: f64 = 9.33263618503218878990e-302;
    let mut hi = 0.0;
    let mut lo = 0.0;
    let k: i32;
    let mut hx = high_word(x);
    let xsb = ((hx >> 31) & 1) as i32;
    hx &= 0x7fff_ffff;
    if hx >= 0x4086_2E42 {
        if hx >= 0x7ff0_0000 {
            if ((hx & 0xfffff) | low_word(x)) != 0 {
                return x + x;
            }
            return if xsb == 0 { x } else { 0.0 };
        }
        if x > O_THRESHOLD {
            return f64::INFINITY;
        }
        if x < U_THRESHOLD {
            return 0.0;
        }
    }
    if hx > 0x3fd6_2e42 {
        if hx < 0x3FF0_A2B2 {
            if xsb == 0 {
                hi = x - LN2_HI;
                lo = LN2_LO;
            } else {
                hi = x + LN2_HI;
                lo = -LN2_LO;
            }
            k = 1 - xsb - xsb;
        } else {
            k = (INVLN2 * x + if xsb == 0 { 0.5 } else { -0.5 }) as i32;
            let t = k as f64;
            hi = x - t * LN2_HI;
            lo = t * LN2_LO;
        }
        x = hi - lo;
    } else if hx < 0x3e30_0000 {
        return 1.0 + x;
    } else {
        k = 0;
    }
    let t = x * x;
    let twopk = if k >= -1021 {
        from_words((0x3ff0_0000i32 + (k << 20)) as u32, 0)
    } else {
        from_words((0x3ff0_0000i32 + ((k + 1000) << 20)) as u32, 0)
    };
    let c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if k >= -1021 {
        if k == 1024 {
            return y * 2.0 * f64::from_bits(0x7fe0_0000_0000_0000);
        }
        y * twopk
    } else {
        y * twopk * TWOM1000
    }
}

/// fdlibm `expm1`.
fn expm1_f64(mut x: f64) -> f64 {
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const INVLN2: f64 = 1.44269504088896338700e+00;
    const Q1: f64 = -3.33333333333331316428e-02;
    const Q2: f64 = 1.58730158725481460165e-03;
    const Q3: f64 = -7.93650757867487942473e-05;
    const Q4: f64 = 4.00821782732936239552e-06;
    const Q5: f64 = -2.01099218183624371326e-07;
    let mut hx = high_word(x);
    let xsb = hx & 0x8000_0000;
    hx &= 0x7fff_ffff;
    if hx >= 0x4043_687A {
        if hx >= 0x4086_2E42 {
            if hx >= 0x7ff0_0000 {
                if ((hx & 0xfffff) | low_word(x)) != 0 {
                    return x + x;
                }
                return if xsb == 0 { x } else { -1.0 };
            }
            if x > O_THRESHOLD {
                return f64::INFINITY;
            }
        }
        if xsb != 0 {
            return -1.0;
        }
    }
    let k: i32;
    let mut c = 0.0;
    if hx > 0x3fd6_2e42 {
        let hi;
        let lo;
        if hx < 0x3FF0_A2B2 {
            if xsb == 0 {
                hi = x - LN2_HI;
                lo = LN2_LO;
                k = 1;
            } else {
                hi = x + LN2_HI;
                lo = -LN2_LO;
                k = -1;
            }
        } else {
            k = (INVLN2 * x + if xsb == 0 { 0.5 } else { -0.5 }) as i32;
            let t = k as f64;
            hi = x - t * LN2_HI;
            lo = t * LN2_LO;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if hx < 0x3c90_0000 {
        return x;
    } else {
        k = 0;
    }
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        return x - (x * e - hxs);
    }
    let twopk = from_words((0x3ff0_0000i32 + (k << 20)) as u32, 0);
    e = x * (e - c) - c;
    e -= hxs;
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        return if x < -0.25 { -2.0 * (e - (x + 0.5)) } else { 1.0 + 2.0 * (x - e) };
    }
    if k <= -2 || k > 56 {
        let mut y = 1.0 - (e - x);
        if k == 1024 {
            y = y * 2.0 * f64::from_bits(0x7fe0_0000_0000_0000);
        } else {
            y *= twopk;
        }
        return y - 1.0;
    }
    let y;
    if k < 20 {
        let t = with_high_word(1.0, (0x3ff0_0000i32 - (0x20_0000i32 >> k)) as u32);
        y = (t - (e - x)) * twopk;
    } else {
        let t = from_words(((0x3ff - k) << 20) as u32, 0);
        y = ((x - (e + t)) + 1.0) * twopk;
    }
    y
}

/// fdlibm `tanh`.
fn tanh_f64(x: f64) -> f64 {
    let jx = high_word(x) as i32;
    let ix = jx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        return if jx >= 0 { 1.0 / x + 1.0 } else { 1.0 / x - 1.0 };
    }
    let z;
    if ix < 0x4036_0000 {
        if ix < 0x3e30_0000 {
            return x;
        }
        if ix >= 0x3ff0_0000 {
            let t = expm1_f64(2.0 * fabs(x));
            z = 1.0 - 2.0 / (t + 2.0);
        } else {
            let t = expm1_f64(-2.0 * fabs(x));
            z = -t / (t + 2.0);
        }
    } else {
        z = 1.0;
    }
    if jx >= 0 { z } else { -z }
}

#[inline(always)]
fn fabs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

#[inline(always)]
fn sqrt_f64(x: f64) -> f64 {
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

/// `erfValue` of `src/nn/ops/elementwise.ts`.
fn erf_f64(value: f64) -> f64 {
    const TWO_OVER_SQRT_PI: f64 = 1.1283791670955126;
    if value != value {
        return value;
    }
    let x = fabs(value);
    let result;
    if x < 2.5 {
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 1;
        while n < 60 {
            term *= -x2 / n as f64;
            let contribution = term / (2 * n + 1) as f64;
            sum += contribution;
            if fabs(contribution) < 1e-17 * fabs(sum) {
                break;
            }
            n += 1;
        }
        result = TWO_OVER_SQRT_PI * sum;
    } else if x > 6.0 {
        result = 1.0;
    } else {
        let tiny = 1e-300;
        let mut f = x;
        let mut c = x;
        let mut d = 0.0;
        let mut n = 1;
        while n < 200 {
            let an = n as f64 / 2.0;
            d = x + an * d;
            d = if fabs(d) < tiny { tiny } else { d };
            c = x + an / c;
            c = if fabs(c) < tiny { tiny } else { c };
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if fabs(delta - 1.0) < 1e-16 {
                break;
            }
            n += 1;
        }
        let erfc = exp_f64(-x * x) / (f * sqrt_f64(core::f64::consts::PI));
        result = 1.0 - erfc;
    }
    if value < 0.0 { -result } else { result }
}

/// `sigmoidValue` of `src/nn/ops/elementwise.ts`.
fn sigmoid_f64(value: f64) -> f64 {
    if value >= 0.0 {
        return 1.0 / (1.0 + exp_f64(-value));
    }
    let e = exp_f64(value);
    e / (1.0 + e)
}

const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
const INV_SQRT_2: f64 = 0.7071067811865476;

const INV_SQRT_2PI: f64 = 0.3989422804014327;

/// Elementwise float32 activation. args: 0 src, 1 dst, 2 n, 3 elements per
/// task, 4 op (0 gelu tanh, 1 gelu erf, 2 sigmoid, 3 silu, 4 tanh, 5 exp,
/// 6 gelu tanh derivative, 7 gelu erf derivative).
#[no_mangle]
pub unsafe extern "C" fn task_unary(args: *const u32, task: u32, _thread: u32) {
    let n = arg(args, 2);
    let per = arg(args, 3);
    let op = arg(args, 4);
    let i0 = task as usize * per;
    let i1 = if i0 + per < n { i0 + per } else { n };
    let src = arg(args, 0) as *const f32;
    let dst = arg(args, 1) as *mut f32;
    let mut i = i0;
    match op {
        0 => while i < i1 {
            let v = *src.add(i) as f64;
            *dst.add(i) = (0.5 * v * (1.0 + tanh_f64(SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))) as f32;
            i += 1;
        },
        1 => while i < i1 {
            let v = *src.add(i) as f64;
            *dst.add(i) = (0.5 * v * (1.0 + erf_f64(v * INV_SQRT_2))) as f32;
            i += 1;
        },
        2 => while i < i1 {
            *dst.add(i) = sigmoid_f64(*src.add(i) as f64) as f32;
            i += 1;
        },
        3 => while i < i1 {
            let v = *src.add(i);
            *dst.add(i) = v * (sigmoid_f64(v as f64) as f32);
            i += 1;
        },
        4 => while i < i1 {
            *dst.add(i) = tanh_f64(*src.add(i) as f64) as f32;
            i += 1;
        },
        5 => while i < i1 {
            *dst.add(i) = exp_f64(*src.add(i) as f64) as f32;
            i += 1;
        },
        6 => while i < i1 {
            let v = *src.add(i) as f64;
            let inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
            let t = tanh_f64(inner);
            let derivative = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * v * v);
            *dst.add(i) = (0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * derivative) as f32;
            i += 1;
        },
        _ => while i < i1 {
            let v = *src.add(i) as f64;
            *dst.add(i) = (0.5 * (1.0 + erf_f64(v * INV_SQRT_2)) + v * INV_SQRT_2PI * exp_f64(-0.5 * v * v)) as f32;
            i += 1;
        },
    }
}

/// Layer normalization over contiguous float32 rows, computed in float64
/// exactly as `layerNorm` in `src/nn/ops/nn.ts`. args: 0 src, 1 dst, 2 rows,
/// 3 width, 4 weight (0: none), 5 bias (0: none), 6-7 eps (f64 bits, low
/// word first), 8 rows per task.
#[no_mangle]
pub unsafe extern "C" fn task_layernorm(args: *const u32, task: u32, _thread: u32) {
    let rows = arg(args, 2);
    let width = arg(args, 3);
    let per = arg(args, 8);
    let eps = f64::from_bits((*args.add(6) as u64) | ((*args.add(7) as u64) << 32));
    let src = arg(args, 0) as *const f32;
    let dst = arg(args, 1) as *mut f32;
    let weight = arg(args, 4) as *const f32;
    let bias = arg(args, 5) as *const f32;
    let r0 = task as usize * per;
    let r1 = if r0 + per < rows { r0 + per } else { rows };
    let widthf = width as f64;
    let mut r = r0;
    while r < r1 {
        let row = src.add(r * width);
        let out = dst.add(r * width);
        let mut total = 0.0f64;
        let mut i = 0;
        while i < width {
            total += *row.add(i) as f64;
            i += 1;
        }
        let average = total / widthf;
        let mut squares = 0.0f64;
        i = 0;
        while i < width {
            let centered = *row.add(i) as f64 - average;
            squares += centered * centered;
            i += 1;
        }
        let inv = 1.0 / sqrt_f64(squares / widthf + eps);
        i = 0;
        while i < width {
            let value = (*row.add(i) as f64 - average) * inv;
            let w = if weight.is_null() { 1.0 } else { *weight.add(i) as f64 };
            let b = if bias.is_null() { 0.0 } else { *bias.add(i) as f64 };
            *out.add(i) = (value * w + b) as f32;
            i += 1;
        }
        r += 1;
    }
}

/// Swap two adjacent axes: `[count, rows, cols, block] -> [count, cols, rows, block]`.
/// args: 0 src, 1 dst, 2 count, 3 rows, 4 cols, 5 block, 6 columns per task.
#[no_mangle]
pub unsafe extern "C" fn task_swap_axes(args: *const u32, task: u32, _thread: u32) {
    let count = arg(args, 2);
    let rows = arg(args, 3);
    let cols = arg(args, 4);
    let block = arg(args, 5);
    let per = arg(args, 6);
    let tasks_per_batch = if per == 0 { 0 } else { (cols + per - 1) / per };
    if tasks_per_batch == 0 {
        return;
    }
    let batch = task as usize / tasks_per_batch;
    if batch >= count {
        return;
    }
    let c0 = (task as usize % tasks_per_batch) * per;
    let c1 = if c0 + per < cols { c0 + per } else { cols };
    let src = (arg(args, 0) as *const f32).add(batch * rows * cols * block);
    let dst = (arg(args, 1) as *mut f32).add(batch * rows * cols * block);
    let mut c = c0;
    while c < c1 {
        let mut r = 0;
        while r < rows {
            let from = src.add((r * cols + c) * block);
            let to = dst.add((c * rows + r) * block);
            let mut i = 0;
            while i + 4 <= block {
                v128_store(to.add(i) as *mut v128, v128_load(from.add(i) as *const v128));
                i += 4;
            }
            while i < block {
                *to.add(i) = *from.add(i);
                i += 1;
            }
            r += 1;
        }
        c += 1;
    }
}

/// Elementwise float32 `a op b` for equal shapes, or a scalar `b` (`b_step`
/// 0). args: 0 a, 1 b, 2 dst, 3 n, 4 elements per task, 5 op (0 add, 1 sub,
/// 2 mul, 3 div), 6 b_step (1 or 0).
#[no_mangle]
pub unsafe extern "C" fn task_binary(args: *const u32, task: u32, _thread: u32) {
    let n = arg(args, 3);
    let per = arg(args, 4);
    let op = arg(args, 5);
    let scalar = arg(args, 6) == 0;
    let i0 = task as usize * per;
    let i1 = if i0 + per < n { i0 + per } else { n };
    let a = arg(args, 0) as *const f32;
    let b = arg(args, 1) as *const f32;
    let dst = arg(args, 2) as *mut f32;
    let bs = f32x4_splat(*b);
    let mut i = i0;
    // Unaligned starts are handled by the scalar loops below.
    while i < i1 && (i & 3) != 0 {
        let y = if scalar { *b } else { *b.add(i) };
        *dst.add(i) = binary1(op, *a.add(i), y);
        i += 1;
    }
    while i + 4 <= i1 {
        let x = load(a.add(i));
        let y = if scalar { bs } else { load(b.add(i)) };
        let z = match op {
            0 => f32x4_add(x, y),
            1 => f32x4_sub(x, y),
            2 => f32x4_mul(x, y),
            _ => f32x4_div(x, y),
        };
        store(dst.add(i), z);
        i += 4;
    }
    while i < i1 {
        let y = if scalar { *b } else { *b.add(i) };
        *dst.add(i) = binary1(op, *a.add(i), y);
        i += 1;
    }
}

#[inline(always)]
fn binary1(op: usize, x: f32, y: f32) -> f32 {
    match op {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        _ => x / y,
    }
}
