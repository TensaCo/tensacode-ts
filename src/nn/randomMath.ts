/**
 * Bit-exact C math library functions, shared by the PyTorch-compatible
 * samplers (``random.ts``) and the torchvision/Pillow resampling filters
 * (``_internal/image/resample.ts``, ``_internal/vec/imageProcessing.ts``).
 *
 * PyTorch's CPU samplers and resize kernels and Pillow's filters call the C
 * math library (``logf``, ``sinf``, ``cosf``, ``log``, ``log1p``, ``sin``,
 * ``cos``) and ``std::fma``. JavaScript's
 * ``Math`` functions round differently in roughly one case in a hundred, so
 * these are exact ports of the reference platform's implementations: GNU libc
 * 2.39 on AArch64, the platform the Python reference package is verified on.
 * The ports follow the compiled instruction sequences, including every fused
 * multiply-add the compiler formed, so results are bitwise identical.
 *
 * Fused multiply-add is emulated exactly (Boldo and Melquiond's algorithm with
 * rounding to odd), with an exact BigInt path for operands whose products
 * would overflow or underflow the error-free transformations.
 */

/**
 * Build the functions. The factory refers to no binding outside itself, so its
 * source text also runs in worker threads (``Kernel.NormalFill`` in
 * ``backend/engine.ts``) with bitwise-identical results.
 */
export function createLibm() {
  const F64 = new Float64Array(1);
  const U32 = new Uint32Array(F64.buffer);
  const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  const LO = LITTLE_ENDIAN ? 0 : 1;
  const HI = LITTLE_ENDIAN ? 1 : 0;
  const F32 = new Float32Array(1);
  const U32F = new Uint32Array(F32.buffer);

  const SPLIT = 134217729; // 2 ** 27 + 1
  const TWO_POW_900 = 2 ** 900;
  const TWO_POW_M900 = 2 ** -900;

  /** Low 32 bits of a double's IEEE representation. */
  function lowWord(x: number): number {
    F64[0] = x;
    return U32[LO]!;
  }

  /** High 32 bits of a double's IEEE representation. */
  function highWord(x: number): number {
    F64[0] = x;
    return U32[HI]!;
  }

  function fromWords(high: number, low: number): number {
    U32[HI] = high >>> 0;
    U32[LO] = low >>> 0;
    return F64[0]!;
  }

  /** IEEE bits of a float32 value. */
  function floatBits(x: number): number {
    F32[0] = x;
    return U32F[0]!;
  }

  /** The float32 with the given IEEE bits. */
  function floatFromBits(bits: number): number {
    U32F[0] = bits >>> 0;
    return F32[0]!;
  }

  /**
   * ``s`` rounded to odd, given the exact error ``error`` of ``s`` (so the exact
   * value is ``s + error``). Used to avoid double rounding.
   */
  function roundToOdd(s: number, error: number): number {
    if (error === 0) return s;
    F64[0] = s;
    if ((U32[LO]! & 1) === 1) return s;
    // Move one unit in the last place toward the exact value.
    if ((s > 0) === (error > 0)) {
      U32[LO] = U32[LO]! + 1;
    } else if (U32[LO] === 0) {
      U32[LO] = 0xffffffff;
      U32[HI] = U32[HI]! - 1;
    } else {
      U32[LO] = U32[LO]! - 1;
    }
    return F64[0]!;
  }

  function decompose(x: number): [bigint, number] {
    F64[0] = x;
    const high = U32[HI]!;
    const low = U32[LO]!;
    const exponent = (high >>> 20) & 0x7ff;
    let mantissa = (BigInt(high & 0xfffff) << 32n) | BigInt(low);
    let power: number;
    if (exponent === 0) {
      power = -1074;
    } else {
      mantissa |= 1n << 52n;
      power = exponent - 1075;
    }
    return [high >>> 31 ? -mantissa : mantissa, power];
  }

  function bitLength(value: bigint): number {
    return value.toString(2).length;
  }

  /** Round ``mantissa * 2 ** power`` to the nearest double, ties to even. */
  function roundExact(mantissa: bigint, power: number, zeroSign: number): number {
    if (mantissa === 0n) return zeroSign < 0 ? -0 : 0;
    const negative = mantissa < 0n;
    let magnitude = negative ? -mantissa : mantissa;
    const top = power + bitLength(magnitude) - 1;
    const lsb = Math.max(top - 52, -1074);
    let exponent = power;
    if (lsb > power) {
      const shift = BigInt(lsb - power);
      const quotient = magnitude >> shift;
      const remainder = magnitude - (quotient << shift);
      const half = 1n << (shift - 1n);
      magnitude = remainder > half || (remainder === half && (quotient & 1n) === 1n) ? quotient + 1n : quotient;
      exponent = lsb;
    }
    let value = Number(magnitude);
    // Scale in steps so intermediate powers of two stay finite.
    while (exponent > 0) {
      const step = Math.min(exponent, 1000);
      value *= 2 ** step;
      exponent -= step;
    }
    while (exponent < 0) {
      const step = Math.max(exponent, -1000);
      value *= 2 ** step;
      exponent -= step;
    }
    return negative ? -value : value;
  }

  function fmaExact(a: number, b: number, c: number): number {
    const [ma, ea] = decompose(a);
    const [mb, eb] = decompose(b);
    const [mc, ec] = decompose(c);
    let product = ma * mb;
    let productPower = ea + eb;
    let addend = mc;
    let addendPower = ec;
    if (productPower > addendPower) {
      product <<= BigInt(productPower - addendPower);
      productPower = addendPower;
    } else {
      addend <<= BigInt(addendPower - productPower);
      addendPower = productPower;
    }
    const sum = product + addend;
    const productNegative = (a < 0 || Object.is(a, -0)) !== (b < 0 || Object.is(b, -0));
    const cNegative = c < 0 || Object.is(c, -0);
    return roundExact(sum, productPower, productNegative && cNegative ? -1 : 1);
  }

  /** Exact ``fma(a, b, c)``: ``a * b + c`` with a single rounding. */
  function fma(a: number, b: number, c: number): number {
    const p = a * b;
    if (!Number.isFinite(a) || !Number.isFinite(b)) return p + c;
    if (!Number.isFinite(c)) return c; // a finite exact product never overflows the sum's infinity
    if (a === 0 || b === 0) return p + c;
    if (c === 0) {
      if (p !== 0 && Math.abs(p) >= TWO_POW_M900 && Math.abs(p) <= TWO_POW_900) return p;
      return fmaExact(a, b, c);
    }
    const ap = Math.abs(p);
    if (ap < TWO_POW_M900 || ap > TWO_POW_900 || Math.abs(a) > TWO_POW_900 || Math.abs(b) > TWO_POW_900
      || Math.abs(c) > TWO_POW_900) {
      return fmaExact(a, b, c);
    }
    let t = SPLIT * a;
    const ah = t - (t - a);
    const al = a - ah;
    t = SPLIT * b;
    const bh = t - (t - b);
    const bl = b - bh;
    const ul = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    const th = c + p;
    const tb = th - c;
    const tl = (c - (th - tb)) + (p - tb);
    const v = tl + ul;
    const vb = v - tl;
    const ve = (tl - (v - vb)) + (ul - vb);
    return th + roundToOdd(v, ve);
  }

  /** Exact single-precision ``fmaf(a, b, c)`` for float32 operands. */
  function fmaf(a: number, b: number, c: number): number {
    const p = a * b; // exact: two 24-bit significands
    const s = p + c;
    if (!Number.isFinite(s)) return Math.fround(s);
    const sb = s - p;
    const error = (p - (s - sb)) + (c - sb);
    return Math.fround(roundToOdd(s, error));
  }

  // ---------------------------------------------------------------------------
  // logf (glibc sysdeps/ieee754/flt-32/e_logf.c)
  // ---------------------------------------------------------------------------

  const LOGF_INVC = [
    1.398907162146528, 1.3403141896637998, 1.286432210124115, 1.2367150214269895,
    1.1906977166711752, 1.1479821020556429, 1.1082251448272158, 1.0711297413057381,
    1.036437278977283, 1.0, 0.9492859795739057, 0.8951049428609004,
    0.8476821620351103, 0.8050314851692001, 0.7664671008843108, 0.731428603316328,
  ];
  const LOGF_LOGC = [
    -0.33569133332882284, -0.2929040563774074, -0.2518726580937369, -0.21245868807117255,
    -0.17453945183745634, -0.1380057072319758, -0.10275976698545139, -0.06871392447020525,
    -0.0357891387398228, 0.0, 0.05204517742929496, 0.11081431298787942,
    0.1652495223695143, 0.21687389031699977, 0.2659635028121397, 0.3127556664073557,
  ];
  const LOGF_LN2 = 0.6931471805599453;
  const LOGF_A0 = -0.25089342214237154;
  const LOGF_A1 = 0.333456765744066;
  const LOGF_A2 = -0.4999997485802103;

  /** glibc ``logf`` for a float32 argument. */
  function logf(x: number): number {
    let ix = floatBits(x);
    if (ix === 0x3f800000) return 0;
    if (((ix - 0x00800000) >>> 0) >= 0x7f800000 - 0x00800000) {
      if (((ix << 1) >>> 0) === 0) return -Infinity;
      if (ix === 0x7f800000) return x;
      if ((ix & 0x80000000) !== 0 || ((ix << 1) >>> 0) >= 0xff000000) return NaN;
      ix = floatBits(Math.fround(x * 8388608.0));
      ix = (ix - (23 << 23)) >>> 0;
    }
    const tmp = (ix - 0x3f330000) >>> 0;
    const i = (tmp >>> 19) & 15;
    const k = tmp >> 23;
    const iz = (ix - (tmp & 0xff800000)) >>> 0;
    const z = floatFromBits(iz);
    const r = fma(LOGF_INVC[i]!, z, -1);
    const y0 = fma(k, LOGF_LN2, LOGF_LOGC[i]!);
    let y = fma(LOGF_A1, r, LOGF_A2);
    const r2 = r * r;
    const t = r + y0;
    y = fma(LOGF_A0, r2, y);
    y = fma(r2, y, t);
    return Math.fround(y);
  }

  // ---------------------------------------------------------------------------
  // sinf / cosf (glibc sysdeps/ieee754/flt-32/s_sinf.c, s_cosf.c)
  // ---------------------------------------------------------------------------

  interface SincosfTable {
    hpiInv: number; hpi: number;
    c0: number; c1: number; c2: number; c3: number; c4: number;
    s1: number; s2: number; s3: number;
  }
  const SINCOSF_SIGN = [1, -1, -1, 1];
  const SINCOSF: readonly [SincosfTable, SincosfTable] = [
    {
      hpiInv: 0.6366197723675814, hpi: 1.5707963267948966,
      c0: 1, c1: -0.49999999725108224, c2: 0.041666623324344516, c3: -0.001388676379437604, c4: 2.4390450703564542e-05,
      s1: -0.16666654943701084, s2: 0.008332178146138854, s3: -0.00019517298981385725,
    },
    {
      hpiInv: 0.6366197723675814, hpi: 1.5707963267948966,
      c0: -1, c1: 0.49999999725108224, c2: -0.041666623324344516, c3: 0.001388676379437604, c4: -2.4390450703564542e-05,
      s1: -0.16666654943701084, s2: 0.008332178146138854, s3: -0.00019517298981385725,
    },
  ];
  const INV_PIO4 = [
    0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529,
    0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0,
    0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041,
  ];
  const PI63 = 3.4061215800865545e-19;

  function sinfPoly(x: number, x2: number, p: SincosfTable, n: number): number {
    if ((n & 1) === 0) {
      const x3 = x * x2;
      const s1 = fma(x2, p.s3, p.s2);
      const x7 = x2 * x3;
      const s = fma(x3, p.s1, x);
      return Math.fround(fma(s1, x7, s));
    }
    const x4 = x2 * x2;
    const c1 = fma(x2, p.c1, p.c0);
    const c2 = fma(x2, p.c4, p.c3);
    const x6 = x2 * x4;
    const c = fma(x4, p.c2, c1);
    return Math.fround(fma(c2, x6, c));
  }

  /** C ``round``: nearest integer, ties away from zero. */
  function roundAway(x: number): number {
    const r = Math.round(Math.abs(x));
    const value = Math.abs(x) - Math.floor(Math.abs(x)) === 0.5 ? Math.floor(Math.abs(x)) + 1 : r;
    return x < 0 ? -value : value;
  }

  /** Quadrant and reduced argument for |y| >= 120 (``reduce_large``). */
  function reduceLarge(xi: number): [number, number] {
    const base = (xi >>> 26) & 15;
    const shift = (xi >>> 23) & 7;
    let m = BigInt(((xi & 0xffffff) | 0x800000) >>> 0);
    m = (m << BigInt(shift)) & 0xffffffffn;
    const mask = 0xffffffffffffffffn;
    let res0 = (m * BigInt(INV_PIO4[base]!)) & 0xffffffffn;
    const res1 = m * BigInt(INV_PIO4[base + 4]!);
    const res2 = m * BigInt(INV_PIO4[base + 8]!);
    res0 = ((res2 >> 32n) | (res0 << 32n)) & mask;
    res0 = (res0 + res1) & mask;
    const n = ((res0 + (1n << 61n)) & mask) >> 62n;
    res0 = (res0 - (n << 62n)) & mask;
    const signed = res0 >= 1n << 63n ? res0 - (1n << 64n) : res0;
    return [Number(signed) * PI63, Number(n)];
  }

  function sincosf(y: number, cosine: boolean): number {
    const bits = floatBits(y);
    const top = (bits >>> 20) & 0x7ff;
    let x = y;
    if (top < 0x3f4) {
      const x2 = x * x;
      if (top < 0x398) return cosine ? 1 : y;
      return sinfPoly(x, x2, SINCOSF[0], cosine ? 1 : 0);
    }
    if (top < 0x42f) {
      const p0 = SINCOSF[0];
      const r = x * p0.hpiInv;
      const rounded = roundAway(r);
      const n = rounded;
      x = fma(-rounded, p0.hpi, x);
      const s = SINCOSF_SIGN[((n % 4) + 4) % 4]!;
      const p = (n & 2) !== 0 ? SINCOSF[1] : p0;
      return sinfPoly(x * s, x * x, p, cosine ? n ^ 1 : n);
    }
    if (top < 0x7f8) {
      const sign = bits >>> 31;
      const [reduced, n] = reduceLarge(bits);
      const quadrant = (n + sign) & 3;
      const s = SINCOSF_SIGN[quadrant]!;
      const p = (quadrant & 2) !== 0 ? SINCOSF[1] : SINCOSF[0];
      return sinfPoly(reduced * s, reduced * reduced, p, cosine ? n ^ 1 : n);
    }
    return NaN;
  }

  /** Results of {@link sincosfPair}. */
  const sincosfResult = { sin: 0, cos: 0 };

  /**
   * glibc ``sinf(y)`` and ``cosf(y)`` together (each identical to the separate
   * call; the range reduction is shared). Results land in {@link sincosfResult}.
   */
  function sincosfPair(y: number): void {
    const bits = floatBits(y);
    const top = (bits >>> 20) & 0x7ff;
    if (top < 0x3f4) {
      const x2 = y * y;
      if (top < 0x398) {
        sincosfResult.sin = y;
        sincosfResult.cos = 1;
        return;
      }
      sincosfResult.sin = sinfPoly(y, x2, SINCOSF[0], 0);
      sincosfResult.cos = sinfPoly(y, x2, SINCOSF[0], 1);
      return;
    }
    if (top < 0x42f) {
      const p0 = SINCOSF[0];
      const rounded = roundAway(y * p0.hpiInv);
      const x = fma(-rounded, p0.hpi, y);
      const n = rounded;
      const sign = SINCOSF_SIGN[((n % 4) + 4) % 4]!;
      const p = (n & 2) !== 0 ? SINCOSF[1] : p0;
      const xs = x * sign;
      const x2 = x * x;
      sincosfResult.sin = sinfPoly(xs, x2, p, n);
      sincosfResult.cos = sinfPoly(xs, x2, p, n ^ 1);
      return;
    }
    sincosfResult.sin = sincosf(y, false);
    sincosfResult.cos = sincosf(y, true);
  }

  /** glibc ``sinf`` for a float32 argument. */
  function sinf(y: number): number {
    return sincosf(y, false);
  }

  /** glibc ``cosf`` for a float32 argument. */
  function cosf(y: number): number {
    return sincosf(y, true);
  }

  // ---------------------------------------------------------------------------
  // log (glibc sysdeps/ieee754/dbl-64/e_log.c, LOG_TABLE_BITS = 7)
  // ---------------------------------------------------------------------------

  const LOG_LN2HI = 0.6931471805598903;
  const LOG_LN2LO = 5.497923018708371e-14;
  const LOG_A = [-0.5000000000000001, 0.33333333331825593, -0.2499999999622955, 0.20000304511814496, -0.16667054827627667];
  const LOG_B = [
    -0.5, 0.3333333333333352, -0.24999999999998432, 0.19999999999320328, -0.16666666669929706,
    0.14285715076560868, -0.12499997863982555, 0.11110712032936046, -0.10000486757818193,
    0.09181994006195467, -0.08328363062289341,
  ];
  const LOG_TABLE = [
    1.4504249240398293, -0.3718565645633589, 1.442253508327276, -0.36620682668944937,
    1.4341736174350004, -0.3605888069791945, 1.426183816329995, -0.3550022171419869,
    1.4182825527052965, -0.34944666968829097, 1.4104682921759335, -0.3439217713603284,
    1.4027396147468003, -0.3384271921261188, 1.3950954438932313, -0.332962831494342,
    1.3875338232485754, -0.32752794345742586, 1.3800539211058593, -0.32212257167088865,
    1.3726542695419708, -0.3167462884799761, 1.3653332798446802, -0.3113985598928366,
    1.358090204587874, -0.3060794515165526, 1.3509234892132138, -0.300788424667644,
    1.3438320840699889, -0.2955252968476998, 1.3368146974742003, -0.29028969275850613,
    1.329870114677736, -0.2850812793277555, 1.322997339161106, -0.27989987391470095,
    1.316195352741367, -0.27474526621870154, 1.3094628125672239, -0.2696169863701243,
    1.3027990455471041, -0.26451506180308115, 1.2962024229438942, -0.2594387762767383,
    1.2896726275815547, -0.2543884090981692, 1.2832080305745537, -0.24936321635129843,
    1.276807885983376, -0.24436312405975968, 1.2704714060687552, -0.23938801747897287,
    1.2641976054949482, -0.23443761696705678, 1.257985357514882, -0.22951151871518505,
    1.2518337750655457, -0.2246094963439873, 1.2457421919097305, -0.21973149037705753,
    1.2397094966625508, -0.2148770752847895, 1.2337348463589233, -0.210046029103637,
    1.2278176973028803, -0.20523836373934046, 1.2219570190618474, -0.20045368751368642,
    1.2161519732977757, -0.1956917537758045, 1.2104018095009725, -0.19095237845203883,
    1.204705805718973, -0.18623539250290833, 1.1990631185441964, -0.18154051731551135,
    1.1934733004462308, -0.1768677957431919, 1.1879350812847385, -0.17221657406412305,
    1.1824481322833125, -0.16758697765942543, 1.1770114976921955, -0.16297859687290384,
    1.1716248121809465, -0.15839151377804228, 1.1662869231674715, -0.15382513241456763,
    1.1609977486762766, -0.14927976358922024, 1.1557563220795803, -0.14475495398119165,
    1.1505619105480347, -0.14025044090817573, 1.1454138888505974, -0.1357660466685502,
    1.140311877374656, -0.1313018016355727, 1.1352550225747513, -0.12685731518763532,
    1.1302429094831266, -0.1224325737671279, 1.1252747693068048, -0.11802724521862729,
    1.1203501571039876, -0.11364127671663482, 1.1154683327680124, -0.10927434611278386,
    1.1106291463292157, -0.10492665324943573, 1.1058315813301596, -0.10059761422644442,
    1.1010752177696026, -0.09628717309055901, 1.0963597137952512, -0.09199534069557558,
    1.0916844827550398, -0.08772190036688698, 1.0870487291277784, -0.08346643613867855,
    1.082452357388312, -0.07922916827544668, 1.0778948225025884, -0.0750099004750382,
    1.0733751731601076, -0.07080805133352897, 1.068893585073351, -0.06662408085151128,
    1.0644491706655506, -0.06245745471915143, 1.0600414846328305, -0.0583080438042316,
    1.0556701316181605, -0.05417576112313327, 1.051334750556926, -0.050060547896805474,
    1.0470347288442157, -0.045962101199052086, 1.0427699229652954, -0.04188056008865715,
    1.0385395013738175, -0.03781540056183985, 1.034343418940345, -0.03376684757915882,
    1.0301811073173315, -0.029734619131772888, 1.026052043621297, -0.025718470239212365,
    1.0219561082336197, -0.021718543925430822, 1.0178926505784922, -0.01773446126981071,
    1.0138614436244586, -0.013766252464051831, 1.0098620186501341, -0.009813706322574944,
    1.0058938559734134, -0.005876555150052809, 1.00195696235014, -0.0019550499938532084,
    0.9961089923088509, 0.0038985973556009412, 0.9884170338185201, 0.011650571286395461,
    0.9808429191005297, 0.019342955478919066, 0.9733840169987446, 0.0269766014846482,
    0.9660377568876556, 0.034552359728422744, 0.9588014945307369, 0.04207121767183253,
    0.9516728569073111, 0.049533940950141186, 0.9446494635965822, 0.056941358295944156,
    0.9377288993026223, 0.06429439168346107, 0.9309091073790681, 0.0715936354946507,
    0.924187681612722, 0.07884010933776153, 0.9175626765599192, 0.08603438905970506,
    0.9110320403624034, 0.09317721180013905, 0.9045935839762024, 0.10026951462748457,
    0.8982456375922825, 0.10731170956330516, 0.8919860966782501, 0.11430473320717738,
    0.8858131121185129, 0.12124928503033061, 0.879725075760676, 0.12814583422959913,
    0.8737201372634685, 0.1349951636851756, 0.8677966405782273, 0.1417978768189414,
    0.8619528050060739, 0.14855476039031146, 0.8561872354420692, 0.1552661937658968,
    0.8504983927816893, 0.16193275688146969, 0.8448844572790304, 0.16855539792220497,
    0.8393442741575965, 0.1751343179947753, 0.8338762249349438, 0.1816702989864325,
    0.8284789320557778, 0.18816387146023317, 0.8231510800065832, 0.1946155228479256,
    0.8178913903778707, 0.20102572579389744, 0.8126984007245374, 0.2073952090795501,
    0.8075710029460227, 0.21372429840596396, 0.8025078881160415, 0.2200135945981856,
    0.7975077379364331, 0.22626374162859975, 0.792569604966373, 0.23247494747693054,
    0.7876923641254114, 0.23864766620658884, 0.7828746724940998, 0.24478265647405806,
    0.7781155388790811, 0.25088025827324145, 0.7734139557869777, 0.2569408552510595,
    0.7687687179914933, 0.26296511155101143, 0.7641790698041854, 0.2689531327189343,
    0.7596438763692399, 0.27490553924610595, 0.7551621951078668, 0.2808227248478943,
    0.7507331780216866, 0.286704979267256, 0.7463557196361751, 0.29255295645509705,
    0.7420289364869653, 0.2983670386142876, 0.7377521537065876, 0.30414734587282055,
    0.7335242966002608, 0.30989455774829366, 0.729344777457841, 0.31560871301871884,
  ];

  /** glibc ``log`` for a double argument. */
  function log(x: number): number {
    F64[0] = x;
    let high = U32[HI]!;
    let low = U32[LO]!;
    // ix - asuint64(1 - 0.0625) < asuint64(1 + 0.064697265625) - asuint64(1 - 0.0625)
    // asuint64(x) - asuint64(1 - 0x1p-4) < asuint64(1 + 0x1.09p-4) - asuint64(1 - 0x1p-4)
    if (x >= 0.9375 && x < 1.064697265625) {
      if (x === 1) return 0;
      const r = x - 1;
      const rhi = fma(-r, 134217728.0, fma(r, 134217728.0, r));
      const r2 = r * r;
      const r3 = r * r2;
      let d4 = fma(LOG_B[8]!, r, LOG_B[7]!);
      d4 = fma(LOG_B[9]!, r2, d4);
      d4 = fma(LOG_B[10]!, r3, d4);
      let d3 = fma(LOG_B[5]!, r, LOG_B[4]!);
      d3 = fma(LOG_B[6]!, r2, d3);
      const rhi2 = rhi * rhi;
      const hi = fma(rhi2, LOG_B[0]!, r);
      let d2 = fma(LOG_B[2]!, r, LOG_B[1]!);
      d3 = fma(d4, r3, d3);
      const rlo = r - rhi;
      const rsum = r + rhi;
      d2 = fma(LOG_B[3]!, r2, d2);
      const scaled = LOG_B[0]! * rlo;
      let lo = fma(rhi2, LOG_B[0]!, r - hi);
      d2 = fma(d3, r3, d2);
      lo = fma(scaled, rsum, lo);
      const y = fma(d2, r3, lo);
      return hi + y;
    }
    const top = high >>> 16;
    if (((top - 0x0010) >>> 0) >= 0x7ff0 - 0x0010) {
      if (x === 0) return -Infinity;
      if (x === Infinity) return x;
      if ((top & 0x8000) !== 0 || (top & 0x7ff0) === 0x7ff0) return NaN;
      F64[0] = x * 4503599627370496.0;
      high = (U32[HI]! - (52 << 20)) >>> 0;
      low = U32[LO]!;
    }
    // tmp = ix - 0x3fe6000000000000 (high word arithmetic suffices: OFF has zero low word).
    const tmpHigh = (high - 0x3fe60000) >>> 0;
    const i = (tmpHigh >>> 13) & 127;
    const k = tmpHigh >> 20;
    const zHigh = (high - (tmpHigh & 0xfff00000)) >>> 0;
    const z = fromWords(zHigh, low);
    const invc = LOG_TABLE[2 * i]!;
    const logc = LOG_TABLE[2 * i + 1]!;
    const w = fma(LOG_LN2HI, k, logc);
    const r = fma(invc, z, -1);
    const hi = r + w;
    const p34 = fma(LOG_A[4]!, r, LOG_A[3]!);
    const p12 = fma(LOG_A[2]!, r, LOG_A[1]!);
    const r2 = r * r;
    let lo = w - hi;
    const r3 = r * r2;
    lo = lo + r;
    const p = fma(p34, r2, p12);
    lo = fma(LOG_LN2LO, k, lo);
    let y = fma(LOG_A[0]!, r2, lo);
    y = fma(r3, p, y);
    return y + hi;
  }

  // ---------------------------------------------------------------------------
  // log1p (glibc sysdeps/ieee754/dbl-64/s_log1p.c)
  // ---------------------------------------------------------------------------

  const LOG1P_LN2_HI = 0.6931471803691238;
  const LOG1P_LN2_LO = 1.9082149292705877e-10;
  const LP1 = 0.6666666666666735;
  const LP2 = 0.3999999999940942;
  const LP3 = 0.2857142874366239;
  const LP4 = 0.22222198432149784;
  const LP5 = 0.1818357216161805;
  const LP6 = 0.15313837699209373;
  const LP7 = 0.14798198605116586;

  /** glibc ``log1p`` for a double argument. */
  function log1p(x: number): number {
    const hx = highWord(x) | 0;
    const ax = hx & 0x7fffffff;
    let k = 1;
    let f = 0;
    let hu = 0;
    let c = 0;
    if (hx < 0x3fda827a) {
      if (ax >= 0x3ff00000) return x === -1 ? -Infinity : NaN;
      if (ax < 0x3e200000) {
        if (ax < 0x3c900000) return x;
        return fma(-(x * x), 0.5, x);
      }
      if (hx > 0 || hx <= (0xbfd2bec3 | 0)) {
        k = 0;
        f = x;
        hu = 1;
      }
    } else if (hx >= 0x7ff00000) {
      return x + x;
    }
    if (k !== 0) {
      let u: number;
      if (hx < 0x43400000) {
        u = 1 + x;
        hu = highWord(u) | 0;
        k = (hu >> 20) - 1023;
        c = k > 0 ? 1 - (u - x) : x - (u - 1);
        c /= u;
      } else {
        u = x;
        hu = highWord(u) | 0;
        k = (hu >> 20) - 1023;
        c = 0;
      }
      hu &= 0x000fffff;
      const low = lowWord(u);
      if (hu < 0x6a09e) {
        u = fromWords(hu | 0x3ff00000, low);
      } else {
        k += 1;
        u = fromWords(hu | 0x3fe00000, low);
        hu = (0x00100000 - hu) >> 2;
      }
      f = u - 1;
    }
    const hfsq = (f * 0.5) * f;
    if (hu === 0) {
      if (f === 0) {
        if (k === 0) return 0;
        return fma(k, LOG1P_LN2_HI, fma(k, LOG1P_LN2_LO, c));
      }
      const R = fma(-f, 0.66666666666666666, 1) * hfsq;
      if (k === 0) return f - R;
      return fma(k, LOG1P_LN2_HI, -((R - fma(k, LOG1P_LN2_LO, c)) - f));
    }
    const s = f / (2 + f);
    const z = s * s;
    const R2 = fma(z, LP3, LP2);
    const z2 = z * z;
    const R3 = fma(z, LP5, LP4);
    const R4 = fma(z, LP7, LP6);
    const z4 = z2 * z2;
    let R = z2 * R2;
    const z6 = z2 * z4;
    R = fma(z, LP1, R);
    R = fma(z4, R3, R);
    R = fma(z6, R4, R);
    const t = (R + hfsq) * s;
    if (k === 0) return f - (hfsq - t);
    const correction = fma(k, LOG1P_LN2_LO, c) + t;
    return fma(k, LOG1P_LN2_HI, -((hfsq - correction) - f));
  }

  // ---------------------------------------------------------------------------
  // sin / cos (glibc sysdeps/ieee754/dbl-64/s_sin.c)
  // ---------------------------------------------------------------------------

  const SN3 = -0.16666666666666488;
  const SN5 = 0.008333332142857223;
  const CS2 = 0.5;
  const CS4 = -0.04166666666666644;
  const CS6 = 0.001388888740079376;
  const S1 = -0.16666666666666666;
  const S2 = 0.008333333333332329;
  const S3 = -0.00019841269834414642;
  const S4 = 2.755729806860771e-06;
  const S5 = -2.5022014848318398e-08;
  const BIG = 52776558133248.0;
  const HP0 = 1.5707963267948966;
  const HP1 = 6.123233995736766e-17;
  const MP1 = 1.5707963407039642;
  const MP2 = -1.3909067564377153e-08;
  const PP3 = -4.97899623147991e-17;
  const PP4 = -1.9034889620193266e-25;
  const HPINV = 0.6366197723675814;
  const TOINT = 6755399441055744.0;

  let sincosTable: Float64Array | null = null;

  function table(): Float64Array {
    if (sincosTable === null) sincosTable = buildSincosTable();
    return sincosTable;
  }

  function copySign(magnitude: number, sign: number): number {
    const negative = sign < 0 || Object.is(sign, -0);
    const absolute = Math.abs(magnitude);
    return negative ? -absolute : absolute;
  }

  function taylorSin(xx: number, x: number, dx: number): number {
    let poly = fma(xx, S5, S4);
    poly = fma(xx, poly, S3);
    poly = fma(xx, poly, S2);
    poly = fma(xx, poly, S1);
    const inner = fma(poly, x, -(dx * 0.5));
    const t = fma(xx, inner, dx);
    return t + x;
  }

  function doSin(xIn: number, dxIn: number): number {
    const xold = xIn;
    if (Math.abs(xIn) < 0.126) return taylorSin(xIn * xIn, xIn, dxIn);
    const dx = xIn <= 0 ? -dxIn : dxIn;
    const u = BIG + Math.abs(xIn);
    const x = Math.abs(xIn) - (u - BIG);
    const xx = x * x;
    const csPoly = fma(xx, CS6, CS4);
    const snPoly = fma(xx, SN5, SN3);
    const x3 = x * xx;
    const tab = table();
    const k = lowWord(u) << 2;
    const sn = tab[k]!;
    const ssn = tab[k + 1]!;
    const cs = tab[k + 2]!;
    const ccs = tab[k + 3]!;
    const cPoly = fma(xx, csPoly, CS2);
    const sInner = fma(x3, snPoly, dx);
    const cTail = xx * cPoly;
    const s = x + sInner;
    const c = fma(x, dx, cTail);
    let cor = fma(s, ccs, ssn);
    cor = fma(-c, sn, cor);
    cor = fma(s, cs, cor);
    return copySign(sn + cor, xold);
  }

  function doCos(xIn: number, dxIn: number): number {
    const dx = xIn < 0 ? -dxIn : dxIn;
    const u = BIG + Math.abs(xIn);
    const x = (Math.abs(xIn) - (u - BIG)) + dx;
    const tab = table();
    const k = lowWord(u) << 2;
    const sn = tab[k]!;
    const ssn = tab[k + 1]!;
    const cs = tab[k + 2]!;
    const ccs = tab[k + 3]!;
    const xx = x * x;
    const csPoly = fma(xx, CS6, CS4);
    const snPoly = fma(xx, SN5, SN3);
    const x3 = x * xx;
    const cPoly = fma(xx, csPoly, CS2);
    const s = fma(x3, snPoly, x);
    const c = xx * cPoly;
    let cor = fma(-s, ssn, ccs);
    cor = fma(-c, cs, cor);
    cor = fma(-s, sn, cor);
    return cs + cor;
  }

  function reduceSincos(x: number): [number, number, number] {
    const t = fma(x, HPINV, TOINT);
    const xn = t - TOINT;
    const n = lowWord(t) & 3;
    let y = fma(-xn, MP1, x);
    y = fma(-xn, MP2, y);
    const t2 = fma(-xn, PP3, y);
    const b = fma(-xn, PP4, t2);
    let db = fma(-xn, PP3, y - t2);
    const tail = fma(-xn, PP4, t2 - b);
    db = db + tail;
    return [b, db, n];
  }

  function doSincos(a: number, da: number, n: number): number {
    const value = (n & 1) !== 0 ? doCos(a, da) : doSin(a, da);
    return (n & 2) !== 0 ? -value : value;
  }

  /** glibc ``sin`` for a double argument (|x| below 105414350). */
  function sin(x: number): number {
    const k = highWord(x) & 0x7fffffff;
    if (k < 0x3e500000) return x;
    if (k < 0x3feb6000) return doSin(x, 0);
    if (k < 0x400368fd) {
      const t = HP0 - Math.abs(x);
      return copySign(doCos(t, HP1), x);
    }
    if (k < 0x419921fb) {
      const [a, da, n] = reduceSincos(x);
      return doSincos(a, da, n);
    }
    if (k < 0x7ff00000) return Math.sin(x);
    return NaN;
  }

  /** glibc ``cos`` for a double argument (|x| below 105414350). */
  function cos(x: number): number {
    const k = highWord(x) & 0x7fffffff;
    if (k < 0x3e400000) return 1;
    if (k < 0x3feb6000) return doCos(x, 0);
    if (k < 0x400368fd) {
      const y = HP0 - Math.abs(x);
      const a = y + HP1;
      const da = (y - a) + HP1;
      return doSin(a, da);
    }
    if (k < 0x419921fb) {
      const [a, da, n] = reduceSincos(x);
      return doSincos(a, da, n + 1);
    }
    if (k < 0x7ff00000) return Math.cos(x);
    return NaN;
  }

  function buildSincosTable(): Float64Array {
    const words = SINCOS_WORDS;
    const result = new Float64Array(words.length / 2);
    for (let index = 0; index < result.length; index += 1) {
      result[index] = fromWords(words[2 * index]!, words[2 * index + 1]!);
    }
    return result;
  }

  // __sincostab from glibc sysdeps/ieee754/dbl-64/sincostab.c as (high, low) words.
  const SINCOS_WORDS: readonly number[] = [
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3ff00000, 0x00000000, 0x00000000, 0x00000000,
    0x3f7fffea, 0xaaaeeeef, 0xbc1e45e2, 0xec67b77c, 0x3fefffc0, 0x00155552, 0x3c8f4a01, 0xa0196dae,
    0x3f8fffaa, 0xaaeeeed5, 0xbc02ab63, 0x9a9f0777, 0x3fefff00, 0x0155549f, 0x3c828a28, 0xa03a5ef3,
    0x3f97ff70, 0x01033255, 0x3bfefe2b, 0x51527336, 0x3feffdc0, 0x06bff7e6, 0x3c8ae6da, 0xe86977bd,
    0x3f9ffeaa, 0xaeeee86f, 0xbc3cd406, 0xfb224ae2, 0x3feffc00, 0x155527d3, 0xbc83b544, 0x92d89b5b,
    0x3fa3feb2, 0xb12d45d5, 0x3c34ec54, 0x203d1c11, 0x3feff9c0, 0x3414a7ba, 0x3c6991f4, 0xbe6c59bf,
    0x3fa7fdc0, 0x1032fba9, 0xbc4599bd, 0xf46e997a, 0x3feff700, 0x6bfdf99f, 0xbc78b3b5, 0x60648d5f,
    0x3fabfc6d, 0x78586dac, 0x3c18e4fd, 0x03dbf236, 0x3feff3c0, 0xc8103a31, 0x3c74856d, 0xbddc0e66,
    0x3faffaaa, 0xeeed4edb, 0xbc42d16d, 0x32684b69, 0x3feff001, 0x5549f4d3, 0x3c832838, 0x7b99426f,
    0x3fb1fc34, 0x3d808bef, 0xbc5f3d32, 0xe6f3be4f, 0x3fefebc2, 0x22a8ef9f, 0x3c579349, 0x34f54c77,
    0x3fb3facb, 0x12d1755b, 0xbc592191, 0x5299468c, 0x3fefe703, 0x4129ef6f, 0xbc6cbf43, 0x37c96f97,
    0x3fb5f911, 0xfd10b737, 0xbc50184f, 0x02be9102, 0x3fefe1c4, 0xc3c873eb, 0xbc35a9c9, 0x057c4a02,
    0x3fb7f701, 0x032550e4, 0x3c3afc2d, 0x1800501a, 0x3fefdc06, 0xbf7e6b9b, 0x3c831902, 0xb535f8db,
    0x3fb9f490, 0x2d55d1f9, 0x3c52696d, 0x7eac1dc1, 0x3fefd5c9, 0x4b43e000, 0xbc62e768, 0xcb4f92f9,
    0x3fbbf1b7, 0x8568391d, 0x3c5e9184, 0x1dea4cc8, 0x3fefcf0c, 0x800e99b1, 0x3c6ea3d7, 0x86d186ac,
    0x3fbdee6f, 0x16c1cce6, 0xbc450f8e, 0x2fb71673, 0x3fefc7d0, 0x78d1bc88, 0x3c8075d2, 0x447db685,
    0x3fbfeaae, 0xee86ee36, 0xbc4afcb2, 0xbcc6f03b, 0x3fefc015, 0x527d5bd3, 0x3c8b68f3, 0x5094efb8,
    0x3fc0f337, 0x8ddd71d1, 0x3c6d8468, 0x724f0f9e, 0x3fefb7db, 0x2bfe0695, 0x3c821dad, 0xf4f65ab1,
    0x3fc1f0d3, 0xd7afceaf, 0xbc66ef95, 0x099769a5, 0x3fefaf22, 0x263c4bd3, 0xbc552ace, 0x133a2769,
    0x3fc2ee28, 0x5e4ab88f, 0xbc6e4d0f, 0x05dee058, 0x3fefa5ea, 0x641c36f2, 0x3c404da6, 0xed17cc7c,
    0x3fc3eb31, 0x2c5d66cb, 0x3c647d66, 0x6b66cb91, 0x3fef9c34, 0x0a7cc428, 0x3c8c5b6b, 0x063b7462,
    0x3fc4e7ea, 0x4dc5f27b, 0x3c5949db, 0x2ac072fc, 0x3fef91ff, 0x40374d01, 0xbc67d03f, 0x4d3a9e4c,
    0x3fc5e44f, 0xcfa126f3, 0xbc66f443, 0x063f89b6, 0x3fef874c, 0x2e1eecf6, 0xbc8c6514, 0xe1332b16,
    0x3fc6e05d, 0xc05a4d4c, 0xbbd32c5c, 0x8b81c940, 0x3fef7c1a, 0xfeffde24, 0xbc78f55b, 0xc47540b1,
    0x3fc7dc10, 0x2fbaf2b5, 0x3c45ab50, 0xe23c97c3, 0x3fef706b, 0xdf9ece1c, 0xbc8698c8, 0x0c36dcb4,
    0x3fc8d763, 0x2efaa944, 0xbc620fa2, 0x62cbb953, 0x3fef643e, 0xfeb82acd, 0x3c76b00a, 0xc1fe28ac,
    0x3fc9d252, 0xd0cec312, 0x3c59c43d, 0x80b1137d, 0x3fef5794, 0x8cff6797, 0x3c6e3a0d, 0x3e03b1d5,
    0x3fcaccdb, 0x297a0765, 0xbc59883b, 0x57d6cdeb, 0x3fef4a6c, 0xbd1e3a79, 0x3c813df0, 0xedaebb57,
    0x3fcbc6f8, 0x4edc6199, 0x3c69c1a5, 0x6a7b0cab, 0x3fef3cc7, 0xc3b3d16e, 0xbc621a3a, 0xd28a3494,
    0x3fccc0a6, 0x588289a3, 0xbc6868d0, 0x9bc87c6b, 0x3fef2ea5, 0xd753ffed, 0x3c8cc421, 0x5f56d583,
    0x3fcdb9e1, 0x5fb5a5d0, 0xbc632e20, 0xd6cc6fc2, 0x3fef2007, 0x3086649f, 0x3c7b9404, 0x16c1984b,
    0x3fceb2a5, 0x7f8ae5a3, 0xbc60be06, 0xaf572ceb, 0x3fef10ec, 0x09c5873b, 0x3c8d9072, 0x762c1283,
    0x3fcfaaee, 0xd4f31577, 0xbc615d88, 0x508e32b8, 0x3fef0154, 0x9f7deea1, 0x3c8d3c1e, 0x99e5cafd,
    0x3fd0515c, 0xbf65155c, 0xbc79b8c2, 0x9dfd8ec8, 0x3feef141, 0x300d2f26, 0xbc82aa1b, 0x08ded372,
    0x3fd0cd00, 0xcef36436, 0xbc79fb0a, 0x0c93e2b5, 0x3feee0b1, 0xfbc0f11c, 0xbc4bfd23, 0x80bbc3b1,
    0x3fd14861, 0xaa94ddeb, 0xbc6be881, 0xb5b615a4, 0x3feecfa7, 0x44d5efa1, 0xbc556d0a, 0x4af541d0,
    0x3fd1c37d, 0x64c6b876, 0x3c746076, 0xfe0dcff5, 0x3feebe21, 0x4f76efa8, 0xbc802f9f, 0x12ba543e,
    0x3fd23e52, 0x111aaf36, 0xbc74f080, 0x334eff18, 0x3feeac20, 0x61bbaf4f, 0x3c62c1d5, 0x3e94658d,
    0x3fd2b8dd, 0xc43eb49f, 0x3c615538, 0x99f2d807, 0x3fee99a4, 0xc3a7cd83, 0xbc82264b, 0x1bc53ce8,
    0x3fd3331e, 0x94049f87, 0x3c7e0cb6, 0xb40c302c, 0x3fee86ae, 0xbf29a9ed, 0x3c89397a, 0xfdbb58a7,
    0x3fd3ad12, 0x9769d3d8, 0x3c003d55, 0x04878398, 0x3fee733e, 0xa0193d40, 0xbc86428b, 0x3546ce13,
    0x3fd426b7, 0xe69ee697, 0xbc7f09c7, 0x5705c59f, 0x3fee5f54, 0xb436e9d0, 0x3c87eb0f, 0xd02fc8bc,
    0x3fd4a00c, 0x9b0f3d20, 0x3c7823ba, 0x6bb08ead, 0x3fee4af1, 0x4b2a449c, 0xbc868ca0, 0x2e8a6833,
    0x3fd5190e, 0xcf68a77a, 0x3c7b3571, 0x55eef0f3, 0x3fee3614, 0xb680d6a5, 0xbc727793, 0xaa015237,
    0x3fd591bc, 0x9fa2f597, 0x3c67c74b, 0xac3fe0cb, 0x3fee20bf, 0x49acd6c1, 0xbc5660ae, 0xc7ef636c,
    0x3fd60a14, 0x29078775, 0x3c5b1fd8, 0x0ba89133, 0x3fee0af1, 0x5a03dbce, 0x3c5fe8e7, 0x02771ae6,
    0x3fd68213, 0x8a38d7f7, 0xbc7d8892, 0x02444aad, 0x3fedf4ab, 0x3ebd875e, 0xbc8e2d8a, 0x7e6736c4,
    0x3fd6f9b8, 0xe33a0255, 0x3c742bc1, 0x4ee9da0d, 0x3feddded, 0x50f228d6, 0xbc6e80c8, 0xd42ba2bf,
    0x3fd77102, 0x55764214, 0xbc66ead7, 0x314bb6ce, 0x3fedc6b7, 0xeb995912, 0x3c54b364, 0x776dcd35,
    0x3fd7e7ee, 0x03c86d4e, 0xbc7b63bc, 0xdabf5af2, 0x3fedaf0b, 0x6b888e83, 0x3c8a249e, 0x2b5e5cea,
    0x3fd85e7a, 0x12826949, 0x3c78a40e, 0x9b5face0, 0x3fed96e8, 0x2f71a9dc, 0x3c8ff61b, 0xd5d2039d,
    0x3fd8d4a4, 0xa774992f, 0x3c744a02, 0xea766326, 0x3fed7e4e, 0x97e17b4a, 0xbc63b770, 0x352bed94,
    0x3fd94a6b, 0xe9f546c5, 0xbc769ce1, 0x3e683f58, 0x3fed653f, 0x073e4040, 0xbc876236, 0x434bec37,
    0x3fd9bfce, 0x02e80510, 0x3c709e39, 0xa320b0a4, 0x3fed4bb9, 0xe1c619e0, 0x3c8f34bb, 0x77858f61,
    0x3fda34c9, 0x1cc50cca, 0xbc5a310e, 0x3b50cecd, 0x3fed31bf, 0x8d8d7c06, 0x3c7e60dd, 0x3089cbdd,
    0x3fdaa95b, 0x63a09277, 0xbc66293e, 0xb13c0381, 0x3fed1750, 0x727d94f0, 0x3c80d52b, 0x1ec1a48e,
    0x3fdb1d83, 0x05321617, 0xbc7ae242, 0xcb99f519, 0x3fecfc6c, 0xfa52ad9f, 0x3c88b5b5, 0x508f2a0d,
    0x3fdb913e, 0x30dbac43, 0xbc7e38ad, 0x2f6c3ff1, 0x3fece115, 0x909a82e5, 0x3c81f139, 0xbb31109a,
    0x3fdc048b, 0x17b140a3, 0x3c619fe6, 0x757e9fa7, 0x3fecc54a, 0xa2b2972e, 0x3c64ee16, 0x2ba83a98,
    0x3fdc7767, 0xec7fd19e, 0xbc5eb14d, 0x1a3d5826, 0x3feca90c, 0x9fc67d0b, 0xbc646a81, 0x485e3462,
    0x3fdce9d2, 0xe3d4a51f, 0xbc62fc8a, 0x12dae298, 0x3fec8c5b, 0xf8ce1a84, 0x3c7ab3d1, 0xa1590123,
    0x3fdd5bca, 0x34047661, 0x3c728a44, 0xa75fc29c, 0x3fec6f39, 0x208be53b, 0xbc8741db, 0xfbaadb42,
    0x3fddcd4c, 0x15329c9a, 0x3c70d4c6, 0xe171fd9a, 0x3fec51a4, 0x8b8b175e, 0xbc61bbb4, 0x3b9aa880,
    0x3fde3e56, 0xc1582a69, 0xbc50a482, 0x1099f88f, 0x3fec339e, 0xb01ddd81, 0xbc8caaf5, 0xee82c5c0,
    0x3fdeaee8, 0x744b05f0, 0xbc5789b4, 0x3c9b027d, 0x3fec1528, 0x065b7d50, 0xbc889211, 0x1312e828,
    0x3fdf1eff, 0x6bc4f97b, 0x3c717212, 0xf8a7525c, 0x3febf641, 0x081e7536, 0x3c8b7bd7, 0x1628a9a1,
    0x3fdf8e99, 0xe76abc97, 0x3c59d950, 0xaf2d00a3, 0x3febd6ea, 0x310294f5, 0x3c731bbc, 0xc88c109d,
    0x3fdffdb6, 0x28d2f57a, 0x3c6f4a99, 0x2e905b6a, 0x3febb723, 0xfe630f32, 0x3c772bd2, 0x452d0a39,
    0x3fe03629, 0x39c69955, 0xbc82d8cd, 0x78397b01, 0x3feb96ee, 0xef58840e, 0x3c545a3c, 0xc78fade0,
    0x3fe06d36, 0x86946e5b, 0x3c83f5ae, 0x4538ff1b, 0x3feb764b, 0x84b704c2, 0xbc8f5848, 0xc21b389b,
    0x3fe0a402, 0x1e9e1001, 0xbc86f643, 0xa13914f6, 0x3feb553a, 0x410c104e, 0x3c58ff79, 0x47027a16,
    0x3fe0da8b, 0x26b5672e, 0xbc8a58de, 0xf0bee909, 0x3feb33bb, 0xa89c8948, 0x3c8ea6a5, 0x1d1f6ca9,
    0x3fe110d0, 0xc4b69c3b, 0x3c8d9189, 0x98809981, 0x3feb11d0, 0x4162a4c6, 0x3c71dd56, 0x1efbc0c2,
    0x3fe146d2, 0x1f8b7f82, 0x3c7bf953, 0x5e2739a8, 0x3feaef78, 0x930bd275, 0xbc7f8362, 0x79746f94,
    0x3fe17c8e, 0x5f2eedb0, 0x3c635e57, 0x102e2488, 0x3feaccb5, 0x26f69de5, 0x3c88fb6a, 0x8dd6b6cc,
    0x3fe1b204, 0xacb02fdd, 0xbc5f190c, 0x70cbb5ff, 0x3feaa986, 0x88308913, 0xbc0b83d6, 0x07cd5070,
    0x3fe1e734, 0x3236574c, 0x3c722a3f, 0xa4f41d5a, 0x3fea85ed, 0x4373e02d, 0x3c69be06, 0x385ec792,
    0x3fe21c1c, 0x1b0394cf, 0x3c5e5b32, 0x4b23aa31, 0x3fea61e9, 0xe72586af, 0x3c858330, 0xe2fd453f,
    0x3fe250bb, 0x93788bbb, 0x3c7ea3d0, 0x2457bcce, 0x3fea3d7d, 0x0352bdcf, 0xbc868dba, 0xeca19669,
    0x3fe28511, 0xc917a067, 0xbc801df1, 0xd9a16b70, 0x3fea18a7, 0x29aee445, 0x3c395e25, 0x736c0358,
    0x3fe2b91d, 0xea88421e, 0xbc8fa371, 0xdb216ab0, 0x3fe9f368, 0xed912f85, 0xbc81d200, 0xc5791606,
    0x3fe2ecdf, 0x279a3082, 0x3c8d3557, 0xe0e7e37e, 0x3fe9cdc2, 0xe3f25e5c, 0x3c83f991, 0x12993f62,
    0x3fe32054, 0xb148bc4f, 0x3c8f6b42, 0x095a135b, 0x3fe9a7b5, 0xa36a6514, 0x3c8722cf, 0xcc9fa7a9,
    0x3fe3537d, 0xb9be0367, 0x3c6b327e, 0x7af040f0, 0x3fe98141, 0xc42e1310, 0x3c8d1ff8, 0x0488f08d,
    0x3fe38659, 0x7456282b, 0xbc710fad, 0xa93b07a8, 0x3fe95a67, 0xe00cb1fd, 0xbc80befd, 0xa21f862d,
    0x3fe3b8e7, 0x15a2840a, 0xbc797653, 0xa7d2f07b, 0x3fe93328, 0x926d9e92, 0xbc8bb770, 0x03600cda,
    0x3fe3eb25, 0xd36cd53a, 0xbc5be570, 0xe1570fc0, 0x3fe90b84, 0x784ddaf7, 0xbc70feb1, 0x0ab93b87,
    0x3fe41d14, 0xe4ba6790, 0x3c84608f, 0xd287ecf5, 0x3fe8e37c, 0x303d9ad1, 0xbc6463a4, 0xb53d4bf8,
    0x3fe44eb3, 0x81cf386b, 0xbc83ed6c, 0x1e6a5505, 0x3fe8bb10, 0x5a5dc900, 0x3c8863e0, 0x3e9474c1,
    0x3fe48000, 0xe431159f, 0xbc8b194a, 0x7463ed10, 0x3fe89241, 0x985d871f, 0x3c8c48d9, 0xc413ed84,
    0x3fe4b0fc, 0x46aab761, 0x3c20da05, 0x738cc59a, 0x3fe86910, 0x8d77a6c6, 0x3c7338ff, 0xe2bfe9dd,
    0x3fe4e1a4, 0xe54ed51b, 0xbc8a492f, 0x89b7c76a, 0x3fe83f7d, 0xde701ca0, 0xbc4152cf, 0x609bc6e8,
    0x3fe511f9, 0xfd7b351c, 0xbc85c0e8, 0x61c48831, 0x3fe8158a, 0x31916d5d, 0xbc6de8b9, 0x0b8228de,
    0x3fe541fa, 0xcddbb724, 0x3c7232c2, 0x8520d391, 0x3fe7eb36, 0x2eaa1488, 0x3c5a1d65, 0xa4a5959f,
    0x3fe571a6, 0x966d59b3, 0x3c5c843b, 0x4d0fb198, 0x3fe7c082, 0x7f09e54f, 0xbc6c73d6, 0xd72aee68,
    0x3fe5a0fc, 0x98813a12, 0xbc8d82e2, 0xb7d4227b, 0x3fe7956f, 0xcd7f6543, 0xbc8ab276, 0xe9d45ae4,
    0x3fe5cffc, 0x16bf8f0d, 0x3c896cb3, 0x70eb578a, 0x3fe769fe, 0xc655211f, 0xbc6827d5, 0xcf8c68c5,
    0x3fe5fea4, 0x552a9e57, 0x3c80b6ce, 0xf7ee20b7, 0x3fe73e30, 0x174efba1, 0xbc65d3ae, 0x3d94ad5f,
    0x3fe62cf4, 0x9921ac79, 0xbc8edd98, 0x55b6241a, 0x3fe71204, 0x6fa77678, 0x3c8425b0, 0xa5029c81,
    0x3fe65aec, 0x2963e755, 0x3c8126f9, 0x6b71053c, 0x3fe6e57c, 0x800cf55e, 0x3c860286, 0xdedbd0a6,
    0x3fe6888a, 0x4e134b2f, 0xbc86b7d3, 0x7644d5e6, 0x3fe6b898, 0xfa9efb5d, 0x3c715ac7, 0x86ccf4b2,
    0x3fe6b5ce, 0x50b7821a, 0xbc65d515, 0x8f702e0f, 0x3fe68b5a, 0x92eb6253, 0xbc89a91a, 0xd985f89c,
    0x3fe6e2b7, 0x7c40bde1, 0xbc70e729, 0x857fad53, 0x3fe65dc1, 0xfdeb8cba, 0xbc597c1b, 0x47337c77,
    0x3fe70f45, 0x1d0a8c40, 0x3c697ede, 0x3885770d, 0x3fe62fcf, 0xf20191c7, 0x3c6d9143, 0x895756ef,
    0x3fe73b76, 0x80dea578, 0xbc722483, 0x06dc12a2, 0x3fe60185, 0x26f563df, 0x3c846ca5, 0xe0e432d0,
    0x3fe7674a, 0xf6f7b524, 0x3c7e9d3f, 0x94ac84a8, 0x3fe5d2e2, 0x55f1f17a, 0x3c803141, 0x04c8892b,
    0x3fe792c1, 0xd0041d52, 0xbc8abf05, 0xeeb354eb, 0x3fe5a3e8, 0x39824077, 0x3c8428aa, 0x2759be62,
    0x3fe7bdda, 0x5e28b3c2, 0x3c4ad119, 0x7ccd0393, 0x3fe57497, 0x8d8e83f2, 0x3c8f4714, 0xaf282d23,
    0x3fe7e893, 0xf5037959, 0x3c80eefb, 0xaa650c4c, 0x3fe544f1, 0x0f592ca5, 0xbc8e7ae8, 0xe6c7a62f,
    0x3fe812ed, 0xe9ae4ba4, 0xbc87830a, 0xdf402dda, 0x3fe514f5, 0x7d7bf3da, 0x3c747a10, 0x8073c259,
  ];

  // ---------------------------------------------------------------------------
  // erf (fdlibm s_erf.c, as in glibc). Only used to choose ``trunc_normal_``'s
  // sampling method, where last-bit differences cannot change the outcome.
  // ---------------------------------------------------------------------------

  const ERX = 8.45062911510467529297e-01;
  const EFX = 1.28379167095512586316e-01;
  const PP = [1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02, -5.77027029648944159157e-03, -2.37630166566501626084e-05];
  const QQ = [0, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03, 1.32494738004321644526e-04, -3.96022827877536812320e-06];
  const PA = [-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01, 3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02, -2.16637559486879084300e-03];
  const QA = [0, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02, 1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02];
  const RA = [-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e+01, -6.23753324503260060396e+01, -1.62396669462573470355e+02, -1.84605092906711035994e+02, -8.12874355063065934246e+01, -9.81432934416914548592e+00];
  const SA = [0, 1.96512716674392571292e+01, 1.37657754143519042600e+02, 4.34565877475229228821e+02, 6.45387271733267880336e+02, 4.29008140027567833386e+02, 1.08635005541779435134e+02, 6.57024977031928170135e+00, -6.04244152148580987438e-02];
  const RB = [-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e+01, -1.60636384855821916062e+02, -6.37566443368389627722e+02, -1.02509513161107724954e+03, -4.83519191608651397019e+02];
  const SB = [0, 3.03380607434824582924e+01, 3.25792512996573918826e+02, 1.53672958608443695994e+03, 3.19985821950859553908e+03, 2.55305040643316442583e+03, 4.74528541206955367215e+02, -2.24409524465858183362e+01];

  /** The error function (fdlibm algorithm). */
  function erf(x: number): number {
    if (Number.isNaN(x)) return x;
    if (x === Infinity) return 1;
    if (x === -Infinity) return -1;
    const hx = highWord(x) | 0;
    const ix = hx & 0x7fffffff;
    if (ix < 0x3feb0000) {
      if (ix < 0x3e300000) return x + EFX * x;
      const z = x * x;
      const r = PP[0]! + z * (PP[1]! + z * (PP[2]! + z * (PP[3]! + z * PP[4]!)));
      const s = 1 + z * (QQ[1]! + z * (QQ[2]! + z * (QQ[3]! + z * (QQ[4]! + z * QQ[5]!))));
      return x + x * (r / s);
    }
    if (ix < 0x3ff40000) {
      const s = Math.abs(x) - 1;
      const P = PA[0]! + s * (PA[1]! + s * (PA[2]! + s * (PA[3]! + s * (PA[4]! + s * (PA[5]! + s * PA[6]!)))));
      const Q = 1 + s * (QA[1]! + s * (QA[2]! + s * (QA[3]! + s * (QA[4]! + s * (QA[5]! + s * QA[6]!)))));
      return hx >= 0 ? ERX + P / Q : -ERX - P / Q;
    }
    if (ix >= 0x40180000) return hx >= 0 ? 1 : -1;
    const ax = Math.abs(x);
    const s = 1 / (ax * ax);
    let R: number;
    let S: number;
    if (ix < 0x4006db6e) {
      R = RA[0]! + s * (RA[1]! + s * (RA[2]! + s * (RA[3]! + s * (RA[4]! + s * (RA[5]! + s * (RA[6]! + s * RA[7]!))))));
      S = 1 + s * (SA[1]! + s * (SA[2]! + s * (SA[3]! + s * (SA[4]! + s * (SA[5]! + s * (SA[6]! + s * (SA[7]! + s * SA[8]!)))))));
    } else {
      R = RB[0]! + s * (RB[1]! + s * (RB[2]! + s * (RB[3]! + s * (RB[4]! + s * (RB[5]! + s * RB[6]!)))));
      S = 1 + s * (SB[1]! + s * (SB[2]! + s * (SB[3]! + s * (SB[4]! + s * (SB[5]! + s * (SB[6]! + s * SB[7]!))))));
    }
    const z = fromWords(highWord(ax), 0);
    const r = Math.exp(-z * z - 0.5625) * Math.exp((z - ax) * (z + ax) + R / S);
    return hx >= 0 ? 1 - r / ax : r / ax - 1;
  }

  // ---------------------------------------------------------------------------
  // Box-Muller blocks of ATen's 16-wide ``normal_`` fill (``normal_fill_16``).
  // ---------------------------------------------------------------------------

  const TWO_PI = 2 * Math.PI;

  /** Transform 16 float32 uniforms at ``offset`` (first half ``u1``, second ``u2``) into normal samples. */
  function normalFill16Float(buffer: Float64Array, offset: number, mean: number, std: number): void {
    const identity = std === 1 && mean === 0;
    for (let j = 0; j < 8; j += 1) {
      const u1 = Math.fround(1 - buffer[offset + j]!);
      const u2 = buffer[offset + j + 8]!;
      const radius = Math.fround(Math.sqrt(Math.fround(-2 * logf(u1))));
      sincosfPair(Math.fround(TWO_PI * u2));
      const first = Math.fround(radius * sincosfResult.cos);
      const second = Math.fround(radius * sincosfResult.sin);
      buffer[offset + j] = identity ? first : fmaf(first, std, mean);
      buffer[offset + j + 8] = identity ? second : fmaf(second, std, mean);
    }
  }

  /** Transform 16 float64 uniforms at ``offset`` into normal samples. */
  function normalFill16Double(buffer: Float64Array, offset: number, mean: number, std: number): void {
    for (let j = 0; j < 8; j += 1) {
      const u1 = 1 - buffer[offset + j]!;
      const u2 = buffer[offset + j + 8]!;
      const radius = Math.sqrt(-2 * log(u1));
      const theta = TWO_PI * u2;
      buffer[offset + j] = fma(radius * cos(theta), std, mean);
      buffer[offset + j + 8] = fma(radius * sin(theta), std, mean);
    }
  }

  return {
    lowWord, highWord, floatBits, floatFromBits, fma, fmaf, logf, sincosfResult, sincosfPair, sinf, cosf, log, log1p, sin, cos, erf, normalFill16Float, normalFill16Double,
  };
}

/** The functions of {@link createLibm}, bound once for this thread. */
export type Libm = ReturnType<typeof createLibm>;

const libm: Libm = createLibm();

export const {
  lowWord, highWord, floatBits, floatFromBits, fma, fmaf, logf, sincosfResult, sincosfPair, sinf, cosf, log, log1p, sin, cos, erf, normalFill16Float, normalFill16Double,
} = libm;
