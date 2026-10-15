import type { Program } from './core.js';
export interface Interval { min: number; max: number; integer: boolean }
/** Conservative interval diagnostics. Runtime guards remain authoritative. */
export function analyze(program: Program): { ranges: Interval[]; warnings: { node: number; reason: string }[]; nodeCount: number; depth: number } {
  const ir = program.graph(), ranges: Interval[] = [], warnings: { node: number; reason: string }[] = [];
  const outward = (x: number, up: boolean): number => {
    if (!Number.isFinite(x)) return x;
    if (x === 0) return up ? Number.MIN_VALUE : -Number.MIN_VALUE;
    const data = new DataView(new ArrayBuffer(8)); data.setFloat64(0, x); let bits = data.getBigUint64(0); bits += (x > 0) === up ? 1n : -1n; data.setBigUint64(0, bits); return data.getFloat64(0);
  };
  for (const n of ir.nodes) {
    const op = n.op, args = n.args.map(a => ranges[a]!), d = n.domain;
    let lo = d.min, hi = d.max, integer = ['int32', 'bool'].includes(d.dtype);
    if (op === 'const') { lo = hi = n.value!; integer = Number.isInteger(lo); }
    else if (['add', 'sub', 'mul', 'div'].includes(op)) {
      const a = args[0]!, b = args[1]!; integer = a.integer && b.integer && op !== 'div';
      if (op === 'add') { lo = a.min + b.min; hi = a.max + b.max; }
      else if (op === 'sub') { lo = a.min - b.max; hi = a.max - b.min; }
      else if (op === 'mul') { const products = [a.min * b.min, a.min * b.max, a.max * b.min, a.max * b.max]; lo = Math.min(...products); hi = Math.max(...products); }
      else if (b.min <= 0 && b.max >= 0) warnings.push({ node: n.id, reason: 'divisor may be zero' });
      else { const values = [a.min / b.min, a.min / b.max, a.max / b.min, a.max / b.max]; lo = Math.min(...values); hi = Math.max(...values); }
      if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo < d.min || hi > d.max) warnings.push({ node: n.id, reason: 'arithmetic may exceed finite domain' });
      if (integer && (lo < -Number.MAX_SAFE_INTEGER || hi > Number.MAX_SAFE_INTEGER)) warnings.push({ node: n.id, reason: 'integer arithmetic may lose precision' });
      if (!integer) { lo = outward(lo, false); hi = outward(hi, true); }
    } else if (['gt', 'ge', 'lt', 'le', 'eq', 'ne'].includes(op)) { lo = 0; hi = 1; integer = (n.semantics ?? ir.semantics) !== 'relaxed'; }
    else if (op === 'select') { lo = Math.min(args[1]!.min, args[2]!.min); hi = Math.max(args[1]!.max, args[2]!.max); integer = args[1]!.integer && args[2]!.integer && (n.semantics ?? ir.semantics) !== 'relaxed'; }
    else if (op === 'gather') { lo = Math.min(...args.slice(1).map(a => a.min)); hi = Math.max(...args.slice(1).map(a => a.max)); integer = args.slice(1).every(a => a.integer); if (!args[0]!.integer || args[0]!.min < 0 || args[0]!.max >= args.length - 1) warnings.push({ node: n.id, reason: 'gather index may be invalid' }); }
    else if (op === 'relu') { lo = Math.max(0, args[0]!.min); hi = Math.max(0, args[0]!.max); integer = args[0]!.integer; }
    else if (op === 'neg') { lo = -args[0]!.max; hi = -args[0]!.min; integer = args[0]!.integer; }
    else if (op === 'sigmoid') { lo = 0; hi = 1; }
    else if (op === 'call') warnings.push({ node: n.id, reason: 'explicit external computation boundary' });
    if (d.dtype === 'float32') { lo = Math.fround(lo); hi = Math.fround(hi); }
    ranges.push({ min: Math.max(d.min, lo), max: Math.min(d.max, hi), integer });
  }
  return { ranges, warnings, nodeCount: ir.nodes.length, depth: program.schedule().length };
}
