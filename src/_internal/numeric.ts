/**
 * Python-exact float summation shared by the runtime modules.
 *
 * - {@link fsum}: ``math.fsum`` (exactly rounded, Shewchuk partials).
 * - {@link pythonSum}: builtin ``sum`` over floats. CPython 3.12+ uses
 *   Neumaier compensated summation, which the reference fixtures reflect.
 */

/** Python ``math.fsum``: exactly rounded floating point sum (Shewchuk partials). */
export function fsum(values: Iterable<number>): number {
  const partials: number[] = [];
  for (let x of values) {
    let index = 0;
    for (let y of partials) {
      if (Math.abs(x) < Math.abs(y)) [x, y] = [y, x];
      const high = x + y;
      const low = y - (high - x);
      if (low) partials[index++] = low;
      x = high;
    }
    partials.length = index;
    partials.push(x);
  }
  let total = 0;
  let index = partials.length;
  if (index) {
    total = partials[--index]!;
    while (index > 0) {
      const x = total;
      const y = partials[--index]!;
      total = x + y;
      const low = y - (total - x);
      if (low) {
        if (index > 0 && ((low < 0 && partials[index - 1]! < 0) || (low > 0 && partials[index - 1]! > 0))) {
          const doubled = low * 2;
          const candidate = total + doubled;
          if (doubled === candidate - total) total = candidate;
        }
        break;
      }
    }
  }
  return total;
}

/** Python (>= 3.12) ``sum`` of floats: left to right with Neumaier compensation. */
export function pythonSum(values: Iterable<number>): number {
  let total = 0;
  let compensation = 0;
  for (const value of values) {
    const next = total + value;
    if (Math.abs(total) >= Math.abs(value)) compensation += (total - next) + value;
    else compensation += (value - next) + total;
    total = next;
  }
  // ``if (c && isfinite(c)) f_result += c;`` (Python/bltinmodule.c)
  return compensation !== 0 && Number.isFinite(compensation) ? total + compensation : total;
}
