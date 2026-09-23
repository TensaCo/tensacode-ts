/** Reserved symbolic operation contracts; execution is not implemented (Python ``ops/graph/encode.py``). */
import { SymbolicOperation } from './symbolic.js';
import type { Graph } from './representation.js';

/** Encode input evidence into source-grounded symbolic structure. Reserved API: calling it throws ``NotImplementedError``. */
export class Encode extends SymbolicOperation<unknown, Graph> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.encode.Encode';
}

/**
 * Interpret text as symbolic relationships while preserving its evidence and
 * uncertainty. Reserved API: calling it throws ``NotImplementedError``.
 */
export class TextEncode extends SymbolicOperation<string, Graph> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.encode.TextEncode';
}
