/** Reserved symbolic operation contracts; execution is not implemented (Python ``ops/graph/decode.py``). */
import { SymbolicOperation } from './symbolic.js';
import type { Graph } from './representation.js';

/** Realize symbolic structure in an output representation. Reserved API: calling it throws ``NotImplementedError``. */
export class Decode extends SymbolicOperation<Graph, unknown> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.decode.Decode';
}

/**
 * Realize symbolic content as text without inventing unsupported
 * relationships. Reserved API: calling it throws ``NotImplementedError``.
 */
export class TextDecode extends SymbolicOperation<Graph, string> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.decode.TextDecode';
}
