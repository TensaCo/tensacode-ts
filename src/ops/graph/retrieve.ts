/** Reserved symbolic operation contracts; execution is not implemented (Python ``ops/graph/retrieve.py``). */
import { SymbolicOperation } from './symbolic.js';
import type { Graph } from './representation.js';

/** Retrieve symbolic evidence relevant to a graph query. Reserved API: calling it throws ``NotImplementedError``. */
export class Retrieve extends SymbolicOperation<Graph, readonly Graph[]> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.retrieve.Retrieve';
}
