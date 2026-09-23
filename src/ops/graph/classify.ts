/** Reserved symbolic operation contracts; execution is not implemented (Python ``ops/graph/classify.py``). */
import { SymbolicOperation } from './symbolic.js';
import type { Graph } from './representation.js';

/** Classify symbolic structure using explicitly supplied categories. Reserved API: calling it throws ``NotImplementedError``. */
export class Classify extends SymbolicOperation<Graph, string> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.classify.Classify';
}
