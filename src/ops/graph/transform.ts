/** Reserved symbolic operation contracts; execution is not implemented (Python ``ops/graph/transform.py``). */
import { SymbolicOperation } from './symbolic.js';
import type { Graph } from './representation.js';

/** Revise symbolic structure in light of explicitly supplied context. Reserved API: calling it throws ``NotImplementedError``. */
export class Transform extends SymbolicOperation<Graph, Graph> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.transform.Transform';
}
