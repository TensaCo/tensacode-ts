/** Reserved symbolic operation contracts; execution is not implemented (Python ``ops/graph/score.py``). */
import { SymbolicOperation } from './symbolic.js';
import type { Graph } from './representation.js';

/** Assess symbolic structure under an explicitly configured objective. Reserved API: calling it throws ``NotImplementedError``. */
export class Score extends SymbolicOperation<Graph, number> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.score.Score';
}
