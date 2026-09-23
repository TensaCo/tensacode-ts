/** Reserved symbolic operation contracts; execution is not implemented (Python ``ops/graph/decide.py``). */
import { ValueError } from '../../errors.js';
import { SymbolicOperation } from './symbolic.js';
import { Graph } from './representation.js';

/** Explicit objective and alternatives; this record supplies no policy. */
export class ChoiceInput {
  static readonly qualifiedName: string = 'tensorcode.ops.graph.decide.ChoiceInput';
  static readonly recordFields = ['objective', 'options'] as const;
  readonly objective: Graph;
  readonly options: readonly Graph[];

  constructor(objective: Graph, options: Iterable<Graph>) {
    const alternatives = [...options];
    if (!(objective instanceof Graph)) throw new TypeError('Choice objective must be a Graph');
    if (!alternatives.length || !alternatives.every((option) => option instanceof Graph)) {
      throw new ValueError('Choice options must contain at least one Graph');
    }
    this.objective = objective;
    this.options = Object.freeze(alternatives);
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): ChoiceInput {
    return new ChoiceInput(fields.objective as Graph, fields.options as readonly Graph[]);
  }

  toRecord(): Record<string, unknown> {
    return { objective: this.objective, options: this.options };
  }
}

/** Select among explicit symbolic alternatives for an objective. Reserved API: calling it throws ``NotImplementedError``. */
export class Decide extends SymbolicOperation<ChoiceInput, Graph> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph.decide.Decide';
}
