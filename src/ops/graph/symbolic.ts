/**
 * Shared declaration for symbolic operations that have no implementation yet
 * (Python ``tensorcode/ops/graph/_symbolic.py``).
 */
import { NotImplementedError } from '../../errors.js';
import { validatedConfig } from '../../_internal/operationConfig.js';
import type { JsonObject } from '../../_internal/json.js';
import { Operation, type Context } from '../base.js';

/** A callable API declaration, never a callback or neural fallback. */
export abstract class SymbolicOperation<I = unknown, O = unknown> extends Operation<I, O> {
  static override readonly qualifiedName: string = 'tensorcode.ops.graph._symbolic.SymbolicOperation';
  readonly config: JsonObject;

  constructor(config: JsonObject | null = null) {
    super();
    this.config = validatedConfig(config, []);
  }

  /** Reserved; always throws ``NotImplementedError``. */
  static async fromFoundation(...args: unknown[]): Promise<never> {
    void args;
    throw unimplemented(this.name);
  }

  /** Reserved; always throws ``NotImplementedError``. */
  static async fromPretrained(...args: unknown[]): Promise<never> {
    void args;
    throw unimplemented(this.name);
  }

  /** Reserved; always throws ``NotImplementedError`` and writes nothing. */
  async savePretrained(...args: unknown[]): Promise<never> {
    void args;
    throw unimplemented(this.constructor.name);
  }

  configuration(): { operation: string; implementation: 'unimplemented' } {
    return { operation: `graph.${this.constructor.name}`, implementation: 'unimplemented' };
  }

  /** Reserved symbolic API; always throws ``NotImplementedError``. */
  forward(value: I, context: Context | null): O {
    void value;
    void context;
    throw unimplemented(this.constructor.name);
  }
}

function unimplemented(name: string): NotImplementedError {
  return new NotImplementedError(
    `graph.${name} symbolic semantics are not implemented. `
    + 'Graph values can store explicitly supplied structure; they do not infer meaning from inputs.',
  );
}
