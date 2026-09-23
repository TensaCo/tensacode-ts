/**
 * Error types shared across the package.
 *
 * Python raises built-in ``ValueError``/``TypeError``/``NotImplementedError``.
 * TypeScript maps them to:
 *
 * - ``TypeError`` (built in) for wrong value kinds,
 * - {@link ValueError} for well-typed but invalid values (Python ``ValueError``),
 * - {@link NotImplementedError} for reserved, unimplemented interfaces,
 * - ``RangeError`` only inside the numerical core (shape/index errors).
 *
 * Library-specific subclasses (``InvalidModelOutput``, ``ProviderError`` ...)
 * extend these so ``instanceof ValueError`` keeps matching Python semantics.
 */

export class ValueError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

export class NotImplementedError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** A required optional dependency (for example ``@huggingface/transformers``) is absent. */
export class MissingDependencyError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}
