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

/** Python ``KeyError``: a missing mapping key (for example an unknown ``rope_type``). */
export class KeyError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** Python ``IndexError``: an index outside a sequence (for example an empty image batch). */
export class IndexError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** Python ``RuntimeError``, where Python's own code raises it by name. */
export class RuntimeError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** Python ``AttributeError``, where transformers raises it for a configuration. */
export class AttributeError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** Python ``ImportError``, where transformers raises it for a missing optional package. */
export class ImportError extends Error {
  constructor(message?: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}
