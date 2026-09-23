/**
 * TensorCode: composable cognitive operations and traceable, trainable programs.
 *
 * The root entry mirrors Python ``tensorcode``: tracing types only. Importing it
 * performs no I/O and loads no model code. Subpath modules provide the rest:
 * ``tensorcode/ops``, ``tensorcode/ops/vec``, ``tensorcode/ops/text``,
 * ``tensorcode/ops/graph``, ``tensorcode/tools``, ``tensorcode/training``,
 * ``tensorcode/integrations`` and the numerical core ``tensorcode/nn``.
 */
export { Trace, trace, InputRef, OutputRef, type ReplayOptions, type SuperviseOptions, type SaveOptions } from './_internal/tracing.js';
export { ValueError, NotImplementedError, MissingDependencyError } from './errors.js';
/**
 * Python values in JavaScript data: ``float(0)`` is written ``0.0`` in
 * configurations, fingerprints, targets and saved files (Python ``float``);
 * ``int(0)`` keeps ``0`` even under a float configuration field;
 * ``orderedObject(entries)`` keeps a Python dict's insertion order for
 * integer-like keys (a ``Map`` works where a mapping is expected).
 */
export { float, int, orderedObject } from './_internal/json.js';
export type { PythonFloat, PythonInt } from './_internal/json.js';

/** Package version (Python ``tensorcode.__version__ == '0.4.0a3'``). */
export const version = '0.4.0-alpha.3';
