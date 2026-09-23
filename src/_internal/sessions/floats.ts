/**
 * Float-typed fields of persisted tool sessions. JavaScript numbers cannot
 * distinguish ``1`` from ``1.0``; model outputs that Python always stores as
 * floats (probabilities, scores, attention weights, NLI distributions) are
 * spelled as floats by the session writers, in addition to the global
 * {@link PYTHON_FLOAT_KEYS}. The global registry is never mutated.
 */
import { PYTHON_FLOAT_KEYS, SESSION_FLOAT_FIELDS } from '../json.js';

export { SESSION_FLOAT_FIELDS };

/** Float keys for session JSON writers (computed per call; includes application-registered keys). */
export function sessionFloatKeys(): ReadonlySet<string> {
  return new Set([...PYTHON_FLOAT_KEYS, ...SESSION_FLOAT_FIELDS]);
}
