/**
 * Stable persisted identities for classes.
 *
 * Artifacts, operation fingerprints and experience bindings name classes by the
 * Python qualified name of the equivalent class (for example
 * ``tensorcode.tools.investigator.Investigator``) so that TypeScript and Python
 * artifacts stay interchangeable. Every public class declares its own
 * ``static readonly qualifiedName``. The lookup only honours a class's *own*
 * static property: an application subclass never silently inherits the
 * identity of a library class. Classes without one are identified as
 * ``js:<ClassName>``.
 */

export const QUALIFIED_NAME = 'qualifiedName';

type Constructor = abstract new (...args: never[]) => unknown;

/** Qualified identity of a class or of an instance's class. */
export function qualifiedName(value: unknown): string {
  const cls = (typeof value === 'function' ? value : (value as { constructor?: unknown } | null)?.constructor) as
    | (Constructor & { qualifiedName?: unknown; name?: string })
    | undefined;
  if (!cls) return 'js:Object';
  if (Object.prototype.hasOwnProperty.call(cls, QUALIFIED_NAME) && typeof cls.qualifiedName === 'string') {
    return cls.qualifiedName;
  }
  return `js:${cls.name || 'anonymous'}`;
}

/** Python ``type(value).__name__`` equivalent for diagnostics and manifests. */
export function className(value: unknown): string {
  const cls = (typeof value === 'function' ? value : (value as { constructor?: unknown } | null)?.constructor) as
    | { name?: string }
    | undefined;
  return cls?.name || 'Object';
}
