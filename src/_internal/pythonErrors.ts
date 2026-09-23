/**
 * Python exception names for JavaScript errors.
 *
 * Persisted observations (for example plan ``error_type`` fields and replan
 * ``policy_errors``) record ``type(exc).__name__`` in Python. JavaScript's
 * built-in errors have different names, so they are mapped to the exception
 * Python raises in the same situation. Library and user error classes keep
 * their own class names, exactly as Python records a custom exception class.
 */

/** Built-in JavaScript error names and the Python exception raised for the same failure. */
const BUILTIN_NAMES: Readonly<Record<string, string>> = Object.freeze({
  Error: 'RuntimeError',
  EvalError: 'RuntimeError',
  RangeError: 'ValueError',
  ReferenceError: 'NameError',
  SyntaxError: 'SyntaxError',
  TypeError: 'TypeError',
  URIError: 'ValueError',
  AggregateError: 'ExceptionGroup',
  InternalError: 'RecursionError',
  AbortError: 'CancelledError',
  TimeoutError: 'TimeoutError',
  MissingDependencyError: 'ImportError',
});

/** Node.js system error codes and the matching Python ``OSError`` subclass. */
const SYSTEM_CODES: Readonly<Record<string, string>> = Object.freeze({
  ENOENT: 'FileNotFoundError',
  EEXIST: 'FileExistsError',
  EACCES: 'PermissionError',
  EPERM: 'PermissionError',
  EISDIR: 'IsADirectoryError',
  ENOTDIR: 'NotADirectoryError',
  ETIMEDOUT: 'TimeoutError',
  ECONNREFUSED: 'ConnectionRefusedError',
  ECONNRESET: 'ConnectionResetError',
  ECONNABORTED: 'ConnectionAbortedError',
  EPIPE: 'BrokenPipeError',
  ECHILD: 'ChildProcessError',
  ESRCH: 'ProcessLookupError',
  EINTR: 'InterruptedError',
  EAGAIN: 'BlockingIOError',
});

const BUILTIN_CONSTRUCTORS = new Set<unknown>([
  Error, EvalError, RangeError, ReferenceError, SyntaxError, TypeError, URIError,
  ...(typeof AggregateError === 'function' ? [AggregateError] : []),
  ...(typeof DOMException === 'function' ? [DOMException] : []),
]);

/**
 * ``type(exc).__name__`` for a thrown value.
 *
 * - A class that extends ``Error`` without setting ``name`` reports its class
 *   name (``class Declined extends Error {}`` → ``Declined``).
 * - Built-in errors map to Python's equivalent (``Error`` → ``RuntimeError``,
 *   ``RangeError`` → ``ValueError``, ``ReferenceError`` → ``NameError``, ...).
 * - Node system errors map by ``code`` (``ENOENT`` → ``FileNotFoundError``,
 *   other ``errno`` errors → ``OSError``).
 * - A thrown non-error value reports ``Exception``.
 */
export function pythonErrorName(error: unknown): string {
  if (!(error instanceof Error) && !(typeof DOMException === 'function' && error instanceof DOMException)) {
    if (error !== null && typeof error === 'object' && typeof (error as { name?: unknown }).name === 'string'
      && typeof (error as { message?: unknown }).message === 'string') {
      return mapName((error as { name: string }).name);
    }
    return 'Exception';
  }
  const candidate = error as Error & { code?: unknown; errno?: unknown; syscall?: unknown };
  const constructor = (candidate as { constructor?: { name?: unknown } }).constructor;
  let name = typeof candidate.name === 'string' && candidate.name ? candidate.name : 'Error';
  const className = typeof constructor?.name === 'string' ? constructor.name : '';
  if (className && !BUILTIN_CONSTRUCTORS.has(constructor) && (name === 'Error' || name === builtinBase(candidate))) {
    // ``class Declined extends Error {}`` keeps ``name === 'Error'``; Python records the class.
    name = className;
  }
  if (name === 'Error' || BUILTIN_CONSTRUCTORS.has(constructor)) {
    const code = typeof candidate.code === 'string' ? candidate.code : null;
    if (code !== null && Object.hasOwn(SYSTEM_CODES, code)) return SYSTEM_CODES[code]!;
    if (code !== null && (typeof candidate.errno === 'number' || typeof candidate.syscall === 'string')) return 'OSError';
  }
  return mapName(name);
}

function builtinBase(error: Error): string | null {
  for (const base of BUILTIN_CONSTRUCTORS) {
    if (typeof base === 'function' && error instanceof (base as typeof Error)) {
      const name = (base as { name?: unknown }).name;
      if (name !== 'Error' && name !== 'DOMException' && typeof name === 'string') return name;
    }
  }
  return null;
}

function mapName(name: string): string {
  return Object.hasOwn(BUILTIN_NAMES, name) ? BUILTIN_NAMES[name]! : name;
}

/** ``str(exc)`` for a thrown value. */
export function pythonErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (error !== null && typeof error === 'object' && typeof (error as { message?: unknown }).message === 'string') {
    return (error as { message: string }).message;
  }
  return String(error);
}

/** Python's ``f'{type(exc).__name__}: {exc}'``. */
export function describePythonError(error: unknown): string {
  return `${pythonErrorName(error)}: ${pythonErrorMessage(error)}`;
}
