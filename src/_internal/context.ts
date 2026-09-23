/**
 * An async-aware context variable (the equivalent of Python's ``ContextVar``).
 *
 * On Node.js it uses ``AsyncLocalStorage`` so values follow ``await``
 * continuations. Where that is unavailable a synchronous stack is used: values
 * are then visible only until the first ``await`` inside ``run``.
 */

interface Storage<T> {
  getStore(): T | undefined;
  run<R>(store: T, callback: () => R): R;
}

type AsyncLocalStorageConstructor = new <T>() => Storage<T>;

function loadAsyncLocalStorage(): AsyncLocalStorageConstructor | null {
  const processLike = (globalThis as { process?: { getBuiltinModule?: (name: string) => unknown } }).process;
  const loader = processLike?.getBuiltinModule;
  if (typeof loader !== 'function') return null;
  try {
    const module = loader('node:async_hooks') as { AsyncLocalStorage?: AsyncLocalStorageConstructor } | undefined;
    return module?.AsyncLocalStorage ?? null;
  } catch {
    return null;
  }
}

const AsyncLocalStorageClass = loadAsyncLocalStorage();

class StackStorage<T> implements Storage<T> {
  private readonly stack: (T | undefined)[] = [];

  getStore(): T | undefined {
    return this.stack.length ? this.stack[this.stack.length - 1] : undefined;
  }

  run<R>(store: T, callback: () => R): R {
    this.stack.push(store);
    try {
      return callback();
    } finally {
      this.stack.pop();
    }
  }
}

/** Whether context values propagate across ``await`` in this runtime. */
export const asyncContextSupported = AsyncLocalStorageClass !== null;

export class ContextVariable<T> {
  private readonly storage: Storage<{ value: T | null }>;

  constructor() {
    this.storage = AsyncLocalStorageClass ? new AsyncLocalStorageClass() : new StackStorage();
  }

  get(): T | null {
    return this.storage.getStore()?.value ?? null;
  }

  /** Run ``callback`` with this variable set to ``value`` (``null`` clears it). */
  run<R>(value: T | null, callback: () => R): R {
    return this.storage.run({ value }, callback);
  }
}
