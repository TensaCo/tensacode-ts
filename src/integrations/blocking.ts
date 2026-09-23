/**
 * Synchronous calls for asynchronous providers (Python's blocking
 * ``complete``): the work runs in a worker thread while the calling thread
 * blocks on ``Atomics.wait`` and then reads the reply with
 * ``receiveMessageOnPort``.
 *
 * One worker per process is started on first use and never keeps the process
 * alive. Environments without ``node:worker_threads`` (browsers, edge
 * runtimes) cannot block a thread on a network or model call; there the
 * synchronous methods raise {@link SynchronousCallUnavailable} and the
 * asynchronous ones (``acomplete``/``acall``) remain available.
 */
import { MissingDependencyError, NotImplementedError, ValueError } from '../errors.js';
import { runLocalEngine } from './localEngine.js';

/** Synchronous provider calls are impossible in this JavaScript environment. */
export class SynchronousCallUnavailable extends NotImplementedError {}

type Threads = typeof import('node:worker_threads');

interface Bridge {
  readonly threads: Threads;
  readonly worker: import('node:worker_threads').Worker;
  readonly port: import('node:worker_threads').MessagePort;
  readonly signal: Int32Array;
  next: number;
}

/** A worker-side failure, rebuilt with its original name. */
export interface WorkerFailure {
  readonly name: string;
  readonly message: string;
}

let bridge: Bridge | null = null;

type ProcessLike = { getBuiltinModule?: (id: string) => unknown; versions?: { node?: string } };

function builtin<T>(id: string): T | null {
  const processLike = (globalThis as { process?: ProcessLike }).process;
  if (typeof processLike?.getBuiltinModule !== 'function') return null;
  try {
    return (processLike.getBuiltinModule(id) as T | undefined) ?? null;
  } catch {
    return null;
  }
}

/**
 * The worker program. It must stay self-contained: its source text is what
 * the worker evaluates (with ``runLocalEngine`` passed in as source text and
 * a dynamic ``import`` written in the worker's own source, which bundlers and
 * test transformers never rewrite).
 */
function workerMain(threads: Threads, engineSource: string, load: (specifier: string) => Promise<any>): void {
  const { port, signal } = threads.workerData as { port: import('node:worker_threads').MessagePort; signal: Int32Array };
  // eslint-disable-next-line no-eval
  const engine = (0, eval)(`(${engineSource})`) as typeof runLocalEngine;
  const models = new Map<number, { model: any; processor: any; loadImage: (image: any) => unknown }>();
  const notify = (): void => {
    Atomics.store(signal, 0, 1);
    Atomics.notify(signal, 0);
  };
  process.on('exit', () => {
    Atomics.store(signal, 0, 2);
    Atomics.notify(signal, 0);
  });
  // A CommonJS build imported from ESM exposes its exports on ``default``.
  const loadTransformers = async (url: string) => {
    const module = await load(url);
    return module.AutoProcessor ? module : (module.default ?? module);
  };
  const rawImageLoader = (transformersUrl: string | null) => async (image: { data: Uint8Array; mediaType: string | null }) => {
    if (!transformersUrl) {
      const error = new Error("LocalModel requires the optional peer dependency '@huggingface/transformers'; install it explicitly");
      error.name = 'MissingDependencyError';
      throw error;
    }
    const transformers = await loadTransformers(transformersUrl);
    const decoded = await transformers.RawImage.fromBlob(new Blob([image.data], image.mediaType ? { type: image.mediaType } : {}));
    return decoded.rgb();
  };
  const http = async (task: {
    url: string; headers?: Record<string, string>; body?: string | null; timeoutMs: number | null; method?: string; redirect?: 'follow' | 'manual' | 'error';
  }) => {
    const controller = new AbortController();
    let timedOut = false;
    const timer = task.timeoutMs === null ? null : setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, task.timeoutMs);
    try {
      let response: Response;
      try {
        response = await fetch(task.url, {
          method: task.method ?? 'POST', headers: task.headers ?? {}, ...(task.body === null || task.body === undefined ? {} : { body: task.body }),
          redirect: task.redirect ?? 'manual', signal: controller.signal,
        });
      } catch (error) {
        if (timedOut) return { outcome: 'timeout' };
        const reason = error instanceof Error ? ((error.cause as Error | undefined)?.message ?? error.message) : String(error);
        return { outcome: 'failed', reason };
      }
      let bytes: Uint8Array;
      try {
        bytes = new Uint8Array(await response.arrayBuffer());
      } catch {
        if (timedOut) return { outcome: 'timeout' };
        return { outcome: 'unreadable' };
      }
      let status = response.status;
      if (response.type === 'opaqueredirect' && !status) status = 302;
      return { outcome: 'response', status, bytes };
    } finally {
      if (timer !== null) clearTimeout(timer);
    }
  };
  const handle = async (task: any): Promise<unknown> => {
    switch (task.kind) {
      case 'http':
        return http(task);
      case 'local-load': {
        let loaded: { model: any; processor: any; loadImage?: (image: any) => unknown };
        if (task.loaderUrl) {
          const module = await load(task.loaderUrl);
          const factory = module[task.exportName ?? 'default'];
          if (typeof factory !== 'function') throw new TypeError(`${task.loaderUrl} does not export a loader function ${task.exportName ?? 'default'}`);
          loaded = await factory(task.args ?? null);
          if (loaded === null || typeof loaded !== 'object' || typeof loaded.model?.generate !== 'function') {
            throw new TypeError('the worker loader must return { model, processor } with model.generate()');
          }
        } else {
          const transformers = await loadTransformers(task.transformersUrl);
          const processor = await transformers.AutoProcessor.from_pretrained(task.modelId, task.settings);
          const modelClass = transformers.AutoModelForImageTextToText ?? transformers.AutoModelForVision2Seq;
          const model = await modelClass.from_pretrained(task.modelId, task.settings);
          loaded = { model, processor };
        }
        if (typeof loaded.model.eval === 'function') loaded.model.eval();
        models.set(task.handle, {
          model: loaded.model, processor: loaded.processor,
          loadImage: typeof loaded.loadImage === 'function' ? loaded.loadImage : rawImageLoader(task.transformersUrl ?? null),
        });
        return null;
      }
      case 'local-run': {
        const entry = models.get(task.handle);
        if (!entry) throw new Error('LocalModel worker model is not loaded');
        return engine(entry.model, entry.processor, entry.loadImage, task.input);
      }
      case 'local-dispose':
        models.delete(task.handle);
        return null;
      default:
        throw new TypeError(`unknown blocking task ${String(task.kind)}`);
    }
  };
  port.on('message', (message: { id: number; task: unknown }) => {
    handle(message.task).then(
      (value) => ({ ok: true, value }),
      (error: unknown) => ({
        ok: false,
        error: {
          name: error instanceof Error ? (error.name === 'Error' && error.constructor?.name ? error.constructor.name : error.name) : 'Error',
          message: error instanceof Error ? error.message : String(error),
        },
      }),
    ).then((reply) => {
      try {
        port.postMessage({ id: message.id, reply });
      } catch (error) {
        port.postMessage({ id: message.id, reply: { ok: false, error: { name: 'TypeError', message: `worker result is not transferable: ${(error as Error).message}` } } });
      }
      notify();
    });
  });
}

function startBridge(): Bridge {
  const threads = builtin<Threads>('node:worker_threads');
  if (threads === null || typeof threads.Worker !== 'function' || typeof threads.receiveMessageOnPort !== 'function'
    || typeof SharedArrayBuffer !== 'function') {
    throw new SynchronousCallUnavailable(
      'Synchronous provider calls need Node.js worker threads (node:worker_threads) to block on the request; '
      + 'this environment cannot block, so use await model.acomplete(...) or await operation.acall(...)',
    );
  }
  const { port1, port2 } = new threads.MessageChannel();
  const signal = new Int32Array(new SharedArrayBuffer(4));
  // Transpilers may reference a ``__name`` helper inside function bodies.
  const source = `"use strict";\nvar __name = (target) => target;\n(${workerMain.toString()})(require('node:worker_threads'), ${JSON.stringify(runLocalEngine.toString())}, (specifier) => import(specifier));\n`;
  const worker = new threads.Worker(source, { eval: true, workerData: { port: port2, signal }, transferList: [port2] });
  worker.unref();
  (port1 as { unref?: () => void }).unref?.();
  worker.on('error', () => {
    if (bridge?.worker === worker) bridge = null;
  });
  worker.on('exit', () => {
    if (bridge?.worker === worker) bridge = null;
  });
  return { threads, worker, port: port1, signal, next: 1 };
}

function reset(current: Bridge): void {
  if (bridge === current) bridge = null;
  void current.worker.terminate().catch(() => undefined);
}

/** Thrown when a blocking call exceeds its ``waitSeconds`` (the worker is discarded). */
export class BlockingTimeout extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BlockingTimeout';
  }
}

/**
 * Run ``task`` in the provider worker and block until it answers. A worker
 * failure is rethrown as the matching error class (``TypeError``,
 * ``ValueError``, ``MissingDependencyError``, or an ``Error`` with the
 * worker's error name).
 */
export function blockingCall<T>(task: Record<string, unknown>, options: { waitSeconds?: number | null; transfer?: readonly unknown[] } = {}): T {
  const current = bridge ?? (bridge = startBridge());
  const id = current.next;
  current.next += 1;
  Atomics.store(current.signal, 0, 0);
  current.port.postMessage({ id, task }, (options.transfer ?? []) as never);
  const deadline = options.waitSeconds === null || options.waitSeconds === undefined ? Infinity : Date.now() + options.waitSeconds * 1000;
  for (;;) {
    for (;;) {
      const received = current.threads.receiveMessageOnPort(current.port) as { message: { id: number; reply: { ok: boolean; value?: unknown; error?: WorkerFailure } } } | undefined;
      if (received === undefined) break;
      if (received.message.id !== id) continue; // a stale reply from an abandoned call
      const { reply } = received.message;
      if (reply.ok) return reply.value as T;
      throw rebuildError(reply.error!);
    }
    if (Atomics.load(current.signal, 0) === 2) {
      reset(current);
      throw new Error('the provider worker thread exited unexpectedly');
    }
    // A notification for a reply that is not ours: clear it and look again.
    if (Atomics.compareExchange(current.signal, 0, 1, 0) === 1) continue;
    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      reset(current);
      throw new BlockingTimeout('blocking call timed out');
    }
    Atomics.wait(current.signal, 0, 0, Number.isFinite(remaining) ? remaining : undefined);
  }
}

/**
 * Send ``task`` to the worker without waiting (its reply is discarded as
 * stale). Does nothing when no worker is running.
 */
export function postWithoutWaiting(task: Record<string, unknown>): void {
  const current = bridge;
  if (current === null) return;
  const id = current.next;
  current.next += 1;
  try {
    current.port.postMessage({ id, task });
  } catch {
    // The worker is gone; nothing to release.
  }
}

function rebuildError(failure: WorkerFailure): Error {
  switch (failure.name) {
    case 'TypeError': return new TypeError(failure.message);
    case 'RangeError': return new RangeError(failure.message);
    case 'ValueError': return new ValueError(failure.message);
    case 'MissingDependencyError': return new MissingDependencyError(failure.message);
    default: {
      const error = new Error(failure.message);
      error.name = failure.name || 'Error';
      return error;
    }
  }
}

/** Resolve a module specifier to a URL the worker can import (``null`` when absent). */
export function resolveModuleUrl(specifier: string, parentUrl: string): string | null {
  const meta = import.meta as { resolve?: (specifier: string, parent?: string) => string };
  try {
    if (typeof meta.resolve === 'function') return meta.resolve(specifier);
  } catch {
    // Fall through to CommonJS resolution.
  }
  const moduleApi = builtin<typeof import('node:module')>('node:module');
  const url = builtin<typeof import('node:url')>('node:url');
  if (moduleApi === null || url === null) return null;
  try {
    const resolved = moduleApi.createRequire(parentUrl).resolve(specifier);
    return url.pathToFileURL(resolved).href;
  } catch {
    return null;
  }
}

/** A file path or URL string as an importable URL. */
export function moduleUrl(specifier: string | URL): string {
  if (specifier instanceof URL) return specifier.href;
  if (/^[A-Za-z][A-Za-z0-9+.-]*:/.test(specifier) && !/^[A-Za-z]:[\\/]/.test(specifier)) return specifier;
  const url = builtin<typeof import('node:url')>('node:url');
  const path = builtin<typeof import('node:path')>('node:path');
  if (url === null || path === null) throw new SynchronousCallUnavailable('worker loader modules need Node.js file URLs');
  return url.pathToFileURL(path.resolve(specifier)).href;
}
