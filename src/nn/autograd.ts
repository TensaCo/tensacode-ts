/**
 * Reverse-mode automatic differentiation.
 *
 * Operations record a {@link GradNode} on their result when gradient recording
 * is enabled and at least one floating input requires gradients. ``backward``
 * visits nodes in reverse topological order and accumulates gradients into leaf
 * tensors' ``grad``. Backward closures run with recording disabled, so
 * higher-order gradients are not supported.
 */
import type { Tensor } from './tensor.js';

export interface GradNode {
  readonly name: string;
  readonly inputs: readonly (Tensor | null)[];
  /** Return one gradient per input (``null`` when an input receives none). */
  backward(grad: Tensor): readonly (Tensor | null | undefined)[];
}

let gradEnabled = true;

export function isGradEnabled(): boolean {
  return gradEnabled;
}

export function setGradEnabled(enabled: boolean): void {
  gradEnabled = enabled;
}

/**
 * Run ``fn`` without recording gradients (like ``torch.no_grad``).
 *
 * The mode is process-global and restored when ``fn`` returns or throws. For an
 * async ``fn`` the mode is restored only after its promise settles, so do not
 * interleave concurrent gradient-recording work with it.
 */
export function noGrad<T>(fn: () => T): T {
  return withGradMode(false, fn);
}

/** Run ``fn`` with gradient recording enabled. */
export function enableGrad<T>(fn: () => T): T {
  return withGradMode(true, fn);
}

function withGradMode<T>(mode: boolean, fn: () => T): T {
  const previous = gradEnabled;
  gradEnabled = mode;
  let result: T;
  try {
    result = fn();
  } catch (error) {
    gradEnabled = previous;
    throw error;
  }
  if (result && typeof (result as { then?: unknown }).then === 'function') {
    return (result as unknown as Promise<unknown>).finally(() => {
      gradEnabled = previous;
    }) as unknown as T;
  }
  gradEnabled = previous;
  return result;
}
