/**
 * EXTENSION POINT — owned by the training builder (module "training").
 *
 * Versioned experience JSON artifacts (Python
 * ``tensorcode/_internal/training/persistence.py``). The foundation fixes these
 * signatures because ``Trace.save`` dynamically imports ``saveExperience``;
 * the training builder replaces the bodies and may add exports, but must keep
 * these two signatures.
 */
import { NotImplementedError } from '../../errors.js';
import type { OperationLike } from '../../ops/base.js';
import type { RecordClass } from '../records.js';
import type { Trace } from '../tracing.js';

export interface ExperienceOptions {
  /** Stable names bound to already-constructed operation instances. */
  operations: Record<string, OperationLike>;
  /** Explicit allowlist of trusted record classes keyed by codec name. */
  codecs?: Record<string, RecordClass> | null;
}

/** Atomically write ``trace`` as a ``tensorcode.experience`` v1 JSON artifact. */
export async function saveExperience(trace: Trace, path: string, options: ExperienceOptions): Promise<void> {
  void trace; void path; void options;
  throw new NotImplementedError('experience persistence is provided by the training module');
}

/** Load an experience artifact bound to the supplied operations. */
export async function loadExperience(path: string, options: ExperienceOptions): Promise<Trace> {
  void path; void options;
  throw new NotImplementedError('experience persistence is provided by the training module');
}
