/**
 * Deep copies of caller values (Python ``copy.deepcopy``) for receipts and plan
 * execution snapshots. Plain data uses ``structuredClone``; frozen arrays
 * (Python tuples) and immutable records keep their shape.
 */
import { ValueError } from '../../errors.js';

export function deepClone<T>(value: T): T {
  return clone(value) as T;
}

function clone(value: unknown): unknown {
  if (value === null || typeof value !== 'object') {
    if (typeof value === 'function') throw new ValueError('values must support deep copy; functions cannot be copied');
    return value;
  }
  if (Array.isArray(value)) {
    const items = value.map(clone);
    return Object.isFrozen(value) ? Object.freeze(items) : items;
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype === Object.prototype || prototype === null) {
    const result: Record<string, unknown> = prototype === null ? Object.create(null) : {};
    for (const [key, item] of Object.entries(value)) result[key] = clone(item);
    return result;
  }
  const cls = (value as { constructor?: { fromRecord?: unknown; recordFields?: unknown } }).constructor;
  if (cls && typeof cls.fromRecord === 'function' && Array.isArray(cls.recordFields)
    && typeof (value as { toRecord?: unknown }).toRecord === 'function') {
    const fields = (value as { toRecord(): Record<string, unknown> }).toRecord();
    const copied: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(fields)) copied[key] = clone(item);
    return (cls.fromRecord as (fields: Record<string, unknown>) => unknown)(copied);
  }
  try {
    return structuredClone(value);
  } catch (error) {
    throw new ValueError('values must support deep copy (structured clone)', { cause: error });
  }
}
