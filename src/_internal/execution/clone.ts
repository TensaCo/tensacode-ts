/**
 * Deep copies of caller values (Python ``copy.deepcopy``) for receipts and plan
 * execution snapshots. Plain data uses ``structuredClone``; frozen arrays
 * (Python tuples) and immutable records keep their shape.
 */
import { ValueError } from '../../errors.js';
import { isPythonNumber, mapKeyKind, orderedEntries, setKeyOrder, setMapKeyKind, transferPythonNumberKind } from '../json.js';

export function deepClone<T>(value: T): T {
  return clone(value) as T;
}

function clone(value: unknown): unknown {
  if (value === null || typeof value !== 'object') {
    if (typeof value === 'function') throw new ValueError('values must support deep copy; functions cannot be copied');
    return value;
  }
  // Copies keep Python number kinds (``float(1)``, values read as ``1.0``) and dict order.
  if (isPythonNumber(value)) return value;
  if (Array.isArray(value)) {
    const items = value.map(clone);
    value.forEach((item, index) => transferPythonNumberKind(items, index, value, index, item));
    return Object.isFrozen(value) ? Object.freeze(items) : items;
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype === Object.prototype || prototype === null) {
    const result: Record<string, unknown> = prototype === null ? Object.create(null) : {};
    const entries = orderedEntries(value as Record<string, unknown>);
    for (const [key, item] of entries) {
      Object.defineProperty(result, key, { value: clone(item), enumerable: true, writable: true, configurable: true });
      transferPythonNumberKind(result, key, value, key, item);
    }
    setKeyOrder(result, entries.map(([key]) => key));
    return result;
  }
  if (value instanceof Map) {
    const result = new Map([...value].map(([key, item]) => [key, clone(item)]));
    for (const [key, item] of value) {
      transferPythonNumberKind(result, key, value, key, item);
      if (typeof key === 'number') setMapKeyKind(result, key, mapKeyKind(value, key));
    }
    return result;
  }
  const cls = (value as { constructor?: { fromRecord?: unknown; recordFields?: unknown } }).constructor;
  if (cls && typeof cls.fromRecord === 'function' && Array.isArray(cls.recordFields)
    && typeof (value as { toRecord?: unknown }).toRecord === 'function') {
    const fields = (value as { toRecord(): Record<string, unknown> }).toRecord();
    const copied: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(fields)) copied[key] = clone(item);
    const result = (cls.fromRecord as (fields: Record<string, unknown>) => unknown)(copied);
    if (result !== null && typeof result === 'object') {
      for (const [key, item] of Object.entries(fields)) if (typeof item === 'number') transferPythonNumberKind(result, key, value, key, item);
    }
    return result;
  }
  try {
    return structuredClone(value);
  } catch (error) {
    throw new ValueError('values must support deep copy (structured clone)', { cause: error });
  }
}
