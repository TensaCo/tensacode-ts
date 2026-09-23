/**
 * Immutable value records: the TypeScript counterpart of frozen dataclasses.
 *
 * A record class exposes ``static recordFields`` (persisted field names, which
 * keep the Python snake_case spelling), ``static fromRecord(fields)`` and an
 * instance ``toRecord()`` returning those fields. Tracing walks record fields
 * for dependencies and experience codecs persist them through an explicit
 * allowlist; nothing is ever reconstructed from an artifact-named class.
 */

export type RecordFields = Record<string, unknown>;

export interface RecordClass<T = unknown> {
  readonly recordFields: readonly string[];
  fromRecord(fields: RecordFields): T;
  readonly name: string;
}

export interface RecordInstance {
  toRecord(): RecordFields;
}

export function isRecordClass(value: unknown): value is RecordClass {
  if (typeof value !== 'function') return false;
  const candidate = value as unknown as Partial<RecordClass>;
  return Array.isArray(candidate.recordFields) && typeof candidate.fromRecord === 'function';
}

/** The record class of ``value``, or ``null`` when it is not a record instance. */
export function recordClassOf(value: unknown): RecordClass | null {
  if (value === null || typeof value !== 'object') return null;
  const constructor = (value as { constructor?: unknown }).constructor;
  if (!isRecordClass(constructor)) return null;
  if (typeof (value as Partial<RecordInstance>).toRecord !== 'function') return null;
  return constructor;
}

export function recordFields(value: unknown): RecordFields {
  const fields = (value as RecordInstance).toRecord();
  const cls = recordClassOf(value)!;
  const keys = Object.keys(fields);
  if (keys.length !== cls.recordFields.length || !cls.recordFields.every((field) => keys.includes(field))) {
    throw new TypeError(`${cls.name}.toRecord() must return exactly its recordFields`);
  }
  return fields;
}

/** Freeze a record instance and its plain nested data. */
export function freezeRecord<T extends object>(value: T): T {
  return Object.freeze(value);
}
