/**
 * Operation configuration identities and fingerprints (Python
 * ``tensorcode/_internal/training/persistence.py``: ``configuration``,
 * ``fingerprint``, ``bindings``, ``validate_bindings``). FOUNDATION-OWNED:
 * experience files and checkpoints written by either implementation validate
 * against the same fingerprints.
 */
import { Module, publicAttributes } from '../nn/module.js';
import { ValueError } from '../errors.js';
import { qualifiedName } from './identity.js';
import { canonicalJson, isPlainObject, sha256Hex, type JsonObject, type JsonValue } from './json.js';
import { stateTopology } from './vec/configuration.js';
import type { OperationLike } from '../ops/base.js';

/** Optional hook: an explicit persisted identity (Python ``_operation_identity``). */
export interface IdentifiedOperation {
  operationIdentity(): string;
}

function configurationValue(value: unknown): JsonValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('Operation configuration requires finite JSON numbers');
    return value;
  }
  if (Array.isArray(value)) return value.map(configurationValue);
  if (isPlainObject(value)) {
    const result: JsonObject = {};
    for (const [key, item] of Object.entries(value)) result[key] = configurationValue(item);
    return result;
  }
  throw new TypeError('Operation configuration requires JSON data; custom callbacks need explicit configuration() metadata');
}

/** ``{type, config, state_topology, replayable}`` for an operation binding. */
export function operationConfiguration(operation: OperationLike): JsonObject {
  const hook = (operation as Partial<IdentifiedOperation>).operationIdentity;
  const identity = typeof hook === 'function' ? hook.call(operation) : qualifiedName(operation);
  if (typeof identity !== 'string' || !identity.trim()) throw new ValueError('Operation identity must be a nonempty string');
  let config: JsonValue;
  if (typeof operation.configuration === 'function') {
    config = configurationValue(operation.configuration());
  } else {
    const entries: JsonObject = {};
    const modules: [string, object][] = operation instanceof Module ? operation.namedModules() : [['', operation]];
    for (const [name, module] of modules) {
      const attributes = module instanceof Module ? module.configurationAttributes() : publicAttributes(module);
      const values: JsonObject = {};
      for (const [key, value] of Object.entries(attributes)) values[key] = configurationValue(value);
      entries[name] = { type: qualifiedName(module), attributes: values };
    }
    config = entries;
  }
  return {
    type: identity,
    config,
    state_topology: operation instanceof Module ? stateTopology(operation) : null,
    replayable: Boolean(operation.replayable),
  };
}

/** SHA-256 of ``json.dumps(config, sort_keys=True, separators=(',', ':'))``. */
export function fingerprint(config: JsonValue): string {
  return sha256Hex(canonicalJson(config));
}

export interface BindingRecord {
  configuration: JsonObject;
  fingerprint: string;
}

/** Configuration and fingerprint of every named operation binding. */
export function bindingRecords(operations: Record<string, OperationLike>): Record<string, BindingRecord> {
  if (!isPlainObject(operations) || !Object.keys(operations).length || Object.keys(operations).some((key) => !key)) {
    throw new ValueError('operations must provide nonempty named bindings');
  }
  const records: Record<string, BindingRecord> = {};
  for (const [name, operation] of Object.entries(operations)) {
    const configuration = operationConfiguration(operation);
    records[name] = { configuration, fingerprint: fingerprint(configuration) };
  }
  return records;
}

/** Reject corrupt records and bindings whose configuration differs from the artifact. */
export function validateBindings(saved: unknown, operations: Record<string, OperationLike>): void {
  if (!isPlainObject(saved) || !isPlainObject(operations)) throw new ValueError('Malformed operation bindings');
  for (const [name, record] of Object.entries(saved)) {
    if (!(name in operations)) throw new ValueError(`Missing operation binding: ${name}`);
    if (!isPlainObject(record) || Object.keys(record).length !== 2 || !('configuration' in record) || !('fingerprint' in record)
      || record.fingerprint !== fingerprint(record.configuration as JsonValue)) {
      throw new ValueError('Corrupt operation configuration fingerprint');
    }
    if (record.fingerprint !== fingerprint(operationConfiguration(operations[name]!))) {
      throw new ValueError(`Incompatible operation configuration: ${name}`);
    }
  }
}
