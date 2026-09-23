/**
 * JSON-safe configuration identities that deliberately exclude learned values
 * (Python ``tensorcode/ops/vec/_configuration.py``). FOUNDATION-OWNED.
 */
import { Module } from '../../nn/module.js';
import { torchDTypeName } from '../../nn/dtype.js';
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { qualifiedName } from '../identity.js';
import { isPlainObject, type JsonObject, type JsonValue } from '../json.js';
import type { Space } from '../../ops/vec/latent.js';


export { qualifiedName } from '../identity.js';

/** Python ``_json_value``: finite JSON with sorted mapping keys; tuples become lists. */
export function jsonConfigValue(value: unknown, path: string): JsonValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new ValueError(`${path} is not JSON-safe configuration metadata`);
    return value;
  }
  if (Array.isArray(value)) return value.map((item) => jsonConfigValue(item, `${path}[]`));
  if (isPlainObject(value)) {
    const result: JsonObject = {};
    for (const key of Object.keys(value).sort()) result[key] = jsonConfigValue(value[key], `${path}.${key}`);
    return result;
  }
  throw new ValueError(`${path} is not JSON-safe configuration metadata; provide an explicit configuration() that excludes learned tensor values`);
}

/** The value's explicit ``configuration()`` as JSON, or ``null`` when it has none. */
export function explicitConfiguration(value: unknown): JsonValue | null {
  const method = (value as { configuration?: unknown } | null)?.configuration;
  if (typeof method !== 'function') return null;
  return jsonConfigValue((method as () => unknown).call(value), `${qualifiedName(value)}.configuration`);
}

/**
 * Identity of a supplied callback. JavaScript cannot distinguish closures from
 * module-level functions, so every callback must carry explicit
 * ``configuration()`` metadata to be persisted.
 */
export function callableIdentity(value: unknown): JsonObject | null {
  if (value === null || value === undefined) return null;
  const explicit = explicitConfiguration(value);
  if (explicit !== null) return { callable: qualifiedName(value), configuration: explicit };
  throw new ValueError('Callbacks require explicit configuration() metadata for persistence');
}

export function spaceConfiguration(space: Space | null | undefined): JsonObject | null {
  return space ? (space.configuration() as unknown as JsonObject) : null;
}

/** Describe module architecture and tensor schemas, never tensor values. */
export function moduleConfiguration(module: Module): JsonObject {
  const modules: JsonObject[] = [];
  for (const [path, child] of module.namedModules()) {
    const explicit = explicitConfiguration(child);
    const entry: JsonObject = { path, type: qualifiedName(child), configuration: explicit };
    if (explicit === null) entry.attributes = jsonConfigValue(child.configurationAttributes(), `${qualifiedName(child)}`);
    modules.push(entry);
  }
  return {
    type: qualifiedName(module),
    configuration: explicitConfiguration(module),
    modules,
    parameters: module.namedParameters().map(([name, parameter]) => ({
      name, shape: [...parameter.shape], dtype: torchDTypeName(parameter.dtype), requires_grad: parameter.requiresGrad,
    })),
    buffers: module.namedBuffers().map(([name, buffer]) => ({ name, shape: [...buffer.shape], dtype: torchDTypeName(buffer.dtype) })),
  };
}

/** ``{name: {shape, dtype}}`` of a module's state (fingerprint ``state_topology``). */
export function stateTopology(module: Module): JsonObject {
  const result: JsonObject = {};
  for (const [name, value] of module.stateDict()) {
    if (value instanceof Tensor) result[name] = { shape: [...value.shape], dtype: torchDTypeName(value.dtype) };
  }
  return result;
}
