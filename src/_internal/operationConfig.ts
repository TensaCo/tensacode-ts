/**
 * Data-only configuration artifacts for parameter-free operations (Python
 * ``tensorcode/_internal/operation_config.py``).
 *
 * Public operations accept JSON configuration, never executable models.
 * {@link validatedConfig} rejects unknown (including obsolete) fields and
 * non-JSON values. Configuration-only artifacts are a single
 * ``tensorcode_config.json`` manifest of format ``tensorcode.operation``.
 */
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ValueError } from '../errors.js';
import { Operation, ModuleOperation } from '../ops/base.js';
import { atomicWriteFile, isSymlink } from './files.js';
import { resolveArtifactDirectory, type HubOptions } from './hub.js';
import { qualifiedName } from './identity.js';
import { deepCopy, isPlainObject, jsonEqual, parseJsonStrict, pythonJsonDumps, validatedJson, type JsonObject } from './json.js';

export const OPERATION_ARTIFACT_FORMAT = 'tensorcode.operation';

/**
 * Validate a JSON operation configuration against the allowed ``keys`` and
 * merge it over ``defaults``. ``null``/``undefined`` means an empty config.
 */
export function validatedConfig(
  config: unknown, keys: Iterable<string>, defaults: JsonObject = {},
): JsonObject {
  const source = config === null || config === undefined ? {} : config;
  if (!isPlainObject(source)) throw new TypeError('Operation config must be a JSON object');
  const allowed = new Set(keys);
  const unknown = Object.keys(source).filter((key) => !allowed.has(key)).sort();
  if (unknown.length) throw new ValueError(`Unknown configuration fields: ${JSON.stringify(unknown)}`);
  let copied: JsonObject;
  try {
    copied = validatedJson<JsonObject>(source, 'configuration');
  } catch (error) {
    throw new ValueError('Configuration must contain finite JSON data', { cause: error });
  }
  if (!jsonEqual(copied, source)) throw new ValueError('Configuration must use JSON types');
  return { ...deepCopy(defaults), ...copied };
}

/** Reject any key outside ``allowed`` (for configs validated field by field). */
export function rejectUnknownFields(config: Record<string, unknown>, allowed: Iterable<string>, message = 'Unknown configuration fields'): void {
  const known = new Set(allowed);
  const unknown = Object.keys(config).filter((key) => !known.has(key)).sort();
  if (unknown.length) throw new ValueError(`${message}: ${JSON.stringify(unknown)}`);
}

export async function saveOperationConfig(identity: string, configuration: JsonObject, directory: string): Promise<string> {
  if (await isSymlink(directory)) throw new ValueError('Artifact directory must not be a symlink');
  const target = join(directory, 'tensorcode_config.json');
  if (await isSymlink(target)) throw new ValueError('Artifact manifest must not be a symlink');
  const payload = { format: OPERATION_ARTIFACT_FORMAT, version: 1, operation: identity, config: configuration };
  await atomicWriteFile(target, `${pythonJsonDumps(payload, { indent: 2, allowNan: false })}\n`);
  return directory;
}

export async function loadOperationConfig(identity: string, source: string, options: HubOptions = {}): Promise<JsonObject> {
  const { path, remote } = await resolveArtifactDirectory(source, { ...options, allowPatterns: ['tensorcode_config.json'] });
  const manifest = join(path, 'tensorcode_config.json');
  if (!remote && ((await isSymlink(path)) || (await isSymlink(manifest)))) {
    throw new ValueError('Local configuration artifacts must not be symlinks');
  }
  const data = parseJsonStrict(await readFile(manifest, 'utf8'));
  if (!isPlainObject(data) || data.format !== OPERATION_ARTIFACT_FORMAT || data.version !== 1 || data.operation !== identity) {
    throw new ValueError('Configuration artifact identity or format mismatch');
  }
  return data.config as JsonObject;
}

/** Static side shared by configuration-only operation classes. */
export interface ConfigOperationClass<T> {
  new (config?: JsonObject | null): T;
}

/** A weightless operation configured by validated JSON (Python ``ConfigOperationMixin``). */
export abstract class ConfigOperation<I = unknown, O = unknown> extends Operation<I, O> {
  readonly config: JsonObject;

  constructor(config: unknown, keys: Iterable<string>, defaults: JsonObject = {}) {
    super();
    this.config = validatedConfig(config, keys, defaults);
  }

  configuration(): JsonObject {
    return deepCopy(this.config);
  }

  savePretrained(directory: string): Promise<string> {
    return saveOperationConfig(qualifiedName(this), this.configuration(), directory);
  }

  static async fromPretrained<T>(this: ConfigOperationClass<T>, source: string, options: HubOptions = {}): Promise<T> {
    return new this(await loadOperationConfig(qualifiedName(this), source, options));
  }
}

/** A configured parameter-free operation that is also an ``nn.Module`` (Python ``ConfigOperationMixin, nn.Module``). */
export abstract class ConfigModuleOperation<I = unknown, O = unknown> extends ModuleOperation<I, O> {
  readonly config: JsonObject;

  constructor(config: unknown, keys: Iterable<string>, defaults: JsonObject = {}) {
    super();
    this.config = validatedConfig(config, keys, defaults);
  }

  configuration(): JsonObject {
    return deepCopy(this.config);
  }

  savePretrained(directory: string): Promise<string> {
    return saveOperationConfig(qualifiedName(this), this.configuration(), directory);
  }

  static async fromPretrained<T>(this: ConfigOperationClass<T>, source: string, options: HubOptions = {}): Promise<T> {
    return new this(await loadOperationConfig(qualifiedName(this), source, options));
  }
}
