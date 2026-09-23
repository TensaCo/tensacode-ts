/**
 * Versioned, non-executable model artifacts and explicit Hugging Face transport
 * (Python ``tensorcode/_internal/pretrained.py``).
 *
 * An artifact directory contains ``tensorcode_config.json`` (format, version,
 * the known concrete class identity and its JSON configuration),
 * ``model.safetensors`` (weights, ``safetensors.torch.save_model`` layout with
 * tied tensors stored once) and a ``README.md`` model card. Loading rejects
 * incompatible artifacts; it never imports or executes artifact-selected code.
 * Artifacts written by the Python and TypeScript implementations are
 * interchangeable when both reconstruct the same architecture and state names.
 */
import { cp, mkdir, mkdtemp, readFile, rename, rm, stat, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { Module } from '../nn/module.js';
import { Parameter, Tensor } from '../nn/tensor.js';
import type { DType } from '../nn/dtype.js';
import {
  deserializeSafetensors, loadModelFromBytes, serializeModel, type SafetensorsContents,
} from '../nn/safetensors.js';
import { ValueError } from '../errors.js';
import { canonicalJson, comparePythonStrings, parseJsonStrict, pythonJsonDumps, validatedJson, type JsonObject } from './json.js';
import { qualifiedName } from './identity.js';
import { resolveArtifactDirectory, type HubOptions } from './hub.js';
import { isSymlink, pathExists } from './files.js';
import { isOperation, type CallOptions, type Context, type OperationLike } from '../ops/base.js';

export const ARTIFACT_MANIFEST = 'tensorcode_config.json';
export const ARTIFACT_WEIGHTS = 'model.safetensors';
export const ARTIFACT_CARD = 'README.md';

export interface FromPretrainedOptions extends Omit<HubOptions, 'allowPatterns'> {
  /** Only ``'cpu'`` is supported by the TypeScript numerical core. */
  device?: string;
}

export interface PushToHubOptions {
  private?: boolean;
  revision?: string | null;
  token?: string | null;
  commitMessage?: string;
  /** Replaces the generated model card. */
  modelCard?: string | null;
}

/** Validate an owned model configuration: a finite JSON object (deep copy). */
export function validatedModelConfig(config: unknown): JsonObject {
  if (config === null || typeof config !== 'object' || Array.isArray(config)) {
    throw new ValueError('model config must be a JSON object');
  }
  try {
    return validatedJson<JsonObject>(config, 'model config');
  } catch (error) {
    throw new ValueError('model config must contain finite JSON values of JSON types', { cause: error });
  }
}

/** CPython ``repr`` of a string (quote choice and escapes of ``str.__repr__``). */
export function pythonStringRepr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  let body = '';
  for (const char of value) {
    const code = char.codePointAt(0)!;
    if (char === '\\') body += '\\\\';
    else if (char === quote) body += `\\${quote}`;
    else if (char === '\n') body += '\\n';
    else if (char === '\r') body += '\\r';
    else if (char === '\t') body += '\\t';
    else if (code < 0x20 || code === 0x7f) body += `\\x${code.toString(16).padStart(2, '0')}`;
    else body += char;
  }
  return quote + body + quote;
}

/** The Python class name (``type(self).__name__``) of a class with a ``qualifiedName``. */
export function pythonClassName(cls: { name: string; qualifiedName?: string }): string {
  return cls.qualifiedName?.split('.').pop() ?? cls.name;
}

/**
 * Reject obsolete or misspelled configuration fields, naming the valid ones
 * (Python ``reject_unknown_fields``). Tools own complete architectures;
 * silently ignoring an unknown field would let a typo or an obsolete artifact
 * construct a different model than the configuration describes.
 */
export function rejectUnknownToolFields(config: Readonly<Record<string, unknown>>, valid: Iterable<string>, owner: string): void {
  const allowed = new Set(valid);
  const unknown = Object.keys(config).filter((key) => !allowed.has(key)).sort(comparePythonStrings);
  if (!unknown.length) return;
  const list = (items: readonly string[]): string => `[${items.map(pythonStringRepr).join(', ')}]`;
  throw new ValueError(`Unknown ${owner} configuration fields: ${list(unknown)}; valid fields: ${list([...allowed].sort(comparePythonStrings))}`);
}

export function requireCpu(device: string | undefined): void {
  if (device !== undefined && device !== 'cpu') {
    throw new ValueError(`device ${JSON.stringify(device)} is unsupported; the TypeScript core computes on the CPU`);
  }
}

// ---------------------------------------------------------------------------
// Artifact writing.
// ---------------------------------------------------------------------------

export interface ArtifactWriter {
  /** Module whose state is serialized into ``model.safetensors``. */
  stateModule: Module;
  manifest: { format: string; version: number; identityKey: 'tool' | 'operation'; identity: string; config: JsonObject };
  /** Save extra assets into the staging directory. */
  saveAssets?(directory: string): Promise<void>;
  modelCard(): string;
}

/**
 * Stage a complete artifact then replace ``directory`` with rollback.
 * Existing unrelated files are preserved; concurrent writers are unsupported.
 */
export async function writePretrainedArtifact(writer: ArtifactWriter, directory: string): Promise<string> {
  const target = resolve(directory);
  if (await isSymlink(target)) throw new ValueError('model destination must be a directory, not a symlink');
  if (await pathExists(target) && !(await stat(target)).isDirectory()) {
    throw new ValueError('model destination must be a directory, not a symlink');
  }
  await mkdir(dirname(target), { recursive: true });
  const stage = await mkdtemp(join(dirname(target), `.${basename(target)}.stage-`));
  const backup = join(dirname(target), `.${basename(target)}.backup-${randomHex()}`);
  let moved = false;
  try {
    if (await pathExists(target)) await cp(target, stage, { recursive: true, verbatimSymlinks: true, force: true });
    for (const name of [ARTIFACT_WEIGHTS, ARTIFACT_MANIFEST]) {
      if (await isSymlink(join(stage, name))) await unlink(join(stage, name));
    }
    if (writer.saveAssets) await writer.saveAssets(stage);
    const card = join(stage, ARTIFACT_CARD);
    if (!(await pathExists(card))) await writeFile(card, writer.modelCard(), 'utf8');
    const { format, version, identityKey, identity, config } = writer.manifest;
    const manifest = { format, version, [identityKey]: identity, config };
    await writeFile(join(stage, ARTIFACT_MANIFEST), `${pythonJsonDumps(manifest, { indent: 2, sortKeys: true, allowNan: false })}\n`, 'utf8');
    await writeFile(join(stage, ARTIFACT_WEIGHTS), serializeModel(writer.stateModule));
    if (await pathExists(target)) {
      await rename(target, backup);
      moved = true;
    }
    try {
      await rename(stage, target);
    } catch (error) {
      if (moved) {
        await rename(backup, target);
        moved = false;
      }
      throw error;
    }
    if (moved) await rm(backup, { recursive: true, force: true });
  } finally {
    if (await pathExists(stage)) await rm(stage, { recursive: true, force: true });
  }
  return target;
}

// ---------------------------------------------------------------------------
// Artifact reading.
// ---------------------------------------------------------------------------

export interface ArtifactReader<T> {
  format: string;
  version: number;
  identityKey: 'tool' | 'operation';
  identity: string;
  /** Bind saved assets before construction (may add private construction inputs). */
  loadConfig?(config: JsonObject, directory: string): Promise<JsonObject> | JsonObject;
  construct(config: JsonObject): T;
  configuration(model: T): JsonObject;
  stateModule(model: T): Module;
}

export interface LoadedArtifact<T> {
  model: T;
  directory: string;
  remote: boolean;
}

/**
 * Load a known class from a local directory or a pinned Hub snapshot.
 * The saved configuration must reconstruct exactly (canonical JSON), so
 * architecture changes cannot silently reinterpret old weights.
 */
export async function readPretrainedArtifact<T>(
  reader: ArtifactReader<T>, source: string, options: FromPretrainedOptions = {},
): Promise<LoadedArtifact<T>> {
  requireCpu(options.device);
  const { device: _device, ...hub } = options;
  void _device;
  const { path, remote } = await resolveArtifactDirectory(source, hub);
  let manifest: unknown;
  try {
    manifest = parseJsonStrict(await readFile(join(path, ARTIFACT_MANIFEST), 'utf8'));
  } catch (error) {
    throw new ValueError(`invalid TensorCode model manifest in ${path}`, { cause: error });
  }
  if (manifest === null || typeof manifest !== 'object' || Array.isArray(manifest)) {
    throw new ValueError('model manifest must be an object');
  }
  const record = manifest as Record<string, unknown>;
  const expectations: [string, unknown][] = [
    ['format', reader.format], ['version', reader.version], [reader.identityKey, reader.identity],
  ];
  for (const [name, expected] of expectations) {
    const actual = record[name];
    if (actual !== expected || (typeof expected === 'number' && !Number.isInteger(actual))) {
      throw new ValueError(`incompatible model ${name}: expected ${JSON.stringify(expected)}`);
    }
  }
  const config = validatedModelConfig(record.config);
  const bound = reader.loadConfig ? await reader.loadConfig(validatedJson<JsonObject>(config), path) : validatedJson<JsonObject>(config);
  const model = reader.construct(bound);
  if (canonicalJson(reader.configuration(model)) !== canonicalJson(config)) {
    throw new ValueError('saved model configuration differs from reconstructed architecture; '
      + 'the artifact is incompatible with this implementation');
  }
  const bytes = await readFile(join(path, ARTIFACT_WEIGHTS));
  const view = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const module = reader.stateModule(model);
  restoreArtifactDtypes(module, deserializeSafetensors(view));
  loadModelFromBytes(module, view, { strict: true });
  module.eval();
  return { model, directory: path, remote };
}

/**
 * Convert registered tensors to the dtypes stored in the artifact before
 * copy-loading, preserving tied parameters (Python ``_restore_artifact_dtypes``).
 */
export function restoreArtifactDtypes(module: Module, contents: SafetensorsContents): void {
  const aliases = contents.metadata ?? {};
  const dtypes = new Map<string, DType>();
  for (const [name, value] of contents.tensors) dtypes.set(name, value.dtype);
  const converted = new Map<object, Tensor>();
  for (const [name, value] of module.stateDict()) {
    const dtype = dtypes.get(name) ?? (aliases[name] !== undefined ? dtypes.get(aliases[name]!) : undefined);
    if (dtype === undefined || value.dtype === dtype) continue;
    let replacement = converted.get(value._storage);
    if (!replacement) {
      replacement = value instanceof Parameter ? new Parameter(value.detach().to(dtype), value.requiresGrad) : value.detach().to(dtype);
      converted.set(value._storage, replacement);
    }
    if (replacement.dtype !== dtype) throw new ValueError('shared model storage has incompatible artifact dtypes');
    if (value instanceof Parameter) module.setParameterAt(name, replacement as Parameter);
    else module.setBufferAt(name, replacement);
  }
}

/** Public TypeScript entry point that exports the class with this Python identity, if any. */
function typescriptImportPath(identity: string): string | null {
  if (identity.startsWith('tensorcode.tools.')) return 'tensorcode/tools';
  if (identity.startsWith('tensorcode.ops.vec.') || identity.startsWith('tensorcode._internal.vec.')) return 'tensorcode/ops/vec';
  if (identity.startsWith('tensorcode.ops.text.') || identity.startsWith('tensorcode._internal.text.')) return 'tensorcode/ops/text';
  return null;
}

export function defaultModelCard(identity: string, className: string, importPath: string | null = typescriptImportPath(identity)): string {
  const pythonModule = identity.slice(0, identity.lastIndexOf('.'));
  return '---\nlibrary_name: tensorcode\ntags:\n- tensorcode\n---\n\n'
    + `# ${className}\n\n`
    + `This artifact stores the \`${identity}\` architecture configuration and model weights.\n\n`
    + '## Loading\n\n'
    + 'Install the compatible TensorCode library and model dependencies, '
    + 'then load this local directory or its Hugging Face repository ID:\n\n'
    + `\`\`\`python\nfrom ${pythonModule} import ${className}\n\n`
    + `model = ${className}.from_pretrained("./model")\n\`\`\`\n\n`
    + '```ts\n'
    + (importPath === null ? `// ${className} is the class that saved this artifact.\n` : `import { ${className} } from '${importPath}';\n\n`)
    + `const model = await ${className}.fromPretrained('./model');\n\`\`\`\n\n`
    + '## Training and evaluation\n\n'
    + 'This generated card does not establish training provenance, task '
    + 'competence, evaluation results, or license. The publisher should '
    + 'document these before distributing a trained model. Session history '
    + 'and optimizer state are not included in the model artifact.\n';
}

/** Write an artifact into a temporary folder and publish it with ``uploadFolder``. */
export async function publishArtifact(
  save: (directory: string) => Promise<unknown>, repoId: string, options: PushToHubOptions = {},
): Promise<unknown> {
  if (options.modelCard !== undefined && options.modelCard !== null && typeof options.modelCard !== 'string') {
    throw new TypeError('modelCard must be a string or null');
  }
  const { uploadFolder } = await import('./hubUpload.js');
  const directory = await mkdtemp(join(tmpdir(), 'tensorcode-publish-'));
  try {
    const folder = join(directory, 'model');
    await save(folder);
    if (typeof options.modelCard === 'string') await writeFile(join(folder, ARTIFACT_CARD), options.modelCard, 'utf8');
    return await uploadFolder({
      repoId, folderPath: folder, private: options.private ?? false, revision: options.revision ?? null,
      token: options.token ?? null, commitMessage: options.commitMessage ?? 'Upload TensorCode model',
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

function randomHex(): string {
  return Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
}

// ---------------------------------------------------------------------------
// Base class.
// ---------------------------------------------------------------------------

/** Static side of a concrete {@link PretrainedModule} subclass. */
export interface PretrainedClass<T extends PretrainedModule> {
  new (config: JsonObject): T;
  readonly artifactFormat: string;
  readonly artifactVersion: number;
  loadPretrainedConfig(config: JsonObject, directory: string): Promise<JsonObject> | JsonObject;
}

/**
 * Base for owned trainable tools and model-backed operations (Python
 * ``PretrainedTool``). Construction never performs Hub I/O and initializes all
 * parameters from a JSON configuration. ``call`` runs ``forward`` directly (like
 * ``nn.Module.__call__``); it is not a trace boundary — {@link LatentOperation}
 * subclasses opt into tracing.
 */
export abstract class PretrainedModule<I = unknown, O = unknown> extends Module {
  static readonly artifactFormat: string = 'tensorcode.pretrained';
  static readonly artifactVersion: number = 1;
  /** Stateful tool calls are not replayable; owned objective operations opt in. */
  get replayable(): boolean {
    return false;
  }

  config: JsonObject;

  constructor(config: unknown) {
    super();
    this.config = validatedModelConfig(config);
  }

  /** Persisted identity of this concrete class (Python qualified name). */
  static toolIdentity(): string {
    return qualifiedName(this);
  }

  /** Independent JSON configuration, without learned weights. */
  configuration(): JsonObject {
    return validatedModelConfig(this.config);
  }

  /** Stable paths of registered operations; compositions may extend this. */
  operationBindings(): Record<string, OperationLike> {
    const result: Record<string, OperationLike> = {};
    for (const [name, module] of this.namedModules()) {
      if (!name) continue;
      if (isOperation(module) || (module as { replayable?: unknown }).replayable === true) {
        result[name] = module as unknown as OperationLike;
      }
    }
    return result;
  }

  call(value: I, options?: CallOptions): O {
    return this.forward(value, options?.context ?? null);
  }

  abstract forward(value: I, context: Context | null): O;

  /** Subclass hook: save local tokenizer/encoder assets into the staging directory. */
  protected async savePretrainedAssets(directory: string): Promise<void> {
    void directory;
  }

  /** Subclass hook: bind saved assets locally before construction. */
  static loadPretrainedConfig(config: JsonObject, directory: string): Promise<JsonObject> | JsonObject {
    void directory;
    return config;
  }

  defaultModelCard(): string {
    const cls = this.constructor as typeof PretrainedModule;
    return defaultModelCard(cls.toolIdentity(), cls.name);
  }

  /** Stage a complete model directory then atomically replace ``directory``. */
  async savePretrained(directory: string): Promise<string> {
    const cls = this.constructor as typeof PretrainedModule;
    return writePretrainedArtifact({
      stateModule: this,
      manifest: {
        format: cls.artifactFormat, version: cls.artifactVersion, identityKey: 'tool',
        identity: cls.toolIdentity(), config: this.configuration(),
      },
      saveAssets: (stage) => this.savePretrainedAssets(stage),
      modelCard: () => this.defaultModelCard(),
    }, directory);
  }

  /** Load this known class from a local directory or a pinned Hub snapshot. */
  static async fromPretrained<T extends PretrainedModule>(
    this: PretrainedClass<T>, source: string, options: FromPretrainedOptions = {},
  ): Promise<T> {
    const cls = this;
    const { model } = await readPretrainedArtifact<T>({
      format: cls.artifactFormat,
      version: cls.artifactVersion,
      identityKey: 'tool',
      identity: qualifiedName(cls),
      loadConfig: (config, directory) => cls.loadPretrainedConfig(config, directory),
      construct: (config) => new cls(config),
      configuration: (model) => model.configuration(),
      stateModule: (model) => model,
    }, source, options);
    return model;
  }

  /** Explicitly publish model artifacts; never sessions or optimizer state. */
  async pushToHub(repoId: string, options: PushToHubOptions = {}): Promise<unknown> {
    return publishArtifact((directory) => this.savePretrained(directory), repoId, options);
  }
}
