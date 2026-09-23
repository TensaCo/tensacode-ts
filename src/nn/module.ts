/**
 * Parameter containers with PyTorch-compatible naming.
 *
 * Children, parameters and buffers are registered explicitly and in order, so
 * ``stateDict()`` keys and ordering match the equivalent ``torch.nn.Module``.
 * Assigning the same {@link Parameter} object under several names ties them;
 * ``namedParameters()`` then reports it once (under its first name) unless
 * ``removeDuplicate`` is false.
 */
import { noGrad } from './autograd.js';
import { castValue, isFloatingDType, roundToDType, type DType } from './dtype.js';
import { formatShape, shapesEqual } from './shape.js';
import { Parameter, Tensor } from './tensor.js';

export type StateDict = Map<string, Tensor>;

export interface LoadStateDictResult {
  missingKeys: string[];
  unexpectedKeys: string[];
}

export interface NamedOptions {
  prefix?: string;
  recurse?: boolean;
  removeDuplicate?: boolean;
}

export class Module {
  /** PyTorch-equivalent qualified class name used in configuration fingerprints. */
  static readonly qualifiedName: string = 'torch.nn.modules.module.Module';
  /** Training mode flag (affects dropout and similar behavior). */
  training = true;
  /** @internal */
  readonly _parameters = new Map<string, Parameter | null>();
  /** @internal */
  readonly _buffers = new Map<string, Tensor | null>();
  /** @internal */
  readonly _nonPersistentBuffers = new Set<string>();
  /** @internal */
  readonly _modules = new Map<string, Module | null>();

  // ------------------------------------------------------------ registration
  /** Register (or replace) a parameter. Pass an existing Parameter to tie weights. */
  registerParameter<T extends Parameter | null>(name: string, value: T): T {
    assertName(name);
    if (value !== null && !(value instanceof Parameter)) throw new TypeError(`${name} must be a Parameter or null`);
    if (this._buffers.has(name) || this._modules.has(name)) throw new Error(`attribute ${name} already registered`);
    this._parameters.set(name, value);
    this.onRegistryChange();
    return value;
  }

  registerBuffer<T extends Tensor | null>(name: string, value: T, persistent = true): T {
    assertName(name);
    if (value !== null && !(value instanceof Tensor)) throw new TypeError(`${name} must be a Tensor or null`);
    if (this._parameters.has(name) || this._modules.has(name)) throw new Error(`attribute ${name} already registered`);
    this._buffers.set(name, value);
    if (persistent) this._nonPersistentBuffers.delete(name);
    else this._nonPersistentBuffers.add(name);
    this.onRegistryChange();
    return value;
  }

  registerModule<T extends Module | null>(name: string, value: T): T {
    assertName(name);
    if (value !== null && !(value instanceof Module)) throw new TypeError(`${name} must be a Module or null`);
    if (this._parameters.has(name) || this._buffers.has(name)) throw new Error(`attribute ${name} already registered`);
    this._modules.set(name, value);
    this.onRegistryChange();
    return value;
  }

  /** Hook for subclasses that cache derived state (for example encodings). */
  protected onRegistryChange(): void {}

  /**
   * JSON attributes describing this module's behavior, equal to the public
   * ``vars(module)`` entries PyTorch reports for the equivalent module (used by
   * operation configuration fingerprints). Excludes tensors and children.
   */
  configurationAttributes(): Record<string, unknown> {
    return publicAttributes(this);
  }

  getParameter(name: string): Parameter | null {
    return this._parameters.get(name) ?? null;
  }

  getBuffer(name: string): Tensor | null {
    return this._buffers.get(name) ?? null;
  }

  getModule(name: string): Module | null {
    return this._modules.get(name) ?? null;
  }

  /** Resolve a dotted path such as ``encoder.layer.0``. */
  getSubmodule(path: string): Module {
    if (!path) return this;
    let current: Module = this;
    for (const part of path.split('.')) {
      const next = current._modules.get(part);
      if (!next) throw new Error(`no submodule ${path}`);
      current = next;
    }
    return current;
  }

  /** Replace the parameter at a dotted path (used to restore tied/untied topologies). */
  setParameterAt(path: string, parameter: Parameter): void {
    const index = path.lastIndexOf('.');
    const owner = index < 0 ? this : this.getSubmodule(path.slice(0, index));
    const name = index < 0 ? path : path.slice(index + 1);
    if (!owner._parameters.has(name)) throw new Error(`no parameter ${path}`);
    const previous = owner._parameters.get(name) ?? null;
    owner._parameters.set(name, parameter);
    rebindFields(owner, previous, parameter);
    owner.onRegistryChange();
  }

  /** Replace the buffer at a dotted path (used to restore artifact dtypes). */
  setBufferAt(path: string, buffer: Tensor): void {
    const index = path.lastIndexOf('.');
    const owner = index < 0 ? this : this.getSubmodule(path.slice(0, index));
    const name = index < 0 ? path : path.slice(index + 1);
    if (!owner._buffers.has(name)) throw new Error(`no buffer ${path}`);
    const previous = owner._buffers.get(name) ?? null;
    owner._buffers.set(name, buffer);
    rebindFields(owner, previous, buffer);
    owner.onRegistryChange();
  }

  // ------------------------------------------------------------ traversal
  *namedChildren(): Generator<[string, Module]> {
    const seen = new Set<Module>();
    for (const [name, module] of this._modules) {
      if (module && !seen.has(module)) {
        seen.add(module);
        yield [name, module];
      }
    }
  }

  children(): Module[] {
    return [...this.namedChildren()].map(([, module]) => module);
  }

  namedModules(options: { prefix?: string; removeDuplicate?: boolean } = {}): [string, Module][] {
    const prefix = options.prefix ?? '';
    const removeDuplicate = options.removeDuplicate ?? true;
    const result: [string, Module][] = [];
    const seen = new Set<Module>();
    const visit = (module: Module, path: string): void => {
      if (removeDuplicate) {
        if (seen.has(module)) return;
        seen.add(module);
      }
      result.push([path, module]);
      for (const [name, child] of module._modules) {
        if (child) visit(child, path ? `${path}.${name}` : name);
      }
    };
    visit(this, prefix);
    return result;
  }

  modules(): Module[] {
    return this.namedModules().map(([, module]) => module);
  }

  namedParameters(options: NamedOptions = {}): [string, Parameter][] {
    const recurse = options.recurse ?? true;
    const removeDuplicate = options.removeDuplicate ?? true;
    const modules = recurse
      ? this.namedModules({ prefix: options.prefix ?? '', removeDuplicate })
      : [[options.prefix ?? '', this] as [string, Module]];
    const result: [string, Parameter][] = [];
    const seen = new Set<Parameter>();
    for (const [path, module] of modules) {
      for (const [name, parameter] of module._parameters) {
        if (!parameter) continue;
        if (removeDuplicate) {
          if (seen.has(parameter)) continue;
          seen.add(parameter);
        }
        result.push([path ? `${path}.${name}` : name, parameter]);
      }
    }
    return result;
  }

  parameters(recurse = true): Parameter[] {
    return this.namedParameters({ recurse }).map(([, parameter]) => parameter);
  }

  namedBuffers(options: NamedOptions = {}): [string, Tensor][] {
    const recurse = options.recurse ?? true;
    const removeDuplicate = options.removeDuplicate ?? true;
    const modules = recurse
      ? this.namedModules({ prefix: options.prefix ?? '', removeDuplicate })
      : [[options.prefix ?? '', this] as [string, Module]];
    const result: [string, Tensor][] = [];
    const seen = new Set<Tensor>();
    for (const [path, module] of modules) {
      for (const [name, buffer] of module._buffers) {
        if (!buffer) continue;
        if (removeDuplicate) {
          if (seen.has(buffer)) continue;
          seen.add(buffer);
        }
        result.push([path ? `${path}.${name}` : name, buffer]);
      }
    }
    return result;
  }

  buffers(recurse = true): Tensor[] {
    return this.namedBuffers({ recurse }).map(([, buffer]) => buffer);
  }

  // ------------------------------------------------------------ state
  /**
   * Parameters and persistent buffers by name, including every name of a tied
   * parameter (like ``torch.nn.Module.state_dict``). Values are the live tensors.
   */
  stateDict(prefix = ''): StateDict {
    const result: StateDict = new Map();
    const visit = (module: Module, path: string): void => {
      for (const [name, parameter] of module._parameters) if (parameter) result.set(path + name, parameter);
      for (const [name, buffer] of module._buffers) {
        if (buffer && !module._nonPersistentBuffers.has(name)) result.set(path + name, buffer);
      }
      for (const [name, child] of module._modules) if (child) visit(child, `${path}${name}.`);
    };
    visit(this, prefix);
    return result;
  }

  /**
   * Copy values into the registered tensors. Shapes must match; values are cast
   * to each destination dtype. With ``strict`` any missing or unexpected key
   * throws before anything is modified.
   */
  loadStateDict(
    state: StateDict | Record<string, Tensor>,
    options: { strict?: boolean } = {},
  ): LoadStateDictResult {
    const strict = options.strict ?? true;
    const entries = state instanceof Map ? state : new Map(Object.entries(state));
    const own = this.stateDict();
    const missingKeys = [...own.keys()].filter((key) => !entries.has(key));
    const unexpectedKeys = [...entries.keys()].filter((key) => !own.has(key));
    if (strict && (missingKeys.length || unexpectedKeys.length)) {
      throw new Error(
        `Error loading state dict: missing keys ${JSON.stringify(missingKeys)}, unexpected keys ${JSON.stringify(unexpectedKeys)}`,
      );
    }
    for (const [key, target] of own) {
      const value = entries.get(key);
      if (!value) continue;
      if (!shapesEqual(value.shape, target.shape)) {
        throw new Error(`size mismatch for ${key}: checkpoint ${formatShape(value.shape)} vs model ${formatShape(target.shape)}`);
      }
    }
    noGrad(() => {
      for (const [key, target] of own) {
        const value = entries.get(key);
        if (!value || value === target) continue;
        const destination = target.data;
        const source = value.data;
        for (let index = 0; index < destination.length; index += 1) {
          destination[index] = castValue(target.dtype, roundToDType(target.dtype, source[index]!));
        }
        target._storage.version += 1;
      }
    });
    // Python ``_load_from_state_dict`` runs per module: every submodule may
    // invalidate caches derived from its tensors.
    for (const module of this.modules()) module.onRegistryChange();
    return { missingKeys, unexpectedKeys };
  }

  // ------------------------------------------------------------ dtype
  /**
   * Cast floating-point parameters and buffers to ``dtype`` (``module.to(dtype)``).
   * Integer and boolean tensors are unchanged. As in PyTorch, parameters keep
   * their identity (their data is replaced in place), so ties with modules
   * outside this one and optimizer references are preserved; buffers are
   * replaced by converted tensors.
   */
  to(dtype: DType): this {
    if (!isFloatingDType(dtype)) throw new TypeError(`Module.to expects a floating-point dtype, got ${dtype}`);
    const converted = new Map<Tensor, Tensor>();
    const convert = (value: Tensor, make: () => Tensor): Tensor => {
      let replacement = converted.get(value);
      if (!replacement) {
        replacement = make();
        converted.set(value, replacement);
      }
      return replacement;
    };
    for (const parameter of this.parameters()) {
      if (!parameter.isFloatingPoint || parameter.dtype === dtype) continue;
      parameter._replaceData(noGrad(() => parameter.to(dtype)).data, dtype);
    }
    for (const module of this.modules()) {
      let changed = false;
      for (const [name, value] of module._buffers) {
        if (value === null || !value.isFloatingPoint || value.dtype === dtype) continue;
        const replacement = convert(value, () => noGrad(() => value.to(dtype)));
        module._buffers.set(name, replacement);
        rebindFields(module, value, replacement);
        changed = true;
      }
      if (changed) module.onRegistryChange();
    }
    return this;
  }

  /** ``module.double()``. */
  double(): this { return this.to('float64'); }
  /** ``module.float()``. */
  float(): this { return this.to('float32'); }
  /** ``module.half()``. */
  half(): this { return this.to('float16'); }
  /** ``module.bfloat16()``. */
  bfloat16(): this { return this.to('bfloat16'); }

  // ------------------------------------------------------------ modes
  /** Set training mode recursively. Subclasses may override to pin modes. */
  train(mode = true): this {
    this.training = mode;
    for (const child of this.children()) child.train(mode);
    return this;
  }

  eval(): this {
    return this.train(false);
  }

  requiresGrad_(requiresGrad = true): this {
    for (const parameter of this.parameters()) parameter.requiresGrad = requiresGrad;
    return this;
  }

  /** Clear gradients of every parameter. */
  zeroGrad(): void {
    for (const parameter of this.parameters()) parameter.grad = null;
  }

  apply(fn: (module: Module) => void): this {
    for (const child of this.children()) child.apply(fn);
    fn(this);
    return this;
  }

  /** Total number of parameter elements (deduplicated). */
  parameterCount(onlyTrainable = false): number {
    return this.parameters().filter((p) => !onlyTrainable || p.requiresGrad).reduce((total, p) => total + p.numel, 0);
  }
}

/**
 * Public data fields of an object (Python ``vars()`` minus private names):
 * own enumerable string keys not starting with ``_``, excluding ``training``,
 * tensors, modules and functions (which Python keeps outside ``__dict__``).
 */
export function publicAttributes(value: object): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const key of Object.keys(value)) {
    if (key.startsWith('_') || key === 'training') continue;
    const item = (value as Record<string, unknown>)[key];
    if (item instanceof Tensor || item instanceof Module || typeof item === 'function') continue;
    result[key] = item;
  }
  return result;
}

/**
 * Point own fields that cached a replaced registered tensor at its
 * replacement, so subclasses holding ``readonly weight: Parameter`` fields
 * never read a stale tensor after ``setParameterAt``/``setBufferAt``/``to``.
 */
function rebindFields(owner: Module, previous: Tensor | null, replacement: Tensor | null): void {
  if (previous === null || previous === replacement) return;
  for (const key of Object.keys(owner)) {
    if ((owner as unknown as Record<string, unknown>)[key] === previous) {
      (owner as unknown as Record<string, unknown>)[key] = replacement;
    }
  }
}

function assertName(name: string): void {
  if (typeof name !== 'string' || !name || name.includes('.')) throw new TypeError(`invalid attribute name ${JSON.stringify(name)}`);
}

/** Wrap a tensor as a trainable parameter. */
export function parameter(value: Tensor, requiresGrad = true): Parameter {
  return new Parameter(value, requiresGrad);
}
