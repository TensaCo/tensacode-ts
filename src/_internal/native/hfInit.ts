/**
 * transformers 5 weight initialization order (``PreTrainedModel.post_init`` →
 * ``initialize_weights``), reproduced so seeded fresh models draw the same
 * random numbers in the same order as Python.
 *
 * ``initialize_weights`` walks the module tree post-order (children before
 * their parent, in registration order). Every module not yet marked
 * initialized gets the model's ``_init_weights`` and is then marked, so a
 * nested ``PreTrainedModel`` (for example ``T5Stack``) that already ran its own
 * ``post_init`` during construction is skipped by its parent. When the walk
 * reaches a nested pretrained model it switches to that model's own
 * ``_init_weights`` (``smart_apply``).
 */
import { noGrad } from '../../nn/autograd.js';
import { randomInitSuppressed } from '../../nn/random.js';
import * as init from '../../nn/init.js';
import { Conv2d, Embedding, GroupNorm, Linear, MultiheadAttention } from '../../nn/layers.js';
import { Tensor } from '../../nn/tensor.js';
import { Module } from '../../nn/module.js';
import type { NativeConfig } from './config.js';

/** A model's ``_init_weights(module)``. */
export type InitWeights = (module: Module) => void;

const initialized = new WeakSet<Module>();
const pretrainedInit = new WeakMap<Module, InitWeights>();

/** The Python class name of a module (``module.__class__.__name__``). */
export function pythonClassName(module: Module): string {
  const qualified = (module.constructor as { qualifiedName?: unknown }).qualifiedName;
  if (typeof qualified === 'string' && qualified && qualified !== (Module as { qualifiedName?: unknown }).qualifiedName) {
    const parts = qualified.split('.');
    return parts[parts.length - 1]!;
  }
  return module.constructor.name;
}

function smartApply(module: Module, initWeights: InitWeights): void {
  for (const child of module.children()) smartApply(child, pretrainedInit.get(child) ?? initWeights);
  if (initialized.has(module)) return;
  initWeights(module);
  initialized.add(module);
}

/**
 * ``PreTrainedModel.post_init()`` weight initialization for ``model`` with its
 * ``_init_weights``. Registers ``model`` as a pretrained (sub-)model so an
 * enclosing model's walk uses this function for its subtree.
 */
export function postInit(model: Module, initWeights: InitWeights): void {
  pretrainedInit.set(model, initWeights);
  // ``init_weights`` does nothing for models built on the meta device (``from_pretrained``).
  if (randomInitSuppressed()) return;
  noGrad(() => smartApply(model, initWeights));
}

/**
 * ``_initialize_missing_keys`` after loading a checkpoint into a model built
 * with random initialization suppressed: the model's ``_init_weights`` walks
 * every module, but tensors in ``loaded`` (and anything sharing their storage,
 * such as tied weights) are protected, so only missing weights draw random
 * numbers. ``fallback`` is used when the model registered no initializer.
 */
export function initializeMissingWeights(model: Module, loaded: ReadonlySet<object>, fallback: InitWeights): void {
  const initWeights = pretrainedInit.get(model) ?? fallback;
  init.withInitGuard((tensor) => loaded.has(tensor._storage), () => noGrad(() => smartApply(model, initWeights)));
}

/** Mark modules as initialized without running an initializer (``_is_hf_initialized``). */
export function markInitialized(...modules: Module[]): void {
  for (const module of modules) initialized.add(module);
}

/**
 * The standard deviation ``PreTrainedModel._init_weights`` uses:
 * ``initializer_range`` (or 0.02 when falsy), else ``init_std``, else
 * ``initializer_factor``, else the text configuration's ``initializer_range``.
 */
export function initializerStd(config: NativeConfig, textConfig: NativeConfig | null = null): number {
  const range = config.get('initializer_range');
  if (range !== undefined) return typeof range === 'number' && range ? range : 0.02;
  const initStd = config.get('init_std');
  if (typeof initStd === 'number') return initStd;
  const factor = config.get('initializer_factor');
  if (typeof factor === 'number') return factor;
  const text = textConfig?.get('initializer_range');
  return typeof text === 'number' ? text : 0.02;
}

/** Modules that ``_init_weights`` treats as normalization layers (ones/zeros). */
function isNorm(module: Module): boolean {
  const name = pythonClassName(module);
  return module instanceof GroupNorm || name.includes('LayerNorm') || name.includes('RMSNorm');
}

/**
 * ``PreTrainedModel._init_weights``: normal(0, std) Linear/Conv/Embedding
 * weights with zero biases and zeroed padding rows, torch's own reset for
 * ``nn.MultiheadAttention``, and unit/zero normalization parameters.
 */
export function baseInitWeights(module: Module, std: number): void {
  if (module instanceof Linear || module instanceof Conv2d) {
    init.normal_(module.weight, 0, std);
    if (module.bias) init.zeros_(module.bias);
  } else if (module instanceof Embedding) {
    init.normal_(module.weight, 0, std);
    if (module.paddingIdx !== null && !init.isInitGuarded(module.weight)) {
      const width = module.embeddingDim;
      module.weight.data.fill(0, module.paddingIdx * width, (module.paddingIdx + 1) * width);
    }
  } else if (module instanceof MultiheadAttention) {
    module.resetParameters();
  } else if (isNorm(module)) {
    const weight = (module as { weight?: unknown }).weight;
    const bias = (module as { bias?: unknown }).bias;
    if (weight instanceof Tensor) init.ones_(weight);
    if (bias instanceof Tensor) init.zeros_(bias);
  }
}
