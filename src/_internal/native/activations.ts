/**
 * Parameter-free activation modules with Hugging Face ``ACT2FN`` identities.
 *
 * Transformers registers activations as submodules (``intermediate_act_fn``,
 * ``DenseReluDense.act``, ``mlp.activation_fn``, ...). Registering the same
 * modules here keeps ``namedModules()`` equal to Python's ``named_modules()``,
 * which checkpoint module-mode topologies depend on.
 */
import { Module } from '../../nn/module.js';
import { activation } from '../../nn/ops/nn.js';
import type { Tensor } from '../../nn/tensor.js';

/** A registered activation; ``forward`` applies the ``ACT2FN`` function. */
export class ActivationModule extends Module {
  static override readonly qualifiedName: string = 'transformers.activations.GELUActivation';
  readonly #fn: (x: Tensor) => Tensor;

  constructor(name: string) {
    super();
    this.#fn = activation(name);
  }

  override configurationAttributes(): Record<string, unknown> {
    return {};
  }

  forward(input: Tensor): Tensor {
    return this.#fn(input);
  }
}

class GELUActivation extends ActivationModule {
  static override readonly qualifiedName: string = 'transformers.activations.GELUActivation';
}
class NewGELUActivation extends ActivationModule {
  static override readonly qualifiedName: string = 'transformers.activations.NewGELUActivation';
}
class GELUTanh extends ActivationModule {
  static override readonly qualifiedName: string = 'transformers.activations.GELUTanh';
}
class FastGELUActivation extends ActivationModule {
  static override readonly qualifiedName: string = 'transformers.activations.FastGELUActivation';
}
class QuickGELUActivation extends ActivationModule {
  static override readonly qualifiedName: string = 'transformers.activations.QuickGELUActivation';
}
class SiLUActivation extends ActivationModule {
  static override readonly qualifiedName: string = 'transformers.activations.SiLUActivation';
}
class LinearActivation extends ActivationModule {
  static override readonly qualifiedName: string = 'transformers.activations.LinearActivation';
}
class ReLU extends ActivationModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.ReLU';
  override configurationAttributes(): Record<string, unknown> { return { inplace: false }; }
}
class SiLU extends ActivationModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.SiLU';
  override configurationAttributes(): Record<string, unknown> { return { inplace: false }; }
}
class Tanh extends ActivationModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.Tanh';
}
class Sigmoid extends ActivationModule {
  static override readonly qualifiedName: string = 'torch.nn.modules.activation.Sigmoid';
}

const CLASSES: Record<string, new (name: string) => ActivationModule> = {
  gelu: GELUActivation,
  gelu_python: GELUActivation,
  gelu_new: NewGELUActivation,
  gelu_pytorch_tanh: GELUTanh,
  gelu_fast: FastGELUActivation,
  quick_gelu: QuickGELUActivation,
  relu: ReLU,
  silu: SiLUActivation,
  swish: SiLU,
  tanh: Tanh,
  sigmoid: Sigmoid,
  linear: LinearActivation,
  identity: LinearActivation,
};

/** ``ACT2FN[name]`` as a registrable module. */
export function activationModule(name: string): ActivationModule {
  const cls = CLASSES[name];
  if (!cls) return new ActivationModule(name); // activation() raises for unknown names
  return new cls(name);
}
