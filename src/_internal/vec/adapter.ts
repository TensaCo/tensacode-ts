/**
 * Tensor-native operation wrapper (Python ``tensorcode/_internal/vec/adapter.py``).
 * Preserves module registration and autograd; optional spaces tag results.
 * FOUNDATION-OWNED: used by the workspace, ranking tools and vector operations.
 */
import { Module } from '../../nn/module.js';
import { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { ModuleOperation, type Context } from '../../ops/base.js';
import { Latent, requireCompatible, type Space } from '../../ops/vec/latent.js';
import { callableIdentity, moduleConfiguration, qualifiedName, spaceConfiguration } from './configuration.js';
import type { JsonObject } from '../json.js';

/** A module with a single-argument forward (tensor or structured input). */
export type ForwardModule = Module & { forward(input: any): any };

/** Explicit context combination: ``combine(value, context) -> module input``. */
export type Combine = ((value: unknown, context: Context) => unknown) & { configuration?(): unknown };

export interface TensorAdapterOptions {
  combine?: Combine | null;
  inputSpace?: Space | null;
  outputSpace?: Space | null;
}

export class TensorAdapter<M extends ForwardModule = ForwardModule> extends ModuleOperation<unknown, unknown> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.adapter.TensorAdapter';
  readonly module: M;
  readonly combine: Combine | null;
  readonly inputSpace: Space | null;
  readonly outputSpace: Space | null;

  constructor(module: M, options: TensorAdapterOptions = {}) {
    super();
    if (!(module instanceof Module) || typeof (module as { forward?: unknown }).forward !== 'function') {
      throw new TypeError('Transform module must be an nn.Module with forward()');
    }
    this.module = this.registerModule('module', module);
    this.combine = options.combine ?? null;
    this.inputSpace = options.inputSpace ?? null;
    this.outputSpace = options.outputSpace ?? null;
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: unknown, context: Context | null): unknown {
    const source = value instanceof Latent ? value : null;
    let input: unknown = value;
    if (source) {
      if (this.inputSpace) requireCompatible(this.inputSpace, source.space);
      input = source.tensor;
    } else if (this.inputSpace) {
      throw new TypeError('A configured input_space requires a Latent input');
    }
    if (this.combine) input = this.combine(input, context ?? {});
    else if (context) throw new ValueError('Context requires an explicit combine function');
    const result = this.module.forward(input);
    if (result instanceof Latent) {
      if (this.outputSpace) requireCompatible(this.outputSpace, result.space, { role: 'output' });
      return result;
    }
    if (!this.outputSpace) return result;
    if (!(result instanceof Tensor)) throw new TypeError('output_space requires a tensor result');
    if (!source) return new Latent(result, this.outputSpace);
    const retains = result.shape.length === source.tensor.shape.length
      && result.shape.slice(0, -1).every((size, index) => size === source.tensor.shape[index]);
    return source.withTensor(result, {
      space: this.outputSpace, mask: retains ? source.mask : null, coordinates: retains ? source.coordinates : null,
    });
  }

  configuration(): JsonObject {
    return {
      operation: qualifiedName(this),
      module: moduleConfiguration(this.module),
      combine: callableIdentity(this.combine),
      input_space: spaceConfiguration(this.inputSpace),
      output_space: spaceConfiguration(this.outputSpace),
    };
  }
}
