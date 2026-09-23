/**
 * Gradient-descent optimizers with PyTorch update rules and state layouts.
 *
 * ``stateDict()`` mirrors ``torch.optim.Optimizer.state_dict()``: parameters are
 * referenced by integer position across all groups, hyperparameters use the
 * PyTorch snake_case names, and per-parameter slots are tensors.
 */
import { noGrad } from './autograd.js';
import { Tensor, zerosLike, scalar } from './tensor.js';
import { shapesEqual } from './shape.js';

export type ParamGroupOptions = Record<string, number | boolean | readonly number[]>;

export interface ParamGroup {
  params: Tensor[];
  [option: string]: unknown;
}

export interface ParamGroupInput {
  params: Iterable<Tensor>;
  [option: string]: unknown;
}

export interface OptimizerStateDict {
  state: Record<string, Record<string, Tensor>>;
  param_groups: Array<Record<string, unknown> & { params: number[] }>;
}

export abstract class Optimizer {
  /** Identity recorded in checkpoints (PyTorch-compatible class path). */
  static readonly identity: string = 'tensorcode.nn.optim.Optimizer';
  readonly paramGroups: ParamGroup[];
  readonly state = new Map<Tensor, Record<string, Tensor>>();
  protected readonly defaults: Record<string, unknown>;

  constructor(params: Iterable<Tensor> | Iterable<ParamGroupInput>, defaults: Record<string, unknown>) {
    this.defaults = defaults;
    const list = [...(params as Iterable<Tensor | ParamGroupInput>)];
    if (!list.length) throw new Error('optimizer got an empty parameter list');
    const groups: ParamGroupInput[] = list[0] instanceof Tensor
      ? [{ params: list as Tensor[] }]
      : (list as ParamGroupInput[]);
    this.paramGroups = groups.map((group) => {
      const { params, ...options } = group;
      for (const key of Object.keys(options)) {
        if (!(key in defaults)) throw new Error(`unknown optimizer option ${key}`);
      }
      const values = [...params];
      for (const value of values) {
        if (!(value instanceof Tensor)) throw new TypeError('optimizer parameters must be tensors');
        if (!value.requiresGrad) throw new Error('optimizer parameters must require gradients');
      }
      return { ...structuredCloneOptions(defaults), ...options, params: values };
    });
    const all = this.paramGroups.flatMap((group) => group.params);
    if (new Set(all).size !== all.length) throw new Error('some parameters appear in more than one parameter group');
  }

  get identity(): string {
    return (this.constructor as typeof Optimizer).identity;
  }

  /** Clear gradients (``set_to_none`` semantics). */
  zeroGrad(): void {
    for (const group of this.paramGroups) for (const param of group.params) param.grad = null;
  }

  abstract step(): void;

  /** Names of the state slots this optimizer may create for a parameter. */
  abstract stateSlots(): readonly string[];

  stateDict(): OptimizerStateDict {
    const index = new Map<Tensor, number>();
    let next = 0;
    const param_groups = this.paramGroups.map((group) => {
      const { params, ...options } = group;
      const ids = params.map((param) => {
        if (!index.has(param)) index.set(param, next++);
        return index.get(param)!;
      });
      return { ...structuredCloneOptions(options), params: ids };
    });
    const state: Record<string, Record<string, Tensor>> = {};
    for (const [param, slots] of this.state) {
      const id = index.get(param);
      if (id === undefined) continue;
      state[String(id)] = Object.fromEntries(Object.entries(slots).map(([key, value]) => [key, value.clone().detach()]));
    }
    return { state, param_groups };
  }

  loadStateDict(stateDict: OptimizerStateDict): void {
    const groups = stateDict.param_groups;
    if (!Array.isArray(groups) || groups.length !== this.paramGroups.length) {
      throw new Error('loaded optimizer state has a different number of parameter groups');
    }
    const lookup = new Map<number, Tensor>();
    groups.forEach((saved, groupIndex) => {
      const current = this.paramGroups[groupIndex]!;
      if (!Array.isArray(saved.params) || saved.params.length !== current.params.length) {
        throw new Error('loaded optimizer parameter group sizes differ');
      }
      saved.params.forEach((id, position) => lookup.set(id, current.params[position]!));
    });
    const nextState = new Map<Tensor, Record<string, Tensor>>();
    for (const [key, slots] of Object.entries(stateDict.state ?? {})) {
      const param = lookup.get(Number(key));
      if (!param) throw new Error(`optimizer state refers to unknown parameter ${key}`);
      const copied: Record<string, Tensor> = {};
      for (const [name, value] of Object.entries(slots)) {
        if (!this.stateSlots().includes(name)) throw new Error(`unknown optimizer state slot ${name}`);
        if (name !== 'step' && !shapesEqual(value.shape, param.shape)) throw new Error(`optimizer slot ${name} has the wrong shape`);
        copied[name] = value.clone().detach();
      }
      nextState.set(param, copied);
    }
    groups.forEach((saved, groupIndex) => {
      const current = this.paramGroups[groupIndex]!;
      for (const [key, value] of Object.entries(saved)) {
        if (key !== 'params') current[key] = structuredCloneValue(value);
      }
    });
    this.state.clear();
    for (const [param, slots] of nextState) this.state.set(param, slots);
  }

  protected slots(param: Tensor): Record<string, Tensor> {
    let slots = this.state.get(param);
    if (!slots) {
      slots = {};
      this.state.set(param, slots);
    }
    return slots;
  }
}

function structuredCloneValue(value: unknown): unknown {
  return Array.isArray(value) ? [...value] : value;
}

function structuredCloneOptions(options: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(options).map(([key, value]) => [key, structuredCloneValue(value)]));
}

function number(group: ParamGroup, key: string): number {
  const value = group[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`optimizer option ${key} must be a finite number`);
  return value;
}

export interface SGDOptions {
  lr?: number;
  momentum?: number;
  dampening?: number;
  weightDecay?: number;
  nesterov?: boolean;
  maximize?: boolean;
}

export class SGD extends Optimizer {
  static override readonly identity = 'torch.optim.sgd.SGD';

  constructor(params: Iterable<Tensor> | Iterable<ParamGroupInput>, options: SGDOptions = {}) {
    const lr = options.lr ?? 1e-3;
    if (!(lr >= 0)) throw new RangeError('invalid learning rate');
    super(params, {
      lr, momentum: options.momentum ?? 0, dampening: options.dampening ?? 0,
      weight_decay: options.weightDecay ?? 0, nesterov: options.nesterov ?? false, maximize: options.maximize ?? false,
      foreach: null, differentiable: false, fused: null,
    });
    for (const group of this.paramGroups) {
      if (group.nesterov && (number(group, 'momentum') <= 0 || number(group, 'dampening') !== 0)) {
        throw new Error('Nesterov momentum requires a momentum and zero dampening');
      }
    }
  }

  stateSlots(): readonly string[] {
    return ['momentum_buffer'];
  }

  step(): void {
    noGrad(() => {
      for (const group of this.paramGroups) {
        const lr = number(group, 'lr');
        const momentum = number(group, 'momentum');
        const dampening = number(group, 'dampening');
        const weightDecay = number(group, 'weight_decay');
        for (const param of group.params) {
          if (!param.grad) continue;
          let grad = param.grad;
          if (group.maximize) grad = grad.neg();
          if (weightDecay !== 0) grad = grad.add(param.detach().mul(weightDecay));
          if (momentum !== 0) {
            const slots = this.slots(param);
            let buffer = slots.momentum_buffer;
            if (!buffer) {
              buffer = grad.clone().detach();
              slots.momentum_buffer = buffer;
            } else {
              buffer.mul_(momentum).add_(grad, 1 - dampening);
            }
            grad = group.nesterov ? grad.add(buffer.mul(momentum)) : buffer;
          }
          param.add_(grad, -lr);
        }
      }
    });
  }
}

export interface AdamOptions {
  lr?: number;
  betas?: readonly [number, number];
  eps?: number;
  weightDecay?: number;
  amsgrad?: boolean;
  maximize?: boolean;
}

export class Adam extends Optimizer {
  static override readonly identity: string = 'torch.optim.adam.Adam';

  constructor(params: Iterable<Tensor> | Iterable<ParamGroupInput>, options: AdamOptions = {}, decoupledDefaultDecay = 0) {
    const betas = options.betas ?? [0.9, 0.999];
    const lr = options.lr ?? 1e-3;
    if (!(lr >= 0) || !(betas[0] >= 0 && betas[0] < 1) || !(betas[1] >= 0 && betas[1] < 1)) {
      throw new RangeError('invalid Adam hyperparameters');
    }
    super(params, {
      lr, betas: [betas[0], betas[1]], eps: options.eps ?? 1e-8,
      weight_decay: options.weightDecay ?? decoupledDefaultDecay, amsgrad: options.amsgrad ?? false,
      maximize: options.maximize ?? false, foreach: null, capturable: false, differentiable: false, fused: null,
      decoupled_weight_decay: decoupledDefaultDecay !== 0,
    });
  }

  stateSlots(): readonly string[] {
    return ['step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'];
  }

  step(): void {
    noGrad(() => {
      for (const group of this.paramGroups) {
        const lr = number(group, 'lr');
        const [beta1, beta2] = group.betas as [number, number];
        const eps = number(group, 'eps');
        const weightDecay = number(group, 'weight_decay');
        for (const param of group.params) {
          if (!param.grad) continue;
          let grad = param.grad;
          if (group.maximize) grad = grad.neg();
          const fusable = !group.amsgrad && param.dtype === 'float32' && grad.dtype === 'float32'
            && (group.decoupled_weight_decay || weightDecay === 0);
          if (group.decoupled_weight_decay) {
            if (weightDecay !== 0 && !fusable) param.mul_(1 - lr * weightDecay);
          } else if (weightDecay !== 0) {
            grad = grad.add(param.detach().mul(weightDecay));
          }
          const slots = this.slots(param);
          if (!slots.step) {
            slots.step = scalar(0);
            slots.exp_avg = zerosLike(param);
            slots.exp_avg_sq = zerosLike(param);
            if (group.amsgrad) slots.max_exp_avg_sq = zerosLike(param);
          }
          const stepTensor = slots.step;
          stepTensor.add_(1);
          const step = stepTensor.item();
          const expAvg = slots.exp_avg!;
          const expAvgSq = slots.exp_avg_sq!;
          if (fusable && fusedAdamUpdate(param, grad, expAvg, expAvgSq, {
            lr, beta1, beta2, eps, step, decay: group.decoupled_weight_decay && weightDecay !== 0 ? 1 - lr * weightDecay : null,
          })) continue;
          expAvg.mul_(beta1).add_(grad, 1 - beta1);
          expAvgSq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);
          const biasCorrection1 = 1 - beta1 ** step;
          const biasCorrection2Sqrt = Math.sqrt(1 - beta2 ** step);
          const stepSize = lr / biasCorrection1;
          let second = expAvgSq;
          if (group.amsgrad) {
            const maximum = slots.max_exp_avg_sq!;
            const target = maximum.data;
            const values = expAvgSq.data;
            for (let index = 0; index < target.length; index += 1) if (values[index]! > target[index]!) target[index] = values[index]!;
            maximum._storage.version += 1;
            second = maximum;
          }
          const denominator = second.sqrt().div(biasCorrection2Sqrt).add(eps);
          param.addcdiv_(expAvg, denominator, -stepSize);
        }
      }
    });
  }
}

/**
 * One Adam(W) update of a float32 parameter in a single pass, performing
 * exactly the float64 operations and float32 roundings of the chained in-place
 * ops (``mul_``, ``add_``, ``addcmul_``, ``sqrt().div().add()``, ``addcdiv_``),
 * so results are bit-identical without temporaries. Returns ``false`` when
 * the storages are not all float32.
 */
function fusedAdamUpdate(
  param: Tensor, grad: Tensor, expAvg: Tensor, expAvgSq: Tensor,
  options: { lr: number; beta1: number; beta2: number; eps: number; step: number; decay: number | null },
): boolean {
  const p = param.data;
  const g = grad.data;
  const m = expAvg.data;
  const v = expAvgSq.data;
  if (!(p instanceof Float32Array) || !(g instanceof Float32Array) || !(m instanceof Float32Array) || !(v instanceof Float32Array)) return false;
  if (g.length !== p.length || m.length !== p.length || v.length !== p.length) return false;
  const { lr, beta1, beta2, eps, step, decay } = options;
  const keep1 = 1 - beta1;
  const keep2 = 1 - beta2;
  // Tensor-scalar arithmetic stores the scalar operand in the tensor's float32 dtype.
  const correction = Math.fround(Math.sqrt(1 - beta2 ** step));
  const epsilon = Math.fround(eps);
  const scale = -(lr / (1 - beta1 ** step));
  const f32 = Math.fround;
  for (let index = 0; index < p.length; index += 1) {
    const gradient = g[index]!;
    const first = f32(f32(m[index]! * beta1) + keep1 * gradient);
    const second = f32(f32(v[index]! * beta2) + keep2 * gradient * gradient);
    m[index] = first;
    v[index] = second;
    const denominator = f32(f32(f32(Math.sqrt(second)) / correction) + epsilon);
    const current = decay === null ? p[index]! : f32(p[index]! * decay);
    p[index] = current + scale * first / denominator;
  }
  param._storage.version += decay === null ? 1 : 2;
  expAvg._storage.version += 2;
  expAvgSq._storage.version += 2;
  return true;
}

export class AdamW extends Adam {
  static override readonly identity = 'torch.optim.adamw.AdamW';

  constructor(params: Iterable<Tensor> | Iterable<ParamGroupInput>, options: AdamOptions = {}) {
    super(params, options, 1e-2);
  }
}
