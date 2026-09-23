/**
 * Differentiable evidence workspace shared by owned cognitive models (Python
 * ``tensorcode/_internal/workspace.py``). Attention links index source
 * positions; they are learned routing weights, not claims of factual support.
 * Slots carry no authored semantic roles. FOUNDATION-OWNED.
 */
import { Module } from '../nn/module.js';
import { Parameter, Tensor, ones, randn } from '../nn/tensor.js';
import { GELU, GRUCell, Linear, Sequential } from '../nn/layers.js';
import { noGrad } from '../nn/autograd.js';
import { ValueError } from '../errors.js';
import { TensorAdapter as Transform } from './vec/adapter.js';
import type { JsonObject } from './json.js';

export interface WorkspaceOutput {
  /** ``[batch, slots, dimensions]`` refined slot states. */
  conditioning: Tensor;
  /** ``[batch, slots]`` all-true boolean mask. */
  mask: Tensor;
  /** ``[batch, slots, tokens]`` final-step source attention. */
  attention: Tensor;
  /** ``[batch, slots, slots]`` final-step slot relations. */
  relations: Tensor;
}

function positive(value: number, name: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) throw new ValueError(`${name} must be a positive integer`);
  return value;
}

export class Workspace extends Module {
  static override readonly qualifiedName: string = 'tensorcode._internal.workspace.Workspace';
  readonly dimensions: number;
  readonly slots: number;
  readonly steps: number;
  readonly queries: Parameter;
  readonly key: Transform<Linear>;
  readonly value: Transform<Linear>;
  readonly query: Transform<Linear>;
  readonly refine: GRUCell;
  readonly relation_query: Transform<Linear>;
  readonly relation_key: Transform<Linear>;
  readonly relation_value: Transform<Linear>;
  readonly update: Transform<Sequential>;

  constructor(dimensions: number, slots = 8, steps = 2) {
    super();
    this.dimensions = positive(dimensions, 'dimensions');
    this.slots = positive(slots, 'slots');
    this.steps = positive(steps, 'steps');
    const d = dimensions;
    const initial = noGrad(() => randn([slots, d]).div(Math.sqrt(d)));
    this.queries = this.registerParameter('queries', new Parameter(initial));
    const linear = () => new Transform(new Linear(d, d, { bias: false }));
    this.key = this.registerModule('key', linear());
    this.value = this.registerModule('value', linear());
    this.query = this.registerModule('query', linear());
    this.refine = this.registerModule('refine', new GRUCell(d, d));
    this.relation_query = this.registerModule('relation_query', linear());
    this.relation_key = this.registerModule('relation_key', linear());
    this.relation_value = this.registerModule('relation_value', linear());
    this.update = this.registerModule('update', new Transform(new Sequential(new Linear(d, d * 2), new GELU(), new Linear(d * 2, d))));
  }

  configuration(): JsonObject {
    return {
      architecture: 'tensorcode._internal.workspace.Workspace', version: 1,
      dimensions: this.dimensions, slots: this.slots, steps: this.steps,
    };
  }

  forward(encoded: Tensor, mask: Tensor | null = null): WorkspaceOutput {
    if (!(encoded instanceof Tensor) || encoded.ndim !== 3) throw new ValueError('encoded must be a tensor with shape [batch, tokens, dimensions]');
    const [batch, tokens, dimensions] = encoded.shape as [number, number, number];
    if (batch < 1 || tokens < 1 || dimensions !== this.dimensions) {
      throw new ValueError('encoded needs nonempty batch/tokens and the configured dimensions');
    }
    if (!encoded.isFloatingPoint || !encoded.allFinite()) throw new ValueError('encoded must contain finite floating point values');
    let valid = mask;
    if (valid === null) valid = ones([batch, tokens], { dtype: 'bool' });
    else if (!(valid instanceof Tensor) || valid.dtype !== 'bool' || valid.shape[0] !== batch || valid.shape[1] !== tokens || valid.ndim !== 2) {
      throw new ValueError('mask must be boolean with shape [batch, tokens]');
    }
    if (!valid.any(1).all().item()) throw new ValueError('every input must contain at least one unmasked token');
    // Mask before projections as well as softmax: excluded sources contribute
    // neither values nor input gradients, even when very large but finite.
    const evidence = encoded.maskedFill(valid.logicalNot().unsqueeze(-1), 0);
    const keys = this.key.call(evidence) as Tensor;
    const values = this.value.call(evidence) as Tensor;
    let state = this.queries.unsqueeze(0).expand(batch, this.slots, dimensions);
    const scale = Math.sqrt(this.dimensions);
    const blocked = valid.logicalNot().unsqueeze(1);
    let attention: Tensor | null = null;
    let relations: Tensor | null = null;
    for (let step = 0; step < this.steps; step += 1) {
      const logits = (this.query.call(state) as Tensor).matmul(keys.transpose(-2, -1)).div(scale);
      attention = logits.maskedFill(blocked, Number.NEGATIVE_INFINITY).softmax(-1);
      const received = attention.matmul(values);
      state = this.refine.forward(received.reshape(-1, dimensions), state.reshape(-1, dimensions)).reshape(batch, this.slots, dimensions);
      relations = (this.relation_query.call(state) as Tensor).matmul((this.relation_key.call(state) as Tensor).transpose(-2, -1)).div(scale).softmax(-1);
      const related = relations.matmul(this.relation_value.call(state) as Tensor);
      state = state.add(this.update.call(related) as Tensor);
    }
    return { conditioning: state, mask: ones([batch, this.slots], { dtype: 'bool' }), attention: attention!, relations: relations! };
  }
}
