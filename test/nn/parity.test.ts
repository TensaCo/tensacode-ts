import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  Adam, AdamW, Conv2d, EmbeddingBag, F, GRU, GRUCell, LayerNorm, Linear, Module, Parameter, SGD, Tensor,
  deserializeSafetensors, loadModelFromBytes, serializeModel, serializeSafetensors, tensor, Embedding,
  type DType,
} from '../../src/nn/index.js';
import { expectClose } from '../helpers/gradcheck.js';

interface TensorJson { shape: number[]; dtype: string; data: number[] }

const fixture = (name: string) => JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8'));
const load = (value: TensorJson, dtype: DType = 'float64') => tensor(value.data, { shape: value.shape, dtype });
const state = (record: Record<string, TensorJson>) => new Map(Object.entries(record).map(([k, v]) => [k, load(v)]));

function toDouble(module: Module): void {
  for (const [name, param] of module.namedParameters()) {
    module.setParameterAt(name, new Parameter(param.to('float64')));
  }
}

describe('layers match PyTorch outputs', () => {
  const cases = fixture('nn_layers.json');

  it('Linear', () => {
    const layer = new Linear(4, 3);
    toDouble(layer);
    layer.loadStateDict(state(cases.linear.state));
    expectClose(layer.forward(load(cases.linear.input)).data, cases.linear.output.data, 1e-12);
  });

  it('LayerNorm', () => {
    const layer = new LayerNorm(4);
    toDouble(layer);
    layer.loadStateDict(state(cases.layer_norm.state));
    expectClose(layer.forward(load(cases.layer_norm.input)).data, cases.layer_norm.output.data, 1e-12);
  });

  it('GRUCell and GRU', () => {
    const cell = new GRUCell(4, 3);
    toDouble(cell);
    cell.loadStateDict(state(cases.gru_cell.state));
    expectClose(cell.forward(load(cases.gru_cell.input), load(cases.gru_cell.hidden)).data, cases.gru_cell.output.data, 1e-12);
    const gru = new GRU(4, 3, { batchFirst: true });
    toDouble(gru);
    gru.loadStateDict(state(cases.gru.state));
    const result = gru.forward(load(cases.gru.input));
    expectClose(result.output.data, cases.gru.output.data, 1e-12);
    expectClose(result.hidden.data, cases.gru.hidden.data, 1e-12);
    expect(result.hidden.shape).toEqual(cases.gru.hidden.shape);
  });

  it('Conv2d', () => {
    const conv = new Conv2d(2, 3, [3, 2], { stride: 2, padding: 1 });
    toDouble(conv);
    conv.loadStateDict(state(cases.conv2d.state));
    const output = conv.forward(load(cases.conv2d.input));
    expect(output.shape).toEqual(cases.conv2d.output.shape);
    expectClose(output.data, cases.conv2d.output.data, 1e-12);
  });

  it('EmbeddingBag', () => {
    const bag = new EmbeddingBag(5, 3);
    toDouble(bag);
    bag.loadStateDict(state(cases.embedding_bag.state));
    expectClose(bag.forward(cases.embedding_bag.indices, cases.embedding_bag.offsets).data, cases.embedding_bag.output.data, 1e-12);
  });

  it('cross entropy with ignore index and GELU variants', () => {
    const loss = F.crossEntropy(load(cases.cross_entropy.logits), tensor(cases.cross_entropy.targets, { dtype: 'int64' }));
    expect(loss.item()).toBeCloseTo(cases.cross_entropy.loss, 12);
    expectClose(F.gelu(load(cases.gelu.input)).data, cases.gelu.erf.data, 1e-12);
    expectClose(F.gelu(load(cases.gelu.input), 'tanh').data, cases.gelu.tanh.data, 1e-12);
  });
});

describe('optimizers follow PyTorch update rules', () => {
  const cases = fixture('nn_optimizers.json');
  for (const name of Object.keys(cases)) {
    it(name, () => {
      const record = cases[name];
      const param = new Parameter(load(record.start));
      const options = record.options as Record<string, unknown>;
      const optimizer = name.startsWith('sgd')
        ? new SGD([param], {
          lr: options.lr as number, momentum: options.momentum as number | undefined,
          dampening: options.dampening as number | undefined, weightDecay: options.weight_decay as number | undefined,
          nesterov: options.nesterov as boolean | undefined,
        })
        : new (name === 'adamw' ? AdamW : Adam)([param], {
          lr: options.lr as number, weightDecay: options.weight_decay as number | undefined,
          amsgrad: options.amsgrad as boolean | undefined, betas: options.betas as [number, number] | undefined,
        });
      const coefficientSets: Tensor[] = record.coefficients.map((value: TensorJson) => load(value));
      for (const coefficient of coefficientSets) {
        optimizer.zeroGrad();
        param.square().mul(coefficient).sum().add(param.sum().mul(0.3)).backward();
        optimizer.step();
      }
      expectClose(param.data, record.final.data, 1e-12, 1e-10);
      const saved = optimizer.stateDict();
      expect(Object.keys(saved.state['0'] ?? {}).sort()).toEqual(record.state_keys);
      expect(Object.keys(saved.param_groups[0]!).sort()).toEqual(record.param_group_keys);
    });
  }

  it('round-trips optimizer state', () => {
    const param = new Parameter(tensor([1, 2, 3]));
    const optimizer = new AdamW([param], { lr: 0.1 });
    param.sum().backward();
    optimizer.step();
    const other = new AdamW([param], { lr: 0.5 });
    other.loadStateDict(optimizer.stateDict());
    expect(other.paramGroups[0]!.lr).toBe(0.1);
    expect(other.state.get(param)!.step!.item()).toBe(1);
  });
});

describe('safetensors', () => {
  const bytes = new Uint8Array(readFileSync(new URL('../fixtures/safetensors_mixed.safetensors', import.meta.url)));
  const expected = fixture('safetensors_mixed.json');

  it('reads every supported dtype written by the reference library', () => {
    const { tensors } = deserializeSafetensors(bytes);
    for (const [name, value] of Object.entries(expected) as [string, TensorJson][]) {
      const actual = tensors.get(name)!;
      expect(actual.shape).toEqual(value.shape);
      expect(actual.dtype).toBe(value.dtype);
      if (value.dtype === 'bool') expect(Array.from(actual.data)).toEqual(value.data.map(Number));
      else expectClose(actual.data, value.data, 0, 0);
    }
  });

  it('writes byte-identical files', () => {
    const { tensors } = deserializeSafetensors(bytes);
    expect(Buffer.from(serializeSafetensors(tensors)).equals(Buffer.from(bytes))).toBe(true);
  });

  it('stores tied weights once with alias metadata like save_model', () => {
    const reference = new Uint8Array(readFileSync(new URL('../fixtures/safetensors_tied.safetensors', import.meta.url)));
    class Tied extends Module {
      shared = this.registerModule('shared', new Embedding(4, 2));
      head = this.registerModule('head', new Linear(2, 4, { bias: false }));
      other = this.registerModule('other', new Linear(2, 2));
      constructor() {
        super();
        this.head.registerParameter('weight', this.shared.weight);
      }
    }
    const model = new Tied();
    loadModelFromBytes(model, reference);
    expect(model.head.weight).toBe(model.shared.weight);
    const values = fixture('safetensors_tied.json');
    expectClose(model.shared.weight.data, values['shared.weight'].data, 1e-7);
    const written = serializeModel(model);
    expect(Buffer.from(written).equals(Buffer.from(reference))).toBe(true);
  });
});
