import { describe, expect, it } from 'vitest';
import { F, GRUCell, Parameter, noGrad, tensor, type Tensor } from '../../src/nn/index.js';
import { gradcheck, randomTensor } from '../helpers/gradcheck.js';
import { NativeConfig } from '../../src/_internal/native/config.js';
import { BertModel } from '../../src/_internal/native/bert.js';
import { T5ForConditionalGeneration } from '../../src/_internal/native/t5.js';
import { keyPaddingBias } from '../../src/_internal/native/modules.js';

describe('additional finite-difference gradient checks', () => {
  it('activations', () => {
    const x = randomTensor([3, 4], 3, 2);
    for (const fn of [
      (t: Tensor) => F.gelu(t), (t: Tensor) => F.gelu(t, 'tanh'), F.quickGelu, F.silu, F.softplus,
      (t: Tensor) => t.tanh(), (t: Tensor) => t.sigmoid(), (t: Tensor) => t.erf(), (t: Tensor) => t.exp(),
    ]) gradcheck(fn, [x.detach().clone()]);
  });

  it('elementwise math with domain restrictions', () => {
    const positive = randomTensor([2, 3], 5).abs().add(0.5);
    gradcheck((t) => t.sqrt(), [positive.detach().clone()]);
    gradcheck((t) => t.rsqrt(), [positive.detach().clone()]);
    gradcheck((t) => t.log(), [positive.detach().clone()]);
    gradcheck((t) => t.pow(2.5), [positive.detach().clone()]);
    gradcheck((a, b) => a.pow(b), [positive.detach().clone(), randomTensor([2, 3], 6)]);
    gradcheck((t) => t.clamp(-0.3, 0.4), [randomTensor([2, 3], 7)]);
    gradcheck((a, b) => F.where(a.gt(0), a, b), [randomTensor([2, 3], 8), randomTensor([2, 3], 9)]);
    gradcheck((a, b) => a.maximum(b), [randomTensor([2, 3], 10), randomTensor([2, 3], 11)]);
  });

  it('normalization and similarity', () => {
    gradcheck((t) => F.normalize(t, 2, -1), [randomTensor([3, 4], 12)]);
    gradcheck((a, b) => F.cosineSimilarity(a, b, -1), [randomTensor([3, 4], 13), randomTensor([3, 4], 14)]);
  });

  it('scaled dot-product attention with an additive mask', () => {
    const mask = keyPaddingBias(tensor([[1, 1, 0], [1, 1, 1]], { dtype: 'int64' }), 'float64');
    gradcheck((q, k, v) => F.scaledDotProductAttention(q, k, v, { bias: mask }).output,
      [randomTensor([2, 2, 3, 4], 15), randomTensor([2, 2, 3, 4], 16), randomTensor([2, 2, 3, 4], 17)]);
  });

  it('GRU cell', () => {
    const cell = new GRUCell(3, 2);
    for (const [name, parameter] of cell.namedParameters()) cell.setParameterAt(name, new Parameter(parameter.to('float64')));
    gradcheck((x, h) => cell.forward(x, h), [randomTensor([2, 3], 18), randomTensor([2, 2], 19)]);
  });

  it('end-to-end BERT and T5 parameter gradients match finite differences', () => {
    const bert = new BertModel(NativeConfig.fromDict({
      model_type: 'bert', hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2, intermediate_size: 12,
      vocab_size: 20, max_position_embeddings: 8, hidden_dropout_prob: 0, attention_probs_dropout_prob: 0,
    })).eval();
    const t5 = new T5ForConditionalGeneration(NativeConfig.fromDict({
      model_type: 't5', d_model: 8, d_ff: 12, d_kv: 4, num_heads: 2, num_layers: 1, vocab_size: 20,
      feed_forward_proj: 'gated-gelu', dropout_rate: 0, decoder_start_token_id: 0,
    })).eval();
    for (const model of [bert, t5]) {
      for (const [name, parameter] of model.namedParameters()) model.setParameterAt(name, new Parameter(parameter.to('float64')));
    }
    const ids = tensor([[3, 5, 7, 0], [4, 6, 0, 0]], { dtype: 'int64' });
    const mask = tensor([[1, 1, 1, 0], [1, 1, 0, 0]], { dtype: 'int64' });
    const checks: [() => Tensor, Parameter][] = [
      [() => bert.forward({ inputIds: ids, attentionMask: mask }).lastHiddenState.sum(), bert.encoder.layer.at(0).attention.self.query.weight],
      [() => t5.forward({ inputIds: ids, attentionMask: mask, labels: tensor([[2, 9, 1], [8, 1, -100]], { dtype: 'int64' }) }).loss!,
        t5.encoder.block.at(0).layer.at(0).getSubmodule('SelfAttention.relative_attention_bias').getParameter('weight')!],
    ];
    for (const [loss, parameter] of checks) {
      parameter.grad = null;
      loss().backward();
      const analytic = parameter.grad!.toArray();
      const eps = 1e-6;
      noGrad(() => {
        for (const index of [0, 1, Math.floor(parameter.numel / 2), parameter.numel - 1]) {
          const original = parameter.data[index]!;
          parameter.data[index] = original + eps;
          const plus = loss().item();
          parameter.data[index] = original - eps;
          const minus = loss().item();
          parameter.data[index] = original;
          expect(Math.abs((plus - minus) / (2 * eps) - analytic[index]!)).toBeLessThan(1e-6 + 1e-4 * Math.abs(analytic[index]!));
        }
      });
    }
  });
});
