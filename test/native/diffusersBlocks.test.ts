/**
 * Every diffusers 0.40 UNet/VAE block family ImageDecoder can load, against
 * PyTorch reference models (``scripts/fixtures/diffusers_blocks_fixtures.py``):
 * identical module trees and state-dict keys, and forward outputs within
 * float32 tolerance.
 */
import { describe, expect, it } from 'vitest';
import { deserializeSafetensors, noGrad, tensor, type Tensor } from '../../src/nn/index.js';
import {
  AutoencoderKL, DDIMScheduler, UNet2DConditionModel, convertDeprecatedAttentionKey, deprecatedAttentionPaths,
} from '../../src/_internal/native/diffusers.js';
import { ValueError } from '../../src/errors.js';
import { ImageDecoder, Latent } from '../../src/ops/vec/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureBytes, fixtureJson, fromJson } from '../helpers/fixtures.js';

const records = fixtureJson('diffusers_blocks/records.json');

function load(module: { loadStateDict(state: Map<string, Tensor>): unknown }, file: string, prefix = ''): void {
  const { tensors } = deserializeSafetensors(fixtureBytes(`diffusers_blocks/${file}`));
  const selected = new Map([...tensors].filter(([key]) => key.startsWith(prefix)).map(([key, value]) => [key.slice(prefix.length), value]));
  module.loadStateDict(selected);
}

function maxRelative(actual: Tensor, expected: number[]): number {
  const scale = Math.max(...expected.map(Math.abs), 1e-6);
  return Math.max(...Array.from(actual.toArray(), (value, index) => Math.abs(value - expected[index]!))) / scale;
}

describe('diffusers UNet2DConditionModel block families match PyTorch', () => {
  for (const name of Object.keys(records.unets)) {
    it(name, () => {
      const record = records.unets[name];
      const model = new UNet2DConditionModel(record.config).eval();
      expect(model.namedModules().map(([path]) => path)).toEqual(record.modules);
      expect([...model.stateDict().keys()]).toEqual(record.state_keys);
      load(model, record.weights, record.prefix);
      const sample = fromJson(record.sample);
      const encoder = fromJson(record.encoder);
      const mask = tensor(record.mask.flat().map((value: number) => value === 1), { shape: [2, 3], dtype: 'bool' });
      const output = noGrad(() => model.forward(sample, tensor(record.timesteps, { dtype: 'int64' }), encoder, mask));
      expect(output.shape).toEqual(record.output.shape);
      expect(maxRelative(output, record.output.data)).toBeLessThan(2e-5);
      const scalar = noGrad(() => model.forward(sample, tensor(5, { dtype: 'int64' }), encoder));
      expect(maxRelative(scalar, record.scalar_output.data)).toBeLessThan(2e-5);
      expect(model.config).toEqual(record.config);
    });
  }

  it('rejects what ImageDecoder rejects and what diffusers cannot run', () => {
    const base = records.unets.odd.config;
    expect(() => new UNet2DConditionModel({ ...base, class_embed_type: 'timestep' })).toThrow(ValueError);
    expect(() => new UNet2DConditionModel({ ...base, encoder_hid_dim: 5 })).toThrow(/unsupported diffusion pipeline/);
    expect(() => new UNet2DConditionModel({ ...base, down_block_types: ['SkipDownBlock2D', 'DownBlock2D'] })).toThrow(/cannot run inside UNet2DConditionModel/);
    expect(() => new UNet2DConditionModel({ ...base, down_block_types: ['Nope', 'DownBlock2D'] })).toThrow('Nope does not exist.');
    expect(() => new UNet2DConditionModel({ ...base, act_fn: 'tanh' }))
      .toThrow("activation function tanh not found in ACT2FN mapping ['swish', 'silu', 'mish', 'gelu', 'relu']");
    expect(() => new UNet2DConditionModel({ ...base, resnet_time_scale_shift: 'ada_group' }))
      .toThrow('This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead');
    expect(() => new UNet2DConditionModel({ ...base, layers_per_block: [1] }))
      .toThrow("Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: [1]. `down_block_types`: ['CrossAttnDownBlock2D', 'DownBlock2D'].");
    expect(() => new UNet2DConditionModel({ ...base, time_embedding_type: 'fourier', time_embedding_dim: 7 }))
      .toThrow('`time_embed_dim` should be divisible by 2, but is 7.');
  });
});

describe('diffusers AutoencoderKL block families match PyTorch', () => {
  for (const name of Object.keys(records.vaes)) {
    it(name, () => {
      const record = records.vaes[name];
      const model = new AutoencoderKL(record.config).eval();
      expect(model.namedModules().map(([path]) => path)).toEqual(record.modules);
      expect([...model.stateDict().keys()]).toEqual(record.state_keys);
      load(model, `vae_${name}.safetensors`);
      const mode = noGrad(() => model.encodeMode(fromJson(record.pixels)));
      expectClose(mode.toArray(), record.mode.data, 1e-5, 1e-5);
      const decoded = noGrad(() => model.decode(fromJson(record.mode)));
      expect(maxRelative(decoded, record.decoded.data)).toBeLessThan(2e-5);
    });
  }

  it('renames legacy attention parameters of deprecated attention blocks only', () => {
    const model = new AutoencoderKL(records.vaes.attention.config);
    const paths = deprecatedAttentionPaths(model);
    expect([...paths].sort()).toEqual(['decoder.mid_block.attentions.0', 'encoder.down_blocks.0.attentions.0', 'encoder.mid_block.attentions.0']);
    expect(convertDeprecatedAttentionKey('encoder.down_blocks.0.attentions.0.query.weight', paths)).toBe('encoder.down_blocks.0.attentions.0.to_q.weight');
    expect(convertDeprecatedAttentionKey('encoder.mid_block.attentions.0.proj_attn.bias', paths)).toBe('encoder.mid_block.attentions.0.to_out.0.bias');
    // AttnUpDecoderBlock2D attentions are not deprecated blocks in diffusers.
    expect(convertDeprecatedAttentionKey('decoder.up_blocks.1.attentions.0.key.weight', paths)).toBe('decoder.up_blocks.1.attentions.0.key.weight');
    expect(() => new AutoencoderKL({ ...records.vaes.attention.config, down_block_types: ['DownBlock2D', 'DownEncoderBlock2D'] }))
      .toThrow(/cannot run inside the AutoencoderKL encoder/);
  });
});

describe('DDIMScheduler beta schedules', () => {
  it('supports every diffusers 0.40 schedule and rejects others like diffusers', () => {
    for (const schedule of ['linear', 'scaled_linear', 'squaredcos_cap_v2']) {
      expect(new DDIMScheduler({ num_train_timesteps: 10, beta_schedule: schedule }).betas.length).toBe(10);
    }
    expect(() => new DDIMScheduler({ beta_schedule: 'sigmoid' })).toThrow("sigmoid is not implemented for DDIMScheduler");
  });
});

describe('ImageDecoder over the new block families matches Python ImageDecoder', () => {
  for (const name of Object.keys(records.decoders)) {
    it(name, async () => {
      const record = records.decoders[name];
      const directory = new URL(`../fixtures/diffusers_blocks/decoder_${name}`, import.meta.url).pathname;
      const model = await ImageDecoder.fromPretrained(directory);
      const value = new Latent(fromJson(record.value), model.inputSpace, { mask: tensor([[true, true]]) });
      const image = model.call(value, { context: { noise: fromJson(record.noise) } });
      expect(image.shape).toEqual(record.image.shape);
      expectClose(image.toArray(), record.image.data, 2e-5, 1e-4);
      // ``context.seed`` draws the same noise as ``torch.randn(generator=torch.Generator().manual_seed(seed))``.
      const seeded = model.call(value, { context: { seed: record.seed } });
      expectClose(seeded.toArray(), record.seeded_image.data, 2e-5, 1e-4);
      const loss = noGrad(() => model.loss(value, fromJson(record.target), {
        noise: fromJson(record.noise), timesteps: tensor(3, { dtype: 'int64' }).reshape(1),
      }));
      expect(Math.abs(loss.item() - record.loss) / Math.abs(record.loss)).toBeLessThan(1e-5);
    });
  }
});
