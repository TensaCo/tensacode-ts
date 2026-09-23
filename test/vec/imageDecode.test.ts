/**
 * Latent diffusion ``ImageDecoder``: port of Python ``tests/vec/test_image_decode.py``
 * plus parity with artifacts, outputs, losses, gradients, experience and
 * checkpoints written by Python (``scripts/fixtures/diffusion_fixtures.py``).
 */
import { cpSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  deserializeSafetensors, getRngState, noGrad, rand, randn, serializeSafetensors, tensor, type Tensor,
} from '../../src/nn/index.js';
import { ImageDecoder, Latent, Space } from '../../src/ops/vec/index.js';
import { DDIMScheduler } from '../../src/_internal/native/diffusers.js';
import { DIFFUSERS_FOUNDATION_FILES } from '../../src/_internal/vec/diffusion.js';
import { globMatch } from '../../src/_internal/hub.js';
import { bindingRecords } from '../../src/_internal/fingerprint.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson, fromJson } from '../helpers/fixtures.js';
import { scratchDirectory } from '../training/helpers.js';

const scratch = scratchDirectory('tensorcode-diffusion-');
const fixtures = new URL('../fixtures/diffusion/', import.meta.url).pathname;
const records = fixtureJson('diffusion/records.json');
const codecs = { latent: Latent, space: Space };

function tiny(): ImageDecoder {
  return new ImageDecoder({
    input_space: new Space('image-conditioning', 6, { organization: 'sequence' }).configuration(),
    unet_config: {
      sample_size: 4, in_channels: 4, out_channels: 4, down_block_types: ['CrossAttnDownBlock2D'], up_block_types: ['CrossAttnUpBlock2D'],
      block_out_channels: [8], layers_per_block: 1, norm_num_groups: 4, cross_attention_dim: 8, attention_head_dim: 2,
    },
    vae_config: {
      in_channels: 3, out_channels: 3, latent_channels: 4, down_block_types: ['DownEncoderBlock2D'], up_block_types: ['UpDecoderBlock2D'],
      block_out_channels: [8], layers_per_block: 1, norm_num_groups: 4, sample_size: 4,
    },
    scheduler_config: { num_train_timesteps: 10, clip_sample: false },
    num_inference_steps: 2,
  });
}

function sameTensor(a: Tensor, b: Tensor): boolean {
  return a.equal(b);
}

describe('ImageDecoder parity with Python', () => {
  for (const name of ['a', 'b'] as const) {
    it(`decoder ${name}: artifact, sampling, conditioning, loss and gradients`, async () => {
      const record = records.decoders[name];
      const model = await ImageDecoder.fromPretrained(join(fixtures, `decoder_${name}`));
      expect(model.configuration()).toEqual(record.configuration);
      expect(model.unet.namedModules().map(([path]) => path)).toEqual(record.unet_modules);
      expect(model.vae.namedModules().map(([path]) => path)).toEqual(record.vae_modules);
      expect([...model.stateDict().keys()]).toEqual(record.state_keys);
      const space = model.inputSpace;
      const value = new Latent(fromJson(record.value), space, { mask: tensor(record.value_mask) });
      const prefix = new Latent(fromJson(record.prefix), space, { mask: tensor(record.prefix_mask) });
      const [conditioning, mask] = noGrad(() => model.conditioning(value, { latents: [prefix] }));
      expectClose(conditioning.data, record.conditioning.data, 1e-6, 1e-5);
      expect(mask.tolist()).toEqual(record.conditioning_mask);
      const image = model.call(value, { context: { noise: fromJson(record.noise), latents: [prefix] } });
      expectClose(image.data, record.image.data, 2e-6, 1e-5);
      const raw = fromJson(record.value);
      raw.requiresGrad = true;
      const loss = model.loss(new Latent(raw, space, { mask: tensor(record.value_mask) }), fromJson(record.target), {
        context: { latents: [prefix] }, noise: fromJson(record.noise), timesteps: tensor(record.timesteps, { dtype: 'int64' }),
      });
      expect(loss.item()).toBeCloseTo(record.loss, 5);
      loss.backward();
      expectClose(raw.grad!.data, record.value_grad.data, 1e-6, 1e-4);
      expectClose((model.projection as any).weight.grad.data, record.projection_grad.data, 1e-6, 1e-4);
      expectClose(model.unet.conv_in.weight.grad!.data, record.conv_in_grad.data, 1e-6, 1e-4);
    });
  }

  it('DDIM schedules, steps, noise and velocity are bit-identical', () => {
    for (const record of records.schedulers) {
      const scheduler = new DDIMScheduler(record.config);
      expect(scheduler.config).toEqual(record.resolved);
      expect(scheduler.betas).toEqual(record.betas);
      expect(scheduler.alphasCumprod).toEqual(record.alphas_cumprod);
      expect(scheduler.finalAlphaCumprod).toBe(record.final_alpha_cumprod);
      scheduler.setTimesteps(4);
      expect(scheduler.timesteps).toEqual(record.timesteps);
      const step = scheduler.step(fromJson(record.model_output), record.timestep, fromJson(record.sample));
      expect([...step.data]).toEqual(record.step.data);
      const steps = tensor(record.noise_timesteps, { dtype: 'int64' });
      expect([...scheduler.addNoise(fromJson(record.sample), fromJson(record.model_output), steps).data]).toEqual(record.add_noise.data);
      expect([...scheduler.getVelocity(fromJson(record.sample), fromJson(record.model_output), steps).data]).toEqual(record.velocity.data);
    }
  });

  it('Python experience replays, and a Python directory checkpoint resumes', async () => {
    const model = await ImageDecoder.fromPretrained(join(fixtures, 'decoder_a'));
    const trainer = Trainer.fromTool(model, { lr: 0.05 });
    const experience = await loadExperience(join(fixtures, 'experience.json'), { operations: trainer.operations, codecs });
    const losses = trainer.fit([experience], { epochs: 2 });
    losses.forEach((loss, index) => expect(loss).toBeCloseTo(records.training.losses[index], 5));
    const trained = await ImageDecoder.fromPretrained(join(fixtures, 'trained'));
    for (const [key, value] of trained.stateDict()) expectClose(model.stateDict().get(key)!.data, value.data, 1e-6, 1e-4);
    const resumed = await ImageDecoder.fromPretrained(join(fixtures, 'decoder_a'));
    const resumedTrainer = Trainer.fromTool(resumed, { lr: 0.05 });
    expect(await resumedTrainer.loadCheckpoint(join(fixtures, 'checkpoint'))).toEqual({ epochs: 2 });
    expect(resumedTrainer.steps).toBe(records.training.steps);
    for (const [key, value] of trained.stateDict()) expect(sameTensor(resumed.stateDict().get(key)!, value), key).toBe(true);
  });

  it('a diffusers-format foundation imports exactly (identity bridge)', async () => {
    const record = records.foundation;
    const directory = join(fixtures, 'foundation');
    const native = await ImageDecoder.fromFoundation(directory, {
      inputSpace: new Space('native', 8, { organization: 'sequence' }), bridge: 'identity', numInferenceSteps: 2, localFilesOnly: true,
    });
    const configuration = native.configuration();
    expect((configuration.foundation as any).source).toBe(directory);
    (configuration.foundation as any).source = 'FOUNDATION';
    expect(configuration).toEqual(record.configuration);
    const prefix = new Latent(fromJson(record.prefix), native.inputSpace, { mask: tensor([[false, true]]) });
    const value = new Latent(fromJson(record.embedding), native.inputSpace, { mask: tensor([[true, false]]) });
    const image = native.call(value, { context: { noise: fromJson(record.noise), latents: [prefix] } });
    expectClose(image.data, record.image.data, 2e-6, 1e-5);
    const source = await ImageDecoder.fromPretrained(join(fixtures, 'decoder_a'));
    for (const component of ['unet', 'vae'] as const) {
      const original = source[component].stateDict();
      const imported = native[component].stateDict();
      expect([...imported.keys()]).toEqual([...original.keys()]);
      for (const [key, tensorValue] of original) expect(sameTensor(imported.get(key)!, tensorValue), key).toBe(true);
    }
    const target = scratch();
    await native.savePretrained(target);
    const restored = await ImageDecoder.fromPretrained(target);
    expect(restored.configuration()).toEqual(native.configuration());
    expect(sameTensor(restored.call(value, { context: { noise: fromJson(record.noise), latents: [prefix] } }), image)).toBe(true);
  });
});

describe('ImageDecoder (tests/vec/test_image_decode.py)', () => {
  it('real reverse diffusion: replay, masked context and generator state', () => {
    const model = tiny();
    const value = new Latent(randn([1, 2, 6]), model.inputSpace, { mask: tensor([[true, false]]) });
    const noise = randn([1, 4, 4, 4]);
    model.train();
    const state = getRngState();
    const image = model.call(value, { context: { noise } });
    expect(model.training).toBe(true);
    expect(getRngState()).toEqual(state);
    expect(image.shape).toEqual([1, 3, 4, 4]);
    expect(image.min().item()).toBeGreaterThanOrEqual(0);
    expect(image.max().item()).toBeLessThanOrEqual(1);
    expect(sameTensor(image, model.call(value, { context: { noise } }))).toBe(true);
    const altered = value.tensor.clone();
    noGrad(() => altered.select(1, 1).add_(100));
    expect(sameTensor(image, model.call(new Latent(altered, value.space, { mask: value.mask }), { context: { noise } }))).toBe(true);
    const withContext = model.call(value, { context: { noise, latents: [new Latent(randn([1, 1, 6]), value.space)] } });
    expect(image.sub(withContext).abs().max().item()).toBeGreaterThan(1e-6);
    expect(sameTensor(model.call(value, { context: { seed: 42 } }), model.call(value, { context: { seed: 42 } }))).toBe(true);
    expect(() => model.call(value)).toThrow(/seed|noise/);
  });

  it('diffusion loss trains conditioning and denoiser', () => {
    const model = tiny();
    const raw = randn([1, 2, 6]);
    raw.requiresGrad = true;
    const prefixRaw = randn([1, 2, 6]);
    prefixRaw.requiresGrad = true;
    const prefix = new Latent(prefixRaw, model.inputSpace, { mask: tensor([[true, false]]) });
    const loss = model.loss(new Latent(raw, model.inputSpace), rand([1, 3, 4, 4]), {
      context: { latents: [prefix] }, noise: randn([1, 4, 4, 4]), timesteps: tensor([4], { dtype: 'int64' }),
    });
    loss.backward();
    expect(loss.allFinite()).toBe(true);
    expect(raw.grad!.abs().sum().item()).toBeGreaterThan(0);
    expect(prefixRaw.grad!.select(1, 0).abs().sum().item()).toBeGreaterThan(0);
    expect(prefixRaw.grad!.select(1, 1).abs().sum().item()).toBe(0);
    expect((model.projection as any).weight.grad.abs().sum().item()).toBeGreaterThan(0);
    expect(model.unet.conv_in.weight.grad!.abs().sum().item()).toBeGreaterThan(0);
  });

  it('complete diffusion artifact round trip', async () => {
    const model = tiny();
    const value = new Latent(randn([1, 2, 6]), model.inputSpace);
    const expected = model.call(value, { context: { seed: 3 } });
    const directory = scratch();
    await model.savePretrained(directory);
    const restored = await ImageDecoder.fromPretrained(directory);
    expect(sameTensor(expected, restored.call(value, { context: { seed: 3 } }))).toBe(true);
    expect(restored.config.conditioning_status).toBe('requires_training');
    expect(restored.configuration()).toEqual(model.configuration());
    expect([...restored.stateDict().keys()]).toEqual([...model.stateDict().keys()]);
    for (const [key, value] of model.stateDict()) expect(sameTensor(restored.stateDict().get(key)!, value), key).toBe(true);
  });

  it('diffusion objective trace restart', async () => {
    const model = tiny();
    const inputs = { value: new Latent(randn([1, 2, 6]), model.inputSpace), noise: randn([1, 4, 4, 4]), timesteps: tensor([3], { dtype: 'int64' }) };
    const target = rand([1, 3, 4, 4]);
    const trainer = Trainer.fromTool(model);
    const session = trainer.capture(inputs, target, { source: 'tiny real RGB fixture' });
    const directory = scratch();
    await session.save(join(directory, 'trace.json'), { operations: trainer.operations, codecs });
    await model.savePretrained(join(directory, 'model'));
    const restarted = await ImageDecoder.fromPretrained(join(directory, 'model'));
    const resumed = Trainer.fromTool(restarted);
    const restored = await loadExperience(join(directory, 'trace.json'), { operations: resumed.operations, codecs });
    const previous = (restarted.projection as any).weight.detach().clone();
    expect(Number.isFinite(resumed.step(restored))).toBe(true);
    expect(sameTensor(previous, (restarted.projection as any).weight)).toBe(false);
  });

  it('rejects unsupported and malformed diffusion', () => {
    const model = tiny();
    const config = structuredClone(model.config) as any;
    config.unet_config.addition_embed_type = 'text';
    expect(() => new ImageDecoder(config)).toThrow(/unsupported/);
    const value = new Latent(randn([1, 2, 6]), model.inputSpace);
    expect(() => model.call(value, { context: { noise: randn([1, 4, 5, 5]) } })).toThrow(/noise shape/);
    expect(() => model.call(new Latent(value.tensor, new Space('wrong', 6, { organization: 'sequence' })), { context: { seed: 2 } })).toThrow(/compatible/);
    expect(() => model.call(new Latent(value.tensor, value.space, { mask: tensor([[false, false]]) }), { context: { seed: 2 } })).toThrow(/valid|unmasked/);
    expect(() => model.loss(value, rand([1, 3, 4, 4]).add(2), { noise: randn([1, 4, 4, 4]), timesteps: tensor([0], { dtype: 'int64' }) }))
      .toThrow(/target_pixels/);
  });

  it('foundation rejects missing weights', async () => {
    const directory = scratch();
    cpSync(join(fixtures, 'foundation'), directory, { recursive: true });
    const file = join(directory, 'unet', 'diffusion_pytorch_model.safetensors');
    const weights = deserializeSafetensors(new Uint8Array(readFileSync(file)));
    weights.tensors.delete('conv_in.weight');
    writeFileSync(file, serializeSafetensors(weights.tensors, weights.metadata));
    await expect(ImageDecoder.fromFoundation(directory, {
      inputSpace: new Space('image-conditioning', 6, { organization: 'sequence' }), localFilesOnly: true, numInferenceSteps: 2,
    })).rejects.toThrow(/missing|incomplete/);
  });

  it('rejects non-conditioning context and unconditional UNets', () => {
    const model = tiny();
    const value = new Latent(randn([1, 2, 6]), model.inputSpace);
    expect(() => model.call(value, { context: { seed: 0, targets: rand([1, 3, 4, 4]) } })).toThrow(/context/);
    const config = structuredClone(model.config) as any;
    Object.assign(config.unet_config, { down_block_types: ['DownBlock2D'], up_block_types: ['UpBlock2D'], mid_block_type: 'UNetMidBlock2D' });
    expect(() => new ImageDecoder(config)).toThrow(/cross-attention/);
  });

  it('public identity, bridge and objective contract', async () => {
    const model = tiny();
    expect(model.config.bridge).toBe('linear');
    expect('conditioning_projection' in model.config).toBe(false);
    const objective = model.trainingOperation.configuration();
    expect(objective.operation).toBe('tensorcode.ops.vec.decode.ImageDecoder');
    expect(objective.role).toBe('objective');
    expect(objective.model).toEqual(model.configuration());
    const saved = bindingRecords(model.operationBindings());
    expect(saved.objective!.configuration.type).toBe('tensorcode.ops.vec.decode.ImageDecoder.objective');
    expect(JSON.stringify(saved)).not.toContain('tensorcode._internal.vec');
    const directory = scratch();
    await model.savePretrained(directory);
    expect(JSON.parse(readFileSync(join(directory, 'tensorcode_config.json'), 'utf8')).tool).toBe('tensorcode.ops.vec.decode.ImageDecoder');
  });

  it('conditioning context is an ordered, masked prefix', () => {
    const model = tiny();
    const value = new Latent(randn([1, 2, 6]), model.inputSpace, { mask: tensor([[true, false]]) });
    const first = new Latent(randn([1, 2, 6]), model.inputSpace, { mask: tensor([[false, true]]) });
    const second = new Latent(randn([1, 1, 6]), model.inputSpace);
    const [actual, actualMask] = model.conditioning(value, { latents: [first, second] });
    const sequence = noGrad(() => first.tensor.clone());
    const mask = tensor([[false, true, true, true, false]]);
    const combined = noGrad(() => sequence.reshape(1, 2, 6));
    const all = noGrad(() => tensor([...combined.data, ...second.tensor.data, ...value.tensor.data], { shape: [1, 5, 6] }));
    const expected = noGrad(() => (model.projection as any).forward(all.maskedFill(mask.logicalNot().unsqueeze(-1), 0)));
    expectClose(actual.data, expected.data, 0, 0);
    expect(actualMask.tolist()).toEqual(mask.tolist());
    expect(() => model.conditioning(value, { latents: [new Latent(randn([2, 1, 6]), model.inputSpace)] })).toThrow(/batch/);
  });

  it('rejects invalid and obsolete bridge configuration', () => {
    const config = tiny().config;
    expect(() => new ImageDecoder({ ...config, bridge: 'unknown' })).toThrow(/bridge/);
    expect(() => new ImageDecoder({ ...config, bridge: 'identity' })).toThrow(/native/);
    const { bridge: _bridge, ...old } = config;
    expect(() => new ImageDecoder({ ...old, conditioning_projection: 'identity' })).toThrow(/bridge/);
  });
});

describe('ImageDecoder foundation downloads', () => {
  it('fetch only the files diffusers loads (no fp16 variants, ONNX or other components)', () => {
    const wanted = (name: string): boolean => DIFFUSERS_FOUNDATION_FILES.some((pattern) => globMatch(pattern, name));
    for (const name of [
      'unet/config.json', 'unet/diffusion_pytorch_model.safetensors', 'vae/diffusion_pytorch_model.safetensors',
      'unet/diffusion_pytorch_model.safetensors.index.json', 'unet/diffusion_pytorch_model-00001-of-00002.safetensors',
      'scheduler/scheduler_config.json',
    ]) expect(wanted(name), name).toBe(true);
    for (const name of [
      'unet/diffusion_pytorch_model.fp16.safetensors', 'unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors',
      'unet/diffusion_pytorch_model.bin', 'unet/model.onnx', 'text_encoder/model.safetensors', 'model_index.json',
    ]) expect(wanted(name), name).toBe(false);
  });
});
