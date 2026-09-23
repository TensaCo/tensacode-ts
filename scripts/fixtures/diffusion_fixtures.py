"""Latent diffusion ImageDecoder parity (diffusers UNet2DConditionModel, AutoencoderKL, DDIMScheduler)."""
from __future__ import annotations

import json
import shutil

import torch

from generate import OUT, tensor_json, write_json

ROOT = OUT / 'diffusion'
SPACE = {'name': 'image-conditioning', 'dimensions': 6, 'organization': 'sequence'}
CONFIGS = {
    # tests/vec/test_image_decode.py
    'a': {
        'input_space': SPACE,
        'unet_config': dict(sample_size=4, in_channels=4, out_channels=4, down_block_types=['CrossAttnDownBlock2D'],
                            up_block_types=['CrossAttnUpBlock2D'], block_out_channels=[8], layers_per_block=1,
                            norm_num_groups=4, cross_attention_dim=8, attention_head_dim=2),
        'vae_config': dict(in_channels=3, out_channels=3, latent_channels=4, down_block_types=['DownEncoderBlock2D'],
                           up_block_types=['UpDecoderBlock2D'], block_out_channels=[8], layers_per_block=1,
                           norm_num_groups=4, sample_size=4),
        'scheduler_config': dict(num_train_timesteps=10, clip_sample=False),
        'num_inference_steps': 2,
    },
    # Down/upsamplers, plain resnet blocks, linear projections, v-prediction.
    'b': {
        'input_space': SPACE,
        'unet_config': dict(sample_size=4, in_channels=4, out_channels=4,
                            down_block_types=['CrossAttnDownBlock2D', 'DownBlock2D'],
                            up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'], block_out_channels=[8, 16],
                            layers_per_block=1, norm_num_groups=4, cross_attention_dim=8, attention_head_dim=2,
                            use_linear_projection=True, transformer_layers_per_block=1),
        'vae_config': dict(in_channels=3, out_channels=3, latent_channels=4,
                           down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D'],
                           up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D'], block_out_channels=[8, 8],
                           layers_per_block=1, norm_num_groups=4, sample_size=8),
        'scheduler_config': dict(num_train_timesteps=12, beta_schedule='scaled_linear', prediction_type='v_prediction',
                                 steps_offset=1, clip_sample=True, clip_sample_range=1.5),
        'num_inference_steps': 3,
    },
}
SCHEDULERS = [
    dict(num_train_timesteps=10, clip_sample=False),
    dict(num_train_timesteps=20, beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, timestep_spacing='trailing'),
    dict(num_train_timesteps=16, beta_schedule='squaredcos_cap_v2', timestep_spacing='linspace', set_alpha_to_one=False),
    dict(num_train_timesteps=12, prediction_type='sample', rescale_betas_zero_snr=True, steps_offset=1),
]


def seeded(shape, seed, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * scale


def decoders():
    from tensorcode.ops.vec.decode import ImageDecoder
    from tensorcode.ops.vec.latent import Latent, Space
    records = {}
    for index, (name, config) in enumerate(CONFIGS.items()):
        torch.manual_seed(400 + index)
        model = ImageDecoder(config)
        with torch.no_grad():  # nonzero, well-scaled weights exercise every path
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.05)
        model.save_pretrained(ROOT / f'decoder_{name}')
        space = model.input_space
        size = model.latent_size
        pixels = tuple(s * model.vae_scale_factor for s in size)
        value = Latent(seeded((1, 2, 6), 10 + index), space, mask=torch.tensor([[True, False]]))
        prefix = Latent(seeded((1, 2, 6), 20 + index), space, mask=torch.tensor([[False, True]]))
        noise = seeded((1, 4, *size), 30 + index)
        with torch.no_grad():
            image = model(value, context={'noise': noise, 'latents': [prefix]})
            conditioning, mask = model._conditioning(value, {'latents': [prefix]})
        target = torch.rand((1, 3, *pixels), generator=torch.Generator().manual_seed(40 + index))
        raw = value.tensor.clone().requires_grad_()
        loss = model.loss(Latent(raw, space, mask=value.mask), target, context={'latents': [prefix]},
                          noise=noise, timesteps=torch.tensor([3]))
        loss.backward()
        records[name] = {
            'configuration': model.configuration(),
            'unet_modules': [path for path, _ in model.unet.named_modules()],
            'vae_modules': [path for path, _ in model.vae.named_modules()],
            'state_keys': list(model.state_dict().keys()),
            'value': tensor_json(value.tensor), 'value_mask': value.mask.tolist(),
            'prefix': tensor_json(prefix.tensor), 'prefix_mask': prefix.mask.tolist(),
            'noise': tensor_json(noise), 'image': tensor_json(image),
            'conditioning': tensor_json(conditioning), 'conditioning_mask': mask.tolist(),
            'target': tensor_json(target), 'timesteps': [3], 'loss': float(loss),
            'value_grad': tensor_json(raw.grad), 'projection_grad': tensor_json(model.projection.weight.grad),
            'conv_in_grad': tensor_json(model.unet.conv_in.weight.grad),
        }
        model.zero_grad()
    return records


def schedulers():
    from diffusers import DDIMScheduler
    records = []
    for index, config in enumerate(SCHEDULERS):
        scheduler = DDIMScheduler.from_config(config)
        scheduler.set_timesteps(4)
        sample = seeded((1, 2, 2, 2), 50 + index)
        output = seeded((1, 2, 2, 2), 60 + index)
        timestep = int(scheduler.timesteps[1])
        timesteps = torch.tensor([int(scheduler.timesteps[0])])
        records.append({
            'config': config, 'resolved': json.loads(json.dumps({k: v for k, v in scheduler.config.items() if not k.startswith('_')})),
            'betas': scheduler.betas.double().tolist(), 'alphas_cumprod': scheduler.alphas_cumprod.double().tolist(),
            'final_alpha_cumprod': float(scheduler.final_alpha_cumprod), 'timesteps': scheduler.timesteps.tolist(),
            'sample': tensor_json(sample), 'model_output': tensor_json(output), 'timestep': timestep,
            'step': tensor_json(scheduler.step(output, timestep, sample, eta=0).prev_sample),
            'add_noise': tensor_json(scheduler.add_noise(sample, output, timesteps)),
            'velocity': tensor_json(scheduler.get_velocity(sample, output, timesteps)),
            'noise_timesteps': timesteps.tolist(),
        })
    return records


def foundation():
    """A diffusers-format foundation and the Python identity-bridge result."""
    from tensorcode.ops.vec.decode import ImageDecoder
    from tensorcode.ops.vec.latent import Latent, Space
    source = ImageDecoder.from_pretrained(ROOT / 'decoder_a')
    directory = ROOT / 'foundation'
    source.unet.save_pretrained(directory / 'unet')
    source.vae.save_pretrained(directory / 'vae')
    source.scheduler.save_pretrained(directory / 'scheduler')
    native = ImageDecoder.from_foundation(directory, input_space=Space('native', 8, organization='sequence'),
                                          bridge='identity', num_inference_steps=2, local_files_only=True)
    embedding = seeded((1, 2, 8), 70)
    prefix = Latent(seeded((1, 2, 8), 71), native.input_space, mask=torch.tensor([[False, True]]))
    noise = seeded((1, 4, 4, 4), 72)
    with torch.no_grad():
        image = native(Latent(embedding, native.input_space, mask=torch.tensor([[True, False]])),
                       context={'noise': noise, 'latents': [prefix]})
    configuration = native.configuration()
    configuration['foundation']['source'] = 'FOUNDATION'
    return {'configuration': configuration, 'embedding': tensor_json(embedding), 'prefix': tensor_json(prefix.tensor),
            'noise': tensor_json(noise), 'image': tensor_json(image)}


def training():
    """A Python experience and directory checkpoint for decoder ``a``."""
    from tensorcode.ops.vec.decode import ImageDecoder
    from tensorcode.ops.vec.latent import Latent, Space
    from tensorcode.training import Trainer
    model = ImageDecoder.from_pretrained(ROOT / 'decoder_a')
    trainer = Trainer.from_tool(model, lr=0.05)
    inputs = {'value': Latent(seeded((1, 2, 6), 80), model.input_space), 'noise': seeded((1, 4, 4, 4), 81),
              'timesteps': torch.tensor([7])}
    target = torch.rand((1, 3, 4, 4), generator=torch.Generator().manual_seed(82))
    experience = trainer.capture(inputs, target, source='fixture:diffusion')
    experience.save(ROOT / 'experience.json', operations=trainer.operations, codecs={'latent': Latent, 'space': Space})
    losses = trainer.fit([experience], epochs=2)
    trainer.save_checkpoint(ROOT / 'checkpoint', progress={'epochs': 2})
    model.save_pretrained(ROOT / 'trained')
    return {'losses': [float(loss) for loss in losses], 'steps': trainer.steps}


def generate():
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    write_json('diffusion/records.json', {'decoders': decoders(), 'schedulers': schedulers(), 'foundation': foundation(),
                                          'training': training()})
