"""Every diffusers 0.40 UNet/VAE block family reachable from ImageDecoder (module trees, weights, outputs).

For each configuration the reference UNet2DConditionModel / AutoencoderKL is built
with a fixed seed, its weights saved as safetensors, and its forward pass run on
fixed inputs. TypeScript loads the weights and must reproduce the module tree,
state-dict keys and outputs (``test/native/diffusersBlocks.test.ts``).
"""
from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from generate import OUT, tensor_json, write_json

ROOT = OUT / 'diffusers_blocks'

UNETS = {
    # UnCLIP-style simple cross-attention (added K/V), resnet up/downsamplers, skipped time
    # activation, output scaling, group-normed encoder states, mish and scale-shift conditioning.
    'simple_cross': dict(
        sample_size=4, in_channels=4, out_channels=4, block_out_channels=[16, 32], layers_per_block=1, norm_num_groups=8,
        down_block_types=['SimpleCrossAttnDownBlock2D', 'ResnetDownsampleBlock2D'],
        up_block_types=['ResnetUpsampleBlock2D', 'SimpleCrossAttnUpBlock2D'], mid_block_type='UNetMidBlock2DSimpleCrossAttn',
        cross_attention_dim=32, attention_head_dim=8, act_fn='mish', resnet_time_scale_shift='scale_shift',
        resnet_skip_time_act=True, resnet_out_scale_factor=1.5, cross_attention_norm='group_norm',
        time_embedding_act_fn='gelu', timestep_post_act='mish', mid_block_scale_factor=1.25),
    # Layer-normed encoder states (cross_attention_dim equals the block width).
    'simple_cross_ln': dict(
        sample_size=4, in_channels=4, out_channels=4, block_out_channels=[16, 16], layers_per_block=1, norm_num_groups=4,
        down_block_types=['SimpleCrossAttnDownBlock2D', 'DownBlock2D'],
        up_block_types=['UpBlock2D', 'SimpleCrossAttnUpBlock2D'], mid_block_type='UNetMidBlock2DSimpleCrossAttn',
        cross_attention_dim=16, attention_head_dim=4, cross_attention_norm='layer_norm', act_fn='swish'),
    # Spatial self-attention blocks, Gaussian Fourier time embedding, the plain UNet mid block,
    # per-block head dims, relu and odd latent sizes through the cross-attention path.
    'attention': dict(
        sample_size=4, in_channels=4, out_channels=4, block_out_channels=[16, 32], layers_per_block=[1, 2], norm_num_groups=4,
        down_block_types=['AttnDownBlock2D', 'CrossAttnDownBlock2D'], up_block_types=['CrossAttnUpBlock2D', 'AttnUpBlock2D'],
        mid_block_type='UNetMidBlock2D', cross_attention_dim=12, attention_head_dim=[4, 8], act_fn='relu',
        time_embedding_type='fourier', flip_sin_to_cos=False, time_embedding_dim=24, downsample_padding=0,
        conv_in_kernel=1, conv_out_kernel=5, center_input_sample=True, dropout=0.1),
    # k-upscaler style K blocks (AdaGroupNorm ResNets, K attention, K samplers) without norm groups.
    'k_diffusion': dict(
        sample_size=16, in_channels=4, out_channels=4, block_out_channels=[32, 32, 64, 64], layers_per_block=1,
        norm_num_groups=None, down_block_types=['KDownBlock2D', 'KCrossAttnDownBlock2D', 'KCrossAttnDownBlock2D', 'KCrossAttnDownBlock2D'],
        up_block_types=['KCrossAttnUpBlock2D', 'KCrossAttnUpBlock2D', 'KCrossAttnUpBlock2D', 'KUpBlock2D'], mid_block_type=None,
        cross_attention_dim=12, attention_head_dim=8, act_fn='gelu', resnet_time_scale_shift='scale_shift',
        time_embedding_type='fourier', timestep_post_act='gelu', conv_in_kernel=1, conv_out_kernel=1),
    # A K downsampler after a cross-attention block, positional embedding.
    'k_mixed': dict(
        sample_size=8, in_channels=4, out_channels=4, block_out_channels=[32, 32], layers_per_block=1, norm_num_groups=32,
        down_block_types=['CrossAttnDownBlock2D', 'KDownBlock2D'], up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'],
        cross_attention_dim=12, attention_head_dim=8, mid_block_type='UNetMidBlock2DCrossAttn'),
    # Odd latent sizes through cross-attention and plain blocks (forwarded upsample sizes).
    'odd': dict(
        sample_size=[5, 7], in_channels=4, out_channels=4, block_out_channels=[8, 16], layers_per_block=1, norm_num_groups=4,
        down_block_types=['CrossAttnDownBlock2D', 'DownBlock2D'], up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'],
        cross_attention_dim=6, attention_head_dim=2, time_embedding_act_fn='silu', only_cross_attention=[True, False]),
}

VAES = {
    'attention': dict(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=[8, 16], layers_per_block=1,
                      norm_num_groups=4, down_block_types=['AttnDownEncoderBlock2D', 'DownEncoderBlock2D'],
                      up_block_types=['UpDecoderBlock2D', 'AttnUpDecoderBlock2D'], act_fn='mish', sample_size=8),
    'no_mid_attention': dict(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=[8], layers_per_block=2,
                             norm_num_groups=4, down_block_types=['AttnDownEncoderBlock2D'], up_block_types=['AttnUpDecoderBlock2D'],
                             act_fn='gelu', mid_block_add_attention=False, use_quant_conv=False, use_post_quant_conv=False,
                             sample_size=4),
}


def _native(config):
    return json.loads(json.dumps({key: value for key, value in dict(config).items() if not key.startswith('_')}))


def generate() -> None:
    from diffusers import AutoencoderKL, UNet2DConditionModel

    ROOT.mkdir(parents=True, exist_ok=True)
    records = {'unets': {}, 'vaes': {}}
    decoder_records, decoder_unets = decoders()
    records['decoders'] = decoder_records
    for index, (name, config) in enumerate(UNETS.items()):
        torch.manual_seed(100 + index)
        if name in decoder_unets:
            # Weights live in ``decoder_<name>/model.safetensors`` under ``unet.``.
            model = decoder_unets[name].eval()
        else:
            model = UNet2DConditionModel(**config).eval()
            with torch.no_grad():
                # Nonzero everything, including zero-initialized output projections.
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.05)
        size = config['sample_size']
        height, width = (size, size) if isinstance(size, int) else size
        sample = torch.randn(2, 4, height, width)
        encoder = torch.randn(2, 3, config['cross_attention_dim'])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        timesteps = torch.tensor([3, 17])
        with torch.no_grad():
            output = model(sample, timesteps, encoder_hidden_states=encoder, encoder_attention_mask=mask).sample
            scalar = model(sample, torch.tensor(5), encoder_hidden_states=encoder).sample
        if name not in decoder_unets:
            save_file({key: value.contiguous() for key, value in model.state_dict().items()}, str(ROOT / f'unet_{name}.safetensors'))
        records['unets'][name] = {
            'weights': f'decoder_{name}/model.safetensors' if name in decoder_unets else f'unet_{name}.safetensors',
            'prefix': 'unet.' if name in decoder_unets else '',
            'config': _native(model.config),
            'modules': [path for path, _ in model.named_modules()],
            'state_keys': list(model.state_dict().keys()),
            'sample': tensor_json(sample), 'encoder': tensor_json(encoder), 'mask': mask.tolist(), 'timesteps': timesteps.tolist(),
            'output': tensor_json(output), 'scalar_output': tensor_json(scalar),
        }
    for index, (name, config) in enumerate(VAES.items()):
        torch.manual_seed(200 + index)
        model = AutoencoderKL(**config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.05)
        pixels = torch.rand(2, 3, config['sample_size'], config['sample_size']) * 2 - 1
        with torch.no_grad():
            mode = model.encode(pixels).latent_dist.mode()
            decoded = model.decode(mode).sample
        save_file({key: value.contiguous() for key, value in model.state_dict().items()}, str(ROOT / f'vae_{name}.safetensors'))
        records['vaes'][name] = {
            'config': _native(model.config),
            'modules': [path for path, _ in model.named_modules()],
            'state_keys': list(model.state_dict().keys()),
            'pixels': tensor_json(pixels), 'mode': tensor_json(mode), 'decoded': tensor_json(decoded),
        }
    write_json('diffusers_blocks/records.json', records)


# ImageDecoder end to end over the new block families (Python ``tensorcode`` ImageDecoder).
DECODERS = {
    'k_diffusion': ('k_diffusion', 'attention', dict(num_train_timesteps=10, beta_schedule='squaredcos_cap_v2', clip_sample=False)),
    'simple_cross': ('simple_cross', 'no_mid_attention', dict(num_train_timesteps=12, beta_schedule='scaled_linear',
                                                            prediction_type='v_prediction', timestep_spacing='trailing')),
}


def decoders():
    from tensorcode.ops.vec.decode import ImageDecoder
    from tensorcode.ops.vec.latent import Latent, Space
    results, unets = {}, {}
    for index, (name, (unet, vae, scheduler)) in enumerate(DECODERS.items()):
        torch.manual_seed(300 + index)
        vae_config = {key: value for key, value in VAES[vae].items()}
        model = ImageDecoder({
            'input_space': {'name': 'image-conditioning', 'dimensions': 6, 'organization': 'sequence'},
            'unet_config': UNETS[unet], 'vae_config': vae_config, 'scheduler_config': scheduler, 'num_inference_steps': 2,
        })
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.05)
        model.save_pretrained(ROOT / f'decoder_{name}')
        space = model.input_space
        size = model.latent_size
        value = Latent(torch.randn(1, 2, 6, generator=torch.Generator().manual_seed(10 + index)), space,
                       mask=torch.tensor([[True, True]]))
        noise = torch.randn((1, 4, *size), generator=torch.Generator().manual_seed(30 + index))
        with torch.no_grad():
            image = model(value, context={'noise': noise})
            seeded = model(value, context={'seed': 7})
        pixels = tuple(s * model.vae_scale_factor for s in size)
        target = torch.rand((1, 3, *pixels), generator=torch.Generator().manual_seed(40 + index))
        with torch.no_grad():
            loss = model.loss(value, target, noise=noise, timesteps=torch.tensor([3]))
        results[name] = {
            'value': tensor_json(value.tensor), 'noise': tensor_json(noise), 'image': tensor_json(image),
            'seed': 7, 'seeded_image': tensor_json(seeded), 'target': tensor_json(target), 'loss': float(loss),
        }
        unets[unet] = model.unet
    return results, unets
