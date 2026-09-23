"""Llama rotary position embeddings of every transformers RoPE type (``llama.ts``): full
forwards and cached incremental decoding across ``original_max_position_embeddings``."""
from __future__ import annotations

import json

import torch
from safetensors.torch import save_model
from transformers import LlamaConfig, LlamaModel

from generate import OUT, tensor_json, write_json

BASE = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            vocab_size=50, max_position_embeddings=16, rope_theta=10000.0)
VARIANTS = {
    'default': {},
    'linear': {'rope_parameters': {'rope_type': 'linear', 'factor': 2.0, 'rope_theta': 10000.0}},
    'dynamic': {'rope_parameters': {'rope_type': 'dynamic', 'factor': 2.0, 'rope_theta': 10000.0}},
    'yarn': {'rope_parameters': {'rope_type': 'yarn', 'factor': 4.0, 'rope_theta': 10000.0, 'original_max_position_embeddings': 8}},
    'yarn_mscale': {'rope_parameters': {'rope_type': 'yarn', 'factor': 3.0, 'rope_theta': 500.0, 'original_max_position_embeddings': 8,
                                        'mscale': 0.8, 'mscale_all_dim': 0.5, 'beta_fast': 8.0, 'beta_slow': 2.0, 'truncate': False}},
    'yarn_attention': {'rope_parameters': {'rope_type': 'yarn', 'factor': 2.0, 'rope_theta': 10000.0, 'original_max_position_embeddings': 4,
                                           'attention_factor': 1.3}},
    'longrope': {'rope_parameters': {'rope_type': 'longrope', 'rope_theta': 10000.0, 'original_max_position_embeddings': 8,
                                     'short_factor': [1.0, 1.1, 1.3, 1.6], 'long_factor': [1.2, 2.0, 3.5, 5.0]}},
    'longrope_factor': {'rope_parameters': {'rope_type': 'longrope', 'rope_theta': 1000.0, 'original_max_position_embeddings': 8, 'factor': 3.0,
                                            'short_factor': [1.0, 1.0, 1.2, 1.4], 'long_factor': [1.5, 2.5, 3.0, 4.0]}},
    'llama3': {'rope_parameters': {'rope_type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
                                   'original_max_position_embeddings': 8, 'rope_theta': 10000.0}},
    'proportional': {'rope_parameters': {'rope_type': 'proportional', 'factor': 1.5, 'partial_rotary_factor': 0.5, 'rope_theta': 10000.0}},
}


def generate():
    root = OUT / 'rope'
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    ids = torch.tensor([[3, 9, 14, 1, 22, 7, 30, 41, 5, 18, 26, 33, 2, 11, 47, 8, 19, 36, 4, 27]])
    for index, (name, extra) in enumerate(VARIANTS.items()):
        torch.manual_seed(100 + index)
        config = LlamaConfig(**{**BASE, **json.loads(json.dumps(extra))})
        model = LlamaModel(config).eval()
        save_model(model, str(root / f'{name}.safetensors'))
        with torch.no_grad():
            full = model(input_ids=ids).last_hidden_state
            out = model(input_ids=ids[:, :6], use_cache=True)
            cache = out.past_key_values
            steps = [out.last_hidden_state[:, -1]]
            for position in range(6, ids.shape[1]):
                out = model(input_ids=ids[:, position:position + 1], past_key_values=cache, use_cache=True)
                steps.append(out.last_hidden_state[:, -1])
            # A short sequence after the long ones (``dynamic`` resets to the original frequencies).
            short = model(input_ids=ids[:, :5]).last_hidden_state
        records[name] = {
            'config': json.loads(json.dumps(config.to_dict())),
            'inv_freq': model.rotary_emb.original_inv_freq.tolist(), 'attention_scaling': model.rotary_emb.attention_scaling,
            'full': tensor_json(full), 'steps': tensor_json(torch.stack(steps, 1)), 'short': tensor_json(short),
        }
    write_json('rope/records.json', {'input_ids': ids.tolist(), 'variants': records})
