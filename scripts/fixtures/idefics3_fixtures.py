"""Idefics3 (SmolVLM) parity: vision tower, connector, Llama text model, loss and greedy generation."""
from __future__ import annotations

import json

import torch
from safetensors.torch import save_model
from transformers import GenerationConfig, Idefics3Config, Idefics3ForConditionalGeneration

from generate import OUT, tensor_json, write_json

VISION = dict(hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, image_size=8,
              patch_size=2)
TEXT = dict(model_type='llama', hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=24, pad_token_id=0, rope_theta=10000, max_position_embeddings=128)
CASES = {
    # Grouped-query attention, default RoPE, untied head.
    'gqa': dict(vision_config=VISION, text_config=TEXT, pad_token_id=0, scale_factor=2, image_token_id=20),
    # llama3 RoPE, tied embeddings, head_dim differing from hidden / heads.
    'llama3': dict(vision_config=dict(VISION, hidden_act='gelu'),
                   text_config=dict(TEXT, head_dim=6, num_key_value_heads=1, tie_word_embeddings=True,
                                    rope_scaling={'rope_type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0,
                                                  'high_freq_factor': 4.0, 'original_max_position_embeddings': 16}),
                   pad_token_id=0, scale_factor=2, image_token_id=20, tie_word_embeddings=True),
}


def prompt(images):
    """Token ids with ``images * 4`` image tokens (image_seq_len = (8 / 2) ** 2 / 2 ** 2)."""
    ids = [1, 5, 6]
    for _ in range(images):
        ids += [21] + [20] * 4 + [21]
    return ids + [7, 8, 9]


def generate():
    record = {}
    for name, spec in CASES.items():
        torch.manual_seed(len(name) * 101)
        config = Idefics3Config(**json.loads(json.dumps(spec)))
        model = Idefics3ForConditionalGeneration(config).eval()
        save_model(model, str(OUT / f'native_idefics3_{name}.safetensors'))
        # Two images; the second is zero-padded on the right and bottom, plus an all-zero padding image.
        pixels = torch.zeros(1, 3, 3, 8, 8)
        pixels[0, 0] = torch.rand(3, 8, 8) * 2 - 1
        pixels[0, 1, :, :6, :4] = torch.rand(3, 6, 4) * 2 - 1
        pixel_mask = torch.zeros(1, 3, 8, 8, dtype=torch.long)
        pixel_mask[0, 0] = 1
        pixel_mask[0, 1, :6, :4] = 1
        ids = torch.tensor([prompt(2)])
        mask = torch.ones_like(ids)
        labels = ids.clone()
        labels[0, :10] = -100
        with torch.no_grad():
            features = model.model.get_image_features(pixels, pixel_mask, return_dict=True).pooler_output
            vision = model.model.vision_model(pixel_values=pixels[0, :2], patch_attention_mask=None).last_hidden_state
            output = model(input_ids=ids, attention_mask=mask, pixel_values=pixels, pixel_attention_mask=pixel_mask,
                           labels=labels, use_cache=False)
            generation = GenerationConfig(bos_token_id=1, eos_token_id=[2, 3], pad_token_id=0, suppress_tokens=[20, 21],
                                          repetition_penalty=1.3, no_repeat_ngram_size=2, min_new_tokens=2)
            generated = model.generate(input_ids=ids, attention_mask=mask, pixel_values=pixels,
                                       pixel_attention_mask=pixel_mask, generation_config=generation,
                                       max_new_tokens=6, do_sample=False)
            plain = model.generate(input_ids=ids, attention_mask=mask, image_hidden_states=features,
                                   generation_config=GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0),
                                   max_new_tokens=5, do_sample=False)
        record[name] = {
            'config': json.loads(json.dumps(config.to_dict())),
            'state_keys': list(model.state_dict().keys()),
            'modules': [path for path, _ in model.named_modules()],
            'pixel_values': tensor_json(pixels), 'pixel_attention_mask': pixel_mask.tolist(),
            'input_ids': ids.tolist(), 'labels': labels.tolist(),
            'vision': tensor_json(vision),
            'image_features': tensor_json(features),
            'logits': tensor_json(output.logits),
            'loss': output.loss.item(),
            'generation_config': generation.to_dict(),
            'generated': generated[0].tolist(),
            'plain_generated': plain[0].tolist(),
        }
    write_json('native_idefics3.json', record)
