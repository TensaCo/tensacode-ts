"""Native transformer parity fixtures: configs, weights, forward outputs and generation."""
from __future__ import annotations

import json

import torch
from safetensors.torch import save_model
from transformers import (
    AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, CLIPConfig, CLIPModel,
    ViTConfig, ViTModel,
)

from generate import OUT, tensor_json, write_json
from tensorcode._internal.vec.text import _native_config

CONFIG_CASES = [
    {'model_type': 'bert', 'hidden_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'intermediate_size': 32, 'vocab_size': 50, 'max_position_embeddings': 32},
    {'model_type': 'bert', 'hidden_size': 16, 'num_labels': 3, 'layer_norm_eps': 1e-5, 'hidden_act': 'gelu_new'},
    {'model_type': 'bert', 'hidden_size': 16, 'id2label': {'0': 'entailment', '1': 'neutral', '2': 'contradiction'}, 'label2id': {'entailment': 0, 'neutral': 1, 'contradiction': 2}},
    {'model_type': 'roberta', 'hidden_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'intermediate_size': 32, 'vocab_size': 50, 'max_position_embeddings': 34, 'pad_token_id': 1, 'bos_token_id': 0, 'eos_token_id': 2, 'type_vocab_size': 1},
    {'model_type': 'electra', 'embedding_size': 8, 'hidden_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'intermediate_size': 32, 'vocab_size': 50, 'max_position_embeddings': 32},
    {'model_type': 'distilbert', 'dim': 16, 'n_layers': 2, 'n_heads': 2, 'hidden_dim': 32, 'vocab_size': 50, 'max_position_embeddings': 32},
    {'model_type': 't5', 'd_model': 16, 'd_ff': 32, 'd_kv': 8, 'num_heads': 2, 'num_layers': 2, 'vocab_size': 40, 'feed_forward_proj': 'gated-gelu', 'tie_word_embeddings': False, 'decoder_start_token_id': 0},
    {'model_type': 't5', 'd_model': 16, 'd_ff': 32, 'd_kv': 8, 'num_heads': 2, 'num_layers': 2, 'num_decoder_layers': 1, 'vocab_size': 40, 'relative_attention_num_buckets': 8, 'relative_attention_max_distance': 16, 'decoder_start_token_id': 0},
    {'model_type': 'vit', 'hidden_size': 16, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'intermediate_size': 32, 'image_size': 8, 'patch_size': 4, 'num_channels': 3},
    # Legacy generation parameters are dropped and ``torch_dtype`` becomes ``dtype``.
    {'model_type': 'bert', 'hidden_size': 16, 'do_sample': True, 'max_length': 30, 'num_beams': 2, 'temperature': 0.5, 'max_new_tokens': 5, 'torch_dtype': 'float16'},
]


def configs():
    cases = []
    for data in CONFIG_CASES:
        config = AutoConfig.for_model(**data) if data['model_type'] != 't5' else _native_config(data)
        cases.append({'input': data, 'to_dict': json.loads(json.dumps(config.to_dict())), 'diff': json.loads(config.to_json_string())})
    clip = CLIPConfig(text_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'vocab_size': 60, 'max_position_embeddings': 12},
                      vision_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'image_size': 8, 'patch_size': 4},
                      projection_dim=8)
    cases.append({'input': {'model_type': 'clip', 'text_config': {'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'vocab_size': 60, 'max_position_embeddings': 12},
                            'vision_config': {'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'image_size': 8, 'patch_size': 4}, 'projection_dim': 8},
                  'to_dict': json.loads(json.dumps(clip.to_dict())), 'diff': json.loads(clip.to_json_string())})
    # openai/clip-vit-base-patch32 style: nested legacy generation keys and ``*_config_dict``.
    legacy = {'model_type': 'clip', 'projection_dim': 8, 'torch_dtype': 'float32',
              'text_config': {'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'vocab_size': 60,
                              'max_position_embeddings': 12, 'do_sample': False, 'top_k': 50, 'bad_words_ids': None, 'torch_dtype': None},
              'text_config_dict': {'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'vocab_size': 60,
                                   'max_position_embeddings': 12, 'hidden_act': 'quick_gelu'},
              'vision_config': {'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1, 'num_attention_heads': 2, 'image_size': 8,
                                'patch_size': 4, 'num_beams': 1},
              'vision_config_dict': None}
    config = AutoConfig.for_model(**legacy)
    cases.append({'input': legacy, 'to_dict': json.loads(json.dumps(config.to_dict())), 'diff': json.loads(config.to_json_string())})
    write_json('native_configs.json', cases)


def seeded(shape, seed, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=generator) * 2 - 1) * scale


def save(name, model):
    save_model(model, str(OUT / f'native_{name}.safetensors'))
    return {'state_keys': list(model.state_dict().keys()),
            'shapes': {k: list(v.shape) for k, v in model.state_dict().items()}}


def encoders():
    cases = {}
    ids = torch.tensor([[3, 7, 11, 5, 9, 2], [4, 8, 6, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    types = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]])
    for index, data in enumerate(CONFIG_CASES[:6]):
        if index in (1, 2):
            continue
        torch.manual_seed(100 + index)
        config = AutoConfig.for_model(**data)
        model = AutoModel.from_config(config).eval()
        name = data['model_type']
        record = save(name, model)
        batch_ids = ids.clone()
        if name == 'roberta':
            batch_ids = torch.where(mask.bool(), batch_ids + 3, torch.ones_like(batch_ids))
        record.update(config=json.loads(config.to_json_string()), input_ids=batch_ids.tolist(), attention_mask=mask.tolist())
        with torch.no_grad():
            kwargs = {'input_ids': batch_ids, 'attention_mask': mask}
            if name in ('bert', 'electra'):
                kwargs['token_type_ids'] = types
                record['token_type_ids'] = types.tolist()
            out = model(**kwargs)
            record['last_hidden_state'] = tensor_json(out.last_hidden_state)
            if getattr(out, 'pooler_output', None) is not None:
                record['pooler_output'] = tensor_json(out.pooler_output)
            embeds = seeded((2, 5, model.get_input_embeddings().weight.shape[1]), 7 + index, 0.5)
            embeds_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
            record['inputs_embeds'] = tensor_json(embeds)
            record['inputs_embeds_mask'] = embeds_mask.tolist()
            record['inputs_embeds_output'] = tensor_json(model(inputs_embeds=embeds, attention_mask=embeds_mask).last_hidden_state)
        classifier_config = AutoConfig.for_model(**{**data, 'num_labels': 3})
        torch.manual_seed(200 + index)
        classifier = AutoModelForSequenceClassification.from_config(classifier_config).eval()
        record['classifier'] = save(f'{name}_classifier', classifier)
        record['classifier']['config'] = json.loads(classifier_config.to_json_string())
        with torch.no_grad():
            kwargs = {'input_ids': batch_ids, 'attention_mask': mask}
            if name in ('bert', 'electra'):
                kwargs['token_type_ids'] = types
            record['classifier']['logits'] = tensor_json(classifier(**kwargs).logits)
        cases[name] = record
    write_json('native_encoders.json', cases)


def t5():
    cases = {}
    for index, data in enumerate(CONFIG_CASES[6:8]):
        torch.manual_seed(300 + index)
        config = _native_config(data)
        model = AutoModelForSeq2SeqLM.from_config(config).eval()
        name = f't5_{index}'
        record = save(name, model)
        record['config'] = json.loads(model.config.to_json_string())
        record['generation_config'] = json.loads(model.generation_config.to_json_string())
        ids = torch.tensor([[5, 9, 13, 7, 1], [6, 8, 1, 0, 0]])
        mask = (ids != 0).long()
        mask[0, :] = 1
        labels = torch.tensor([[4, 12, 9, 1], [7, 1, -100, -100]])
        record.update(input_ids=ids.tolist(), attention_mask=mask.tolist(), labels=labels.tolist())
        with torch.no_grad():
            encoder = model.get_encoder()(input_ids=ids, attention_mask=mask).last_hidden_state
            record['encoder_last_hidden_state'] = tensor_json(encoder)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            record['logits'] = tensor_json(out.logits)
            record['loss'] = out.loss.item()
            record['greedy'] = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False).tolist()
            record['greedy_min'] = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, min_new_tokens=4, do_sample=False).tolist()
            record['beam'] = model.generate(input_ids=ids[:1], attention_mask=mask[:1], max_new_tokens=5, num_beams=3,
                                            num_return_sequences=3, do_sample=False).tolist()
            record['beam_batch'] = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5, num_beams=2,
                                                  do_sample=False, length_penalty=0.5, early_stopping=True).tolist()
            record['repetition'] = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, repetition_penalty=1.5, do_sample=False).tolist()
            embeds = seeded((2, 4, config.d_model), 17 + index, 0.5)
            embeds_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
            record['inputs_embeds'] = tensor_json(embeds)
            record['inputs_embeds_mask'] = embeds_mask.tolist()
            record['embeds_greedy'] = model.generate(inputs_embeds=embeds, attention_mask=embeds_mask, max_new_tokens=5, do_sample=False).tolist()
            record['embeds_loss'] = model(inputs_embeds=embeds, attention_mask=embeds_mask, labels=labels).loss.item()
        cases[name] = record
    write_json('native_t5.json', cases)


def vision():
    torch.manual_seed(400)
    config = ViTConfig(**{k: v for k, v in CONFIG_CASES[8].items() if k != 'model_type'})
    model = ViTModel(config, add_pooling_layer=False).eval()
    record = save('vit', model)
    record['config'] = json.loads(config.to_json_string())
    pixels = seeded((2, 3, 8, 8), 41)
    record['pixel_values'] = tensor_json(pixels)
    with torch.no_grad():
        record['last_hidden_state'] = tensor_json(model(pixel_values=pixels).last_hidden_state)
    torch.manual_seed(401)
    clip_config = CLIPConfig(text_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'vocab_size': 60, 'max_position_embeddings': 12, 'eos_token_id': 59, 'bos_token_id': 58, 'pad_token_id': 1},
                             vision_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 2, 'image_size': 8, 'patch_size': 4},
                             projection_dim=8)
    clip = CLIPModel(clip_config).eval()
    clip_record = save('clip', clip)
    clip_record['config'] = json.loads(json.dumps(clip_config.to_dict()))
    ids = torch.tensor([[58, 5, 9, 13, 59, 1], [58, 7, 59, 1, 1, 1]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]])
    clip_record.update(input_ids=ids.tolist(), attention_mask=mask.tolist(), pixel_values=tensor_json(pixels))
    with torch.no_grad():
        out = clip(input_ids=ids, attention_mask=mask, pixel_values=pixels)
        clip_record['text_embeds'] = tensor_json(out.text_embeds)
        clip_record['image_embeds'] = tensor_json(out.image_embeds)
        clip_record['logits_per_image'] = tensor_json(out.logits_per_image)
        clip_record['text_last_hidden_state'] = tensor_json(out.text_model_output.last_hidden_state)
        clip_record['vision_last_hidden_state'] = tensor_json(out.vision_model_output.last_hidden_state)
        clip_record['vision_pooler_output'] = tensor_json(out.vision_model_output.pooler_output)
    write_json('native_vision.json', {'vit': record, 'clip': clip_record})


def generate():
    configs()
    encoders()
    t5()
    vision()
