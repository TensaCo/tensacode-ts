"""Reference outputs for real cached foundations (tests skip when the Hub cache lacks them)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModel, AutoTokenizer, CLIPModel

from generate import write_json
from tensorcode._internal.vec.text import _load_foundation, _parameter_aliases, _tokenizer_config

PROMPTS = ['translate English to German: The house is wonderful.', 'Answer the question: what color is the sky?']


def snapshot_with(repo, name):
    root = Path(HF_HUB_CACHE) / ('models--' + repo.replace('/', '--')) / 'snapshots'
    matches = sorted(p for p in root.glob('*') if (p / name).exists()) if root.exists() else []
    return matches[0] if matches else None


def summary(tensor):
    flat = tensor.detach().double().reshape(-1)
    return {'shape': list(tensor.shape), 'head': flat[:24].tolist(), 'sum': flat.sum().item(), 'abs_sum': flat.abs().sum().item()}


def t5(result):
    repo = 'google/flan-t5-small'
    snap = snapshot_with(repo, 'model.safetensors')
    if snap is None or not (snap / 'config.json').exists():
        return
    model, tokenizer = _load_foundation(repo, snap.name, {'local_files_only': True}, decoder=True)
    model.eval()
    batch = tokenizer(PROMPTS, padding=True, return_tensors='pt')
    with torch.no_grad():
        encoded = model.get_encoder()(**batch).last_hidden_state
        generated = model.generate(**batch, max_new_tokens=8, do_sample=False)
        beams = model.generate(**{k: v[:1] for k, v in batch.items()}, max_new_tokens=6, num_beams=3, num_return_sequences=3, do_sample=False)
        labels = tokenizer(['Das Haus ist wunderbar.', 'blue'], padding=True, return_tensors='pt')['input_ids']
        labels = labels.masked_fill(labels == 0, -100)
        loss = model(**batch, labels=labels).loss.item()
    result[repo] = {
        'snapshot': snap.name, 'config': json.loads(model.config.to_json_string()),
        'generation_config': json.loads(model.generation_config.to_json_string()),
        'aliases': {k: v for k, v in _parameter_aliases(model).items() if k != v},
        'tokenizer_sha': __import__('hashlib').sha256(_tokenizer_config(tokenizer)['json'].encode()).hexdigest(),
        'input_ids': batch['input_ids'].tolist(), 'attention_mask': batch['attention_mask'].tolist(),
        'encoder': summary(encoded), 'greedy': generated.tolist(), 'beam': beams.tolist(),
        'decoded': tokenizer.batch_decode(generated, skip_special_tokens=True), 'loss': loss,
    }


def electra(result):
    repo = 'google/electra-small-discriminator'
    config_snap = snapshot_with(repo, 'config.json')
    weights_snap = snapshot_with(repo, 'model.safetensors')
    if config_snap is None or weights_snap is None:
        return
    model = AutoModel.from_pretrained(repo, revision=config_snap.name, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=config_snap.name, local_files_only=True)
    batch = tokenizer(['The service recovered.', 'A kettle boils water quickly today.'], padding=True, return_tensors='pt')
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    result[repo] = {'config_snapshot': config_snap.name, 'weights_snapshot': weights_snap.name,
                    'to_dict': json.loads(json.dumps(model.config.to_dict())),
                    'input_ids': batch['input_ids'].tolist(), 'attention_mask': batch['attention_mask'].tolist(),
                    'token_type_ids': batch['token_type_ids'].tolist(), 'hidden': summary(hidden)}


def clip(result):
    repo = 'openai/clip-vit-base-patch32'
    config_snap = snapshot_with(repo, 'config.json')
    weights_snap = snapshot_with(repo, 'model.safetensors')
    if config_snap is None or weights_snap is None:
        return
    model = CLIPModel.from_pretrained(repo, revision=config_snap.name, local_files_only=True).eval()
    # Deterministic pixels reproducible in TypeScript: (sin(0.37 * i) + 1) / 2.
    pixels = ((torch.arange(3 * 224 * 224, dtype=torch.float64) * 0.37).sin().add(1).div(2)).float().reshape(1, 3, 224, 224)
    ids = torch.tensor([[49406, 320, 1125, 539, 320, 2368, 49407]])
    with torch.no_grad():
        out = model(input_ids=ids, pixel_values=pixels)
    result[repo] = {'config_snapshot': config_snap.name, 'weights_snapshot': weights_snap.name,
                    'input_ids': ids.tolist(),
                    'text_embeds': out.text_embeds.reshape(-1).tolist(), 'image_embeds': out.image_embeds.reshape(-1).tolist(),
                    'logits_per_image': out.logits_per_image.reshape(-1).tolist()}


def generate():
    result = {}
    for build in (t5, electra, clip):
        build(result)
    write_json('foundations_cached.json', result)
