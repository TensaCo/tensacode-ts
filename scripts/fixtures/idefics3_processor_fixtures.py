"""Idefics3Processor batches with several images per prompt, the special tokens it adds to
tokenizers that lack them, and Idefics3 forwards/generation over such batches."""
from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Idefics3ImageProcessor, Idefics3Processor, PreTrainedTokenizerFast
from transformers.utils import logging

from generate import OUT, tensor_json, write_json

ROOT = OUT / 'scene_language'
SMOLVLM = 'HuggingFaceTB/SmolVLM-256M-Instruct'


def _snapshot():
    from huggingface_hub.constants import HF_HUB_CACHE
    root = Path(HF_HUB_CACHE) / ('models--' + SMOLVLM.replace('/', '--')) / 'snapshots'
    found = sorted(p for p in root.glob('*') if (p / 'model.safetensors').exists()) if root.exists() else []
    return found[0] if found else None


def _image(generator, height, width):
    pixels = torch.rand(3, height, width, generator=generator)
    uint8 = (pixels.clamp(0, 1) * 255).round().to(torch.uint8)
    return uint8, Image.frombytes('RGB', (width, height), uint8.permute(1, 2, 0).contiguous().numpy().tobytes())


def _batch_record(batch):
    record = {'input_ids': batch['input_ids'].tolist(), 'attention_mask': batch['attention_mask'].tolist()}
    if 'pixel_values' in batch:
        values = batch['pixel_values'].contiguous()
        record.update(pixel_shape=list(values.shape), pixel_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest(),
                      mask_sum=batch['pixel_attention_mask'].double().sum().item(),
                      mask_sha256=hashlib.sha256(batch['pixel_attention_mask'].to(torch.int64).contiguous().numpy().tobytes()).hexdigest())
    return record


def _error(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - errors are part of the fixture
        return {'error': type(exc).__name__, 'message': str(exc)}
    return {'error': None}


def smol_batches():
    snapshot = _snapshot()
    if snapshot is None:
        print(f'skipping uncached {SMOLVLM}')
        return None
    processor = Idefics3Processor.from_pretrained(snapshot, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    generator = torch.Generator().manual_seed(31)
    raw = [_image(generator, h, w) for h, w in [(30, 45), (20, 64), (21, 21)]]
    images = [item[0].tolist() for item in raw]
    pil = [item[1] for item in raw]
    prompts = ['<|im_start|>User:<image>and<image>compare<end_of_utterance>\nAssistant:',
               '<|im_start|>User:<image>describe<end_of_utterance>\nAssistant:']
    records = {'images': images, 'prompts': prompts}
    records['nested'] = _batch_record(processor(text=prompts, images=[[pil[0], pil[1]], [pil[2]]], padding=True, return_tensors='pt'))
    records['flat'] = _batch_record(processor(text=prompts, images=pil, padding=True, return_tensors='pt'))
    records['text_only'] = _batch_record(processor(text=['<|im_start|>User: hello', 'hi'], padding=True, return_tensors='pt'))
    records['errors'] = {
        'mismatch': _error(lambda: processor(text=prompts, images=[[pil[0]], [pil[1], pil[2]]], padding=True)),
        'no_images': _error(lambda: processor(text=prompts, padding=True)),
        'extra_image': _error(lambda: processor(text=[prompts[1]], images=[pil[0], pil[1]])),
        'ragged': _error(lambda: processor(text=prompts, images=[[pil[0], pil[1]], [pil[2]]], return_tensors='pt')),
    }
    # Forward and generation of the small random SmolVLM-shaped model over the multi-image batch.
    from tensorcode.tools.scene import Scene
    tool = Scene.from_pretrained(ROOT / 'smol').eval()
    model = tool.language.model
    batch = processor(text=prompts, images=[[pil[0], pil[1]], [pil[2]]], padding=True, return_tensors='pt')
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        output = model(**batch)
        generated = model.generate(**batch, max_new_tokens=4, do_sample=False)
    records['model'] = {'last_logits_head': tensor_json(output.logits[:, -1, :256]), 'argmax': output.logits[:, -1].argmax(-1).tolist(),
                        'generated': generated.tolist()}
    return records


def added_tokens():
    """A tokenizer without the Idefics3 tokens: the processor adds them (Rust ``add_special_tokens``)."""
    vocabulary = {'[UNK]': 0, '[BOS]': 1, '[EOS]': 2, 'describe': 3, 'left': 4, '<image>': 5, 'object': 6}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    from tokenizers import AddedToken
    backend.add_tokens([AddedToken('<end_of_utterance>', normalized=True, special=False), AddedToken('<global-img>', special=True)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]')
    with tempfile.TemporaryDirectory() as directory:
        tokenizer.save_pretrained(directory)
        source = (Path(directory) / 'tokenizer.json').read_text()
        config = (Path(directory) / 'tokenizer_config.json').read_text()
    processor = Idefics3Processor(Idefics3ImageProcessor(do_resize=False, do_image_splitting=False, max_image_size={'longest_edge': 8},
                                                         size={'longest_edge': 8}), tokenizer, image_seq_len=2)
    with tempfile.TemporaryDirectory() as directory:
        processor.save_pretrained(directory)
        saved = {path.name: path.read_text() for path in Path(directory).iterdir() if path.is_file()}
    generator = torch.Generator().manual_seed(4)
    image, pil = _image(generator, 8, 8)
    batch = processor(text='describe <image> left <end_of_utterance> object<fake_token_around_image>', images=[pil], return_tensors='pt')
    return {'tokenizer_json': source, 'tokenizer_config': config, 'saved': saved, 'image': image.tolist(),
            'input_ids': batch['input_ids'].tolist(), 'image_token_id': processor.image_token_id}


def generate():
    logging.set_verbosity_error()
    write_json('scene_language/processor.json', {'smol': smol_batches(), 'added': added_tokens()})
