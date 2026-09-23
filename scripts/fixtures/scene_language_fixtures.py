"""Scene language mode (Idefics3/SmolVLM) parity: artifacts, processing, interpretation, loss and foundations."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (GenerationConfig, Idefics3Config, Idefics3ForConditionalGeneration, Idefics3ImageProcessor,
                          Idefics3Processor, PreTrainedTokenizerFast)

from generate import OUT, tensor_json, write_json

ROOT = OUT / 'scene_language'
SMOLVLM = 'HuggingFaceTB/SmolVLM-256M-Instruct'


def _snapshot():
    """The cached SmolVLM snapshot directory, or None."""
    from huggingface_hub.constants import HF_HUB_CACHE
    root = Path(HF_HUB_CACHE) / ('models--' + SMOLVLM.replace('/', '--')) / 'snapshots'
    found = sorted(p for p in root.glob('*') if (p / 'model.safetensors').exists()) if root.exists() else []
    return found[0] if found else None


def _fresh(name):
    path = ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    return path


def _bindings(tool):
    from tensorcode._internal.training.persistence import bindings
    return {name: record['fingerprint'] for name, record in bindings(tool.operation_bindings()).items()}


def tiny_processor():
    """The processor of Python's ``tests/models/test_scene_language.py``."""
    vocabulary = {'[UNK]': 0, '[BOS]': 1, '[EOS]': 2, 'describe': 3, 'left': 4, 'right': 5, 'object': 6, 'user': 7,
                  'assistant': 8, ':': 9, '<image>': 10, '<fake_token_around_image>': 11, '<end_of_utterance>': 12,
                  '<global-img>': 13}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]',
                                        pad_token='[UNK]', additional_special_tokens=['<image>', '<fake_token_around_image>',
                                                                                      '<end_of_utterance>', '<global-img>'])
    template = ("{% for message in messages %}{{ message['role'] }}: {% for part in message['content'] %}{% if part['type'] == "
                "'image' %}<image>{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endfor %}{% if add_generation_prompt %}"
                " assistant:{% endif %}")
    return Idefics3Processor(Idefics3ImageProcessor(do_resize=False, do_image_splitting=False, max_image_size={'longest_edge': 8},
                                                    size={'longest_edge': 8}), tokenizer, image_seq_len=4, chat_template=template)


def _assets(processor):
    with tempfile.TemporaryDirectory() as directory:
        processor.save_pretrained(directory)
        return {p.relative_to(directory).as_posix(): p.read_text(encoding='utf-8')
                for p in Path(directory).rglob('*') if p.is_file()}


def _scene(language, generation, assets, **options):
    from tensorcode.tools.scene import Scene
    config = {'mode': 'language', 'language_config': json.loads(json.dumps(language.to_dict())),
              'generation_config': generation.to_dict(), '_language_assets': assets,
              'processor_hashes': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in assets.items()}, **options}
    return Scene(config).eval()


def _receipt_record(tool, value, **kwargs):
    batch, workspace, visual = tool.language.prepare(value)
    return {'input_ids': batch['input_ids'].tolist(), 'image_hidden_states': tensor_json(batch['image_hidden_states']),
            'visual_tokens': visual}


def tiny():
    torch.manual_seed(7)
    processor = tiny_processor()
    assets = _assets(processor)
    language = Idefics3Config(vision_config={'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2,
                                             'image_size': 8, 'patch_size': 2},
                              text_config={'model_type': 'llama', 'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1,
                                           'num_attention_heads': 2, 'num_key_value_heads': 2, 'vocab_size': 16, 'pad_token_id': 0,
                                           'rope_theta': 10000},
                              pad_token_id=0, scale_factor=2, image_token_id=10)
    generation = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0, suppress_tokens=[10, 11, 12, 13])
    tool = _scene(language, generation, assets, workspace_dimensions=4, workspace_slots=2, max_new_tokens=8)
    generator = torch.Generator().manual_seed(11)
    pixels = torch.rand(3, 8, 8, generator=generator)
    value = {'pixels': pixels, 'question': 'describe object', 'source_id': 'fixture:image'}
    with torch.no_grad():
        # Give the residual path a nonzero effect so the workspace reaches the language model.
        tool.language.gate.fill_(.3)
        record = {'pixels': tensor_json(pixels), 'configuration': tool.configuration(), 'bindings': _bindings(tool),
                  'state_keys': list(tool.state_dict()), 'modules': [name for name, _ in tool.named_modules()],
                  'prepared': _receipt_record(tool, value),
                  'receipt': tool.interpret(value, max_new_tokens=8), 'receipt_short': tool.interpret(value, max_new_tokens=2),
                  'loss': tool.loss(value, 'left object').item()}
    loss = tool.loss(value, 'right object')
    loss.backward()
    record['gate_grad'] = tool.language.gate.grad.item()
    tool.zero_grad()
    tool.save_pretrained(_fresh('tiny'))
    return record


def smol_processor_scene():
    """Real SmolVLM processor assets (image splitting, LANCZOS) with a small random Idefics3 model."""
    snapshot = _snapshot()
    if snapshot is None:
        print(f'skipping uncached {SMOLVLM}')
        return None
    processor = Idefics3Processor.from_pretrained(snapshot, local_files_only=True)
    assets = _assets(processor)
    torch.manual_seed(21)
    language = Idefics3Config(vision_config={'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2,
                                             'image_size': 512, 'patch_size': 32},
                              text_config={'model_type': 'llama', 'hidden_size': 8, 'intermediate_size': 12, 'num_hidden_layers': 1,
                                           'num_attention_heads': 2, 'num_key_value_heads': 1, 'vocab_size': 49280, 'pad_token_id': 2,
                                           'rope_theta': 100000, 'max_position_embeddings': 4096, 'tie_word_embeddings': True},
                              pad_token_id=2, scale_factor=2, image_token_id=49190, tie_word_embeddings=True)
    generation = GenerationConfig(bos_token_id=0, eos_token_id=49279, pad_token_id=2)
    tool = _scene(language, generation, assets, workspace_dimensions=4, workspace_slots=2, max_new_tokens=8)
    generator = torch.Generator().manual_seed(12)
    records = {'snapshot': Path(snapshot).name, 'processing': []}
    # Processor outputs for several image sizes (pixel values are hashed: they are large).
    for height, width in [(30, 45), (64, 20), (21, 21), (70, 30)]:
        pixels = torch.rand(3, height, width, generator=generator)
        uint8 = (pixels.clamp(0, 1) * 255).round().to(torch.uint8)
        from PIL import Image
        image = Image.frombytes('RGB', (width, height), uint8.permute(1, 2, 0).contiguous().numpy().tobytes())
        batch = processor(text='<|im_start|>User:<image>describe<end_of_utterance>\nAssistant:', images=[image], return_tensors='pt')
        values = batch['pixel_values'].contiguous()
        records['processing'].append({
            'pixels': tensor_json(pixels), 'input_ids': batch['input_ids'][0].tolist(),
            'pixel_shape': list(values.shape), 'pixel_sha256': hashlib.sha256(values.numpy().tobytes()).hexdigest(),
            'pixel_sum': values.double().sum().item(), 'pixel_head': values.reshape(-1)[:64].tolist(),
            'mask_sum': batch['pixel_attention_mask'].double().sum().item(),
        })
    pixels = torch.rand(3, 30, 45, generator=generator)
    value = {'pixels': pixels, 'question': 'Describe the image.', 'source_id': 'fixture:smol'}
    with torch.no_grad():
        tool.language.gate.fill_(-.2)
        records['scene'] = {'pixels': tensor_json(pixels), 'prepared': _receipt_record(tool, value),
                            'receipt': tool.interpret(value), 'loss': tool.loss(value, 'A small image.').item(),
                            'configuration': tool.configuration(), 'bindings': _bindings(tool)}
    tool.save_pretrained(_fresh('smol'))
    return records


def foundations():
    """``Scene.from_language_foundation`` configurations (processor asset hashes, configs) for local and Hub sources."""
    from tensorcode.tools.scene import Scene
    records = {}
    torch.manual_seed(5)
    local = _fresh('tiny_foundation')
    processor = tiny_processor()
    language = Idefics3Config(vision_config={'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2,
                                             'image_size': 8, 'patch_size': 2},
                              text_config={'model_type': 'llama', 'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1,
                                           'num_attention_heads': 2, 'vocab_size': 16, 'pad_token_id': 0, 'rope_theta': 10000},
                              pad_token_id=0, scale_factor=2, image_token_id=10)
    model = Idefics3ForConditionalGeneration(language)
    model.generation_config = GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model.save_pretrained(local)
    processor.save_pretrained(local)
    relative = 'test/fixtures/scene_language/tiny_foundation'  # the TypeScript test passes this repo-relative path
    import os
    cwd = os.getcwd()
    os.chdir(OUT.parent.parent)
    try:
        tool = Scene.from_language_foundation(relative, revision=None, local_files_only=True)
    finally:
        os.chdir(cwd)
    records['local'] = {'configuration': tool.configuration(), 'assets': tool.language.assets}
    # Named chat templates (Python ``test_named_processor_templates_preserved``).
    named = _fresh('tiny_foundation_named')
    model.save_pretrained(named)
    processor.chat_template = {'default': processor.chat_template, 'alternative': processor.chat_template + ' alternative'}
    processor.save_pretrained(named)
    os.chdir(OUT.parent.parent)
    try:
        tool = Scene.from_language_foundation('test/fixtures/scene_language/tiny_foundation_named', revision='pinned',
                                              local_files_only=True)
    finally:
        os.chdir(cwd)
    records['local_named'] = {'configuration': tool.configuration(), 'assets': tool.language.assets}
    snapshot = _snapshot()
    if snapshot is None:
        print(f'skipping uncached {SMOLVLM}')
        return records
    for flag in (True, False):
        hub = Scene.from_language_foundation(SMOLVLM, revision=snapshot.name, local_files_only=flag)
        records[f'hub_{str(flag).lower()}'] = {'configuration': hub.configuration(), 'snapshot': snapshot.name}
    return records


def generate():
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json('scene_language/records.json', {'tiny': tiny(), 'smol': smol_processor_scene(), 'foundations': foundations()})
