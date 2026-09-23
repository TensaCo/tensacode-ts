"""Scene language ``interpret`` under generation configurations (beam search, guidance,
sequence bias, errors) for the tiny and SmolVLM-processor Scene fixtures."""
from __future__ import annotations

import copy
import json
import warnings

import torch
from transformers import GenerationConfig
from transformers.utils import logging

from generate import OUT, write_json

ROOT = OUT / 'scene_language'

TINY_CASES = [
    {'name': 'beams3', 'generation': {'num_beams': 3}},
    {'name': 'beams3_return', 'generation': {'num_beams': 3, 'num_return_sequences': 2, 'length_penalty': 0.5}},
    {'name': 'beams2_early', 'generation': {'num_beams': 2, 'early_stopping': True, 'length_penalty': 2.0}},
    {'name': 'beams2_never', 'generation': {'num_beams': 2, 'early_stopping': 'never', 'length_penalty': -1.0}},
    {'name': 'guidance', 'generation': {'guidance_scale': 2.0}},
    {'name': 'guidance_beams', 'generation': {'guidance_scale': 1.5, 'num_beams': 3}},
    {'name': 'sequence_bias', 'generation': {'sequence_bias': [[[4], 5.0], [[4, 6], 3.0]]}},
    {'name': 'repetition', 'generation': {'repetition_penalty': 1.8, 'no_repeat_ngram_size': 2}},
    {'name': 'sampling_ignored', 'generation': {'do_sample': True, 'temperature': 0.3, 'top_k': 2}},
    {'name': 'prompt_lookup', 'generation': {'prompt_lookup_num_tokens': 2}},
    {'name': 'watermark', 'generation': {'watermarking_config': {'greenlist_ratio': 0.5, 'bias': 3.0, 'hashing_key': 7, 'seeding_scheme': 'lefthash', 'context_width': 1}}},
    {'name': 'err_stop_strings', 'generation': {'stop_strings': ['left']}},
    {'name': 'err_token_healing', 'generation': {'token_healing': True}},
    {'name': 'err_return_greedy', 'generation': {'num_return_sequences': 2}},
    {'name': 'err_contrastive', 'generation': {'penalty_alpha': 0.5, 'top_k': 4}},
    {'name': 'err_low_memory', 'generation': {'num_beams': 2, 'low_memory': True}},
]
SMOL_CASES = [
    {'name': 'beams2', 'generation': {'num_beams': 2}},
    {'name': 'beams3_return', 'generation': {'num_beams': 3, 'num_return_sequences': 3}},
    {'name': 'guidance', 'generation': {'guidance_scale': 1.7}},
]
CONSTRUCTION_ERRORS = [
    {'name': 'early_stopping', 'generation': {'early_stopping': 'soon'}},
    {'name': 'return_sequences', 'generation': {'num_return_sequences': 4, 'num_beams': 2}},
    {'name': 'watermark', 'generation': {'watermarking_config': {'seeding_scheme': 'other'}}},
]


def _pixels(record):
    return torch.tensor(record['data'], dtype=torch.float32).reshape(record['shape'])


def _interpret(tool, base, value, case, max_new_tokens):
    generation = copy.deepcopy(base)
    for key, item in case['generation'].items():
        setattr(generation, key, item)
    tool.language.model.generation_config = generation
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            receipt = tool.interpret(value, max_new_tokens=max_new_tokens)
        return {'name': case['name'], 'receipt': receipt}
    except Exception as exc:  # noqa: BLE001 - errors are part of the fixture
        return {'name': case['name'], 'error': type(exc).__name__, 'message': str(exc)}


def generate():
    from tensorcode.tools.scene import Scene
    logging.set_verbosity_error()
    records = json.loads((ROOT / 'records.json').read_text())
    result = {'tiny': [], 'smol': [], 'construction': []}
    tiny = Scene.from_pretrained(ROOT / 'tiny').eval()
    with torch.no_grad():
        tiny.language.gate.fill_(.3)
    base = copy.deepcopy(tiny.language.model.generation_config)
    value = {'pixels': _pixels(records['tiny']['pixels']), 'question': 'describe object', 'source_id': 'fixture:image'}
    for case in TINY_CASES:
        result['tiny'].append(_interpret(tiny, base, value, case, 8))
    if (ROOT / 'smol').exists() and records.get('smol'):
        smol = Scene.from_pretrained(ROOT / 'smol').eval()
        with torch.no_grad():
            smol.language.gate.fill_(-.2)
        base = copy.deepcopy(smol.language.model.generation_config)
        value = {'pixels': _pixels(records['smol']['scene']['pixels']), 'question': 'Describe the image.', 'source_id': 'fixture:smol'}
        for case in SMOL_CASES:
            result['smol'].append(_interpret(smol, base, value, case, 6))
    config = json.loads((ROOT / 'tiny' / 'tensorcode_config.json').read_text())['config']
    assets = {name: (ROOT / 'tiny' / 'processor' / name).read_text(encoding='utf-8') for name in config['processor_hashes']}
    for case in CONSTRUCTION_ERRORS:
        generation = GenerationConfig.from_dict(config['generation_config']).to_dict()
        generation.update(case['generation'])
        try:
            Scene(dict(config, generation_config=generation, _language_assets=assets))
            result['construction'].append({'name': case['name'], 'generation': case['generation']})
        except Exception as exc:  # noqa: BLE001
            result['construction'].append({'name': case['name'], 'generation': case['generation'], 'error': type(exc).__name__, 'message': str(exc)})
    result['tiny_cases'] = TINY_CASES
    result['smol_cases'] = SMOL_CASES
    write_json('scene_language/generation.json', result)
