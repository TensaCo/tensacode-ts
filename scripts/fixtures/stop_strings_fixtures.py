"""``StopStringCriteria`` (byte-level, byte-fallback and plain token strings), ``stop_strings``
and ``token_healing`` in ``generate`` with a tokenizer (``causalGeneration.ts``)."""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.generation.stopping_criteria import StopStringCriteria
from transformers.utils import logging

from generate import OUT, write_json

ROOT = OUT / 'scene_language'


def _criteria(tokenizer, stops, texts):
    try:
        criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=stops)
    except Exception as exc:  # noqa: BLE001 - errors are part of the fixture
        return {'error': type(exc).__name__, 'message': str(exc)}
    rows = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        prefixes = [ids[:end] for end in range(1, len(ids) + 1)]
        rows.append({'ids': ids, 'done': [bool(criteria(torch.tensor([prefix]), None)[0]) for prefix in prefixes]})
    return {'rows': rows}


def generate():
    logging.set_verbosity_error()
    records = {}
    smol = PreTrainedTokenizerFast.from_pretrained(ROOT / 'smol' / 'processor')
    texts = ['the cat sat', 'at the end. Done', 'concatenate', 'hello world', 'Émile et café.']
    records['byte_level'] = _criteria(smol, ['cat', ' the', 'end.', 'é.'], texts)
    records['byte_level_single'] = _criteria(smol, 'world', texts)
    tiny = PreTrainedTokenizerFast.from_pretrained(ROOT / 'tiny' / 'processor')
    records['word_level'] = _criteria(tiny, ['left'], ['describe left object'])
    llama = AutoTokenizer.from_pretrained(ROOT / 'tokenizer_classes' / 'llama')
    records['byte_fallback'] = _criteria(llama, ['llo', 'bé'], ['hello abc', 'ab hello', 'hé bé'])
    t5 = AutoTokenizer.from_pretrained(ROOT / 'tokenizer_classes' / 't5_fast')
    records['metaspace'] = _criteria(t5, ['a b', 'hello'], ['a b c', 'hello a'])
    # generate() with stop strings and token healing on the small SmolVLM-shaped Scene model (text only).
    from tensorcode.tools.scene import Scene
    tool = Scene.from_pretrained(ROOT / 'smol').eval()
    model = tool.language.model
    prompt = smol('<|im_start|>User: tell me about the', add_special_tokens=False, return_tensors='pt')['input_ids']
    runs = {}
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        free = model.generate(input_ids=prompt, max_new_tokens=8, do_sample=False)
        text = smol.decode(free[0, prompt.shape[1]:])
        runs['free'] = {'sequences': free.tolist(), 'text': text}
        # Stop at a string that the free generation produces.
        stop = text[len(text) // 2: len(text) // 2 + 3] or text
        runs['stop'] = {'stop': stop, 'sequences': model.generate(input_ids=prompt, max_new_tokens=8, do_sample=False,
                                                                   stop_strings=[stop], tokenizer=smol).tolist()}
        healed = smol('<|im_start|>User: the quick bro', add_special_tokens=False, return_tensors='pt')['input_ids']
        runs['healing'] = {'prompt': healed.tolist(), 'sequences': model.generate(input_ids=healed, max_new_tokens=3, do_sample=False,
                                                                                   token_healing=True, tokenizer=smol,
                                                                                   pad_token_id=smol.pad_token_id).tolist()}
    records['generate'] = runs
    write_json('scene_language/stop_strings.json', records)
