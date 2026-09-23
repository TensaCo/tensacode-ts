"""Tokenizers ``AutoTokenizer`` builds for checkpoints without tokenizer files.

``test/tokenizers/blankTokenizers.test.ts`` checks that TypeScript builds the
same backend, special tokens, TensorCode tokenizer configuration and ids.
"""
from __future__ import annotations

import json
import tempfile

from transformers import AutoConfig, AutoTokenizer

from generate import OUT
from tensorcode._internal.vec.text import _tokenizer_config

TYPES = ['albert', 'bert', 'clip', 'deberta-v2', 'distilbert', 'electra', 'roberta', 't5']


def generate():
    records = {}
    for model_type in TYPES:
        with tempfile.TemporaryDirectory() as directory:
            config = AutoConfig.for_model(model_type)
            config.save_pretrained(directory)
            config_text = open(f'{directory}/config.json').read()
            tokenizer = AutoTokenizer.from_pretrained(directory)
        records[model_type] = {
            'config_json': config_text, 'class': type(tokenizer).__name__,
            'to_str': tokenizer.backend_tokenizer.to_str(),
            'special_tokens_map': {k: str(v) for k, v in tokenizer.special_tokens_map.items() if isinstance(v, str)},
            'tokenizer_config': _tokenizer_config(tokenizer),
            'ids': tokenizer('hello world [UNK] </s>')['input_ids'],
        }
    (OUT / 'blank_tokenizers.json').write_text(json.dumps(records, indent=1) + '\n')
