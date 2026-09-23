"""``AutoTokenizer.from_pretrained`` over checkpoints with slow vocabulary files only.

BERT-family checkpoints with ``vocab.txt`` and RoBERTa/CLIP/GPT-2 checkpoints with
``vocab.json`` and ``merges.txt`` (no ``tokenizer.json``) load in Python, which
builds the class's backend from them. ``test/tokenizers/slowTokenizers.test.ts``
checks the TypeScript backend JSON, tokenizer configuration and encodings.
"""
from __future__ import annotations

import json
import shutil

from transformers import AutoTokenizer

from generate import OUT
from tensorcode._internal.vec.text import _tokenizer_config

ROOT = OUT / 'slow_tokenizers'
TEXTS = ['Hello WORLD naïve 中文 [MASK] hellos', 'a pair']
WORDS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'world', '##s', 'naive', 'naïve', 'Hello', 'WORLD', '中', '文',
         'a', 'pair', 'hello  ']
BYTES = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'Ġ': 4, 'h': 5, 'e': 6, 'l': 7, 'o': 8, 'Ġh': 9, 'll': 10, 'Ġhe': 11,
         'Ġhell': 12, 'Ġhello': 13, 'a': 14, 'Ġa': 15, '<mask>': 16}
MERGES = '#version: 0.2\nĠ h\nl l\nĠh e\nĠhe ll\nĠhell o\nĠ a\n'
CLIP = {'<|startoftext|>': 0, '<|endoftext|>': 1, 'h': 2, 'e': 3, 'l': 4, 'o': 5, 'o</w>': 6, 'hello</w>': 7, 'a</w>': 8, 'he': 9,
        'll': 10, 'hell': 11}
CLIP_MERGES = '#version: 0.2\nh e\nl l\nhe ll\nhell o</w>\n'


def cases():
    yield 'bert', {'vocab.txt': '\n'.join(WORDS) + '\n', 'tokenizer_config.json': json.dumps({'tokenizer_class': 'BertTokenizer'})}
    yield 'bert_cased', {'vocab.txt': '\n'.join(WORDS) + '\n', 'config.json': json.dumps({'model_type': 'electra'}),
                         'tokenizer_config.json': json.dumps({'do_lower_case': False, 'model_max_length': 32}),
                         'special_tokens_map.json': json.dumps({'additional_special_tokens': ['<note>']}),
                         'added_tokens.json': json.dumps({'<note>': 17})}
    yield 'roberta', {'vocab.json': json.dumps(BYTES), 'merges.txt': MERGES,
                      'tokenizer_config.json': json.dumps({'tokenizer_class': 'RobertaTokenizer', 'add_prefix_space': True})}
    yield 'clip', {'vocab.json': json.dumps(CLIP), 'merges.txt': CLIP_MERGES, 'config.json': json.dumps({'model_type': 'clip'})}
    gpt2 = {key: value for key, value in BYTES.items() if key not in ('<s>', '<pad>', '</s>', '<unk>', '<mask>')}
    gpt2 = {'<|endoftext|>': 0, **{key: index + 1 for index, key in enumerate(gpt2)}}
    yield 'gpt2', {'vocab.json': json.dumps(gpt2), 'merges.txt': MERGES, 'tokenizer_config.json': json.dumps({'tokenizer_class': 'GPT2Tokenizer'})}
    yield 'gpt2_bos', {'vocab.json': json.dumps(gpt2), 'merges.txt': MERGES,
                       'tokenizer_config.json': json.dumps({'tokenizer_class': 'GPT2TokenizerFast', 'add_bos_token': True, 'add_eos_token': True})}


def generate():
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    records = {'texts': TEXTS, 'cases': {}}
    for name, files in cases():
        directory = ROOT / name
        directory.mkdir()
        for file, text in files.items():
            (directory / file).write_text(text, encoding='utf-8')
        tokenizer = AutoTokenizer.from_pretrained(directory)
        records['cases'][name] = {
            'class': type(tokenizer).__name__, 'to_str': tokenizer.backend_tokenizer.to_str(), 'config': _tokenizer_config(tokenizer),
            'single': tokenizer(TEXTS[0])['input_ids'], 'pair': tokenizer(TEXTS[0], TEXTS[1])['input_ids'],
        }
    (ROOT / 'records.json').write_text(json.dumps(records, indent=1) + '\n')
