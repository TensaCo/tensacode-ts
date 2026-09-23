"""Processor assets ``Idefics3Processor.from_pretrained(dir).save_pretrained(...)`` writes for
foundations whose tokenizers are LlamaTokenizer(Fast), GPT2Tokenizer(Fast), T5Tokenizer(Fast),
AlbertTokenizer, DebertaV2Tokenizer or the generic TokenizersBackend (including unknown class
names), with the Idefics3 special tokens added to tokenizers that lack them."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import warnings
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import Idefics3Processor
from transformers.utils import logging

from generate import OUT, write_json

ROOT = OUT / 'scene_language' / 'tokenizer_classes'
PROCESSOR_CONFIG = {
    'image_processor': {'do_convert_rgb': True, 'do_image_splitting': False, 'do_normalize': True, 'do_pad': True, 'do_rescale': True,
                        'do_resize': False, 'image_mean': [0.5, 0.5, 0.5], 'image_processor_type': 'Idefics3ImageProcessor',
                        'image_std': [0.5, 0.5, 0.5], 'max_image_size': {'longest_edge': 8}, 'resample': 1,
                        'rescale_factor': 0.00392156862745098, 'size': {'longest_edge': 8}},
    'image_seq_len': 4, 'processor_class': 'Idefics3Processor',
}
TEMPLATE = "{% for message in messages %}{{ message['role'] }}: {{ message['content'][0]['text'] }}{% endfor %}"
HUB = {
    'flan_t5': 'google/flan-t5-small', 'albert': 'albert/albert-base-v2', 'deberta': 'cross-encoder/nli-deberta-v3-small',
    'smolvlm': 'HuggingFaceTB/SmolVLM-256M-Instruct',
}


def _snapshot(repo):
    from huggingface_hub.constants import HF_HUB_CACHE
    root = Path(HF_HUB_CACHE) / ('models--' + repo.replace('/', '--')) / 'snapshots'
    found = sorted(root.glob('*')) if root.exists() else []
    return found[0] if found else None


def _bpe_json(byte_fallback):
    vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}
    for index, piece in enumerate(['▁', 'a', 'b', 'c', 'd', 'e', 'h', 'l', 'o', '▁h', 'll', 'he', 'hell', 'hello', '▁hello', 'ab', 'abc']):
        vocab[piece] = 3 + index
    for byte in range(256):
        vocab[f'<0x{byte:02X}>'] = len(vocab)
    merges = [('▁', 'h'), ('l', 'l'), ('h', 'e'), ('he', 'll'), ('hell', 'o'), ('▁', 'hello'), ('a', 'b'), ('ab', 'c')]
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token='<unk>', byte_fallback=byte_fallback, fuse_unk=True))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Prepend('▁'), normalizers.Replace(' ', '▁')])
    tokenizer.decoder = decoders.Sequence([decoders.Replace('▁', ' '), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(' ', 1, 0)])
    return tokenizer.to_str()


def _gpt2_json():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {token: index for index, token in enumerate(['<|endoftext|>'] + sorted(alphabet))}
    for piece in ['Ġh', 'el', 'lo', 'Ġhel', 'Ġhello']:
        vocab[piece] = len(vocab)
    merges = [('Ġ', 'h'), ('e', 'l'), ('l', 'o'), ('Ġh', 'el'), ('Ġhel', 'lo')]
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer.to_str()


def _unigram_json():
    vocab = [('<pad>', 0.0), ('</s>', 0.0), ('<unk>', 0.0), ('▁', -2.0), ('▁a', -3.0), ('b', -3.5), ('▁hello', -4.25), ('c', -5.125)]
    vocab += [(f'<extra_id_{index}>', 0.0) for index in range(3, -1, -1)]
    tokenizer = Tokenizer(models.Unigram(vocab, unk_id=2))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    return tokenizer.to_str()


def _wordpiece_json():
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'hello': 5, 'abc': 6, 'a': 7, '##bc': 8}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token='[UNK]'))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer.to_str()


def _clip_json():
    vocab = {'<|startoftext|>': 0, '<|endoftext|>': 1, 'h': 2, 'e': 3, 'l': 4, 'o': 5, 'o</w>': 6, 'hello</w>': 7, 'a': 8, 'b': 9,
             'c</w>': 10, 'he': 11, 'll': 12, 'hell': 13}
    tokenizer = Tokenizer(models.BPE(vocab, [('h', 'e'), ('l', 'l'), ('he', 'll'), ('hell', 'o</w>')], end_of_word_suffix='</w>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer.to_str()


SYNTHETIC = {
    'llama': (lambda: _bpe_json(True), {'tokenizer_class': 'LlamaTokenizer', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>',
                                        'add_bos_token': True, 'legacy': False, 'model_max_length': 64, 'padding_side': 'left'}),
    'llama_fast_prefix': (lambda: _bpe_json(True), {'tokenizer_class': 'LlamaTokenizerFast', 'add_prefix_space': False, 'add_eos_token': True,
                                                    'legacy': True, 'clean_up_tokenization_spaces': True}),
    'llama_defaults': (lambda: _bpe_json(False), {'tokenizer_class': 'LlamaTokenizer'}),
    'gpt2_fast': (_gpt2_json, {'tokenizer_class': 'GPT2TokenizerFast', 'model_max_length': 1024, 'add_prefix_space': False}),
    't5_fast': (_unigram_json, {'tokenizer_class': 'T5TokenizerFast', 'extra_ids': 4, 'model_max_length': 512, 'sp_model_kwargs': {}}),
    'generic': (lambda: _bpe_json(True), {'tokenizer_class': 'PreTrainedTokenizerFast', 'bos_token': '<s>', 'eos_token': '</s>', 'truncation_side': 'left'}),
    'unknown_class': (lambda: _bpe_json(True), {'tokenizer_class': 'MysteryTokenizer', 'unk_token': '<unk>', 'custom_flag': 3}),
    'bert': (_wordpiece_json, {'tokenizer_class': 'BertTokenizer', 'do_lower_case': False, 'model_max_length': 128}),
    'roberta': (_gpt2_json, {'tokenizer_class': 'RobertaTokenizerFast', 'add_prefix_space': True, 'errors': 'strict'}),
    'clip': (_clip_json, {'tokenizer_class': 'CLIPTokenizer'}),
}
CACHED = {
    'flan_t5': ('flan_t5', ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'], None),
    'albert': ('albert', ['tokenizer.json', 'tokenizer_config.json', 'config.json'], None),
    'deberta': ('deberta', ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json'], None),
    'smol_llama': ('smolvlm', ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'], 'LlamaTokenizerFast'),
    'smol_gpt2_fast': ('smolvlm', ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'], 'GPT2TokenizerFast'),
}


def _save(directory):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        processor = Idefics3Processor.from_pretrained(directory, local_files_only=True)
    with tempfile.TemporaryDirectory() as out:
        processor.save_pretrained(out)
        assets = {p.relative_to(out).as_posix(): p.read_text(encoding='utf-8') for p in Path(out).rglob('*') if p.is_file()}
    ids = processor.tokenizer('hello <image> abc<end_of_utterance>', add_special_tokens=True)['input_ids']
    return assets, ids


def _common(directory):
    (directory / 'processor_config.json').write_text(json.dumps(PROCESSOR_CONFIG))
    (directory / 'chat_template.jinja').write_text(TEMPLATE)


def generate():
    logging.set_verbosity_error()
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    records = {'synthetic': {}, 'cached': {}}
    for name, (build, config) in SYNTHETIC.items():
        directory = ROOT / name
        directory.mkdir()
        (directory / 'tokenizer.json').write_text(build())
        (directory / 'tokenizer_config.json').write_text(json.dumps(config))
        _common(directory)
        assets, ids = _save(directory)
        records['synthetic'][name] = {'assets': assets, 'ids': ids}
    for name, (source, files, override) in CACHED.items():
        snapshot = _snapshot(HUB[source])
        if snapshot is None:
            print(f'skipping uncached {HUB[source]}')
            continue
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for file in files:
                if (snapshot / file).exists():
                    shutil.copy(snapshot / file, directory / file)
            if override:
                config = json.loads((directory / 'tokenizer_config.json').read_text())
                config['tokenizer_class'] = override
                (directory / 'tokenizer_config.json').write_text(json.dumps(config))
            _common(directory)
            assets, ids = _save(directory)
        records['cached'][name] = {
            'repo': HUB[source], 'snapshot': snapshot.name, 'files': files, 'override': override, 'ids': ids,
            'hashes': {key: hashlib.sha256(value.encode()).hexdigest() for key, value in assets.items()},
            'tokenizer_config': assets['tokenizer_config.json'],
        }
    write_json('scene_language/tokenizer_classes.json', records)
