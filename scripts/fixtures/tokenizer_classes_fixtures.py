"""``AutoTokenizer.from_pretrained`` over tokenizer directories whose files differ
from what the transformers 5 tokenizer class builds.

Classes with a custom ``__init__`` (BERT, RoBERTa, CLIP, T5, ...) rebuild their
backend from the ``tokenizer.json`` vocabulary and their construction flags,
and ``TokenizersBackend.__init__`` registers special tokens and the
``added_tokens_decoder`` of ``tokenizer_config.json``. The cases below make
every step visible; real cached tokenizers are included when present.
``test/tokenizers/tokenizerClasses.test.ts`` compares the TypeScript result
(``backend_tokenizer.to_str()``, TensorCode's tokenizer configuration and
encodings) with these.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import AutoTokenizer

from generate import OUT
from tensorcode._internal.vec.text import _tokenizer_config

ROOT = OUT / 'tokenizer_classes'
TEXTS = ['Hello WORLD naïve 中文 <mask> [MASK] hellos', 'a pair of words']
FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']


def wordpiece(vocab, *, specials=True, extra=False, processor=True):
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token='[UNK]', max_input_chars_per_word=50))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    if processor:
        tokenizer.post_processor = processors.BertProcessing(('[SEP]', vocab['[SEP]']), ('[CLS]', vocab['[CLS]']))
    tokenizer.decoder = decoders.WordPiece()
    if specials:
        tokenizer.add_special_tokens([token for token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] if token in vocab])
    if extra:
        tokenizer.add_tokens([AddedToken('<extra>', lstrip=True)])
    return tokenizer


def bpe(vocab, merges, *, suffix=''):
    tokenizer = Tokenizer(models.BPE(vocab, merges, **({'end_of_word_suffix': suffix} if suffix else {})))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def unigram(pieces):
    tokenizer = Tokenizer(models.Unigram(pieces, unk_id=2))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()
    return tokenizer


BERT_VOCAB = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4, 'hello': 5, 'world': 6, '##s': 7, 'naive': 8,
              'naïve': 9, 'Hello': 10, 'WORLD': 11, '中': 12, '文': 13, 'a': 14, 'pair': 15, 'of': 16, 'words': 17}
BYTE_VOCAB = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'Ġ': 4, 'h': 5, 'e': 6, 'l': 7, 'o': 8, 'Ġh': 9, 'll': 10, 'Ġhe': 11,
              'Ġhell': 12, 'Ġhello': 13, 'a': 14, 'Ġa': 15}
BYTE_MERGES = [('Ġ', 'h'), ('l', 'l'), ('Ġh', 'e'), ('Ġhe', 'll'), ('Ġhell', 'o'), ('Ġ', 'a')]
CLIP_VOCAB = {'<|startoftext|>': 0, '<|endoftext|>': 1, 'h': 2, 'e': 3, 'l': 4, 'o': 5, 'o</w>': 6, 'hello</w>': 7, 'a</w>': 8,
              'he': 9, 'll': 10, 'hell': 11}
CLIP_MERGES = [('h', 'e'), ('l', 'l'), ('he', 'll'), ('hell', 'o</w>')]


def cases():
    yield 'bert_plain', wordpiece(BERT_VOCAB, specials=False), {'tokenizer_class': 'BertTokenizer'}, None
    yield 'bert_flags', wordpiece(BERT_VOCAB), {
        'tokenizer_class': 'BertTokenizer', 'do_lower_case': False, 'strip_accents': True, 'tokenize_chinese_chars': False}, None
    yield 'bert_extra', wordpiece(BERT_VOCAB, extra=True, processor=False), {
        'tokenizer_class': 'BertTokenizerFast', 'added_tokens_decoder': {
            '4': {'content': '[MASK]', 'lstrip': True, 'normalized': False, 'rstrip': False, 'single_word': False, 'special': True}}}, None
    missing = {key: value for key, value in BERT_VOCAB.items() if key != '[MASK]'}
    missing = {key: index for index, key in enumerate(missing)}
    yield 'bert_missing_mask', wordpiece(missing), {'tokenizer_class': 'BertTokenizer', 'model_max_length': 64}, {'additional_special_tokens': ['<note>']}
    yield 'distilbert_model_type', wordpiece(BERT_VOCAB), {}, None
    yield 'roberta', bpe(BYTE_VOCAB, BYTE_MERGES), {'tokenizer_class': 'RobertaTokenizer', 'add_prefix_space': True, 'trim_offsets': False}, None
    no_mask = bpe(BYTE_VOCAB, BYTE_MERGES)
    yield 'roberta_no_mask', no_mask, {'tokenizer_class': 'RobertaTokenizerFast'}, {'mask_token': {
        'content': '<mask>', 'lstrip': True, 'normalized': False, 'rstrip': False, 'single_word': False}}
    yield 'clip', bpe(CLIP_VOCAB, CLIP_MERGES, suffix='</w>'), {'tokenizer_class': 'CLIPTokenizer'}, None
    t5 = unigram([('<pad>', 0.0), ('</s>', 0.0), ('<unk>', 0.0), ('▁', -2.0), ('▁hello', -3.0), ('▁a', -4.0), ('<extra_id_1>', 0.0),
                  ('<extra_id_0>', 0.0)])
    yield 't5', t5, {'tokenizer_class': 'T5Tokenizer', 'extra_ids': 2}, None
    generic = bpe(BYTE_VOCAB, BYTE_MERGES)
    yield 'generic', generic, {'tokenizer_class': 'PreTrainedTokenizerFast', 'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'}, None


def real():
    from huggingface_hub.constants import HF_HUB_CACHE
    for name, repo in {'electra': 'google/electra-small-discriminator', 'clip_hub': 'openai/clip-vit-base-patch32',
                       'flan_t5': 'google/flan-t5-small', 'deberta_v3': 'cross-encoder/nli-deberta-v3-small',
                       'albert': 'albert/albert-base-v2', 'smolvlm': 'HuggingFaceTB/SmolVLM-256M-Instruct'}.items():
        root = Path(HF_HUB_CACHE) / ('models--' + repo.replace('/', '--')) / 'snapshots'
        snapshots = sorted(root.glob('*')) if root.exists() else []
        if snapshots and (snapshots[0] / 'tokenizer.json').exists():
            yield name, repo, snapshots[0]


def record(directory, digest=False):
    tokenizer = AutoTokenizer.from_pretrained(directory)
    backend = tokenizer.backend_tokenizer.to_str()
    config = _tokenizer_config(tokenizer)
    if digest:  # real vocabularies are large: keep hashes of the JSON texts
        backend = hashlib.sha256(backend.encode()).hexdigest()
        config = {**config, 'json': hashlib.sha256(config['json'].encode()).hexdigest()}
    return {'class': type(tokenizer).__name__, 'to_str': backend, 'config': config,
            'single': tokenizer(TEXTS[0])['input_ids'], 'pair': tokenizer(TEXTS[0], TEXTS[1])['input_ids'],
            'decoded': tokenizer.decode(tokenizer(TEXTS[0])['input_ids'], skip_special_tokens=True)}


def generate():
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    records = {'texts': TEXTS, 'synthetic': {}, 'hub': {}}
    for name, tokenizer, config, special in cases():
        directory = ROOT / name
        directory.mkdir()
        tokenizer.save(str(directory / 'tokenizer.json'))
        (directory / 'tokenizer_config.json').write_text(json.dumps(config))
        if special is not None:
            (directory / 'special_tokens_map.json').write_text(json.dumps(special))
        if name == 'distilbert_model_type':
            (directory / 'config.json').write_text(json.dumps({'model_type': 'distilbert'}))
        records['synthetic'][name] = record(directory)
    for name, repo, snapshot in real():
        records['hub'][name] = {'repo': repo, 'snapshot': snapshot.name, **record(snapshot, digest=True)}
    (ROOT / 'records.json').write_text(json.dumps(records, indent=1) + '\n')
