"""Tokenizer parity fixtures: synthetic tokenizers (committed) and cached real ones (by hash)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers, Regex
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from generate import OUT, write_json
from tensorcode._internal.vec.text import _tokenizer_config

CORPUS = [
    "The service recovered after reconnecting the database.",
    "Héllo wörld! Naïve café owners don't pay 3.50 dollars.",
    "Which document describes the sky? The sky appears blue during the day.",
    "A kettle boils water; tea is brewed at 90 degrees.",
    "日本語のテキストと中文字符 mixed with English words.",
    "Timeouts, retries and back-off policies matter for resilient systems.",
    "I'm sure we've seen they're right, it's fine.",
] * 20

TEXTS = [
    "The service recovered after reconnecting the database.",
    "  Multiple   spaces\tand\nnewlines  ",
    "Héllo WÖRLD, naïve café!",
    "中文 and 日本語 text",
    "I'm sure we've seen it's they're fine...",
    "unknownwordzzq 12345 3.14",
    "emoji 🙂 test",
    "",
]


def canonical(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend['padding'] = None
    backend['truncation'] = None
    return json.dumps(backend, sort_keys=True, separators=(',', ':'))


def cases(tokenizer, texts, *, special_inline, pairs=False):
    out = {'config': {k: v for k, v in _tokenizer_config(tokenizer).items() if k != 'json'},
           'json_sha256': hashlib.sha256(_tokenizer_config(tokenizer)['json'].encode()).hexdigest(),
           'encodings': []}
    for text in texts + [special_inline]:
        ids = tokenizer(text)['input_ids']
        out['encodings'].append({
            'text': text, 'ids': ids, 'tokens': tokenizer.convert_ids_to_tokens(ids),
            'no_special': tokenizer(text, add_special_tokens=False)['input_ids'],
            'decoded': tokenizer.decode(ids), 'decoded_skip': tokenizer.decode(ids, skip_special_tokens=True),
        })
    batch = tokenizer(texts[:3], padding=True, truncation=True, max_length=9)
    out['batch'] = {k: v for k, v in batch.items()}
    if pairs:
        pair = tokenizer(texts[:2], texts[2:4], padding=True, truncation=True, max_length=12)
        out['pairs'] = {k: v for k, v in pair.items()}
    return out


def train(model, trainer, normalizer, pre_tokenizer, post_processor, decoder):
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoder
    tokenizer.train_from_iterator(CORPUS, trainer=trainer)
    if post_processor is not None:
        tokenizer.post_processor = post_processor(tokenizer)
    return tokenizer


def synthetic():
    specs = {}
    wordpiece = train(
        models.WordPiece(unk_token='[UNK]'),
        trainers.WordPieceTrainer(vocab_size=180, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']),
        normalizers.BertNormalizer(lowercase=True), pre_tokenizers.BertPreTokenizer(),
        lambda t: processors.TemplateProcessing(single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
                                                special_tokens=[('[CLS]', t.token_to_id('[CLS]')), ('[SEP]', t.token_to_id('[SEP]'))]),
        decoders.WordPiece())
    specs['wordpiece'] = (wordpiece, dict(cls_token='[CLS]', sep_token='[SEP]', pad_token='[PAD]', unk_token='[UNK]', mask_token='[MASK]'), 'hello [SEP] world', True)
    bytelevel = train(
        models.BPE(), trainers.BpeTrainer(vocab_size=300, special_tokens=['<s>', '<pad>', '</s>', '<unk>'],
                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
        None, pre_tokenizers.ByteLevel(add_prefix_space=False),
        lambda t: processors.RobertaProcessing(('</s>', t.token_to_id('</s>')), ('<s>', t.token_to_id('<s>'))),
        decoders.ByteLevel())
    specs['bytelevel_bpe'] = (bytelevel, dict(bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>'), 'hello</s> world', False)
    unigram = train(
        models.Unigram(), trainers.UnigramTrainer(vocab_size=150, special_tokens=['<pad>', '</s>', '<unk>'], unk_token='<unk>'),
        normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(Regex(' {2,}'), ' ')]),
        pre_tokenizers.Metaspace(replacement='▁', prepend_scheme='always'),
        lambda t: processors.TemplateProcessing(single='$A </s>', pair='$A </s> $B </s>', special_tokens=[('</s>', t.token_to_id('</s>'))]),
        decoders.Metaspace(replacement='▁', prepend_scheme='always'))
    specs['unigram'] = (unigram, dict(eos_token='</s>', pad_token='<pad>', unk_token='<unk>'), 'hello </s> world', False)
    clip_like = train(
        models.BPE(end_of_word_suffix='</w>', unk_token='<|endoftext|>'),
        trainers.BpeTrainer(vocab_size=300, special_tokens=['<|startoftext|>', '<|endoftext|>'], end_of_word_suffix='</w>'),
        normalizers.Sequence([normalizers.NFC(), normalizers.Replace(Regex(r'\s+'), ' '), normalizers.Lowercase()]),
        pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""), behavior='removed', invert=True),
                                 pre_tokenizers.ByteLevel(add_prefix_space=False)]),
        lambda t: processors.RobertaProcessing(('<|endoftext|>', t.token_to_id('<|endoftext|>')), ('<|startoftext|>', t.token_to_id('<|startoftext|>'))),
        decoders.ByteLevel())
    specs['clip_bpe'] = (clip_like, dict(bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|endoftext|>', unk_token='<|endoftext|>'), 'hello <|endoftext|> world', False)
    result = {}
    for name, (backend, special, inline, pairs) in specs.items():
        wrapper = PreTrainedTokenizerFast(tokenizer_object=backend, **special)
        record = cases(wrapper, TEXTS, special_inline=inline, pairs=pairs)
        record['tokenizer_json'] = wrapper.backend_tokenizer.to_str()
        record['special_tokens'] = special
        record['canonical_json'] = canonical(wrapper)
        result[name] = record
    write_json('tokenizers_synthetic.json', result)


def cached():
    repos = {'google/electra-small-discriminator': ('hello [SEP] world', True),
             'google/flan-t5-small': ('Translate </s> to German: <extra_id_0> house', False),
             'openai/clip-vit-base-patch32': ('a photo of a <|endoftext|> cat', False)}
    from huggingface_hub.constants import HF_HUB_CACHE
    result = {}
    for repo, (inline, pairs) in repos.items():
        root = Path(HF_HUB_CACHE) / ('models--' + repo.replace('/', '--')) / 'snapshots'
        snapshots = sorted(p for p in root.glob('*') if (p / 'tokenizer.json').exists()) if root.exists() else []
        if not snapshots:
            print(f'skipping uncached {repo}')
            continue
        snapshot = snapshots[0]
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=snapshot.name, local_files_only=True, use_fast=True)
        record = cases(tokenizer, TEXTS, special_inline=inline, pairs=pairs)
        record['tokenizer_class'] = type(tokenizer).__name__
        record['snapshot'] = snapshot.name
        record['python_canonical_equals_file_canonical'] = canonical(tokenizer) == json.dumps(
            {**json.loads((snapshot / 'tokenizer.json').read_text()), 'padding': None, 'truncation': None}, sort_keys=True, separators=(',', ':'))
        result[repo] = record
    write_json('tokenizers_cached.json', result)


def generate():
    synthetic()
    cached()
