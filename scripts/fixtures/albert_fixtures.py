"""ALBERT parity: shared layer groups, factorized embeddings and the classifier head."""
from __future__ import annotations

import json
import shutil

import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (AlbertConfig, AlbertForSequenceClassification, AlbertModel, AutoConfig,
                          PreTrainedTokenizerFast)

import os
from pathlib import Path

from generate import OUT, tensor_json, write_json

ROOT = Path(__file__).resolve().parents[2]
FOUNDATION = 'test/fixtures/vec/albert_foundation'  # repo-relative, as the TypeScript test passes it

CONFIG = dict(model_type='albert', vocab_size=12, embedding_size=4, hidden_size=8, intermediate_size=16,
              num_hidden_layers=3, num_hidden_groups=2, inner_group_num=2, num_attention_heads=2,
              max_position_embeddings=16, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)


def tokenizer():
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, 'hello': 2, 'world': 3, 'prefix': 4, 'answer': 5},
                                         unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')


def generate():
    ids = torch.tensor([[3, 7, 11, 5, 9, 2], [4, 8, 6, 2, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    types = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0]])
    torch.manual_seed(300)
    config = AutoConfig.for_model(**CONFIG)
    model = AlbertModel(config).eval()
    save_model(model, str(OUT / 'native_albert.safetensors'))
    record = {'config': json.loads(config.to_json_string()), 'state_keys': list(model.state_dict().keys()),
              'input_ids': ids.tolist(), 'attention_mask': mask.tolist(), 'token_type_ids': types.tolist()}
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, token_type_ids=types)
        record['last_hidden_state'] = tensor_json(out.last_hidden_state)
        record['pooler_output'] = tensor_json(out.pooler_output)
        embeds = torch.linspace(-1, 1, 2 * 5 * 4).reshape(2, 5, 4)
        embeds_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        record['inputs_embeds'] = tensor_json(embeds)
        record['inputs_embeds_mask'] = embeds_mask.tolist()
        record['inputs_embeds_output'] = tensor_json(model(inputs_embeds=embeds, attention_mask=embeds_mask).last_hidden_state)
    torch.manual_seed(301)
    classifier_config = AutoConfig.for_model(**{**CONFIG, 'num_labels': 3})
    classifier = AlbertForSequenceClassification(classifier_config).eval()
    save_model(classifier, str(OUT / 'native_albert_classifier.safetensors'))
    with torch.no_grad():
        record['classifier'] = {'config': json.loads(classifier_config.to_json_string()),
                                'state_keys': list(classifier.state_dict().keys()),
                                'logits': tensor_json(classifier(input_ids=ids, attention_mask=mask, token_type_ids=types).logits)}
    write_json('native_albert.json', record)

    # A foundation directory as in Python's tests/vec/test_pretrained_text.py.
    os.chdir(ROOT)
    foundation = FOUNDATION
    shutil.rmtree(foundation, ignore_errors=True)
    torch.manual_seed(302)
    AlbertModel(AlbertConfig(vocab_size=6, embedding_size=4, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                             num_hidden_groups=1, num_attention_heads=2, max_position_embeddings=8, hidden_dropout_prob=0.,
                             attention_probs_dropout_prob=0.)).save_pretrained(foundation)
    tokenizer().save_pretrained(foundation)
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.latent import Latent, Space
    space = Space('albert-input', 4, organization='sequence')
    encoder = TextEncoder.from_foundation(foundation, context_space=space)
    prefix = torch.linspace(-1, 1, 8).reshape(1, 2, 4)
    with torch.no_grad():
        output = encoder('hello world', context={'latents': [Latent(prefix, space, mask=torch.tensor([[True, False]]))]})
        plain = encoder(['hello world', 'hello'])
    write_json('vec/albert.json', {'configuration': encoder.configuration(), 'prefix': tensor_json(prefix),
                                   'output': tensor_json(output.tensor), 'plain': tensor_json(plain.tensor),
                                   'plain_mask': plain.mask.tolist()})
