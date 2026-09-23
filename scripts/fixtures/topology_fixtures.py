"""Module topologies of native architectures and a native-backed tool checkpoint.

Checkpoint module modes are keyed by ``named_modules()``, so the TypeScript
native models must register exactly the modules transformers registers
(including parameter-free activations).
"""
from __future__ import annotations

import json
import shutil

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          T5Config, T5EncoderModel)

from generate import OUT, write_json

SMALL = dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=50,
             max_position_embeddings=32)
T5 = dict(model_type='t5', vocab_size=20, d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8)
CASES = {
    'bert': ('base', dict(model_type='bert', **SMALL)),
    'roberta': ('base', dict(model_type='roberta', **SMALL)),
    'electra': ('base', dict(model_type='electra', embedding_size=8, **SMALL)),
    'distilbert': ('base', dict(model_type='distilbert', dim=16, n_layers=1, n_heads=2, hidden_dim=32, vocab_size=50,
                                max_position_embeddings=32)),
    'deberta-v2': ('base', dict(model_type='deberta-v2', relative_attention=True, pos_att_type=['p2c', 'c2p'], **SMALL)),
    'vit': ('base', dict(model_type='vit', hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, image_size=8, patch_size=4)),
    't5': ('seq2seq', T5),
    't5-gated': ('seq2seq', dict(T5, feed_forward_proj='gated-gelu')),
    't5-encoder': ('encoder', T5),
    'clip': ('base', dict(model_type='clip', projection_dim=8,
                          text_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                                           vocab_size=50, max_position_embeddings=16),
                          vision_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                             intermediate_size=32, image_size=8, patch_size=4))),
}
for _name in ['bert', 'roberta', 'electra', 'distilbert', 'deberta-v2']:
    CASES[f'{_name}-classifier'] = ('sequence-classification', CASES[_name][1])


def build(head, config):
    if head == 'encoder':
        return T5EncoderModel(config)
    if head == 'base' and config.model_type == 'vit':
        return AutoModel.from_config(config, add_pooling_layer=False)  # tensorcode's ViT usage
    factory = {'base': AutoModel, 'sequence-classification': AutoModelForSequenceClassification,
               'seq2seq': AutoModelForSeq2SeqLM}[head]
    return factory.from_config(config)


def chatbot_checkpoint():
    from tensorcode import training
    from tensorcode.tools.chatbot import Chatbot
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '</s>': 1, '<unk>': 2, 'user': 3, ':': 4, 'hello': 5,
                                            'world': 6, 'answer': 7}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    foundation = T5Config(vocab_size=8, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, dropout_rate=0.0)
    torch.manual_seed(0)
    model = Chatbot({'foundation_config': foundation.to_dict(), 'tokenizer_json': tokenizer.to_str(),
                     'tokenizer_special_tokens': {'pad_token': '<pad>', 'eos_token': '</s>', 'unk_token': '<unk>'},
                     'workspace': {'slots': 3, 'steps': 2}, 'max_new_tokens': 3, 'max_input_tokens': 32,
                     'max_turns': 2})
    root = OUT / 'training' / 'python_chatbot_resume'
    shutil.rmtree(root, ignore_errors=True)
    model.save_pretrained(root / 'initial')
    trainer = training.Trainer.from_tool(model, lr=0.05)
    experience = trainer.capture(['hello world'], ['answer'], source='fixture:review')
    trainer.fit([experience], epochs=2)
    trainer.save_checkpoint(root / 'checkpoint', progress={'next_example': 1})
    model.save_pretrained(root / 'trained')
    (root / 'expected.json').write_text(json.dumps({'steps': trainer.steps, 'progress': {'next_example': 1},
                                                    'generation': model.generate_batch(['hello world'])}) + '\n')


def generate():
    cases = {}
    for name, (head, data) in CASES.items():
        config = AutoConfig.for_model(**data)
        model = build(head, config)
        cases[name] = {'head': head, 'config': json.loads(json.dumps(config.to_dict())),
                       'modules': [path for path, _ in model.named_modules()],
                       'parameters': list(model.state_dict().keys())}
    write_json('native_modules.json', cases)
    chatbot_checkpoint()
