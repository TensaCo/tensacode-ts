"""Artifacts whose parameters are not float32.

Loading converts registered parameters to the stored dtypes. Modules that
cache a parameter in a field must then read the converted parameter, so these
fixtures record Python outputs for float64 tools.
"""
from __future__ import annotations

import json
import shutil

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import T5Config

from generate import OUT


def generate():
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.tools.investigator import Investigator
    root = OUT / 'dtype'
    shutil.rmtree(root, ignore_errors=True)
    torch.manual_seed(0)
    investigator = Investigator({'vocabulary': ['a', 'b', 'c'], 'dimensions': 8, 'slots': 2, 'steps': 1}).double()
    with torch.no_grad():
        for parameter in investigator.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.3)
    investigator.save_pretrained(root / 'investigator_float64')
    inputs = {'question': 'a b', 'evidence': [{'source_id': 's', 'text': 'a c'}],
              'hypotheses': [{'id': 'x', 'text': 'a'}, {'id': 'y', 'text': 'b c'}]}
    receipt = investigator.eval()(inputs)

    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '</s>': 1, '<unk>': 2, 'user': 3, ':': 4, 'hello': 5,
                                            'world': 6, 'answer': 7}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    foundation = T5Config(vocab_size=8, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, dropout_rate=0.0)
    chatbot = Chatbot({'foundation_config': foundation.to_dict(), 'tokenizer_json': tokenizer.to_str(),
                       'tokenizer_special_tokens': {'pad_token': '<pad>', 'eos_token': '</s>', 'unk_token': '<unk>'},
                       'workspace': {'slots': 3, 'steps': 2}, 'max_new_tokens': 3, 'max_input_tokens': 32,
                       'max_turns': 2}).double()
    with torch.no_grad():
        for parameter in chatbot.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    chatbot.save_pretrained(root / 'chatbot_float64')
    chatbot.eval()
    with torch.no_grad():
        loss = chatbot.loss_batch(['hello world'], ['answer'])
        encoded = chatbot.foundation.get_encoder()(input_ids=torch.tensor([[5, 6, 1]])).last_hidden_state
    (root / 'expected.json').write_text(json.dumps({
        'investigator': {'inputs': inputs, 'scores': [c['predicted_score'] for c in receipt['candidates']],
                         'probabilities': [c['probability'] for c in receipt['candidates']]},
        'chatbot': {'loss': float(loss), 'generation': chatbot.generate_batch(['hello world', 'hello']),
                    'encoder_ids': [[5, 6, 1]], 'encoder_states': encoded.reshape(-1).tolist()},
    }) + '\n')
