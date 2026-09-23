"""Tool configuration field validation and foundation rank encoder identities.

Python 0.4.0a4 rejects unknown or obsolete tool configuration fields with a
message naming each field and listing the valid ones, and describes a
foundation-backed ranking encoder by its native configuration and tensor
schemas (``FoundationTransform``). ``test/tools/toolConfigurationFields.test.ts``
checks the TypeScript messages and operation configurations/fingerprints
against these outputs.
"""
from __future__ import annotations

import copy
import json

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from generate import OUT
from tensorcode._internal.response_quality import ResponseQualityAssessor
from tensorcode._internal.retrieval import RetrievalEncoder
from tensorcode._internal.training.persistence import configuration as op_configuration, fingerprint as op_fingerprint
from tensorcode.tools.chatbot import Chatbot
from tensorcode.tools.decision import Decision
from tensorcode.tools.investigator import Investigator
from tensorcode.tools.planner import Planner
from tensorcode.tools.scene import Scene


def tools_json():
    return json.loads((OUT / 'tools' / 'tools.json').read_text())


def foundation_config(model_type):
    """Python ``test_response_quality.tiny_config(model_type)``."""
    tokenizer = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
                                     'question': 4, 'evidence': 5, 'candidate': 6, 'yes': 7}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
                                                  special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    return {'foundation_config': {'model_type': model_type, 'vocab_size': 8, 'hidden_size': 8,
                                  'embedding_size': 8, 'num_hidden_layers': 1, 'num_attention_heads': 2,
                                  'intermediate_size': 16, 'max_position_embeddings': 128,
                                  'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.},
            'tokenizer_json': tokenizer.to_str(),
            'tokenizer_special_tokens': {'pad_token': '[PAD]', 'unk_token': '[UNK]',
                                         'cls_token': '[CLS]', 'sep_token': '[SEP]'},
            'max_tokens': 128}


def message(tool, config):
    try:
        tool(config)
    except ValueError as error:
        return str(error)
    raise AssertionError(f'{tool.__name__} accepted {sorted(config)}')


def generate():
    tools = tools_json()
    ranking = {'vocabulary': ['hello', 'world'], 'dimensions': 8, 'slots': 2, 'steps': 1}
    scene = {'vocabulary': ['object', 'left'], 'dimensions': 8, 'slots': 3}
    quality = foundation_config('bert')
    cases = {
        'Chatbot': (Chatbot, tools['chatbot']['config']),
        'Investigator': (Investigator, tools['investigator']['config']),
        'Decision': (Decision, tools['investigator']['config']),
        'Planner': (Planner, ranking),
        'Scene': (Scene, scene),
        'RetrievalEncoder': (RetrievalEncoder, tools['retrieval']['config']),
        'ResponseQualityAssessor': (ResponseQualityAssessor, quality),
    }
    messages = {name: message(tool, dict(copy.deepcopy(config), colour='blue', obsolete_head=1, **{"it's": 2}))
                for name, (tool, config) in cases.items()}
    cognition = copy.deepcopy(tools['chatbot']['config'])
    cognition['cognition'] = {'investigator': copy.deepcopy(tools['investigator']['config']), 'proposal_limit': 3}
    messages['Chatbot cognition'] = message(Chatbot, cognition)
    messages['Scene language'] = message(Scene, {'mode': 'language', 'vocabulary': ['object']})

    encoders = {}
    for model_type in ('bert', 'electra'):
        native = foundation_config(model_type)
        config = {key: native[key] for key in ('foundation_config', 'tokenizer_json', 'tokenizer_special_tokens')}
        for tool in (Planner, Investigator, Decision):
            torch.manual_seed(0)
            model = tool(copy.deepcopy(config))
            encoders[f'{tool.__name__}/{model_type}'] = {
                'encode': model.rank.encode.configuration(),
                'fingerprints': {name: op_fingerprint(op_configuration(operation))
                                 for name, operation in model.operation_bindings().items()},
            }
    payload = {'messages': messages, 'quality_config': quality, 'ranking_config': ranking,
               'scene_config': scene, 'foundation_configs': {t: foundation_config(t) for t in ('bert', 'electra')},
               'encoders': encoders}
    (OUT / 'tool_fields.json').write_text(json.dumps(payload, separators=(',', ':')) + '\n')
