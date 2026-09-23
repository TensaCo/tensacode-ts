"""Workspace, RankOperation and TensorAdapter configuration parity fixtures."""
from __future__ import annotations

import json

import torch
from safetensors.torch import save_model

from generate import OUT, tensor_json, write_json
from tensorcode._internal.ranking import RankOperation, normalize_config
from tensorcode._internal.vec.adapter import TensorAdapter
from tensorcode._internal.workspace import Workspace
from tensorcode._internal.training.persistence import configuration, fingerprint


def generate():
    torch.manual_seed(7)
    workspace = Workspace(4, slots=2, steps=2)
    save_model(workspace, str(OUT / 'workspace.safetensors'))
    encoded = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    with torch.no_grad():
        out = workspace(encoded, mask)
    config = normalize_config({'vocabulary': ['service', 'database', 'timeout', 'sky', 'blue'], 'dimensions': 8, 'slots': 2, 'steps': 2})
    torch.manual_seed(8)
    rank = RankOperation(config, task_key='question', candidates_key='hypotheses')
    save_model(rank, str(OUT / 'rank.safetensors'))
    inputs = {'question': 'Which document describes the sky?',
              'evidence': [{'source_id': 'report:1', 'text': 'The database timeout recovered.'}],
              'conversation_context': [{'role': 'user', 'text': 'We saw a blue sky.'}],
              'hypotheses': [{'id': 'a', 'text': 'The sky is blue.'}, {'id': 'b', 'text': 'A database timeout.'}]}
    with torch.no_grad():
        receipt = rank.receipt(inputs, probabilities=True)
    adapter = TensorAdapter(torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.GELU(), torch.nn.LayerNorm(2), torch.nn.Dropout(0.1)))
    write_json('ranking.json', {
        'workspace': {'state_keys': list(workspace.state_dict().keys()), 'encoded': tensor_json(encoded), 'mask': mask.tolist(),
                      'conditioning': tensor_json(out['conditioning']), 'attention': tensor_json(out['attention']),
                      'relations': tensor_json(out['relations']), 'configuration': workspace.configuration()},
        'rank': {'config': config, 'state_keys': list(rank.state_dict().keys()), 'inputs': inputs, 'receipt': receipt,
                 'configuration': rank.configuration()},
        'fingerprints': {
            'adapter': {'configuration': configuration(adapter), 'fingerprint': fingerprint(configuration(adapter))},
            'workspace_key': {'configuration': configuration(workspace.key), 'fingerprint': fingerprint(configuration(workspace.key))},
            'rank': {'configuration': configuration(rank), 'fingerprint': fingerprint(configuration(rank))},
        },
    })
