"""PyTorch ``.bin`` checkpoints (``torch.save``) for the weights-only loader.

``test/native/torchCheckpoint.test.ts`` reads these with
``loadTorchStateDict`` and ``loadNativeFoundation`` and compares them with the
tensors ``torch.load(weights_only=True)`` returns, and with the logits
transformers computes after ``from_pretrained`` of a ``.bin``-only BERT
directory (legacy ``LayerNorm.gamma``/``beta`` names included).
"""
from __future__ import annotations

import collections
import json
import os
import pickle
import shutil

import torch
from transformers import BertConfig, BertForSequenceClassification

from generate import OUT, tensor_json

ROOT = OUT / 'torch_checkpoint'


def state():
    torch.manual_seed(3)
    base = torch.randn(4, 6)
    values = collections.OrderedDict()
    values['float32'] = torch.randn(3, 5)
    values['float64'] = torch.randn(2, 3, dtype=torch.float64)
    values['float16'] = torch.randn(7).half()
    values['bfloat16'] = torch.randn(2, 2).bfloat16()
    values['int64'] = torch.arange(-3, 9).reshape(3, 4)
    values['int32'] = torch.tensor([1, -2, 3], dtype=torch.int32)
    values['uint8'] = torch.tensor([0, 7, 255], dtype=torch.uint8)
    values['bool'] = torch.tensor([True, False, True])
    values['transposed'] = base.t()            # non-contiguous view
    values['slice'] = base[1:3, 2:5]          # offset view sharing storage
    values['shared'] = base                   # the same storage again
    values['parameter'] = torch.nn.Parameter(torch.randn(2, 3))
    values['scalar'] = torch.tensor(2.5)
    values['empty'] = torch.zeros(0, 3)
    return values


def generate():
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    values = state()
    torch.save(values, ROOT / 'zip.bin')
    torch.save(values, ROOT / 'legacy.bin', _use_new_zipfile_serialization=False)
    loaded = torch.load(ROOT / 'zip.bin', weights_only=True)
    expected = {name: {'dtype': str(value.dtype).removeprefix('torch.'), 'shape': list(value.shape),
                       'data': value.double().reshape(-1).tolist() if value.is_floating_point() else value.reshape(-1).tolist()}
                for name, value in loaded.items()}

    class Exploit:
        def __reduce__(self):
            return (os.system, ('echo pwned',))
    (ROOT / 'exploit.bin').write_bytes(pickle.dumps({'weight': Exploit()}, protocol=2))

    # A .bin-only BERT classifier directory with legacy LayerNorm names.
    torch.manual_seed(0)
    config = BertConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=16, max_position_embeddings=32, num_labels=3)
    model = BertForSequenceClassification(config).eval()
    directory = ROOT / 'bert_bin'
    directory.mkdir()
    config.save_pretrained(directory)
    legacy = collections.OrderedDict(
        (name.replace('LayerNorm.weight', 'LayerNorm.gamma').replace('LayerNorm.bias', 'LayerNorm.beta'), value)
        for name, value in model.state_dict().items())
    torch.save(legacy, directory / 'pytorch_model.bin', _use_new_zipfile_serialization=False)
    reloaded = BertForSequenceClassification.from_pretrained(directory).eval()
    ids = torch.tensor([[1, 5, 7, 2, 0]])
    with torch.no_grad():
        logits = reloaded(input_ids=ids, attention_mask=(ids != 0).long()).logits

    # The same weights split across two zip-format shards with an index.
    sharded = ROOT / 'bert_sharded'
    sharded.mkdir()
    config.save_pretrained(sharded)
    names = list(legacy)
    halves = [names[: len(names) // 2], names[len(names) // 2:]]
    weight_map = {}
    for index, part in enumerate(halves, start=1):
        file = f'pytorch_model-0000{index}-of-00002.bin'
        torch.save(collections.OrderedDict((name, legacy[name]) for name in part), sharded / file)
        weight_map.update({name: file for name in part})
    (sharded / 'pytorch_model.bin.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': weight_map}))

    # The ALBERT foundation's weights as a PyTorch checkpoint (the Hub conversion test's main branch).
    from safetensors.torch import load_file
    albert = load_file(OUT / 'vec' / 'albert_foundation' / 'model.safetensors')
    torch.save(collections.OrderedDict(albert), ROOT / 'albert_pytorch_model.bin')

    (ROOT / 'expected.json').write_text(json.dumps({
        'tensors': expected, 'input_ids': ids.tolist(), 'logits': tensor_json(logits)}))
