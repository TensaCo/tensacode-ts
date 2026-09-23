"""Checkpoints whose optimizer hyperparameters are whole-number floats.

Python writes ``torch.optim.AdamW(params, lr=1.0, weight_decay=0.0)`` as
``1.0``/``0.0``. ``test/training/optimizerKinds.test.ts`` checks that
TypeScript re-saves such checkpoints byte for byte and writes the same bytes
from ``float()`` markers.
"""
from __future__ import annotations

import torch

from generate import OUT
from tensorcode._internal.training.checkpoint import save_checkpoint
from tensorcode._internal.vec.adapter import TensorAdapter

ROOT = OUT / 'training'


def generate():
    for name, make in {
        'adamw': lambda params: torch.optim.AdamW(params, lr=1.0, betas=(0.5, 1.0 - 0.25), weight_decay=0.0),
        'sgd': lambda params: torch.optim.SGD(params, lr=1.0, momentum=0.0, dampening=0, weight_decay=0.0),
    }.items():
        torch.manual_seed(5)
        head = TensorAdapter(torch.nn.Linear(2, 2))
        save_checkpoint(ROOT / f'optimizer_kinds_{name}.json', operations={'head': head}, optimizer=make(head.parameters()))
