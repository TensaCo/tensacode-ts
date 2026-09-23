"""A Python-written pretrained artifact that TypeScript must load and re-save byte-identically."""
from __future__ import annotations

import shutil

import torch
from torch import nn

from generate import OUT, write_json
from tensorcode._internal.pretrained import PretrainedTool
from tensorcode._internal.workspace import Workspace
from tensorcode.training.calibration import TemperatureCalibration


class FixtureTool(PretrainedTool):
    """Identity ``artifact_fixtures.FixtureTool``; mirrored by a TypeScript test class."""

    def __init__(self, config):
        super().__init__(config)
        dimensions = config['dimensions']
        self.workspace = Workspace(dimensions, 2, 1)
        self.head = nn.Linear(dimensions, 3).half()
        self.shared = nn.Linear(3, 3, bias=False)
        self.alias = self.shared
        self.calibration = TemperatureCalibration()

    def forward(self, inputs, *, context=None):
        return self.head(inputs)


def generate():
    torch.manual_seed(11)
    tool = FixtureTool({'dimensions': 4, 'label': 'fixture', 'rate': 1.0 + 0.5})
    target = OUT / 'artifact_python'
    if target.exists():
        shutil.rmtree(target)
    tool.save_pretrained(target)
    restored = FixtureTool.from_pretrained(target)
    write_json('artifact_python.json', {'state_keys': list(restored.state_dict().keys()),
                                        'dtypes': {k: str(v.dtype) for k, v in restored.state_dict().items()}})
