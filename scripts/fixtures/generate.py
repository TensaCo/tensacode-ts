"""Generate golden fixtures from the Python reference stack (PyTorch/transformers).

Run with the Python package's virtual environment:

    ../python/.venv/bin/python scripts/fixtures/generate.py

Fixtures are small, deterministic and committed under ``test/fixtures``. They let
the TypeScript tests check numerical parity without Python at test time.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Fixtures are CPU references. Hiding CUDA keeps CUDA generator states out of
# saved training checkpoints (a CPU-only runtime cannot restore them).
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'test' / 'fixtures'
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Every ``*_fixtures.py`` module beside this script exposes ``generate()``.
# Module builders add their own fixture modules; this file needs no edits.
GENERATORS = sorted(path.stem for path in Path(__file__).resolve().parent.glob('*_fixtures.py'))


def tensor_json(value: torch.Tensor) -> dict:
    value = value.detach().cpu()
    return {'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.'),
            'data': value.double().reshape(-1).tolist() if value.is_floating_point() else value.reshape(-1).tolist()}


def write_json(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=None, separators=(',', ':')) + '\n')


def main(selected: list[str]) -> None:
    torch.manual_seed(0)
    for name in GENERATORS:
        if selected and name not in selected:
            continue
        module = __import__(name)
        print(f'generating {name}')
        module.generate()


if __name__ == '__main__':
    main(sys.argv[1:])
