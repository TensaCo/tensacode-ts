"""``Scene.from_foundation`` over the tiny CLIP snapshot (``test/fixtures/vec/clip_foundation``).

Python loads the snapshot's tokenizer with ``CLIPTokenizerFast``, which rebuilds
the backend as CLIP's byte-level BPE; ``test/tools/scene.test.ts`` checks that
TypeScript persists the same tokenizer (hash and saved ``tokenizer.json``).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch

from generate import OUT
from tensorcode.tools import Scene


def generate():
    torch.manual_seed(0)
    scene = Scene.from_foundation(str(OUT / 'vec' / 'clip_foundation'), revision='pinned', dimensions=8)
    with tempfile.TemporaryDirectory() as directory:
        scene.save_pretrained(directory)
        saved = (Path(directory) / 'tokenizer.json').read_text(encoding='utf-8')
    record = {'tokenizer_sha256': scene.configuration()['tokenizer_sha256'],
              'saved_tokenizer_sha256': hashlib.sha256(saved.encode()).hexdigest()}
    (OUT / 'vec' / 'scene_clip_foundation.json').write_text(json.dumps(record, indent=1) + '\n')
