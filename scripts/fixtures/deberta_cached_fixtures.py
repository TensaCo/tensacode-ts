"""Real cached ``cross-encoder/nli-deberta-v3-small`` (DeBERTa-v3) reference logits.

Skipped when the checkpoint is not in the local Hugging Face cache.
"""
from __future__ import annotations

import torch

from generate import tensor_json, write_json

REPO = 'cross-encoder/nli-deberta-v3-small'
SNAPSHOT = 'fa2804872c3b4bd748f38c0185cc85775361e735'
PAIRS = [
    ('A man is playing a guitar on stage.', 'A person is making music.'),
    ('The database recovered after the restart.', 'The database is permanently lost.'),
    ('Two dogs run through a field.', 'The animals are sleeping indoors.'),
]


def generate() -> None:
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not isinstance(try_to_load_from_cache(REPO, 'model.safetensors', revision=SNAPSHOT), str):
        print(f'skipping: {REPO} is not cached')
        return
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=SNAPSHOT, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(REPO, revision=SNAPSHOT, local_files_only=True).eval()
    batch = tokenizer([p for p, _ in PAIRS], [h for _, h in PAIRS], padding=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**batch).logits
    write_json('deberta_cached.json', {
        'repo': REPO, 'snapshot': SNAPSHOT, 'pairs': PAIRS,
        'input_ids': batch['input_ids'].tolist(), 'attention_mask': batch['attention_mask'].tolist(),
        'token_type_ids': batch['token_type_ids'].tolist(), 'logits': tensor_json(logits),
        'id2label': {str(k): v for k, v in model.config.id2label.items()},
        'predictions': [model.config.id2label[int(i)] for i in logits.argmax(-1)],
    })
