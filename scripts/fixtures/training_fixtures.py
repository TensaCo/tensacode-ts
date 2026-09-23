"""Training interoperability fixtures: experiences, checkpoints, response quality and DeBERTa-v2.

Outputs go to ``test/fixtures/training/``:

- ``experience_python.json``: a Python-written ``tensorcode.experience`` (external
  boundary, tuple/bytes/record codecs, TensorAdapter replay).
- ``checkpoint_python.json``: a standalone Adam ``tensorcode.checkpoint``.
- ``python_resume/`` + ``experience_resume.json``: a Python directory checkpoint
  (``tensorcode.tool_training``) and its experience.
- ``response_quality_{bert,electra}/``: saved ``ResponseQualityAssessor`` artifacts.
- ``interop.json``: expected weights, replays, losses and receipts for the above.
- ``deberta_*.safetensors`` + ``deberta_v2.json`` + ``deberta_v3_foundation/``:
  tiny seeded DeBERTa-v2/v3 models, configurations and outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import shutil

import torch
from safetensors.torch import save_model

from generate import OUT, tensor_json
from tensorcode import trace, training
from tensorcode._internal.training.checkpoint import save_checkpoint
from tensorcode._internal.vec.adapter import TensorAdapter
from tensorcode.ops.base import Operation

ROOT = OUT / 'training'


def write(name, payload):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / name).write_text(json.dumps(payload, separators=(',', ':')) + '\n')


@dataclass(frozen=True)
class Note:
    text: str
    weight: float


class External(Operation):
    """A non-replayable boundary; its result is recorded in experiences."""

    def forward(self, value, *, context=None):
        return torch.tensor([value['x'] * 2.0, float(value['y'])])


def linear_state(module):
    return {key: tensor_json(value) for key, value in module.state_dict().items()}


def experience():
    torch.manual_seed(11)
    external, head = External(), TensorAdapter(torch.nn.Linear(2, 2))
    weights = linear_state(head.module)
    value = {'x': 1.5, 'y': -2, 'tags': ('a', 'b'), 'raw': b'\x00\xffhi'}
    context = {'note': Note('hello é', 0.5)}
    with trace() as session:
        boundary = external(value, context=context)
        output = head(boundary)
    session.supervise(output, torch.tensor([0.5, 0.25]), loss='mse', source='review:é')
    session.save(ROOT / 'experience_python.json', operations={'external': external, 'head': head}, codecs={'note': Note})
    replay_head = TensorAdapter(torch.nn.Linear(2, 2))
    replay_head.load_state_dict(head.state_dict())
    loaded = training.load_experience(ROOT / 'experience_python.json',
                                      operations={'external': external, 'head': replay_head}, codecs={'note': Note})
    prediction = loaded.replay(loaded.supervisions[0].output, boundary='recorded')
    trainer = training.Trainer.from_ops({'external': external, 'head': replay_head}, lr=0.1)
    loss = trainer.step(loaded)
    return {'weights': weights, 'prediction': tensor_json(prediction), 'loss': loss,
            'trained': linear_state(replay_head.module)}


def standalone_checkpoint():
    torch.manual_seed(12)
    head = TensorAdapter(torch.nn.Linear(2, 2))
    trainer = training.Trainer.from_ops({'head': head}, optimizer=lambda ps: torch.optim.Adam(ps, lr=.01))
    inputs, target = torch.tensor([1., -2.]), torch.tensor([.5, -.25])
    with trace() as session:
        output = head(inputs)
    session.supervise(output, target, loss='mse')
    first = trainer.step(session)
    save_checkpoint(ROOT / 'checkpoint_python.json', operations={'head': head}, optimizer=trainer.optimizer)
    saved = linear_state(head.module)
    second = trainer.step(session)
    return {'inputs': inputs.tolist(), 'target': target.tolist(), 'first_loss': first, 'saved': saved,
            'next_loss': second, 'next': linear_state(head.module)}


def directory_checkpoint():
    torch.manual_seed(13)
    head = TensorAdapter(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.4)))
    head.eval()
    trainer = training.Trainer.from_ops({'head': head}, optimizer=lambda ps: torch.optim.Adam(ps, lr=.01))
    with trace() as session:
        output = head(torch.ones(2))
    session.supervise(output, torch.zeros(2), loss='mse')
    trainer.step(session)
    session.save(ROOT / 'experience_resume.json', operations={'head': head})
    target = ROOT / 'python_resume'
    shutil.rmtree(target, ignore_errors=True)
    trainer.save_checkpoint(target, progress={'cursor': 3, 'note': 'python'})
    state = {key: tensor_json(value) for key, value in head.state_dict().items()}
    loss = trainer.step(session)
    return {'state': state, 'next_loss': loss,
            'next': {key: tensor_json(value) for key, value in head.state_dict().items()}}


def response_quality():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from tensorcode._internal.response_quality import ResponseQualityAssessor, AXES
    tokenizer = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
                                     'question': 4, 'evidence': 5, 'candidate': 6, 'yes': 7}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
                                                  special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    inputs = {'question': 'question', 'evidence': [{'source_id': 's1', 'text': 'evidence yes'}], 'candidate': 'yes'}
    targets = {'support': True, 'completeness': False, 'constraints': None}
    records = {}
    for index, (model_type, input_format) in enumerate([('bert', None), ('electra', 'paired')]):
        config = {'foundation_config': {'model_type': model_type, 'vocab_size': 8, 'hidden_size': 8,
                  'embedding_size': 8, 'num_hidden_layers': 1, 'num_attention_heads': 2,
                  'intermediate_size': 16, 'max_position_embeddings': 128,
                  'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.},
                  'tokenizer_json': tokenizer.to_str(), 'tokenizer_special_tokens': {'pad_token': '[PAD]',
                  'unk_token': '[UNK]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'}, 'max_tokens': 64}
        if input_format is not None:
            config['input_format'] = input_format
        torch.manual_seed(21 + index)
        model = ResponseQualityAssessor(config).eval()
        with torch.no_grad():
            model.fit_calibration(torch.tensor([[1., 2., 3.], [-1., -2., -3.], [.5, -.5, 2.]]),
                                  [dict.fromkeys(AXES, True), dict.fromkeys(AXES, False),
                                   {'support': True, 'completeness': None, 'constraints': False}])
        name = f'response_quality_{model_type}'
        shutil.rmtree(ROOT / name, ignore_errors=True)
        model.save_pretrained(ROOT / name)
        with torch.no_grad():
            logits = model(inputs)
            batch = model([inputs, dict(inputs, candidate='question yes')])
        receipt = model.receipt(inputs)
        metadata = model.input_metadata(dict(inputs, candidate='yes ' * 40))
        loss = model.loss(inputs, targets).item()  # invalidates calibration; receipt taken first
        records[model_type] = {'directory': name, 'receipt': receipt, 'logits': tensor_json(logits),
                               'batch_logits': tensor_json(batch), 'loss': loss,
                               'metadata': metadata}
    return {'inputs': inputs, 'targets': targets, 'models': records}


def deberta():
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
    ids = torch.tensor([[1, 7, 11, 5, 9, 2, 4, 13], [1, 8, 6, 2, 0, 0, 0, 0]])
    mask = (ids != 0).long()
    types = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 0, 0]])
    cases = {
        # deberta-v3-small-like: relative attention with log buckets, shared keys, no absolute positions.
        'v3': {'model_type': 'deberta-v2', 'vocab_size': 40, 'hidden_size': 16, 'num_hidden_layers': 2,
               'num_attention_heads': 2, 'intermediate_size': 32, 'max_position_embeddings': 64,
               'relative_attention': True, 'position_buckets': 8, 'max_relative_positions': -1,
               'pos_att_type': ['p2c', 'c2p'], 'share_att_key': True, 'norm_rel_ebd': 'layer_norm',
               'position_biased_input': False, 'type_vocab_size': 0, 'layer_norm_eps': 1e-7,
               'hidden_act': 'gelu', 'pad_token_id': 0},
        # deberta-v2-like: separate positional projections, absolute positions, token types, embedding projection.
        'v2': {'model_type': 'deberta-v2', 'vocab_size': 40, 'hidden_size': 16, 'embedding_size': 12,
               'num_hidden_layers': 2, 'num_attention_heads': 4, 'intermediate_size': 24,
               'max_position_embeddings': 32, 'relative_attention': True, 'position_buckets': -1,
               'max_relative_positions': 12, 'pos_att_type': 'c2p|p2c', 'share_att_key': False,
               'position_biased_input': True, 'type_vocab_size': 2, 'hidden_act': 'gelu_new',
               'conv_kernel_size': 3, 'conv_act': 'tanh', 'pooler_hidden_act': 'tanh', 'pooler_dropout': 0.1},
        # plain transformer without relative attention.
        'plain': {'model_type': 'deberta-v2', 'vocab_size': 40, 'hidden_size': 16, 'num_hidden_layers': 1,
                  'num_attention_heads': 2, 'intermediate_size': 32, 'max_position_embeddings': 32,
                  'relative_attention': False, 'position_biased_input': True, 'type_vocab_size': 0},
    }
    records = {}
    for index, (name, data) in enumerate(cases.items()):
        config = AutoConfig.for_model(**data)
        torch.manual_seed(500 + index)
        model = AutoModel.from_config(config).eval()
        save_model(model, str(ROOT / f'deberta_{name}.safetensors'))
        record = {'input': data, 'to_dict': json.loads(json.dumps(config.to_dict())), 'diff': json.loads(config.to_json_string()),
                  'state_keys': list(model.state_dict().keys()), 'input_ids': ids.tolist(), 'attention_mask': mask.tolist()}
        kwargs = {'input_ids': ids, 'attention_mask': mask}
        if config.type_vocab_size:
            kwargs['token_type_ids'] = types
            record['token_type_ids'] = types.tolist()
        with torch.no_grad():
            record['last_hidden_state'] = tensor_json(model(**kwargs).last_hidden_state)
        labelled = AutoConfig.for_model(**{**data, 'id2label': {0: 'entailment', 1: 'neutral', 2: 'contradiction'}})
        torch.manual_seed(600 + index)
        classifier = AutoModelForSequenceClassification.from_config(labelled).eval()
        save_model(classifier, str(ROOT / f'deberta_{name}_classifier.safetensors'))
        record['classifier'] = {'config': json.loads(labelled.to_json_string()),
                                'state_keys': list(classifier.state_dict().keys())}
        with torch.no_grad():
            record['classifier']['logits'] = tensor_json(classifier(**kwargs).logits)
        # A pretrained-style directory (config.json + model.safetensors) for foundation loading.
        if name == 'v3':
            directory = ROOT / 'deberta_v3_foundation'
            shutil.rmtree(directory, ignore_errors=True)
            classifier.save_pretrained(directory, safe_serialization=True)
        records[name] = record
    from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position
    relative = torch.arange(-1100, 1101)
    records['buckets'] = [{'bucket_size': size, 'max_position': limit, 'start': -1100,
                           'values': make_log_bucket_position(relative, size, limit).long().tolist()}
                          for size, limit in ((256, 512), (8, 64), (128, 1024), (6, 20))]
    write('deberta_v2.json', records)


def generate():
    ROOT.mkdir(parents=True, exist_ok=True)
    write('interop.json', {'experience': experience(), 'checkpoint': standalone_checkpoint(),
                           'resume': directory_checkpoint(), 'response_quality': response_quality()})
    deberta()
