"""Python JSON values in TypeScript: int/float kinds, dict order, dict key types and
transformers' non-persistent embedding buffers.

A Python caller may pass an int to a float field (``hidden_dropout_prob=0``),
write integral floats under keys that are not float-typed (``custom_scale=2.0``),
use integer-like keys in insertion order (``{'2': ..., '1': ...}``) and non-string
dictionary keys in traced values. These artifacts, operation fingerprints,
experience files and ``named_buffers()`` content fingerprints are what the
TypeScript loaders must reproduce byte for byte (``test/internal/jsonValues.test.ts``).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoConfig, BertConfig, T5Config

from generate import OUT
from topology_fixtures import CASES, build
from tensorcode import trace
from tensorcode._internal.cognition.locking import _ModelFingerprint
from tensorcode._internal.retrieval import RetrievalEncoder
from tensorcode._internal.training.persistence import configuration as op_configuration, fingerprint as op_fingerprint
from tensorcode.ops.base import Operation
from tensorcode.ops.text import Retrieve
from tensorcode.ops.text.score import ScoreResult
from tensorcode.tools.chatbot import Chatbot
from tensorcode.tools.investigator import Investigator

ROOT = OUT / 'json_values'
SPECIAL = {'pad_token': '<pad>', 'eos_token': '</s>', 'unk_token': '<unk>'}


def word_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '</s>': 1, '<unk>': 2, 'user': 3, ':': 4, 'hello': 5,
                                            'world': 6, 'answer': 7}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def chatbot_config():
    # Python caller ints in float fields and an integral float under a non-float key.
    foundation = T5Config(vocab_size=8, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, dropout_rate=0,
                          layer_norm_epsilon=1e-6, custom_scale=2.0)
    return {'foundation_config': foundation.to_dict(), 'tokenizer_json': word_tokenizer().to_str(),
            'tokenizer_special_tokens': dict(SPECIAL), 'workspace': {'slots': 3, 'steps': 2}, 'max_new_tokens': 3,
            'max_input_tokens': 32, 'max_turns': 2}


def bindings(tool):
    return {name: op_fingerprint(op_configuration(operation)) for name, operation in tool.operation_bindings().items()}


def save(tool, name):
    target = ROOT / name
    tool.save_pretrained(target)
    (target / 'README.md').unlink()


def tools(record):
    torch.manual_seed(1)
    chatbot = Chatbot(chatbot_config()).eval()
    save(chatbot, 'chatbot')
    record['chatbot'] = {'bindings': bindings(chatbot), 'fingerprint': chatbot.fingerprint,
                         'configuration': json.dumps(chatbot.configuration(), sort_keys=True)}

    torch.manual_seed(2)
    generator = chatbot_config()
    verifier = BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                          hidden_dropout_prob=0, attention_probs_dropout_prob=0,
                          id2label={2: 'support', 0: 'contradiction', 1: 'unknown'}, custom_weight=3.0).to_dict()
    investigator = Investigator({'vocabulary': ['hello', 'world'], 'dimensions': 8, 'slots': 2, 'steps': 1,
                                 'generator': generator, 'verifier_config': verifier,
                                 'verifier_tokenizer_json': generator['tokenizer_json'],
                                 'verifier_tokenizer_special_tokens': generator['tokenizer_special_tokens'],
                                 'verifier_labels': {'support': 2, 'contradiction': 0, 'unknown': 1}}).eval()
    save(investigator, 'investigator')
    record['investigator'] = {'bindings': bindings(investigator),
                              'configuration': json.dumps(investigator.configuration(), sort_keys=True)}

    torch.manual_seed(3)
    retrieval = RetrievalEncoder({
        'foundation_config': BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                                        intermediate_size=16, hidden_dropout_prob=0, attention_probs_dropout_prob=0).to_dict(),
        'tokenizer_json': word_tokenizer().to_str(), 'tokenizer_special_tokens': {'pad_token': '<pad>', 'unk_token': '<unk>'},
        'pooling': 'masked_mean', 'normalize': True, 'max_tokens': 4}).eval()
    save(retrieval, 'retrieval')
    record['retrieval'] = {'bindings': bindings(retrieval),
                           'configuration': json.dumps(retrieval.configuration(), sort_keys=True)}


def operations(record):
    """Owned operations whose saved configurations carry Python ints in float fields."""
    from transformers import PreTrainedTokenizerFast, ViTConfig, ViTImageProcessor, ViTModel
    from tensorcode.ops import text
    from tensorcode.ops.vec import ImageDecoder, ImageEncoder, TextDecoder, Transform
    from tensorcode.ops.vec.latent import Space
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, 'hello': 2, 'world': 3, 'answer': 4, '</s>': 5}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='</s>')
    foundations = ROOT / 'foundations'
    torch.manual_seed(6)
    from transformers import T5ForConditionalGeneration
    T5ForConditionalGeneration(T5Config(vocab_size=6, d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=2,
                                        dropout_rate=0, decoder_start_token_id=0, eos_token_id=5, pad_token_id=0,
                                        custom_scale=2.0)).save_pretrained(foundations / 't5')
    tokenizer.save_pretrained(foundations / 't5')
    ViTModel(ViTConfig(image_size=8, patch_size=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=16, hidden_dropout_prob=0), add_pooling_layer=False).eval().save_pretrained(foundations / 'vit')
    ViTImageProcessor(size={'height': 8, 'width': 8}, image_mean=[0, 0, 0], image_std=[1, 1, 1]).save_pretrained(foundations / 'vit')
    cases = {
        'transform': Transform({'architecture': 'transformer', 'input_space': Space('tokens', 3, organization='sequence').configuration(),
                                'output_space': Space('states', 2, organization='sequence').configuration(),
                                'native_config': {'model_type': 'bert', 'hidden_size': 4, 'num_hidden_layers': 1,
                                                  'num_attention_heads': 2, 'intermediate_size': 6, 'hidden_dropout_prob': 0,
                                                  'attention_probs_dropout_prob': 0, 'vocab_size': 8, 'custom_scale': 2.0}}),
        'text_decoder': TextDecoder.from_foundation(foundations / 't5', input_space=Space('arbitrary', 3, organization='sequence'),
                                                    generation={'max_new_tokens': 3, 'temperature': 1, 'repetition_penalty': 1}),
        'image_encoder': ImageEncoder.from_foundation(foundations / 'vit', readout='output_encoding', output_space=Space('visual', 8),
                                                      context_space=Space('context', 8, organization='sequence')),
        'image_decoder': ImageDecoder({
            'input_space': {'name': 'image-conditioning', 'dimensions': 6, 'organization': 'sequence'},
            'unet_config': dict(sample_size=4, in_channels=4, out_channels=4, down_block_types=['CrossAttnDownBlock2D'],
                                up_block_types=['CrossAttnUpBlock2D'], block_out_channels=[8], layers_per_block=1,
                                norm_num_groups=4, cross_attention_dim=8, attention_head_dim=2, norm_eps=1,
                                mid_block_scale_factor=2.0),
            'vae_config': dict(in_channels=3, out_channels=3, latent_channels=4, down_block_types=['DownEncoderBlock2D'],
                               up_block_types=['UpDecoderBlock2D'], block_out_channels=[8], layers_per_block=1,
                               norm_num_groups=4, sample_size=4, scaling_factor=1),
            'scheduler_config': dict(num_train_timesteps=10, clip_sample=False, clip_sample_range=1),
            'num_inference_steps': 2}),
        'text_classify': text.Classify.from_foundation(foundations / 't5', config={
            'labels': ['2', '1', 'a'], 'instructions': 'x', 'generation': {'max_new_tokens': 4, 'temperature': 1}}),
        # Item values are caller data: ``timeout`` is not a configuration float here.
        'text_retrieve': text.Retrieve.from_foundation(foundations / 't5', config={
            'items': {'2': {'timeout': 5, 'p': 1.0, 'n': [1, 2.0]}, '1': 'one'}, 'descriptions': {'2': 'two', '1': 'one'},
            'limit': 1, 'decoding': 'likelihood'}),
    }
    record['operations'] = {}
    for name, operation in cases.items():
        operation.eval()
        operation.save_pretrained(ROOT / 'operations' / name)
        record['operations'][name] = {'class': type(operation).__name__, 'bindings': bindings_of(operation)}
    shutil.rmtree(foundations)


def bindings_of(operation):
    getter = getattr(operation, 'operation_bindings', None)
    return {name: op_fingerprint(op_configuration(value)) for name, value in (getter() if getter else {'operation': operation}).items()}


class Fixed:
    """An external model whose configuration mixes ints in float fields and ordered integer-like keys."""

    def complete(self, request):
        raise AssertionError('not called')

    def configuration(self):
        return {'type': 'fixed', 'temperature': 1, 'top_p': 1.0, 'weights': {'2': 1.0, '1': 0, 'x': 0.5}}


def retrieve(record):
    operation = Retrieve.from_model(Fixed(), items={'2': 'two', '1': 'one', 'x': 'ex'}, limit=2)
    record['retrieve'] = {'configuration': json.dumps(op_configuration(operation), sort_keys=True),
                          'fingerprint': op_fingerprint(op_configuration(operation)),
                          'item_keys': list(operation.items), 'schema': json.dumps(operation.response_schema())}


@dataclass(frozen=True)
class Note:
    text: str
    weight: float


@dataclass(frozen=True)
class Bag:
    data: dict


class Echo(Operation):
    def configuration(self):
        return {'kind': 'echo', 'scale': 1.0, 'offset': 0, 'weights': {'2': 1.0, '10': 0, '1': 2}, 'stride': [8.0, 8]}

    def forward(self, value, *, context=None):
        return {'score': 1.0, 'count': 2, 'by_id': {1: 'a', 0: 'b', 10: 'c'}, 'pair': (1.0, 2), 'weights': [0.0, 1.0, 0.5],
                'big': 12345678901234567890, 'note': Note('n', 2.0)}


Echo.__module__ = 'parity'
Echo.__qualname__ = 'Echo'


def experience(record):
    echo = Echo()
    with trace() as session:
        output = echo({'x': 1.0, 'n': 2, 't': (1.0, 2), '2': 'two', '1': 'one',
                       'bag': Bag({3: 1.0, 1: 'bool', 2: 0, 'k': [1.0, 2], 1.5: 'x', None: 'none', (1, 2.0): 'tuple'})})
    session.supervise(output, Note('target', 1.0), loss='custom', source='review:1')
    session.supervise(output, ScoreResult(value=1.0, distribution={1: 1.0, 0: 0.0}, confidence=1.0),
                      loss='custom', source='review:2')
    session.save(ROOT / 'experience.json', operations={'echo': echo}, codecs={'note': Note, 'score': ScoreResult, 'bag': Bag})
    record['echo'] = {'fingerprint': op_fingerprint(op_configuration(echo))}


def memory():
    """A JSON memory file: caller values keep Python's kinds and key order (no float schema)."""
    from tensorcode._internal.memory.json import JsonMemory
    path = ROOT / 'memory.json'
    store = JsonMemory(path, retrieve=lambda search: search.candidates)
    store.append(1.0, kind='note', metadata={'p': 1, '2': 1.0, '1': 'x'})
    store.append({'10': 1.0, '9': 2, 'timeout': 0}, kind='note')
    shutil.copy(path, ROOT / 'memory_before.json')
    store.append({'x': 0.5}, kind='note', metadata={'timeout': 3})
    path.rename(ROOT / 'memory_after.json')


def trajectory(record):
    """A plan trajectory: caller state, observations and arguments are plain ``json.dumps`` data."""
    from tensorcode._internal.execution.planning import OutcomeExperience, PlanExecutionResult
    first = OutcomeExperience('c1', 'look', 'obs:1', {'2': 1.0, '1': 0, 'timeout': 3, 'p': 1}, 'observed',
                              {'k': 2.0, 'timeout': 1}, 1.0)
    second = OutcomeExperience('c1', 'look', 'obs:2', 2.0, 'observed', {}, None)
    PlanExecutionResult({'10': 1.0, '9': 2, 'score': 1}, (first, second), 'completed').save(ROOT / 'trajectory.json')
    record['trajectory'] = {'evidence': [first.as_evidence()['text'], second.as_evidence()['text']]}


def buffers(record):
    """``named_buffers()`` (non-persistent position/token-type ids) and content fingerprints."""
    cases = {}
    for name in ['bert', 'roberta', 'electra', 'albert', 'distilbert', 'deberta-v2', 'clip']:
        head, data = CASES[name]
        torch.manual_seed(5)
        model = build(head, AutoConfig.for_model(**data)).eval()
        save_file({key: value.contiguous() for key, value in model.state_dict().items()}, ROOT / f'buffers_{name}.safetensors')
        cases[name] = {
            'head': head, 'config': json.loads(json.dumps(model.config.to_dict())),
            'buffers': [[key, list(value.shape), str(value.dtype)] for key, value in model.named_buffers()],
            'fingerprint': _ModelFingerprint()([('model', model)], {'probe': 1.0, 'ids': {'2': 0, '1': 1}}),
        }
    record['buffers'] = cases


def generate():
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir(parents=True)
    record = {}
    tools(record)
    operations(record)
    retrieve(record)
    experience(record)
    memory()
    trajectory(record)
    buffers(record)
    (ROOT / 'record.json').write_text(json.dumps(record, indent=1) + '\n')
