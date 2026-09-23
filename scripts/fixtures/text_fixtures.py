"""Text operation parity fixtures: prompts, schemas, owned configurations and a tiny seeded T5.

Writes ``test/fixtures/text/``:

- ``foundation/`` — a tiny seeded T5 + WordLevel tokenizer saved with ``save_pretrained``
  (loaded by both implementations through ``from_foundation``);
- ``classify_python/`` — an owned ``Classify`` artifact saved by Python;
- ``text.json`` — prompts, response schemas, configurations, fingerprints,
  generated text, likelihood scores, losses and provider wire payloads.
"""
from __future__ import annotations

import json
import os
import shutil

import torch

from generate import OUT, ROOT, write_json

FOUNDATION = 'test/fixtures/text/foundation'  # relative to the package root (both implementations)
VOCAB = [
    '[PAD]', '[UNK]', '</s>', 'question', 'answer', 'yes', 'no', 'bad', 'good', 'label', 'choice', 'score',
    'keys', 'abstained', 'distribution', 'confidence', 'scores', 'true', 'false', 'null', 'a', 'b', '0', '1',
    'user', 'system', 'assistant', 'content', 'role', 'messages', 'instructions', 'Options', 'Answer',
    '{"', '":', ',', '"', '}', '[{"', '"}]', '-', ':', 'be', 'brief', 'affirmative', 'refund', 'policy',
]


def build_foundation(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
    backend = Tokenizer(models.WordLevel({token: index for index, token in enumerate(VOCAB)}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', eos_token='</s>')
    torch.manual_seed(1234)
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(VOCAB), d_model=8, d_ff=16, d_kv=4, num_layers=1, num_decoder_layers=1, num_heads=2,
        dropout_rate=0.0, decoder_start_token_id=0, eos_token_id=2, pad_token_id=0))
    with torch.no_grad():  # sharper logits so greedy decoding is not dominated by near ties
        for parameter in model.parameters():
            parameter.mul_(4.0)
    if path.exists():
        shutil.rmtree(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def result_json(result):
    from dataclasses import fields, is_dataclass
    if isinstance(result, tuple):  # Transform: the message tuple with the appended reply
        return [{'role': message.role, 'content': message.content} for message in result]
    if not is_dataclass(result):
        return result
    data = {field.name: getattr(result, field.name) for field in fields(result)}
    for key, value in data.items():
        if hasattr(value, 'items'):
            data[key] = {str(k): v for k, v in value.items()}
        elif isinstance(value, tuple):
            data[key] = list(value)
    return data


def generate():
    from tensorcode.ops import text
    from tensorcode.ops.text.model import ModelRequest
    from tensorcode._internal.text.native import alternative_prompt
    from tensorcode._internal.training.persistence import configuration, fingerprint
    from tensorcode.integrations import JevModel, OpenAICompatibleModel
    from tensorcode.integrations import jev as jev_module

    os.chdir(ROOT)
    target = OUT / 'text'
    target.mkdir(parents=True, exist_ok=True)
    build_foundation(ROOT / FOUNDATION)

    value = (text.Message('user', 'question'),)
    context = {'policy': (text.Message('system', 'be brief'),)}
    cases = {
        'transform': (text.Transform, {'instructions': 'Answer briefly', 'generation': {'max_new_tokens': 4}}, 'answer'),
        'classify': (text.Classify, {'labels': ['yes', 'no'], 'descriptions': {'yes': 'affirmative answer'},
                                     'instructions': 'Is it a question?', 'generation': {'max_new_tokens': 5}},
                     {'label': 'yes', 'distribution': {'yes': 0.75, 'no': 0.25}, 'confidence': None, 'abstained': False}),
        'decide': (text.Decide, {'options': ['yes', 'no'], 'generation': {'max_new_tokens': 3, 'num_beams': 2}},
                   {'choice': 'no', 'distribution': None, 'confidence': 0.5, 'abstained': False}),
        'score': (text.Score, {'rubric': ['bad', 'good']},
                  {'score': 1, 'distribution': None, 'confidence': None, 'abstained': False}),
        'retrieve': (text.Retrieve, {'items': {'a': 'answer', 'b': 'question'}, 'limit': 1},
                     {'keys': ['a'], 'scores': None, 'abstained': False}),
        'classify_likelihood': (text.Classify, {'labels': ['question', 'answer'], 'descriptions': {'answer': 'answer'},
                                                'decoding': 'likelihood'},
                                {'label': 'answer', 'distribution': None, 'confidence': None, 'abstained': False}),
        'decide_likelihood_mean': (text.Decide, {'options': ['yes', 'no answer'], 'decoding': 'likelihood',
                                                 'likelihood_normalization': 'mean'},
                                   {'choice': 'yes', 'distribution': {'yes': 0.25, 'no answer': 0.75},
                                    'confidence': None, 'abstained': False}),
        'score_likelihood': (text.Score, {'rubric': ['bad', 'good', 'yes'], 'decoding': 'likelihood'},
                             {'score': 2, 'distribution': None, 'confidence': None, 'abstained': False}),
        'retrieve_likelihood': (text.Retrieve, {'items': {'a': 'answer', 'b': 'question', 'c': 'refund policy'},
                                                'limit': 2, 'decoding': 'likelihood'},
                                {'keys': ['b', 'c'], 'scores': None, 'abstained': False}),
    }
    owned = {}
    for name, (cls, config, target_value) in cases.items():
        op = cls.from_foundation(FOUNDATION, config=config)
        request = op._request(value, None)
        inputs = op.model.inputs(request)
        with torch.no_grad():
            ids = op.model.model.generate(**inputs, **op.model.generation)
        record = {
            'class': cls.__name__,
            'config': config,
            'target': target_value,
            'configuration': op.configuration(),
            'operation': {'configuration': configuration(op), 'fingerprint': fingerprint(configuration(op))},
            'objective': {'configuration': configuration(op.training_operation),
                          'fingerprint': fingerprint(configuration(op.training_operation))},
            'prompt': op.model.prompt(request),
            'context_prompt': op.model.prompt(op._request(value, context)),
            'input_ids': inputs['input_ids'].tolist(),
            'generated_ids': ids.tolist(),
            'generated_text': op.model.tokenizer.decode(ids[0], skip_special_tokens=True),
            'loss': float(op.loss(value, target_value)),
            'replayable': op.replayable,
        }
        if hasattr(op, 'response_schema'):
            record['response_schema'] = op.response_schema()
        if getattr(op, 'decoding', 'generate') == 'likelihood':
            alternatives = op._alternatives()
            record['alternatives'] = [list(pair) for pair in alternatives]
            record['alternative_prompt'] = alternative_prompt(op._scoring_request(value, None), alternatives)
            record['scores'] = op.model.score_alternatives(op._scoring_request(value, None), alternatives,
                                                           normalization=op.likelihood_normalization)
            record['result'] = result_json(op(value))
        else:
            try:
                record['result'] = result_json(op(value))
            except Exception as error:  # noqa: BLE001 - record the exact outcome for parity
                record['error'] = {'type': type(error).__name__, 'message': str(error)}
        owned[name] = record

    # A Python-saved owned artifact that TypeScript loads and re-saves byte-identically.
    artifact = target / 'classify_python'
    if artifact.exists():
        shutil.rmtree(artifact)
    classify = text.Classify.from_foundation(FOUNDATION, config=cases['classify'][1])
    classify.save_pretrained(artifact)

    # Prompt serialization edge cases.
    prompts = []
    requests = [
        ModelRequest((text.Message('user', 'héllo ✓ "quoted"\nline'),)),
        ModelRequest((text.Message('system', 'rules'), text.Message('user', (text.TextPart('one '), text.TextPart('two', source_ref='s:1')))),
                     instructions='Do it', response_schema={'type': 'object', 'required': ['x'], 'properties': {'x': {'type': 'number', 'minimum': 0}}},
                     schema_name='custom.name'),
    ]
    native = text.Transform.from_foundation(FOUNDATION).model
    for request in requests:
        prompts.append({
            'prompt': native.prompt(request),
            'alternative_prompt': alternative_prompt(request, [('x: first', 'x'), ('y', 'y')]),
        })

    # Response schemas for external operations.
    class Provider:
        def complete(self, request):
            raise AssertionError('unused')

        def configuration(self):
            return {'type': 'fixture_provider'}

    provider = Provider()
    external_ops = {
        'classify': text.Classify.from_model(provider, labels=('billing', 'technical', 'other'), instructions='Route',
                                             descriptions={'billing': 'payments and charges'}),
        'decide': text.Decide.from_model(provider, options=('archive', 'reply')),
        'score': text.Score.from_model(provider, rubric=('low', 'medium', 'high'), instructions='Urgency'),
        'retrieve': text.Retrieve.from_model(provider, items={'policy': {'text': 'refund policy'}, 'faq': 'general'},
                                             descriptions={'policy': 'refund policy', 'faq': 'general questions'}, limit=2),
        'transform': text.Transform.from_model(provider, instructions='Summarize'),
    }
    external = {}
    for name, op in external_ops.items():
        external[name] = {
            'configuration': op.configuration(),
            'operation': {'configuration': configuration(op), 'fingerprint': fingerprint(configuration(op))},
        }
        if hasattr(op, 'response_schema'):
            external[name]['response_schema'] = op.response_schema()
    openai_op = text.Classify.from_model(OpenAICompatibleModel(base_url='https://example.test/v1/', model='m'),
                                         labels=('a', 'b'), instructions='x')
    external['openai_classify'] = {
        'configuration': openai_op.configuration(),
        'operation': {'configuration': configuration(openai_op), 'fingerprint': fingerprint(configuration(openai_op))},
    }

    # Provider wire payloads.
    message = text.Message('user', (
        text.TextPart('What animal?', source_ref='prompt:1'),
        text.ImagePart(data=b'image-bytes', media_type='image/png', source_ref='upload:1'),
        text.ImagePart(url='https://example.test/cat.jpg', detail='low'),
    ))
    request = external_ops['classify']._request((message,), None)
    chat = OpenAICompatibleModel(base_url='https://example.test/v1', model='vision-test')
    responses = OpenAICompatibleModel(base_url='https://example.test/v1', model='vision-test', api='responses')
    jev_request = external_ops['score']._request((text.Message('user', (text.TextPart('help now', source_ref='t:1'),)),), None)
    wire = {
        'chat_payload': chat._chat_payload(request),
        'responses_payload': responses._responses_payload(request),
        'chat_body': json.dumps(chat._chat_payload(request), separators=(',', ':'), ensure_ascii=False),
        'jev_score_question': list(jev_module._question(jev_request)),
        'jev_state': jev_module._state(jev_request),
        'schema_names': {name: __import__('tensorcode.integrations.openai', fromlist=['_schema_name'])._schema_name(name)
                         for name in ['tensorcode.classify', '', 'x' * 80, 'a b/c']},
    }
    del JevModel

    write_json('text/text.json', {'foundation': FOUNDATION, 'value': 'question', 'owned': owned,
                                  'prompts': prompts, 'external': external, 'wire': wire})
    generate_flan()


FLAN = 'google/flan-t5-small'
FLAN_SNAPSHOT = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'


def generate_flan():
    """Owned operations on the cached flan-T5-small snapshot (skipped when absent)."""
    import hashlib
    from huggingface_hub import try_to_load_from_cache
    from tensorcode.ops import text

    if not isinstance(try_to_load_from_cache(FLAN, 'model.safetensors', revision=FLAN_SNAPSHOT), str):
        print('skipping flan-t5-small text fixtures (not cached)')
        return
    value = (text.Message('user', 'I was charged twice for my subscription.'),)
    classify = text.Classify.from_foundation(FLAN, revision=FLAN_SNAPSHOT, local_files_only=True, config={
        'labels': ['billing', 'technical'], 'descriptions': {'billing': 'payments, charges and refunds'},
        'instructions': 'Route the support ticket', 'decoding': 'likelihood'})
    transform = text.Transform.from_foundation(FLAN, revision=FLAN_SNAPSHOT, local_files_only=True, config={
        'instructions': 'Answer the question.', 'generation': {'max_new_tokens': 8}})
    question = (text.Message('user', 'What color is the sky?'),)
    from tensorcode._internal.training.persistence import bindings
    configured = classify.configuration()
    # The embedded tokenizer JSON went through Rust ``Tokenizer.from_str``,
    # whose float parsing moves some Unigram scores by one ULP; TypeScript
    # reproduces it, so configurations and fingerprints match exactly.
    tokenizer_json = configured['tokenizer']['json']
    write_json('text/flan.json', {
        'snapshot': FLAN_SNAPSHOT,
        'classify': {
            'configuration_sha256': hashlib.sha256(json.dumps(configured, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'fingerprints': {name: record['fingerprint'] for name, record in bindings(classify.operation_bindings()).items()},
            'tokenizer_vocab_size': len(json.loads(tokenizer_json)['model']['vocab']),
            'foundation': configured['foundation'],
            'scores': classify.model.score_alternatives(classify._scoring_request(value, None), classify._alternatives()),
            'loss': float(classify.loss(value, {'label': 'billing', 'distribution': None, 'confidence': None, 'abstained': False}).detach()),
        },
        'transform': {
            'text': transform(question)[-1].content,
            'loss': float(transform.loss(question, 'blue').detach()),
        },
    })
