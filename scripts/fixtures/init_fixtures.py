"""Fresh-construction parity: seeded initialization of every tool, operation and native model.

Each case seeds ``torch.manual_seed`` (and ``random.seed``), constructs the
object from a JSON configuration and records a SHA-256 of every state tensor's
raw bytes plus the generator state afterwards. The TypeScript test constructs
the same objects after ``manualSeed`` and must reproduce every hash, so freshly
initialized weights are bitwise identical in both languages.

Regenerate with ``npm run fixtures -- init_fixtures``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import random

import torch

from generate import OUT, write_json


def _load(path):
    return json.loads((OUT / path).read_text())


def _digest(tensor):
    tensor = tensor.detach().contiguous()
    data = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def _record(obj, seed):
    return {'seed': seed, 'state': {key: [str(value.dtype).removeprefix('torch.'), list(value.shape), _digest(value)]
                                    for key, value in obj.state_dict().items()},
            'rng': _digest(torch.get_rng_state())}


def _tiny_response_config(model_type, input_format=None):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    tokenizer = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
                                     'question': 4, 'evidence': 5, 'candidate': 6, 'yes': 7}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
                                                  special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    config = {'foundation_config': {'model_type': model_type, 'vocab_size': 8, 'hidden_size': 8, 'embedding_size': 8,
                                    'num_hidden_layers': 1, 'num_attention_heads': 2, 'intermediate_size': 16,
                                    'max_position_embeddings': 128, 'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.},
              'tokenizer_json': tokenizer.to_str(),
              'tokenizer_special_tokens': {'pad_token': '[PAD]', 'unk_token': '[UNK]', 'cls_token': '[CLS]', 'sep_token': '[SEP]'},
              'max_tokens': 64}
    if input_format is not None:
        config['input_format'] = input_format
    return config


def tool_cases():
    owned = _load('vec/owned.json')
    text = _load('vec/text.json')
    vision = _load('vec/vision.json')
    scene = _load('vec/scene.json')
    tools = _load('tools/tools.json')
    ranking = _load('ranking.json')
    owned_text = _load('text/text.json')['owned']
    diffusion = _load('diffusion/records.json')
    albert = _load('vec/albert.json')
    language = _load('scene_language/records.json')
    foundation = dict(scene['foundation']['configuration'])
    foundation['_tokenizer_json'] = (OUT / 'vec' / 'clip_foundation' / 'tokenizer.json').read_text(encoding='utf-8')
    tiny_language = dict(language['tiny']['configuration'])
    tiny_language['_language_assets'] = {path.name: path.read_text(encoding='utf-8')
                                         for path in sorted((OUT / 'scene_language' / 'tiny' / 'processor').iterdir())}
    return [
        ('transform_linear', 'tensorcode.ops.vec.transform:Transform', owned['transform_linear']['configuration']),
        ('classify_mlp', 'tensorcode.ops.vec.classify:Classify', owned['classify_mlp']['configuration']),
        ('decode_sequence', 'tensorcode.ops.vec.decode:Decode', owned['decode_sequence']['configuration']),
        ('score_mlp', 'tensorcode.ops.vec.score:Score', owned['score_mlp']['configuration']),
        ('transform_bert', 'tensorcode.ops.vec.transform:Transform', owned['transform_bert']['configuration']),
        ('vocabulary', 'tensorcode.ops.vec.encode:VocabularyEncoder', owned['vocabulary']['configuration']),
        ('patch', 'tensorcode.ops.vec.encode:PatchEncoder', owned['patch']['configuration']),
        ('text_encoder_t5', 'tensorcode.ops.vec.encode:TextEncoder', text['encoder_t5_foundation_configuration']),
        ('text_encoder_bert', 'tensorcode.ops.vec.encode:TextEncoder', text['encoder_bert_foundation_configuration']),
        ('text_encoder_albert', 'tensorcode.ops.vec.encode:TextEncoder', albert['configuration']),
        ('text_decoder_t5', 'tensorcode.ops.vec.decode:TextDecoder', text['decoder_foundation_configuration']),
        ('image_encoder', 'tensorcode.ops.vec.encode:ImageEncoder', vision['foundation_configuration']),
        ('image_decoder', 'tensorcode.ops.vec.decode:ImageDecoder', diffusion['decoders']['a']['configuration']),
        ('scene_rank', 'tensorcode.tools.scene:Scene', scene['rank']['configuration']),
        ('scene_foundation', 'tensorcode.tools.scene:Scene', foundation),
        ('scene_language', 'tensorcode.tools.scene:Scene', tiny_language),
        ('chatbot', 'tensorcode.tools.chatbot:Chatbot', tools['chatbot']['configuration']),
        ('investigator', 'tensorcode.tools.investigator:Investigator', tools['investigator']['configuration']),
        ('planner', 'tensorcode.tools.planner:Planner', tools['planner']['configuration']),
        ('decision', 'tensorcode.tools.decision:Decision', tools['planner']['decision']['configuration']),
        ('response_quality_bert', 'tensorcode._internal.response_quality:ResponseQualityAssessor', _tiny_response_config('bert')),
        ('response_quality_electra', 'tensorcode._internal.response_quality:ResponseQualityAssessor',
         _tiny_response_config('electra', 'paired')),
        ('rank_operation', 'tensorcode._internal.ranking:RankOperation', ranking['rank']['config']),
        ('text_transform', 'tensorcode.ops.text.transform:Transform', owned_text['transform']['configuration']),
        ('text_classify', 'tensorcode.ops.text.classify:Classify', owned_text['classify']['configuration']),
        ('text_decide', 'tensorcode.ops.text.decide:Decide', owned_text['decide']['configuration']),
        ('text_score', 'tensorcode.ops.text.score:Score', owned_text['score']['configuration']),
        ('text_retrieve', 'tensorcode.ops.text.retrieve:Retrieve', owned_text['retrieve']['configuration']),
    ]


def tools():
    records = []
    for index, (name, target, config) in enumerate(tool_cases()):
        module, cls = target.split(':')
        seed = 1000 + index
        torch.manual_seed(seed)
        random.seed(seed)
        constructor = getattr(importlib.import_module(module), cls)
        if cls == 'RankOperation':
            obj = constructor(config, task_key='question', candidates_key='hypotheses')
        else:
            obj = constructor(config)
        records.append({'name': name, 'class': target, 'configuration': config, **_record(obj, seed)})
    torch.manual_seed(1100)
    from tensorcode._internal.workspace import Workspace
    records.append({'name': 'workspace', 'class': 'tensorcode._internal.workspace:Workspace',
                    'configuration': {'dimensions': 6, 'slots': 3, 'steps': 2}, **_record(Workspace(6, slots=3, steps=2), 1100)})
    return records


ENCODER = dict(hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16, vocab_size=20, max_position_embeddings=16)
NATIVE = [
    ('base', dict(model_type='bert', **ENCODER)),
    ('sequence-classification', dict(model_type='bert', num_labels=3, **ENCODER)),
    ('base', dict(model_type='roberta', pad_token_id=1, **ENCODER)),
    ('sequence-classification', dict(model_type='roberta', pad_token_id=1, **ENCODER)),
    ('base', dict(model_type='electra', embedding_size=4, **ENCODER)),
    ('sequence-classification', dict(model_type='electra', embedding_size=4, num_labels=3, **ENCODER)),
    ('base', dict(model_type='distilbert', dim=8, n_layers=2, n_heads=2, hidden_dim=16, vocab_size=20, max_position_embeddings=16)),
    ('sequence-classification', dict(model_type='distilbert', dim=8, n_layers=2, n_heads=2, hidden_dim=16, vocab_size=20,
                                     max_position_embeddings=16, num_labels=2)),
    ('base', dict(model_type='albert', embedding_size=4, num_hidden_groups=1, **ENCODER)),
    ('sequence-classification', dict(model_type='albert', embedding_size=4, num_labels=3, **ENCODER)),
    ('base', dict(model_type='deberta-v2', relative_attention=True, position_buckets=4, max_relative_positions=8,
                  pos_att_type=['p2c', 'c2p'], position_biased_input=False, **ENCODER)),
    ('sequence-classification', dict(model_type='deberta-v2', num_labels=3, relative_attention=True, position_buckets=4,
                                     pos_att_type=['p2c', 'c2p'], **ENCODER)),
    ('seq2seq', dict(model_type='t5', vocab_size=20, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=2, num_decoder_layers=1)),
    ('seq2seq', dict(model_type='t5', vocab_size=20, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=1,
                     feed_forward_proj='gated-gelu', tie_word_embeddings=False)),
    ('encoder', dict(model_type='t5', vocab_size=20, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=2)),
    ('base', dict(model_type='vit', hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                  image_size=8, patch_size=4)),
    ('base', dict(model_type='clip', projection_dim=8,
                  text_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, vocab_size=20,
                                   max_position_embeddings=16),
                  vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, image_size=8,
                                     patch_size=4))),
    ('base', dict(model_type='llama', hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=1, vocab_size=20, max_position_embeddings=32)),
]


def native():
    from transformers import (AutoConfig, AutoModel, AutoModelForImageTextToText, AutoModelForSeq2SeqLM,
                              AutoModelForSequenceClassification, T5EncoderModel, ViTModel)
    language = _load('scene_language/records.json')['tiny']['configuration']['language_config']
    cases = NATIVE + [('image-text-to-text', dict(language))]
    records = []
    for index, (head, spec) in enumerate(cases):
        spec = dict(spec)
        model_type = spec.pop('model_type')
        config = AutoConfig.for_model(model_type, **spec)
        seed = 300 + index
        torch.manual_seed(seed)
        if head == 'base':
            model = ViTModel(config, add_pooling_layer=False) if model_type == 'vit' else AutoModel.from_config(config)
        elif head == 'sequence-classification':
            model = AutoModelForSequenceClassification.from_config(config)
        elif head == 'seq2seq':
            model = AutoModelForSeq2SeqLM.from_config(config)
        elif head == 'encoder':
            model = T5EncoderModel(config)
        else:
            model = AutoModelForImageTextToText.from_config(config)
        records.append({'name': f'{model_type}:{head}:{index}', 'head': head, 'configuration': json.loads(config.to_json_string()),
                        **_record(model, seed)})
    return records


def diffusers():
    from diffusers import AutoencoderKL, UNet2DConditionModel
    base = dict(sample_size=8, in_channels=4, out_channels=4, norm_num_groups=4, cross_attention_dim=8, attention_head_dim=2)
    unets = {
        'cross_attention': dict(base, block_out_channels=[8, 16], layers_per_block=2, down_block_types=['CrossAttnDownBlock2D', 'DownBlock2D'],
                                up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'], transformer_layers_per_block=[1, 2]),
        'self_attention': dict(base, block_out_channels=[8, 8], layers_per_block=2, down_block_types=['AttnDownBlock2D', 'DownBlock2D'],
                               up_block_types=['UpBlock2D', 'AttnUpBlock2D'], mid_block_type='UNetMidBlock2D'),
        'simple_cross_attention': dict(base, block_out_channels=[8, 8], layers_per_block=1,
                                       down_block_types=['SimpleCrossAttnDownBlock2D', 'SimpleCrossAttnDownBlock2D'],
                                       up_block_types=['SimpleCrossAttnUpBlock2D', 'SimpleCrossAttnUpBlock2D'],
                                       mid_block_type='UNetMidBlock2DSimpleCrossAttn'),
        'resnet_sampling': dict(base, block_out_channels=[8, 8], layers_per_block=1, down_block_types=['ResnetDownsampleBlock2D', 'CrossAttnDownBlock2D'],
                                up_block_types=['CrossAttnUpBlock2D', 'ResnetUpsampleBlock2D']),
        'k_diffusion': dict(base, block_out_channels=[8, 8], layers_per_block=1, down_block_types=['KDownBlock2D', 'KCrossAttnDownBlock2D'],
                            up_block_types=['KCrossAttnUpBlock2D', 'KUpBlock2D'], mid_block_type=None, resnet_time_scale_shift='ada_group'),
        'linear_projection': dict(base, block_out_channels=[8], layers_per_block=1, down_block_types=['CrossAttnDownBlock2D'],
                                  up_block_types=['CrossAttnUpBlock2D'], use_linear_projection=True, only_cross_attention=True),
    }
    vaes = {
        'plain': dict(in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=4, block_out_channels=[8, 8], layers_per_block=1,
                      down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D'], up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D']),
        'attention': dict(in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=4, block_out_channels=[8], layers_per_block=2,
                          down_block_types=['AttnDownEncoderBlock2D'], up_block_types=['AttnUpDecoderBlock2D'],
                          mid_block_add_attention=False, use_quant_conv=False),
    }
    records = []
    for kind, table, cls in (('unet', unets, UNet2DConditionModel), ('vae', vaes, AutoencoderKL)):
        for name, config in table.items():
            torch.manual_seed(7)
            model = cls(**config)
            configuration = {key: value for key, value in dict(model.config).items() if not key.startswith('_')}
            records.append({'name': f'{kind}:{name}', 'kind': kind,
                            'configuration': json.loads(json.dumps(configuration, default=list)), **_record(model, 7)})
    return records


def sampling():
    """Seeded ``generate`` (multinomial, nucleus, beam sampling) and training-mode dropout forwards."""
    from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM
    config = AutoConfig.for_model('t5', vocab_size=40, d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=2, num_decoder_layers=2,
                                  decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    torch.manual_seed(5)
    model = AutoModelForSeq2SeqLM.from_config(config).eval()
    ids = torch.tensor([[3, 5, 7, 9, 1], [4, 4, 8, 6, 1]])
    runs = []
    for settings in [dict(do_sample=True, max_new_tokens=10, num_return_sequences=2),
                     dict(do_sample=True, max_new_tokens=6, top_k=0, top_p=0.9, temperature=0.7),
                     dict(do_sample=True, num_beams=3, max_new_tokens=6)]:
        torch.manual_seed(11)
        with torch.no_grad():
            sequences = model.generate(input_ids=ids, **settings)
        runs.append({'settings': settings, 'sequences': sequences.tolist(), 'rng': _digest(torch.get_rng_state())})
    generation = {'configuration': json.loads(config.to_json_string()), 'seed': 5, 'input_ids': ids.tolist(), 'runs': runs}
    forwards = []
    for head, spec in [('base', dict(model_type='bert', hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                                     vocab_size=30, max_position_embeddings=32, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.3)),
                       ('seq2seq', dict(model_type='t5', vocab_size=30, d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=2,
                                        num_decoder_layers=1, dropout_rate=0.25))]:
        spec = dict(spec)
        native_config = AutoConfig.for_model(spec.pop('model_type'), **spec)
        torch.manual_seed(5)
        network = (AutoModelForSeq2SeqLM if head == 'seq2seq' else AutoModel).from_config(native_config).train()
        tokens = torch.tensor([[1, 5, 7, 9, 2, 0], [3, 4, 4, 8, 6, 2]])
        torch.manual_seed(6)
        with torch.no_grad():
            if head == 'seq2seq':
                output = network(input_ids=tokens, decoder_input_ids=tokens[:, :4]).logits
            else:
                output = network(input_ids=tokens).last_hidden_state
        forwards.append({'head': head, 'configuration': json.loads(native_config.to_json_string()), 'input_ids': tokens.tolist(),
                         'output': output.flatten().tolist(), 'rng': _digest(torch.get_rng_state())})
    return {'generation': generation, 'dropout_forwards': forwards}


def seeded_decode():
    """``ImageDecoder`` with ``context={'seed': ...}``: noise from ``torch.Generator().manual_seed(seed)``."""
    from tensorcode.ops.vec.decode import ImageDecoder
    from tensorcode.ops.vec.latent import Latent, Space
    records = _load('diffusion/records.json')['decoders']['a']
    torch.manual_seed(17)
    decoder = ImageDecoder(records['configuration']).eval()
    space = decoder.input_space
    value = Latent(torch.tensor(records['value']['data'], dtype=torch.float32).reshape(records['value']['shape']), space)
    with torch.no_grad():
        pixels = decoder(value, context={'seed': 1234})
    return {'configuration': records['configuration'], 'seed': 17, 'value': records['value'], 'context_seed': 1234,
            'pixels': pixels.flatten().tolist(), 'shape': list(pixels.shape)}


def _verifier_foundation():
    """A three-label sequence classifier config over the BERT fixture weights (the head is missing)."""
    import shutil
    target = OUT / 'init' / 'verifier_foundation'
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(OUT / 'vec' / 'bert_foundation', target)
    config = json.loads((target / 'config.json').read_text())
    config['id2label'] = {'0': 'support', '1': 'unknown', '2': 'contradiction'}
    config['label2id'] = {'support': 0, 'unknown': 1, 'contradiction': 2}
    (target / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    return target


def foundation_flows():
    """``from_foundation`` flows: loading draws nothing (meta construction); new heads draw like Python."""
    import transformers
    transformers.logging.set_verbosity_error()
    from tensorcode.ops.vec import Classify, Score, Transform
    from tensorcode.ops.vec.decode import ImageDecoder, TextDecoder
    from tensorcode.ops.vec.encode import ImageEncoder, TextEncoder
    from tensorcode.ops.vec.latent import Space
    import tensorcode.ops.text as text
    from tensorcode.tools.chatbot import Chatbot
    from tensorcode.tools.investigator import Investigator
    from tensorcode.tools.planner import Planner
    from tensorcode.tools.scene import Scene
    from tensorcode._internal.response_quality import ResponseQualityAssessor
    from tensorcode._internal.retrieval import RetrievalEncoder
    root = 'test/fixtures'
    verifier = _verifier_foundation().relative_to(OUT.parent.parent).as_posix()
    labels = {'support': 0, 'unknown': 1, 'contradiction': 2}
    flows = {
        'vec_transform': lambda: Transform.from_foundation(f'{root}/vec/bert_foundation', input_space=Space('in', 3), output_space=Space('out', 2)),
        'vec_classify': lambda: Classify.from_foundation(f'{root}/vec/roberta_foundation', input_space=Space('in', 3), labels=['a', 'b']),
        'vec_score': lambda: Score.from_foundation(f'{root}/vec/bert_foundation', query_space=Space('q', 3), candidate_space=Space('c', 3),
                                                   meaning='pair utility'),
        'text_encoder_t5': lambda: TextEncoder.from_foundation(f'{root}/vec/t5_foundation'),
        'text_encoder_bert': lambda: TextEncoder.from_foundation(f'{root}/vec/bert_foundation', readout='output_encoding',
                                                                 context_space=Space('ctx', 8, organization='sequence')),
        'text_decoder_t5': lambda: TextDecoder.from_foundation(f'{root}/vec/t5_foundation', input_space=Space('in', 3, organization='sequence')),
        'image_encoder': lambda: ImageEncoder.from_foundation(f'{root}/vec/vit_foundation', output_space=Space('v', 8), readout='output_encoding'),
        'image_decoder': lambda: ImageDecoder.from_foundation(f'{root}/diffusion/foundation', input_space=Space('in', 6, organization='sequence'),
                                                              num_inference_steps=2),
        'text_transform': lambda: text.Transform.from_foundation(f'{root}/text/foundation'),
        'text_classify': lambda: text.Classify.from_foundation(f'{root}/text/foundation', config={'labels': ['yes', 'no']}),
        'scene_clip': lambda: Scene.from_foundation(f'{root}/vec/clip_foundation', revision='pinned', dimensions=8),
        'scene_language': lambda: Scene.from_language_foundation(f'{root}/scene_language/tiny_foundation', revision=None, local_files_only=True),
        'chatbot': lambda: Chatbot.from_foundation(f'{root}/vec/t5_foundation'),
        'planner': lambda: Planner.from_foundations(f'{root}/vec/bert_foundation', f'{root}/vec/t5_foundation'),
        'investigator_missing_head': lambda: Investigator.from_foundations(f'{root}/vec/bert_foundation', f'{root}/vec/t5_foundation', verifier,
                                                                         verifier_labels=labels),
        'retrieval': lambda: RetrievalEncoder.from_foundation(f'{root}/vec/bert_foundation', pooling='masked_mean', normalize=True),
        'response_quality': lambda: ResponseQualityAssessor.from_foundation(f'{root}/vec/bert_foundation', max_tokens=8),
    }
    records = {}
    for index, (name, build) in enumerate(flows.items()):
        seed = 500 + index
        torch.manual_seed(seed)
        random.seed(seed)
        records[name] = _record(build(), seed)
    return records


def generate():
    write_json('init_parity.json', {'tools': tools(), 'native': native(), 'diffusers': diffusers(), **sampling(),
                                    'seeded_decode': seeded_decode(), 'foundation_flows': foundation_flows()})
