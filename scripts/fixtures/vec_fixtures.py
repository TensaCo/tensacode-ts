"""Vector operation, graph and Scene parity fixtures (``npm run fixtures -- vec_fixtures``).

Writes Python-saved artifacts that TypeScript must load, evaluate identically
and re-save byte-identically, plus tiny native foundations (BERT, T5, ViT, CLIP)
that TypeScript ``fromFoundation`` imports offline. Everything lives under
``test/fixtures/vec``.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from torch.nn import functional as F

from generate import OUT, tensor_json

VEC = OUT / 'vec'


def _write(name, payload):
    VEC.mkdir(parents=True, exist_ok=True)
    (VEC / name).write_text(json.dumps(payload, separators=(',', ':')) + '\n')


def _fresh(name):
    path = VEC / name
    if path.exists():
        shutil.rmtree(path)
    return path


def _bindings(operation):
    from tensorcode._internal.training.persistence import bindings
    return {name: record['fingerprint'] for name, record in bindings(operation.operation_bindings()).items()}


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({'[PAD]': 0, '[UNK]': 1, 'hello': 2, 'world': 3, 'prefix': 4, 'answer': 5}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]')


def owned():
    from tensorcode.ops.vec import CandidateSet, Classify, Decode, Latent, Score, Space, Transform
    from tensorcode.ops.vec.encode import PatchEncoder, VocabularyEncoder
    S = Space('owned-input', 3)
    O = Space('owned-output', 2)
    records = {}

    torch.manual_seed(21)
    value = torch.randn(2, 3)
    target = torch.randn(2, 2)
    op = Transform({'architecture': 'linear', 'input_space': S.configuration(), 'output_space': O.configuration()})
    op.save_pretrained(_fresh('transform_linear'))
    with torch.no_grad():
        records['transform_linear'] = {
            'value': tensor_json(value), 'target': tensor_json(target),
            'output': tensor_json(op(Latent(value, S)).tensor),
            'loss': op.loss(Latent(value, S), Latent(target, O)).item(),
            'configuration': op.configuration(), 'bindings': _bindings(op),
            'state_keys': list(op.state_dict()),
        }

    op = Classify({'architecture': 'mlp', 'input_space': S.configuration(), 'hidden_dimensions': [4], 'labels': ['a', 'b']})
    op.save_pretrained(_fresh('classify_mlp'))
    with torch.no_grad():
        prediction = op(Latent(value, S))
        records['classify_mlp'] = {
            'value': tensor_json(value), 'logits': tensor_json(prediction.logits), 'values': list(prediction.values),
            'loss': op.loss(Latent(value, S), torch.tensor([0, 1])).item(),
            'single': tensor_json(op(Latent(value[0], S)).logits),
            'configuration': op.configuration(), 'bindings': _bindings(op), 'state_keys': list(op.state_dict()),
        }

    seq = Space('decode-source', 3, organization='sequence')
    op = Decode({'input_space': seq.configuration(), 'output_dimensions': 2, 'output': 'regression values', 'readout': 'sequence'})
    op.save_pretrained(_fresh('decode_sequence'))
    mask = torch.tensor([True, False])
    with torch.no_grad():
        records['decode_sequence'] = {
            'value': tensor_json(value), 'mask': mask.tolist(),
            'output': tensor_json(op(Latent(value, seq, mask=mask))),
            'loss': op.loss(Latent(value, seq, mask=mask), torch.tensor([[1., 1.], [100., 100.]])).item(),
            'configuration': op.configuration(), 'bindings': _bindings(op),
        }

    query = torch.randn(3)
    candidates = torch.randn(4, 3)
    op = Score({'architecture': 'mlp', 'query_space': S.configuration(), 'candidate_space': S.configuration(),
                'hidden_dimensions': [5], 'meaning': 'authored relevance logits'})
    op.save_pretrained(_fresh('score_mlp'))
    candidate_mask = torch.tensor([True, True, False, True])
    scores_target = torch.randn(4)
    with torch.no_grad():
        values = CandidateSet(Latent(query, S), Latent(candidates, S, mask=candidate_mask), ('a', 'b', 'c', 'd'))
        records['score_mlp'] = {
            'query': tensor_json(query), 'candidates': tensor_json(candidates), 'mask': candidate_mask.tolist(),
            'scores': tensor_json(op(values).values), 'target': tensor_json(scores_target),
            'loss': op.loss(values, scores_target).item(),
            'configuration': op.configuration(), 'bindings': _bindings(op), 'state_keys': list(op.state_dict()),
        }

    tokens = Space('tokens', 3, organization='sequence')
    states = Space('states', 2, organization='sequence')
    native = {'model_type': 'bert', 'hidden_size': 4, 'num_hidden_layers': 1, 'num_attention_heads': 2,
              'intermediate_size': 6, 'hidden_dropout_prob': 0.0, 'attention_probs_dropout_prob': 0.0, 'vocab_size': 8}
    op = Transform({'architecture': 'transformer', 'input_space': tokens.configuration(),
                    'output_space': states.configuration(), 'native_config': native}).eval()
    op.save_pretrained(_fresh('transform_bert'))
    x = torch.randn(2, 3)
    prefix = torch.randn(2, 3)
    prefix_mask = torch.tensor([True, False])
    with torch.no_grad():
        records['transform_bert'] = {
            'value': tensor_json(x), 'prefix': tensor_json(prefix), 'prefix_mask': prefix_mask.tolist(),
            'output': tensor_json(op(Latent(x, tokens), context={'latents': [Latent(prefix, tokens, mask=prefix_mask)]}).tensor),
            'plain': tensor_json(op(Latent(x, tokens)).tensor),
            'configuration': op.configuration(), 'bindings': _bindings(op), 'state_keys': list(op.state_dict()),
        }

    op = VocabularyEncoder({'vocabulary': ['hello', 'world'], 'dimensions': 4, 'output_space': Space('words', 4).configuration()})
    op.save_pretrained(_fresh('vocabulary'))
    with torch.no_grad():
        records['vocabulary'] = {
            'texts': ['hello world', 'unknown', '', 'Hello, WORLD!'],
            'output': tensor_json(op(('hello world', 'unknown', '', 'Hello, WORLD!')).tensor),
            'configuration': op.configuration(), 'bindings': _bindings(op),
        }

    op = PatchEncoder({'in_channels': 1, 'patch_size': [2, 3],
                       'output_space': Space('owned-patches', 2, organization='spatial').configuration(),
                       'coordinate_stride': [3, 4], 'coordinate_offset': [-1, 2]})
    op.save_pretrained(_fresh('patch'))
    images = torch.arange(48.).reshape(2, 1, 4, 6) / 48
    with torch.no_grad():
        result = op(images)
        records['patch'] = {
            'images': tensor_json(images), 'output': tensor_json(result.tensor), 'coordinates': tensor_json(result.coordinates),
            'configuration': op.configuration(), 'bindings': _bindings(op),
        }
    default = PatchEncoder({'in_channels': 3, 'patch_size': 2, 'output_space': Space('patches', 4, organization='spatial').configuration()})
    records['patch_default_configuration'] = default.configuration()
    _write('owned.json', records)


def text():
    from transformers import BertConfig, BertModel, T5Config, T5ForConditionalGeneration
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.latent import Latent, Space
    torch.manual_seed(31)
    t5 = _fresh('t5_foundation')
    T5ForConditionalGeneration(T5Config(vocab_size=6, d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=2,
                                        dropout_rate=0., decoder_start_token_id=0, eos_token_id=5, pad_token_id=0)).save_pretrained(t5)
    _tokenizer().save_pretrained(t5)
    bert = _fresh('bert_foundation')
    BertModel(BertConfig(vocab_size=6, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                         max_position_embeddings=8, hidden_dropout_prob=0., attention_probs_dropout_prob=0.)).save_pretrained(bert)
    _tokenizer().save_pretrained(bert)
    texts = ['hello world', 'hello']
    records = {'t5_path': str(t5), 'bert_path': str(bert)}

    encoder = TextEncoder.from_foundation(t5)
    records['encoder_t5_foundation_configuration'] = encoder.configuration()
    encoder.save_pretrained(_fresh('text_encoder_t5'))
    with torch.no_grad():
        result = encoder(texts)
        records['encoder_t5'] = {'output': tensor_json(result.tensor), 'mask': result.mask.tolist(), 'bindings': _bindings(encoder)}

    space = Space('context', 8, organization='sequence')
    encoder = TextEncoder.from_foundation(bert, readout='output_encoding', context_space=space)
    records['encoder_bert_foundation_configuration'] = encoder.configuration()
    encoder.save_pretrained(_fresh('text_encoder_bert'))
    prefix = torch.randn(2, 2, 8)
    prefix_mask = torch.tensor([[True, False], [True, True]])
    with torch.no_grad():
        records['encoder_bert'] = {
            'prefix': tensor_json(prefix), 'prefix_mask': prefix_mask.tolist(),
            'output': tensor_json(encoder(texts, context={'latents': [Latent(prefix, space, mask=prefix_mask)]}).tensor),
            'plain': tensor_json(encoder(texts).tensor), 'bindings': _bindings(encoder),
            'state_keys': list(encoder.state_dict()),
        }

    pooled = TextEncoder.from_foundation(bert, readout='pooled', context_space=space)
    with torch.no_grad():
        records['encoder_bert_pooled'] = {
            'output': tensor_json(pooled(texts, context={'latents': [Latent(prefix, space, mask=prefix_mask)]}).tensor),
            'plain': tensor_json(pooled(texts).tensor),
        }

    decoder = TextDecoder.from_foundation(t5, input_space=Space('arbitrary', 3, organization='sequence'), generation={'max_new_tokens': 3})
    records['decoder_foundation_configuration'] = decoder.configuration()
    decoder.save_pretrained(_fresh('text_decoder_t5'))
    x = torch.randn(2, 2, 3)
    mask = torch.tensor([[True, True], [True, False]])
    with torch.no_grad():
        value = Latent(x, decoder.input_space, mask=mask)
        embeds, packed = decoder._inputs(value, None)
        records['decoder_t5'] = {
            'value': tensor_json(x), 'mask': mask.tolist(),
            'loss': decoder.loss(value, ['answer', 'hello']).item(),
            'generated': decoder(value),
            'sequences': decoder.model.generate(inputs_embeds=embeds, attention_mask=packed, max_new_tokens=3, do_sample=False).tolist(),
            'logits': tensor_json(decoder.model(inputs_embeds=embeds, attention_mask=packed,
                                                decoder_input_ids=torch.tensor([[0, 2, 3], [0, 5, 4]])).logits),
            'bindings': _bindings(decoder), 'state_keys': list(decoder.state_dict()),
        }
    native = TextDecoder.from_foundation(t5, input_space=decoder.native_input_space, bridge='identity', generation={'max_new_tokens': 3})
    with torch.no_grad():
        embedded = native.embed_text(texts)
        records['decoder_identity'] = {
            'embeddings': tensor_json(embedded.tensor), 'mask': embedded.mask.tolist(),
            'generated': native(embedded), 'loss': native.loss(embedded, ['answer', 'hello']).item(),
        }
    _write('text.json', records)


def vision():
    from transformers import ViTConfig, ViTImageProcessor, ViTModel
    from tensorcode.ops.vec.encode import ImageEncoder
    from tensorcode.ops.vec.latent import Latent, Space
    torch.manual_seed(41)
    vit = _fresh('vit_foundation')
    ViTModel(ViTConfig(image_size=8, patch_size=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=16), add_pooling_layer=False).eval().save_pretrained(vit)
    ViTImageProcessor(size={'height': 8, 'width': 8}).save_pretrained(vit)
    records = {'vit_path': str(vit)}
    encoder = ImageEncoder.from_foundation(vit, readout='output_encoding', output_space=Space('visual-readout', 8),
                                           context_space=Space('context', 8, organization='sequence'))
    records['foundation_configuration'] = encoder.configuration()
    encoder.save_pretrained(_fresh('image_encoder'))
    pixels = torch.rand(2, 3, 8, 8)
    prefix = torch.randn(2, 2, 8)
    prefix_mask = torch.tensor([[True, False], [True, True]])
    with torch.no_grad():
        records['output_encoding'] = {
            'pixels': tensor_json(pixels), 'prefix': tensor_json(prefix), 'prefix_mask': prefix_mask.tolist(),
            'output': tensor_json(encoder(pixels, context={'latents': [Latent(prefix, encoder.context_space, mask=prefix_mask)]}).tensor),
            'plain': tensor_json(encoder(pixels).tensor), 'bindings': _bindings(encoder), 'state_keys': list(encoder.state_dict()),
        }
    sequence = ImageEncoder({'model': json.loads(encoder.model.config.to_json_string()), 'processor': encoder.configuration()['processor'],
                             'readout': 'sequence', 'output_space': Space('vision', 8, organization='sequence').configuration()})
    sequence.model.load_state_dict(encoder.model.state_dict())
    sequence.eval()
    with torch.no_grad():
        result = sequence(pixels)
        records['sequence'] = {'output': tensor_json(result.tensor), 'coordinates': tensor_json(result.coordinates)}

    generator = torch.Generator().manual_seed(5)
    image = torch.randint(0, 256, (3, 13, 17), dtype=torch.uint8, generator=generator)
    floats = torch.rand(3, 13, 17, generator=generator)
    processing = {'uint8': image.tolist(), 'float': tensor_json(floats)}
    for resample in (0, 2, 3):
        processor = ViTImageProcessor(size={'height': 8, 'width': 8}, resample=resample, image_mean=[0.1, 0.2, 0.3], image_std=[0.7, 0.8, 0.9])
        processing[f'processor_{resample}'] = {
            'config': json.loads(processor.to_json_string()),
            'pixel_values': tensor_json(processor(images=image, return_tensors='pt')['pixel_values']),
        }
    upscale = ViTImageProcessor(size={'height': 20, 'width': 24})
    processing['upscale'] = {'config': json.loads(upscale.to_json_string()),
                             'pixel_values': tensor_json(upscale(images=image, return_tensors='pt')['pixel_values'])}
    for mode in ('bilinear', 'bicubic'):
        for size in ((8, 8), (20, 24), (9, 30)):
            key = f'interpolate_{mode}_{size[0]}x{size[1]}'
            processing[key] = tensor_json(F.interpolate(floats[None], size=size, mode=mode, align_corners=False, antialias=True)[0])
    _write('vision.json', {**records, 'processing': processing})


def scene():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel
    from tensorcode.tools.scene import Scene
    records = {}
    torch.manual_seed(4)
    tool = Scene({'vocabulary': ['object', 'left', 'of', 'other', 'supported', 'unsupported'], 'dimensions': 8, 'slots': 3})
    generator = torch.Generator().manual_seed(9)
    pixels = torch.rand(3, 16, 16, generator=generator)
    inputs = {'question': 'object left of other', 'source_id': 'photo:1', 'pixels': pixels,
              'candidates': [{'id': 'yes', 'text': 'supported'}, {'id': 'no', 'text': 'unsupported'}]}
    tool.eval()
    with torch.no_grad():
        receipt = tool(inputs)
        loss = tool.loss(inputs, 'yes').item()
    tool.save_pretrained(_fresh('scene_python'))
    records['rank'] = {'pixels': tensor_json(pixels), 'receipt': receipt, 'loss': loss, 'configuration': tool.configuration(),
                       'bindings': _bindings(tool), 'state_keys': list(tool.state_dict())}

    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '[BOS]': 1, 'object': 2, 'left': 3, 'of': 4, 'other': 5, 'supported': 6,
                                            'unsupported': 7, '[EOS]': 15}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single='[BOS] $A [EOS]', special_tokens=[('[BOS]', 1), ('[EOS]', 15)])
    serialized = tokenizer.to_str()
    config = CLIPConfig(text_config={'vocab_size': 16, 'hidden_size': 8, 'intermediate_size': 16, 'num_hidden_layers': 1,
                                     'num_attention_heads': 2, 'max_position_embeddings': 16, 'eos_token_id': 15, 'bos_token_id': 1,
                                     'pad_token_id': 0},
                        vision_config={'image_size': 16, 'patch_size': 8, 'hidden_size': 8, 'intermediate_size': 16,
                                       'num_hidden_layers': 1, 'num_attention_heads': 2}, projection_dim=8)
    torch.manual_seed(6)
    foundation = Scene({'vocabulary': ['<foundation>'], 'dimensions': 8, 'slots': 3, 'foundation_config': json.loads(json.dumps(config.to_dict())),
                        '_tokenizer_json': serialized, 'tokenizer_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
                        'image_mean': [.5] * 3, 'image_std': [.5] * 3, 'patch_size': 8})
    foundation.eval()
    wide = torch.rand(3, 16, 20, generator=generator)
    with torch.no_grad():
        receipt = foundation(inputs)
        wide_receipt = foundation(dict(inputs, pixels=wide))
    foundation.save_pretrained(_fresh('scene_foundation_python'))
    records['foundation'] = {'receipt': receipt, 'wide_pixels': tensor_json(wide), 'wide_receipt': wide_receipt,
                             'configuration': foundation.configuration(), 'bindings': _bindings(foundation),
                             'state_keys': list(foundation.state_dict()),
                             'tokens': foundation.rank.tokens('object left of other').tolist()}

    clip = _fresh('clip_foundation')
    foundation.rank.foundation.save_pretrained(clip)
    (clip / 'tokenizer.json').write_text(serialized, encoding='utf-8')
    CLIPImageProcessor(image_mean=[.5, .5, .5], image_std=[.5, .5, .5]).save_pretrained(clip)
    records['clip_path'] = str(clip)
    _write('scene.json', records)


def foundations():
    """Extra tiny foundations: RoBERTa (padding-offset positions) and untied T5 heads."""
    from transformers import RobertaConfig, RobertaModel, T5Config, T5ForConditionalGeneration
    torch.manual_seed(51)
    roberta = _fresh('roberta_foundation')
    RobertaModel(RobertaConfig(vocab_size=6, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                               max_position_embeddings=8, pad_token_id=0, hidden_dropout_prob=0., attention_probs_dropout_prob=0.)).save_pretrained(roberta)
    _tokenizer().save_pretrained(roberta)
    records = {}
    for scale in (False, True):
        name = f't5_untied_{"scaled" if scale else "unscaled"}'
        config = T5Config(vocab_size=6, d_model=8, d_ff=16, num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0.,
                          decoder_start_token_id=0, eos_token_id=5, pad_token_id=0, tie_word_embeddings=False)
        config.tie_word_embeddings = False
        config.scale_decoder_outputs = scale
        native = T5ForConditionalGeneration(config)
        with torch.no_grad():
            native.shared.weight.fill_(0.25)
            native.lm_head.weight.fill_(0.75)
        native.eval()
        path = _fresh(name)
        native.save_pretrained(path)
        _tokenizer().save_pretrained(path)
        with torch.no_grad():
            records[name] = tensor_json(native(input_ids=torch.tensor([[2, 3]]), decoder_input_ids=torch.tensor([[0, 2]])).logits)
    _write('foundations.json', records)


T5_SNAPSHOT = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
CLIP_CONFIG_SNAPSHOT = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CLIP_WEIGHTS_SNAPSHOT = 'c237dc49a33fc61debc9276459120b7eac67e7ef'


def _snapshot(repo, snapshot):
    from huggingface_hub.constants import HF_HUB_CACHE
    from pathlib import Path
    path = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / 'snapshots' / snapshot
    return path if path.is_dir() else None


def clip_directory():
    """A single local CLIP directory assembled from the cached config and weights snapshots."""
    config = _snapshot('openai/clip-vit-base-patch32', CLIP_CONFIG_SNAPSHOT)
    weights = _snapshot('openai/clip-vit-base-patch32', CLIP_WEIGHTS_SNAPSHOT)
    if config is None or weights is None:
        return None
    import tempfile
    directory = Path(tempfile.mkdtemp(prefix='tensorcode-clip-'))
    for name in ('config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'):
        shutil.copy(config / name, directory / name)
    shutil.copy(weights / 'model.safetensors', directory / 'model.safetensors')
    return directory


def cached():
    """Real cached foundations (skipped when absent); records outputs only, never weights."""
    from tensorcode.ops.vec.decode import TextDecoder
    from tensorcode.ops.vec.encode import TextEncoder
    from tensorcode.ops.vec.latent import Space
    from tensorcode.tools.scene import Scene
    records = {}
    if _snapshot('google/flan-t5-small', T5_SNAPSHOT) is not None:
        prompts = ['translate English to German: The house is wonderful.', 'What is the capital of France?']
        options = dict(revision=T5_SNAPSHOT, local_files_only=True)
        probe = TextDecoder.from_foundation('google/flan-t5-small', input_space=Space('unused', 4, organization='sequence'), **options)
        decoder = TextDecoder.from_foundation('google/flan-t5-small', input_space=probe.native_input_space, bridge='identity',
                                              generation={'max_new_tokens': 12}, **options)
        encoder = TextEncoder.from_foundation('google/flan-t5-small', readout='pooled', **options)
        with torch.no_grad():
            embedded = decoder.embed_text(prompts)
            pooled = encoder(prompts).tensor
            config = decoder.configuration()
            records['flan_t5_small'] = {
                'snapshot': T5_SNAPSHOT, 'prompts': prompts, 'generated': decoder(embedded),
                'loss': decoder.loss(embedded, ['Das Haus ist wunderbar.', 'Paris']).item(),
                'pooled': tensor_json(pooled[:, :16]), 'pooled_abs_mean': pooled.abs().mean().item(),
                'decoder_configuration': {k: v for k, v in config.items() if k != 'tokenizer'},
                'encoder_output_space': encoder.configuration()['output_space'],
            }
    directory = clip_directory()
    if directory is not None:
        torch.manual_seed(3)
        scene = Scene.from_foundation(str(directory), revision=None, local_files_only=True, dimensions=8)
        generator = torch.Generator().manual_seed(12)
        pixels = torch.rand(3, 40, 56, generator=generator)
        with torch.no_grad():
            patches, image_global = scene.rank._encode_image(pixels)
            question, question_global = scene.rank._encode_text('a photo of a cat')
        config = scene.configuration()
        records['clip'] = {
            'pixels': tensor_json(pixels), 'image_global': tensor_json(image_global), 'patch_head': tensor_json(patches[:3, :8]),
            'question_global': tensor_json(question_global), 'tokens': scene.rank.tokens('a photo of a cat').tolist(),
            'configuration': {k: v for k, v in config.items() if k not in ('tokenizer_sha256',)},
        }
        shutil.rmtree(directory)
    _write('cached.json', records)


def generate():
    VEC.mkdir(parents=True, exist_ok=True)
    cached()
    foundations()
    owned()
    text()
    vision()
    scene()
