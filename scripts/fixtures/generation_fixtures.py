"""transformers ``generate`` parity for the owned Idefics3 text model: decoding strategies,
logits processors, warpers, stopping criteria and their errors (``src/_internal/native/causalGeneration.ts``)."""
from __future__ import annotations

import copy
import json
import warnings

import torch
from safetensors.torch import save_model
from transformers import GenerationConfig, Idefics3Config, Idefics3ForConditionalGeneration
from transformers.generation import logits_process as lp
from transformers.utils import logging

import math

from generate import OUT, write_json
from generate import tensor_json as _tensor_json


def tensor_json(value):
    """``tensor_json`` with non-finite floats as strings (strict JSON)."""
    record = _tensor_json(value)
    record['data'] = [('nan' if math.isnan(x) else 'inf' if x > 0 else '-inf') if isinstance(x, float) and not math.isfinite(x) else x
                      for x in record['data']]
    return record

ROOT = OUT / 'generation'

CONFIG = dict(
    vision_config=dict(hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=2),
    text_config=dict(model_type='llama', hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, vocab_size=64, pad_token_id=0, rope_theta=10000, max_position_embeddings=128),
    pad_token_id=0, scale_factor=2, image_token_id=60,
)
MODEL_GENERATION = dict(bos_token_id=1, eos_token_id=2, pad_token_id=0)
PROMPT = [1, 5, 6, 61, 60, 60, 60, 60, 61, 7, 8, 9, 10]
SHORT = [1, 11, 12, 13]


def build():
    torch.manual_seed(1234)
    model = Idefics3ForConditionalGeneration(Idefics3Config(**copy.deepcopy(CONFIG))).eval()
    # Wider weights make continuations depend on the context instead of repeating one token.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and 'norm' not in name:
                parameter.normal_(0, 0.3)
    return model


def error_record(exc):
    return {'error': type(exc).__name__, 'message': str(exc)}


def run_case(model, features, pixels, case):
    settings = dict(case.get('settings', {}))
    record = {'name': case['name']}
    try:
        model.generation_config = GenerationConfig(**{**MODEL_GENERATION, **case.get('generation', {})})
    except Exception as exc:  # noqa: BLE001 - errors are part of the fixture
        record.update(error_record(exc))
        record['stage'] = 'config'
        return record
    inputs = case.get('inputs', 'states')
    kwargs = {}
    if inputs == 'states':
        ids = torch.tensor([case.get('prompt', PROMPT)])
        kwargs['image_hidden_states'] = features
    elif inputs == 'pixels':
        ids = torch.tensor([case.get('prompt', PROMPT)])
        kwargs['pixel_values'] = pixels
    elif inputs == 'text':
        ids = torch.tensor([case.get('prompt', SHORT)])
    elif inputs == 'batch':
        ids = torch.tensor([[0, 0, 0, 1, 11, 12, 13], [1, 14, 15, 16, 17, 18, 19]])
        kwargs['attention_mask'] = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    else:
        raise ValueError(inputs)
    if 'negative' in case:
        kwargs['negative_prompt_ids'] = torch.tensor(case['negative'])
    if 'seed' in case:
        torch.manual_seed(case['seed'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            output = model.generate(input_ids=ids, **kwargs, **settings)
    except Exception as exc:  # noqa: BLE001 - errors are part of the fixture
        record.update(error_record(exc))
        return record
    if isinstance(output, torch.Tensor):
        record['sequences'] = output.tolist()
    else:
        record['sequences'] = output.sequences.tolist()
        if getattr(output, 'scores', None) is not None:
            record['scores'] = [tensor_json(step) for step in output.scores]
        if getattr(output, 'logits', None) is not None:
            record['logits'] = [tensor_json(step) for step in output.logits]
        if getattr(output, 'sequences_scores', None) is not None:
            record['sequences_scores'] = output.sequences_scores.tolist()
        if getattr(output, 'beam_indices', None) is not None:
            record['beam_indices'] = output.beam_indices.tolist()
    return record


CASES = [
    {'name': 'greedy', 'settings': {'max_new_tokens': 8}},
    {'name': 'greedy_pixels', 'inputs': 'pixels', 'settings': {'max_new_tokens': 6}},
    {'name': 'greedy_text', 'inputs': 'text', 'settings': {'max_new_tokens': 6}},
    {'name': 'default_length', 'inputs': 'text', 'settings': {}},
    {'name': 'max_length', 'inputs': 'text', 'settings': {'max_length': 9}},
    {'name': 'no_cache', 'settings': {'max_new_tokens': 6, 'use_cache': False}},
    {'name': 'chunked_prefill', 'settings': {'max_new_tokens': 6, 'prefill_chunk_size': 5}},
    {'name': 'repetition', 'settings': {'max_new_tokens': 8, 'repetition_penalty': 1.7}},
    {'name': 'no_repeat_ngram', 'settings': {'max_new_tokens': 10, 'no_repeat_ngram_size': 2}},
    {'name': 'encoder_repetition', 'settings': {'max_new_tokens': 8, 'encoder_repetition_penalty': 1.8}},
    {'name': 'encoder_no_repeat', 'settings': {'max_new_tokens': 8, 'encoder_no_repeat_ngram_size': 1}},
    {'name': 'encoder_no_repeat2', 'inputs': 'text', 'prompt': [1, 49, 23, 11, 51, 49, 23], 'settings': {'max_new_tokens': 8, 'encoder_no_repeat_ngram_size': 2}},
    {'name': 'bad_words', 'settings': {'max_new_tokens': 8, 'bad_words_ids': [[2], [15, 15], [4, 25], [28, 15, 4]]}},
    {'name': 'min_length', 'generation': {'eos_token_id': [25, 28]}, 'settings': {'max_new_tokens': 8, 'min_length': 18}},
    {'name': 'min_new_tokens', 'generation': {'eos_token_id': [25, 28]}, 'settings': {'max_new_tokens': 8, 'min_new_tokens': 5}},
    {'name': 'eos_list', 'generation': {'eos_token_id': [25, 28]}, 'settings': {'max_new_tokens': 8}},
    {'name': 'forced', 'settings': {'max_new_tokens': 5, 'forced_eos_token_id': 2}, 'inputs': 'text', 'prompt': [1]},
    {'name': 'forced_bos', 'settings': {'max_new_tokens': 4, 'forced_bos_token_id': 33}, 'inputs': 'text', 'prompt': [1]},
    {'name': 'suppress', 'settings': {'max_new_tokens': 8, 'suppress_tokens': [2, 4, 60], 'begin_suppress_tokens': [15, 9]}},
    {'name': 'remove_invalid', 'settings': {'max_new_tokens': 6, 'remove_invalid_values': True, 'suppress_tokens': [3]}},
    {'name': 'exponential_decay', 'settings': {'max_new_tokens': 8, 'exponential_decay_length_penalty': [2, 3.5]}},
    {'name': 'renormalize', 'settings': {'max_new_tokens': 6, 'renormalize_logits': True, 'return_dict_in_generate': True, 'output_scores': True}},
    {'name': 'sequence_bias', 'settings': {'max_new_tokens': 8, 'sequence_bias': [[[25], 9.5], [[25, 26], 30.0], [[25, 26, 27], -12.0], [[40, 41, 42, 43, 44, 45, 46, 47], 5.0]]}},
    {'name': 'sequence_bias_duplicates', 'settings': {'max_new_tokens': 6, 'sequence_bias': [[[25], 9.5], [[25], -20.0], [[31], 4.0]]}},
    {'name': 'guidance', 'settings': {'max_new_tokens': 8, 'guidance_scale': 1.5}},
    {'name': 'guidance_low', 'settings': {'max_new_tokens': 8, 'guidance_scale': 0.5}},
    {'name': 'guidance_negative', 'settings': {'max_new_tokens': 8, 'guidance_scale': 3.0}, 'negative': [[1, 30, 31]]},
    {'name': 'guidance_no_cache', 'settings': {'max_new_tokens': 6, 'guidance_scale': 2.0, 'use_cache': False}},
    {'name': 'scores', 'settings': {'max_new_tokens': 5, 'return_dict_in_generate': True, 'output_scores': True, 'output_logits': True, 'repetition_penalty': 1.2}},
    {'name': 'beam2', 'settings': {'max_new_tokens': 8, 'num_beams': 2}},
    {'name': 'beam3_return', 'settings': {'max_new_tokens': 8, 'num_beams': 3, 'num_return_sequences': 3}},
    {'name': 'beam4_lp0', 'settings': {'max_new_tokens': 8, 'num_beams': 4, 'length_penalty': 0.0, 'num_return_sequences': 2}},
    {'name': 'beam4_lp2', 'settings': {'max_new_tokens': 8, 'num_beams': 4, 'length_penalty': 2.0, 'num_return_sequences': 4}},
    {'name': 'beam3_negative_lp', 'settings': {'max_new_tokens': 8, 'num_beams': 3, 'length_penalty': -1.5, 'num_return_sequences': 3}},
    {'name': 'beam3_early', 'settings': {'max_new_tokens': 8, 'num_beams': 3, 'early_stopping': True, 'num_return_sequences': 3}},
    {'name': 'beam3_never', 'settings': {'max_new_tokens': 8, 'num_beams': 3, 'early_stopping': 'never', 'num_return_sequences': 3}},
    {'name': 'beam_eos_list', 'generation': {'eos_token_id': [2, 3, 4]}, 'settings': {'max_new_tokens': 8, 'num_beams': 3, 'num_return_sequences': 2}},
    {'name': 'beam_no_eos', 'generation': {'eos_token_id': None}, 'settings': {'max_new_tokens': 5, 'num_beams': 2}},
    {'name': 'beam_scores', 'settings': {'max_new_tokens': 6, 'num_beams': 3, 'num_return_sequences': 2, 'return_dict_in_generate': True, 'output_scores': True, 'output_logits': True}},
    {'name': 'beam_guidance', 'settings': {'max_new_tokens': 6, 'num_beams': 3, 'guidance_scale': 1.5}},
    {'name': 'beam_bias', 'settings': {'max_new_tokens': 6, 'num_beams': 2, 'sequence_bias': [[[2], 6.0]], 'repetition_penalty': 1.3}},
    {'name': 'beam_encoder_penalty', 'settings': {'max_new_tokens': 6, 'num_beams': 3, 'encoder_repetition_penalty': 2.0}},
    {'name': 'beam_pixels', 'inputs': 'pixels', 'settings': {'max_new_tokens': 6, 'num_beams': 2}},
    {'name': 'beam_no_cache', 'settings': {'max_new_tokens': 5, 'num_beams': 2, 'use_cache': False}},
    {'name': 'batch_greedy', 'inputs': 'batch', 'settings': {'max_new_tokens': 6}},
    {'name': 'batch_beam', 'inputs': 'batch', 'settings': {'max_new_tokens': 6, 'num_beams': 3, 'num_return_sequences': 2}},
    {'name': 'prompt_lookup', 'inputs': 'text', 'prompt': [1, 20, 21, 22, 23, 20, 21], 'settings': {'max_new_tokens': 10, 'prompt_lookup_num_tokens': 3}},
    {'name': 'prompt_lookup_image', 'settings': {'max_new_tokens': 8, 'prompt_lookup_num_tokens': 2, 'max_matching_ngram_size': 1}},
    {'name': 'prompt_lookup_bias', 'inputs': 'text', 'prompt': [1, 20, 21, 22, 23, 20, 21], 'settings': {'max_new_tokens': 8, 'prompt_lookup_num_tokens': 4, 'sequence_bias': [[[22], 50.0], [[21, 22], 50.0], [[22, 23], 60.0]]}},
    {'name': 'watermark_left', 'settings': {'max_new_tokens': 8, 'watermarking_config': {'greenlist_ratio': 0.25, 'bias': 4.0, 'hashing_key': 15485863, 'seeding_scheme': 'lefthash', 'context_width': 1}}},
    {'name': 'watermark_self', 'settings': {'max_new_tokens': 6, 'watermarking_config': {'greenlist_ratio': 0.5, 'bias': 6.0, 'hashing_key': 15485863, 'seeding_scheme': 'selfhash', 'context_width': 2}}},
    {'name': 'watermark_context', 'settings': {'max_new_tokens': 6, 'watermarking_config': {'greenlist_ratio': 0.4, 'bias': 3.0, 'hashing_key': 97, 'seeding_scheme': 'lefthash', 'context_width': 3}}},
    {'name': 'max_time_zero', 'settings': {'max_new_tokens': 6, 'max_time': 0.0}},
    {'name': 'cache_static', 'settings': {'max_new_tokens': 6, 'cache_implementation': 'static'}},
    {'name': 'sample', 'seed': 7, 'settings': {'max_new_tokens': 8, 'do_sample': True}},
    {'name': 'sample_warped', 'seed': 11, 'settings': {'max_new_tokens': 8, 'do_sample': True, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.9}},
    {'name': 'sample_min_p', 'seed': 3, 'settings': {'max_new_tokens': 8, 'do_sample': True, 'min_p': 0.05, 'top_k': 0}},
    {'name': 'sample_typical', 'seed': 5, 'settings': {'max_new_tokens': 8, 'do_sample': True, 'typical_p': 0.8, 'top_k': 0}},
    {'name': 'sample_epsilon_eta', 'seed': 9, 'settings': {'max_new_tokens': 8, 'do_sample': True, 'epsilon_cutoff': 0.01, 'eta_cutoff': 0.02}},
    {'name': 'sample_top_h', 'seed': 13, 'settings': {'max_new_tokens': 8, 'do_sample': True, 'top_h': 0.6, 'top_k': 0}},
    {'name': 'sample_return', 'seed': 17, 'settings': {'max_new_tokens': 6, 'do_sample': True, 'num_return_sequences': 3}},
    {'name': 'beam_sample', 'seed': 19, 'settings': {'max_new_tokens': 6, 'do_sample': True, 'num_beams': 3, 'num_return_sequences': 2}},
    {'name': 'sample_prompt_lookup', 'seed': 23, 'inputs': 'text', 'prompt': [1, 20, 21, 22, 23, 20, 21], 'settings': {'max_new_tokens': 8, 'do_sample': True, 'prompt_lookup_num_tokens': 3}},
    # Errors raised by transformers.
    {'name': 'err_contrastive', 'settings': {'max_new_tokens': 4, 'penalty_alpha': 0.6, 'top_k': 4}},
    {'name': 'err_group_beam', 'settings': {'max_new_tokens': 4, 'num_beams': 4, 'num_beam_groups': 2, 'diversity_penalty': 1.0}},
    {'name': 'err_constrained', 'settings': {'max_new_tokens': 4, 'num_beams': 2, 'force_words_ids': [[5]]}},
    {'name': 'err_dola', 'settings': {'max_new_tokens': 4, 'dola_layers': 'high'}},
    {'name': 'err_return_greedy', 'settings': {'max_new_tokens': 4, 'num_return_sequences': 2}},
    {'name': 'err_return_beams', 'settings': {'max_new_tokens': 4, 'num_beams': 2, 'num_return_sequences': 3}},
    {'name': 'err_low_memory', 'settings': {'max_new_tokens': 4, 'num_beams': 2, 'low_memory': True}},
    {'name': 'err_early_stopping', 'settings': {'max_new_tokens': 4, 'early_stopping': 'sometimes'}},
    {'name': 'err_bias_zero', 'settings': {'max_new_tokens': 4, 'sequence_bias': [[[0], 1.0]]}},
    {'name': 'err_bias_vocab', 'settings': {'max_new_tokens': 4, 'sequence_bias': [[[64], 1.0]]}},
    {'name': 'err_bad_words', 'settings': {'max_new_tokens': 4, 'bad_words_ids': [[-1]]}},
    {'name': 'err_forced_suppressed', 'settings': {'max_new_tokens': 4, 'forced_eos_token_id': [2, 3], 'suppress_tokens': [2, 3, 5]}},
    {'name': 'err_stop_strings', 'settings': {'max_new_tokens': 4, 'stop_strings': ['x']}},
    {'name': 'err_token_healing', 'settings': {'max_new_tokens': 4, 'token_healing': True}},
    {'name': 'err_offloaded', 'settings': {'max_new_tokens': 4, 'cache_implementation': 'offloaded'}},
    {'name': 'err_quantized', 'settings': {'max_new_tokens': 4, 'cache_implementation': 'quantized'}},
    {'name': 'err_cache_name', 'settings': {'max_new_tokens': 4, 'cache_implementation': 'mystery'}},
    {'name': 'err_early_exit', 'settings': {'max_new_tokens': 4, 'assistant_early_exit': 1}},
    {'name': 'err_mtp', 'settings': {'max_new_tokens': 4, 'use_mtp': True}},
    {'name': 'err_ensemble_lookup', 'settings': {'max_new_tokens': 4, 'prompt_lookup_num_tokens': 2, 'assistant_ensemble_weight': 0.5}},
    {'name': 'err_ensemble_range', 'settings': {'max_new_tokens': 4, 'assistant_ensemble_weight': 1.5}},
    {'name': 'err_selfhash_small', 'settings': {'max_new_tokens': 4, 'watermarking_config': {'seeding_scheme': 'selfhash', 'greenlist_ratio': 0.1}}},
    {'name': 'err_watermark_scheme', 'settings': {'max_new_tokens': 4, 'watermarking_config': {'seeding_scheme': 'nohash'}}},
    {'name': 'err_watermark_key', 'settings': {'max_new_tokens': 4, 'watermarking_config': {'salt': 3}}},
    {'name': 'err_temperature', 'settings': {'max_new_tokens': 4, 'do_sample': True, 'temperature': 0.0}},
    {'name': 'err_top_p', 'seed': 29, 'settings': {'max_new_tokens': 4, 'do_sample': True, 'top_p': 1.5}},
    {'name': 'err_top_h', 'settings': {'max_new_tokens': 4, 'do_sample': True, 'top_h': 0.0}},
    {'name': 'err_max_length', 'inputs': 'text', 'settings': {'max_length': 3}},
    {'name': 'err_max_new_tokens', 'settings': {'max_new_tokens': 0}},
    {'name': 'err_model_kwargs', 'settings': {'max_new_tokens': 4, 'not_a_setting': 3}},
    {'name': 'err_generate_argument', 'generation': {'streamer': 1}, 'settings': {'max_new_tokens': 4}},
    {'name': 'err_compile_config', 'settings': {'max_new_tokens': 4, 'compile_config': {'fullgraph': True}}},
    {'name': 'err_exponential_no_eos', 'generation': {'eos_token_id': None}, 'settings': {'max_new_tokens': 4, 'exponential_decay_length_penalty': [1, 2.0]}},
]


def processor_records():
    """Individual processors and warpers on fixed score rows (float32)."""
    generator = torch.Generator().manual_seed(99)
    scores = torch.randn(3, 40, generator=generator) * 3
    scores[1, 5] = scores[1, 7]  # a tie
    ids = torch.tensor([[1, 5, 5, 9, 12, 5], [1, 7, 30, 7, 30, 7], [2, 3, 4, 5, 6, 3]])
    specs = {
        'repetition': lp.RepetitionPenaltyLogitsProcessor(1.3),
        'temperature': lp.TemperatureLogitsWarper(0.7),
        'top_k': lp.TopKLogitsWarper(5),
        'top_k_keep': lp.TopKLogitsWarper(1, min_tokens_to_keep=3),
        'top_p': lp.TopPLogitsWarper(0.8),
        'top_p_keep': lp.TopPLogitsWarper(0.1, min_tokens_to_keep=4),
        'top_h': lp.TopHLogitsWarper(0.5),
        'min_p': lp.MinPLogitsWarper(0.1),
        'min_p_keep': lp.MinPLogitsWarper(0.9, min_tokens_to_keep=3),
        'typical': lp.TypicalLogitsWarper(0.7),
        'typical_keep': lp.TypicalLogitsWarper(0.2, min_tokens_to_keep=5),
        'epsilon': lp.EpsilonLogitsWarper(0.02),
        'eta': lp.EtaLogitsWarper(0.03),
        'no_repeat': lp.NoRepeatNGramLogitsProcessor(2),
        'min_length': lp.MinLengthLogitsProcessor(8, [2, 3]),
        'min_new_tokens': lp.MinNewTokensLengthLogitsProcessor(2, 5, [4]),
        'infnan': lp.InfNanRemoveLogitsProcessor(),
        'normalize': lp.LogitNormalization(),
        'exponential': lp.ExponentialDecayLengthPenalty((1, 1.5), [2, 3], 3),
        'suppress': lp.SuppressTokensLogitsProcessor([1, 2, 39]),
        'suppress_begin': lp.SuppressTokensAtBeginLogitsProcessor([4, 5], 6),
        'sequence_bias': lp.SequenceBiasLogitsProcessor([[[5], 2.5], [[5, 9], -3.0], [[30, 7], 1.25], [[9, 12, 5], 0.5]]),
        'bad_words': lp.NoBadWordsLogitsProcessor([[3], [7, 30], [5, 5, 9], [2]], [2]),
        'watermark_left': lp.WatermarkLogitsProcessor(40, 'cpu', greenlist_ratio=0.25, bias=2.0),
        'watermark_self': lp.WatermarkLogitsProcessor(40, 'cpu', greenlist_ratio=0.5, bias=1.5, seeding_scheme='selfhash', context_width=3),
    }
    special = scores.clone()
    special[0, 3] = float('inf')
    special[1, 4] = float('-inf')
    special[2, 6] = float('nan')
    records = {'scores': tensor_json(scores), 'ids': ids.tolist(), 'special': tensor_json(special), 'outputs': {}}
    for name, processor in specs.items():
        source = special if name == 'infnan' else scores
        records['outputs'][name] = tensor_json(processor(ids, source.clone()))
    records['encoder'] = {
        'penalty': tensor_json(lp.EncoderRepetitionPenaltyLogitsProcessor(1.6, torch.tensor([[1, 9, 20, 20]]))(ids, scores.clone())),
        'ngram': tensor_json(lp.EncoderNoRepeatNGramLogitsProcessor(2, torch.tensor([[5, 9, 12, 5, 30, 1, 7, 33]]))(ids, scores.clone())),
    }
    table = torch.Generator().manual_seed(15485863)
    records['randperm'] = {'seed': 15485863, 'n': 50, 'values': torch.randperm(50, generator=table).tolist()}
    return records


def generate():
    logging.set_verbosity_error()
    ROOT.mkdir(parents=True, exist_ok=True)
    model = build()
    save_model(model, str(ROOT / 'idefics3_generation.safetensors'))
    torch.manual_seed(5)
    pixels = torch.zeros(1, 1, 3, 8, 8)
    pixels[0, 0] = torch.rand(3, 8, 8) * 2 - 1
    with torch.no_grad():
        features = model.model.get_image_features(pixels, None, return_dict=True).pooler_output
    records = {
        'config': json.loads(json.dumps(model.config.to_dict())),
        'model_generation': MODEL_GENERATION, 'prompt': PROMPT, 'short': SHORT,
        'pixel_values': tensor_json(pixels), 'image_hidden_states': tensor_json(features),
        'cases': [run_case(model, features, pixels, case) for case in CASES],
        'case_specs': CASES,
        'processors': processor_records(),
    }
    write_json('generation/records.json', records)
