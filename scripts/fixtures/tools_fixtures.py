"""Tools module parity fixtures: tiny Chatbot/Investigator/Planner/Decision artifacts,
receipts, proposals, sessions and persisted runtime records written by Python.

Everything uses tiny random models (mechanism checks, not quality). The cached
Hub investigator (``jacob-valdez/tensorcode-investigator-hotpot-001``) is only
read when present locally; its receipts are recorded for the acceptance test.
"""
from __future__ import annotations

import copy
import json
import shutil

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, T5Config

from generate import OUT, tensor_json, write_json
from tensorcode._internal.training.persistence import configuration as op_configuration, fingerprint as op_fingerprint
from tensorcode._internal.proposals import proposal_prompt
from tensorcode._internal.sessions.ranking import RankingSession
from tensorcode._internal.cognition.state import CognitiveState
from tensorcode._internal.execution.planning import ExecutablePlan, PlanExecutor, PlanStep
from tensorcode._internal.memory.json import JsonMemory
from tensorcode._internal.retrieval import RetrievalEncoder
from tensorcode.tools.actions import ActionOutcome
from tensorcode.tools.chatbot import Chatbot, _bounded_memory_update
from tensorcode.tools.cognition import Assessment, Evidence, Goal, Hypothesis, Observation, Plan
from tensorcode.tools.decision import Decision
from tensorcode.tools.investigator import Investigator
from tensorcode.tools.planner import Planner

TOOLS = OUT / 'tools'
HOTPOT_REPO = 'jacob-valdez/tensorcode-investigator-hotpot-001'
HOTPOT_REVISION = '1bc225917c3646fcb9702df91ff5e445846c1dc7'


def tiny_config():
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '</s>': 1, '<unk>': 2, 'user': 3, ':': 4, 'hello': 5,
                                            'world': 6, 'answer': 7}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    foundation = T5Config(vocab_size=8, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=8,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, dropout_rate=0.0)
    return {'foundation_config': foundation.to_dict(), 'tokenizer_json': tokenizer.to_str(),
            'tokenizer_special_tokens': {'pad_token': '<pad>', 'eos_token': '</s>', 'unk_token': '<unk>'},
            'workspace': {'slots': 3, 'steps': 2}, 'max_new_tokens': 3, 'max_input_tokens': 32, 'max_turns': 2}


def investigator_config():
    generator = tiny_config()
    return {'vocabulary': ['hello', 'world'], 'dimensions': 8, 'slots': 2, 'steps': 1, 'generator': generator,
            'verifier_config': BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                                          intermediate_size=16, num_labels=3).to_dict(),
            'verifier_tokenizer_json': generator['tokenizer_json'],
            'verifier_tokenizer_special_tokens': generator['tokenizer_special_tokens'],
            'verifier_labels': {'support': 2, 'contradiction': 0, 'unknown': 1}}


def cognition_investigator_config():
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '<unk>': 1, 'alpha': 2, 'beta': 3}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return {'vocabulary': ['alpha', 'beta'], 'dimensions': 8, 'slots': 2, 'steps': 1,
            'verifier_config': BertConfig(vocab_size=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                                          intermediate_size=16, num_labels=3).to_dict(),
            'verifier_tokenizer_json': tokenizer.to_str(),
            'verifier_tokenizer_special_tokens': {'pad_token': '<pad>', 'unk_token': '<unk>'},
            'verifier_labels': {'support': 0, 'contradiction': 1, 'unknown': 2}}


def retrieval_config():
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '<unk>': 1, 'alpha': 2, 'beta': 3}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return {'foundation_config': BertConfig(vocab_size=4, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                                            intermediate_size=16, hidden_dropout_prob=0.0,
                                            attention_probs_dropout_prob=0.0).to_dict(),
            'tokenizer_json': tokenizer.to_str(), 'tokenizer_special_tokens': {'pad_token': '<pad>', 'unk_token': '<unk>'},
            'pooling': 'masked_mean', 'normalize': True, 'max_tokens': 4}


INPUT = {'question': 'hello', 'evidence': [{'source_id': 'a', 'text': 'world'}, {'source_id': 'b', 'text': 'hello'}]}


def bindings(tool):
    return {name: op_fingerprint(op_configuration(operation)) for name, operation in tool.operation_bindings().items()}


def save(tool, name):
    target = TOOLS / name
    if target.exists():
        shutil.rmtree(target)
    tool.save_pretrained(target)
    (target / 'README.md').unlink()


def chatbot_fixtures():
    torch.manual_seed(1)
    model = Chatbot(tiny_config()).eval()
    save(model, 'chatbot')
    with torch.no_grad():
        losses = {
            'single': model.loss_batch(['hello world'], ['answer']).item(),
            'batch': model.loss_batch(['hello', 'hello world user'], ['answer', 'world answer']).item(),
            'bypass': model.loss_batch(['hello'], ['answer'], workspace_ablation='bypass').item(),
            'zero': model.loss_batch(['hello'], ['answer'], workspace_ablation='zero').item(),
        }
        encoded = model.encoder(['hello', 'hello world'])
        state = model.encode_workspace(['hello', 'hello world'])
        tokens = model.decoder(model.encode_workspace(['hello world']), context={'max_new_tokens': 3, 'do_sample': False})
        beams = model.decoder(model.encode_workspace(['hello world']), context={
            'max_new_tokens': 3, 'do_sample': False, 'num_beams': 3, 'num_return_sequences': 3})
    generations = model.generate_batch(['hello', 'hello world', 'user : hello'])
    session = model.new_session()
    session('hello')
    session('world answer')
    session.save(TOOLS / 'chatbot_session.json')
    return {
        'config': tiny_config(), 'configuration': model.configuration(), 'fingerprint': model.fingerprint,
        'state_keys': list(model.state_dict().keys()), 'bindings': bindings(model),
        'encoder_configuration': model.encoder.configuration(), 'decoder_configuration': model.decoder.configuration(),
        'losses': losses, 'encoded': tensor_json(encoded['encoded']), 'conditioning': tensor_json(state['conditioning']),
        'greedy_tokens': tokens.tolist(), 'beam_tokens': beams.tolist(), 'generations': generations,
        'session_history': session.history, 'session_last_result': session.last_result,
    }


def investigator_fixtures():
    torch.manual_seed(2)
    tool = Investigator(investigator_config()).eval()
    save(tool, 'investigator')
    conversation = dict(INPUT, conversation_context=[{'role': 'user', 'text': 'which café ☕ incident 𝄞?'},
                                                     {'role': 'assistant', 'text': 'unverified "quoted" answer'}])
    prompts = {
        'plain': proposal_prompt(INPUT, 'question'),
        'goal': proposal_prompt({'goal': 'hello', 'evidence': INPUT['evidence']}, 'goal'),
        'conversation': proposal_prompt(conversation, 'question'),
        'unicode': proposal_prompt({'question': 'naïve café — ☕?', 'evidence': [{'source_id': 'ü', 'text': 'ß\n"x"'}]}, 'question'),
        'v2': proposal_prompt(INPUT, 'question', template_version=2),
    }
    proposals = tool.propose(INPUT, count=3)
    single = tool.propose(INPUT, count=1)
    pairs = [{'premise': 'world', 'hypothesis': 'hello'}, {'premise': 'hello world', 'hypothesis': 'world'}]
    with torch.no_grad():
        logits = tool.verifier(pairs)
    verified = tool.verifier.verify('hello', INPUT['evidence'])
    joint = tool.verifier.verify_joint('hello', INPUT['evidence'])
    receipt = tool.investigate(dict(INPUT, hypotheses=[{'id': 'h1', 'text': 'hello'}, {'id': 'h2', 'text': 'world'}]))
    with torch.no_grad():
        proposal_loss = tool.proposal_loss(INPUT, 'answer').item()
    session = tool.new_session()
    session(dict(INPUT, hypotheses=[{'id': 'h1', 'text': 'hello'}, {'id': 'h2', 'text': 'world'}]))
    session(copy.deepcopy(INPUT))
    session.save(TOOLS / 'investigator_session.json')
    calibrated = Investigator(investigator_config()).eval()
    calibrated.load_state_dict(tool.state_dict())
    calibrated.verifier.calibration.fit(torch.tensor([[5., 0., 0.], [5., 0., 0.]]), torch.tensor([0, 1]))
    return {
        'config': investigator_config(), 'configuration': tool.configuration(), 'state_keys': list(tool.state_dict().keys()),
        'bindings': bindings(tool), 'prompts': prompts, 'proposals': proposals, 'single_proposal': single,
        'verifier_logits': tensor_json(logits), 'verifications': verified, 'joint_verification': joint, 'receipt': receipt,
        'proposal_loss': proposal_loss, 'session_history': session.history,
        'calibration_digest': calibrated.verifier.calibration_weight_digest.tolist(),
        'calibration_temperature': calibrated.verifier.calibration.temperature.item(),
        'generator_fingerprint': tool.generator.fingerprint,
    }


def planner_fixtures():
    torch.manual_seed(3)
    planner = Planner({'vocabulary': ['goal', 'step', 'hello'], 'dimensions': 8, 'generator': tiny_config()}).eval()
    save(planner, 'planner')
    inputs = {'goal': 'hello', 'evidence': [{'source_id': 's1', 'text': 'hello step'}],
              'plans': [{'id': 'a', 'text': 'step hello'}, {'id': 'b', 'text': 'goal'}]}
    torch.manual_seed(4)
    decision = Decision({'vocabulary': ['red', 'blue', 'find'], 'dimensions': 8, 'slots': 2, 'steps': 1}).eval()
    save(decision, 'decision')
    case = {'question': 'find red', 'evidence': [{'source_id': 'x', 'text': 'red'}],
            'hypotheses': [{'id': 'r', 'text': 'red'}, {'id': 'b', 'text': 'blue'}]}
    return {
        'configuration': planner.configuration(), 'state_keys': list(planner.state_dict().keys()), 'bindings': bindings(planner),
        'inputs': inputs, 'receipt': planner(inputs), 'proposals': planner.propose({'goal': 'hello'}, count=2),
        'decision': {'configuration': decision.configuration(), 'case': case, 'receipt': decision(case), 'bindings': bindings(decision)},
    }


def cognition_fixtures():
    torch.manual_seed(5)
    tool = Investigator(cognition_investigator_config()).eval()
    with torch.no_grad():
        tool.verifier.model.classifier.weight.zero_()
        tool.verifier.model.classifier.bias.copy_(torch.tensor([8., 0., 0.]))
    save(tool, 'cognitive_investigator')
    session = tool.new_cognitive_session(memory={'capacity': 4, 'top_k': 2}, max_records=64)
    session.ingest([Evidence('a', 'alpha', 'doc-a'), Evidence('b', 'beta beta', 'doc-b')])
    session.remember('a', question='alpha?', outcome='supplied outcome')
    first = session.investigate('alpha', hypotheses=[{'id': 'h', 'text': 'beta'}])
    session.revise_evidence('a', 'alpha beta', 'doc-a2')
    session.new_episode()
    second = session.investigate('alpha', hypotheses=[{'id': 'h', 'text': 'alpha'}])
    session.save(TOOLS / 'cognitive_session.json')
    state = (CognitiveState(max_records=16)
             .add_evidence([Evidence('e', 'source words', 'document')])
             .add_hypotheses([Hypothesis('h', 'interpretation', model_provenance='weights-a')])
             .assess([Assessment('e', 'h', {'support': .25, 'contradiction': .5, 'unknown': .25}, 'judge-a')])
             .select(['h'])
             .add_goals([Goal('g', 'investigate', 'user')])
             .add_plans([Plan('p', ('inspect', 'revise'), ('new evidence',))])
             .observe([Observation('o', 'actual response ☕', 'receipt-1')]))
    state.save(TOOLS / 'cognitive_state.json')
    return {'identity': session._model_identity(), 'memory_fingerprint': session.memory.fingerprint,
            'first': first, 'second': second, 'snapshot': session.snapshot()}


def runtime_fixtures():
    def inspect(state):
        return ActionOutcome(dict(state, visited=state['visited'] + ['inspect']), {'measurement': 1, 'note': 'café'})

    def finish(state, value=None):
        return ActionOutcome(dict(state, finished=value), {'result': 'finished'}, done=True)

    executor = PlanExecutor(actions={'inspect': inspect, 'finish': finish},
                            replan=lambda request: ExecutablePlan('revised', (PlanStep('finish', {'value': 2.5}),)), max_steps=3)
    result = executor({'visited': []}, ExecutablePlan('initial', (PlanStep('inspect', {}, {'measurement': 1}),)))
    result.save(TOOLS / 'trajectory.json')
    path = TOOLS / 'memory.json'
    if path.exists():
        path.unlink()
    memory = JsonMemory(path, retrieve=lambda request: request.candidates[:request.limit])
    memory.append('Customer asked about a card fee ☕', kind='observation', metadata={'turn': 1})
    memory.append({'nested': [1, 2.5, None]}, kind='response', source_id='explicit', metadata={'turn': 2})
    torch.manual_seed(6)
    native = torch.randn(2, 3, 4)
    update = torch.randn(2, 3, 4) * 1e3
    mask = torch.tensor([[True, True, False], [True, True, True]])
    residual = _bounded_memory_update(native, update, mask, torch.tensor(0.3))
    return {'trajectory': json.loads((TOOLS / 'trajectory.json').read_text()),
            'bounded': {'native': tensor_json(native), 'update': tensor_json(update), 'mask': mask.tolist(),
                        'gate': 0.3, 'residual': tensor_json(residual)}}


def retrieval_fixtures():
    torch.manual_seed(7)
    encoder = RetrievalEncoder(retrieval_config()).eval()
    save(encoder, 'retrieval')
    receipt = encoder.receipt(['alpha', 'alpha beta beta', 'beta alpha alpha alpha alpha'])
    return {'config': retrieval_config(), 'receipt': receipt, 'encode_configuration': encoder.encode.configuration(),
            'bindings': bindings(encoder)}


def hotpot_fixtures():
    try:
        tool = Investigator.from_pretrained(HOTPOT_REPO, revision=HOTPOT_REVISION, local_files_only=True).eval()
    except Exception as error:  # noqa: BLE001 - the cached snapshot is optional
        print(f'skipping hotpot fixture: {error}')
        return None
    cases = [
        {'question': 'Which magazine was started first, Arthur\'s Magazine or First for Women?',
         'evidence': [{'source_id': 'arthur', 'text': "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia."},
                      {'source_id': 'women', 'text': 'First for Women is a woman\'s magazine published by Bauer Media Group in the USA. The magazine was started in 1989.'}],
         'hypotheses': [{'id': 'arthur', 'text': "Arthur's Magazine was started first."},
                        {'id': 'women', 'text': 'First for Women was started first.'}]},
        {'question': 'Were Scott Derrickson and Ed Wood of the same nationality?',
         'evidence': [{'source_id': 's1', 'text': 'Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.'},
                      {'source_id': 's2', 'text': 'Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.'}],
         'hypotheses': [{'id': 'yes', 'text': 'Yes, both were American.'}, {'id': 'no', 'text': 'No, they had different nationalities.'},
                        {'id': 'unknown', 'text': 'Their nationalities are not stated.'}]},
    ]
    receipts = []
    with torch.no_grad():
        for case in cases:
            receipts.append(tool(copy.deepcopy(case)))
    return {'repo': HOTPOT_REPO, 'revision': HOTPOT_REVISION, 'cases': cases, 'receipts': receipts,
            'configuration_keys': sorted(tool.configuration().keys())}


def generate():
    TOOLS.mkdir(parents=True, exist_ok=True)
    payload = {
        'chatbot': chatbot_fixtures(),
        'investigator': investigator_fixtures(),
        'planner': planner_fixtures(),
        'cognition': cognition_fixtures(),
        'runtime': runtime_fixtures(),
        'retrieval': retrieval_fixtures(),
        'hotpot': hotpot_fixtures(),
    }
    write_json('tools/tools.json', payload)
