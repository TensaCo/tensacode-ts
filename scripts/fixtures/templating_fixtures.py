"""Chat templates rendered by transformers, and LANCZOS resizing by PyTorch."""
from __future__ import annotations

import glob
import json

import torch
import torch.nn.functional as F

from generate import tensor_json, write_json

FEATURES = """{%- set ns = namespace(count=0, last='') -%}
{%- macro fmt(m, prefix='>') -%}{{ prefix }} {{ m.role|upper }}={{ m.content|trim|length }}{%- endmacro -%}
{% for message in messages if message.role != 'system' %}
  {% set ns.count = ns.count + 1 %}
  {{ loop.index }}/{{ loop.length }} {{ fmt(message) }}{% if loop.first %} first{% endif %}{% if loop.last %} last{% endif %}
  {%- if message.content is string and 'x' in message.content %} has-x{% endif %}
{% endfor %}
count={{ ns.count }} {{ messages[1:]|map(attribute='role')|join(',') }} {{ messages|length > 1 }} {{ (messages|first).role|capitalize }}
{{ {'a': 1, 'b': [1, 'two', none, true]}|tojson }} {{ 7 // 2 }} {{ 7 % 3 }} {{ 2 ** 3 }} {{ 'abc'[::-1] }} {{ [1,2,3][-1] }}
{%- if tools is defined %} tools{% else %} notools{% endif %}
{% for k, v in {'x': 1, 'y': 2}.items() %}{{ k }}={{ v }};{% endfor %}
{{ 'Hello World'.split(' ')[1].lower() }} {{ '  pad  '.strip() }}|{{ none }}|{{ undefined_name }}|{{ 1 if false else 2 }}
{% for m in messages %}{% if loop.index > 2 %}{% break %}{% endif %}[{{ m.role }}]{% endfor %}"""


def templates():
    from transformers.utils.chat_template_utils import _compile_jinja_template
    sources = {
        'smolvlm': "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
        'tiny': "{% for message in messages %}{{ message['role'] }}: {% for part in message['content'] %}{% if part['type'] == 'image' %}<image>{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endfor %}{% if add_generation_prompt %} assistant:{% endif %}",
        'chatml': "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
        'features': FEATURES,
    }
    for path in glob.glob('/home/*/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/chat_template.json'):
        sources['qwen3vl'] = json.load(open(path))['chat_template']
    multimodal = [
        [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'describe object'}]}],
        [{'role': 'system', 'content': [{'type': 'text', 'text': 'Be brief.'}]},
         {'role': 'user', 'content': [{'type': 'text', 'text': 'What is here?'}, {'type': 'image'}]},
         {'role': 'assistant', 'content': [{'type': 'text', 'text': 'A cat.'}]}],
    ]
    plain = [[{'role': 'system', 'content': 'sys  '}, {'role': 'user', 'content': 'hi x'},
              {'role': 'assistant', 'content': ' hello '}, {'role': 'user', 'content': 'bye'}]]
    cases = []
    for name, source in sources.items():
        for messages in (plain if name in ('chatml', 'features') else multimodal):
            for generation in (True, False):
                output = _compile_jinja_template(source).render(messages=messages, add_generation_prompt=generation,
                                                                 bos_token='<s>', eos_token='</s>')
                cases.append({'name': name, 'source': source, 'messages': messages, 'add_generation_prompt': generation,
                              'output': output})
    write_json('templates.json', cases)


def lanczos():
    torch.manual_seed(0)
    cases = []
    for h, w, oh, ow in [(13, 17, 8, 11), (8, 8, 20, 30), (31, 7, 12, 5), (64, 48, 17, 23)]:
        x = torch.rand(1, 3, h, w)
        u = (torch.rand(1, 3, h, w) * 255).round().to(torch.uint8)
        cases.append({'size': [oh, ow], 'float': tensor_json(x), 'uint8': tensor_json(u),
                      'float_out': tensor_json(F.interpolate(x, size=(oh, ow), mode='lanczos', antialias=True, align_corners=False)),
                      'uint8_out': tensor_json(F.interpolate(u, size=(oh, ow), mode='lanczos', antialias=True, align_corners=False))})
    write_json('lanczos.json', cases)


def generate():
    templates()
    lanczos()
