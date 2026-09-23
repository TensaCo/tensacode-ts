/**
 * Chat-template rendering against jinja2 as transformers configures it
 * (``_compile_jinja_template``): filters, tests, statements and value
 * rendering. Expected outputs were produced by transformers 5.17 / Jinja2 with
 * the ``messages`` below; ``ERR <name>`` marks templates Python rejects.
 */
import { describe, expect, it } from 'vitest';
import { ChatTemplate } from '../../src/_internal/text/jinja.js';

const messages = [{ role: 'system', content: 'You are x' }, { role: 'user', content: ' hi there ' }, { role: 'assistant', content: 'hello' }];

const CASES: [string, string][] = [
  ["{% for m in messages %}{{ m.role | upper }}: {{ m.content | trim }}\n{% endfor %}", "SYSTEM: You are x\nUSER: hi there\nASSISTANT: hello\n"],
  ["{{ messages | map(attribute='role') | join(',') }}", "system,user,assistant"],
  ["{{ messages | rejectattr('role', 'equalto', 'system') | list | length }}", "2"],
  ["{{ messages | selectattr('role', 'equalto', 'user') | list | count }}", "1"],
  ["{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}", "3"],
  ["{{ messages[0].content.split(' ') | join('|') }}", "You|are|x"],
  ["{{ messages[0].content.startswith('You') }} {{ messages[1].content.strip() }}", "True hi there"],
  ["{% if tools is defined and tools %}T{% else %}N{% endif %}", "N"],
  ["{{ [3,1,2] | sort | first }} {{ [3,1,2] | max }} {{ [3,1,2] | min }} {{ [3,1,2] | sum }}", "1 3 1 6"],
  ["{{ {'b':1,'a':2} | dictsort }}", "[('a', 2), ('b', 1)]"],
  ["{{ '%s-%d' | format('x', 3) }}", "x-3"],
  ["{{ messages[0] | tojson }}", "{\"role\": \"system\", \"content\": \"You are x\"}"],
  ["{{ messages[0] | tojson(indent=2) }}", "{\n  \"role\": \"system\",\n  \"content\": \"You are x\"\n}"],
  ["{{ 'abc' | reverse }}{{ 'a<b' | e }}", "cbaa&lt;b"],
  ["{% for x in [1,2,3] %}{{ loop.index }}{{ loop.first }}{{ loop.last }}{{ loop.length }}{{ loop.revindex }}{% endfor %}", "1TrueFalse332FalseFalse323FalseTrue31"],
  ["{% for a, b in {'x': 1}.items() %}{{ a }}={{ b }}{% endfor %}", "x=1"],
  ["{{ raise_exception('boom') if false else 'ok' }}", "ok"],
  ["{% macro f(x, y=2) %}{{ x + y }}{% endmacro %}{{ f(1) }}{{ f(1, y=5) }}", "36"],
  ["{{ 'x' ~ 1 ~ none }}", "x1None"],
  ["{{ messages | length is divisibleby 3 }}", "True"],
  ["{{ (messages | last).content[-3:] }}", "llo"],
  ["{{ 'hello world' | title }} {{ 'a,b' .split(',') }}", "Hello World ['a', 'b']"],
  ["{% for m in messages if m.role != 'system' %}{{ m.role }}{% else %}empty{% endfor %}", "userassistant"],
  ["{{ [1,2,3,4,5] | batch(2) | list }}", "[[1, 2], [3, 4], [5]]"],
  ["{{ 'abc' | truncate(2) }}|{{ 'a b' | wordcount }}|{{ '  x ' | center(6) }}|", "ERR AssertionError"],
  ["{% filter upper %}abc{% endfilter %}", "ABC"],
  ["{% with a = 1 %}{{ a }}{% endwith %}", "1"],
  ["{% raw %}{{ x }}{% endraw %}", "{{ x }}"],
  ["{{ 3 // 2 }} {{ 2 ** 3 }} {{ 7 % 3 }} {{ 1 / 2 }}", "1 8 1 0.5"],
  ["{{ 'abc'.replace('b', 'x').upper() }}", "AXC"],
  ["{{ messages[0]['content'] | lower }}", "you are x"],
  ["{{ x | default('d', true) }}", "d"],
  ["{{ [1, 2] + [3] }}", "[1, 2, 3]"],
  ["{{ 1.0 }} {{ 1e3 }} {{ 0.1 + 0.2 }}", "1.0 1000.0 0.30000000000000004"],
  ["{{ {'a': 1} }} {{ [none, true] }}", "{'a': 1} [None, True]"],
  ["{% for i in range(3) %}{{ i }}{% endfor %}", "012"],
  ["{{ 'x' in 'xyz' }} {{ 'role' in messages[0] }}", "True True"],
  ["{{ messages | map('string') | first }}", "{'role': 'system', 'content': 'You are x'}"],
  ["{{ {'a': 1}.items() | list }}", "[('a', 1)]"],
  ["{{ (1,) }} {{ (1, 'x') }} {{ () }}", "(1,) (1, 'x') ()"],
  ["{% for k, v in {'a': 1} | dictsort %}{{ k }}{{ v }}{% endfor %}", "a1"],
  ["{% filter trim | upper %}  ab  {% endfilter %}", "AB"],
  ["{% with a = [1, 2], b = 'x' %}{{ a }}{{ b }}{% endwith %}{{ a }}", "[1, 2]x"],
  ["x\n  {% raw -%}\n  {{ y }}\n  {%- endraw %}\nz", "x\n{{ y }}z"],
  ["{{ '%5.2f|%-4d|%04d|%r|%%' | format(3.14159, 7, 42, 'q') }}", " 3.14|7   |0042|'q'|%"],
  ["{{ messages | max(attribute='content') }}", "{'role': 'system', 'content': 'You are x'}"],
  ["{{ ['B', 'a', 'C'] | max }} {{ ['B', 'a', 'C'] | min(true) }}", "C B"],
  ["{{ messages | map(attribute='content') | map('length') | sum }}", "24"],
  ["{{ [1,2,3,4,5] | batch(2, 'x') | list }}", "[[1, 2], [3, 4], [5, 'x']]"],
  ["{{ 10 is divisibleby 3 }}{{ 3 is gt 2 }}{{ 2 is le 2 }}", "FalseTrueTrue"],
  ["{{ '<a href=\"x\">&' | escape }}", "&lt;a href=&#34;x&#34;&gt;&amp;"],
  ["{{ 'hello big world' | truncate(9) }}|{{ 'hello big world' | truncate(9, true) }}|{{ 'hello big world' | truncate(12, false, '!', 0) }}|{{ 'short' | truncate(3) }}", "hello...|hello ...|hello big!|short"],
  ["[{{ 'x' | center(6) }}][{{ 'ab' | center(7) }}][{{ 'abc' | center(2) }}]", "[  x   ][   ab  ][abc]"],
  ["{{ '<b>' | forceescape }}", "&lt;b&gt;"],
  ["{{ 'abc' | truncate(2) }}", "ERR AssertionError"],
];

describe('chat templates render like jinja2', () => {
  for (const [source, expected] of CASES) {
    it(source, () => {
      let rendered: string;
      try {
        rendered = new ChatTemplate(source).render({ messages, add_generation_prompt: false });
      } catch (error) {
        rendered = `ERR ${(error as Error).name}`;
      }
      if (expected.startsWith('ERR ')) expect(rendered.startsWith('ERR ')).toBe(true);
      else expect(rendered).toBe(expected);
    });
  }
});
