/** Chat template rendering matches transformers (``scripts/fixtures/templating_fixtures.py``). */
import { describe, expect, it } from 'vitest';
import { ChatTemplate, TemplateError } from '../../src/_internal/text/jinja.js';
import { fixtureJson } from '../helpers/fixtures.js';

const cases = fixtureJson<{ name: string; source: string; messages: unknown[]; add_generation_prompt: boolean; output: string }[]>('templates.json');

describe('chat templates', () => {
  cases.forEach((record, index) => {
    it(`${record.name} #${index} (generation prompt: ${record.add_generation_prompt})`, () => {
      const rendered = new ChatTemplate(record.source).render({
        messages: record.messages, add_generation_prompt: record.add_generation_prompt, bos_token: '<s>', eos_token: '</s>',
      });
      expect(rendered).toBe(record.output);
    });
  });

  it('raise_exception and unsupported syntax fail explicitly', () => {
    expect(() => new ChatTemplate("{{ raise_exception('bad role') }}").render({})).toThrow(/bad role/);
    expect(() => new ChatTemplate('{% include "x" %}')).toThrow(TemplateError);
  });
});
