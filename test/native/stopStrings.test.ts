/**
 * ``StopStringCriteria`` for byte-level, byte-fallback and plain token
 * strings, and ``stop_strings``/``token_healing`` in ``generate`` with a
 * tokenizer (``scripts/fixtures/stop_strings_fixtures.py``).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { FastTokenizer } from '../../src/_internal/tokenizers/index.js';
import { stopStringTokenizer } from '../../src/_internal/native/causalGeneration.js';
import { stopStringCriteria } from '../../src/_internal/native/logitsProcessors.js';
import { tensor } from '../../src/nn/index.js';
import { Scene } from '../../src/tools/scene.js';

const root = new URL('../fixtures/scene_language/', import.meta.url).pathname;
const records = JSON.parse(readFileSync(join(root, 'stop_strings.json'), 'utf8'));

async function load(folder: string): Promise<FastTokenizer> {
  return FastTokenizer.fromDirectory(join(root, folder));
}

const cases: [string, string, string | string[], string[]][] = [
  ['byte_level', 'smol/processor', ['cat', ' the', 'end.', 'é.'], ['the cat sat', 'at the end. Done', 'concatenate', 'hello world', 'Émile et café.']],
  ['byte_level_single', 'smol/processor', 'world', ['the cat sat', 'at the end. Done', 'concatenate', 'hello world', 'Émile et café.']],
  ['word_level', 'tiny/processor', ['left'], ['describe left object']],
  ['byte_fallback', 'tokenizer_classes/llama', ['llo', 'bé'], ['hello abc', 'ab hello', 'hé bé']],
  ['metaspace', 'tokenizer_classes/t5_fast', ['a b', 'hello'], ['a b c', 'hello a']],
];

describe('StopStringCriteria matches transformers', () => {
  for (const [name, folder, stops, texts] of cases) {
    it(name, async () => {
      const tokenizer = await load(folder);
      const record = records[name];
      if (record.error) {
        expect(() => stopStringCriteria(stopStringTokenizer(tokenizer), stops)).toThrow(record.message);
        return;
      }
      const criterion = stopStringCriteria(stopStringTokenizer(tokenizer), stops);
      texts.forEach((text, index) => {
        const ids = tokenizer.encode(text, { addSpecialTokens: false }).inputIds[0]!;
        expect(ids).toEqual(record.rows[index].ids);
        const done = ids.map((_, end) => criterion([ids.slice(0, end + 1)])[0]);
        expect(done, text).toEqual(record.rows[index].done);
      });
    });
  }
});

describe('generate with a tokenizer', () => {
  const runs = records.generate;
  it('stop_strings and token_healing equal transformers', async () => {
    const tool = await Scene.fromPretrained(join(root, 'smol'));
    const model = tool.language!.model;
    const tokenizer = await load('smol/processor');
    const prompt = runs.free.sequences[0].slice(0, 7);
    const ids = tensor(prompt, { shape: [1, prompt.length], dtype: 'int64' });
    const options = { generationConfig: tool.language!.generationConfig, tokenizer };
    const free = model.generate({ inputIds: ids }, { ...options, settings: { max_new_tokens: 8, do_sample: false } });
    expect(free.sequences).toEqual(runs.free.sequences);
    const stopped = model.generate({ inputIds: ids }, { ...options, settings: { max_new_tokens: 8, do_sample: false, stop_strings: [runs.stop.stop] } });
    expect(stopped.sequences).toEqual(runs.stop.sequences);
    const healing = tensor(runs.healing.prompt.flat(), { shape: [1, runs.healing.prompt[0].length], dtype: 'int64' });
    const healed = model.generate({ inputIds: healing }, {
      ...options, settings: { max_new_tokens: 3, do_sample: false, token_healing: true, pad_token_id: tokenizer.padTokenId },
    });
    expect(healed.sequences).toEqual(runs.healing.sequences);
  }, 300_000);
});
