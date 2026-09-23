import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { NativeConfig, nativeConfig } from '../../src/_internal/native/config.js';
import { emitJsonRaw, parseJsonRaw, pythonJsonDumps, rawGet, type RawNode } from '../../src/_internal/json.js';

const text = readFileSync(new URL('../fixtures/native_configs.json', import.meta.url), 'utf8');
const cases = JSON.parse(text) as { input: Record<string, unknown>; to_dict: Record<string, unknown>; diff: Record<string, unknown> }[];
const raw = parseJsonRaw(text) as Extract<RawNode, { t: 'a' }>;
const python = (index: number, key: string) => emitJsonRaw(rawGet(raw.items[index], key)!, { sortKeys: true, separators: [',', ':'] });
const ours = (value: unknown) => pythonJsonDumps(value, { sortKeys: true, separators: [',', ':'] });

describe('native configuration matches transformers serialization', () => {
  cases.forEach((record, index) => {
    it(`${record.input.model_type} case ${index}`, () => {
      const config = record.input.model_type === 't5' ? nativeConfig(record.input) : NativeConfig.fromDict(record.input);
      expect(ours(config.toDict())).toBe(python(index, 'to_dict'));
      expect(ours(config.toDiffDict())).toBe(python(index, 'diff'));
      // Normalized forms reconstruct verbatim (idempotent artifacts).
      expect(NativeConfig.fromDict(record.diff).toDiffDict()).toEqual(record.diff);
      expect(NativeConfig.fromDict(record.to_dict).toDict()).toEqual(record.to_dict);
    });
  });

  it('resolves attribute aliases and derived T5 settings', () => {
    const t5 = nativeConfig({ model_type: 't5', d_model: 16, num_layers: 3, feed_forward_proj: 'gated-gelu', tie_word_embeddings: false });
    expect(t5.get('hidden_size')).toBe(16);
    expect(t5.get('num_hidden_layers')).toBe(3);
    expect(t5.get('num_decoder_layers')).toBe(3);
    expect(t5.get('dense_act_fn')).toBe('gelu_new');
    expect(t5.get('tie_word_embeddings')).toBe(false);
    expect(t5.get('scale_decoder_outputs')).toBe(false);
    expect(() => NativeConfig.fromDict({ model_type: 'gpt2' })).toThrow(/unsupported/);
  });
});
