/** Shared tiny configurations and comparison helpers for tools/runtime tests. */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect } from 'vitest';
import { fixtureJson } from '../helpers/fixtures.js';
import { NativeConfig } from '../../src/_internal/native/config.js';
import type { JsonObject } from '../../src/_internal/json.js';

export const TOOLS = fixtureJson('tools/tools.json');

export function fixturePath(name: string): string {
  return new URL(`../fixtures/tools/${name}`, import.meta.url).pathname;
}

/** Python ``test_chatbot_model.tiny_config()`` (T5 vocab 8, d_model 16). */
export function tinyConfig(): JsonObject {
  return structuredClone(TOOLS.chatbot.config) as JsonObject;
}

/** Python ``test_investigation.config()``: rank vocabulary, owned generator and BERT verifier. */
export function investigatorConfig(): JsonObject {
  return structuredClone(TOOLS.investigator.config) as JsonObject;
}

/** Python ``test_retrieval_encoder.retrieval_config()``. */
export function retrievalConfig(): JsonObject {
  return structuredClone(TOOLS.retrieval.config) as JsonObject;
}

/** A WordLevel/Whitespace ``tokenizer.json`` string. */
export function wordLevelTokenizer(vocab: Record<string, number>, unk: string): string {
  return JSON.stringify({
    version: '1.0', truncation: null, padding: null, added_tokens: [], normalizer: null,
    pre_tokenizer: { type: 'Whitespace' }, post_processor: null, decoder: null,
    model: { type: 'WordLevel', vocab, unk_token: unk },
  });
}

/** ``BertConfig(**options).to_dict()``. */
export function bertConfig(options: JsonObject): JsonObject {
  return NativeConfig.fromDict({ model_type: 'bert', ...options }).toDict();
}

/** Python ``test_cognition.investigator()`` configuration (alpha/beta, BERT verifier). */
export function cognitionConfig(): JsonObject {
  return {
    vocabulary: ['alpha', 'beta'], dimensions: 8, slots: 2, steps: 1,
    verifier_config: bertConfig({ vocab_size: 4, hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2, intermediate_size: 16, num_labels: 3 }),
    verifier_tokenizer_json: wordLevelTokenizer({ '<pad>': 0, '<unk>': 1, alpha: 2, beta: 3 }, '<unk>'),
    verifier_tokenizer_special_tokens: { pad_token: '<pad>', unk_token: '<unk>' },
    verifier_labels: { support: 0, contradiction: 1, unknown: 2 },
  };
}

export function scratch(prefix = 'tensorcode-tools-'): { dir: string; cleanup(): void } {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  return { dir, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

/** Deep equality with numeric tolerance (strings, booleans and structure exact). */
export function expectDeepClose(actual: unknown, expected: unknown, atol = 1e-5, path = '$'): void {
  if (typeof expected === 'number') {
    if (typeof actual !== 'number' || !(Math.abs(actual - expected) <= atol + atol * Math.abs(expected))) {
      expect.fail(`${path}: ${String(actual)} != ${expected}`);
    }
    return;
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) expect.fail(`${path}: array length ${JSON.stringify(actual)?.slice(0, 200)} vs ${expected.length}`);
    expected.forEach((item, index) => expectDeepClose((actual as unknown[])[index], item, atol, `${path}[${index}]`));
    return;
  }
  if (expected !== null && typeof expected === 'object') {
    if (actual === null || typeof actual !== 'object') expect.fail(`${path}: expected object, got ${String(actual)}`);
    const keys = Object.keys(expected as object).sort();
    expect(Object.keys(actual as object).sort(), `${path} keys`).toEqual(keys);
    for (const key of keys) expectDeepClose((actual as Record<string, unknown>)[key], (expected as Record<string, unknown>)[key], atol, `${path}.${key}`);
    return;
  }
  expect(actual, path).toEqual(expected);
}
