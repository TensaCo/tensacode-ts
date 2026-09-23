/** Parity with Python-generated fixtures (``scripts/fixtures/text_fixtures.py``). */
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import {
  Classify, Decide, ImagePart, Message, ModelRequest, Retrieve, Score, TextPart, Transform,
} from '../../src/ops/text/index.js';
import { OpenAICompatibleModel } from '../../src/integrations/index.js';
import { alternativePrompt } from '../../src/_internal/text/native.js';
import { fingerprint, operationConfiguration } from '../../src/_internal/fingerprint.js';
import { PYTHON_FLOAT_KEYS, canonicalJson, pythonJsonDumps, sha256Hex, type JsonValue } from '../../src/_internal/json.js';
import { jevQuestion, jevState } from '../../src/integrations/jev.js';
import { schemaName } from '../../src/integrations/openai.js';
import { fixtureJson } from '../helpers/fixtures.js';

const fixture = fixtureJson('text/text.json');
const CLASSES: Record<string, any> = { Transform, Classify, Decide, Score, Retrieve };
const VALUE = Object.freeze([new Message('user', 'question')]);
const CONTEXT = { policy: Object.freeze([new Message('system', 'be brief')]) };
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-text-parity-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

function closeTo(actual: number, expected: number, tolerance = 2e-4): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance * Math.max(1, Math.abs(expected)));
}

/**
 * Python's safetensors writes ``__metadata__`` (tied-alias names) from a Rust
 * ``HashMap``, so its key order differs between Python runs. Everything else
 * (header length, tensor entries and order, data bytes) is byte-identical.
 */
function expectSameSafetensors(actual: Buffer, expected: Buffer): void {
  expect(actual.length).toBe(expected.length);
  const length = Number(expected.readBigUInt64LE(0));
  expect(Number(actual.readBigUInt64LE(0))).toBe(length);
  const header = (bytes: Buffer) => JSON.parse(bytes.subarray(8, 8 + length).toString('utf8'));
  const [left, right] = [header(actual), header(expected)];
  expect(Object.keys(left)).toEqual(Object.keys(right));
  expect(left).toEqual(right);
  expect(Buffer.compare(actual.subarray(8 + length), expected.subarray(8 + length))).toBe(0);
}

function resultJson(result: any): unknown {
  if (Array.isArray(result)) return result.map((message: Message) => ({ role: message.role, content: message.content }));
  return JSON.parse(JSON.stringify(result.toRecord()));
}

describe('owned text operations match Python', () => {
  for (const [name, record] of Object.entries<any>(fixture.owned)) {
    describe(name, () => {
      const load = () => CLASSES[record.class].fromFoundation(fixture.foundation, { config: record.config });

      it('reconstructs the Python configuration, fingerprints and prompts', async () => {
        const op = await load();
        expect(op.configuration()).toEqual(record.configuration);
        expect(canonicalJson(op.configuration())).toBe(canonicalJson(record.configuration));
        expect(operationConfiguration(op)).toEqual(record.operation.configuration);
        expect(fingerprint(operationConfiguration(op))).toBe(record.operation.fingerprint);
        expect(operationConfiguration(op.trainingOperation)).toEqual(record.objective.configuration);
        expect(fingerprint(operationConfiguration(op.trainingOperation))).toBe(record.objective.fingerprint);
        expect(op.replayable).toBe(record.replayable);
        const request = op._request(VALUE, null);
        expect(op.nativeModel.prompt(request)).toBe(record.prompt);
        expect(op.nativeModel.prompt(op._request(VALUE, CONTEXT))).toBe(record.context_prompt);
        expect(op.nativeModel.inputs(request).inputIds.tolist()).toEqual(record.input_ids);
        if (record.response_schema) expect(op.responseSchema()).toEqual(record.response_schema);
      });

      it('generates, scores and computes losses like Python', async () => {
        const op = await load();
        const request = op._request(VALUE, null);
        expect(op.nativeModel.generateIds(request).tolist()).toEqual(record.generated_ids);
        expect(op.nativeModel.generateText(request)).toBe(record.generated_text);
        closeTo(op.loss(VALUE, record.target).item(), record.loss);
        if (record.scores) {
          expect(op._alternatives().map((pair: readonly string[]) => [...pair])).toEqual(record.alternatives);
          expect(alternativePrompt(op._scoringRequest(VALUE, null), op._alternatives())).toBe(record.alternative_prompt);
          const scores = op.nativeModel.scoreAlternatives(op._scoringRequest(VALUE, null), op._alternatives(), { normalization: op.likelihoodNormalization });
          scores.forEach((score: number, index: number) => closeTo(score, record.scores[index]));
        }
        if (record.error) {
          expect(() => op.call(VALUE)).toThrow(record.error.message);
        } else {
          const actual = resultJson(op.call(VALUE)) as any;
          const expected = record.result;
          if (Array.isArray(expected)) expect(actual).toEqual(expected);
          else {
            for (const [key, value] of Object.entries<any>(expected)) {
              if (typeof value === 'number') closeTo(actual[key], value, 1e-3);
              else if (value && typeof value === 'object' && !Array.isArray(value)) {
                expect(Object.keys(actual[key])).toEqual(Object.keys(value));
                for (const [inner, number] of Object.entries<any>(value)) closeTo(actual[key][inner], number, 1e-3);
              } else expect(actual[key]).toEqual(value);
            }
          }
        }
      });
    });
  }
});

describe('Python-saved owned artifacts', () => {
  const source = new URL('../fixtures/text/classify_python', import.meta.url).pathname;

  it('load and re-save byte-identically', async () => {
    const op = await Classify.fromPretrained(source);
    expect(op.configuration()).toEqual(fixture.owned.classify.configuration);
    expect(fingerprint(operationConfiguration(op))).toBe(fixture.owned.classify.operation.fingerprint);
    const request = op._request(VALUE, null);
    expect(op.nativeModel.generateIds(request).tolist()).toEqual(fixture.owned.classify.generated_ids);
    closeTo(op.loss(VALUE, fixture.owned.classify.target).item(), fixture.owned.classify.loss);
    const target = join(scratch, 'resaved');
    await op.savePretrained(target);
    expect(Buffer.compare(readFileSync(join(target, 'tensorcode_config.json')), readFileSync(join(source, 'tensorcode_config.json')))).toBe(0);
    expectSameSafetensors(readFileSync(join(target, 'model.safetensors')), readFileSync(join(source, 'model.safetensors')));
  });

  it('TypeScript-saved foundation artifacts match the Python artifact', async () => {
    const op = await Classify.fromFoundation(fixture.foundation, { config: fixture.owned.classify.config });
    const target = join(scratch, 'typescript');
    await op.savePretrained(target);
    expect(Buffer.compare(readFileSync(join(target, 'tensorcode_config.json')), readFileSync(join(source, 'tensorcode_config.json')))).toBe(0);
    expectSameSafetensors(readFileSync(join(target, 'model.safetensors')), readFileSync(join(source, 'model.safetensors')));
  });
});

describe('serialization parity', () => {
  it('prompts and alternative prompts', async () => {
    const op = await Transform.fromFoundation(fixture.foundation);
    const requests = [
      new ModelRequest([new Message('user', 'héllo ✓ "quoted"\nline')]),
      new ModelRequest([new Message('system', 'rules'), new Message('user', [new TextPart('one '), new TextPart('two', { sourceRef: 's:1' })])], {
        instructions: 'Do it', responseSchema: { type: 'object', required: ['x'], properties: { x: { type: 'number', minimum: 0 } } },
        schemaName: 'custom.name',
      }),
    ];
    requests.forEach((request, index) => {
      expect(op.nativeModel.prompt(request)).toBe(fixture.prompts[index].prompt);
      expect(alternativePrompt(request, [['x: first', 'x'], ['y', 'y']])).toBe(fixture.prompts[index].alternative_prompt);
    });
  });

  it('external configurations, schemas and fingerprints', () => {
    const provider = { complete() { throw new Error('unused'); }, configuration: () => ({ type: 'fixture_provider' }) };
    const ops: Record<string, any> = {
      classify: Classify.fromModel(provider, { labels: ['billing', 'technical', 'other'], instructions: 'Route', descriptions: { billing: 'payments and charges' } }),
      decide: Decide.fromModel(provider, { options: ['archive', 'reply'] }),
      score: Score.fromModel(provider, { rubric: ['low', 'medium', 'high'], instructions: 'Urgency' }),
      retrieve: Retrieve.fromModel(provider, {
        items: { policy: { text: 'refund policy' }, faq: 'general' }, descriptions: { policy: 'refund policy', faq: 'general questions' }, limit: 2,
      }),
      transform: Transform.fromModel(provider, { instructions: 'Summarize' }),
    };
    for (const [name, op] of Object.entries(ops)) {
      const expected = fixture.external[name];
      expect(op.configuration()).toEqual(expected.configuration);
      expect(operationConfiguration(op)).toEqual(expected.operation.configuration);
      expect(fingerprint(operationConfiguration(op))).toBe(expected.operation.fingerprint);
      if (expected.response_schema) expect(op.responseSchema()).toEqual(expected.response_schema);
    }
  });

  it('provider configurations need the float key "timeout" for fingerprint parity', () => {
    const op = Classify.fromModel(new OpenAICompatibleModel({ baseUrl: 'https://example.test/v1/', model: 'm' }), { labels: ['a', 'b'], instructions: 'x' });
    const expected = fixture.external.openai_classify;
    expect(op.configuration()).toEqual(expected.configuration);
    const configuration = operationConfiguration(op);
    expect(configuration).toEqual(expected.operation.configuration);
    const floatKeys = new Set([...PYTHON_FLOAT_KEYS, 'timeout']);
    const text = pythonJsonDumps(configuration, { sortKeys: true, separators: [',', ':'], allowNan: false, floatKeys });
    expect(sha256Hex(text)).toBe(expected.operation.fingerprint);
    if (PYTHON_FLOAT_KEYS.has('timeout')) expect(fingerprint(configuration as JsonValue)).toBe(expected.operation.fingerprint);
  });

  it('provider wire payloads', () => {
    const multimodal = new Message('user', [
      new TextPart('What animal?', { sourceRef: 'prompt:1' }),
      new ImagePart({ data: new Uint8Array(Buffer.from('image-bytes')), mediaType: 'image/png', sourceRef: 'upload:1' }),
      new ImagePart({ url: 'https://example.test/cat.jpg', detail: 'low' }),
    ]);
    const provider = { complete() { throw new Error('unused'); }, configuration: () => ({ type: 'fixture_provider' }) };
    const classify = Classify.fromModel(provider, { labels: ['billing', 'technical', 'other'], instructions: 'Route', descriptions: { billing: 'payments and charges' } });
    const request = classify._request([multimodal], null);
    const chat = new OpenAICompatibleModel({ baseUrl: 'https://example.test/v1', model: 'vision-test' });
    const responses = new OpenAICompatibleModel({ baseUrl: 'https://example.test/v1', model: 'vision-test', api: 'responses' });
    expect(chat.chatPayload(request)).toEqual(fixture.wire.chat_payload);
    expect(responses.responsesPayload(request)).toEqual(fixture.wire.responses_payload);
    expect(pythonJsonDumps(chat.chatPayload(request), { separators: [',', ':'], ensureAscii: false, floatKeys: new Set() })).toBe(fixture.wire.chat_body);
    const score = Score.fromModel(provider, { rubric: ['low', 'medium', 'high'], instructions: 'Urgency' });
    const jevRequest = score._request([new Message('user', [new TextPart('help now', { sourceRef: 't:1' })])], null);
    expect(jevQuestion(jevRequest)).toEqual(fixture.wire.jev_score_question);
    expect(jevState(jevRequest)).toEqual(fixture.wire.jev_state);
    for (const [name, expected] of Object.entries(fixture.wire.schema_names)) expect(schemaName(name)).toBe(expected);
  });
});
