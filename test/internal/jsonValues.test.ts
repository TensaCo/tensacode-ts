/**
 * Python JSON values in TypeScript: the int/float distinction, dictionary
 * insertion order (integer-like keys included), non-string dictionary key
 * types and transformers' non-persistent buffers, against artifacts,
 * fingerprints and experience files written by Python
 * (``scripts/fixtures/json_values_fixtures.py``).
 */
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { float, int } from '../../src/index.js';
import {
  canonicalJson, deepCopy, graftPythonNumbers, jsonEqual, mergeJson, orderedKeys, orderedObject, parseJsonStrict,
  pythonJsonDumps, pythonJsonLoads, pythonNumberKind, rawToValue, parseJsonRaw, setPythonNumberKind, validatedJson,
  type JsonObject,
} from '../../src/_internal/json.js';
import { fingerprint, operationConfiguration } from '../../src/_internal/fingerprint.js';
import { ModelFingerprint } from '../../src/_internal/cognition/locking.js';
import { NativeConfig, nativeConfig } from '../../src/_internal/native/config.js';
import { createNativeModel, type NativeHead } from '../../src/_internal/native/registry.js';
import { JsonMemory } from '../../src/_internal/memory/json.js';
import { OutcomeExperience, PlanExecutionResult } from '../../src/_internal/execution/planning.js';
import { RetrievalEncoder } from '../../src/_internal/retrieval.js';
import { Codec } from '../../src/_internal/training/persistence.js';
import { trace } from '../../src/_internal/tracing.js';
import { loadModelFromBytes } from '../../src/nn/safetensors.js';
import { Linear, tensor } from '../../src/nn/index.js';
import { Operation } from '../../src/ops/base.js';
import * as text from '../../src/ops/text/index.js';
import * as vec from '../../src/ops/vec/index.js';
const { Retrieve } = text;
import { ScoreResult } from '../../src/ops/text/score.js';
import { Chatbot, Investigator } from '../../src/tools/index.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import { fixtureBytes } from '../helpers/fixtures.js';
import { LabelHead } from '../training/helpers.js';

const fixtures = new URL('../fixtures/json_values/', import.meta.url).pathname;
const record = parseJsonStrict(readFileSync(join(fixtures, 'record.json'), 'utf8')) as Record<string, any>;
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-json-values-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const read = (path: string) => readFileSync(path, 'utf8');

describe('lossless Python JSON', () => {
  it('keeps the int/float kind of parsed numbers, big integers included', () => {
    const text = '{"2": 1.0, "1": 0, "x": [1.0, 2, 12345678901234567890, -0.0, 1e16, 1E2]}';
    // Python: json.dumps(json.loads(text))
    expect(pythonJsonDumps(parseJsonStrict(text))).toBe('{"2": 1.0, "1": 0, "x": [1.0, 2, 12345678901234567890, -0.0, 1e+16, 100.0]}');
    expect(pythonJsonDumps(rawToValue(parseJsonRaw(text)))).toBe(pythonJsonDumps(parseJsonStrict(text)));
  });

  it('keeps Python insertion order of integer-like keys', () => {
    const parsed = parseJsonStrict('{"b": 1, "10": 2, "2": 3, "a": 4}') as JsonObject;
    expect(Object.keys(parsed)).toEqual(['2', '10', 'b', 'a']); // JavaScript's own order
    expect(orderedKeys(parsed)).toEqual(['b', '10', '2', 'a']);
    expect(pythonJsonDumps(parsed)).toBe('{"b": 1, "10": 2, "2": 3, "a": 4}');
    parsed['1'] = 5;
    expect(orderedKeys(parsed)).toEqual(['b', '10', '2', 'a', '1']);
    expect(pythonJsonDumps(orderedObject([['2', 'two'], ['1', 'one']]))).toBe('{"2": "two", "1": "one"}');
  });

  it('marks numbers with float() and int(), overriding the schema default', () => {
    expect(pythonJsonDumps({ weight: float(1), hidden_dropout_prob: 0, attention_dropout: int(0) }))
      .toBe('{"weight": 1.0, "hidden_dropout_prob": 0.0, "attention_dropout": 0}');
    expect(canonicalJson([float(-0), float(1e16), int('123456789012345678901234567890')])).toBe('[-0.0,1e+16,123456789012345678901234567890]');
    expect(jsonEqual({ a: float(1) }, { a: 1 })).toBe(true); // Python 1 == 1.0
  });

  it('copies, merges and grafts recorded kinds and order', () => {
    const parsed = parseJsonStrict('{"2": 1.0, "1": [0.0, 1], "hidden_dropout_prob": 0}') as JsonObject;
    const copy = deepCopy(parsed);
    expect(pythonJsonDumps(copy)).toBe('{"2": 1.0, "1": [0.0, 1], "hidden_dropout_prob": 0}');
    expect(pythonNumberKind(copy, 'hidden_dropout_prob')).toBe('int');
    expect(pythonJsonDumps(mergeJson({ z: float(3) }, copy))).toBe('{"z": 3.0, "2": 1.0, "1": [0.0, 1], "hidden_dropout_prob": 0}');
    const rebuilt: JsonObject = { hidden_dropout_prob: 0, 1: [0, 1], 2: 1 };
    graftPythonNumbers(rebuilt, parsed);
    expect(pythonJsonDumps(rebuilt)).toBe('{"2": 1.0, "1": [0.0, 1], "hidden_dropout_prob": 0}');
    setPythonNumberKind(rebuilt, '2', null);
    expect(pythonJsonDumps(rebuilt)).toBe('{"2": 1, "1": [0.0, 1], "hidden_dropout_prob": 0}');
    // A Map with string keys is an ordered Python dict in configurations.
    expect(pythonJsonDumps(validatedJson(new Map<string, unknown>([['2', 1], ['1', float(2)]])))).toBe('{"2": 1, "1": 2.0}');
  });

  it('writes Map keys like Python json.dumps', () => {
    // Python: json.dumps({1: 'b', None: 'c', 1.5: 'd', 2.0: 'e'}) and sort_keys=True over {2: 'a', 1: 'b'}.
    const floatKey = new Map<unknown, string>([[1, 'b'], [null, 'c'], [1.5, 'd']]);
    expect(pythonJsonDumps(floatKey)).toBe('{"1": "b", "null": "c", "1.5": "d"}');
    expect(pythonJsonDumps(new Map([[2, 'a'], [1, 'b']]), { sortKeys: true })).toBe('{"1": "b", "2": "a"}');
  });

  it('loads like json.loads (last duplicate wins in first position, NaN accepted)', () => {
    const value = pythonJsonLoads('{"a": 1, "b": 2, "a": 3.0}') as JsonObject;
    expect(orderedKeys(value)).toEqual(['a', 'b']);
    expect(pythonJsonDumps(value)).toBe('{"a": 3.0, "b": 2}');
    expect(pythonJsonDumps(pythonJsonLoads('[NaN, Infinity, -Infinity]'))).toBe('[NaN, Infinity, -Infinity]');
    expect(() => parseJsonStrict('{"a": 1, "a": 2}')).toThrow(/Duplicate/);
  });

  it('native configurations keep Python ints in float fields and float defaults', () => {
    const config = NativeConfig.fromDict(parseJsonStrict('{"model_type": "bert", "hidden_dropout_prob": 0, "custom": 2.0}'));
    expect(pythonJsonDumps(config.toDiffDict(), { sortKeys: true })).toContain('"custom": 2.0');
    expect(pythonJsonDumps(config.toDiffDict(), { sortKeys: true })).toContain('"hidden_dropout_prob": 0,');
    // transformers default ``LlamaConfig.rope_theta`` is the float 10000.0.
    const llama = NativeConfig.fromDict({ model_type: 'llama', hidden_size: 16, num_attention_heads: 2 });
    expect(pythonJsonDumps(llama.toDict().rope_parameters)).toBe('{"rope_theta": 10000.0, "rope_type": "default"}');
    expect(pythonJsonDumps(llama.toDict().rope_parameters, { floatKeys: new Set() })).toBe('{"rope_theta": 10000.0, "rope_type": "default"}');
    // SmolVLM's config.json spells ``rope_theta`` as the int 100000, which Python keeps.
    const smol = NativeConfig.fromPretrainedDict(parseJsonStrict('{"model_type": "llama", "rope_theta": 100000, "hidden_size": 16, "num_attention_heads": 2}'), 'x');
    expect(pythonJsonDumps(smol.toDict().rope_parameters)).toBe('{"rope_theta": 100000, "rope_type": "default"}');
    expect(pythonJsonDumps(nativeConfig({ model_type: 'clip' }).toDict(), { sortKeys: true })).toContain('"initializer_factor": 1.0');
  });
});

describe('Python artifacts with ints in float fields', () => {
  for (const [name, load] of [
    ['chatbot', (path: string) => Chatbot.fromPretrained(path)],
    ['investigator', (path: string) => Investigator.fromPretrained(path)],
    ['retrieval', (path: string) => RetrievalEncoder.fromPretrained(path)],
  ] as const) {
    it(`${name}: re-saved manifest, configuration and binding fingerprints equal Python's`, async () => {
      const tool = await load(join(fixtures, name));
      const target = join(scratch, name);
      await tool.savePretrained(target);
      expect(read(join(target, 'tensorcode_config.json'))).toBe(read(join(fixtures, name, 'tensorcode_config.json')));
      expect(pythonJsonDumps(tool.configuration(), { sortKeys: true })).toBe(record[name].configuration);
      const bindings = Object.fromEntries(Object.entries(tool.operationBindings())
        .map(([key, operation]) => [key, fingerprint(operationConfiguration(operation))]));
      expect(bindings).toEqual(record[name].bindings);
      if (tool instanceof Chatbot) expect(tool.fingerprint).toBe(record.chatbot.fingerprint);
    }, 60_000);
  }
});

describe('Python operation artifacts with ints in float fields', () => {
  const classes: Record<string, { fromPretrained(path: string): Promise<{ savePretrained(path: string): Promise<unknown> }> }> = {
    Transform: vec.Transform, TextDecoder: vec.TextDecoder, ImageEncoder: vec.ImageEncoder, ImageDecoder: vec.ImageDecoder,
    Classify: text.Classify, Retrieve: text.Retrieve,
  };
  for (const [name, entry] of Object.entries(record.operations as Record<string, { class: string; bindings: Record<string, string> }>)) {
    it(`${name}: loads, re-saves byte-identically and fingerprints like Python`, async () => {
      const source = join(fixtures, 'operations', name);
      const operation = await classes[entry.class]!.fromPretrained(source);
      const target = join(scratch, 'operations', name);
      await operation.savePretrained(target);
      expect(read(join(target, 'tensorcode_config.json'))).toBe(read(join(source, 'tensorcode_config.json')));
      const bound = typeof (operation as { operationBindings?: unknown }).operationBindings === 'function'
        ? (operation as unknown as { operationBindings(): Record<string, Operation> }).operationBindings() : { operation: operation as unknown as Operation };
      expect(Object.fromEntries(Object.entries(bound).map(([key, value]) => [key, fingerprint(operationConfiguration(value))]))).toEqual(entry.bindings);
    }, 60_000);
  }
});

describe('caller data inside configurations', () => {
  it('Retrieve item values keep Python kinds without the configuration float schema', async () => {
    const loaded = await text.Retrieve.fromPretrained(join(fixtures, 'operations', 'text_retrieve'));
    const config = loaded.configuration();
    // The same items authored in TypeScript: ``timeout: 5`` stays an int, ``float(1)`` is 1.0.
    config.items = orderedObject<unknown>([['2', { timeout: 5, p: float(1), n: [1, float(2)] }], ['1', 'one']]) as JsonObject;
    const authored = new text.Retrieve(config);
    expect(fingerprint(operationConfiguration(authored))).toBe(record.operations.text_retrieve.bindings.operation);
    expect(canonicalJson(authored.configuration())).toContain('"items":{"1":"one","2":{"n":[1,2.0],"p":1.0,"timeout":5}}');
  });
});

describe('Python dict order in operation configurations', () => {
  it('Retrieve items keep their order (a Map), and marked model settings fingerprint like Python', () => {
    const model = {
      complete: () => { throw new Error('not called'); },
      configuration: () => ({ type: 'fixed', temperature: int(1), top_p: 1, weights: orderedObject<unknown>([['2', float(1)], ['1', 0], ['x', 0.5]]) }),
    };
    const operation = Retrieve.fromModel(model, { items: new Map([['2', 'two'], ['1', 'one'], ['x', 'ex']]), limit: 2 });
    expect(orderedKeys(operation.items)).toEqual(record.retrieve.item_keys);
    expect(pythonJsonDumps(operationConfiguration(operation), { sortKeys: true })).toBe(record.retrieve.configuration);
    expect(fingerprint(operationConfiguration(operation))).toBe(record.retrieve.fingerprint);
    expect(pythonJsonDumps(operation.responseSchema())).toBe(record.retrieve.schema);
  });
});

class Note {
  static readonly qualifiedName = 'parity.Note';
  static readonly recordFields = ['text', 'weight'] as const;
  static readonly recordFloatFields = ['weight'] as const;
  constructor(readonly text: string, readonly weight: number) {
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Note {
    return new Note(fields.text as string, fields.weight as number);
  }

  toRecord(): Record<string, unknown> {
    return { text: this.text, weight: this.weight };
  }
}

class Bag {
  static readonly qualifiedName = 'parity.Bag';
  static readonly recordFields = ['data'] as const;
  constructor(readonly data: unknown) {
    Object.freeze(this);
  }

  static fromRecord(fields: Record<string, unknown>): Bag {
    return new Bag(fields.data);
  }

  toRecord(): Record<string, unknown> {
    return { data: this.data };
  }
}

class Echo extends Operation<unknown, unknown> {
  static override readonly qualifiedName: string = 'parity.Echo';
  configuration(): Record<string, unknown> {
    return { kind: 'echo', scale: float(1), offset: 0, weights: orderedObject<unknown>([['2', float(1)], ['10', 0], ['1', 2]]), stride: [float(8), int(8)] };
  }

  forward(): unknown {
    return orderedObject<unknown>([
      ['score', float(1)], ['count', 2], ['by_id', new Map([[1, 'a'], [0, 'b'], [10, 'c']])], ['pair', Object.freeze([float(1), 2])],
      ['weights', [float(0), float(1), 0.5]], ['big', int('12345678901234567890')], ['note', new Note('n', 2)],
    ]);
  }
}

describe('experience files with Python values', () => {
  const codecs = { note: Note, score: ScoreResult, bag: Bag };
  const expected = () => read(join(fixtures, 'experience.json'));

  it('a Python experience loads with Python key types and re-saves byte-identically', async () => {
    const echo = new Echo();
    expect(fingerprint(operationConfiguration(echo))).toBe(record.echo.fingerprint);
    const loaded = await loadExperience(join(fixtures, 'experience.json'), { operations: { echo }, codecs });
    const bag = loaded.inputs.get(6) as Bag;
    expect(bag.data).toBeInstanceOf(Map);
    expect([...(bag.data as Map<unknown, unknown>).keys()]).toEqual([3, 1, 2, 'k', 1.5, null, [1, 2]]);
    expect(loaded.supervisions[1]!.target).toBeInstanceOf(ScoreResult);
    expect(orderedKeys((loaded.supervisions[1]!.target as ScoreResult).distribution!)).toEqual(['1', '0']);
    const path = join(scratch, 'resaved.json');
    await loaded.save(path, { operations: { echo }, codecs });
    expect(read(path)).toBe(expected());
  });

  it('the same trace authored in TypeScript writes Python bytes', async () => {
    const echo = new Echo();
    const session = trace();
    const output = session.run(() => echo.call(orderedObject<unknown>([
      ['x', float(1)], ['n', 2], ['t', Object.freeze([float(1), 2])], ['2', 'two'], ['1', 'one'],
      ['bag', new Bag(new Map<unknown, unknown>([[3, float(1)], [1, 'bool'], [2, 0], ['k', [float(1), 2]], [1.5, 'x'], [null, 'none'],
        [Object.freeze([1, float(2)]), 'tuple']]))],
    ])));
    session.supervise(output, new Note('target', 1), { loss: 'custom', source: 'review:1' });
    session.supervise(output, new ScoreResult(1, { distribution: new Map([[1, 1], [0, 0]]), confidence: 1 }), { loss: 'custom', source: 'review:2' });
    const path = join(scratch, 'authored.json');
    await session.save(path, { operations: { echo }, codecs });
    expect(read(path)).toBe(expected());
  });

  it('traces a Map as a Python dict: string keys bind as a dict tree, other keys are rejected', async () => {
    const echo = new Echo();
    const session = trace();
    expect(() => session.run(() => echo.call(new Map<unknown, unknown>([[1, 'a']])))).toThrow(/Traced mapping keys must be strings/);
    const ordered = trace();
    ordered.run(() => echo.call(new Map<string, unknown>([['2', float(1)], ['1', 0]])));
    const plain = trace();
    plain.run(() => echo.call(orderedObject<unknown>([['2', float(1)], ['1', 0]])));
    const [a, b] = [join(scratch, 'map.json'), join(scratch, 'object.json')];
    await ordered.save(a, { operations: { echo }, codecs });
    await plain.save(b, { operations: { echo }, codecs });
    expect(read(a)).toBe(read(b));
    expect(read(a)).toContain('"inputs": {"0": 1.0, "1": 0}');
  });

  it('decodes non-string dict keys as a Map and re-encodes their Python types', () => {
    const codec = new Codec();
    const encoded = parseJsonStrict('{"type": "dict", "items": [[2, 1.0], [true, "t"], [null, 0], [1.5, "x"]]}');
    const decoded = codec.decode(encoded) as Map<unknown, unknown>;
    expect([...decoded.keys()]).toEqual([2, true, null, 1.5]);
    expect(pythonJsonDumps(codec.encode(decoded))).toBe(pythonJsonDumps(encoded));
    expect(() => codec.decode(parseJsonStrict('{"type": "dict", "items": [[1, "a"], [1.0, "b"]]}'))).toThrow(/Duplicate/);
  });

  it('a float target is not an integer class index (torch.as_tensor(1.0) is floating)', () => {
    const head = new LabelHead(new Linear(1, 2), ['a', 'b']);
    for (const target of [float(1), [float(1)]]) {
      const session = trace();
      const output = session.run(() => head.call(tensor([1])));
      session.supervise(output, target);
      expect(() => Trainer.fromOps({ head }).step(session)).toThrow(/integer indices/);
    }
    const session = trace();
    const output = session.run(() => head.call(tensor([1])));
    session.supervise(output, int(1));
    expect(Trainer.fromOps({ head }).step(session)).toBeGreaterThan(0);
  });
});

describe('JSON memory files with Python values', () => {
  it('appending to a Python memory file writes the bytes Python writes', async () => {
    const path = join(scratch, 'memory.json');
    copyFileSync(join(fixtures, 'memory_before.json'), path);
    const store = await JsonMemory.open(path, { retrieve: (search) => search.candidates });
    expect(pythonJsonDumps(store.records.map((item) => item.metadata))).toBe('[{"p": 1, "2": 1.0, "1": "x"}, {}]');
    await store.append({ x: 0.5 }, { kind: 'note', metadata: { timeout: 3 } });
    expect(read(path)).toBe(read(join(fixtures, 'memory_after.json')));
  });
});

describe('plan trajectories with Python values', () => {
  it('a Python trajectory re-saves byte-identically and renders Python evidence text', async () => {
    const loaded = await PlanExecutionResult.load(join(fixtures, 'trajectory.json'));
    expect(loaded.experiences.map((item) => item.asEvidence().text)).toEqual(record.trajectory.evidence);
    const path = join(scratch, 'trajectory.json');
    await loaded.save(path);
    expect(read(path)).toBe(read(join(fixtures, 'trajectory.json')));
  });

  it('a trajectory authored in TypeScript writes the same bytes', async () => {
    const first = new OutcomeExperience('c1', 'look', 'obs:1', orderedObject<unknown>([['2', float(1)], ['1', 0], ['timeout', 3], ['p', 1]]),
      'observed', { k: float(2), timeout: 1 }, float(1));
    const second = new OutcomeExperience('c1', 'look', 'obs:2', float(2), 'observed', {}, null);
    const result = new PlanExecutionResult(orderedObject<unknown>([['10', float(1)], ['9', 2], ['score', 1]]), [first, second], 'completed');
    expect([first, second].map((item) => item.asEvidence().text)).toEqual(record.trajectory.evidence);
    const path = join(scratch, 'authored-trajectory.json');
    await result.save(path);
    expect(read(path)).toBe(read(join(fixtures, 'trajectory.json')));
  });
});

describe('transformers non-persistent embedding buffers', () => {
  for (const [name, entry] of Object.entries(record.buffers as Record<string, { head: NativeHead; config: JsonObject; buffers: [string, number[], string][]; fingerprint: string }>)) {
    it(`${name}: named_buffers() and content fingerprints equal Python's`, () => {
      const model = createNativeModel(nativeConfig(entry.config), entry.head);
      loadModelFromBytes(model, fixtureBytes(`json_values/buffers_${name}.safetensors`), { strict: true });
      model.eval();
      expect(model.namedBuffers().map(([key, value]) => [key, [...value.shape], `torch.${value.dtype}`])).toEqual(entry.buffers);
      expect([...model.stateDict().keys()].some((key) => key.endsWith('position_ids'))).toBe(false);
      const config = parseJsonStrict('{"probe": 1.0, "ids": {"2": 0, "1": 1}}');
      expect(new ModelFingerprint().compute([['model', model]], config)).toBe(entry.fingerprint);
    });
  }
});
