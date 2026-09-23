/**
 * Tools reject unknown or obsolete configuration fields, at construction and
 * load, with Python's message; foundation rank encoders describe their native
 * configuration and tensor schemas (Python ``tests/models/test_tool_configuration_fields.py``;
 * fixtures: scripts/fixtures/tool_fields_fixtures.py).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { manualSeed } from '../../src/nn/index.js';
import { fingerprint, operationConfiguration } from '../../src/_internal/fingerprint.js';
import { pythonJsonLoads, type JsonObject } from '../../src/_internal/json.js';
import { RetrievalEncoder } from '../../src/_internal/retrieval.js';
import { ResponseQualityAssessor } from '../../src/_internal/responseQuality.js';
import { Chatbot } from '../../src/tools/chatbot.js';
import { Decision } from '../../src/tools/decision.js';
import { Investigator } from '../../src/tools/investigator.js';
import { Planner } from '../../src/tools/planner.js';
import { Scene } from '../../src/tools/scene.js';
import { investigatorConfig, retrievalConfig, scratch, tinyConfig } from './helpers.js';

const FIXTURE = pythonJsonLoads(readFileSync(new URL('../fixtures/tool_fields.json', import.meta.url), 'utf8')) as any;
const temp = scratch('tensorcode-tool-fields-');
afterAll(() => temp.cleanup());

const clone = <T>(value: T): T => structuredClone(value);
const ranking = (): JsonObject => clone(FIXTURE.ranking_config);
const scene = (): JsonObject => clone(FIXTURE.scene_config);

type Construct = new (config: any) => unknown;
const CONSTRUCTORS: [string, Construct, () => JsonObject][] = [
  ['Chatbot', Chatbot, tinyConfig],
  ['Investigator', Investigator, investigatorConfig],
  ['Decision', Decision, investigatorConfig],
  ['Planner', Planner, ranking],
  ['Scene', Scene, scene],
  ['RetrievalEncoder', RetrievalEncoder, retrievalConfig],
  ['ResponseQualityAssessor', ResponseQualityAssessor, () => clone(FIXTURE.quality_config)],
];

function messageOf(build: () => unknown): string {
  try {
    build();
  } catch (error) {
    expect((error as Error).name).toBe('ValueError');
    return (error as Error).message;
  }
  throw new Error('construction succeeded');
}

describe('tool configuration fields', () => {
  it.each(CONSTRUCTORS)('%s rejects unknown fields with Python\'s message', (name, Tool, make) => {
    const config = { ...make(), colour: 'blue', obsolete_head: 1, "it's": 2 };
    expect(messageOf(() => new Tool(config))).toBe(FIXTURE.messages[name]);
  });

  it('checks nested Chatbot cognition fields', () => {
    const config = tinyConfig();
    config.cognition = { investigator: investigatorConfig(), proposal_limit: 3 };
    expect(messageOf(() => new Chatbot(config))).toBe(FIXTURE.messages['Chatbot cognition']);
  });

  it('checks Scene language fields before loading assets', () => {
    expect(messageOf(() => new Scene({ mode: 'language', vocabulary: ['object'] }))).toBe(FIXTURE.messages['Scene language']);
  });

  it('keeps valid configurations constructible', () => {
    expect(Object.keys(new Planner(ranking()).configuration()).every((key) => Planner.configFields.includes(key))).toBe(true);
    expect(Object.keys(new Investigator(investigatorConfig()).configuration()).every((key) => Investigator.configFields.includes(key))).toBe(true);
    expect(Object.keys(new Chatbot(tinyConfig()).configuration()).every((key) => Chatbot.configFields.includes(key))).toBe(true);
    expect(Object.keys(new Scene(scene()).configuration()).every((key) => Scene.rankingFields.includes(key))).toBe(true);
  });

  it.each([
    ['Planner', Planner, ranking], ['Investigator', Investigator, investigatorConfig],
    ['Chatbot', Chatbot, tinyConfig], ['Scene', Scene, scene],
  ] as const)('%s.fromPretrained rejects a saved unknown field', async (name, Tool, make) => {
    const directory = join(temp.dir, name);
    await (new (Tool as Construct)(make()) as { savePretrained(path: string): Promise<string> }).savePretrained(directory);
    const load = (Tool as unknown as { fromPretrained(path: string, options: object): Promise<unknown> }).fromPretrained;
    expect(await load.call(Tool, directory, { localFilesOnly: true })).toBeInstanceOf(Tool);
    const manifestPath = join(directory, 'tensorcode_config.json');
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    manifest.config.obsolete_field = true;
    writeFileSync(manifestPath, JSON.stringify(manifest));
    await expect(load.call(Tool, directory, { localFilesOnly: true })).rejects.toThrow(/configuration fields: \['obsolete_field'\]; valid fields/);
  });
});

describe('foundation rank encoders', () => {
  it.each(['bert', 'electra'])('describe the %s native configuration and tensor schemas like Python', (modelType) => {
    const native = FIXTURE.foundation_configs[modelType];
    for (const [name, Tool] of [['Planner', Planner], ['Investigator', Investigator], ['Decision', Decision]] as const) {
      manualSeed(0);
      const model = new Tool({
        foundation_config: clone(native.foundation_config), tokenizer_json: native.tokenizer_json,
        tokenizer_special_tokens: clone(native.tokenizer_special_tokens),
      });
      const expected = FIXTURE.encoders[`${name}/${modelType}`];
      expect(JSON.parse(JSON.stringify(model.rank.encode.configuration()))).toEqual(JSON.parse(JSON.stringify(expected.encode)));
      const fingerprints = Object.fromEntries(Object.entries(model.operationBindings())
        .map(([key, operation]) => [key, fingerprint(operationConfiguration(operation))]));
      expect(fingerprints).toEqual(expected.fingerprints);
    }
  });
});
