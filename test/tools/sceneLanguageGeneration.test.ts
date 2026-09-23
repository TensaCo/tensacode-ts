/**
 * Scene language ``interpret`` under generation configurations: beam search
 * (with transformers' ``repeat_interleave`` of image hidden states across split
 * image tiles), guidance, sequence bias, prompt lookup, watermarking and the
 * errors Python raises (``scripts/fixtures/scene_generation_fixtures.py``).
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { noGrad } from '../../src/nn/index.js';
import { Scene, type SceneInterpretation } from '../../src/tools/scene.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fromJson } from '../helpers/fixtures.js';

const root = new URL('../fixtures/scene_language/', import.meta.url).pathname;
const records = JSON.parse(readFileSync(join(root, 'records.json'), 'utf8'));
const generation = JSON.parse(readFileSync(join(root, 'generation.json'), 'utf8'));

function expectReceipt(actual: SceneInterpretation, expected: SceneInterpretation): void {
  const strip = (receipt: SceneInterpretation) => ({ ...receipt, workspace: { ...receipt.workspace, attention: null, relations: null } });
  expect(strip(actual)).toEqual(strip(expected));
  expectClose(actual.workspace.attention.flat(), expected.workspace.attention.flat(), 1e-5, 1e-5);
}

interface Case { name: string; generation: Record<string, unknown> }
interface Result { name: string; receipt?: SceneInterpretation; error?: string; message?: string }

function check(tool: Scene, base: Record<string, unknown>, value: Parameters<Scene['interpret']>[0], spec: Case, result: Result, maxNewTokens: number): void {
  tool.language!.generationConfig = { ...base, ...spec.generation } as never;
  if (result.error) {
    let caught: unknown = null;
    try {
      tool.interpret(value, { maxNewTokens });
    } catch (error) {
      caught = error;
    }
    expect((caught as Error | null)?.name, spec.name).toBe(result.error);
    expect((caught as Error).message).toBe(result.message);
    return;
  }
  expectReceipt(tool.interpret(value, { maxNewTokens }), result.receipt!);
}

describe('Scene interpret with generation settings (tiny)', () => {
  const pixels = fromJson(records.tiny.pixels);
  const value = { pixels, question: 'describe object', source_id: 'fixture:image' };
  for (const spec of generation.tiny_cases as Case[]) {
    const result = (generation.tiny as Result[]).find((item) => item.name === spec.name)!;
    it(spec.name, async () => {
      const tool = await Scene.fromPretrained(join(root, 'tiny'));
      noGrad(() => tool.language!.gate.fill_(0.3));
      check(tool, { ...tool.language!.generationConfig }, value, spec, result, 8);
    });
  }
});

describe('Scene interpret with generation settings (SmolVLM processor, split image)', () => {
  const smol = records.smol;
  for (const spec of generation.smol_cases as Case[]) {
    const result = (generation.smol as Result[]).find((item) => item.name === spec.name);
    it.skipIf(!smol || !result)(spec.name, async () => {
      const tool = await Scene.fromPretrained(join(root, 'smol'));
      noGrad(() => tool.language!.gate.fill_(-0.2));
      const value = { pixels: fromJson(smol.scene.pixels), question: 'Describe the image.', source_id: 'fixture:smol' };
      check(tool, { ...tool.language!.generationConfig }, value, spec, result!, 6);
    }, 300_000);
  }
});

describe('Scene language construction validates the generation configuration', () => {
  const config = JSON.parse(readFileSync(join(root, 'tiny', 'tensorcode_config.json'), 'utf8')).config;
  const assets = Object.fromEntries(Object.keys(config.processor_hashes).map((name) => [name, readFileSync(join(root, 'tiny', 'processor', name), 'utf8')]));
  for (const result of generation.construction as (Case & { error?: string; message?: string })[]) {
    it(result.name, () => {
      const build = () => new Scene({ ...config, generation_config: { ...config.generation_config, ...result.generation }, _language_assets: assets });
      if (!result.error) {
        expect(build).not.toThrow();
        return;
      }
      let caught: unknown = null;
      try {
        build();
      } catch (error) {
        caught = error;
      }
      expect((caught as Error | null)?.name).toBe(result.error);
      expect((caught as Error).message).toBe(result.message);
    });
  }
});
