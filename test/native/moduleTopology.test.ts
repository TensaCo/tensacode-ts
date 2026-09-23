/**
 * Native module trees equal transformers' ``named_modules()``, including the
 * parameter-free activation modules, and a Python directory checkpoint of a
 * native-backed tool restores in TypeScript
 * (``scripts/fixtures/topology_fixtures.py``).
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { nativeConfig } from '../../src/_internal/native/config.js';
import { createNativeModel, type NativeHead } from '../../src/_internal/native/registry.js';
import { Chatbot } from '../../src/tools/chatbot.js';
import { Trainer } from '../../src/training/index.js';
import { fixtureJson } from '../helpers/fixtures.js';
import { tensor, type TensorModule } from '../../src/nn/index.js';

interface TopologyCase { head: NativeHead; config: Record<string, unknown>; modules: string[]; parameters: string[] }

const cases = fixtureJson<Record<string, TopologyCase>>('native_modules.json');
const resume = new URL('../fixtures/training/python_chatbot_resume/', import.meta.url).pathname;

describe('native module topology matches transformers', () => {
  for (const [name, record] of Object.entries(cases)) {
    it(name, () => {
      const model = createNativeModel(nativeConfig(record.config), record.head);
      expect(model.namedModules().map(([path]) => path)).toEqual(record.modules);
      expect([...model.stateDict().keys()]).toEqual(record.parameters);
    });
  }

  it('activation modules compute the configured activation', () => {
    const model = createNativeModel(nativeConfig(cases.bert!.config), 'base');
    const act = model.getSubmodule('encoder.layer.0.intermediate.intermediate_act_fn') as unknown as TensorModule;
    const out = act.forward(tensor([-1, 0, 1])).tolist() as number[];
    expect(out[0]).toBeCloseTo(-0.15865525, 6);
    expect(out[1]).toBe(0);
    expect(out[2]).toBeCloseTo(0.84134475, 6);
  });
});

describe('Python directory checkpoint of a native-backed tool', () => {
  it('restores modes, steps, progress, weights and optimizer state', async () => {
    const expected = JSON.parse(readFileSync(`${resume}expected.json`, 'utf8'));
    const model = await Chatbot.fromPretrained(`${resume}initial`);
    const trainer = Trainer.fromTool(model, { lr: 0.05 });
    const progress = await trainer.loadCheckpoint(`${resume}checkpoint`);
    expect(progress).toEqual(expected.progress);
    expect(trainer.steps).toBe(expected.steps);
    expect(model.training).toBe(true);
    const trained = await Chatbot.fromPretrained(`${resume}trained`);
    const reference = trained.stateDict();
    for (const [key, value] of model.stateDict()) {
      expect(value.equal(reference.get(key)!), key).toBe(true);
    }
    model.eval();
    expect(model.generateBatch(['hello world'])).toEqual(expected.generation);
  });
});
