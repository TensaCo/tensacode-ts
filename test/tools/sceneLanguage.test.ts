/**
 * Scene language mode (Python ``tests/models/test_scene_language.py``) with
 * parity fixtures from ``scripts/fixtures/scene_language_fixtures.py``:
 * Python artifacts load and interpret identically, the SmolVLM processor is
 * reproduced exactly, and ``fromLanguageFoundation`` writes the same processor
 * assets (and hashes) as Python.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { noGrad, tensor, type Tensor } from '../../src/nn/index.js';
import { ValueError } from '../../src/errors.js';
import { sha256Hex } from '../../src/_internal/json.js';
import { tensorBytes } from '../../src/nn/safetensors.js';
import { Scene, type SceneInterpretation } from '../../src/tools/scene.js';
import { pixelsToUint8 } from '../../src/tools/sceneLanguage.js';
import { Idefics3Processor } from '../../src/_internal/native/idefics3Processing.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fromJson } from '../helpers/fixtures.js';
import { fingerprints, scratchDirectory } from '../vec/helpers.js';


const root = new URL('../fixtures/scene_language/', import.meta.url).pathname;
const records = JSON.parse(readFileSync(join(root, 'records.json'), 'utf8'));
const scratch = scratchDirectory('tensorcode-scene-language-');

function inputs(pixels: Tensor, question = 'describe object', sourceId = 'fixture:image') {
  return { pixels, question, source_id: sourceId };
}

function expectReceipt(actual: SceneInterpretation, expected: SceneInterpretation): void {
  const strip = (receipt: SceneInterpretation) => ({ ...receipt, workspace: { ...receipt.workspace, attention: null, relations: null } });
  expect(strip(actual)).toEqual(strip(expected));
  expectClose(actual.workspace.attention.flat(), expected.workspace.attention.flat(), 1e-5, 1e-5);
  expectClose(actual.workspace.relations.flat(), expected.workspace.relations.flat(), 1e-5, 1e-5);
}

async function tiny(): Promise<Scene> {
  return Scene.fromPretrained(join(root, 'tiny'));
}

describe('Scene language artifacts written by Python', () => {
  const record = records.tiny;
  const pixels = fromJson(record.pixels);

  it('configuration, topology, bindings and byte-identical re-save', async () => {
    const tool = await tiny();
    expect(tool.configuration()).toEqual(record.configuration);
    expect([...tool.stateDict().keys()]).toEqual(record.state_keys);
    expect(tool.namedModules().map(([name]) => name)).toEqual(record.modules);
    expect(fingerprints(tool.operationBindings())).toEqual(record.bindings);
    const target = join(scratch, 'tiny-resave');
    await tool.savePretrained(target);
    for (const file of ['tensorcode_config.json', 'model.safetensors']) {
      expect(readFileSync(join(target, file)).equals(readFileSync(join(root, 'tiny', file))), file).toBe(true);
    }
    for (const name of Object.keys(record.configuration.processor_hashes)) {
      expect(readFileSync(join(target, 'processor', name), 'utf8')).toBe(readFileSync(join(root, 'tiny', 'processor', name), 'utf8'));
    }
  });

  it('prepare, interpret, loss and residual gradient match Python', async () => {
    const tool = await tiny();
    const value = inputs(pixels);
    const prepared = noGrad(() => tool.language!.prepare(value));
    expect(prepared.batch.inputIds.tolist()).toEqual(record.prepared.input_ids);
    expect(prepared.visualTokens).toBe(record.prepared.visual_tokens);
    expectClose(prepared.batch.imageHiddenStates.data, record.prepared.image_hidden_states.data, 1e-5, 1e-5);
    expectReceipt(tool.interpret(value, { maxNewTokens: 8 }), record.receipt);
    expectReceipt(tool.call(value) as SceneInterpretation, record.receipt);
    expectReceipt(tool.interpret(value, { maxNewTokens: 2 }), record.receipt_short);
    expect(noGrad(() => tool.loss(value, 'left object')).item()).toBeCloseTo(record.loss, 5);
    tool.loss(value, 'right object').backward();
    expect(tool.language!.gate.grad!.item()).toBeCloseTo(record.gate_grad, 6);
    expect(tool.language!.model.parameters().every((parameter) => parameter.grad === null)).toBe(true);
  });
});

describe('Scene language mechanics (tests/models/test_scene_language.py)', () => {
  const pixels = fromJson(records.tiny.pixels);

  it('receipts are unverified, full-image and uncalibrated; artifacts reload identically', async () => {
    const tool = await tiny();
    noGrad(() => tool.language!.gate.fill_(0));
    const result = tool.interpret(inputs(pixels), { maxNewTokens: 2 });
    expect(result.verification).toBe('unverified');
    expect(result.uncertainty).toEqual({ status: 'uncalibrated', confidence: null });
    expect(result.source.kind).toBe('full-image');
    expect(result.source.shape).toEqual([3, 8, 8]);
    expect(result.source.source_id).toBe('fixture:image');
    expect(result.workspace.active).toBe(false);
    expect(result.workspace.visual_tokens).toBe(4);
    expect('boxes' in result || 'claims' in result).toBe(false);
    await tool.savePretrained(join(scratch, 'mechanics'));
    const loaded = await Scene.fromPretrained(join(scratch, 'mechanics'));
    expect(loaded.interpret(inputs(pixels), { maxNewTokens: 2 })).toEqual(result);
    expect(existsSync(join(scratch, 'mechanics', 'processor', 'tokenizer.json'))).toBe(true);
    const ids = [...new Uint8Array(tensorBytes(pixels))];
    expect(result.source.sha256).toBe(records.tiny.receipt.source.sha256);
    expect(ids.length).toBe(3 * 8 * 8 * 4);
  });

  it('teacher feedback trains the residual without entering the workspace', async () => {
    const tool = await tiny();
    noGrad(() => tool.language!.gate.fill_(0));
    const captured: Tensor[] = [];
    const workspace = tool.language!.workspace;
    const forward = workspace.forward.bind(workspace);
    workspace.forward = (encoded: Tensor, mask: Tensor | null = null) => {
      captured.push(encoded.detach().clone());
      return forward(encoded, mask);
    };
    const a = tool.loss(inputs(pixels), 'left object');
    tool.loss(inputs(pixels), 'right object');
    workspace.forward = forward;
    expect(captured[0]!.equal(captured[1]!)).toBe(true);
    a.backward();
    expect(Math.abs(tool.language!.gate.grad!.item())).toBeGreaterThan(0);
    expect(tool.language!.model.parameters().every((parameter) => parameter.grad === null)).toBe(true);
    tool.zeroGrad();
    noGrad(() => tool.language!.gate.fill_(0.2));
    tool.loss(inputs(pixels), 'left object').backward();
    expect(tool.language!.workspace.queries.grad!.abs().sum().item()).toBeGreaterThan(0);
  });

  it('a zero gate preserves the foundation image encoding; a nonzero gate revises it', async () => {
    const tool = await tiny();
    noGrad(() => tool.language!.gate.fill_(0));
    const { batch } = noGrad(() => tool.language!.prepare(inputs(pixels)));
    const processor = tool.language!.processor;
    const processed = processor.call(processor.applyChatTemplate(
      [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: 'describe object' }] }], { addGenerationPrompt: true }), [pixelsToUint8(pixels)]);
    const features = noGrad(() => tool.language!.model.getImageFeatures(processed.pixelValues, processed.pixelAttentionMask));
    expect(batch.imageHiddenStates.equal(features)).toBe(true);
    expect(noGrad(() => tool.language!.model.forward(batch)).logits.allFinite()).toBe(true);
    noGrad(() => tool.language!.gate.fill_(0.5));
    const revised = noGrad(() => tool.language!.prepare(inputs(pixels)));
    expect(revised.batch.imageHiddenStates.equal(batch.imageHiddenStates)).toBe(false);
  });

  it('generation dictionary settings are accepted; inference then learning keeps gradients', async () => {
    const tool = await tiny();
    tool.language!.generationConfig.return_dict_in_generate = true;
    expect(typeof tool.interpret(inputs(pixels), { maxNewTokens: 1 }).interpretation).toBe('string');
    tool.loss(inputs(pixels), 'left object').backward();
    expect(Number.isFinite(tool.language!.gate.grad!.item())).toBe(true);
  });

  it('rejects invalid inputs', async () => {
    const tool = await tiny();
    const bad = [
      { question: '' }, { source_id: '' }, { pixels: tensor(new Array(64).fill(0), { shape: [1, 8, 8] }) },
      { pixels: tensor(new Array(192).fill(Number.NaN), { shape: [3, 8, 8] }) },
    ];
    for (const replacement of bad) expect(() => tool.interpret({ ...inputs(pixels), ...replacement } as never)).toThrow(ValueError);
    expect(() => tool.interpret(inputs(pixels), { maxNewTokens: 9 })).toThrow(/max_new_tokens/);
    expect(() => tool.loss(inputs(pixels), ' ')).toThrow(/nonempty/);
  });

  it('processor asset integrity and path names are verified before use', async () => {
    const tool = await tiny();
    await tool.savePretrained(join(scratch, 'integrity'));
    const path = join(scratch, 'integrity', 'processor', 'tokenizer.json');
    writeFileSync(path, `${readFileSync(path, 'utf8')} `);
    await expect(Scene.fromPretrained(join(scratch, 'integrity'))).rejects.toThrow(/checksum/);
    await tool.savePretrained(join(scratch, 'names'));
    const configPath = join(scratch, 'names', 'tensorcode_config.json');
    const saved = JSON.parse(readFileSync(configPath, 'utf8'));
    saved.config.processor_hashes = { '../../outside.json': 'invalid' };
    writeFileSync(configPath, JSON.stringify(saved));
    await expect(Scene.fromPretrained(join(scratch, 'names'))).rejects.toThrow(/asset names/);
  });

  it('language feedback replays into a freshly loaded tool', async () => {
    const tool = await tiny();
    noGrad(() => tool.language!.gate.fill_(0));
    const trainer = Trainer.fromTool(tool, { lr: 0.1 });
    const experience = trainer.capture(inputs(pixels), 'left object', { source: 'authored mechanism fixture' });
    await experience.save(join(scratch, 'experience.json'), { operations: trainer.operations });
    await tool.savePretrained(join(scratch, 'replay'));
    const restored = await Scene.fromPretrained(join(scratch, 'replay'));
    const replay = Trainer.fromTool(restored, { lr: 0.1 });
    const record = await loadExperience(join(scratch, 'experience.json'), { operations: replay.operations });
    replay.step(record);
    expect(Math.abs(restored.language!.gate.item())).toBeGreaterThan(0);
  });
});

describe('SmolVLM processor and a small Idefics3 model', () => {
  const smol = records.smol;
  it.skipIf(!smol)('processor outputs equal transformers (image splitting, LANCZOS, prompt expansion)', async () => {
    const tool = await Scene.fromPretrained(join(root, 'smol'));
    const processor: Idefics3Processor = tool.language!.processor;
    for (const record of smol.processing) {
      const out = processor.call('<|im_start|>User:<image>describe<end_of_utterance>\nAssistant:', [pixelsToUint8(fromJson(record.pixels))]);
      expect([...out.inputIds.data]).toEqual(record.input_ids);
      expect(out.pixelValues.shape).toEqual(record.pixel_shape);
      expect(sha256Hex(tensorBytes(out.pixelValues))).toBe(record.pixel_sha256);
      expect(out.pixelAttentionMask!.sum().item()).toBe(record.mask_sum);
    }
  }, 120_000);

  it.skipIf(!smol)('prepare, interpret and loss equal Python', async () => {
    const tool = await Scene.fromPretrained(join(root, 'smol'));
    expect(tool.configuration()).toEqual(smol.scene.configuration);
    expect(fingerprints(tool.operationBindings())).toEqual(smol.scene.bindings);
    const value = inputs(fromJson(smol.scene.pixels), 'Describe the image.', 'fixture:smol');
    const prepared = noGrad(() => tool.language!.prepare(value));
    expect(prepared.batch.inputIds.tolist()).toEqual(smol.scene.prepared.input_ids);
    expectClose(prepared.batch.imageHiddenStates.data, smol.scene.prepared.image_hidden_states.data, 1e-5, 1e-5);
    expectReceipt(tool.interpret(value), smol.scene.receipt);
    expect(noGrad(() => tool.loss(value, 'A small image.')).item()).toBeCloseTo(smol.scene.loss, 4);
  }, 120_000);
});

describe('Scene.fromLanguageFoundation', () => {
  const foundations = records.foundations;

  it('a local foundation directory yields Python\'s configuration and processor assets', async () => {
    const tool = await Scene.fromLanguageFoundation('test/fixtures/scene_language/tiny_foundation', { revision: null, localFilesOnly: true });
    expect(tool.language!.assets).toEqual(foundations.local.assets);
    expect(tool.configuration()).toEqual(foundations.local.configuration);
    expect(tool.language!.gate.item()).toBe(0);
    expect(tool.training).toBe(false);
  });

  it('named chat templates are preserved as processor assets', async () => {
    const tool = await Scene.fromLanguageFoundation('test/fixtures/scene_language/tiny_foundation_named', { revision: 'pinned', localFilesOnly: true });
    expect(tool.language!.assets).toEqual(foundations.local_named.assets);
    expect(tool.configuration()).toEqual(foundations.local_named.configuration);
    await tool.savePretrained(join(scratch, 'named'));
    const loaded = await Scene.fromPretrained(join(scratch, 'named'));
    expect(loaded.language!.assets['additional_chat_templates/alternative.jinja']).toBe(tool.language!.assets['additional_chat_templates/alternative.jinja']);
  });

  const snapshot = foundations.hub_true?.snapshot as string | undefined;
  const cached = snapshot !== undefined
    && existsSync(join(homedir(), '.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots', snapshot));
  it.skipIf(!cached)('the cached SmolVLM-256M snapshot yields Python\'s configuration (asset hashes, configs)', async () => {
    // The fixtures were generated offline, where transformers records local_files_only=True.
    vi.stubEnv('HF_HUB_OFFLINE', '1');
    for (const flag of [true, false]) {
      const tool = await Scene.fromLanguageFoundation('HuggingFaceTB/SmolVLM-256M-Instruct', { revision: snapshot!, localFilesOnly: flag });
      expect(tool.configuration()).toEqual(foundations[`hub_${flag}`].configuration);
    }
    vi.unstubAllEnvs();
  }, 300_000);
});
