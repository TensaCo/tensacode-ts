/** Port of python/tests/models/test_chatbot_model.py (tiny random T5; mechanisms, not quality). */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Parameter, Tensor, cat, full, manualSeed, noGrad, ones, randn, serializeModel, tensor, zeros } from '../../src/nn/index.js';
import { Chatbot, boundedMemoryUpdate } from '../../src/tools/chatbot.js';
import { ValueError } from '../../src/errors.js';
import { scratch, tinyConfig } from './helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

function rms(values: number[]): number {
  return Math.sqrt(values.reduce((total, value) => total + value * value, 0) / values.length);
}

describe('Chatbot model', () => {
  it('owns eager parameters that receive gradients', () => {
    const model = new Chatbot(tinyConfig());
    const params = new Set(model.parameters());
    const loss = model.lossBatch(['hello world'], ['answer']);
    loss.backward();
    expect(Number.isFinite(loss.item())).toBe(true);
    expect(model.workspace.parameters().some((p) => p.grad !== null && p.grad.abs().sum().item() > 0)).toBe(true);
    expect(new Set(model.parameters())).toEqual(params);
  });

  it('evidence and workspace ablation affect computation', () => {
    const model = new Chatbot(tinyConfig()).eval();
    noGrad(() => {
      const first = model.encodeWorkspace(['hello']).conditioning;
      const second = model.encodeWorkspace(['world']).conditioning;
      expect(first.equal(second)).toBe(false);
      const zero = model.encodeWorkspace(['hello'], { workspaceAblation: 'zero' }).conditioning;
      expect(zero.ne(0).sum().item()).toBe(0);
      expect(model.lossBatch(['hello'], ['answer']).item()).not.toBe(model.lossBatch(['hello'], ['answer'], { workspaceAblation: 'zero' }).item());
    });
  });

  it('never passes targets to the input encoder', () => {
    const model = new Chatbot(tinyConfig());
    const seen: unknown[] = [];
    const original = model.encoder.forward.bind(model.encoder);
    model.encoder.forward = (value, context) => {
      seen.push(value);
      return original(value, context);
    };
    model.lossBatch(['hello'], ['answer']);
    expect(seen).toEqual([['hello']]);
  });

  it('round-trips checkpoints and keeps sessions separate', async () => {
    const model = new Chatbot(tinyConfig()).eval();
    const expected = noGrad(() => model.lossBatch(['hello'], ['answer']).item());
    await model.savePretrained(join(temp.dir, 'model'));
    const loaded = (await Chatbot.fromPretrained(join(temp.dir, 'model'))).eval();
    expect(noGrad(() => loaded.lossBatch(['hello'], ['answer']).item())).toBe(expected);
    const a = loaded.newSession();
    const b = loaded.newSession();
    a.call('hello');
    expect(a.history.length).toBe(2);
    expect(b.history).toEqual([]);
    expect(loaded.history).toEqual([]);
    await a.save(join(temp.dir, 'session.json'));
    await b.load(join(temp.dir, 'session.json'));
    expect(b.history).toEqual(a.history);
    b.history[0]!.text = 'world';
    expect(a.history[0]!.text).toBe('hello');
    expect('history' in JSON.parse(readFileSync(join(temp.dir, 'model/tensorcode_config.json'), 'utf8'))).toBe(false);
  });

  it('rolls back a failed decode and restores module modes', () => {
    const model = new Chatbot(tinyConfig()).train();
    model.decoder.forward = () => {
      throw new Error('decode failed');
    };
    expect(() => model.call('hello')).toThrow('decode failed');
    expect(model.history).toEqual([]);
    expect(model.lastResult).toBeNull();
    expect(model.training).toBe(true);
  });

  it('implements the objective training protocol', () => {
    const model = new Chatbot(tinyConfig());
    const loss = model.trainingOperation.call({ inputs: ['hello'], targets: ['answer'] });
    expect(loss.ndim).toBe(0);
    expect(loss.requiresGrad).toBe(true);
    expect(model.operationBindings().objective).toBe(model.trainingOperation);
    expect(model.trainingOperation.parameters()).toEqual(model.parameters());
    expect(model.trainingInputsIncludeTargets).toBe(true);
  });

  it('bypass is the exact foundation memory and unknown ablations are rejected', () => {
    const model = new Chatbot(tinyConfig()).eval();
    noGrad(() => {
      const raw = model.encoder.call(['hello']).encoded;
      const bypass = model.encodeWorkspace(['hello'], { workspaceAblation: 'bypass' });
      expect(bypass.conditioning.equal(raw)).toBe(true);
      expect(() => model.encodeWorkspace(['hello'], { workspaceAblation: 'unknown' })).toThrow(/ablation/);
    });
  });

  it('checks session compatibility and bounded capacity', async () => {
    const model = new Chatbot(tinyConfig());
    model.generateBatch = (inputs) => inputs.map(() => 'answer');
    for (let turn = 0; turn < 4; turn += 1) model.call('hello');
    expect(model.history.length).toBe(4);
    expect(model.history[3]!.source_id).toBe('turn-7');
    const path = join(temp.dir, 'capacity.json');
    await model.saveSession(path);
    const state = JSON.parse(readFileSync(path, 'utf8'));
    state.model = 'different';
    writeFileSync(path, JSON.stringify(state));
    const before = model.history;
    await expect(model.loadSession(path)).rejects.toThrow(/incompatible/);
    expect(model.history).toEqual(before);
  });

  it('rejects corrupt source IDs transactionally', async () => {
    const model = new Chatbot(tinyConfig());
    model.generateBatch = (inputs) => inputs.map(() => 'answer');
    model.call('hello');
    const path = join(temp.dir, 'corrupt.json');
    await model.saveSession(path);
    const state = JSON.parse(readFileSync(path, 'utf8'));
    state.history[0].source_id = 'invalid';
    writeFileSync(path, JSON.stringify(state));
    const original = model.history;
    await expect(model.loadSession(path)).rejects.toThrow(/source IDs/);
    expect(model.history).toEqual(original);
  });

  it('commits complete turns for consecutive session calls', async () => {
    const model = new Chatbot(tinyConfig());
    model.generateBatch = (inputs) => inputs.map(() => 'answer');
    await Promise.all(['hello', 'world'].map(async (value) => model.call(value)));
    expect(model.history.map((row) => row.source_id)).toEqual(['turn-0', 'turn-1', 'turn-2', 'turn-3']);
    expect(model.history.map((row) => row.role)).toEqual(['user', 'assistant', 'user', 'assistant']);
  });

  it('foundation bootstrap preserves actual untied weights', async () => {
    const model = new Chatbot(tinyConfig());
    noGrad(() => model.foundation.setParameterAt('lm_head.weight', new Parameter(model.foundation.lm_head.weight.detach().clone().add(1))));
    const root = join(temp.dir, 'foundation');
    mkdirSync(root, { recursive: true });
    writeFileSync(join(root, 'config.json'), JSON.stringify(model.foundation.config.toDict()));
    writeFileSync(join(root, 'model.safetensors'), serializeModel(model.foundation));
    writeFileSync(join(root, 'tokenizer.json'), tinyConfig().tokenizer_json as string);
    writeFileSync(join(root, 'tokenizer_config.json'), JSON.stringify({
      tokenizer_class: 'PreTrainedTokenizerFast', pad_token: '<pad>', eos_token: '</s>', unk_token: '<unk>',
    }));
    const restored = await Chatbot.fromFoundation(root);
    expect(restored.foundation.config.get('tie_word_embeddings')).toBe(true);
    expect(restored.configuration().untied_lm_head).toBe(true);
    expect(restored.foundation.shared.weight.equal(model.foundation.shared.weight)).toBe(true);
    expect(restored.foundation.lm_head.weight.equal(model.foundation.lm_head.weight)).toBe(true);
    expect(restored.foundation.shared.weight).not.toBe(restored.foundation.lm_head.weight);
    model.eval();
    restored.eval();
    noGrad(() => {
      const inputs = model.tokenizer.encodeTensors(['hello']).input_ids;
      const target = model.tokenizer.encodeTensors(['answer']).input_ids;
      const expected = model.foundation.forward({ inputIds: inputs, labels: target }).loss!.item();
      expect(restored.lossBatch(['hello'], ['answer'], { workspaceAblation: 'bypass' }).item()).toBe(expected);
    });
  });

  it('encoder fingerprint distinguishes tokenizers', () => {
    const model = new Chatbot(tinyConfig());
    const config = tinyConfig();
    const tokenizer = JSON.parse(config.tokenizer_json as string);
    const vocab = tokenizer.model.vocab;
    [vocab.hello, vocab.world] = [vocab.world, vocab.hello];
    config.tokenizer_json = JSON.stringify(tokenizer);
    expect(model.encoder.configuration()).not.toEqual(new Chatbot(config).encoder.configuration());
  });

  it('generation configuration survives checkpoints', async () => {
    const model = new Chatbot(tinyConfig());
    model.generationConfig.num_beams = 3;
    const original = model.decoder.configuration();
    await model.savePretrained(join(temp.dir, 'generation'));
    const restored = await Chatbot.fromPretrained(join(temp.dir, 'generation'));
    expect(restored.generationConfig.num_beams).toBe(3);
    expect(restored.decoder.configuration()).toEqual(original);
  });

  it('generation preserves mixed module modes', () => {
    const model = new Chatbot(tinyConfig()).train();
    model.foundation.eval();
    const expected = model.namedModules().map(([name, module]) => [name, module.training]);
    model.generateBatch(['hello']);
    expect(model.namedModules().map(([name, module]) => [name, module.training])).toEqual(expected);
  });

  it('memory update configuration is explicit', () => {
    expect(new Chatbot(tinyConfig()).configuration().memory_update).toBe('relative_rms_bounded');
    expect(() => new Chatbot({ ...tinyConfig(), memory_update: 'unbounded' })).toThrow(/memory_update/);
    expect(() => new Chatbot({ ...tinyConfig(), max_turns: 0 })).toThrow(ValueError);
    expect(() => new Chatbot({ ...tinyConfig(), memory_mode: 'other' })).toThrow(/memory_mode/);
  });
});

describe('bounded memory update', () => {
  it('bounds relative RMS under pathological scale and gate', () => {
    const native = tensor([[[1, -2], [3, 4]], [[0.01, -0.02], [0.03, 0.04]]]);
    const update = tensor([[[1, 2], [-3, 4]], [[4, -3], [2, 1]]]).mul(1e30);
    const mask = ones([2, 2], { dtype: 'bool' });
    for (const gate of [-1e6, 1e6, 0.004235]) {
      const residual = boundedMemoryUpdate(native, update, mask, tensor(gate));
      expect(residual.allFinite()).toBe(true);
      const nativeRms = native.square().mean([1, 2]).sqrt().toArray();
      const residualRms = residual.square().mean([1, 2]).sqrt().toArray();
      residualRms.forEach((value, index) => expect(value).toBeLessThanOrEqual(nativeRms[index]! * Math.abs(Math.tanh(gate)) + 1e-6));
    }
  });

  it('is per example and ignores masked padding', () => {
    const native = tensor([[[1, 2], [3, 4]]]);
    const update = tensor([[[2, -1], [4, -3]]]);
    const gate = tensor(0.2);
    const expected = boundedMemoryUpdate(native, update, ones([1, 2], { dtype: 'bool' }), gate);
    const paddedNative = cat([native, full([1, 3, 2], Number.NaN)], 1);
    const paddedUpdate = cat([update, full([1, 3, 2], Number.POSITIVE_INFINITY)], 1);
    const mask = tensor([[1, 1, 0, 0, 0]], { dtype: 'bool' });
    const actual = boundedMemoryUpdate(paddedNative, paddedUpdate, mask, gate);
    const values = actual.toArray();
    expect(values.slice(0, 4).map((value, index) => Math.abs(value - expected.data[index]!) < 1e-6).every(Boolean)).toBe(true);
    expect(values.slice(4).every((value) => value === 0)).toBe(true);
    const batched = boundedMemoryUpdate(cat([native, native.mul(1e8)]), cat([update, update.mul(1e20)]), ones([2, 2], { dtype: 'bool' }), gate);
    const rows = batched.toArray();
    rows.slice(0, 4).forEach((value, index) => expect(Math.abs(value - expected.data[index]!)).toBeLessThan(1e-6));
    rows.slice(4).forEach((value, index) => expect(Math.abs(value / 1e8 - expected.data[index]!)).toBeLessThanOrEqual(1e-5 * Math.abs(expected.data[index]!) + 1e-8));
  });

  it('is finite and differentiable at zeros and with an empty mask', () => {
    for (const [nativeZero, updateZero, emptyMask] of [[true, false, false], [false, true, false], [true, true, false], [false, false, true]]) {
      const native = (nativeZero ? zeros([1, 3, 2]) : ones([1, 3, 2])).requiresGrad_();
      const update = (updateZero ? zeros([1, 3, 2]) : ones([1, 3, 2])).requiresGrad_();
      const gate = tensor(0.2).requiresGrad_();
      const residual = boundedMemoryUpdate(native, update, full([1, 3], emptyMask ? 0 : 1, { dtype: 'bool' }), gate);
      expect(residual.allFinite()).toBe(true);
      expect(residual.ne(0).sum().item()).toBe(0);
      residual.sum().backward();
      for (const value of [native, update, gate]) {
        expect(value.grad).not.toBeNull();
        expect(value.grad!.allFinite()).toBe(true);
      }
    }
  });

  it('backpropagates through every owned component', () => {
    const model = new Chatbot(tinyConfig());
    model.lossBatch(['hello world'], ['answer']).backward();
    for (const module of [model.foundation, model.workspace, model.memoryProjection]) {
      expect(module.parameters().some((p) => p.grad !== null && p.grad.allFinite() && p.grad.abs().sum().item() > 0)).toBe(true);
    }
    expect(model.memoryGate.grad).not.toBeNull();
    expect(Math.abs(model.memoryGate.grad!.item())).toBeGreaterThan(0);
  });

  it('bounds actual conditioning with extreme projection and padding', () => {
    const model = new Chatbot(tinyConfig()).eval();
    noGrad(() => {
      model.memoryProjection.weight.fill_(1e25);
      model.memoryProjection.bias!.fill_(-1e25);
      model.memoryGate.fill_(1e6);
    });
    const inputs = ['hello', 'hello world'];
    noGrad(() => {
      const encoded = model.encoder.call(inputs);
      const output = model.encodeWorkspace(inputs).conditioning;
      const residual = output.sub(encoded.encoded);
      expect(output.allFinite()).toBe(true);
      const [batch, tokens, width] = encoded.encoded.shape as [number, number, number];
      const mask = encoded.mask.toArray();
      for (let index = 0; index < batch; index += 1) {
        const nativeValues: number[] = [];
        const residualValues: number[] = [];
        for (let token = 0; token < tokens; token += 1) {
          for (let d = 0; d < width; d += 1) {
            const position = (index * tokens + token) * width + d;
            if (mask[index * tokens + token]) {
              nativeValues.push(encoded.encoded.data[position]!);
              residualValues.push(residual.data[position]!);
            } else {
              expect(residual.data[position]).toBe(0);
            }
          }
        }
        expect(rms(residualValues)).toBeLessThanOrEqual(rms(nativeValues) * (1 + 1e-6));
      }
      expect(model.encodeWorkspace(inputs, { workspaceAblation: 'bypass' }).conditioning.equal(encoded.encoded)).toBe(true);
    });
  });

  it('zero projection keeps backward finite; slot memory stays separate', () => {
    const model = new Chatbot(tinyConfig());
    noGrad(() => {
      model.memoryProjection.weight.zero_();
      model.memoryProjection.bias!.zero_();
    });
    model.lossBatch(['hello world'], ['answer']).backward();
    expect(model.parameters().every((p) => p.grad === null || p.grad.allFinite())).toBe(true);
    const slots = new Chatbot({ ...tinyConfig(), memory_mode: 'slots' }).eval();
    noGrad(() => {
      const encoded = slots.encoder.call(['hello world']);
      const expected = slots.workspace.forward(encoded.encoded, encoded.mask.bool()).conditioning;
      expect(slots.encodeWorkspace(['hello world']).conditioning.equal(expected)).toBe(true);
    });
  });

  it('keeps gradients finite for subnormal projected updates', () => {
    const native = ones([1, 2, 2]).requiresGrad_();
    const update = tensor([[[1, 2], [3, 4]]]).mul(1e-40).detach().requiresGrad_();
    const gate = tensor(0.2).requiresGrad_();
    const residual = boundedMemoryUpdate(native, update, ones([1, 2], { dtype: 'bool' }), gate);
    residual.select(0, 0).select(0, 0).select(0, 0).backward();
    expect(residual.allFinite()).toBe(true);
    for (const value of [native, update, gate]) expect(value.grad!.allFinite()).toBe(true);
    expect(residual.square().mean().sqrt().item()).toBeLessThanOrEqual(native.square().mean().sqrt().item());
  });

  it('bypass does not depend on an invalid disabled projection', () => {
    const model = new Chatbot(tinyConfig()).eval();
    noGrad(() => model.memoryProjection.weight.fill_(Number.POSITIVE_INFINITY));
    noGrad(() => {
      const expected = model.encoder.call(['hello']).encoded;
      expect(model.encodeWorkspace(['hello'], { workspaceAblation: 'bypass' }).conditioning.equal(expected)).toBe(true);
      expect(() => model.encodeWorkspace(['hello'])).toThrow(/finite/);
    });
  });

  it('bfloat16 bound allows only destination rounding', () => {
    manualSeed(1);
    const native = randn([128, 5, 16]).to('bfloat16').requiresGrad_();
    const update = randn([128, 5, 16]).mul(1e20).to('bfloat16').requiresGrad_();
    const maskValues = Array.from({ length: 128 * 5 }, (_, index) => (index % 5 === 4 ? 0 : 1));
    const mask = tensor(maskValues, { shape: [128, 5], dtype: 'bool' });
    const gate = tensor(1e6).requiresGrad_();
    const residual = boundedMemoryUpdate(native, update, mask, gate);
    const valid = (value: Tensor) => value.slice(1, 0, 4).float();
    const ratio = valid(residual).square().mean([1, 2]).sqrt().div(valid(native).square().mean([1, 2]).sqrt()).toArray();
    expect(ratio.every((value) => value <= 1 + 0.0078125)).toBe(true);
    expect(residual.slice(1, 4, 5).ne(0).sum().item()).toBe(0);
    residual.select(1, 0).select(1, 0).float().sum().backward();
    expect(native.grad!.allFinite() && update.grad!.allFinite()).toBe(true);
    expect(native.grad!.slice(1, 4, 5).ne(0).sum().item()).toBe(0);
    expect(update.grad!.slice(1, 4, 5).ne(0).sum().item()).toBe(0);
  });
});
