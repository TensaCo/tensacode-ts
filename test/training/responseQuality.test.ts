/** Port of Python ``tests/models/test_response_quality.py`` plus Python artifact parity. */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { F, Tensor, noGrad, serializeModel, tensor } from '../../src/nn/index.js';
import { AXES, ResponseQualityAssessor, type QualityInputs, type QualityTargets } from '../../src/_internal/responseQuality.js';
import { NativeConfig } from '../../src/_internal/native/config.js';
import { createNativeModel } from '../../src/_internal/native/registry.js';
import { Trainer, loadExperience } from '../../src/training/index.js';
import type { JsonObject } from '../../src/_internal/json.js';
import { expectClose } from '../helpers/gradcheck.js';
import { fixtureJson } from '../helpers/fixtures.js';
import { allEqual, clones, scratchDirectory } from './helpers.js';

const scratch = scratchDirectory('tensorcode-response-quality-');
const fixtures = new URL('../fixtures/training/', import.meta.url).pathname;
const interop = fixtureJson('training/interop.json').response_quality;
const TOKENIZER_JSON = JSON.parse(readFileSync(join(fixtures, 'response_quality_bert', 'tensorcode_config.json'), 'utf8')).config.tokenizer_json as string;

function tinyConfig(modelType = 'bert', maxTokens = 128, inputFormat: string | null = null): JsonObject {
  const config: JsonObject = {
    foundation_config: {
      model_type: modelType, vocab_size: 8, hidden_size: 8, embedding_size: 8, num_hidden_layers: 1, num_attention_heads: 2,
      intermediate_size: 16, max_position_embeddings: 128, hidden_dropout_prob: 0, attention_probs_dropout_prob: 0,
    },
    tokenizer_json: TOKENIZER_JSON,
    tokenizer_special_tokens: { pad_token: '[PAD]', unk_token: '[UNK]', cls_token: '[CLS]', sep_token: '[SEP]' },
    max_tokens: maxTokens,
  };
  if (inputFormat !== null) config.input_format = inputFormat;
  return config;
}

const INPUT: QualityInputs = { question: 'question', evidence: [{ source_id: 's1', text: 'evidence yes' }], candidate: 'yes' };
const TARGET: QualityTargets = { support: true, completeness: false, constraints: null };

function withCandidate(candidate: string): QualityInputs {
  return { ...INPUT, candidate };
}

describe('ResponseQualityAssessor', () => {
  for (const inputFormat of [null, 'json', 'paired']) {
    for (const modelType of ['bert', 'electra']) {
      it(`masks unreviewed axes with native gradients (${modelType}, ${inputFormat})`, () => {
        const model = new ResponseQualityAssessor(tinyConfig(modelType, 128, inputFormat));
        const logits = model.call(INPUT);
        expect(logits.shape).toEqual([3]);
        const expected = noGrad(() => F.binaryCrossEntropyWithLogits(logits.slice(0, 0, 2), tensor([1, 0])));
        const loss = model.loss(INPUT, TARGET);
        expect(loss.item()).toBeCloseTo(expected.item(), 6);
        loss.backward();
        const grad = model.head.weight.grad!;
        expect(grad.slice(0, 0, 2).abs().sum().item()).toBeGreaterThan(0);
        expect(grad.select(0, 2).abs().sum().item()).toBe(0);
        expect(model.encoder.parameters()[0]!.grad!.abs().sum().item()).toBeGreaterThan(0);
      });
    }
  }

  for (const inputFormat of [null, 'paired']) {
    it(`enforces the strict input and target boundary (${inputFormat})`, () => {
      const model = new ResponseQualityAssessor(tinyConfig('bert', 128, inputFormat));
      for (const extra of ['targets', 'rationale', 'gold_answer']) {
        expect(() => model.call({ ...INPUT, [extra]: 'leak' } as never)).toThrow(/labels belong in targets/);
      }
      for (const bad of [{}, { support: 1, completeness: false, constraints: null }, { support: null, completeness: null, constraints: null }]) {
        expect(() => model.loss(INPUT, bad as never)).toThrow();
      }
      expect(() => model.call({ ...INPUT, evidence: [...INPUT.evidence, ...INPUT.evidence] })).toThrow(/unique/);
      expect(() => new ResponseQualityAssessor({ ...tinyConfig(), model: 'obsolete' })).toThrow(/Unknown ResponseQualityAssessor configuration fields: \['model'\]/);
    });
  }

  it('reports truncation and applies caller-owned acceptance', () => {
    const model = new ResponseQualityAssessor(tinyConfig('bert', 16));
    const receipt = model.receipt(withCandidate('yes '.repeat(100)));
    expect(receipt.input_truncated).toBe(true);
    expect(receipt.input_token_count).toBeGreaterThan(16);
    expect(receipt.source_ids).toEqual(['s1']);
    expect(ResponseQualityAssessor.accepts(receipt, 0)).toBe(false);
    const short = new ResponseQualityAssessor(tinyConfig()).receipt(INPUT);
    expect(Object.keys(short.scores).sort()).toEqual([...AXES].sort());
    expect(short.input_truncated).toBe(false);
    expect(ResponseQualityAssessor.accepts(short, 0)).toBe(true);
    for (const bad of [{ scores: { support: 1, completeness: 1, constraints: 1 } }, { ...short, scores: { support: Number.NaN, completeness: 1, constraints: 1 } }]) {
      expect(() => ResponseQualityAssessor.accepts(bad)).toThrow();
    }
    expect(() => ResponseQualityAssessor.accepts(short, 2)).toThrow(/threshold/);
  });

  for (const inputFormat of [null, 'json', 'paired']) {
    it(`calibration invalidates and survives an artifact round trip (${inputFormat})`, async () => {
      const model = new ResponseQualityAssessor(tinyConfig('bert', 128, inputFormat)).eval();
      model.fitCalibration(tensor([[1, 2, 3], [-1, -2, -3]]), [
        { support: true, completeness: true, constraints: true }, { support: false, completeness: false, constraints: false },
      ]);
      const before = model.receipt(INPUT);
      expect(Object.values(before.calibrated).every(Boolean)).toBe(true);
      const directory = join(scratch(), 'model');
      await model.savePretrained(directory);
      const loaded = await ResponseQualityAssessor.fromPretrained(directory);
      expect(loaded.configuration()).toEqual(model.configuration());
      expect('input_format' in loaded.config).toBe(inputFormat !== null);
      expect(loaded.receipt(INPUT)).toEqual(before);
      expect(noGrad(() => model.call(INPUT)).equal(noGrad(() => loaded.call(INPUT)))).toBe(true);
      noGrad(() => loaded.head.weight.add_(0.01));
      expect(Object.values(loaded.receipt(INPUT).calibrated).some(Boolean)).toBe(false);
      model.loss(INPUT, TARGET);
      expect(Object.values(model.receipt(INPUT).calibrated).some(Boolean)).toBe(false);
    });
  }

  it('resumes tool training from a checkpoint with an exact next step', async () => {
    const config = tinyConfig();
    const first = Trainer.fromTool(new ResponseQualityAssessor(config));
    const session = first.capture(INPUT, TARGET, { source: 'test:authored' });
    first.step(session);
    const directory = scratch();
    mkdirSync(directory, { recursive: true });
    await session.save(join(directory, 'experience.json'), { operations: first.operations });
    await first.saveCheckpoint(join(directory, 'resume'), { progress: { epoch: 1 } });
    const expectedLoss = first.step(session);
    const expected = clones(first.parameters);
    const second = Trainer.fromTool(new ResponseQualityAssessor(config));
    const restored = await loadExperience(join(directory, 'experience.json'), { operations: second.operations });
    expect(await second.loadCheckpoint(join(directory, 'resume'))).toEqual({ epoch: 1 });
    expect(second.step(restored)).toBe(expectedLoss);
    expect(allEqual(expected, second.parameters)).toBe(true);
    expect(Object.keys(first.operations).sort()).toEqual(['objective', 'operation']);
  });

  it('averages masked examples in a batch and rejects truncated supervision', () => {
    const model = new ResponseQualityAssessor(tinyConfig());
    const second: QualityTargets = { support: false, completeness: null, constraints: null };
    const expected = (model.loss(INPUT, TARGET).item() + model.loss(INPUT, second).item()) / 2;
    expect(model.loss([INPUT, INPUT], [TARGET, second]).item()).toBeCloseTo(expected, 6);
    expect(() => model.loss([INPUT, INPUT], [TARGET])).toThrow();
    expect(() => model.loss([INPUT, INPUT], [TARGET, { support: null, completeness: null, constraints: null }])).toThrow();
    const metadata = model.inputMetadata(INPUT);
    expect(metadata.source_ids).toEqual(['s1']);
    expect(metadata.input_truncated).toBe(false);
    expect(metadata.input_token_count).toBe(model.receipt(INPUT).input_token_count);
    expect(() => model.loss(withCandidate('yes '.repeat(200)), TARGET)).toThrow(/truncat/);
  });

  for (const inputFormat of ['json', 'paired'] as const) {
    for (const modelType of ['bert', 'electra']) {
      it(`owns imported native weights and tokenizer (${modelType}, ${inputFormat})`, async () => {
        const configured = new ResponseQualityAssessor(tinyConfig(modelType));
        const nativeConfig = NativeConfig.fromDict((configured.config.foundation_config as JsonObject));
        const native = createNativeModel(nativeConfig, 'sequence-classification');
        const directory = join(scratch(), 'foundation');
        mkdirSync(directory, { recursive: true });
        writeFileSync(join(directory, 'config.json'), JSON.stringify(nativeConfig.toDiffDict()));
        writeFileSync(join(directory, 'model.safetensors'), serializeModel(native));
        writeFileSync(join(directory, 'tokenizer.json'), TOKENIZER_JSON);
        writeFileSync(join(directory, 'tokenizer_config.json'), JSON.stringify({
          tokenizer_class: 'PreTrainedTokenizerFast', pad_token: '[PAD]', unk_token: '[UNK]', cls_token: '[CLS]', sep_token: '[SEP]',
        }));
        const loaded = await ResponseQualityAssessor.fromFoundation(directory, { maxTokens: 128, localFilesOnly: true, inputFormat });
        expect(loaded.config.input_format).toBe(inputFormat);
        const original = (native as unknown as { base: { stateDict(): Map<string, Tensor> } }).base.stateDict();
        for (const [key, value] of loaded.encoder.stateDict()) expect(value.equal(original.get(key)!)).toBe(true);
        const receipt = loaded.receipt(INPUT);
        expect((receipt.model as JsonObject).response_quality_heads_pretrained).toBe(false);
        expect((receipt.model as JsonObject).initialization).toBe('pretrained_encoder_only');
        await loaded.savePretrained(join(directory, '..', 'owned'));
        const restored = await ResponseQualityAssessor.fromPretrained(join(directory, '..', 'owned'));
        expect(restored.receipt(INPUT)).toEqual(receipt);
      });
    }
  }

  it('requires consistent coverage for acceptance', () => {
    const receipt = new ResponseQualityAssessor(tinyConfig()).receipt(INPUT);
    const { input_token_count: _count, ...missing } = receipt;
    void _count;
    for (const malformed of [{ ...receipt, input_token_count: 1000 }, { ...receipt, max_tokens: 0 }, missing]) {
      expect(() => ResponseQualityAssessor.accepts(malformed, 0)).toThrow();
    }
  });

  it('enforces the tokenizer contract and rejects nonfinite receipts', () => {
    for (const options of [{ json: 'override' }, { padding_side: 'left' }, { truncation_side: 'left' }]) {
      expect(() => new ResponseQualityAssessor({ ...tinyConfig(), tokenizer_options: options })).toThrow();
    }
    const model = new ResponseQualityAssessor(tinyConfig());
    noGrad(() => model.calibration('support').temperature.fill_(Number.NaN));
    expect(() => model.receipt(INPUT)).toThrow(/finite/);
  });

  for (const maxTokens of [3, 16, 128]) {
    for (const length of [10, 200]) {
      it(`paired encoding keeps native segments and full coverage (max ${maxTokens}, ${length} words)`, () => {
        const model = new ResponseQualityAssessor(tinyConfig('bert', maxTokens, 'paired'));
        const inputs = withCandidate('yes '.repeat(length));
        const first = `Question: question\nCandidate: ${inputs.candidate}`;
        const second = JSON.stringify(inputs.evidence).replace(/":"/g, '": "').replace(/","/g, '", "');
        const captured = model.encoderInputs(inputs);
        expect(Object.keys(captured).sort()).toEqual(['attention_mask', 'input_ids', 'token_type_ids']);
        const ids = captured.input_ids.toArray();
        expect(ids[0]).toBe(model.tokenizer.clsTokenId);
        expect(ids.filter((id) => id === model.tokenizer.sepTokenId).length).toBe(2);
        expect(captured.token_type_ids!.toArray()).toContain(1);
        expect(ids.length).toBeLessThanOrEqual(maxTokens);
        const full = model.tokenizer.encode(first, { textPair: second }).inputIds[0]!;
        const metadata = model.inputMetadata(inputs);
        expect(metadata.input_token_count).toBe(full.length);
        expect(metadata.input_truncated).toBe(full.length > maxTokens);
        if (metadata.input_truncated) expect(() => model.loss(inputs, TARGET)).toThrow(/truncat/);
        noGrad(() => model.call(inputs));
      });
    }
  }

  it('validates input_format and paired special-token capacity', () => {
    for (const invalid of ['other', null, 2, [], {}]) {
      expect(() => new ResponseQualityAssessor({ ...tinyConfig(), input_format: invalid as never })).toThrow(/input_format/);
    }
    expect(() => new ResponseQualityAssessor(tinyConfig('bert', 2, 'paired'))).toThrow(/special tokens/);
  });

  it('paired inputs reject review labels inside evidence', () => {
    const model = new ResponseQualityAssessor(tinyConfig('bert', 128, 'paired'));
    for (const field of ['targets', 'rationale', 'reference_answer']) {
      const inputs = { ...INPUT, evidence: [{ ...INPUT.evidence[0]!, [field]: 'secret' }] };
      expect(() => model.inputMetadata(inputs)).toThrow(/source_id\/text/);
      expect(() => model.call(inputs)).toThrow(/source_id\/text/);
    }
  });

  it('json format keeps the serialized-input tokenization and receipts', () => {
    const original = new ResponseQualityAssessor(tinyConfig()).eval();
    const explicit = new ResponseQualityAssessor(tinyConfig('bert', 128, 'json')).eval();
    explicit.loadStateDict(original.stateDict());
    const text = '{"question": "question", "evidence": [{"source_id": "s1", "text": "evidence yes"}], "candidate": "yes"}';
    const expected = original.tokenizer.encodeTensors([text], { padding: true, truncation: true, maxLength: 128 });
    const captured = original.encoderInputs(INPUT);
    expect(Object.keys(captured).sort()).toEqual(Object.keys(expected).sort());
    expect(captured.input_ids.equal(expected.input_ids)).toBe(true);
    expect(original.inputMetadata(INPUT).input_token_count).toBe(original.tokenizer.encode(text).inputIds[0]!.length);
    expect(original.receipt(INPUT)).toEqual(explicit.receipt(INPUT));
    expect('input_format' in original.config).toBe(false);
  });

  for (const modelType of ['bert', 'electra']) {
    it(`loads the Python-saved ${modelType} artifact with matching logits and receipts`, async () => {
      const record = interop.models[modelType];
      const model = await ResponseQualityAssessor.fromPretrained(join(fixtures, record.directory));
      const logits = noGrad(() => model.call(interop.inputs));
      expectClose(logits.data, record.logits.data, 1e-5, 1e-5);
      const batch = noGrad(() => model.call([interop.inputs, withCandidate('question yes')]));
      expectClose(batch.data, record.batch_logits.data, 1e-5, 1e-5);
      const receipt = model.receipt(interop.inputs);
      expect(receipt.calibrated).toEqual(record.receipt.calibrated);
      expect(receipt.calibration_sample_count).toEqual(record.receipt.calibration_sample_count);
      for (const axis of AXES) {
        expect(receipt.scores[axis]).toBeCloseTo(record.receipt.scores[axis], 5);
        expect(receipt.logits[axis]).toBeCloseTo(record.receipt.logits[axis], 5);
      }
      const { scores: _s, logits: _l, ...rest } = receipt;
      const { scores: _ps, logits: _pl, ...expectedRest } = record.receipt;
      void _s; void _l; void _ps; void _pl;
      expect(rest).toEqual(expectedRest);
      expect(model.inputMetadata(withCandidate('yes '.repeat(40)))).toEqual(record.metadata);
      expect(model.loss(interop.inputs, interop.targets).item()).toBeCloseTo(record.loss, 5);
    });
  }
});
