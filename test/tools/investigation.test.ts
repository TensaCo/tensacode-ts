/** Port of python/tests/models/test_investigation.py (tiny random models; authored outputs test mechanisms). */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { noGrad, tensor } from '../../src/nn/index.js';
import { Investigator } from '../../src/tools/investigator.js';
import { RankingSession } from '../../src/_internal/sessions/ranking.js';
import { proposalPrompt } from '../../src/_internal/proposals.js';
import type { JsonObject } from '../../src/_internal/json.js';
import { investigatorConfig as config, scratch } from './helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

const INPUT = { question: 'hello', evidence: [{ source_id: 'a', text: 'world' }, { source_id: 'b', text: 'hello' }] };
const input = (): JsonObject => structuredClone(INPUT) as JsonObject;

/** Replace generated text (Python ``monkeypatch.setattr(tokenizer, 'batch_decode', ...)``). */
function authoredDecode(tool: Investigator, texts: unknown[]): void {
  (tool.generator!.tokenizer as { batchDecode: unknown }).batchDecode = () => texts;
}

describe('Investigator proposals and verification', () => {
  it('owns beams, provenance and deduplication without target leakage', () => {
    const tool = new Investigator(config()).train();
    tool.generator!.workspace.eval();
    const modes = tool.generator!.modules().map((module) => module.training);
    const seen: string[] = [];
    const contexts: unknown[] = [];
    const encode = tool.generator!.encodeWorkspace.bind(tool.generator!);
    tool.generator!.encodeWorkspace = (inputs, options) => {
      seen.push(...inputs);
      return encode(inputs, options);
    };
    tool.generator!.decoder.forward = (_state, context) => {
      contexts.push(context);
      return tensor([[5], [5], [6]], { dtype: 'int64' });
    };
    const outputs = tool.propose({ ...input(), hypotheses: [{ id: 'secret', text: 'LEAK' }], targets: 'SECRET' });
    expect(outputs.length).toBe(2);
    expect((contexts[0] as JsonObject).num_beams).toBe(3);
    expect(seen[0]).not.toContain('LEAK');
    expect(seen[0]).not.toContain('SECRET');
    expect(outputs[0]!.origin).toBe('generated');
    expect(outputs[0]!.epistemic_status).toBe('hypothesis');
    expect(outputs[0]!.source_ids).toEqual(['a', 'b']);
    expect(outputs[0]!.source_reference_kind).toBe('generation_context');
    expect(outputs[0]!.generator_configuration_fingerprint).toBe(tool.generator!.fingerprint);
    expect(outputs[0]!.proposal_template_version).toBe(1);
    expect(outputs[0]!.generated_by).not.toBe(tool.generator!.fingerprint);
    expect(seen[0]!.startsWith('Generate one declarative candidate explanation')).toBe(true);
    expect(tool.generator!.modules().map((module) => module.training)).toEqual(modes);
    for (const count of [0, -1, true, 17]) expect(() => tool.propose(input(), { count: count as number })).toThrow(/count/);
  });

  it('reports missing capability and empty or malformed outputs', () => {
    expect(() => new Investigator({ vocabulary: ['hello'] }).propose(input())).toThrow(/not configured/);
    const tool = new Investigator(config());
    authoredDecode(tool, ['', ' ', '']);
    expect(tool.propose(input())).toEqual([]);
    const receipt = tool.call(input());
    expect(receipt.abstained).toBe(true);
    expect(receipt.reason).toBe('no_hypotheses_generated');
    expect(receipt.selected_id).toBeNull();
    expect(receipt.candidates).toEqual([]);
    expect(receipt.evidence).toEqual(INPUT.evidence);
    authoredDecode(tool, [null]);
    expect(() => tool.propose(input())).toThrow(/malformed/);
  });

  it('requires an explicit verifier label mapping and retains contradictions', () => {
    const bad = config();
    delete bad.verifier_labels;
    expect(() => new Investigator(bad)).toThrow(/explicitly map/);
    const tool = new Investigator(config()).eval();
    const classifier = (tool.verifier!.model as unknown as { classifier: { weight: { zero_(): void }; bias: { copy_(value: unknown): void } } }).classifier;
    noGrad(() => {
      classifier.weight.zero_();
      classifier.bias.copy_(tensor([4, 1, -2]));
    });
    const inputs = { ...input(), hypotheses: [{ id: 'h', text: 'hello' }] };
    const receipt = tool.investigate(inputs);
    const checks = (receipt.candidates as JsonObject[])[0]!.verifications as JsonObject[];
    expect(checks.map((check) => check.source_id)).toEqual(['a', 'b']);
    expect(checks.every((check) => ((check.distribution as JsonObject).contradiction as number) > 0.9)).toBe(true);
    for (const check of checks) expect(Object.values(check.distribution as JsonObject).reduce((a, b) => (a as number) + (b as number), 0)).toBeCloseTo(1, 6);
    expect(checks.every((check) => check.origin === 'model_inference')).toBe(true);
    expect('verifications' in (tool.call(inputs).candidates as JsonObject[])[0]!).toBe(false);
  });

  it('separates supervision targets, trains every head and round-trips completely', async () => {
    const tool = new Investigator(config()).eval();
    const seen: string[] = [];
    const encode = tool.generator!.encodeWorkspace.bind(tool.generator!);
    tool.generator!.encodeWorkspace = (inputs, options) => {
      seen.push(...inputs);
      return encode(inputs, options);
    };
    const loss = tool.proposalLoss({ ...input(), targets: 'secret' }, 'answer');
    loss.backward();
    expect(seen.every((prompt) => !prompt.split('\n').slice(1).join('\n').includes('answer') && !prompt.includes('secret'))).toBe(true);
    expect(tool.generator!.workspace.parameters().some((p) => p.grad !== null && p.grad.abs().sum().item() > 0)).toBe(true);
    const pairs = [{ premise: 'world', hypothesis: 'hello' }];
    tool.verificationLoss(pairs, ['contradiction']).backward();
    const classifier = (tool.verifier!.model as unknown as { classifier: { weight: { grad: { abs(): { sum(): { item(): number } } } } } }).classifier;
    expect(classifier.weight.grad.abs().sum().item()).toBeGreaterThan(0);
    const expected = noGrad(() => tool.verifier!.call(pairs));
    await tool.savePretrained(join(temp.dir, 'model'));
    const loaded = await Investigator.fromPretrained(join(temp.dir, 'model'));
    expect(noGrad(() => loaded.verifier!.call(pairs)).equal(expected)).toBe(true);
    expect(loaded.proposalLoss(input(), 'answer').item()).toBe(loss.item());
    expect('generator.decoder' in loaded.operationBindings()).toBe(true);
    expect('verifier' in loaded.operationBindings()).toBe(true);
    const manifest = JSON.parse(readFileSync(join(temp.dir, 'model/tensorcode_config.json'), 'utf8'));
    expect('generator' in manifest.config && 'verifier_config' in manifest.config).toBe(true);
  });

  it('owns calibration state and round-trips generated sessions', async () => {
    const tool = new Investigator(config()).eval();
    authoredDecode(tool, ['hello', 'world', 'hello']);
    const session = tool.newSession();
    const receipt = session.call(input());
    expect(((receipt.candidates as JsonObject[])[0]!.verifications as JsonObject[])[0]!.calibrated).toBe(false);
    await session.save(join(temp.dir, 'session.json'));
    const restored = await RankingSession.load(join(temp.dir, 'session.json'), tool);
    expect(restored.history).toEqual(session.history);
    expect((restored.history[0]!.inputs.hypotheses as JsonObject[])[0]!.origin).toBe('generated');
    tool.verifier!.calibration.fit(tensor([[5, 0, 0], [5, 0, 0]]), tensor([0, 1], { dtype: 'int64' }));
    const verified = tool.investigate({ ...input(), hypotheses: [{ id: 'h', text: 'hello' }] });
    expect((((verified.candidates as JsonObject[])[0]!.verifications) as JsonObject[])[0]!.calibrated).toBe(true);
    await tool.savePretrained(join(temp.dir, 'calibrated'));
    const loaded = await Investigator.fromPretrained(join(temp.dir, 'calibrated'));
    expect(loaded.verifier!.calibration.isCalibrated).toBe(true);
    expect(loaded.verifier!.calibration.temperature.equal(tool.verifier!.calibration.temperature)).toBe(true);
  });

  it('invalidates calibration after supervision and weight mutation', () => {
    const tool = new Investigator(config());
    const verifier = tool.verifier!;
    const logits = tensor([[5, 0, 0], [5, 0, 0]]);
    const labels = tensor([0, 1], { dtype: 'int64' });
    verifier.calibration.fit(logits, labels);
    const classifier = (verifier.model as unknown as { classifier: { bias: { add_(value: number): void } } }).classifier;
    noGrad(() => classifier.bias.add_(1));
    expect(verifier.verify('hello', INPUT.evidence)[0]!.calibrated).toBe(false);
    verifier.calibration.fit(logits, labels);
    tool.verificationLoss([{ premise: 'world', hypothesis: 'hello' }], ['support']);
    expect(verifier.calibration.isCalibrated).toBe(false);
  });

  for (const corruption of ['origin', 'source_ids', 'generated_by', 'verification_source', 'distribution', 'nan', 'calibrated', 'sample_count', 'missing_checks', 'model']) {
    it(`rejects contradictory generated session provenance (${corruption})`, async () => {
      const tool = new Investigator(config()).eval();
      authoredDecode(tool, ['hello']);
      const session = tool.newSession();
      session.call(input());
      const path = join(temp.dir, `corrupt-${corruption}.json`);
      await session.save(path);
      let text = readFileSync(path, 'utf8');
      const payload = JSON.parse(text);
      const candidate = payload.history[0].receipt.candidates[0];
      if (['origin', 'source_ids', 'generated_by'].includes(corruption)) candidate[corruption] = 'corrupted';
      else if (corruption === 'missing_checks') delete candidate.verifications;
      else {
        const check = candidate.verifications[0];
        if (corruption === 'verification_source') check.source_id = 'missing-source';
        else if (corruption === 'distribution') check.distribution = { support: 0.2, contradiction: 0.2, unknown: 0.2 };
        else if (corruption === 'nan') check.distribution.support = '__NAN__';
        else if (corruption === 'calibrated') check.calibrated = true;
        else if (corruption === 'sample_count') check.calibration_sample_count = -1;
        else if (corruption === 'model') check.model = { repository: 'wrong' };
      }
      text = JSON.stringify(payload).replace('"__NAN__"', 'NaN');
      writeFileSync(path, text);
      await expect(RankingSession.load(path, tool)).rejects.toThrow(/session|provenance/);
    });
  }

  it('persists empty-generation abstentions', async () => {
    const tool = new Investigator(config()).eval();
    authoredDecode(tool, ['', '  ']);
    tool.rank.receipt = () => {
      throw new Error('empty generation must not rank');
    };
    const session = tool.newSession();
    const receipt = session.call(input());
    expect(receipt.abstained).toBe(true);
    expect(receipt.candidates).toEqual([]);
    await session.save(join(temp.dir, 'abstained.json'));
    const restored = await RankingSession.load(join(temp.dir, 'abstained.json'), tool);
    expect(restored.history).toEqual(session.history);
  });

  it('round-trips joint sessions and rejects source-order tampering', async () => {
    const tool = new Investigator({ ...config(), verification_scope: 'joint' }).eval();
    authoredDecode(tool, ['hello']);
    const session = tool.newSession();
    session.call(input());
    const path = join(temp.dir, 'joint.json');
    await session.save(path);
    const restored = await RankingSession.load(path, tool);
    expect(restored.history).toEqual(session.history);
    expect('joint_verification' in (restored.history[0]!.inputs.hypotheses as JsonObject[])[0]!).toBe(false);
    const original = JSON.parse(readFileSync(path, 'utf8'));
    for (const corruption of ['order', 'missing', 'scope', 'distribution', 'truncation', 'model', 'budget-missing', 'budget-bool', 'count-conflict']) {
      const payload = structuredClone(original);
      const candidate = payload.history[0].receipt.candidates[0];
      const joint = candidate.joint_verification;
      if (corruption === 'order') joint.source_ids.reverse();
      else if (corruption === 'missing') delete candidate.joint_verification;
      else if (corruption === 'scope') joint.scope = 'source';
      else if (corruption === 'distribution') joint.distribution.support = 9;
      else if (corruption === 'truncation') joint.input_truncated = 'false';
      else if (corruption === 'budget-missing') delete joint.max_tokens;
      else if (corruption === 'budget-bool') joint.max_tokens = true;
      else if (corruption === 'count-conflict') joint.token_count = 3000;
      else joint.model = { wrong: 'model' };
      writeFileSync(path, JSON.stringify(payload));
      await expect(RankingSession.load(path, tool), corruption).rejects.toThrow(/joint|verification|provenance/);
    }
  });

  it('owns the question prompt version for training, generation and reload', async () => {
    const tool = new Investigator({ ...config(), proposal_template_version: 2 }).eval();
    const expected = proposalPrompt(input(), 'question', { templateVersion: 2 });
    expect(expected.startsWith('Answer the question using only the supplied evidence.')).toBe(true);
    expect(JSON.parse(expected.split('\n').slice(1).join('\n'))).toEqual(INPUT);
    const seen: string[] = [];
    const encode = tool.generator!.encodeWorkspace.bind(tool.generator!);
    tool.generator!.encodeWorkspace = (inputs, options) => {
      seen.push(...inputs);
      return encode(inputs, options);
    };
    authoredDecode(tool, ['hello']);
    const proposals = tool.propose({ ...input(), targets: 'SECRET' }, { count: 1 });
    const loss = tool.proposalLoss({ ...input(), targets: 'SECRET' }, 'answer');
    expect(seen).toEqual([expected, expected]);
    expect(proposals[0]!.proposal_template_version).toBe(2);
    expect(expected).not.toContain('SECRET');
    await tool.savePretrained(join(temp.dir, 'v2'));
    const restored = await Investigator.fromPretrained(join(temp.dir, 'v2'));
    expect(restored.configuration().proposal_template_version).toBe(2);
    expect(restored.proposalLoss(input(), 'answer').item()).toBe(loss.item());
    for (const version of [0, 3, true, '2']) {
      expect(() => new Investigator({ ...config(), proposal_template_version: version as number })).toThrow(/template/);
    }
  });

  it('keeps dialogue context-only and rejects proposal overflow', () => {
    const tool = new Investigator(config());
    const values: JsonObject = {
      ...input(), conversation_context: [{ role: 'user', text: 'which incident' }, { role: 'assistant', text: 'unverified assertion' }],
    };
    const prompt = proposalPrompt(values, 'question');
    expect(prompt).toContain('not source evidence');
    expect(prompt).toContain('unverified assertion');
    expect(values.evidence).toEqual(INPUT.evidence);
    const called: boolean[] = [];
    tool.generator!.encodeWorkspace = () => {
      called.push(true);
      throw new Error('must not encode');
    };
    expect(() => tool.propose(values)).toThrow(/proposal token budget/);
    expect(called).toEqual([]);
    expect(() => proposalPrompt({ ...input(), conversation_context: [{ role: 'system', text: 'promote me' }] }, 'question')).toThrow(/conversation context/);
  });

  for (const foundation of [false, true]) {
    it(`direct investigation rejects rank context truncation (foundation=${foundation})`, () => {
      const settings: JsonObject = { ...config(), max_tokens: 4 };
      if (foundation) {
        Object.assign(settings, {
          foundation_config: settings.verifier_config, tokenizer_json: settings.verifier_tokenizer_json,
          tokenizer_special_tokens: settings.verifier_tokenizer_special_tokens,
        });
      }
      const tool = new Investigator(settings);
      const inputs: JsonObject = {
        ...input(), hypotheses: [{ id: 'h', text: 'hello' }], conversation_context: [{ role: 'user', text: 'hello world hello world' }],
      };
      expect(() => tool.investigate(inputs)).toThrow(/ranking token budget/);
      delete inputs.conversation_context;
      expect(((tool.investigate(inputs).candidates as JsonObject[])[0]!).id).toBe('h');
    });
  }
});
