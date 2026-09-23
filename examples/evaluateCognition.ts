/**
 * Evaluate an owned cognitive Chatbot on explicit evidence, with honest
 * controls (Python ``examples/evaluate_cognition.py``).
 *
 *     npm run build
 *     node examples/evaluateCognition.ts --model ./cognitive-chatbot --cases hotpot-cases.jsonl --output report.json
 *     node examples/evaluateCognition.ts --model ./assembled --ranker ./investigator --language-repo google/flan-t5-base \
 *       --language-revision REV --verifier-directory ./verifier --cases cases.jsonl --output report.json
 *
 * Real HotpotQA rows use oracle supporting passages. Answer coverage and
 * response fidelity are lexical diagnostics, not semantic correctness
 * judgments. Authored contradictions are reported separately from public-data
 * measurements. `--ranker` assembles a complete cognitive model from its
 * components first (every component is owned and serialized; the fitted
 * verifier calibration is retained).
 */
import { createHash } from 'node:crypto';
import { existsSync, appendFileSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { isDeepStrictEqual, parseArgs } from 'node:util';
import { manualSeed, noGrad, readSafetensorsFile } from 'tensorcode/nn';
import { Chatbot, Investigator } from 'tensorcode/tools';
import { Evidence } from 'tensorcode/tools/cognition';

type Json = Record<string, any>;
type Case = { id: string; question: string; target: string; evidence: Json[]; source_kind?: string };

/** SQuAD normalization with Python ``str.casefold`` and ``string.punctuation``. */
function normalize(value: string): string {
  const stripped = value.toLowerCase().replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, '');
  return stripped.replace(/\b(a|an|the)\b/g, ' ').split(/\s+/).filter(Boolean).join(' ');
}

function sha256(bytes: Uint8Array | string): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function loadCases(path: string): Case[] {
  const records = readFileSync(path, 'utf8').split('\n').filter((line) => line.trim()).map((line) => JSON.parse(line) as Case);
  const ids = records.map((record) => record.id);
  if (!records.length || new Set(ids).size !== ids.length) throw new Error('evaluation requires unique nonempty records');
  return records;
}

function summarize(records: Json[]): Json {
  const n = records.length;
  if (!n) return { count: 0 };
  const mean = (key: string) => records.reduce((total, record) => total + Number(Boolean(record[key])), 0) / n;
  const nli = records.filter((record) => record.realization_nli?.length);
  return {
    count: n, failed_calls: records.filter((record) => 'error' in record).length,
    short_answer_exact_match_format_sensitive: mean('answer_exact_match'),
    answer_wholeword_containment_not_accuracy: mean('answer_wholeword_containment'),
    candidate_answer_substring_coverage: mean('candidate_answer_substring_coverage'),
    cognitive_abstention_rate: mean('abstained'), final_enforced_abstention_rate: mean('abstention_enforced'),
    default_abstention_text_rate: mean('default_abstention_text'), selected_text_preserved_rate: mean('selected_text_preserved'),
    selected_contradiction_veto_violations: records.filter((record) => record.contradicted_selection).length,
    generation_truncations: records.filter((record) => record.input_truncated).length,
    realization_nli_support_mean_on_selected: nli.length ? nli.reduce((total, record) => total + record.realization_nli[0].distribution.support, 0) / nli.length : null,
    uncalibrated_verifier_records: records.filter((record) => record.verifier_calibrated === false).length,
  };
}

function measure(value: Case, answer: string, receipt: Json): Json {
  const cognition = receipt.cognition;
  const candidates = cognition.candidates as Json[];
  const selectedId = cognition.selected_id;
  const selected = candidates.find((candidate) => candidate.id === selectedId) ?? null;
  // The full selected declaration must appear in the realization; abstentions do
  // not count as faithful realizations. This intentionally undercounts paraphrases.
  const target = normalize(value.target);
  const verifications = candidates.flatMap((candidate) => candidate.verifications ?? []);
  return {
    id: value.id, question: value.question, target: value.target, answer, answer_exact_match: normalize(answer) === target,
    answer_wholeword_containment: Boolean(target) && ` ${normalize(answer)} `.includes(` ${target} `),
    candidate_answer_substring_coverage: Boolean(target) && candidates.some((candidate) => ` ${normalize(candidate.text)} `.includes(` ${target} `)),
    selected_text_preserved: selected !== null && normalize(answer).includes(normalize(selected.text)),
    abstained: cognition.abstained, abstention_enforced: receipt.abstention_enforced ?? cognition.abstained,
    default_abstention_text: answer === 'I do not have enough supported evidence to answer.',
    contradicted_selection: selected !== null && selected.verifications.some((verification: Json) => verification.distribution.contradiction > cognition.policy.max_contradiction),
    input_truncated: (receipt.input_truncated ?? false) || candidates.some((candidate) => candidate.input_truncated ?? false),
    verifier_calibrated: verifications.length ? verifications.every((verification: Json) => verification.calibrated ?? false) : null,
    receipt,
  };
}

function evaluate(bot: Chatbot, cases: Case[], options: { progressPath?: string; controlCount?: number } = {}) {
  const controlCount = options.controlCount ?? cases.length;
  if (!Number.isInteger(controlCount) || controlCount < 0 || controlCount > cases.length) throw new Error('control_count must be between zero and the case count');
  const records: Json[] = [];
  const controls: Json[] = [];
  const call = (session: ReturnType<Chatbot['newSession']>, value: Json, item: Case): Json => {
    try {
      const answer = session.call(value);
      const receipt = session.lastResult as Json;
      const result = measure(item, answer, receipt);
      const cognition = receipt.cognition;
      const selected = (cognition.candidates as Json[]).find((candidate) => candidate.id === cognition.selected_id);
      if (selected) result.realization_nli = bot.investigator!.verifier!.verify(answer, [{ source_id: selected.id, text: selected.text }]);
      return result;
    } catch (error) {
      const name = (error as Error)?.name;
      if (name !== 'ValueError' && !(error instanceof Error && error.constructor === Error)) throw error;
      return {
        id: item.id, question: item.question, target: item.target, answer: null,
        error: { type: name === 'ValueError' ? 'ValueError' : 'RuntimeError', message: (error as Error).message },
        answer_exact_match: false, candidate_answer_substring_coverage: false, selected_text_preserved: false, abstained: false,
        contradicted_selection: false, input_truncated: false,
      };
    }
  };
  cases.forEach((item, index) => {
    const session = bot.newSession();
    records.push(call(session, { question: item.question, evidence: item.evidence }, item));
    if (index < controlCount) {
      // Fresh-session source omission isolates evidence availability without
      // claiming to implement historical source deletion.
      const omission = call(bot.newSession(), { question: item.question }, item);
      const replacement = { text: 'This source is unavailable and supplies no evidence about the question.' };
      const revised = item.evidence.map((entry) => ({ ...replacement, evidence_id: entry.id }));
      const replacementResult = call(session, { question: item.question, revisions: revised }, item);
      controls.push({
        id: item.id, omission, replacement: replacementResult, replacement_kind: 'authored source-withdrawal notice, not real-world evidence',
        state_revision_before: records[records.length - 1]!.receipt?.cognition?.state_revision ?? null,
        state_revision_after: replacementResult.receipt?.cognition?.state_revision ?? null,
      });
    }
    if (options.progressPath) {
      appendFileSync(options.progressPath, `${JSON.stringify({ record: records[records.length - 1], controls: index < controlCount ? controls[controls.length - 1] : null })}\n`);
    }
    const last = records[records.length - 1]!;
    console.log(JSON.stringify({ id: item.id, answer: last.answer, abstained: last.abstained, error: last.error ?? null }));
  });
  return {
    real_data: { metrics: summarize(records), records },
    controls: { records: controls, omission: summarize(controls.map((row) => row.omission)), replacement: summarize(controls.map((row) => row.replacement)) },
  };
}

/** Retrieval over oracle passages from these diagnostic questions only. */
function evaluateMemory(bot: Chatbot, cases: Case[]) {
  const corpus = cases.flatMap((item) => item.evidence.map((entry) => [item, entry, `${item.id}:${entry.id}`] as const));
  const memory = bot.investigator!.newCognitiveSession({ memory: { capacity: Math.max(1, corpus.length) }, maxRecords: Math.max(1, corpus.length) });
  for (const [item, entry, identity] of corpus) {
    memory.ingest([new Evidence(identity, entry.text, entry.source_id)]);
    memory.remember(identity, { episodeId: item.id });
  }
  const tokenSet = (value: string) => new Set(value.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? []);
  const records = cases.map((item) => {
    const relevant = new Set(item.evidence.map((entry) => `${item.id}:${entry.id}`));
    const learned = memory.retrieve(item.question, { k: 5 }).map((hit) => hit.evidence.id);
    const tokens = tokenSet(item.question);
    const lexical = [...corpus].map((row, position) => [row, position] as const).sort((a, b) => {
      const overlap = (row: typeof corpus[number]) => [...tokenSet(row[1].text)].filter((token) => tokens.has(token)).length;
      return overlap(b[0]) - overlap(a[0]) || a[1] - b[1];
    }).slice(0, 5).map(([row]) => row[2]);
    return {
      id: item.id, relevant_ids: [...relevant].sort(), learned_ids: learned, lexical_ids: lexical,
      learned_hit_at_1: Boolean(learned.length && relevant.has(learned[0]!)), learned_hit_at_5: learned.some((id) => relevant.has(id)),
      lexical_hit_at_1: Boolean(lexical.length && relevant.has(lexical[0]!)), lexical_hit_at_5: lexical.some((id) => relevant.has(id)),
    };
  });
  const metrics = Object.fromEntries(['learned_hit_at_1', 'learned_hit_at_5', 'lexical_hit_at_1', 'lexical_hit_at_5']
    .map((key) => [key, records.reduce((total, record) => total + Number((record as Json)[key]), 0) / records.length]));
  return {
    metrics, records, corpus_size: corpus.length,
    limitations: 'Oracle supporting-passage corpus built from these diagnostic questions, not open-corpus retrieval. Ranker encoder was trained on a separate HotpotQA subset; no fitting here.',
  };
}

/** Actual repository documentation is runtime evidence, never a core seed. */
async function repositorySmoke(bot: Chatbot, document: string, sessionPath: string) {
  const fullText = readFileSync(document, 'utf8');
  const paragraphs = fullText.split('\n\n');
  if (paragraphs.length < 2) throw new Error('documentation smoke expects a heading and first paragraph');
  const excerpt = paragraphs[1]!;
  const question = 'Which service is the preferred model host for TensorCode?';
  const session = bot.newSession();
  const first = session.call({ question, evidence: [{ id: 'developer-guide', source_id: document, text: excerpt }] });
  const firstReceipt = session.lastResult;
  session.newEpisode();
  const recalled = session.call({ question });
  const recalledReceipt = session.lastResult;
  await session.save(sessionPath);
  const restored = await bot.newSession().load(sessionPath);
  const snapshotEqual = isDeepStrictEqual(restored.cognition!.snapshot(), session.cognition!.snapshot());
  restored.newEpisode();
  const reloaded = restored.call({ question });
  return {
    kind: 'real repository document smoke, separate from the statistical benchmark', document, sha256: sha256(fullText),
    excerpt, question, first: { text: first, receipt: firstReceipt }, new_episode: { text: recalled, receipt: recalledReceipt },
    save_load_snapshot_exact: snapshotEqual, reloaded_new_episode: { text: reloaded, receipt: restored.lastResult },
    limitations: 'One actual document; source retrieval and persistence do not establish answer correctness.',
  };
}

function authoredCases(): Case[] {
  return [{
    id: 'authored-conflict', question: 'Is the door open?', target: 'insufficient evidence',
    evidence: [{ id: 'report-1', source_id: 'authored-observer-A', text: 'The door is open.' },
      { id: 'report-2', source_id: 'authored-observer-B', text: 'The door is closed and is not open.' }],
    source_kind: 'authored_mechanism_fixture',
  }];
}

/** Own and serialize every component, retaining fitted verifier calibration. */
async function assemble(rankerPath: string, languageRepo: string, languageRevision: string | null, verifierDirectory: string, output: string,
  options: { generatorPath?: string | null; retrievalPath?: string | null } = {}): Promise<Chatbot> {
  manualSeed(17);
  const ranker = await Investigator.fromPretrained(rankerPath);
  const language = existsSync(join(languageRepo, 'tensorcode_config.json'))
    ? await Chatbot.fromPretrained(languageRepo)
    : await Chatbot.fromFoundation(languageRepo, { revision: languageRevision, localFilesOnly: true, options: { max_input_tokens: 1024, max_new_tokens: 96 } });
  const generator = options.generatorPath ? await Chatbot.fromPretrained(options.generatorPath) : language;
  const verifierConfig = JSON.parse(readFileSync(join(verifierDirectory, 'verifier_config.json'), 'utf8'));
  const config: Json = { ...ranker.configuration(), generator: generator.configuration(), ...verifierConfig };
  let retrieval: { configuration(): unknown; stateDict(): Map<string, unknown> } | null = null;
  if (options.retrievalPath) {
    // Python imports ``tensorcode._internal.retrieval`` for this step as well.
    const { RetrievalEncoder } = await import('../dist/_internal/retrieval.js');
    retrieval = await RetrievalEncoder.fromFoundation(options.retrievalPath, {
      pooling: 'masked_mean', normalize: true, revision: '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', maxTokens: 256, localFilesOnly: true,
    });
    config.retrieval_encoder = retrieval.configuration();
  }
  const investigator = new Investigator(config);
  investigator.rank.loadStateDict(ranker.rank.stateDict());
  investigator.generator!.loadStateDict(generator.stateDict());
  if (retrieval !== null) investigator.episodicEncoder!.loadStateDict(retrieval.stateDict() as never);
  investigator.verifier!.loadStateDict((await readSafetensorsFile(join(verifierDirectory, 'verifier.safetensors'))).tensors);
  const cognitive = {
    ...language.configuration(),
    cognition: {
      investigator: investigator.configuration(), proposal_count: 3, memory: { capacity: 256, top_k: 5 }, max_records: 1024,
      policy: { min_support: 0.7, max_contradiction: 0.2, max_unknown: 0.3 },
    },
  };
  const model = new Chatbot(cognitive);
  const { missingKeys, unexpectedKeys } = model.loadStateDict(language.stateDict(), { strict: false });
  if (unexpectedKeys.length || missingKeys.some((key) => !key.startsWith('investigator.'))) throw new Error('language component transfer mismatch');
  model.investigator!.loadStateDict(investigator.stateDict());
  await model.savePretrained(output);
  const weights: [string, string][] = [['ranker', join(rankerPath, 'model.safetensors')], ['language', join(languageRepo, 'model.safetensors')],
    ['generator', join(options.generatorPath ?? languageRepo, 'model.safetensors')], ['verifier', join(verifierDirectory, 'verifier.safetensors')]];
  const provenance = {
    initialization_seed: 17, ranker: rankerPath, language: languageRepo, language_revision: languageRevision,
    generator: options.generatorPath ?? languageRepo, retrieval: options.retrievalPath ?? null, verifier_directory: verifierDirectory,
    verifier_calibrated: model.investigator!.verifier!.calibration.isCalibrated,
    policy_origin: 'authored fixed thresholds; no evaluation tuning',
    memory_policy: { capacity: 256, top_k: 5, max_records: 1024, origin: 'authored retention/retrieval policy' },
    component_weight_sha256: Object.fromEntries(weights.filter(([, path]) => existsSync(path)).map(([name, path]) => [name, sha256(readFileSync(path))])),
  };
  writeFileSync(join(output, 'assembly-provenance.json'), `${JSON.stringify(provenance, null, 2)}\n`);
  return model;
}

const { values } = parseArgs({
  options: {
    model: { type: 'string' }, cases: { type: 'string' }, output: { type: 'string' }, device: { type: 'string', default: 'cpu' },
    'control-count': { type: 'string' }, 'repo-smoke': { type: 'string' }, role: { type: 'string', default: 'diagnostic' },
    ranker: { type: 'string' }, 'language-repo': { type: 'string', default: 'google/flan-t5-base' }, 'language-revision': { type: 'string' },
    'generator-model': { type: 'string' }, 'retrieval-foundation': { type: 'string' }, 'verifier-directory': { type: 'string' },
  },
});
if (!values.model || !values.cases || !values.output) throw new Error('--model, --cases and --output are required');
if (!['diagnostic', 'final'].includes(values.role!)) throw new Error('--role must be diagnostic or final');
let bot: Chatbot;
if (values.ranker) {
  if (!values['verifier-directory'] || (!values['language-revision'] && !(existsSync(values['language-repo']!) && statSync(values['language-repo']!).isDirectory()))) {
    throw new Error('assembly requires verifier directory and pinned language revision');
  }
  bot = await assemble(values.ranker, values['language-repo']!, values['language-revision'] ?? null, values['verifier-directory'], values.model, {
    generatorPath: values['generator-model'] ?? null, retrievalPath: values['retrieval-foundation'] ?? null,
  });
} else {
  bot = await Chatbot.fromPretrained(values.model);
}
bot.eval();
const cases = loadCases(values.cases);
mkdirSync(dirname(values.output), { recursive: true });
const progress = values.output.replace(/\.[^./]*$/, '') + '.progress.jsonl';
writeFileSync(progress, '');
const controlCount = values['control-count'] === undefined ? undefined : Number(values['control-count']);
const report: Json = noGrad(() => evaluate(bot, cases, { progressPath: progress, ...(controlCount === undefined ? {} : { controlCount }) }));
const fixtures: Json = noGrad(() => evaluate(bot, authoredCases()));
fixtures.conflict_cases_with_selected_candidate = fixtures.real_data.records.filter((row: Json) => !row.abstained && !('error' in row)).length;
fixtures.expected_behavior = 'Authored mutually conflicting sources should prevent an unqualified selection; this is a fixture, not a public benchmark.';
report.authored_fixtures = fixtures;
report.episodic_retrieval = noGrad(() => evaluateMemory(bot, cases));
if (values['repo-smoke']) report.repository_smoke = await repositorySmoke(bot, values['repo-smoke'], values.output.replace(/\.[^./]*$/, '') + '.session.json');
const manifest = join(dirname(values.cases), 'data-manifest.json');
if (existsSync(manifest)) report.data_manifest = JSON.parse(readFileSync(manifest, 'utf8'));
report.protocol = { primary_count: cases.length, control_count: controlCount ?? cases.length, control_selection: 'first fixed cases in input order' };
report.model_fingerprint = bot.fingerprint;
report.foundation = (bot.configuration() as Json).foundation ?? null;
report.evaluation_role = values.role === 'diagnostic'
  ? 'diagnostic development: observed results may motivate later model changes; not a final untouched benchmark'
  : 'fixed-configuration final evaluation; do not tune on these results';
report.cases_sha256 = sha256(readFileSync(values.cases));
report.manual_factual_review = { status: 'pending', scope: 'Review every non-abstained primary response against the complete source passages and gold answer; containment is not correctness.' };
report.limitations = ['Exact short-answer EM is format-mismatched for declarative responses and must not be presented as factual accuracy.', 'Oracle supporting passages, not learned retrieval.',
  'Lexical answer coverage and fidelity diagnostics do not establish semantic correctness; realization NLI uses the same separately calibrated SNLI verifier and is an additional model judgment.',
  'NLI calibration is inherited from a separate SNLI calibration split, not evidence-QA calibration.',
  'Authored threshold policy screens model scores; acceptance is not proof of truth.',
  'Foundation generation workspace is unadapted unless separately documented.'];
writeFileSync(values.output, `${JSON.stringify(report, null, 2)}\n`);
