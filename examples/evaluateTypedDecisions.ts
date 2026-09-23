/**
 * Compare generated-JSON and likelihood decoding for owned typed decisions
 * (Python ``examples/evaluate_typed_decisions.py``).
 *
 *     npm run build
 *     node examples/evaluateTypedDecisions.ts --foundation google/flan-t5-small \
 *       --banking77 banking_data/test.csv --per-class 5 --output typed-decisions.json
 *     node examples/evaluateTypedDecisions.ts --foundation google/flan-t5-small \
 *       --candidates response-quality-candidates.jsonl --labels labels-a.jsonl labels-b.jsonl
 *
 * Zero-shot diagnostic of the *decoding mechanism* on a supplied foundation:
 * no training, threshold selection or prompt search occurs. Inputs are
 * Banking77 test rows (the first `--per-class` per intent, raw intent labels)
 * and/or response-quality candidates with reviewed labels, asked as three
 * true/false questions (support, completeness, constraints). The latter are
 * development data, not a benchmark.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';
import { parseArgs } from 'node:util';
import * as text from 'tensorcode/ops/text';

const AXES: Record<string, string> = {
  support: 'Do the evidence passages support every factual assertion in the candidate answer?',
  completeness: 'Does the candidate answer supply the kind of value the question asks for?',
  constraints: 'Does the candidate answer satisfy every restriction stated in the question?',
};
const MODES: Record<string, Record<string, string>> = {
  generate: {},
  likelihood_sum: { decoding: 'likelihood' },
  likelihood_mean: { decoding: 'likelihood', likelihood_normalization: 'mean' },
};

function expectedCalibrationError(pairs: [number, boolean][], bins = 10): number | null {
  if (!pairs.length) return null;
  const grouped = new Map<number, [number, boolean][]>();
  for (const [confidence, correct] of pairs) {
    const bin = Math.min(Math.floor(confidence * bins), bins - 1);
    grouped.set(bin, [...(grouped.get(bin) ?? []), [confidence, correct]]);
  }
  let total = 0;
  for (const rows of grouped.values()) {
    const confidence = rows.reduce((sum, [value]) => sum + value, 0) / rows.length;
    const accuracy = rows.reduce((sum, [, hit]) => sum + Number(hit), 0) / rows.length;
    total += (rows.length / pairs.length) * Math.abs(confidence - accuracy);
  }
  return total;
}

function auroc(scored: [number, boolean][]): number | null {
  const positives = scored.filter(([, label]) => label).map(([score]) => score);
  const negatives = scored.filter(([, label]) => !label).map(([score]) => score);
  if (!positives.length || !negatives.length) return null;
  let wins = 0;
  for (const p of positives) for (const n of negatives) wins += Number(p > n) + 0.5 * Number(p === n);
  return wins / (positives.length * negatives.length);
}

function build(foundation: string, config: Record<string, unknown>): Promise<text.Classify> {
  return text.Classify.fromFoundation(foundation, { config: config as never }) as Promise<text.Classify>;
}

/** ``csv.DictReader`` over RFC 4180 CSV. */
function readCsv(path: string): Record<string, string>[] {
  const source = readFileSync(path, 'utf8');
  const records: string[][] = [];
  let field = '';
  let record: string[] = [];
  let quoted = false;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index]!;
    if (quoted) {
      if (char === '"' && source[index + 1] === '"') { field += '"'; index += 1; } else if (char === '"') quoted = false; else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { record.push(field); field = ''; }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && source[index + 1] === '\n') index += 1;
      record.push(field); records.push(record); field = ''; record = [];
    } else field += char;
  }
  if (field || record.length) { record.push(field); records.push(record); }
  const [header, ...rows] = records.filter((row) => row.length > 1 || row[0] !== '');
  return rows.map((row) => Object.fromEntries(header!.map((key, index) => [key, row[index] ?? ''])));
}

const invalid = (error: unknown) => error instanceof text.InvalidModelOutput || (error as Error)?.name === 'ValueError';

async function banking77(path: string, perClass: number, foundation: string) {
  const rows: Record<string, string>[] = [];
  const counts = new Map<string, number>();
  for (const row of readCsv(path.startsWith('~/') ? `${process.env.HOME}${path.slice(1)}` : path)) {
    if ((counts.get(row.category!) ?? 0) < perClass) {
      counts.set(row.category!, (counts.get(row.category!) ?? 0) + 1);
      rows.push(row);
    }
  }
  const labels = [...counts.keys()].sort();
  const report: Record<string, unknown> = { rows: rows.length, labels: labels.length, majority_accuracy: Math.max(...counts.values()) / rows.length };
  for (const [mode, decoding] of Object.entries(MODES)) {
    const op = await build(foundation, {
      labels, instructions: 'Which banking customer intent does this message express?', generation: { max_new_tokens: 64 }, ...decoding,
    });
    let correct = 0;
    let valid = 0;
    const calibration: [number, boolean][] = [];
    const started = performance.now();
    for (const row of rows) {
      let result: text.ClassificationResult;
      try {
        result = op.call([new text.Message('user', row.text!)]);
      } catch (error) {
        if (invalid(error)) continue;
        throw error;
      }
      valid += 1;
      const hit = result.label === row.category;
      correct += Number(hit);
      if (result.confidence !== null) calibration.push([result.confidence, hit]);
    }
    report[mode] = {
      valid_rate: valid / rows.length, accuracy: correct / rows.length, ece: expectedCalibrationError(calibration),
      seconds_per_row: (performance.now() - started) / 1000 / rows.length,
    };
    console.log('banking77', mode, JSON.stringify(report[mode]));
  }
  return report;
}

async function responseQuality(candidatesPath: string, labelPaths: string[], foundation: string) {
  const labels: Record<string, Record<string, boolean | null>> = {};
  for (const path of labelPaths) {
    for (const line of readFileSync(path, 'utf8').split('\n').filter((item) => item.trim())) {
      const row = JSON.parse(line);
      labels[row.id] = row.targets;
    }
  }
  const candidates = readFileSync(candidatesPath, 'utf8').split('\n').filter((line) => line.trim()).map((line) => JSON.parse(line));
  const report: Record<string, unknown> = { candidates: candidates.length, label_authorship: 'assistant-reviewed development labels' };
  for (const [mode, decoding] of Object.entries(MODES)) {
    const questions: Record<string, text.Classify> = {};
    for (const [axis, instruction] of Object.entries(AXES)) {
      questions[axis] = await build(foundation, { labels: ['true', 'false'], instructions: instruction, generation: { max_new_tokens: 64 }, ...decoding });
    }
    const outcomes = Object.fromEntries(Object.keys(AXES).map((axis) => [axis, { valid: 0, correct: 0, labelled: 0, scored: [] as [number, boolean][] }]));
    const started = performance.now();
    for (const row of candidates) {
      const evidence = (row.evidence as { source_id: string; text: string }[]).map((item) => `[${item.source_id}] ${item.text}`).join('\n');
      const message = new text.Message('user', `Question: ${row.question}\nCandidate answer: ${row.candidate}\nEvidence:\n${evidence}`);
      // Each axis is scored separately so one invalid response does not discard
      // the others; text.ask would raise for the whole set.
      for (const [axis, operation] of Object.entries(questions)) {
        let answer: text.ClassificationResult | null;
        try {
          answer = operation.call([message]);
        } catch (error) {
          if (!invalid(error)) throw error;
          answer = null;
        }
        const target = labels[row.id]?.[axis];
        if (target === null || target === undefined) continue;
        const stats = outcomes[axis]!;
        stats.labelled += 1;
        if (answer === null) continue;
        stats.valid += 1;
        stats.correct += Number((answer.label === 'true') === target);
        if (answer.distribution !== null) stats.scored.push([answer.distribution.true!, target]);
      }
    }
    const summary: Record<string, unknown> = {};
    for (const [axis, stats] of Object.entries(outcomes)) {
      const positives = stats.scored.filter(([, label]) => label).length;
      summary[axis] = {
        labelled: stats.labelled, valid_rate: stats.valid / stats.labelled, accuracy: stats.correct / stats.labelled,
        auroc_true: auroc(stats.scored), scored_rows: stats.scored.length,
        scored_positive_rate: stats.scored.length ? positives / stats.scored.length : null,
      };
    }
    summary.seconds_per_candidate = (performance.now() - started) / 1000 / candidates.length;
    report[mode] = summary;
    console.log('response_quality', mode, JSON.stringify(summary));
  }
  const majority: Record<string, number> = {};
  for (const axis of Object.keys(AXES)) {
    const values = Object.values(labels).map((targets) => targets[axis]).filter((value): value is boolean => value !== null && value !== undefined);
    const positives = values.filter(Boolean).length;
    majority[axis] = Math.max(positives, values.length - positives) / values.length;
  }
  report.majority_accuracy = majority;
  return report;
}

const { values, positionals } = parseArgs({
  options: {
    foundation: { type: 'string' }, device: { type: 'string', default: 'cpu' }, banking77: { type: 'string' },
    'per-class': { type: 'string', default: '5' }, candidates: { type: 'string' }, labels: { type: 'string', multiple: true, default: [] },
    output: { type: 'string' },
  },
  allowPositionals: true,
});
if (!values.foundation) throw new Error('--foundation is required');
const report: Record<string, unknown> = {
  foundation: existsSync(values.foundation) ? basename(values.foundation) : values.foundation, training: 'none (zero-shot)', modes: MODES,
};
if (values.banking77) report.banking77 = await banking77(values.banking77, Number(values['per-class']), values.foundation);
// Python's `--labels a b c` (nargs='*'): extra positional paths are label files too.
if (values.candidates) report.response_quality = await responseQuality(values.candidates, [...values.labels!, ...positionals], values.foundation);
if (values.output) writeFileSync(values.output, `${JSON.stringify(report, null, 2)}\n`);
