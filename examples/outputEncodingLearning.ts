/**
 * Collect a text → OUTPUT_ENCODING → text program, train it, and reload its
 * operations (Python ``examples/output_encoding_learning.py``).
 *
 *     npm run build
 *     node examples/outputEncodingLearning.ts --foundation google/flan-t5-small \
 *       --revision 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab --data reviewed.jsonl --output /tmp/readout-run
 *
 * Supply reviewed JSONL rows `{"text": "input text", "target": "reviewed output"}`.
 * The foundation must be cached (loading is `localFilesOnly`). The loss is
 * measured on the supplied training examples, not held-out generalization;
 * native foundation weights stay frozen. The example saves complete operation
 * weights, not optimizer or random-generator continuation.
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { trace, type Trace } from 'tensorcode';
import { AdamW, manualSeed, noGrad, type Tensor } from 'tensorcode/nn';
import { TextDecoder, TextEncoder, latentCodecs, type Latent } from 'tensorcode/ops/vec';
import { Trainer, loadExperience } from 'tensorcode/training';

const { values } = parseArgs({
  options: {
    foundation: { type: 'string' }, revision: { type: 'string' }, data: { type: 'string' }, output: { type: 'string' },
    device: { type: 'string', default: 'cpu' }, steps: { type: 'string', default: '50' },
    'batch-size': { type: 'string', default: '4' }, lr: { type: 'string', default: '0.001' },
  },
});
if (!values.foundation || !values.revision || !values.data || !values.output) {
  throw new Error('--foundation, --revision, --data and --output are required');
}
const steps = Number(values.steps);
const batchSize = Number(values['batch-size']);
const lr = Number(values.lr);
if (!(steps >= 1) || !(batchSize >= 1) || !(lr > 0 && lr < 1)) throw new Error('positive steps/batch_size and lr in (0,1) required');

type Row = { text: string; target: string };
const raw = readFileSync(values.data);
const rows = raw.toString('utf8').split('\n').filter((line) => line.trim()).map((line) => JSON.parse(line) as Row);
if (!rows.length || rows.some((row) => row === null || typeof row !== 'object' || Object.keys(row).sort().join() !== 'target,text'
  || [row.text, row.target].some((value) => typeof value !== 'string' || !value.trim()))) {
  throw new Error('supply nonempty JSONL rows containing text and target strings');
}
const output = values.output;
mkdirSync(output, { recursive: false });
manualSeed(17);

// All trainable parameters exist before capture. Import cached foundations explicitly.
const hub = { revision: values.revision, localFilesOnly: true };
const encode = (await TextEncoder.fromFoundation(values.foundation, { ...hub, readout: 'output_encoding' })).eval();
const decode = (await TextDecoder.fromFoundation(values.foundation, {
  ...hub, inputSpace: encode.outputSpace, bridge: 'linear', generation: { max_new_tokens: 32 },
})).eval();
encode.model.requiresGrad_(false);
decode.model.requiresGrad_(false);
if (rows.some((row) => encode.tokenizer.encode(row.text).inputIds[0]!.length > 255 || decode.tokenizer.encode(row.target).inputIds[0]!.length > 128)) {
  throw new Error('example budget is 255 input and 128 target tokens; split long examples explicitly');
}
const operations = { encode, objective: decode.trainingOperation };
const trainer = Trainer.fromOps(operations, {
  optimizer: (parameters) => new AdamW(parameters, { lr }), losses: { objective: (loss: Tensor) => loss },
});
const beforeReadout = encode.outputEncoding!.detach().clone();
const beforeBridge = (decode.projection as { weight: Tensor }).weight.detach().clone();
const digest = createHash('sha256').update(raw).digest('hex');
const source = `supplied JSONL supervision sha256:${digest}`;
const paths: string[] = [];
for (let start = 0; start < rows.length; start += batchSize) {
  const batch = rows.slice(start, start + batchSize);
  const experience = trace();
  const loss = experience.run(() => {
    const latent = encode.call(batch.map((row) => row.text));
    return decode.trainingOperation.call({ inputs: latent, targets: batch.map((row) => row.target) });
  });
  experience.supervise(loss, batch.map((row) => row.target), { loss: 'objective', source });
  const path = join(output, `experience-${String(paths.length).padStart(5, '0')}.json`);
  await experience.save(path, { operations, codecs: latentCodecs(), release: true });
  paths.push(path);
}
const losses: number[] = [];
for (let step = 0; step < steps; step += 1) {
  const experience: Trace = await loadExperience(paths[step % paths.length]!, { operations, codecs: latentCodecs() });
  losses.push(trainer.step(experience));
}
await encode.savePretrained(join(output, 'encoder'));
await decode.savePretrained(join(output, 'decoder'));
const sample = rows.slice(0, batchSize);
const expected = noGrad(() => decode.loss(encode.call(sample.map((row) => row.text)), sample.map((row) => row.target)));
const restoredEncode = await TextEncoder.fromPretrained(join(output, 'encoder'));
const restoredDecode = await TextDecoder.fromPretrained(join(output, 'decoder'));
const actual = noGrad(() => restoredDecode.loss(restoredEncode.call(sample.map((row) => row.text)) as Latent, sample.map((row) => row.target)));
const report = {
  examples: rows.length, steps, losses,
  readout_changed: !beforeReadout.equal(encode.outputEncoding!),
  bridge_changed: !beforeBridge.equal((decode.projection as { weight: Tensor }).weight),
  reloaded_loss_exact: expected.equal(actual), supervision_sha256: digest,
  foundation: values.foundation, revision: values.revision,
  limitations: 'Supplied training pairs only; frozen native foundations; no shared-semantic-space or generalization claim; weights-only reload.',
};
if (!report.reloaded_loss_exact) throw new Error('restored operation loss differs');
writeFileSync(join(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
