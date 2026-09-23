/**
 * Evaluate a supplied local vision-language model on one image (Python
 * ``examples/local_multimodal.py``; not a benchmark).
 *
 *     npm run build
 *     node examples/localMultimodal.ts --image candy.jpg --source "https://example.org/candy.jpg" --output report.json
 *
 * Install the optional peer `@huggingface/transformers` and make the model
 * available locally first (this script never downloads images or weights:
 * loading is `localFilesOnly`). Free-form answers and strict structured
 * operations run through the same model; structured failures are measured
 * outcomes, never replaced with made-up answers.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { parseArgs } from 'node:util';
import { trace } from 'tensorcode';
import { LocalModel } from 'tensorcode/integrations';
import * as text from 'tensorcode/ops/text';

const { values } = parseArgs({
  options: {
    image: { type: 'string' }, source: { type: 'string' }, model: { type: 'string', default: 'HuggingFaceTB/SmolVLM-256M-Instruct' },
    revision: { type: 'string', default: '7e3e67edbbed1bf9888184d9df282b700a323964' }, device: { type: 'string', default: 'cpu' },
    dtype: { type: 'string' }, output: { type: 'string' },
  },
});
if (!values.image || !values.source || !values.output) throw new Error('--image, --source and --output are required');
const raw = new Uint8Array(readFileSync(values.image));
// Transformers.js loads by model ID from its own cache; the revision is recorded, not re-resolved.
const model = await LocalModel.fromPretrained(values.model, { device: values.device, maxNewTokens: 160, ...(values.dtype ? { dtype: values.dtype } : {}) });
const image = new text.ImageEncoder({ media_type: 'image/jpeg', source_ref: values.source }).call(raw)[0]!.content[0] as text.ImagePart;
const records: Record<string, unknown>[] = [];
const seconds = (started: number) => (performance.now() - started) / 1000;
const rejected = (error: unknown) => error instanceof TypeError || (error as Error)?.name === 'ValueError' || (error as Error)?.name === 'InvalidModelOutput';
for (const question of ['Describe what the hand is holding.', 'How many candies are in the hand?', 'What animal is drawn on the candy?']) {
  const value = [new text.Message('user', [image, new text.TextPart(question)])];
  const started = performance.now();
  try {
    const session = trace();
    const response = session.run(() => text.Transform.fromModel(model).call(value));
    records.push({ task: question, answer: response[response.length - 1]!.content, seconds: seconds(started), trace_calls: session.calls.length });
  } catch (error) {
    if (!rejected(error)) throw error;
    records.push({ task: question, status: 'rejected', error: (error as Error).message, seconds: seconds(started) });
  }
}
// These deliberately exercise strict structured operations with a small model.
// Failure is a measured outcome, not replaced with a made-up answer.
const structured: [string, text.Classify | text.Retrieve][] = [
  ['classify', text.Classify.fromModel(model, {
    labels: ['food', 'vehicle'], instructions: 'Classify the pictured objects. Return JSON. Use null for unknown confidence and distribution.',
  })],
  ['retrieve', text.Retrieve.fromModel(model, {
    items: { food: 'Candy and other sweets', vehicle: 'Cars and trucks' },
    instructions: 'Find the item describing the pictured objects. Return JSON. Use null for scores.',
  })],
];
for (const [task, operation] of structured) {
  const started = performance.now();
  try {
    const result = operation.call([new text.Message('user', [image, new text.TextPart('Identify the best match.')])]) as { value?: unknown; keys?: readonly string[] };
    records.push({ task, value: result.value ?? null, keys: [...(result.keys ?? [])], status: 'valid', seconds: seconds(started) });
  } catch (error) {
    if (!rejected(error)) throw error;
    records.push({ task, status: 'rejected', error: (error as Error).message, seconds: seconds(started) });
  }
}
const report = {
  model_id: values.model, revision: values.revision, image_source: values.source,
  image_sha256: createHash('sha256').update(raw).digest('hex'), runtime: `node ${process.version}`, device: values.device, records,
  limitations: 'Single published image, not held-out benchmark. Semantics come from the supplied pretrained model. No training, calibrated confidence, or general visual understanding claim. Structured failures are retained.',
};
writeFileSync(values.output, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
