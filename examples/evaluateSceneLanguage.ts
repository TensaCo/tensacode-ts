/**
 * Evaluate owned scene interpretations with real, blank and shuffled image
 * evidence (Python ``examples/evaluate_scene_language.py``).
 *
 *     npm run build
 *     node examples/evaluateSceneLanguage.ts --data vsr.jsonl --model ./scene-language --report report.json
 *     node examples/evaluateSceneLanguage.ts --data vsr.jsonl --model ./scene-language \
 *       --foundation HuggingFaceTB/SmolVLM-256M-Instruct --revision 7e3e67edbbed1bf9888184d9df282b700a323964 --report report.json
 *
 * Input JSONL: `image_path`, `source_id`, `question` (a spatial caption),
 * `target` ('0'/'1'). VSR labels evaluate explicit yes/no judgments. Free
 * descriptions are preserved for review, not automatically scored as factual.
 * No training occurs. Images are decoded and thumbnailed exactly as Pillow's
 * `Image.open(...).convert('RGB').thumbnail((1024, 1024))`. Pure JavaScript
 * compute makes each SmolVLM interpretation take minutes.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { manualSeed, noGrad, zeros, type Tensor } from 'tensorcode/nn';
import { Scene } from 'tensorcode/tools';
// The TypeScript counterpart of ``PIL.Image.open`` (Pillow-exact decoding and resampling).
import { openImage, type RasterImage } from 'tensorcode/ops/vec';

type Row = { image_path: string; source_id: string; question: string; target: string | number };

/** Pillow ``Image.thumbnail(size)``: aspect-preserving BICUBIC resize with ``reducing_gap=2.0``. */
function thumbnail(image: RasterImage, [x0, y0]: [number, number]): RasterImage {
  let x = x0;
  let y = y0;
  if (x >= image.width && y >= image.height) return image;
  const aspect = image.width / image.height;
  const roundAspect = (value: number, key: (n: number) => number) => {
    const low = Math.floor(value);
    const high = Math.ceil(value);
    return Math.max(key(low) <= key(high) ? low : high, 1);
  };
  if (x / y >= aspect) x = roundAspect(y * aspect, (n) => Math.abs(aspect - n / y));
  else y = roundAspect(x / aspect, (n) => (n === 0 ? 0 : Math.abs(aspect - x / n)));
  if (x === image.width && y === image.height) return image;
  return image.resize([x, y], 3, null, 2.0);
}

function readRows(path: string, limit: number): [Row, Tensor][] {
  const result: [Row, Tensor][] = [];
  for (const line of readFileSync(path, 'utf8').split('\n').slice(0, limit)) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as Row;
    const location = isAbsolute(row.image_path) ? row.image_path : join(dirname(path), row.image_path);
    const image = thumbnail(openImage(location).convert('RGB'), [1024, 1024]);
    result.push([row, image.toTensor().to('float32').div(255)]);
  }
  if (result.length < 2) throw new Error('at least two images are required for shuffle evaluation');
  return result;
}

function fingerprint(pixels: Tensor): string {
  const bytes = new Uint8Array(Float32Array.from(pixels.toArray()).buffer);
  return createHash('sha256').update(bytes).update(`(${pixels.shape.join(', ')})`).digest('hex');
}

function evaluate(model: Scene, rows: [Row, Tensor][], maxNewTokens: number) {
  const fingerprints = rows.map(([, pixels]) => fingerprint(pixels));
  if (new Set(fingerprints).size < 2) throw new Error('shuffle evaluation requires distinct image contents');
  const report: Record<string, any> = {
    judgments: [], descriptions: [], metrics: {},
    limitations: ['Descriptions are unverified model proposals, not extracted facts.', 'Foundation pretraining overlap with these images is unknown.',
      'Blank/shuffle judgments are compared with original image labels; this is label retention, not altered-image ground-truth accuracy.',
      'An inactive workspace residual preserves foundation behavior and establishes no learned TensorCode workspace benefit.'],
  };
  for (const mode of ['full', 'blank', 'shuffle']) {
    let correct = 0;
    let parsed = 0;
    rows.forEach(([row, original], index) => {
      let pixels = original;
      let sourceId = row.source_id;
      if (mode === 'blank') {
        pixels = zeros(original.shape);
        sourceId = `blank:${row.source_id}`;
      } else if (mode === 'shuffle') {
        let replacement = index;
        for (let offset = 1; offset < rows.length; offset += 1) {
          const candidate = (index + offset) % rows.length;
          if (fingerprints[candidate] !== fingerprints[index]) { replacement = candidate; break; }
        }
        [{ source_id: sourceId }, pixels] = [rows[replacement]![0], rows[replacement]![1]];
      }
      const question = `Does the image support this description? Answer yes or no, then explain the spatial relationship: ${row.question}`;
      const receipt = model.interpret({ pixels, source_id: sourceId, question }, { maxNewTokens });
      const match = /^\s*(yes|no)\b/i.exec(receipt.interpretation as string);
      const prediction = match === null ? null : String(Number(match[1]!.toLowerCase() === 'yes'));
      parsed += Number(prediction !== null);
      correct += Number(prediction === String(row.target));
      report.judgments.push({ mode, original_source_id: row.source_id, caption: row.question, target: row.target, prediction, receipt });
      if (index < 4) {
        const description = model.interpret({
          pixels, source_id: sourceId,
          question: 'Describe the overall scene, spatial arrangement, and interactions. Mention uncertainty where details are unclear.',
        }, { maxNewTokens });
        report.descriptions.push({ mode, original_source_id: row.source_id, receipt: description });
      }
    });
    report.metrics[mode] = { count: rows.length, [mode === 'full' ? 'accuracy' : 'original_label_agreement']: correct / rows.length, parsed_fraction: parsed / rows.length };
  }
  return report;
}

const { values } = parseArgs({
  options: {
    data: { type: 'string' }, model: { type: 'string' }, foundation: { type: 'string' }, revision: { type: 'string' },
    device: { type: 'string', default: 'cpu' }, limit: { type: 'string', default: '32' }, 'max-new-tokens': { type: 'string', default: '64' },
    report: { type: 'string' },
  },
});
if (!values.data || !values.model || !values.report) throw new Error('--data, --model and --report are required');
manualSeed(23);
if (values.foundation) {
  if (!values.revision) throw new Error('--foundation requires --revision');
  const imported = await Scene.fromLanguageFoundation(values.foundation, { revision: values.revision });
  await imported.savePretrained(values.model);
}
const model = await Scene.fromPretrained(values.model);
const rows = readRows(values.data, Number(values.limit));
const report = noGrad(() => evaluate(model, rows, Number(values['max-new-tokens'])));
report.model_config = model.configuration();
report.data = resolve(values.data);
writeFileSync(values.report, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report.metrics, null, 2));
