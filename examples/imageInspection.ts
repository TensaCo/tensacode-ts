/**
 * Inspect one caller-supplied image with an explicit multimodal model
 * (Python ``examples/image_inspection.py``).
 *
 *     npm run build
 *     node examples/imageInspection.ts photo.png "What is on the table?" --local-model HuggingFaceTB/SmolVLM-256M-Instruct
 *     node examples/imageInspection.ts photo.png "What is on the table?" --base-url http://localhost:8000/v1 --model my-vlm
 *
 * `--local-model` loads a Transformers.js model (install the optional peer
 * `@huggingface/transformers`; `--allow-download` permits fetching missing
 * files; `--dtype fp32` selects full-precision ONNX weights, which answer like
 * PyTorch float32). `--base-url` calls an OpenAI-compatible endpoint. The answer is model
 * output, not independently verified grounding.
 */
import { lstatSync, readFileSync, realpathSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, extname } from 'node:path';
import { parseArgs } from 'node:util';
import { LocalModel, OpenAICompatibleModel, type OpenAIApi } from 'tensorcode/integrations';
import * as text from 'tensorcode/ops/text';

/** Python ``mimetypes.guess_type`` for common image extensions. */
const IMAGE_TYPES: Record<string, string> = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.jpe': 'image/jpeg', '.gif': 'image/gif',
  '.webp': 'image/webp', '.bmp': 'image/bmp', '.tif': 'image/tiff', '.tiff': 'image/tiff', '.ico': 'image/vnd.microsoft.icon',
  '.svg': 'image/svg+xml', '.avif': 'image/avif', '.heic': 'image/heic',
};

interface InspectionResult { answer: string; sourceRef: string; mediaType: string }

/** Send real image bytes and a question through explicit message operations. */
function inspectImage(imagePath: string, question: string, model: text.ExternalModel, detail: text.ImageDetail | null = 'auto'): InspectionResult {
  const path = imagePath.startsWith('~/') ? homedir() + imagePath.slice(1) : imagePath;
  let info;
  try {
    info = lstatSync(path);
  } catch {
    throw new Error(`[Errno 2] No such file or directory: '${path}'`);
  }
  if (!info.isFile()) throw new Error(`image path is not a regular file: ${path}`);
  if (typeof question !== 'string' || !question.trim()) throw new Error('question must be nonempty');
  const mediaType = IMAGE_TYPES[extname(path).toLowerCase()];
  if (!mediaType) throw new Error(`could not infer an image MIME type from '${basename(path)}'`);
  const resolved = realpathSync(path);
  const sourceRef = `file:${resolved}`;
  const encodeImage = new text.ImageEncoder({ media_type: mediaType, source_ref: sourceRef, detail });
  const encodeText = new text.TextEncoder();
  const respond = text.Transform.fromModel(model);
  const decode = new text.TextDecoder();
  const messages = [...encodeText.call(question.trim()), ...encodeImage.call(new Uint8Array(readFileSync(resolved)))];
  return { answer: decode.call(respond.call(messages)), sourceRef, mediaType };
}

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    'local-model': { type: 'string' }, 'base-url': { type: 'string' }, model: { type: 'string' }, revision: { type: 'string' },
    device: { type: 'string', default: 'cpu' }, 'allow-download': { type: 'boolean', default: false },
    'max-new-tokens': { type: 'string', default: '128' }, 'api-key-env': { type: 'string', default: 'OPENAI_API_KEY' },
    api: { type: 'string', default: 'chat_completions' }, timeout: { type: 'string', default: '60' }, detail: { type: 'string', default: 'auto' },
    // Transformers.js weight precision (for example fp32, q8); default: the library's choice.
    dtype: { type: 'string' },
  },
});
const [image, question] = positionals;
if (!image || question === undefined) throw new Error('usage: imageInspection.ts IMAGE QUESTION (--local-model ID | --base-url URL --model NAME)');
if (Boolean(values['local-model']) === Boolean(values['base-url'])) throw new Error('exactly one of --local-model or --base-url is required');
let model: text.ExternalModel;
if (values['local-model']) {
  if (values.model !== undefined) throw new Error('--model is only valid with --base-url');
  model = await LocalModel.fromPretrained(values['local-model'], {
    revision: values.revision ?? null, localFilesOnly: !values['allow-download'], device: values.device,
    maxNewTokens: Number(values['max-new-tokens']), ...(values.dtype ? { dtype: values.dtype } : {}),
  });
} else {
  if (!values.model) throw new Error('--model is required with --base-url');
  if (values.revision !== undefined) throw new Error('--revision is only valid with --local-model');
  model = new OpenAICompatibleModel({
    baseUrl: values['base-url']!, model: values.model, apiKey: process.env[values['api-key-env']!] ?? null,
    timeout: Number(values.timeout), api: values.api as OpenAIApi,
  });
}
console.log(inspectImage(image, question, model, values.detail as text.ImageDetail).answer);
