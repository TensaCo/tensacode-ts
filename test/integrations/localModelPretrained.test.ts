/**
 * ``LocalModel.fromPretrained`` with the real optional peer
 * ``@huggingface/transformers`` (a devDependency here) and a cached
 * SmolVLM-256M-Instruct ONNX export. Expected answers come from Python's
 * ``LocalModel.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct',
 * max_new_tokens=24).complete(...)`` with PyTorch float32 weights on the same
 * prompts and ``test/fixtures/local/red_square.png``.
 *
 * Runs when the Transformers.js cache holds the model (download once with
 * ``localFilesOnly: false``) and skips otherwise.
 */
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { LocalModel } from '../../src/integrations/index.js';
import { Classify, ImagePart, Message, ModelRequest, TextPart, Transform } from '../../src/ops/text/index.js';

const MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct';
const require = createRequire(import.meta.url);
const packageDir = (() => {
  try {
    return dirname(dirname(require.resolve('@huggingface/transformers')));
  } catch {
    return null;
  }
})();
const cached = packageDir !== null && existsSync(join(packageDir, '.cache', MODEL, 'onnx', 'decoder_model_merged.onnx'));
const image = readFileSync(new URL('../fixtures/local/red_square.png', import.meta.url));

// Python (PyTorch float32): {"text": ["City", 2], "image": ["Red.", 3]}
const textRequest = () => new ModelRequest([new Message('user', 'What is the capital of France? Answer in one word.')]);
const imageRequest = () => new ModelRequest([new Message('user', [
  new ImagePart({ data: new Uint8Array(image), mediaType: 'image/png' }), new TextPart('What color is the square in the middle? Answer in one word.'),
])]);

describe.skipIf(!cached)('LocalModel.fromPretrained with Transformers.js', () => {
  it('generates what Python generates, asynchronously and synchronously', async () => {
    const model = await LocalModel.fromPretrained(MODEL, { maxNewTokens: 24, dtype: 'fp32' });
    expect(model.configuration()).toEqual({ model_id: MODEL, revision: null, max_new_tokens: 24 });
    const text = await model.acomplete(textRequest());
    expect([text.text, text.providerMetadata!.generated_tokens]).toEqual(['City', 2]);
    const described = await model.acomplete(imageRequest());
    expect([described.text, described.providerMetadata!.generated_tokens]).toEqual(['Red.', 3]);
    expect(described.providerMetadata).toMatchObject({ backend: 'transformers.js', source: 'supplied_pretrained_model', finish_reason: 'stop' });
    // The blocking path loads its own copy in the worker thread and answers identically.
    expect(model.complete(textRequest()).text).toBe('City');
    expect(model.complete(imageRequest()).text).toBe('Red.');
    expect(Transform.fromModel(model).call([new Message('user', 'What is the capital of France? Answer in one word.')]).at(-1)!.content).toBe('City');
  }, 600_000);

  it('reports structured-output failures as Python does', async () => {
    const model = await LocalModel.fromPretrained(MODEL, { maxNewTokens: 4, dtype: 'fp32' });
    const classify = Classify.fromModel(model, { labels: ['red', 'blue'] });
    await expect(classify.acall([new Message('user', 'Which?')])).rejects.toThrow(/Local model/);
  }, 600_000);
});
