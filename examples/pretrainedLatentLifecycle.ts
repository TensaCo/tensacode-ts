/**
 * Bounded, explicitly supervised text-latent lifecycle on a pretrained model
 * (Python ``examples/pretrained_latent_lifecycle.py``, text path).
 *
 *     npm run build
 *     node examples/pretrainedLatentLifecycle.ts --output /tmp/latent-run
 *
 * Downloads the pinned `google/flan-t5-small` revision unless it is cached (or
 * pass `--local-files-only`). The four authored pairs demonstrate adapter
 * optimization, not held-out ability. Native token embeddings and final
 * encoder states are different Spaces: only the former support a native
 * identity decoder; the latter require a trained bridge. The optional
 * ViT/diffusion image path of the Python example needs `--image-input`.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { AdamW, manualSeed, noGrad, tensor, type Tensor } from 'tensorcode/nn';
import { Space, TextDecoder, TextEncoder, latentCodecs, type Latent } from 'tensorcode/ops/vec';
import { Trainer, loadExperience } from 'tensorcode/training';

const REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab';

interface Seq2Seq {
  config: { get(key: string): unknown };
  encode(inputs: { inputIds?: Tensor | null; inputsEmbeds?: Tensor | null; attentionMask?: Tensor | null }): Tensor;
  forward(inputs: { encoderHiddenStates: Tensor; attentionMask: Tensor | null; decoderInputIds: Tensor }): { logits: Tensor };
}

/**
 * Greedy ``model.generate`` for the native generation settings (FLAN-T5:
 * greedy, no processors): rows start with ``decoder_start_token_id``, stop at
 * EOS and are padded afterwards, as transformers returns them.
 */
function greedyGenerate(model: Seq2Seq, inputs: { inputIds?: Tensor; inputsEmbeds?: Tensor; attentionMask: Tensor }, maxNewTokens: number): number[][] {
  return noGrad(() => {
    const encoderHiddenStates = model.encode(inputs);
    const start = Number(model.config.get('decoder_start_token_id'));
    const eos = Number(model.config.get('eos_token_id'));
    const pad = Number(model.config.get('pad_token_id'));
    const batch = encoderHiddenStates.shape[0]!;
    const rows = Array.from({ length: batch }, () => [start]);
    const finished = new Array<boolean>(batch).fill(false);
    for (let step = 0; step < maxNewTokens && !finished.every(Boolean); step += 1) {
      const decoderInputIds = tensor(rows, { dtype: 'int64' });
      const { logits } = model.forward({ encoderHiddenStates, attentionMask: inputs.attentionMask, decoderInputIds });
      const next = logits.select(1, logits.shape[1]! - 1).argmax(-1).toArray();
      rows.forEach((row, index) => {
        const token = finished[index] ? pad : Number(next[index]);
        row.push(token);
        if (token === eos) finished[index] = true;
      });
    }
    return rows;
  });
}

function generatedLengths(rows: number[][], eos: number): number[] {
  return rows.map((row) => (row.includes(eos) ? row.indexOf(eos) : row.length - 1));
}

const { values } = parseArgs({
  options: {
    foundation: { type: 'string', default: 'google/flan-t5-small' }, revision: { type: 'string', default: REVISION },
    output: { type: 'string' }, device: { type: 'string', default: 'cpu' }, steps: { type: 'string', default: '4' },
    'local-files-only': { type: 'boolean', default: false }, 'image-input': { type: 'string' },
  },
});
const steps = Number(values.steps);
if (!values.output) throw new Error('--output is required');
if (!(steps >= 1)) throw new Error('--steps must be positive');
if (values['image-input']) {
  throw new Error('--image-input: the ViT/diffusion image path decodes the image file; run it with the Python example or supply decoded pixels through ops.vec.ImageEncoder.preprocess');
}
manualSeed(17);
const output = values.output;
mkdirSync(output, { recursive: false });
const hub = { revision: values.revision, localFilesOnly: values['local-files-only'] };
const nativeSpace = new Space(`${values.foundation}:encoder:input_embeddings`, 512, { version: values.revision, organization: 'sequence' });
const identity = (await TextDecoder.fromFoundation(values.foundation!, {
  ...hub, inputSpace: nativeSpace, bridge: 'identity', generation: { max_new_tokens: 24 },
})).eval();
const prompts = ['Translate to German: The house is wonderful.', 'What is the capital of France?', 'What is 2 plus 2?',
  'Answer briefly: What color is a ripe banana?'];
const targets = ['Das Haus ist wunderbar.', 'Paris', '4', 'yellow'];
const tokens = identity.tokenizer.encodeTensors(prompts, { padding: true });
const inputIds = tokens.input_ids!;
const attentionMask = tokens.attention_mask!;
const embedded = noGrad(() => identity.embedText(prompts));
const wrapped = noGrad(() => identity.call(embedded)) as string[];
const model = identity.model as unknown as Seq2Seq;
const ids = greedyGenerate(model, { inputIds, attentionMask }, 24);
const native = identity.tokenizer.batchDecode(ids, { skipSpecialTokens: true });
const encoder = (await TextEncoder.fromFoundation(values.foundation!, hub)).eval();
const latent = noGrad(() => encoder.call(prompts)) as Latent;
const nativeEncoder = (encoder.model as unknown as { getEncoder?(): { forward(i: unknown): { lastHiddenState: Tensor } } });
const hidden = noGrad(() => (nativeEncoder.getEncoder ? nativeEncoder.getEncoder() : encoder.model as unknown as { forward(i: unknown): { lastHiddenState: Tensor } })
  .forward({ inputIds, attentionMask }).lastHiddenState);
let rejected = false;
try {
  identity.call(latent);
} catch (error) {
  rejected = (error as Error).name === 'ValueError';
}
const decoder = await TextDecoder.fromFoundation(values.foundation!, {
  ...hub, inputSpace: latent.space, bridge: 'linear', generation: { max_new_tokens: 24 },
});
decoder.model.requiresGrad_(false);
const optimizer = new AdamW(decoder.projection.parameters(), { lr: 0.001 });
const trainer = Trainer.fromTool(decoder, { optimizer });
// Disable foundation dropout for this fixed-data adapter diagnostic.
decoder.model.eval();
const before = noGrad(() => decoder.loss(latent, targets).item());
const experience = trainer.capture(latent, targets, { source: 'four authored demonstration pairs; no held-out split' });
await experience.save(join(output, 'experience.json'), { operations: trainer.operations, codecs: latentCodecs() });
const replay = await loadExperience(join(output, 'experience.json'), { operations: trainer.operations, codecs: latentCodecs() });
const losses = Array.from({ length: steps }, () => trainer.step(replay));
decoder.eval();
const after = noGrad(() => decoder.loss(latent, targets).item());
const prediction = noGrad(() => decoder.call(latent)) as string[];
// Evaluator-only native token inspection records whether generation hit its cap.
const adapterIds = greedyGenerate(decoder.model as unknown as Seq2Seq, {
  inputsEmbeds: noGrad(() => decoder.projection.forward(latent.tensor)), attentionMask: latent.mask!,
}, 24);
await decoder.savePretrained(join(output, 'decoder'));
await trainer.saveCheckpoint(join(output, 'training'), { progress: { steps, data: 'authored fixed pairs' } });
const restored = await TextDecoder.fromPretrained(join(output, 'decoder'));
restored.model.requiresGrad_(false);
const resumed = Trainer.fromTool(restored, { optimizer: new AdamW(restored.projection.parameters(), { lr: 0.001 }) });
const progress = await resumed.loadCheckpoint(join(output, 'training'));
restored.eval();
const eos = Number(identity.model.config.get('eos_token_id'));
const maxAbs = noGrad(() => latent.tensor.sub(hidden).abs().max().item());
const inputLengths = Array.from(noGrad(() => attentionMask.sum(1)).toArray(), Number);
const report = {
  foundation: values.foundation, revision: values.revision, device: values.device, prompts, authored_targets: targets,
  native_predictions: native, identity_predictions: wrapped, identity_native_equal: JSON.stringify(wrapped) === JSON.stringify(native),
  encoder_native_max_abs_error: maxAbs, encoder_final_hidden_identity_rejected: rejected, input_tokens: inputLengths,
  truncation: {
    input: false, target: false, generation_max_new_tokens: 24,
    native_generation_missing_eos: ids.map((row) => !row.includes(eos)),
    adapter_generation_missing_eos: adapterIds.map((row) => !row.includes(eos)),
    native_generated_tokens: generatedLengths(ids, eos), adapter_generated_tokens: generatedLengths(adapterIds, eos),
  },
  adapter: {
    steps, before_cross_entropy: before, step_losses: losses, after_cross_entropy: after, predictions: prediction,
    reload_equal: JSON.stringify(prediction) === JSON.stringify(noGrad(() => restored.call(latent))), resumed_steps: resumed.steps,
    resumed_progress: progress, trainable_parameters: decoder.projection.parameters().reduce((total, parameter) => total + parameter.numel, 0),
  },
  limitations: ['Native behavior is inherited from supplied FLAN-T5 weights.',
    'Only a linear bridge learned from four authored pairs; no held-out generalization claim.',
    'Final encoder states are not native decoder input embeddings.'],
};
writeFileSync(join(output, 'results.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
