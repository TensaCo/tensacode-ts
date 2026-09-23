/**
 * Owned local seq2seq language realization from explicit memory tensors
 * (Python ``tensorcode/_internal/text/realization.py``).
 */
import type { Tensor } from '../../nn/tensor.js';
import { Operation, type Context } from '../../ops/base.js';
import { generateSeq2Seq, type GenerationSettings } from '../native/generation.js';
import { generationDefaults } from '../native/config.js';
import type { T5ForConditionalGeneration } from '../native/t5.js';
import type { JsonObject } from '../json.js';

export interface DecoderMemory {
  /** ``[batch, positions, d_model]`` decoder memory (encoder outputs). */
  conditioning: Tensor;
  /** ``[batch, positions]`` memory mask. */
  mask: Tensor;
  /** Teacher-forced target ids (``-100`` ignored); when present, returns the loss. */
  labels?: Tensor;
}

export interface DecoderLoss {
  loss: Tensor;
  logits: Tensor;
}

/**
 * Owned local seq2seq language realization from explicit memory tensors.
 *
 * The owner registers the model parameters; this operation keeps a
 * reference so a shared encoder/decoder model is saved exactly once. With
 * ``labels`` it returns the teacher-forced ``{loss, logits}``; otherwise it
 * generates token ids with the owner's generation configuration overlaid by
 * the call context (``GenerationConfig``-style snake_case settings).
 */
export class SequenceDecoder extends Operation<DecoderMemory, Tensor | DecoderLoss> {
  static override readonly qualifiedName: string = 'tensorcode._internal.text.realization.SequenceDecoder';
  private readonly model: T5ForConditionalGeneration;
  private readonly generationConfig: () => JsonObject;

  constructor(model: T5ForConditionalGeneration, generationConfig: () => JsonObject) {
    super();
    this.model = model;
    this.generationConfig = generationConfig;
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: DecoderMemory, context: Context | null): Tensor | DecoderLoss {
    const settings = { ...(context ?? {}) } as GenerationSettings;
    if (value.labels !== undefined && value.labels !== null) {
      const result = this.model.forward({ encoderHiddenStates: value.conditioning, attentionMask: value.mask, labels: value.labels });
      return { loss: result.loss!, logits: result.logits };
    }
    return generateSeq2Seq(this.model, { encoderHiddenStates: value.conditioning, attentionMask: value.mask }, settings, {
      generationConfig: this.generationConfig(),
    });
  }

  configuration(): JsonObject {
    return {
      operation: 'tensorcode.tools.chatbot.Chatbot.decoder', memory: 'explicit-sequence-v1',
      model: this.model.config.toDiffDict(),
      generation: generationDefaults(this.generationConfig()),
    };
  }
}
