/**
 * Scene language mode (Python ``tensorcode.tools.scene.SceneLanguage``): an
 * owned Idefics3 (SmolVLM) foundation that interprets a full image, with a
 * trainable workspace residual on its visual tokens. Interpretations are
 * explicitly unverified; they are not extracted facts or scene graphs.
 */
import { Module } from '../nn/module.js';
import { Parameter, Tensor, ones, tensor, zeros } from '../nn/tensor.js';
import { noGrad } from '../nn/autograd.js';
import { Linear, MultiheadAttention } from '../nn/layers.js';
import { cat } from '../nn/ops/shape.js';
import { tensorBytes } from '../nn/safetensors.js';
import { ValueError } from '../errors.js';
import { ModuleOperation, type Context } from '../ops/base.js';
import { Workspace, type WorkspaceOutput } from '../_internal/workspace.js';
import { TensorAdapter as Transform } from '../_internal/vec/adapter.js';
import { deepCopy, isPlainObject, sha256Hex, type JsonObject, type JsonValue } from '../_internal/json.js';
import { NativeConfig } from '../_internal/native/config.js';
import { Idefics3ForConditionalGeneration } from '../_internal/native/idefics3.js';
import { generationConfigFromDict } from '../_internal/native/causalGeneration.js';
import { Idefics3Processor, safeAssetName } from '../_internal/native/idefics3Processing.js';

/** Scene language-mode input: a CHW RGB image in ``[0, 1]`` and a question. */
export interface SceneLanguageInputs {
  question: string;
  source_id: string;
  pixels: Tensor;
}

/** An unverified full-image interpretation receipt. */
export interface SceneInterpretation extends JsonObject {
  interpretation: string;
  verification: 'unverified';
  uncertainty: { status: 'uncalibrated'; confidence: null };
  source: { source_id: string; kind: 'full-image'; shape: number[]; sha256: string };
  question: string;
  completion_status: 'complete' | 'token_limit';
  workspace: { active: boolean; visual_tokens: number; attention: number[][]; relations: number[][] };
  foundation_source: JsonValue;
}

/** Language-mode defaults (Python ``Scene.__init__``). */
export const LANGUAGE_DEFAULTS: readonly (readonly [string, number])[] = [
  ['workspace_dimensions', 64], ['workspace_slots', 8], ['workspace_steps', 2], ['max_image_size', 4096],
  ['max_question_chars', 4096], ['max_input_tokens', 4096], ['max_target_chars', 4096], ['max_new_tokens', 256],
];

/** Python ``len(text)`` (code points). */
function pythonLength(text: string): number {
  let count = 0;
  for (const _ of text) count += 1;
  return count;
}

/** ``(x * 255).round().to(torch.uint8)`` in float32 with round-half-to-even. */
export function pixelsToUint8(pixels: Tensor): Tensor {
  const source = pixels.detach().to('float32').data;
  const out = new Uint8Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    const value = Math.fround(Math.min(Math.max(source[index]!, 0), 1) * 255);
    const floor = Math.floor(value);
    const fraction = value - floor;
    out[index] = fraction > 0.5 || (fraction === 0.5 && floor % 2 === 1) ? floor + 1 : floor;
  }
  return tensor(out, { shape: [...pixels.shape], dtype: 'uint8' });
}

/** Python ``str((tuple(shape), dtype))`` for the pixel fingerprint. */
function shapeDtypeRepr(pixels: Tensor): string {
  const shape = pixels.shape.length === 1 ? `(${pixels.shape[0]},)` : `(${pixels.shape.join(', ')})`;
  return `(${shape}, torch.${pixels.dtype})`;
}

export interface PreparedBatch {
  inputIds: Tensor;
  attentionMask: Tensor;
  imageHiddenStates: Tensor;
}

/**
 * Owned Idefics3 perception/realization with a trainable visual residual. The
 * residual gate initializes to zero, so importing a foundation preserves its
 * behavior; a fresh workspace is not a learned improvement over it.
 */
export class SceneLanguage extends Module {
  static override readonly qualifiedName: string = 'tensorcode.tools.scene.SceneLanguage';
  readonly config: JsonObject;
  readonly assets: Record<string, string>;
  readonly processor: Idefics3Processor;
  readonly gate: Parameter;
  readonly model: Idefics3ForConditionalGeneration;
  readonly down: Transform<Linear>;
  readonly workspace: Workspace;
  readonly read: MultiheadAttention;
  readonly up: Transform<Linear>;
  generationConfig: JsonObject;

  constructor(config: JsonObject, assets: unknown) {
    super();
    const hashes = config.processor_hashes;
    if (!isPlainObject(hashes)) throw new ValueError('language construction requires processor_hashes');
    if (!isPlainObject(assets)) throw new ValueError('language construction requires complete processor assets');
    const expected = Object.keys(hashes).sort();
    const supplied = Object.keys(assets).sort();
    if (expected.length !== supplied.length || expected.some((name, index) => name !== supplied[index])) {
      throw new ValueError('language construction requires complete processor assets');
    }
    for (const [name, value] of Object.entries(assets)) {
      if (!safeAssetName(name) || typeof value !== 'string') throw new ValueError('invalid processor asset');
      if (sha256Hex(value) !== hashes[name]) throw new ValueError('processor asset checksum mismatch');
    }
    this.assets = { ...(assets as Record<string, string>) };
    this.processor = Idefics3Processor.fromAssets(this.assets);
    this.config = deepCopy(config);
    // Parameters of this module precede its submodules (PyTorch registration order).
    this.gate = this.registerParameter('gate', new Parameter(zeros([])));
    const nativeConfig = NativeConfig.fromDict(config.language_config);
    this.model = this.registerModule('model', new Idefics3ForConditionalGeneration(nativeConfig));
    if (!isPlainObject(config.generation_config)) throw new ValueError('language construction requires generation_config');
    // ``GenerationConfig.from_dict`` validates the settings (raising transformers' errors).
    generationConfigFromDict(config.generation_config);
    this.generationConfig = deepCopy(config.generation_config as JsonObject);
    if (config.freeze_foundation === true) {
      this.model.requiresGrad_(false);
      this.model.eval();
    }
    const width = nativeConfig.sub('text_config').number('hidden_size');
    const dimensions = config.workspace_dimensions as number;
    this.down = this.registerModule('down', new Transform(new Linear(width, dimensions)));
    this.workspace = this.registerModule('workspace', new Workspace(dimensions, config.workspace_slots as number, config.workspace_steps as number));
    this.read = this.registerModule('read', new MultiheadAttention(dimensions, 1, { batchFirst: true }));
    this.up = this.registerModule('up', new Transform(new Linear(dimensions, width)));
  }

  override train(mode = true): this {
    super.train(mode);
    if (this.config.freeze_foundation === true) this.model?.eval();
    return this;
  }

  validate(value: unknown): Tensor {
    if (!isPlainObject(value)) throw new ValueError('scene inputs must be a dictionary');
    for (const key of ['question', 'source_id']) {
      if (typeof value[key] !== 'string' || !(value[key] as string).trim()) throw new ValueError(`${key} must be nonempty text`);
    }
    if (pythonLength(value.question as string) > (this.config.max_question_chars as number)) throw new ValueError('question exceeds configured limit');
    const pixels = value.pixels;
    if (!(pixels instanceof Tensor) || pixels.ndim !== 3 || pixels.shape[0] !== 3 || Math.min(pixels.shape[1]!, pixels.shape[2]!) < 1
      || Math.max(pixels.shape[1]!, pixels.shape[2]!) > (this.config.max_image_size as number)) {
      throw new ValueError('pixels must be nonempty RGB CHW within configured bounds');
    }
    if (!pixels.isFloatingPoint || !pixels.allFinite() || pixels.min().item() < 0 || pixels.max().item() > 1) {
      throw new ValueError('pixels must contain finite floating values in [0, 1]');
    }
    return pixels;
  }

  /** Processor batch with workspace-revised ``image_hidden_states`` (Python ``prepare``). */
  prepare(value: SceneLanguageInputs): { batch: PreparedBatch; workspace: WorkspaceOutput; visualTokens: number } {
    const pixels = this.validate(value);
    const image = pixelsToUint8(pixels);
    const messages: JsonValue[] = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: value.question }] }];
    const prompt = this.processor.applyChatTemplate(messages, { addGenerationPrompt: true });
    const processed = this.processor.call(prompt, [image]);
    if (processed.inputIds.shape[1]! > (this.config.max_input_tokens as number)) {
      throw new ValueError('processed image/question exceeds configured token limit');
    }
    const dtype = this.down.module.weight.dtype;
    const pixelValues = processed.pixelValues!.to(dtype);
    const pixelMask = processed.pixelAttentionMask ? processed.pixelAttentionMask.to(dtype) : null;
    // Foundation visual tokens retain their pretrained spatial organization.
    const visual = this.model.model.getImageFeatures(pixelValues, pixelMask);
    const originalShape = [...visual.shape];
    const width = originalShape[originalShape.length - 1]!;
    const visualSequence = visual.reshape(1, -1, width);
    const count = visualSequence.shape[1]!;
    const text = this.model.getInputEmbeddings().forward(processed.inputIds);
    const encoded = this.down.call(cat([visualSequence, text], 1)) as Tensor;
    const mask = cat([ones([1, count], { dtype: 'bool' }), processed.attentionMask.ne(0)], 1);
    const workspace = this.workspace.forward(encoded, mask);
    const [read] = this.read.forward(encoded.slice(1, 0, count), workspace.conditioning, workspace.conditioning, { needWeights: false });
    const revised = visualSequence.add(this.gate.tanh().mul(this.up.call(read) as Tensor));
    return {
      batch: { inputIds: processed.inputIds, attentionMask: processed.attentionMask, imageHiddenStates: revised.reshape(originalShape) },
      workspace,
      visualTokens: count,
    };
  }

  private eosIds(): number[] {
    const eos = this.generationConfig.eos_token_id;
    if (eos === null || eos === undefined) return [];
    return Array.isArray(eos) ? eos.map(Number) : [Number(eos)];
  }

  /** Teacher-forced loss of reviewer-supplied ``target`` text after the workspace interpretation. */
  loss(value: SceneLanguageInputs, target: unknown): Tensor {
    if (typeof target !== 'string' || !target.trim()) throw new ValueError('language target must be nonempty reviewer-supplied text');
    if (pythonLength(target) > (this.config.max_target_chars as number)) throw new ValueError('language target exceeds configured limit');
    const { batch } = this.prepare(value);
    // The target is appended only after workspace interpretation is complete.
    const ids = [...this.processor.tokenizer.encode(target, { addSpecialTokens: false }).inputIds[0]!];
    const eos = this.eosIds();
    if (eos.length) ids.push(eos[0]!);
    if (ids.length > (this.config.max_new_tokens as number)) throw new ValueError('language target exceeds configured token limit');
    const prefix = batch.inputIds.shape[1]!;
    if (prefix + ids.length > this.model.config.sub('text_config').number('max_position_embeddings')) {
      throw new ValueError('target and input exceed model context capacity');
    }
    const targetIds = tensor(ids, { shape: [1, ids.length], dtype: 'int64' });
    const inputIds = cat([batch.inputIds, targetIds], 1);
    const attentionMask = cat([batch.attentionMask, ones([1, ids.length], { dtype: batch.attentionMask.dtype })], 1);
    const labels = tensor([...new Array<number>(prefix).fill(-100), ...ids], { shape: [1, prefix + ids.length], dtype: 'int64' });
    return this.model.forward({ inputIds, attentionMask, imageHiddenStates: batch.imageHiddenStates, labels }).loss!;
  }

  /** Unverified full-image interpretation (transformers ``generate`` with ``do_sample=False``). */
  interpret(value: SceneLanguageInputs, options: { maxNewTokens?: number | null } = {}): SceneInterpretation {
    const limit = options.maxNewTokens ?? (this.config.max_new_tokens as number);
    if (typeof limit !== 'number' || !Number.isInteger(limit) || limit < 1 || limit > (this.config.max_new_tokens as number)) {
      throw new ValueError('max_new_tokens exceeds configured bounds');
    }
    return noGrad(() => {
      const { batch, workspace, visualTokens } = this.prepare(value);
      const prompt = batch.inputIds.shape[1]!;
      if (prompt + limit > this.model.config.sub('text_config').number('max_position_embeddings')) {
        throw new ValueError('generation and input exceed model context capacity');
      }
      // ``self.model.generate(**batch, max_new_tokens=limit, do_sample=False, return_dict_in_generate=False)``.
      const generated = this.model.generate(
        { inputIds: batch.inputIds, attentionMask: batch.attentionMask, imageHiddenStates: batch.imageHiddenStates },
        { generationConfig: this.generationConfig, settings: { max_new_tokens: limit, do_sample: false, return_dict_in_generate: false } },
      ).sequences[0]!;
      const answer = generated.slice(prompt);
      const description = this.processor.tokenizer.decode(answer, { skipSpecialTokens: true }).trim();
      const eos = this.eosIds();
      const complete = answer.length > 0 && eos.includes(answer[answer.length - 1]!);
      const pixels = value.pixels.detach();
      const bytes = tensorBytes(pixels);
      const suffix = new TextEncoder().encode(shapeDtypeRepr(pixels));
      const payload = new Uint8Array(bytes.length + suffix.length);
      payload.set(bytes, 0);
      payload.set(suffix, bytes.length);
      const foundation = this.config.foundation_source;
      return {
        interpretation: description,
        verification: 'unverified',
        uncertainty: { status: 'uncalibrated', confidence: null },
        source: { source_id: value.source_id, kind: 'full-image', shape: [...pixels.shape], sha256: sha256Hex(payload) },
        question: value.question,
        completion_status: complete ? 'complete' : 'token_limit',
        workspace: {
          active: Math.abs(this.gate.item()) > 0,
          visual_tokens: visualTokens,
          attention: workspace.attention.select(0, 0).tolist() as number[][],
          relations: workspace.relations.select(0, 0).tolist() as number[][],
        },
        foundation_source: foundation === undefined ? null : deepCopy(foundation),
      };
    });
  }
}

interface LanguageTool {
  readonly language: SceneLanguage | null;
  configuration(): JsonObject;
  loss(inputs: never, targets: unknown): Tensor;
}

/** Replayable language-mode objective (Python ``SceneLanguageObjective``). */
export class SceneLanguageObjective extends ModuleOperation<Record<string, unknown>, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode.tools.scene.SceneLanguageObjective';
  private readonly tool: LanguageTool;

  constructor(tool: LanguageTool) {
    super();
    this.tool = tool;
  }

  override get replayable(): boolean {
    return true;
  }

  override parameters(): Parameter[] {
    return this.tool.language!.parameters();
  }

  configuration(): JsonObject {
    return { operation: 'tensorcode.tools.scene.SceneLanguageObjective', config: this.tool.configuration() };
  }

  forward(value: Record<string, unknown>, context: Context | null): Tensor {
    if (context && Object.keys(context).length) throw new ValueError('Scene language objective does not accept context');
    return (this.tool.loss as (inputs: unknown, targets: unknown) => Tensor).call(this.tool, value.inputs, value.targets);
  }
}
