/**
 * Owned image-and-text model with spatial workspace organization (Python
 * ``tensorcode/tools/scene.py``).
 *
 * Ranking mode scores supplied descriptions; attention identifies patch
 * routing, not factual support. It does not construct scene graphs. Fresh
 * models have random visual weights.
 *
 * Language mode (Python ``Scene.from_language_foundation`` / ``interpret`` over
 * an Idefics3/SmolVLM foundation) is not available in the TypeScript port:
 * those methods, and loading a language-mode artifact, throw
 * ``NotImplementedError``.
 */
import { readFile, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Parameter, Tensor, arange, onesLike, randn, tensor, zerosLike } from '../nn/tensor.js';
import { noGrad } from '../nn/autograd.js';
import { Embedding, GELU, GRU, Linear, Sequential } from '../nn/layers.js';
import { cat, meshgrid, stack } from '../nn/ops/shape.js';
import { cosineSimilarity, crossEntropy } from '../nn/ops/nn.js';
import { NotImplementedError, ValueError } from '../errors.js';
import { ModuleOperation, type Context, type OperationLike } from '../ops/base.js';
import { Space } from '../ops/vec/latent.js';
import { PatchEncoder } from '../ops/vec/encode.js';
import { PretrainedModule } from '../_internal/pretrained.js';
import { RankingObjective, replayableBindings } from '../_internal/ranking.js';
import { Workspace, type WorkspaceOutput } from '../_internal/workspace.js';
import { TensorAdapter as Transform } from '../_internal/vec/adapter.js';
import { resizeImage } from '../_internal/vec/imageProcessing.js';
import { casefold, wordTokens } from '../_internal/text/casefold.js';
import { isSymlink } from '../_internal/files.js';
import { deepCopy, isPlainObject, parseJsonStrict, sha256Hex, type JsonObject, type JsonValue } from '../_internal/json.js';
import { NativeConfig } from '../_internal/native/config.js';
import { CLIPModel } from '../_internal/native/clip.js';
import { createNativeModel } from '../_internal/native/registry.js';
import { loadNativeFoundation } from '../_internal/native/foundation.js';
import { Tokenizer } from '../_internal/tokenizers/index.js';
import { canonicalBackendJson, rustTokenizerString } from '../_internal/tokenizers/serialization.js';

export const SCENE_LANGUAGE_UNAVAILABLE = 'Scene language mode (Idefics3/SmolVLM interpretation) is not available in the TypeScript port; use the Python package';

/** One ranking candidate. */
export interface SceneCandidate {
  id: string;
  text: string;
  [key: string]: JsonValue;
}

/** Scene ranking input: a CHW image in ``[0, 1]``, a question and candidates. */
export interface SceneInputs {
  question: string;
  source_id: string;
  pixels: Tensor;
  candidates: SceneCandidate[];
}

export type SceneAblation = 'zero' | 'bypass' | null;

export interface SceneComputation {
  logits: Tensor;
  workspace: WorkspaceOutput;
  coordinates: Tensor;
}

function validateInputs(value: unknown, config: JsonObject, options: { channels: number | null; minimum: number }): [Tensor, SceneCandidate[]] {
  if (!isPlainObject(value)) throw new ValueError('scene inputs must be a dictionary');
  for (const key of ['question', 'source_id']) {
    if (typeof value[key] !== 'string' || !(value[key] as string).trim()) throw new ValueError(`${key} must be nonempty text`);
  }
  const pixels = value.pixels;
  if (!(pixels instanceof Tensor) || pixels.ndim !== 3 || (options.channels !== null && pixels.shape[0] !== options.channels)) {
    throw new ValueError('pixels must be CHW with configured channels');
  }
  const [, height, width] = pixels.shape as [number, number, number];
  if (Math.min(height, width) < options.minimum || Math.max(height, width) > (config.max_image_size as number)) {
    throw new ValueError('image dimensions outside configured bounds');
  }
  if (!pixels.isFloatingPoint || !pixels.allFinite() || pixels.min().item() < 0 || pixels.max().item() > 1) {
    throw new ValueError('pixels must be finite floating values in [0, 1]');
  }
  const candidates = value.candidates;
  if (!Array.isArray(candidates) || !candidates.length || candidates.length > (config.max_candidates as number)) {
    throw new ValueError('candidates must be a nonempty bounded list');
  }
  const ids: string[] = [];
  for (const item of candidates) {
    if (!isPlainObject(item) || ['id', 'text'].some((key) => typeof item[key] !== 'string' || !(item[key] as string).trim())) {
      throw new ValueError('candidates require nonempty id and text');
    }
    ids.push(item.id as string);
  }
  if (new Set(ids).size !== ids.length) throw new ValueError('candidate IDs must be unique');
  return [pixels, candidates as SceneCandidate[]];
}

function applyAblation(query: Tensor, encoded: Tensor, ablation: SceneAblation | undefined): Tensor {
  if (ablation === 'zero') return zerosLike(query);
  if (ablation === 'bypass') return encoded.mean(1);
  if (ablation !== null && ablation !== undefined) throw new ValueError('workspace_ablation must be None, zero or bypass');
  return query;
}

function interaction(query: Tensor, candidate: Tensor): Tensor {
  return cat([query, candidate, query.mul(candidate), query.sub(candidate).abs()], -1);
}

/** Owned patch/vocabulary ranking model (random initialization). */
export class SceneRank extends ModuleOperation<SceneInputs, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode.tools.scene.SceneRank';
  readonly config: JsonObject;
  readonly modality: Parameter;
  readonly workspace: Workspace;
  readonly score: Transform<Sequential>;
  readonly vocabulary: ReadonlyMap<string, number>;
  readonly image: PatchEncoder | null = null;
  readonly text: Transform<Embedding> | null = null;
  readonly position: Transform<Linear> | null = null;
  readonly text_position: Embedding | null = null;
  readonly candidate_sequence: GRU | null = null;

  constructor(config: JsonObject, options: { foundation?: boolean } = {}) {
    super();
    this.config = deepCopy(config);
    const d = config.dimensions as number;
    this.vocabulary = new Map((config.vocabulary as string[]).map((word, index) => [word, index + 1]));
    if (!options.foundation) {
      this.image = this.registerModule('image', new PatchEncoder({
        patch_size: config.patch_size!, in_channels: config.in_channels!,
        output_space: new Space('scene-patches', d, { organization: 'spatial' }).configuration() as unknown as JsonObject,
      }));
      this.text = this.registerModule('text', new Transform(new Embedding(this.vocabulary.size + 1, d)));
      this.position = this.registerModule('position', new Transform(new Linear(2, d)));
      this.text_position = this.registerModule('text_position', new Embedding(config.max_tokens as number, d));
      this.candidate_sequence = this.registerModule('candidate_sequence', new GRU(d, d, { batchFirst: true }));
    } else {
      this.registerFoundationModules(config);
    }
    this.modality = this.registerParameter('modality', new Parameter(noGrad(() => randn([2, d]).mul(0.02))));
    this.workspace = this.registerModule('workspace', new Workspace(d, config.slots as number, config.steps as number));
    this.score = this.registerModule('score', new Transform(new Sequential(new Linear(d * 4, d), new GELU(), new Linear(d, 1))));
  }

  /** Subclass hook registering perception modules before the shared workspace. */
  protected registerFoundationModules(config: JsonObject): void {
    void config;
  }

  override get replayable(): boolean {
    return true;
  }

  configuration(): JsonObject {
    return { operation: 'tensorcode.tools.scene.SceneRank', config: deepCopy(this.config) };
  }

  /** Vocabulary ids of ``text`` (casefolded word tokens, truncated to ``max_tokens``). */
  tokens(text: string): Tensor {
    const words = wordTokens(casefold(text)).slice(0, this.config.max_tokens as number);
    const ids = words.map((word) => this.vocabulary.get(word) ?? 0);
    return tensor(ids.length ? ids : [0], { dtype: 'int64' });
  }

  validate(value: unknown): [Tensor, SceneCandidate[]] {
    return validateInputs(value, this.config, { channels: this.config.in_channels as number, minimum: this.config.patch_size as number });
  }

  compute(value: SceneInputs, options: { workspaceAblation?: SceneAblation } = {}): SceneComputation {
    const [raw, candidates] = this.validate(value);
    const d = this.config.dimensions as number;
    const pixels = raw.dtype === this.modality.dtype ? raw : raw.to(this.modality.dtype);
    const visual = this.image!.call(pixels);
    const coordinates = visual.coordinates!.reshape(-1, 2);
    const normalizer = tensor([pixels.shape[1]!, pixels.shape[2]!], { dtype: coordinates.dtype });
    const patches = visual.tensor.reshape(-1, d)
      .add(this.position!.call(coordinates.div(normalizer)) as Tensor)
      .add(this.modality.select(0, 0));
    const questionTokens = this.tokens(value.question);
    const question = (this.text!.call(questionTokens) as Tensor)
      .add(this.text_position!.forward(arange(0, questionTokens.shape[0]!)))
      .add(this.modality.select(0, 1));
    const encoded = cat([patches, question], 0).unsqueeze(0);
    const workspace = this.workspace.forward(encoded);
    const query = applyAblation(workspace.conditioning.mean(1), encoded, options.workspaceAblation);
    const candidate = stack(candidates.map((item) => {
      const embedded = (this.text!.call(this.tokens(item.text)) as Tensor).unsqueeze(0);
      return this.candidate_sequence!.forward(embedded).hidden.select(0, 0).select(0, 0);
    }));
    const logits = (this.score.call(interaction(query.expandAs(candidate), candidate)) as Tensor).squeeze(-1);
    return { logits, workspace, coordinates };
  }

  forward(value: SceneInputs, context: Context | null): Tensor {
    if (context) throw new ValueError('SceneRank does not accept context');
    return this.compute(value).logits;
  }
}

/** Least-recently-used feature cache holding detached copies. */
class FeatureCache {
  private readonly entries = new Map<string, Tensor[]>();

  get size(): number {
    return this.entries.size;
  }

  get(key: string): Tensor[] | undefined {
    const value = this.entries.get(key);
    if (value) {
      this.entries.delete(key);
      this.entries.set(key, value);
    }
    return value;
  }

  set(key: string, values: Tensor[]): void {
    this.entries.delete(key);
    this.entries.set(key, values.map((item) => item.detach().clone()));
    if (this.entries.size > 1024) this.entries.delete(this.entries.keys().next().value as string);
  }

  clear(): void {
    this.entries.clear();
  }
}

/**
 * Frozen owned CLIP perception with a trainable crossmodal workspace. CLIP's
 * inherited representations supply perceptual competence; TensorCode learns
 * the workspace and candidate ranking. Cached features are session-free
 * runtime optimizations and never appear in model artifacts.
 */
export class FoundationSceneRank extends SceneRank {
  static override readonly qualifiedName: string = 'tensorcode.tools.scene.FoundationSceneRank';
  readonly tokenizerJson: string;
  readonly tokenizer: Tokenizer;
  declare foundation: CLIPModel;
  declare image_projection: Transform<Linear>;
  declare text_projection: Transform<Linear>;
  declare candidate_projection: Transform<Linear>;
  declare global_projection: Transform<Linear>;
  declare private visionCache: FeatureCache | undefined;
  declare private textCache: FeatureCache | undefined;

  constructor(config: JsonObject, tokenizerJson: string) {
    super(config, { foundation: true });
    this.tokenizerJson = tokenizerJson;
    this.tokenizer = Tokenizer.fromString(tokenizerJson);
    this.visionCache = new FeatureCache();
    this.textCache = new FeatureCache();
  }

  protected override registerFoundationModules(config: JsonObject): void {
    const d = config.dimensions as number;
    const foundation = createNativeModel(NativeConfig.fromDict(config.foundation_config), 'base');
    if (!(foundation instanceof CLIPModel)) throw new ValueError('foundation_config must describe a CLIP model');
    this.foundation = this.registerModule('foundation', foundation);
    this.foundation.requiresGrad_(false);
    this.foundation.eval();
    const vision = this.foundation.config.sub('vision_config');
    const text = this.foundation.config.sub('text_config');
    const projection = this.foundation.config.number('projection_dim');
    this.image_projection = this.registerModule('image_projection', new Transform(new Linear(vision.number('hidden_size'), d)));
    this.text_projection = this.registerModule('text_projection', new Transform(new Linear(text.number('hidden_size'), d)));
    this.candidate_projection = this.registerModule('candidate_projection', new Transform(new Linear(projection, d)));
    this.global_projection = this.registerModule('global_projection', new Transform(new Linear(2 * projection + 1, d)));
  }

  override configuration(): JsonObject {
    return { operation: 'tensorcode.tools.scene.FoundationSceneRank', config: deepCopy(this.config) };
  }

  /** Number of cached image / text feature entries. */
  get cacheSizes(): { vision: number; text: number } {
    return { vision: this.visionCache?.size ?? 0, text: this.textCache?.size ?? 0 };
  }

  clearFeatures(): void {
    this.visionCache?.clear();
    this.textCache?.clear();
  }

  protected override onRegistryChange(): void {
    this.clearFeatures();
  }

  override train(mode = true): this {
    super.train(mode);
    this.foundation?.eval();
    return this;
  }

  /** CLIP token ids (with special tokens), truncated to the text position capacity. */
  override tokens(text: string): Tensor {
    let ids = this.tokenizer.postProcess(this.tokenizer.encodeText(text), null, true).ids;
    const textConfig = this.foundation.config.sub('text_config');
    const limit = textConfig.number('max_position_embeddings');
    if (ids.length > limit) ids = [...ids.slice(0, limit - 1), textConfig.number('eos_token_id')];
    return tensor(ids, { dtype: 'int64' });
  }

  override validate(value: unknown): [Tensor, SceneCandidate[]] {
    return validateInputs(value, this.config, { channels: this.config.in_channels as number, minimum: this.config.patch_size as number });
  }

  /** Frozen CLIP text states and projected global feature (cached; Python ``_encode_text``). */
  encodeText(text: string): [Tensor, Tensor] {
    let cached = this.textCache!.get(text);
    if (!cached) {
      cached = noGrad(() => {
        const ids = this.tokens(text).unsqueeze(0);
        const output = this.foundation.text_model.forward(ids, onesLike(ids));
        const pooled = this.foundation.text_projection.forward(output.poolerOutput).select(0, 0);
        return [output.lastHiddenState.select(0, 0), pooled];
      });
      this.textCache!.set(text, cached);
    }
    return [cached[0]!.clone(), cached[1]!.clone()];
  }

  /** Frozen CLIP patch states and projected global feature (cached; Python ``_encode_image``). */
  encodeImage(pixels: Tensor): [Tensor, Tensor] {
    // Include pixels rather than trusting a possibly reused external source ID.
    const bytes = new Uint8Array(pixels.data.buffer, pixels.data.byteOffset, pixels.data.byteLength);
    const key = `${sha256Hex(bytes)}:${pixels.shape.join('x')}:${pixels.dtype}`;
    let cached = this.visionCache!.get(key);
    if (!cached) {
      cached = noGrad(() => {
        const vision = this.foundation.config.sub('vision_config');
        const size = vision.number('image_size');
        const image = resizeImage(pixels.detach().unsqueeze(0).to('float32'), [size, size], 'bicubic');
        const mean = tensor(this.config.image_mean as number[]).reshape(1, -1, 1, 1);
        const std = tensor(this.config.image_std as number[]).reshape(1, -1, 1, 1);
        const output = this.foundation.vision_model.forward(image.sub(mean).div(std));
        const pooled = this.foundation.visual_projection.forward(output.poolerOutput).select(0, 0);
        return [output.lastHiddenState.select(0, 0).slice(0, 1), pooled];
      });
      this.visionCache!.set(key, cached);
    }
    return [cached[0]!.clone(), cached[1]!.clone()];
  }

  override compute(value: SceneInputs, options: { workspaceAblation?: SceneAblation } = {}): SceneComputation {
    const [pixels, candidates] = this.validate(value);
    const [patches, imageGlobal] = this.encodeImage(pixels);
    const [question, questionGlobal] = this.encodeText(value.question);
    const encoded = cat([
      (this.image_projection.call(patches) as Tensor).add(this.modality.select(0, 0)),
      (this.text_projection.call(question) as Tensor).add(this.modality.select(0, 1)),
    ], 0).unsqueeze(0);
    const workspace = this.workspace.forward(encoded);
    let query = applyAblation(workspace.conditioning.mean(1), encoded, options.workspaceAblation);
    // Aligned pretrained globals are explicit foundation features; their
    // relevance is learned by the supplied-data objective.
    const similarity = cosineSimilarity(imageGlobal, questionGlobal, 0).reshape(1);
    query = query.add((this.global_projection.call(cat([imageGlobal, questionGlobal, similarity], 0)) as Tensor).unsqueeze(0));
    const candidate = this.candidate_projection.call(stack(candidates.map((item) => this.encodeText(item.text)[1]))) as Tensor;
    const logits = (this.score.call(interaction(query.expandAs(candidate), candidate)) as Tensor).squeeze(-1);
    const vision = this.foundation.config.sub('vision_config');
    const imageSize = vision.number('image_size');
    const patchSize = vision.number('patch_size');
    const side = Math.floor(imageSize / patchSize);
    const row = arange(0, side, 1, { dtype: query.dtype }).add(0.5).mul(patchSize * pixels.shape[1]! / imageSize);
    const column = arange(0, side, 1, { dtype: query.dtype }).add(0.5).mul(patchSize * pixels.shape[2]! / imageSize);
    const [rows, columns] = meshgrid(row, column);
    const coordinates = stack([rows!, columns!], -1).reshape(-1, 2);
    return { logits, workspace, coordinates };
  }
}

const RANK_DEFAULTS: readonly (readonly [string, number])[] = [['dimensions', 32], ['slots', 4], ['steps', 2], ['max_tokens', 256]];
const IMAGE_DEFAULTS: readonly (readonly [string, number])[] = [['patch_size', 8], ['in_channels', 3], ['max_image_size', 256], ['max_candidates', 64]];

function positiveInteger(value: unknown): boolean {
  return typeof value === 'number' && Number.isInteger(value) && value >= 1;
}

/** Validate and default a ranking-mode Scene configuration (private keys removed). */
function normalizeSceneConfig(input: unknown): JsonObject {
  if (!isPlainObject(input)) throw new ValueError('model config must be a JSON object');
  const config: Record<string, unknown> = { ...input };
  delete config._tokenizer_json;
  delete config._language_assets;
  if (config.mode === 'language') throw new NotImplementedError(SCENE_LANGUAGE_UNAVAILABLE);
  const vocabulary = config.vocabulary;
  if (!Array.isArray(vocabulary) || !vocabulary.length || vocabulary.some((word) => typeof word !== 'string' || !word)
    || new Set(vocabulary).size !== vocabulary.length) {
    throw new ValueError('vocabulary must contain unique nonempty strings');
  }
  for (const [name, fallback] of RANK_DEFAULTS) {
    if (!(name in config)) config[name] = fallback;
    if (!positiveInteger(config[name])) throw new ValueError(`${name} must be a positive integer`);
  }
  if ((config.architecture_version ?? 1) !== 1) throw new ValueError('unsupported scene architecture_version');
  config.architecture_version = 1;
  for (const [name, fallback] of IMAGE_DEFAULTS) {
    if (!(name in config)) config[name] = fallback;
    if (!positiveInteger(config[name])) throw new ValueError(`${name} must be a positive integer`);
  }
  if ((config.max_image_size as number) < (config.patch_size as number)) throw new ValueError('max_image_size must accommodate a patch');
  return config as JsonObject;
}

export interface SceneFoundationOptions {
  revision: string | null;
  localFilesOnly?: boolean;
  cacheDir?: string | null;
  token?: string | null;
  endpoint?: string | null;
  dimensions?: number;
  slots?: number;
  steps?: number;
}

/** A ranking receipt returned by {@link Scene.call}. */
export interface SceneReceipt extends JsonObject {
  selected_id: string;
  candidates: JsonObject[];
  source_id: string;
  patch_coordinates: number[][];
  attention: number[][];
  attention_source_ids: (string | null)[];
  relations: number[][];
}

/**
 * Rank supplied descriptions for an image and question. Ranking uses a
 * learned image/text workspace; selected candidates remain fallible
 * interpretations. This interface supplies no object vocabulary or spatial
 * truth rules.
 */
export class Scene extends PretrainedModule<SceneInputs, SceneReceipt> {
  static override readonly qualifiedName: string = 'tensorcode.tools.scene.Scene';
  readonly rank: SceneRank;
  readonly objective: RankingObjective;

  constructor(config: unknown) {
    super(normalizeSceneConfig(config));
    const tokenizerJson = (config as Record<string, unknown>)._tokenizer_json;
    if ('foundation_config' in this.config) {
      if (typeof tokenizerJson !== 'string' || sha256Hex(tokenizerJson) !== this.config.tokenizer_sha256) {
        throw new ValueError('foundation construction requires matching tokenizer assets');
      }
      this.rank = this.registerModule('rank', new FoundationSceneRank(this.config, tokenizerJson));
    } else {
      this.rank = this.registerModule('rank', new SceneRank(this.config));
    }
    this.objective = this.registerModule('objective', new RankingObjective(this));
  }

  /** Rank ``candidates`` for an image and question; returns an inspectable receipt. */
  forward(inputs: SceneInputs, context: Context | null): SceneReceipt {
    if (context) throw new ValueError('Scene does not accept context');
    const { logits, workspace, coordinates } = this.rank.compute(inputs);
    const scores = logits.detach().toArray();
    const probabilities = logits.detach().softmax(-1).toArray();
    let selected = 0;
    for (let index = 1; index < scores.length; index += 1) if (scores[index]! > scores[selected]!) selected = index;
    const patchCoordinates = coordinates.detach().tolist() as number[][];
    return {
      selected_id: inputs.candidates[selected]!.id,
      candidates: inputs.candidates.map((item, index) => ({ ...deepCopy(item as JsonObject), predicted_score: scores[index]!, probability: probabilities[index]! })),
      source_id: inputs.source_id,
      patch_coordinates: patchCoordinates,
      attention: workspace.attention.select(0, 0).detach().tolist() as number[][],
      attention_source_ids: [
        ...patchCoordinates.map(() => inputs.source_id),
        ...Array.from({ length: this.rank.tokens(inputs.question).shape[0]! }, () => null),
      ],
      relations: workspace.relations.select(0, 0).detach().tolist() as number[][],
    };
  }

  /** Alias of {@link Scene.call}. */
  predict(inputs: SceneInputs): SceneReceipt {
    return this.call(inputs);
  }

  /** Candidate ranking loss against a candidate ID or index. */
  loss(inputs: SceneInputs, targets: unknown): Tensor {
    const logits = this.rank.call(inputs);
    let target = targets;
    if (typeof target === 'string') {
      const ids = inputs.candidates.map((item) => item.id);
      if (!ids.includes(target)) throw new ValueError('target must identify a supplied candidate');
      target = ids.indexOf(target);
    }
    if (typeof target !== 'number' || !Number.isInteger(target) || target < 0 || target >= logits.numel) {
      throw new ValueError('target must be a valid candidate index or ID');
    }
    return crossEntropy(logits.unsqueeze(0), tensor([target], { dtype: 'int64' }));
  }

  /** Objective used by ``Trainer.fromTool``. */
  get trainingOperation(): RankingObjective {
    return this.objective;
  }

  get trainingInputsIncludeTargets(): boolean {
    return true;
  }

  /** Named operations for tracing, experience and checkpoints. */
  override operationBindings(): Record<string, OperationLike> {
    return replayableBindings(this);
  }

  /** Unavailable: language-mode interpretation requires the Python package. */
  interpret(inputs: unknown, options: { maxNewTokens?: number | null } = {}): never {
    void inputs;
    void options;
    throw new NotImplementedError(SCENE_LANGUAGE_UNAVAILABLE);
  }

  /** Unavailable: Idefics3/SmolVLM language foundations require the Python package. */
  static async fromLanguageFoundation(...args: unknown[]): Promise<never> {
    void args;
    throw new NotImplementedError(SCENE_LANGUAGE_UNAVAILABLE);
  }

  /** Explicitly import pinned pretrained CLIP perception; ranking starts random. */
  static async fromFoundation(repoId = 'openai/clip-vit-base-patch32', options: SceneFoundationOptions): Promise<Scene> {
    const { revision, dimensions = 32, slots = 4, steps = 2, ...hub } = options;
    const loaded = await loadNativeFoundation(repoId, { ...hub, revision, head: 'base' });
    if (!(loaded.model instanceof CLIPModel)) throw new ValueError('Scene foundation must be a CLIP checkpoint');
    if (!loaded.tokenizer) throw new ValueError('Scene foundation requires a tokenizer.json');
    const processor = parseJsonStrict(await readFile(join(loaded.directory, 'preprocessor_config.json'), 'utf8'));
    if (!isPlainObject(processor)) throw new ValueError('preprocessor_config.json must contain a JSON object');
    const tokenizerJson = loaded.tokenizer.rustJsonText;
    const config: JsonObject = {
      vocabulary: ['<foundation>'], dimensions, slots, steps,
      foundation_config: loaded.config.toDict(),
      foundation_source: { repo_id: repoId, revision },
      _tokenizer_json: tokenizerJson,
      tokenizer_sha256: sha256Hex(tokenizerJson),
      image_mean: processor.image_mean as JsonValue,
      image_std: processor.image_std as JsonValue,
      preprocessing: 'bicubic-antialiased-square-resize',
      patch_size: loaded.config.sub('vision_config').number('patch_size'),
    };
    const model = new Scene(config);
    (model.rank as FoundationSceneRank).foundation.loadStateDict(loaded.model.stateDict());
    return model;
  }

  protected override async savePretrainedAssets(directory: string): Promise<void> {
    if (this.rank instanceof FoundationSceneRank) {
      const path = join(directory, 'tokenizer.json');
      if (await isSymlink(path)) await unlink(path);
      // Python writes ``Tokenizer.from_str(tokenizer_json).to_str()``.
      await writeFile(path, rustTokenizerString(canonicalBackendJson(this.rank.tokenizerJson, { rustParsed: true })), 'utf8');
    }
  }

  static override async loadPretrainedConfig(config: JsonObject, directory: string): Promise<JsonObject> {
    if (config.mode === 'language') throw new NotImplementedError(SCENE_LANGUAGE_UNAVAILABLE);
    if ('foundation_config' in config) {
      return { ...config, _tokenizer_json: await readFile(join(directory, 'tokenizer.json'), 'utf8') };
    }
    return config;
  }
}

