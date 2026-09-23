/**
 * Owned text candidate models shared by Investigator, Decision, Planner and
 * Scene (Python ``tensorcode/_internal/ranking.py``). Predictions are not
 * observations or actions. FOUNDATION-OWNED.
 */
import { Module } from '../nn/module.js';
import { Parameter, Tensor, tensor, zerosLike } from '../nn/tensor.js';
import { Embedding, GELU, Linear, Sequential } from '../nn/layers.js';
import { cat, stack } from '../nn/ops/shape.js';
import { ValueError } from '../errors.js';
import { ModuleOperation, type Context, type OperationLike } from '../ops/base.js';
import { TensorAdapter as Transform } from './vec/adapter.js';
import { Workspace, type WorkspaceOutput } from './workspace.js';
import { conversationBlock, conversationContext } from './conversation.js';
import { casefold, wordTokens } from './text/casefold.js';
import { deepCopy, isPlainObject, type JsonObject, type JsonValue } from './json.js';
import { className, qualifiedName } from './identity.js';
import { FastTokenizer } from './tokenizers/index.js';
import { NativeConfig } from './native/config.js';
import { createNativeModel } from './native/registry.js';
import { loadNativeFoundation, type FoundationOptions } from './native/foundation.js';
import type { EncoderOutput, NativeModel } from './native/modules.js';

/** Validate and default a ranking tool configuration (Python ``normalize_config``). */
export function normalizeRankingConfig(config: JsonObject): JsonObject {
  const result: JsonObject = deepCopy(config);
  if ('foundation_config' in result) {
    if (typeof result.tokenizer_json !== 'string') throw new ValueError('foundation requires tokenizer_json');
    if (!('freeze_foundation' in result)) result.freeze_foundation = true;
    if (typeof result.freeze_foundation !== 'boolean') throw new ValueError('freeze_foundation must be boolean');
    if (!('vocabulary' in result)) result.vocabulary = ['<foundation>'];
  }
  const vocabulary = result.vocabulary;
  if (!Array.isArray(vocabulary) || !vocabulary.length || vocabulary.some((item) => typeof item !== 'string' || !item)) {
    throw new ValueError('vocabulary must be a nonempty list of unique nonempty strings');
  }
  if (new Set(vocabulary).size !== vocabulary.length) throw new ValueError('vocabulary must contain unique strings');
  for (const [key, fallback] of [['dimensions', 32], ['slots', 4], ['steps', 2], ['max_tokens', 256]] as const) {
    if (!(key in result)) result[key] = fallback;
    const value = result[key];
    if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) throw new ValueError(`${key} must be a positive integer`);
  }
  if ((result.architecture_version ?? 1) !== 1) throw new ValueError('unsupported ranking architecture_version');
  if (!('cache_records' in result)) result.cache_records = 0;
  const records = result.cache_records;
  if (typeof records !== 'number' || !Number.isInteger(records) || records < 0) throw new ValueError('cache_records must be a nonnegative integer');
  result.architecture_version = 1;
  return result;
}

/** Native contextual encoder; configuration and tokenizer are checkpoint-owned. */
export class FoundationEncoding extends Module {
  static override readonly qualifiedName: string = 'tensorcode._internal.ranking.FoundationEncoding';
  readonly config: JsonObject;
  readonly model: NativeModel & { forward(inputs: unknown): EncoderOutput };
  readonly frozen: boolean;

  constructor(config: JsonObject) {
    super();
    this.config = deepCopy(config);
    const native = NativeConfig.fromDict(config.foundation_config);
    this.model = this.registerModule('model', createNativeModel(native, 'base') as FoundationEncoding['model']);
    this.frozen = config.freeze_foundation === true;
    this.model.requiresGrad_(!this.frozen);
    if (this.frozen) this.model.eval();
  }

  configuration(): JsonObject {
    return { foundation_config: deepCopy(this.config.foundation_config!), freeze_foundation: this.frozen };
  }

  override train(mode = true): this {
    super.train(mode);
    if (this.frozen) this.model.eval();
    return this;
  }

  /** ``model(**inputs).last_hidden_state`` for tokenizer tensors. */
  forward(inputs: { input_ids: Tensor; attention_mask?: Tensor; token_type_ids?: Tensor }): Tensor {
    return this.model.forward({
      inputIds: inputs.input_ids, attentionMask: inputs.attention_mask ?? null, tokenTypeIds: inputs.token_type_ids ?? null,
    }).lastHiddenState;
  }
}

export interface RankingRecord {
  [key: string]: JsonValue;
}

export interface RankComputation {
  scores: Tensor;
  workspace: WorkspaceOutput;
  sources: (string | null)[];
}

export type WorkspaceAblation = 'zero' | 'bypass' | null;

/** Encode task, dialogue, evidence and candidates; score candidates through the workspace. */
export class RankOperation extends ModuleOperation<Record<string, unknown>, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.ranking.RankOperation';
  readonly config: JsonObject;
  readonly taskKey: string;
  readonly candidatesKey: string;
  readonly vocabulary: Map<string, number>;
  readonly tokenizer: FastTokenizer | null;
  readonly encode: Transform;
  readonly projection: Transform<Linear> | null;
  readonly workspace: Workspace;
  readonly query: Transform<Linear>;
  readonly candidate: Transform<Linear>;
  readonly score: Transform<Sequential>;
  private readonly encodingCache = new Map<string, Tensor>();

  constructor(config: JsonObject, options: { taskKey: string; candidatesKey: string }) {
    super();
    this.config = deepCopy(config);
    this.taskKey = options.taskKey;
    this.candidatesKey = options.candidatesKey;
    this.vocabulary = new Map((config.vocabulary as string[]).map((token, index) => [token, index + 1]));
    const dimensions = config.dimensions as number;
    if ('foundation_config' in config) {
      this.tokenizer = FastTokenizer.fromJsonString(config.tokenizer_json as string, (config.tokenizer_special_tokens as Record<string, string> | undefined) ?? {});
      const encoding = new FoundationEncoding(config);
      this.encode = this.registerModule('encode', new Transform(encoding));
      this.projection = this.registerModule('projection', new Transform(new Linear(encoding.model.config.hiddenSize, dimensions)));
    } else {
      this.tokenizer = null;
      this.encode = this.registerModule('encode', new Transform(new Embedding(this.vocabulary.size + 1, dimensions)));
      this.projection = null;
    }
    this.workspace = this.registerModule('workspace', new Workspace(dimensions, config.slots as number, config.steps as number));
    this.query = this.registerModule('query', new Transform(new Linear(dimensions, dimensions)));
    this.candidate = this.registerModule('candidate', new Transform(new Linear(dimensions, dimensions)));
    this.score = this.registerModule('score', new Transform(new Sequential(new Linear(dimensions * 4, dimensions), new GELU(), new Linear(dimensions, 1))));
  }

  override get replayable(): boolean {
    return true;
  }

  configuration(): JsonObject {
    return { operation: qualifiedName(this), config: deepCopy(this.config), task_key: this.taskKey, candidates_key: this.candidatesKey };
  }

  /** Validate the ranking input; returns ``[evidence, candidates]``. */
  validate(value: unknown): [RankingRecord[], RankingRecord[]] {
    if (!isPlainObject(value) || typeof value[this.taskKey] !== 'string' || !(value[this.taskKey] as string).trim()) {
      throw new ValueError(`${this.taskKey} must be nonempty text`);
    }
    const evidence = value.evidence ?? [];
    const candidates = value[this.candidatesKey];
    if (!Array.isArray(evidence) || !Array.isArray(candidates) || !candidates.length) {
      throw new ValueError('evidence must be a list and candidates a nonempty list');
    }
    for (const [records, idKey] of [[evidence, 'source_id'], [candidates, 'id']] as const) {
      const ids: string[] = [];
      for (const record of records) {
        if (!isPlainObject(record) || typeof record[idKey] !== 'string' || !record[idKey] || typeof record.text !== 'string' || !(record.text as string).trim()) {
          throw new ValueError(`each record requires nonempty ${idKey} and text`);
        }
        ids.push(record[idKey] as string);
      }
      if (new Set(ids).size !== ids.length) throw new ValueError(`${idKey} values must be unique`);
    }
    return [evidence as RankingRecord[], candidates as RankingRecord[]];
  }

  clearEncodingCache(): void {
    this.encodingCache.clear();
  }

  protected override onRegistryChange(): void {
    this.encodingCache?.clear();
  }

  /** Vocabulary ids of ``text`` (casefolded word tokens, truncated to ``max_tokens``). */
  tokens(text: string): Tensor {
    const words = wordTokens(casefold(text)).slice(0, this.config.max_tokens as number);
    const ids = words.map((word) => this.vocabulary.get(word) ?? 0);
    return tensor(ids.length ? ids : [0], { dtype: 'int64' });
  }

  compute(value: Record<string, unknown>, options: { workspaceAblation?: WorkspaceAblation } = {}): RankComputation {
    const [evidence, candidates] = this.validate(value);
    const dialogue = conversationContext(value);
    const dialogueText = conversationBlock(dialogue);
    const maxTokens = this.config.max_tokens as number;
    if (dialogue.length) {
      const length = this.tokenizer ? this.tokenizer.encode(dialogueText).inputIds[0]!.length : wordTokens(casefold(dialogueText)).length;
      if (length > maxTokens) throw new ValueError('Conversation context exceeds ranking token budget');
    }
    const segments: [string | null, string][] = [[null, value[this.taskKey] as string]];
    if (dialogue.length) segments.push([null, dialogueText]);
    for (const item of evidence) segments.push([item.source_id as string, item.text as string]);
    const texts = [...segments.map(([, text]) => text), ...candidates.map((item) => item.text as string)];
    let encodedTexts: Tensor[];
    if (this.tokenizer) {
      const inputs = this.tokenizer.encodeTensors(texts, { padding: true, truncation: true, maxLength: maxTokens });
      const encoding = this.encode.module as FoundationEncoding;
      const cacheEnabled = (this.config.cache_records as number) > 0
        && !this.encode.parameters().some((parameter) => parameter.requiresGrad) && !encoding.model.training;
      const key = JSON.stringify(texts);
      let hidden = cacheEnabled ? this.encodingCache.get(key) ?? null : null;
      if (hidden === null) {
        hidden = this.encode.call(inputs) as Tensor;
        if (cacheEnabled) {
          this.encodingCache.set(key, hidden.detach().clone());
          if (this.encodingCache.size > (this.config.cache_records as number)) {
            this.encodingCache.delete(this.encodingCache.keys().next().value as string);
          }
        }
      } else {
        this.encodingCache.delete(key);
        this.encodingCache.set(key, hidden);
        hidden = hidden.clone();
      }
      const batch = this.projection!.call(hidden) as Tensor;
      const mask = inputs.attention_mask;
      encodedTexts = texts.map((_, index) => batch.select(0, index).maskedSelect(mask.select(0, index).bool()));
    } else {
      encodedTexts = texts.map((text) => this.encode.call(this.tokens(text)) as Tensor);
    }
    const states: Tensor[] = [];
    const sources: (string | null)[] = [];
    segments.forEach(([sourceId], index) => {
      const encoded = encodedTexts[index]!;
      states.push(encoded);
      for (let token = 0; token < encoded.shape[0]!; token += 1) sources.push(sourceId);
    });
    const workspace = this.workspace.forward(cat(states, 0).unsqueeze(0));
    const ablation = options.workspaceAblation ?? null;
    if (ablation !== null && ablation !== 'zero' && ablation !== 'bypass') throw new ValueError('workspace_ablation must be None, zero, or bypass');
    let conditioning = workspace.conditioning.mean(1);
    if (ablation === 'zero') conditioning = zerosLike(conditioning);
    else if (ablation === 'bypass') conditioning = cat(states, 0).mean(0, true);
    const width = conditioning.shape[1]!;
    const query = (this.query.call(conditioning) as Tensor).expand(candidates.length, width);
    const candidate = this.candidate.call(stack(encodedTexts.slice(segments.length).map((encoded) => encoded.mean(0)))) as Tensor;
    const features = cat([query, candidate, query.mul(candidate), query.sub(candidate).abs()], -1);
    const scores = (this.score.call(features) as Tensor).squeeze(-1);
    return { scores, workspace, sources };
  }

  forward(value: Record<string, unknown>, context: Context | null): Tensor {
    if (context) throw new ValueError('RankOperation does not accept context');
    return this.compute(value).scores;
  }

  /** Inspectable ranking receipt (selected id, scores, attention, relations). */
  receipt(value: Record<string, unknown>, options: { probabilities?: boolean } = {}): JsonObject {
    const { scores, workspace, sources } = this.compute(value);
    const values = scores.toArray();
    let selected = 0;
    for (let index = 1; index < values.length; index += 1) if (values[index]! > values[selected]!) selected = index;
    const candidates = value[this.candidatesKey] as RankingRecord[];
    const result: JsonObject = {
      selected_id: candidates[selected]!.id!,
      candidates: candidates.map((item, index) => ({ ...deepCopy(item), predicted_score: values[index]! })),
      evidence: deepCopy((value.evidence ?? []) as JsonValue),
      attention: workspace.attention.select(0, 0).detach().tolist() as JsonValue,
      attention_source_ids: sources,
      relations: workspace.relations.select(0, 0).detach().tolist() as JsonValue,
    };
    if (options.probabilities) {
      const probabilities = scores.softmax(-1).toArray();
      (result.candidates as JsonObject[]).forEach((item, index) => { item.probability = probabilities[index]!; });
    }
    return result;
  }
}

/** A tool trained through {@link RankingObjective}. */
export interface RankingTool extends Module {
  configuration(): JsonObject;
  loss(inputs: unknown, targets: unknown): Tensor;
  proposalLoss?(inputs: unknown, targets: unknown): Tensor;
  generationLoss?(inputs: unknown, targets: unknown): Tensor;
  verificationLoss?(inputs: unknown, targets: unknown): Tensor;
  retrievalLoss?(inputs: unknown, targets: unknown): Tensor;
}

/** Objective envelope ``{inputs, targets}`` routed by an optional ``mode``. */
export class RankingObjective extends ModuleOperation<Record<string, unknown>, Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.ranking.RankingObjective';
  private readonly tool: RankingTool;

  constructor(tool: RankingTool) {
    super();
    this.tool = tool;
  }

  override get replayable(): boolean {
    return true;
  }

  override parameters(): Parameter[] {
    return this.tool.parameters();
  }

  configuration(): JsonObject {
    return { operation: qualifiedName(this), tool: className(this.tool), config: this.tool.configuration() };
  }

  forward(value: Record<string, unknown>, context: Context | null): Tensor {
    if (context) throw new ValueError('RankingObjective does not accept context');
    let inputs = value.inputs;
    let mode = (value.mode as string | undefined) ?? 'rank';
    if (isPlainObject(inputs) && Object.keys(inputs).length === 2 && 'mode' in inputs && 'inputs' in inputs) {
      mode = inputs.mode as string;
      inputs = inputs.inputs;
    }
    if (mode === 'rank') return this.tool.loss(inputs, value.targets);
    let objective: ((inputs: unknown, targets: unknown) => Tensor) | undefined;
    if (mode === 'proposal') objective = this.tool.proposalLoss ?? this.tool.generationLoss;
    else if (mode === 'verification') objective = this.tool.verificationLoss;
    else if (mode === 'retrieval') objective = this.tool.retrievalLoss;
    else throw new ValueError('training mode must be rank, proposal, verification, or retrieval');
    if (!objective) throw new ValueError(`${mode} training capability is not configured`);
    return objective.call(this.tool, inputs, value.targets).clone();
  }
}

/** Replayable registered operations of a tool (Python ranking ``bindings(tool)``). */
export function replayableBindings(tool: Module): Record<string, OperationLike> {
  const result: Record<string, OperationLike> = {};
  for (const [name, module] of tool.namedModules()) {
    if (name && (module as { replayable?: unknown }).replayable === true) result[name] = module as unknown as OperationLike;
  }
  return result;
}

/**
 * Build a ranking tool from a pretrained encoder foundation (Python
 * ``ranking.from_foundation``). The workspace and ranking head start random;
 * the encoder weights, configuration and tokenizer become part of the tool.
 */
export async function rankingFromFoundation<T extends Module & { rank: RankOperation }>(
  construct: (config: JsonObject) => T, repo: string,
  options: Omit<FoundationOptions, 'head'> & { options?: JsonObject } = {},
): Promise<T> {
  const { options: toolOptions, ...load } = options;
  const loaded = await loadNativeFoundation(repo, { ...load, head: 'base', initializeMissing: true });
  if (!loaded.tokenizer) throw new ValueError('foundation requires a serializable fast tokenizer');
  const resolved = loaded.commitHash ?? load.revision ?? null;
  const special: JsonObject = {};
  for (const [key, value] of Object.entries(loaded.tokenizer.specialTokensMap)) if (typeof value === 'string') special[key] = value;
  const config: JsonObject = {
    ...(toolOptions ?? {}),
    foundation_config: loaded.config.toDict(),
    tokenizer_json: loaded.tokenizer.rustJsonText,
    tokenizer_special_tokens: special,
    foundation: { repository: repo, revision: resolved, workspace_initialization: 'random' },
  };
  const result = construct(config);
  const encoding = result.rank.encode.module as FoundationEncoding;
  encoding.model.loadStateDict(loaded.model.stateDict());
  return result;
}
