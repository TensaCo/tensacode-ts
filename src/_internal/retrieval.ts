/**
 * Owned sentence embeddings with an explicit masked-mean/L2 model contract
 * (Python ``tensorcode/_internal/retrieval.py``).
 *
 * Only foundations documented for this pooling contract are compatible.
 * Loading a plain language encoder does not confer contrastively learned
 * retrieval ability.
 */
import { isDirectory } from './hub.js';
import { noGrad } from '../nn/autograd.js';
import { Tensor, tensor } from '../nn/tensor.js';
import { normalize } from '../nn/ops/nn.js';
import { torchDTypeName } from '../nn/dtype.js';
import { ValueError } from '../errors.js';
import type { Context } from '../ops/base.js';
import { PretrainedModule, validatedModelConfig } from './pretrained.js';
import { FoundationEncoding } from './ranking.js';
import { TensorAdapter as Transform } from './vec/adapter.js';
import { deepCopy, isPlainObject, type JsonObject } from './json.js';
import { qualifiedName } from './identity.js';
import { FastTokenizer } from './tokenizers/index.js';
import { loadNativeFoundation, type FoundationOptions } from './native/foundation.js';
import { withEvalModes } from './memory/learned.js';
import { pythonNamedBuffers } from './cognition/locking.js';

/**
 * Private tensor execution with an explicit native-model identity: its
 * complete native configuration and registered tensor schemas describe the
 * owned architecture.
 */
export class RetrievalTransform extends Transform<FoundationEncoding> {
  static override readonly qualifiedName: string = 'tensorcode._internal.retrieval._RetrievalTransform';

  override configuration(): JsonObject {
    return {
      operation: qualifiedName(this),
      module: this.module.configuration(),
      parameters: this.namedParameters().map(([name, value]) => ({
        name, shape: [...value.shape], dtype: torchDTypeName(value.dtype), requires_grad: value.requiresGrad,
      })),
      buffers: pythonNamedBuffers(this).map(([name, value]) => ({ name, shape: [...value.shape], dtype: torchDTypeName(value.dtype) })),
    };
  }
}

const ALLOWED = new Set([
  'foundation_config', 'tokenizer_json', 'tokenizer_special_tokens', 'pooling', 'normalize', 'max_tokens', 'freeze_foundation', 'foundation',
]);

function validateTexts(texts: unknown): asserts texts is readonly string[] {
  if (!Array.isArray(texts) || !texts.length || texts.some((text) => typeof text !== 'string' || !text.trim())) {
    throw new ValueError('retrieval input must be a nonempty sequence of nonempty strings');
  }
}

export interface RetrievalFoundationOptions extends Omit<FoundationOptions, 'head'> {
  pooling: string;
  normalize: boolean;
  maxTokens?: number;
  freezeFoundation?: boolean;
}

/**
 * A complete encoder/tokenizer artifact, without an added projection.
 *
 * Configuration construction initializes random weights. {@link fromFoundation}
 * explicitly imports weights; the caller must select a foundation whose model
 * card specifies masked mean pooling and L2 normalization at ``max_tokens``.
 */
export class RetrievalEncoder extends PretrainedModule<readonly string[], Tensor> {
  static override readonly qualifiedName: string = 'tensorcode._internal.retrieval.RetrievalEncoder';
  declare readonly tokenizer: FastTokenizer;
  declare readonly encode: RetrievalTransform;

  constructor(config: unknown) {
    const value = validatedModelConfig(config);
    if (Object.keys(value).some((key) => !ALLOWED.has(key))) throw new ValueError('unsupported retrieval encoder configuration fields');
    if (value.pooling !== 'masked_mean' || value.normalize !== true) {
      throw new ValueError('retrieval requires explicit pooling=masked_mean and normalize=True');
    }
    if (!isPlainObject(value.foundation_config) || typeof value.tokenizer_json !== 'string') {
      throw new ValueError('retrieval requires complete native encoder and fast tokenizer configuration');
    }
    if (!('max_tokens' in value)) value.max_tokens = 256;
    if (!('freeze_foundation' in value)) value.freeze_foundation = false;
    if (!('tokenizer_special_tokens' in value)) value.tokenizer_special_tokens = {};
    if (typeof value.max_tokens !== 'number' || !Number.isInteger(value.max_tokens) || value.max_tokens < 1) {
      throw new ValueError('max_tokens must be a positive integer');
    }
    if (typeof value.freeze_foundation !== 'boolean') throw new ValueError('freeze_foundation must be boolean');
    if (!isPlainObject(value.tokenizer_special_tokens)) throw new ValueError('tokenizer_special_tokens must be an object');
    if ('foundation' in value && !isPlainObject(value.foundation)) throw new ValueError('foundation provenance must be an object');
    const native = value.foundation_config as JsonObject;
    if (native.is_encoder_decoder || native.is_decoder) throw new ValueError('retrieval currently supports encoder-only foundations');
    super(value);
    const tokenizer = FastTokenizer.fromJsonString(value.tokenizer_json as string, value.tokenizer_special_tokens as Record<string, string>);
    if (tokenizer.padTokenId === null) throw new ValueError('retrieval tokenizer requires a padding token');
    (this as { tokenizer: FastTokenizer }).tokenizer = tokenizer;
    (this as { encode: RetrievalTransform }).encode = this.registerModule('encode', new RetrievalTransform(new FoundationEncoding(value)));
  }

  get metadata(): JsonObject {
    return {
      encoder: 'owned_retrieval_encoder', pooling: 'masked_mean', normalized: true,
      max_tokens: this.config.max_tokens!, dimensions: this.encode.module.model.config.hiddenSize,
      foundation: deepCopy(this.config.foundation ?? { initialization: 'configured_weights' }),
      compatibility: 'requires a foundation trained/documented for masked-mean pooling followed by L2 normalization',
      semantics: 'embedding proximity, not truth or calibrated evidence support',
    };
  }

  forward(inputs: readonly string[], context: Context | null): Tensor {
    if (context && Object.keys(context).length) throw new ValueError('retrieval encoder does not accept context');
    validateTexts(inputs);
    const tokens = this.tokenizer.encodeTensors([...inputs], { padding: true, truncation: true, maxLength: this.config.max_tokens as number });
    const hidden = (this.encode.call(tokens) as Tensor).float();
    const mask = tokens.attention_mask.unsqueeze(-1).to(hidden.dtype);
    const pooled = hidden.mul(mask).sum(1).div(mask.sum(1).clampMin(1));
    return normalize(pooled, 2, -1);
  }

  /** Inference vectors and explicit source truncation/pooling metadata. */
  receipt(texts: readonly string[]): JsonObject {
    validateTexts(texts);
    const vectors = withEvalModes(this, () => noGrad(() => this.call(texts).tolist())) as number[][];
    const lengths = texts.map((text) => this.tokenizer.encode(text, { truncation: false }).inputIds[0]!.length);
    return {
      ...this.metadata, embeddings: vectors,
      input_truncated: lengths.map((length) => length > (this.config.max_tokens as number)),
      input_token_counts: lengths,
    };
  }

  /** Explicit multi-positive contrastive supervision; no inferred positives. */
  contrastiveLoss(queries: readonly string[], documents: readonly string[], positiveMask: unknown, options: { temperature?: number } = {}): Tensor {
    validateTexts(queries);
    validateTexts(documents);
    const temperature = options.temperature ?? 0.05;
    if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature <= 0) {
      throw new ValueError('temperature must be finite and positive');
    }
    let positives: Tensor;
    if (positiveMask instanceof Tensor) positives = positiveMask;
    else {
      const rows = positiveMask as unknown[];
      const boolean = Array.isArray(rows) && rows.every((row) => Array.isArray(row) && row.every((item) => typeof item === 'boolean'));
      if (!boolean) throw new ValueError('positive_mask must be boolean [queries, documents] with a positive for each query');
      const width = (rows[0] as unknown[] | undefined)?.length ?? 0;
      if (rows.some((row) => (row as unknown[]).length !== width)) {
        throw new ValueError('positive_mask must be boolean [queries, documents] with a positive for each query');
      }
      positives = tensor((rows as boolean[][]).flat().map(Number), { shape: [rows.length, width], dtype: 'bool' });
    }
    if (positives.dtype !== 'bool' || positives.ndim !== 2 || positives.shape[0] !== queries.length
      || positives.shape[1] !== documents.length || !positives.any(1).all().item()) {
      throw new ValueError('positive_mask must be boolean [queries, documents] with a positive for each query');
    }
    const logits = this.call(queries).matmul(this.call(documents).transpose(0, 1)).div(temperature);
    const weights = positives.to(logits.dtype).div(positives.sum(1, true).to(logits.dtype));
    return weights.mul(logits.logSoftmax(-1)).sum(-1).mean().neg();
  }

  /** Alias of {@link contrastiveLoss}. */
  loss(queries: readonly string[], documents: readonly string[], positiveMask: unknown, options: { temperature?: number } = {}): Tensor {
    return this.contrastiveLoss(queries, documents, positiveMask, options);
  }

  /**
   * Load an explicitly selected compatible model; never downloads on construction.
   *
   * ``pooling: 'masked_mean', normalize: true`` is a compatibility declaration,
   * not automatic discovery of arbitrary SentenceTransformers module graphs.
   */
  static async fromFoundation(repo: string, options: RetrievalFoundationOptions): Promise<RetrievalEncoder> {
    const { pooling, normalize: normalizeOption, maxTokens = 256, freezeFoundation = false, ...load } = options;
    if (pooling !== 'masked_mean' || normalizeOption !== true) throw new ValueError('only masked_mean followed by L2 normalization is supported');
    const loaded = await loadNativeFoundation(repo, { ...load, head: 'base' });
    if (!loaded.tokenizer) throw new ValueError('retrieval foundation requires a serializable fast tokenizer');
    const resolved = loaded.commitHash ?? load.revision ?? null;
    if (!(await isDirectory(repo)) && !resolved) throw new ValueError('retrieval foundation provenance requires a resolved revision');
    const special: JsonObject = {};
    for (const [key, value] of Object.entries(loaded.tokenizer.specialTokensMap)) if (typeof value === 'string') special[key] = value;
    const result = new RetrievalEncoder({
      foundation_config: loaded.config.toDict(), tokenizer_json: loaded.tokenizer.jsonText, tokenizer_special_tokens: special,
      pooling, normalize: normalizeOption, max_tokens: maxTokens, freeze_foundation: freezeFoundation,
      foundation: { repository: repo, revision: resolved, pooling_contract: 'caller_declared_masked_mean_l2', weights: 'loaded_foundation' },
    });
    result.encode.module.model.loadStateDict(loaded.model.stateDict());
    result.eval();
    return result;
  }
}

export { isDirectory };
