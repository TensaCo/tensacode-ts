/**
 * Private non-owning encoder for complete tool models (Python
 * ``tensorcode/_internal/vec/sequence.py``).
 */
import type { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import { Operation, type Context } from '../../ops/base.js';
import {
  emitJsonRaw, parseJsonRaw, rawFromValue, rawGet, rawSet, sha256Hex, type JsonObject, type RawNode,
} from '../json.js';
import type { FastTokenizer } from '../tokenizers/index.js';
import type { T5ForConditionalGeneration } from '../native/t5.js';
import { NativeConfig } from '../native/config.js';

/**
 * Python ``json.loads(model.config.to_json_string())``: the diff against
 * transformers defaults, even when the configuration was supplied already
 * normalized (``NativeConfig`` keeps such configurations verbatim).
 */
export function configJsonString(config: NativeConfig): JsonObject {
  if (!config.isVerbatim) return config.toDiffDict();
  const data = config.toDict();
  delete data.transformers_version;
  return NativeConfig.fromDict(data).toDiffDict();
}

export interface SequenceEncoding {
  /** ``[batch, tokens, d_model]`` encoder states. */
  encoded: Tensor;
  /** ``[batch, tokens]`` int64 attention mask. */
  mask: Tensor;
}

const SPECIAL_ORDER = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token'];

/**
 * ``json.loads(PreTrainedTokenizerFast(tokenizer_object=..., **special).backend_tokenizer.to_str())``
 * as raw JSON: transformers 5 adds the special tokens to the backend (as
 * special, non-normalized added tokens) and installs a default
 * ``TemplateProcessing`` post-processor when none is configured.
 */
export function backendTokenizerNode(tokenizer: FastTokenizer): RawNode {
  const root = parseJsonRaw(tokenizer.jsonText);
  rawSet(root, 'padding', rawFromValue(null));
  rawSet(root, 'truncation', rawFromValue(null));
  const processor = rawGet(root, 'post_processor');
  if (processor === undefined || (processor.t === 'l' && processor.v === null)) {
    rawSet(root, 'post_processor', rawFromValue({
      type: 'TemplateProcessing',
      single: [{ Sequence: { id: 'A', type_id: 0 } }],
      pair: [{ Sequence: { id: 'A', type_id: 0 } }, { Sequence: { id: 'B', type_id: 1 } }],
      special_tokens: {},
    }));
  }
  const existing = rawGet(root, 'added_tokens');
  const added: RawNode[] = existing?.t === 'a' ? [...existing.items] : [];
  const specials = tokenizer.specialTokensMap;
  const ordered = [...SPECIAL_ORDER, ...Object.keys(specials).filter((key) => !SPECIAL_ORDER.includes(key))];
  const tokens: string[] = [];
  for (const key of ordered) {
    const value = specials[key];
    if (typeof value === 'string') tokens.push(value);
    else if (Array.isArray(value)) tokens.push(...value.map(String));
  }
  for (const content of tokens) {
    const id = tokenizer.backend.tokenToId(content);
    if (id === undefined) continue;
    const token = rawFromValue({ id, content, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true });
    const index = added.findIndex((item) => rawGet(item, 'content')?.t === 's' && (rawGet(item, 'content') as { v: string }).v === content);
    if (index >= 0) {
      const special = rawGet(added[index]!, 'special');
      if (special?.t === 'l' && special.v === true) continue;
      added[index] = token;
    } else {
      added.push(token);
    }
  }
  const idOf = (node: RawNode): number => {
    const id = rawGet(node, 'id');
    return id?.t === 'n' ? Number(id.raw) : 0;
  };
  rawSet(root, 'added_tokens', { t: 'a', items: added.sort((a, b) => idOf(a) - idOf(b)) });
  return root;
}

/** Python ``sha256(json.dumps(spec_with_padding_and_truncation_none, sort_keys=True))``. */
export function tokenizerSha256(tokenizer: FastTokenizer): string {
  return sha256Hex(emitJsonRaw(backendTokenizerNode(tokenizer), { sortKeys: true }));
}

/**
 * Tokenize text and run an owner's local pretrained sequence encoder.
 *
 * Ownership stays with the enclosing model; this callable exposes its encoding
 * boundary without registering the shared encoder/decoder weights twice.
 */
export class SequenceEncoder extends Operation<readonly string[], SequenceEncoding> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.sequence.SequenceEncoder';
  private readonly model: T5ForConditionalGeneration;
  readonly tokenizer: FastTokenizer;
  readonly maxTokens: number;
  readonly tokenizerSha256: string;

  constructor(model: T5ForConditionalGeneration, tokenizer: FastTokenizer, options: { maxTokens?: number } = {}) {
    super();
    this.model = model;
    this.tokenizer = tokenizer;
    this.maxTokens = options.maxTokens ?? 512;
    this.tokenizerSha256 = tokenizerSha256(tokenizer);
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: readonly string[], context: Context | null): SequenceEncoding {
    if (context && Object.keys(context).length) throw new ValueError('SequenceEncoder does not consume context');
    const batch = this.tokenizer.encodeTensors(value, { padding: true, truncation: true, maxLength: this.maxTokens });
    const encoded = this.model.getEncoder().forward({ inputIds: batch.input_ids, attentionMask: batch.attention_mask }).lastHiddenState;
    return { encoded, mask: batch.attention_mask };
  }

  configuration(): JsonObject {
    return {
      operation: 'tensorcode._internal.vec.sequence.SequenceEncoder',
      max_tokens: this.maxTokens,
      model: configJsonString(this.model.config),
      tokenizer_sha256: this.tokenizerSha256,
      special_tokens: this.tokenizer.specialTokensMap as JsonObject,
    };
  }
}
