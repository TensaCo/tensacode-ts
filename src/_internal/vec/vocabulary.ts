/**
 * Small trainable text encoder with an explicit, caller-supplied vocabulary
 * (Python ``tensorcode/_internal/vec/vocabulary.py``).
 *
 * Lowercase regex tokenization and mean pooling are authored mechanics. The
 * word embeddings are learned parameters; this is not a pretrained language
 * model.
 */
import { EmbeddingBag } from '../../nn/layers.js';
import type { Tensor } from '../../nn/tensor.js';
import { ValueError } from '../../errors.js';
import type { Context } from '../../ops/base.js';
import { Latent, Space } from '../../ops/vec/latent.js';
import { LatentOperation } from '../latentOps.js';
import { validatedConfig } from '../operationConfig.js';
import { lower, wordTokens } from '../text/casefold.js';

/** Python ``re.findall(r"\w+|[^\w\s]", text.lower())``. */
export function tokenize(text: unknown): string[] {
  if (typeof text !== 'string') throw new TypeError('VocabularyEncoder expects strings');
  return wordTokens(lower(text));
}

/** Mean-pooled word embeddings over an explicit vocabulary (index 0 is unknown). */
export class VocabularyEncoder extends LatentOperation<string | readonly string[], Tensor | Latent> {
  static override readonly qualifiedName: string = 'tensorcode._internal.vec.vocabulary.VocabularyEncoder';
  readonly vocabulary: readonly string[];
  readonly lookup: ReadonlyMap<string, number>;
  readonly embedding: EmbeddingBag;
  readonly outputSpace: Space | null;

  constructor(config: unknown) {
    const validated = validatedConfig(config, ['vocabulary', 'dimensions', 'output_space'], { dimensions: 64, output_space: null });
    const vocabulary = validated.vocabulary;
    const dimensions = validated.dimensions;
    if (!Array.isArray(vocabulary) || !vocabulary.every((word) => typeof word === 'string')) {
      throw new ValueError('Vocabulary must be a list of unique strings');
    }
    if (new Set(vocabulary).size !== vocabulary.length) throw new ValueError('Vocabulary must contain unique strings');
    if (typeof dimensions !== 'number' || !Number.isInteger(dimensions) || dimensions <= 0) {
      throw new ValueError('dimensions must be a positive integer');
    }
    const outputSpace = validated.output_space === null ? null : Space.fromConfig(validated.output_space);
    if (outputSpace !== null && outputSpace.dimensions !== dimensions) {
      throw new ValueError('VocabularyEncoder output_space dimensions must match dimensions');
    }
    super(validated);
    this.vocabulary = Object.freeze([...(vocabulary as string[])]);
    this.lookup = new Map(this.vocabulary.map((word, index) => [word, index + 1]));
    this.embedding = this.registerModule('embedding', new EmbeddingBag(this.vocabulary.length + 1, dimensions, { mode: 'mean' }));
    this.outputSpace = outputSpace;
  }

  forward(value: string | readonly string[], context: Context | null): Tensor | Latent {
    if (context) throw new ValueError('VocabularyEncoder does not consume context');
    const single = typeof value === 'string';
    const texts: readonly unknown[] = single ? [value] : [...(value as readonly unknown[])];
    if (!texts.length) throw new ValueError('Empty batch');
    const indices: number[] = [];
    const offsets: number[] = [];
    for (const text of texts) {
      offsets.push(indices.length);
      const ids = tokenize(text).map((token) => this.lookup.get(token) ?? 0);
      indices.push(...(ids.length ? ids : [0]));
    }
    const encoded = this.embedding.forward(indices, offsets);
    const result = single ? encoded.select(0, 0) : encoded;
    return this.outputSpace === null ? result : new Latent(result, this.outputSpace);
  }
}
