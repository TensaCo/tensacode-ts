/** Port of the vector parts of ``tests/integration/test_operation_boundaries.py``. */
import { describe, expect, it } from 'vitest';
import { zeros } from '../../src/nn/index.js';
import * as vec from '../../src/ops/vec/index.js';
import * as encode from '../../src/ops/vec/encode.js';
import * as decode from '../../src/ops/vec/decode.js';
import * as transform from '../../src/ops/vec/transform.js';
import * as classify from '../../src/ops/vec/classify.js';
import * as score from '../../src/ops/vec/score.js';
import { qualifiedName } from '../../src/_internal/identity.js';

describe('vector operation boundaries', () => {
  it('vector classes have public operation identities', () => {
    const modules: Record<string, [Record<string, unknown>, string[]]> = {
      encode: [encode, ['TextEncoder', 'ImageEncoder', 'VocabularyEncoder', 'PatchEncoder']],
      decode: [decode, ['Decode', 'TextDecoder', 'ImageDecoder']],
      transform: [transform, ['Transform']],
      classify: [classify, ['Classify']],
      score: [score, ['Score']],
    };
    for (const [module, [namespace, names]] of Object.entries(modules)) {
      for (const name of names) {
        const cls = namespace[name];
        expect(cls).toBe((vec as Record<string, unknown>)[name]);
        expect(qualifiedName(cls)).toBe(`tensorcode.ops.vec.${module}.${name}`);
      }
    }
    for (const [name, identity] of [
      ['Decide', 'tensorcode.ops.vec.decide.Decide'], ['Retrieve', 'tensorcode.ops.vec.retrieve.Retrieve'],
      ['CandidateSet', 'tensorcode.ops.vec.candidates.CandidateSet'], ['Prediction', 'tensorcode.ops.vec.classify.Prediction'],
    ]) expect(qualifiedName((vec as Record<string, unknown>)[name])).toBe(identity);
  });

  it('exports every Python name, alias and latentCodecs', () => {
    for (const name of [
      'Latent', 'Space', 'Transform', 'Classify', 'Prediction', 'CandidateSet', 'Scores', 'Score', 'Decide', 'Decision',
      'Retrieve', 'Retrieval', 'TextEncoder', 'TextEncode', 'ImageEncoder', 'ImageEncode', 'VocabularyEncoder', 'PatchEncoder',
      'Decode', 'Decoder', 'TextDecoder', 'TextDecode', 'ImageDecoder', 'ImageDecode', 'latentCodecs',
    ]) expect((vec as Record<string, unknown>)[name], name).toBeDefined();
    expect(vec.TextEncode).toBe(vec.TextEncoder);
    expect(vec.Decoder).toBe(vec.Decode);
    expect(vec.latentCodecs()).toEqual({ 'tensorcode.Latent': vec.Latent, 'tensorcode.Space': vec.Space });
  });

  it('backend helpers and tool-only encoders are not public', () => {
    expect('SequenceEncoder' in encode).toBe(false);
    expect('tokenize' in encode).toBe(false);
  });

  it('specialized encoders declare their output space', () => {
    const textSpace = new vec.Space('vocabulary', 4);
    const text = new vec.VocabularyEncoder({ vocabulary: ['hello'], dimensions: 4, output_space: textSpace.configuration() as any });
    expect(text.outputSpace!.equals(textSpace)).toBe(true);
    expect((text.call('hello') as vec.Latent).space.equals(textSpace)).toBe(true);
    expect(vec.VocabularyEncoder.toolIdentity()).toBe('tensorcode.ops.vec.encode.VocabularyEncoder');
    expect(text.configuration().output_space).toEqual(textSpace.configuration());
    const imageSpace = new vec.Space('patches', 4, { organization: 'spatial' });
    const image = new vec.PatchEncoder({ patch_size: 2, in_channels: 3, output_space: imageSpace.configuration() as any });
    expect(image.call(zeros([3, 4, 4])).space.equals(imageSpace)).toBe(true);
    expect(vec.PatchEncoder.toolIdentity()).toBe('tensorcode.ops.vec.encode.PatchEncoder');
    expect(image.configuration().output_space).toEqual(imageSpace.configuration());
  });
});
