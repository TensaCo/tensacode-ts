/**
 * PyTorch ``.bin`` checkpoints load like ``torch.load(weights_only=True)`` and
 * ``from_pretrained`` of a ``.bin``-only directory (fixtures:
 * scripts/fixtures/torch_checkpoint_fixtures.py).
 */
import { cpSync, readFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { noGrad } from '../../src/nn/index.js';
import { loadTorchStateDict } from '../../src/_internal/native/torchCheckpoint.js';
import { loadNativeFoundation } from '../../src/_internal/native/foundation.js';
import type { NativeSequenceClassifier } from '../../src/_internal/native/bert.js';
import { expectClose } from '../helpers/gradcheck.js';
import { ints } from '../helpers/fixtures.js';
import { scratchDirectory } from '../training/helpers.js';

const root = new URL('../fixtures/torch_checkpoint/', import.meta.url).pathname;
const expected = JSON.parse(readFileSync(join(root, 'expected.json'), 'utf8')) as {
  tensors: Record<string, { dtype: string; shape: number[]; data: number[] }>;
  input_ids: number[][];
  logits: { data: number[] };
};
const scratch = scratchDirectory('tensorcode-torch-checkpoint-');

describe('loadTorchStateDict', () => {
  it.each(['zip.bin', 'legacy.bin'])('reads %s like torch.load(weights_only=True)', (file) => {
    const tensors = loadTorchStateDict(new Uint8Array(readFileSync(join(root, file))));
    expect([...tensors.keys()]).toEqual(Object.keys(expected.tensors));
    for (const [name, record] of Object.entries(expected.tensors)) {
      const value = tensors.get(name)!;
      expect(value.dtype, name).toBe(record.dtype);
      expect(value.shape, name).toEqual(record.shape);
      expect(Array.from(value.data), name).toEqual(record.data.map(Number));
    }
  });

  it('refuses pickles that would call anything but tensor rebuilders', () => {
    expect(() => loadTorchStateDict(new Uint8Array(readFileSync(join(root, 'exploit.bin'))))).toThrow(
      /Weights only load failed\. Unsupported global: GLOBAL (posix|os)\.system/,
    );
  });
});

describe('.bin foundations', () => {
  const classify = async (directory: string): Promise<void> => {
    const loaded = await loadNativeFoundation(directory, { head: 'sequence-classification', localFilesOnly: true });
    const ids = ints(expected.input_ids);
    noGrad(() => {
      const output = (loaded.model as unknown as NativeSequenceClassifier).forward({ inputIds: ids, attentionMask: ids.ne(0).to('int64') });
      expectClose(output.logits.data, expected.logits.data, 1e-5, 1e-5);
    });
  };

  it('load legacy-format weights with LayerNorm gamma/beta names', () => classify(join(root, 'bert_bin')));
  it('load sharded zip-format weights', () => classify(join(root, 'bert_sharded')));

  it('keep safetensors-only loading where Python passes use_safetensors=True', async () => {
    await expect(loadNativeFoundation(join(root, 'bert_bin'), { head: 'sequence-classification', localFilesOnly: true, useSafetensors: true }))
      .rejects.toThrow(`Error no file named model.safetensors found in directory ${join(root, 'bert_bin')}.`);
    const empty = scratch();
    cpSync(join(root, 'bert_bin'), empty, { recursive: true });
    rmSync(join(empty, 'pytorch_model.bin'));
    await expect(loadNativeFoundation(empty, { head: 'sequence-classification', localFilesOnly: true }))
      .rejects.toThrow(`Error no file named model.safetensors, or pytorch_model.bin, found in directory ${empty}.`);
  });
});
