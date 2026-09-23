/** uint16 tensors behave like torch.uint16: a storage type promoted only to floating types. */
import { describe, expect, it } from 'vitest';
import { tensor } from '../../src/nn/index.js';
import { deserializeSafetensors, serializeSafetensors } from '../../src/nn/safetensors.js';

const PYTHON_HEX = '70000000000000007b2262223a7b226474797065223a22463332222c227368617065223a5b315d2c22646174615f6f666673657473223a5b302c345d7d2c2261223a7b226474797065223a22553136222c227368617065223a5b335d2c22646174615f6f666673657473223a5b342c31305d7d7d202020200000c03f0100ffff0700';

describe('uint16 tensors', () => {
  it('promote only to floating types, like torch.uint16', () => {
    const values = tensor([1, 65535], { dtype: 'uint16' });
    expect(values.add(tensor([0.5, 0.5])).dtype).toBe('float32');
    expect(values.to('float32').tolist()).toEqual([1, 65535]);
    expect(() => values.add(tensor([1, 2], { dtype: 'int32' }))).toThrow('Promotion for uint16, uint32, uint64 types is not supported, attempted to promote UInt16 and Int');
  });
});

describe('uint16 safetensors', () => {
  it('serializes U16 exactly as safetensors.torch.save does', () => {
    // Python: save({'a': torch.tensor([1, 65535, 7], dtype=torch.uint16), 'b': torch.tensor([1.5])}).hex()
    const bytes = serializeSafetensors({ a: tensor([1, 65535, 7], { dtype: 'uint16' }), b: tensor([1.5]) });
    expect(Buffer.from(bytes).toString('hex')).toBe(PYTHON_HEX);
    const loaded = deserializeSafetensors(bytes).tensors.get('a')!;
    expect(loaded.dtype).toBe('uint16');
    expect(loaded.tolist()).toEqual([1, 65535, 7]);
  });
});
