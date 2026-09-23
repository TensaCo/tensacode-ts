/** Port of Python ``tests/training/test_operation_identity.py``. */
import { describe, expect, it } from 'vitest';
import { Linear } from '../../src/nn/index.js';
import { Operation } from '../../src/ops/base.js';
import { bindingRecords, operationConfiguration, validateBindings } from '../../src/_internal/fingerprint.js';
import { transform } from './helpers.js';

class Objective extends Operation {
  override get replayable(): boolean {
    return true;
  }

  forward(value: unknown): unknown {
    return value;
  }

  configuration(): unknown {
    return { role: 'objective', model: { width: 4 } };
  }

  operationIdentity(): unknown {
    return 'tensorcode.ops.vec.decode.TextDecoder.objective';
  }
}

describe('owned objective identities', () => {
  it('survive private class moves; configuration changes stay incompatible', () => {
    class RelocatedObjective extends Objective {}
    const source = new Objective();
    const restored = new RelocatedObjective();
    expect(operationConfiguration(source).type).toBe('tensorcode.ops.vec.decode.TextDecoder.objective');
    validateBindings(bindingRecords({ objective: source }), { objective: restored });
    restored.configuration = () => ({ role: 'objective', model: { width: 8 } });
    expect(() => validateBindings(bindingRecords({ objective: source }), { objective: restored })).toThrow(/Incompatible/);
  });

  for (const identity of ['', null, 12]) {
    it(`rejects invalid explicit identity ${JSON.stringify(identity)}`, () => {
      const source = new Objective();
      source.operationIdentity = () => identity;
      expect(() => operationConfiguration(source)).toThrow(/identity/);
    });
  }

  it('keeps the concrete type identity without an override', () => {
    expect(operationConfiguration(transform(new Linear(1, 1))).type).toBe('tensorcode._internal.vec.adapter.TensorAdapter');
  });
});
