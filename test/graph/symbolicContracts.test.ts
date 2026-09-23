/** Port of ``tests/graph/test_symbolic_contracts.py``. */
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import * as graph from '../../src/ops/graph/index.js';
import { isOperation } from '../../src/ops/base.js';
import { NotImplementedError, ValueError } from '../../src/errors.js';
import { qualifiedName } from '../../src/_internal/identity.js';
import { trace } from '../../src/_internal/tracing.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-graph-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const names = ['Encode', 'TextEncode', 'Decode', 'TextDecode', 'Transform', 'Score', 'Retrieve', 'Classify', 'Decide'] as const;
const values: Record<(typeof names)[number], unknown> = {
  Encode: {}, TextEncode: 'Evidence with two competing interpretations.',
  Decode: new graph.Graph(['a']), TextDecode: new graph.Graph(['a']), Transform: new graph.Graph(['a']),
  Score: new graph.Graph(['a']), Retrieve: new graph.Graph(['a']), Classify: new graph.Graph(['a']),
  Decide: new graph.ChoiceInput(new graph.Graph(['a']), [new graph.Graph(['b'])]),
};
interface SymbolicClass {
  new (config?: unknown): graph.SymbolicOperation;
  fromFoundation(...args: unknown[]): Promise<never>;
  fromPretrained(...args: unknown[]): Promise<never>;
}

describe('symbolic graph contracts', () => {
  it.each(names)('%s fails explicitly in sync, async and traced calls', async (name) => {
    const operation = new (graph[name] as unknown as SymbolicClass)();
    expect(isOperation(operation)).toBe(true);
    expect(operation.replayable).toBe(false);
    expect(operation.configuration()).toEqual({ operation: `graph.${name}`, implementation: 'unimplemented' });
    expect(qualifiedName(operation)).toBe(`tensorcode.ops.graph.${name.replace(/^Text/, '').toLowerCase()}.${name}`);
    expect(() => operation.call(values[name] as never)).toThrow(/symbolic semantics are not implemented/);
    await expect(operation.acall(values[name] as never)).rejects.toThrow(/symbolic semantics are not implemented/);
    const session = trace();
    expect(() => session.run(() => operation.call(values[name] as never))).toThrow(NotImplementedError);
  });

  it('symbolic transform does not accept callback compatibility', () => {
    expect(() => new graph.Transform(((value: unknown) => value) as never)).toThrow(TypeError);
  });

  it('no neural graph implementation or JSON operation aliases exist', () => {
    expect('JSONEncoder' in graph).toBe(false);
    expect('JSONDecoder' in graph).toBe(false);
  });

  it('choice input requires actual graph alternatives', () => {
    const objective = new graph.Graph(['goal']);
    expect(() => new graph.ChoiceInput(objective, [])).toThrow(/at least one Graph/);
    expect(() => new graph.ChoiceInput('goal' as never, [objective])).toThrow(/objective/);
    expect(new graph.ChoiceInput(objective, [objective]).options).toEqual([objective]);
  });

  it.each(names)('%s configuration does not enable artifact or foundation loading', async (name) => {
    const cls = graph[name] as unknown as SymbolicClass;
    const operation = new cls({});
    expect(operation.configuration()).toEqual(new cls().configuration());
    expect(() => new cls({ model: 'implicit' })).toThrow(ValueError);
    expect(() => new cls({ model: 'implicit' })).toThrow(/Unknown configuration/);
    await expect(cls.fromFoundation('unused')).rejects.toThrow(NotImplementedError);
    await expect(cls.fromPretrained('unused')).rejects.toThrow(NotImplementedError);
    await expect(operation.savePretrained(join(scratch, name))).rejects.toThrow(NotImplementedError);
    expect(existsSync(join(scratch, name))).toBe(false);
  });
});
