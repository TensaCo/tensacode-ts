/** Port of python/tests/runtime/test_memory.py (JSON memory with caller-defined retrieval). */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { JsonMemory, MemoryRecord, type MemorySearch } from '../../src/_internal/memory/json.js';
import { ImagePart, Message, TextPart } from '../../src/ops/text/messages.js';
import { scratch } from '../tools/helpers.js';

const temp = scratch();
afterAll(() => temp.cleanup());

function containsText(request: MemorySearch): MemoryRecord[] {
  return request.candidates.filter((record) => String(record.value).toLowerCase().includes(String(request.query).toLowerCase()))
    .slice(0, request.limit) as MemoryRecord[];
}

describe('JsonMemory', () => {
  it('uses caller retrieval and survives restart', async () => {
    const path = join(temp.dir, 'memory.json');
    const memory = await JsonMemory.open(path, { retrieve: containsText });
    const first = await memory.append('Customer asked about a card fee', { kind: 'observation', metadata: { turn: 1 } });
    const response = await memory.append('The assistant discussed fee policy', {
      kind: 'response', metadata: { turn: 1, observation_source_id: first.sourceId },
    });
    const restarted = await JsonMemory.open(path, { retrieve: containsText });
    expect(restarted.records).toEqual([first, response]);
    expect(restarted.search('card', { limit: 3 })).toEqual([first]);
    expect(first.sourceId).toBe('memory-00000001');
    expect(response.sourceId).toBe('memory-00000002');
    expect(first.kind).toBe('observation');
    expect(response.kind).toBe('response');
    expect(JSON.parse(readFileSync(path, 'utf8')).format).toBe('tensorcode-memory');
  });

  it('rolls back every staged record in a failed transaction', async () => {
    const path = join(temp.dir, 'rollback.json');
    const memory = await JsonMemory.open(path, { retrieve: containsText });
    const existing = await memory.append('kept', { kind: 'observation' });
    await expect(memory.transaction((transaction) => {
      transaction.append('not committed', { kind: 'observation' });
      transaction.append('also not committed', { kind: 'response' });
      throw new Error('turn failed');
    })).rejects.toThrow('turn failed');
    expect(memory.records).toEqual([existing]);
    expect((await JsonMemory.open(path, { retrieve: containsText })).records).toEqual([existing]);
  });

  it('rejects retrieval results outside the supplied candidates', async () => {
    const invented = new MemoryRecord('not-stored', 'observation', 'invented');
    const memory = new JsonMemory({ retrieve: () => [invented] });
    await memory.append('stored', { kind: 'observation' });
    expect(() => memory.search('anything')).toThrow(/candidate/);
  });

  it('rejects values its JSON codec cannot persist', async () => {
    const memory = await JsonMemory.open(join(temp.dir, 'codec.json'), { retrieve: containsText });
    await expect(memory.append(new Date() as unknown, { kind: 'observation' })).rejects.toThrow(TypeError);
    await expect(memory.append({ value: Number.NaN }, { kind: 'observation' })).rejects.toThrow(/JSON/);
    expect(memory.records).toEqual([]);
  });

  it('snapshots mutable inputs and returned records', async () => {
    const memory = new JsonMemory<{ nested: string[] }>({ retrieve: containsText as never });
    const supplied = { nested: ['original'] };
    await memory.append(supplied, { kind: 'observation' });
    supplied.nested.push('caller mutation');
    const exposed = memory.records;
    exposed[0]!.value.nested.push('reader mutation');
    expect(memory.records[0]!.value).toEqual({ nested: ['original'] });
  });

  it('failed transactions cannot mutate preexisting values', async () => {
    const memory = new JsonMemory<{ status: string }>({ retrieve: containsText as never });
    await memory.append({ status: 'kept' }, { kind: 'observation' });
    await expect(memory.transaction((transaction) => {
      transaction.records[0]!.value.status = 'mutated';
      throw new Error('rollback');
    })).rejects.toThrow();
    expect(memory.records[0]!.value).toEqual({ status: 'kept' });
  });

  it('rejects nested transactions instead of losing the inner commit', async () => {
    const memory = new JsonMemory({ retrieve: containsText });
    await memory.transaction(async (outer) => {
      outer.append('outer', { kind: 'observation' });
      await expect(memory.append('inner', { kind: 'observation' })).rejects.toThrow(/nested/);
    });
    expect(memory.records.map((record) => record.value)).toEqual(['outer']);
  });

  it('automatic source IDs skip explicitly occupied IDs', async () => {
    const memory = new JsonMemory({ retrieve: containsText });
    const explicit = await memory.append('explicit', { kind: 'observation', sourceId: 'memory-00000001' });
    const automatic = await memory.append('automatic', { kind: 'observation' });
    expect(explicit.sourceId).toBe('memory-00000001');
    expect(automatic.sourceId).toBe('memory-00000002');
  });

  it('retrieval cannot mutate a candidate into new evidence', async () => {
    const mutate = (request: MemorySearch<{ text: string }>) => {
      (request.candidates[0]!.value as { text: string }).text = 'invented';
      return request.candidates;
    };
    const memory = new JsonMemory<{ text: string }>({ retrieve: mutate });
    await memory.append({ text: 'original' }, { kind: 'observation' });
    expect(() => memory.search('anything')).toThrow(/candidate/);
    expect(memory.records[0]!.value).toEqual({ text: 'original' });
  });

  it('stores message sequences through the message codec', async () => {
    const path = join(temp.dir, 'messages.json');
    const memory = await JsonMemory.forMessages(path, { retrieve: (request) => request.candidates });
    const messages = [new Message('user', [new TextPart('look', { sourceRef: 'doc:1' }), new ImagePart({ data: new Uint8Array([0, 255]), mediaType: 'image/png' })])];
    await memory.append(messages, { kind: 'turn' });
    const restarted = await JsonMemory.forMessages(path, { retrieve: (request) => request.candidates });
    expect(restarted.records[0]!.value).toEqual(messages);
    expect(restarted.search(null)).toEqual(restarted.records);
  });
});
