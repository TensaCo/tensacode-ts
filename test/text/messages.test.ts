/** Port of ``tests/text/test_messages.py`` plus pure-operation contracts. */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import * as text from '../../src/ops/text/index.js';
import { ValueError } from '../../src/errors.js';

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-text-messages-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));
const bytes = (value: string | number[]) => new Uint8Array(typeof value === 'string' ? Buffer.from(value) : value);

describe('messages', () => {
  it('keeps legacy string content and freezes part sequences', () => {
    const legacy = new text.Message('user', 'hello');
    const parts = [new text.TextPart('look', { sourceRef: 'ticket:1' })];
    const multipart = new text.Message('user', parts);
    parts.push(new text.TextPart('later'));
    expect(legacy.content).toBe('hello');
    expect(multipart.content).toEqual([new text.TextPart('look', { sourceRef: 'ticket:1' })]);
    expect(Object.isFrozen(multipart.content)).toBe(true);
  });

  it('image encoder preserves bytes and urls without fetching', () => {
    const encoder = new text.ImageEncoder({ media_type: 'image/png', source_ref: 'upload:7' });
    const encodedBytes = encoder.call(bytes([0x89, 0x50, 0x4e, 0x47]));
    const encodedUrl = encoder.call('https://example.test/image.png');
    const bytePart = (encodedBytes[0]!.content as readonly text.ImagePart[])[0];
    const urlPart = (encodedUrl[0]!.content as readonly text.ImagePart[])[0];
    expect(bytePart).toEqual(new text.ImagePart({ data: bytes([0x89, 0x50, 0x4e, 0x47]), mediaType: 'image/png', sourceRef: 'upload:7' }));
    expect(urlPart).toEqual(new text.ImagePart({ url: 'https://example.test/image.png', mediaType: 'image/png', sourceRef: 'upload:7' }));
  });

  it('image encoder preserves an explicit image part without overrides', () => {
    const explicit = new text.ImagePart({ data: bytes('image'), mediaType: 'image/jpeg', sourceRef: 'camera:1', detail: 'high' });
    const encoded = new text.ImageEncoder({ media_type: 'image/png', source_ref: 'encoder-default', detail: 'low' }).call(explicit);
    expect(encoded).toEqual([new text.Message('user', [explicit])]);
    expect((encoded[0]!.content as readonly text.ImagePart[])[0]).toBe(explicit);
  });

  it('image part requires exactly one source', () => {
    expect(() => new text.ImagePart({})).toThrow(/exactly one/);
    expect(() => new text.ImagePart({ data: bytes('x'), url: 'https://example.test/x' })).toThrow(/exactly one/);
    expect(() => new text.ImagePart({ url: 'file:///tmp/private' })).toThrow(/URL/);
  });

  it('text decoder handles multipart assistant text only', () => {
    const messages = [
      new text.Message('user', 'hello'),
      new text.Message('assistant', [new text.TextPart('one'), new text.TextPart('two')]),
    ];
    expect(new text.TextDecoder().call(messages)).toBe('onetwo');
    const image = [new text.Message('assistant', [new text.ImagePart({ url: 'https://example.test/x.png' })])];
    expect(() => new text.TextDecoder().call(image)).toThrow(/image/);
    expect(() => new text.TextDecoder().call([new text.Message('user', 'x')])).toThrow(/final assistant/);
    expect(() => new text.TextDecoder().call([])).toThrow(/final assistant/);
  });

  it('image encoder artifact preserves current source settings', async () => {
    const encoder = new text.ImageEncoder({ source_ref: 'source:before' });
    encoder.sourceRef = 'source:after';
    await encoder.savePretrained(join(scratch, 'image'));
    const restored = await text.ImageEncoder.fromPretrained(join(scratch, 'image'));
    expect(restored.call(bytes('image'))).toEqual(encoder.call(bytes('image')));
    expect(restored.configuration().source_ref).toBe('source:after');
  });

  it('encoders reject context and invalid inputs', () => {
    expect(() => new text.TextEncoder().call('x', { context: { a: 1 } })).toThrow(/context/);
    expect(new text.TextEncoder().call('x', { context: {} })).toEqual([new text.Message('user', 'x')]);
    expect(() => new text.ImageEncoder().call(42 as never)).toThrow(TypeError);
    expect(() => new text.ImageEncoder({ detail: 'medium' })).toThrow(ValueError);
    expect(() => new text.ImageEncoder({ unknown: 1 })).toThrow(/Unknown configuration fields/);
    expect(new text.TextEncoder().replayable && new text.TextDecoder().replayable && new text.ImageEncoder().replayable).toBe(true);
  });
});

describe('pure configuration round trip (test_owned_operations.test_pure_config_roundtrip)', () => {
  const cases: [string, new (config?: unknown) => any, Record<string, unknown>][] = [
    ['TextEncoder', text.TextEncoder, {}], ['TextDecoder', text.TextDecoder, {}], ['ImageEncoder', text.ImageEncoder, { detail: 'high' }],
  ];
  for (const [name, cls, config] of cases) {
    it(name, async () => {
      const op = new cls(config);
      await op.savePretrained(join(scratch, `pure-${name}`));
      expect((await (cls as any).fromPretrained(join(scratch, `pure-${name}`))).configuration()).toEqual(op.configuration());
      expect(() => new cls({ model: 'obsolete' })).toThrow(ValueError);
    });
  }
});
