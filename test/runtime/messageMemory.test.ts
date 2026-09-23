/** Port of python/tests/runtime/test_runtime_message_memory.py. */
import { describe, expect, it } from 'vitest';
import { ImagePart, Message, TextPart } from '../../src/ops/text/messages.js';
import { decodeMessageSequence, encodeMessageSequence } from '../../src/_internal/memory/messages.js';

describe('message memory codec', () => {
  it('round-trips text, images and sources', () => {
    const messages = [
      new Message('user', [
        new TextPart('inspect this', { sourceRef: 'document:1' }),
        new ImagePart({ data: new Uint8Array([0, 255, ...new TextEncoder().encode('image')]), mediaType: 'image/png', sourceRef: 'image:1' }),
      ]),
      new Message('assistant', 'Two possibilities remain.'),
    ];
    const encoded = encodeMessageSequence(messages);
    expect(decodeMessageSequence(JSON.parse(JSON.stringify(encoded)))).toEqual(messages);
    expect(((encoded.messages[0]!.content.parts as Record<string, unknown>[])[1]!).data).toBe(Buffer.from([0, 255, ...Buffer.from('image')]).toString('base64'));
  });

  it('rejects corrupt image bytes', () => {
    const payload = encodeMessageSequence([new Message('user', [new ImagePart({ data: new TextEncoder().encode('image'), mediaType: 'image/png' })])]);
    ((payload.messages[0]!.content.parts as Record<string, unknown>[])[0]!).data = '%%%';
    expect(() => decodeMessageSequence(payload)).toThrow(/base64/);
  });

  it('rejects non-message values and malformed payloads', () => {
    expect(() => encodeMessageSequence('text')).toThrow(TypeError);
    expect(() => encodeMessageSequence([{ role: 'user' }])).toThrow(TypeError);
    expect(() => decodeMessageSequence({ format: 'other' })).toThrow(/invalid/);
    expect(() => decodeMessageSequence({ format: 'tensorcode-messages', version: 2, messages: [] })).toThrow(/unsupported/);
    expect(() => decodeMessageSequence({ format: 'tensorcode-messages', version: 1, messages: [{ role: 'user', content: { kind: 'video' } }] })).toThrow(/unsupported/);
  });
});
