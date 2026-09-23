/**
 * JSON data codec for public text message sequences (Python
 * ``tensorcode/_internal/memory/messages.py``). Image bytes are base64 text.
 */
import { ValueError } from '../../errors.js';
import { isPlainObject } from '../json.js';
import { ImagePart, Message, TextPart, type ImageDetail, type MessagePart, type MessageRole } from '../../ops/text/messages.js';

export interface EncodedMessages {
  format: 'tensorcode-messages';
  version: 1;
  messages: { role: string; content: Record<string, unknown> }[];
}

function toBase64(data: Uint8Array): string {
  return Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString('base64');
}

/** Python ``base64.b64decode(value, validate=True)``. */
function fromBase64(value: unknown): Uint8Array {
  if (typeof value !== 'string' || value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) {
    throw new ValueError('malformed base64 image data');
  }
  const buffer = Buffer.from(value, 'base64');
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength).slice();
}

export function encodeMessageSequence(value: unknown): EncodedMessages {
  if (typeof value === 'string' || value instanceof Uint8Array || !Array.isArray(value)) {
    throw new TypeError('message memory values must be Message sequences');
  }
  const messages = value.map((message) => {
    if (!(message instanceof Message)) throw new TypeError('message memory values must contain Message objects');
    let content: Record<string, unknown>;
    if (typeof message.content === 'string') {
      content = { kind: 'string', text: message.content };
    } else {
      const parts = message.content.map((part: MessagePart) => {
        if (part instanceof TextPart) return { kind: 'text', text: part.text, source_ref: part.sourceRef };
        if (part instanceof ImagePart) {
          return {
            kind: 'image', data: part.data !== null ? toBase64(part.data) : null, url: part.url,
            media_type: part.mediaType, source_ref: part.sourceRef, detail: part.detail,
          };
        }
        throw new TypeError('unsupported message part');
      });
      content = { kind: 'parts', parts };
    }
    return { role: message.role, content };
  });
  return { format: 'tensorcode-messages', version: 1, messages };
}

export function decodeMessageSequence(payload: unknown): readonly Message[] {
  if (!isPlainObject(payload) || payload.format !== 'tensorcode-messages') throw new ValueError('invalid message memory value');
  if (payload.version !== 1 || !Array.isArray(payload.messages)) throw new ValueError('unsupported or malformed message memory value');
  const messages = payload.messages.map((raw) => {
    if (!isPlainObject(raw)) throw new ValueError('malformed message memory entry');
    const content = raw.content;
    if (!isPlainObject(content)) throw new ValueError('malformed message memory content');
    let decoded: string | MessagePart[];
    if (content.kind === 'string') {
      decoded = content.text as string;
    } else if (content.kind === 'parts') {
      if (!Array.isArray(content.parts)) throw new ValueError('malformed message parts');
      decoded = content.parts.map((part) => {
        if (!isPlainObject(part)) throw new ValueError('malformed message part');
        if (part.kind === 'text') return new TextPart(part.text as string, { sourceRef: (part.source_ref as string | null) ?? null });
        if (part.kind === 'image') {
          const data = part.data === null || part.data === undefined ? null : fromBase64(part.data);
          return new ImagePart({
            data, url: (part.url as string | null) ?? null, mediaType: (part.media_type as string | null) ?? null,
            sourceRef: (part.source_ref as string | null) ?? null, detail: (part.detail as ImageDetail | null) ?? null,
          });
        }
        throw new ValueError('unsupported message part kind');
      });
    } else {
      throw new ValueError('unsupported message content kind');
    }
    return new Message(raw.role as MessageRole, decoded);
  });
  return Object.freeze(messages);
}
