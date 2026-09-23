/** Pure message serialization (Python ``tensorcode/ops/text/encode.py``). */
import { ConfigOperation } from '../../_internal/operationConfig.js';
import { validatedConfig } from '../../_internal/operationConfig.js';
import { ValueError } from '../../errors.js';
import type { JsonObject } from '../../_internal/json.js';
import type { Context } from '../base.js';
import { ImagePart, Message, type ImageDetail, type MessagePart } from './messages.js';

/** Serialize a string as one user message without interpreting it. */
export class TextEncoder extends ConfigOperation<string | Iterable<MessagePart>, readonly Message[]> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text.encode.TextEncoder';

  constructor(config: unknown = null) {
    super(config, []);
  }

  override get replayable(): boolean {
    return true;
  }

  /** A one-message tuple wrapping the input; context is rejected. */
  forward(value: string | Iterable<MessagePart>, context: Context | null): readonly Message[] {
    if (context) throw new ValueError('TextEncoder only serializes text; context belongs on a transform');
    return Object.freeze([new Message('user', value)]);
  }
}

const IMAGE_KEYS = ['media_type', 'source_ref', 'detail'];

/** Serialize image bytes or a URL without interpreting or fetching it. */
export class ImageEncoder extends ConfigOperation<ImagePart | Uint8Array | string, readonly Message[]> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text.encode.ImageEncoder';
  mediaType: string | null;
  sourceRef: string | null;
  detail: ImageDetail | null;

  constructor(config: unknown = null) {
    super(config, IMAGE_KEYS, { media_type: null, source_ref: null, detail: null });
    this.mediaType = this.config.media_type as string | null;
    this.sourceRef = this.config.source_ref as string | null;
    this.detail = this.config.detail as ImageDetail | null;
    // Validate the settings exactly as ImagePart would.
    new ImagePart({ url: 'https://example.invalid/image', mediaType: this.mediaType, sourceRef: this.sourceRef, detail: this.detail });
  }

  override get replayable(): boolean {
    return true;
  }

  /** JSON configuration that reconstructs this operation (current settings). */
  override configuration(): JsonObject {
    return validatedConfig({ media_type: this.mediaType, source_ref: this.sourceRef, detail: this.detail }, IMAGE_KEYS);
  }

  /** A one-message tuple wrapping the image; context is rejected. */
  forward(value: ImagePart | Uint8Array | string, context: Context | null): readonly Message[] {
    if (context) throw new ValueError('ImageEncoder only serializes an image; context belongs on a transform');
    let part: ImagePart;
    if (value instanceof ImagePart) {
      // An explicit part is authoritative; encoder defaults apply only to raw
      // bytes/URLs and never overwrite source metadata.
      part = value;
    } else if (value instanceof Uint8Array) {
      part = new ImagePart({ data: value, mediaType: this.mediaType, sourceRef: this.sourceRef, detail: this.detail });
    } else if (typeof value === 'string') {
      part = new ImagePart({ url: value, mediaType: this.mediaType, sourceRef: this.sourceRef, detail: this.detail });
    } else {
      throw new TypeError('ImageEncoder expects ImagePart, bytes or an image URL');
    }
    return Object.freeze([new Message('user', [part])]);
  }
}
