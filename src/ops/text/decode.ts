/** Assistant text decoding (Python ``tensorcode/ops/text/decode.py``). */
import { ConfigOperation } from '../../_internal/operationConfig.js';
import { ValueError } from '../../errors.js';
import type { Context } from '../base.js';
import { ImagePart, TextPart, type Message } from './messages.js';

/** Return the text of a final assistant message; image content is rejected. */
export class TextDecoder extends ConfigOperation<readonly Message[], string> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text.decode.TextDecoder';

  constructor(config: unknown = null) {
    super(config, []);
  }

  override get replayable(): boolean {
    return true;
  }

  forward(value: readonly Message[], context: Context | null): string {
    if (context) throw new ValueError('TextDecoder does not consume context');
    const last = value?.length ? value[value.length - 1] : undefined;
    if (!last || last.role !== 'assistant') throw new ValueError('Expected a final assistant message');
    const content = last.content;
    if (typeof content === 'string') return content;
    if (content.some((part) => part instanceof ImagePart)) {
      throw new ValueError('Assistant message contains an image and cannot be decoded as text');
    }
    return content.filter((part): part is TextPart => part instanceof TextPart).map((part) => part.text).join('');
  }
}
