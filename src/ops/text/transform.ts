/** Assistant replies from an owned native seq2seq model (Python ``tensorcode/ops/text/transform.py``). */
import type { Context } from '../base.js';
import type { JsonObject } from '../../_internal/json.js';
import { OwnedTextOperation } from '../../_internal/text/owned.js';
import { Message } from './messages.js';
import { ModelOutput, ModelRequest, isAsyncModel, isModel } from './model.js';
import { messageSequence, modelConfiguration } from './structured.js';

function isPromise(value: unknown): value is PromiseLike<unknown> {
  return value !== null && typeof value === 'object' && typeof (value as { then?: unknown }).then === 'function';
}

function messageTuple(value: unknown): readonly Message[] {
  if (value === null || value === undefined || typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] !== 'function'
    || typeof value === 'string') {
    throw new TypeError('Expected Message objects');
  }
  const messages = [...(value as Iterable<unknown>)];
  if (!messages.every((message) => message instanceof Message)) throw new TypeError('Expected Message objects');
  return messages as Message[];
}

function reply(messages: readonly Message[], answer: unknown): readonly Message[] {
  if (typeof answer !== 'string') throw new TypeError('Model must return a string');
  return Object.freeze([...messages, new Message('assistant', answer)]);
}

/**
 * Generate an assistant reply using an owned native seq2seq model; returns the
 * input messages followed by the reply.
 *
 * ``fromModel`` explicitly wraps an external provider (a ``Model``,
 * ``AsyncModel`` or ``messages → text`` function) without owned artifacts.
 */
export class Transform extends OwnedTextOperation<readonly Message[]> {
  static override readonly qualifiedName: string = 'tensorcode.ops.text.transform.Transform';

  _request(value: readonly Message[], context: Context | null): ModelRequest {
    return new ModelRequest(messageSequence(value, context), { instructions: this.instructions });
  }

  forward(value: readonly Message[], context: Context | null): readonly Message[] {
    const messages = messageTuple(value);
    // Context roles remain intact; context precedes the primary conversation.
    const combined = messageSequence(messages, context);
    let answer: unknown;
    if (isModel(this.model)) {
      const output = this.model.complete(this._request(messages, context));
      if (!(output instanceof ModelOutput)) throw new TypeError('model.complete must return ModelOutput');
      answer = output.text;
    } else if (typeof this.model === 'function') {
      answer = (this.model as (messages: readonly Message[]) => unknown)(combined);
      if (isPromise(answer)) throw new TypeError('The model returned a promise; use await operation.acall(...)');
    } else if (isAsyncModel(this.model)) {
      throw new TypeError('This model is asynchronous (acomplete only); use await operation.acall(...)');
    } else {
      throw new TypeError('Transform requires a model.complete(ModelRequest) method or a messages → text function');
    }
    return reply(messages, answer);
  }

  /** Asynchronous ``forward`` using ``model.acomplete`` (or an async function) when available. */
  override async aforward(value: readonly Message[], context: Context | null): Promise<readonly Message[]> {
    if (isAsyncModel(this.model)) {
      const messages = messageTuple(value);
      const output = await this.model.acomplete(this._request(messages, context));
      if (!(output instanceof ModelOutput) || typeof output.text !== 'string') {
        throw new TypeError('model.acomplete must return ModelOutput with text');
      }
      return reply(messages, output.text);
    }
    if (typeof this.model === 'function' && !isModel(this.model)) {
      const messages = messageTuple(value);
      const answer = await (this.model as (messages: readonly Message[]) => unknown)(messageSequence(messages, context));
      return reply(messages, answer);
    }
    return this.forward(value, context);
  }

  override configuration(): JsonObject {
    if (this._owned) return super.configuration();
    return { type: 'text_transform', instructions: this.instructions, model: modelConfiguration(this.model) as JsonObject };
  }
}
