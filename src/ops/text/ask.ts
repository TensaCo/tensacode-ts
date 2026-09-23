/** Ask several structured operations about the same messages (Python ``tensorcode/ops/text/ask.py``). */
import { isPlainObject } from '../../_internal/json.js';
import { activeSession } from '../../_internal/tracing.js';
import type { Context } from '../base.js';
import type { Message } from './messages.js';
import { ModelOutput, type ModelRequest } from './model.js';
import { InvalidModelOutput, StructuredOperation, requireStructured } from './structured.js';

/** Named structured questions (``Classify``, ``Decide``, ``Score`` or ``Retrieve``). */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Questions = Readonly<Record<string, StructuredOperation<any>>>;

/** Read-only answers keyed by question name. */
export type Answers<Q extends Questions> = Readonly<{ [K in keyof Q]: Q[K] extends StructuredOperation<infer R> ? R : never }>;

function validatedQuestions<Q extends Questions>(questions: Q): Q {
  if (!isPlainObject(questions) || !Object.keys(questions).length) {
    throw new TypeError('questions must be a nonempty mapping of names to structured operations');
  }
  for (const [name, operation] of Object.entries(questions)) {
    if (!name) throw new TypeError('question names must be nonempty strings');
    if (!(operation instanceof StructuredOperation)) throw new TypeError('questions must be Classify, Decide, Score or Retrieve operations');
  }
  return questions;
}

/** The single external model that can answer every question at once, if any. */
function sharedModel(questions: Questions, method: string): Record<string, unknown> | null {
  const operations = Object.values(questions);
  const models = new Set(operations.map((operation) => operation.model));
  if (models.size !== 1 || operations.some((operation) => operation._owned)) return null;
  const [model] = models;
  return model !== null && typeof model === 'object' && typeof (model as Record<string, unknown>)[method] === 'function'
    ? model as Record<string, unknown>
    : null;
}

function results<Q extends Questions>(questions: Q, outputs: unknown): Answers<Q> {
  const names = Object.keys(questions);
  if (!isPlainObject(outputs) || Object.keys(outputs).length !== names.length || !names.every((name) => Object.hasOwn(outputs, name))) {
    throw new InvalidModelOutput('Model answers must match the question names');
  }
  if (!Object.values(outputs).every((output) => output instanceof ModelOutput)) {
    throw new TypeError('model.completeQuestions must return ModelOutput values');
  }
  const answers: Record<string, unknown> = {};
  for (const [name, operation] of Object.entries(questions)) {
    answers[name] = operation._parse(requireStructured(outputs[name] as ModelOutput));
  }
  return Object.freeze(answers) as Answers<Q>;
}

function requests(questions: Questions, messages: readonly Message[], context: Context | null): Record<string, ModelRequest> {
  const result: Record<string, ModelRequest> = {};
  for (const [name, operation] of Object.entries(questions)) result[name] = operation._request(messages, context);
  return result;
}

export interface AskOptions {
  context?: Context | null;
}

/**
 * Answer named structured ``questions`` about one message sequence.
 *
 * When every question wraps the same external model and that model implements
 * ``completeQuestions``, all questions travel in one request. Otherwise, and
 * whenever a trace is active, each operation is called normally. Returns a
 * read-only mapping from question name to that operation's result.
 */
export function ask<Q extends Questions>(messages: Iterable<Message>, questions: Q, options: AskOptions = {}): Answers<Q> {
  validatedQuestions(questions);
  const values = Object.freeze([...messages]);
  const context = options.context ?? null;
  const model = activeSession() !== null ? null : sharedModel(questions, 'completeQuestions');
  if (model === null) {
    const answers: Record<string, unknown> = {};
    for (const [name, operation] of Object.entries(questions)) answers[name] = operation.call(values, { context });
    return Object.freeze(answers) as Answers<Q>;
  }
  const complete = model.completeQuestions as (requests: Record<string, ModelRequest>) => unknown;
  return results(questions, complete.call(model, requests(questions, values, context)));
}

/** Asynchronous {@link ask}; uses ``acompleteQuestions`` when available. */
export async function aask<Q extends Questions>(messages: Iterable<Message>, questions: Q, options: AskOptions = {}): Promise<Answers<Q>> {
  validatedQuestions(questions);
  const values = Object.freeze([...messages]);
  const context = options.context ?? null;
  const model = activeSession() !== null ? null : sharedModel(questions, 'acompleteQuestions');
  if (model === null) {
    const names = Object.keys(questions);
    const answers = await Promise.all(names.map((name) => questions[name]!.acall(values, { context })));
    return Object.freeze(Object.fromEntries(names.map((name, index) => [name, answers[index]]))) as Answers<Q>;
  }
  const complete = model.acompleteQuestions as (requests: Record<string, ModelRequest>) => Promise<unknown>;
  return results(questions, await complete.call(model, requests(questions, values, context)));
}
