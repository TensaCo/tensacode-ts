/** TypeSafe Jev System One HTTP adapter based on its published OpenAPI schema (Python ``tensorcode/integrations/jev.py``). */
import { KeyError, ValueError } from '../errors.js';
import { isPlainObject } from '../_internal/json.js';
import { ImagePart, TextPart, type Message } from '../ops/text/messages.js';
import { ModelOutput, ModelRequest } from '../ops/text/model.js';
import { ProviderProtocolError, endpoint, postJson, postJsonSync, pythonRepr, validateTimeout, type FetchLike } from './http.js';

export interface JevModelOptions {
  apiKey: string;
  /** Default ``https://api.typesafe.ai``. */
  baseUrl?: string;
  /** Default ``jev-latest``. */
  model?: string;
  /** Request timeout in seconds (default 30). */
  timeout?: number;
  fetch?: FetchLike | null;
}

type Json = Record<string, unknown>;
type QuestionKind = 'noul' | 'choice' | 'score';

const SELECTION: Record<string, string> = { 'tensorcode.classify': 'label', 'tensorcode.decide': 'choice' };

function dig(value: unknown, ...path: string[]): unknown {
  let current = value;
  for (const key of path) {
    if (!isPlainObject(current) || !Object.hasOwn(current, key)) throw new KeyError(key);
    current = current[key];
  }
  return current;
}

/** Configured alternatives and optional descriptions from a selection schema. */
function alternatives(request: ModelRequest): [string[], Record<string, string | null>] {
  const field = SELECTION[request.schemaName!]!;
  try {
    const values = dig(request.responseSchema, 'properties', field, 'enum');
    if (!Array.isArray(values)) throw new KeyError('enum');
    const choices = values.filter((item) => item !== null) as string[];
    const described = dig(request.responseSchema, 'properties', 'distribution', 'properties');
    const descriptions: Record<string, string | null> = {};
    for (const alternative of choices) {
      const entry = dig(described, alternative);
      if (!isPlainObject(entry)) throw new KeyError(alternative);
      descriptions[alternative] = (entry.description as string | undefined) ?? null;
    }
    return [choices, descriptions];
  } catch (error) {
    throw new ProviderProtocolError('Selection schema is not compatible with Jev Choice', { cause: error });
  }
}

/** The Jev question for one request and its expected answer type. */
export function jevQuestion(request: ModelRequest): [Json, QuestionKind] {
  const name = request.schemaName;
  if (name !== null && Object.hasOwn(SELECTION, name)) {
    const [choices, descriptions] = alternatives(request);
    if (choices.length === 2 && choices.includes('true') && choices.includes('false')) {
      // Exactly the labels true/false map to Jev's yes/no probability.
      const question: Json = { type: 'noul', instructions: request.instructions };
      if (Object.values(descriptions).some(Boolean)) question.criteria = { true: descriptions.true, false: descriptions.false };
      return [question, 'noul'];
    }
    return [{ type: 'choice', instructions: request.instructions, criteria: descriptions }, 'choice'];
  }
  if (name === 'tensorcode.score') {
    let rubric: unknown[];
    try {
      const properties = dig(request.responseSchema, 'properties', 'distribution', 'properties');
      if (!isPlainObject(properties)) throw new KeyError('properties');
      rubric = Object.keys(properties).map((_, index) => dig(properties, String(index), 'description'));
    } catch (error) {
      throw new ProviderProtocolError('Score schema is not compatible with Jev Score', { cause: error });
    }
    return [{ type: 'score', instructions: request.instructions, criteria: rubric }, 'score'];
  }
  if (name === 'tensorcode.retrieve') throw new ProviderProtocolError('Jev adapter does not map multi-item retrieve requests');
  throw new ProviderProtocolError('Jev requires a supported structured decision request');
}

/** The documented Jev conversation state of a request's messages. */
export function jevState(request: ModelRequest): Json[] {
  return request.messages.map((message) => {
    if (typeof message.content === 'string') return { role: message.role, content: message.content };
    const content = message.content.map((part) => {
      if (part instanceof ImagePart) throw new ProviderProtocolError('Jev documented state does not establish image input support');
      if (!(part instanceof TextPart)) throw new ProviderProtocolError('Unsupported Jev message part');
      const entry: Json = { type: 'text', text: part.text };
      if (part.sourceRef !== null) entry.source_ref = part.sourceRef;
      return entry;
    });
    return { role: message.role, content };
  });
}

function canonical(request: ModelRequest, answer: Json): Json {
  const name = request.schemaName;
  if (name !== null && Object.hasOwn(SELECTION, name)) {
    const field = SELECTION[name]!;
    if (answer.type === 'noul') {
      const probability = answer.noul;
      if (typeof probability !== 'number' || !(probability >= 0 && probability <= 1)) {
        throw new ProviderProtocolError('Jev noul answer is not a probability');
      }
      return {
        [field]: probability >= 0.5 ? 'true' : 'false',
        distribution: { true: probability, false: 1 - probability },
        confidence: null,
        abstained: false,
      };
    }
    if (!['choice', 'probabilities', 'confidence'].every((key) => Object.hasOwn(answer, key))) {
      throw new ProviderProtocolError('Jev Choice answer is missing required fields');
    }
    return { [field]: answer.choice, distribution: answer.probabilities, confidence: answer.confidence, abstained: false };
  }
  if (name === 'tensorcode.score') {
    if (!['score', 'probabilities', 'confidence'].every((key) => Object.hasOwn(answer, key))) {
      throw new ProviderProtocolError('Jev Score answer is missing required fields');
    }
    return { score: answer.score, distribution: answer.probabilities, confidence: answer.confidence, abstained: false };
  }
  throw new ProviderProtocolError('Unsupported Jev result');
}

function bytesEqual(a: Uint8Array | null, b: Uint8Array | null): boolean {
  if (a === null || b === null) return a === b;
  return a.length === b.length && a.every((byte, index) => byte === b[index]);
}

function messagesEqual(a: readonly Message[], b: readonly Message[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((message, index) => {
    const other = b[index]!;
    if (message.role !== other.role) return false;
    if (typeof message.content === 'string' || typeof other.content === 'string') return message.content === other.content;
    const left = message.content;
    const right = other.content;
    return left.length === right.length && left.every((part, position) => {
      const peer = right[position]!;
      if (part instanceof TextPart) return peer instanceof TextPart && part.text === peer.text && part.sourceRef === peer.sourceRef;
      return peer instanceof ImagePart && part.url === peer.url && part.mediaType === peer.mediaType
        && part.sourceRef === peer.sourceRef && part.detail === peer.detail && bytesEqual(part.data, peer.data);
    });
  });
}

/**
 * Adapt TensorCode decision requests to ``POST /v1/systemone``.
 *
 * Jev is a typed evaluation model rather than a chat model. This adapter
 * supports classification, decisions and rubric scores; labels exactly
 * ``true``/``false`` use Jev's yes/no question. ``completeQuestions`` (and
 * ``acompleteQuestions``) sends several questions about the same messages in
 * one request. Unsupported
 * request shapes fail before making an HTTP request.
 */
export class JevModel {
  static readonly qualifiedName: string = 'tensorcode.integrations.jev.JevModel';
  readonly baseUrl: string;
  readonly model: string;
  readonly timeout: number;
  readonly #apiKey: string;
  readonly #fetch: FetchLike | null;

  constructor(options: JevModelOptions) {
    const { apiKey, baseUrl = 'https://api.typesafe.ai', model = 'jev-latest', timeout = 30 } = options;
    if (typeof apiKey !== 'string' || !apiKey) throw new ValueError('api_key must be a nonempty string');
    if (typeof model !== 'string' || !model) throw new ValueError('model must be a nonempty string');
    if (typeof baseUrl !== 'string') throw new TypeError('base_url must be a string');
    this.#apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.model = model;
    this.timeout = validateTimeout(timeout);
    this.#fetch = options.fetch ?? null;
    endpoint(this.baseUrl, 'v1/systemone');
  }

  toString(): string {
    return `JevModel(base_url=${pythonRepr(this.baseUrl)}, model=${pythonRepr(this.model)}, timeout=${this.timeout})`;
  }

  /** JSON description of this adapter; never includes credentials. */
  configuration(): Json {
    return { type: 'jev', base_url: this.baseUrl, model: this.model, timeout: this.timeout };
  }

  /** Send one request and return a ``ModelOutput`` (blocks). */
  complete(request: ModelRequest): ModelOutput {
    return this.completeQuestions({ result: request }).result!;
  }

  /** Asynchronous {@link complete}. */
  async acomplete(request: ModelRequest): Promise<ModelOutput> {
    return (await this.acompleteQuestions({ result: request })).result!;
  }

  /** Send named decision requests about identical messages in one call (blocks). */
  completeQuestions(requests: Readonly<Record<string, ModelRequest>>): Readonly<Record<string, ModelOutput>> {
    const [entries, questions, payload] = this.questions(requests, 'completeQuestions');
    const response = postJsonSync(endpoint(this.baseUrl, 'v1/systemone'), payload, { apiKey: this.#apiKey, timeout: this.timeout, fetch: this.#fetch });
    return this.answers(entries, questions, response);
  }

  /** Asynchronous {@link completeQuestions}. */
  async acompleteQuestions(requests: Readonly<Record<string, ModelRequest>>): Promise<Readonly<Record<string, ModelOutput>>> {
    const [entries, questions, payload] = this.questions(requests, 'acompleteQuestions');
    const response = await postJson(endpoint(this.baseUrl, 'v1/systemone'), payload, { apiKey: this.#apiKey, timeout: this.timeout, fetch: this.#fetch });
    return this.answers(entries, questions, response);
  }

  private questions(requests: Readonly<Record<string, ModelRequest>>, method: string): [[string, ModelRequest][], Record<string, [Json, QuestionKind]>, Json] {
    if (!isPlainObject(requests) || !Object.keys(requests).length) {
      throw new TypeError(`${method} expects a nonempty mapping of ModelRequest`);
    }
    const entries = Object.entries(requests);
    if (!entries.every(([, request]) => request instanceof ModelRequest)) throw new TypeError(`${method} expects ModelRequest values`);
    const first = entries[0]![1];
    if (entries.some(([, request]) => !messagesEqual(request.messages, first.messages))) {
      throw new ProviderProtocolError('Jev questions in one request must share the same messages');
    }
    const questions = Object.fromEntries(entries.map(([name, request]) => [name, jevQuestion(request)]));
    const payload = {
      state: jevState(first),
      model: this.model,
      questions: Object.fromEntries(Object.entries(questions).map(([name, [question]]) => [name, question])),
    };
    return [entries, questions, payload];
  }

  private answers(entries: [string, ModelRequest][], questions: Record<string, [Json, QuestionKind]>, response: Json): Readonly<Record<string, ModelOutput>> {
    if (!Object.hasOwn(response, 'answers')) throw new ProviderProtocolError('Jev response has no answers');
    const answers = response.answers;
    if (!isPlainObject(answers) || Object.keys(answers).length !== entries.length || !entries.every(([name]) => Object.hasOwn(answers, name))) {
      throw new ProviderProtocolError('Jev answers do not match the requested questions');
    }
    const metadata: Json = {};
    for (const key of ['model', 'usage']) if (Object.hasOwn(response, key)) metadata[key] = response[key];
    const providerMetadata = Object.keys(metadata).length ? metadata : null;
    const outputs: Record<string, ModelOutput> = {};
    for (const [name, request] of entries) {
      const answer = answers[name];
      if (!isPlainObject(answer) || answer.type !== questions[name]![1]) throw new ProviderProtocolError('Jev answer has the wrong type');
      outputs[name] = new ModelOutput({ structured: canonical(request, answer), providerMetadata });
    }
    return outputs;
  }
}
