/**
 * Private conversation state shared with one owned Chatbot (Python
 * ``tensorcode/_internal/sessions/chat.py``). Conversation evidence is owned
 * by one session, never by a weight artifact.
 */
import { readFile } from 'node:fs/promises';
import { ValueError } from '../../errors.js';
import { isPlainObject, jsonEqual, parseJsonStrict, pythonJsonDumps, type JsonObject } from '../json.js';
import { atomicWriteFile } from '../files.js';
import { SerialQueue } from '../cognition/locking.js';
import { sessionFloatKeys } from './floats.js';
import { CognitiveSession } from '../cognition/session.js';
import type { CognitiveState } from '../cognition/state.js';
import type { Chatbot } from '../../tools/chatbot.js';

export interface ChatTurn {
  source_id: string;
  role: 'user' | 'assistant';
  text: string;
}

export class ChatSession {
  static readonly qualifiedName: string = 'tensorcode._internal.sessions.chat.ChatSession';
  readonly model: Chatbot;
  history: ChatTurn[] = [];
  lastResult: JsonObject | null = null;
  cognition: CognitiveSession | null;
  private readonly queue = new SerialQueue();

  constructor(model: Chatbot) {
    this.model = model;
    this.cognition = model.newCognitiveSessionForChat();
  }

  /** Reply to ``value`` and commit the turn only when every step succeeds. */
  call(value: unknown): string {
    return this.model.respond(value, this);
  }

  reset(): void {
    this.history = [];
    this.lastResult = null;
    this.cognition = this.model.newCognitiveSessionForChat();
  }

  newEpisode(): CognitiveState {
    if (this.cognition === null) throw new ValueError('Cognitive episodes are not configured');
    const state = this.cognition.newEpisode();
    this.history = [];
    this.lastResult = null;
    return state;
  }

  rebuildMemory(): void {
    if (this.cognition === null || this.cognition.memory === null) throw new ValueError('Episodic memory is not configured');
    this.cognition.memory.rebuildIndex();
  }

  /** The persisted session document (Python ``save`` payload). */
  toData(): JsonObject {
    const value: JsonObject = {
      format: this.cognition !== null ? 2 : 1, model: this.model.fingerprint,
      history: this.history.map((item) => ({ ...item })),
    };
    if (this.cognition !== null) value.cognition = this.cognition.snapshot() as unknown as JsonObject;
    return value;
  }

  async save(path: string): Promise<void> {
    const text = pythonJsonDumps(this.toData(), { ensureAscii: false, floatKeys: sessionFloatKeys() });
    await this.queue.run(() => atomicWriteFile(path, text));
  }

  async load(path: string): Promise<this> {
    return this.queue.run(async () => {
      const text = await readFile(path, 'utf8');
      let data: unknown;
      try {
        data = parseJsonStrict(text);
      } catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
        throw new ValueError(`invalid session JSON: ${error.message}`, { cause: error });
      }
      this.restore(data);
      return this;
    });
  }

  /** Validate a session document and commit it (transactional). */
  restore(value: unknown): this {
    const expectedFormat = this.cognition !== null ? 2 : 1;
    if (!isPlainObject(value) || value.format !== expectedFormat || value.model !== this.model.fingerprint) {
      throw new ValueError('Session is incompatible with this model configuration');
    }
    const history = value.history;
    if (!Array.isArray(history) || history.length > (this.model.config.max_turns as number) * 2) {
      throw new ValueError('Invalid session history capacity');
    }
    history.forEach((item, index) => {
      if (!isPlainObject(item) || Object.keys(item).length !== 3 || !['source_id', 'role', 'text'].every((key) => key in item)
        || item.role !== (index % 2 === 0 ? 'user' : 'assistant') || typeof item.text !== 'string' || typeof item.source_id !== 'string') {
        throw new ValueError('Invalid session evidence');
      }
    });
    if (history.length % 2) throw new ValueError('Session contains an incomplete turn');
    const ids = history.map((item) => {
      const text = (item.source_id as string).startsWith('turn-') ? (item.source_id as string).slice(5) : (item.source_id as string);
      if (!/^\s*[+-]?\d+\s*$/.test(text)) throw new ValueError('Invalid session source IDs');
      return Number(text.trim());
    });
    if (history.some((item, index) => item.source_id !== `turn-${ids[index]}` || ids[index]! < 0)
      || (ids.length && (ids[0]! % 2 || ids.some((id, index) => id !== ids[0]! + index)))) {
      throw new ValueError('Invalid session source IDs');
    }
    let cognitive: CognitiveSession | null = null;
    if (this.cognition !== null) {
      cognitive = CognitiveSession.fromSnapshot(value.cognition, { investigator: this.model.investigator! });
      if (cognitive.state.maxRecords !== this.cognition.state.maxRecords) {
        throw new ValueError('Session cognitive capacity differs from model configuration');
      }
      if (!jsonEqual(cognitive.snapshot().policy, this.cognition.snapshot().policy)) {
        throw new ValueError('Session cognitive policy differs from model configuration');
      }
    }
    this.history = history as unknown as ChatTurn[];
    this.cognition = cognitive;
    this.lastResult = null;
    return this;
  }
}
