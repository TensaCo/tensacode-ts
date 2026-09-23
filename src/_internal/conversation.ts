/**
 * Dialogue context helpers shared by ranking and proposal generation (Python
 * ``tensorcode/_internal/proposals.py``: ``conversation_context`` and
 * ``conversation_block``). Dialogue informs interpretation; it is never factual
 * evidence. FOUNDATION-OWNED.
 */
import { ValueError } from '../errors.js';
import { isPlainObject, pythonJsonDumps } from './json.js';

export interface ConversationRow {
  role: 'user' | 'assistant';
  text: string;
}

export function conversationContext(inputs: Record<string, unknown>): ConversationRow[] {
  const rows = inputs.conversation_context ?? [];
  if (!Array.isArray(rows)) throw new ValueError('conversation_context must be a list');
  for (const row of rows) {
    if (!isPlainObject(row) || Object.keys(row).length !== 2 || !('role' in row) || !('text' in row)
      || (row.role !== 'user' && row.role !== 'assistant') || typeof row.text !== 'string' || !row.text.trim()) {
      throw new ValueError('conversation context requires user/assistant role and nonempty text');
    }
  }
  // Copies keep the supplied key order, which the serialized block preserves.
  return rows.map((row) => ({ ...(row as ConversationRow) }));
}

export function conversationBlock(rows: readonly ConversationRow[]): string {
  if (!rows.length) return '';
  return 'Prior dialogue (context only; not source evidence; assistant statements are unverified):\n'
    + `${pythonJsonDumps(rows, { ensureAscii: false })}\n`;
}
