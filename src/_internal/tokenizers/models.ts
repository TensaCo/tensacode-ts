/** Tokenization models from ``tokenizer.json`` (Rust ``tokenizers`` semantics). */
import { UnsupportedTokenizerError, type Json } from './pipeline.js';
import { utf8Bytes } from './unicode.js';

export interface TokenModel {
  /** Token strings for one pre-tokenized piece. */
  tokenize(piece: string): string[];
  tokenToId(token: string): number | undefined;
  /**
   * Id of a token this model's ``tokenize`` produced, when it differs from a
   * vocabulary lookup (Rust Unigram maps pieces outside the vocabulary to
   * ``unk_id`` while encoding, but ``token_to_id`` does not).
   */
  tokenizedId?(token: string): number | undefined;
  idToToken(id: number): string | undefined;
  readonly vocabSize: number;
}

function vocabularyFrom(vocab: Record<string, number>): { toId: Map<string, number>; toToken: Map<number, string> } {
  const toId = new Map<string, number>();
  const toToken = new Map<number, string>();
  for (const [token, id] of Object.entries(vocab)) {
    toId.set(token, id);
    toToken.set(id, token);
  }
  return { toId, toToken };
}

class WordPiece implements TokenModel {
  private readonly toId: Map<string, number>;
  private readonly toToken: Map<number, string>;
  private readonly unk: string;
  private readonly prefix: string;
  private readonly maxChars: number;

  constructor(spec: Json) {
    ({ toId: this.toId, toToken: this.toToken } = vocabularyFrom(spec.vocab as Record<string, number>));
    this.unk = String(spec.unk_token ?? '[UNK]');
    this.prefix = String(spec.continuing_subword_prefix ?? '##');
    this.maxChars = Number(spec.max_input_chars_per_word ?? 100);
  }

  get vocabSize(): number { return this.toId.size; }

  tokenize(piece: string): string[] {
    const chars = Array.from(piece);
    if (chars.length > this.maxChars) return [this.unk];
    const tokens: string[] = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let current: string | null = null;
      while (start < end) {
        let candidate = chars.slice(start, end).join('');
        if (start > 0) candidate = this.prefix + candidate;
        if (this.toId.has(candidate)) {
          current = candidate;
          break;
        }
        end -= 1;
      }
      if (current === null) return [this.unk];
      tokens.push(current);
      start = end;
    }
    return tokens;
  }

  tokenToId(token: string): number | undefined { return this.toId.get(token); }
  idToToken(id: number): string | undefined { return this.toToken.get(id); }
}

class WordLevel implements TokenModel {
  private readonly toId: Map<string, number>;
  private readonly toToken: Map<number, string>;
  private readonly unk: string;

  constructor(spec: Json) {
    ({ toId: this.toId, toToken: this.toToken } = vocabularyFrom(spec.vocab as Record<string, number>));
    this.unk = String(spec.unk_token ?? '<unk>');
  }

  get vocabSize(): number { return this.toId.size; }
  tokenize(piece: string): string[] { return [this.toId.has(piece) ? piece : this.unk]; }
  tokenToId(token: string): number | undefined { return this.toId.get(token); }
  idToToken(id: number): string | undefined { return this.toToken.get(id); }
}

class BPE implements TokenModel {
  private readonly toId: Map<string, number>;
  private readonly toToken: Map<number, string>;
  private readonly ranks = new Map<string, number>();
  private readonly unk: string | null;
  private readonly prefix: string;
  private readonly suffix: string;
  private readonly fuseUnk: boolean;
  private readonly byteFallback: boolean;
  private readonly ignoreMerges: boolean;
  private readonly cache = new Map<string, string[]>();

  constructor(spec: Json) {
    ({ toId: this.toId, toToken: this.toToken } = vocabularyFrom(spec.vocab as Record<string, number>));
    const merges = (spec.merges as (string | [string, string])[] | undefined) ?? [];
    merges.forEach((merge, rank) => {
      let pair: [string, string];
      if (typeof merge === 'string') {
        const index = merge.indexOf(' ', 1);
        pair = [merge.slice(0, index), merge.slice(index + 1)];
      } else pair = merge;
      const key = `${pair[0]}\u0000${pair[1]}`;
      if (!this.ranks.has(key)) this.ranks.set(key, rank);
    });
    this.unk = typeof spec.unk_token === 'string' ? spec.unk_token : null;
    this.prefix = typeof spec.continuing_subword_prefix === 'string' ? spec.continuing_subword_prefix : '';
    this.suffix = typeof spec.end_of_word_suffix === 'string' ? spec.end_of_word_suffix : '';
    this.fuseUnk = spec.fuse_unk === true;
    this.byteFallback = spec.byte_fallback === true;
    this.ignoreMerges = spec.ignore_merges === true;
    if (spec.dropout !== null && spec.dropout !== undefined && spec.dropout !== 0) {
      throw new UnsupportedTokenizerError('BPE dropout is not supported');
    }
  }

  get vocabSize(): number { return this.toId.size; }

  tokenize(piece: string): string[] {
    if (!piece) return [];
    const cached = this.cache.get(piece);
    if (cached) return cached;
    const result = this.merge(piece);
    if (this.cache.size < 100_000) this.cache.set(piece, result);
    return result;
  }

  private merge(piece: string): string[] {
    if (this.ignoreMerges && this.toId.has(piece)) return [piece];
    const chars = Array.from(piece);
    let symbols: string[] = [];
    let unkRun = false;
    chars.forEach((char, index) => {
      let symbol = char;
      if (index > 0) symbol = this.prefix + symbol;
      if (index === chars.length - 1) symbol += this.suffix;
      if (this.toId.has(symbol)) {
        symbols.push(symbol);
        unkRun = false;
        return;
      }
      if (this.byteFallback) {
        const bytes = Array.from(utf8Bytes(char), (byte) => `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`);
        if (bytes.every((token) => this.toId.has(token))) {
          symbols.push(...bytes);
          unkRun = false;
          return;
        }
      }
      if (this.unk === null) return; // Unknown characters are dropped without an unk token.
      if (this.fuseUnk && unkRun) return;
      symbols.push(this.unk);
      unkRun = true;
    });
    const unknown = this.unk;
    for (;;) {
      let best = Number.POSITIVE_INFINITY;
      let position = -1;
      for (let index = 0; index + 1 < symbols.length; index += 1) {
        const left = symbols[index]!;
        const right = symbols[index + 1]!;
        if (left === unknown || right === unknown) continue;
        const rank = this.ranks.get(`${left}\u0000${right}`);
        if (rank !== undefined && rank < best) {
          best = rank;
          position = index;
        }
      }
      if (position < 0) break;
      const right = symbols[position + 1]!;
      const merged = symbols[position]! + (this.prefix && right.startsWith(this.prefix) ? right.slice(this.prefix.length) : right);
      symbols = [...symbols.slice(0, position), merged, ...symbols.slice(position + 2)];
    }
    return symbols;
  }

  tokenToId(token: string): number | undefined { return this.toId.get(token); }
  idToToken(id: number): string | undefined { return this.toToken.get(id); }
}

interface TrieNode {
  children: Map<string, TrieNode>;
  id: number;
}

const UNK_PENALTY = 10;

class Unigram implements TokenModel {
  private readonly pieces: string[];
  private readonly scores: number[];
  private readonly toId = new Map<string, number>();
  private readonly root: TrieNode = { children: new Map(), id: -1 };
  private readonly unkId: number | null;
  private readonly minScore: number;
  private readonly byteFallback: boolean;
  private readonly fuseUnk: boolean;
  private readonly cache = new Map<string, string[]>();

  constructor(spec: Json) {
    const vocab = spec.vocab as [string, number][];
    this.pieces = vocab.map(([piece]) => piece);
    this.scores = vocab.map(([, score]) => score);
    this.unkId = typeof spec.unk_id === 'number' ? spec.unk_id : null;
    this.byteFallback = spec.byte_fallback === true;
    this.fuseUnk = spec.fuse_unk !== false;
    let minScore = Number.POSITIVE_INFINITY;
    vocab.forEach(([piece, score], id) => {
      // Rust fills a HashMap in vocabulary order: a repeated piece maps to its last id.
      this.toId.set(piece, id);
      if (score < minScore) minScore = score;
      let node = this.root;
      for (const char of piece) {
        let next = node.children.get(char);
        if (!next) {
          next = { children: new Map(), id: -1 };
          node.children.set(char, next);
        }
        node = next;
      }
      if (node.id < 0) node.id = id;
    });
    this.minScore = minScore;
  }

  get vocabSize(): number { return this.pieces.length; }

  tokenize(piece: string): string[] {
    if (!piece) return [];
    const cached = this.cache.get(piece);
    if (cached) return cached;
    const result = this.viterbi(piece);
    if (this.cache.size < 100_000) this.cache.set(piece, result);
    return result;
  }

  private viterbi(text: string): string[] {
    const chars = Array.from(text);
    const size = chars.length;
    const unkScore = this.minScore - UNK_PENALTY;
    const bestScore = new Float64Array(size + 1);
    const startsAt = new Int32Array(size + 1).fill(-1);
    const ids = new Int32Array(size + 1).fill(-1);
    for (let start = 0; start < size; start += 1) {
      if (start > 0 && startsAt[start] < 0) continue;
      const base = bestScore[start]!;
      let hasSingle = false;
      let node: TrieNode | undefined = this.root;
      for (let end = start; end < size; end += 1) {
        node = node.children.get(chars[end]!);
        if (!node) break;
        if (node.id >= 0) {
          const target = end + 1;
          const candidate = this.scores[node.id]! + base;
          if (startsAt[target]! < 0 || candidate > bestScore[target]!) {
            bestScore[target] = candidate;
            startsAt[target] = start;
            ids[target] = node.id;
          }
          if (target === start + 1) hasSingle = true;
        }
      }
      if (!hasSingle) {
        const target = start + 1;
        const candidate = unkScore + base;
        if (startsAt[target]! < 0 || candidate > bestScore[target]!) {
          bestScore[target] = candidate;
          startsAt[target] = start;
          ids[target] = this.unkId ?? -2;
        }
      }
    }
    const results: string[] = [];
    let fused: string[] = [];
    let end = size;
    while (end > 0) {
      const start = startsAt[end]!;
      const id = ids[end]!;
      const token = chars.slice(start, end).join('');
      if (this.fuseUnk && this.unkId !== null && id === this.unkId) {
        fused.push(token);
      } else {
        if (fused.length) {
          results.push(fused.reverse().join(''));
          fused = [];
        }
        results.push(token);
      }
      end = start;
    }
    if (fused.length) results.push(fused.reverse().join(''));
    results.reverse();
    if (!this.byteFallback) return results;
    return results.flatMap((token) => {
      if (this.toId.has(token)) return [token];
      const bytes = Array.from(utf8Bytes(token), (byte) => `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`);
      return bytes.every((item) => this.toId.has(item)) ? bytes : [token];
    });
  }

  /** Rust ``Unigram::token_to_id``: vocabulary pieces only. */
  tokenToId(token: string): number | undefined {
    return this.toId.get(token);
  }

  tokenizedId(token: string): number | undefined {
    return this.toId.get(token) ?? this.unkId ?? undefined;
  }

  idToToken(id: number): string | undefined { return this.pieces[id]; }
}

export function buildModel(spec: Json): TokenModel {
  const type = spec.type ?? (spec.continuing_subword_prefix !== undefined && spec.merges === undefined ? 'WordPiece' : undefined);
  switch (type) {
    case 'WordPiece': return new WordPiece(spec);
    case 'BPE': return new BPE(spec);
    case 'Unigram': return new Unigram(spec);
    case 'WordLevel': return new WordLevel(spec);
    default: throw new UnsupportedTokenizerError(`unsupported tokenizer model: ${JSON.stringify(spec.type)}`);
  }
}
