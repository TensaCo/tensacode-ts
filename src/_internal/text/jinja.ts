/**
 * A sandboxed Jinja2 subset for Hugging Face chat templates, rendered the way
 * transformers does (``ImmutableSandboxedEnvironment(trim_blocks=True,
 * lstrip_blocks=True)`` with ``loopcontrols``, ``raise_exception``,
 * ``namespace`` and ``tojson``).
 *
 * Supported: text, ``{{ }}``, ``{# #}``, whitespace control (``-``),
 * ``for``/``else``/``break``/``continue`` (with ``loop``), ``if``/``elif``/``else``,
 * ``set`` (including ``namespace`` attributes), ``generation`` blocks, macros,
 * Python-style expressions (arithmetic, comparisons, ``in``, ``is`` tests,
 * conditional expressions, slicing, filters and string/dict methods).
 * Unsupported syntax raises a {@link TemplateError}; templates never execute
 * host code.
 */

export class TemplateError extends Error {
  override name = 'TemplateError';
}

/** Jinja ``Undefined``: renders as ``''``, iterates as empty, is falsy. */
class Undefined {
  constructor(readonly hint: string) {}
}

class Namespace {
  constructor(readonly values: Map<string, unknown>) {}
}

class Macro {
  constructor(readonly params: { name: string; fallback: Expr | null }[], readonly body: Node[], readonly scope: Scope) {}
}

type Value = unknown;

// ---------------------------------------------------------------------------
// Lexing.
// ---------------------------------------------------------------------------

type Segment =
  | { kind: 'text'; value: string }
  | { kind: 'output'; source: string }
  | { kind: 'statement'; source: string };

function lex(template: string): Segment[] {
  const segments: Segment[] = [];
  let position = 0;
  let trimNextNewline = false;
  let stripNextWhitespace = false;
  const pushText = (text: string): void => {
    let value = text;
    if (stripNextWhitespace) value = value.replace(/^\s+/, '');
    else if (trimNextNewline && value.startsWith('\n')) value = value.slice(1);
    else if (trimNextNewline && value.startsWith('\r\n')) value = value.slice(2);
    stripNextWhitespace = false;
    trimNextNewline = false;
    if (value) segments.push({ kind: 'text', value });
  };
  while (position < template.length) {
    const next = template.slice(position).search(/\{[{%#]/);
    if (next < 0) {
      pushText(template.slice(position));
      break;
    }
    const start = position + next;
    const opener = template[start + 1]!;
    const closer = opener === '{' ? '}}' : opener === '%' ? '%}' : '#}';
    let inner = start + 2;
    const stripBefore = template[inner] === '-';
    const keepBefore = template[inner] === '+';
    if (stripBefore || keepBefore) inner += 1;
    const end = findTagEnd(template, inner, closer);
    let body = template.slice(inner, end);
    const stripAfter = body.endsWith('-');
    if (stripAfter || body.endsWith('+')) body = body.slice(0, -1);
    let text = template.slice(position, start);
    const block = opener !== '{';
    if (stripBefore) text = text.replace(/\s+$/, '');
    else if (block && !keepBefore) {
      // lstrip_blocks: remove spaces/tabs between a line start and a block tag.
      const lineStart = text.lastIndexOf('\n');
      const tail = text.slice(lineStart + 1);
      if (/^[ \t]*$/.test(tail) && (lineStart >= 0 || isAtLineStart(template, position))) text = text.slice(0, lineStart + 1);
    }
    pushText(text);
    if (opener === '{') segments.push({ kind: 'output', source: body.trim() });
    else if (opener === '%') segments.push({ kind: 'statement', source: body.trim() });
    position = end + 2;
    if (stripAfter) stripNextWhitespace = true;
    else if (block) trimNextNewline = true; // trim_blocks
  }
  return segments;
}

function isAtLineStart(template: string, position: number): boolean {
  return position === 0 || template[position - 1] === '\n';
}

function findTagEnd(template: string, from: number, closer: string): number {
  let quote: string | null = null;
  for (let index = from; index < template.length; index += 1) {
    const char = template[index]!;
    if (quote) {
      if (char === '\\') index += 1;
      else if (char === quote) quote = null;
      continue;
    }
    if (closer !== '#}' && (char === '"' || char === "'")) quote = char;
    else if (template.startsWith(closer, index)) {
      return index;
    }
  }
  throw new TemplateError(`unterminated template tag ${closer}`);
}

// ---------------------------------------------------------------------------
// Expressions.
// ---------------------------------------------------------------------------

type Expr =
  | { type: 'literal'; value: Value }
  | { type: 'name'; name: string }
  | { type: 'list'; items: Expr[] }
  | { type: 'tuple'; items: Expr[] }
  | { type: 'dict'; entries: [Expr, Expr][] }
  | { type: 'attr'; target: Expr; name: string }
  | { type: 'item'; target: Expr; key: Expr }
  | { type: 'slice'; target: Expr; start: Expr | null; stop: Expr | null; step: Expr | null }
  | { type: 'call'; callee: Expr; args: Expr[]; kwargs: [string, Expr][] }
  | { type: 'filter'; target: Expr; name: string; args: Expr[]; kwargs: [string, Expr][] }
  | { type: 'test'; target: Expr; name: string; args: Expr[]; negated: boolean }
  | { type: 'unary'; op: string; operand: Expr }
  | { type: 'binary'; op: string; left: Expr; right: Expr }
  | { type: 'conditional'; test: Expr; yes: Expr; no: Expr | null };

type Token = { kind: 'name' | 'string' | 'number' | 'op'; value: string };

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let index = 0;
  while (index < source.length) {
    const char = source[index]!;
    if (/\s/.test(char)) { index += 1; continue; }
    if (char === '"' || char === "'") {
      let value = '';
      index += 1;
      while (index < source.length && source[index] !== char) {
        if (source[index] === '\\') {
          const escaped = source[index + 1]!;
          value += ({ n: '\n', t: '\t', r: '\r', '\\': '\\', "'": "'", '"': '"' } as Record<string, string>)[escaped] ?? `\\${escaped}`;
          index += 2;
        } else {
          value += source[index];
          index += 1;
        }
      }
      if (index >= source.length) throw new TemplateError('unterminated string literal');
      index += 1;
      tokens.push({ kind: 'string', value });
      continue;
    }
    const number = /^\d+(\.\d+)?([eE][+-]?\d+)?/.exec(source.slice(index));
    if (number) {
      tokens.push({ kind: 'number', value: number[0] });
      index += number[0].length;
      continue;
    }
    const name = /^[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(index));
    if (name) {
      tokens.push({ kind: 'name', value: name[0] });
      index += name[0].length;
      continue;
    }
    const op = ['//', '**', '==', '!=', '<=', '>='].find((candidate) => source.startsWith(candidate, index)) ?? char;
    if (!'//**==!=<=>=+-*/%~<>()[]{}.,:|='.includes(op[0]!)) throw new TemplateError(`unexpected character ${JSON.stringify(char)}`);
    tokens.push({ kind: 'op', value: op });
    index += op.length;
  }
  return tokens;
}

class Parser {
  private index = 0;

  constructor(private readonly tokens: Token[]) {}

  get done(): boolean {
    return this.index >= this.tokens.length;
  }

  peek(offset = 0): Token | undefined {
    return this.tokens[this.index + offset];
  }

  isOp(value: string): boolean {
    const token = this.peek();
    return token?.kind === 'op' && token.value === value;
  }

  isName(value: string): boolean {
    const token = this.peek();
    return token?.kind === 'name' && token.value === value;
  }

  take(): Token {
    const token = this.tokens[this.index];
    if (!token) throw new TemplateError('unexpected end of expression');
    this.index += 1;
    return token;
  }

  expectOp(value: string): void {
    const token = this.take();
    if (token.kind !== 'op' || token.value !== value) throw new TemplateError(`expected ${value}, found ${token.value}`);
  }

  expectName(value?: string): string {
    const token = this.take();
    if (token.kind !== 'name' || (value !== undefined && token.value !== value)) throw new TemplateError(`expected ${value ?? 'a name'}, found ${token.value}`);
    return token.value;
  }

  /** A full expression, allowing an unparenthesized tuple. */
  parseTuple(): Expr {
    const first = this.parseExpression();
    if (!this.isOp(',')) return first;
    const items = [first];
    while (this.isOp(',')) {
      this.take();
      if (this.done || this.isOp(')')) break;
      items.push(this.parseExpression());
    }
    return { type: 'tuple', items };
  }

  parseExpression(): Expr {
    const value = this.parseOr();
    if (this.isName('if')) {
      this.take();
      const test = this.parseOr();
      let no: Expr | null = null;
      if (this.isName('else')) {
        this.take();
        no = this.parseExpression();
      }
      return { type: 'conditional', test, yes: value, no };
    }
    return value;
  }

  private parseOr(): Expr {
    let left = this.parseAnd();
    while (this.isName('or')) {
      this.take();
      left = { type: 'binary', op: 'or', left, right: this.parseAnd() };
    }
    return left;
  }

  private parseAnd(): Expr {
    let left = this.parseNot();
    while (this.isName('and')) {
      this.take();
      left = { type: 'binary', op: 'and', left, right: this.parseNot() };
    }
    return left;
  }

  private parseNot(): Expr {
    if (this.isName('not')) {
      this.take();
      return { type: 'unary', op: 'not', operand: this.parseNot() };
    }
    return this.parseComparison();
  }

  private parseComparison(): Expr {
    let left = this.parseConcat();
    for (;;) {
      const token = this.peek();
      if (token?.kind === 'op' && ['==', '!=', '<', '>', '<=', '>='].includes(token.value)) {
        this.take();
        left = { type: 'binary', op: token.value, left, right: this.parseConcat() };
      } else if (this.isName('in')) {
        this.take();
        left = { type: 'binary', op: 'in', left, right: this.parseConcat() };
      } else if (this.isName('not') && this.peek(1)?.kind === 'name' && this.peek(1)?.value === 'in') {
        this.take();
        this.take();
        left = { type: 'binary', op: 'not in', left, right: this.parseConcat() };
      } else if (this.isName('is')) {
        this.take();
        let negated = false;
        if (this.isName('not')) {
          this.take();
          negated = true;
        }
        const name = this.expectName();
        const args: Expr[] = [];
        if (this.isOp('(')) {
          this.take();
          while (!this.isOp(')')) {
            args.push(this.parseExpression());
            if (this.isOp(',')) this.take();
          }
          this.take();
        } else if (!this.done && !this.isOp(')') && !this.isName('and') && !this.isName('or') && !this.isName('else')
          && !this.isName('if') && !this.isOp(',') && !this.isOp(']') && !this.isOp('}') && !this.isOp(':')) {
          args.push(this.parseConcat());
        }
        left = { type: 'test', target: left, name, args, negated };
      } else {
        return left;
      }
    }
  }

  private parseConcat(): Expr {
    let left = this.parseAdditive();
    while (this.isOp('~')) {
      this.take();
      left = { type: 'binary', op: '~', left, right: this.parseAdditive() };
    }
    return left;
  }

  private parseAdditive(): Expr {
    let left = this.parseMultiplicative();
    while (this.isOp('+') || this.isOp('-')) {
      const op = this.take().value;
      left = { type: 'binary', op, left, right: this.parseMultiplicative() };
    }
    return left;
  }

  private parseMultiplicative(): Expr {
    let left = this.parseUnary();
    while (this.isOp('*') || this.isOp('/') || this.isOp('//') || this.isOp('%')) {
      const op = this.take().value;
      left = { type: 'binary', op, left, right: this.parseUnary() };
    }
    return left;
  }

  private parseUnary(): Expr {
    if (this.isOp('-') || this.isOp('+')) {
      const op = this.take().value;
      return { type: 'unary', op, operand: this.parseUnary() };
    }
    return this.parsePower();
  }

  private parsePower(): Expr {
    const base = this.parsePostfix(this.parsePrimary());
    if (this.isOp('**')) {
      this.take();
      return { type: 'binary', op: '**', left: base, right: this.parseUnary() };
    }
    return base;
  }

  private parseArguments(): [Expr[], [string, Expr][]] {
    const args: Expr[] = [];
    const kwargs: [string, Expr][] = [];
    this.expectOp('(');
    while (!this.isOp(')')) {
      const token = this.peek();
      const following = this.peek(1);
      if (token?.kind === 'name' && following?.kind === 'op' && following.value === '=') {
        this.take();
        this.take();
        kwargs.push([token.value, this.parseExpression()]);
      } else {
        args.push(this.parseExpression());
      }
      if (this.isOp(',')) this.take();
      else if (!this.isOp(')')) throw new TemplateError('expected , or ) in argument list');
    }
    this.take();
    return [args, kwargs];
  }

  private parsePostfix(initial: Expr): Expr {
    let expr = initial;
    for (;;) {
      if (this.isOp('.')) {
        this.take();
        const token = this.take();
        if (token.kind === 'number') expr = { type: 'item', target: expr, key: { type: 'literal', value: Number(token.value) } };
        else expr = { type: 'attr', target: expr, name: token.value };
      } else if (this.isOp('[')) {
        this.take();
        let start: Expr | null = null;
        if (!this.isOp(':')) start = this.parseExpression();
        if (this.isOp(':')) {
          this.take();
          const stop = this.isOp(']') || this.isOp(':') ? null : this.parseExpression();
          let step: Expr | null = null;
          if (this.isOp(':')) {
            this.take();
            if (!this.isOp(']')) step = this.parseExpression();
          }
          this.expectOp(']');
          expr = { type: 'slice', target: expr, start, stop, step };
        } else {
          this.expectOp(']');
          expr = { type: 'item', target: expr, key: start! };
        }
      } else if (this.isOp('(')) {
        const [args, kwargs] = this.parseArguments();
        expr = { type: 'call', callee: expr, args, kwargs };
      } else if (this.isOp('|')) {
        this.take();
        const name = this.expectName();
        let args: Expr[] = [];
        let kwargs: [string, Expr][] = [];
        if (this.isOp('(')) [args, kwargs] = this.parseArguments();
        expr = { type: 'filter', target: expr, name, args, kwargs };
      } else {
        return expr;
      }
    }
  }

  private parsePrimary(): Expr {
    const token = this.take();
    if (token.kind === 'string') {
      let value = token.value;
      while (this.peek()?.kind === 'string') value += this.take().value; // implicit concatenation
      return { type: 'literal', value };
    }
    if (token.kind === 'number') return { type: 'literal', value: /[.eE]/.test(token.value) ? new PyFloat(Number(token.value)) : Number(token.value) };
    if (token.kind === 'name') {
      if (token.value === 'true' || token.value === 'True') return { type: 'literal', value: true };
      if (token.value === 'false' || token.value === 'False') return { type: 'literal', value: false };
      if (token.value === 'none' || token.value === 'None') return { type: 'literal', value: null };
      return { type: 'name', name: token.value };
    }
    if (token.value === '(') {
      if (this.isOp(')')) {
        this.take();
        return { type: 'tuple', items: [] };
      }
      const inner = this.parseTuple();
      this.expectOp(')');
      return inner;
    }
    if (token.value === '[') {
      const items: Expr[] = [];
      while (!this.isOp(']')) {
        items.push(this.parseExpression());
        if (this.isOp(',')) this.take();
      }
      this.take();
      return { type: 'list', items };
    }
    if (token.value === '{') {
      const entries: [Expr, Expr][] = [];
      while (!this.isOp('}')) {
        const key = this.parseExpression();
        this.expectOp(':');
        entries.push([key, this.parseExpression()]);
        if (this.isOp(',')) this.take();
      }
      this.take();
      return { type: 'dict', entries };
    }
    throw new TemplateError(`unexpected token ${token.value}`);
  }
}

/** A Python ``float`` literal (keeps ``1.0`` distinct from ``1`` when rendered). */
class PyFloat {
  constructor(readonly value: number) {}
}

function parseExpression(source: string): Expr {
  const parser = new Parser(tokenize(source));
  const expr = parser.parseTuple();
  if (!parser.done) throw new TemplateError(`unexpected ${parser.peek()!.value} in expression ${JSON.stringify(source)}`);
  return expr;
}

// ---------------------------------------------------------------------------
// Statements.
// ---------------------------------------------------------------------------

type Node =
  | { type: 'text'; value: string }
  | { type: 'output'; expr: Expr }
  | { type: 'if'; branches: [Expr, Node[]][]; otherwise: Node[] }
  | { type: 'for'; targets: string[]; iterable: Expr; filter: Expr | null; body: Node[]; otherwise: Node[] }
  | { type: 'set'; target: string; attribute: string | null; value: Expr | null; body: Node[] | null }
  | { type: 'macro'; name: string; params: { name: string; fallback: Expr | null }[]; body: Node[] }
  | { type: 'break' }
  | { type: 'continue' };

function statementHead(source: string): [string, string] {
  const match = /^([A-Za-z_]+)\s*([\s\S]*)$/.exec(source);
  if (!match) throw new TemplateError(`invalid statement ${JSON.stringify(source)}`);
  return [match[1]!, match[2]!];
}

function parseTemplate(segments: Segment[]): Node[] {
  let index = 0;
  const parseBlock = (terminators: string[]): [Node[], string | null, string] => {
    const nodes: Node[] = [];
    while (index < segments.length) {
      const segment = segments[index]!;
      index += 1;
      if (segment.kind === 'text') {
        nodes.push({ type: 'text', value: segment.value });
        continue;
      }
      if (segment.kind === 'output') {
        nodes.push({ type: 'output', expr: parseExpression(segment.source) });
        continue;
      }
      const [keyword, rest] = statementHead(segment.source);
      if (terminators.includes(keyword)) return [nodes, keyword, rest];
      switch (keyword) {
        case 'if': {
          const branches: [Expr, Node[]][] = [];
          let condition = parseExpression(rest);
          let otherwise: Node[] = [];
          for (;;) {
            const [body, end, tail] = parseBlock(['elif', 'else', 'endif']);
            branches.push([condition, body]);
            if (end === 'elif') {
              condition = parseExpression(tail);
              continue;
            }
            if (end === 'else') {
              const [elseBody, close] = parseBlock(['endif']);
              if (close !== 'endif') throw new TemplateError('missing endif');
              otherwise = elseBody;
            } else if (end !== 'endif') {
              throw new TemplateError('missing endif');
            }
            break;
          }
          nodes.push({ type: 'if', branches, otherwise });
          break;
        }
        case 'for': {
          const match = /^([\s\S]+?)\s+in\s+([\s\S]+)$/.exec(rest);
          if (!match) throw new TemplateError(`invalid for statement ${JSON.stringify(rest)}`);
          const targets = match[1]!.replace(/[()]/g, '').split(',').map((item) => item.trim()).filter(Boolean);
          let iterableSource = match[2]!;
          let filter: Expr | null = null;
          const filtered = /^([\s\S]+?)\s+if\s+([\s\S]+)$/.exec(iterableSource);
          if (filtered && !/\belse\b/.test(filtered[2]!)) {
            iterableSource = filtered[1]!;
            filter = parseExpression(filtered[2]!);
          }
          if (/\s+recursive\s*$/.test(iterableSource)) throw new TemplateError('recursive loops are not supported');
          const [body, end] = parseBlock(['else', 'endfor']);
          let otherwise: Node[] = [];
          if (end === 'else') {
            const [elseBody, close] = parseBlock(['endfor']);
            if (close !== 'endfor') throw new TemplateError('missing endfor');
            otherwise = elseBody;
          } else if (end !== 'endfor') {
            throw new TemplateError('missing endfor');
          }
          nodes.push({ type: 'for', targets, iterable: parseExpression(iterableSource), filter, body, otherwise });
          break;
        }
        case 'set': {
          const assignment = /^([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\s*(?:=\s*([\s\S]+))?$/.exec(rest);
          if (!assignment) throw new TemplateError(`invalid set statement ${JSON.stringify(rest)}`);
          if (assignment[3] === undefined) {
            const [body, end] = parseBlock(['endset']);
            if (end !== 'endset') throw new TemplateError('missing endset');
            nodes.push({ type: 'set', target: assignment[1]!, attribute: assignment[2] ?? null, value: null, body });
          } else {
            nodes.push({ type: 'set', target: assignment[1]!, attribute: assignment[2] ?? null, value: parseExpression(assignment[3]), body: null });
          }
          break;
        }
        case 'macro': {
          const signature = /^([A-Za-z_][A-Za-z0-9_]*)\s*\(([\s\S]*)\)$/.exec(rest);
          if (!signature) throw new TemplateError(`invalid macro ${JSON.stringify(rest)}`);
          const params = signature[2]!.split(',').map((item) => item.trim()).filter(Boolean).map((item) => {
            const [name, fallback] = item.split('=').map((part) => part.trim());
            return { name: name!, fallback: fallback === undefined ? null : parseExpression(fallback) };
          });
          const [body, end] = parseBlock(['endmacro']);
          if (end !== 'endmacro') throw new TemplateError('missing endmacro');
          nodes.push({ type: 'macro', name: signature[1]!, params, body });
          break;
        }
        case 'generation': {
          const [body, end] = parseBlock(['endgeneration']);
          if (end !== 'endgeneration') throw new TemplateError('missing endgeneration');
          nodes.push(...body);
          break;
        }
        case 'break':
          nodes.push({ type: 'break' });
          break;
        case 'continue':
          nodes.push({ type: 'continue' });
          break;
        default:
          throw new TemplateError(`unsupported template statement ${JSON.stringify(keyword)}`);
      }
    }
    return [nodes, null, ''];
  };
  const [nodes, end] = parseBlock([]);
  if (end !== null) throw new TemplateError(`unexpected ${end}`);
  return nodes;
}

// ---------------------------------------------------------------------------
// Evaluation.
// ---------------------------------------------------------------------------

class Scope {
  private readonly values = new Map<string, Value>();

  constructor(private readonly parent: Scope | null) {}

  get(name: string): Value {
    if (this.values.has(name)) return this.values.get(name);
    return this.parent ? this.parent.get(name) : new Undefined(name);
  }

  set(name: string, value: Value): void {
    this.values.set(name, value);
  }

  child(): Scope {
    return new Scope(this);
  }
}

class LoopControl {
  constructor(readonly kind: 'break' | 'continue') {}
}

function isUndefined(value: Value): value is Undefined {
  return value instanceof Undefined;
}

function unwrap(value: Value): Value {
  return value instanceof PyFloat ? value.value : value;
}

function truthy(value: Value): boolean {
  const v = unwrap(value);
  if (v === null || v === undefined || isUndefined(v) || v === false) return false;
  if (typeof v === 'number') return v !== 0;
  if (typeof v === 'string') return v.length > 0;
  if (Array.isArray(v)) return v.length > 0;
  if (v instanceof Map) return v.size > 0;
  if (typeof v === 'object' && !(v instanceof Namespace) && !(v instanceof Macro)) return Object.keys(v as object).length > 0;
  return true;
}

function pythonRepr(value: Value): string {
  const v = value;
  if (v instanceof PyFloat) return floatRepr(v.value);
  if (typeof v === 'string') return `'${v.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n')}'`;
  return toText(v, true);
}

function floatRepr(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) < 1e16) return `${value}.0`;
  return String(value);
}

/** ``str(value)`` as Python/Jinja renders it. */
function toText(value: Value, nested = false): string {
  if (value instanceof PyFloat) return floatRepr(value.value);
  if (value === null || value === undefined) return 'None';
  if (isUndefined(value)) return '';
  if (typeof value === 'string') return nested ? pythonRepr(value) : value;
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return String(value);
  if (Array.isArray(value)) return `[${value.map((item) => (typeof item === 'string' ? pythonRepr(item) : toText(item, true))).join(', ')}]`;
  if (value instanceof Namespace) return '<Namespace>';
  if (typeof value === 'object') {
    return `{${Object.entries(value as Record<string, Value>).map(([key, item]) => `${pythonRepr(key)}: ${typeof item === 'string' ? pythonRepr(item) : toText(item, true)}`).join(', ')}}`;
  }
  return String(value);
}

function toJson(value: Value, indent: number | null): string {
  const normalize = (item: Value): unknown => {
    const v = unwrap(item);
    if (isUndefined(v)) return null;
    if (Array.isArray(v)) return v.map(normalize);
    if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v as Record<string, Value>).map(([key, entry]) => [key, normalize(entry)]));
    return v;
  };
  // transformers' tojson: json.dumps(x, ensure_ascii=False, indent=indent)
  const text = JSON.stringify(normalize(value), null, indent ?? undefined);
  return indent === null ? text.replace(/"(?:[^"\\]|\\.)*"|[:,]/g, (token) => (token === ':' ? ': ' : token === ',' ? ', ' : token)) : text;
}

function equals(a: Value, b: Value): boolean {
  const x = unwrap(a);
  const y = unwrap(b);
  if (isUndefined(x) || isUndefined(y)) return isUndefined(x) && isUndefined(y);
  if (Array.isArray(x) && Array.isArray(y)) return x.length === y.length && x.every((item, index) => equals(item, y[index]));
  if (x && y && typeof x === 'object' && typeof y === 'object') {
    const kx = Object.keys(x as object);
    const ky = Object.keys(y as object);
    return kx.length === ky.length && kx.every((key) => equals((x as Record<string, Value>)[key], (y as Record<string, Value>)[key]));
  }
  if (typeof x === 'boolean' && typeof y === 'number') return Number(x) === y;
  if (typeof y === 'boolean' && typeof x === 'number') return Number(y) === x;
  return x === y;
}

function length(value: Value): number {
  const v = unwrap(value);
  if (typeof v === 'string' || Array.isArray(v)) return v.length;
  if (v && typeof v === 'object') return Object.keys(v as object).length;
  throw new TemplateError('object has no len()');
}

function iterate(value: Value): Value[] {
  const v = unwrap(value);
  if (isUndefined(v) || v === null || v === undefined) {
    if (v === null || v === undefined) throw new TemplateError("'NoneType' object is not iterable");
    return [];
  }
  if (Array.isArray(v)) return v;
  if (typeof v === 'string') return [...v];
  if (typeof v === 'object') return Object.keys(v as object);
  throw new TemplateError('object is not iterable');
}

function attribute(target: Value, name: string): Value {
  const v = unwrap(target);
  if (isUndefined(v)) return new Undefined(`${v.hint}.${name}`);
  if (v instanceof Namespace) return v.values.has(name) ? v.values.get(name) : new Undefined(name);
  if (typeof v === 'string') return stringMethod(v, name);
  if (Array.isArray(v)) return listMethod(v, name);
  if (v && typeof v === 'object') {
    const record = v as Record<string, Value>;
    if (Object.prototype.hasOwnProperty.call(record, name)) return record[name];
    const method = dictMethod(record, name);
    return method ?? new Undefined(name);
  }
  return new Undefined(name);
}

function item(target: Value, key: Value): Value {
  const v = unwrap(target);
  const k = unwrap(key);
  if (isUndefined(v)) return new Undefined(`${v.hint}[...]`);
  if (Array.isArray(v) || typeof v === 'string') {
    if (typeof k !== 'number' || !Number.isInteger(k)) return new Undefined('index');
    const index = k < 0 ? v.length + k : k;
    if (index < 0 || index >= v.length) return new Undefined('index');
    return v[index];
  }
  if (v instanceof Namespace) return v.values.has(String(k)) ? v.values.get(String(k)) : new Undefined(String(k));
  if (v && typeof v === 'object') {
    const record = v as Record<string, Value>;
    return Object.prototype.hasOwnProperty.call(record, String(k)) ? record[String(k)] : new Undefined(String(k));
  }
  return new Undefined('item');
}

function sliceValue(target: Value, start: Value, stop: Value, step: Value): Value {
  const v = unwrap(target);
  if (!Array.isArray(v) && typeof v !== 'string') return new Undefined('slice');
  const size = v.length;
  const stride = step === null ? 1 : Number(unwrap(step));
  if (stride === 0) throw new TemplateError('slice step cannot be zero');
  const clamp = (value: Value, fallback: number): number => {
    if (value === null) return fallback;
    let index = Number(unwrap(value));
    if (index < 0) index += size;
    return stride > 0 ? Math.min(Math.max(index, 0), size) : Math.min(Math.max(index, -1), size - 1);
  };
  const from = clamp(start, stride > 0 ? 0 : size - 1);
  const to = clamp(stop, stride > 0 ? size : -1);
  const items: Value[] = [];
  for (let index = from; stride > 0 ? index < to : index > to; index += stride) items.push(v[index]);
  return typeof v === 'string' ? (items as string[]).join('') : items;
}

type Callable = (args: Value[], kwargs: Record<string, Value>) => Value;

function stringMethod(value: string, name: string): Value {
  const methods: Record<string, Callable> = {
    strip: ([chars]) => stripChars(value, chars as string | undefined, true, true),
    lstrip: ([chars]) => stripChars(value, chars as string | undefined, true, false),
    rstrip: ([chars]) => stripChars(value, chars as string | undefined, false, true),
    lower: () => value.toLowerCase(),
    upper: () => value.toUpperCase(),
    title: () => titleCase(value),
    capitalize: () => capitalize(value),
    startswith: ([prefix]) => (Array.isArray(prefix) ? prefix.some((item) => value.startsWith(String(item))) : value.startsWith(String(prefix))),
    endswith: ([suffix]) => (Array.isArray(suffix) ? suffix.some((item) => value.endsWith(String(item))) : value.endsWith(String(suffix))),
    split: ([separator, limit]) => pythonSplit(value, separator as string | null | undefined, limit === undefined ? -1 : Number(limit)),
    replace: ([old, replacement, count]) => replaceCount(value, String(old), String(replacement), count === undefined ? -1 : Number(count)),
    find: ([needle]) => value.indexOf(String(needle)),
    count: ([needle]) => value.split(String(needle)).length - 1,
    join: ([items]) => iterate(items).map((part) => toText(part)).join(value),
    format: (args) => { let index = 0; return value.replace(/\{\}/g, () => toText(args[index++])); },
  };
  const method = methods[name];
  return method ? { __callable__: method } : new Undefined(name);
}

function listMethod(value: Value[], name: string): Value {
  if (name === 'index') return { __callable__: ([needle]: Value[]) => value.findIndex((entry) => equals(entry, needle)) };
  if (name === 'count') return { __callable__: ([needle]: Value[]) => value.filter((entry) => equals(entry, needle)).length };
  return new Undefined(name);
}

function dictMethod(value: Record<string, Value>, name: string): Value | null {
  switch (name) {
    case 'items': return { __callable__: () => Object.entries(value).map(([key, entry]) => [key, entry]) };
    case 'keys': return { __callable__: () => Object.keys(value) };
    case 'values': return { __callable__: () => Object.values(value) };
    case 'get': return { __callable__: ([key, fallback]: Value[]) => (Object.prototype.hasOwnProperty.call(value, String(unwrap(key))) ? value[String(unwrap(key))] : (fallback ?? null)) };
    default: return null;
  }
}

function stripChars(value: string, chars: string | null | undefined, left: boolean, right: boolean): string {
  const set = chars === undefined || chars === null ? null : new Set([...chars]);
  const isStrip = (char: string): boolean => (set ? set.has(char) : /\s/.test(char));
  let start = 0;
  let end = value.length;
  if (left) while (start < end && isStrip(value[start]!)) start += 1;
  if (right) while (end > start && isStrip(value[end - 1]!)) end -= 1;
  return value.slice(start, end);
}

function pythonSplit(value: string, separator: string | null | undefined, limit: number): string[] {
  if (separator === undefined || separator === null) {
    const parts = value.trim().split(/\s+/).filter(Boolean);
    return limit >= 0 && parts.length > limit + 1 ? [...parts.slice(0, limit), parts.slice(limit).join(' ')] : parts;
  }
  const parts = value.split(separator);
  return limit >= 0 && parts.length > limit + 1 ? [...parts.slice(0, limit), parts.slice(limit).join(separator)] : parts;
}

function replaceCount(value: string, old: string, replacement: string, count: number): string {
  if (count < 0) return value.split(old).join(replacement);
  let result = value;
  let offset = 0;
  for (let index = 0; index < count; index += 1) {
    const position = result.indexOf(old, offset);
    if (position < 0) break;
    result = result.slice(0, position) + replacement + result.slice(position + old.length);
    offset = position + replacement.length;
  }
  return result;
}

function capitalize(value: string): string {
  return value ? value[0]!.toUpperCase() + value.slice(1).toLowerCase() : value;
}

function titleCase(value: string): string {
  return value.replace(/[A-Za-z]+/g, (word) => capitalize(word));
}

function callValue(callee: Value, args: Value[], kwargs: Record<string, Value>): Value {
  if (callee && typeof callee === 'object' && '__callable__' in (callee as object)) {
    return ((callee as { __callable__: Callable }).__callable__)(args, kwargs);
  }
  throw new TemplateError('object is not callable');
}

const TESTS: Record<string, (value: Value, args: Value[]) => boolean> = {
  defined: (value) => !isUndefined(value),
  undefined: (value) => isUndefined(value),
  none: (value) => value === null,
  string: (value) => typeof value === 'string',
  number: (value) => typeof unwrap(value) === 'number',
  integer: (value) => typeof value === 'number' && Number.isInteger(value),
  float: (value) => value instanceof PyFloat,
  boolean: (value) => typeof value === 'boolean',
  true: (value) => value === true,
  false: (value) => value === false,
  mapping: (value) => !!value && typeof value === 'object' && !Array.isArray(value) && !(value instanceof PyFloat) && !isUndefined(value),
  sequence: (value) => Array.isArray(value) || typeof value === 'string',
  iterable: (value) => Array.isArray(value) || typeof value === 'string' || (!!value && typeof value === 'object' && !isUndefined(value)),
  callable: (value) => !!value && typeof value === 'object' && '__callable__' in (value as object),
  even: (value) => Number(unwrap(value)) % 2 === 0,
  odd: (value) => Math.abs(Number(unwrap(value)) % 2) === 1,
  eq: (value, [other]) => equals(value, other),
  equalto: (value, [other]) => equals(value, other),
  ne: (value, [other]) => !equals(value, other),
  in: (value, [other]) => contains(other, value),
  lower: (value) => typeof value === 'string' && value === value.toLowerCase(),
  upper: (value) => typeof value === 'string' && value === value.toUpperCase(),
};

function contains(container: Value, needle: Value): boolean {
  const c = unwrap(container);
  if (typeof c === 'string') return c.includes(toText(needle));
  if (Array.isArray(c)) return c.some((entry) => equals(entry, needle));
  if (c && typeof c === 'object' && !isUndefined(c)) return Object.prototype.hasOwnProperty.call(c, String(unwrap(needle)));
  if (isUndefined(c)) throw new TemplateError("argument of type 'Undefined' is not iterable");
  return false;
}

function compare(a: Value, b: Value): number {
  const x = unwrap(a);
  const y = unwrap(b);
  if (typeof x === 'string' && typeof y === 'string') return x < y ? -1 : x > y ? 1 : 0;
  const nx = typeof x === 'boolean' ? Number(x) : x;
  const ny = typeof y === 'boolean' ? Number(y) : y;
  if (typeof nx === 'number' && typeof ny === 'number') return nx - ny;
  throw new TemplateError('unorderable types');
}

function filterValue(name: string, value: Value, args: Value[], kwargs: Record<string, Value>, env: Renderer): Value {
  const v = unwrap(value);
  switch (name) {
    case 'capitalize': return capitalize(toText(v));
    case 'lower': return toText(v).toLowerCase();
    case 'upper': return toText(v).toUpperCase();
    case 'title': return titleCase(toText(v));
    case 'trim': return stripChars(toText(v), (args[0] as string | undefined) ?? null, true, true);
    case 'length': case 'count': return length(v);
    case 'first': { const items = iterate(v); return items.length ? items[0] : new Undefined('first'); }
    case 'last': { const items = iterate(v); return items.length ? items[items.length - 1] : new Undefined('last'); }
    case 'join': {
      const attributeName = kwargs.attribute as string | undefined;
      return iterate(v).map((entry) => toText(attributeName ? attribute(entry, attributeName) : entry)).join(toText(args[0] ?? ''));
    }
    case 'default': case 'd': {
      const fallback = args[0] ?? '';
      const boolean = truthy(args[1] ?? kwargs.boolean ?? false);
      return isUndefined(v) || (boolean && !truthy(v)) ? fallback : value;
    }
    case 'tojson': return toJson(v, (kwargs.indent ?? args[0] ?? null) as number | null);
    case 'string': return toText(v);
    case 'int': { const parsed = Number.parseInt(toText(v), 10); return Number.isNaN(parsed) ? 0 : parsed; }
    case 'float': return new PyFloat(Number(toText(v)));
    case 'abs': return Math.abs(Number(v));
    case 'list': return [...iterate(v)];
    case 'reverse': return typeof v === 'string' ? [...v].reverse().join('') : [...iterate(v)].reverse();
    case 'replace': return replaceCount(toText(v), toText(args[0]), toText(args[1]), args[2] === undefined ? -1 : Number(args[2]));
    case 'items': return Object.entries((v ?? {}) as Record<string, Value>).map(([key, entry]) => [key, entry]);
    case 'unique': { const seen: Value[] = []; for (const entry of iterate(v)) if (!seen.some((other) => equals(other, entry))) seen.push(entry); return seen; }
    case 'sort': {
      const attributeName = kwargs.attribute as string | undefined;
      const reverse = truthy(kwargs.reverse ?? false);
      const items = [...iterate(v)].sort((a, b) => compare(attributeName ? attribute(a, attributeName) : a, attributeName ? attribute(b, attributeName) : b));
      return reverse ? items.reverse() : items;
    }
    case 'map': {
      const attributeName = kwargs.attribute as string | undefined;
      if (attributeName) return iterate(v).map((entry) => attribute(entry, attributeName));
      const filterName = toText(args[0]);
      return iterate(v).map((entry) => filterValue(filterName, entry, args.slice(1), {}, env));
    }
    case 'select': case 'reject': {
      const testName = args.length ? toText(args[0]) : null;
      const keep = name === 'select';
      return iterate(v).filter((entry) => (testName ? runTest(testName, entry, args.slice(1)) : truthy(entry)) === keep);
    }
    case 'selectattr': case 'rejectattr': {
      const attributeName = toText(args[0]);
      const testName = args.length > 1 ? toText(args[1]) : null;
      const keep = name === 'selectattr';
      return iterate(v).filter((entry) => {
        const field = attribute(entry, attributeName);
        return (testName ? runTest(testName, field, args.slice(2)) : truthy(field)) === keep;
      });
    }
    case 'indent': {
      const width = Number(args[0] ?? kwargs.width ?? 4);
      const first = truthy(args[1] ?? kwargs.first ?? false);
      const pad = ' '.repeat(width);
      const lines = toText(v).split('\n');
      return lines.map((line, index) => ((index === 0 && !first) || !line ? line : pad + line)).join('\n');
    }
    case 'safe': case 'e': case 'escape': return name === 'safe' ? value : toText(v);
    case 'wordcount': return toText(v).split(/\s+/).filter(Boolean).length;
    case 'round': {
      const precision = Number(args[0] ?? 0);
      return new PyFloat(Math.round(Number(v) * 10 ** precision) / 10 ** precision);
    }
    default: throw new TemplateError(`unsupported template filter ${JSON.stringify(name)}`);
  }
}

function runTest(name: string, value: Value, args: Value[]): boolean {
  const test = TESTS[name];
  if (!test) throw new TemplateError(`unsupported template test ${JSON.stringify(name)}`);
  return test(value, args);
}

class Renderer {
  constructor(private readonly nodes: Node[]) {}

  render(context: Record<string, Value>): string {
    const scope = new Scope(null);
    scope.set('raise_exception', { __callable__: ([message]: Value[]) => { throw new TemplateError(toText(message)); } });
    scope.set('range', { __callable__: (args: Value[]) => {
      const numbers = args.map((arg) => Number(unwrap(arg)));
      const [start, stop, step] = numbers.length === 1 ? [0, numbers[0]!, 1] : [numbers[0]!, numbers[1]!, numbers[2] ?? 1];
      const result: number[] = [];
      for (let index = start; step > 0 ? index < stop : index > stop; index += step) result.push(index);
      return result;
    } });
    scope.set('namespace', { __callable__: (_args: Value[], kwargs: Record<string, Value>) => new Namespace(new Map(Object.entries(kwargs))) });
    scope.set('dict', { __callable__: (_args: Value[], kwargs: Record<string, Value>) => ({ ...kwargs }) });
    scope.set('strftime_now', { __callable__: ([format]: Value[]) => strftime(toText(format), new Date()) });
    for (const [key, value] of Object.entries(context)) scope.set(key, value);
    const output: string[] = [];
    this.renderNodes(this.nodes, scope, output);
    return output.join('');
  }

  private renderNodes(nodes: Node[], scope: Scope, output: string[]): void {
    for (const node of nodes) {
      switch (node.type) {
        case 'text': output.push(node.value); break;
        case 'output': output.push(toText(this.evaluate(node.expr, scope))); break;
        case 'if': {
          const branch = node.branches.find(([condition]) => truthy(this.evaluate(condition, scope)));
          this.renderNodes(branch ? branch[1] : node.otherwise, scope, output);
          break;
        }
        case 'for': this.renderFor(node, scope, output); break;
        case 'set': {
          let value: Value;
          if (node.body) {
            const buffer: string[] = [];
            this.renderNodes(node.body, scope, buffer);
            value = buffer.join('');
          } else {
            value = this.evaluate(node.value!, scope);
          }
          if (node.attribute) {
            const target = scope.get(node.target);
            if (!(target instanceof Namespace)) throw new TemplateError('cannot assign attribute on non-namespace object');
            target.values.set(node.attribute, value);
          } else {
            scope.set(node.target, value);
          }
          break;
        }
        case 'macro': scope.set(node.name, new Macro(node.params, node.body, scope)); break;
        case 'break': throw new LoopControl('break');
        case 'continue': throw new LoopControl('continue');
      }
    }
  }

  private renderFor(node: Extract<Node, { type: 'for' }>, scope: Scope, output: string[]): void {
    let items = iterate(this.evaluate(node.iterable, scope));
    const bind = (target: Scope, entry: Value): void => {
      if (node.targets.length === 1) target.set(node.targets[0]!, entry);
      else {
        const parts = iterate(entry);
        node.targets.forEach((name, index) => target.set(name, parts[index]));
      }
    };
    if (node.filter) {
      items = items.filter((entry) => {
        const probe = scope.child();
        bind(probe, entry);
        return truthy(this.evaluate(node.filter!, probe));
      });
    }
    if (!items.length) {
      this.renderNodes(node.otherwise, scope, output);
      return;
    }
    for (let index = 0; index < items.length; index += 1) {
      const inner = scope.child();
      bind(inner, items[index]);
      inner.set('loop', {
        index: index + 1, index0: index, revindex: items.length - index, revindex0: items.length - index - 1,
        first: index === 0, last: index === items.length - 1, length: items.length,
        previtem: index > 0 ? items[index - 1] : new Undefined('previtem'),
        nextitem: index < items.length - 1 ? items[index + 1] : new Undefined('nextitem'),
        cycle: { __callable__: (args: Value[]) => args[index % args.length] },
      });
      try {
        this.renderNodes(node.body, inner, output);
      } catch (error) {
        if (error instanceof LoopControl) {
          if (error.kind === 'break') break;
          continue;
        }
        throw error;
      }
    }
  }

  private evaluate(expr: Expr, scope: Scope): Value {
    switch (expr.type) {
      case 'literal': return expr.value;
      case 'name': return scope.get(expr.name);
      case 'list': return expr.items.map((entry) => this.evaluate(entry, scope));
      case 'tuple': return expr.items.map((entry) => this.evaluate(entry, scope));
      case 'dict': return Object.fromEntries(expr.entries.map(([key, value]) => [toText(this.evaluate(key, scope)), this.evaluate(value, scope)]));
      case 'attr': return attribute(this.evaluate(expr.target, scope), expr.name);
      case 'item': return item(this.evaluate(expr.target, scope), this.evaluate(expr.key, scope));
      case 'slice': return sliceValue(this.evaluate(expr.target, scope),
        expr.start ? this.evaluate(expr.start, scope) : null, expr.stop ? this.evaluate(expr.stop, scope) : null,
        expr.step ? this.evaluate(expr.step, scope) : null);
      case 'call': {
        const callee = this.evaluate(expr.callee, scope);
        const args = expr.args.map((entry) => this.evaluate(entry, scope));
        const kwargs = Object.fromEntries(expr.kwargs.map(([key, value]) => [key, this.evaluate(value, scope)]));
        if (callee instanceof Macro) return this.callMacro(callee, args, kwargs);
        if (isUndefined(callee)) throw new TemplateError(`'${callee.hint}' is undefined`);
        return callValue(callee, args, kwargs);
      }
      case 'filter': {
        const args = expr.args.map((entry) => this.evaluate(entry, scope));
        const kwargs = Object.fromEntries(expr.kwargs.map(([key, value]) => [key, this.evaluate(value, scope)]));
        return filterValue(expr.name, this.evaluate(expr.target, scope), args, kwargs, this);
      }
      case 'test': {
        const result = runTest(expr.name, this.evaluate(expr.target, scope), expr.args.map((entry) => this.evaluate(entry, scope)));
        return expr.negated ? !result : result;
      }
      case 'unary': {
        const operand = this.evaluate(expr.operand, scope);
        if (expr.op === 'not') return !truthy(operand);
        const number = Number(unwrap(operand));
        return expr.op === '-' ? (operand instanceof PyFloat ? new PyFloat(-number) : -number) : operand;
      }
      case 'conditional': return truthy(this.evaluate(expr.test, scope))
        ? this.evaluate(expr.yes, scope)
        : (expr.no ? this.evaluate(expr.no, scope) : new Undefined('conditional'));
      case 'binary': return this.binary(expr, scope);
    }
  }

  private binary(expr: Extract<Expr, { type: 'binary' }>, scope: Scope): Value {
    if (expr.op === 'and') {
      const left = this.evaluate(expr.left, scope);
      return truthy(left) ? this.evaluate(expr.right, scope) : left;
    }
    if (expr.op === 'or') {
      const left = this.evaluate(expr.left, scope);
      return truthy(left) ? left : this.evaluate(expr.right, scope);
    }
    const left = this.evaluate(expr.left, scope);
    const right = this.evaluate(expr.right, scope);
    const l = unwrap(left);
    const r = unwrap(right);
    const floatResult = left instanceof PyFloat || right instanceof PyFloat;
    const numeric = (value: number): Value => (floatResult ? new PyFloat(value) : value);
    switch (expr.op) {
      case '==': return equals(left, right);
      case '!=': return !equals(left, right);
      case '<': return compare(left, right) < 0;
      case '>': return compare(left, right) > 0;
      case '<=': return compare(left, right) <= 0;
      case '>=': return compare(left, right) >= 0;
      case 'in': return contains(right, left);
      case 'not in': return !contains(right, left);
      case '~': return toText(left) + toText(right);
      case '+':
        if (typeof l === 'string' && typeof r === 'string') return l + r;
        if (Array.isArray(l) && Array.isArray(r)) return [...l, ...r];
        if (typeof l === 'number' && typeof r === 'number') return numeric(l + r);
        throw new TemplateError('unsupported operand types for +');
      case '-': return numeric(Number(l) - Number(r));
      case '*':
        if (typeof l === 'string' && typeof r === 'number') return l.repeat(r);
        if (Array.isArray(l) && typeof r === 'number') return Array.from({ length: r }, () => l).flat();
        return numeric(Number(l) * Number(r));
      case '/': return new PyFloat(Number(l) / Number(r));
      case '//': return numeric(Math.floor(Number(l) / Number(r)));
      case '%': {
        const a = Number(l);
        const b = Number(r);
        return numeric(((a % b) + b) % b);
      }
      case '**': return numeric(Number(l) ** Number(r));
      default: throw new TemplateError(`unsupported operator ${expr.op}`);
    }
  }

  private callMacro(macro: Macro, args: Value[], kwargs: Record<string, Value>): Value {
    const scope = macro.scope.child();
    macro.params.forEach((param, index) => {
      if (index < args.length) scope.set(param.name, args[index]);
      else if (param.name in kwargs) scope.set(param.name, kwargs[param.name]);
      else if (param.fallback) scope.set(param.name, this.evaluate(param.fallback, macro.scope));
      else scope.set(param.name, new Undefined(param.name));
    });
    const output: string[] = [];
    this.renderNodes(macro.body, scope, output);
    return output.join('');
  }
}

function strftime(format: string, date: Date): string {
  const pad = (value: number, width = 2): string => String(value).padStart(width, '0');
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  return format.replace(/%([YmdHMSbBaAy%])/g, (_, code: string) => {
    switch (code) {
      case 'Y': return String(date.getFullYear());
      case 'y': return pad(date.getFullYear() % 100);
      case 'm': return pad(date.getMonth() + 1);
      case 'd': return pad(date.getDate());
      case 'H': return pad(date.getHours());
      case 'M': return pad(date.getMinutes());
      case 'S': return pad(date.getSeconds());
      case 'b': return months[date.getMonth()]!.slice(0, 3);
      case 'B': return months[date.getMonth()]!;
      case 'a': return days[date.getDay()]!.slice(0, 3);
      case 'A': return days[date.getDay()]!;
      default: return '%';
    }
  });
}

/** A compiled chat template. */
export class ChatTemplate {
  private readonly renderer: Renderer;

  constructor(readonly source: string) {
    this.renderer = new Renderer(parseTemplate(lex(source)));
  }

  /** Render with the given variables (``messages``, ``add_generation_prompt``, special tokens ...). */
  render(context: Record<string, unknown>): string {
    return this.renderer.render(context);
  }
}
