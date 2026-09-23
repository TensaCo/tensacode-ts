/**
 * Keyword-argument binding for plan actions (Python
 * ``inspect.signature(action).bind(None, **arguments)``).
 *
 * A Python registry action is ``action(state, **arguments)``; a TypeScript one
 * is ``action(state, arguments)``. The keyword parameters of a TypeScript
 * action are the properties it destructures from its second parameter::
 *
 *     (state, { amount, note = 'none' }) => ...   // amount required, note optional
 *     (state, { amount, ...rest }) => ...         // also accepts any other keyword
 *     (state, args) => ...                        // accepts any keyword (``**kwargs``)
 *     (state) => ...                              // accepts no keyword
 *     () => ...                                   // accepts no keyword
 *
 * They are read from the function's source. Functions whose source does not
 * show their parameters (bound or native functions, transpiled code) accept
 * every keyword unless an explicit signature is declared with
 * {@link withSignature}. Binding errors are ``TypeError``s with Python's
 * messages.
 */

/** One declared keyword parameter. */
export interface ActionParameter {
  readonly name: string;
  /** ``true`` when the parameter has no default (Python ``Parameter.empty``). */
  readonly required: boolean;
  /** Python ``KEYWORD_ONLY`` (only changes the missing-argument message). */
  readonly keywordOnly?: boolean;
}

/** An explicit action signature (Python ``def action(state, *, ...)``). */
export interface ActionSignature {
  /** The parameter receiving the state (default ``state``); ``null`` declares no positional parameter. */
  readonly state?: string | null;
  /**
   * Keyword parameters in declaration order. A string names a required
   * parameter; a trailing ``?`` (``'note?'``) marks one with a default.
   */
  readonly parameters?: readonly (string | ActionParameter)[];
  /** Accept undeclared keywords (Python ``**kwargs``). */
  readonly variadic?: boolean;
  /** Accept any number of positional arguments (Python ``*args``). */
  readonly variadicPositional?: boolean;
}

/** A normalized signature. */
export interface ResolvedSignature {
  readonly state: string | null;
  readonly parameters: readonly ActionParameter[];
  readonly variadic: boolean;
  readonly variadicPositional: boolean;
}

const declared = new WeakMap<object, ResolvedSignature>();

function normalize(signature: ActionSignature): ResolvedSignature {
  if (signature === null || typeof signature !== 'object') throw new TypeError('signature must be an object');
  const parameters = (signature.parameters ?? []).map((item): ActionParameter => {
    if (typeof item === 'string') {
      const optional = item.endsWith('?');
      const name = optional ? item.slice(0, -1) : item;
      if (!name) throw new TypeError('signature parameter names must be nonempty strings');
      return Object.freeze({ name, required: !optional });
    }
    if (item === null || typeof item !== 'object' || typeof item.name !== 'string' || !item.name) {
      throw new TypeError('signature parameters must be names or {name, required} objects');
    }
    return Object.freeze({ name: item.name, required: item.required !== false, keywordOnly: item.keywordOnly === true });
  });
  const names = new Set<string>();
  for (const parameter of parameters) {
    if (names.has(parameter.name)) throw new TypeError(`duplicate argument '${parameter.name}' in function definition`);
    names.add(parameter.name);
  }
  const state = signature.state === undefined ? 'state' : signature.state;
  if (state !== null && (typeof state !== 'string' || !state)) throw new TypeError('signature state must be a nonempty string or null');
  return Object.freeze({
    state, parameters: Object.freeze(parameters), variadic: signature.variadic === true,
    variadicPositional: signature.variadicPositional === true,
  });
}

/**
 * Declare an action's keyword parameters explicitly (returns ``fn``).
 * Takes precedence over the parameters read from the function's source.
 */
export function withSignature<F extends (...args: never[]) => unknown>(fn: F, signature: ActionSignature): F {
  if (typeof fn !== 'function') throw new TypeError('withSignature expects a function');
  declared.set(fn, normalize(signature));
  return fn;
}

/** The explicit or inferred signature of ``fn``. */
export function actionSignature(fn: (...args: never[]) => unknown, explicit?: ActionSignature | null): ResolvedSignature {
  if (explicit) return normalize(explicit);
  const known = declared.get(fn);
  if (known) return known;
  return inferSignature(fn);
}

const PERMISSIVE: ResolvedSignature = Object.freeze({ state: null, parameters: Object.freeze([]), variadic: true, variadicPositional: true });

function quote(name: string): string {
  // Python ``repr`` of an identifier-like string.
  return name.includes("'") && !name.includes('"') ? `"${name}"` : `'${name.replaceAll('\\', '\\\\').replaceAll("'", "\\'")}'`;
}

/**
 * Bind ``(state, **args)`` like Python ``Signature.bind(None, **args)``;
 * throws ``TypeError`` with Python's message when binding fails.
 */
export function bindArguments(signature: ResolvedSignature, args: Readonly<Record<string, unknown>>): void {
  const keys = Object.keys(args);
  if (signature.state === null) {
    if (!signature.variadicPositional) throw new TypeError('too many positional arguments');
  } else if (keys.includes(signature.state) && !signature.parameters.some((parameter) => parameter.name === signature.state)) {
    throw new TypeError(`multiple values for argument ${quote(signature.state)}`);
  }
  const remaining = new Set(keys);
  for (const parameter of signature.parameters) {
    if (remaining.delete(parameter.name)) continue;
    if (parameter.required) {
      throw new TypeError(`missing a required${parameter.keywordOnly ? ' keyword-only' : ''} argument: ${quote(parameter.name)}`);
    }
  }
  if (remaining.size && !signature.variadic) {
    throw new TypeError(`got an unexpected keyword argument ${quote(remaining.values().next().value as string)}`);
  }
}

// ---------------------------------------------------------------------------
// Function source parsing.
// ---------------------------------------------------------------------------

/** Index just past a string, template literal or comment starting at ``index``; ``index`` otherwise. */
function skipLiteral(source: string, index: number): number {
  const char = source[index];
  if (char === '/' && source[index + 1] === '/') {
    const end = source.indexOf('\n', index + 2);
    return end < 0 ? source.length : end + 1;
  }
  if (char === '/' && source[index + 1] === '*') {
    const end = source.indexOf('*/', index + 2);
    return end < 0 ? source.length : end + 2;
  }
  if (char === '"' || char === "'") {
    let position = index + 1;
    while (position < source.length && source[position] !== char) position += source[position] === '\\' ? 2 : 1;
    return position + 1;
  }
  if (char === '`') {
    let position = index + 1;
    while (position < source.length && source[position] !== '`') {
      if (source[position] === '\\') position += 2;
      else if (source[position] === '$' && source[position + 1] === '{') {
        const end = matching(source, position + 1);
        if (end < 0) return source.length;
        position = end + 1;
      }
      else position += 1;
    }
    return position + 1;
  }
  return index;
}

const CLOSERS: Record<string, string> = { '(': ')', '[': ']', '{': '}' };

/** Index of the bracket closing the one at ``open``. */
function matching(source: string, open: number): number {
  const stack: string[] = [CLOSERS[source[open]!]!];
  let index = open + 1;
  while (index < source.length) {
    const skipped = skipLiteral(source, index);
    if (skipped !== index) {
      index = skipped;
      continue;
    }
    const char = source[index]!;
    if (Object.hasOwn(CLOSERS, char)) stack.push(CLOSERS[char]!);
    else if (char === stack[stack.length - 1]) {
      stack.pop();
      if (!stack.length) return index;
    }
    index += 1;
  }
  return -1;
}

/** Split at top-level commas, dropping comments. */
function splitTopLevel(source: string): string[] {
  const parts: string[] = [];
  let current = '';
  let index = 0;
  while (index < source.length) {
    const char = source[index]!;
    if ((char === '/' && (source[index + 1] === '/' || source[index + 1] === '*'))) {
      index = skipLiteral(source, index);
      current += ' ';
      continue;
    }
    const skipped = skipLiteral(source, index);
    if (skipped !== index) {
      current += source.slice(index, skipped);
      index = skipped;
      continue;
    }
    if (Object.hasOwn(CLOSERS, char)) {
      const end = matching(source, index);
      if (end < 0) return [...parts, current + source.slice(index)].map((part) => part.trim());
      current += source.slice(index, end + 1);
      index = end + 1;
      continue;
    }
    if (char === ',') {
      parts.push(current.trim());
      current = '';
    } else {
      current += char;
    }
    index += 1;
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

/** Index of the first top-level ``=`` that is not part of ``==``/``=>``/``<=``/``>=``/``!=``, or -1. */
function topLevelAssignment(source: string): number {
  let index = 0;
  while (index < source.length) {
    const skipped = skipLiteral(source, index);
    if (skipped !== index) {
      index = skipped;
      continue;
    }
    const char = source[index]!;
    if (Object.hasOwn(CLOSERS, char)) {
      const end = matching(source, index);
      if (end < 0) return -1;
      index = end + 1;
      continue;
    }
    if (char === '=' && source[index + 1] !== '=' && source[index + 1] !== '>' && !'=!<>'.includes(source[index - 1] ?? '')) return index;
    index += 1;
  }
  return -1;
}

const IDENTIFIER = /^[A-Za-z_$ -￿][\w$ -￿]*$/u;

/** The parameter list text of a function's source, or ``null``. */
function parameterList(source: string): string | null {
  let text = source.trim();
  if (/\{\s*\[native code\]\s*\}\s*$/.test(text)) return null;
  if (/^class\b/.test(text)) return null;
  const arrow = /^(?:async\s+)?([A-Za-z_$ -￿][\w$ -￿]*)\s*=>/u.exec(text);
  if (arrow) return arrow[1]!;
  // Method keys may be computed (``[key](state) {}``): skip them.
  text = text.replace(/^(?:async\s+)?(?:function\b\s*)?\*?\s*/, '');
  if (text.startsWith('[')) {
    const end = matching(text, 0);
    if (end < 0) return null;
    text = text.slice(end + 1);
  }
  let index = 0;
  while (index < text.length && text[index] !== '(') {
    const skipped = skipLiteral(text, index);
    if (skipped !== index) {
      index = skipped;
      continue;
    }
    if ('{=>'.includes(text[index]!)) return null;
    index += 1;
  }
  if (index >= text.length) return null;
  const end = matching(text, index);
  return end < 0 ? null : text.slice(index + 1, end);
}

/** The keyword parameters destructured by an object pattern, or ``null`` when unknowable. */
function objectPattern(pattern: string): { parameters: ActionParameter[]; variadic: boolean } | null {
  const body = pattern.trim().slice(1, -1);
  const parameters: ActionParameter[] = [];
  let variadic = false;
  for (const property of splitTopLevel(body)) {
    if (!property) continue;
    if (property.startsWith('...')) {
      variadic = true;
      continue;
    }
    let key: string;
    let rest: string;
    const keyMatch = /^([A-Za-z_$ -￿][\w$ -￿]*|\d+)\s*/u.exec(property);
    if (keyMatch) {
      key = keyMatch[1]!;
      rest = property.slice(keyMatch[0].length);
    } else if (property[0] === '"' || property[0] === "'") {
      const end = skipLiteral(property, 0);
      try {
        key = JSON.parse(property[0] === "'" ? `"${property.slice(1, end - 1).replaceAll('"', '\\"').replaceAll("\\'", "'")}"` : property.slice(0, end)) as string;
      } catch {
        return null;
      }
      rest = property.slice(end).trim();
    } else {
      return null; // computed key
    }
    let required: boolean;
    if (rest.startsWith(':')) required = topLevelAssignment(rest.slice(1)) < 0;
    else if (rest.startsWith('=')) required = false;
    else if (!rest) required = true;
    else return null;
    parameters.push({ name: key, required });
  }
  return { parameters, variadic };
}

/** Infer ``(state, {keywords})`` from a function's source. */
export function inferSignature(fn: (...args: never[]) => unknown): ResolvedSignature {
  let source: string;
  try {
    source = Function.prototype.toString.call(fn);
  } catch {
    return PERMISSIVE;
  }
  const list = parameterList(source);
  if (list === null) return PERMISSIVE;
  const parameters = splitTopLevel(list).filter(Boolean);
  if (!parameters.length) {
    // JavaScript functions may ignore their arguments: the state is accepted, keywords are not.
    return Object.freeze({ state: null, parameters: Object.freeze([]), variadic: false, variadicPositional: true });
  }
  const first = parameters[0]!;
  if (first.startsWith('...')) return PERMISSIVE;
  const firstAssignment = topLevelAssignment(first);
  const firstName = (firstAssignment < 0 ? first : first.slice(0, firstAssignment)).trim();
  const state = IDENTIFIER.test(firstName) ? firstName : 'state';
  const second = parameters[1];
  if (second === undefined) {
    return Object.freeze({ state, parameters: Object.freeze([]), variadic: false, variadicPositional: false });
  }
  if (second.startsWith('...')) return Object.freeze({ state, parameters: Object.freeze([]), variadic: true, variadicPositional: true });
  const assignment = topLevelAssignment(second);
  const target = (assignment < 0 ? second : second.slice(0, assignment)).trim();
  if (target.startsWith('{')) {
    const pattern = objectPattern(target);
    if (pattern === null) return Object.freeze({ state, parameters: Object.freeze([]), variadic: true, variadicPositional: false });
    return Object.freeze({
      state, parameters: Object.freeze(pattern.parameters.map((item) => Object.freeze(item))), variadic: pattern.variadic,
      variadicPositional: false,
    });
  }
  // A plain identifier (or an array pattern) receives the whole keyword mapping.
  return Object.freeze({ state, parameters: Object.freeze([]), variadic: true, variadicPositional: false });
}
