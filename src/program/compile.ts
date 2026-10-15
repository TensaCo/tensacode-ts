/** Compile source syntax, never execute a function to discover its branches. */
import ts from 'typescript';
import { validateScope } from './scope.js';
import { Builder, Value, type Domain, type Options, type Program, type ReferenceResult } from './core.js';
export interface CompileOptions extends Options { inputs: Record<string, Domain>; maxIterations?: number; backend?: 'reference' | 'wasm' }
type Symbolic = Value | Symbolic[];
export function compile(source: string | Function, options: CompileOptions): Program {
  const file = ts.createSourceFile('program.ts', `(${String(source)})`, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const diagnostics = (file as ts.SourceFile & { parseDiagnostics: readonly ts.Diagnostic[] }).parseDiagnostics;
  if (diagnostics.length) throw new Error('invalid source syntax');
  const stmt = file.statements[0];
  if (file.statements.length !== 1 || !stmt || !ts.isExpressionStatement(stmt) || !ts.isParenthesizedExpression(stmt.expression)) throw new Error('expected a function');
  const fn = stmt.expression.expression;
  if (!ts.isArrowFunction(fn) && !ts.isFunctionExpression(fn)) throw new Error('expected a function');
  if (fn.modifiers?.length || fn.asteriskToken || fn.parameters.some(p => !ts.isIdentifier(p.name) || p.initializer || p.dotDotDotToken || p.questionToken)) throw new Error('async/generator/default/variadic functions unsupported');
  validateScope(fn);
  const names = fn.parameters.map(p => (p.name as ts.Identifier).text);
  if (names.length !== Object.keys(options.inputs).length || names.some(n => !Object.hasOwn(options.inputs, n))) throw new Error('input signature mismatch');
  const b = new Builder(options), env = new Map<string, Symbolic>(names.map(n => [n, b.input(n, options.inputs[n]!)]));
  const binary = new Map<ts.SyntaxKind, string>([[ts.SyntaxKind.PlusToken, 'add'], [ts.SyntaxKind.MinusToken, 'sub'], [ts.SyntaxKind.AsteriskToken, 'mul'], [ts.SyntaxKind.SlashToken, 'div'], [ts.SyntaxKind.GreaterThanToken, 'gt'], [ts.SyntaxKind.GreaterThanEqualsToken, 'ge'], [ts.SyntaxKind.LessThanToken, 'lt'], [ts.SyntaxKind.LessThanEqualsToken, 'le'], [ts.SyntaxKind.EqualsEqualsEqualsToken, 'eq'], [ts.SyntaxKind.ExclamationEqualsEqualsToken, 'ne']]);
  let iterations = 0;
  const scalar = (v: Symbolic): Value => { if (!(v instanceof Value)) throw new Error('expected scalar'); return v; };
  const lowerExpr = (n: ts.Expression, scope: Map<string, Symbolic>): Symbolic => {
    const loc = file.getLineAndCharacterOfPosition(n.getStart(file)); b.source = { language: 'typescript', line: loc.line + 1, column: loc.character };
    if (ts.isParenthesizedExpression(n)) return expr(n.expression, scope);
    if (ts.isNumericLiteral(n)) return b.constant(Number(n.text));
    if (n.kind === ts.SyntaxKind.TrueKeyword || n.kind === ts.SyntaxKind.FalseKeyword) return b.constant(Number(n.kind === ts.SyntaxKind.TrueKeyword));
    if (ts.isIdentifier(n) && scope.has(n.text)) return scope.get(n.text)!;
    if (ts.isArrayLiteralExpression(n)) return n.elements.map(x => expr(x, scope));
    if (ts.isElementAccessExpression(n)) { const a = expr(n.expression, scope); if (!Array.isArray(a)) throw new Error('indexing requires a fixed array'); return b.gather(a.map(scalar), scalar(expr(n.argumentExpression, scope))); }
    if (ts.isBinaryExpression(n) && binary.has(n.operatorToken.kind)) return b.op(binary.get(n.operatorToken.kind)!, scalar(expr(n.left, scope)), scalar(expr(n.right, scope)));
    if (ts.isPrefixUnaryExpression(n)) {
      if (n.operator === ts.SyntaxKind.MinusToken) return b.op('neg', scalar(expr(n.operand, scope)));
      if (n.operator === ts.SyntaxKind.ExclamationToken) return b.op('eq', scalar(expr(n.operand, scope)), 0);
    }
    if (ts.isConditionalExpression(n)) return merge(condition(n.condition, scope), expr(n.whenTrue, scope), expr(n.whenFalse, scope));
    if (ts.isCallExpression(n)) {
      if (ts.isIdentifier(n.expression) && ['relu', 'sigmoid'].includes(n.expression.text) && n.arguments.length === 1) return b.op(n.expression.text, scalar(expr(n.arguments[0]!, scope)));
      if (ts.isPropertyAccessExpression(n.expression) && ts.isIdentifier(n.expression.expression) && n.expression.expression.text === 'Math' && n.expression.name.text === 'max' && n.arguments.length === 2) {
        const zero = n.arguments.findIndex(a => ts.isNumericLiteral(a) && Number(a.text) === 0);
        if (zero < 0) throw new Error('Math.max is supported only as ReLU with a literal zero');
        return b.op('relu', scalar(expr(n.arguments[1 - zero]!, scope)));
      }
    }
    throw new Error(`unsupported expression at line ${loc.line + 1}: ${n.getText(file)}`);
  };
  const expr = (n: ts.Expression, scope: Map<string, Symbolic>): Symbolic => {
    const before = b.ir.nodes.length, result = lowerExpr(n, scope);
    if (result instanceof Value && result.id >= before) { const loc = file.getLineAndCharacterOfPosition(n.getStart(file)); b.ir.nodes[result.id]!.source = { language: 'typescript', line: loc.line + 1, column: loc.character }; }
    return result;
  };
  const condition = (n: ts.Expression, scope: Map<string, Symbolic>): Value => {
    if (ts.isParenthesizedExpression(n)) return condition(n.expression, scope);
    const v = scalar(expr(n, scope));
    const bool = ts.isBinaryExpression(n) && ['gt', 'ge', 'lt', 'le', 'eq', 'ne'].includes(binary.get(n.operatorToken.kind) ?? '') || ts.isPrefixUnaryExpression(n) && n.operator === ts.SyntaxKind.ExclamationToken;
    return bool ? v : b.op('ne', v, 0);
  };
  const merge = (c: Value, a: Symbolic, z: Symbolic): Symbolic => {
    if (Array.isArray(a) && Array.isArray(z) && a.length === z.length) return a.map((x, i) => merge(c, x, z[i]!));
    return b.select(c, scalar(a), scalar(z));
  };
  const statements = (s: ts.Statement): ts.Statement[] => ts.isBlock(s) ? [...s.statements] : [s];
  const literal = (n: ts.Expression): number => {
    if (ts.isNumericLiteral(n)) return Number(n.text);
    if (ts.isPrefixUnaryExpression(n) && n.operator === ts.SyntaxKind.MinusToken) return -literal(n.operand);
    throw new Error('loop bounds must be integer literals');
  };
  const block = (stmts: ts.Statement[], outer: Map<string, Symbolic>): Symbolic => {
    const scope = new Map(outer);
    for (let pos = 0; pos < stmts.length; pos++) {
      const s = stmts[pos]!;
      if (ts.isReturnStatement(s) && s.expression) return expr(s.expression, scope);
      if (ts.isIfStatement(s)) return merge(condition(s.expression, scope), block([...statements(s.thenStatement), ...stmts.slice(pos + 1)], scope), block([...(s.elseStatement ? statements(s.elseStatement) : []), ...stmts.slice(pos + 1)], scope));
      if (ts.isVariableStatement(s)) {
        for (const d of s.declarationList.declarations) { if (!ts.isIdentifier(d.name) || !d.initializer) throw new Error('locals require scalar names and initializers'); scope.set(d.name.text, expr(d.initializer, scope)); }
      } else if (ts.isExpressionStatement(s) && ts.isBinaryExpression(s.expression) && ts.isIdentifier(s.expression.left)) {
        const e = s.expression, name = (e.left as ts.Identifier).text;
        if (!scope.has(name)) throw new Error('assignment to undeclared local');
        const rhs = expr(e.right, scope);
        if (e.operatorToken.kind === ts.SyntaxKind.EqualsToken) scope.set(name, rhs);
        else {
          const op = new Map([[ts.SyntaxKind.PlusEqualsToken, 'add'], [ts.SyntaxKind.MinusEqualsToken, 'sub'], [ts.SyntaxKind.AsteriskEqualsToken, 'mul'], [ts.SyntaxKind.SlashEqualsToken, 'div']]).get(e.operatorToken.kind);
          if (!op) throw new Error('unsupported assignment');
          scope.set(name, b.op(op, scalar(scope.get(name)!), scalar(rhs)));
        }
      } else if (ts.isForStatement(s)) {
        const init = s.initializer, cond = s.condition, inc = s.incrementor;
        if (!init || !ts.isVariableDeclarationList(init) || init.declarations.length !== 1 || !cond || !ts.isBinaryExpression(cond) || !inc || !ts.isPostfixUnaryExpression(inc)) throw new Error('loop requires literal bounds and i++/i--');
        const d = init.declarations[0]!;
        if (!ts.isIdentifier(d.name) || !d.initializer || !ts.isIdentifier(cond.left) || cond.left.text !== d.name.text || !ts.isIdentifier(inc.operand) || inc.operand.text !== d.name.text) throw new Error('invalid loop induction variable');
        const start = literal(d.initializer), end = literal(cond.right), step = inc.operator === ts.SyntaxKind.PlusPlusToken ? 1 : -1;
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) throw new Error('integer loop bounds required');
        const cmp = cond.operatorToken.kind;
        if (!(step === 1 && [ts.SyntaxKind.LessThanToken, ts.SyntaxKind.LessThanEqualsToken].includes(cmp) || step === -1 && [ts.SyntaxKind.GreaterThanToken, ts.SyntaxKind.GreaterThanEqualsToken].includes(cmp))) throw new Error('loop must converge');
        const more = (i: number): boolean => cmp === ts.SyntaxKind.LessThanToken ? i < end : cmp === ts.SyntaxKind.LessThanEqualsToken ? i <= end : cmp === ts.SyntaxKind.GreaterThanToken ? i > end : i >= end;
        // Lower through repeated blocks with a synthetic continuation. No source eval.
        const expanded: ts.Statement[] = [];
        for (let i = start; more(i); i += step) {
          if (++iterations > (options.maxIterations ?? 1024)) throw new Error('loop budget exceeded');
          // Parse literal binding to retain normal AST handling and source positions.
          const binding = ts.createSourceFile('loop.ts', `let ${d.name.text} = ${i};`, ts.ScriptTarget.Latest, true).statements[0]!;
          const locate = (n: ts.Node): void => { ts.setTextRange(n, s); ts.forEachChild(n, locate); }; locate(binding);
          expanded.push(binding, ...statements(s.statement));
        }
        return block([...expanded, ...stmts.slice(pos + 1)], scope);
      } else if (ts.isEmptyStatement(s)) { /* no-op */ }
      else throw new Error(`unsupported statement: ${s.getText(file)}`);
    }
    throw new Error('every path must return a value');
  };
  const result = ts.isBlock(fn.body) ? block([...fn.body.statements], env) : expr(fn.body, env);
  const program = b.build(Array.isArray(result) ? Object.fromEntries(result.map((v, i) => [String(i), scalar(v)])) : result).optimize().withBackend(options.backend ?? 'reference');
  if (typeof source === 'function') program.withReference(inputs => source(...names.map(n => inputs[n])) as ReferenceResult);
  return program;
}
