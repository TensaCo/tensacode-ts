import ts from 'typescript';
/** Reject unsupported lexical effects before scalar lowering duplicates continuations. */
export function validateScope(fn: ts.ArrowFunction | ts.FunctionExpression): void {
  type Scope = Map<string, boolean>[];
  const lookup = (name: string, scopes: Scope): boolean | undefined => { for (const s of scopes.toReversed()) if (s.has(name)) return s.get(name)!; return undefined; };
  const assign = (name: string, scopes: Scope, induction: Set<string>): void => {
    if (induction.has(name)) throw new Error('loop induction variable cannot be mutated in its body');
    const immutable = lookup(name, scopes);
    if (immutable === undefined) throw new Error('assignment to undeclared local');
    if (immutable) throw new Error('assignment to const local');
  };
  const expression = (node: ts.Node, scopes: Scope, induction: Set<string>): void => {
    if (ts.isBinaryExpression(node) && [ts.SyntaxKind.EqualsToken, ts.SyntaxKind.PlusEqualsToken, ts.SyntaxKind.MinusEqualsToken, ts.SyntaxKind.AsteriskEqualsToken, ts.SyntaxKind.SlashEqualsToken].includes(node.operatorToken.kind)) {
      if (!ts.isIdentifier(node.left)) throw new Error('mutation requires a local scalar');
      assign(node.left.text, scopes, induction); expression(node.right, scopes, induction); return;
    }
    if (ts.isIdentifier(node)) {
      if (lookup(node.text, scopes) === undefined && !['Math', 'relu', 'sigmoid'].includes(node.text)) throw new Error(`unbound name: ${node.text}`);
      return;
    }
    if (ts.isPropertyAccessExpression(node)) { expression(node.expression, scopes, induction); return; }
    ts.forEachChild(node, child => expression(child, scopes, induction));
  };
  const declarations = (list: ts.VariableDeclarationList, scopes: Scope, induction: Set<string>): void => {
    if (!(list.flags & (ts.NodeFlags.Let | ts.NodeFlags.Const))) throw new Error('use let or const; var hoisting is unsupported');
    for (const d of list.declarations) {
      if (!ts.isIdentifier(d.name) || !d.initializer) throw new Error('locals require initializers');
      // Shadowing is deliberately rejected: scalar SSA has no implicit scope aliases.
      if (lookup(d.name.text, scopes) !== undefined || ['Math', 'relu', 'sigmoid'].includes(d.name.text)) throw new Error('local shadowing is unsupported');
      expression(d.initializer, scopes, induction); scopes.at(-1)!.set(d.name.text, !!(list.flags & ts.NodeFlags.Const));
    }
  };
  const statement = (s: ts.Statement, scopes: Scope, induction: Set<string>): void => {
    if (ts.isBlock(s)) { const child = [...scopes, new Map<string, boolean>()]; for (const inner of s.statements) statement(inner, child, induction); }
    else if (ts.isVariableStatement(s)) declarations(s.declarationList, scopes, induction);
    else if (ts.isIfStatement(s)) { expression(s.expression, scopes, induction); statement(s.thenStatement, [...scopes, new Map()], induction); if (s.elseStatement) statement(s.elseStatement, [...scopes, new Map()], induction); }
    else if (ts.isForStatement(s)) {
      if (!s.initializer || !ts.isVariableDeclarationList(s.initializer) || s.initializer.declarations.length !== 1 || !(s.initializer.flags & ts.NodeFlags.Let)) throw new Error('loop induction requires let');
      const loop = [...scopes, new Map<string, boolean>()]; declarations(s.initializer, loop, induction);
      const d = s.initializer.declarations[0]!; if (!ts.isIdentifier(d.name)) throw new Error('invalid loop binding');
      if (s.condition) expression(s.condition, loop, induction);
      statement(s.statement, loop, new Set([...induction, d.name.text]));
    } else if (ts.isReturnStatement(s)) { if (s.expression) expression(s.expression, scopes, induction); }
    else if (ts.isExpressionStatement(s)) expression(s.expression, scopes, induction);
    else if (!ts.isEmptyStatement(s)) throw new Error('unsupported statement');
  };
  const root = new Map<string, boolean>();
  for (const p of fn.parameters) { if (!ts.isIdentifier(p.name) || root.has(p.name.text) || ['Math', 'relu', 'sigmoid'].includes(p.name.text)) throw new Error('invalid or reserved parameter name'); root.set(p.name.text, false); }
  if (ts.isBlock(fn.body)) statement(fn.body, [root], new Set()); else expression(fn.body, [root], new Set());
}
