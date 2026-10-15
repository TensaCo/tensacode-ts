import { Program } from './core.js';
/** Accept a candidate only after exhaustive agreement over the original finite input domain.
 * Stateful behavior, floating domains and external calls require separate validation.
 */
export function verifiedReplacement(reference: Program, candidate: Program, maxCases = 100000): Program {
  const a = reference.graph(), b = candidate.graph();
  if (a.semantics !== 'exact' || b.semantics !== 'exact') throw new Error('replacement requires exact semantics');
  if ([...a.nodes, ...b.nodes].some(n => ['state', 'call', 'parameter'].includes(n.op))) throw new Error('freeze parameters and remove state/host boundaries before replacement');
  const signature = (p: Program) => JSON.stringify(p.graph().nodes.filter(n => n.op === 'input').map(n => [n.name, n.domain.dtype, n.domain.min, n.domain.max]).sort());
  if (signature(reference) !== signature(candidate) || JSON.stringify(Object.keys(a.outputs).sort()) !== JSON.stringify(Object.keys(b.outputs).sort())) throw new Error('replacement signature mismatch');
  const report = candidate.verify(inputs => reference.run(inputs), { maxCases });
  if (!report.passed) throw new Error(`replacement counterexample: ${JSON.stringify(report.counterexample)}`);
  return candidate;
}
