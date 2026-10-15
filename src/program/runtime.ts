import { Program, checked, type Artifact, type RunOptions } from './core.js';
export interface Snapshot { format: 'tensorcode.program-state'; version: 1; program: Artifact; state: Record<string, number>; steps: number; maxSteps: number }
/** Event updates commit only after all outputs and next-state domains validate. */
export class Runtime {
  private current: Record<string, number>;
  private count = 0;
  constructor(readonly program: Program, readonly maxSteps = 100000) {
    if (!Number.isSafeInteger(maxSteps) || maxSteps < 0) throw new Error('invalid step budget');
    this.current = Object.fromEntries(program.graph().nodes.filter(n => n.op === 'state').map(n => [n.name!, n.value!]));
  }
  get state(): Record<string, number> { return { ...this.current }; }
  get steps(): number { return this.count; }
  dispatch(inputs: Record<string, number>, options: Omit<RunOptions, 'state'> = {}): Record<string, number> {
    if (this.count >= this.maxSteps) throw new Error('execution step budget exceeded');
    const result = this.program.step(inputs, { ...options, state: this.current });
    this.current = result.state; this.count++; return result.outputs;
  }
  snapshot(): Snapshot { return { format: 'tensorcode.program-state', version: 1, program: this.program.graph(), state: this.state, steps: this.steps, maxSteps: this.maxSteps }; }
  static restore(snapshot: Snapshot): Runtime {
    if (snapshot.format !== 'tensorcode.program-state' || snapshot.version !== 1) throw new Error('invalid state artifact');
    const result = new Runtime(new Program(snapshot.program), snapshot.maxSteps);
    if (Object.keys(snapshot.state).length !== Object.keys(result.current).length || Object.keys(result.current).some(k => !Object.hasOwn(snapshot.state, k))) throw new Error('state signature mismatch');
    if (!Number.isSafeInteger(snapshot.steps) || snapshot.steps < 0 || snapshot.steps > snapshot.maxSteps) throw new Error('invalid step count');
    for (const n of snapshot.program.nodes) if (n.op === 'state') checked(snapshot.state[n.name!]!, n.domain);
    result.current = { ...snapshot.state }; result.count = snapshot.steps; return result;
  }
  replay(events: Iterable<Record<string, number>>): Record<string, number>[] { return Array.from(events, event => this.dispatch(event)); }
}
