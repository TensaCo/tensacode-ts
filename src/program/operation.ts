import { ModuleOperation, type Context } from '../ops/base.js';
import type { Tensor } from '../nn/tensor.js';
import { Program, type Artifact } from './core.js';
import { ProgramModule } from './tensor.js';
/** JSON-constructed, replayable operation for trace()/Trainer.fromOps(). */
export class ProgramOperation extends ModuleOperation<Record<string, Tensor>, Record<string, Tensor>> {
  static override readonly qualifiedName: string = 'tensorcode.program.ProgramOperation';
  private readonly module: ProgramModule;
  constructor(artifact: Artifact) {
    super();
    const p = new Program(artifact);
    if (artifact.nodes.some(n => n.op === 'call' || n.op === 'state')) throw new Error('replayable operations require pure, stateless programs');
    this.module = this.registerModule('program', new ProgramModule(p));
  }
  override get replayable(): boolean { return true; }
  override forward(inputs: Record<string, Tensor>, context: Context | null = null): Record<string, Tensor> {
    if (context !== null && Object.keys(context).length) throw new Error('program inputs must be explicit; context is unsupported');
    return this.module.forward(inputs);
  }
  configuration(): Artifact { return this.module.program.graph(); }
  artifact(): Artifact { return this.module.artifact().graph(); }
}
