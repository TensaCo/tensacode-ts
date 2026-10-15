import { Module } from '../nn/module.js';
import { Parameter, Tensor, tensor } from '../nn/tensor.js';
import { isGradEnabled } from '../nn/autograd.js';
import { Program, type Binding } from './core.js';
/** An owned module; registered parameters participate in native optimizers/checkpoints. */
export class ProgramModule extends Module {
  static override readonly qualifiedName: string = 'tensorcode.program.ProgramModule';
  private readonly names: string[] = [];
  constructor(readonly program: Program) {
    super();
    for (const n of program.graph().nodes) if (n.op === 'parameter') {
      this.names.push(n.name!);
      this.registerParameter(`p${this.names.length - 1}`, new Parameter(tensor(n.value!, { dtype: n.domain.dtype === 'float64' ? 'float64' : 'float32' })));
    }
  }
  forward(inputs: Record<string, Tensor>, options: { bindings?: Record<string, Binding> } = {}): Record<string, Tensor> {
    const names = Object.keys(inputs), xs = [...Object.values(inputs), ...this.parameters()];
    if (xs.some(x => x.ndim !== 0 || !x.isFloatingPoint)) throw new Error('program module expects scalar floating tensors');
    const values = Object.fromEntries(names.map((n, i) => [n, xs[i]!.item()]));
    const parameters = Object.fromEntries(this.names.map((n, i) => [n, xs[names.length + i]!.item()]));
    const recording = isGradEnabled() && xs.some(x => x.requiresGrad);
    const { outputs: results, gradients } = recording ? this.program.linearize(values, { parameters, ...options }) : { outputs: this.program.run(values, { parameters, ...options }), gradients: {} as Record<string, Record<string, number>> };
    const allNames = [...names, ...this.names];
    return Object.fromEntries(Object.entries(results).map(([name, y]) => {
      const result = tensor(y, { dtype: this.program.graph().dtype });
      if (recording) {
        // Capture the VJP at forward time, so mutable host bindings cannot change it.
        const gradient = gradients[name]!;
        const versions = xs.map(x => x.version);
        result._requiresGrad = true;
        result._gradFn = { name: 'Program', inputs: xs, backward: grad => {
          if (xs.some((x, i) => x.version !== versions[i])) throw new Error('program input modified before backward');
          return xs.map((x, i) => x.requiresGrad ? tensor(grad.item() * gradient[allNames[i]!]!, { dtype: x.dtype }) : null);
        } };
      }
      return [name, result];
    }));
  }
  artifact(): Program {
    const ir = this.program.graph(), params = new Map(this.names.map((n, i) => [n, this.parameters()[i]!.item()]));
    for (const n of ir.nodes) if (n.op === 'parameter') n.value = params.get(n.name!)!;
    return new Program(ir);
  }
}
