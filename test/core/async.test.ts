import { describe, expect, it } from 'vitest';
import { trace } from '../../src/_internal/tracing.js';
import { ModuleOperation, Operation, type Context } from '../../src/ops/base.js';
import { Parameter, Tensor, noGrad, tensor, scalar } from '../../src/nn/index.js';

class Add extends Operation<number[], number[]> {
  override get replayable(): boolean { return true; }
  forward(value: number[], context: Context | null): number[] {
    return [value[0]! + ((context?.amount as number | undefined) ?? 1)];
  }
}

class Payload {
  static readonly recordFields = ['tensor'] as const;
  tensor: Tensor;
  constructor(value: Tensor) { this.tensor = value; }
  static fromRecord(fields: Record<string, unknown>): Payload { return new Payload(fields.tensor as Tensor); }
  toRecord(): Record<string, unknown> { return { tensor: this.tensor }; }
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

class AwaitedScale extends ModuleOperation<{ payload: Payload }, Tensor> {
  entered = deferred();
  resume = deferred();
  weight: Parameter;
  lastResult: Tensor | null = null;
  constructor(readonly failure = false) {
    super();
    this.weight = this.registerParameter('weight', new Parameter(scalar(2)));
  }
  override get replayable(): boolean { return true; }
  forward(value: { payload: Payload }, context: Context | null): Tensor {
    const amount = (context!.scale as { amount: Tensor }).amount;
    return value.payload.tensor.mul(amount).mul(this.weight);
  }
  override async aforward(value: { payload: Payload }, context: Context | null): Promise<Tensor> {
    this.entered.resolve();
    await this.resume.promise;
    if (this.failure) throw new Error('provider failed after await');
    this.lastResult = this.forward(value, context);
    return this.lastResult;
  }
}

describe('asynchronous operations', () => {
  it('preserve capture and conditioning', async () => {
    const add = new Add();
    const session = trace();
    const second = await session.run(async () => {
      const first = await add.acall([2], { context: { amount: 3 } });
      return add.acall(first);
    });
    expect(second).toEqual([6]);
    expect(session.calls.length).toBe(2);
    expect(session.replay(session.ref(second))).toEqual([6]);
  });

  for (const mutation of ['tensor', 'record', 'context_tensor', 'context_container'] as const) {
    it(`reject ${mutation} mutation during an awaited call before registering its output`, async () => {
      const operation = new AwaitedScale();
      const value = { payload: new Payload(tensor([1], { requiresGrad: true })) };
      const context = { scale: { amount: tensor(3, { requiresGrad: true }) } as { amount: Tensor } };
      const session = trace();
      await session.run(async () => {
        const pending = operation.acall(value, { context });
        await operation.entered.promise;
        noGrad(() => {
          if (mutation === 'tensor') value.payload.tensor.add_(5);
          else if (mutation === 'record') value.payload.tensor = tensor([7]);
          else if (mutation === 'context_tensor') context.scale.amount.add_(5);
          else context.scale.amount = tensor(7);
        });
        operation.resume.resolve();
        await expect(pending).rejects.toThrow(/mutated.*async/);
        const failed = session.calls[0]!;
        expect(failed.pending).toBe(false);
        expect(failed.result).toBeNull();
        expect(failed.error).toMatch(/ValueError/);
        expect(() => session.example(failed.output)).toThrow(/Failed call/);
        expect(() => session.ref(operation.lastResult)).toThrow(/Unknown/);
        const recovered = await operation.acall(value, { context });
        expect((session.replay(recovered) as Tensor).toArray()).toEqual(recovered.toArray());
        expect(session.example(recovered).calls).toEqual([1]);
      });
    });
  }

  it('preserve live gradients and replay gradients', async () => {
    const operation = new AwaitedScale();
    const valueTensor = tensor([2], { requiresGrad: true });
    const amount = tensor(3, { requiresGrad: true });
    const session = trace();
    const result = await session.run(async () => {
      const pending = operation.acall({ payload: new Payload(valueTensor) }, { context: { scale: { amount } } });
      await operation.entered.promise;
      operation.resume.resolve();
      return pending;
    });
    result.sum().backward();
    expect(valueTensor.grad!.item()).toBe(6);
    expect(amount.grad!.item()).toBe(4);
    expect(operation.weight.grad!.item()).toBe(6);
    operation.weight.grad = null;
    const replayed = session.replay(result) as Tensor;
    expect(replayed.toArray()).toEqual(result.toArray());
    replayed.sum().backward();
    expect(operation.weight.grad!.item()).toBe(6);
  });

  it('preserve provider errors raised after input mutation', async () => {
    const operation = new AwaitedScale(true);
    const input = tensor([1]);
    const session = trace();
    await session.run(async () => {
      const task = operation.acall({ payload: new Payload(input) }, { context: { scale: { amount: tensor(3) } } });
      await operation.entered.promise;
      input.add_(1);
      operation.resume.resolve();
      await expect(task).rejects.toThrow('provider failed after await');
    });
    expect(session.calls[0]!.error).toBe('Error: provider failed after await');
    expect(session.calls[0]!.result).toBeNull();
  });

  it('reject a synchronous forward that returns a promise under tracing', () => {
    class Bad extends Operation<number, Promise<number>> {
      forward(): Promise<number> { return Promise.resolve(1); }
    }
    trace().run(() => {
      expect(() => new Bad().call(1)).toThrow(/aforward/);
    });
  });
});
