/**
 * Public package surface and cross-module regressions, ported from
 * python/tests/integration/test_text_namespace.py, test_regressions.py and
 * models/test_public_tool_boundaries.py::test_runtime_namespace_is_removed.
 */
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { trace } from '../../src/index.js';
import { Linear, noGrad, ones, tensor, type Tensor } from '../../src/nn/index.js';
import { Operation } from '../../src/ops/index.js';
import * as vec from '../../src/ops/vec/index.js';

const root = new URL('../../', import.meta.url);
const pkg = JSON.parse(readFileSync(new URL('package.json', root), 'utf8')) as {
  exports: Record<string, string | { types: string; import: string }>;
  homepage: string;
};

/** Map a package export target (``./dist/x.js``) to its source file. */
function sourceOf(target: string): URL {
  return new URL(target.replace(/^\.\/dist\//, 'src/').replace(/\.js$/, '.ts'), root);
}

describe('package entry points', () => {
  it('every export target has a source module and a declaration target', () => {
    for (const [key, target] of Object.entries(pkg.exports)) {
      if (key === './package.json') continue;
      expect(typeof target, key).toBe('object');
      const { types, import: module } = target as { types: string; import: string };
      expect(types.replace(/\.d\.ts$/, '.js'), key).toBe(module);
      if (key.includes('*')) continue;
      expect(existsSync(sourceOf(module)), `${key} -> ${module}`).toBe(true);
    }
  });

  it('representation namespaces are explicit entries (not the ./ops/* file pattern)', () => {
    for (const name of ['vec', 'text', 'graph']) {
      expect(pkg.exports[`./ops/${name}`]).toEqual({ types: `./dist/ops/${name}/index.d.ts`, import: `./dist/ops/${name}/index.js` });
    }
  });

  it('removed namespaces are not importable compatibility paths', () => {
    for (const name of ['./ops/llm', './runtime']) expect(pkg.exports[name]).toBeUndefined();
    expect(existsSync(new URL('src/ops/llm', root))).toBe(false);
    expect(existsSync(new URL('src/runtime', root))).toBe(false);
  });

  it('the product site is tensorcode.dev', () => {
    expect(pkg.homepage).toBe('https://tensorcode.dev');
  });

  it('entry points export the Python public names', async () => {
    const names = async (path: string) => Object.keys(await import(path)).sort();
    // Python ``tensorcode.__all__`` plus the version, error classes and the
    // Python value helpers (``float``/``int`` markers, ordered dicts).
    expect(await names('../../src/index.js')).toEqual(
      ['InputRef', 'MissingDependencyError', 'NotImplementedError', 'OutputRef', 'Trace', 'ValueError', 'float', 'int', 'orderedObject', 'trace', 'version']);
    // Python ``tensorcode.tools.__all__`` plus the shared pretrained base.
    expect(await names('../../src/tools/index.js')).toEqual(
      ['Chatbot', 'Decision', 'Investigator', 'Planner', 'PretrainedModule', 'Scene']);
    expect(await names('../../src/training/index.js')).toEqual(
      ['TemperatureCalibration', 'Trainer', 'evaluateCalibration', 'fitThreshold', 'loadExperience']);
    expect(await names('../../src/integrations/index.js')).toEqual(
      ['JevModel', 'LocalModel', 'OpenAICompatibleModel', 'ProviderError', 'ProviderHTTPError', 'ProviderProtocolError', 'ProviderTimeout', 'SynchronousCallUnavailable']);
    const vecNames = await names('../../src/ops/vec/index.js');
    for (const name of ['Latent', 'Space', 'Transform', 'Classify', 'Score', 'Decide', 'Retrieve', 'TextEncoder',
      'ImageEncoder', 'VocabularyEncoder', 'PatchEncoder', 'TextDecoder', 'ImageDecoder', 'Decode']) {
      expect(vecNames).toContain(name);
    }
  });
});

describe('text namespace', () => {
  it('composes with graph, cognition and actions without optional dependencies', async () => {
    const text = await import('../../src/ops/text/index.js');
    const graph = await import('../../src/ops/graph/index.js');
    const decode = await import('../../src/ops/text/decode.js');
    const { Evidence } = await import('../../src/tools/cognition.js');
    const { ActionOutcome, actionLoop } = await import('../../src/tools/actions.js');
    expect(typeof Evidence).toBe('function');
    expect(typeof ActionOutcome).toBe('function');
    expect(typeof actionLoop).toBe('function');
    const messages = new text.TextEncoder().call('hello');
    const response = text.Transform.fromModel(() => 'answer').call(messages);
    expect(new text.TextDecoder().call(response)).toBe('answer');
    expect('Decode' in text).toBe(false);
    expect('Decode' in decode).toBe(false);
    expect(new graph.Transform({}).configuration().implementation).toBe('unimplemented');
  });
});

describe('tracing regressions', () => {
  it('a record operand keeps the encoder dependency and gradient', () => {
    class Box {
      static readonly recordFields = ['tensor'] as const;
      constructor(readonly tensor: Tensor) { Object.freeze(this); }
      static fromRecord(fields: Record<string, unknown>): Box { return new Box(fields.tensor as Tensor); }
      toRecord(): Record<string, unknown> { return { tensor: this.tensor }; }
    }
    class Consume extends Operation<Box, Tensor> {
      override get replayable(): boolean { return true; }
      forward(value: Box): Tensor { return value.tensor.mul(2); }
    }
    const encoder = vec.Transform.fromModule(new Linear(2, 1));
    const episode = trace();
    const output = episode.run(() => new Consume().call(new Box(encoder.call(ones([2])) as Tensor)));
    const port = episode.ref(output);
    expect(episode.example(port).calls.length).toBe(2);
    (episode.replay(port) as Tensor).sum().backward();
    expect((encoder.module as Linear).weight.grad).not.toBeNull();
  });

  it('container tensor replacement is a mutation', () => {
    class Make extends Operation<Tensor, Tensor[]> {
      override get replayable(): boolean { return true; }
      forward(value: Tensor): Tensor[] { return [value.mul(1)]; }
    }
    class Consume extends Operation<Tensor[], Tensor> {
      override get replayable(): boolean { return true; }
      forward(value: Tensor[]): Tensor { return value[0]!.mul(2); }
    }
    trace().run(() => {
      const result = new Make().call(tensor([1]));
      result[0] = tensor([10]);
      expect(() => new Consume().call(result)).toThrow(/mutat/);
    });
  });

  it('an explicit reference does not bypass the mutation check', () => {
    const op = vec.Transform.fromModule(new Linear(1, 1));
    const episode = trace();
    episode.run(() => {
      const result = op.call(ones([1])) as Tensor;
      const port = episode.ref(result);
      result.add_(1);
      expect(() => op.call(port as never)).toThrow(/mutat/);
    });
  });

  it('tracing does not replace the primary input container', () => {
    class Append extends Operation<string[], number> {
      forward(value: string[]): number {
        value.push('changed');
        return value.length;
      }
    }
    const items: string[] = [];
    trace().run(() => expect(new Append().call(items)).toBe(1));
    expect(items).toEqual(['changed']);
  });

  it('gradient-free mode traces and detects tensor mutation', () => {
    const op = vec.Transform.fromModule(new Linear(1, 1));
    const episode = trace();
    noGrad(() => episode.run(() => {
      const result = op.call(ones([1])) as Tensor;
      op.call(result);
      result.add_(1);
      expect(() => op.call(result)).toThrow(/mutat/);
    }));
    expect(episode.calls.length).toBe(2);
  });
});
