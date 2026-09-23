import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { PretrainedModule } from '../../src/_internal/pretrained.js';
import { Workspace } from '../../src/_internal/workspace.js';
import { TemperatureCalibration } from '../../src/training/calibration.js';
import { Linear, type Tensor } from '../../src/nn/index.js';
import { torchDTypeName } from '../../src/nn/dtype.js';
import { ValueError } from '../../src/errors.js';
import { fixtureJson } from '../helpers/fixtures.js';
import type { JsonObject } from '../../src/_internal/json.js';

class FixtureTool extends PretrainedModule<Tensor, Tensor> {
  static override readonly qualifiedName: string = 'artifact_fixtures.FixtureTool';
  readonly workspace: Workspace;
  readonly head: Linear;
  readonly shared: Linear;
  readonly calibration: TemperatureCalibration;

  constructor(config: JsonObject) {
    super(config);
    const dimensions = this.config.dimensions as number;
    this.workspace = this.registerModule('workspace', new Workspace(dimensions, 2, 1));
    this.head = this.registerModule('head', new Linear(dimensions, 3));
    this.shared = this.registerModule('shared', new Linear(3, 3, { bias: false }));
    this.registerModule('alias', this.shared);
    this.calibration = this.registerModule('calibration', new TemperatureCalibration());
  }

  forward(inputs: Tensor): Tensor {
    return this.head.forward(inputs);
  }
}

const source = new URL('../fixtures/artifact_python', import.meta.url).pathname;
const expected = fixtureJson('artifact_python.json');
const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-artifact-'));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

describe('pretrained artifacts interoperate with Python', () => {
  it('loads a Python artifact, restoring dtypes and ties', async () => {
    const tool = await FixtureTool.fromPretrained(source);
    expect([...tool.stateDict().keys()]).toEqual(expected.state_keys);
    const dtypes = Object.fromEntries([...tool.stateDict()].map(([key, value]) => [key, torchDTypeName(value.dtype)]));
    expect(dtypes).toEqual(expected.dtypes);
    expect(tool.getSubmodule('alias')).toBe(tool.shared);
    expect(tool.training).toBe(false);
  });

  it('re-saves byte-identical manifest and weights', async () => {
    const tool = await FixtureTool.fromPretrained(source);
    const target = join(scratch, 'resaved');
    await tool.savePretrained(target);
    for (const file of ['tensorcode_config.json', 'model.safetensors']) {
      expect(Buffer.compare(readFileSync(join(target, file)), readFileSync(join(source, file)))).toBe(0);
    }
    const again = await FixtureTool.fromPretrained(target, { localFilesOnly: true });
    expect([...again.stateDict().keys()]).toEqual(expected.state_keys);
  });

  it('rejects incompatible identities and configuration drift', async () => {
    class OtherTool extends FixtureTool {
      static override readonly qualifiedName: string = 'artifact_fixtures.OtherTool';
    }
    await expect(OtherTool.fromPretrained(source)).rejects.toThrow(/incompatible model tool/);
    class DriftTool extends FixtureTool {
      static override readonly qualifiedName: string = 'artifact_fixtures.FixtureTool';
      override configuration(): JsonObject {
        return { ...super.configuration(), extra: true };
      }
    }
    await expect(DriftTool.fromPretrained(source)).rejects.toThrow(ValueError);
    await expect(FixtureTool.fromPretrained('./does-not-exist')).rejects.toThrow(/not found/);
  });
});
