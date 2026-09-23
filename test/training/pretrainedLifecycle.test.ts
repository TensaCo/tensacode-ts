/** Port of Python ``tests/models/test_pretrained.py`` (PretrainedModule lifecycle and Hub publication). */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  Linear, Tensor, deserializeSafetensors, noGrad, ones, randn, serializeSafetensors, zeros,
} from '../../src/nn/index.js';
import { PretrainedModule } from '../../src/_internal/pretrained.js';
import { FileNotFoundError } from '../../src/_internal/hub.js';
import { ValueError } from '../../src/errors.js';
import type { JsonObject } from '../../src/_internal/json.js';
import { scratchDirectory, transform } from './helpers.js';
import { expectClose } from '../helpers/gradcheck.js';

const scratch = scratchDirectory('tensorcode-pretrained-');

class Tiny extends PretrainedModule<Tensor, Tensor> {
  static override readonly qualifiedName: string = 'tests.models.test_pretrained.Tiny';
  readonly encoder: Linear;
  readonly decoder: Linear;
  session: string[] = [];

  constructor(config: JsonObject) {
    super(config);
    const width = this.config.width as number;
    this.encoder = this.registerModule('encoder', new Linear(width, width, { bias: false }));
    this.decoder = this.registerModule('decoder', new Linear(width, width, { bias: false }));
    this.decoder.setParameterAt('weight', this.encoder.weight);
  }

  forward(value: Tensor): Tensor {
    return this.decoder.forward(this.encoder.forward(value));
  }
}

class Other extends Tiny {
  static override readonly qualifiedName: string = 'tests.models.test_pretrained.Other';
}

function manifestPath(directory: string): string {
  return join(directory, 'tensorcode_config.json');
}

function editManifest(directory: string, edit: (data: any) => void): void {
  const data = JSON.parse(readFileSync(manifestPath(directory), 'utf8'));
  edit(data);
  writeFileSync(manifestPath(directory), JSON.stringify(data));
}

function tiny(width = 2): Tiny {
  return new Tiny({ width });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('PretrainedModule lifecycle', () => {
  it('round-trips shared weights without session state', async () => {
    const config = { width: 3 };
    const model = new Tiny(config);
    config.width = 9;
    model.session.push('private message');
    const value = randn([2, 3]);
    const directory = join(scratch(), 'model');
    await model.savePretrained(directory);
    const restored = await Tiny.fromPretrained(directory, { localFilesOnly: true });
    expect(restored.config).toEqual({ width: 3 });
    expect(restored.training).toBe(false);
    expect(restored.decoder.weight).toBe(restored.encoder.weight);
    expect(restored.session).toEqual([]);
    expectClose(noGrad(() => restored.forward(value)).data, noGrad(() => model.forward(value)).data, 0, 0);
    expect(readFileSync(manifestPath(directory), 'utf8')).not.toContain('private message');
    expect(model.configuration()).toEqual({ width: 3 });
  });

  const changes: Record<string, unknown>[] = [{ version: 999 }, { tool: 'untrusted.Other' }, { config: [] }, { format: 'other' }];
  for (const change of changes) {
    it(`rejects an incompatible manifest ${JSON.stringify(change)}`, async () => {
      const directory = scratch();
      await tiny().savePretrained(directory);
      editManifest(directory, (data) => Object.assign(data, change));
      await expect(Tiny.fromPretrained(directory)).rejects.toThrow(ValueError);
    });
  }

  it('rejects the wrong class and shape changes', async () => {
    const directory = scratch();
    await tiny().savePretrained(directory);
    await expect(Other.fromPretrained(directory)).rejects.toThrow(/tool/);
    editManifest(directory, (data) => { data.config.width = 3; });
    await expect(Tiny.fromPretrained(directory)).rejects.toThrow();
  });

  it('requires JSON configuration', () => {
    expect(() => new Tiny({ width: 2, value: Number.NaN })).toThrow(ValueError);
  });

  it('preserves the previous model when saving fails', async () => {
    const directory = scratch();
    const model = tiny();
    await model.savePretrained(directory);
    const previous = readFileSync(join(directory, 'model.safetensors'));
    (model as unknown as { savePretrainedAssets: () => Promise<void> }).savePretrainedAssets = async () => {
      throw new Error('disk failure');
    };
    await expect(model.savePretrained(directory)).rejects.toThrow(/disk failure/);
    expect(readFileSync(join(directory, 'model.safetensors')).equals(previous)).toBe(true);
    await Tiny.fromPretrained(directory);
  });

  it('resolves Hub ids from a pinned cached snapshot without network', async () => {
    const cache = scratch();
    const sha = 'a'.repeat(40);
    const snapshot = join(cache, 'models--owner--model', 'snapshots', sha);
    await tiny().savePretrained(snapshot);
    mkdirSync(join(cache, 'models--owner--model', 'refs'), { recursive: true });
    writeFileSync(join(cache, 'models--owner--model', 'refs', 'main'), sha);
    const network = vi.fn(() => { throw new Error('network used'); });
    vi.stubGlobal('fetch', network);
    const pinned = await Tiny.fromPretrained('owner/model', { revision: sha, localFilesOnly: true, cacheDir: cache, token: 'token' });
    expect(pinned.config).toEqual({ width: 2 });
    const main = await Tiny.fromPretrained('owner/model', { localFilesOnly: true, cacheDir: cache });
    expect(main.config).toEqual({ width: 2 });
    await expect(Tiny.fromPretrained('owner/model', { revision: 'b'.repeat(40), localFilesOnly: true, cacheDir: cache }))
      .rejects.toThrow(FileNotFoundError);
    expect(network).not.toHaveBeenCalled();
  });

  it('does not turn a missing local directory into a Hub request', async () => {
    const network = vi.fn(() => { throw new Error('network used'); });
    vi.stubGlobal('fetch', network);
    await expect(Tiny.fromPretrained(join(scratch(), 'missing'))).rejects.toThrow(FileNotFoundError);
    expect(network).not.toHaveBeenCalled();
  });

  it('publishes only model assets through the Hub transport', async () => {
    const uploaded: { names?: string[]; restored?: Tiny } = {};
    const commits: string[] = [];
    const fake = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      const headers = new Headers(init?.headers);
      if (!url.startsWith('https://storage.test/')) expect(headers.get('authorization')).toBe('Bearer token');
      if (url.endsWith('/api/repos/create')) {
        expect(JSON.parse(init!.body as string)).toMatchObject({ name: 'model', organization: 'owner', private: true });
        return new Response('{}', { status: 200 });
      }
      if (url.includes('/preupload/')) {
        const files = (JSON.parse(init!.body as string) as { files: { path: string }[] }).files;
        uploaded.names = files.map((file) => file.path).sort();
        return Response.json({ files: files.map((file) => ({ path: file.path, uploadMode: file.path.endsWith('.safetensors') ? 'lfs' : 'regular' })) });
      }
      if (url.endsWith('.git/info/lfs/objects/batch')) {
        const objects = (JSON.parse(init!.body as string) as { objects: { oid: string; size: number }[] }).objects;
        return Response.json({ objects: objects.map((object) => ({ ...object, actions: { upload: { href: `https://storage.test/${object.oid}` } } })) });
      }
      if (url.startsWith('https://storage.test/')) {
        // Rebuild the uploaded artifact from the transported bytes.
        const folder = scratch();
        mkdirSync(folder, { recursive: true });
        writeFileSync(join(folder, 'model.safetensors'), new Uint8Array(init!.body as ArrayBuffer));
        (uploaded as Record<string, unknown>).folder = folder;
        return new Response(null, { status: 200 });
      }
      if (url.includes('/commit/')) {
        commits.push(url);
        const folder = (uploaded as Record<string, string>).folder!;
        for (const line of (init!.body as string).trim().split('\n').map((text) => JSON.parse(text))) {
          if (line.key === 'file') writeFileSync(join(folder, line.value.path), Buffer.from(line.value.content, 'base64'));
        }
        uploaded.restored = await Tiny.fromPretrained(folder);
        return Response.json({ commitUrl: 'commit-url', commitOid: 'abc' });
      }
      throw new Error(`unexpected request ${url}`);
    });
    vi.stubGlobal('fetch', fake);
    const model = tiny();
    model.session.push('private');
    const result = await model.pushToHub('owner/model', { private: true, revision: 'main', token: 'token' }) as { commit: string };
    expect(result.commit).toBe('commit-url');
    expect(uploaded.names).toEqual(['README.md', 'model.safetensors', 'tensorcode_config.json']);
    expect(uploaded.restored!.session).toEqual([]);
    expect(commits).toEqual(['https://huggingface.co/api/models/owner/model/commit/main']);
  });

  it('rejects extra weights', async () => {
    const directory = scratch();
    await tiny().savePretrained(directory);
    const path = join(directory, 'model.safetensors');
    const contents = deserializeSafetensors(new Uint8Array(readFileSync(path)));
    contents.tensors.set('unexpected', zeros([1]));
    writeFileSync(path, serializeSafetensors(contents.tensors, contents.metadata));
    await expect(Tiny.fromPretrained(directory)).rejects.toThrow(/unexpected/);
  });

  it('operation bindings find traced operations and deduplicate aliases', () => {
    const model = tiny();
    const interpret = model.registerModule('interpret', transform(new Linear(2, 2)));
    model.registerModule('alias', interpret);
    expect(model.operationBindings()).toEqual({ interpret });
    expect(model.operationBindings().interpret).toBe(interpret);
  });

  it('round-trips float64 weights with aliases', async () => {
    const model = tiny();
    const doubled = model.encoder.weight.detach().to('float64');
    const parameter = new (model.encoder.weight.constructor as typeof import('../../src/nn/index.js').Parameter)(doubled);
    model.encoder.setParameterAt('weight', parameter);
    model.decoder.setParameterAt('weight', parameter);
    const directory = scratch();
    await model.savePretrained(directory);
    const loaded = await Tiny.fromPretrained(directory);
    expect(loaded.encoder.weight.dtype).toBe('float64');
    expect(loaded.decoder.weight).toBe(loaded.encoder.weight);
    const value = randn([1, 2]).to('float64');
    expectClose(noGrad(() => loaded.forward(value)).data, noGrad(() => model.forward(value)).data, 0, 0);
  });

  it('writes a factual model card and preserves an authored one', async () => {
    const directory = scratch();
    const model = tiny();
    await model.savePretrained(directory);
    const card = join(directory, 'README.md');
    const text = readFileSync(card, 'utf8');
    expect(text).toContain('library_name: tensorcode');
    expect(text).toContain('fromPretrained');
    expect(text.toLowerCase()).toContain('training');
    writeFileSync(card, '# Authored model card\n');
    await model.savePretrained(directory);
    expect(readFileSync(card, 'utf8')).toBe('# Authored model card\n');
  });

  it('validates an explicit model card before any network use', async () => {
    const calls: string[] = [];
    let readme = '';
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      calls.push(url);
      if (url.includes('/preupload/')) {
        const files = (JSON.parse(init!.body as string) as { files: { path: string }[] }).files;
        return Response.json({ files: files.map((file) => ({ path: file.path, uploadMode: 'regular' })) });
      }
      if (url.includes('/commit/')) {
        for (const line of (init!.body as string).trim().split('\n').map((text) => JSON.parse(text))) {
          if (line.key === 'file' && line.value.path === 'README.md') readme = Buffer.from(line.value.content, 'base64').toString('utf8');
        }
        return Response.json({ commitUrl: 'published' });
      }
      return Response.json({});
    }));
    vi.stubEnv('HF_TOKEN', 'token');
    const model = tiny();
    await expect(model.pushToHub('owner/model', { modelCard: 123 as never })).rejects.toThrow(/modelCard/);
    expect(calls).toEqual([]);
    const result = await model.pushToHub('owner/model', { modelCard: '# Evaluated model\n' }) as { commit: string };
    expect(result.commit).toBe('published');
    expect(readme).toBe('# Evaluated model\n');
  });

  class Defaulted extends Tiny {
    static override readonly qualifiedName: string = 'tests.models.test_pretrained.Defaulted';
    constructor(config: JsonObject) {
      super({ activation_scale: 0.5, ...config });
    }
  }

  class NestedDefaulted extends Tiny {
    static override readonly qualifiedName: string = 'tests.models.test_pretrained.NestedDefaulted';
    readonly child: Defaulted;
    constructor(config: JsonObject) {
      super(config);
      this.child = this.registerModule('child', new Defaulted(config.child as JsonObject));
    }
    override configuration(): JsonObject {
      return { ...super.configuration(), child: this.child.configuration() };
    }
  }

  for (const nested of [false, true]) {
    it(`rejects default drift before reading weights (nested=${nested})`, async () => {
      const directory = scratch();
      const model = nested ? new NestedDefaulted({ width: 2, child: { width: 2 } }) : new Defaulted({ width: 2 });
      await model.savePretrained(directory);
      editManifest(directory, (data) => {
        delete (nested ? data.config.child : data.config).activation_scale;
      });
      rmSync(join(directory, 'model.safetensors'));
      const cls = nested ? NestedDefaulted : Defaulted;
      await expect(cls.fromPretrained(directory)).rejects.toThrow(/configuration.*architecture/);
    });
  }

  it('round-trips constructor defaults and nested canonical artifacts', async () => {
    const model = new NestedDefaulted({ width: 2, child: { width: 2 }, metadata: { 0: [null, true, 1, 1.5, 'value'] } });
    expect((model.configuration().child as JsonObject).activation_scale).toBe(0.5);
    const directory = scratch();
    await model.savePretrained(directory);
    const restored = await NestedDefaulted.fromPretrained(directory);
    expect(restored.configuration()).toEqual(model.configuration());
    for (const [key, value] of model.stateDict()) expect(value.equal(restored.stateDict().get(key)!)).toBe(true);
  });

  class LocalAsset extends Tiny {
    static override readonly qualifiedName: string = 'tests.models.test_pretrained.LocalAsset';
    readonly asset: string;
    constructor(config: JsonObject) {
      const { _asset: asset, ...rest } = config;
      super(rest);
      this.asset = typeof asset === 'string' ? asset : 'local content';
    }
    protected override async savePretrainedAssets(directory: string): Promise<void> {
      writeFileSync(join(directory, 'asset.txt'), this.asset);
    }
    static override loadPretrainedConfig(config: JsonObject, directory: string): JsonObject {
      return { ...config, _asset: readFileSync(join(directory, 'asset.txt'), 'utf8') };
    }
  }

  it('excludes private asset bindings from canonical comparison', async () => {
    const directory = scratch();
    await new LocalAsset({ width: 2 }).savePretrained(directory);
    const restored = await LocalAsset.fromPretrained(directory);
    expect(restored.configuration()).toEqual({ width: 2 });
    expect(restored.asset).toBe('local content');
    expect(existsSync(join(directory, 'asset.txt'))).toBe(true);
  });

  class NumericNormalized extends Tiny {
    static override readonly qualifiedName: string = 'tests.models.test_pretrained.NumericNormalized';
    constructor(config: JsonObject) {
      const value = config.scale;
      super({ ...config, scale: typeof value === 'boolean' ? Number(value) : Number(value) });
    }
  }

  it('rejects JSON type drift (bool for number) before weight reads', async () => {
    // JavaScript cannot distinguish 1 from 1.0; the boolean drift remains detectable.
    const directory = scratch();
    await new NumericNormalized({ width: 2, scale: 1.0 }).savePretrained(directory);
    editManifest(directory, (data) => { data.config.scale = true; });
    rmSync(join(directory, 'model.safetensors'));
    await expect(NumericNormalized.fromPretrained(directory)).rejects.toThrow(/configuration.*architecture/);
    expect(readdirSync(directory)).not.toContain('model.safetensors');
    expect(ones([1]).item()).toBe(1);
  });
});
