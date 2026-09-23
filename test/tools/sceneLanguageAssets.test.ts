/**
 * ``Scene.fromLanguageFoundation`` processor assets for every emulated
 * tokenizer class (``scripts/fixtures/processor_assets_fixtures.py``): the
 * ``tokenizer.json`` (class rebuild plus the Idefics3 tokens the processor
 * adds), ``tokenizer_config.json`` and the other files equal
 * ``Idefics3Processor.from_pretrained(dir).save_pretrained(...)``.
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { sha256Hex } from '../../src/_internal/json.js';
import { idefics3ProcessorAssets } from '../../src/tools/sceneLanguageFoundation.js';
import { Idefics3Processor } from '../../src/_internal/native/idefics3Processing.js';
import { scratchDirectory } from '../vec/helpers.js';

const root = new URL('../fixtures/scene_language/', import.meta.url).pathname;
const records = JSON.parse(readFileSync(join(root, 'tokenizer_classes.json'), 'utf8'));
const scratch = scratchDirectory('tensorcode-scene-assets-');
const hub = join(homedir(), '.cache/huggingface/hub');

describe('processor assets of synthetic tokenizers equal transformers', () => {
  for (const [name, record] of Object.entries<any>(records.synthetic)) {
    it(name, async () => {
      const assets = await idefics3ProcessorAssets(join(root, 'tokenizer_classes', name), { isLocal: true, localFilesOnly: true });
      expect(Object.keys(assets).sort()).toEqual(Object.keys(record.assets).sort());
      for (const [file, text] of Object.entries<string>(record.assets)) expect(assets[file], `${name}/${file}`).toBe(text);
      const processor = Idefics3Processor.fromAssets(assets);
      expect(processor.tokenizer.encode('hello <image> abc<end_of_utterance>').inputIds[0]).toEqual(record.ids);
    });
  }
});

describe('processor assets of cached Hub tokenizers equal transformers', () => {
  for (const [name, record] of Object.entries<any>(records.cached)) {
    const snapshot = join(hub, `models--${record.repo.replace('/', '--')}`, 'snapshots', record.snapshot);
    it.skipIf(!existsSync(snapshot))(name, async () => {
      const directory = join(scratch, name);
      mkdirSync(directory, { recursive: true });
      for (const file of record.files as string[]) if (existsSync(join(snapshot, file))) copyFileSync(join(snapshot, file), join(directory, file));
      if (record.override) {
        const config = JSON.parse(readFileSync(join(directory, 'tokenizer_config.json'), 'utf8'));
        config.tokenizer_class = record.override;
        writeFileSync(join(directory, 'tokenizer_config.json'), JSON.stringify(config));
      }
      for (const file of ['processor_config.json', 'chat_template.jinja']) copyFileSync(join(root, 'tokenizer_classes', 'llama', file), join(directory, file));
      const assets = await idefics3ProcessorAssets(directory, { isLocal: true, localFilesOnly: true });
      expect(assets['tokenizer_config.json']).toBe(record.tokenizer_config);
      const hashes = Object.fromEntries(Object.entries(assets).map(([file, text]) => [file, sha256Hex(text)]));
      expect(hashes).toEqual(record.hashes);
      const processor = Idefics3Processor.fromAssets(assets);
      expect(processor.tokenizer.encode('hello <image> abc<end_of_utterance>').inputIds[0]).toEqual(record.ids);
    }, 300_000);
  }
});
