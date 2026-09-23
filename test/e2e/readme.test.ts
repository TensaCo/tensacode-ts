/**
 * The README's 30-second example runs as written: its package imports are
 * pointed at the sources and its output directory at a scratch folder.
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';

const here = fileURLToPath(new URL('.', import.meta.url));
const source = (path: string) => fileURLToPath(new URL(`../../src/${path}`, import.meta.url));

const cleanup: string[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  for (const path of cleanup.splice(0)) rmSync(path, { recursive: true, force: true });
});

function readmeSnippet(): string {
  const readme = readFileSync(new URL('../../README.md', import.meta.url), 'utf8');
  const section = readme.slice(readme.indexOf('## 30-second example'));
  const match = /```ts\n([\s\S]*?)```/.exec(section);
  if (!match) throw new Error('README has no 30-second TypeScript example');
  return match[1]!;
}

describe('README', () => {
  it('30-second example selects the network hypothesis after reloading', async () => {
    const output = mkdtempSync(join(tmpdir(), 'tensorcode-readme-'));
    cleanup.push(output);
    const code = readmeSnippet()
      .replace(/from 'tensorcode\/(nn|tools|training)'/g, (_, entry: string) => `from ${JSON.stringify(source(`${entry}/index.ts`))}`)
      .replaceAll("'./investigator'", JSON.stringify(join(output, 'investigator')));
    expect(code).not.toMatch(/from 'tensorcode/);
    // Written next to this test so vitest transforms it like any other module.
    const module = join(here, `.readme-snippet-${process.pid}.ts`);
    cleanup.push(module);
    writeFileSync(module, code);
    const log = vi.spyOn(console, 'log').mockImplementation(() => {});
    await import(/* @vite-ignore */ module);
    expect(log).toHaveBeenCalledWith('network');
  });
});
