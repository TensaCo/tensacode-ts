import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { defaultCacheDir } from '../../src/_internal/hub.js';

/** Local snapshot directory of a cached Hub repository, or null when absent. */
export function cachedSnapshot(repo: string, snapshot: string): string | null {
  const path = join(defaultCacheDir(), `models--${repo.replace('/', '--')}`, 'snapshots', snapshot);
  return existsSync(path) ? path : null;
}
