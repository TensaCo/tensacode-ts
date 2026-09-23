/** Atomic local file operations shared by artifact, experience and checkpoint writers. */
import { lstat, mkdir, mkdtemp, open, rename, rm } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

export async function isSymlink(path: string): Promise<boolean> {
  try {
    return (await lstat(path)).isSymbolicLink();
  } catch {
    return false;
  }
}

export async function pathExists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Write ``content`` to a temporary sibling, flush it and atomically replace
 * ``path``. The destination is untouched when writing fails.
 */
export async function atomicWriteFile(path: string, content: string | Uint8Array): Promise<void> {
  const directory = dirname(path);
  await mkdir(directory, { recursive: true });
  const temporaryDirectory = await mkdtemp(join(directory, `.${basename(path)}.`));
  const temporary = join(temporaryDirectory, 'content');
  try {
    const handle = await open(temporary, 'w');
    try {
      await handle.writeFile(content);
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(temporary, path);
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}
