/** Shared helpers for vector parity tests. */
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, expect } from 'vitest';
import { bindingRecords } from '../../src/_internal/fingerprint.js';
import { deserializeSafetensors } from '../../src/nn/safetensors.js';
import { jsonEqual } from '../../src/_internal/json.js';
import type { OperationLike } from '../../src/ops/base.js';

export function fixturePath(name: string): string {
  return new URL(`../fixtures/vec/${name}`, import.meta.url).pathname.replace(/\/$/, '');
}

/** A scratch directory removed after the current test file. */
export function scratchDirectory(prefix = 'tensorcode-vec-'): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  afterAll(() => rmSync(directory, { recursive: true, force: true }));
  return directory;
}

let counter = 0;

/**
 * Safetensors files equal up to the ordering of ``__metadata__`` aliases.
 * Python serializes tied-tensor aliases from a Rust ``HashMap``, so with more
 * than one alias its own byte order varies between runs.
 */
function equalUpToAliasOrder(actual: Buffer, expected: Buffer): boolean {
  if (actual.length !== expected.length) return false;
  const view = (buffer: Buffer) => deserializeSafetensors(new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength));
  const a = view(actual);
  const b = view(expected);
  if (!jsonEqual(a.metadata ?? {}, b.metadata ?? {}) || Object.keys(b.metadata ?? {}).length < 2) return false;
  const headerLength = Number(expected.readBigUInt64LE(0));
  if (Buffer.compare(actual.subarray(8 + headerLength), expected.subarray(8 + headerLength)) !== 0) return false;
  return [...a.tensors.keys()].join() === [...b.tensors.keys()].join();
}

/** Save ``model`` and compare the manifest and weights with a Python-written fixture artifact. */
export async function expectByteIdenticalResave(
  model: { savePretrained(directory: string): Promise<string> }, name: string, scratch: string,
): Promise<void> {
  counter += 1;
  const target = join(scratch, `${name}-${counter}`);
  await model.savePretrained(target);
  for (const file of ['tensorcode_config.json', 'model.safetensors']) {
    const actual = readFileSync(join(target, file));
    const expected = readFileSync(join(fixturePath(name), file));
    if (Buffer.compare(actual, expected) !== 0 && !(file.endsWith('.safetensors') && equalUpToAliasOrder(actual, expected))) {
      const encoding = file.endsWith('.json') ? 'utf8' : 'hex';
      expect(actual.toString(encoding)).toBe(expected.toString(encoding));
    }
  }
}

export function fingerprints(operations: Record<string, OperationLike>): Record<string, string> {
  return Object.fromEntries(Object.entries(bindingRecords(operations)).map(([name, record]) => [name, record.fingerprint]));
}

/** Replace an absolute path recorded by Python with the local fixture path. */
export function relocate<T>(value: T, from: string, to: string): T {
  return JSON.parse(JSON.stringify(value).split(from).join(to)) as T;
}
