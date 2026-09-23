#!/usr/bin/env node
/**
 * Build the WebAssembly SIMD kernels (`scripts/wasm/kernels.rs`) and embed
 * them as base64 in `src/nn/backend/kernels.generated.ts`.
 *
 * Requires `rustc` with the `wasm32-unknown-unknown` target (only when the
 * kernels change; the generated file is checked in, so installing or building
 * the package needs no toolchain):
 *
 *     rustup target add wasm32-unknown-unknown
 *     node scripts/wasm/build.mjs
 *
 * Two variants are built: baseline SIMD128 and relaxed SIMD (fused
 * multiply-add). The kernels import their memory (`env.memory`); the build
 * marks that import as shared so worker threads can run the same module over
 * one memory, and records the offset of the limits flag so the runtime can
 * clear it where shared memory is unavailable.
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, 'kernels.rs');
const target = join(here, '..', '..', 'src', 'nn', 'backend', 'kernels.generated.ts');

function compile(features, cfg, output) {
  const args = [
    '--edition', '2021', '--target', 'wasm32-unknown-unknown', '--crate-type', 'cdylib',
    '-C', 'opt-level=3', '-C', 'panic=abort', '-C', 'debuginfo=0',
    '-C', `target-feature=${features}`,
    '-C', 'link-arg=--import-memory', '-C', 'link-arg=--max-memory=4294967296',
    '-C', 'link-arg=-zstack-size=65536', '-C', 'link-arg=--strip-all',
    ...cfg.flatMap((name) => ['--cfg', name]),
    '-o', output, source,
  ];
  execFileSync('rustc', args, { stdio: 'inherit' });
  return new Uint8Array(readFileSync(output));
}

/**
 * Reject code that touches a global. Rust keeps its shadow stack pointer in a
 * global; every thread instantiates the module over the same memory, so a
 * kernel that spilled to the shadow stack would race with the other threads.
 */
function checkNoGlobals(bytes, start, end) {
  let position = start;
  const leb = () => {
    let result = 0n;
    let shift = 0n;
    let byte;
    do {
      byte = bytes[position++];
      result |= BigInt(byte & 0x7f) << shift;
      shift += 7n;
    } while (byte & 0x80);
    return Number(result & 0xffffffffn);
  };
  const memarg = () => { leb(); leb(); };
  const count = leb();
  for (let fn = 0; fn < count; fn += 1) {
    const size = leb();
    const bodyEnd = position + size;
    const groups = leb();
    for (let group = 0; group < groups; group += 1) { leb(); position += 1; }
    while (position < bodyEnd) {
      const op = bytes[position++];
      if (op === 0x23 || op === 0x24) throw new Error(`kernel function ${fn} uses a global (shadow stack): keep kernels free of spills`);
      if (op === 0x02 || op === 0x03 || op === 0x04) {
        const type = bytes[position];
        if (type === 0x40 || (type >= 0x6f && type <= 0x7f)) position += 1; else leb();
      } else if (op === 0x0c || op === 0x0d || op === 0x10 || (op >= 0x20 && op <= 0x22) || op === 0x25 || op === 0x26 || op === 0xd2) {
        leb();
      } else if (op === 0x0e) {
        const targets = leb();
        for (let index = 0; index <= targets; index += 1) leb();
      } else if (op === 0x11) {
        leb(); leb();
      } else if (op === 0x1c) {
        const types = leb();
        position += types;
      } else if (op >= 0x28 && op <= 0x3e) {
        memarg();
      } else if (op === 0x3f || op === 0x40 || op === 0xd0) {
        position += 1;
      } else if (op === 0x41 || op === 0x42) {
        leb();
      } else if (op === 0x43) {
        position += 4;
      } else if (op === 0x44) {
        position += 8;
      } else if (op === 0xfc) {
        const sub = leb();
        if (sub === 8) { leb(); position += 1; } else if (sub === 9) leb();
        else if (sub === 10) position += 2;
        else if (sub === 11) position += 1;
        else if (sub >= 12 && sub <= 17) { leb(); if (sub === 12 || sub === 14) leb(); }
      } else if (op === 0xfd) {
        const sub = leb();
        if (sub <= 11 || sub === 92 || sub === 93) memarg();
        else if (sub === 12 || sub === 13) position += 16;
        else if (sub >= 21 && sub <= 34) position += 1;
        else if (sub >= 84 && sub <= 91) { memarg(); position += 1; }
      } else if (op === 0xfe) {
        leb();
        memarg();
      }
    }
    if (position !== bodyEnd) throw new Error(`could not decode kernel function ${fn}`);
  }
}

/** Parse the module, check it has no data or shadow-stack use, and mark the memory import shared. */
function shareMemory(bytes) {
  let position = 8;
  const leb = () => {
    let result = 0;
    let shift = 0;
    let byte;
    do {
      byte = bytes[position++];
      result |= (byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);
    return result >>> 0;
  };
  const name = () => {
    const length = leb();
    const text = new TextDecoder().decode(bytes.subarray(position, position + length));
    position += length;
    return text;
  };
  let flagsOffset = -1;
  while (position < bytes.length) {
    const id = bytes[position++];
    const size = leb();
    const end = position + size;
    if (id === 11) throw new Error('kernels must not have data segments (shared memory would re-initialize them)');
    if (id === 10) checkNoGlobals(bytes, position, end);
    if (id === 2) {
      const count = leb();
      for (let index = 0; index < count; index += 1) {
        const module = name();
        const field = name();
        const kind = bytes[position++];
        if (kind !== 2 || module !== 'env' || field !== 'memory') throw new Error(`unexpected import ${module}.${field}`);
        flagsOffset = position;
        const flags = bytes[position++];
        if (flags !== 0x01) throw new Error(`memory import must declare a maximum (flags ${flags})`);
        leb();
        leb();
      }
    }
    position = end;
  }
  if (flagsOffset < 0) throw new Error('memory import not found');
  const shared = Uint8Array.from(bytes);
  shared[flagsOffset] = 0x03;
  if (!WebAssembly.validate(shared)) throw new Error('shared-memory module does not validate');
  const module = new WebAssembly.Module(shared);
  const globals = WebAssembly.Module.exports(module).filter((entry) => entry.kind !== 'function');
  if (globals.length) throw new Error(`unexpected non-function exports ${JSON.stringify(globals)}`);
  return { bytes: shared, flagsOffset };
}

const scratch = mkdtempSync(join(tmpdir(), 'tensorcode-wasm-'));
try {
  const simd = shareMemory(compile('+simd128,+bulk-memory', [], join(scratch, 'simd.wasm')));
  const relaxed = shareMemory(compile('+simd128,+relaxed-simd,+bulk-memory', ['relaxed'], join(scratch, 'relaxed.wasm')));
  const base64 = (bytes) => Buffer.from(bytes).toString('base64');
  const wrap = (text) => text.match(/.{1,120}/g).map((line) => `  '${line}'`).join(' +\n');
  const output = `/**
 * Generated by \`node scripts/wasm/build.mjs\` from \`scripts/wasm/kernels.rs\`. Do not edit.
 *
 * WebAssembly SIMD kernels (base64). The memory import is declared shared; the
 * byte at \`*_MEMORY_FLAGS\` is the import's limits flag (0x03 shared, 0x01 not).
 */

export const SIMD_MEMORY_FLAGS = ${simd.flagsOffset};

export const SIMD_WASM =
${wrap(base64(simd.bytes))};

export const RELAXED_MEMORY_FLAGS = ${relaxed.flagsOffset};

export const RELAXED_WASM =
${wrap(base64(relaxed.bytes))};
`;
  writeFileSync(target, output);
  console.log(`wrote ${target} (simd ${simd.bytes.length} bytes, relaxed ${relaxed.bytes.length} bytes)`);
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
