import { readFileSync } from 'node:fs';
import { tensor, type Tensor, type DType } from '../../src/nn/index.js';

export interface TensorJson { shape: number[]; dtype: string; data: number[] }

export function fixtureJson<T = any>(name: string): T {
  return JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), 'utf8')) as T;
}

export function fixtureBytes(name: string): Uint8Array {
  const buffer = readFileSync(new URL(`../fixtures/${name}`, import.meta.url));
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

export function fromJson(value: TensorJson, dtype: DType = 'float32'): Tensor {
  return tensor(value.data, { shape: value.shape, dtype });
}

export function ints(rows: number[][]): Tensor {
  return tensor(rows.flat(), { shape: [rows.length, rows[0]!.length], dtype: 'int64' });
}
