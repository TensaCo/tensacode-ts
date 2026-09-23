/**
 * Image sources: reading files and encoded strings the way transformers'
 * ``load_image_as_tensor`` / ``load_image`` resolve them (an ``http(s)://``
 * URL, an existing file path, or base64 text with an optional
 * ``data:image/...;base64,`` prefix), plus Python's lenient
 * ``base64.decodebytes``.
 */
import { blockingCall } from '../../integrations/blocking.js';
import { ValueError } from '../../errors.js';

interface FsModule {
  statSync(path: string, options?: { throwIfNoEntry?: boolean }): { isFile(): boolean } | undefined;
  readFileSync(path: string): Uint8Array;
}

function nodeFs(): FsModule | null {
  try {
    const processLike = (globalThis as { process?: { getBuiltinModule?: (name: string) => unknown } }).process;
    const module = processLike?.getBuiltinModule?.('node:fs') as FsModule | undefined;
    return module && typeof module.readFileSync === 'function' ? module : null;
  } catch {
    return null;
  }
}

/** ``os.path.isfile(path)`` (``false`` without a file system). */
export function isFile(path: string): boolean {
  const fs = nodeFs();
  if (!fs || !path) return false;
  try {
    return fs.statSync(path, { throwIfNoEntry: false })?.isFile() ?? false;
  } catch {
    return false;
  }
}

/** Read a file's bytes (Node.js only). */
export function readFileBytes(path: string): Uint8Array {
  const fs = nodeFs();
  if (!fs) throw new ValueError('reading image files requires a Node.js-compatible file system; pass encoded bytes instead');
  const buffer = fs.readFileSync(path);
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

const BASE64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const BASE64_TABLE = new Int16Array(256).fill(-1);
for (let i = 0; i < BASE64.length; i += 1) BASE64_TABLE[BASE64.charCodeAt(i)] = i;

/**
 * CPython 3.13 ``binascii.a2b_base64`` in non-strict mode
 * (``base64.decodebytes``): characters outside the alphabet and ``=`` pads
 * are skipped; a trailing partial quad needs enough pads.
 */
export function decodeBase64(text: string): Uint8Array {
  const out: number[] = [];
  let quadPosition = 0;
  let leftChar = 0;
  let pads = 0;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code === 0x3d) { // '='
      pads += 1;
      continue;
    }
    const value = code < 256 ? BASE64_TABLE[code]! : -1;
    if (value < 0) continue; // Non-alphabet data (including UTF-8 bytes of non-ASCII text).
    pads = 0;
    switch (quadPosition) {
      case 0:
        quadPosition = 1;
        leftChar = value;
        break;
      case 1:
        quadPosition = 2;
        out.push(((leftChar << 2) | (value >> 4)) & 0xff);
        leftChar = value & 0x0f;
        break;
      case 2:
        quadPosition = 3;
        out.push(((leftChar << 4) | (value >> 2)) & 0xff);
        leftChar = value & 0x03;
        break;
      default:
        quadPosition = 0;
        out.push(((leftChar << 6) | value) & 0xff);
        leftChar = 0;
        break;
    }
  }
  if (quadPosition === 1) {
    const characters = Math.floor(out.length / 3) * 4 + 1;
    throw new ValueError(`Invalid base64-encoded string: number of data characters (${characters}) cannot be 1 more than a multiple of 4`);
  }
  if (quadPosition !== 0 && quadPosition + pads < 4) throw new ValueError('Incorrect padding');
  return Uint8Array.from(out);
}

export function isUrl(source: string): boolean {
  return source.startsWith('http://') || source.startsWith('https://');
}

function incorrectSource(source: string, error: unknown): ValueError {
  return new ValueError(
    'Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, '
    + `or a base64 encoded string. Got ${source}. Failed with ${(error as Error).message ?? String(error)}`,
  );
}

/**
 * ``httpx.get(url, timeout=timeout, follow_redirects=True).content``, blocking
 * the calling thread while a worker thread fetches (as Python's call blocks).
 */
function fetchBlocking(source: string, timeout: number | null): Uint8Array {
  const reply = blockingCall<{ outcome: string; status?: number; bytes?: Uint8Array; reason?: string }>({
    kind: 'http', method: 'GET', url: source, redirect: 'follow', timeoutMs: timeout ? timeout * 1000 : null,
  }, { waitSeconds: timeout ? timeout + 30 : null });
  if (reply.outcome === 'timeout') throw new Error(`timed out fetching ${source}`, { cause: 'timeout' });
  if (reply.outcome !== 'response') throw new Error(`fetching ${source} failed: ${reply.reason ?? reply.outcome}`);
  const bytes = reply.bytes!;
  if (!bytes.length) throw new ValueError('both buffer length (0) and count (-1) must not be 0');
  return bytes;
}

/**
 * Encoded bytes for a string source (``load_image_as_tensor``): an
 * ``http(s)://`` URL (fetched while the caller blocks, see
 * {@link fetchSourceBytes} for the asynchronous form), a file path or base64
 * (``data:image/...`` prefix allowed).
 */
export function sourceBytes(source: string, options: { timeout?: number | null } = {}): Uint8Array {
  if (isUrl(source)) return fetchBlocking(source, options.timeout ?? null);
  if (isFile(source)) return readFileBytes(source);
  let text = source;
  if (text.startsWith('data:image/')) text = text.split(',')[1] ?? '';
  let bytes: Uint8Array;
  try {
    bytes = decodeBase64(text);
  } catch (error) {
    throw incorrectSource(source, error);
  }
  if (!bytes.length) throw new ValueError('both buffer length (0) and count (-1) must not be 0');
  return bytes;
}

/** Like {@link sourceBytes}, also fetching ``http(s)://`` URLs (redirects followed). */
export async function fetchSourceBytes(source: string, options: { timeout?: number | null } = {}): Promise<Uint8Array> {
  if (!isUrl(source)) return sourceBytes(source);
  const controller = options.timeout ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), options.timeout! * 1000) : null;
  try {
    const response = await fetch(source, { redirect: 'follow', ...(controller ? { signal: controller.signal } : {}) });
    const buffer = new Uint8Array(await response.arrayBuffer());
    if (!buffer.length) throw new ValueError('both buffer length (0) and count (-1) must not be 0');
    return buffer;
  } finally {
    if (timer) clearTimeout(timer);
  }
}
