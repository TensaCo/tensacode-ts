/** Optional real-browser acceptance test. CHROME selects a Chromium executable. */
import { build } from 'esbuild';
import { spawn } from 'node:child_process';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url));
const dir = await mkdtemp(join(tmpdir(), 'tensorcode-browser-'));
let chrome, ws;
try {
  await build({ entryPoints: [join(root, 'test/program/browser-entry.ts')], bundle: true, platform: 'browser', format: 'iife', outfile: join(dir, 'bundle.js') });
  await writeFile(join(dir, 'index.html'), '<!doctype html><script src="bundle.js"></script>');
  chrome = spawn(process.env.CHROME ?? 'google-chrome', ['--headless', '--no-sandbox', '--disable-gpu-sandbox', '--enable-unsafe-webgpu', '--use-angle=swiftshader', '--remote-debugging-port=0', `--user-data-dir=${join(dir, 'profile')}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Chrome startup timed out')), 20000);
    chrome.on('error', reject);
    chrome.stderr.on('data', data => { const match = String(data).match(/DevTools listening on (ws:\/\/\S+)/); if (match) { clearTimeout(timer); resolve(match[1]); } });
  });
  ws = new WebSocket(url); await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  let sequence = 0; const pending = new Map();
  ws.onmessage = event => { const data = JSON.parse(event.data); if (data.id) { const p = pending.get(data.id); pending.delete(data.id); if (data.error) p.reject(new Error(JSON.stringify(data.error))); else p.resolve(data.result); } };
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => { const id = ++sequence; pending.set(id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params, sessionId })); });
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  await send('Page.enable', {}, sessionId);
  await send('Page.navigate', { url: `file://${join(dir, 'index.html')}` }, sessionId);
  const result = await send('Runtime.evaluate', { expression: 'new Promise((resolve,reject) => { const wait = () => { if (globalThis.smoke) globalThis.smoke.then(resolve,reject); else setTimeout(wait, 10); }; wait(); setTimeout(() => reject(new Error("smoke timed out")), 30000); })', awaitPromise: true, returnByValue: true }, sessionId);
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  console.log(JSON.stringify(result.result.value));
  if (!result.result.value.cpu || !result.result.value.wasm) process.exitCode = 1;
} finally {
  ws?.close();
  if (chrome) { chrome.kill(); await new Promise(resolve => chrome.once('exit', resolve)); }
  await rm(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
