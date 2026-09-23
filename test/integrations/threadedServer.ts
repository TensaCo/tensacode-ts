/**
 * A loopback HTTP server running in its own worker thread, standing in for the
 * Python tests' ``ThreadingHTTPServer``. Synchronous provider calls block the
 * test's thread, so the server cannot share its event loop.
 */
import { Worker } from 'node:worker_threads';

export interface RecordedRequest {
  path: string;
  headers: Record<string, string>;
  json: any;
}

/** ``(request) => [status, payload, delaySeconds, headers?]``; must be self-contained (it runs in the worker). */
export type ThreadedResponder = (request: RecordedRequest) => [number, unknown, number, Record<string, string>?];

export interface ThreadedServer {
  url: string;
  requests(): Promise<RecordedRequest[]>;
  close(): Promise<void>;
}

const program = `
var __name = (target) => target;
const { parentPort, workerData } = require('node:worker_threads');
const { createServer } = require('node:http');
const responder = (0, eval)('(' + workerData.responder + ')');
const requests = [];
const server = createServer((request, response) => {
  const chunks = [];
  request.on('data', (chunk) => chunks.push(chunk));
  request.on('end', () => {
    const body = Buffer.concat(chunks).toString('utf8');
    const recorded = { path: request.url, headers: { ...request.headers }, json: body ? JSON.parse(body) : null };
    requests.push(recorded);
    const [status, payload, delay, headers] = responder(recorded);
    setTimeout(() => {
      const encoded = Buffer.from(JSON.stringify(payload));
      response.writeHead(status, { 'content-type': 'application/json', 'content-length': encoded.length, ...(headers || {}) });
      response.end(encoded);
    }, (delay || 0) * 1000);
  });
});
server.listen(0, '127.0.0.1', () => parentPort.postMessage({ port: server.address().port }));
parentPort.on('message', (message) => {
  if (message === 'requests') parentPort.postMessage({ requests });
  if (message === 'close') { server.closeAllConnections(); server.close(() => process.exit(0)); }
});
`;

export async function threadedServer(responder: ThreadedResponder): Promise<ThreadedServer> {
  const worker = new Worker(program, { eval: true, workerData: { responder: responder.toString() } });
  const port = await new Promise<number>((resolve, reject) => {
    worker.once('message', (message: { port: number }) => resolve(message.port));
    worker.once('error', reject);
  });
  return {
    url: `http://127.0.0.1:${port}`,
    requests: () => new Promise((resolve) => {
      worker.once('message', (message: { requests: RecordedRequest[] }) => resolve(message.requests));
      worker.postMessage('requests');
    }),
    close: async () => {
      worker.postMessage('close');
      await new Promise((resolve) => worker.once('exit', resolve));
    },
  };
}
