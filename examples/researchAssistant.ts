/**
 * Bounded research over caller-supplied local text documents (Python
 * ``examples/research_assistant.py``).
 *
 *     npm run build
 *     node examples/researchAssistant.ts ./handbook "How do I rotate the API key?" \
 *       --base-url http://localhost:8000/v1 --model local-model
 *
 * A `text.Decide` chooser selects one supplied action per step inside
 * `tools/actions.actionLoop`: `search` ranks local documents by term counts,
 * each `read:<id>` action opens only its fixed file, and `finish` answers only
 * from read excerpts and must cite read source IDs. Everything is bounded:
 * steps, documents and characters per document. Selected file contents are sent
 * to the endpoint you configure.
 */
import { lstatSync, readdirSync, readFileSync, realpathSync } from 'node:fs';
import { extname, join, relative, resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { OpenAICompatibleModel } from 'tensorcode/integrations';
import * as text from 'tensorcode/ops/text';
import { ActionOutcome, actionLoop, type ActionRequest } from 'tensorcode/tools/actions';

const SUPPORTED_SUFFIXES = new Set(['.txt', '.md', '.rst', '.csv', '.json']);

interface Document { source_id: string; relative_path: string }
interface Source { source_id: string; relative_path: string; text: string }
interface ResearchState { question: string; matches: string[]; sources: Source[]; answer: string | null }

function documents(root: string, maxDocuments: number): Document[] {
  const paths: string[] = [];
  const walk = (directory: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      const rel = relative(root, path);
      if (entry.isDirectory() && !entry.isSymbolicLink()) walk(path);
      else if (entry.isFile() && !rel.split(sep).some((part) => part.startsWith('.'))
        && SUPPORTED_SUFFIXES.has(extname(entry.name).toLowerCase()) && realpathSync(path).startsWith(root + sep)) {
        paths.push(rel.split(sep).join('/'));
      }
    }
  };
  walk(root);
  paths.sort();
  if (!paths.length) throw new Error('document directory contains no supported documents');
  if (paths.length > maxDocuments) {
    throw new Error(`document directory contains ${paths.length} supported documents; increase max_documents above ${maxDocuments} explicitly`);
  }
  return paths.map((path, index) => ({ source_id: `doc-${String(index + 1).padStart(4, '0')}`, relative_path: path }));
}

function read(root: string, document: Document, maxChars: number): string {
  const path = realpathSync(join(root, document.relative_path));
  if (!path.startsWith(root + sep) || !lstatSync(path).isFile()) throw new Error(`document escaped the configured directory: ${document.relative_path}`);
  return Array.from(readFileSync(path, 'utf8')).slice(0, maxChars).join('');
}

/** Python ``json.dumps(value, sort_keys=True)``. */
function pythonJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(', ')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${asciiJson(key)}: ${pythonJson((value as Record<string, unknown>)[key])}`).join(', ')}}`;
  }
  return typeof value === 'string' ? asciiJson(value) : JSON.stringify(value);
}

function asciiJson(value: string): string {
  return JSON.stringify(value).replace(/[\u0080-\uffff]/g, (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
}

async function research(docsDir: string, question: string, options: {
  model: text.ExternalModel; maxSteps?: number; maxDocuments?: number; maxChars?: number;
}) {
  const { model, maxSteps = 6, maxDocuments = 40, maxChars = 8_000 } = options;
  const root = realpathSync(resolve(docsDir));
  if (!lstatSync(root).isDirectory()) throw new Error('docs_dir must be a directory');
  if (!question.trim()) throw new Error('question must be nonempty');
  if (!Number.isInteger(maxDocuments) || maxDocuments < 1) throw new Error('max_documents must be a positive integer');
  if (!Number.isInteger(maxChars) || maxChars < 1) throw new Error('max_chars must be a positive integer');
  const docs = documents(root, maxDocuments);

  const search = (state: ResearchState): ActionOutcome<ResearchState> => {
    const terms = [...new Set(state.question.toLowerCase().match(/[a-z0-9]+/g) ?? [])];
    const ranked = docs.map((document) => {
      const content = read(root, document, maxChars * 4).toLowerCase();
      const score = terms.reduce((total, term) => total + content.split(term).length - 1, 0);
      return [-score, document.relative_path, document.source_id] as const;
    }).sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
    const matches = ranked.slice(0, Math.min(5, ranked.length)).map((item) => item[2]);
    return new ActionOutcome({ ...state, matches }, { matches });
  };
  const readAction = (document: Document) => (state: ResearchState): ActionOutcome<ResearchState> => {
    if (state.sources.some((source) => source.source_id === document.source_id)) {
      return new ActionOutcome(state, { source_id: document.source_id, cached: true });
    }
    const source = { source_id: document.source_id, relative_path: document.relative_path, text: read(root, document, maxChars) };
    return new ActionOutcome({ ...state, sources: [...state.sources, source] }, { source_id: source.source_id, relative_path: source.relative_path });
  };
  const finish = (state: ResearchState): ActionOutcome<ResearchState> => {
    if (!state.sources.length) return new ActionOutcome(state, { answered: false, sources: [] }, true);
    const evidence = state.sources.map((source) => `[SOURCE ${source.source_id}: ${source.relative_path}]\n${source.text}`).join('\n\n');
    const prompt = 'Answer the question using only the supplied source excerpts. '
      + 'Cite source IDs in square brackets. If the excerpts do not answer it, say so.\n\n'
      + `Question: ${state.question}\n\n${evidence}`;
    const answer = new text.TextDecoder().call(text.Transform.fromModel(model).call([new text.Message('user', prompt)]));
    const known = new Set(state.sources.map((source) => source.source_id));
    const citations = new Set([...answer.matchAll(/\[([^[\]]+)\]/g)].map((match) => match[1]!).filter((value) => value.startsWith('doc-')));
    const unknown = [...citations].filter((value) => !known.has(value)).sort();
    if (unknown.length) throw new Error(`answer cited unknown source IDs: [${unknown.map((value) => `'${value}'`).join(', ')}]`);
    if (!citations.size) throw new Error('answer must cite at least one read source ID');
    return new ActionOutcome({ ...state, answer }, { answered: true, sources: [...known].sort() }, true);
  };

  const actions: Record<string, (state: ResearchState) => ActionOutcome<ResearchState>> = { search };
  for (const document of docs) actions[`read:${document.source_id}`] = readAction(document);
  actions.finish = finish;
  const decide = text.Decide.fromModel(model, {
    options: Object.keys(actions),
    instructions: 'Choose exactly one supplied action. Search ranks local documents; read actions '
      + 'open only their fixed file; finish answers only from read excerpts.',
  });
  const choose = (request: ActionRequest<ResearchState>) => {
    const state = request.state;
    const manifest = {
      question: state.question, step: request.step, documents: docs, search_matches: [...state.matches],
      read_sources: state.sources.map((source) => ({
        source_id: source.source_id, relative_path: source.relative_path, excerpt: Array.from(source.text).slice(0, 1000).join(''),
      })),
      options: [...request.options],
    };
    return decide.call([new text.Message('user', pythonJson(manifest))]);
  };
  const initial: ResearchState = { question: question.trim(), matches: [], sources: [], answer: null };
  const result = await actionLoop<ResearchState>({ chooser: choose, actions, maxSteps }).call(initial);
  return { answer: result.state.answer, stopReason: result.stopReason, sources: result.state.sources, receipts: result.receipts };
}

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    'base-url': { type: 'string' }, model: { type: 'string' }, 'api-key-env': { type: 'string', default: 'OPENAI_API_KEY' },
    'max-steps': { type: 'string', default: '6' }, 'max-documents': { type: 'string', default: '40' }, 'max-chars': { type: 'string', default: '8000' },
  },
});
const [docsDir, question] = positionals;
if (!docsDir || question === undefined || !values['base-url'] || !values.model) {
  throw new Error('usage: researchAssistant.ts DOCS_DIR QUESTION --base-url URL --model NAME');
}
const model = new OpenAICompatibleModel({ baseUrl: values['base-url'], model: values.model, apiKey: process.env[values['api-key-env']!] ?? null });
const report = await research(docsDir, question, {
  model, maxSteps: Number(values['max-steps']), maxDocuments: Number(values['max-documents']), maxChars: Number(values['max-chars']),
});
console.log(JSON.stringify({
  answer: report.answer, stop_reason: report.stopReason, sources: report.sources.map((source) => source.relative_path),
  receipts: report.receipts.map((receipt) => ({ step: receipt.step, action: receipt.action, effect: receipt.effect })),
}, null, 2));
