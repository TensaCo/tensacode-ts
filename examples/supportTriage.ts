/**
 * Structured text operations over an explicit model provider (Python
 * ``examples/support_triage.py`` plus an offline tour).
 *
 *     npm run build
 *     node examples/supportTriage.ts
 *     OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=my-model node examples/supportTriage.ts
 *     node examples/supportTriage.ts --input tickets.jsonl --policy policy.txt \
 *       --label billing --label incident --label question --base-url http://localhost:8000/v1 --model local-model
 *
 * With `--input`, this is the Python CLI: route real support tickets (UTF-8
 * JSONL, one `{"id": "case-1", "text": "..."}` per line) with a
 * caller-supplied policy and labels, writing JSONL in the same order. Missing
 * provider confidence stays absent; missing distributions stay null.
 *
 * Without arguments it classifies, decides and scores one ticket, asks several
 * questions at once and traces a message composition. Offline, a small keyword
 * provider stands in for a language model so the tour is deterministic. It
 * implements the same `Model` protocol (`complete(ModelRequest) -> ModelOutput`)
 * as the HTTP adapters. When `OPENAI_BASE_URL` and `OPENAI_MODEL` are set, the
 * same operations call any OpenAI-compatible endpoint instead (synchronously,
 * as in Python; `acall`/`aask` are the asynchronous forms). Every structured
 * response is validated; nothing is repaired or invented.
 */
import { closeSync, openSync, readFileSync, readSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { trace } from 'tensorcode';
import { OpenAICompatibleModel, type OpenAIApi } from 'tensorcode/integrations';
import * as text from 'tensorcode/ops/text';

/** Deterministic keyword rules that answer TensorCode's structured schemas. */
class KeywordModel implements text.Model {
  complete(request: text.ModelRequest): text.ModelOutput {
    const content = request.messages.map((message) => String(message.content)).join(' ').toLowerCase();
    const billing = /charge|refund|invoice/.test(content);
    switch (request.schemaName) {
      case 'tensorcode.classify':
        return new text.ModelOutput({
          structured: { label: billing ? 'billing' : 'technical', distribution: billing ? { billing: 0.9, technical: 0.1 } : { billing: 0.2, technical: 0.8 }, abstained: false },
        });
      case 'tensorcode.decide':
        return new text.ModelOutput({ structured: { choice: billing ? 'refund' : 'escalate', abstained: false } });
      case 'tensorcode.score':
        return new text.ModelOutput({ structured: { score: /twice|urgent|down/.test(content) ? 2 : 0, abstained: false } });
      default:
        return new text.ModelOutput({ text: billing ? 'We are refunding the duplicate charge.' : 'We are looking into it.' });
    }
  }
}

function tour(): void {
  const endpoint = process.env.OPENAI_BASE_URL;
  const remote = endpoint && process.env.OPENAI_MODEL
    ? new OpenAICompatibleModel({ baseUrl: endpoint, model: process.env.OPENAI_MODEL, apiKey: process.env.OPENAI_API_KEY ?? null })
    : null;
  const model: text.ExternalModel = remote ?? new KeywordModel();

  const route = text.Classify.fromModel(model, {
    labels: ['billing', 'technical'],
    descriptions: { billing: 'payments, charges and refunds' },
    instructions: 'Route the support ticket',
  });
  const action = text.Decide.fromModel(model, { options: ['refund', 'escalate', 'reply'], instructions: 'Choose the next action' });
  const urgency = text.Score.fromModel(model, { rubric: ['can wait', 'this week', 'today'], instructions: 'Assess urgency' });

  const ticket = [new text.Message('user', 'I was charged twice for my subscription this month.')];

  {
    const classified = route.call(ticket);
    console.log('route:', classified.label, classified.distribution);
    console.log('action:', action.call(ticket).choice);
    console.log('urgency:', urgency.call(ticket).value);

    // Several named questions about the same messages.
    const answers = text.ask(ticket, { route, action, urgency });
    console.log('ask:', { route: answers.route.label, action: answers.action.choice, urgency: answers.urgency.value });

    // Message compositions are traced like any other operation.
    const encode = new text.TextEncoder();
    const respond = text.Transform.fromModel(model);
    const decode = new text.TextDecoder();
    const session = trace();
    const reply = session.run(() => decode.call(respond.call(encode.call('Please refund the duplicate invoice.'))));
    console.log('reply:', reply, `(${session.calls.length} traced calls)`);
  }
}

interface Ticket { id: string; text: string }

function loadTickets(path: string, maxTickets: number, maxTicketChars: number): Ticket[] {
  if (maxTickets < 1 || maxTicketChars < 1) throw new Error('ticket limits must be positive');
  const tickets: Ticket[] = [];
  const seen = new Set<string>();
  readFileSync(path, 'utf8').split('\n').forEach((line, index) => {
    const number = index + 1;
    if (!line.trim()) return;
    if (tickets.length >= maxTickets) throw new Error(`input exceeds max_tickets=${maxTickets}`);
    let record: unknown;
    try {
      record = JSON.parse(line);
    } catch {
      throw new Error(`invalid JSON on line ${number}`);
    }
    if (record === null || typeof record !== 'object' || Array.isArray(record) || Object.keys(record).sort().join() !== 'id,text') {
      throw new Error(`line ${number} must contain only id and text`);
    }
    const { id, text: body } = record as Record<string, unknown>;
    if (typeof id !== 'string' || !id.trim()) throw new Error(`line ${number} id must be a nonempty string`);
    if (seen.has(id)) throw new Error(`duplicate ticket id: ${id}`);
    if (typeof body !== 'string' || !body.trim()) throw new Error(`line ${number} text must be a nonempty string`);
    if (Array.from(body).length > maxTicketChars) throw new Error(`line ${number} exceeds max_ticket_chars=${maxTicketChars}`);
    seen.add(id);
    tickets.push({ id, text: body });
  });
  return tickets;
}

function routeTickets(tickets: Ticket[], labels: string[], policy: string, model: text.ExternalModel): Record<string, unknown>[] {
  if (labels.length < 2 || new Set(labels).size !== labels.length || !labels.every((label) => label.trim())) {
    throw new Error('labels must contain at least two unique nonempty strings');
  }
  if (!policy.trim()) throw new Error('policy must be nonempty caller-supplied text');
  const classify = text.Classify.fromModel(model, {
    labels,
    instructions: `Route the supplied support ticket under this caller-owned policy. Abstain when the policy does not support one route.\n\nPOLICY:\n${policy}`,
  });
  const values = tickets.map((ticket) => new text.TextEncoder().call(ticket.text));
  const results = classify.batch(values);
  return tickets.map((ticket, index) => {
    const result = results[index]!;
    const record: Record<string, unknown> = {
      id: ticket.id, route: result.label, abstained: result.abstained,
      distribution: result.distribution === null ? null : { ...result.distribution },
    };
    if (result.confidence !== null) record.confidence = result.confidence;
    return record;
  });
}

/** Python ``json.dumps(record, ensure_ascii=False)`` (", " and ": " separators). */
function pythonJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(', ')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.entries(value).map(([key, item]) => `${JSON.stringify(key)}: ${pythonJson(item)}`).join(', ')}}`;
  }
  if (typeof value === 'number' && Number.isInteger(value) && !Object.is(value, -0)) return String(value);
  return JSON.stringify(value);
}

function cli(argv: string[]): void {
  const { values } = parseArgs({
    args: argv,
    options: {
      input: { type: 'string' }, policy: { type: 'string' }, label: { type: 'string', multiple: true }, 'base-url': { type: 'string' },
      model: { type: 'string' }, api: { type: 'string', default: 'chat_completions' }, 'api-key-env': { type: 'string', default: 'OPENAI_API_KEY' },
      timeout: { type: 'string', default: '30' }, 'max-tickets': { type: 'string', default: '1000' },
      'max-ticket-chars': { type: 'string', default: '10000' }, 'max-policy-chars': { type: 'string', default: '20000' }, output: { type: 'string' },
    },
  });
  if (!values.input || !values.policy || !values.label || !values['base-url'] || !values.model) {
    throw new Error('--input, --policy, --label, --base-url and --model are required');
  }
  const maxPolicyChars = Number(values['max-policy-chars']);
  if (maxPolicyChars < 1) throw new Error('--max-policy-chars must be positive');
  const buffer = Buffer.alloc(maxPolicyChars * 4 + 4);
  const descriptor = openSync(values.policy, 'r');
  let size: number;
  try {
    size = readSync(descriptor, buffer, 0, buffer.length, 0);
  } finally {
    closeSync(descriptor);
  }
  const policy = Array.from(buffer.subarray(0, size).toString('utf8'));
  if (policy.length > maxPolicyChars) throw new Error(`policy exceeds --max-policy-chars=${maxPolicyChars}`);
  const tickets = loadTickets(values.input, Number(values['max-tickets']), Number(values['max-ticket-chars']));
  const model = new OpenAICompatibleModel({
    baseUrl: values['base-url'], model: values.model, api: values.api as OpenAIApi,
    apiKey: process.env[values['api-key-env']!] ?? null, timeout: Number(values.timeout),
  });
  const records = routeTickets(tickets, values.label, policy.join(''), model);
  const rendered = records.map((record) => `${pythonJson(record)}\n`).join('');
  if (values.output) writeFileSync(values.output, rendered, 'utf8');
  else process.stdout.write(rendered);
}

if (process.argv.includes('--input')) cli(process.argv.slice(2));
else tour();
