/**
 * Structured text operations over an explicit model provider: classify, decide
 * and score one support ticket, ask several questions at once, and trace a
 * message composition.
 *
 *     npm run build
 *     node examples/supportTriage.ts
 *     OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=my-model node examples/supportTriage.ts
 *
 * Offline, a small keyword provider stands in for a language model so the
 * example is deterministic. It implements the same `Model` protocol
 * (`complete(ModelRequest) -> ModelOutput`) as the HTTP adapters. When
 * `OPENAI_BASE_URL` and `OPENAI_MODEL` are set, the same operations call any
 * OpenAI-compatible endpoint instead (asynchronously). Every structured
 * response is validated; nothing is repaired or invented.
 */
import { trace } from 'tensorcode';
import { OpenAICompatibleModel } from 'tensorcode/integrations';
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

if (remote) {
  // HTTP providers are asynchronous: use `acall` / `aask`.
  console.log('route:', await route.acall(ticket));
  console.log('answers:', await text.aask(ticket, { route, action, urgency }));
} else {
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
