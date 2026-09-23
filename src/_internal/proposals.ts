/**
 * Owned text generation shared by cognitive tools; outputs remain proposals
 * (Python ``tensorcode/_internal/proposals.py``). ``conversationContext`` and
 * ``conversationBlock`` live in the core ``conversation.ts``.
 */
import { noGrad } from '../nn/autograd.js';
import type { Tensor } from '../nn/tensor.js';
import { ValueError } from '../errors.js';
import { isPlainObject, pythonJsonDumps, sha256Hex, type JsonObject, type JsonValue } from './json.js';
import { conversationBlock, conversationContext } from './conversation.js';
import { casefold } from './text/casefold.js';
import { withEvalModes } from './memory/learned.js';
import type { Chatbot } from '../tools/chatbot.js';

export { conversationBlock, conversationContext };

export type ProposalTaskKey = 'question' | 'goal' | (string & {});

export interface ProposalRecord extends JsonObject {
  id: string;
  text: string;
  origin: 'generated';
  generated_by: string;
  proposal_template_version: number;
  generator_identity_kind: string;
  generator_configuration_fingerprint: string;
  generator_foundation: JsonValue;
  epistemic_status: string;
  source_ids: string[];
  source_reference_kind: string;
  input_truncated: boolean;
}

function isTemplateVersion(value: unknown): value is 1 | 2 {
  return value === 1 || value === 2;
}

/** Keep dialogue contextual and identified evidence factual; exclude targets. */
export function proposalPrompt(inputs: unknown, taskKey: ProposalTaskKey, options: { templateVersion?: number } = {}): string {
  const templateVersion = options.templateVersion ?? 1;
  if (!isPlainObject(inputs) || typeof inputs[taskKey] !== 'string' || !(inputs[taskKey] as string).trim()) {
    throw new ValueError(`${taskKey} must be nonempty text`);
  }
  if (!isTemplateVersion(templateVersion) || (templateVersion === 2 && taskKey !== 'question')) {
    throw new ValueError('unsupported proposal template version for task');
  }
  const evidence = inputs.evidence ?? [];
  if (!Array.isArray(evidence)) throw new ValueError('evidence must be a list');
  const ids = new Set<string>();
  for (const item of evidence) {
    if (!isPlainObject(item) || typeof item.source_id !== 'string' || !item.source_id || ids.has(item.source_id)
      || typeof item.text !== 'string' || !item.text.trim()) {
      throw new ValueError('evidence requires unique source_id and nonempty text');
    }
    ids.add(item.source_id);
  }
  let instruction = taskKey === 'question'
    ? 'Generate one declarative candidate explanation or answer to the question using the supplied evidence.'
    : 'Generate one proposed plan for the goal using the supplied evidence.';
  instruction += ' This is an uncertain proposal. Do not fabricate observations. Return only the candidate text.\n';
  if (templateVersion === 2) {
    instruction = 'Answer the question using only the supplied evidence. Write the answer as one complete sentence. '
      + 'Do not include JSON or repeat the evidence.\n';
  }
  const context = conversationContext(inputs);
  const payload = {
    [taskKey]: inputs[taskKey],
    evidence: (evidence as Record<string, unknown>[]).map((item) => ({ source_id: item.source_id, text: item.text })),
  };
  return instruction + conversationBlock(context) + pythonJsonDumps(payload, { ensureAscii: false });
}

function tokenCount(generator: Chatbot, text: string): number {
  return generator.tokenizer.encode(text).inputIds[0]!.length;
}

function budgetCheck(generator: Chatbot, inputs: Record<string, unknown>, prompt: string): void {
  if (conversationContext(inputs).length && tokenCount(generator, prompt) > (generator.config.max_input_tokens as number)) {
    throw new ValueError('Conversation and source evidence exceed proposal token budget; increase max_input_tokens or reduce conversation context');
  }
}

/** Python ``' '.join(text.split())``. */
function collapseWhitespace(text: string): string {
  return text.split(/\s+/u).filter(Boolean).join(' ');
}

/**
 * Generate up to ``count`` distinct candidate texts with the owned generator
 * (beam search, one beam per requested candidate). Records are proposals with
 * explicit generator provenance, never evidence.
 */
export function generateProposals(generator: Chatbot | null, inputs: Record<string, unknown>, options: {
  taskKey: ProposalTaskKey; count?: number; kind?: string; maxCount?: number; templateVersion?: number;
}): ProposalRecord[] {
  if (generator === null || generator === undefined) {
    throw new ValueError('proposal generation capability is not configured; configure an owned generator');
  }
  const { taskKey } = options;
  const count = options.count ?? 3;
  const kind = options.kind ?? 'hypothesis';
  const maxCount = options.maxCount ?? 16;
  const templateVersion = options.templateVersion ?? 1;
  if (typeof maxCount !== 'number' || !Number.isInteger(maxCount) || maxCount < 1
    || typeof count !== 'number' || !Number.isInteger(count) || count < 1 || count > maxCount) {
    throw new ValueError(`count must be an integer between 1 and ${maxCount}`);
  }
  const prompt = proposalPrompt(inputs, taskKey, { templateVersion });
  budgetCheck(generator, inputs, prompt);
  const texts: unknown = withEvalModes(generator, () => noGrad(() => {
    const state = generator.encodeWorkspace([prompt]);
    const tokens = generator.decoder.call(state, {
      context: {
        max_new_tokens: generator.config.max_new_tokens, do_sample: false,
        num_beams: count, num_return_sequences: count, return_dict_in_generate: false,
      },
    }) as Tensor;
    return generator.tokenizer.batchDecode(tokens, { skipSpecialTokens: true });
  }));
  if (!Array.isArray(texts) || texts.length > count || texts.some((text) => typeof text !== 'string')) {
    throw new ValueError('generator returned malformed text sequences');
  }
  const fingerprint = generator.fingerprint;
  const identity = sha256Hex(pythonJsonDumps({
    generator: fingerprint, proposal_template_version: templateVersion, task_key: taskKey,
  }, { sortKeys: true }));
  const foundation = (generator.configuration().foundation ?? null) as JsonValue;
  const truncated = tokenCount(generator, prompt) > (generator.config.max_input_tokens as number);
  const sourceIds = ((inputs.evidence ?? []) as Record<string, unknown>[]).map((item) => item.source_id as string);
  const records: ProposalRecord[] = [];
  const seen = new Set<string>();
  for (const raw of texts as string[]) {
    const text = raw.trim();
    const normalized = casefold(collapseWhitespace(text));
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    records.push({
      id: `${kind}-${sha256Hex(`${prompt}\n${text}`).slice(0, 20)}`, text,
      origin: 'generated', generated_by: identity,
      proposal_template_version: templateVersion,
      generator_identity_kind: 'configuration_and_prompt_template_fingerprint',
      generator_configuration_fingerprint: fingerprint,
      generator_foundation: foundation === null ? null : JSON.parse(JSON.stringify(foundation)) as JsonValue,
      epistemic_status: kind,
      source_ids: [...sourceIds],
      source_reference_kind: 'generation_context',
      input_truncated: truncated,
    });
  }
  return records;
}

/** Teacher-forced loss of the owned generator on explicit target texts. */
export function proposalLoss(generator: Chatbot | null, inputs: Record<string, unknown>, targets: unknown, options: {
  taskKey: ProposalTaskKey; templateVersion?: number;
}): Tensor {
  if (generator === null || generator === undefined) {
    throw new ValueError('proposal generation capability is not configured; configure an owned generator');
  }
  const values = typeof targets === 'string' ? [targets] : targets;
  if (!Array.isArray(values) || !values.length || values.some((item) => typeof item !== 'string' || !item.trim())) {
    throw new ValueError('targets must be nonempty generation text');
  }
  const prompt = proposalPrompt(inputs, options.taskKey, { templateVersion: options.templateVersion ?? 1 });
  budgetCheck(generator, inputs, prompt);
  return generator.lossBatch(new Array<string>(values.length).fill(prompt), values as string[]);
}
