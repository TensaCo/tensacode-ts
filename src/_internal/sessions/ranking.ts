/**
 * Private independent ranking receipt history (Python
 * ``tensorcode/_internal/sessions/ranking.py``). Each call supplies its
 * complete evidence.
 */
import { readFile } from 'node:fs/promises';
import { ValueError } from '../../errors.js';
import type { Context } from '../../ops/base.js';
import { canonicalJson, deepCopy, isPlainObject, jsonEqual, parseJsonStrict, pythonJsonDumps, sha256Hex, type JsonObject, type JsonValue } from '../json.js';
import { atomicWriteFile } from '../files.js';
import { qualifiedName } from '../identity.js';
import { proposalPrompt } from '../proposals.js';
import { sessionFloatKeys } from './floats.js';
import type { RankOperation } from '../ranking.js';

/** A ranking tool (Investigator, Decision, Planner) as seen by its sessions. */
export interface RankingSessionTool {
  readonly rank: RankOperation;
  readonly config: JsonObject;
  readonly verifier?: { readonly config: JsonObject } | null;
  call(value: Record<string, unknown>): JsonObject;
  configuration(): JsonObject;
}

export interface RankingHistoryItem {
  inputs: JsonObject;
  receipt: JsonObject;
}

const SCORE_KEYS = new Set(['predicted_score', 'probability', 'verifications', 'joint_verification']);

function withoutScores(candidate: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(candidate).filter(([key]) => !SCORE_KEYS.has(key)));
}

function isInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value);
}

/** Python ``math.isclose(a, b, abs_tol=...)``. */
function isClose(a: number, b: number, absTol: number): boolean {
  return Math.abs(a - b) <= Math.max(1e-9 * Math.max(Math.abs(a), Math.abs(b)), absTol);
}

export class RankingSession {
  static readonly qualifiedName: string = 'tensorcode._internal.sessions.ranking.RankingSession';
  readonly tool: RankingSessionTool;
  private historyItems: RankingHistoryItem[] = [];

  constructor(tool: RankingSessionTool) {
    this.tool = tool;
  }

  /** Independent copy of the retained ``{inputs, receipt}`` records. */
  get history(): RankingHistoryItem[] {
    return deepCopy(this.historyItems as unknown as JsonValue) as unknown as RankingHistoryItem[];
  }

  call(inputs: Record<string, unknown>, options: { context?: Context | null } = {}): JsonObject {
    if (options.context && Object.keys(options.context).length) throw new ValueError('RankingSession does not accept context');
    const snapshot = deepCopy(inputs) as JsonObject;
    const receipt = this.tool.call(snapshot);
    const key = this.tool.rank.candidatesKey;
    if (!(key in snapshot)) {
      // Persist the actual generated alternatives, never regenerate on load.
      snapshot[key] = (receipt.candidates as JsonObject[]).map((candidate) => deepCopy(withoutScores(candidate)) as JsonObject);
    }
    const last = this.historyItems[this.historyItems.length - 1];
    const previous = last ? (last.receipt.selected_id as string | null) : null;
    receipt.previous_selected_id = previous;
    receipt.revised = previous !== null && previous !== receipt.selected_id;
    this.historyItems.push({ inputs: snapshot, receipt: deepCopy(receipt) });
    return receipt;
  }

  private identity(): string {
    return sha256Hex(canonicalJson({ tool: qualifiedName(this.tool), config: this.tool.configuration() }));
  }

  async save(path: string): Promise<void> {
    const text = pythonJsonDumps({ version: 1, identity: this.identity(), history: this.historyItems }, { allowNan: false, floatKeys: sessionFloatKeys() });
    await atomicWriteFile(path, text);
  }

  static async load(path: string, tool: RankingSessionTool): Promise<RankingSession> {
    let data: unknown;
    try {
      data = parseJsonStrict(await readFile(path, 'utf8'));
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      throw new ValueError(`invalid session JSON: ${error.message}`, { cause: error });
    }
    return RankingSession.fromData(data, tool);
  }

  /** Validate a persisted session document against ``tool``. */
  static fromData(data: unknown, tool: RankingSessionTool): RankingSession {
    const result = new RankingSession(tool);
    if (!isPlainObject(data) || data.version !== 1 || data.identity !== result.identity() || !Array.isArray(data.history)) {
      throw new ValueError('session architecture identity or format mismatch');
    }
    let previous: string | null = null;
    for (const item of data.history) {
      if (!isPlainObject(item) || Object.keys(item).length !== 2 || !('inputs' in item) || !('receipt' in item)) {
        throw new ValueError('invalid session record');
      }
      const receipt = item.receipt as Record<string, unknown>;
      const inputs = item.inputs as Record<string, unknown>;
      if (isPlainObject(receipt) && receipt.abstained === true && receipt.selected_id === null
        && Array.isArray(receipt.candidates) && !receipt.candidates.length
        && isPlainObject(inputs) && Array.isArray(inputs[tool.rank.candidatesKey]) && !(inputs[tool.rank.candidatesKey] as unknown[]).length) {
        proposalPrompt(inputs, tool.rank.taskKey);
        if (!jsonEqual(receipt.evidence, inputs.evidence ?? []) || receipt.previous_selected_id !== previous
          || receipt.revised !== (previous !== null)) {
          throw new ValueError('invalid abstention receipt');
        }
        previous = null;
        continue;
      }
      const [evidence, candidates] = tool.rank.validate(inputs);
      if (!isPlainObject(receipt) || !jsonEqual(receipt.evidence, evidence)
        || !candidates.some((candidate) => candidate.id === receipt.selected_id)) {
        throw new ValueError('session receipt does not match its evidence/candidates');
      }
      const sourceIds = receipt.attention_source_ids;
      const known = new Set(evidence.map((entry) => entry.source_id));
      if (!Array.isArray(sourceIds) || sourceIds.some((source) => source !== null && !known.has(source as string))) {
        throw new ValueError('session attention source mismatch');
      }
      const recorded = receipt.candidates;
      if (!Array.isArray(recorded) || recorded.length !== candidates.length
        || recorded.some((candidate, index) => !isPlainObject(candidate)
          || !jsonEqual(withoutScores(candidate), withoutScores(candidates[index] as Record<string, unknown>)))) {
        throw new ValueError('session candidate receipt mismatch');
      }
      if (receipt.previous_selected_id !== previous || receipt.revised !== (previous !== null && previous !== receipt.selected_id)) {
        throw new ValueError('session revision history mismatch');
      }
      if ((recorded as Record<string, unknown>[]).some((candidate) => typeof candidate.predicted_score !== 'number' || !Number.isFinite(candidate.predicted_score))) {
        throw new ValueError('session contains invalid predicted scores');
      }
      for (const candidate of recorded as Record<string, unknown>[]) {
        const verified = 'verification_semantics' in receipt || (candidate.origin === 'generated' && (tool.verifier ?? null) !== null);
        if (!verified && !('verifications' in candidate)) continue;
        const checks = candidate.verifications;
        if (!Array.isArray(checks) || checks.length !== evidence.length
          || checks.some((check, index) => !isPlainObject(check) || check.source_id !== evidence[index]!.source_id)) {
          throw new ValueError('session verification source mismatch');
        }
        const extra: Record<string, unknown>[] = [];
        if (tool.config.verification_scope === 'joint') {
          const joint = candidate.joint_verification;
          if (evidence.length) {
            if (!isPlainObject(joint) || !jsonEqual(joint.source_ids, evidence.map((source) => source.source_id))
              || joint.scope !== 'joint' || !isInteger(joint.token_count) || joint.token_count < 1
              || !isInteger(joint.max_tokens) || joint.max_tokens < 1 || typeof joint.input_truncated !== 'boolean'
              || joint.input_truncated !== (joint.token_count > joint.max_tokens)) {
              throw new ValueError('invalid session joint verification coverage');
            }
            extra.push(joint);
          } else if (joint !== null && joint !== undefined) {
            throw new ValueError('invalid session joint verification without evidence');
          }
        } else if ('joint_verification' in candidate) {
          throw new ValueError('session joint verification conflicts with configured scope');
        }
        for (const check of [...checks, ...extra] as Record<string, unknown>[]) {
          const distribution = check.distribution;
          if (!isPlainObject(distribution) || Object.keys(distribution).length !== 3
            || !['support', 'contradiction', 'unknown'].every((key) => key in distribution)
            || Object.values(distribution).some((score) => typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1)
            || !isClose(Object.values(distribution as Record<string, number>).reduce((a, b) => a + b, 0), 1, 1e-5)) {
            throw new ValueError('invalid session verification distribution');
          }
          const calibrated = check.calibrated;
          const count = check.calibration_sample_count;
          if (typeof calibrated !== 'boolean' || !isInteger(count) || count < 0 || calibrated !== (count > 0)
            || check.origin !== 'model_inference' || typeof check.input_truncated !== 'boolean' || !isPlainObject(check.model)) {
            throw new ValueError('invalid session verification provenance or calibration');
          }
          const verifier = tool.verifier ?? null;
          if (verifier !== null && !jsonEqual(check.model, verifier.config.verifier_foundation ?? { initialization: 'configured_weights' })) {
            throw new ValueError('session verifier model provenance mismatch');
          }
        }
      }
      previous = receipt.selected_id as string;
    }
    result.historyItems = deepCopy(data.history as JsonValue) as unknown as RankingHistoryItem[];
    return result;
  }
}
