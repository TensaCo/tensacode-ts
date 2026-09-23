/**
 * Owned plan generation and learned outcome prediction; text never executes
 * (Python ``tensorcode/tools/planner.py``).
 */
import { Tensor, tensor } from '../nn/tensor.js';
import { mseLoss } from '../nn/ops/nn.js';
import { ValueError } from '../errors.js';
import type { Context, OperationLike } from '../ops/base.js';
import { PretrainedModule } from '../_internal/pretrained.js';
import { deepCopy, isPlainObject, type JsonObject, type JsonValue } from '../_internal/json.js';
import {
  RankOperation, RankingObjective, normalizeRankingConfig, rankingFromFoundation, replayableBindings,
} from '../_internal/ranking.js';
import type { FoundationOptions } from '../_internal/native/foundation.js';
import { generateProposals, proposalLoss, type ProposalRecord } from '../_internal/proposals.js';
import { RankingSession } from '../_internal/sessions/ranking.js';
import { isDirectory } from '../_internal/retrieval.js';
import { PlanExecutor, type PlanAction, type ReplanPolicy } from '../_internal/execution/planning.js';
import { Chatbot } from './chatbot.js';

export {
  PlanStep, ExecutablePlan, OutcomeExperience, ReplanRequest, PlanExecutionResult, PlanExecutor,
} from '../_internal/execution/planning.js';

export interface PlannerFoundationOptions extends Omit<FoundationOptions, 'head'> {
  options?: JsonObject;
}

export interface PlannerFoundationsOptions extends Omit<FoundationOptions, 'head' | 'revision'> {
  encoderRevision?: string | null;
  generatorRevision?: string | null;
  generatorOptions?: JsonObject | null;
  options?: JsonObject;
}

/**
 * Generate inert textual plans and rank predicted retrospective outcomes.
 *
 * A configured owned generator proposes candidate text when plans are omitted.
 * Training can supervise one observed plan without assigning fabricated
 * outcomes to unobserved alternatives. Predictions are not causal treatment
 * estimates.
 */
export class Planner extends PretrainedModule<Record<string, unknown>, JsonObject> {
  static override readonly qualifiedName: string = 'tensorcode.tools.planner.Planner';
  declare readonly generator: Chatbot | null;
  declare readonly rank: RankOperation;
  declare readonly objective: RankingObjective;
  /** Planners own no verifier; ranking sessions read this as absent. */
  get verifier(): null {
    return null;
  }

  constructor(config: unknown) {
    if (!isPlainObject(config)) throw new ValueError('model config must be a JSON object');
    const value = { ...(config as JsonObject) };
    let generator: Chatbot | null = null;
    if (value.generator !== undefined && value.generator !== null) {
      generator = new Chatbot(value.generator);
      value.generator = generator.configuration();
    }
    super(normalizeRankingConfig(value));
    const writable = this as unknown as Record<string, unknown>;
    writable.generator = generator !== null ? this.registerModule('generator', generator) : null;
    writable.rank = this.registerModule('rank', new RankOperation(this.config, { taskKey: 'goal', candidatesKey: 'plans' }));
    writable.objective = this.registerModule('objective', new RankingObjective(this));
  }

  /** Outcome-regression objective used by ``Trainer.fromTool``. */
  get trainingOperation(): RankingObjective {
    return this.objective;
  }

  get trainingInputsIncludeTargets(): boolean {
    return true;
  }

  /** Score supplied ``plans`` (or generated ones) for a ``goal``; never executes them. */
  forward(inputs: Record<string, unknown>, context: Context | null): JsonObject {
    if (context && Object.keys(context).length) throw new ValueError('This tool does not accept context');
    let value = inputs;
    if (!('plans' in value)) {
      value = { ...value, plans: this.propose(value) };
      if (!(value.plans as unknown[]).length) {
        return {
          selected_id: null, candidates: [], evidence: deepCopy((value.evidence ?? []) as JsonValue),
          abstained: true, reason: 'no_generated_plans',
        };
      }
    }
    const receipt = this.rank.receipt(value);
    if ((receipt.candidates as JsonObject[]).some((item) => !Number.isFinite(item.predicted_score as number))) {
      throw new ValueError('planner predicted nonfinite scores; no plan selected');
    }
    return receipt;
  }

  /** Alias of {@link forward} without context. */
  predict(inputs: Record<string, unknown>): JsonObject {
    return this.call(inputs);
  }

  /** The complete JSON configuration, including the generator. */
  override configuration(): JsonObject {
    const config = super.configuration();
    if (this.generator !== null) config.generator = this.generator.configuration();
    return config;
  }

  /** Generate inert plan text, retaining its full authored step text. */
  propose(inputs: Record<string, unknown>, options: { count?: number } = {}): ProposalRecord[] {
    return generateProposals(this.generator, inputs, { taskKey: 'goal', count: options.count ?? 3, kind: 'plan' });
  }

  /** Teacher-forced loss for the owned plan generator. */
  generationLoss(inputs: Record<string, unknown>, targets: unknown): Tensor {
    return proposalLoss(this.generator, inputs, targets, { taskKey: 'goal' });
  }

  /** Create a ranking-history session sharing this tool's weights. */
  newSession(): RankingSession {
    return new RankingSession(this);
  }

  /**
   * Construct an executor without running actions or the replan policy.
   * Calling it with state and an ExecutablePlan validates every step before
   * any effect, then executes at most ``maxSteps`` actions.
   */
  newExecutor<S = any>(options: { actions: Record<string, PlanAction<S>> | Map<string, PlanAction<S>>; replan: ReplanPolicy<S>; maxSteps: number }): PlanExecutor<S> {
    return new PlanExecutor<S>(options);
  }

  /** Outcome regression for one observed plan or one outcome per plan. */
  loss(inputs: unknown, targets: unknown): Tensor {
    const predictions = this.rank.call(inputs as Record<string, unknown>);
    if (isPlainObject(targets)) {
      const ids = ((inputs as Record<string, unknown>).plans as JsonObject[]).map((item) => item.id);
      if (!ids.includes(targets.candidate_id as JsonValue)) throw new ValueError('candidate_id must identify a supplied plan');
      const outcome = targets.outcome;
      if (typeof outcome !== 'number' || !Number.isFinite(outcome)) throw new ValueError('outcome must be a finite number');
      return mseLoss(predictions.select(0, ids.indexOf(targets.candidate_id as JsonValue)), tensor(outcome, { dtype: predictions.dtype }));
    }
    const values = targets instanceof Tensor ? targets.toArray() : Array.isArray(targets) ? targets : [targets];
    const shapeMatches = targets instanceof Tensor ? targets.ndim === 1 && targets.numel === predictions.numel
      : Array.isArray(targets) && targets.length === predictions.numel;
    if (!shapeMatches || values.some((item) => typeof item !== 'number' || !Number.isFinite(item))) {
      throw new ValueError('targets must provide one finite observed outcome per plan');
    }
    return mseLoss(predictions, tensor(values as number[], { dtype: predictions.dtype }));
  }

  /** Named operations for tracing, experience and checkpoints. */
  override operationBindings(): Record<string, OperationLike> {
    const result = replayableBindings(this);
    if (this.generator !== null) {
      for (const [key, value] of Object.entries(this.generator.operationBindings())) result[`generator.${key}`] = value;
    }
    return result;
  }

  /** Explicitly load a pretrained encoder; workspace and ranking head start random. */
  static async fromFoundation<T extends Planner>(
    this: new (config: JsonObject) => T, repo: string, options: PlannerFoundationOptions = {},
  ): Promise<T> {
    const { options: extra, ...load } = options;
    const construct = (config: JsonObject) => new this(config);
    const result = await rankingFromFoundation(construct, repo, { ...load, options: extra ?? {} });
    if (!(await isDirectory(repo)) && !(result.config.foundation as JsonObject).revision) {
      throw new ValueError('foundation provenance requires a resolved revision');
    }
    return result;
  }

  /** Bootstrap owned foundation weights; ranking/workspace heads need training. */
  static async fromFoundations<T extends Planner>(
    this: new (config: JsonObject) => T, encoderRepo: string, generatorRepo: string, options: PlannerFoundationsOptions = {},
  ): Promise<T> {
    const { encoderRevision = null, generatorRevision = null, generatorOptions = null, options: extra = {}, ...load } = options;
    const generator = await Chatbot.fromFoundation(generatorRepo, { ...load, revision: generatorRevision, options: generatorOptions ?? {} });
    const result = await (Planner.fromFoundation as (this: new (config: JsonObject) => T, repo: string, options: PlannerFoundationOptions) => Promise<T>)
      .call(this, encoderRepo, { ...load, revision: encoderRevision, options: { ...deepCopy(extra), generator: generator.configuration() } });
    result.generator!.loadStateDict(generator.stateDict());
    return result;
  }
}
