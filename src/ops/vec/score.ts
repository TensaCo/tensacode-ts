/**
 * Owned pairwise vector scoring with an explicit authored score meaning
 * (Python ``tensorcode/ops/vec/score.py``).
 */
import { Module } from '../../nn/module.js';
import { Tensor } from '../../nn/tensor.js';
import { Linear } from '../../nn/layers.js';
import { cat } from '../../nn/ops/shape.js';
import { ValueError } from '../../errors.js';
import { LatentOperation } from '../../_internal/latentOps.js';
import {
  Objective, OwnedMap, positive, pythonList, unknownKeys, type OwnedKind, type OwnedFoundationOptions,
} from '../../_internal/vec/owned.js';
import { moduleConfiguration } from '../../_internal/vec/configuration.js';
import { qualifiedName } from '../../_internal/identity.js';
import type { JsonObject } from '../../_internal/json.js';
import type { Context, OperationLike } from '../base.js';
import { CandidateSet, Scores } from './candidates.js';
import { Latent, Space, requireCompatible } from './latent.js';

/** Private readout over interacting query/candidate pairs (Python ``_PairReadout``). */
export class PairReadout extends OwnedMap<Tensor> {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.score._PairReadout';
  static override readonly kind: OwnedKind = 'decode';
}

/** A supplied scoring module: ``module.forward(query, candidates) -> scores``. */
export type PairScoreModule = Module & { forward(query: Tensor, candidates: Tensor): Tensor };

const PAIR_SPACE = 'tensorcode:score:interacting-pair';
const SCORE_INTERNAL: unique symbol = Symbol('tensorcode.vec.score.internal');

interface ScoreInternals {
  readonly [SCORE_INTERNAL]: true;
  readonly supplied?: { module: PairScoreModule; querySpace: Space; candidateSpace: Space; meaning: string };
  readonly readout?: PairReadout;
}

export interface ScoreFoundationOptions extends Omit<OwnedFoundationOptions, 'inputSpace' | 'outputSpace' | 'labels' | 'outputDimensions' | 'output' | 'readout'> {
  querySpace: Space | JsonObject;
  candidateSpace: Space | JsonObject;
  meaning: string;
  pairDimensions?: number;
}

/**
 * Owned pairwise scoring of candidates against a query. Returns {@link Scores}
 * carrying the configured authored ``meaning``; query and candidate vectors
 * must match their declared spaces.
 */
export class Score extends LatentOperation<CandidateSet, Scores> {
  static override readonly qualifiedName: string = 'tensorcode.ops.vec.score.Score';
  readonly querySpace: Space;
  readonly candidateSpace: Space;
  readonly meaning: string;
  readonly pairSpace: Space | null = null;
  readonly queryProjection: Linear | null = null;
  readonly candidateProjection: Linear | null = null;
  readonly module: PairReadout | PairScoreModule;
  readonly #supplied: boolean;
  readonly #objective: Objective;

  constructor(config: unknown, internals?: ScoreInternals) {
    const supplied = internals?.[SCORE_INTERNAL] ? internals.supplied : undefined;
    super(supplied ? {} : config);
    this.#objective = new Objective(this);
    if (supplied) {
      this.#supplied = true;
      this.module = this.registerModule('module', supplied.module);
      this.querySpace = supplied.querySpace;
      this.candidateSpace = supplied.candidateSpace;
      this.meaning = supplied.meaning;
      return;
    }
    this.#supplied = false;
    const cfg = this.config;
    const allowed = ['architecture', 'query_space', 'candidate_space', 'meaning', 'hidden_dimensions', 'native_config', 'foundation', 'pair_dimensions'];
    const unknown = unknownKeys(cfg, allowed);
    if (unknown.length) throw new ValueError(`unknown configuration fields: ${pythonList(unknown)}`);
    this.querySpace = Space.fromConfig(cfg.query_space);
    this.candidateSpace = Space.fromConfig(cfg.candidate_space);
    const meaning = cfg.meaning;
    if (typeof meaning !== 'string' || !meaning.trim()) throw new ValueError('Score meaning must be a nonempty string');
    this.meaning = meaning;
    const width = positive(cfg.pair_dimensions ?? Math.min(this.querySpace.dimensions, this.candidateSpace.dimensions), 'pair_dimensions');
    cfg.pair_dimensions = width;
    this.queryProjection = this.registerModule('query_projection', new Linear(this.querySpace.dimensions, width));
    this.candidateProjection = this.registerModule('candidate_projection', new Linear(this.candidateSpace.dimensions, width));
    this.pairSpace = new Space(PAIR_SPACE, width * 3);
    const readout: JsonObject = {};
    for (const key of ['architecture', 'hidden_dimensions', 'native_config', 'foundation']) if (key in cfg) readout[key] = cfg[key]!;
    Object.assign(readout, { input_space: this.pairSpace.configuration() as unknown as JsonObject, output_dimensions: 1, output: this.meaning });
    const module = internals?.[SCORE_INTERNAL] && internals.readout ? internals.readout : new PairReadout(readout);
    this.module = this.registerModule('module', module);
    cfg.query_space = this.querySpace.configuration() as unknown as JsonObject;
    cfg.candidate_space = this.candidateSpace.configuration() as unknown as JsonObject;
    cfg.architecture = module.config.architecture!;
    if ('native_config' in readout) cfg.native_config = module.config.native_config!;
  }

  /** Advanced: wrap a supplied ``module(query, candidates) -> scores``; cannot ``savePretrained``. */
  static fromModule(module: PairScoreModule, options: { querySpace: Space; candidateSpace: Space; meaning: string }): Score {
    if (!(module instanceof Module) || typeof module.forward !== 'function') throw new TypeError('Score module must be an nn.Module');
    if (!(options.querySpace instanceof Space) || !(options.candidateSpace instanceof Space)) throw new TypeError('Score spaces must be Space objects');
    if (typeof options.meaning !== 'string' || !options.meaning.trim()) throw new ValueError('Score meaning must be nonempty');
    return new this({}, {
      [SCORE_INTERNAL]: true,
      supplied: { module, querySpace: options.querySpace, candidateSpace: options.candidateSpace, meaning: options.meaning },
    });
  }

  /** Build the pair readout from a supported pretrained transformer ``repo``. */
  static async fromFoundation(repo: string, options: ScoreFoundationOptions): Promise<Score> {
    const { querySpace, candidateSpace, meaning, pairDimensions, revision = null, ...rest } = options;
    const q = Space.fromConfig(querySpace);
    const c = Space.fromConfig(candidateSpace);
    const width = positive(pairDimensions ?? Math.min(q.dimensions, c.dimensions), 'pair_dimensions');
    const pair = new Space(PAIR_SPACE, width * 3);
    const module = await PairReadout.fromFoundation(repo, { ...rest, revision, inputSpace: pair, outputDimensions: 1, output: meaning });
    const saved = module.configuration();
    const config: JsonObject = {};
    for (const key of ['architecture', 'native_config', 'foundation']) if (key in saved) config[key] = saved[key]!;
    Object.assign(config, {
      query_space: q.configuration() as unknown as JsonObject, candidate_space: c.configuration() as unknown as JsonObject,
      meaning, pair_dimensions: width,
    });
    return new this(config, { [SCORE_INTERNAL]: true, readout: module }).eval();
  }

  get supplied(): boolean {
    return this.#supplied;
  }

  get trainingInputsIncludeTargets(): boolean {
    return true;
  }

  get objective(): Objective {
    return this.#objective;
  }

  /** Objective operation used by ``Trainer.fromTool``. */
  get trainingOperation(): Objective {
    return this.#objective;
  }

  /** Score every candidate in a {@link CandidateSet}; returns {@link Scores}. */
  forward(value: CandidateSet, context: Context | null): Scores {
    if (!(value instanceof CandidateSet)) throw new TypeError('Score expects a CandidateSet');
    requireCompatible(this.querySpace, value.query.space, { role: 'query' });
    requireCompatible(this.candidateSpace, value.candidates.space, { role: 'candidates' });
    let scores: Tensor;
    if (this.#supplied) {
      if (context) throw new ValueError('supplied Score does not consume context');
      scores = (this.module as PairScoreModule).forward(value.query.tensor, value.candidates.tensor);
    } else {
      if (context) throw new ValueError('Score does not consume context; supply explicit query and candidates');
      const query = value.query.tensor;
      const candidates = value.candidates.tensor;
      if (query.dtype !== candidates.dtype) throw new ValueError('query and candidates require matching dtype and device');
      const queryMask = value.query.mask;
      if (queryMask !== null && (queryMask.dtype !== 'bool' || !queryMask.all().item())) {
        throw new ValueError('every query must be valid with a boolean mask');
      }
      const candidateFeatures = this.candidateProjection!.forward(candidates);
      const queryFeatures = this.queryProjection!.forward(query).unsqueeze(-2).expandAs(candidateFeatures);
      const pairs = cat([queryFeatures, candidateFeatures, queryFeatures.mul(candidateFeatures)], -1);
      const flat = pairs.reshape(-1, pairs.shape[pairs.ndim - 1]!);
      const readout = (this.module as PairReadout).call(new Latent(flat, this.pairSpace!));
      scores = readout.reshape(candidates.shape.slice(0, -1));
      if (value.candidates.mask !== null) scores = scores.maskedFill(value.candidates.mask.logicalNot(), 0);
    }
    return new Scores(scores, this.meaning, value);
  }

  /** Mean squared error against per-candidate target scores (masked). */
  loss(value: CandidateSet, targets: unknown, options: { context?: Context | null } = {}): Tensor {
    let scores = this.forward(value, options.context ?? null).values;
    if (!(targets instanceof Tensor) || targets.shape.length !== scores.shape.length
      || targets.shape.some((size, index) => size !== scores.shape[index])) {
      throw new ValueError('score targets must match candidate scores');
    }
    let expected = targets.dtype === scores.dtype ? targets : targets.to(scores.dtype);
    const mask = value.candidates.mask;
    if (mask !== null) {
      scores = scores.maskedSelect(mask);
      expected = expected.maskedSelect(mask);
    }
    if (!expected.allFinite()) throw new ValueError('valid targets must be finite');
    return scores.sub(expected).square().mean();
  }

  override configuration(): JsonObject {
    if (this.#supplied) {
      return {
        operation: qualifiedName(this),
        query_space: this.querySpace.configuration() as unknown as JsonObject,
        candidate_space: this.candidateSpace.configuration() as unknown as JsonObject,
        meaning: this.meaning,
        module: moduleConfiguration(this.module),
      };
    }
    return super.configuration();
  }

  /** Named operations for tracing, experience and checkpoints. */
  override operationBindings(): Record<string, OperationLike> {
    return { ...super.operationBindings(), objective: this.#objective };
  }

  /** Save configuration and weights; rejected for ``fromModule`` scores. */
  override async savePretrained(directory: string): Promise<string> {
    if (this.#supplied) throw new ValueError('supplied modules have no declarative reconstruction; cannot save_pretrained');
    return super.savePretrained(directory);
  }
}
