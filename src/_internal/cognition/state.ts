/**
 * Bounded, immutable session data (Python ``tensorcode/_internal/cognition/state.py``).
 * Records do not infer facts or execute policies.
 */
import { readFile } from 'node:fs/promises';
import { ValueError } from '../../errors.js';
import { isPlainObject, parseJsonStrict, pythonJsonDumps, type JsonObject } from '../json.js';
import { atomicWriteFile } from '../files.js';
import { sessionFloatKeys } from '../sessions/floats.js';
import {
  Assessment, Evidence, Goal, Hypothesis, Observation, Plan, recordEquals,
} from '../../tools/cognition.js';

export { Assessment, Evidence, Goal, Hypothesis, Observation, Plan };

type RecordKind = 'evidence' | 'hypotheses' | 'assessments' | 'goals' | 'plans' | 'observations';
type AnyRecord = Evidence | Hypothesis | Assessment | Goal | Plan | Observation;

const RECORD_TYPES: Record<RecordKind, { new (...args: any[]): AnyRecord; recordFields: readonly string[]; fromRecord(fields: Record<string, unknown>): AnyRecord }> = {
  evidence: Evidence, hypotheses: Hypothesis, assessments: Assessment, goals: Goal, plans: Plan, observations: Observation,
};
const KINDS = Object.keys(RECORD_TYPES) as RecordKind[];
const STATE_FIELDS = ['max_records', 'revision', ...KINDS, 'selection', 'selection_revision'];

export interface CognitiveStateOptions {
  maxRecords?: number;
  revision?: number;
  evidence?: readonly Evidence[];
  hypotheses?: readonly Hypothesis[];
  assessments?: readonly Assessment[];
  goals?: readonly Goal[];
  plans?: readonly Plan[];
  observations?: readonly Observation[];
  selection?: readonly string[];
  selectionRevision?: number | null;
}

function isInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value);
}

function sameRecords(a: readonly AnyRecord[], b: readonly AnyRecord[]): boolean {
  return a.length === b.length && a.every((record, index) => recordEquals(record, b[index]));
}

function recordId(record: AnyRecord): string {
  return (record as { id?: string }).id!;
}

export class CognitiveState {
  static readonly qualifiedName: string = 'tensorcode._internal.cognition.state.CognitiveState';
  readonly maxRecords: number;
  readonly revision: number;
  readonly evidence: readonly Evidence[];
  readonly hypotheses: readonly Hypothesis[];
  readonly assessments: readonly Assessment[];
  readonly goals: readonly Goal[];
  readonly plans: readonly Plan[];
  readonly observations: readonly Observation[];
  readonly selection: readonly string[];
  readonly selectionRevision: number | null;

  constructor(options: CognitiveStateOptions = {}) {
    const maxRecords = options.maxRecords ?? 256;
    const revision = options.revision ?? 0;
    if (!isInteger(maxRecords) || maxRecords < 1) throw new ValueError('max_records must be positive');
    if (!isInteger(revision) || revision < 0) throw new ValueError('invalid state revision');
    this.maxRecords = maxRecords;
    this.revision = revision;
    const values: Partial<Record<RecordKind, readonly AnyRecord[]>> = {};
    for (const name of KINDS) {
      const records = options[name] ?? [];
      const kind = RECORD_TYPES[name];
      if (!Array.isArray(records) || records.some((record) => record === null || typeof record !== 'object' || record.constructor !== kind)) {
        throw new ValueError(`invalid ${name} records`);
      }
      values[name] = Object.freeze([...records]);
      if (name !== 'assessments' && new Set(records.map(recordId)).size !== records.length) throw new ValueError(`duplicate ${name} ids`);
    }
    this.evidence = values.evidence as readonly Evidence[];
    this.hypotheses = values.hypotheses as readonly Hypothesis[];
    this.assessments = values.assessments as readonly Assessment[];
    this.goals = values.goals as readonly Goal[];
    this.plans = values.plans as readonly Plan[];
    this.observations = values.observations as readonly Observation[];
    if (KINDS.reduce((total, name) => total + values[name]!.length, 0) > maxRecords) throw new ValueError('state capacity exceeded');
    const evidenceIds = new Set(this.evidence.map((record) => record.id));
    const hypothesisIds = new Set(this.hypotheses.map((record) => record.id));
    for (const assessment of this.assessments) {
      if (!evidenceIds.has(assessment.evidenceId) || !hypothesisIds.has(assessment.hypothesisId)) {
        throw new ValueError('assessment refers to absent record');
      }
      if (assessment.revision > revision) throw new ValueError('assessment revision is in the future');
    }
    const selection = options.selection ?? [];
    if (!Array.isArray(selection) || selection.some((id) => typeof id !== 'string' || !hypothesisIds.has(id))) {
      throw new ValueError('selection refers to absent hypothesis');
    }
    this.selection = Object.freeze([...selection]);
    const selectionRevision = options.selectionRevision ?? null;
    if (selectionRevision !== null && (!isInteger(selectionRevision) || selectionRevision < 0 || selectionRevision > revision)) {
      throw new ValueError('invalid selection revision');
    }
    this.selectionRevision = selectionRevision;
    if (this.selection.length && selectionRevision === null) throw new ValueError('selection needs revision');
    Object.freeze(this);
  }

  private options(): CognitiveStateOptions {
    return {
      maxRecords: this.maxRecords, revision: this.revision, evidence: this.evidence, hypotheses: this.hypotheses,
      assessments: this.assessments, goals: this.goals, plans: this.plans, observations: this.observations,
      selection: this.selection, selectionRevision: this.selectionRevision,
    };
  }

  /** Python ``dataclasses.replace``: a validated copy with some fields changed. */
  replace(changes: CognitiveStateOptions): CognitiveState {
    return new CognitiveState({ ...this.options(), ...changes });
  }

  private add(name: RecordKind, records: Iterable<AnyRecord>, revisable = false): CognitiveState {
    const items = new Map<string, AnyRecord>(this[name].map((record) => [recordId(record), record]));
    const batch = [...records];
    if (batch.some((record) => record === null || typeof record !== 'object' || record.constructor !== RECORD_TYPES[name])) {
      throw new ValueError(`invalid ${name} record`);
    }
    if (new Set(batch.map(recordId)).size !== batch.length) throw new ValueError('duplicate ids in batch');
    for (const record of batch) {
      const existing = items.get(recordId(record));
      if (existing !== undefined && !recordEquals(existing, record) && !revisable) throw new ValueError('immutable record id conflict');
      items.set(recordId(record), record);
    }
    const updated = [...items.values()];
    if (sameRecords(updated, this[name])) return this;
    return this.replace({ [toOption(name)]: updated, revision: this.revision + 1 } as CognitiveStateOptions);
  }

  addEvidence(records: Iterable<Evidence>): CognitiveState {
    return this.add('evidence', records);
  }

  addHypotheses(records: Iterable<Hypothesis>): CognitiveState {
    return this.add('hypotheses', records, true);
  }

  addGoals(records: Iterable<Goal>): CognitiveState {
    return this.add('goals', records, true);
  }

  addPlans(records: Iterable<Plan>): CognitiveState {
    return this.add('plans', records, true);
  }

  observe(records: Iterable<Observation>): CognitiveState {
    return this.add('observations', records);
  }

  assess(records: Iterable<Assessment>): CognitiveState {
    const batch = [...records];
    if (batch.some((record) => !(record instanceof Assessment) || record.constructor !== Assessment)) throw new ValueError('invalid assessment');
    const stamped = batch.map((record) => record.replace({ revision: this.revision + 1 }));
    return this.replace({
      revision: this.revision + 1, assessments: [...this.assessments, ...stamped], selectionRevision: null, selection: [],
    });
  }

  select(hypothesisIds: Iterable<string>): CognitiveState {
    return this.replace({ selection: [...hypothesisIds], selectionRevision: this.revision });
  }

  isStale(assessment: Assessment): boolean {
    return assessment.revision !== this.revision;
  }

  get selectionStale(): boolean {
    return this.selectionRevision !== this.revision;
  }

  equals(other: unknown): boolean {
    if (!(other instanceof CognitiveState)) return false;
    return pythonJsonDumps(this.toDict(), { sortKeys: true }) === pythonJsonDumps(other.toDict(), { sortKeys: true });
  }

  /** Python ``to_dict``: the persisted schema (version 1). */
  toDict(): JsonObject {
    const result: JsonObject = {
      schema_version: 1, max_records: this.maxRecords, revision: this.revision,
      selection: [...this.selection], selection_revision: this.selectionRevision,
    };
    for (const name of KINDS) {
      result[name] = this[name].map((record) => {
        const data = record.toRecord() as Record<string, unknown>;
        const copy: JsonObject = {};
        for (const [key, value] of Object.entries(data)) copy[key] = (Array.isArray(value) ? [...value] : value) as JsonObject[string];
        return copy;
      });
    }
    return result;
  }

  static fromDict(data: unknown): CognitiveState {
    const expected = [...STATE_FIELDS, 'schema_version'];
    if (!isPlainObject(data) || Object.keys(data).length !== expected.length || !expected.every((key) => key in data)
      || !isInteger(data.schema_version) || data.schema_version !== 1) {
      throw new ValueError('unsupported session schema');
    }
    const options: CognitiveStateOptions = {
      maxRecords: data.max_records as number, revision: data.revision as number,
      selection: data.selection as string[], selectionRevision: data.selection_revision as number | null,
    };
    if (options.selectionRevision === undefined) options.selectionRevision = null;
    for (const name of KINDS) {
      const rows = data[name];
      if (!Array.isArray(rows)) throw new ValueError(`${name} must be an array`);
      const kind = RECORD_TYPES[name];
      const records = rows.map((row) => {
        if (!isPlainObject(row) || Object.keys(row).length !== kind.recordFields.length || !kind.recordFields.every((key) => key in row)) {
          throw new ValueError(`invalid ${name} schema`);
        }
        return kind.fromRecord(row);
      });
      (options as Record<string, unknown>)[toOption(name)] = records;
    }
    return new CognitiveState(options);
  }

  async save(path: string): Promise<void> {
    await atomicWriteFile(path, pythonJsonDumps(this.toDict(), { allowNan: false, indent: 2, floatKeys: sessionFloatKeys() }));
  }

  static async load(path: string): Promise<CognitiveState> {
    let data: unknown;
    try {
      data = parseJsonStrict(await readFile(path, 'utf8'));
    } catch (error) {
      if (error instanceof SyntaxError && /Duplicate/.test(error.message)) throw new ValueError('duplicate JSON key', { cause: error });
      throw error;
    }
    return CognitiveState.fromDict(data);
  }
}

function toOption(name: RecordKind): keyof CognitiveStateOptions {
  return name;
}
