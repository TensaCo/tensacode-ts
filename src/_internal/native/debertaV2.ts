/**
 * EXTENSION POINT — owned by the tools builder (module "tools").
 *
 * DeBERTa-v2/v3 (``model_type='deberta-v2'``) native architecture with
 * transformers 5.17 parameter names, used by NLI verifiers such as
 * ``cross-encoder/nli-deberta-v3-small``. The builder implements
 * ``DebertaV2Model`` and ``DebertaV2ForSequenceClassification`` here (disentangled
 * attention with log-bucketed relative positions) and keeps these factory
 * signatures; ``registry.ts`` already routes ``deberta-v2`` to them.
 */
import { NotImplementedError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import type { NativeModel } from './modules.js';

export function createDebertaV2Model(config: NativeConfig): NativeModel {
  void config;
  throw new NotImplementedError('DeBERTa-v2 native architecture is provided by the tools module');
}

export function createDebertaV2ForSequenceClassification(config: NativeConfig): NativeModel {
  void config;
  throw new NotImplementedError('DeBERTa-v2 native architecture is provided by the tools module');
}
