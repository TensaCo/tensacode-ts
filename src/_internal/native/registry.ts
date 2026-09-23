/**
 * ``AutoModel.from_config`` equivalents for the supported native architectures.
 *
 * | head                      | architectures                                              |
 * |---------------------------|------------------------------------------------------------|
 * | ``base``                  | bert, roberta, electra, distilbert, deberta-v2, vit, clip  |
 * | ``seq2seq``               | t5 (``AutoModelForSeq2SeqLM``)                             |
 * | ``encoder``               | t5 (``T5EncoderModel``)                                    |
 * | ``sequence-classification`` | bert, roberta, electra, distilbert, deberta-v2           |
 */
import { ValueError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import type { NativeModel } from './modules.js';
import {
  BertForSequenceClassification, BertModel, DistilBertForSequenceClassification, DistilBertModel,
  ElectraForSequenceClassification, ElectraModel, RobertaForSequenceClassification, RobertaModel,
} from './bert.js';
import { T5EncoderModel, T5ForConditionalGeneration } from './t5.js';
import { ViTModel } from './vit.js';
import { CLIPModel } from './clip.js';
import { createDebertaV2ForSequenceClassification, createDebertaV2Model } from './debertaV2.js';

export type NativeHead = 'base' | 'seq2seq' | 'encoder' | 'sequence-classification';

export interface CreateOptions {
  /** BERT/RoBERTa/ViT pooler (``add_pooling_layer``); default true for BERT/RoBERTa, false for ViT. */
  addPoolingLayer?: boolean;
}

/** ``base_model_prefix`` of the architecture's pretrained head models. */
export const BASE_MODEL_PREFIX: Record<string, string> = {
  bert: 'bert', roberta: 'roberta', electra: 'electra', distilbert: 'distilbert', 'deberta-v2': 'deberta',
  t5: 'transformer', vit: 'vit', clip: 'clip',
};

export function createNativeModel(config: NativeConfig, head: NativeHead = 'base', options: CreateOptions = {}): NativeModel {
  const type = config.modelType;
  if (head === 'seq2seq') {
    if (type === 't5') return new T5ForConditionalGeneration(config);
    throw new ValueError(`${type} is not a supported native sequence-to-sequence architecture (supported: t5)`);
  }
  if (head === 'encoder') {
    if (type === 't5') return new T5EncoderModel(config);
    return createNativeModel(config, 'base', options);
  }
  if (head === 'sequence-classification') {
    switch (type) {
      case 'bert': return new BertForSequenceClassification(config);
      case 'roberta': return new RobertaForSequenceClassification(config);
      case 'electra': return new ElectraForSequenceClassification(config);
      case 'distilbert': return new DistilBertForSequenceClassification(config);
      case 'deberta-v2': return createDebertaV2ForSequenceClassification(config);
      default: throw new ValueError(`${type} has no supported native sequence classification head`);
    }
  }
  switch (type) {
    case 'bert': return new BertModel(config, { addPoolingLayer: options.addPoolingLayer ?? true });
    case 'roberta': return new RobertaModel(config, { addPoolingLayer: options.addPoolingLayer ?? true });
    case 'electra': return new ElectraModel(config);
    case 'distilbert': return new DistilBertModel(config);
    case 'deberta-v2': return createDebertaV2Model(config);
    case 'vit': return new ViTModel(config, { addPoolingLayer: options.addPoolingLayer ?? false });
    case 'clip': return new CLIPModel(config);
    case 't5': throw new ValueError('T5 base models are used through the seq2seq or encoder heads');
    default: throw new ValueError(`unsupported native architecture ${JSON.stringify(type)}`);
  }
}
