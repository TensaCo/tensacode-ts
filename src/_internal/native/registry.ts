/**
 * ``AutoModel.from_config`` equivalents for the supported native architectures.
 *
 * | head                      | architectures                                              |
 * |---------------------------|------------------------------------------------------------|
 * | ``base``                  | albert, bert, roberta, electra, distilbert, deberta-v2, vit, clip, llama |
 * | ``seq2seq``               | t5 (``AutoModelForSeq2SeqLM``)                             |
 * | ``encoder``               | t5 (``T5EncoderModel``)                                    |
 * | ``sequence-classification`` | albert, bert, roberta, electra, distilbert, deberta-v2   |
 * | ``image-text-to-text``    | idefics3 (``AutoModelForImageTextToText``)                 |
 */
import { ValueError } from '../../errors.js';
import type { NativeConfig } from './config.js';
import type { NativeModel } from './modules.js';
import {
  AlbertForSequenceClassification, AlbertModel, BertForSequenceClassification, BertModel, DistilBertForSequenceClassification, DistilBertModel,
  ElectraForSequenceClassification, ElectraModel, RobertaForSequenceClassification, RobertaModel,
} from './bert.js';
import { T5EncoderModel, T5ForConditionalGeneration } from './t5.js';
import { ViTModel } from './vit.js';
import { CLIPModel } from './clip.js';
import { createDebertaV2ForSequenceClassification, createDebertaV2Model } from './debertaV2.js';
import { LlamaModel } from './llama.js';
import { Idefics3ForConditionalGeneration } from './idefics3.js';

export type NativeHead = 'base' | 'seq2seq' | 'encoder' | 'sequence-classification' | 'image-text-to-text';

export interface CreateOptions {
  /** BERT/RoBERTa/ViT pooler (``add_pooling_layer``); default true for BERT/RoBERTa, false for ViT. */
  addPoolingLayer?: boolean;
}

/** ``base_model_prefix`` of the architecture's pretrained head models. */
export const BASE_MODEL_PREFIX: Record<string, string> = {
  albert: 'albert', bert: 'bert', roberta: 'roberta', electra: 'electra', distilbert: 'distilbert', 'deberta-v2': 'deberta',
  t5: 'transformer', vit: 'vit', clip: 'clip', llama: 'model', idefics3: 'model',
};

export function createNativeModel(config: NativeConfig, head: NativeHead = 'base', options: CreateOptions = {}): NativeModel {
  const type = config.modelType;
  if (head === 'seq2seq') {
    if (type === 't5') return new T5ForConditionalGeneration(config);
    throw new ValueError(`${type} is not a supported native sequence-to-sequence architecture (supported: t5)`);
  }
  if (head === 'image-text-to-text') {
    if (type === 'idefics3') return new Idefics3ForConditionalGeneration(config);
    throw new ValueError(`${type} is not a supported native image-text-to-text architecture (supported: idefics3)`);
  }
  if (head === 'encoder') {
    if (type === 't5') return new T5EncoderModel(config);
    return createNativeModel(config, 'base', options);
  }
  if (head === 'sequence-classification') {
    switch (type) {
      case 'albert': return new AlbertForSequenceClassification(config);
      case 'bert': return new BertForSequenceClassification(config);
      case 'roberta': return new RobertaForSequenceClassification(config);
      case 'electra': return new ElectraForSequenceClassification(config);
      case 'distilbert': return new DistilBertForSequenceClassification(config);
      case 'deberta-v2': return createDebertaV2ForSequenceClassification(config);
      default: throw new ValueError(`${type} has no supported native sequence classification head`);
    }
  }
  switch (type) {
    case 'albert': return new AlbertModel(config, { addPoolingLayer: options.addPoolingLayer ?? true });
    case 'bert': return new BertModel(config, { addPoolingLayer: options.addPoolingLayer ?? true });
    case 'roberta': return new RobertaModel(config, { addPoolingLayer: options.addPoolingLayer ?? true });
    case 'electra': return new ElectraModel(config);
    case 'distilbert': return new DistilBertModel(config);
    case 'deberta-v2': return createDebertaV2Model(config);
    case 'vit': return new ViTModel(config, { addPoolingLayer: options.addPoolingLayer ?? false });
    case 'clip': return new CLIPModel(config);
    case 'llama': return new LlamaModel(config);
    case 'idefics3': throw new ValueError('Idefics3 models are used through the image-text-to-text head');
    case 't5': throw new ValueError('T5 base models are used through the seq2seq or encoder heads');
    default: throw new ValueError(`unsupported native architecture ${JSON.stringify(type)}`);
  }
}
