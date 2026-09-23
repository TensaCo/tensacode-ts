/**
 * ``Idefics3ImageProcessor`` (torchvision backend) and ``Idefics3Processor``
 * of transformers 5.17 over decoded ``uint8`` images: longest-edge resizing,
 * image splitting into ``max_image_size`` tiles plus a global image, fused
 * rescale/normalize, padding, and the prompt expansion of ``<image>`` into
 * ``<fake_token_around_image>``/row-column/``<image>`` token runs.
 */
import { Tensor, tensor } from '../../nn/tensor.js';
import { NotImplementedError, ValueError } from '../../errors.js';
import { deepCopy, isPlainObject, parseJsonStrict, type JsonObject, type JsonValue } from '../json.js';
import { FastTokenizer } from '../tokenizers/index.js';
import { ChatTemplate } from '../text/jinja.js';
import { PIL_RESAMPLING, resizeImage, type InterpolationMode } from '../vec/imageProcessing.js';

/** ``Idefics3ImageProcessor`` class defaults. */
export const IDEFICS3_IMAGE_PROCESSOR_DEFAULTS: Readonly<JsonObject> = Object.freeze({
  do_convert_rgb: true, do_image_splitting: true, do_normalize: true, do_pad: true, do_rescale: true, do_resize: true,
  image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], max_image_size: { longest_edge: 364 }, resample: 1,
  rescale_factor: 1 / 255, size: { longest_edge: 4 * 364 },
});

const MAX_IMAGE_SIZE = 4096;

/** Python ``int(x)`` of a float (truncation toward zero). */
const pyInt = Math.trunc;

function rescaleToMaxLength(height: number, width: number, maxLength: number): [number, number] {
  const aspect = width / height;
  let h = height;
  let w = width;
  if (w >= h) {
    w = maxLength;
    h = pyInt(w / aspect);
    if (h % 2 !== 0) h += 1;
  } else {
    h = maxLength;
    w = pyInt(h * aspect);
    if (w % 2 !== 0) w += 1;
  }
  return [Math.max(h, 1), Math.max(w, 1)];
}

function scaleBelowUpperBound(height: number, width: number, maxLength: number): [number, number] {
  const aspect = width / height;
  let h = height;
  let w = width;
  if (w >= h && w > maxLength) {
    w = maxLength;
    h = pyInt(w / aspect);
  } else if (h > w && h > maxLength) {
    h = maxLength;
    w = pyInt(h * aspect);
  }
  return [Math.max(h, 1), Math.max(w, 1)];
}

function longestEdge(value: JsonValue | undefined, name: string): number {
  if (!isPlainObject(value) || typeof value.longest_edge !== 'number') throw new ValueError(`${name} must be a dictionary with key 'longest_edge'`);
  return value.longest_edge;
}

export interface Idefics3ImageBatch {
  /** ``[1, images, channels, height, width]`` float32. */
  pixelValues: Tensor;
  /** ``[1, images, height, width]`` (1 valid, 0 padding). */
  pixelAttentionMask: Tensor | null;
  rows: number[];
  cols: number[];
}

/** ``Idefics3ImageProcessor`` over decoded ``uint8`` ``CHW`` images. */
export class Idefics3ImageProcessor {
  readonly values: JsonObject;

  constructor(config: unknown = {}) {
    if (!isPlainObject(config)) throw new ValueError('image processor configuration must be a JSON object');
    this.values = { ...deepCopy(IDEFICS3_IMAGE_PROCESSOR_DEFAULTS as JsonObject), ...deepCopy(config as JsonObject) };
  }

  private flag(key: string): boolean {
    return this.values[key] === true;
  }

  private interpolation(): InterpolationMode {
    const code = this.values.resample ?? 1;
    const mode = typeof code === 'number' ? PIL_RESAMPLING[code] : undefined;
    if (mode === undefined) throw new ValueError(`unsupported resample ${JSON.stringify(code)}`);
    if (mode === null) throw new NotImplementedError(`resample ${code} (BOX/HAMMING) is not available in the TypeScript port`);
    return mode;
  }

  /** ``resize`` with ``longest_edge`` (aspect preserving) or an explicit ``[height, width]``. */
  private resize(image: Tensor, size: { longestEdge?: number; height?: number; width?: number }): Tensor {
    const height = image.shape[image.ndim - 2]!;
    const width = image.shape[image.ndim - 1]!;
    let target: [number, number];
    if (size.longestEdge) {
      target = rescaleToMaxLength(height, width, size.longestEdge);
      target = scaleBelowUpperBound(target[0], target[1], MAX_IMAGE_SIZE);
    } else if (size.height && size.width) target = [size.height, size.width];
    else throw new ValueError("size must be a dictionary with key 'longest_edge' or 'height' and 'width'.");
    return resizeImage(image, target, this.interpolation());
  }

  private resizeForVisionEncoder(image: Tensor, maxSize: number): Tensor {
    let height = image.shape[image.ndim - 2]!;
    let width = image.shape[image.ndim - 1]!;
    const aspect = width / height;
    if (width >= height) {
      width = Math.ceil(width / maxSize) * maxSize;
      height = pyInt(width / aspect);
      height = Math.ceil(height / maxSize) * maxSize;
    } else {
      height = Math.ceil(height / maxSize) * maxSize;
      width = pyInt(height * aspect);
      width = Math.ceil(width / maxSize) * maxSize;
    }
    return this.resize(image, { height, width });
  }

  /** Tiles (row-major) followed by the global image, and the split counts. */
  private splitImage(image: Tensor, maxSize: number): { frames: Tensor[]; rows: number; cols: number } {
    const height = image.shape[1]!;
    const width = image.shape[2]!;
    if (height <= maxSize && width <= maxSize) return { frames: [image], rows: 0, cols: 0 };
    const rows = Math.ceil(height / maxSize);
    const cols = Math.ceil(width / maxSize);
    const frames: Tensor[] = [];
    // ``unfold(size=max, step=max)`` keeps only complete tiles.
    for (let row = 0; row + 1 <= Math.floor(height / maxSize); row += 1) {
      for (let col = 0; col + 1 <= Math.floor(width / maxSize); col += 1) {
        frames.push(image.slice(1, row * maxSize, (row + 1) * maxSize).slice(2, col * maxSize, (col + 1) * maxSize));
      }
    }
    frames.push(this.resize(image, { height: maxSize, width: maxSize }));
    return { frames, rows, cols };
  }

  private normalize(image: Tensor): Tensor {
    const channels = image.shape[0]!;
    const plane = image.shape[1]! * image.shape[2]!;
    const mean = this.values.image_mean as number[] | number;
    const std = this.values.image_std as number[] | number;
    const at = (value: number[] | number, channel: number): number => (Array.isArray(value) ? value[value.length === 1 ? 0 : channel]! : value);
    const doRescale = this.flag('do_rescale');
    const doNormalize = this.flag('do_normalize');
    const factor = this.values.rescale_factor as number;
    const out = new Float32Array(image.numel);
    for (let channel = 0; channel < channels; channel += 1) {
      let m = Math.fround(doNormalize ? at(mean, channel) : 0);
      let s = Math.fround(doNormalize ? at(std, channel) : 1);
      if (doNormalize && doRescale) {
        // transformers fuses rescale into normalize: (x - mean / f) / (std / f).
        m = Math.fround(m * Math.fround(1 / factor));
        s = Math.fround(s * Math.fround(1 / factor));
      }
      for (let index = 0; index < plane; index += 1) {
        const position = channel * plane + index;
        const value = image.data[position]!;
        if (doNormalize) out[position] = Math.fround(Math.fround(value - m) / s);
        else if (doRescale) out[position] = Math.fround(value * factor);
        else out[position] = value;
      }
    }
    return tensor(out, { shape: [...image.shape] });
  }

  /** Process one decoded ``uint8`` ``CHW`` RGB image. */
  preprocess(image: Tensor): Idefics3ImageBatch {
    if (!(image instanceof Tensor) || image.ndim !== 3 || image.dtype !== 'uint8' || image.shape[0] !== 3) {
      throw new ValueError('Idefics3 images must be decoded uint8 RGB CHW tensors');
    }
    let current = image;
    if (this.flag('do_resize')) {
      const size = this.values.size;
      if (isPlainObject(size) && typeof size.longest_edge === 'number') current = this.resize(current, { longestEdge: size.longest_edge });
      else if (isPlainObject(size) && typeof size.height === 'number' && typeof size.width === 'number') current = this.resize(current, { height: size.height, width: size.width });
      else throw new ValueError("size must be a dictionary with key 'longest_edge' or 'height' and 'width'.");
    }
    const maxSize = longestEdge(this.values.max_image_size, 'max_image_size');
    let frames: Tensor[];
    let rows = 0;
    let cols = 0;
    if (this.flag('do_image_splitting')) {
      const split = this.splitImage(this.resizeForVisionEncoder(current, maxSize), maxSize);
      frames = split.frames;
      rows = split.rows;
      cols = split.cols;
    } else {
      frames = [this.resize(current, { height: maxSize, width: maxSize })];
    }
    const processed = frames.map((frame) => this.normalize(frame));
    const height = Math.max(...processed.map((frame) => frame.shape[1]!));
    const width = Math.max(...processed.map((frame) => frame.shape[2]!));
    if (!this.flag('do_pad')) {
      if (processed.some((frame) => frame.shape[1] !== height || frame.shape[2] !== width)) throw new ValueError('processed images must share one size');
      const flat = new Float32Array(processed.length * 3 * height * width);
      processed.forEach((frame, index) => flat.set(frame.data as Float32Array, index * frame.numel));
      return { pixelValues: tensor(flat, { shape: [1, processed.length, 3, height, width] }), pixelAttentionMask: null, rows: [rows], cols: [cols] };
    }
    const pixels = new Float32Array(processed.length * 3 * height * width);
    const mask = new Float32Array(processed.length * height * width);
    processed.forEach((frame, index) => {
      const [channels, h, w] = frame.shape as [number, number, number];
      for (let c = 0; c < channels; c += 1) {
        for (let y = 0; y < h; y += 1) {
          for (let x = 0; x < w; x += 1) pixels[((index * channels + c) * height + y) * width + x] = frame.data[(c * h + y) * w + x]!;
        }
      }
      for (let y = 0; y < h; y += 1) for (let x = 0; x < w; x += 1) mask[(index * height + y) * width + x] = 1;
    });
    return {
      pixelValues: tensor(pixels, { shape: [1, processed.length, 3, height, width] }),
      pixelAttentionMask: tensor(mask, { shape: [1, processed.length, height, width] }),
      rows: [rows], cols: [cols],
    };
  }
}

export interface Idefics3ProcessorOutput {
  inputIds: Tensor;
  attentionMask: Tensor;
  pixelValues: Tensor;
  pixelAttentionMask: Tensor | null;
}

const ASSET_SUFFIXES = new Set(['.json', '.jinja', '.txt']);

/** Whether a processor asset name is a safe relative ``.json``/``.jinja``/``.txt`` path. */
export function safeAssetName(name: unknown): name is string {
  if (typeof name !== 'string' || !name || name.startsWith('/') || name.startsWith('\\') || /^[A-Za-z]:/.test(name)) return false;
  const parts = name.split(/[\\/]/);
  if (parts.some((part) => part === '..')) return false;
  const base = parts[parts.length - 1]!;
  const dot = base.lastIndexOf('.');
  return dot > 0 && ASSET_SUFFIXES.has(base.slice(dot));
}

/** ``Idefics3Processor`` built from the files ``processor.save_pretrained`` writes. */
export class Idefics3Processor {
  readonly imageProcessor: Idefics3ImageProcessor;
  readonly tokenizer: FastTokenizer;
  readonly imageSeqLen: number;
  readonly chatTemplate: ChatTemplate | null;
  readonly fakeImageToken = '<fake_token_around_image>';
  readonly imageToken = '<image>';
  readonly globalImageTag = '<global-img>';

  constructor(imageProcessor: Idefics3ImageProcessor, tokenizer: FastTokenizer, imageSeqLen = 169, chatTemplate: string | null = null) {
    this.imageProcessor = imageProcessor;
    this.tokenizer = tokenizer;
    this.imageSeqLen = imageSeqLen;
    this.chatTemplate = chatTemplate === null ? null : new ChatTemplate(chatTemplate);
    for (const token of [this.fakeImageToken, this.imageToken, '<end_of_utterance>']) {
      if (tokenizer.backend.tokenToId(token) === undefined) throw new NotImplementedError(`the processor tokenizer lacks ${token}; adding tokens is not supported`);
    }
  }

  /** ``Idefics3Processor.from_pretrained`` over saved processor assets (name to text). */
  static fromAssets(assets: Record<string, string>): Idefics3Processor {
    const json = (name: string): JsonObject | null => {
      const text = assets[name];
      if (typeof text !== 'string') return null;
      const value = parseJsonStrict(text);
      if (!isPlainObject(value)) throw new ValueError(`${name} must contain a JSON object`);
      return value as JsonObject;
    };
    const processorConfig = json('processor_config.json') ?? {};
    const imageConfig = isPlainObject(processorConfig.image_processor)
      ? processorConfig.image_processor as JsonObject : json('preprocessor_config.json') ?? {};
    const type = imageConfig.image_processor_type;
    if (type !== undefined && type !== 'Idefics3ImageProcessor' && type !== 'Idefics3ImageProcessorFast') {
      throw new NotImplementedError(`image processor ${String(type)} is not supported (Idefics3ImageProcessor only)`);
    }
    const { image_processor_type: _type, processor_class: _class, ...settings } = imageConfig;
    const tokenizer = FastTokenizer.fromFiles(assets);
    let template: string | null = typeof assets['chat_template.jinja'] === 'string' ? assets['chat_template.jinja'] : null;
    if (template === null) {
      const legacy = json('chat_template.json');
      if (legacy && typeof legacy.chat_template === 'string') template = legacy.chat_template;
      else if (typeof processorConfig.chat_template === 'string') template = processorConfig.chat_template;
    }
    const seq = typeof processorConfig.image_seq_len === 'number' ? processorConfig.image_seq_len : 169;
    return new Idefics3Processor(new Idefics3ImageProcessor(settings), tokenizer, seq, template);
  }

  /** ``processor.apply_chat_template(messages, add_generation_prompt=..., tokenize=False)``. */
  applyChatTemplate(messages: JsonValue[], options: { addGenerationPrompt?: boolean } = {}): string {
    if (!this.chatTemplate) throw new ValueError('Cannot use apply_chat_template because this processor does not have a chat template.');
    return this.chatTemplate.render({
      ...this.tokenizer.specialTokensMap, messages, tools: null, documents: null, add_generation_prompt: options.addGenerationPrompt ?? false,
    });
  }

  /** ``replace_image_token`` for one image with ``rows``/``cols`` splits. */
  imagePromptString(rows: number, cols: number): string {
    const image = this.imageToken.repeat(this.imageSeqLen);
    const global = `${this.fakeImageToken}${this.globalImageTag}${image}${this.fakeImageToken}`;
    if (rows === 0 && cols === 0) return global;
    let text = '';
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) text += `${this.fakeImageToken}<row_${row + 1}_col_${col + 1}>${image}`;
      text += '\n';
    }
    return `${text}\n${global}`;
  }

  /** ``processor(text=prompt, images=[image], return_tensors='pt')`` for one prompt and ``uint8`` image. */
  call(prompt: string, images: readonly Tensor[]): Idefics3ProcessorOutput {
    const count = prompt.split(this.imageToken).length - 1;
    if (count !== images.length) {
      throw new ValueError(`The total number of ${this.imageToken} tokens in the prompts should be the same as the number of images passed. Found [${count}] ${this.imageToken} tokens and [${images.length}] images per sample.`);
    }
    if (images.length !== 1) throw new NotImplementedError('the Idefics3 processor port handles exactly one image');
    const batch = this.imageProcessor.preprocess(images[0]!);
    const expanded = prompt.split(this.imageToken).join(this.imagePromptString(batch.rows[0]!, batch.cols[0]!));
    const encoded = this.tokenizer.encode(expanded, { addSpecialTokens: true });
    const ids = encoded.inputIds[0]!;
    return {
      inputIds: tensor(ids, { shape: [1, ids.length], dtype: 'int64' }),
      attentionMask: tensor(encoded.attentionMask[0]!, { shape: [1, ids.length], dtype: 'int64' }),
      pixelValues: batch.pixelValues,
      pixelAttentionMask: batch.pixelAttentionMask,
    };
  }
}
