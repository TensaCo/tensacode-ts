/** In-place parameter initializers (``torch.nn.init`` equivalents). */
import { noGrad } from './autograd.js';
import { getDefaultGenerator, type Generator } from './random.js';
import type { Tensor } from './tensor.js';

export function fanInOut(tensor: Tensor): [number, number] {
  if (tensor.ndim < 2) throw new RangeError('fan in/out requires at least two dimensions');
  const receptive = tensor.shape.slice(2).reduce((product, size) => product * size, 1);
  return [tensor.shape[1]! * receptive, tensor.shape[0]! * receptive];
}

export function normal_(tensor: Tensor, mean = 0, std = 1, generator?: Generator): Tensor {
  return noGrad(() => tensor.normal_(mean, std, generator ?? getDefaultGenerator()));
}

export function uniform_(tensor: Tensor, low = 0, high = 1, generator?: Generator): Tensor {
  return noGrad(() => tensor.uniform_(low, high, generator ?? getDefaultGenerator()));
}

export function constant_(tensor: Tensor, value: number): Tensor {
  return noGrad(() => tensor.fill_(value));
}

export function zeros_(tensor: Tensor): Tensor {
  return constant_(tensor, 0);
}

export function ones_(tensor: Tensor): Tensor {
  return constant_(tensor, 1);
}

/** ``kaiming_uniform_(a=sqrt(5))``: the default ``nn.Linear``/``nn.Conv2d`` weight init. */
export function kaimingUniformDefault_(tensor: Tensor, generator?: Generator): Tensor {
  const [fanIn] = fanInOut(tensor);
  const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0;
  return uniform_(tensor, -bound, bound, generator);
}

export function kaimingUniform_(tensor: Tensor, a = 0, generator?: Generator): Tensor {
  const [fanIn] = fanInOut(tensor);
  const gain = Math.sqrt(2 / (1 + a * a));
  const bound = Math.sqrt(3) * gain / Math.sqrt(fanIn);
  return uniform_(tensor, -bound, bound, generator);
}

export function xavierUniform_(tensor: Tensor, gain = 1, generator?: Generator): Tensor {
  const [fanIn, fanOut] = fanInOut(tensor);
  const bound = gain * Math.sqrt(6 / (fanIn + fanOut));
  return uniform_(tensor, -bound, bound, generator);
}

export function truncatedNormal_(tensor: Tensor, mean = 0, std = 1, low = -2, high = 2, generator?: Generator): Tensor {
  const source = generator ?? getDefaultGenerator();
  return noGrad(() => {
    const data = tensor.data;
    for (let index = 0; index < data.length; index += 1) {
      let value: number;
      do value = mean + std * source.normal(); while (value < low || value > high);
      data[index] = value;
    }
    tensor._storage.version += 1;
    return tensor;
  });
}
