"""Compute-backend parity: float32 products, convolution and attention (including
``enable_gqa``) with gradients, and float16/bfloat16 results, from PyTorch."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from generate import tensor_json, write_json


def _randn(*shape: int, seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _grads(fn, inputs: list[torch.Tensor], seed: int) -> dict:
    leaves = [value.clone().requires_grad_() for value in inputs]
    out = fn(*leaves)
    probe = _randn(*out.shape, seed=seed).to(out.dtype)
    (out * probe).sum().backward()
    return {'out': tensor_json(out), 'probe': tensor_json(probe), 'grads': [tensor_json(leaf.grad) for leaf in leaves]}


def float32_cases() -> dict:
    x, w, b = _randn(17, 42, seed=1), _randn(21, 42, seed=2), _randn(21, seed=3)
    a, m = _randn(2, 3, 9, 14, seed=4), _randn(3, 14, 7, seed=5)
    image, kernel, kernel_bias = _randn(2, 3, 9, 8, seed=6), _randn(4, 3, 3, 3, seed=7), _randn(4, seed=8)
    q, k, v = _randn(2, 4, 13, 16, seed=9), _randn(2, 2, 15, 16, seed=10), _randn(2, 2, 15, 12, seed=11)
    padding = torch.zeros(2, 1, 1, 15)
    padding[0, ..., 5] = torch.finfo(torch.float32).min
    padding[1, ..., 11:] = torch.finfo(torch.float32).min
    full_bias = _randn(2, 4, 13, 15, seed=12)
    with torch.no_grad():
        gqa = F.scaled_dot_product_attention(q, k, v, attn_mask=padding, enable_gqa=True)
        biased = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), attn_mask=full_bias, scale=0.3)
    return {
        'linear': {'inputs': [tensor_json(t) for t in (x, w, b)], **_grads(F.linear, [x, w, b], 20)},
        'matmul': {'inputs': [tensor_json(t) for t in (a, m)], **_grads(torch.matmul, [a, m], 21)},
        'conv2d': {'inputs': [tensor_json(t) for t in (image, kernel, kernel_bias)],
                   **_grads(lambda i, k_, b_: F.conv2d(i, k_, b_, stride=(2, 1), padding=1), [image, kernel, kernel_bias], 22)},
        'attention': {'q': tensor_json(q), 'k': tensor_json(k), 'v': tensor_json(v), 'padding': tensor_json(padding),
                      'full_bias': tensor_json(full_bias), 'gqa': tensor_json(gqa), 'biased': tensor_json(biased)},
    }


def half_cases(dtype: torch.dtype) -> dict:
    x, w, b = (_randn(9, 40, seed=30).to(dtype), _randn(11, 40, seed=31).to(dtype), _randn(11, seed=32).to(dtype))
    a, m = _randn(2, 5, 24, seed=33).to(dtype), _randn(24, 7, seed=34).to(dtype)
    image, kernel = _randn(1, 2, 7, 6, seed=35).to(dtype), _randn(3, 2, 3, 3, seed=36).to(dtype)
    scores = (_randn(6, 29, seed=37) * 3).to(dtype)
    norm_weight, norm_bias = _randn(40, seed=38).to(dtype), _randn(40, seed=39).to(dtype)
    q, k, v = _randn(1, 2, 5, 8, seed=40).to(dtype), _randn(1, 2, 6, 8, seed=41).to(dtype), _randn(1, 2, 6, 8, seed=42).to(dtype)
    with torch.no_grad():
        # The math attention formulation (each op rounds to the dtype).
        attention = torch.softmax(q @ k.transpose(-2, -1) * 0.25, -1) @ v
        outputs = {
            'linear': F.linear(x, w, b), 'matmul': a @ m, 'conv2d': F.conv2d(image, kernel, padding=1),
            'softmax': torch.softmax(scores, -1), 'log_softmax': torch.log_softmax(scores, -1),
            'layer_norm': F.layer_norm(x, (40,), norm_weight, norm_bias), 'attention': attention,
        }
    inputs = {'x': x, 'w': w, 'b': b, 'a': a, 'm': m, 'image': image, 'kernel': kernel, 'scores': scores,
              'norm_weight': norm_weight, 'norm_bias': norm_bias, 'q': q, 'k': k, 'v': v}
    return {'inputs': {name: tensor_json(value) for name, value in inputs.items()},
            'outputs': {name: tensor_json(value) for name, value in outputs.items()}}


def generate() -> None:
    write_json('backend.json', {
        'float32': float32_cases(),
        'bfloat16': half_cases(torch.bfloat16),
        'float16': half_cases(torch.float16),
    })


if __name__ == '__main__':
    generate()
