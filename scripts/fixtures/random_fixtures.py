"""PyTorch CPU generator and CPython ``random`` parity fixtures.

Each case seeds ``torch.manual_seed`` and runs a sequence of sampling calls;
outputs are stored as raw little-endian bytes (base64) so the TypeScript tests
compare bit for bit. The generator state after the sequence is stored too.
"""
from __future__ import annotations

import base64
import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from generate import write_json


def raw(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().contiguous()
    data = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
    return {'dtype': str(tensor.dtype).removeprefix('torch.'), 'shape': list(tensor.shape),
            'bytes': base64.b64encode(data.numpy().tobytes()).decode('ascii')}


def dtype_of(name: str) -> torch.dtype:
    return getattr(torch, name)


def run(op: dict) -> torch.Tensor:
    kind = op['op']
    if kind == 'rand':
        return torch.rand(op['shape'], dtype=dtype_of(op['dtype']))
    if kind == 'randn':
        return torch.randn(op['shape'], dtype=dtype_of(op['dtype']))
    if kind == 'uniform':
        return torch.empty(op['shape'], dtype=dtype_of(op['dtype'])).uniform_(op['low'], op['high'])
    if kind == 'normal':
        return torch.empty(op['shape'], dtype=dtype_of(op['dtype'])).normal_(op['mean'], op['std'])
    if kind == 'randint':
        return torch.randint(op['low'], op['high'], op['shape'])
    if kind == 'randperm':
        return torch.randperm(op['n'])
    if kind == 'bernoulli':
        return torch.empty(op['shape'], dtype=dtype_of(op['dtype'])).bernoulli_(op['p'])
    if kind == 'bernoulli_tensor':
        probabilities = torch.arange(op['size'], dtype=dtype_of(op['dtype'])) / (op['size'] - 1)
        return torch.bernoulli(probabilities)
    if kind == 'exponential':
        return torch.empty(op['shape'], dtype=dtype_of(op['dtype'])).exponential_(op['lambd'])
    if kind == 'dropout':
        x = (torch.arange(math.prod(op['shape']), dtype=dtype_of(op['dtype'])) / 4 - 3).reshape(op['shape'])
        return F.dropout(x, op['p'], training=True)
    if kind == 'multinomial':
        weights = torch.arange(1, op['categories'] + 1, dtype=dtype_of(op['dtype'])).repeat(op['rows'], 1)
        weights = weights * weights / 8
        return torch.multinomial(weights, op['samples'], replacement=op['replacement'])
    raise ValueError(kind)


SEQUENCE = [
    {'op': 'rand', 'shape': [5], 'dtype': 'float32'},
    {'op': 'randn', 'shape': [3], 'dtype': 'float32'},
    {'op': 'randn', 'shape': [16], 'dtype': 'float32'},
    {'op': 'randn', 'shape': [37], 'dtype': 'float32'},
    {'op': 'randn', 'shape': [4, 5], 'dtype': 'float64'},
    {'op': 'randn', 'shape': [7], 'dtype': 'float64'},
    {'op': 'rand', 'shape': [9], 'dtype': 'float64'},
    {'op': 'normal', 'shape': [300], 'dtype': 'float32', 'mean': 0.25, 'std': 0.02},
    {'op': 'normal', 'shape': [5], 'dtype': 'float32', 'mean': -1.5, 'std': 3.0},
    {'op': 'uniform', 'shape': [64, 3], 'dtype': 'float32', 'low': -0.0441941738241592, 'high': 0.0441941738241592},
    {'op': 'uniform', 'shape': [33], 'dtype': 'float64', 'low': -0.123, 'high': 0.456},
    {'op': 'randn', 'shape': [40], 'dtype': 'float16'},
    {'op': 'randn', 'shape': [21], 'dtype': 'bfloat16'},
    {'op': 'rand', 'shape': [10], 'dtype': 'float16'},
    {'op': 'randint', 'low': -5, 'high': 17, 'shape': [12]},
    {'op': 'randint', 'low': 0, 'high': 2 ** 40, 'shape': [6]},
    {'op': 'randperm', 'n': 25},
    {'op': 'bernoulli', 'shape': [20], 'dtype': 'float32', 'p': 0.3},
    {'op': 'bernoulli_tensor', 'size': 17, 'dtype': 'float32'},
    {'op': 'bernoulli_tensor', 'size': 9, 'dtype': 'float64'},
    {'op': 'exponential', 'shape': [11], 'dtype': 'float32', 'lambd': 1.5},
    {'op': 'dropout', 'shape': [4, 6], 'dtype': 'float32', 'p': 0.1},
    {'op': 'dropout', 'shape': [3, 7], 'dtype': 'float64', 'p': 0.35},
    {'op': 'multinomial', 'rows': 2, 'categories': 6, 'samples': 1, 'replacement': False, 'dtype': 'float32'},
    {'op': 'multinomial', 'rows': 1, 'categories': 8, 'samples': 4, 'replacement': False, 'dtype': 'float32'},
    {'op': 'multinomial', 'rows': 2, 'categories': 5, 'samples': 7, 'replacement': True, 'dtype': 'float32'},
    {'op': 'multinomial', 'rows': 1, 'categories': 5, 'samples': 3, 'replacement': True, 'dtype': 'float64'},
    {'op': 'randn', 'shape': [3], 'dtype': 'float64'},
]


def generator_cases() -> list[dict]:
    cases = []
    for seed in [0, 1, 42, 2 ** 40 + 5, -7]:
        torch.manual_seed(seed)
        initial = raw(torch.get_rng_state())
        outputs = [raw(run(op)) for op in SEQUENCE]
        cases.append({'seed': str(seed), 'initial_state': initial, 'outputs': outputs,
                      'final_state': raw(torch.get_rng_state())})
    return cases


def generator_object_case() -> dict:
    generator = torch.Generator()
    default = raw(generator.get_state())
    generator.manual_seed(99)
    values = torch.randn(20, generator=generator)
    return {'default_state': default, 'seed_99_randn_20': raw(values), 'state': raw(generator.get_state())}


def large_case() -> dict:
    torch.manual_seed(3)
    values = torch.randn(20_003)
    doubles = torch.randn(4_001, dtype=torch.float64)
    return {'float32': raw(values), 'float64': raw(doubles), 'state': raw(torch.get_rng_state())}


def init_cases() -> dict:
    torch.manual_seed(5)
    linear = nn.Linear(37, 11)
    embedding = nn.Embedding(13, 6, padding_idx=2)
    bag = nn.EmbeddingBag(9, 4, mode='mean')
    conv = nn.Conv2d(3, 5, kernel_size=3, stride=2)
    gru = nn.GRU(6, 4, batch_first=True)
    cell = nn.GRUCell(5, 3)
    attention = nn.MultiheadAttention(8, 1, batch_first=True)
    norm = nn.LayerNorm(7)
    xavier = nn.init.xavier_uniform_(torch.empty(9, 4))
    kaiming = nn.init.kaiming_uniform_(torch.empty(6, 10), a=math.sqrt(5))
    trunc = nn.init.trunc_normal_(torch.empty(50), std=0.02)
    modules = {'linear': linear, 'embedding': embedding, 'bag': bag, 'conv': conv, 'gru': gru, 'cell': cell,
               'attention': attention, 'norm': norm}
    states = {name: {key: raw(value) for key, value in module.state_dict().items()} for name, module in modules.items()}
    return {'states': states, 'xavier': raw(xavier), 'kaiming': raw(kaiming), 'trunc_normal': raw(trunc),
            'state': raw(torch.get_rng_state())}


def python_random_cases() -> list[dict]:
    cases = []
    for seed in [0, 12345, 2 ** 70 + 3, -9, 'tensorcode', b'\x00\x01bytes', 3.5, -2.25]:
        rng = random.Random(seed)
        encoded_seed = ({'type': 'str', 'value': seed} if isinstance(seed, str)
                        else {'type': 'bytes', 'value': base64.b64encode(seed).decode()} if isinstance(seed, bytes)
                        else {'type': 'float', 'value': seed} if isinstance(seed, float)
                        else {'type': 'int', 'value': str(seed)})
        values = {
            'random': [rng.random() for _ in range(5)],
            'getrandbits': [str(rng.getrandbits(k)) for k in (1, 7, 32, 33, 64, 100)],
            'randrange': [rng.randrange(10), rng.randrange(3, 1000), rng.randrange(-50, 50, 7), rng.randrange(100, 0, -3),
                          rng.randint(1, 6)],
            'big_randrange': str(rng.randrange(2 ** 80)),
            'choice': rng.choice('abcdefghij'),
            'shuffle': (lambda items: (rng.shuffle(items), items)[1])(list(range(15))),
            'sample_small': rng.sample(range(30), 5),
            'sample_large': rng.sample(range(1000), 12),
            'sample_counts': rng.sample(['a', 'b', 'c'], 4, counts=[3, 1, 2]),
            'choices': rng.choices('xyz', k=4),
            'weighted_choices': rng.choices('xyz', weights=[1, 5, 2], k=5),
            'uniform': rng.uniform(-2.0, 3.0),
            'gauss': [rng.gauss(0.5, 2.0) for _ in range(5)],
        }
        state = rng.getstate()
        cases.append({'seed': encoded_seed, 'values': values,
                      'state': [state[0], list(state[1]), state[2]]})
    return cases


def libm_cases() -> dict:
    """C library results the samplers depend on (glibc on the reference AArch64 platform), as raw bits."""
    import ctypes
    import struct
    libm = ctypes.CDLL('libm.so.6')
    for name in ('logf', 'sinf', 'cosf'):
        getattr(libm, name).restype = ctypes.c_float
        getattr(libm, name).argtypes = [ctypes.c_float]
    for name in ('log', 'log1p', 'sin', 'cos', 'fma'):
        getattr(libm, name).restype = ctypes.c_double
    for name in ('log', 'log1p', 'sin', 'cos'):
        getattr(libm, name).argtypes = [ctypes.c_double]
    libm.fma.argtypes = [ctypes.c_double] * 3
    rng = random.Random(2024)
    f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
    f32bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
    f64bits = lambda x: format(struct.unpack('<Q', struct.pack('<d', x))[0], '016x')
    uniforms = [f32(rng.getrandbits(24) * 2.0 ** -24) for _ in range(600)]
    floats = {
        'logf': [f32(1 - u) for u in uniforms], 'sinf': [f32(2 * math.pi * u) for u in uniforms],
        'cosf': [f32(2 * math.pi * u) for u in uniforms],
    }
    doubles = {
        'log': [1 - rng.getrandbits(53) * 2.0 ** -53 for _ in range(600)] + [0.9375 + rng.random() * 0.13 for _ in range(200)],
        'log1p': [-(rng.getrandbits(53) * 2.0 ** -53) for _ in range(600)] + [(rng.random() - 0.3) * 50 for _ in range(200)],
        'sin': [2 * math.pi * rng.random() for _ in range(600)] + [(rng.random() - 0.5) * 1e6 for _ in range(200)],
        'cos': [2 * math.pi * rng.random() for _ in range(600)] + [(rng.random() - 0.5) * 1e6 for _ in range(200)],
    }
    triples = [[(rng.random() - .5) * 2 ** rng.randint(-30, 30) for _ in range(3)] for _ in range(600)]
    return {
        'float': {name: [[f32bits(x), f32bits(getattr(libm, name)(x))] for x in xs] for name, xs in floats.items()},
        'double': {name: [[f64bits(x), f64bits(getattr(libm, name)(x))] for x in xs] for name, xs in doubles.items()},
        'fma': [[*map(f64bits, t), f64bits(libm.fma(*t))] for t in triples],
    }


def generate() -> None:
    write_json('random.json', {
        'sequence': SEQUENCE,
        'generator': generator_cases(),
        'generator_object': generator_object_case(),
        'large': large_case(),
        'init': init_cases(),
        'python_random': python_random_cases(),
        'libm': libm_cases(),
    })
