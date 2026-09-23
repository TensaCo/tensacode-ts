"""Layer, optimizer and safetensors parity fixtures."""
from __future__ import annotations

import torch
from torch import nn
from safetensors.torch import save, save_model

from generate import OUT, tensor_json, write_json


def seeded(shape, seed, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=generator, dtype=torch.float64) * 2 - 1) * scale


def layers():
    torch.manual_seed(1)
    cases = {}
    linear = nn.Linear(4, 3).double()
    x = seeded((2, 4), 1)
    cases['linear'] = {'state': {k: tensor_json(v) for k, v in linear.state_dict().items()},
                       'input': tensor_json(x), 'output': tensor_json(linear(x))}
    norm = nn.LayerNorm(4, eps=1e-5).double()
    with torch.no_grad():
        norm.weight.copy_(seeded((4,), 2))
        norm.bias.copy_(seeded((4,), 3))
    cases['layer_norm'] = {'state': {k: tensor_json(v) for k, v in norm.state_dict().items()},
                           'input': tensor_json(x), 'output': tensor_json(norm(x))}
    cell = nn.GRUCell(4, 3).double()
    h = seeded((2, 3), 4)
    cases['gru_cell'] = {'state': {k: tensor_json(v) for k, v in cell.state_dict().items()},
                         'input': tensor_json(x), 'hidden': tensor_json(h), 'output': tensor_json(cell(x, h))}
    gru = nn.GRU(4, 3, batch_first=True).double()
    sequence = seeded((2, 5, 4), 5)
    output, hidden = gru(sequence)
    cases['gru'] = {'state': {k: tensor_json(v) for k, v in gru.state_dict().items()},
                    'input': tensor_json(sequence), 'output': tensor_json(output), 'hidden': tensor_json(hidden)}
    conv = nn.Conv2d(2, 3, kernel_size=(3, 2), stride=2, padding=1).double()
    image = seeded((1, 2, 5, 6), 6)
    cases['conv2d'] = {'state': {k: tensor_json(v) for k, v in conv.state_dict().items()},
                       'input': tensor_json(image), 'output': tensor_json(conv(image))}
    bag = nn.EmbeddingBag(5, 3, mode='mean').double()
    cases['embedding_bag'] = {'state': {k: tensor_json(v) for k, v in bag.state_dict().items()},
                              'indices': [1, 4, 4, 0, 2], 'offsets': [0, 2],
                              'output': tensor_json(bag(torch.tensor([1, 4, 4, 0, 2]), torch.tensor([0, 2])))}
    logits = seeded((3, 4), 7)
    targets = torch.tensor([1, -100, 3])
    cases['cross_entropy'] = {'logits': tensor_json(logits), 'targets': targets.tolist(),
                              'loss': torch.nn.functional.cross_entropy(logits, targets).item()}
    values = seeded((6,), 8, 3.0)
    cases['gelu'] = {'input': tensor_json(values), 'erf': tensor_json(torch.nn.functional.gelu(values)),
                     'tanh': tensor_json(torch.nn.functional.gelu(values, approximate='tanh'))}
    write_json('nn_layers.json', cases)


def optimizers():
    cases = {}
    configurations = {
        'sgd': (torch.optim.SGD, {'lr': 0.1}),
        'sgd_momentum': (torch.optim.SGD, {'lr': 0.1, 'momentum': 0.9, 'dampening': 0.1, 'weight_decay': 0.01}),
        'sgd_nesterov': (torch.optim.SGD, {'lr': 0.05, 'momentum': 0.8, 'nesterov': True}),
        'adam': (torch.optim.Adam, {'lr': 0.01, 'weight_decay': 0.1}),
        'adam_amsgrad': (torch.optim.Adam, {'lr': 0.02, 'amsgrad': True, 'betas': (0.8, 0.9)}),
        'adamw': (torch.optim.AdamW, {'lr': 0.01}),
    }
    for name, (cls, options) in configurations.items():
        param = torch.nn.Parameter(seeded((2, 3), 11))
        start = param.detach().clone()
        optimizer = cls([param], **options)
        coefficients = []
        for step in range(4):
            optimizer.zero_grad()
            coefficient = seeded((2, 3), 20 + step)
            coefficients.append(tensor_json(coefficient))
            loss = (param.square() * coefficient).sum() + param.sum() * 0.3
            loss.backward()
            optimizer.step()
        cases[name] = {'options': {k: list(v) if isinstance(v, tuple) else v for k, v in options.items()},
                       'start': tensor_json(start), 'final': tensor_json(param), 'coefficients': coefficients,
                       'state_keys': sorted(optimizer.state_dict()['state'].get(0, {}).keys()),
                       'param_group_keys': sorted(k for k in optimizer.state_dict()['param_groups'][0].keys())}
    write_json('nn_optimizers.json', cases)


def safetensors_bytes():
    tensors = {
        'b.weight': torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7,
        'a.bias': torch.tensor([1.5, -2.25], dtype=torch.float64),
        'ids': torch.tensor([[1, 2], [3, -4]], dtype=torch.int64),
        'flags': torch.tensor([True, False, True]),
        'half': torch.tensor([0.1, -65504.0, 3.0], dtype=torch.float16),
        'brain': torch.tensor([0.1, 1e30, -3.0], dtype=torch.bfloat16),
        'small': torch.tensor([7, 250], dtype=torch.uint8),
        'mid': torch.tensor([-7, 30000], dtype=torch.int16),
        'i32': torch.tensor([-70000, 5], dtype=torch.int32),
    }
    (OUT / 'safetensors_mixed.safetensors').write_bytes(save(tensors))
    write_json('safetensors_mixed.json', {k: tensor_json(v) for k, v in tensors.items()})

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Embedding(4, 2)
            self.head = nn.Linear(2, 4, bias=False)
            self.head.weight = self.shared.weight
            self.other = nn.Linear(2, 2)

    torch.manual_seed(3)
    model = Tied()
    save_model(model, str(OUT / 'safetensors_tied.safetensors'))
    write_json('safetensors_tied.json', {k: tensor_json(v) for k, v in model.state_dict().items()})


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    layers()
    optimizers()
    safetensors_bytes()
