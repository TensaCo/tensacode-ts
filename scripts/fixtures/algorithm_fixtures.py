"""Portable algorithm parity fixtures; regenerate via generate.py algorithm_fixtures."""

import json
import math

import torch
from tensorcode.algorithm import AlgorithmBuilder, compile_schedule, from_attention

from generate import write_json


def generate():
    b = AlgorithmBuilder(name="vector_feedback_parity")
    p = b.input("position", (20,))
    v = b.input("velocity", (20,))
    t = b.input("target", (20,))
    a = b.build({"result": 4 * (t - p) - 0.5 * v})
    inputs = {
        "position": [math.sin(i) * 0.2 for i in range(20)],
        "velocity": [math.cos(i) * 0.4 for i in range(20)],
        "target": [0.1] * 20,
    }
    write_json("algorithm_feedback.json", {
        "artifact": json.loads(a.to_json()), "inputs": inputs,
        "expected": a.run(inputs)["result"].tolist(),
    })

    torch.manual_seed(9)
    m = torch.nn.MultiheadAttention(4, 2, batch_first=True, dropout=0.0).double()
    a = from_attention(m, tokens=3, causal=True)
    x = torch.randn(3, 4, dtype=torch.float64)
    write_json("algorithm_attention.json", {
        "algorithm": a.graph(), "x": x.tolist(),
        "expected": a.run({"x": x.numpy()})["result"].tolist(),
    })

    a = compile_schedule({"inspect": [], "pick": ["inspect"], "pack": ["pick"]})
    cases = []
    for completed in range(8):
        for enabled in range(8):
            inputs = {
                "completed": [(completed >> i) & 1 for i in range(3)],
                "enabled": [(enabled >> i) & 1 for i in range(3)],
            }
            cases.append({
                "inputs": inputs,
                "outputs": {k: v.tolist() for k, v in a.run(inputs).items()},
            })
    write_json("algorithm_schedule.json", {"artifact": a.to_json(), "cases": cases})
