# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic tensors for the engine's CPU tests and fixture generators: tiny stand-in networks get
weights whose seed is the CRC of the tensor name, so a fixture can be rebuilt bit for bit anywhere."""
from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Tiny stand-ins with the structure of Z-Image-Turbo (diffusers ZImageTransformer2DModel config) and of its
# Qwen3 text encoder (transformers Qwen3 config).
TINY_TRANSFORMER = {"all_patch_size": [2], "all_f_patch_size": [1], "in_channels": 16, "dim": 256, "n_layers": 2,
                    "n_refiner_layers": 2, "n_heads": 8, "n_kv_heads": 8, "norm_eps": 1e-5, "qk_norm": True,
                    "cap_feat_dim": 48, "rope_theta": 256.0, "t_scale": 1000.0, "axes_dims": [8, 12, 12],
                    "axes_lens": [256, 64, 64]}
TINY_TEXT = {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
             "head_dim": 16, "hidden_act": "silu", "hidden_size": 64, "intermediate_size": 128,
             "max_position_embeddings": 40960, "model_type": "qwen3", "num_attention_heads": 4,
             "num_hidden_layers": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_scaling": None,
             "rope_theta": 1000000, "sliding_window": None, "tie_word_embeddings": True, "use_cache": False,
             "use_sliding_window": False, "vocab_size": 151936}


def seeded_state_dict(shapes: Dict[str, Sequence[int]]):
    """Deterministic float32 tensors for a {name: shape} map: the seed is the CRC of the name.

    Norm scales are about 1, other vectors small, matrices scaled by the inverse square root of their
    fan-in, so activations stay well-behaved through the whole network.
    """
    import torch

    state = {}
    for name, shape in shapes.items():
        generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        values = torch.randn(tuple(shape), generator=generator)
        if len(shape) == 1:
            is_norm_scale = name.endswith(".weight") and "norm" in name
            values = 1.0 + 0.1 * values if is_norm_scale else 0.05 * values
        else:
            values = values / math.sqrt(math.prod(shape[1:]))
        state[name] = values
    return state


def seeded_inputs(shape: Sequence[int], seed: int):
    import torch

    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed))


def num(value: float) -> float:
    """A float rounded to nine significant digits, as the fixtures store numbers."""
    return float(f"{float(value):.9g}")


def flat(tensor) -> List[float]:
    return [num(v) for v in tensor.reshape(-1).tolist()]


def shapes(module) -> Dict[str, List[int]]:
    return {name: list(t.shape) for name, t in module.state_dict().items()}
