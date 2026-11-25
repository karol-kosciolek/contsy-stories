# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Z-Image-Turbo diffusion transformer (S3-DiT) from its bf16 checkpoint or the Lite GGUF.

Files: the product reads one bf16 safetensors file that the installer merges from Tongyi-MAI's official float32
shards (`checkpoint.merge`; the names stay the official diffusers names). The shards themselves, through their
index, load as well. The Lite GGUF stores the weights in a single-file layout with a fused `attention.qkv`; those
names go through diffusers' converter (`diffusers_tensors`), as does a safetensors file in that layout.

Conventions (diffusers 0.42 `ZImagePipeline` and `ZImageTransformer2DModel`, Apache-2.0):

* Sampling state is a velocity flow: `x = sigma * noise + (1 - sigma) * latent`, sigma from 1 to 0.
* The network is fed `t = 1 - sigma` (the pipeline's `(1000 - timestep) / 1000`) and multiplies it by
  `t_scale` = 1000 itself.
* The raw output has the opposite sign of the velocity (the pipeline's `noise_pred = -noise_pred`), so
  `velocity = -output` and the denoised estimate is `x - sigma * velocity`.
* Z-Image-Turbo samples without guidance, so one forward pass per step with the prompt embedding is the
  whole model call.
* The input is the latent as it comes from the sampler, in the dtype of the weights; the output is
  widened to float32.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor

from ...core.quant import materialize_quantized
from ...formats.ggml import GGUFFile, QTensor
from ...sampling.flow import denoised_from_velocity
from . import assert_loaded, checkpoint, materialize

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "zimage_transformer.json"
# A tensor that only the official (diffusers) naming has; the fused-qkv layout calls it `x_embedder.weight`.
OFFICIAL_NAME = "all_x_embedder.2-1.weight"


def _build(config: Optional[Dict] = None):
    """A ZImageTransformer2DModel on the meta device (default: the shipped Z-Image-Turbo config)."""
    from diffusers import ZImageTransformer2DModel

    if config is None:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    with torch.device("meta"):
        return ZImageTransformer2DModel.from_config({k: v for k, v in config.items() if not k.startswith("_")})


def diffusers_tensors(checkpoint: Iterator[Tuple[str, Tensor]]) -> Iterator[Tuple[str, Tensor]]:
    """(name, tensor) pairs of the fused-qkv single-file layout (the Lite GGUF's) to diffusers names, one tensor
    at a time.

    diffusers' converter renames per key and splits the fused `attention.qkv` into to_q/to_k/to_v, so it
    can run on one-entry dicts; a whole file is never converted (or held) at once.
    """
    from diffusers.loaders.single_file_utils import convert_z_image_transformer_checkpoint_to_diffusers

    for name, tensor in checkpoint:
        yield from convert_z_image_transformer_checkpoint_to_diffusers({name: tensor}).items()


def _file_tensors(path: Path) -> Iterator[Tuple[str, Tensor]]:
    """Tensors of a safetensors file or sharded set under diffusers names: official names pass through, the
    fused-qkv layout is converted."""
    if OFFICIAL_NAME in checkpoint.names(path):
        yield from checkpoint.tensors(path)
    else:
        yield from diffusers_tensors(checkpoint.tensors(path))


def _gguf_tensors(file: GGUFFile):
    for name, tensor in file.tensors.items():
        if name.endswith(".attention.qkv.weight"):
            if len(tensor.shape) != 2 or tensor.shape[0] % 3:
                raise ValueError(f"Invalid fused Z-Image projection {name}: {tensor.shape}")
            rows, columns = tensor.shape
            for projection, data in zip(("to_q", "to_k", "to_v"), tensor.data.chunk(3, dim=0), strict=True):
                key = name.replace(".attention.qkv.weight", f".attention.{projection}.weight")
                yield from diffusers_tensors(iter([(key, QTensor(key, tensor.qtype, (rows // 3, columns), data))]))
        else:
            yield from diffusers_tensors(iter([(name, tensor)]))


def load_transformer(path: str | Path, device: torch.device | str = "cpu",
                     dtype: torch.dtype = torch.bfloat16, config: Optional[Dict] = None):
    """Build the transformer on the meta device and fill it from safetensors (one file, or a sharded set's index)
    or a Lite GGUF.

    Tensors are read, converted and copied to `device` one at a time. `config` replaces the shipped
    config (used with tiny models in tests).
    """
    model = _build(config)
    if Path(path).suffix.lower() == ".gguf":
        with GGUFFile(path) as file:
            materialize_quantized(model, _gguf_tensors(file), device, dtype, "Z-Image transformer")
    else:
        materialize(model, _file_tensors(Path(path)), device, dtype, "transformer")
    assert_loaded(model, "transformer")
    return model.eval()


def predict_velocity(model, latent: Tensor, sigma: Tensor, context: Tensor) -> Tensor:
    """One forward pass: the velocity of the flow at noise level `sigma` (float32, shape of `latent`).

    `latent` is [1, C, H, W] float32, `sigma` a scalar tensor, `context` the prompt embedding [L, D].
    """
    dtype = next(parameter.dtype for parameter in model.parameters() if not parameter.is_meta)
    device = latent.device
    t = (1.0 - sigma.to(device=device, dtype=torch.float32)).reshape(1)
    output = model([latent[0].unsqueeze(1).to(dtype)], t, [context.to(device=device, dtype=dtype)],
                   return_dict=False)[0]
    return -output[0].squeeze(1).unsqueeze(0).float()


def predict_denoised(model, latent: Tensor, sigma: Tensor, context: Tensor) -> Tensor:
    """The sampler's view of the model: the estimate of the clean latent, `x - sigma * velocity`."""
    return denoised_from_velocity(latent, sigma.to(latent.device), predict_velocity(model, latent, sigma, context))
