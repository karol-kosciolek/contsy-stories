# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The CLIP ViT-H/14 image encoder of Wan 2.1 image-to-video, as InfiniteTalk uses it.

Wan 2.1 I2V (Wan-Video/Wan2.1, Apache-2.0, `wan/modules/clip.py`) gives its cross-attention the tokens of the
open-clip XLM-RoBERTa ViT-H/14 vision tower: the start picture is resized to 224 x 224 (bicubic, the aspect ratio is
not kept), normalised with CLIP's mean and deviation, and run through the patch embedding, the class token, the
position embedding, the pre-norm and the first 31 of the 32 blocks (`use_31_block`): the 257 x 1280 tokens before the
last block, without the post-norm or the projection head. The blocks are pre-norm with exact GELU and LayerNorm eps
1e-5; the attention is one fused `to_qkv` projection.

The official checkpoint (`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` of Wan-AI/Wan2.1-I2V-14B-480P) is a
4.8 GB float32 pickle holding the text tower too. `extract_visual` (the catalog's install step) writes the part this
encoder reads - the vision tower without its last block, post-norm and head - as float16 safetensors (Wan runs this
encoder in float16), 1.2 GB.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SIZE, PATCH, DIM, HEADS, LAYERS, USED_LAYERS = 224, 14, 1280, 16, 32, 31
TOKENS = (SIZE // PATCH) ** 2 + 1  # 257
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
EPS = 1e-5


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=EPS)
        self.attn = nn.Module()
        self.attn.to_qkv = nn.Linear(DIM, 3 * DIM)
        self.attn.proj = nn.Linear(DIM, DIM)
        self.norm2 = nn.LayerNorm(DIM, eps=EPS)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))

    def forward(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        normed = F.layer_norm(x.float(), (DIM,), self.norm1.weight.float(), self.norm1.bias.float(), EPS).type_as(x)
        q, k, v = self.attn.to_qkv(normed).view(b, s, 3, HEADS, DIM // HEADS).unbind(2)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + self.attn.proj(out.transpose(1, 2).reshape(b, s, DIM))
        normed = F.layer_norm(x.float(), (DIM,), self.norm2.weight.float(), self.norm2.bias.float(), EPS).type_as(x)
        return x + self.mlp(normed)


class ClipVision(nn.Module):
    """The vision tower up to its 31st block; `forward` takes pictures already prepared by `prepare`."""

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH, bias=False)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embedding = nn.Parameter(torch.zeros(1, TOKENS, DIM))
        self.pre_norm = nn.LayerNorm(DIM, eps=EPS)
        self.transformer = nn.Sequential(*(_Block() for _ in range(USED_LAYERS)))

    def forward(self, pictures: Tensor) -> Tensor:
        """[B, 3, 224, 224] normalised pictures -> [B, 257, 1280] tokens (the output of block 31)."""
        x = self.patch_embedding(pictures.to(self.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_embedding.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        x = x + self.pos_embedding.to(x.dtype)
        x = F.layer_norm(x.float(), (DIM,), self.pre_norm.weight.float(), self.pre_norm.bias.float(), EPS).type_as(x)
        return self.transformer(x)


def prepare(pictures: Tensor) -> Tensor:
    """Pictures [B, 3, H, W] in -1..1 (the range the VAE gets) to the encoder's input, as Wan's `CLIPModel.visual`:
    bicubic resize to 224 x 224, back to 0..1, then CLIP's normalisation."""
    resized = F.interpolate(pictures.float(), size=(SIZE, SIZE), mode="bicubic", align_corners=False)
    resized = resized.mul(0.5).add(0.5)
    mean = torch.tensor(MEAN, device=resized.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=resized.device).view(1, 3, 1, 1)
    return (resized - mean) / std


def load(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float16) -> ClipVision:
    """The encoder from the file `extract_visual` wrote."""
    from safetensors.torch import load_file

    with torch.device("meta"):  # no random fill of 611M weights only to overwrite them
        model = ClipVision()
    state = load_file(str(path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"{Path(path).name}: missing {missing[:3]}, unexpected {unexpected[:3]}")
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def visual_tensors(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """The tensors of an official checkpoint the encoder reads, under its own names, as float16."""
    prefix, dropped = "visual.", f"transformer.{LAYERS - 1}."
    out = {}
    for name, tensor in state.items():
        if not name.startswith(prefix):
            continue
        local = name[len(prefix):]
        if local in ("head", "post_norm.weight", "post_norm.bias") or local.startswith(dropped):
            continue
        out[local] = tensor.to(torch.float16).contiguous()
    return out


def extract_visual(source: str | Path, output: str | Path, *, expected_size: Optional[int] = None,
                   expected_sha256: Optional[str] = None,
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   cancel_flag: Optional[Callable[[], bool]] = None) -> str:
    """The catalog's install step: write the encoder's tensors of the official checkpoint `source` to `output`
    (safetensors, float16, sorted names, no metadata, so the bytes are the same on every PC). The pickle is read with
    `weights_only`, which loads tensors and plain containers only. Written to `<output>.part` and renamed when its
    size and SHA-256 equal the expected ones (when given). Returns the SHA-256."""
    from safetensors.torch import save_file

    output = Path(output)
    part = output.with_name(output.name + ".part")
    if part.is_symlink() or output.is_symlink():
        raise ValueError("refusing to write through a link")
    state = torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
    if cancel_flag is not None and cancel_flag():
        raise ValueError("conversion cancelled; run it again to start over")
    tensors = dict(sorted(visual_tensors(state).items()))
    del state
    if len(tensors) != 3 + 2 + USED_LAYERS * 12:  # embeddings, pre-norm, 12 per block
        raise ValueError(f"{Path(source).name} holds {len(tensors)} vision tensors, not those of ViT-H/14")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(part))
    size = part.stat().st_size
    digest = hashlib.sha256()
    with part.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
            if progress_cb is not None:
                progress_cb(stream.tell(), size)
    sha256 = digest.hexdigest()
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{part.name} has {size} bytes, the catalog expects {expected_size}")
    if expected_sha256 is not None and sha256 != expected_sha256.lower():
        raise ValueError(f"{part.name} has SHA-256 {sha256}, the catalog expects {expected_sha256}")
    os.replace(part, output)
    return sha256
