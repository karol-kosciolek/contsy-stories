# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""FLAN-T5 prompt encoding for EzAudio (google/flan-t5-xl, encoder half only).

The reference (EzAudio src/inference.py) runs `T5Tokenizer(prompt, max_length=100, padding="max_length",
truncation=True)` and `T5EncoderModel(input_ids, attention_mask).last_hidden_state` in float32. Here:

* Tokens: SentencePiece pieces of the prompt (the slow, "legacy" T5 tokenizer the reference was written for does
  exactly this), cut to `max_tokens - 1`, then the end token (id 1), then padding (id 0) up to `max_tokens`. The
  mask is True up to and including the end token. The empty prompt, the reference's unconditional text, is the
  end token alone.
* What differs from the UMT5 of Wan (engine/models/wan22/text.py): T5 v1.1 has one relative position bias in the
  first layer, shared by all layers (transformers' T5 class handles it), GELU-tanh instead of GELU-erf in the gated
  feed-forward layer, no start or emphasis syntax, a fixed length of 100 tokens, and the padding is not zeroed (the
  diffusion transformer masks it in its cross-attention, as in the reference).
* The official checkpoint is float32 and split in two files; the encoder and the embedding table are all in the
  first one, the second holds decoder tensors only. Only the encoder is read (about 4.9 GB of the 9.45 GB file).
* With a 16-bit `dtype` the matrices are stored as float16 (FLAN-T5-XL's largest value is about 150, far inside
  the float16 range, and float16 keeps three more mantissa bits than bfloat16) and every linear layer computes in
  float32 (`CastLinear`), like the Wan encoder.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from torch import Tensor, nn

from ..wan22.text import CastLinear, float32_gated
from ..zimage import assert_loaded, materialize

END, PAD = 1, 0
INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
# Parts of a full T5 checkpoint that the encoder does not use; any other tensor must be the encoder's.
_DECODER_PARTS = ("decoder.", "lm_head.")
_ENCODER_PARTS = ("encoder.", "shared.")
_FLOAT16_LIMIT = 60000.0


def clean_prompt(prompt: str) -> str:
    """The prompt as one line of plain text: HTML entities decoded (twice: a prompt can arrive escaped twice), runs
    of whitespace collapsed. The tokenizer then never sees layout characters the scene text happened to contain."""
    return " ".join(html.unescape(html.unescape(prompt)).split())


def load_tokenizer(folder: str | Path):
    """SentencePiece from the `spiece.model` of the FLAN-T5 folder."""
    import sentencepiece

    return sentencepiece.SentencePieceProcessor(model_file=str(Path(folder) / "spiece.model"))


def tokenize(sp, prompts: Sequence[str], max_tokens: int) -> Tuple[Tensor, Tensor]:
    """(ids [B, max_tokens] int64, mask [B, max_tokens] bool) as the reference tokenizer call produces them."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    ids = torch.full((len(prompts), max_tokens), PAD, dtype=torch.long)
    mask = torch.zeros((len(prompts), max_tokens), dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        tokens = [int(t) for t in sp.encode(prompt)][:max_tokens - 1] + [END]
        ids[row, :len(tokens)] = torch.tensor(tokens)
        mask[row, :len(tokens)] = True
    return ids, mask


def shard_files(folder: Path) -> List[Path]:
    """The checkpoint files that hold encoder tensors, from the shard index (or the single-file checkpoint)."""
    index = folder / INDEX_FILE
    if not index.is_file():
        return [folder / SINGLE_FILE]
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    names = list(dict.fromkeys(file for name, file in weight_map.items() if name.startswith(_ENCODER_PARTS)))
    # A shard is a plain file name next to the index; anything else (a path, "..") is refused.
    if not names or any(Path(name).name != name or name in (".", "..") for name in names):
        raise ValueError(f"{index}: unexpected shard names {names}")
    return [folder / name for name in names]


def _tensors(files: Sequence[Path]) -> Iterator[Tuple[str, Tensor]]:
    """The encoder's tensors in the files; the decoder's are skipped, anything else is passed on and then refused by
    `materialize` (the index may list a tied `encoder.embed_tokens.weight` that safetensors stored only once)."""
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not name.startswith(_DECODER_PARTS):
                    yield name, handle.get_tensor(name)


def _storage(dtype: torch.dtype):
    """`materialize`'s precision rule: vectors and the relative position bias stay float32; with a 16-bit dtype a
    matrix becomes float16 when its values fit (all of FLAN-T5's do), otherwise bfloat16."""
    def rule(name: str, tensor: Tensor) -> Optional[torch.dtype]:
        if tensor.ndim == 1 or "relative_attention_bias" in name:
            return torch.float32
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float16 if float(tensor.abs().max()) < _FLOAT16_LIMIT else torch.bfloat16
        return None
    return rule


def load_encoder(folder: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
    """The T5 encoder of a Hugging Face FLAN-T5 folder (config.json + safetensors shards); every encoder tensor of
    the files must find its place and every parameter must be filled."""
    from transformers import T5Config, T5EncoderModel
    from transformers.models.t5.modeling_t5 import T5DenseGatedActDense

    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "t5" or config.get("feed_forward_proj") != "gated-gelu":
        raise ValueError(f"{folder}: expected a FLAN-T5 (T5 v1.1, gated GELU) configuration")
    with torch.device("meta"):
        model = T5EncoderModel(T5Config(**config))
    materialize(model, _tensors(shard_files(folder)), device, dtype, "FLAN-T5 encoder",
                optional=["encoder.embed_tokens.weight"], dtype_for=_storage(dtype))
    model.encoder.embed_tokens.weight = model.shared.weight  # the file stores the tied table once
    for module in model.modules():
        if type(module) is nn.Linear:
            module.__class__ = CastLinear
        elif type(module) is T5DenseGatedActDense:
            module.__class__ = float32_gated(T5DenseGatedActDense)
    assert_loaded(model, "FLAN-T5 encoder")
    return model.eval()


@torch.inference_mode()
def encode(encoder, ids: Tensor, mask: Tensor) -> Tensor:
    """Hidden states [B, L, d_model] in float32 (the padding positions as the encoder leaves them)."""
    device = encoder.shared.weight.device
    embeds = encoder.shared(ids.to(device)).float()
    return encoder(inputs_embeds=embeds, attention_mask=mask.to(device).long()).last_hidden_state.float()
