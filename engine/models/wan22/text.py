# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Wan 2.2 prompt encoding: UMT5-XXL as the text encoder.

What the official code does (Wan-Video/Wan2.2, Apache-2.0: `wan/modules/t5.py` `T5EncoderModel` and
`wan/modules/tokenizers.py` `HuggingfaceTokenizer` with seq_len 512; diffusers 0.42 `WanPipeline._get_t5_prompt_embeds`)
and how this module follows it:

* The weights are Wan-AI's own encoder: the bfloat16 `text_encoder/` shards of Wan-AI/Wan2.2-T2V-A14B-Diffusers
  (transformers names, read through their `model.safetensors.index.json`; the I2V repository holds the same files).
  They are kept in bfloat16 as stored, which takes the same memory as the 16-bit expansion of an fp8 file. A single
  safetensors file in the same key layout and the Lite GGUF are accepted too.
* The prompt's SentencePiece tokens, then the end token `</s>` (id 1, what `add_special_tokens` adds), then padding
  with id 0 up to 512 tokens. The SentencePiece model is the official google/umt5-xxl `spiece.model`, which Wan-AI
  ships as `tokenizer/spiece.model` (SHA-256 e3909a67...); it is read from beside the shards, or from a
  `spiece_model` tensor of a single-file encoder.
* The attention mask is 1 up to and including the end token and 0 on the padding. The official code keeps the hidden
  states up to the true length and pads them with zeros to 512; here the hidden state after the final layer norm is
  multiplied by the mask, which gives the same exact zeros.
* Parentheses and every other character are plain prompt text: there is no weighting syntax.
* Differences from the official code: no text cleaning before tokenizing (the official `prompt_clean` collapses
  whitespace and fixes HTML entities) and no truncation at 512 tokens (a longer prompt keeps all its tokens).
* Every linear layer computes in float32 whatever the weights are stored in (`CastLinear`, expanding each matrix
  only for the moment of its use). The feed-forward block of the transformers class would round its activation to
  the dtype of `wo` first, so its class is replaced (`float32_gated`) to keep float32 up to the last product.
* The Lite GGUF stores its tokenizer as metadata and its linear layers as quantized blocks.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor, nn

from ...core.quant import materialize_quantized
from ...formats import fp8
from ...formats.ggml import GGUFFile
from ..zimage import assert_loaded, materialize

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "wan_text_encoder.json"
END, PAD = 1, 0
MIN_LENGTH = 512
INDEX_SUFFIX = ".safetensors.index.json"  # the index of a sharded safetensors set (transformers' naming)
TOKENIZER_FILE = "spiece.model"


def _sharded(path: str | Path) -> bool:
    return Path(path).name.lower().endswith(INDEX_SUFFIX)


def check_format(path: str | Path) -> None:
    if not _sharded(path) and Path(path).suffix.lower() not in (".gguf", ".safetensors"):
        raise fp8.UnsupportedQuantization(f"{Path(path).name}: expected a GGUF or safetensors UMT5 encoder "
                                          f"(one file, or the {INDEX_SUFFIX} of a sharded set)")


def _shards(index: Path) -> List[Path]:
    """The shard files that a sharded set's index names, in order; each must be a plain file name beside the index."""
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index.name}: no weight_map in the index")
    shards = []
    for name in dict.fromkeys(sorted(weight_map.values())):
        if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".safetensors"):
            raise ValueError(f"{index.name}: shard {name!r} must be a .safetensors file beside the index")
        shards.append(index.parent / name)
    return shards


def _tokenizer_beside(path: Path) -> Optional[Path]:
    """`spiece.model` beside the encoder, or in the `tokenizer/` folder next to it (the Diffusers repository layout)."""
    for candidate in (path.parent / TOKENIZER_FILE, path.parent.parent / "tokenizer" / TOKENIZER_FILE):
        if candidate.is_file():
            return candidate
    return None


def files(path: str | Path) -> List[Path]:
    """Every file the encoder at `path` is read from (the index, its shards and the tokenizer for a sharded set)."""
    check_format(path)
    path = Path(path)
    if not _sharded(path):
        return [path]
    tokenizer = _tokenizer_beside(path)
    return [path, *_shards(path), *([tokenizer] if tokenizer else [])]


_GGUF_LAYER_NAMES = {
    "attn_q": "layer.0.SelfAttention.q", "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v", "attn_o": "layer.0.SelfAttention.o",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias", "attn_norm": "layer.0.layer_norm",
    "ffn_up": "layer.1.DenseReluDense.wi_1", "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_down": "layer.1.DenseReluDense.wo", "ffn_norm": "layer.1.layer_norm",
}


def _gguf_name(name: str) -> str:
    if name == "token_embd.weight":
        return "shared.weight"
    if name == "enc.output_norm.weight":
        return "encoder.final_layer_norm.weight"
    block = re.fullmatch(r"enc\.blk\.(\d+)\.([a-z_]+)\.weight", name)
    if block and block[2] in _GGUF_LAYER_NAMES:
        return f"encoder.block.{block[1]}.{_GGUF_LAYER_NAMES[block[2]]}.weight"
    return name


def _gguf_spiece(file: GGUFFile) -> bytes:
    """Restore the unigram vocabulary and flags used by the Lite UMT5 tokenizer."""
    from sentencepiece import sentencepiece_model_pb2 as sp

    def field(key, default=None):
        return file.field(f"tokenizer.ggml.{key}", default)

    if file.arch not in ("t5", "t5encoder") or field("model") != "t5":
        raise ValueError("Expected T5 tokenizer metadata in the UMT5 GGUF file.")
    model = sp.ModelProto()
    model.normalizer_spec.add_dummy_prefix = field("add_space_prefix", True)
    model.normalizer_spec.remove_extra_whitespaces = field("remove_extra_whitespaces", False)
    tokens, scores, types = (field(key) for key in ("tokens", "scores", "token_type"))
    if not tokens or scores is None or types is None or not len(tokens) == len(scores) == len(types):
        raise ValueError("The UMT5 GGUF tokenizer vocabulary is incomplete.")
    for token, score, kind in zip(tokens, scores, types):
        model.pieces.add(piece=token, score=score, type=kind)
    model.trainer_spec.model_type = sp.TrainerSpec.UNIGRAM
    model.trainer_spec.byte_fallback = any(kind == sp.ModelProto.SentencePiece.BYTE for kind in types)
    model.trainer_spec.vocab_size = len(tokens)
    model.trainer_spec.eos_id = field("eos_token_id", END)
    model.trainer_spec.pad_id = field("padding_token_id", PAD)
    model.trainer_spec.unk_id = field("unknown_token_id", 2)
    return model.SerializeToString()


def read_spiece(path: str | Path) -> bytes:
    """The SentencePiece model of an encoder: `spiece.model` beside a sharded set (or in the `tokenizer/` folder next
    to it), the `spiece_model` tensor of a single safetensors file, or the vocabulary of the Lite GGUF."""
    check_format(path)
    path = Path(path)
    if path.suffix.lower() == ".gguf":
        with GGUFFile(path) as file:
            return _gguf_spiece(file)
    if not _sharded(path):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if "spiece_model" in handle.keys():
                return bytes(handle.get_tensor("spiece_model").numpy().tobytes())
    tokenizer = _tokenizer_beside(path)
    if tokenizer is None:
        raise FileNotFoundError(f"{path.name}: no {TOKENIZER_FILE} beside the encoder or in ../tokenizer/")
    return tokenizer.read_bytes()


def load_tokenizer(model: bytes | str | Path):
    """A SentencePiece processor from model bytes or a `.model` file."""
    import sentencepiece

    if isinstance(model, (bytes, bytearray)):
        return sentencepiece.SentencePieceProcessor(model_proto=bytes(model))
    return sentencepiece.SentencePieceProcessor(model_file=str(model))


def tokenize(sp, prompt: str, min_length: int = MIN_LENGTH) -> List[int]:
    """Token ids of the prompt: its SentencePiece pieces, the end token, padding up to `min_length`."""
    tokens = [int(t) for t in sp.encode(prompt)] + [END]
    return tokens + [PAD] * max(0, min_length - len(tokens))


class CastLinear(nn.Linear):
    """A linear layer that computes in float32 whatever its weight is stored in."""

    def forward(self, x: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.float()
        return F.linear(x.float(), self.weight.float(), bias)


@lru_cache(maxsize=None)
def float32_gated(base: type) -> type:
    """`base` (the gated feed-forward block of transformers' T5 or UMT5) with its forward minus the cast of the
    activation to the dtype of `wo`.

    The cast (made for 8-bit loading) would round the float32 activation to bfloat16 in every layer when the weights
    are stored in bfloat16; without it the activation stays float32 up to the last matrix product. A subclass is
    swapped in, never a bound method stored on the module: that would be a reference cycle, and the module's GPU
    weights would stay allocated after every call until the garbage collector happened to run."""
    class GatedDense(base):
        def forward(self, hidden_states: Tensor) -> Tensor:
            return self.wo(self.dropout(self.act(self.wi_0(hidden_states)) * self.wi_1(hidden_states)))

    return GatedDense


def _stored_precision(dtype: torch.dtype):
    """The rule for `materialize`: small tensors keep float32 (norm scales, relative position biases) and, when the
    weights are to be bfloat16, every 16-bit matrix keeps the 16-bit dtype it arrived in (the embedding table, and
    the weights that `_file_tensors` expanded from fp8), so that nothing is rounded to bfloat16 on the way."""
    def rule(name: str, tensor: Tensor) -> Optional[torch.dtype]:
        if tensor.ndim == 1 or "relative_attention_bias" in name:
            return torch.float32
        if dtype is torch.bfloat16 and tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor.dtype
        return None
    return rule


def _build(config: Optional[Dict] = None):
    """A UMT5EncoderModel on the meta device (default: the shipped Wan 2.2 text encoder config)."""
    from transformers import UMT5Config, UMT5EncoderModel

    if config is None:
        config = {k: v for k, v in json.loads(CONFIG_FILE.read_text(encoding="utf-8")).items()
                  if not k.startswith("_")}
    with torch.device("meta"):
        return UMT5EncoderModel(UMT5Config(**config))


def _file_tensors(path: Path, narrow: bool) -> Iterator[Tuple[str, Tensor]]:
    """Tensors of the file, or of every shard of a sharded set, without an embedded tokenizer (a scaled fp8 matrix
    comes out expanded in float32; bf16 and fp16 tensors as stored).

    With `narrow`, a float32 matrix (an expanded fp8 weight) is stored as float16 when its values fit: a weight times
    its scale then keeps 11 significant bits, where bfloat16 keeps 8. Measured on nine real matrices of the fp8 file, the
    output of a layer is off by 0.14 percent with bfloat16 and 0.02 percent with float16; the conditioning of the whole
    encoder was 0.6 percent off the float32 product (mean) with bfloat16. Memory is the same. A matrix that does not fit float16
    becomes bfloat16."""
    for part in (_shards(path) if _sharded(path) else [path]):
        for name, tensor in fp8.iter_tensors(part, torch.float32):
            if name == "spiece_model":
                continue
            if narrow and tensor.dtype is torch.float32 and tensor.ndim == 2:
                tensor = tensor.to(torch.float16 if float(tensor.abs().max()) < 60000 else torch.bfloat16)
            yield name, tensor


def load_text_encoder(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16,
                      config: Optional[Dict] = None):
    """UMT5 encoder from safetensors in the transformers key layout (Wan-AI's bf16 shards through their
    `model.safetensors.index.json`, or one file in bf16, fp16, fp32 or scaled fp8) or from the Lite GGUF.

    Weights are stored in `dtype` (for bfloat16: in the 16-bit dtype they arrive in, see `_file_tensors`); the small
    tensors (norm scales, relative position biases) stay float32, and every linear layer computes in float32 (see
    `CastLinear`). Loaded to the CPU (the product streams the blocks to the GPU), the bfloat16 matrices of Wan-AI's
    shards are not copied: they stay safetensors' maps of the files, so the 10.6 GiB are read once, as the blocks are
    first streamed, and live in the file cache rather than in the process (measured: load and first prompt 9.6 s
    instead of 12.4 s, the same encodings bit for bit)."""
    check_format(path)
    model = _build(config)
    optional = ["encoder.embed_tokens.weight"]
    if Path(path).suffix.lower() == ".gguf":
        def storage(name, _tensor):
            return torch.float16 if name == "shared.weight" and dtype != torch.float32 else torch.float32

        with GGUFFile(path) as file:
            if file.arch not in ("t5", "t5encoder"):
                raise ValueError(f"Expected a T5 encoder, got {file.arch!r}")
            materialize_quantized(model, ((_gguf_name(name), tensor) for name, tensor in file.tensors.items()),
                                  device, dtype, "umt5", optional=optional, dtype_for=storage)
    else:
        materialize(model, _file_tensors(Path(path), dtype is torch.bfloat16), device, dtype, "umt5",
                    optional=optional, dtype_for=_stored_precision(dtype), copy=False)
    model.encoder.embed_tokens.weight = model.shared.weight  # the file stores the shared table once
    for module in model.modules():
        if type(module) is nn.Linear:
            module.__class__ = CastLinear
        elif type(module).__name__ == "UMT5DenseGatedActDense":
            module.__class__ = float32_gated(type(module))
    assert_loaded(model, "umt5")
    return model.eval()


@torch.inference_mode()
def _encode_tokens(encoder, tokens: List[int], device: torch.device) -> Tensor:
    """Hidden states [1, L, D] in float32, zero where the attention mask is 0."""
    ids = torch.tensor([tokens], device=device)
    mask = torch.zeros_like(ids)
    mask[:, :tokens.index(END) + 1] = 1  # the end token itself still counts
    embeds = encoder.shared(ids).float()
    hidden = encoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state.float()
    return hidden * mask.unsqueeze(-1).float()


def encode_prompt(sp, encoder, prompt: str, min_length: int = MIN_LENGTH) -> Tensor:
    """The conditioning [1, L, 4096] float32 for a prompt."""
    return _encode_tokens(encoder, tokenize(sp, prompt, min_length), next(encoder.parameters()).device)
