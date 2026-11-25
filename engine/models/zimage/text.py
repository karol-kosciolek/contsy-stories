# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Z-Image-Turbo prompt encoding: Qwen3-4B as the text encoder.

Conventions (diffusers 0.42 `ZImagePipeline`, Apache-2.0, as the Tongyi-MAI Z-Image release uses it):

* The prompt goes into the Qwen chat template with a user turn and an open assistant turn, no system
  prompt and no `<think>` block (`apply_chat_template(..., add_generation_prompt=True,
  enable_thinking=True)`).
* The conditioning is the hidden state after the second-to-last decoder layer, before the final norm
  (`hidden_states[-2]`).
* No truncation and no padding: the sequence has exactly the length of the templated prompt, so the
  attention mask is all ones (the pipeline pads and then keeps only the unmasked rows, which is the same).
* The encoder is stored in bf16 but computes in float32 (Contsy's own choice; the pipeline runs it in
  bf16): `encode` upcasts one decoder layer at a time.

Files: the product reads one bf16 safetensors file that the installer merges from Tongyi-MAI's official
text_encoder shards (`checkpoint.merge`, the tensors unchanged); the shards themselves, through their index, load as
well. The Lite set reads Qwen's own Qwen3-4B GGUF. The tokenizer files (vocab.json, merges.txt,
tokenizer_config.json of the Qwen2 tokenizer shared by Qwen3, from the same Tongyi-MAI repository) are not part of
the weights; `load_tokenizer` takes the directory that holds them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from ...core.quant import materialize_quantized
from ...formats.ggml import GGUFFile
from . import assert_loaded, checkpoint, materialize

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "zimage_text_encoder.json"
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
HIDDEN_LAYER = -2  # index into the decoder layers counted from the end: the output of layers[-2]
# The transformer's position table has 1536 entries on the sequence axis and the caption is padded
# to a multiple of 32, so a prompt must stay well below that (real prompts use a few hundred tokens).
MAX_PROMPT_TOKENS = 1024


def load_tokenizer(directory: str | Path):
    """The Qwen2 tokenizer from a directory with vocab.json, merges.txt and tokenizer_config.json."""
    from transformers import Qwen2Tokenizer

    return Qwen2Tokenizer.from_pretrained(str(directory))


def tokenize(tokenizer, prompt: str) -> List[int]:
    """Token ids of the templated prompt."""
    ids = tokenizer(TEMPLATE.format(prompt))["input_ids"]
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f"the prompt has {len(ids)} tokens, the limit is {MAX_PROMPT_TOKENS}; shorten it")
    return ids


def _build(config: Optional[Dict] = None):
    """A Qwen3Model on the meta device. `config` defaults to the shipped Z-Image text encoder config."""
    from transformers import Qwen3Config, Qwen3Model

    if config is None:
        config = {k: v for k, v in json.loads(CONFIG_FILE.read_text(encoding="utf-8")).items()
                  if not k.startswith("_")}
    with torch.device("meta"):
        return Qwen3Model(Qwen3Config(**config))


def _checkpoint_tensors(path: Path) -> Iterator[Tuple[str, Tensor]]:
    """Tensors of the encoder (a safetensors file or a sharded set's index, official `model.*` names) under the
    names of `Qwen3Model` (no `model.` prefix)."""
    for key, tensor in checkpoint.tensors(path):
        yield key.removeprefix("model."), tensor


_GGUF_LAYERS = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}


def _gguf_name(name: str) -> str:
    if name == "token_embd.weight":
        return "embed_tokens.weight"
    if name == "output_norm.weight":
        return "norm.weight"
    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "blk" and parts[2] in _GGUF_LAYERS:
        return f"layers.{parts[1]}.{_GGUF_LAYERS[parts[2]]}.{parts[3]}"
    return name


def load_text_encoder(path: str | Path, device: torch.device | str = "cpu",
                      dtype: torch.dtype = torch.bfloat16, config: Optional[Dict] = None):
    """Build Qwen3Model on the meta device and fill it tensor by tensor from safetensors (one file, or a sharded
    set's index) or GGUF.

    The rotary table is not stored in the file, so it is rebuilt on `device`. `config` replaces the
    shipped config (used with tiny models in tests).
    """
    model = _build(config)
    if Path(path).suffix.lower() == ".gguf":
        with GGUFFile(path) as file:
            if file.arch != "qwen3":
                raise ValueError(f"Expected a Qwen3 encoder, got {file.arch!r}")
            tensors = ((_gguf_name(name), tensor) for name, tensor in file.tensors.items() if name != "output.weight")
            materialize_quantized(model, tensors, device, dtype, "Qwen3 text encoder")
    else:
        materialize(model, _checkpoint_tensors(Path(path)), device, dtype, "text encoder")
    with torch.device(device):
        model.rotary_emb = type(model.rotary_emb)(model.config)
    assert_loaded(model, "text encoder")
    return model.eval()


class _FloatLayers:
    """Context manager: each decoder layer runs in float32 while its weights rest in the storage dtype.

    A forward pre-hook widens the layer, a forward hook narrows it again. Widening and narrowing bf16
    weights is exact, so the model is unchanged afterwards and only one layer is ever widened.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.handles: list = []

    def __enter__(self):
        storage = self.model.embed_tokens.weight.dtype
        if storage == torch.float32:
            return self

        def widen(module, args, kwargs):
            module.float()

        def narrow(module, args, kwargs, output):
            module.to(storage)

        for layer in self.model.layers:
            self.handles.append(layer.register_forward_pre_hook(widen, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(narrow, with_kwargs=True))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def encode(model, ids: List[int]) -> Tensor:
    """Conditioning for one tokenized prompt: float32 [len(ids), hidden_size] from layer -2."""
    device = model.embed_tokens.weight.device
    embeds = model.embed_tokens(torch.tensor([ids], device=device)).float()
    with _FloatLayers(model):
        out = model(inputs_embeds=embeds, output_hidden_states=True)
    return out.hidden_states[HIDDEN_LAYER][0].float()


def encode_prompt(tokenizer, model, prompt: str) -> Tensor:
    return encode(model, tokenize(tokenizer, prompt))

