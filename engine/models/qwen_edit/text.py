# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Qwen-Image-Edit-2511 prompt and picture encoding: Qwen2.5-VL-7B (text and vision tower) as the encoder.

The network classes are transformers' `Qwen2_5_VLForConditionalGeneration` (Apache-2.0); this module owns the file
loading, the token and picture layout and the glue. What diffusers 0.42 `QwenImageEditPlusPipeline` (Apache-2.0)
does with the same model, and how this module follows it:

1. Text. The editing template (`TEMPLATE_EDIT`, the system prompt of the pipeline) holds the user turn
   `Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: ...` followed by the prompt, one announcement per
   picture in the order given (1 to 3). The Qwen2 BPE tokenizer adds no start or end token and nothing is padded.
   Every `<|image_pad|>` (151655) is replaced by the embeddings of its picture. The tokenizer files are the official
   `processor/` files of Qwen/Qwen-Image-Edit-2511 (vocab.json, merges.txt, tokenizer_config.json): 22 added tokens,
   151643 to 151664, so the literal text `<think>`, `</think>`, `<tool_response>` or `</tool_response>` in a prompt
   is ordinary text (BPE pieces), as in the official tokenizer.json.
2. Positions. Transformers' multimodal rope (`Qwen2_5_VLModel.get_rope_index`): the tokens of a picture whose merged
   grid is gh x gw get (time, height, width) = (P, P + row, P + column), where P continues the text before it; the text
   after it continues at P + max(gh, gw); text alone counts 0, 1, 2 ... on all three axes.
3. Hidden state. The output of the last decoder layer after the final RMSNorm (`hidden_states[-1]` of the pipeline).
   The template up to and including `<|im_start|>user\n` is dropped (the pipeline drops the same 64 tokens), so the
   conditioning starts with the picture announcements.
4. Pictures. The encoder sees a copy of about 384 x 384 pixels of area of every picture (`CONDITION_IMAGE_SIZE` of the
   pipeline). Here: `scale = sqrt(384 * 384 / (w * h))`, sides `round(w * scale)` and `round(h * scale)`, torch "area"
   interpolation on the 0..1 tensor; then each side is rounded to a multiple of 28 (Python `round`), the copy is
   resized bilinearly (no antialias), normalised with the CLIP mean and standard deviation, repeated twice in time
   (temporal patch 2) and cut into 14 x 14 patches in the merge-2 order: `[grid_h * grid_w, 3 * 2 * 14 * 14]` and the
   grid (1, h / 14, w / 14). The pipeline rounds the copy to multiples of 32 and lets the image processor resize it
   bicubically; the bilinear resize and the multiple of 28 are Contsy's own choices.
5. Mask. The pad token `<|endoftext|>` (151643) marks the end of the text: it and everything after it are masked
   (pictures always count) and cut from the conditioning, so a prompt that contains that literal text ends there.

Compute and memory. The catalog file holds the encoder in scaled fp8 (`X.weight` float8 next to a 0-dimensional
float32 `X.scale_weight`); the installer makes it from the official bfloat16 shards (`convert.quantize_text_encoder`).
The linear layers keep their fp8 bytes on the device with their float32 scale and expand one matrix per call
(`_Fp8Linear`, at most 270 MB at a time) and compute in float32; the embedding table stays bfloat16; norms, biases and
the patch embedding are float32. Resident weights of the real file: about 8.3 GB (the output head `lm_head.weight`,
1.1 GB, is not used by the conditioning: the converted file leaves it out and a file that has it does not load it).
Float32 compute is Contsy's own choice (the pipeline runs the encoder in bfloat16): it is not slower here and keeps
the conditioning closer to the full-precision result.

Transitional: the scaled-fp8 encoder of the previous catalog (`qwen_2.5_vl_7b_fp8_scaled.safetensors`) also carries a
`scaled_fp8` marker and `X.scale_input` tensors, which are skipped (`_is_bookkeeping`). A running Studio that started
on the previous catalog still hands that file to this module until it restarts; remove the skip with the other
previous-catalog paths after that restart.
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from torch import Tensor, nn

from ...formats import fp8
from ..zimage import assert_loaded, materialize

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "qwen_vl_text_encoder.json"

# The editing prompt of diffusers' QwenImageEditPlusPipeline (pictures are announced inside the user turn).
TEMPLATE_EDIT = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the "
    "user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
PICTURE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
MAX_IMAGES = 3  # pictures per pass

PAD, IM_START, USER, NEWLINE, IMAGE_PAD = 151643, 151644, 872, 198, 151655

AREA = 384 * 384  # pixels of the copy the encoder sees
PATCH, MERGE, TEMPORAL = 14, 2, 2
MIN_PIXELS, MAX_PIXELS = 3136, 12845056
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
PATCH_VALUES = 3 * TEMPORAL * PATCH * PATCH  # numbers per patch row

COMPUTE_DTYPE = torch.float32  # what the linear layers compute in; see the docstring
FILE_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
               "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}

__all__ = ["load_tokenizer", "load_text_encoder", "prepare_images", "encode_prompt", "token_ids",
           "VisionInput", "TEMPLATE_EDIT", "PICTURE", "MAX_IMAGES"]


# ----------------------------------------------------------------------------- tokens

def load_tokenizer(tokenizer_dir: str | Path):
    """Load the Qwen2 tokenizer from local vocab, merges and configuration files."""
    from transformers import Qwen2Tokenizer

    directory = Path(tokenizer_dir)
    if not (directory / "vocab.json").is_file():
        raise FileNotFoundError(f"no Qwen2 tokenizer files (vocab.json, merges.txt) in {directory}")
    return Qwen2Tokenizer.from_pretrained(str(directory))


def _compose(prompt: str, n_images: int) -> str:
    """The text that goes to the tokenizer: picture announcements and prompt inside the editing template."""
    if isinstance(n_images, bool) or not isinstance(n_images, int) or not 1 <= n_images <= MAX_IMAGES:
        raise ValueError(f"the editing model takes 1 to {MAX_IMAGES} pictures, got {n_images!r}")
    return TEMPLATE_EDIT.format("".join(PICTURE.format(i + 1) for i in range(n_images)) + prompt)


def token_ids(tokenizer, prompt: str, n_images: int) -> List[int]:
    """Token ids of the templated prompt; each picture shows up as the placeholder 151655 (`<|image_pad|>`), which
    `encode_prompt` replaces by the picture's embeddings."""
    return [int(token) for token in tokenizer(_compose(prompt, n_images))["input_ids"]]


def _template_end(ids: Sequence[int]) -> int:
    """Where the conditioning starts: right after the template's `<|im_start|>user\\n` (the second `<|im_start|>`)."""
    starts = [i for i, token in enumerate(ids) if token == IM_START]
    if len(starts) < 2 or list(ids[starts[1] + 1:starts[1] + 3]) != [USER, NEWLINE]:
        raise ValueError("the prompt template has no user turn")
    return starts[1] + 3


def _token_mask(ids: Sequence[int], replaced: Sequence[int]) -> List[int]:
    """The attention mask of the tokens that are not replaced by pictures (the list has `len(ids) - len(replaced)`
    entries). The pad token is the end marker: it and everything after it get 0 (a pad at the very start counts as
    left padding and is skipped)."""
    skipped = set(replaced)
    mask: List[int] = []
    ended = left_pad = False
    for index, token in enumerate(ids):
        if index in skipped:
            continue
        if index == 0 and token == PAD:
            left_pad = True
        if ended or (left_pad and token == PAD):
            mask.append(0)
        else:
            mask.append(1)
            left_pad = False
        if not ended and token == PAD and not left_pad:
            mask[-1] = 0
            ended = True
    return mask


# ----------------------------------------------------------------------------- pictures

class VisionInput(NamedTuple):
    """What the vision tower needs for a list of pictures: all patch rows one after another and one grid per picture."""
    pixel_values: Tensor  # [sum of grid_h * grid_w, 1176] float32
    grid_thw: Tensor  # [n, 3] long: (1, rows, columns) in patches of 14 x 14


def _picture_tensor(picture: Image.Image) -> Tensor:
    """[1, H, W, 3] float32 in 0..1 (RGB, an alpha channel dropped)."""
    if not hasattr(picture, "convert"):
        raise TypeError(f"expected a PIL image, got {type(picture).__name__}")
    if picture.mode == "I":
        picture = picture.point(lambda value: value * (1 / 255))
    return torch.from_numpy(np.array(picture.convert("RGB")).astype(np.float32) / 255.0)[None]


def _vl_copy(picture: Tensor) -> Tensor:
    """The copy of a [1, H, W, 3] picture the encoder looks at: 384 x 384 pixels of area, averaged ('area')."""
    samples = picture.movedim(-1, 1)
    scale = math.sqrt(AREA / (samples.shape[3] * samples.shape[2]))
    width, height = round(samples.shape[3] * scale), round(samples.shape[2] * scale)
    return F.interpolate(samples, size=(height, width), mode="area").movedim(1, -1)


def _process_image(picture: Tensor, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Tuple[Tensor, Tensor]:
    """Patch rows and grid of one [1, H, W, 3] picture (see the docstring of the module). Returns
    ([grid_h * grid_w, 1176], [3])."""
    height, width = picture.shape[1], picture.shape[2]
    image = picture[0, ..., :3].permute(2, 0, 1)
    factor = PATCH * MERGE
    rows, cols = round(height / factor) * factor, round(width / factor) * factor
    if rows * cols > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        rows = max(factor, math.floor(height / beta / factor) * factor)
        cols = max(factor, math.floor(width / beta / factor) * factor)
    elif rows * cols < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        rows, cols = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    resized = F.interpolate(image[None], size=(rows, cols), mode="bilinear", align_corners=False)[0]
    normalised = resized.clone()
    for channel in range(3):
        normalised[channel] = (resized[channel] - MEAN[channel]) / STD[channel]
    grid_h, grid_w = rows // PATCH, cols // PATCH
    frames = normalised[None].repeat(TEMPORAL, 1, 1, 1)  # one still picture: the same frame twice
    patches = frames.reshape(1, TEMPORAL, 3, grid_h // MERGE, MERGE, PATCH, grid_w // MERGE, MERGE, PATCH)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(grid_h * grid_w, PATCH_VALUES)
    return patches, torch.tensor([1, grid_h, grid_w], dtype=torch.long)


def prepare_images(images: Sequence[Image.Image]) -> VisionInput:
    """The 384 x 384-area copies of the pictures as the patch rows and grids the vision tower takes (see the docstring)."""
    rows, grids = [], []
    for picture in images:
        patches, grid = _process_image(_vl_copy(_picture_tensor(picture)))
        rows.append(patches)
        grids.append(grid)
    if not rows:
        return VisionInput(torch.zeros((0, PATCH_VALUES)), torch.zeros((0, 3), dtype=torch.long))
    return VisionInput(torch.cat(rows), torch.stack(grids))


# ----------------------------------------------------------------------------- the network

class _Fp8Linear(nn.Linear):
    """A linear layer whose weight stays float8 on the device next to a float32 scale (`scale_weight`).

    The matrix is expanded for the call only, `weight.to(float32) * scale`, then `F.linear` runs in float32."""

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(torch.float32).mul_(self.scale_weight)
        bias = None if self.bias is None else self.bias.to(COMPUTE_DTYPE)
        return F.linear(x.to(COMPUTE_DTYPE), weight, bias)


class _CastLinear(nn.Linear):
    """A linear layer that computes in float32 whatever its weight is stored in (bfloat16 files, tests)."""

    def forward(self, x: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.to(COMPUTE_DTYPE)
        return F.linear(x.to(COMPUTE_DTYPE), self.weight.to(COMPUTE_DTYPE), bias)


class _NoHead(nn.Module):
    """Stands in for `lm_head` (1.1 GB, unused by the conditioning): the encoder is not for generating text."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("this encoder was loaded without the output head; call encoder.model for hidden states")


@contextlib.contextmanager
def _quiet_transformers() -> Iterator[None]:
    """The shipped config keeps `rope_scaling` in the older spelling; transformers 5 warns about it on every build."""
    from transformers.utils import logging as hf_logging

    previous = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(previous)


def _config_dict(config: Optional[Dict] = None) -> Dict:
    if config is not None:
        return config
    return {k: v for k, v in json.loads(CONFIG_FILE.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def _build(config: Optional[Dict] = None):
    """Qwen2_5_VLForConditionalGeneration on the meta device (default: the shipped Qwen-Image-Edit-2511 config)."""
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    with _quiet_transformers():
        configuration = Qwen2_5_VLConfig(**_config_dict(config))
    with torch.device("meta"):
        return Qwen2_5_VLForConditionalGeneration(configuration)


def _convert_name(key: str) -> str:
    """Tensor names of the encoder file (the transformers 4 layout of the official shards) -> the names of
    transformers 5: the decoder moves under `model.language_model`, the vision tower under `model.visual`. Names that
    already follow transformers 5 pass."""
    if key.startswith(("model.language_model.", "model.visual.")):
        return key
    if key in ("lm_head.weight", "model.lm_head.weight"):
        return "lm_head.weight"
    if key.startswith("visual."):
        return "model." + key
    if key.startswith("model."):
        return "model.language_model." + key[len("model."):]
    return key


def _is_bookkeeping(key: str) -> bool:
    """The marker and input scales of the previous catalog's encoder file (see the module docstring): not weights."""
    return key == "scaled_fp8" or key.endswith(".scale_input")


class _Header(NamedTuple):
    tensors: Dict[str, Tuple[Tuple[int, ...], torch.dtype]]  # transformers name -> (shape, dtype) of every real tensor
    scaled: Set[str]  # module paths whose weight is scaled fp8


def _check_format(path: str | Path) -> None:
    """Refuse a file the loader cannot read, with a message that says what to use instead."""
    if Path(path).suffix.lower() == ".gguf":
        raise fp8.UnsupportedQuantization(
            f"{Path(path).name}: a GGUF text encoder is not supported for Qwen-Image-Edit yet. Use the catalog's "
            "scaled-fp8 encoder (made from the official Qwen/Qwen-Image-Edit-2511 shards by the installer) or a "
            "bf16/fp16 safetensors file of the same model.")


def _read_header(path: str | Path) -> _Header:
    """Names, shapes and dtypes of the file without reading any weight."""
    _check_format(path)
    tensors, scaled = {}, set()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if _is_bookkeeping(key):
                continue
            if key.endswith(".scale_weight"):
                scaled.add(_convert_name(key[:-len(".scale_weight")] + ".weight")[:-len(".weight")])
            else:
                piece = handle.get_slice(key)
                tensors[_convert_name(key)] = (tuple(piece.get_shape()), FILE_DTYPES[piece.get_dtype()])
    return _Header(tensors, scaled)


def _stored(name: str, ndim: int, source: torch.dtype, dtype: torch.dtype) -> torch.dtype:
    """Storage dtype of one tensor on the device. fp8 bytes and their scales are kept as they are; small tensors (norms,
    biases, the 5-D patch embedding) are float32, exact for 16-bit sources; the big matrices (the embedding table, the
    weights of unscaled linears) are `dtype`, but 16-bit sources keep their own 16-bit type when `dtype` is bfloat16."""
    if name.endswith(".scale_weight"):
        return torch.float32
    if source in fp8.FP8_TYPES:
        return source
    if ndim != 2:
        return torch.float32
    if dtype is torch.bfloat16 and source in (torch.float16, torch.bfloat16):
        return source
    return dtype


def _file_tensors(path: Path, expand: bool) -> Iterator[Tuple[str, Tensor]]:
    """Tensors of the file under transformers names; `lm_head.weight` and the bookkeeping tensors are left out. With
    `expand`, scaled fp8 weights come out dequantized in float32; otherwise they stay fp8 and their scales follow as
    `X.scale_weight` (0-dimensional)."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        present = set(keys)
        for key in keys:
            name = _convert_name(key)
            if name == "lm_head.weight" or _is_bookkeeping(key):
                continue
            if key.endswith(".scale_weight"):
                if not expand:
                    yield name, handle.get_tensor(key).to(torch.float32).reshape(())
                continue
            tensor = handle.get_tensor(key)
            if expand and tensor.dtype in fp8.FP8_TYPES:
                scale_key = key[:-len("weight")] + "scale_weight" if key.endswith(".weight") else None
                tensor = (fp8.dequantize_fp8(tensor, handle.get_tensor(scale_key).reshape(()), torch.float32)
                          if scale_key in present else tensor.to(torch.float32))
            yield name, tensor


def _convert_linears(model: nn.Module, scaled: Set[str]) -> None:
    """Every `nn.Linear` computes in float32 whatever it stores; those named in `scaled` also keep fp8 plus a scale."""
    for name, module in list(model.named_modules()):
        if type(module) is not nn.Linear:
            continue
        if name in scaled:
            module.__class__ = _Fp8Linear
            module.register_buffer("scale_weight", torch.empty((), dtype=torch.float32, device="meta"))
        else:
            module.__class__ = _CastLinear


def load_text_encoder(path: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16,
                      config: Optional[Dict] = None):
    """Qwen2_5_VLForConditionalGeneration (decoder and vision tower) from a single safetensors file.

    The scaled-fp8 file of `convert.quantize_text_encoder` (`qwen_2.5_vl_7b_edit_2511_fp8.safetensors`), or a bf16 /
    fp16 file with the same names (transformers' own names work too). The model is built on the meta device and filled
    tensor by tensor. For `dtype` bfloat16 or float16 the scaled fp8 weights stay fp8 on `device` and are expanded one
    matrix at a time (see the docstring); `torch.float32` expands every weight when loading (tests, CPU). The decoder's
    `lm_head` is not loaded. `config` replaces the shipped config (tiny models in tests)."""
    header = _read_header(path)
    keep_fp8 = dtype is not torch.float32
    model = _build(config)
    model.lm_head = _NoHead()
    _convert_linears(model, header.scaled if keep_fp8 else set())
    # copy=False: a tensor already in its stored dtype on `device` (the CPU when the encoder is streamed) stays the
    # file's mapping, read from the file cache as its layer streams, instead of a 7.7 GB private copy made and filled
    # on every composition (as UMT5, engine/models/wan22/text.py)
    materialize(model, _file_tensors(Path(path), expand=not keep_fp8), device, dtype, "qwen2.5-vl encoder",
                dtype_for=lambda name, tensor: _stored(name, tensor.ndim, tensor.dtype, dtype), copy=False)
    with torch.device(device):  # the rotary tables are not in the file
        decoder, vision = model.model.language_model, model.model.visual
        decoder.rotary_emb = type(decoder.rotary_emb)(decoder.config)
        vision.rotary_pos_emb = type(vision.rotary_pos_emb)(vision.config)
    assert_loaded(model, "qwen2.5-vl encoder")
    return model.eval()


# ----------------------------------------------------------------------------- positions and encoding

def _rope_positions(model, ids: Sequence[int], slots: Sequence[int], sizes: Sequence[int], grids: Tensor,
                    mask: Tensor) -> Tensor:
    """The (3, L) multimodal rope positions of the sequence from transformers' `get_rope_index`: `ids` with the
    picture placeholder at `slots` written out once per picture embedding (`sizes`), `grids` the (1, rows, columns)
    patch grids, `mask` the [1, L] attention mask."""
    expanded: List[int] = []
    kinds: List[int] = []
    cursor = 0
    for slot, size in zip(slots, sizes):
        expanded += list(ids[cursor:slot]) + [IMAGE_PAD] * size
        kinds += [0] * (slot - cursor) + [1] * size
        cursor = slot + 1
    expanded += list(ids[cursor:])
    kinds += [0] * (len(ids) - cursor)
    device = mask.device
    positions, _ = model.get_rope_index(input_ids=torch.tensor([expanded], device=device),
                                        mm_token_type_ids=torch.tensor([kinds], device=device),
                                        image_grid_thw=grids.to(device), attention_mask=mask)
    return positions[:, 0]


@torch.inference_mode()
def encode_prompt(tokenizer, encoder, prompt: str, images: Sequence[Image.Image]) -> Tensor:
    """The conditioning of a prompt and one to three pictures: float32 [1, L, 3584].

    The tokens of the template up to and including `<|im_start|>user\\n` are dropped (they only provide context); the
    pictures, their `Picture i:` announcements, the prompt and the closing turn markers remain. No picture or more than
    three raise ValueError."""
    pictures = list(images)
    ids = token_ids(tokenizer, prompt, len(pictures))
    vision = prepare_images(pictures)
    model = encoder.model
    table = model.language_model.embed_tokens
    device = table.weight.device
    slots = [i for i, token in enumerate(ids) if token == IMAGE_PAD][:len(pictures)]
    if len(slots) != len(pictures):
        raise ValueError("the template lost a picture placeholder")

    features, offset = [], 0
    for grid in vision.grid_thw:  # one picture at a time: the vision tower's attention never crosses pictures
        count = int(grid.prod())
        pixels = vision.pixel_values[offset:offset + count].to(device)
        offset += count
        features.append(model.visual(pixels, grid_thw=grid[None].to(device)).pooler_output.to(torch.float32))

    embedded = table(torch.tensor(ids, device=device)).to(torch.float32)
    kept = iter(_token_mask(ids, slots))
    rows, flags, cursor = [], [], 0
    for slot, feature in zip(slots, features):
        rows += [embedded[cursor:slot], feature]
        flags += [next(kept) for _ in range(slot - cursor)] + [1] * feature.shape[0]
        cursor = slot + 1
    rows.append(embedded[cursor:])
    flags += list(kept)
    embeds = torch.cat(rows)[None]
    mask = torch.tensor([flags], dtype=torch.long)

    positions = _rope_positions(model, ids, slots, [f.shape[0] for f in features], vision.grid_thw, mask.to(device))
    masked = bool((mask == 0).any())
    hidden = model.language_model(inputs_embeds=embeds,
                                  attention_mask=mask.to(device) if masked else None,
                                  position_ids=positions[:, None, :].to(device),
                                  use_cache=False).last_hidden_state.to(torch.float32)

    start = _template_end(ids)
    tail = mask[0, start:]
    valid = int(tail.sum())
    if not bool((tail[:valid] == 1).all()):
        raise ValueError("the attention mask is not a run of ones followed by zeros")
    return hidden[:, start:][:, :valid]
