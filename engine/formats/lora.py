# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""LoRA files: reading them for a diffusers model and applying them without touching the weights.

`load_lora` turns the key style of a file into module names of the target model. The styles that
occur in Contsy's files (the two accelerators of the video and edit models; Contsy has no LoRAs of its own):

* LightX2V Wan 2.2: `diffusion_model.<original Wan name>.lora_down.weight` / `lora_up.weight`, no alpha.
  The I2V files also hold full parameter changes, which the LoRA authors' own loader (ModelTC/LightX2V,
  Apache-2.0, `lightx2v/utils/lora_loader.py`) applies as `parameter + strength * change`: `.diff` for a
  weight (norms, the patch embedding, the output head), `.diff_b` for a bias and `.diff_m` for the
  modulation table of a block or of the output head (`scale_shift_table` of diffusers' Wan blocks,
  [1, 6, dim], and of the model itself, [1, 2, dim]);
* Qwen-Image-Edit Lightning: `<diffusers module>.lora_down.weight` / `lora_up.weight` / `.alpha`.
* PEFT style (the style LoRAs of the animated films, app/styles.py: Qwen-Image-Edit-2511 Anime / 3D,
  Z-Image-Turbo anime / 3D render): `<module>.lora_A[.default].weight` (down) / `lora_B[.default].weight` (up),
  optionally behind `diffusion_model.`, without an alpha (a scale of 1, PEFT's `lora_alpha` = rank).

No file uses a fused qkv projection against separate to_q/to_k/to_v modules, so that layout is not
supported: such a key is reported as unmatched. The image-attention keys of Wan 2.1 I2V that Wan 2.2
I2V no longer has are skipped.

`attach` adds the LoRA output next to the output of each target module with a forward hook, so the
base weights are never modified and it works on top of quantized, fp8 or GGUF layers. A block's
modulation table is a small dense parameter read inside the block, not a module output: it is changed
in place and restored by the function `attach` returns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor, nn


class LoRAKeyError(ValueError):
    """A key in a LoRA file does not fit the target model."""


@dataclass(frozen=True)
class LoRAWeights:
    """One low-rank update of a Linear layer: output += scale * (x @ down.T) @ up.T."""
    down: Tensor  # [rank, in_features]
    up: Tensor  # [out_features, rank]
    alpha: Optional[float] = None

    @property
    def scale(self) -> float:
        return 1.0 if self.alpha is None else self.alpha / self.down.shape[0]


@dataclass
class LoRA:
    modules: Dict[str, LoRAWeights] = field(default_factory=dict)
    # Full changes of parameters, {module path: {"weight" or "bias": change}} (the `.diff` keys).
    deltas: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)
    # Changes of a modulation table, {module path: change} (the `.diff_m` keys; "" is the model itself).
    modulations: Dict[str, Tensor] = field(default_factory=dict)
    # Keys of the file that are deliberately not applied.
    skipped: Tuple[str, ...] = ()

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "LoRA":
        """A copy with every tensor on `device` in `dtype`, so that steps do not upload them again."""
        return LoRA(modules={name: LoRAWeights(w.down.to(device, dtype), w.up.to(device, dtype), w.alpha)
                             for name, w in self.modules.items()},
                    deltas={name: {key: value.to(device, dtype) for key, value in change.items()}
                            for name, change in self.deltas.items()},
                    modulations={name: change.to(device, dtype) for name, change in self.modulations.items()},
                    skipped=self.skipped)


MODULATION = "scale_shift_table"  # diffusers' name of the modulation tables of Wan (blocks and output head)


# Roles of the key suffixes: down/up matrices, alpha, full weight change, full bias change.
_SUFFIXES = {".lora_down.weight": "down", ".lora_up.weight": "up",
             ".lora_A.weight": "down", ".lora_B.weight": "up",
             ".lora_A.default.weight": "down", ".lora_B.default.weight": "up",
             ".alpha": "alpha", ".diff": "weight", ".diff_b": "bias", ".diff_m": "modulation"}

# Original Wan names -> diffusers names, as in the Wan checkpoints and LightX2V LoRAs.
_WAN_RENAMES = tuple((re.compile(p), t) for p, t in (
    (r"blocks\.(\d+)\.self_attn\.([qkv])", r"blocks.\1.attn1.to_\2"),
    (r"blocks\.(\d+)\.cross_attn\.([qkv])", r"blocks.\1.attn2.to_\2"),
    (r"blocks\.(\d+)\.self_attn\.o", r"blocks.\1.attn1.to_out.0"),
    (r"blocks\.(\d+)\.cross_attn\.o", r"blocks.\1.attn2.to_out.0"),
    (r"blocks\.(\d+)\.self_attn\.(norm_[qk])", r"blocks.\1.attn1.\2"),
    (r"blocks\.(\d+)\.cross_attn\.(norm_[qk])", r"blocks.\1.attn2.\2"),
    (r"blocks\.(\d+)\.cross_attn\.k_img", r"blocks.\1.attn2.add_k_proj"),
    (r"blocks\.(\d+)\.cross_attn\.v_img", r"blocks.\1.attn2.add_v_proj"),
    (r"blocks\.(\d+)\.cross_attn\.norm_k_img", r"blocks.\1.attn2.norm_added_k"),
    (r"blocks\.(\d+)\.ffn\.0", r"blocks.\1.ffn.net.0.proj"),
    (r"blocks\.(\d+)\.ffn\.2", r"blocks.\1.ffn.net.2"),
    (r"blocks\.(\d+)\.norm3", r"blocks.\1.norm2"),
    (r"text_embedding\.([02])", lambda m: f"condition_embedder.text_embedder.linear_{int(m[1]) // 2 + 1}"),
    (r"time_embedding\.([02])", lambda m: f"condition_embedder.time_embedder.linear_{int(m[1]) // 2 + 1}"),
    (r"time_projection\.1", "condition_embedder.time_proj"),
    (r"head\.head", "proj_out"),
    (r"head", ""),  # the modulation table of the output head is the model's own scale_shift_table
))

# Wan 2.1 I2V has an image branch in cross attention that Wan 2.2 I2V lacks; its keys are dead weight.
_WAN21_IMAGE_BRANCH = re.compile(r"(^|\.)(k_img|v_img|norm_k_img|img_emb)(\.|$)")


def _rename_wan(name: str) -> str:
    for pattern, template in _WAN_RENAMES:
        match = pattern.fullmatch(name)
        if match:
            return match.expand(template) if isinstance(template, str) else template(match)
    return name


def _split_key(key: str) -> Tuple[str, str]:
    """`diffusion_model.a.b.lora_down.weight` -> ('a.b', 'down')."""
    for suffix, role in _SUFFIXES.items():
        if key.endswith(suffix):
            return key.removeprefix("diffusion_model.")[:-len(suffix)], role
    raise LoRAKeyError(f"unsupported LoRA key {key!r}")


def load_lora(path: str | Path, module_names: Iterable[str]) -> LoRA:
    """Read a LoRA file for a model whose modules are named `module_names` (its `named_modules`).

    Raises LoRAKeyError for any key that fits no module, listing the offenders.
    """
    names = frozenset(module_names)
    parts: Dict[str, Dict[str, Tensor]] = {}
    skipped: List[str] = []
    unmatched: List[str] = []
    with safe_open(str(path), "pt") as file:
        for key in file.keys():
            try:
                stem, role = _split_key(key)
            except LoRAKeyError:
                unmatched.append(key)
                continue
            # The renamed form first: diffusers' Wan blocks have a parameterless `norm3`, while the
            # original `norm3` (the affine cross-attention norm) is their `norm2`.
            target = next((n for n in (_rename_wan(stem), stem) if n in names), None)
            if target is None and _WAN21_IMAGE_BRANCH.search(stem):
                skipped.append(key)
            elif target is None:
                unmatched.append(key)
            else:
                parts.setdefault(target, {})[role] = file.get_tensor(key)
    if unmatched:
        raise LoRAKeyError(f"{Path(path).name}: {len(unmatched)} keys fit no module of the model, "
                           f"for example {unmatched[:3]}")

    lora = LoRA(skipped=tuple(skipped))
    for target, tensors in parts.items():
        if "down" in tensors or "up" in tensors:
            if "down" not in tensors or "up" not in tensors:
                raise LoRAKeyError(f"{target}: only one of the two LoRA matrices is in the file")
            down, up = tensors["down"], tensors["up"]
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise LoRAKeyError(f"{target}: LoRA matrices {tuple(down.shape)} and {tuple(up.shape)} do not fit")
            alpha = tensors.get("alpha")
            lora.modules[target] = LoRAWeights(down, up, None if alpha is None else float(alpha))
        elif "alpha" in tensors:
            raise LoRAKeyError(f"{target}: an alpha without LoRA matrices")
        if "modulation" in tensors:
            lora.modulations[target] = tensors["modulation"]
        delta = {k: tensors[k] for k in ("weight", "bias") if k in tensors}
        if delta:
            lora.deltas[target] = delta
    return lora


def _param_change(module: nn.Module, x: Tensor, delta: Dict[str, Tensor]) -> Tensor:
    """How the output of `module` changes when `delta` is added to its weight and bias."""
    weight, bias = delta.get("weight"), delta.get("bias")
    if isinstance(module, nn.Linear):
        out = F.linear(x, weight) if weight is not None else 0
    elif isinstance(module, nn.modules.conv._ConvNd):
        out = module._conv_forward(x, weight, None) if weight is not None else 0
        if bias is not None:
            bias = bias.reshape(1, -1, *([1] * (x.ndim - 2)))
    elif isinstance(module, nn.RMSNorm) and bias is None:
        out = F.rms_norm(x, module.normalized_shape, None, module.eps) * weight
    elif isinstance(module, nn.LayerNorm):
        normed = F.layer_norm(x.float(), module.normalized_shape, None, None, module.eps)
        out = normed * weight.float() if weight is not None else 0
    else:
        raise TypeError(f"cannot apply a parameter change to {type(module).__name__}")
    return out if bias is None else out + bias


@dataclass
class _Term:
    strength: float
    lora: Optional[LoRAWeights]
    delta: Dict[str, Tensor]


def _check(name: str, module: nn.Module, lora: LoRA) -> None:
    if name in lora.modules:
        weights = lora.modules[name]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name}: LoRA matrices need a Linear layer, the model has {type(module).__name__}")
        if (weights.down.shape[1], weights.up.shape[0]) != (module.in_features, module.out_features):
            raise ValueError(f"{name}: LoRA fits a {weights.down.shape[1]} -> {weights.up.shape[0]} layer, "
                             f"the model has {module.in_features} -> {module.out_features}")
    changes = dict(lora.deltas.get(name, {}))
    if name in lora.modulations:
        changes[MODULATION] = lora.modulations[name]
    for param, change in changes.items():
        current = getattr(module, param, None)
        if current is None or current.shape != change.shape:
            raise ValueError(f"{name}.{param}: change of shape {tuple(change.shape)} does not fit "
                             f"{None if current is None else tuple(current.shape)}")


def _stochastic_bf16(value: Tensor, generator: torch.Generator) -> Tensor:
    """Round float32 `value` to bfloat16 at random, so that the average result equals `value`.

    A bfloat16 number keeps the upper 16 bits of a float32. Adding a random 16-bit number to the
    lower half before clearing it rounds up with a probability proportional to the discarded part."""
    bits = value.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device, generator=generator)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


# Activations with more elements than _SLICE_THRESHOLD (Wan video: 75,600 tokens x 13,824 channels) are processed
# in slices of about _SLICE_ELEMENTS elements, so that the temporaries of the hook (float32 copies for the norm
# changes and for stochastic rounding, the LoRA product) stay small instead of being several times the output.
# Smaller activations are computed in one piece.
_SLICE_ELEMENTS = 1 << 25
_SLICE_THRESHOLD = 1 << 27


def _sliceable(module: nn.Module, x: Tensor, output: Tensor) -> bool:
    """Every token is handled on its own by these layers, so any slice of tokens gives the same numbers."""
    one_dimension = not isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or len(module.normalized_shape) == 1  # else a slice cuts a normalised axis
    return (isinstance(module, (nn.Linear, nn.LayerNorm, nn.RMSNorm)) and one_dimension and output.ndim >= 2
            and x.ndim == output.ndim and x.shape[:-1] == output.shape[:-1] and output.numel() > _SLICE_THRESHOLD)


def _may_write_in_place(output: Tensor) -> bool:
    if output.requires_grad and torch.is_grad_enabled():
        return False
    return not output.is_inference() or torch.is_inference_mode_enabled()


def _in_slices(compute: Callable[[Tensor, Tensor], Tensor], x: Tensor, output: Tensor) -> Tensor:
    """`compute(x_slice, output_slice)` for slices of tokens; the results land in `output` itself when that is
    safe (inference), else in a new tensor, so that no second output-sized tensor is needed."""
    result = output if _may_write_in_place(output) else torch.empty_like(output)
    step = max(1, _SLICE_ELEMENTS // (output.numel() // output.shape[-2]))
    for start in range(0, output.shape[-2], step):
        length = min(step, output.shape[-2] - start)
        piece = compute(x.narrow(-2, start, length), output.narrow(-2, start, length))
        result.narrow(-2, start, length).copy_(piece)
    return result


def attach(model: nn.Module, loras: Sequence[Tuple[LoRA, float]], rounding: str = "nearest",
           seed: int = 0) -> Callable[[], None]:
    """Apply LoRAs to `model` with forward hooks; returns a function that removes them again.

    Each target module's output gets `strength * scale * ((x @ down.T) @ up.T)` added, computed in
    the dtype and on the device of the input. Several LoRAs on one module add up. Nothing is
    registered unless every LoRA fits the model.

    `rounding` matters for bfloat16 outputs. "nearest" adds the term in the output's dtype, so a
    term smaller than half a bfloat16 step is lost (weak LoRAs in early layers lose part of their
    effect). "stochastic" adds it in float32 and rounds the sum at random: the average effect is
    exact, at the cost of rounding noise; it is reproducible for a given `seed` and call order.
    """
    if rounding not in ("nearest", "stochastic"):
        raise ValueError(f"rounding must be 'nearest' or 'stochastic', not {rounding!r}")
    generators: Dict[torch.device, torch.Generator] = {}

    def add(output: Tensor, extra: Tensor) -> Tensor:
        if rounding == "stochastic" and output.dtype is torch.bfloat16:
            generator = generators.get(output.device)
            if generator is None:
                generator = generators[output.device] = torch.Generator(device=output.device)
                generator.manual_seed(seed)
            return _stochastic_bf16(output.float() + extra.float(), generator)
        return output + extra.to(output.dtype)

    modules = dict(model.named_modules())
    terms: Dict[str, List[_Term]] = {}
    modulation: Dict[str, Tensor] = {}  # block path -> summed float32 change of its modulation table
    for lora, strength in loras:
        for name in {*lora.modules, *lora.deltas, *lora.modulations}:
            if name not in modules:
                raise LoRAKeyError(f"the model has no module {name!r}")
            _check(name, modules[name], lora)
            if name in lora.modules or name in lora.deltas:
                terms.setdefault(name, []).append(_Term(strength, lora.modules.get(name), lora.deltas.get(name, {})))
            if name in lora.modulations:
                change = strength * lora.modulations[name].float()
                modulation[name] = change if name not in modulation else modulation[name] + change

    def combine(entries: List[_Term], module: nn.Module, x: Tensor, output: Tensor) -> Tensor:
        extra = 0
        for term in entries:
            if term.lora is not None:
                down = term.lora.down.to(x.device, x.dtype)
                up = term.lora.up.to(x.device, x.dtype)
                extra = extra + term.strength * term.lora.scale * F.linear(F.linear(x, down), up)
            if term.delta:
                change = {k: v.to(x.device, x.dtype) for k, v in term.delta.items()}
                extra = extra + term.strength * _param_change(module, x, change)
        return add(output, extra)

    def make_hook(entries: List[_Term]):
        def hook(module: nn.Module, args: tuple, kwargs: dict, output: Tensor) -> Tensor:
            x = args[0] if args else kwargs["input"]
            if not _sliceable(module, x, output):
                return combine(entries, module, x, output)
            return _in_slices(lambda xs, outs: combine(entries, module, xs, outs), x, output)
        return hook

    handles = [modules[name].register_forward_hook(make_hook(entries), with_kwargs=True)
               for name, entries in terms.items()]
    originals: List[Tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for name, change in modulation.items():
            table = getattr(modules[name], MODULATION)
            originals.append((table, table.data.clone()))
            table.data.copy_((table.data.float() + change.to(table.device)).to(table.dtype))

    def remove() -> None:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for table, original in originals:
                table.data.copy_(original)
        originals.clear()
    return remove
