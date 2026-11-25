# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The settings of an EzAudio checkpoint (`ckpts/ezaudio-xl.yml`), checked against what the engine implements.

The network's sizes are read from the shapes of its tensors (transformer.py); what the shapes cannot show (the
kind of time conditioning, the activation, the scaling of the low-rank modulation, the diffusion schedule) is read
here, and a file asking for anything else is refused instead of being run wrongly. The YAML files are flat (two
levels of `key: value`), so a small reader is enough and no YAML package is needed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# The options of the published EzAudio-XL and EzAudio-L configurations; the engine implements exactly these.
_MODEL = {
    "mae": True, "patch_size": 1, "input_type": "1d", "qkv_bias": False, "qk_scale": None, "qk_norm": "layernorm",
    "norm_layer": "layernorm", "act_layer": "geglu", "context_norm": True, "time_fusion": "ada_sola_bias",
    "cls_dim": None, "context_fusion": "cross", "context_pe_method": "none", "pe_method": "none",
    "rope_mode": "shared", "use_conv": True, "skip": True, "skip_norm": True,
}
_AUTOENCODER = {"name": "stable_vae", "q_first": True}
_DIFFUSION = {"beta_schedule": "scaled_linear", "prediction_type": "v_prediction", "rescale_betas_zero_snr": True,
              "timestep_spacing": "trailing", "clip_sample": False}
_TEXT_ENCODERS = ("google/flan-t5-xl", "google/flan-t5-large")


@dataclass(frozen=True)
class Settings:
    """What the pipeline needs from the YAML file, plus the sizes the checkpoint must have."""
    text_encoder: str       # the Hugging Face name of the FLAN-T5 encoder the network was trained with
    max_tokens: int         # prompts are padded or cut to this many tokens
    sample_rate: int        # of the waveform decoder
    latent_rate: int        # latent frames per second
    window_frames: int      # the training length (10 s)
    latent_scale: float
    latent_shift: float
    train_timesteps: int
    beta_start: float
    beta_end: float
    sizes: Dict[str, int]   # the network's sizes as transformer.Transformer names them


def _scalar(text: str) -> Any:
    """One YAML scalar or flow list of the kinds these files use."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return [_scalar(part) for part in text[1:-1].split(",") if part.strip()]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "~", ""):
        return None
    try:
        return ast.literal_eval(text)  # ints and floats (0.00085, 4.0)
    except (ValueError, SyntaxError):
        return text


def read_yaml(path: str | Path) -> Dict[str, Any]:
    """The two-level `key: value` YAML of the EzAudio checkpoints (comments and blank lines allowed)."""
    data: Dict[str, Any] = {}
    section = None
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        key, colon, value = line.strip().partition(":")
        if not colon or not key:
            raise ValueError(f"{Path(path).name}:{number}: expected 'key: value'")
        if line[0] != " ":
            section = None
            if value.strip():
                data[key] = _scalar(value)
            else:
                section = data[key] = {}
        elif section is not None:
            section[key] = _scalar(value)
        else:
            raise ValueError(f"{Path(path).name}:{number}: indented line outside a section")
    return data


def _require(found: Dict[str, Any], wanted: Dict[str, Any], what: str) -> None:
    wrong = {key: found.get(key, "<missing>") for key, value in wanted.items() if found.get(key, "<missing>") != value}
    if wrong:
        raise ValueError(f"EzAudio {what} settings not supported by the engine: {wrong}")


def load(path: str | Path) -> Settings:
    """The settings of `ezaudio-*.yml`; ValueError when the file asks for something the engine does not do."""
    data = read_yaml(path)
    try:
        model, autoencoder, text, diffusion = (data[key] for key in ("model", "autoencoder", "text_encoder", "diff"))
        _require(model, _MODEL, "model")
        _require(autoencoder, _AUTOENCODER, "autoencoder")
        _require(diffusion, _DIFFUSION, "diffusion")
        # The blocks add the low-rank modulation unscaled (alpha / rank = 1), as both published files have it.
        if model["ada_sola_alpha"] != model["ada_sola_rank"]:
            raise ValueError("EzAudio: ada_sola_alpha must equal ada_sola_rank")
        if model["in_chans"] != 2 * model["out_chans"] + 1:
            raise ValueError("EzAudio: in_chans must be the noisy latent, the known latent and the mask channel")
        if text["model"] not in _TEXT_ENCODERS:
            raise ValueError(f"EzAudio: unsupported text encoder {text['model']!r}")
        sizes = dict(dim=model["embed_dim"], latent_channels=model["out_chans"], context_dim=model["context_dim"],
                     heads=model["num_heads"], hidden=int(model["embed_dim"] * model["mlp_ratio"]),
                     rank=model["ada_sola_rank"], depth=model["depth"])
        return Settings(text_encoder=text["model"], max_tokens=int(text["max_length"]),
                        sample_rate=int(autoencoder["sr"]), latent_rate=int(autoencoder["latent_sr"]),
                        window_frames=int(model["img_size"]), latent_scale=float(autoencoder["scale"]),
                        latent_shift=float(autoencoder["shift"]),
                        train_timesteps=int(diffusion["num_train_timesteps"]),
                        beta_start=float(diffusion["beta_start"]), beta_end=float(diffusion["beta_end"]),
                        sizes={key: int(value) for key, value in sizes.items()})
    except (KeyError, TypeError) as error:
        raise ValueError(f"{Path(path).name} is not an EzAudio checkpoint configuration ({error})") from error


def check_sizes(settings: Settings, sizes: Dict[str, int]) -> None:
    """Refuse a checkpoint whose shapes do not match its configuration (another model's file, a wrong pairing)."""
    if dict(sizes) != settings.sizes:
        raise ValueError(f"EzAudio: the checkpoint has sizes {dict(sizes)}, its configuration {settings.sizes}")
