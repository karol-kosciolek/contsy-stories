# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Wan 2.2 A14B adapter: the UMT5 text encoder (text.py), the video autoencoder (vae.py), the
image-to-video conditioning (conditioning.py) and the diffusion transformers (transformer.py).

The network classes come from `transformers` and `diffusers`; this package owns the file loading (the
original Wan key layout in safetensors, and GGUF), the conventions of the model and the glue.
"""
