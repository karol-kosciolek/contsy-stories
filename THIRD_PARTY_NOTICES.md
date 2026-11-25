# Third-party notices

Contsy's own code is licensed under the [GNU GPL v3.0 only](LICENSE) (SPDX: `GPL-3.0-only`). Parts that come from other projects keep their own licenses, listed below. This repository includes no third-party model weights, engine binaries or Python packages. `install.bat` and the in-app model manager download them from their upstream sources at pinned revisions. [docs/LICENSING.md](docs/LICENSING.md) explains how the parts fit together.

If you build a bundle that redistributes any downloaded component, you must meet that component's license terms yourself. The PyAV wheel includes GPL-licensed FFmpeg libraries.

The installer leaves out funasr, which VoxCPM declares but never imports, and with it kaldiio, whose license allows evaluation only. Earlier installations lose both on their next `install.bat` run.

## Included in this repository

| Path | Origin | License | Notes |
| --- | --- | --- | --- |
| `engine/formats/ggml.py`, `engine/models/llm/fitting.py` | The GGUF file layout and the block layouts of Q8_0, Q4_K, Q5_K and Q6_K (`ggml-common.h`, `dequantize_row_*` in `ggml-quants.c`) of [ggml](https://github.com/ggml-org/ggml) and [llama.cpp](https://github.com/ggml-org/llama.cpp), Copyright (c) 2023-2024 The ggml authors | [MIT](LICENSES/MIT.txt) for the format description; Contsy's code is GPL-3.0-only | The dequantizers are written for PyTorch from the block layouts and are checked bit for bit against `gguf.quants` of gguf-py (MIT, the same authors). `fitting.py` follows llama.cpp's rules for offloading layers and its 1024 MiB default margin. Part of Contsy's own engine. |
| `engine/models/rife/__init__.py` | The RIFE 4.7 network architecture, which the RIFE 4.9 weights (`flownet_v4.9.pkl`) use: `IFNet_HDv3` of the 4.7 release and `warplayer.py` from [Practical-RIFE](https://github.com/hzwer/Practical-RIFE), Copyright (c) 2021 hzwer | MIT ([text](LICENSES/MIT.txt)) for the adapted parts; Contsy's changes are GPL-3.0-only | Changes: only the 4.7 generation, inference only, no ensemble pass, a timestamp per picture for exact output frame times. Part of Contsy's own engine. |
| `engine/models/musetalk/`, `engine/pipelines/musetalk.py` | [TMElyralab MuseTalk](https://github.com/TMElyralab/MuseTalk), © 2024 Tencent Music Entertainment Group | [MIT](LICENSES/MIT.txt) for the adapted parts; Contsy's changes are GPL-3.0-only | MuseTalk 1.5 lip sync: the Whisper windows of every frame, face boxes and masks from MediaPipe and YuNet, the single-step network and the blend back into the frame. |
| `engine/models/whisper.py`, `engine/models/sd_unet.py` | [OpenAI Whisper](https://github.com/openai/whisper), © 2022 OpenAI; the Stable Diffusion 1.x UNet of [diffusers](https://github.com/huggingface/diffusers); both first written after [ByteDance LatentSync](https://github.com/bytedance/LatentSync), © 2024 ByteDance Ltd. and/or its affiliates | MIT (Whisper) and [Apache-2.0](LICENSES/Apache-2.0.txt) (diffusers, LatentSync) for the adapted parts; Contsy's changes are GPL-3.0-only | The Whisper tiny audio encoder and the UNet that MuseTalk 1.5 runs. |
| `engine/models/ezaudio/`, `engine/pipelines/ezaudio.py`, `engine/sampling/ddim.py` | [EzAudio](https://github.com/haidog-yaqub/EzAudio), © 2025 Jiarui Hai; its transformer block from [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha), Copyright 2024 Junsong Chen, Jincheng Yu, Enze Xie; its long skip connections from [U-ViT](https://github.com/baofff/U-ViT), Copyright (c) 2022 Fan Bao; its waveform decoder from [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools), © 2023 Stability AI, with the SnakeBeta activation of [BigVGAN](https://github.com/NVIDIA/BigVGAN), © 2022 NVIDIA Corporation; DDIM as diffusers' `DDIMScheduler` computes it | [MIT](LICENSES/MIT.txt) (EzAudio, U-ViT, stable-audio-tools, BigVGAN) and [Apache-2.0](LICENSES/Apache-2.0.txt) (PixArt-alpha, diffusers) for the adapted parts; Contsy's changes are GPL-3.0-only | EzAudio-XL sound: FLAN-T5-XL prompt encoding, the masked diffusion transformer, the 24 kHz decoder and resampling to 48 kHz. The skip merge is written from U-ViT's design; the LayerNorm in front of its projection is there because EzAudio's checkpoint holds one. |
| `engine/kernels/sage/attn_qk_int8_per_block.py`, `engine/kernels/sage/quant_per_block.py` | SageAttention's Triton kernels (INT8 query-key products per block, FP16 probability-value products), `sageattention/triton/` of [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) at commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, Copyright 2024 Jintao Zhang, Haofeng Huang (SageAttention team) | [Apache-2.0](LICENSES/Apache-2.0.txt), kept: the files are unchanged apart from an added SPDX and provenance header | Faster attention in the Wan 2.2 experts (`engine/core/attention.py` calls them; `engine.attention` in the settings switches them off). Triton compiles them on the first video and keeps the result in `models/.cache/triton`. |
| `engine/models/configs/*.json` | The `config.json` files of the transformer, VAE and text encoder from [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers), [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers), [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) and [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo); each file names its repository, path and revision in `_source` | [Apache-2.0](LICENSES/Apache-2.0.txt), like the models | Network shapes the engine builds its models from (data, see `REUSE.toml`). |
| Navigation icons in `app/ui_theme.py` | Geometry of [Lucide](https://lucide.dev) icons, Copyright (c) 2026 Lucide Icons and Contributors | ISC ([notice](LICENSES/ISC-Lucide.txt)) | Inline SVG paths, adjusted. |
| `assets/voices/*.wav` | Generated during development with VoxCPM2 (Apache-2.0) from a written description of each voice (voice design): no reference recording of any person | GPL-3.0-only, like the rest of the repository | Twenty synthetic voices, not recordings of real people. The best of several candidates per description by pitch range and a quality score (UTMOS); see `tools/make_voices.py` and `assets/voices/voices.json`. |

## Downloaded by the installer

| Component | Source (pinned) | License | How Contsy uses it |
| --- | --- | --- | --- |
| RIFE 4.9 weights (`flownet_v4.9.pkl`) | `train_log/flownet.pkl` of the author's archive `RIFE_trained_model_v4.9.2.zip`, linked from the README of [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (Google Drive), taken out of the archive unchanged; archive and file SHA-256 pinned in the model catalog | MIT ([text](LICENSES/MIT.txt)), Copyright (c) 2021 hzwer: the Practical-RIFE README puts its model links under the project's MIT license | Frame interpolation (16 or 25 FPS to 30 FPS). An installation that already has `rife49.pth` keeps it: it holds the same 124 tensors. |
| llama-cpp-python 0.3.36 (Windows Vulkan) | [Official Vulkan release](https://github.com/abetlen/llama-cpp-python/releases/expanded_assets/v0.3.36-vulkan), SHA-256 pinned | MIT | Python binding for the assistant and its picture reviews in an isolated Python worker. |
| PyTorch 2.14.1, torchvision, torchaudio (CUDA 13.0) | download.pytorch.org | BSD-style. The bundled NVIDIA CUDA libraries are under NVIDIA's license terms. | Imported. |
| triton-windows 3.8.0.post29 (Windows build of Triton) | PyPI, pinned to the PyTorch 2.14.1 it is built for | MIT. The wheel also bundles the TinyCC compiler (`tcc.exe`, `libtcc.dll`, LGPL-2.1), which builds Triton's small launcher modules, and NVIDIA's `ptxas.exe` and CUDA headers under the NVIDIA CUDA Toolkit license terms. | Compiles the engine's GPU kernels (SageAttention, the FP8 helpers in `engine/kernels/fp8.py`, Wan's rotary embedding in `engine/kernels/rope.py`). Without it the engine uses PyTorch's own attention and slower PyTorch versions of the helpers. |
| Gradio 6.30 (including its web frontend) | PyPI | Apache-2.0 | Imported; serves the local web UI. |
| anthropic 2.10.0 (Claude API SDK), with jiter 0.19.0 and sniffio 1.3.1; it also uses httpx2, pydantic and opentelemetry-api 1.45.1, which are installed already | PyPI | Apache-2.0 (anthropic, opentelemetry-api), MIT (jiter), MIT OR Apache-2.0 (sniffio) | Imported only by the optional Agent page, and only to call the Claude API with the owner's own key; nothing is sent until a key is saved and a run is started. |
| PyAV 19.0.2 (`av`) | PyPI | BSD-3-Clause for PyAV. The wheel bundles FFmpeg 9.0.2 libraries built with libx264 and libx265, so they are under GPL-2.0-or-later terms, compatible with Contsy's GPL-3.0-only. | Imported by the own engine for video and audio (`engine/media`): decoding, H.264/AAC encoding, filters (loudnorm, sidechain ducking, cross-fades). |
| Diffusers 0.42 and Transformers 5.20 | PyPI | Apache-2.0 | Network classes and text encoders. Contsy supplies its own pipelines. |
| accelerate 1.15.0 | PyPI | Apache-2.0 | Used by Transformers and Diffusers while they load networks. |
| gguf 0.19 | PyPI | MIT | Reads GGUF headers and tensor layouts. Contsy dequantizes blocks in PyTorch. |
| sentencepiece 0.2.2 | PyPI | Apache-2.0 | The UMT5 and FLAN-T5 tokenizers. |
| MediaPipe 1.1.1, with protobuf 7.36.2 | PyPI | Apache-2.0 (MediaPipe), BSD-3-Clause (protobuf) | Face landmarks for face tracking and lip sync. |
| OpenCV 5.0.0.93 (`opencv-python`) | PyPI | Apache-2.0. The wheel bundles third-party libraries listed in its `LICENSE-3RD-PARTY.txt`, among them an FFmpeg build (`opencv_videoio_ffmpeg`) under the LGPL-2.1. | YuNet face detection and SFace identity comparison. |
| soundfile 0.14.0 | PyPI | BSD-3-Clause for soundfile. The Windows wheel bundles libsndfile (`libsndfile_x64.dll`) under the LGPL-2.1. | Reads and writes WAV files. |
| librosa 1.0.0 | PyPI | ISC | Pitch of imported voice samples; VoxCPM uses it too. |
| VoxCPM 2.0.3 | PyPI | Apache-2.0 | Local speech synthesis. |
| Other pinned libraries in `requirements.txt` | PyPI | Apache-2.0: huggingface_hub 2.3.0, requests 2.34.2, safetensors 0.8.0, tokenizers 0.23.3, datasets 3.6.0. BSD-3-Clause: NumPy 2.5.2 (its wheel also contains 0BSD, MIT, Zlib and CC0-1.0 parts), SciPy 1.18.1, psutil 7.2.2, nvidia-ml-py 13.615.71, fsspec 2025.3.0. MIT-CMU: Pillow 12.3.0. MIT: piexif 1.1.3, truststore 0.10.5, einops 0.8.2, filelock 4.2.6. | Downloads, pictures, GPU and memory readings, numerics. datasets, fsspec and einops are needed only by VoxCPM and the Hugging Face libraries. |
| VoxCPM's own dependency list (versions as pip resolves them; tested with these) | PyPI | BSD-3-Clause: torchcodec 0.17.0. Apache-2.0: modelscope 1.40.2, wetext 0.1.8, spaces 0.51.3, sortedcontainers 2.4.0. MIT: inflect 7.6.0, addict 2.4.0, pydantic 2.14.1, argbind 0.3.9. MIT or AFL-2.1: simplejson 4.2.0. MPL-2.0 and MIT: tqdm 4.70.1. Matplotlib license (PSF-based): matplotlib 3.11.2. | Installed because VoxCPM declares them. |
| Python | Windows Package Manager | PSF-2.0 | Interpreter, installed when a supported version is absent. |

pip also installs the dependencies of the packages above. `venv\Scripts\python.exe -m pip list` shows every installed package, and each package's metadata names its license.

## Models

Models download the first time a feature needs them, or all at once through `install.bat`. Contsy does not redistribute any weights. The table summarizes their terms; [MODEL_LICENSES.md](MODEL_LICENSES.md) has the pinned sources.

| Model | License | Commercial use and hosting |
| --- | --- | --- |
| Z-Image Turbo (incl. Unsloth GGUF) | Apache-2.0 | Allowed |
| Qwen3-4B (Z-Image's text encoder, incl. Qwen's GGUF) | Apache-2.0 | Allowed |
| FLUX.1 autoencoder (Black Forest Labs), which Z-Image uses as its VAE (the bf16 copy in Tongyi-MAI's Z-Image-Turbo repository) | Apache-2.0 | Allowed |
| Wan 2.2 A14B, QuantStack GGUF conversions, LightX2V LoRAs, Wan 2.1 VAE and UMT5 encoder (from Wan-AI's Diffusers repository; Lite: city96 GGUF) | Apache-2.0 | Allowed |
| RIFE 4.9 weights (Practical-RIFE, hzwer) | MIT | Allowed |
| Qwen Image Edit 2511 (incl. Unsloth GGUF), Lightning LoRA, Qwen2.5-VL 7B encoder (official bf16 weights, converted to fp8 at install) and its tokenizer, Qwen Image VAE | Apache-2.0 | Allowed |
| Qwen3.5 9B (Unsloth GGUF) | Apache-2.0 | Allowed |
| VoxCPM2 | Apache-2.0 | Allowed |
| MuseTalk 1.5 | MIT | Allowed |
| Whisper tiny, sd-vae-ft-mse (used by MuseTalk) | MIT | Allowed |
| EzAudio-XL (diffusion transformer and waveform decoder) | MIT | Allowed |
| FLAN-T5-XL (EzAudio's text encoder) | Apache-2.0 | Allowed |
| OpenCV YuNet / SFace (face check) | MIT (Copyright (c) 2020 Shiqi Yu) / Apache-2.0 | Allowed |
| MediaPipe Face Landmarker (face detector, face mesh and blendshape models) | Apache-2.0 (each of the three model cards links the Apache License 2.0) | Allowed |
| InfiniteTalk single-speaker audio adapter (MeiGen-AI) | Apache-2.0 | Allowed |
| Wan 2.1 image to video 14B 480p (LightX2V step- and guidance-distilled FP8 release), InfiniteTalk's base | Apache-2.0 | Allowed |
| CLIP ViT-H/14 image encoder of Wan 2.1 | Apache-2.0 | Allowed |
| chinese-wav2vec2-base (InfiniteTalk's speech encoder) | MIT | Allowed |
| Index-AniSora V3.2 (image-to-video experts of Anime and 2D cartoon films) | Apache-2.0 | Allowed |
| Style LoRAs of the animated films (Anime and Pixar-inspired 3D for Qwen Image Edit 2511; anime and 3D render for Z-Image Turbo) | Apache-2.0 (their authors do not say what they were trained on) | Allowed |

Four model files are made on the user's PC from the official files at install time: Z-Image's transformer and text encoder (Tongyi-MAI's shards, joined into single bf16 files without changing any value), the Qwen2.5-VL encoder of Qwen Image Edit 2511 (Qwen's bf16 shards, converted to fp8 with a per-tensor scale) and the RIFE 4.9 weights (taken unchanged out of the author's archive). Each result is checked against the SHA-256 in the model catalog and keeps the license of the files it is made from. The code that does it is Contsy's own (`engine/models/zimage/checkpoint.py`, `engine/models/qwen_edit/convert.py`, `app/downloader.py`), written from the published [safetensors](https://github.com/huggingface/safetensors) and float8 e4m3fn formats; no code is copied.
