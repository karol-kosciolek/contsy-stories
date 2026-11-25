# Licensing

This page explains which license covers which part of Contsy, how the native engine uses libraries and AI models, and what the licenses allow and require. It is the project's own analysis, written to be checked by a lawyer. It is not legal advice.

## Summary

- **Contsy's own code** is licensed under the **GNU General Public License v3.0 only** (SPDX: `GPL-3.0-only`), © 2026 Karol Kosciolek. The full text is in [LICENSE](../LICENSE).
- **Third-party parts keep their own licenses.** The repository contains a few adapted files (listed below) and no bundled Python packages or model weights. `install.bat` and the in-app model manager download those from their upstream sources onto the user's PC.
- **You may** use Contsy, including commercially, and study, change, share and host it.
- **You must** keep the license and copyright notices. If you share Contsy, changed or not, you must provide its source code under the GPL. Hosting it for others over a network is not sharing a copy, so it carries no source-code duty.
- **The AI models have their own licenses.** All of them are under Apache-2.0 or MIT. See [Models](#models-downloaded-automatically).

## What the repository contains

| Part | License | Copyright |
| --- | --- | --- |
| Contsy's own code and files: `app/`, `engine/`, `tests/`, `tools/`, `install.bat`, `start.bat`, `config/`, documentation | GPL-3.0-only | 2026 Karol Kosciolek |
| Engine ports for MuseTalk, EzAudio, RIFE and weight readers | Upstream Apache-2.0 or MIT notices and GPL-3.0-only for Contsy's changes | Upstream authors and 2026 Karol Kosciolek. See [Third-party notices](../THIRD_PARTY_NOTICES.md). |
| SageAttention's Triton kernels in `engine/kernels/sage/`, unchanged from the authors' repository | Apache-2.0 (each file keeps its notice and SPDX header) | 2024 Jintao Zhang, Haofeng Huang (SageAttention team). See [Third-party notices](../THIRD_PARTY_NOTICES.md). |
| Navigation icons in `app/ui_theme.py`, following Lucide geometry | ISC ([notice](../LICENSES/ISC-Lucide.txt)) | 2026 Lucide Icons and Contributors |
| `assets/voices/*.wav`, twenty synthetic voices generated with VoxCPM2 from written descriptions (see [Resolved](#resolved)) | GPL-3.0-only | 2026 Karol Kosciolek (AI-generated) |

Each source file names its license in an SPDX header. [REUSE.toml](../REUSE.toml) covers files that cannot carry a header. License texts are in [LICENSES/](../LICENSES). Apache-2.0, MIT and ISC code may be combined with GPLv3 code. The adapted files keep their original notices, as those licenses require.

## Libraries used by the engine

The installer adds pinned Python libraries. PyTorch runs the networks, Diffusers and Transformers provide network classes, triton-windows (MIT) compiles the engine's GPU kernels, and llama-cpp-python runs the assistant in a Python worker. PyAV handles media through the FFmpeg libraries in its wheel. The engine includes the adapted code and upstream notices listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

A ready-to-run bundle distributes every library and model file it includes. The PyAV wheel includes an FFmpeg build with libx264 and libx265. A distributor of that wheel needs the applicable GPL notices and corresponding source for the exact bundled libraries. Model licenses apply separately. The repository ships the installer and source, and the installer downloads the libraries to the user's PC.

## Models downloaded automatically

Contsy downloads models from their providers' repositories at pinned revisions. It does not redistribute any weights. Each model's license applies to whoever downloads and uses it, independently of Contsy's GPL. [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) lists every model with its license, and [MODEL_LICENSES.md](../MODEL_LICENSES.md) lists the sources.

- **Apache-2.0 or MIT, so commercial use and hosting are allowed:** Z-Image Turbo with its Qwen3-4B text encoder, the FLUX.1 autoencoder it uses as its VAE; Wan 2.2 with its conversions, VAE, UMT5 encoder, LightX2V LoRAs and the FP8 image-to-video experts (LightX2V's low-noise file and the high-noise one made from the GGUF expert and its LightX2V LoRA); Qwen Image Edit 2511 with its Lightning LoRA, Qwen2.5-VL encoder and VAE; Qwen3.5 9B, VoxCPM2, MuseTalk 1.5, Whisper tiny, sd-vae-ft-mse, EzAudio-XL, FLAN-T5-XL, OpenCV YuNet and SFace, the MediaPipe Face Landmarker models, the RIFE 4.9 weights (MIT); InfiniteTalk with its Wan 2.1 image-to-video 480p base, the Wan 2.1 CLIP ViT-H/14 image encoder and chinese-wav2vec2-base (MIT); Index-AniSora V3.2; and the style LoRAs of the animated films.
- **Every file comes from its authors or a named conversion.** Where the official file has another format than the engine reads (Z-Image's sharded transformer and text encoder, Qwen Image Edit's bf16 text encoder, the RIFE archive), and for the FP8 high-noise video expert (made from the GGUF expert with its LightX2V LoRA merged in), the installer converts it on the user's PC with Contsy's own code and checks the result against the SHA-256 in the catalog. The converted files keep the license of the official ones.
- **No model license in use lists forbidden uses.** MuseTalk 1.5 (MIT) replaced a lip-sync model whose license did.

**What you create.** The GPL covers Contsy's code, not the photos, videos and sound you create with it. Whether you may use an output commercially depends on the models that produced it.

## Commercial use

- **Contsy's own code:** allowed without a fee, including for client work, agencies and businesses.
- **Models:** their licenses apply separately.
- **Outputs:** not restricted by Contsy's license.

## Your obligations under the GPL

| What you do | What the GPL requires |
| --- | --- |
| Use Contsy unmodified, privately or commercially, on your own PC | Nothing beyond the third-party licenses |
| Change Contsy for your own use only | Nothing |
| Let people use Contsy, changed or not, over a network without giving them a copy | No duty to publish the source code. GPLv3 section 0 says that interacting with users over a network without transferring a copy is not conveying. The third-party licenses still apply. |
| Share copies of Contsy, changed or not | Pass on the GPL-3.0-only and keep the copyright, SPDX and license notices, NOTICE and the license texts. Provide the complete source code of what you share, as section 6 describes. Do not add further restrictions. |
| Share a changed version | All of the above. Also mark your changes with a date (section 5a), and keep the legal notices visible in the UI: **Settings → About** (section 5d). Setting `app.source_url` makes that card link to your version's source code. |

## Hosting Contsy as a service

- **The GPL allows hosting**, including commercially and for a fee, with or without changes.
- **Operators do not have to publish their changes.** The GPL's source-code duty applies only when copies are shared. Anyone may therefore run a changed Contsy as an online service and keep the changes private.
- **Practical limits are not license limits.** Contsy is a single-user local app. It listens only on `127.0.0.1`, rejects foreign hosts and has no accounts or access control, and its optional Agent page keeps one secret, a Claude API key encrypted for the Windows account. A hosted version needs its own authentication, isolation, key protection and abuse protection (see [SECURITY.md](../SECURITY.md)). It must also respect the models' terms for hosted use.

## How to get the source code

- **Repository:** Contsy's complete source code is the repository this file belongs to.
- **In the app:** **Settings → About** shows the license, the notices and where the source code is: the `app.source_url` setting if it is set, otherwise the installation folder. Contsy runs directly from its Python source files, so every installation contains its complete source code.

## Contributions

Contributors keep the copyright in their code. [CONTRIBUTING.md](../CONTRIBUTING.md) asks each contributor to license a contribution to everyone under the GPL-3.0-only, like the rest of Contsy's own code, and also to grant the copyright holder an additional license to use and relicense it under other terms. That lets the copyright holder offer Contsy under other licenses as well. It takes nothing away from the GPL: every version already published under the GPL stays available under the GPL, and that cannot be withdrawn.

## Resolved

- **Voice samples.** The twenty files in `assets/voices` were generated during development with VoxCPM2 (Apache-2.0) from text alone: each voice comes from a short written description of its age and character (VoxCPM2 voice design), with no reference recording of anyone. For every description several candidates were generated, and the one with the best speech-quality score (UTMOS) within the intended pitch range was kept. They are synthetic, not recordings of real people. Evidence:
  - the generation tool is in the repository (`tools/make_voices.py`), and `assets/voices/voices.json` records each voice's description, seed, pitch and UTMOS score;
  - each file has VoxCPM2's output format (48 kHz mono, 16-bit).
- **kaldiio.** Its license is NTT's "Software License Agreement for Evaluation" (internal evaluation only), not an open-source license. pip used to install it because VoxCPM 2.0.3 declares funasr, which requires kaldiio. Neither Contsy nor VoxCPM ever imports either package.
  - The installer now installs VoxCPM without funasr, and older installations lose funasr and kaldiio on their next `install.bat` run. That removes 18 packages that only funasr needed.
  - The dependency check accepts only the missing funasr requirement.
  - Tested on 1 November 2026: installation, the dependency check, all unit tests and speech generation with the four voices bundled at that time (since replaced by the twenty voices described above), in an environment without any of the 18 packages.
- **RIFE 4.9 weights** (15 November 2026). The weights equal `train_log/flownet.pkl` of the author's Practical-RIFE v4.9, v4.9.1 and v4.9.2 archives tensor for tensor (124 float32 tensors, once the `module.` prefix of the author's names is dropped), so they are MIT: the Practical-RIFE README puts its model links under the project's MIT license, and has done so since v4.9 was first listed. Contsy now downloads the author's v4.9.2 archive from the README's Google Drive link and takes `flownet.pkl` out of it unchanged; both SHA-256 values are pinned. An installation that already has `rife49.pth` keeps it, since it holds the same tensors.
- **CLIP ViT-B/32** (15 November 2026). Its Hugging Face repository named no license and its model card called any deployed use out of scope, so Contsy no longer uses it. Lip sync used it to guess the gender of each face; now a face's gender is the one stored in the profile of the character face recognition finds in it (YuNet and SFace, already in use), and faces nobody is recognised in go to unnamed speakers by size. The installer no longer downloads it; an earlier installation may delete `models\clip-vit-base-patch32` (0.6 GB).
- **Re-hosted downloads** (15 November 2026). Every model file now comes from its authors' repository (Tongyi-MAI, Wan-AI, Qwen, the RIFE author) or from a quantized conversion named in [MODEL_LICENSES.md](../MODEL_LICENSES.md). Where the official file has another format, the installer converts it on the user's PC. The new files were checked against the ones used before: the Z-Image transformer and text encoder, the Qwen Image Edit encoder and VAE and the RIFE weights hold identical tensors; the Z-Image and Wan VAEs are the same weights rounded to bf16, so pictures, which decode in bf16, are identical (the character training, removed later that day, ran the Z-Image VAE in float32, where its latents differed by 2.5% and a decoded picture by at most 2 levels of 255); Wan's Full text encoder is now the exact official bf16 one instead of an fp8 copy that differed from it by about 9%.
