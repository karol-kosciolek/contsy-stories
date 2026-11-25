# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""MuseTalk 1.5 lip sync: one UNet step from a masked face, a reference face and a Whisper window.

`unet` builds and loads the network (our SD-1.x UNet class with MuseTalk's configuration), `audio` turns
16 kHz speech into the per-frame Whisper windows the model was trained on, `faces` finds the face box,
builds the blend mask and pastes the generated face back. The pipeline is `engine.pipelines.musetalk`.
"""
