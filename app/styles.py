# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The visual style of a film: Realistic (the default), 3D animation, 2D cartoon, Anime, or a style of the owner's own.

A style is mostly words: the same models (Z-Image, Qwen-Image-Edit, Wan 2.2, MuseTalk) draw every style, and the
style decides what the prompts say about the picture. Each style holds every text that names the look of a
picture, so no "photorealistic" word reaches a picture of another style:

    video       the suffix of every Wan prompt (and of a start frame's scene text)
    image       the suffix of a Z-Image prompt (prefs.styled, the character creator's first photo)
    keep        the opening of a Qwen-Image-Edit composition (app/refs.py)
    scene       how a composition of several characters names its picture (refs.cast_lead)
    render      the lead of a character's scene (characters.render)
    anchor      the character creator's first photo (app/char_photos.py)
    view        the end of every further view in the creator
    writer_*    how the prompt assistant names the model it writes for (app/llm.py)
    subject     what the photo gate looks for (llm.check_photo)
    creature_*  the anchor and the end of every view of an animal or creature character in the creator; "" in
                Realistic, which draws no creatures (allows_creatures)

Realistic keeps the exact texts the studio used before styles existed, so a realistic film, its clip keys and its
pictures stay as they were. A Timeline project stores its style id (app/timeline.py, "realistic" when absent), a
character the style its photos were drawn in (app/characters.py). The face check thresholds of each style family are
in app/qc.py (FACE_RULES).

The built-in styles (presets) are Realistic, 3D animation, 2D cartoon (classic hand-drawn cartoon) and Anime
(Japanese / Asian anime). Anime keeps the id "2d" of the one 2D style before 2026-11-16 ("2D cartoon / anime"), with
every text and model byte for byte, so stored films and characters keep their look, keys and Ready clips.

The animated styles also bring style LoRAs (the catalog entry STYLE_MODEL_ID, owner decision 2026-11-15), each
named by a part of its file name in that entry: `edit_lora` for every Qwen-Image-Edit composition of the film (its
`edit_trigger`, the sentence the LoRA was trained with, opens the edit prompt), `image_lora` for the Z-Image first
photo of a character, and `video_lora` for Wan 2.2's high-noise expert (no style has one). The drawn styles run the
Index-AniSora experts (`video_experts`, the Full set) and draw a start frame for a shot with nothing to start from.

A custom style is the owner's own words for the look on top of a starting point (a preset, or "Something else" with
no style add-on), and its id holds its whole definition: "custom:1:<base id>:<look words>". It takes the base's
models, lip-sync behaviour, face rules and photo-gate texts, and puts its words into every picture text (template
version 1, frozen: a stored id must always mean the same pictures). The owner's names for such styles live in the
style library (app/style_library.py); a film or character needs only its id, so editing or deleting a library entry
never changes or locks a film.
"""
from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

REALISTIC, ANIMATION_3D, CARTOON, ANIME = "realistic", "3d", "cartoon", "2d"
# The stored id of the anime style: it was "2D cartoon / anime" until 2026-11-16, and the id is stored in projects
# and characters and hashed into their keys. Never rename it.
CARTOON_2D = ANIME
OTHER = "other"  # a starting point of custom styles only, never a film's own style
STYLE_MODEL_ID = "animation-styles"  # the catalog entry holding the style LoRAs
ANISORA_ID = "anisora-2d"  # the catalog entry of the drawn styles' image-to-video experts
CUSTOM_PREFIX = "custom:1:"  # template version 1 of custom styles (frozen)
MIN_LOOK, MAX_LOOK = 3, 200
# What a custom style's look words may contain: letters (Polish ones too), digits, spaces and a little punctuation.
# FROZEN: an id is valid only while its look passes clean_look unchanged, so this may never refuse what it accepts.
_LOOK = re.compile(r"[\w ,.;'()/&+!?-]+")


@dataclass(frozen=True)
class Style:
    id: str
    label: str
    video: str
    image: str
    keep: str
    scene: str
    render: str
    anchor: str
    view: str
    writer_image: str
    writer_video: str
    subject: str
    edit_lora: Optional[Tuple[str, float]] = None   # (part of the file name, strength) in STYLE_MODEL_ID
    edit_trigger: str = ""
    image_lora: Optional[Tuple[str, float]] = None
    video_lora: Optional[Tuple[str, float]] = None  # Wan 2.2's high-noise expert
    # MuseTalk paints a real mouth: on a 2D cartoon face it found the face but left a realistic, blurred mouth patch
    # (2026-11-15, animation_samples/debug/2d_musetalk_strip.png); a 3D face it moves plausibly
    musetalk: bool = True
    # The catalog entry of an image-to-video expert pair that replaces Wan 2.2's own for this style (the drawn styles:
    # the official Index-AniSora V3.2 experts, fine-tuned from Wan 2.2 for anime, with LightX2V's 4-step distillation
    # merged in). Image to video only, so a shot with nothing to start from gets a drawn start frame (draws_start).
    video_experts: Optional[str] = None
    # A shot with no picture, character or reference to start from: Z-Image draws its start frame in the style (its
    # image text and image LoRA), and image to video makes the clip, instead of text to video.
    draws_start: bool = False
    base: str = ""     # a custom style: the id of the preset (or OTHER) it starts from
    medium: str = ""   # a starting point: how a custom style's texts name the medium (frozen)
    about: str = ""    # a preset: one line on what it looks like; a custom style: its look words
    # An animal or creature character (app/characters.py, kind "creature") in the creator: the first photo ("{look}"
    # only: no gender and no default outfit, the look or the photo says what it wears) and the end of every further
    # view. "" = the style draws no creatures (Realistic and every custom style based on it).
    creature_anchor: str = ""
    creature_view: str = ""

    @property
    def has_loras(self) -> bool:
        return any((self.edit_lora, self.image_lora, self.video_lora))

    @property
    def allows_creatures(self) -> bool:
        """Whether an animal or creature character can be drawn in this style (every style family but Realistic)."""
        return bool(self.creature_anchor)

    @property
    def realistic(self) -> bool:
        return self.id == REALISTIC

    @property
    def family(self) -> str:
        """The preset (or OTHER) whose models and rules the style follows: its own id, or a custom style's base."""
        return self.base or self.id

    @property
    def custom(self) -> bool:
        return bool(self.base)


_REALISTIC = Style(
    id=REALISTIC, label="Realistic",
    # the studio's one style before 2026-11-15 (prefs.STYLE keeps these two strings)
    video="ultra realistic, photorealistic real life footage, natural lighting, cinematic, highly detailed, "
          "sharp focus",
    image="ultra realistic photo, photorealistic, natural light, highly detailed, sharp focus",
    keep="Photorealistic photo, one coherent real scene.",
    scene="one coherent real scene",
    render="Make a single live-action scene. ",
    anchor="Photo of an adult {gender}. {look}. Portrait photo, head and shoulders, wearing a plain crew-neck top, "
           "looking straight at the camera, neutral relaxed expression, plain light gray studio background, soft "
           "even daylight",
    view=". Photorealistic photo, natural skin texture.",
    writer_image="a photorealistic image generator",
    writer_video="a realistic text-to-video model",
    subject="one real-looking human",
    medium="photorealistic live-action film",
    about="Looks like a filmed movie: real people and places, natural light.",
)

_ANIMATION_3D = Style(
    id=ANIMATION_3D, label="3D animation",
    video="3D animated film, stylized 3D CGI animation, expressive animated characters, smooth stylized shading, "
          "soft global illumination, vibrant colors, cinematic, highly detailed",
    image="3D animated film still, stylized 3D CGI render, expressive animated character, smooth stylized "
          "shading, soft global illumination, vibrant colors, highly detailed",
    keep="A still from a 3D animated film, one coherent scene. Draw everything, every person too, in the same "
         "stylized 3D CGI animation look.",
    scene="one coherent scene of a 3D animated film",
    render="Make a single scene of a 3D animated film: draw the person and everything around them in the same "
           "stylized 3D CGI animation look. ",
    anchor="3D animated film character, a stylized 3D CGI render of an adult {gender} with adult proportions. "
           "{look}. Head and shoulders portrait, wearing a plain dark gray crew-neck sweater, looking straight at the camera, "
           "neutral relaxed expression, plain light gray studio background, soft even light",
    view=". The same stylized 3D animated character, rendered in exactly the art style of image 1: a grown adult with the "
         "same mature adult face and adult proportions.",
    creature_anchor="3D animated film character, a stylized 3D CGI render of one fully grown non-human creature "
                    "character, not a human: {look}. Alone, no people and no other animals. Head and shoulders "
                    "portrait, looking straight at the camera, neutral relaxed expression, plain light gray studio "
                    "background, soft even light",
    creature_view=". The same stylized 3D animated character, rendered in exactly the art style of image 1: the same "
                  "fully grown non-human creature with the same face, body shape and proportions, not a human.",
    writer_image="an image generator that draws stills of 3D animated films (stylized 3D CGI)",
    writer_video="a text-to-video model for 3D animated films (stylized 3D CGI)",
    subject="one human character (a photo, a 3D animated render or a drawing of a person)",
    edit_lora=("pi3_20", 1.0), edit_trigger="Transform it into Pixar-inspired 3D.",
    medium="3D animated film, stylized 3D CGI animation",
    about="Looks like a 3D animated feature film: rounded stylized characters, soft light, vibrant colors.",
    # no image LoRA: Z-Image draws a cleaner 3D animated character from the prompt alone than with the 3D render LoRA
    # (side by side, same seed, 2026-11-15: animation_samples/debug/anchor_3d_*.png)
)

_CARTOON = Style(
    id=CARTOON, label="2D cartoon",
    video="2D animated cartoon, classic hand-drawn cartoon style, bold clean black outlines, simple rounded shapes, "
          "flat bright colors, expressive exaggerated poses, squash and stretch, simple painted background, cinematic, "
          "highly detailed",
    image="2D cartoon illustration, classic hand-drawn cartoon style, bold clean black outlines, simple rounded shapes, "
          "flat bright colors, expressive face, simple painted background, highly detailed",
    keep="A still from a 2D animated cartoon, one coherent scene. Draw everything, every person too, in the same "
         "classic hand-drawn cartoon style with bold clean black outlines, simple rounded shapes and flat bright colors.",
    scene="one coherent scene of a classic hand-drawn 2D cartoon",
    render="Make a single scene of a 2D animated cartoon: draw the person and everything around them in the same "
           "classic hand-drawn cartoon style with bold clean black outlines, simple rounded shapes and flat bright "
           "colors. ",
    anchor="2D cartoon character, a classic hand-drawn cartoon drawing of an adult {gender} with adult proportions. "
           "{look}. Head and shoulders portrait, wearing a plain dark gray crew-neck sweater, looking straight at the "
           "viewer, neutral relaxed expression, plain light gray background, bold clean black outlines, flat colors",
    view=". The same 2D cartoon character, drawn in exactly the art style of image 1: a grown adult with the same "
         "mature adult face and adult proportions.",
    creature_anchor="2D cartoon character, a classic hand-drawn cartoon drawing of one fully grown non-human creature "
                    "character, not a human: {look}. Alone, no people and no other animals. Head and shoulders "
                    "portrait, looking straight at the viewer, neutral relaxed expression, plain light gray "
                    "background, bold clean black outlines, flat colors",
    creature_view=". The same 2D cartoon character, drawn in exactly the art style of image 1: the same fully grown "
                  "non-human creature with the same face, body shape and proportions, not a human.",
    writer_image="an image generator that draws classic 2D cartoon illustrations (hand-drawn, bold outlines, flat "
                 "colors)",
    writer_video="a text-to-video model for classic 2D animated cartoons (hand-drawn, bold outlines, flat colors)",
    subject="one human character (a photo, a 3D animated render or a drawing of a person)",
    # no LoRA: the only 2D LoRAs installed draw anime faces; the words carry the look
    musetalk=False,
    video_experts=ANISORA_ID,
    draws_start=True,
    medium="2D cartoon, hand-drawn animation",
    about="Looks like a classic hand-drawn TV cartoon: bold black outlines, flat bright colors, simple rounded "
          "shapes, bouncy motion.",
)

_ANIME = Style(
    id=ANIME, label="Anime",
    video="2D animated cartoon, anime style, hand-drawn cel animation, clean bold line art, flat cel shading, "
          "vibrant colors, painted background, cinematic, highly detailed",
    image="2D anime cartoon illustration, hand-drawn cel animation still, clean bold line art, flat cel shading, "
          "vibrant colors, painted background, highly detailed",
    keep="A still from a 2D animated cartoon, one coherent scene. Draw everything, every person too, in the same "
         "hand-drawn anime cartoon style with clean line art and flat cel shading.",
    scene="one coherent scene of a 2D animated cartoon",
    render="Make a single scene of a 2D animated cartoon: draw the person and everything around them in the same "
           "hand-drawn anime cartoon style with clean line art and flat cel shading. ",
    anchor="2D anime cartoon character, a hand-drawn illustration of an adult {gender} with adult proportions. "
           "{look}. Head and shoulders portrait, wearing a plain dark gray crew-neck sweater, looking straight at the viewer, "
           "neutral relaxed expression, plain light gray background, clean line art, flat cel shading",
    view=". The same 2D anime cartoon character, drawn in exactly the art style of image 1: a grown adult with the same "
         "mature adult face and adult proportions.",
    # an anime character with animal ears is a person to the creature gate: ask for an animal head and body
    creature_anchor="2D anime cartoon character, a hand-drawn illustration of one fully grown non-human creature "
                    "character with an animal head and an animal body, not a human with animal ears: {look}. Alone, "
                    "no people and no other animals. Head and shoulders portrait, looking straight at the viewer, "
                    "neutral relaxed expression, plain light gray background, clean line art, flat cel shading",
    creature_view=". The same 2D anime cartoon character, drawn in exactly the art style of image 1: the same fully "
                  "grown non-human creature with the same animal head, animal body and proportions, not a human with "
                  "animal ears.",
    writer_image="an image generator that draws 2D anime cartoon illustrations (hand-drawn cel animation)",
    writer_video="a text-to-video model for 2D animated cartoons in anime style (hand-drawn cel animation)",
    subject="one human character (a photo, a 3D animated render or a drawing of a person)",
    edit_lora=("anime-2000", 1.0), edit_trigger="Transform into anime.",
    image_lora=("sk_anime_style", 0.8),  # 0.8: closer to the look text, steadier anatomy
    musetalk=False,
    video_experts=ANISORA_ID,
    draws_start=True,
    medium="2D anime, hand-drawn cel animation",
    about="Looks like Japanese / Asian anime: clean line art, flat cel shading, expressive eyes, painted backgrounds.",
)

# A starting point for custom styles that want none of the presets' add-ons: the owner's words alone decide the look,
# Wan 2.2 makes the motion. Its own picture texts never reach a picture (a custom style replaces them).
_OTHER = Style(
    id=OTHER, label="Something else",
    video="stylized animated film in one consistent art style, cinematic, highly detailed",
    image="stylized picture in one consistent art style, highly detailed",
    keep="One coherent scene in one consistent art style. Show everything, every person too, in that same style.",
    scene="one coherent scene in one consistent art style",
    render="Make a single scene in one consistent art style: show the person and everything around them in it. ",
    anchor="Stylized character in one consistent art style, a picture of an adult {gender} with adult proportions. "
           "{look}. Head and shoulders portrait, wearing a plain dark gray crew-neck sweater, looking straight at the "
           "viewer, neutral relaxed expression, plain light gray background",
    view=". The same character, drawn in exactly the art style of image 1: a grown adult with the same mature adult "
         "face and adult proportions.",
    creature_anchor="Stylized character in one consistent art style, a picture of one fully grown non-human creature "
                    "character, not a human: {look}. Alone, no people and no other animals. Head and shoulders "
                    "portrait, looking straight at the viewer, neutral relaxed expression, plain light gray background",
    creature_view=". The same character, drawn in exactly the art style of image 1: the same fully grown non-human "
                  "creature with the same face, body shape and proportions, not a human.",
    writer_image="an image generator that draws stylized pictures in one consistent art style",
    writer_video="a text-to-video model for stylized animated films in one consistent art style",
    subject="one human character (a photo, a 3D animated render or a drawing of a person)",
    musetalk=False,
    draws_start=True,
    medium="stylized animated film",
    about="Your words alone decide the look: paint, clay, paper cut-out, comics and more.",
)

# The presets, in the order every picker shows them.
STYLES: Dict[str, Style] = {style.id: style for style in (_REALISTIC, _ANIMATION_3D, _CARTOON, _ANIME)}
# What a custom style can start from: the presets and OTHER.
BASES: Dict[str, Style] = {**STYLES, OTHER: _OTHER}
DEFAULT = REALISTIC


def clean_look(text) -> str:
    """The look words of a custom style in their one canonical form (Unicode NFC, single spaces, no leading or
    trailing space or ",.;"). ValueError with a message for the owner when they cannot be a look.
    FROZEN with template version 1: a stored custom id is valid only while its look comes back unchanged."""
    if not isinstance(text, str):
        raise ValueError("Describe the look in at least 3 characters.")
    look = " ".join(unicodedata.normalize("NFC", text).split()).strip(" ,.;")
    if len(look) < MIN_LOOK:
        raise ValueError("Describe the look in at least 3 characters.")
    if len(look) > MAX_LOOK:
        raise ValueError(f"Use at most {MAX_LOOK} characters for the look.")
    if not _LOOK.fullmatch(look):
        raise ValueError("Use only letters, numbers, spaces and , . ; ' ( ) / & + ! ? - in the look.")
    return look


def custom_id(base: str, look: str) -> str:
    """The id of the custom style with these look words on top of the starting point `base` (a preset or OTHER)."""
    if base not in BASES:
        raise ValueError("Choose what the style starts from.")
    return f"{CUSTOM_PREFIX}{base}:{clean_look(look)}"


@functools.lru_cache(maxsize=256)
def _custom(style_id: str) -> Optional[Style]:
    """The Style a custom id stands for (template version 1), or None when it is not a valid custom id."""
    if not style_id.startswith(CUSTOM_PREFIX):
        return None
    base, separator, look = style_id[len(CUSTOM_PREFIX):].partition(":")
    if not separator or base not in BASES:
        return None
    try:
        if clean_look(look) != look:
            return None
    except ValueError:
        return None
    start = BASES[base]
    short = look if len(look) <= 40 else look[:39].rstrip(" ,.;") + "…"
    return replace(
        start, id=style_id, base=base, about=look, medium="", label=f"Custom: {short}",
        video=f"{look}, {start.medium}, cinematic, highly detailed",
        image=f"{look}, {start.medium} still, highly detailed",
        keep=f"One coherent scene in this look: {look} ({start.medium}). Show everything, every person too, in this "
             "same look.",
        render=f"Make a single scene in this look: {look} ({start.medium}). Show the person and everything around them "
               "in this same look. ")


def get(style_id: Optional[str]) -> Style:
    """The style with this id (a preset or a custom id); Realistic for "" or None. ValueError for anything else,
    OTHER included (it is only a starting point)."""
    if not style_id:
        return STYLES[DEFAULT]
    if isinstance(style_id, str):
        found = STYLES.get(style_id) or _custom(style_id)
        if found is not None:
            return found
    raise ValueError("Choose a valid style.")


def valid(style_id) -> bool:
    return isinstance(style_id, str) and (style_id in STYLES or _custom(style_id) is not None)


def of(style_id) -> Style:
    """Like get, but an unknown or damaged value is Realistic (for stored records read back)."""
    try:
        return get(style_id) if style_id else STYLES[DEFAULT]
    except ValueError:
        return STYLES[DEFAULT]


def for_video_text(text: str) -> Style:
    """The style whose Wan suffix is `text` (what a Timeline clip records and video_gen.generate takes): a preset's,
    or a custom style's ("<look>, <its base's medium>, cinematic, highly detailed"); Realistic for "" or any other
    text."""
    if not text:
        return STYLES[DEFAULT]
    found = next((style for style in STYLES.values() if style.video == text), None)
    if found is not None:
        return found
    for start in BASES.values():
        tail = f", {start.medium}, cinematic, highly detailed"
        if text.endswith(tail):
            custom = _custom(f"{CUSTOM_PREFIX}{start.id}:{text[:-len(tail)]}")
            if custom is not None:
                return custom
    return STYLES[DEFAULT]


def choices() -> List[Tuple[str, str]]:
    """(label, id) of the presets for a dropdown or radio, Realistic first (app/style_library.choices adds the
    owner's own styles)."""
    return [(style.label, style.id) for style in STYLES.values()]


def base_choices() -> List[Tuple[str, str]]:
    """(label, id) of what a custom style can start from: the presets, then "Something else"."""
    return [(style.label, style.id) for style in STYLES.values()] + [
        ("Something else (paint, clay, paper, comics…)", OTHER)]


def base_label(style_id) -> str:
    """The label of the preset (or "Something else") a style follows."""
    return BASES[of(style_id).family].label


def label(style_id) -> str:
    return of(style_id).label
