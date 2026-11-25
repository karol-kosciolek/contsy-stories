# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Dialogi postaci zapisane w prompcie wideo.

Kwestie piszemy w cudzyslowie wprost w opisie sceny, np.:
    A woman in a cafe smiles at the camera and says „Cześć, miło Cię widzieć!”
parse() rozdziela prompt na:
  - opis obrazu dla Wan (bez tekstu kwestii - cytat w prompcie konczy sie czesto napisem
    na ekranie; zamiast niego dopisujemy, ze postac mowi),
  - opis dzwieku tla dla EzAudio-XL (bez mowy - ta dochodzi osobno),
  - liste kwestii z plcia mowiacego (z najblizszego slowa przed cytatem: woman/man,
    kobieta/mezczyzna, she/he...), wedlug ktorej dobieramy glos.

Voices (app/voices.py): "@Hanna says „…”", or a registered voice name as the last cue before
the quote, speaks that line with that voice. Lines without a voice get the character's voice
or the default voice of their gender (voices_for). The picture and background-sound
descriptions keep the name but lose the @ sign.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

# „...”  “...”  "..."  «...»
QUOTE = re.compile(r'[„“"«]([^„“”"«»]{2,}?)[”"»]')
# The voice key of a character's own voice: the voice lives in the character's folder, not in the voice
# library (app/characters.py), and voices.get resolves "character:<slug>" to it.
CHARACTER_VOICE = "character:"
_FEMALE = re.compile(r"\b(woman|women|girl|lady|she|her|female|mother|grandmother|kobiet\w*|"
                     r"dziewczyn\w*|pani|ona|matk\w*|babci\w*)\b", re.I)
# Polish "on" / "pan" only in Polish text: a Polish verb (also the past tense "zapytał") up
# to two words later ("On też mówi", "a on z uśmiechem odpowiada", "starszy pan w kapeluszu
# odpowiada"), a name after "pan", or "On:" before a line. In English prompts "sits on a
# bench" and "holds a pan" made a woman speak with a man's voice.
_PL_VERB = (r"mówi|pyta|krzyczy|szepcze|odpowiada|dodaje|woła|się|jest|ma|stoi|siedzi|patrzy|"
            r"idzie|biegnie|trzyma|uśmiecha|śmieje|wchodzi|podaje|odwraca|spogląda")
_PL_NEXT = rf"(?:[\s,]+\w+){{0,2}}?[\s,]+(?:(?:{_PL_VERB})\b|(?-i:[a-ząćęłńóśźż]\w*ł)\b)"
_MALE = re.compile(r"\b(man|men|boy|guy|he|him|his|male|father|grandfather|m[eę][żz]czyzn\w*|"
                   r"ch[łl]opak\w*|chłopiec|ojciec|dziadek|"
                   rf"pan(?=\s+(?-i:[A-ZŁŚŻŹĆŃÓĘĄ])|{_PL_NEXT})|"
                   rf"on(?={_PL_NEXT}|\s*:))\b", re.I)
# "says:", "mówi:" itp. tuz przed cytatem - usuwamy razem z cytatem
_SAYS = re.compile(r"\b(says|said|asks|asked|shouts|whispers|replies|answers|mówi|pyta|"
                   r"krzyczy|szepcze|odpowiada)\s*:?\s*$", re.I)
# @Hanna before a line = the voice named Hanna (app/voices.py). Never a number: @2 is a
# reference image (app/refs.py).
_TAG = re.compile(r"(?<!\w)@(?!\d)(\w{2,})")
_WORD = re.compile(r"\w{2,}")

# "two women" / "dwie kobiety" w opisie - kwestie tej plci przypisujemy na zmiane dwom osobom
_PAIR = {"female": re.compile(r"\b(two|2)\s+(women|girls|ladies)\b|\bdwie\s+(kobiety|dziewczyny)", re.I),
         "male": re.compile(r"\b(two|2)\s+(men|boys|guys)\b|\bdw(ó|o)ch\s+(m[eę][żz]czyzn|ch[łl]opak)", re.I)}


# "She says" / "he answers": refers to the person of that gender named last before it
_PRONOUN = re.compile(r"(?i:she|he|her|him|his|ona|on)")
# her / him / his after someone named earlier in the same sentence (and nobody of their gender)
# are the person spoken to: in "Kuba takes her hand and says" Kuba speaks
_OBJECT = re.compile(r"(?i:her|him|his)")
# The verb that introduces a spoken line. Its speaker is named before it: in "Ola says to Kuba
# „Hi”" and "Ola mówi do Kuby „Cześć”" Kuba is the listener.
_VERB = re.compile(r"\b(says|said|asks|asked|adds|added|answers|answered|replies|replied|tells|"
                   r"told|shouts|shouted|whispers|whispered|yells|yelled|exclaims|exclaimed|"
                   r"mówi|pyta|odpowiada|dodaje|krzyczy|szepcze|woła|powiedział\w*|zapytał\w*|"
                   r"odpowiedział\w*|dodał\w*|krzykn\w*|szepn\w*|zawołał\w*)\b", re.I)
# A name right after one of these is an object ("looks at Kuba", "do Kuby"), not the speaker
_PREP = re.compile(r"\b(to|at|with|for|from|towards?|beside|behind|near|about|in|do|na|dla|"
                   r"z|ze|od|przy|obok|ku|za|przed|o)\s*$", re.I)
# "Ola hands Kuba a coffee and says": the second verb keeps the sentence's first subject
_AND = re.compile(r"\b(and|then|i|oraz|potem)\s*$", re.I)
_SENTENCE = re.compile(r"[.!?;]\s+(?=\S)")
# The image number Timeline adds after a named place or item ("the TV studio [2]", app/items.py mark): a quote is
# shown or spoken as it is in the text without it, which is the text the editor shows.
_IMAGE_NUMBER = re.compile(r"\s*\[\d{1,2}\]")
# Someone in a TV studio who says something: the studio is a place there, not a TV screen showing a text
_STUDIO_SPEAKS = re.compile(r"\b(?:tv|television)\s+studios?\b(?:\s+[\w'’-]+){0,3}?\s+(?:says?|said|asks?|"
                            r"shouts?|whispers?|replies|answers?)\b", re.I)

# Text shown in the picture, never spoken: a quote right after words that name writing or a display - "a large
# screen displays the exact text "CRYPTO NEWS"", "a sign reads "OPEN"", "a shirt with the word "BOSS"", "z napisem
# „Sklep”". On 2026-11-16 such a quote became the character's first spoken line, and the picture text got "talking"
# in its place, which the picture model painted on the screen. It stays in the picture text as written and is no
# dialogue line; read aloud ("reads it aloud") it is speech.
_DISPLAY = (r"(?:screens?|signs?|signboards?|boards?|billboards?|banners?|posters?|displays?|monitors?|"
            r"tv|television|tickers?|chyrons?|cards?|placards?|plaques?|labels?|shirts?|t-?shirts?|hoodies?|mugs?|"
            r"books?|covers?|newspapers?|magazines?|headlines?|walls?|neon|graphics?|captions?|subtitles?|titles?|"
            r"logos?|notes?|whiteboards?|blackboards?|chalkboards?|ekran\w*|tablic\w*|szyld\w*|baner\w*|"
            r"plakat\w*|transparent\w*|koszul\w*|kartk\w*|monitor\w*)")
_SHOWS = (r"(?:displays?|displaying|displayed|shows?|showing|shown|reads?|reading|says|saying|spells?|spelling|"
          r"with|bearing|wyświetla\w*|pokazuje|głosi)")
# "a sign reads", "the screen behind him shows", "a neon sign that says", "a sign:"; never "looks at the sign and says"
_ON_DISPLAY = re.compile(rf"\b{_DISPLAY}\b(?:(?:\s+(?!(?:and|then|but|while|as|i|oraz)\b)[\w'’-]+){{0,3}}?"
                         rf"\s+(?:(?:that|which|który|która|które)\s+)?{_SHOWS}\s*[:,–—-]?|\s*:)\s*$", re.I)
# "the exact text", "with the words", "titled", "written in red", "z napisem"
_WRITING = re.compile(r"\b(?:text|texts|words?|wording|lettering|letters|caption|title|titled|headline|inscription|"
                      r"slogan|logo|label(?:l?ed)?|written|printed|painted|engraved|embroidered|stitched|spelled|"
                      r"spelling|spells|types?|typing|typed|writes|writing|wrote|napis\w*|tekst\w*|has[łl]\w*|"
                      r"podpis\w*)\b(?:\s+[\w'’-]+){0,3}?\s*[:,–—-]?\s*$", re.I)
_ALOUD = re.compile(r"\b(?:aloud|out\s+loud|reads?\s+out|na\s+g[łl]os)\b", re.I)
# "says the words „…”": a speaking verb right before the writing word keeps the quote speech
_SAID_WORDS = re.compile(rf"{_VERB.pattern}\s+(?:(?:the|a|an|these|those|this|his|her|their|its|some|following|"
                         rf"same|ten|te|tę|tych)\s+)*$", re.I)

# app/cast.py people(): (name pattern, character slug, 'female'/'male', voice slug, name)
Person = Tuple["re.Pattern[str]", str, str, str, str]


@dataclass
class Line:
    speaker: str   # gender: "female" / "male"
    text: str
    who: str = ""  # person: "female#0", "female#1", "male#0"... "female#<voice slug>" or
                   # "female#char:<character slug>"
    voice: str = ""  # voice slug named in the prompt ("" = default voice for the gender)
    character: str = ""  # character speaking the line (app/cast.py), "" = none


def is_gender_word(word: str) -> bool:
    """True for words that already pick a gender in prompts (woman, he, kobieta, pan...)."""
    word = word or ""
    return bool(_FEMALE.fullmatch(word) or _MALE.fullmatch(word)
                or word.casefold() in ("on", "pan"))


def tags(prompt: str) -> List[str]:
    """Voice tags (@Name) outside the quoted lines, in order, without repeats."""
    outside = QUOTE.sub(" ", prompt or "")
    return list(dict.fromkeys(m.group(1) for m in _TAG.finditer(outside)))


def _gender(context: str) -> Optional[str]:
    """Plec z ostatniego slowa-wskazowki w kontekscie przed cytatem."""
    last_f = max((m.end() for m in _FEMALE.finditer(context)), default=-1)
    last_m = max((m.end() for m in _MALE.finditer(context)), default=-1)
    if last_f < 0 and last_m < 0:
        return None
    return "female" if last_f > last_m else "male"


def _cue_gender(word: str) -> str:
    return "female" if _FEMALE.fullmatch(word) else "male"


def _speaker(context: str, history: str, voices: Dict[str, Tuple[str, str]],
             people: Sequence[Person]) -> Tuple[str, str, str]:
    """(voice slug, gender, character slug) of the person speaking the next quote, else
    ("", "", "").

    Only the line's own sentence counts, up to its speaking verb (says, asks, mówi...): the
    speaker is named before the verb, so in "Ola says to Kuba" and "Ola mówi do Kuby" Ola
    speaks. Polish case forms ("Kubie", "Kuby") and names after "to", "at", "do"... are
    objects, never the speaker. Of the people before the verb the one closest to it speaks
    ("Hanna watches as the man says": the man), but "X hugs Y and says" keeps the sentence's
    first person. An explicit @tag in that sentence (a character's before a voice of the same
    name) wins over bare names and gender words. her / him / his after someone of the other
    gender are the listener. A pronoun ("Ola looks at Kuba. He says") means the person of that
    gender named last before it (a character, or an @voice); quoted lines never name anyone."""
    start = max((m.end() for m in _SENTENCE.finditer(context)), default=0)
    verb = max(_VERB.finditer(context, start), key=lambda m: m.start(), default=None)
    stop = verb.start() if verb else len(context)
    first = bool(verb and _AND.search(context, 0, stop))

    def pick(found: list):
        # (start, end, priority, value): the first person of an "and says" sentence, else the
        # one closest to the verb; a character before a voice before a gender word
        return (min(found, key=lambda t: (t[0], -t[2])) if first
                else max(found, key=lambda t: (t[1], t[2])))[3]

    tagged = [(m.start(), m.end(), 1, (voice, gender, slug))
              for pattern, slug, gender, voice, _ in people
              for m in pattern.finditer(context, start)
              if m.group(0).startswith("@") and m.end() <= stop]
    tagged += [(m.start(), m.end(), 0, voices[m.group(1).casefold()] + ("",))
               for m in _TAG.finditer(context, start)
               if m.group(1).casefold() in voices and m.end() <= stop]
    if tagged:
        return pick(tagged)

    def subject(m) -> bool:
        return (m.end() <= stop and not m.groupdict().get("f")
                and not _PREP.search(context, 0, m.start()))

    refs = [(m.start(), m.end(), 2, (voice, gender, slug))
            for pattern, slug, gender, voice, _ in people
            for m in pattern.finditer(context, start) if subject(m)]
    refs += [(m.start(), m.end(), 1, voices[m.group(0).casefold()] + ("",))
             for m in _WORD.finditer(context, start)
             if m.group(0).casefold() in voices and subject(m)]
    cues = [m for rx in (_FEMALE, _MALE) for m in rx.finditer(context, start) if m.end() <= stop]
    named_genders = [(t[0], t[3][1]) for t in refs]
    for cue in cues:
        if _OBJECT.fullmatch(cue.group(0)):
            before = [_cue_gender(c.group(0)) for c in cues
                      if c.start() < cue.start() and not _OBJECT.fullmatch(c.group(0))]
            before += [g for at, g in named_genders if at < cue.start()]
            if before and _cue_gender(cue.group(0)) not in before:
                continue  # "Kuba takes her hand and says": her is the listener
        refs.append((cue.start(), cue.end(), 0, cue))
    if not refs:
        return "", "", ""
    who = pick(refs)
    if not isinstance(who, re.Match):
        return who
    gender = _cue_gender(who.group(0))
    if _PRONOUN.fullmatch(who.group(0)):
        upto = len(history) - len(context) + who.start()
        # quoted lines never name anyone (as in app/cast.py mentions); same length keeps upto
        plain = QUOTE.sub(lambda q: " " * len(q.group(0)), history)[:upto]
        named = [(m.end(), 1, voice, slug) for pattern, slug, g, voice, _ in people if g == gender
                 for m in pattern.finditer(plain)]
        named += [(m.end(), 0, voices[m.group(1).casefold()][0], "")
                  for m in _TAG.finditer(plain)
                  if voices.get(m.group(1).casefold(), ("", ""))[1] == gender]
        if named:
            _, _, voice, slug = max(named, key=lambda t: (t[0], t[1]))
            return voice, gender, slug
    return "", gender, ""


def _is_shown(prompt: str, match: re.Match) -> bool:
    """Whether the quote `match` of `prompt` is text shown in the picture (_ON_DISPLAY, _WRITING), not a spoken
    line. Only its own sentence up to the quote counts, with earlier quotes blanked out."""
    head = QUOTE.sub(lambda q: " " * len(q.group(0)), prompt[:match.start()])
    context = _IMAGE_NUMBER.sub("", head[max((m.end() for m in _SENTENCE.finditer(head)), default=0):])
    if _ALOUD.search(context) or _STUDIO_SPEAKS.search(context):
        return False
    if _ON_DISPLAY.search(context):
        return True
    found = _WRITING.search(context)
    return bool(found) and not _SAID_WORDS.search(context, 0, found.start())


def spoken(prompt: str) -> List[re.Match]:
    """The quotes of a prompt that are spoken lines, in order: every quote but the text shown in the picture."""
    return [match for match in QUOTE.finditer(prompt or "") if not _is_shown(prompt, match)]


def shown(prompt: str) -> tuple[str, ...]:
    """The quoted texts a prompt shows in the picture (a screen's, a sign's...), which are never spoken."""
    return tuple(match.group(1).strip() for match in QUOTE.finditer(prompt or "") if _is_shown(prompt, match))


def replace_shown(prompt: str, old: str, new: str) -> str:
    """The prompt with every text it shows in the picture (shown) that reads `old` reading `new`, its quote marks
    kept; spoken lines and the rest of the text stay as they are."""
    prompt = prompt or ""
    parts, position = [], 0
    for match in QUOTE.finditer(prompt):
        if match.group(1).strip() == old and _is_shown(prompt, match):
            parts.append(prompt[position:match.start(1)])
            parts.append(new)
            position = match.end(1)
    parts.append(prompt[position:])
    return "".join(parts)


def _untag(text: str, people: Sequence[Person]) -> str:
    """'@ola says' -> 'Ola says', '@Hanna says' -> 'Hanna says' (picture and sound text)."""
    for pattern, _, _, _, name in people:
        text = pattern.sub(lambda m: name if m.group(0).startswith("@") else m.group(0), text)
    return _TAG.sub(r"\1", text)


def parse(prompt: str, voices: Optional[Dict[str, Tuple[str, str]]] = None,
          people: Optional[Sequence[Person]] = None) -> Tuple[str, str, List[Line]]:
    """Zwraca (opis_obrazu, opis_dzwieku_tla, kwestie).

    voices: casefolded voice names and slugs -> (slug, gender), see voices.tokens(); people:
    the job's characters (app/cast.py people()) - a character named before a line
    speaks it with the character's voice. Without them only the gender of the speaker is
    recognised. Both descriptions lose the @ of a tag."""
    voices = voices or {}
    people = list(people or [])
    lines: List[Line] = []
    visual, sound, pos = [], [], 0
    has_f, has_m = bool(_FEMALE.search(prompt)), bool(_MALE.search(prompt))
    for m in QUOTE.finditer(prompt):
        before = prompt[pos:m.start()]
        if _is_shown(prompt, m):
            # text shown in the picture (a screen's, a sign's): it stays in the picture text as written, the
            # background sound has no use for it, and nobody says it
            kept = _untag(before, people)
            visual.append(kept + m.group(0))
            sound.append(kept)
            pos = m.end()
            continue
        text = m.group(1).strip()
        voice, gender, character = _speaker(before, prompt[:m.start()], voices, people)
        speaker = gender or _gender(before)
        if speaker is None:
            if lines and has_f and has_m:      # dialog bez wskazowek - na zmiane
                speaker = "male" if lines[-1].speaker == "female" else "female"
            elif lines:
                speaker = lines[-1].speaker
            else:
                speaker = "male" if has_m and not has_f else "female"
        if not voice and not character and not _PAIR[speaker].search(prompt):
            # "@Hanna says „Hi”. She adds „Bye”": the same person keeps talking unless the
            # description announces a second person of that gender
            previous = next((l for l in reversed(lines) if l.speaker == speaker), None)
            if previous is not None:
                voice, character = previous.voice, previous.character
                if character and _gender(before):
                    # "Ola and Nova chat. Ola says „Hi”. The other woman answers „Hey”": a
                    # gender word after a named speaker means the other character of that
                    # gender (the one named last before this line)
                    plain = QUOTE.sub(lambda q: " " * len(q.group(0)), prompt[:m.start()])
                    named = [(n.end(), p_voice, slug) for pattern, slug, g, p_voice, _ in people
                             if g == speaker and slug != character
                             for n in pattern.finditer(plain)]
                    if named:
                        _, voice, character = max(named, key=lambda t: t[0])
        lines.append(Line(speaker, text, voice=voice, character=character))
        before = _untag(before, people)  # "@Hanna says" -> "Hanna says" for picture and sound
        # "says „...”" -> "talks" w tym samym miejscu; kropka, jesli cytat konczyl zdanie
        stop = "." if text[-1] in ".!?" else ""
        if _SAYS.search(before):
            visual.append(_SAYS.sub("talks", before) + stop)
        elif before.strip():
            visual.append(before.rstrip() + " talking" + stop)
        # kolejny cytat tuz po poprzednim - "talking" juz jest w opisie
        sound.append(_SAYS.sub("", before) + stop)
        pos = m.end()
    tail = _untag(prompt[pos:], people)
    visual.append(tail)
    sound.append(tail)

    def tidy(parts: List[str], drop_fragments: bool = False) -> str:
        s = re.sub(r"\s{2,}", " ", "".join(parts))
        if drop_fragments:  # "and." / "he." po usunietej kwestii
            s = re.sub(r"\b(and|i|oraz|he|she|they|on|ona|oni)\s*([.,!?]|$)", r"\2", s)
        s = re.sub(r"\s+([,.!?])", r"\1", s)
        s = re.sub(r"([.!?,])[.!?,]+", r"\1", s)
        if drop_fragments:  # urywki ("The man.", "She.") nic nie wnosza do opisu dzwieku
            s = " ".join(z for z in re.split(r"(?<=[.!?])\s+", s) if len(z.split()) > 2)
        return s.strip(" ,:")

    last: dict = {}
    for line in lines:
        g = line.speaker
        if line.character:
            line.who = f"{g}#char:{line.character}"
            continue
        if line.voice:
            line.who = f"{g}#{line.voice}"
            continue
        idx = (1 - last[g] if g in last else 0) if _PAIR[g].search(prompt) else 0
        last[g] = idx
        line.who = f"{g}#{idx}"
    if not lines:
        clean = re.sub(r"\s{2,}", " ", _untag(prompt, people)).strip()
        if len(visual) == 1:  # no quote at all: both texts as they always were
            return clean, clean, []
        # only text shown in the picture: the picture keeps it as written, the background sound does without it
        quiet = re.sub(r"\s{2,}", " ", QUOTE.sub(" ", _untag(prompt, people))).strip()
        return clean, re.sub(r"\s+([,.!?])", r"\1", quiet), []
    return f"{tidy(visual).rstrip('.!?')}, natural lip movement", tidy(sound, True), lines


def quotes(prompt: str) -> tuple[str, ...]:
    """Dialogue in the same order as the synthesizer parses it: the spoken quotes, never text shown in the
    picture (spoken)."""
    return tuple(match.group(1).strip() for match in spoken(prompt))


def assign_speakers(lines: List[Line], speakers: Sequence[str], people: Sequence[Person]) -> List[Line]:
    chosen = {slug: (gender, voice) for _, slug, gender, voice, _ in people}
    result = list(lines)
    for index, slug in enumerate(speakers):
        if not slug:
            continue
        if index >= len(lines) or slug not in chosen:
            raise ValueError("Choose a character in this scene for the spoken line.")
        gender, voice = chosen[slug]
        result[index] = replace(lines[index], speaker=gender, character=slug, voice=voice,
                                who=f"{gender}#char:{slug}")
    return result


def voices_for(lines: List[Line], catalog: Dict[str, List[str]],
               preferred: Optional[Tuple[str, str]] = None) -> List[str]:
    """Voice slug for every line.

    A line that names a voice keeps it while that voice still exists in catalog; a character keeps
    its own voice (CHARACTER_VOICE). A character without a voice gets the first voice of its gender that no other
    character in the video uses yet (two women never share a voice by default). The others
    get the character's voice (preferred = (slug, gender), when the gender matches) or the
    first voice of their gender in catalog (the default chosen on the Voices page); the second
    unnamed person of the same gender ("two women") gets the next voice in that list. An
    unnamed person never gets a voice that a character or a named @voice already speaks with
    in the video, as long as that gender has another voice."""
    known = {slug for slugs in catalog.values() for slug in slugs}
    by_character: Dict[str, str] = {}
    for line in lines:  # characters with their own voice keep it, whatever comes first
        if line.character and (line.voice in known or line.voice.startswith(CHARACTER_VOICE)):
            by_character.setdefault(line.character, line.voice)
    for line in lines:  # then the others: the first voice of the gender no character uses yet
        if line.character and line.character not in by_character:
            pool = list(catalog.get(line.speaker) or [])
            taken = set(by_character.values())
            by_character[line.character] = next((v for v in pool if v not in taken),
                                                pool[0] if pool else "")
    # voices already heard in the video: the characters' and the named ones (@Hanna)
    used = set(by_character.values()) | {line.voice for line in lines
                                         if not line.character and line.voice in known}
    out: List[str] = []
    for line in lines:
        if line.character:
            out.append(by_character[line.character])
            continue
        if line.voice and line.voice in known:
            out.append(line.voice)
            continue
        pool = list(catalog.get(line.speaker) or [])
        if preferred and preferred[0] and preferred[1] == line.speaker:
            if preferred[0] in pool:
                pool.remove(preferred[0])
            pool.insert(0, preferred[0])
        free = [v for v in pool if v not in used] or pool  # never a voice someone already uses
        key = line.who.split("#")[-1]
        idx = int(key) if key.isdigit() else 0
        out.append(free[idx % len(free)] if free else "")
    return out


def assemble(wavs: List[str], out_path: str | Path, start: float = 0.3,
             gap: float = 0.5) -> Tuple[float, List[Tuple[float, float]]]:
    """Skleja kwestie w jedna sciezke (krotka cisza przed i miedzy).

    Zwraca (dlugosc w s, [(poczatek, koniec) kazdej kwestii]) - czasy sa potrzebne do
    ruchu ust, gdy mowi kilka postaci.
    """
    parts, rate = [], None
    for w in wavs:
        data, sr = sf.read(w, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        rate = rate or sr
        parts.append(data)
    silence = lambda s: np.zeros(int(rate * s), dtype=np.float32)  # noqa: E731
    track, timings, t = [silence(start)], [], start
    for i, p in enumerate(parts):
        timings.append((t, t + len(p) / rate))
        t += len(p) / rate + (gap if i + 1 < len(parts) else 0)
        track += [p] + ([silence(gap)] if i + 1 < len(parts) else [])
    audio = np.concatenate(track)
    sf.write(str(out_path), audio, rate)
    return len(audio) / rate, timings


PLACED_PEAK = 0.98  # place(): a sum of overlapping lines is scaled down below this peak


def durations(wavs: Sequence[str | Path]) -> List[float]:
    """Length in seconds of every line file, read from the file headers."""
    return [float(sf.info(str(path)).duration) for path in wavs]


def place(wavs: Sequence[str | Path], starts: Sequence[float], out_path: str | Path, seconds: float) -> None:
    """Lines on the scene clock (Timeline dialogue with entry times): one mono track exactly
    `seconds` long with every line from its own start. Overlapping lines are summed, never cut;
    only the scene end cuts a line. When the sum would clip, the whole track is scaled down (the
    export sets the dialogue loudness anyway)."""
    if len(wavs) != len(starts):
        raise ValueError("Each spoken line needs a start time.")
    parts, rate = [], None
    for w in wavs:
        data, sr = sf.read(str(w), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if rate is not None and sr != rate:
            raise ValueError("The spoken lines have different sample rates.")
        rate = sr
        parts.append(data)
    rate = rate or 48000
    track = np.zeros(int(round(seconds * rate)), dtype=np.float32)
    for data, start in zip(parts, starts):
        begin = int(round(max(float(start), 0.0) * rate))
        piece = data[:max(track.size - begin, 0)]
        track[begin:begin + piece.size] += piece
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > PLACED_PEAK:
        track *= np.float32(PLACED_PEAK / peak)
    sf.write(str(out_path), track, rate)


def speaker_track(dialog_wav: str | Path, timings: List[Tuple[float, float]], keep: List[int],
                  out_path: str | Path, length: float, lines: Optional[Sequence[str | Path]] = None) -> None:
    """Sciezka jednej postaci: tylko jej kwestie (indeksy `keep`), reszta cisza, dlugosc
    `length` s - na tej podstawie lip sync (MuseTalk) rusza ustami tylko tej osoby.

    lines: every line's own file as synthesized (one per timing). The track is then made of the kept
    lines alone, each from its start: when lines overlap, another speaker's words do not move this
    face's lips (the dialogue track sums overlapping lines). Without them the windows are cut out of
    `dialog_wav`."""
    audio, rate = sf.read(str(dialog_wav), dtype="float32", always_2d=False)
    out = np.zeros(int(length * rate) + 1, dtype=np.float32)
    if lines is not None:
        if len(lines) != len(timings):
            raise ValueError("Each spoken line needs its own file.")
        for i in keep:
            data, sr = sf.read(str(lines[i]), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != rate:
                raise ValueError("The spoken lines have different sample rates.")
            a = max(int(timings[i][0] * rate), 0)
            piece = data[:max(len(out) - a, 0)]
            out[a:a + piece.size] += piece
        sf.write(str(out_path), out, rate)
        return
    for i in keep:
        a, b = (int(x * rate) for x in timings[i])
        b = min(b, len(audio), len(out))
        out[a:b] = audio[a:b]
    sf.write(str(out_path), out, rate)
