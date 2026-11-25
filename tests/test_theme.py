# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Dark grayscale theme: only achromatic colors (R = G = B, alpha allowed) anywhere - the Gradio
theme in both modes, the app's stylesheets, the page script, colors written in app/ui_*.py and
the HTML the built app sends to the browser. Status must stay readable without hue (shapes)."""
from __future__ import annotations

import ast
import importlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

import gradio as gr

from app import ui_theme

ROOT = Path(__file__).resolve().parent.parent

# CSS named colors (CSS Color 4); only a few of them are gray
NAMED = frozenset("""
aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet
brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue
darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange
darkorchid darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey darkturquoise
darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen
fuchsia gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred
indigo ivory khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan
lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen
lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen magenta
maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue
mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin
navajowhite navy oldlace olive olivedrab orange orangered orchid palegoldenrod palegreen
paleturquoise palevioletred papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red
rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue
slategray slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white
whitesmoke yellow yellowgreen""".split())
GRAY_NAMES = frozenset("black white gray grey silver gainsboro whitesmoke dimgray dimgrey darkgray "
                       "darkgrey lightgray lightgrey".split())
# system colors follow the OS (blue links, accent, highlight) - never gray for sure
SYSTEM = frozenset(name.lower() for name in """AccentColor AccentColorText ActiveText ButtonBorder
ButtonFace ButtonText Canvas CanvasText Field FieldText GrayText Highlight HighlightText LinkText
Mark MarkText SelectedItem SelectedItemText VisitedText -webkit-focus-ring-color -webkit-link""".split())
RGB_SPACES = frozenset("srgb srgb-linear display-p3 a98-rgb prophoto-rgb rec2020".split())
# properties whose value may hold a color (custom properties too: they feed colors elsewhere)
COLOR_PROPERTY = re.compile(
    r"(--[\w-]+|[\w-]*-color|(-webkit-)?(accent-color|caret-color|color|fill|stroke|stop-color|flood-color|"
    r"lighting-color|background(-[a-z-]+)?|border(-[a-z-]+)?|outline(-[a-z-]+)?|box-shadow|"
    r"text-shadow|text-decoration(-[a-z-]+)?|text-emphasis(-[a-z-]+)?|column-rule(-[a-z-]+)?|"
    r"scrollbar-color|filter|mask(-[a-z-]+)?|text-fill-color|text-stroke(-[a-z-]+)?))$")

_HEX = re.compile(r"(?<![\w&#])#([0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})(?![\w-])")
_FUNCTION = re.compile(r"(?<![\w-])(rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(", re.I)
_WORD = re.compile(r"(?<![\w-])(-webkit-[\w-]+|[A-Za-z][\w-]*)(?![\w(-])")
_URL = re.compile(r"url\(\s*(\"[^\"]*\"|'[^']*'|[^)]*)\s*\)", re.I)
_DATA = re.compile(r"data:image/svg\+xml[^\"'\s)]*", re.I)
_ATTRIBUTE = re.compile(r"(?<![\w-])(fill|stroke|stop-color|flood-color|lighting-color|color|bgcolor)"
                        r"\s*=\s*(\"[^\"]*\"|'[^']*')", re.I)
_DECLARATION = re.compile(r"(?<![\w-])(--[\w-]+|-?[a-z][a-z0-9-]*)\s*:\s*([^;{}<>\"'`\n]*)")
_JS_STYLE = re.compile(r"\.style\.([A-Za-z]+)\s*=\s*([\"'`])(.*?)\2")
_JS_PROPERTY = re.compile(r"setProperty\(\s*([\"'`])([\w-]+)\1\s*,\s*([\"'`])(.*?)\3")


def _functions(text: str):
    """(name, arguments, whole call) of every color function; nested parentheses are kept."""
    for match in _FUNCTION.finditer(text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[end], 0)
            end += 1
        yield match.group(1).lower(), text[match.end():end - 1], text[match.start():end]


def value_colors(value: str, named: bool = True) -> list[str]:
    """Color tokens in one CSS value: hex, color functions and (optionally) color keywords."""
    found = [m.group(0) for m in _HEX.finditer(value)]
    found += [whole for _, _, whole in _functions(value)]
    if named:
        found += [w for w in _WORD.findall(value) if w.lower() in NAMED or w.lower() in SYSTEM]
    return found


def _number(token: str, percent_of: float) -> float:
    if token == "none":
        return 0.0
    if token.endswith("%"):
        return float(token[:-1]) * percent_of / 100
    return float(re.sub(r"(deg|grad|rad|turn)$", "", token))


def _gray_function(name: str, args: str) -> bool:
    if re.search(r"[a-z-]+\(|\bfrom\b", args):  # var(), calc(), relative colors: cannot be verified
        return False
    parts = [p for p in re.split(r"[\s,/]+", args.strip().lower()) if p]
    try:
        if name in ("rgb", "rgba"):
            r, g, b = (_number(p, 255) for p in parts[:3])
            return abs(r - g) < 1e-9 and abs(g - b) < 1e-9
        if name in ("hsl", "hsla"):
            saturation, lightness = _number(parts[1], 100), _number(parts[2], 100)
            return saturation == 0 or lightness <= 0 or lightness >= 100
        if name == "hwb":
            return _number(parts[1], 100) + _number(parts[2], 100) >= 100
        if name in ("lab", "oklab"):
            return _number(parts[1], 1) == 0 and _number(parts[2], 1) == 0
        if name in ("lch", "oklch"):
            return _number(parts[1], 1) == 0
        if name == "color":
            space, *channels = parts
            return space in RGB_SPACES and len({_number(c, 1) for c in channels[:3]}) == 1 \
                and len(channels) >= 3
    except (ValueError, IndexError):
        return False
    return False


def is_gray(token: str) -> bool:
    """True when the color is achromatic (R = G = B; alpha may be anything)."""
    low = token.strip().lower()
    if low.startswith("#"):
        digits = low[1:]
        if len(digits) in (3, 4):
            return digits[0] == digits[1] == digits[2]
        return digits[0:2] == digits[2:4] == digits[4:6]
    call = re.match(r"([a-z-]+)\((.*)\)$", low, re.S)
    if call:
        return _gray_function(call.group(1), call.group(2))
    return low in GRAY_NAMES


def chromatic(tokens) -> list[str]:
    return sorted({t for t in tokens if not is_gray(t)})


def _embedded(data: str) -> list[str]:
    """Colors in an SVG (or any data) behind url() or a data: URI, after URL decoding."""
    text = re.sub(r"url\(\s*['\"]?#[\w-]+['\"]?\s*\)", "", unquote(data))  # links to SVG ids
    return (value_colors(text, named=False) + _attribute_colors(text)
            + _declaration_colors(text))


def _take_urls(text: str, found: list[str]) -> str:
    def take(match):
        found.extend(_embedded(match.group(1).strip("\"'")))
        return "url()"
    return _URL.sub(take, text)


def _attribute_colors(text: str) -> list[str]:
    return [c for m in _ATTRIBUTE.finditer(text) for c in value_colors(m.group(2)[1:-1])]


def _declaration_colors(text: str) -> list[str]:
    return [c for m in _DECLARATION.finditer(text) if COLOR_PROPERTY.match(m.group(1).lower())
            for c in value_colors(m.group(2))]


def css_colors(css: str) -> list[str]:
    """Every color in a stylesheet: declaration values (of any property) and SVG inside url()."""
    found: list[str] = []
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = _take_urls(css, found)
    css = re.sub(r"\"[^\"]*\"|'[^']*'", '""', css)  # font names, content: text, grid areas
    while True:  # innermost blocks first, so nested rules keep their own declarations
        bodies = re.findall(r"\{([^{}]*)\}", css)
        if not bodies:
            break
        for body in bodies:
            for declaration in body.split(";"):
                prop, colon, value = declaration.partition(":")
                if colon:
                    found += value_colors(value, named=bool(COLOR_PROPERTY.match(prop.strip().lower())))
        css = re.sub(r"\{[^{}]*\}", ";", css)
    return found


def markup_colors(text: str) -> list[str]:
    """Colors in HTML, SVG, CSS or script text: color attributes, declarations of color properties,
    inline style assignments in scripts and SVG behind url()/data: URIs. Prose is ignored, so
    "Job #162" or "a green hoodie" are not colors."""
    found: list[str] = []
    text = _take_urls(text, found)
    for match in _DATA.finditer(text):
        found += _embedded(match.group(0)[len("data:image/svg+xml"):])
    found += _attribute_colors(text) + _declaration_colors(text)
    for match in _JS_STYLE.finditer(text):
        prop = re.sub(r"[A-Z]", lambda c: "-" + c.group(0).lower(), match.group(1))
        if COLOR_PROPERTY.match(prop):
            found += value_colors(match.group(3))
    for match in _JS_PROPERTY.finditer(text):
        if COLOR_PROPERTY.match(match.group(2)):
            found += value_colors(match.group(4))
    return found


_WHOLE_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}|(rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(.*\)", re.S)
# keyword arguments that take a color (waveform_color=, button_primary_background_fill=, ...)
_COLOR_ARGUMENT = re.compile(r"color|fill|stroke|background|border|shadow|outline")


def literal_colors(text: str) -> list[str]:
    """Colors in a string: markup/CSS inside it, the string being a color itself ("#4f46e5") and
    rgb()/hsl()-style calls anywhere ("0 1px 2px rgba(...)"). A bare word ("red") or "#162" is
    only a color where a color is expected - see markup_colors() and source_colors()."""
    found = markup_colors(text)
    if _WHOLE_COLOR.fullmatch(text.strip()):
        found += value_colors(text.strip())
    found += [whole for name, _, whole in _functions(text) if name not in ("color", "lab")]
    return found


def source_colors(source: str) -> list[str]:
    """Colors written in Python source: its string literals and color keyword arguments."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found += literal_colors(node.value)
        elif isinstance(node, ast.keyword) and node.arg and _COLOR_ARGUMENT.search(node.arg) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            found += value_colors(node.value.value)
    return found


def strings(value):
    """Every string inside nested config data (dicts, lists)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def _theme_blocks(css: str) -> tuple[dict, dict]:
    light, dark = css.split(":root.dark, :root .dark {", 1)
    light = light[light.index(":root {") + len(":root {"):]
    parse = lambda body: dict(re.findall(r"--([\w-]+):\s*([^;]*);", body[:body.index("}")]))
    return parse(light), parse(dark)


def _root_tokens(css: str) -> dict:
    tokens = {}
    for body in re.findall(r"(?<![\w-]):root\s*\{([^{}]*)\}", css):
        tokens.update(re.findall(r"--([\w-]+):\s*([^;]+?)\s*;", body))
    return tokens


def _selectors(text: str) -> list[str]:
    """A selector list split on its top-level commas (not those inside :is(...))."""
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return [" ".join(p.split()) for p in parts + [text[start:]]]


def _rule(css: str, selector: str) -> dict:
    """Declarations of the rule whose selector list holds `selector` (whitespace-insensitive)."""
    want = " ".join(selector.split())
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if want in _selectors(selectors):
            return {p.strip(): v.strip() for p, _, v in
                    (d.partition(":") for d in body.split(";")) if v.strip()}
    raise AssertionError(f"no CSS rule for {selector!r}")


class ScannerTests(unittest.TestCase):
    """The checks below are only as good as the color finder - prove it finds hues."""

    def test_finds_hues_in_every_notation(self):
        css = """.a { color: #4f46e5; background: rgba(79, 70, 229, .12); }
            .b { border: 1px solid hsl(240 50% 50%); outline-color: red; }
            .c { --x: oklch(70% 0.1 200); fill: color(display-p3 1 0 0); box-shadow: 0 0 0 1px #fa0; }
            .d { background: url("data:image/svg+xml,%3Csvg fill='%23ff0000'/%3E") }
            .e { mask: url("data:image/svg+xml;utf8,<svg stroke='orange'/>"); }
            .f { color: rgb(var(--x)); } .g { outline: 2px solid Highlight; }
            @media (max-width: 9px) { .h { color: lab(50% 20 0); } }"""
        self.assertEqual(sorted(chromatic(css_colors(css))), sorted([
            "#4f46e5", "rgba(79, 70, 229, .12)", "hsl(240 50% 50%)", "red", "oklch(70% 0.1 200)",
            "color(display-p3 1 0 0)", "#fa0", "#ff0000", "orange", "rgb(var(--x))", "Highlight",
            "lab(50% 20 0)"]))

    def test_accepts_grays_and_alpha(self):
        css = """.a { color: #fff; background: #0a0a0a80; border: 1px solid rgba(255, 255, 255, .08); }
            .b { color: hsl(200 0% 40%); fill: white; stroke: black; --y: rgb(0 0 0 / 0.1); }
            .c { background: transparent; color: currentColor; outline-color: #FFFFFF; }
            .d { background: url("data:image/svg+xml,%3Csvg fill='white' stroke='%23333'/%3E"); }
            .e { content: "red"; font-family: "Orange Sans"; white-space: nowrap; }"""
        self.assertEqual(chromatic(css_colors(css)), [])

    def test_prose_is_not_a_color(self):
        for text in ("Job #162 failed", "For example, an adult in a green hoodie", "Scene 3: a red car",
                     "&#160;", "Background: red sky at dusk", "<a href='#add'>", "Orange", "red car"):
            with self.subTest(text=text):
                self.assertEqual(literal_colors(text), [])

    def test_finds_colors_in_markup_and_scripts(self):
        html = ("<div style=\"width: 3px; background: #22c55e\"></div><svg><path fill='#1d4ed8'/></svg>"
                "<script>el.style.borderColor = 'tomato'; el.style.setProperty('--tl-x', 'rgb(1, 2, 3)');"
                "</script>")
        self.assertEqual(chromatic(markup_colors(html)),
                         sorted(["#22c55e", "#1d4ed8", "tomato", "rgb(1, 2, 3)"]))

    def test_finds_named_colors_in_any_color_property_and_hex_in_scripts(self):
        self.assertEqual(chromatic(css_colors(".a { -webkit-tap-highlight-color: crimson; }")), ["crimson"])
        script = "ctx.fillStyle = \"#e11d48\"; el.setAttribute('fill', '#22c55e'); const WARN = '#f59e0b';"
        self.assertEqual(chromatic(value_colors(script, named=False)), sorted(["#e11d48", "#22c55e", "#f59e0b"]))

    def test_finds_colors_in_python_source(self):
        source = '''
ACCENT = "#4f46e5"
SHADOW = "0 1px 2px rgba(24, 24, 27, .04)"
CSS = ".a { color: red; }"
HUES = ("green", "orange")  # names, not colors
LABEL = f"Job #{162} of 3: green hoodie"
player = gr.Audio(waveform_options=gr.WaveformOptions(waveform_color="orange"))
theme.set(button_cancel_text_color="crimson", block_background_fill="white")
'''
        self.assertEqual(chromatic(source_colors(source)),
                         sorted(["#4f46e5", "rgba(24, 24, 27, .04)", "red", "orange", "crimson"]))


class GradioThemeTests(unittest.TestCase):
    def test_theme_css_is_gray_in_both_modes(self):
        css = ui_theme.THEME._get_theme_css()
        light, dark = _theme_blocks(css)
        self.assertGreater(len(light), 100)
        self.assertEqual(chromatic(css_colors(css)), [])
        for name, value in {**light, **dark}.items():
            with self.subTest(variable=name):
                self.assertEqual(chromatic(css_colors(f"x {{ --{name}: {value}; }}")), [])

    def test_light_variables_equal_the_dark_ones(self):
        light, dark = _theme_blocks(ui_theme.THEME._get_theme_css())
        self.assertEqual(light, dark)
        for name in vars(ui_theme.THEME):
            if name.endswith("_dark") and hasattr(ui_theme.THEME, name[:-5]):
                self.assertEqual(getattr(ui_theme.THEME, name), getattr(ui_theme.THEME, name[:-5]), name)

    def test_page_background_before_the_app_loads_is_dark(self):
        # Gradio's index.html paints body with these before any stylesheet arrives
        for name in ("body_background_fill", "body_background_fill_dark"):
            self.assertEqual(ui_theme.THEME._get_computed_value(name), "#0a0a0a")
        for name in ("body_text_color", "body_text_color_dark"):
            self.assertEqual(ui_theme.THEME._get_computed_value(name), "#f2f2f2")

    def test_hues_are_pure_gray(self):
        gray = gr.themes.colors.neutral
        for hue in ("primary", "secondary", "neutral"):
            self.assertEqual(getattr(ui_theme.THEME, f"{hue}_500"), gray.c500)

    def test_buttons_follow_the_design(self):
        theme = ui_theme.THEME
        self.assertEqual((theme.button_primary_background_fill, theme.button_primary_text_color,
                          theme.button_primary_background_fill_hover), ("#f2f2f2", "#0a0a0a", "#ffffff"))
        self.assertEqual((theme.button_secondary_background_fill, theme.button_secondary_border_color),
                         ("#181818", "#2e2e2e"))
        self.assertEqual((theme.button_cancel_background_fill, theme.button_cancel_border_color,
                          theme.button_cancel_text_color), ("transparent", "#555555", "#f2f2f2"))
        # the tick on a light selected checkbox must be dark
        self.assertIn("black", unquote(theme.checkbox_check))
        self.assertNotIn("white", unquote(theme.checkbox_check))


class StylesheetTests(unittest.TestCase):
    TOKENS = {"c-bg": "#0a0a0a", "c-surface": "#121212", "c-surface-2": "#181818",
              "c-muted": "#1c1c1c", "c-border": "#262626", "c-border-2": "#3a3a3a",
              "c-text": "#f2f2f2", "c-text-2": "#bfbfbf", "c-text-3": "#8f8f8f",
              "c-accent": "#ffffff", "c-accent-soft": "#2a2a2a", "c-ok": "#e6e6e6",
              "c-warn": "#a6a6a6", "c-err": "#ffffff"}

    def test_tokens_are_defined_in_root(self):
        tokens = _root_tokens(ui_theme.CSS)
        self.assertEqual({k: v for k, v in tokens.items() if k.startswith("c-")}, self.TOKENS)
        self.assertEqual(_rule(ui_theme.CSS, ":root").get("color-scheme"), "dark")

    def test_app_css_is_gray(self):
        self.assertEqual(chromatic(css_colors(ui_theme.CSS)), [])
        self.assertEqual(chromatic(markup_colors(ui_theme.CSS)), [])

    def test_timeline_css_is_gray(self):
        # owned by the Timeline editor (app/ui_timeline_view.py); it must use the same grays
        view = importlib.import_module("app.ui_timeline_view")
        self.assertEqual(chromatic(css_colors(view.CSS)), [])

    def test_gradio_palette_used_by_its_frontend_is_gray(self):
        """Gradio's components use its Tailwind palette (--color-red-700 ...) directly; every
        palette color its frontend references must be redefined as a gray by ui_theme.CSS."""
        assets = Path(gr.__file__).parent / "templates" / "frontend" / "assets"
        css = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in assets.glob("*.css"))
        palette = set(re.findall(r"--(color-[a-z]+(?:-\d+)?):", css))  # Gradio's own definitions
        used = set(re.findall(r"var\(--(color-[a-z]+(?:-\d+)?)\s*[,)]", css)) & palette
        self.assertIn("color-red-500", used)  # the scan really sees the toasts
        theme_variables = set(re.findall(r"--([\w-]+):", ui_theme.THEME._get_theme_css()))
        defined = _root_tokens(ui_theme.CSS)
        missing = sorted(used - theme_variables - {"color-black", "color-white"} - set(defined))
        self.assertEqual(missing, [])
        self.assertEqual(chromatic(defined[name] for name in used if name in defined), [])

    def test_status_is_told_apart_by_shape(self):
        css = ui_theme.CSS
        dot, ring, square = (_rule(css, ".st"), _rule(css, ".st-warn"), _rule(css, ".st-err"))
        self.assertEqual(dot["border-radius"], "50%")
        self.assertEqual(ring["background"], "none")
        self.assertIn("inset", ring["box-shadow"])
        self.assertNotEqual(square.get("border-radius", "50%"), "50%")
        self.assertGreaterEqual(int(_rule(css, ":is(p, li):has(> .st-err)")["font-weight"]), 600)
        self.assertIn("inset", _rule(css, ".badge.warn::before")["box-shadow"])
        self.assertNotEqual(_rule(css, ".badge.err::before")["border-radius"], "50%")
        self.assertGreaterEqual(int(_rule(css, ".badge.err")["font-weight"]), 600)
        # the engine chip in the top bar: ready = filled dot, paused = hollow ring, unavailable = square, bold
        self.assertEqual(_rule(css, ".engine-chip .dot")["border-radius"], "50%")
        self.assertIn("inset", _rule(css, ".engine-chip .dot.pause")["box-shadow"])
        self.assertEqual(_rule(css, ".engine-chip .dot.pause")["background"], "none")
        self.assertNotEqual(_rule(css, ".engine-chip .dot.off")["border-radius"], "50%")
        self.assertGreaterEqual(int(_rule(css, '.engine-chip[data-state="off"] b')["font-weight"]), 600)
        for mark, kind in (("✅", "ok"), ("⚠", "warn"), ("❌", "err"), ("⏹", "info")):
            self.assertIn(f"st-{kind}", ui_theme.note(f"{mark} Text."))

    def test_neutral_badges_have_a_marker_of_their_own(self):
        css = ui_theme.CSS
        base = _rule(css, ".badge::before")
        shapes = {kind: {**base, **(_rule(css, f".badge.{kind}::before") if kind else {})}
                  for kind in ("", "ok", "warn", "err", "info")}
        looks = {kind: (shape.get("border-radius"), shape.get("background"), shape.get("box-shadow"),
                        shape.get("height")) for kind, shape in shapes.items()}
        self.assertEqual(len(set(looks.values())), 5, looks)  # neutral never looks like ok
        from app import ui_characters
        self.assertIn("badge ok", ui_characters._state_badge("Ready"))
        self.assertIn("badge warn", ui_characters._state_badge("Needs reference photos"))  # characters.state()

    def test_native_selects_get_a_gray_arrow_that_the_timeline_keeps(self):
        rule = _rule(ui_theme.CSS, ".gradio-container select")
        self.assertIn("data:image/svg+xml", rule["background-image"])
        arrow = unquote(rule["background-image"])
        self.assertIn(f"stroke='{ui_theme.TEXT_3}'", arrow)
        self.assertEqual(chromatic(_embedded(arrow)), [])
        self.assertIn("!important", rule["background-image"])
        view = importlib.import_module("app.ui_timeline_view")
        for selector in ("#timeline-page .tl-row :is(input, select)", "#timeline-page .timeline-speaker-row select"):
            with self.subTest(selector=selector):
                timeline_rule = _rule(view.CSS, selector)
                self.assertIn("background-color", timeline_rule)
                self.assertNotIn("background", timeline_rule)  # the shorthand would drop the arrow
        self.assertEqual(_rule(view.CSS, "#timeline-page .tl-row select")["padding-right"], "26px")

    def test_timeline_focus_is_told_apart_from_pressed_selected_and_failed(self):
        view = importlib.import_module("app.ui_timeline_view")
        outside = _rule(view.CSS, "#timeline-page .timeline-deck :is(.timeline-scale button, .timeline-clip):focus-visible")
        self.assertEqual(outside["outline-offset"], "2px")  # around a light pressed button, not inside it
        selected = _rule(view.CSS, '#timeline-page .timeline-clip[aria-pressed="true"]')
        self.assertNotIn("outline", selected)
        self.assertIn("inset", selected["box-shadow"])
        failed = _rule(view.CSS, '#timeline-page .timeline-clip[data-status="Failed"]')
        self.assertEqual(failed.get("border-style"), "dashed")
        self.assertNotIn("border", failed)  # not the 2 px white frame the selection had
        strips = {status: _rule(view.CSS, f'#timeline-page .timeline-clip[data-status="{status}"]::after')["background"]
                  for status in ("Ready", "Generating", "Queued", "Stopped", "Modified")}
        self.assertEqual(len(set(strips.values())), 5)  # narrow clips hide the label: the strip tells the status
        self.assertEqual(chromatic(css_colors(" ".join(f"x {{ background: {value}; }}" for value in strips.values()))),
                         [])

    def test_gradio_built_in_hues_are_overridden(self):
        css = ui_theme.CSS
        for kind in ("error", "warning", "info", "success"):
            self.assertIn("--toast-color", _rule(css, f".toast-body.{kind}"))
        self.assertIn("var(--c-border-2)", _rule(
            css, '.slider_input_container input[type="range"]::-webkit-slider-runnable-track')["background"])
        self.assertIn("background", _rule(css, "::selection"))
        self.assertIn("color", _rule(css, '.gradio-container :is([type="checkbox"], [type="radio"])'))
        self.assertIn("outline", _rule(css, '.gradio-container input:is([type="checkbox"], '
                                            '[type="radio"], [type="range"]):focus-visible'))


class ScriptTests(unittest.TestCase):
    def test_page_is_always_dark(self):
        js = ui_theme.JS
        self.assertIn("classList.add('dark')", js)
        self.assertIn("classList.contains('dark')", js)  # no observer loop
        self.assertNotIn("classList.remove('dark')", js)
        self.assertIn(ui_theme.STALE_TAB_GUARD, js)
        self.assertIn("syncNavigation", js)
        self.assertEqual(chromatic(markup_colors(js) + value_colors(js, named=False)), [])

    def test_every_page_script_writes_only_grays(self):
        view = importlib.import_module("app.ui_timeline_view")
        editor = importlib.import_module("app.ui_timeline")
        scripts = [value for name, value in vars(view).items() if name.endswith("_JS") and isinstance(value, str)]
        scripts.append(editor._SPEAKER_JS)
        self.assertGreaterEqual(len(scripts), 5)
        for script in scripts:
            self.assertEqual(chromatic(value_colors(script, named=False) + markup_colors(script)), [])

    def test_launch_style_uses_the_theme(self):
        main = importlib.import_module("app.main")
        view = importlib.import_module("app.ui_timeline_view")
        self.assertIs(main.STYLE["theme"], ui_theme.THEME)
        self.assertEqual(main.STYLE["js"], ui_theme.JS)
        # the theme CSS and the Timeline CSS, with selector lists written out for Gradio (ui_theme.flatten_selectors)
        self.assertEqual(main.STYLE["css"], ui_theme.flatten_selectors(ui_theme.CSS + view.CSS))


class SourceTests(unittest.TestCase):
    def test_ui_modules_write_only_grays(self):
        files = sorted((ROOT / "app").glob("ui_*.py")) + [ROOT / "app" / "main.py"]
        self.assertIn(ROOT / "app" / "ui_theme.py", files)
        for path in files:
            with self.subTest(file=path.name):
                self.assertEqual(chromatic(source_colors(path.read_text(encoding="utf-8"))), [])

    def test_rendered_fragments_are_gray(self):
        pages = (("Timeline", "timeline"), ("Gallery", "gallery"), ("Settings", "settings"))
        html = [ui_theme.BRAND, ui_theme.workspace_nav(pages, "Menu", "timeline"),
                ui_theme.topbar_start(pages, "timeline", "panel"),
                ui_theme.kpis([{"label": "Queue", "value": "1", "sub": "x", "progress": 0.4}]),
                ui_theme.table(["A"], [["<b>x</b>"]]), ui_theme.table(["A"], [], "Nothing yet"),
                ui_theme.head_html("Title", "Sub")]
        html += [ui_theme.badge("State", kind) for kind in ("", "ok", "warn", "err", "info")]
        html += [ui_theme.note(mark + " Text.") for mark, _ in ui_theme._MARKS]
        for fragment in html:
            self.assertEqual(chromatic(markup_colors(fragment)), [])
        # the menu icons are SVG masks: they really get decoded and checked
        self.assertIn("black", markup_colors(html[1]))

    def test_audio_waveforms_are_gray(self):
        options = ui_theme.waveform()
        for color in (options.waveform_color, options.waveform_progress_color,
                      options.trim_region_color):
            self.assertTrue(color and is_gray(color), color)


class BuiltAppTests(unittest.TestCase):
    """Everything the built app sends to the browser: HTML values, component scripts, waveforms."""

    def test_rendered_app_is_gray(self):
        from app import characters, prefs, studio, style_library
        from app.vram import GpuInfo
        main = importlib.import_module("app.main")
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(prefs, "_FILE", Path(tmp) / "prefs.json"), \
                patch.object(style_library, "all_custom", return_value=[]), \
                patch.object(characters, "all_characters", return_value=[]), \
                patch("app.items.all_items", return_value=[]), \
                patch.object(studio, "results", return_value=[]), \
                patch("app.ui_settings.model_licenses_html", return_value=""), \
                patch.object(main, "get_gpu_info",
                             return_value=GpuInfo(True, "NVIDIA GeForce RTX 4070", total_gb=16.0,
                                                  free_gb=9.5)):
            demo = main.build_ui()
        config = demo.get_config_file()
        for dependency in config["dependencies"]:  # event scripts (Show in Gallery...): any hex is a color
            script = dependency.get("js")
            if isinstance(script, str):
                self.assertEqual(chromatic(value_colors(script, named=False)), [])
        for component in config["components"]:
            props = component.get("props", {})
            with self.subTest(component=component.get("type"), id=component.get("id"),
                              elem_id=props.get("elem_id")):
                self.assertEqual(chromatic(c for text in strings(props) for c in literal_colors(text)), [])
                if isinstance(props.get("js_on_load"), str):
                    self.assertEqual(chromatic(value_colors(props["js_on_load"], named=False)), [])
                if component.get("type") == "audio":
                    waveform = props.get("waveform_options") or {}
                    for key in ("waveform_color", "waveform_progress_color"):
                        # unset = Gradio's tinted canvas defaults (#9ca3af, orange)
                        self.assertTrue(waveform.get(key) and is_gray(waveform[key]),
                                        f"gr.Audio needs waveform_options=ui_theme.waveform(): {key}")


class EmojiTests(unittest.TestCase):
    """Windows draws the status emoji the app's modules write into messages with its color emoji font
    (⛔ is a red disc): no Emoji_Presentation character may reach notes, tables or the Timeline."""

    @classmethod
    def setUpClass(cls):
        try:
            import regex
        except ImportError as error:  # pragma: no cover - installed with transformers
            raise unittest.SkipTest("the regex module is not installed") from error
        cls.emoji = regex.compile(r"\p{Emoji_Presentation}|️")

    def assertPlain(self, text):
        self.assertIsNone(self.emoji.search(text), text)

    def test_messages_tables_and_the_timeline_show_words_only(self):
        from app import characters, queue_manager, scene_parts, studio, timeline, ui_queue, ui_timeline
        from app import ui_timeline_view

        refusal = characters.check_look("a 15-year-old with long hair")
        blocked = "⛔ The photo appears to show a minor (estimated age 15). Only adults are allowed."
        self.assertIsNotNone(self.emoji.search(refusal))  # the source still has it: the page must not
        for mark, _ in ui_theme._MARKS:
            self.assertPlain(ui_theme.note(f"{mark} Done. {refusal}"))
        self.assertPlain(ui_theme.note(f"❌ #12: {blocked}"))
        self.assertPlain(ui_theme.note(f"Scene 1: {refusal}"))
        self.assertEqual(ui_theme.note("Job #162 finished."), "Job #162 finished.")
        self.assertPlain(ui_timeline_view.issues_html([scene_parts.Issue(scene_parts.ERROR, refusal)]))
        project = timeline.Project("0" * 32, 1, "Film", (timeline.Scene("A cup", 1),))
        failed = queue_manager.Job(4, timeline.PRODUCE, "Export movie · Film", {"project": project.id},
                                   status=queue_manager.ERROR, error=f"Scene 1: {refusal}")
        self.assertPlain(ui_timeline._queue_line(project, {"assets": []}, failed, False, None))
        self.assertPlain(ui_timeline._queue_line(project, {"assets": []}, None, False,
                                                 {"scene": 0, "error": f"Scene 1: {blocked}",
                                                  "kind": timeline.PREPARE}))
        with patch.object(studio, "jobs", return_value=[failed]), patch.object(studio, "current_set", return_value=""):
            finished = ui_queue._finished_html()
        self.assertIn("Only adult characters are supported", finished)
        self.assertPlain(finished)


class FaviconTests(unittest.TestCase):
    def test_the_tab_and_app_icon_is_gray(self):
        import numpy as np
        from PIL import Image

        main = importlib.import_module("app.main")
        with tempfile.TemporaryDirectory() as tmp, patch.object(ui_theme, "get_upload_folder", return_value=tmp):
            options = main.local_launch_options({"host": "127.0.0.1"})
            icon = Path(options["favicon_path"])
            self.assertTrue(icon.is_file())
            self.assertTrue(icon.is_relative_to(Path(tmp)))
            with Image.open(icon) as image:
                self.assertEqual((image.format, image.size), ("PNG", (64, 64)))
                pixels = np.asarray(image.convert("RGBA"))
            self.assertTrue(np.array_equal(pixels[..., 0], pixels[..., 1]))
            self.assertTrue(np.array_equal(pixels[..., 1], pixels[..., 2]))
            self.assertGreater(int(pixels[..., 3].max()), 0)
            written = icon.stat().st_mtime_ns
            self.assertEqual(ui_theme.favicon(tmp), str(icon))
            self.assertEqual(icon.stat().st_mtime_ns, written)  # an unchanged icon is not written again

    def test_the_page_links_the_icon_by_a_version_of_its_bytes(self):
        """The browser caches /favicon.ico on its own; the link's version changes with the icon, so a new icon
        is fetched again instead of the cached old one."""
        import hashlib

        main = importlib.import_module("app.main")
        with tempfile.TemporaryDirectory() as tmp, patch.object(ui_theme, "get_upload_folder", return_value=tmp):
            head = main.local_launch_options({"host": "127.0.0.1"})["head"]
            version = hashlib.sha256(ui_theme.favicon_png()).hexdigest()[:12]
            self.assertEqual(head, f'<link rel="icon" type="image/png" href="/favicon.ico?v={version}">')
            with patch.object(ui_theme, "favicon_png", return_value=b"another icon"):
                changed = main.local_launch_options({"host": "127.0.0.1"})["head"]
            self.assertNotEqual(changed, head)
            self.assertIn(hashlib.sha256(b"another icon").hexdigest()[:12], changed)
        with patch.object(ui_theme, "favicon", return_value=None):  # no icon file: Gradio's own, no stale link
            self.assertEqual(main.local_launch_options({"host": "127.0.0.1"})["head"], "")


if __name__ == "__main__":
    unittest.main()
