# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Gradio prefixes every custom CSS rule and splits selectors at every comma, also inside :is(); the app
writes such lists out first (ui_theme.flatten_selectors) so the cascade is the one the CSS describes."""
import re
import unittest

from app import main, ui_theme

_LIST = re.compile(r":(?:is|where|not|has)\(([^()]*(?:\([^()]*\)[^()]*)*)\)")


class ExpandSelectorTests(unittest.TestCase):
    def test_lists_become_separate_selectors(self):
        self.assertEqual(ui_theme._expand_selector("#a :is(.b, .c) > p"), ["#a :is(.b) > p", "#a :is(.c) > p"])
        self.assertEqual(ui_theme._expand_selector(".x:not(.b, .c)"), [".x:not(.b):not(.c)"])
        self.assertEqual(ui_theme._expand_selector("#a :is(.b, .c) :where(p, li)"),
                         ["#a :is(.b) :where(p)", "#a :is(.b) :where(li)",
                          "#a :is(.c) :where(p)", "#a :is(.c) :where(li)"])
        self.assertEqual(ui_theme._expand_selector('.y:is([data-x="a,b"], :hover)'),
                         ['.y:is([data-x="a,b"])', ".y:is(:hover)"])
        self.assertEqual(ui_theme._expand_selector("#a .b"), ["#a .b"])

    def test_style_sheet_structure_is_kept(self):
        css = ("/* a, comment */\n#p :is(.a, .b) { color: #fff; background: url(\"data:image/svg+xml;utf8,%3Csvg%3E\"); }\n"
               "@media (max-width: 768px) {\n  #p :is(.c, .d) { gap: 1px; }\n}\n"
               "@keyframes pulse { from { opacity: 1; } to { opacity: .5; } }\n.e, .f { margin: 0; }\n")
        flat = ui_theme.flatten_selectors(css)
        self.assertIn("/* a, comment */", flat)
        self.assertIn('#p :is(.a),\n#p :is(.b) { color: #fff; background: url("data:image/svg+xml;utf8,%3Csvg%3E"); }',
                      flat)
        self.assertIn("@media (max-width: 768px) {", flat)
        self.assertIn("#p :is(.c),\n#p :is(.d) { gap: 1px; }", flat)
        self.assertIn("@keyframes pulse {", flat)
        self.assertIn(".e,\n.f { margin: 0; }", flat)
        self.assertEqual((flat.count("{"), flat.count("}")), (css.count("{"), css.count("}")))


class AppStyleTests(unittest.TestCase):
    def test_no_selector_list_reaches_gradio(self):
        css = main.STYLE["css"]
        left = [match.group(0) for match in _LIST.finditer(css) if "," in match.group(1)]
        self.assertEqual(left, [])
        self.assertEqual(css.count("{"), css.count("}"))

    def test_the_narrow_effect_row_override_can_win(self):
        """The wide effect-row columns are written without :is(.a, .b), so the [data-narrow] rule beats them."""
        css = main.STYLE["css"]
        self.assertIn('#timeline-page .tl-rows[data-rows="effects"] :is(.tl-row) {', css)
        self.assertIn('#timeline-page [data-narrow] .tl-rows[data-rows="effects"] .tl-row {', css)


if __name__ == "__main__":
    unittest.main()
