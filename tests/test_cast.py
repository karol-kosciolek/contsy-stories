# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Several characters in one video or photo: names in prompts (also with @ and in Polish case
forms), who is in the job and in each shot, the top picker's universes, and the composed frame
(every character's primary reference photo -> composed scene) without a GPU."""
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

from app import cast, characters, dialogue, prefs, refs, universes
from tests.character_fixtures import store_character

OLA = {"name": "Ola", "slug": "ola", "gender": "woman", "look": "Fair skin, long wavy auburn hair"}
KUBA = {"name": "Kuba", "slug": "kuba", "gender": "man", "look": "Short dark brown hair, a short beard"}
JAN = {"name": "Jan", "slug": "jan", "gender": "man", "look": ""}
NOVA = {"name": "Virtual Nova", "slug": "virtual_nova", "gender": "woman", "look": ""}
ALL = [OLA, KUBA, JAN, NOVA]


class NameTests(unittest.TestCase):
    def test_names_tags_and_polish_case_forms_are_found_in_order(self):
        self.assertEqual([c["name"] for c in cast.mentions(
            "Ola podaje Kubie kawę, a Jan patrzy na Olę.", ALL)], ["Ola", "Kuba", "Jan"])
        self.assertEqual([c["name"] for c in cast.mentions("Janowi i Jana", ALL)], ["Jan"])
        self.assertEqual([c["name"] for c in cast.mentions("@kuba and @virtual_nova", ALL)],
                         ["Kuba", "Virtual Nova"])
        self.assertEqual([c["name"] for c in cast.mentions("@Virtual Nova waves at Ola", ALL)],
                         ["Virtual Nova", "Ola"])

    def test_a_name_with_an_underscore_or_a_digit_is_found_in_any_letter_case(self):
        dog = {"name": "chain_dog", "slug": "chain_dog"}
        for text in ("chain_dog barks", "Chain_dog barks", "CHAIN_DOG barks", "@chain_dog barks"):
            self.assertEqual(cast.mentions(text, [dog]), [dog], text)
        self.assertEqual(cast.mentions("two chain_dogs bark", [dog]), [])
        ola = {"name": "Ola", "slug": "ola"}
        self.assertEqual(cast.mentions("an ola of light", [ola]), [])  # an ordinary name keeps its capital

    def test_common_words_and_quoted_dialogue_never_pick_a_character(self):
        self.assertEqual(cast.mentions("the jan dance; ola and kuba in lower case", ALL), [])
        self.assertEqual(cast.mentions("Kubek stoi na stole. Olaf idzie.", ALL), [])
        self.assertEqual([c["name"] for c in cast.mentions('Ola says „Hi Kuba!”', ALL)], ["Ola"])

    def test_an_exact_name_wins_over_another_name_s_case_form(self):
        ann = {"name": "Ann", "slug": "ann", "gender": "woman"}
        anna = {"name": "Anna", "slug": "anna", "gender": "woman"}
        self.assertEqual([c["slug"] for c in cast.mentions("Anna smiles", [ann, anna])], ["anna"])

    def test_names_become_image_markers_and_labels(self):
        self.assertEqual(cast.mark_names("Ola hands Kuba a coffee. Kuba smiles at Ola.", [OLA, KUBA]),
                         "The woman [1] hands the man [2] a coffee. The man [2] smiles at the woman [1].")
        self.assertEqual(cast.label_names("Ola hugs Nova.", [OLA, NOVA]),
                         "The first woman hugs Nova.")  # "Nova" alone is not the name
        self.assertEqual(cast.label_names("Ola hugs Virtual Nova.", [OLA, NOVA]),
                         "The first woman hugs the second woman.")
        self.assertEqual(cast.look_sentences([OLA, KUBA]),
                         ". The woman has fair skin, long wavy auburn hair. "
                         "The man has short dark brown hair, a short beard")
        self.assertEqual(cast.strip_mentions("@Ola says hi to @Hanna", [OLA]).strip(),
                         "says hi to @Hanna")

    def test_each_shot_shows_the_characters_it_names(self):
        shots = ["A woman walks on the beach.", "Kuba joins Ola.", "They laugh.", "Kuba waves."]
        self.assertEqual([[c["slug"] for c in s] for s in cast.shot_casts(shots, [OLA, KUBA])],
                         [["ola"], ["kuba", "ola"], ["kuba", "ola"], ["kuba"]])
        self.assertEqual(cast.shot_casts(["Anything"], []), [[]])

    def test_the_composed_scene_numbers_characters_first_and_shifts_reference_images(self):
        text = cast.mark_names(refs.renumber("Ola in the hoodie [1] hugs Kuba", 2), [OLA, KUBA])
        prompt = refs.edit_prompt(text, [1, 2, 3], False, False,
                                  base_lead=refs.cast_lead([OLA, KUBA]))
        self.assertTrue(prompt.startswith("These are different people: image 1 shows Ola, a woman; "
                                          "image 2 shows Kuba, a man."))
        self.assertIn("The woman from image 1 in the hoodie from image 3 hugs the man from image 2",
                      prompt)
        self.assertNotIn("The person must be exactly", prompt)

    def test_an_unnumbered_reference_image_is_still_described_with_a_cast(self):
        text = cast.mark_names(refs.renumber("Ola and Kuba walk in the park", 2), [OLA, KUBA])
        prompt = refs.edit_prompt(text, [1, 2, 3], False, False,
                                  base_lead=refs.cast_lead([OLA, KUBA]))
        self.assertIn("Use the exact items, places and any other people shown in image 3.", prompt)
        self.assertNotIn("image 1 and", prompt.split("Use the exact")[1][:40])  # only image 3
        plain = refs.edit_prompt("A woman in a park", [1], False, False)  # no numbers: as before
        self.assertIn("Use the exact person and items shown in image 1", plain)


class _Library(unittest.TestCase):
    """Characters, universes and prefs in a temporary folder."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cast_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for number, ch in enumerate(ALL):
            store_character(self.tmp / "characters", {**ch, "created": number})
        for target in (patch.object(characters, "DIR", self.tmp / "characters"),
                       patch.object(prefs, "_FILE", self.tmp / "prefs.json")):
            target.start()
            self.addCleanup(target.stop)


class ResolveTests(_Library):
    def test_picked_characters_come_first_then_the_named_ones(self):
        found, error = cast.resolve(["ola"], "Kuba and Jan wave at Ola")
        self.assertEqual(error, "")
        self.assertEqual([c["slug"] for c in found], ["ola", "kuba", "jan"])

    def test_more_than_the_limit_is_an_error_not_a_silent_cut(self):
        found, error = cast.resolve(["ola", "kuba"], "Jan and Virtual Nova", cap=3)
        self.assertEqual(len(found), 4)
        self.assertIn("Up to 3 characters", error)

    def test_characters_without_reference_photos_are_reported_not_used(self):
        (self.tmp / "characters" / "jan" / "photos" / "01.png").unlink()
        found, error, notes = cast.check_prompt("Ola talks to Jan", ["ola"])
        self.assertEqual([c["slug"] for c in found], ["ola"])
        self.assertEqual(notes[0], "No reference photos yet, so left out: Jan.")

    def test_unknown_tags_are_refused_but_character_tags_pass(self):
        with patch("app.voices.tokens", return_value={"hanna": ("hanna", "female")}):
            self.assertEqual(cast.check_prompt("@kuba says „Hi” to @Hanna", [])[1], "")
            self.assertIn("@Nobody", cast.check_prompt("@Nobody says „Hi”", [])[1])
            self.assertEqual(cast.check_prompt("@Virtual Nova waves", [])[1], "")

    def test_job_params_and_label(self):
        params = {}
        self.assertEqual(cast.tag(params, [characters.get("ola"), characters.get("kuba")]), " · Ola + Kuba")
        self.assertEqual((params["character"], params["characters"]), ("ola", ["ola", "kuba"]))
        self.assertEqual(cast.tag({}, []), "")

    def test_only_characters_who_appear_are_in_the_job(self):
        universes.save("Family", ["ola", "kuba"])
        picked = cast.picked("universe:family")  # the picker's pre-fill: both members
        found, error, notes = cast.check_prompt("Ola reads a book in a cafe", picked)
        self.assertEqual(([c["slug"] for c in found], error), (["ola"], ""))
        self.assertIn("Not named in the prompt, so not in it: Kuba.", notes)
        found, _, _ = cast.check_prompt("Kuba reads a book", picked)  # Kuba's photo, not Ola's
        self.assertEqual([c["slug"] for c in found], ["kuba"])
        found, _, _ = cast.check_prompt("Ola hands Kuba a coffee", picked)
        self.assertEqual([c["slug"] for c in found], ["ola", "kuba"])
        prompt = "Shot 1: A beach at sunset. Shot 2: Kuba runs."
        found, _, _ = cast.check_prompt(prompt, picked, refs.split_shots(prompt))
        self.assertEqual([c["slug"] for c in found], ["ola", "kuba"])  # shot 1 shows the main one

    def test_the_limit_counts_only_characters_who_appear(self):
        with patch.object(cast, "MAX_CAST", 2):
            found, error, _ = cast.check_prompt("Jan waves at Ola", ["ola", "kuba", "virtual_nova"])
            self.assertEqual(([c["slug"] for c in found], error), (["ola", "jan"], ""))
            self.assertIn("Up to 2", cast.check_prompt("Ola, Kuba and Jan", ["ola"])[1])

    def test_the_estimate_counts_only_characters_composed_into_shared_frames(self):
        from app import queue_manager as qm

        ola, kuba = characters.get("ola"), characters.get("kuba")
        params = {}
        cast.tag(params, [ola, kuba], ["Ola hands Kuba a coffee"])
        self.assertEqual(params["extra_characters"], 1)
        cast.tag(params, [ola, kuba], ["Ola walks.", "Kuba runs."])  # never together
        self.assertEqual(params["extra_characters"], 0)
        self.assertEqual(qm.Job(id=1, kind="obraz", label="", params={"extra_characters": 1}
                                ).cast_extra(), qm.CAST_EXTRA["obraz"])
        for bad in ({"characters": ["ola", "kuba"]}, {"extra_characters": "2"},
                    {"extra_characters": True}, {"extra_characters": -1}):
            self.assertEqual(qm.Job(id=2, kind="obraz", label="", params=bad).cast_extra(), 0)
        # a composed job runs another pipeline: it leaves the learned base rate alone
        runner = unittest.mock.Mock(return_value=(True, [], [], ""))
        for params, changes in (({"extra_characters": 1}, False), ({}, True)):
            job = qm.Job(id=3, kind="obraz", label="", params=params)
            job.started = qm.time.time() - 60.0
            with patch.dict(qm._RUNNERS, {"obraz": runner}), patch.dict(qm._RATE, {"obraz": 30.0}), \
                    patch.object(qm, "_save"), patch.object(qm, "FILES_DIR", self.tmp / "files"):
                qm._run(job)
                self.assertEqual(qm._RATE["obraz"] != 30.0, changes, params)

    def test_photo_prompts_lose_the_at_sign(self):
        chs = [characters.get("kuba"), characters.get("virtual_nova")]
        self.assertEqual(cast.untag("A photo of @kuba with @Virtual_Nova and @Ola", chs),
                         "A photo of Kuba with Virtual Nova and Ola")
        self.assertEqual(cast.untag('@kuba holds a sign reading "@kuba" and the hoodie @2', chs),
                         'Kuba holds a sign reading "@kuba" and the hoodie @2')

    def test_queued_jobs_resolve_names_when_cast_is_not_explicit(self):
        found, _ = cast.for_job({"characters": ["ola", "gone"], "prompt": "x"})
        self.assertEqual([c["slug"] for c in found], ["ola"])
        found, _ = cast.for_job({"character": "ola", "prompt": "Ola meets Kuba"})
        self.assertEqual([c["slug"] for c in found], ["ola", "kuba"])


class UniverseTests(_Library):
    def test_save_rename_and_delete(self):
        u, msg = universes.save("Family", ["ola", "kuba", "missing"])
        self.assertEqual((u["slug"], u["members"]), ("family", ["ola", "kuba"]))
        self.assertIn("2 characters", msg)
        self.assertIn("already exists", universes.save("family", ["ola", "jan"])[1])
        self.assertIn("character name", universes.save("Ola", ["ola", "jan"])[1])
        self.assertIn("at least two", universes.save("Solo", ["ola"])[1])
        self.assertIn("Enter", universes.save("  ", ["ola", "jan"])[1])
        u2, _ = universes.save("Big family", ["ola", "kuba", "jan"], slug="family")
        self.assertEqual((u2["slug"], u2["name"], len(universes.all_universes())), ("family", "Big family", 1))
        self.assertEqual(universes.delete("family")[0], True)
        self.assertEqual(universes.all_universes(), [])
        self.assertEqual(universes.delete("family"), (False, "Universe not found."))

    def test_a_universe_name_that_combines_a_minor_with_sexual_content_is_refused(self):
        with patch.object(universes.llm, "blocked", return_value=True):
            self.assertEqual(universes.check_name("Family"), universes.llm.BLOCK_MSG)
            self.assertEqual(universes.save("Family", ["ola", "kuba"]), (None, universes.llm.BLOCK_MSG))
        self.assertEqual(universes.all_universes(), [])

    def test_a_universe_has_at_most_max_cast_members(self):
        with patch.object(cast, "MAX_CAST", 3):
            self.assertIn("at most 3", universes.save("Crowd", ["ola", "kuba", "jan", "virtual_nova"])[1])
            self.assertIsNotNone(universes.save("Trio", ["ola", "kuba", "jan"])[0])

    def test_a_character_that_is_not_usable_does_not_filter_the_pages(self):
        cast.select("ola")
        self.assertEqual(cast.profiles(), ["ola"])
        (self.tmp / "characters" / "ola" / "photos" / "01.png").unlink()  # its only photo is gone
        self.assertEqual(cast.profiles(), [])  # everything, as the top picker and page headers say

    def test_deleted_character_leaves_every_universe(self):
        universes.save("Family", ["ola", "kuba", "jan"])
        universes.forget("kuba")
        self.assertEqual(universes.get("family")["members"], ["ola", "jan"])

    def test_a_universe_in_the_top_picker_selects_its_main_character_and_fills_the_picker(self):
        universes.save("Family", ["kuba", "ola"])
        self.assertIn(("Universe: Family (2)", "universe:family"), cast.top_picker_choices())
        self.assertEqual(cast.select("universe:family"), "kuba")
        self.assertEqual((prefs.load()["character"], prefs.load()["universe"]), ("kuba", "family"))
        self.assertEqual(cast.top_picker_value(), "universe:family")
        self.assertEqual(cast.picked(), ["kuba", "ola"])
        self.assertEqual(cast.profiles(), ["kuba", "ola"])
        self.assertEqual(cast.picked("ola"), ["ola"])
        self.assertEqual(cast.picked("universe:family"), ["kuba", "ola"])
        cast.select("ola")
        self.assertEqual((prefs.load()["character"], prefs.load()["universe"]), ("ola", ""))
        self.assertEqual((cast.top_picker_value(), cast.picked(), cast.profiles()), ("ola", ["ola"], ["ola"]))

    def test_universe_main_character_follows_deletions(self):
        universes.save("Family", ["kuba", "ola"])
        cast.select("universe:family")
        universes.forget("kuba")
        self.assertEqual(cast.top_picker_value(), "universe:family")
        self.assertEqual(prefs.load()["character"], "ola")


class ComposeTests(_Library):
    def test_a_failed_or_stopped_compose_leaves_no_files(self):
        from app import cancel

        def stop(*_args, **_options):
            raise cancel.Cancelled()

        members = [characters.get("ola"), characters.get("kuba")]
        with patch("app.refs.compose_refs", return_value=(False, "engine down")):
            result = cast.compose_frame(None, members, "Ola and Kuba", 640, 360, self.tmp / "frame.png", 1)
        self.assertEqual(result, (False, "engine down"))
        with patch("app.refs.compose_refs", side_effect=stop), self.assertRaises(cancel.Cancelled):
            cast.compose_frame(None, members, "Ola and Kuba", 640, 360, self.tmp / "frame.png", 1)
        self.assertEqual(list(self.tmp.glob("frame*")), [])
        ola = dict(characters.get("ola"), photos=[])  # its photo went missing meanwhile
        ok, msg = cast.compose_frame(None, [ola, members[1]], "Ola and Kuba", 640, 360, self.tmp / "f.png", 1)
        self.assertEqual((ok, msg), (False, "Ola: this character has no reference photo"))

    def test_compose_frame_composes_every_characters_primary_photo_with_the_references(self):
        calls = []

        def compose(client, base, ref_list, text, w, h, out, seed=None, **kw):
            calls.append((base, ref_list, text, kw["pass_leads"], sorted(kw["looks"]), (w, h), seed))
            Path(out).write_bytes(b"scene")
            return True, str(out)

        out = self.tmp / "frame.png"
        ola, kuba = characters.get("ola"), characters.get("kuba")
        with patch("app.refs.compose_refs", side_effect=compose), patch.object(cast.characters, "render") as render:
            ok, path = cast.compose_frame(None, [ola, kuba], "Ola hands Kuba the hoodie [1] on the beach",
                                          1280, 720, out, 10, extra_refs=[(1, "hoodie.png")], looks={1: ""})
        render.assert_not_called()  # nothing is drawn alone first: the photos are the references
        self.assertEqual((ok, path), (True, str(out)))
        base, ref_list, text, leads, looks, size, seed = calls[0]
        self.assertEqual((base, size, seed), (None, (1280, 720), 10))
        self.assertEqual(ref_list, [(1, ola["photos"][0]), (2, kuba["photos"][0]), (3, "hoodie.png")])
        self.assertEqual(text, "The woman [1] hands the man [2] the hoodie [3] on the beach")
        self.assertTrue(leads[0].startswith("These are different people: image 1 shows Ola"))
        self.assertEqual(looks, [1, 2, 3])
        self.assertEqual(Path(path).read_bytes(), b"scene")

    def test_more_than_three_characters_get_one_lead_per_pass(self):
        five = [characters.get(s) for s in ("ola", "kuba", "jan", "virtual_nova")]
        captured = {}

        def compose(client, base, ref_list, text, w, h, out, seed=None, **kw):
            captured["leads"] = kw["pass_leads"]
            Path(out).write_bytes(b"scene")
            return True, str(out)

        with patch("app.refs.compose_refs", side_effect=compose):
            ok, path = cast.compose_frame(None, five, "Ola, Kuba, Jan and Virtual Nova", 640, 360,
                                          self.tmp / "f.png", 1)
        self.assertTrue(ok)
        self.assertEqual(len(captured["leads"]), 2)
        self.assertIn("image 3 shows Jan", captured["leads"][0])
        self.assertIn("image 2 shows Virtual Nova", captured["leads"][1])
        self.assertIn("Add them to the picture from image 1", captured["leads"][1])
        self.assertEqual(Path(path).read_bytes(), b"scene")


class DialogueCastTests(_Library):
    def test_character_names_pick_the_speaker_and_pronouns_follow_them(self):
        people = cast.people([characters.get("ola"), characters.get("kuba")])
        _, _, lines = dialogue.parse("Ola looks at Kuba. He says „Hi”. She answers „Hey”.", {}, people)
        self.assertEqual([(l.speaker, l.character, l.who) for l in lines],
                         [("male", "kuba", "male#char:kuba"), ("female", "ola", "female#char:ola")])
        visual, _, lines = dialogue.parse("@kuba says „Yo”.", {"kuba": ("kuba_voice", "male")}, people)
        self.assertEqual((lines[0].character, lines[0].voice), ("kuba", ""))  # character wins
        self.assertIn("Kuba talks", visual)

    def test_two_women_without_voices_get_different_voices(self):
        people = cast.people([characters.get("ola"), characters.get("virtual_nova")])
        _, _, lines = dialogue.parse("Ola says „Hi”. Virtual Nova answers „Hey”. Ola adds „Bye”.",
                                     {}, people)
        catalog = {"female": ["hanna", "zoe", "maja"], "male": ["adam"]}
        self.assertEqual(dialogue.voices_for(lines, catalog), ["hanna", "zoe", "hanna"])

    def test_a_character_with_its_own_voice_keeps_it(self):
        data = {**characters.get("ola"), "voice": "maja"}
        with patch("app.voices.get", side_effect=lambda k, pool=None:
                   type("V", (), {"slug": k})() if k == "maja" else None):
            people = cast.people([data, characters.get("virtual_nova")])
        _, _, lines = dialogue.parse("Virtual Nova says „Hi”. Ola answers „Hey”.", {}, people)
        catalog = {"female": ["hanna", "zoe", "maja"], "male": ["adam"]}
        self.assertEqual(dialogue.voices_for(lines, catalog), ["hanna", "maja"])

    def test_a_characters_own_voice_is_kept_although_the_library_does_not_list_it(self):
        store_character(self.tmp / "characters", OLA, voice=b"RIFF-own-voice")
        people = cast.people([characters.get("ola"), characters.get("virtual_nova")])
        self.assertEqual(people[0][3], "character:ola")
        _, _, lines = dialogue.parse("Virtual Nova says „Hi”. Ola answers „Hey”.", {}, people)
        catalog = {"female": ["hanna", "zoe"], "male": ["adam"]}
        self.assertEqual(dialogue.voices_for(lines, catalog), ["hanna", "character:ola"])

    def _characters(self, slugs, prompt):
        people = cast.people([characters.get(s) for s in slugs])
        return [l.character for l in dialogue.parse(prompt, {}, people)[2]]

    def test_the_speaker_is_named_before_the_speaking_verb(self):
        for prompt in ("Ola says to Kuba „Hi”", "Ola hands Kuba a coffee and says „Here you go”",
                       "Ola podaje Kubie kawę i mówi „Proszę”", "Ola mówi do Kuby „Cześć”",
                       "Ola walks in. She hugs Kuba and says „I missed you”",
                       "Ola asks Kuba „Are you ok?”", "@ola says to @kuba „Hi”",
                       "Kuba listens while Ola says „Hi”", "Kuba enters. Ola smiles at him and says „Hi”",
                       "Ola looks at Kuba. Her voice trembles: „Stay”"):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._characters(["ola", "kuba"], prompt), ["ola"])
        self.assertEqual(self._characters(["ola", "kuba"], "Ola says „Hi” to Kuba and adds „Nice”"),
                         ["ola", "ola"])
        self.assertEqual(self._characters(["ola", "kuba"], "Ola looks at Kuba and he says „Hi”"), ["kuba"])
        self.assertEqual(self._characters(["ola", "kuba"], "Ola przytula Kubę. On mówi „Cześć”."), ["kuba"])
        self.assertEqual(self._characters(["ola", "kuba"], "Ola and Kuba sit in a cafe. He takes her hand "
                                          "and says „I love you”. She smiles at him and answers „Me too”."),
                         ["kuba", "ola"])
        self.assertEqual(self._characters(["ola", "kuba"], "Ola and Kuba at a bar. A waiter asks „Hi?”"), [""])

    def test_tags_win_only_in_the_line_s_own_sentence(self):
        self.assertEqual(self._characters(["kuba", "ola"], "@kuba hugs @ola. He says „I missed you”."), ["kuba"])
        self.assertEqual(self._characters(["kuba", "ola"], "@ola hugs @kuba. She says „I missed you”."), ["ola"])
        self.assertEqual(self._characters(["kuba", "ola"], "@kuba hugs @ola. Kuba says „I missed you”."), ["kuba"])
        self.assertEqual(self._characters(["kuba", "ola"], "@kuba hugs Ola and says „I missed you”."), ["kuba"])
        people = cast.people([characters.get("ola")])
        lines = dialogue.parse("@ola waves at a man. The man says „Hi”.", {}, people)[2]
        self.assertEqual((lines[0].speaker, lines[0].character, lines[0].who), ("male", "", "male#0"))

    def test_one_character_s_name_does_not_take_another_person_s_line(self):
        for prompt in ("A woman hugs Kuba and says „I missed you”", "Kobieta przytula Kubę i mówi „Tęskniłam”"):
            with self.subTest(prompt=prompt):
                lines = dialogue.parse(prompt, {}, cast.people([characters.get("kuba")]))[2]
                self.assertEqual(lines[0].who, "female#0")

    def test_quoted_lines_never_name_a_pronoun_s_person(self):
        self.assertEqual(self._characters(["kuba", "ola", "jan"], "Kuba hugs Ola. She says „I missed you, "
                                          "Jan”. He answers „Me too”."), ["ola", "kuba"])

    def test_the_other_woman_is_the_second_named_character(self):
        self.assertEqual(self._characters(["ola", "virtual_nova"], "Ola and Virtual Nova sit in a cafe. Ola "
                                          "says „Hi”. The other woman answers „Hey”. Ola adds „Bye”."),
                         ["ola", "virtual_nova", "ola"])
        self.assertEqual(self._characters(["ola", "virtual_nova"], "Ola and Virtual Nova chat. Ola says „Hi” "
                                          "and adds „Bye”."), ["ola", "ola"])

    def test_a_character_and_an_unnamed_woman_get_different_voices(self):
        catalog = {"female": ["hanna", "zoe", "maja"], "male": ["adam"]}
        prompt = "Two women in a cafe. Ola says „Hi”. The other one says „Hey”."
        _, _, lines = dialogue.parse(prompt, {}, cast.people([characters.get("ola")]))
        self.assertEqual([l.who for l in lines], ["female#char:ola", "female#0"])
        self.assertEqual(dialogue.voices_for(lines, catalog), ["hanna", "zoe"])
        data = {**characters.get("ola"), "voice": "maja"}
        with patch("app.voices.get", side_effect=lambda k, pool=None:
                   type("V", (), {"slug": k})() if k == "maja" else None):
            people = cast.people([data])
        _, _, lines = dialogue.parse(prompt, {}, people)
        self.assertEqual(dialogue.voices_for(lines, catalog, ("maja", "female")), ["maja", "hanna"])


if __name__ == "__main__":
    unittest.main()
