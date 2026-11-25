# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Contsy Stories (docs/CONTSY_STORIES_PRODUCT.md): series, seasons and episodes, the story memory, character sheets,
the storyboard and its continuity warnings, the series-bound agent tools, the episode run and the conversation after
it. Everything in temporary folders with a scripted model: no GPU, no network, no API cost."""
import json
from unittest.mock import patch

from app import agent_runner, agent_store, characters, series, styles, timeline
from tests.agent_fake import tool_turn
from tests.test_agent_runner import FINISH, RunnerCase


class SeriesCase(RunnerCase):
    def setUp(self):
        super().setUp()
        self.context.enter_context(patch.object(series, "DIR", self.root / "series"))

    def make(self, **kw):
        data = series.create("Pip and the Lantern Forest", look=kw.get("look", "cartoon_2d"),
                             logline="A curious fox solves small mysteries.")
        pip = series.save_character(data["id"], {"name": "Pip", "kind": "creature", "description": "a small fox",
                                                 "colors": "orange fur, white chest", "outfit": "green scarf",
                                                 "marks": "round glasses", "voice": "hanna"})
        return data, pip


class StructureTests(SeriesCase):
    def test_series_seasons_and_episode_numbers(self):
        data, _ = self.make()
        one = series.create_episode(data["id"], "Pip finds a glowing lantern.")
        two = series.create_episode(data["id"], "Pip returns the lantern to its owner.", seconds=600, fmt="9:16")
        three = series.create_episode(data["id"], "A new adventure in winter.", season=2)
        self.assertEqual([(e["season"], e["number"]) for e in series.episodes(data["id"])], [(1, 1), (1, 2), (2, 1)])
        self.assertEqual((one["format"], two["format"], three["season"]), ("16:9", "9:16", 2))
        self.assertEqual({s["number"] for s in series.get(data["id"])["seasons"]}, {1, 2})
        self.assertEqual(sum(series.chapters(600)), 600)
        self.assertLessEqual(max(series.chapters(600)), series.CHAPTER_SECONDS)
        with self.assertRaises(series.SeriesError):
            series.create_episode(data["id"], "too long", seconds=5000)

    def test_only_animated_styles_and_the_engine_ids(self):
        for row in series.looks():
            style = styles.get(series.engine_style(row["key"]))
            self.assertNotEqual(style.family, styles.REALISTIC, row)
        self.assertEqual(series.engine_style("cartoon_2d"), styles.CARTOON)
        self.assertTrue(series.engine_style("clay").startswith(styles.CUSTOM_PREFIX))

    def test_character_sheets(self):
        data, pip = self.make()
        self.assertTrue(pip["id"].startswith("c"))
        with self.assertRaises(series.SeriesError):
            series.save_character(data["id"], {"name": "pip"})  # a name is unique in its series
        with self.assertRaises(series.SeriesError):
            series.save_character(data["id"], {"name": "Kid", "kind": "person", "description": "a 7 year old boy"})
        series.set_outfits(data["id"], pip["id"], ["winter coat", "", "pajamas"])
        self.assertEqual(series.get(data["id"])["characters"][0]["outfits"], ["winter coat", "pajamas"])
        self.assertIn("green scarf", series.character_look(pip))
        # the sheet is linked to the library character of the same name once it exists
        self.saved_character(name="Pip", look="A small orange fox")
        linked = series.sync_links(series.get(data["id"]))
        self.assertEqual(linked["characters"][0]["slug"], "pip")

    def test_isolated_delete_keeps_a_copy(self):
        data, _ = self.make()
        series.delete(data["id"])
        self.assertIsNone(series.get(data["id"]))
        self.assertTrue(any((self.root / "series" / ".deleted").iterdir()))


class MemoryTests(SeriesCase):
    def test_story_context_has_the_bible_and_only_earlier_episodes(self):
        data, _ = self.make()
        one = series.create_episode(data["id"], "Pip finds a lantern.")
        two = series.create_episode(data["id"], "Pip meets Bolt the robot.")
        three = series.create_episode(data["id"], "Winter comes.")
        series.set_summary(data["id"], one["id"], "Pip lost the silver key in the river.", source="auto")
        series.set_summary(data["id"], two["id"], "Pip and Bolt became friends.", source="auto")
        fact = series.add_fact(data["id"], "Pip has no silver key any more.", episode=one["id"], source="auto")
        old = series.add_fact(data["id"], "Bolt is afraid of water.")
        series.edit_fact(data["id"], old["id"], active=False)
        context = series.story_context(data["id"], three["id"])
        self.assertIn("lost the silver key", context)
        self.assertIn("became friends", context)
        self.assertIn(fact["text"], context)
        self.assertNotIn("afraid of water", context)  # a retired fact no longer reaches the agent
        self.assertIn("green scarf", context)
        self.assertNotIn("became friends", series.story_context(data["id"], two["id"]))  # not its own future

    def test_the_owners_correction_wins(self):
        data, _ = self.make()
        one = series.create_episode(data["id"], "Pip finds a lantern.")
        series.set_summary(data["id"], one["id"], "Wrong summary from the agent.", source="auto")
        series.set_summary(data["id"], one["id"], "Pip found the lantern in the old mill.")
        series.set_summary(data["id"], one["id"], "Another automatic summary.", source="auto")
        self.assertEqual(series.get_episode(data["id"], one["id"])["summary"], "Pip found the lantern in the old mill.")
        fact = series.add_fact(data["id"], "Pip has a lantern.", source="auto")
        series.edit_fact(data["id"], fact["id"], text="Pip has a blue lantern.")
        self.assertEqual(series.get(data["id"])["facts"][0]["source"], "owner")
        series.update(data["id"], story_state="Pip is at home.")
        self.assertEqual(series.get(data["id"])["story_state_source"], "owner")

    def test_long_histories_fit_the_context(self):
        data, _ = self.make()
        for number in range(40):
            episode = series.create_episode(data["id"], f"Adventure number {number}.")
            series.set_summary(data["id"], episode["id"], f"Episode {number}: " + "something happened. " * 100,
                               source="auto")
        context = series.story_context(data["id"])
        self.assertLessEqual(len(context), series.CONTEXT_CHARS + 500)
        self.assertIn("Episode 39", context)  # the latest one is kept in full


class StoryboardTests(SeriesCase):
    def project(self, data, episode, cast=(), style=None):
        scenes = (timeline.Scene(prompt="Pip walks into the forest.", seconds=5, cast=tuple(cast)),
                  timeline.Scene(prompt="Pip opens the lantern.", seconds=5, cast=tuple(cast)))
        project = timeline.save(None, None, "Episode 1", scenes, "Landscape 16:9", True, list(cast),
                                style=style or series.engine_style(data["look"]))
        series.update_episode(data["id"], episode["id"], projects=[project.id])
        return project

    def test_scenes_and_takes_come_from_the_timeline(self):
        data, _ = self.make()
        episode = series.create_episode(data["id"], "Pip finds a lantern.", seconds=15)
        self.project(data, episode)
        scenes = series.episode_scenes(series.get_episode(data["id"], episode["id"]))
        self.assertEqual([s["index"] for s in scenes], [0, 1])
        self.assertFalse(any(s["has_clip"] for s in scenes))

    def test_warnings_name_extra_characters_style_and_missing_shots(self):
        data, _ = self.make()
        self.saved_character(name="Mia")
        episode = series.create_episode(data["id"], "Pip finds a lantern.", seconds=15)
        self.project(data, episode, cast=["mia"], style=styles.ANIMATION_3D)
        found = "\n".join(series.warnings(data["id"], episode["id"]))
        self.assertIn("Mia is not part of the series cast", found)
        self.assertIn("not the series style", found)
        self.assertIn("shot not made yet", found)
        self.assertIn("Pip is not created yet", found)


class AgentTests(SeriesCase):
    def bound_host(self, data, episode, **settings):
        host = self.host(**settings)
        host.store.update(series={"series": data["id"], "episode": episode["id"]})
        return host

    def test_memory_tool_and_the_series_guards(self):
        data, _ = self.make()
        episode = series.create_episode(data["id"], "Pip finds a lantern.")
        host = self.bound_host(data, episode, **self.allow(write=True))
        self.ok(host, "save_episode_memory", summary="Pip found a lantern and lost his scarf.",
                facts=["Pip lost his green scarf."], story_state="Pip is looking for his scarf.")
        saved = series.get(data["id"])
        self.assertEqual(series.get_episode(data["id"], episode["id"])["summary_source"], "auto")
        self.assertEqual(saved["facts"][0]["text"], "Pip lost his green scarf.")
        self.assertEqual(saved["story_state"], "Pip is looking for his scarf.")
        self.assertIn("lantern", self.ok(host, "get_series_bible")["bible"] + "lantern")
        self.refused(host, "queue_still", "INVALID", prompt="A fox", format="Square 1:1")
        # an existing series character is never designed again
        self.saved_character(name="Pip")
        series.sync_links(series.get(data["id"]))
        self.refused(host, "create_character", "INVALID", name="Pip", gender="man", look="A fox",
                     kind="creature", voice={"preset": "hanna"})
        # a series scene is a shot the studio compiles (docs/QUALITY_ARCHITECTURE.md), never a free prompt
        self.refused(host, "save_project", "INVALID", title="Ep 1", scenes=[{"prompt": "A forest at dawn.", "seconds": 4}])
        series.save_place(data["id"], "The Forest", "An old forest of tall pines and ferns.")
        shot = {"location": "The Forest", "time_of_day": "morning", "camera": "slow tracking",
                "action": "Pip walks from the left to the right between the pines", "key_visual": ["tall pines"]}
        # every project of the episode is drawn in the series style
        saved = self.ok(host, "save_project", title="Ep 1", style=styles.REALISTIC,
                        scenes=[{"shot": shot, "seconds": 4}])
        project = timeline.load(saved["id"])
        self.assertEqual(project.style, styles.CARTOON)
        self.assertEqual(project.scenes[0].cast, ("pip",))
        self.assertTrue(project.scenes[0].prompt.startswith("The Forest: An old forest"))

    def test_an_unbound_run_has_no_series(self):
        host = self.host()
        self.refused(host, "get_series_bible", "NOT_FOUND")

    def test_episode_run_and_the_conversation_after_it(self):
        data, _ = self.make()
        episode = series.create_episode(data["id"], "Pip finds a lantern.")
        memory = ("save_episode_memory", {"summary": "Pip found a lantern in the old mill."})
        self.provider.script = [tool_turn(memory), tool_turn(FINISH)]
        run_id = agent_runner.start_episode(data["id"], episode["id"])
        agent_runner.wait(run_id)
        record = agent_store.RunStore(run_id).read()
        self.assertEqual(record["state"], "completed")
        self.assertEqual(record["series"], {"series": data["id"], "episode": episode["id"]})
        first = json.dumps(agent_store.RunStore(run_id).wire_messages()[0])
        self.assertIn("green scarf", first)  # the bible went into the brief
        refreshed = series.refresh_episode(data["id"], episode["id"])
        self.assertEqual(refreshed["status"], "storyboard")
        self.assertEqual(refreshed["summary"], "Pip found a lantern in the old mill.")
        # the owner asks for a change after the run ended: the same transcript goes on
        self.provider.script = [tool_turn(FINISH)]
        agent_runner.follow_up(run_id, "Change the dialogue in scene 4 and make the fox look happier.")
        agent_runner.wait(run_id)
        messages = agent_store.RunStore(run_id).wire_messages()
        self.assertTrue(any("make the fox look happier" in json.dumps(m) for m in messages[-3:]))
        self.assertEqual(agent_store.RunStore(run_id).read()["state"], "completed")
        chat = series.get_episode(data["id"], episode["id"])["chat"]
        self.assertEqual([m["role"] for m in chat], ["owner", "owner"])
        with self.assertRaises(agent_runner.RunnerError):
            agent_runner.follow_up(run_id, "   ")
