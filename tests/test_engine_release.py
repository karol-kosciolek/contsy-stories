# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Release regressions at direct generation and archive entry points."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from app import characters, config, history, image_gen, llm, model_manager
from app import refs, tts, video_gen


class GenerationGuards(unittest.TestCase):
    def test_minors_content_is_rejected_at_direct_entry_points_before_download(self):
        prompt = "a 15-year-old in a bikini"
        with patch.object(model_manager, "ensure") as ensure:
            for output in (image_gen.txt2img(prompt, 512, 512, 8, 1),
                           image_gen.edit_with_refs([(1, "reference.png")], prompt, 1.0),
                           video_gen.generate(prompt, 5, "Portrait 9:16")):
                self.assertEqual((output.ok, output.error), (False, llm.BLOCK_MSG))
            self.assertEqual(refs.compose(MagicMock(), ["source.png"], prompt, 512, 512, "unused.png"),
                             (False, llm.BLOCK_MSG))
            with tempfile.TemporaryDirectory() as tmp, patch.object(characters, "DRAFTS_DIR", Path(tmp)):
                draft = characters.new_draft()
                characters.update_draft(draft, look="a 15-year-old girl with long hair")
                with self.assertRaisesRegex(ValueError, "Only adult characters are supported"):
                    characters.enqueue_photos(draft, 4)
                ok, _, _, error = characters.run_photo_job({"draft": draft, "count": 4}, lambda *_: None)
                self.assertFalse(ok)
                self.assertIn("Only adult characters are supported", error)
        ensure.assert_not_called()

    def test_an_image_is_saved_and_recorded_without_the_assistant(self):
        """No check stands between the engine and the Gallery: the picture is saved and recorded as it
        is, and the assistant (its runtime may even be missing) is never asked for."""
        data = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(data, format="PNG")
        unused = AssertionError("An image needs no assistant")
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
                patch.object(llm, "ensure", side_effect=unused), \
                patch.object(llm._OWN_LLM, "ask", side_effect=unused), \
                patch.object(image_gen, "available_models", return_value=["z.safetensors"]), \
                patch.object(image_gen.engine_runtime, "render", return_value=data.getvalue()), \
                patch.object(history, "record") as record:
            out = image_gen.txt2img("a topless adult woman on a beach", 512, 512, 8, 7)
            self.assertTrue(out.ok, out.error)
            self.assertEqual(sorted(str(file) for file in Path(tmp).iterdir()), out.files)
            with Image.open(out.files[0]) as saved:
                self.assertEqual(saved.size, (8, 8))
        entry = record.call_args.args[0]
        self.assertEqual((entry["files"], entry["seed"]), (out.files, 7))
        for field in ("qc", "level"):
            self.assertNotIn(field, entry)

    def test_a_composition_without_an_output_cannot_publish(self):
        client = MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
                patch.object(model_manager, "ensure", return_value=None), \
                patch.object(image_gen.engine_runtime, "engine", return_value=client), \
                patch.object(llm, "describe_person", return_value=""), \
                patch.object(refs, "compose_refs", return_value=(True, "draft.png")), \
                patch.object(history, "record") as record:
            out = image_gen.edit_with_refs([(1, "reference.png")], "A city scene", 1.0)
            left = list(Path(tmp).iterdir())
        self.assertFalse(out.ok)
        self.assertEqual(out.error, "The composition returned no image.")
        self.assertEqual((out.files, left), ([], []))
        record.assert_not_called()

    def test_a_composition_keeps_the_size_the_edit_model_draws(self):
        """Reference images are composed at about 1 megapixel in the chosen proportions and the picture
        is kept as composed: no other model is downloaded and nothing enlarges it."""
        sizes = []

        def compose(_client, _base, _refs, _text, width, height, out_path, *args, **kwargs):
            sizes.append((width, height))
            Image.new("RGB", (width, height), "gray").save(out_path)
            return True, str(out_path)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "OUTPUTS_DIR", Path(tmp)), \
                patch.object(model_manager, "ensure", return_value=None) as ensure, \
                patch.object(image_gen.engine_runtime, "engine", return_value=MagicMock()), \
                patch.object(llm, "describe_person", return_value=""), \
                patch.object(refs, "compose_refs", side_effect=compose), \
                patch.object(history, "record") as record:
            out = image_gen.edit_with_refs([(1, "reference.png")], "A city scene", 16 / 9)
            self.assertTrue(out.ok, out.error)
            self.assertEqual([path.name for path in Path(tmp).iterdir()], [Path(out.files[0]).name])
            with Image.open(out.files[0]) as saved:
                self.assertEqual(saved.size, (1360, 768))
        self.assertEqual(sizes, [(1360, 768)])
        self.assertEqual(ensure.call_args.args[0], [image_gen.engine_runtime.QWEN_EDIT_ID])
        entry = record.call_args.args[0]
        self.assertEqual((entry["model"], entry["width"], entry["height"]), ("Qwen-Image-Edit-2511", 1360, 768))

    def test_reference_composition_rejects_excess_images_instead_of_truncating(self):
        client = MagicMock()
        ok, error = refs.compose(client, ["one"] * 4, "landscape", 512, 512, "unused.png")
        self.assertFalse(ok)
        self.assertIn("at most 3", error)
        client.upload_image.assert_not_called()

class ModelAndLanguage(unittest.TestCase):
    def test_missing_assistant_runtime_is_reported_before_model_download(self):
        with (patch.object(llm, "runtime_problem", return_value="Install the assistant runtime."),
              patch.object(model_manager, "ensure") as install):
            self.assertEqual(llm.ensure(), "Install the assistant runtime.")
        install.assert_not_called()

    def test_a_review_asks_again_at_the_next_size_and_reports_an_outage(self):
        import json
        schema = {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]}
        frames = ["before.png", "after.png"]

        def valid(answer):
            return answer.get("verdict") in ("match", "mismatch")

        answers = ["{", json.dumps({"verdict": "match"})]
        with patch.object(llm, "inspect_frames", side_effect=answers) as inspect:
            self.assertEqual(llm.inspect_review(frames, "Compare.", 40, schema, valid, (512, 768)),
                             {"verdict": "match"})
        self.assertEqual([call.args[0] for call in inspect.call_args_list], [frames, frames])
        self.assertEqual([call.kwargs["size"] for call in inspect.call_args_list], [512, 768])
        for answer in (None, "", "[]", json.dumps({"verdict": "perhaps"})):
            with self.subTest(answer=answer), patch.object(llm, "inspect_frames", return_value=answer), \
                    self.assertRaisesRegex(llm.VisualReviewUnavailable, "after 2 attempts"):
                llm.inspect_review(frames, "Compare.", 40, schema, valid, (512, 768))

    def test_a_queued_review_keeps_the_vision_model_only_until_it_ends(self):
        import json
        from app import queue_manager
        schema = {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]}
        with (patch.object(queue_manager, "in_worker", return_value=True),
              patch.object(llm, "ensure", return_value=None),
              patch.object(llm, "_catalog_file", return_value="vision.gguf"),
              patch.object(llm, "_OWN_LLM") as client):
            client.ask.side_effect = ["{", json.dumps({"verdict": "match"}), "A look."]
            verdict = llm.inspect_review(["before.png", "after.png"], "Compare.", 40, schema,
                                         lambda answer: "verdict" in answer, (512, 768))
            self.assertEqual(verdict, {"verdict": "match"})
            # Both attempts reuse the loaded model; the GPU goes back to the queue when the review ends.
            self.assertEqual([call.kwargs["keep_loaded"] for call in client.ask.call_args_list], [True, True])
            client.unload_for_gpu_handover.assert_called_once_with()
            llm._vision("still.png", "Describe it.", 20)
            self.assertFalse(client.ask.call_args.kwargs["keep_loaded"])

    def test_first_vision_request_sees_projector_installed_by_ensure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            photo = folder / "photo.png"
            Image.new("RGB", (8, 8), "red").save(photo)
            bundle = {"files": [{"filename": n, "dest": "llm"}
                                 for n in ("aligned.gguf", "aligned-mmproj.gguf")]}

            def install():
                for name in ("aligned.gguf", "aligned-mmproj.gguf"):
                    (folder / name).touch()

            with patch.object(llm, "ensure", side_effect=install), \
                    patch.object(config, "get_model", return_value=bundle), \
                    patch.object(config, "destination_dir", return_value=folder), \
                    patch.object(llm._OWN_LLM, "ask", return_value="fresh vision response") as ask:
                self.assertEqual(llm._vision(str(photo), "Describe it", 10), "fresh vision response")
                self.assertEqual(ask.call_args.kwargs["mmproj_path"], folder / "aligned-mmproj.gguf")

    def test_reference_description_is_cached_by_image_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp) / "photo.png"
            Image.new("RGB", (8, 8), "red").save(photo)
            with (patch.object(llm, "_LOOKS_FILE", Path(tmp) / "looks.json"),
                  patch.object(llm, "_vision", return_value="Brown hair and dark eyes.") as vision):
                self.assertEqual(llm.describe_person(str(photo)), "Brown hair and dark eyes")
                self.assertEqual(llm.describe_person(str(photo)), "Brown hair and dark eyes")
            vision.assert_called_once()

    def test_reference_photo_check_requires_strict_model_response_types(self):
        import json
        valid = {"person": True, "age": 25, "gender": "woman", "caption": "Wearing a coat."}
        for field, value in (("person", "false"), ("age", True), ("age", -1),
                             ("caption", []), ("caption", ""), ("gender", "unknown")):
            with self.subTest(field=field, value=value), \
                    patch.object(llm, "_vision", return_value=json.dumps({**valid, field: value})):
                self.assertIsNone(llm.check_photo("unused.png"))
        with patch.object(llm, "_vision", return_value=json.dumps({**valid, "age": 16})):
            self.assertEqual(llm.check_photo("unused.png")["age"], 16)

    def test_empty_speech_batch_is_rejected_before_loading_model(self):
        with patch.object(tts, "available") as available:
            self.assertFalse(tts.synthesize_many([])[0])
        available.assert_not_called()

    def test_english_reference_ordinals_are_normalized_without_changing_speech(self):
        self.assertEqual(llm.mark_refs("the woman from the second photo holds image 1", 2),
                         "the woman [2] holds [1]")

    def test_catalog_selection_never_chooses_old_custom_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("old-abliterated.gguf", "aligned.gguf", "aligned-mmproj.gguf"):
                (Path(tmp) / name).write_bytes(b"test")
            bundle = {"files": [{"filename": n, "dest": "llm"}
                                 for n in ("aligned.gguf", "aligned-mmproj.gguf")]}
            with patch.object(config, "get_model", return_value=bundle), \
                    patch.object(config, "destination_dir", return_value=Path(tmp)):
                self.assertEqual(llm._model_file(), "aligned.gguf")
                self.assertEqual(llm._catalog_file(True), "aligned-mmproj.gguf")

    def test_speech_language_is_preserved_by_prompt_instruction(self):
        prompt = 'A woman says „Dzień dobry, świecie!” in a cafe.'
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(llm, "ask", return_value=(True, prompt)) as ask:
            ok, text = llm.write_prompt('She says „Dzień dobry, świecie!”', "video")
        self.assertTrue(ok)
        self.assertIn("Dzień dobry, świecie!", text)
        # a line the user quoted stays word for word; any other spoken line is English (the films are in English)
        self.assertIn("only a line the user wrote in quotes is kept word for word", ask.call_args.args[0])
        self.assertTrue(ask.call_args.kwargs["check"](prompt))
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(llm, "ask", return_value=(True, "A woman says „Cześć”.")) as ask:
            llm.write_prompt("kobieta mówi cześć", "video")
        self.assertFalse(ask.call_args.kwargs["check"]("A woman says „Dzień dobry wszystkim, miłego dnia”."))
        self.assertTrue(ask.call_args.kwargs["check"]("A woman says „Good morning, everyone!”"))
        self.assertNotIn("in Polish", ask.call_args.args[1])
        self.assertTrue(llm.english(prompt))

    def test_the_assistant_keeps_only_the_minors_guard(self):
        """No adult-content rule is given to the model and its answer is not cleaned or refused for
        adult content; text that combines minors with sexual content stops before and after the model."""
        answer = "A topless adult woman lies on a beach at sunset, without revealing too much."
        with patch.object(llm, "ensure", return_value=None), \
                patch.object(llm, "ask", return_value=(True, answer)) as ask:
            self.assertEqual(llm.write_prompt("naga kobieta na plaży o zachodzie słońca", "image"), (True, answer))
            system = ask.call_args.args[0]
            self.assertIn("describe every person as an adult", system)
            self.assertNotIn("Content level", system)
            ask.reset_mock()
            self.assertEqual(llm.write_prompt("a 15-year-old in a bikini", "image"), (False, llm.BLOCK_MSG))
            ask.assert_not_called()
            ask.return_value = (True, "A 15-year-old in lingerie on a beach.")
            self.assertEqual(llm.write_prompt("a person on a beach", "image"), (False, llm.BLOCK_MSG))
            ask.reset_mock()
            ask.return_value = (True, "A 30-year-old woman walks along the beach at sunset, the camera follows her.")
            self.assertTrue(llm.write_prompt("kobieta idzie plażą", "video")[0])
            self.assertIn("describe every person as an adult", ask.call_args.args[0])

    def test_a_vision_answer_that_combines_a_minor_with_sexual_content_is_dropped(self):
        """A look or a reference-photo caption from the vision assistant reaches prompts; a blocked one is
        replaced by nothing, so it can never steer a picture."""
        with tempfile.TemporaryDirectory() as tmp, patch.object(llm, "_LOOKS_FILE", Path(tmp) / "looks.json"):
            picture = Path(tmp) / "person.png"
            Image.new("RGB", (8, 8), "red").save(picture)
            with patch.object(llm, "_vision", return_value="A 15-year-old girl in a bikini"):
                self.assertEqual(llm.describe_person(str(picture)), "")
            answer = ('{"person": true, "age": 25, "gender": "woman", '
                      '"caption": "a 15-year-old in lingerie on a bed"}')
            with patch.object(llm, "_vision", return_value=answer):
                self.assertEqual(llm.check_photo(str(picture)),
                                 {"person": True, "age": 25, "gender": "woman", "caption": ""})
