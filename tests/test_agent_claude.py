# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Claude provider of the Agent page: the exact request it builds for each model, a streamed response read by the
real SDK over a mock transport (no network), the usage and cost of a response, the classification of errors, and the
features it turns off when the server refuses them."""
import json
import os
import unittest
from unittest.mock import patch

import httpx2

from app import agent_claude, agent_config, agent_provider as provider
from app.agent_provider import Features, ProviderError, TurnRequest
from tests import agent_fake as fake

KEY = "sk-" + "ant-" + "api03-" + "A1b2C3d4E5f6G7h8I9j0"
TOOLS = [{"name": "get_app_state", "description": "State.", "input_schema": {"type": "object", "properties": {}}}]


def request(model=fake.MODEL, **options) -> TurnRequest:
    return TurnRequest(model, "You run the studio.", TOOLS, [{"role": "user", "content": "Brief."}], **options)


class _Transport:
    """A mock transport: records each request and answers with the next scripted (status, headers, body)."""

    def __init__(self, *answers):
        self.answers, self.seen = list(answers), []

    def __call__(self, http_request: httpx2.Request) -> httpx2.Response:
        self.seen.append(http_request)
        status, headers, body = self.answers.pop(0)
        if isinstance(body, dict):
            return httpx2.Response(status, json=body, headers=headers, request=http_request)
        return httpx2.Response(status, content=body.encode("utf-8"), request=http_request,
                               headers={"content-type": "text/event-stream", **headers})

    def client(self):
        return httpx2.Client(transport=httpx2.MockTransport(self))

    def body(self, index=0) -> dict:
        return json.loads(self.seen[index].content)


def sdk_provider(*answers) -> tuple[agent_claude.ClaudeProvider, _Transport]:
    transport = _Transport(*answers)
    client = agent_claude.make_client(KEY, http_client=transport.client())
    return agent_claude.ClaudeProvider(key=KEY, client=client), transport


class RequestShapeTests(unittest.TestCase):
    def test_every_model_gets_the_same_safe_request_and_only_its_own_features(self):
        claude = agent_claude.ClaudeProvider(key=KEY, client=object())
        for model, spec in provider.MODELS.items():
            with self.subTest(model=model):
                params = claude.build_params(request(model, effort="high"))
                self.assertEqual(params["model"], model)
                self.assertNotRegex(params["model"], r"-\d{8}$")  # never a dated ID
                self.assertEqual(params["thinking"], {"type": "adaptive"})
                self.assertEqual(params["output_config"], {"effort": "high"})
                self.assertEqual(params["cache_control"], {"type": "ephemeral"})
                for name in ("tool_choice", "temperature", "top_p", "top_k", "budget_tokens"):
                    self.assertNotIn(name, params)
                edit = params["context_management"]["edits"][0]
                self.assertEqual((edit["type"], edit["trigger"]["type"]), ("compact_20260112", "input_tokens"))
                self.assertIn("compact-2026-01-12", params["betas"])
                if spec.fallbacks:
                    self.assertEqual(params["fallbacks"], "default")
                    self.assertIn("server-side-fallback-2026-07-01", params["betas"])
                else:  # the smallest model answers 400 to a fallbacks parameter
                    self.assertNotIn("fallbacks", params)
                    self.assertNotIn("server-side-fallback-2026-07-01", params["betas"])
                self.assertNotIn("thinking-binding-controls-2026-08-01", params["betas"])

    def test_the_cache_lifetime_and_the_compaction_text_follow_the_features(self):
        claude = agent_claude.ClaudeProvider(key=KEY, client=object())
        features = Features(compaction=True, compact_trigger=120_000, compact_instructions="Keep every id.",
                            fallbacks=False, cache_ttl="1h")
        params = claude.build_params(request(features=features))
        self.assertEqual(params["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        edit = params["context_management"]["edits"][0]
        self.assertEqual((edit["trigger"]["value"], edit["instructions"]), (120_000, "Keep every id."))
        self.assertNotIn("fallbacks", params)
        none = claude.build_params(request(features=Features(compaction=False, fallbacks=False)))
        self.assertNotIn("betas", none)
        self.assertNotIn("context_management", none)


class WireTests(unittest.TestCase):
    """The real SDK, a mock transport and a streamed answer."""

    def setUp(self):
        self.context = patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "env-token", "ANTHROPIC_BASE_URL": "https://evil.example",
                                               "ANTHROPIC_CUSTOM_HEADERS": "X-Leak: yes", "ANTHROPIC_PROFILE": "other"})
        self.context.start()
        self.addCleanup(self.context.stop)

    def test_a_streamed_answer_becomes_a_turn_and_the_request_leaves_nothing_to_the_environment(self):
        blocks = [{"type": "thinking", "thinking": "", "signature": "sig-abcdefghijklmnop"},
                  {"type": "text", "text": "I will look."},
                  {"type": "tool_use", "id": "toolu_01", "name": "get_app_state", "input": {"sections": ["queue"]}}]
        usage = {"input_tokens": 120, "output_tokens": 40, "cache_read_input_tokens": 3000,
                 "cache_creation_input_tokens": 500}
        claude, transport = sdk_provider((200, {}, fake.message_sse(blocks, stop_reason="tool_use", usage=usage)))
        events = []
        turn = claude.complete(request(effort="medium"), on_event=events.append)

        sent = transport.seen[0]
        self.assertEqual(sent.url.host, "api.anthropic.com")  # not the base URL of the environment
        self.assertEqual(sent.headers["authorization"], "Bearer " + KEY)
        self.assertNotIn("x-api-key", sent.headers)
        self.assertNotIn("x-leak", sent.headers)
        self.assertEqual(sorted(sent.headers["anthropic-beta"].split(",")),
                         ["compact-2026-01-12", "server-side-fallback-2026-07-01"])
        body = transport.body()
        self.assertTrue(body["stream"])
        self.assertEqual((body["model"], body["thinking"], body["output_config"]), (fake.MODEL, {"type": "adaptive"},
                                                                                    {"effort": "medium"}))
        self.assertEqual(body["messages"], [{"role": "user", "content": "Brief."}])

        self.assertEqual(turn.stop_reason, "tool_use")
        self.assertEqual(turn.text, "I will look.")
        self.assertEqual([(call.id, call.name, call.input) for call in turn.tool_calls],
                         [("toolu_01", "get_app_state", {"sections": ["queue"]})])
        # the blocks are the plain JSON, thinking included, ready to be sent back unchanged
        self.assertEqual([block["type"] for block in turn.content], ["thinking", "text", "tool_use"])
        self.assertEqual(turn.content[0], {"type": "thinking", "thinking": "", "signature": "sig-abcdefghijklmnop"})
        self.assertEqual(turn.context_tokens, 120 + 3000 + 500)
        self.assertIn("message_start", events)
        entry = turn.usage[0]
        self.assertEqual((entry.input_tokens, entry.output_tokens, entry.cache_read, entry.cache_write_5m),
                         (120, 40, 3000, 500))

    def test_blocks_survive_a_round_trip_through_a_second_request(self):
        first = [{"type": "thinking", "thinking": "", "signature": "sig-abcdefghijklmnop"},
                 {"type": "tool_use", "id": "toolu_01", "name": "get_app_state", "input": {"a": 1}}]
        claude, transport = sdk_provider((200, {}, fake.message_sse(first, stop_reason="tool_use")),
                                         (200, {}, fake.message_sse([{"type": "text", "text": "Done."}])))
        turn = claude.complete(request())
        messages = [{"role": "user", "content": "Brief."}, {"role": "assistant", "content": turn.content},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01",
                                                  "content": "{}"}]}]
        claude.complete(request().__class__(fake.MODEL, "S", TOOLS, messages))
        self.assertEqual(transport.body(1)["messages"][1]["content"], turn.content)

    def test_the_stop_flag_closes_the_stream(self):
        claude, _ = sdk_provider((200, {}, fake.message_sse([{"type": "text", "text": "Long answer"}])))
        with self.assertRaises(provider.Aborted):
            claude.complete(request(), should_abort=lambda: True)

    def test_a_server_that_refuses_a_beta_by_name_loses_only_that_feature(self):
        refusal = (400, {}, fake.error_body("invalid_request_error",
                                            "Unexpected value(s) `compact-2026-01-12` for the `anthropic-beta` header. "
                                            "Supported values are: files-api-2025-04-14."))
        claude, transport = sdk_provider(refusal, (200, {}, fake.message_sse([{"type": "text", "text": "ok"}])))
        turn = claude.complete(request())
        self.assertEqual(turn.text, "ok")
        self.assertEqual(claude.off, {"compaction"})
        self.assertEqual(transport.seen[1].headers["anthropic-beta"], "server-side-fallback-2026-07-01")
        self.assertNotIn("context_management", transport.body(1))

    def test_a_thinking_block_the_server_cannot_match_is_retried_once_with_drop_block(self):
        binding = (400, {}, fake.error_body("invalid_request_error",
                                            "Invalid signature in thinking block: bound to a different conversation "
                                            "(anthropic-beta header controls this)."))
        claude, transport = sdk_provider(binding, (200, {}, fake.message_sse([{"type": "text", "text": "ok"}])))
        claude.complete(request())
        self.assertTrue(claude.drop_blocks)
        second = transport.body(1)
        self.assertEqual(second["thinking"]["block_binding"], {"prefix_mismatch_behavior": "drop_block"})
        self.assertIn("thinking-binding-controls-2026-08-01", transport.seen[1].headers["anthropic-beta"])
        self.assertEqual(claude.off, set())  # the word "beta" in the message did not switch a feature off

    def test_an_error_after_the_retry_is_raised_classified(self):
        claude, _ = sdk_provider((401, {}, fake.error_body("authentication_error", "invalid x-api-key")))
        with self.assertRaises(ProviderError) as caught:
            claude.complete(request())
        self.assertEqual((caught.exception.kind, caught.exception.status), ("auth", 401))

    def test_ping_asks_for_the_model_without_sampling(self):
        claude, transport = sdk_provider((200, {}, {"id": "claude-opus-5-5", "type": "model",
                                                    "display_name": "Claude Opus 5.5",
                                                    "created_at": "2026-01-01T00:00:00Z"}))
        self.assertEqual(claude.ping("claude-opus-5-5"), "Claude Opus 5.5")
        self.assertEqual(transport.seen[0].method, "GET")
        self.assertTrue(str(transport.seen[0].url).endswith("/v1/models/claude-opus-5-5"))


class UsageTests(unittest.TestCase):
    def message(self, usage, model=fake.MODEL):
        from anthropic.types.beta import BetaMessage

        return BetaMessage.model_validate({"id": "msg_1", "type": "message", "role": "assistant", "model": model,
                                           "content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn",
                                           "stop_sequence": None, "usage": usage})

    def test_a_fallback_served_response_is_priced_per_iteration_and_by_its_own_model(self):
        usage = {"input_tokens": 10, "output_tokens": 10, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0,
                 "iterations": [
                     {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000, "output_tokens": 50,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
                     {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000,
                      "output_tokens": 500, "cache_read_input_tokens": 20000, "cache_creation_input_tokens": 3000,
                      "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000}},
                     {"type": "compaction", "input_tokens": 100, "output_tokens": 100,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}]}
        turn = agent_claude.to_turn(self.message(usage, model="claude-opus-4-8"), fake.MODEL)
        self.assertTrue(turn.served_by_fallback)
        self.assertEqual([(e.kind, e.model) for e in turn.usage],
                         [("message", "claude-opus-5-5"), ("fallback_message", "claude-opus-4-8"),
                          ("compaction", fake.MODEL)])
        self.assertEqual((turn.usage[1].cache_write_5m, turn.usage[1].cache_write_1h), (1000, 2000))
        opus = provider.MODELS["claude-opus-5-5"]
        worst = max(provider.MODELS.values(), key=lambda spec: spec.output)  # a model the table does not know
        expected = ((1000 * opus.input + 50 * opus.output) + (1000 * worst.input + 500 * worst.output
                    + 20000 * worst.cache_read + 1000 * worst.cache_write_5m + 2000 * worst.cache_write_1h)
                    + (100 * opus.input + 100 * opus.output)) / 1_000_000
        self.assertAlmostEqual(turn.cost(), expected, places=9)

    def test_a_response_without_iterations_uses_its_own_usage_and_cache_split(self):
        turn = agent_claude.to_turn(self.message({"input_tokens": 100, "output_tokens": 20,
                                                  "cache_read_input_tokens": 4000,
                                                  "cache_creation_input_tokens": 600}))
        self.assertEqual(len(turn.usage), 1)
        self.assertEqual((turn.usage[0].cache_read, turn.usage[0].cache_write_5m, turn.usage[0].cache_write_1h),
                         (4000, 600, 0))
        self.assertFalse(turn.served_by_fallback)

    def test_the_smallest_model_has_a_second_rate_card_above_a_hundred_thousand_prompt_tokens(self):
        low = provider.UsageEntry("claude-haiku-5-5", input_tokens=50_000, output_tokens=1000)
        high = provider.UsageEntry("claude-haiku-5-5", input_tokens=150_000, output_tokens=1000)
        spec = provider.MODELS["claude-haiku-5-5"]
        self.assertAlmostEqual(provider.cost([low]), (50_000 * spec.input + 1000 * spec.output) / 1e6)
        self.assertAlmostEqual(provider.cost([high]), (150_000 * spec.high_input + 1000 * spec.high_output) / 1e6)
        self.assertGreater(provider.cost([high]) / 150_000, provider.cost([low]) / 50_000)

    def test_the_cache_is_priced_by_kind_and_a_worst_case_covers_a_cut_off_request(self):
        spec = provider.MODELS[fake.MODEL]
        entry = provider.UsageEntry(fake.MODEL, cache_read=1_000_000, cache_write_5m=1_000_000,
                                    cache_write_1h=1_000_000)
        self.assertAlmostEqual(provider.cost([entry]), spec.cache_read + spec.cache_write_5m + spec.cache_write_1h)
        self.assertGreater(provider.worst_case(fake.MODEL, 200_000), 1.0)


class ErrorTests(unittest.TestCase):
    def raised(self, status, body, headers=None, cls=None):
        import anthropic

        http_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx2.Response(status, json=body, headers=headers or {}, request=http_request)
        factory = cls or anthropic.APIStatusError
        return factory(body["error"]["message"], response=response, body=body)

    def kind(self, status, error_type, message, headers=None, cls=None):
        exc = self.raised(status, fake.error_body(error_type, message), headers, cls)
        return agent_claude.map_error(exc)

    def test_status_errors_are_classified(self):
        import anthropic

        cases = [
            (401, "authentication_error", "invalid x-api-key", anthropic.AuthenticationError, "auth"),
            (403, "permission_error", "forbidden", anthropic.PermissionDeniedError, "auth"),
            (402, "billing_error", "billing problem", None, "billing"),
            (400, "invalid_request_error", "Your credit balance is too low to access the API", anthropic.BadRequestError,
             "billing"),
            (429, "rate_limit_error", "enforced_spend_limit_reached", anthropic.RateLimitError, "billing"),
            (429, "rate_limit_error", "slow down", anthropic.RateLimitError, "rate_limit"),
            (413, "request_too_large", "Request exceeds the maximum allowed size", None, "too_large"),
            (404, "not_found_error", "model: claude-x", anthropic.NotFoundError, "model"),
            (400, "invalid_request_error", "prompt is too long: 1100000 tokens", anthropic.BadRequestError, "context"),
            (400, "invalid_request_error", "Invalid signature: bound to a different conversation",
             anthropic.BadRequestError, "binding"),
            (400, "invalid_request_error", "Claude Fable 5.2 requires data retention", anthropic.BadRequestError,
             "model"),
            (400, "invalid_request_error", "Unexpected value(s) `compact-2026-01-12` for the `anthropic-beta` header.",
             anthropic.BadRequestError, "feature"),
            (400, "invalid_request_error", "messages: field required", anthropic.BadRequestError, "bad_request"),
            (529, "overloaded_error", "Overloaded", None, "overloaded"),
            (500, "api_error", "Internal server error", anthropic.InternalServerError, "server"),
        ]
        for status, etype, message, cls, expected in cases:
            with self.subTest(status=status, message=message[:40]):
                error = self.kind(status, etype, message, cls=cls)
                self.assertEqual(error.kind, expected)
        self.assertTrue(self.kind(429, "rate_limit_error", "slow", {"retry-after": "17"}, anthropic.RateLimitError)
                        .retry_after == 17.0)
        self.assertTrue(self.kind(500, "api_error", "x", cls=anthropic.InternalServerError).retryable)
        self.assertFalse(self.kind(401, "authentication_error", "x", cls=anthropic.AuthenticationError).retryable)

    def test_a_mid_stream_error_frame_without_a_status_is_still_classified(self):
        import anthropic

        exc = anthropic.APIError("Overloaded", httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
                                 body=fake.error_body("overloaded_error", "Overloaded"))
        self.assertEqual(agent_claude.map_error(exc).kind, "overloaded")
        exc = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
        self.assertEqual(agent_claude.map_error(exc).kind, "network")
        self.assertEqual(agent_claude.map_error(RuntimeError("boom")).kind, "unknown")

    def test_the_key_never_appears_in_an_error_text(self):
        other = "sk-" + "ant-abcdefghijklmnop"  # built in two parts, like KEY, so token scanners skip this fake key
        with patch.object(agent_config, "get_key", return_value=(KEY, "saved")):
            error = self.kind(400, "invalid_request_error", f"bad request for {KEY} and {other}",
                              cls=__import__("anthropic").BadRequestError)
        for text in (error.user_message, error.detail, str(error)):
            self.assertNotIn(KEY, text)
            self.assertNotIn(other, text)


if __name__ == "__main__":
    unittest.main()
