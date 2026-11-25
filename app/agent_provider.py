# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""The Agent page's neutral side of a model provider: the models it offers with their prices, the types a request, a
turn and an error are made of, and the registry of providers. Nothing here imports an SDK or touches the network, so
the run loop, the page and the tests work with any provider (or a scripted fake one).

A provider turns a TurnRequest into a Turn. Blocks of a turn (`Turn.content`) are kept as the plain JSON the API sent,
because a later request must send them back exactly as they came (thinking blocks are bound to the conversation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high")
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 64000  # per response; thinking counts toward it
CACHE_TTLS = ("5m", "1h")
PRICES_AS_OF = "2026-11-03"


@dataclass(frozen=True)
class ModelSpec:
    """A model the page offers. Prices are US dollars per million tokens and only an estimate (PRICES_AS_OF): the
    cost meter and the cost limit use them. `tier_tokens`: a prompt longer than this is billed at the `high_*` rates
    (the cheapest model has two rate cards); 0 = one rate card."""
    id: str
    label: str
    hint: str
    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float
    fallbacks: bool = True          # the server-side fallback on a refusal is offered for this model
    tier_tokens: int = 0
    high_input: float = 0.0
    high_output: float = 0.0
    high_cache_read: float = 0.0
    high_cache_write_5m: float = 0.0
    high_cache_write_1h: float = 0.0


MODELS: dict[str, ModelSpec] = {spec.id: spec for spec in (
    ModelSpec("claude-opus-5-5", "Claude Opus 5.5",
              "Most capable for a full client brief. Slowest and the highest cost of the everyday models.",
              4.0, 20.0, 0.20, 5.0, 8.0),
    ModelSpec("claude-sonnet-5-5", "Claude Sonnet 5.5",
              "Balanced: good plans at half the cost of Opus.",
              2.0, 10.0, 0.10, 2.5, 4.0),
    ModelSpec("claude-haiku-5-5", "Claude Haiku 5.5",
              "Fastest and cheapest. Good for trying the page; may miss details in a long brief.",
              0.10, 0.50, 0.01, 0.125, 0.20, fallbacks=False, tier_tokens=100_000,
              high_input=0.50, high_output=2.50, high_cache_read=0.05, high_cache_write_5m=0.625,
              high_cache_write_1h=1.0),
    ModelSpec("claude-fable-5-2", "Claude Fable 5.2",
              "A different Claude model with the highest price. Compare it with Opus on a short brief.",
              8.0, 40.0, 0.20, 10.0, 16.0),
)}


def model_choices() -> list[tuple[str, str]]:
    """(label, id) of the models, for the page's dropdown."""
    return [(spec.label, spec.id) for spec in MODELS.values()]


def spec_of(model: str) -> ModelSpec:
    """The price card of `model`. A model the table does not know (a fallback model the server picked) is priced at
    the most expensive card, so the cost limit can only trip early."""
    found = MODELS.get(model)
    if found is not None:
        return found
    return max(MODELS.values(), key=lambda spec: spec.output)


@dataclass(frozen=True)
class UsageEntry:
    """Tokens of one sampling iteration of a response (the response's own, a compaction, a fallback attempt)."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    kind: str = "message"           # message | compaction | fallback_message

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read + self.cache_write_5m + self.cache_write_1h


def cost(entries: Sequence[UsageEntry]) -> float:
    """US dollars for `entries`, each priced by its own model and, for a two-rate-card model, by its prompt size."""
    total = 0.0
    for entry in entries:
        spec = spec_of(entry.model)
        high = bool(spec.tier_tokens) and entry.prompt_tokens > spec.tier_tokens
        rates = ((spec.high_input, spec.high_output, spec.high_cache_read, spec.high_cache_write_5m,
                  spec.high_cache_write_1h) if high else
                 (spec.input, spec.output, spec.cache_read, spec.cache_write_5m, spec.cache_write_1h))
        tokens = (entry.input_tokens, entry.output_tokens, entry.cache_read, entry.cache_write_5m,
                  entry.cache_write_1h)
        total += sum(rate * count for rate, count in zip(rates, tokens)) / 1_000_000
    return total


def worst_case(model: str, prompt_tokens: int, max_tokens: int = MAX_TOKENS) -> float:
    """What one attempt could cost at most: its whole prompt uncached and a full-length answer. A request that is cut
    off reports no usage, so the loop reserves this much for it."""
    return cost([UsageEntry(model, input_tokens=max(prompt_tokens, 0), output_tokens=max_tokens)])


@dataclass(frozen=True)
class Features:
    """Optional request features of the provider; each can be turned off by the provider itself when the server
    refuses it (the "capability ladder")."""
    compaction: bool = True
    compact_trigger: int = 200_000
    compact_instructions: str = ""
    fallbacks: bool = True
    cache_ttl: str = "5m"


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    """One response. `content` is the list of blocks as plain JSON (stored and sent back unchanged)."""
    content: list[dict]
    stop_reason: str
    model: str = ""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: list[UsageEntry] = field(default_factory=list)
    request_id: str = ""
    stop_details: Optional[dict] = None
    served_by_fallback: bool = False
    context_tokens: int = 0         # the size of the conversation this response was made from
    transformations: list = field(default_factory=list)  # thinking blocks the server dropped (drop_block)

    def cost(self) -> float:
        return cost(self.usage)


@dataclass
class TurnRequest:
    model: str
    system: str
    tools: list[dict]
    messages: list[dict]
    effort: str = DEFAULT_EFFORT
    max_tokens: int = MAX_TOKENS
    features: Features = field(default_factory=Features)


class ProviderError(Exception):
    """A failure of a request, classified for the run loop. kind: auth | billing | rate_limit | overloaded | network |
    timeout | server | too_large | context | binding | feature | model | bad_request | refusal | unknown.
    `user_message` is plain words for the page; `retryable` kinds are retried by the loop with a back-off."""

    RETRYABLE = ("rate_limit", "overloaded", "network", "timeout", "server")

    def __init__(self, kind: str, user_message: str, *, status: Optional[int] = None, retry_after: float = 0.0,
                 request_id: str = "", error_type: str = "", detail: str = ""):
        super().__init__(f"{kind}: {user_message}")
        self.kind, self.user_message, self.status = kind, user_message, status
        self.retry_after, self.request_id, self.error_type, self.detail = retry_after, request_id, error_type, detail

    @property
    def retryable(self) -> bool:
        return self.kind in self.RETRYABLE


class Aborted(Exception):
    """The stop flag was set while a response streamed in; the stream was closed."""


class Provider(Protocol):
    def complete(self, request: TurnRequest, *, should_abort: Callable[[], bool] = ...,
                 on_event: Callable[[str], None] = ...) -> Turn: ...

    def ping(self, model: str) -> str: ...


_PROVIDERS: dict[str, tuple[str, Callable[[], Provider]]] = {}


def register_provider(name: str, label: str, factory: Callable[[], Provider]) -> None:
    _PROVIDERS[name] = (label, factory)


def provider_choices() -> list[tuple[str, str]]:
    _builtin()
    return [(label, name) for name, (label, _) in _PROVIDERS.items()]


def get_provider(name: str = DEFAULT_PROVIDER) -> Provider:
    _builtin()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")
    return _PROVIDERS[name][1]()


def _builtin() -> None:
    """Claude is registered on first use, so importing this module never loads the SDK."""
    if DEFAULT_PROVIDER not in _PROVIDERS:
        def factory() -> Provider:
            from . import agent_claude

            return agent_claude.ClaudeProvider()
        register_provider(DEFAULT_PROVIDER, "Claude (Anthropic)", factory)
