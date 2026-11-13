"""Tokens, costo y latencias por etapa (§4.3, capa Operación).

Cada ejecución del pipeline crea un `UsageTracker` y lo publica en un `ContextVar`; los
wrappers de LLM y embeddings reportan ahí. Así el tracking funciona igual en corridas
secuenciales o con hilos (cada hilo setea su propio tracker).

El costo se reporta **nominal** (lo que costaría sin caché), para que sea comparable entre
corridas; `costo_usd_pagado` excluye los aciertos de caché.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

# USD por 1M de tokens: (input, input cacheado, output). Revisar contra la página de precios
# de OpenAI antes de reportar (ADR 001).
PRICES_USD_PER_1M: dict[str, tuple[float, float, float]] = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "text-embedding-3-small": (0.02, 0.0, 0.0),
    "text-embedding-3-large": (0.13, 0.0, 0.0),
}

_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def price_for(model: str) -> tuple[float, float, float] | None:
    return PRICES_USD_PER_1M.get(model) or PRICES_USD_PER_1M.get(_DATE_SUFFIX.sub("", model))


@dataclass
class LLMCall:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_hit: bool = False
    ms: float = 0.0

    def cost(self) -> float:
        p = price_for(self.model)
        if not p:
            return 0.0
        fresh = self.input_tokens - self.cached_input_tokens
        return (fresh * p[0] + self.cached_input_tokens * p[1] + self.output_tokens * p[2]) / 1e6


@dataclass
class UsageTracker:
    llm_calls: list[LLMCall] = field(default_factory=list)
    embed_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    embed_tokens_cached: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timings_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] += (time.perf_counter() - t0) * 1000

    def add_llm(self, call: LLMCall) -> None:
        self.llm_calls.append(call)

    def add_embedding(self, model: str, tokens: int, cached: bool) -> None:
        self.embed_tokens[model] += tokens
        if cached:
            self.embed_tokens_cached[model] += tokens

    def tokens(self) -> dict[str, int]:
        return {
            "in": sum(c.input_tokens for c in self.llm_calls),
            "out": sum(c.output_tokens for c in self.llm_calls),
            "embed": sum(self.embed_tokens.values()),
        }

    def cost_usd(self, paid_only: bool = False) -> float:
        total = sum(c.cost() for c in self.llm_calls if not (paid_only and c.cache_hit))
        for model, tok in self.embed_tokens.items():
            p = price_for(model)
            if p:
                billable = tok - (self.embed_tokens_cached[model] if paid_only else 0)
                total += billable * p[0] / 1e6
        return total

    def all_cache_hits(self) -> bool:
        return bool(self.llm_calls) and all(c.cache_hit for c in self.llm_calls)

    def summary(self) -> dict:
        return {
            "tokens": self.tokens(),
            "costo_usd": round(self.cost_usd(), 6),
            "costo_usd_pagado": round(self.cost_usd(paid_only=True), 6),
            "llm_calls": len(self.llm_calls),
            "llm_cache_hits": sum(c.cache_hit for c in self.llm_calls),
            "latencias_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "tokens_por_etapa": _tokens_by_stage(self.llm_calls),
        }


def _tokens_by_stage(calls: list[LLMCall]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
    for c in calls:
        out[c.stage]["in"] += c.input_tokens
        out[c.stage]["out"] += c.output_tokens
    return dict(out)


_current: ContextVar[UsageTracker | None] = ContextVar("safefish_tracker", default=None)


def set_tracker(tracker: UsageTracker | None):
    return _current.set(tracker)


def reset_tracker(token) -> None:
    _current.reset(token)


def current_tracker() -> UsageTracker | None:
    return _current.get()
