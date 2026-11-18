"""Common LLM interface plus the token ledger that every agent charges.

Every call states *why* it is made (``purpose``).  The ledger keeps those
purposes apart because the paper's central claim is about validation spend
versus execution spend, so the two must never be mixed in one counter.

Cached responses are still charged in the *accounted* columns: a cache hit
saves us an API request during development, but the system under test must
pay for it exactly as if it had been issued, otherwise baselines that happen
to run second would look artificially cheap.
"""
from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

OPTIMIZATION_PURPOSES = {"generation_agent", "criteria", "validation", "pairwise", "prompt_rewrite", "validation_exec"}


class BudgetExhausted(RuntimeError):
    """Raised when an optimization-time call would cross the token budget."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False
    latency_s: float = 0.0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def json(self, default: Any = None) -> Any:
        return parse_json_loose(self.text, default)


@dataclass
class LedgerEntry:
    calls: int = 0
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    usd: float = 0.0


class TokenLedger:
    """Thread-safe per-purpose accounting with an optional optimization budget."""

    def __init__(self, price_in_per_m: float, price_out_per_m: float,
                 optimization_budget: int | None = None) -> None:
        self.price_in = price_in_per_m / 1e6
        self.price_out = price_out_per_m / 1e6
        self.optimization_budget = optimization_budget
        self._by_purpose: dict[str, LedgerEntry] = defaultdict(LedgerEntry)
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    def usd_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.price_in + completion_tokens * self.price_out

    def optimization_tokens(self) -> int:
        with self._lock:
            return sum(e.prompt_tokens + e.completion_tokens
                       for p, e in self._by_purpose.items() if p in OPTIMIZATION_PURPOSES)

    def remaining(self) -> int | None:
        if self.optimization_budget is None:
            return None
        return max(0, self.optimization_budget - self.optimization_tokens())

    def check(self, purpose: str, estimated_tokens: int) -> None:
        if purpose not in OPTIMIZATION_PURPOSES or self.optimization_budget is None:
            return
        if self.optimization_tokens() + estimated_tokens > self.optimization_budget:
            raise BudgetExhausted(
                f"{purpose} needs ~{estimated_tokens} tokens, "
                f"{self.remaining()} left of {self.optimization_budget}")

    def record(self, purpose: str, result: LLMResult) -> None:
        with self._lock:
            e = self._by_purpose[purpose]
            e.calls += 1
            e.api_calls += 0 if result.cached else 1
            e.prompt_tokens += result.prompt_tokens
            e.completion_tokens += result.completion_tokens
            e.latency_s += result.latency_s
            e.usd += self.usd_for(result.prompt_tokens, result.completion_tokens)

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {p: dict(vars(e)) for p, e in self._by_purpose.items()}

    def totals(self, optimization_only: bool | None = None) -> dict:
        snap = self.snapshot()
        keep = {
            p: e for p, e in snap.items()
            if optimization_only is None
            or (optimization_only and p in OPTIMIZATION_PURPOSES)
            or (optimization_only is False and p not in OPTIMIZATION_PURPOSES)
        }
        out = {"calls": 0, "api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
               "latency_s": 0.0, "usd": 0.0}
        for e in keep.values():
            for k in out:
                out[k] += e[k]
        out["tokens"] = out["prompt_tokens"] + out["completion_tokens"]
        return out

    def reset(self) -> None:
        with self._lock:
            self._by_purpose.clear()


class LLMClient(Protocol):
    model: str
    ledger: TokenLedger

    def complete(self, prompt: str, *, purpose: str, task: str,
                 payload: dict | None = None, system: str | None = None,
                 max_tokens: int = 256, json_mode: bool = False) -> LLMResult: ...


def estimate_tokens(text: str) -> int:
    """Rough tokenizer-free estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)


_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.S)


def parse_json_loose(text: str, default: Any = None) -> Any:
    """Parse JSON that an LLM may have wrapped in prose or code fences."""
    if not text:
        return default
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = cleaned.rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(cleaned)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return default
    return default


@dataclass
class CallRecord:
    """Lightweight trace of one LLM call; the UI shows these in transcripts."""
    purpose: str
    task: str
    prompt: str
    response: str
    tokens: int
    cached: bool
    meta: dict = field(default_factory=dict)
