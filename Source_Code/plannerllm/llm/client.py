"""Concrete LLM clients: Groq (OpenAI-compatible HTTP) and a local simulator."""
from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from typing import Callable

from .base import (BudgetExhausted, CallRecord, LLMResult, TokenLedger,
                   estimate_tokens)
from .cache import ResponseCache

Listener = Callable[[CallRecord], None]


class _BaseClient:
    """Shared plumbing: budget check -> cache -> raw call -> ledger -> listeners."""

    model: str = "base"

    def __init__(self, ledger: TokenLedger, cache: ResponseCache | None = None,
                 temperature: float = 0.0) -> None:
        self.ledger = ledger
        self.cache = cache
        self.temperature = temperature
        self.listeners: list[Listener] = []

    def _raw(self, prompt: str, system: str | None, max_tokens: int, json_mode: bool,
             task: str, payload: dict | None) -> tuple[str, int, int]:
        raise NotImplementedError

    _latency_kind: str | None = "service"

    def _latency(self, prompt_tokens: int, completion_tokens: int, measured_s: float) -> float:
        return measured_s

    def complete(self, prompt: str, *, purpose: str, task: str,
                 payload: dict | None = None, system: str | None = None,
                 max_tokens: int = 256, json_mode: bool = False) -> LLMResult:
        self.ledger.check(purpose, estimate_tokens(prompt) + max_tokens // 2)
        key = None
        hit = None
        if self.cache is not None:
            key = ResponseCache.make_key(self.model, system, prompt, max_tokens,
                                         json_mode, self.temperature)
            hit = self.cache.get(key)
        t0 = time.perf_counter()
        if hit is not None:
            text, pt, ct, lat = hit
            # report the latency the original request had, so timing studies
            # are not distorted by cache hits
            result = LLMResult(text, pt, ct, cached=True, latency_s=lat, model=self.model)
        else:
            text, pt, ct = self._raw(prompt, system, max_tokens, json_mode, task, payload)
            result = LLMResult(text, pt, ct, cached=False,
                               latency_s=self._latency(pt, ct, time.perf_counter() - t0),
                               model=self.model)
            if self.cache is not None and key is not None and text:
                self.cache.put(key, self.model, text, pt, ct, result.latency_s, self._latency_kind)
        self.ledger.record(purpose, result)
        if self.listeners:
            rec = CallRecord(purpose=purpose, task=task, prompt=prompt, response=result.text,
                             tokens=result.total_tokens, cached=result.cached)
            for fn in list(self.listeners):
                fn(rec)
        return result


class RateLimiter:
    """Sliding 60-second window over both requests and tokens."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm, self.tpm = rpm, tpm
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] > 60.0:
                    self._events.popleft()
                used = sum(t for _, t in self._events)
                if len(self._events) < self.rpm and used + tokens <= self.tpm:
                    self._events.append((now, tokens))
                    return
                wait = 60.0 - (now - self._events[0][0]) + 0.05 if self._events else 1.0
            time.sleep(min(max(wait, 0.2), 10.0))


class GroqClient(_BaseClient):
    """Chat-completions client for Groq (works with any OpenAI-compatible URL)."""

    def __init__(self, ledger: TokenLedger, cache: ResponseCache | None, *,
                 model: str, base_url: str, api_key: str | None = None,
                 temperature: float = 0.0, rpm: int = 28, tpm: int = 5500,
                 max_retries: int = 6, timeout_s: float = 60.0, extra_body: dict | None = None,
                 reasoning_tokens: int = 0) -> None:
        super().__init__(ledger, cache, temperature)
        self.extra_body = dict(extra_body or {})
        self.reasoning_tokens = reasoning_tokens
        import httpx
        self.model = model
        # GROQ_API_KEYS may hold several comma-separated keys (from different accounts); when one
        # account's daily token quota is used up the client moves on to the next key.
        keys = [api_key] if api_key else [k.strip() for k in
                (os.environ.get("GROQ_API_KEYS", "") + "," + os.environ.get("GROQ_API_KEY", "")).split(",")]
        self.api_keys = list(dict.fromkeys(k for k in keys if k))
        if not self.api_keys:
            raise RuntimeError("GROQ_API_KEY is not set (put it in .env or the environment)")
        self.key_index = 0
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.limiter = RateLimiter(rpm, tpm)
        self._http = httpx.Client(timeout=timeout_s)

    _service_s: float | None = None

    def _latency(self, prompt_tokens: int, completion_tokens: int, measured_s: float) -> float:
        s, self._service_s = self._service_s, None
        if s is not None:
            return float(s)
        return self.cache.estimate_latency(prompt_tokens, completion_tokens) if self.cache else measured_s

    @property
    def api_key(self) -> str:
        return self.api_keys[self.key_index]

    def _raw(self, prompt, system, max_tokens, json_mode, task, payload):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens + self.reasoning_tokens,
                "temperature": self.temperature, **self.extra_body}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        self.limiter.acquire(estimate_tokens(prompt + (system or "")) + max_tokens + self.reasoning_tokens // 2)
        delay = 2.0
        waited = 0.0
        attempt = 0
        while True:
            attempt += 1
            try:
                r = self._http.post(f"{self.base_url}/chat/completions", json=body,
                                    headers={"Authorization": f"Bearer {self.api_key}"})
            except Exception:  # network hiccup
                if attempt > self.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage") or {}
                # server-side compute time; excludes queueing, network and our rate-limit waits
                self._service_s = usage.get("total_time")
                return (text, int(usage.get("prompt_tokens", estimate_tokens(prompt))),
                        int(usage.get("completion_tokens", estimate_tokens(text))))
            if r.status_code == 400 and json_mode and "json" in r.text.lower():
                # the model failed to emit valid JSON; retry once without json mode
                body.pop("response_format", None)
                json_mode = False
                continue
            if (r.status_code == 429 and "per day" in r.text
                    and self.key_index + 1 < len(self.api_keys)):
                # this account's daily quota is gone: switch to the next key and retry at once
                self.key_index += 1
                continue
            if r.status_code == 429 and waited < 8 * 3600:
                # per-minute and per-day quotas: wait for the window instead of failing the run
                retry_after = r.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else delay
                except ValueError:
                    wait = delay
                wait = min(wait + random.random(), 900)
                waited += wait
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            if r.status_code in (500, 502, 503) and attempt <= self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"Groq API error {r.status_code}: {r.text[:300]}")


def build_client(settings, ledger: TokenLedger | None = None):
    """Factory used everywhere; reads ``settings.llm``."""
    from .simulated import SimulatedLLM
    cfg = settings.llm
    ledger = ledger or TokenLedger(cfg.price_in_per_m, cfg.price_out_per_m,
                                   settings.budget.optimization_tokens)
    if cfg.backend == "sim":
        # the simulator also caches, so a sim run tells us how many *unique*
        # tokens the same experiment would send to Groq (see scripts/estimate_tokens.py)
        return SimulatedLLM(ledger, ResponseCache(cfg.sim_cache_path))
    cache = ResponseCache(cfg.cache_path)
    return GroqClient(ledger, cache, model=cfg.model, base_url=cfg.base_url,
                      temperature=cfg.temperature, rpm=cfg.requests_per_minute,
                      tpm=cfg.tokens_per_minute, max_retries=cfg.max_retries,
                      timeout_s=cfg.timeout_s, extra_body=cfg.extra_body,
                      reasoning_tokens=cfg.reasoning_tokens)


__all__ = ["GroqClient", "RateLimiter", "build_client", "BudgetExhausted"]
