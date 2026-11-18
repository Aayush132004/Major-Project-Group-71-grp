"""Groq client behaviour against a mocked HTTP endpoint (no key or network needed)."""
import json

import httpx
import pytest

from plannerllm.llm.base import BudgetExhausted, TokenLedger, parse_json_loose
from plannerllm.llm.cache import ResponseCache
from plannerllm.llm.client import GroqClient


def make_client(handler, tmp_path, budget=None):
    ledger = TokenLedger(0.05, 0.08, budget)
    c = GroqClient(ledger, ResponseCache(tmp_path / "c.sqlite"), model="llama-3.1-8b-instant",
                   base_url="https://example.test/v1", api_key="test", rpm=1000, tpm=10**7, max_retries=3)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def ok(text, pt=10, ct=5):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                     "usage": {"prompt_tokens": pt, "completion_tokens": ct}})


def test_cache_hit_is_still_charged(tmp_path):
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return ok("hello")
    c = make_client(handler, tmp_path)
    a = c.complete("hi", purpose="validation", task="rate")
    b = c.complete("hi", purpose="validation", task="rate")
    assert len(calls) == 1 and not a.cached and b.cached
    snap = c.ledger.snapshot()["validation"]
    assert snap["calls"] == 2 and snap["api_calls"] == 1 and snap["prompt_tokens"] == 20


def test_retries_on_429(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seq = [httpx.Response(429, headers={"retry-after": "0"}), ok("done")]
    c = make_client(lambda req: seq.pop(0), tmp_path)
    assert c.complete("x", purpose="execute", task="answer").text == "done"


def test_json_mode_falls_back_when_model_breaks_json(tmp_path):
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"code": "json_validate_failed", "message": "json"}})
        return ok('{"winner": "B"}')
    c = make_client(handler, tmp_path)
    r = c.complete("which? reply as JSON", purpose="pairwise", task="pairwise", json_mode=True)
    assert r.json()["winner"] == "B" and "response_format" not in bodies[-1]


def test_budget_is_enforced_for_optimisation_only(tmp_path):
    c = make_client(lambda req: ok("x", 400, 100), tmp_path, budget=600)
    c.complete("a" * 40, purpose="validation", task="rate", max_tokens=10)
    with pytest.raises(BudgetExhausted):
        c.complete("b" * 400, purpose="validation", task="rate", max_tokens=200)
    c.complete("c" * 4000, purpose="execute", task="answer")       # execution is never blocked


def test_parse_json_loose():
    assert parse_json_loose('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose("score: {\"scores\": {\"x\": 4}} thanks") == {"scores": {"x": 4}}
    assert parse_json_loose("no json", default=[]) == []
