"""Offline stand-in for the LLM, used by the test-suite and the no-key demo.

It never sees reference answers.  Each task is answered with a plain lexical
heuristic (extractive QA by word overlap, rating by groundedness, naive SQL),
so its quality is honest-but-weak.  Every number produced in this mode is
labelled SIMULATED by the API and UI; reported results must use Groq.
"""
from __future__ import annotations

import hashlib
import json
import re

from .base import TokenLedger, estimate_tokens
from .client import _BaseClient

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in and or is are was were be for on with by as at from that this "
            "what which who whom how when where why does do did it its into than then there".split())


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(str(text).lower()) if w not in _STOP]


def _overlap(a: str, b: str) -> float:
    wa, wb = set(words(a)), set(words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa)


def _jitter(*parts: str, scale: float = 0.15) -> float:
    h = hashlib.md5("|".join(parts).encode()).hexdigest()
    return (int(h[:6], 16) / 0xFFFFFF - 0.5) * 2 * scale


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text))
    return [p.strip() for p in parts if len(p.strip()) > 20]


class SimulatedLLM(_BaseClient):
    model = "simulated-extractive"

    def __init__(self, ledger: TokenLedger, cache=None) -> None:
        super().__init__(ledger, cache=cache)

    def _latency(self, prompt_tokens, completion_tokens, measured_s):
        # mimic a hosted 8B model: fixed overhead + prefill + decode time
        return 0.12 + prompt_tokens * 0.00015 + completion_tokens * 0.004

    # ------------------------------------------------------------------
    def _raw(self, prompt, system, max_tokens, json_mode, task, payload):
        payload = payload or {}
        handler = getattr(self, f"_t_{task}", None)
        text = handler(payload) if handler else ""
        if not isinstance(text, str):
            text = json.dumps(text)
        return text, estimate_tokens((system or "") + prompt), estimate_tokens(text)

    # ---- generation tasks --------------------------------------------
    def _best_sentence(self, question: str, passages: list[str]) -> str:
        best, best_score = "", -1.0
        for p in passages:
            for s in _sentences(p) or [p]:
                score = _overlap(question, s) + _jitter(question, s, scale=0.05)
                if score > best_score:
                    best, best_score = s, score
        return best[:300]

    def _t_answer(self, p):
        return self._best_sentence(p.get("question", ""), p.get("context", [])) or "Not found."

    def _t_extract(self, p):
        return self._best_sentence(p.get("question", "") + " " + p.get("instruction", ""),
                                   [p.get("text", "")]) or "NONE"

    def _t_reduce(self, p):
        return self._best_sentence(p.get("question", ""), p.get("items", [])) or "NONE"

    # ---- judgement tasks ---------------------------------------------
    def _score(self, question: str, output: str, context: list[str]) -> float:
        if not output or output.strip().upper() in {"NONE", "NOT FOUND."}:
            return 1.5
        grounded = max((_overlap(output, c) for c in context), default=0.0)
        relevant = _overlap(question, output)
        base = 1.0 + 2.5 * grounded + 1.5 * min(1.0, relevant * 2)
        return max(1.0, min(5.0, base + _jitter(question, output, scale=0.4)))

    def _t_rate(self, p):
        crits = p.get("criteria") or ["accuracy"]
        s = self._score(p.get("question", ""), p.get("output", ""), p.get("context", []))
        return {"scores": {c: round(max(1, min(5, s + _jitter(c, p.get("output", ""), scale=0.3))))
                           for c in crits}}

    def _t_rate_batch(self, p):
        crits = p.get("criteria") or ["accuracy"]
        out = []
        for k, it in enumerate(p.get("items", []), 1):
            s = self._score(it.get("question", ""), it.get("output", ""), it.get("context", []))
            out.append({"id": k, "scores": {c: round(max(1, min(5, s + _jitter(c, it.get("output", ""), scale=0.3))))
                                            for c in crits}})
        return {"items": out}

    def _t_pairwise(self, p):
        q, ctx = p.get("question", ""), p.get("context", [])
        sa = self._score(q, p.get("output_a", ""), ctx)
        sb = self._score(q, p.get("output_b", ""), ctx)
        return {"winner": "A" if sa >= sb else "B", "reason": "more grounded answer"}

    def _t_criteria(self, p):
        return {"criteria": [
            {"name": "correctness", "description": "answer is factually supported by the context"},
            {"name": "completeness", "description": "answer covers what the question asks"},
            {"name": "groundedness", "description": "no content absent from the context"},
        ][: p.get("max_criteria", 3)]}

    # ---- planning tasks ----------------------------------------------
    def _t_propose(self, p):
        names = list(p.get("directives", []))
        picked = [n for n in names if _jitter(n, p.get("pipeline", ""), scale=1.0) > -0.3]
        return {"directives": [{"name": n, "why": "may improve accuracy per dollar"} for n in picked[:4]]}

    def _t_prompt_rewrite(self, p):
        base = p.get("prompt", "")
        return {"prompt": base.rstrip(".") + ". Answer in one short sentence and copy key phrases "
                                             "verbatim from the context."}

    def _t_sql(self, p):
        q = p.get("question", "").lower()
        schema: dict = p.get("schema", {})
        if not schema:
            return "SELECT 1"
        table = max(schema, key=lambda t: _overlap(q, t + " " + " ".join(schema[t])) + _jitter(t, scale=0.01))
        cols = schema[table]
        col = max(cols, key=lambda c: _overlap(q, c.replace("_", " "))) if cols else "*"
        if any(k in q for k in ("how many", "number of", "count")):
            return f"SELECT count(*) FROM {table}"
        return f"SELECT {col} FROM {table}"

    def _t_sql_repair(self, p):
        return self._t_sql(p)
