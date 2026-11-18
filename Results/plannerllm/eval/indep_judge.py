"""Independent judge: a second LLM, from a different model family, used only for evaluation.

The optimizer's validation agent and the pipeline's answer generator are the same model
(gpt-oss-20b), so a verdict could share the generator's blind spots.  This module asks a
model from another family (Qwen, by default) two questions the optimizer never sees:

* ``grade``    - is a held-out answer correct with respect to the human reference answers?
                 This gives an accuracy that does not depend on word overlap.
* ``compare``  - given the same two outputs the validation agent compared, which one is better?
                 Agreement with the validation agent's verdict measures judge bias.

Both calls go through the normal client, so they are cached and rate limited, and their
tokens are kept in a separate ledger (they are evaluation cost, not optimisation cost).
"""
from __future__ import annotations

import hashlib
import json

from ..llm.base import TokenLedger
from ..llm.cache import ResponseCache
from ..llm.client import GroqClient

SYSTEM = ("You are an impartial grader of question-answering and information-extraction systems. "
          "You compare outputs carefully and reply only with the JSON object you are asked for.")

GRADE = """Task of the system: {task}

Question: {question}

Reference answer(s) written by people:
{refs}

System answer:
{answer}

Grade the system answer against the references.
- "correct": it states the information of at least one reference answer. Wording may differ and extra correct detail is fine.
- "partial": it contains only part of that information, or is too vague to be fully correct.
- "incorrect": it is wrong, unrelated, or says the information is missing when a reference exists.
A reference of NONE means the source has no such information; then an answer of NONE (or "not found") is correct and any other answer is incorrect.
Reply with JSON: {{"verdict": "correct" | "partial" | "incorrect"}}"""

COMPARE = """Task of the system: {task}

Question: {question}

Output A:
{a}

Output B:
{b}

Which output accomplishes the task better for this question? Judge correctness, completeness and
faithfulness to the task; do not prefer an output only because it is longer.
Reply with JSON: {{"better": "A" | "B" | "tie"}}"""

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class IndependentJudge:
    def __init__(self, settings) -> None:
        cfg = settings.llm
        self.model = cfg.judge_model
        self.ledger = TokenLedger(cfg.price_in_per_m, cfg.price_out_per_m, None)
        self.client = GroqClient(self.ledger, ResponseCache(cfg.cache_path), model=self.model,
                                 base_url=cfg.base_url, temperature=0.0, rpm=cfg.requests_per_minute,
                                 tpm=cfg.tokens_per_minute, max_retries=cfg.max_retries,
                                 timeout_s=cfg.timeout_s, extra_body={"reasoning_effort": "none"},
                                 reasoning_tokens=0)

    def grade(self, task: str, question: str, refs: list[str], answer: str) -> float | None:
        refs_txt = "\n".join(f"- {r[:600]}" for r in refs[:4]) or "- NONE"
        msg = GRADE.format(task=task, question=question, refs=refs_txt, answer=(answer or "(empty)")[:800])
        r = self.client.complete(msg, purpose="judge", task="judge_grade", system=SYSTEM, max_tokens=60,
                                 json_mode=True)
        v = str((r.json({}) or {}).get("verdict", "")).strip().lower()
        return SCORES.get(v)

    def compare(self, task: str, question: str, candidate: str, incumbent: str, key: str) -> str | None:
        """Returns 'candidate', 'incumbent' or 'tie'.  The A/B position is decided by a hash of ``key``
        so that position bias averages out over items."""
        swap = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 1
        a, b = (incumbent, candidate) if swap else (candidate, incumbent)
        msg = COMPARE.format(task=task, question=question, a=(a or "(empty)")[:800], b=(b or "(empty)")[:800])
        r = self.client.complete(msg, purpose="judge", task="judge_compare", system=SYSTEM, max_tokens=60,
                                 json_mode=True)
        v = str((r.json({}) or {}).get("better", "")).strip().upper()
        if v not in ("A", "B", "TIE"):
            return None
        if v == "TIE":
            return "tie"
        first = "incumbent" if swap else "candidate"
        second = "candidate" if swap else "incumbent"
        return first if v == "A" else second

    def usage(self) -> dict:
        return self.ledger.totals()


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def dumps(o) -> str:
    return json.dumps(o, sort_keys=True)
