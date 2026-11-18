"""Local judge: a small CPU model that gives the gate a cheap opinion.

The judge runs only the *non-LLM prefix* of each plan (filters, ANN probe,
rerank, schema linking) on the validation sample and scores what each plan
would hand to the LLM with the MiniLM cross-encoder (22M parameters, well
inside the <=3B budget of Section 6).  It costs no API tokens.  When two
plans differ only in their LLM operators (prompt rewrite, decompose,
few-shot...) the prefixes are identical and the judge abstains (returns
None), which the gate treats as "no information".
"""
from __future__ import annotations

import math

import numpy as np

from .pipeline import Pipeline


def cheap_prefix(plan: Pipeline) -> Pipeline:
    ops = []
    for o in plan.topological():
        if o.uses_llm:
            break
        ops.append(o)
    p = plan.clone()
    p.operators = [o for o in p.operators if o.name in {x.name for x in ops}]
    return p


class LocalJudge:
    def __init__(self, workload, executor, temperature: float = 0.6, top: int = 3) -> None:
        self.wl = workload
        self.ex = executor
        self.temperature = temperature
        self.top = top
        self.calls = 0

    def _context_quality(self, plan: Pipeline, queries: list[dict]) -> float | None:
        prefix = cheap_prefix(plan)
        if not prefix.operators:
            return None
        scores = []
        for q in queries:
            res = self.ex.run(prefix, q, purpose="judge")
            texts = [self.wl.records[self.wl.id_to_pos[i]]["text"] for i in res.retrieved_ids[: self.top]]
            if not texts:
                scores.append(-10.0)
                continue
            s = self.wl.reranker.score(q["question"], texts)
            scores.append(float(np.mean(np.sort(s)[::-1][: self.top])))
        return float(np.mean(scores)) if scores else None

    def prefer_probability(self, cand: Pipeline, inc: Pipeline, queries: list[dict]) -> float | None:
        """P_judge(candidate better than incumbent), or None when uninformative."""
        if cheap_prefix(cand).fingerprint() == cheap_prefix(inc).fingerprint():
            return None
        self.calls += 1
        sc = self._context_quality(cand, queries)
        sp = self._context_quality(inc, queries)
        if sc is None or sp is None:
            return None
        return 1.0 / (1.0 + math.exp(-(sc - sp) / self.temperature))
