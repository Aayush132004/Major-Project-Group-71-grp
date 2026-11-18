"""Selectivity-aware, size-weighted sampling for optimisation-time validation.

Following DocETL, the sample grows when downstream operators are selective
(a filter or an extract-map that drops most records would otherwise leave
too few informative outputs), and items are drawn with probability that
grows with document/question length so long, hard items are not starved.
"""
from __future__ import annotations

import math
import random

from .pipeline import Pipeline


def downstream_selectivity(plan: Pipeline, stats: dict) -> float:
    sel = 1.0
    for op in plan.topological():
        if op.kind == "filter" or (op.kind == "retrieve" and op.params.get("where")):
            sel *= max(stats.get("sel_est", 1.0), 0.02) ** 0.25   # soften: queries, not rows
        if op.kind == "map":
            sel *= 0.5
    return max(sel, 0.1)


def sample_size(n_pool: int, plan: Pipeline, stats: dict, cfg) -> int:
    sel = downstream_selectivity(plan, stats)
    n = math.ceil(max(cfg.min_items, cfg.frac * n_pool) / sel)
    return int(min(max(n, cfg.min_items), cfg.max_items, n_pool))


def sample_queries(queries: list[dict], plan: Pipeline, stats: dict, cfg, salt: str = "") -> list[dict]:
    n = sample_size(len(queries), plan, stats, cfg)
    rng = random.Random(f"{cfg.seed}:{salt}")
    weights = [math.sqrt(len(q["question"]) + len(q.get("contract_text", "")) / 50 + 1) for q in queries]
    pool = list(range(len(queries)))
    picked: list[int] = []
    while pool and len(picked) < n:   # weighted sampling without replacement
        total = sum(weights[i] for i in pool)
        r, acc = rng.random() * total, 0.0
        for i in pool:
            acc += weights[i]
            if acc >= r:
                picked.append(i)
                pool.remove(i)
                break
    return [queries[i] for i in sorted(picked)]
