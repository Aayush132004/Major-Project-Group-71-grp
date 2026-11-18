"""Applicability signatures (Section 5.2).

A rewrite's verdict is only reusable in a context that looks like the one in
which it was validated.  The signature captures that context from four
sources named in the methodology: template type (operator shape), data
statistics (cardinality, selectivity, document length), operator
configuration (index family, top-k), and prompt structure.  Continuous
statistics are bucketed so that "the same kind of query" hashes identically
while a real shift (e.g. selectivity moving an order of magnitude) does not.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

from .pipeline import Pipeline


def _bucket(value: float | None, edges: list[float]) -> int:
    if value is None:
        return -1
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def prompt_structure(plan: Pipeline) -> str:
    """Shape of the instruction text, not its words: length band + format cues."""
    parts = []
    for o in plan.topological():
        if not o.prompt:
            continue
        n_sent = len([s for s in re.split(r"[.!?]+", o.prompt) if s.strip()])
        cues = "".join(c for c, rx in (("N", r"\bNONE\b"), ("Q", r"quote|verbatim|word for word"),
                                       ("F", r"format|json|one sentence|short"))
                       if re.search(rx, o.prompt, re.I))
        parts.append(f"{o.kind}:{min(n_sent, 4)}{cues}")
    return "|".join(parts)


def _retrieve_cfg(plan: Pipeline) -> tuple[int | None, str | None, bool]:
    r = plan.find("retrieve")
    if r is None:
        return None, None, False
    return int(r.params.get("top_k", 10)), r.params.get("index", "hnsw"), bool(r.params.get("where"))


def config_shape(plan: Pipeline) -> str:
    """Physical configuration of a plan with top-k bucketed (part of the context)."""
    k, index, filtered = _retrieve_cfg(plan)
    gen = plan.find("sql_generate")
    shots = int(gen.params.get("fewshot_k", 0)) if gen else 0
    return f"{index}|{int(filtered)}|{_bucket(k, [5, 10, 20])}|{shots}|{prompt_structure(plan)}"


def context_features(parent: Pipeline, cand: Pipeline, directive: str, stats: dict) -> dict:
    k, index, filtered = _retrieve_cfg(cand)
    gen = cand.find("sql_generate")
    return {
        "directive": directive,
        "family": stats.get("family"),
        "parent_shape": "->".join(parent.kinds()),
        "parent_cfg": config_shape(parent),
        "cand_shape": "->".join(cand.kinds()),
        "rows_b": _bucket(math.log2(max(stats.get("n_rows", 1), 1)), [8, 10, 12, 14, 17, 20]),
        "sel_b": _bucket(stats.get("sel_est"), [0.005, 0.02, 0.06, 0.15, 0.4, 0.99]),
        "doclen_b": _bucket(stats.get("avg_doc_tokens"), [64, 128, 256, 512]),
        "topk_b": _bucket(k, [5, 10, 20]),
        "index": index,
        "filtered_probe": filtered,
        "fewshot": int(gen.params.get("fewshot_k", 0)) if gen else None,
        "tables_b": _bucket(stats.get("n_tables"), [3, 5, 8, 12]),
        "prompt": prompt_structure(cand),
    }


def signature_of(features: dict) -> str:
    blob = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def numeric_vector(stats: dict, cand: Pipeline) -> np.ndarray:
    """Continuous description used for near-signature (soft) matches and drift."""
    k, _, filtered = _retrieve_cfg(cand)
    return np.array([
        math.log10(max(stats.get("n_rows", 1), 1)),
        math.log10(max(stats.get("sel_est", 1.0), 1e-4)),
        math.log10(max(stats.get("avg_doc_tokens", 1.0), 1.0)),
        math.log2(max(k or 1, 1)),
        float(filtered),
        math.log2(max(stats.get("n_tables", 1) or 1, 1)),
    ], dtype=float)


def drift(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two contexts in log-space (0 == identical)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / math.sqrt(len(a)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
