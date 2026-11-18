"""Hybrid symbolic-neural cost model (Section 5.3).

    C^(c) = sum_op f_sym(op; stats)  +  f_ANN(index, n, d, k)  +  sum_llm tokens * price
    C^_hybrid(c) = C^(c) * exp(r^_nn(features(c)))

The symbolic part walks the plan once, propagating estimated row counts with
DuckDB's optimizer selectivity and the classical independence assumption.
It is deliberately the "textbook" estimator: it is wrong exactly where the
literature says it is wrong (correlated predicates - e.g. a post-filter after
a semantic probe keeps far more rows than the corpus-wide selectivity says).
The learned residual is a small gradient-boosted model on the log-ratio
between observed and symbolic cost, so it can only *correct* the backbone,
never replace it - unless the ``neural`` ablation is requested.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .pipeline import Pipeline

# token sizes of the fixed parts of our prompt templates (measured once)
TEMPLATE_TOKENS = {"generate": 70, "map": 60, "sql_generate": 45, "sql_repair": 70}
OUTPUT_TOKENS = {"generate": 45, "map": 35, "sql_generate": 40, "sql_repair": 40}
MAP_KEEP_RATE = 0.5
REPAIR_RATE = 0.3


@dataclass
class CostEstimate:
    usd: float
    latency_ms: float
    tokens: float
    llm_calls: float
    breakdown: dict = field(default_factory=dict)
    mode: str = "symbolic"

    def to_dict(self) -> dict:
        return {"usd": self.usd, "latency_ms": self.latency_ms, "tokens": self.tokens,
                "llm_calls": self.llm_calls, "mode": self.mode, "breakdown": self.breakdown}


@dataclass
class Calibration:
    """Machine-specific constants, measured by microbenchmarks at start-up."""
    ann_ms: dict = field(default_factory=lambda: {"hnsw": 0.15, "ivf": 0.10, "flat": 0.6})
    ce_ms_per_pair: float = 2.5
    db_ms_per_krow: float = 0.4
    llm_overhead_s: float = 0.25
    llm_s_per_in_token: float = 0.00015
    llm_s_per_out_token: float = 0.004


class HybridCostModel:
    FEATURES = ["log_rows", "sel_est", "log_doc_tokens", "llm_calls", "log_tokens", "top_k",
                "has_rerank", "has_map", "filtered_probe", "post_filter", "idx_hnsw", "idx_ivf",
                "idx_flat", "fewshot", "has_repair", "fam_retrieval", "fam_mixed", "fam_sql",
                "log_sym_usd", "log_sym_ms"]

    def __init__(self, price_in_per_m: float, price_out_per_m: float, cpu_usd_per_hour: float,
                 calibration: Calibration | None = None, mode: str = "hybrid") -> None:
        self.p_in = price_in_per_m / 1e6
        self.p_out = price_out_per_m / 1e6
        self.cpu_usd_per_ms = cpu_usd_per_hour / 3.6e6
        self.calib = calibration or Calibration()
        self.mode = mode                      # symbolic | neural | hybrid
        self._res_usd = self._res_ms = None   # residual regressors
        self._nn_usd = self._nn_ms = None     # neural-only regressors
        self.n_train = 0

    # ------------------------------------------------------------ symbolic
    def _llm_call(self, kind: str, prompt_tokens: float) -> tuple[float, float, float]:
        out = OUTPUT_TOKENS.get(kind, 40)
        usd = prompt_tokens * self.p_in + out * self.p_out
        sec = self.calib.llm_overhead_s + prompt_tokens * self.calib.llm_s_per_in_token \
            + out * self.calib.llm_s_per_out_token
        return usd, sec * 1000, prompt_tokens + out

    def symbolic(self, plan: Pipeline, stats: dict) -> CostEstimate:
        n = float(stats.get("n_rows", 1000))
        sel = float(stats.get("sel_est", 1.0))
        doc_tok = min(float(stats.get("avg_doc_tokens", 120)), 175.0)
        q_tok = float(stats.get("avg_query_tokens", 12))
        rows: float | None = None            # None == whole corpus still in play
        restricted = 1.0                     # fraction of corpus visible to the probe
        db_ms = ann_ms = ce_ms = llm_ms = usd_tok = tokens = calls = 0.0
        per_op = []
        for op in plan.topological():
            before = (db_ms, ann_ms, ce_ms, llm_ms, usd_tok, tokens)
            k = op.kind
            if k == "filter":
                scanned = n if rows is None else rows
                db_ms += scanned / 1000 * self.calib.db_ms_per_krow + 0.3
                if rows is None:
                    restricted *= sel
                    rows = n * restricted
                else:
                    rows = rows * sel        # independence assumption (the classic error)
            elif k == "retrieve":
                fam = op.params.get("index", "hnsw")
                base = self.calib.ann_ms.get(fam, 0.2) * (n / 1e4) ** (0.3 if fam == "hnsw" else 1.0)
                vis = restricted
                if op.params.get("where"):
                    db_ms += n / 1000 * self.calib.db_ms_per_krow + 0.3
                    vis *= sel
                if vis < 1.0:   # predicate-restricted search: graph/list walks get longer
                    base *= {"hnsw": min(1 / max(vis, 1e-3), 50) ** 0.5, "ivf": 4.0, "flat": 1.0}.get(fam, 1.0)
                ann_ms += base * (int(op.params.get("top_k", 10)) / 10) ** 0.2
                rows = min(float(op.params.get("top_k", 10)), n * vis)
            elif k == "rerank":
                r_in = rows if rows is not None else n
                ce_ms += r_in * self.calib.ce_ms_per_pair
                rows = min(float(op.params.get("top_k", 5)), r_in)
            elif k == "map":
                r_in = rows if rows is not None else n
                if op.params.get("max_records"):
                    r_in = min(r_in, float(op.params["max_records"]))
                u, ms, t = self._llm_call("map", TEMPLATE_TOKENS["map"] + q_tok + min(doc_tok * 1.6, 300))
                usd_tok += u * r_in
                llm_ms += ms * r_in
                tokens += t * r_in
                calls += r_in
                rows = r_in * MAP_KEEP_RATE
            elif k == "generate":
                r_in = min(rows if rows is not None else n, float(op.params.get("max_docs", 6)))
                per_doc = 60 if plan.find("map") is not None else doc_tok
                u, ms, t = self._llm_call("generate", TEMPLATE_TOKENS["generate"] + q_tok + r_in * per_doc)
                usd_tok, llm_ms, tokens, calls = usd_tok + u, llm_ms + ms, tokens + t, calls + 1
            elif k == "sql_generate":
                schema_tok = float(stats.get("schema_tokens", 150))
                n_tab = float(stats.get("n_tables", 5))
                if rows is not None:
                    schema_tok *= min(1.0, rows / max(n_tab, 1.0))
                shots = int(op.params.get("fewshot_k", 0)) * 45
                u, ms, t = self._llm_call("sql_generate", TEMPLATE_TOKENS["sql_generate"] + q_tok + schema_tok + shots)
                usd_tok, llm_ms, tokens, calls = usd_tok + u, llm_ms + ms, tokens + t, calls + 1
            elif k == "sql_repair":
                db_ms += 3.0
                u, ms, t = self._llm_call("sql_repair", TEMPLATE_TOKENS["sql_repair"] + q_tok
                                          + float(stats.get("schema_tokens", 150)))
                usd_tok += REPAIR_RATE * u
                llm_ms += REPAIR_RATE * ms
                tokens += REPAIR_RATE * t
                calls += REPAIR_RATE
            elif k == "sql_execute":
                db_ms += 3.0
            after = (db_ms, ann_ms, ce_ms, llm_ms, usd_tok, tokens)
            per_op.append({"op": op.name, "kind": k, "rows_out": rows,
                           "ms": sum(after[i] - before[i] for i in range(4)),
                           "usd": after[4] - before[4], "tokens": after[5] - before[5]})
        cpu_ms = db_ms + ann_ms + ce_ms
        usd = usd_tok + cpu_ms * self.cpu_usd_per_ms
        return CostEstimate(usd=usd, latency_ms=cpu_ms + llm_ms, tokens=tokens, llm_calls=calls,
                            breakdown={"db_ms": db_ms, "ann_ms": ann_ms, "rerank_ms": ce_ms,
                                       "llm_ms": llm_ms, "token_usd": usd_tok,
                                       "cpu_usd": cpu_ms * self.cpu_usd_per_ms, "per_op": per_op},
                            mode="symbolic")

    # ------------------------------------------------------------ features
    def features(self, plan: Pipeline, stats: dict, sym: CostEstimate | None = None) -> list[float]:
        sym = sym or self.symbolic(plan, stats)
        r = plan.find("retrieve")
        idx = r.params.get("index", "hnsw") if r else None
        g = plan.find("sql_generate")
        fam = stats.get("family", "")
        return [
            math.log10(max(stats.get("n_rows", 1), 1)), float(stats.get("sel_est", 1.0)),
            math.log10(max(stats.get("avg_doc_tokens", 1), 1)), sym.llm_calls,
            math.log10(max(sym.tokens, 1)), float(r.params.get("top_k", 0)) if r else 0.0,
            float(plan.find("rerank") is not None), float(plan.find("map") is not None),
            float(bool(r and r.params.get("where"))),
            float(r is not None and plan.find("filter") is not None
                  and plan.index_of(plan.find("filter").name) > plan.index_of(r.name)),
            float(idx == "hnsw"), float(idx == "ivf"), float(idx == "flat"),
            float(g.params.get("fewshot_k", 0)) if g else 0.0, float(plan.find("sql_repair") is not None),
            float(fam == "retrieval-heavy"), float(fam == "mixed"), float(fam == "sql-heavy"),
            math.log10(max(sym.usd, 1e-9)), math.log10(max(sym.latency_ms, 1e-3)),
        ]

    # ------------------------------------------------------------ learning
    def fit(self, X: list[list[float]], obs_usd: list[float], obs_ms: list[float],
            sym_usd: list[float], sym_ms: list[float], min_samples: int = 12) -> bool:
        if len(X) < min_samples:
            return False
        from sklearn.ensemble import GradientBoostingRegressor
        Xa = np.asarray(X, dtype=float)
        lo = lambda v: np.log(np.maximum(np.asarray(v, dtype=float), 1e-9))
        mk = lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.06,
                                               subsample=0.9, random_state=0)
        self._res_usd = mk().fit(Xa, lo(obs_usd) - lo(sym_usd))
        self._res_ms = mk().fit(Xa, lo(obs_ms) - lo(sym_ms))
        Xn = Xa[:, :-2]   # neural-only model may not see the symbolic estimate
        self._nn_usd = mk().fit(Xn, lo(obs_usd))
        self._nn_ms = mk().fit(Xn, lo(obs_ms))
        self.n_train = len(X)
        return True

    @property
    def fitted(self) -> bool:
        return self._res_usd is not None

    def predict(self, plan: Pipeline, stats: dict, mode: str | None = None) -> CostEstimate:
        mode = mode or self.mode
        sym = self.symbolic(plan, stats)
        if mode == "symbolic" or not self.fitted:
            return sym
        x = np.asarray(self.features(plan, stats, sym), dtype=float).reshape(1, -1)
        if mode == "neural":
            usd = float(np.exp(self._nn_usd.predict(x[:, :-2])[0]))
            ms = float(np.exp(self._nn_ms.predict(x[:, :-2])[0]))
        else:
            usd = sym.usd * float(np.exp(self._res_usd.predict(x)[0]))
            ms = sym.latency_ms * float(np.exp(self._res_ms.predict(x)[0]))
        scale = usd / sym.usd if sym.usd > 0 else 1.0
        return CostEstimate(usd=usd, latency_ms=ms, tokens=sym.tokens * scale,
                            llm_calls=sym.llm_calls, breakdown={**sym.breakdown, "symbolic_usd": sym.usd,
                                                                "symbolic_ms": sym.latency_ms},
                            mode=mode)

    def validation_cost(self, cand: Pipeline, inc: Pipeline, stats: dict, n_items: int,
                        n_criteria: int) -> tuple[float, float]:
        """Expected (usd, tokens) of validating ``cand`` against ``inc`` on a sample.

        Covers executing both plans on the sample plus one rating call per
        output and a few pairwise calls - the spend the VoI gate weighs.
        """
        c = self.predict(cand, stats)
        p = self.predict(inc, stats)
        rate_tok = 420 + 25 * n_criteria
        tokens = n_items * (c.tokens + p.tokens + 2 * rate_tok) + 3 * 520
        usd = n_items * (c.usd + p.usd) + (2 * n_items * rate_tok + 3 * 520) * (0.9 * self.p_in + 0.1 * self.p_out)
        return usd, tokens
