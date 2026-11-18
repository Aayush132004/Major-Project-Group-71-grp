"""Ground-truth scoring of executed plans.

This module is the *only* place that reads reference answers.  The optimizer
and its validation agent never call it; experiments use it to measure what
the optimizer actually achieved (and to build the full-validation oracle that
Section 10.2 compares decision quality against).
"""
from __future__ import annotations

from statistics import mean

from ..engine.executor import ExecResult
from ..engine.sqlrun import results_match
from . import metrics as M


class Oracle:
    def __init__(self, workload) -> None:
        self.wl = workload
        self._gold_rows: dict[str, list | None] = {}

    def score_one(self, query: dict, res: ExecResult) -> dict:
        out = {"qid": query["qid"], "tokens": res.tokens, "latency_ms": res.latency_ms}
        if self.wl.metric == "execution_accuracy":
            gold_sql = query["gold_sql"]
            if query["qid"] not in self._gold_rows:
                rows, err, _ = self.wl.sql.run(query["db_id"], gold_sql)
                self._gold_rows[query["qid"]] = rows if err is None else None
            ok = results_match(res.rows, self._gold_rows[query["qid"]], "order by" in gold_sql.lower())
            out.update(accuracy=float(ok), exec_acc=float(ok), exec_error=res.error is not None,
                       metric="execution_accuracy", answer=(res.sql or "")[:400])
            return out
        refs = query["answers"]
        p, r, f = M.best_prf(res.answer, refs)
        source = " ".join(res.context) + " " + query.get("contract_text", "")
        rl = M.best_rouge(res.answer, refs)
        # primary metric: Rouge-L for MS MARCO (its official QA metric), token F1 for CUAD spans
        primary, name = (rl, "rouge_l") if self.wl.name == "msmarco" else (f, "token_f1")
        out.update(accuracy=primary, metric=name, precision=p, recall=r, f1=f, rouge_l=rl,
                   answer=res.answer[:400],
                   hallucinated=float(M.hallucinated(res.answer, source)),
                   answer_chars=len(res.answer))
        if query.get("gold_ids"):
            out["recall_at_k"] = M.recall_at_k(res.retrieved_ids, query["gold_ids"])
        return out

    def score(self, queries: list[dict], results: list[ExecResult]) -> dict:
        rows = [self.score_one(q, r) for q, r in zip(queries, results)]
        agg = {"n": len(rows)}
        for key in ("accuracy", "precision", "recall", "f1", "rouge_l", "hallucinated", "exec_acc",
                    "recall_at_k", "exec_error", "tokens", "latency_ms", "answer_chars"):
            vals = [r[key] for r in rows if key in r]
            if vals:
                agg[key] = float(mean(vals))
        agg["per_query"] = rows
        return agg
