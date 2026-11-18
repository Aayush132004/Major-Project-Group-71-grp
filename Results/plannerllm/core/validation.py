"""Validation agent (the base-paper evaluation protocol, Sections 5.4 / 8).

1. synthesise task-specific criteria for the pipeline's final operator;
2. execute candidate and incumbent on a selectivity-aware sample;
3. rate every output 1-5 per criterion with the LLM;
4. when the average ratings are close, run a pairwise comparison (with the
   A/B positions swapped on alternate items to cancel position bias);
5. return a verdict plus a transcript the UI and report can show.

The agent never sees reference answers - it judges outputs the way a human
reviewer would, from the question and the context the plan used.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from ..engine import prompts
from ..engine.executor import ExecResult, Executor
from ..llm.base import BudgetExhausted
from .pipeline import Pipeline


@dataclass
class Verdict:
    prefer_candidate: bool
    score_c: float = 0.0
    score_p: float = 0.0
    criteria: list = field(default_factory=list)
    pairwise_win_rate: float | None = None
    source: str = "validation"     # validation | memory | gate-skip | cost-only | budget
    confidence: float = 1.0
    tokens: int = 0
    note: str = ""
    items: list = field(default_factory=list)

    @property
    def acc_delta(self) -> float:
        """Rating gap mapped onto the 0-1 accuracy scale used by the gate."""
        return (self.score_c - self.score_p) / 4.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["acc_delta"] = self.acc_delta
        return d


def output_text(res: ExecResult) -> str:
    if res.sql is not None:
        rows = res.rows[:5] if res.rows else res.rows
        return f"SQL: {res.sql}\nResult (first rows): {rows}\nError: {res.error or 'none'}"
    return res.answer or "(empty)"


def context_text(res: ExecResult, limit: int = 3, chars: int = 350) -> str:
    return "\n".join(f"- {c[:chars]}" for c in res.context[:limit]) or "(none)"


class ValidationAgent:
    def __init__(self, llm, executor: Executor, settings, task_text: str, final_kind: str) -> None:
        self.llm = llm
        self.ex = executor
        self.cfg = settings.validation
        self.task_text = task_text
        self.final_kind = final_kind
        self._criteria: list[dict] | None = None
        self._outputs: dict[tuple[str, str], ExecResult] = {}
        self._ratings: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------ steps
    def criteria(self) -> list[dict]:
        if self._criteria is None:
            msg = prompts.CRITERIA.format(task=self.task_text, kind=self.final_kind, k=self.cfg.max_criteria)
            r = self.llm.complete(msg, purpose="criteria", task="criteria",
                                  payload={"max_criteria": self.cfg.max_criteria},
                                  system=prompts.SYSTEM_AGENT, max_tokens=300, json_mode=True)
            crits = (r.json({}) or {}).get("criteria") or []
            crits = [c for c in crits if isinstance(c, dict) and c.get("name")][: self.cfg.max_criteria]
            self._criteria = crits or [{"name": "accuracy", "description": "the output answers the question correctly"},
                                       {"name": "faithfulness", "description": "the output is supported by the context"}]
        return self._criteria

    def outputs(self, plan: Pipeline, sample: list[dict]) -> list[ExecResult]:
        fp = plan.fingerprint()
        out = []
        for q in sample:
            key = (fp, q["qid"])
            if key not in self._outputs:
                self._outputs[key] = self.ex.run(plan, q, purpose="validation_exec")
            out.append(self._outputs[key])
        return out

    def _rate(self, plan_fp: str, q: dict, res: ExecResult) -> float:
        key = (plan_fp, q["qid"])
        if key in self._ratings:
            return self._ratings[key]
        crits = self.criteria()
        crit_txt = "\n".join(f"- {c['name']}: {c.get('description', '')}" for c in crits)
        msg = prompts.RATE.format(task=self.task_text, question=q["question"], context=context_text(res),
                                  output=output_text(res), criteria=crit_txt)
        r = self.llm.complete(msg, purpose="validation", task="rate", system=prompts.SYSTEM_AGENT,
                              payload={"question": q["question"], "output": output_text(res),
                                       "context": res.context, "criteria": [c["name"] for c in crits]},
                              max_tokens=120, json_mode=True)
        scores = (r.json({}) or {}).get("scores") or {}
        vals = []
        for v in scores.values():
            try:
                vals.append(min(5.0, max(1.0, float(v))))
            except (TypeError, ValueError):
                continue
        score = sum(vals) / len(vals) if vals else 3.0
        self._ratings[key] = score
        return score

    def _rate_many(self, plan_fp: str, sample: list[dict], results: list[ExecResult]) -> list[float]:
        """Rate all not-yet-rated outputs of one plan in a single LLM call.

        The judge still scores every item on every criterion; grouping the items
        only saves round trips, which matters under a requests-per-day quota.
        """
        todo = [(q, r) for q, r in zip(sample, results) if (plan_fp, q["qid"]) not in self._ratings]
        if todo:
            crits = self.criteria()
            names = [c["name"] for c in crits]
            crit_txt = "\n".join(f"- {c['name']}: {c.get('description', '')}" for c in crits)
            items = "\n\n".join(
                f"Item {k}\nQuestion: {q['question']}\nContext the pipeline used:\n{context_text(r, chars=300)}\n"
                f"Output:\n{output_text(r)[:500]}" for k, (q, r) in enumerate(todo, 1))
            msg = prompts.RATE_BATCH.format(task=self.task_text, criteria=crit_txt, items=items, n=len(todo))
            r = self.llm.complete(msg, purpose="validation", task="rate_batch", system=prompts.SYSTEM_AGENT,
                                  payload={"items": [{"question": q["question"], "output": output_text(x),
                                                      "context": x.context} for q, x in todo], "criteria": names},
                                  max_tokens=60 + 45 * len(todo), json_mode=True)
            parsed = (r.json({}) or {}).get("items") or []
            by_id = {}
            for it in parsed if isinstance(parsed, list) else []:
                try:
                    by_id[int(it.get("id"))] = it.get("scores") or {}
                except (TypeError, ValueError, AttributeError):
                    continue
            for k, (q, _) in enumerate(todo, 1):
                vals = []
                for v in (by_id.get(k) or {}).values():
                    try:
                        vals.append(min(5.0, max(1.0, float(v))))
                    except (TypeError, ValueError):
                        continue
                self._ratings[(plan_fp, q["qid"])] = sum(vals) / len(vals) if vals else 3.0
        return [self._ratings[(plan_fp, q["qid"])] for q in sample]

    def _pairwise(self, q: dict, a: ExecResult, b: ExecResult, swap: bool) -> bool:
        """True if the *candidate* (a) wins."""
        first, second = (b, a) if swap else (a, b)
        msg = prompts.PAIRWISE.format(task=self.task_text, question=q["question"],
                                      context=context_text(a), a=output_text(first), b=output_text(second))
        r = self.llm.complete(msg, purpose="pairwise", task="pairwise", system=prompts.SYSTEM_AGENT,
                              payload={"question": q["question"], "context": a.context + b.context,
                                       "output_a": output_text(first), "output_b": output_text(second)},
                              max_tokens=80, json_mode=True)
        winner = str((r.json({}) or {}).get("winner", "A")).strip().upper()[:1]
        return (winner == "B") if swap else (winner == "A")

    # ------------------------------------------------------------ public API
    def validate(self, cand: Pipeline, inc: Pipeline, sample: list[dict]) -> Verdict:
        t0 = time.perf_counter()
        before = self.llm.ledger.optimization_tokens()
        crits = self.criteria()
        out_c, out_p = self.outputs(cand, sample), self.outputs(inc, sample)
        fc, fp = cand.fingerprint(), inc.fingerprint()
        sc_all, sp_all = self._rate_many(fc, sample, out_c), self._rate_many(fp, sample, out_p)
        items = []
        for q, rc, rp, sc, sp in zip(sample, out_c, out_p, sc_all, sp_all):
            items.append({"qid": q["qid"], "question": q["question"], "candidate": output_text(rc)[:400],
                          "incumbent": output_text(rp)[:400], "score_c": sc, "score_p": sp})
        avg_c = sum(sc_all) / max(len(sc_all), 1)
        avg_p = sum(sp_all) / max(len(sp_all), 1)
        win_rate = None
        if abs(avg_c - avg_p) < self.cfg.pairwise_margin:
            # close call: pairwise comparison on the items where the plans disagree most
            order = sorted(range(len(sample)), key=lambda i: -abs(sc_all[i] - sp_all[i]))
            picks = order[: self.cfg.pairwise_items]
            wins = 0
            for n, i in enumerate(picks):
                won = self._pairwise(sample[i], out_c[i], out_p[i], swap=bool(n % 2))
                items[i]["pairwise"] = "candidate" if won else "incumbent"
                wins += int(won)
            win_rate = wins / max(len(picks), 1)
            prefer = win_rate > 0.5 and avg_c >= avg_p - 0.25
        else:
            prefer = avg_c > avg_p
        spent = self.llm.ledger.optimization_tokens() - before
        note = (f"rating c={avg_c:.2f} vs p={avg_p:.2f}"
                + (f", pairwise win-rate {win_rate:.2f}" if win_rate is not None else "")
                + f", {len(sample)} items, {time.perf_counter() - t0:.1f}s")
        return Verdict(prefer, avg_c, avg_p, crits, win_rate, "validation", 1.0, spent, note, items)

    def tournament(self, plans: list[Pipeline], sample: list[dict]) -> tuple[Pipeline, list[dict]]:
        """DocETL-style selection: rate every plan, pairwise round-robin over the top A."""
        table = []
        for plan in plans:
            try:
                outs = self.outputs(plan, sample)
                fp = plan.fingerprint()
                avg = sum(self._rate_many(fp, sample, outs)) / max(len(sample), 1)
            except BudgetExhausted:
                break
            table.append({"plan": plan, "score": avg, "wins": 0})
        if not table:
            return plans[0], []
        table.sort(key=lambda r: -r["score"])
        top = table[: self.cfg.top_a]
        try:
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    a, b = top[i], top[j]
                    oa, ob = self.outputs(a["plan"], sample), self.outputs(b["plan"], sample)
                    k = (i + j) % len(sample)
                    if self._pairwise(sample[k], oa[k], ob[k], swap=bool((i + j) % 2)):
                        a["wins"] += 1
                    else:
                        b["wins"] += 1
        except BudgetExhausted:
            pass
        best = max(top, key=lambda r: (r["wins"], r["score"]))
        return best["plan"], [{"plan": r["plan"].describe(), "score": r["score"], "wins": r["wins"]} for r in table]
