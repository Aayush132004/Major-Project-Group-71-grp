"""Generation agent: chooses which directives to apply (Section 5.1).

The LLM sees the current plan, the data statistics and the catalog of
directives that are actually applicable, and nominates the promising ones.
Its picks are tried first; the remaining applicable directives follow, so a
bad or empty reply never leaves the search without candidates (the
rule-based fallback the base system also relies on).
"""
from __future__ import annotations

from ..engine import prompts
from ..llm.base import BudgetExhausted
from .directives import Candidate, DirectiveContext, DirectiveStore
from .pipeline import Pipeline

STAT_KEYS = ("n_rows", "sel_est", "avg_doc_tokens", "n_tables", "avg_query_tokens")


class GenerationAgent:
    def __init__(self, llm, store: DirectiveStore, use_llm: bool = True) -> None:
        self.llm = llm
        self.store = store
        self.use_llm = use_llm
        self.last_proposal: list[dict] = []

    def propose(self, plan: Pipeline, ctx: DirectiveContext, cap: int) -> list[Candidate]:
        applicable = self.store.applicable(plan, ctx)
        chosen: list[str] = []
        self.last_proposal = []
        if self.use_llm and applicable:
            catalog = "\n".join(f"- {d['name']}: {d['description']}" for d in self.store.catalog()
                                if d["name"] in applicable)
            stats = {k: ctx.stats[k] for k in STAT_KEYS if k in ctx.stats}
            msg = prompts.PROPOSE.format(task=ctx.task_text, pipeline=plan.describe(), stats=stats, catalog=catalog)
            try:
                r = self.llm.complete(msg, purpose="generation_agent", task="propose",
                                      payload={"directives": applicable, "pipeline": plan.describe()},
                                      system=prompts.SYSTEM_AGENT, max_tokens=300, json_mode=True)
                picks = (r.json({}) or {}).get("directives") or []
                for p in picks:
                    name = (p.get("name") if isinstance(p, dict) else str(p)).strip().upper()
                    if name in applicable and name not in chosen:
                        chosen.append(name)
                        self.last_proposal.append({"name": name, "why": p.get("why", "") if isinstance(p, dict) else ""})
            except BudgetExhausted:
                pass
        order = chosen + [n for n in applicable if n not in chosen]
        cands = self.store.expand(plan, ctx, order, cap)
        for c in cands:
            c.proposed_by = "agent" if c.directive in chosen else "rules"
        return cands[:cap]
