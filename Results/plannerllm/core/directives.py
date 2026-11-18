"""Rewrite directives (Section 5.1).

Each directive is a small, named plan transformation.  Logical directives
change *what* the LLM is asked to do (decompose, merge, prompt rewrite);
physical ones change *how* data is accessed (index family, top-k); the
retrieval-subplan family added by PlannerLLM moves predicates in and out of
the ANN probe and adds/removes a rerank stage ("and vice versa").  SQL
directives cover the text-to-SQL workloads.

A directive never decides whether its rewrite is good - that is the job of
the cost model, the plan memory and the gated validation agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import pipeline as P
from .pipeline import Operator, Pipeline

MAP_INSTRUCTION = ("Copy the sentence(s) from the text that help answer the question, word for word. "
                   "If none do, reply NONE.")


@dataclass
class DirectiveContext:
    workload: str
    family: str
    task_text: str
    filter_template: str
    stats: dict
    llm: object | None = None


@dataclass
class Candidate:
    plan: Pipeline
    directive: str
    detail: str = ""
    proposed_by: str = "rules"   # rules | agent


@dataclass
class Directive:
    name: str
    family: str          # retrieval | physical | logical | sql
    description: str
    fn: Callable[[Pipeline, DirectiveContext], list[tuple[Pipeline, str]]] = field(repr=False)

    def apply(self, plan: Pipeline, ctx: DirectiveContext) -> list[Candidate]:
        out = []
        for new, detail in self.fn(plan, ctx):
            new.lineage = plan.lineage + [self.name]
            try:
                new.validate()
            except P.PipelineError:
                continue
            out.append(Candidate(new, self.name, detail))
        return out


# ------------------------------------------------------------------ helpers
def _first_llm_op(plan: Pipeline) -> Operator | None:
    return next((o for o in plan.topological() if o.uses_llm), None)


def _terminal_llm(plan: Pipeline) -> Operator | None:
    llm_ops = [o for o in plan.topological() if o.kind in ("generate", "sql_generate")]
    return llm_ops[-1] if llm_ops else None


# ------------------------------------------------------------------ retrieval family
def add_rerank(plan: Pipeline, ctx: DirectiveContext):
    r = plan.find("retrieve")
    if r is None or plan.find("rerank") is not None or ctx.family == "sql-heavy":
        return []
    p = plan.clone()
    r2 = p.op(r.name)
    keep = int(r2.params.get("top_k", 10))
    r2.params["top_k"] = min(max(keep * 2, 20), 40)
    anchor = _first_llm_op(p)
    p.insert_before(anchor.name, Operator(p.unique_name("rerank"), "rerank", {"top_k": keep}))
    return [(p, f"retrieve top_k {keep}->{r2.params['top_k']}, cross-encoder keeps {keep}")]


def drop_rerank(plan: Pipeline, ctx: DirectiveContext):
    rr, r = plan.find("rerank"), plan.find("retrieve")
    if rr is None or r is None:
        return []
    p = plan.clone()
    p.op(r.name).params["top_k"] = int(rr.params.get("top_k", 5))
    p.remove(rr.name)
    return [(p, "single-stage probe, top_k = former rerank k")]


def push_filter_into_probe(plan: Pipeline, ctx: DirectiveContext):
    """filter (before or after) + retrieve  ==>  retrieve[where]  (filtered vector probe)."""
    r, f = plan.find("retrieve"), plan.find("filter")
    if r is None or f is None or r.params.get("where"):
        return []
    p = plan.clone()
    where = p.op(f.name).params["where"]
    p.remove(f.name)
    p.op(r.name).params["where"] = where
    return [(p, f"predicate '{where}' evaluated inside the ANN probe")]


def pull_filter_out_of_probe(plan: Pipeline, ctx: DirectiveContext):
    """retrieve[where]  ==>  retrieve(over-fetch) -> filter   (post-filtering)."""
    r = plan.find("retrieve")
    if r is None or not r.params.get("where"):
        return []
    p = plan.clone()
    r2 = p.op(r.name)
    where = r2.params.pop("where")
    r2.params["top_k"] = min(int(r2.params.get("top_k", 10)) * 3, 60)
    p.insert_after(r2.name, Operator(p.unique_name("post_filter"), "filter", {"where": where}))
    return [(p, f"probe without predicate (k x3), then filter '{where}'")]


def add_filter(plan: Pipeline, ctx: DirectiveContext):
    """Insert an early metadata predicate taken from the query's structured fields."""
    r = plan.find("retrieve")
    if r is None or plan.find("filter") is not None or r.params.get("where"):
        return []
    p = plan.clone()
    p.insert_before(r.name, Operator(p.unique_name("pre_filter"), "filter", {"where": ctx.filter_template}))
    return [(p, f"pre-filter rows with '{ctx.filter_template}' before the probe")]


# ------------------------------------------------------------------ physical family
def swap_index(plan: Pipeline, ctx: DirectiveContext):
    r = plan.find("retrieve")
    if r is None:
        return []
    cur = r.params.get("index", "hnsw")
    out = []
    for target in {"hnsw": ("ivf", "flat"), "ivf": ("hnsw",), "flat": ("hnsw",)}.get(cur, ("hnsw",)):
        p = plan.clone()
        p.op(r.name).params["index"] = target
        out.append((p, f"index {cur} -> {target}"))
    return out


def tune_topk(plan: Pipeline, ctx: DirectiveContext):
    r = plan.find("rerank") or plan.find("retrieve")
    if r is None:
        return []
    k = int(r.params.get("top_k", 10))
    out = []
    for new_k in sorted({max(3, k // 2), min(40, k * 2)} - {k}):
        p = plan.clone()
        p.op(r.name).params["top_k"] = new_k
        if r.kind == "rerank":
            rr = p.find("retrieve")
            if rr is not None and int(rr.params.get("top_k", 10)) < new_k:
                rr.params["top_k"] = new_k * 2
        out.append((p, f"{r.kind} top_k {k} -> {new_k}"))
    return out


# ------------------------------------------------------------------ logical family
def decompose_generate(plan: Pipeline, ctx: DirectiveContext):
    """generate  ==>  map(extract per record) -> generate   (DocETL chaining)."""
    g = plan.find("generate")
    if g is None or plan.find("map") is not None:
        return []
    p = plan.clone()
    # extract only from the records the final call will read; the rest would be wasted calls
    cap = int(g.params.get("max_docs", 6))
    p.insert_before(g.name, Operator(p.unique_name("extract"), "map", {"max_records": cap}, prompt=MAP_INSTRUCTION))
    return [(p, "per-record extraction before the final answer")]


def merge_map(plan: Pipeline, ctx: DirectiveContext):
    """map -> generate  ==>  generate   (fusion: fewer LLM round-trips)."""
    m, g = plan.find("map"), plan.find("generate")
    if m is None or g is None:
        return []
    p = plan.clone()
    p.remove(m.name)
    return [(p, "fold extraction into the final generate call")]


def prompt_rewrite(plan: Pipeline, ctx: DirectiveContext):
    target = _terminal_llm(plan)
    if target is None or ctx.llm is None or "PROMPT_REWRITE" in plan.lineage:
        return []
    from ..engine import prompts
    from ..llm.base import BudgetExhausted
    msg = prompts.PROMPT_REWRITE.format(kind=target.kind, task=ctx.task_text, prompt=target.prompt)
    try:
        r = ctx.llm.complete(msg, purpose="prompt_rewrite", task="prompt_rewrite",
                             payload={"prompt": target.prompt}, system=prompts.SYSTEM_AGENT,
                             max_tokens=220, json_mode=True)
    except BudgetExhausted:
        return []
    new_prompt = (r.json({}) or {}).get("prompt", "")
    if not isinstance(new_prompt, str) or len(new_prompt) < 20 or new_prompt.strip() == target.prompt.strip():
        return []
    p = plan.clone()
    p.op(target.name).prompt = new_prompt.strip()[:600]
    return [(p, f"agent-rewritten instruction for '{target.name}'")]


# ------------------------------------------------------------------ SQL family
def add_schema_link(plan: Pipeline, ctx: DirectiveContext):
    g = plan.find("sql_generate")
    if g is None or plan.find("retrieve") is not None:
        return []
    p = plan.clone()
    p.insert_before(g.name, Operator(p.unique_name("link_schema"), "retrieve",
                                     {"top_k": 4, "index": "flat", "where": ctx.filter_template}))
    return [(p, "retrieve the 4 most relevant tables instead of the whole schema")]


def add_fewshot(plan: Pipeline, ctx: DirectiveContext):
    g = plan.find("sql_generate")
    if g is None or int(g.params.get("fewshot_k", 0)) > 0:
        return []
    p = plan.clone()
    p.op(g.name).params["fewshot_k"] = 3
    return [(p, "3 nearest training question/SQL pairs as demonstrations")]


def add_sql_repair(plan: Pipeline, ctx: DirectiveContext):
    g = plan.find("sql_generate")
    if g is None or plan.find("sql_repair") is not None:
        return []
    p = plan.clone()
    p.insert_after(g.name, Operator(p.unique_name("repair_sql"), "sql_repair"))
    return [(p, "execute once; on error or empty result ask the LLM to fix the SQL")]


class DirectiveStore:
    """Registry of all directives with a human-readable catalog for the agent."""

    def __init__(self) -> None:
        self._d: dict[str, Directive] = {}
        for name, fam, desc, fn in [
            ("ADD_RERANK", "retrieval", "two-stage plan: over-fetch with ANN, keep top-k by cross-encoder", add_rerank),
            ("DROP_RERANK", "retrieval", "single-stage plan: remove the rerank stage", drop_rerank),
            ("FILTERED_PROBE", "retrieval", "evaluate the metadata predicate inside the ANN probe", push_filter_into_probe),
            ("POST_FILTER", "retrieval", "probe without the predicate, then filter (over-fetch x3)", pull_filter_out_of_probe),
            ("ADD_FILTER", "logical", "add an early metadata predicate from the query's fields", add_filter),
            ("SWAP_INDEX", "physical", "change the ANN index family (hnsw / ivf / flat)", swap_index),
            ("TUNE_TOPK", "physical", "halve or double how many records reach the LLM", tune_topk),
            ("DECOMPOSE", "logical", "extract relevant sentences per record before answering", decompose_generate),
            ("MERGE_MAP", "logical", "fuse the extraction step into the final LLM call", merge_map),
            ("PROMPT_REWRITE", "logical", "let the agent rewrite the final operator's instruction", prompt_rewrite),
            ("SCHEMA_LINK", "sql", "retrieve only the relevant tables before writing SQL", add_schema_link),
            ("FEWSHOT", "sql", "add retrieved question/SQL demonstrations", add_fewshot),
            ("SQL_REPAIR", "sql", "execute and repair failing or empty SQL", add_sql_repair),
        ]:
            self.register(Directive(name, fam, desc, fn))

    def register(self, d: Directive) -> None:
        self._d[d.name] = d

    def get(self, name: str) -> Directive:
        return self._d[name]

    def names(self) -> list[str]:
        return list(self._d)

    def catalog(self) -> list[dict]:
        return [{"name": d.name, "family": d.family, "description": d.description} for d in self._d.values()]

    def applicable(self, plan: Pipeline, ctx: DirectiveContext) -> list[str]:
        """Directives whose cheap (non-LLM) form produces at least one candidate."""
        names = []
        probe_ctx = DirectiveContext(**{**ctx.__dict__, "llm": None})
        for d in self._d.values():
            if d.name == "PROMPT_REWRITE":
                if _terminal_llm(plan) is not None and "PROMPT_REWRITE" not in plan.lineage:
                    names.append(d.name)
                continue
            if d.fn(plan, probe_ctx):
                names.append(d.name)
        return names

    def expand(self, plan: Pipeline, ctx: DirectiveContext, names: list[str] | None = None,
               cap: int | None = None) -> list[Candidate]:
        out: list[Candidate] = []
        seen = {plan.fingerprint()}
        for name in names or self.names():
            if name not in self._d or (cap is not None and len(out) >= cap):
                continue
            for cand in self._d[name].apply(plan, ctx):
                fp = cand.plan.fingerprint()
                if fp not in seen and (cap is None or len(out) < cap):
                    seen.add(fp)
                    out.append(cand)
        return out
