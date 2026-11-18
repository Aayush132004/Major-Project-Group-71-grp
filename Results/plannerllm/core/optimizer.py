"""PlannerLLM - Algorithm 1 of the methodology, plus the comparison systems.

    1: cands <- {P} U ApplyDirectives(P, D)
    2: for c in cands: c.score <- C^(c) - lambda * M.predAccuracyGain(c)
    3: sort cands by score; take top-m
    4: for c in top-m:
    5:    if M.signatureHit(c): verdict <- M.lookup(c)          (no LLM call)
    6:    elif VoI(c, incumbent) > theta * tokenCost(validation):
    7:        verdict <- ValidationAgent(c, sample)
    8:        M.write(c, contextSignature, verdict)
    9:    if verdict == prefer(c): incumbent <- c
   10: return incumbent

The loop is repeated for a few rounds from the new incumbent (DocETL's
opportunistic top-down traversal).  Every decision is recorded as an event;
the UI streams them live and the report quotes them as transcripts.
"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from ..llm.base import BudgetExhausted
from .directives import Candidate
from .pipeline import Pipeline
from .sampling import sample_queries
from .session import Session, mean_observed
from .signature import context_features, numeric_vector, signature_of
from .validation import Verdict

EventFn = Callable[[dict], None]


@dataclass
class AuditEntry:
    round: int
    directive: str
    detail: str
    plan: str
    score: float
    cost_usd: float
    source: str           # memory | validation | gate-skip | cost-only | budget | random | tournament
    verdict: str          # prefer | reject | skip
    note: str = ""
    gate: dict | None = None
    memory: dict | None = None
    validation: dict | None = None
    signature: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizationResult:
    system: str
    task_id: str
    base: Pipeline
    plan: Pipeline
    audit: list[AuditEntry] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    opt_tokens: int = 0
    opt_usd: float = 0.0
    opt_calls: int = 0
    opt_api_calls: int = 0
    n_candidates: int = 0
    n_validations: int = 0
    n_memory_hits: int = 0
    n_gate_skips: int = 0
    wall_s: float = 0.0
    budget_exhausted: bool = False

    def summary(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("base", "plan", "audit", "events")} | {
            "plan": self.plan.describe(), "base": self.base.describe(), "lineage": self.plan.lineage,
            "plan_yaml": self.plan.to_yaml(), "plan_ops": self.plan.to_dict()["operators"],
            "base_ops": self.base.to_dict()["operators"]}


def diverse_top(scored: list, m: int) -> list:
    """Top-m by score, but one candidate per directive first, so two variants of
    the same rewrite (e.g. k/2 and 2k) cannot crowd every other idea out."""
    first, rest, seen = [], [], set()
    for item in scored:
        (rest if item[1].directive in seen else first).append(item)
        seen.add(item[1].directive)
    picked = (first + rest)[:m]
    return sorted(picked, key=lambda t: t[0])


class _Run:
    """Shared bookkeeping: ledger diff, budget window, event emission."""

    def __init__(self, s: Session, system: str, task, on_event: EventFn | None) -> None:
        self.s, self.system, self.task, self.on_event = s, system, task, on_event
        self.base = s.workload.base_pipeline()
        self.res = OptimizationResult(system, task.task_id, self.base, self.base)
        self.t0 = time.perf_counter()
        self.start = s.ledger.totals(optimization_only=True)
        self._saved_budget = s.ledger.optimization_budget
        if s.settings.budget.optimization_tokens is not None:
            s.ledger.optimization_budget = s.ledger.optimization_tokens() + s.settings.budget.optimization_tokens
        s.fresh_validator()

    def emit(self, kind: str, **data) -> None:
        ev = {"type": kind, "t": round(time.perf_counter() - self.t0, 3), **data}
        self.res.events.append(ev)
        if self.on_event:
            self.on_event(ev)

    def finish(self, plan: Pipeline) -> OptimizationResult:
        end = self.s.ledger.totals(optimization_only=True)
        r = self.res
        r.plan = plan
        r.opt_tokens = end["tokens"] - self.start["tokens"]
        r.opt_usd = end["usd"] - self.start["usd"]
        r.opt_calls = end["calls"] - self.start["calls"]
        r.opt_api_calls = end["api_calls"] - self.start["api_calls"]
        r.wall_s = time.perf_counter() - self.t0
        self.s.ledger.optimization_budget = self._saved_budget
        self.emit("done", plan=plan.describe(), lineage=plan.lineage, tokens=r.opt_tokens,
                  validations=r.n_validations, memory_hits=r.n_memory_hits, gate_skips=r.n_gate_skips)
        return r


class PlannerLLM:
    """The proposed optimizer with its ablation switches."""

    def __init__(self, session: Session, *, use_gate: bool = True, use_memory: bool = True,
                 use_validation: bool = True, shadow_gate: bool = False, audit_memory: bool = False,
                 cost_mode: str | None = None, theta: float | None = None, name: str = "plannerllm") -> None:
        self.s = session
        self.use_gate, self.use_memory, self.use_validation = use_gate, use_memory, use_validation
        self.shadow_gate = shadow_gate        # log the gate's decision but validate anyway (Sec. 10.3)
        self.audit_memory = audit_memory      # re-validate memory hits to score reuse accuracy (Sec. 10.4)
        self.cost_mode = cost_mode
        self.theta = theta
        self.name = name
        self.memory_audit: list[dict] = []

    def _cost(self, plan: Pipeline, stats: dict):
        return self.s.cost_model.predict(plan, stats, self.cost_mode)

    def _log_telemetry(self, plan: Pipeline, stats: dict, results) -> None:
        if not results:
            return
        cm = self.s.cost_model
        sym = cm.symbolic(plan, stats)
        usd, ms = mean_observed(self.s, results)
        self.s.memory.add_telemetry(self.s.workload.name, plan.fingerprint(), cm.features(plan, stats, sym),
                                    sym.usd, sym.latency_ms, usd, ms)

    # ------------------------------------------------------------------
    def optimize(self, task, on_event: EventFn | None = None) -> OptimizationResult:
        s, cfg = self.s, self.s.settings
        run = _Run(s, self.name, task, on_event)
        stats, fam = task.stats, s.workload.family
        ctx = s.directive_context(stats)
        incumbent = run.base
        cost_inc = self._cost(incumbent, stats)
        tried = {incumbent.fingerprint()}
        run.emit("start", task=task.task_id, plan=incumbent.describe(), yaml=incumbent.to_yaml(),
                 ops=incumbent.to_dict()["operators"],
                 cost=cost_inc.to_dict(), stats=stats)
        try:
            for rnd in range(cfg.search.rounds):
                cands = [c for c in s.agent.propose(incumbent, ctx, cfg.search.max_candidates)
                         if c.plan.fingerprint() not in tried]
                if not cands:
                    break
                run.res.n_candidates += len(cands)
                scored = []
                for c in cands:
                    cost = self._cost(c.plan, stats)
                    # the agent's nomination is a weak prior on accuracy gain
                    prior = cfg.gate.prior_gain * (1.5 if c.proposed_by == "agent" else 1.0)
                    gain = (s.memory.predicted_gain(c.directive, fam, prior)[0]
                            if self.use_memory else prior)
                    score = cfg.gate.reuse_runs * cost.usd - cfg.search.lambda_gain * cfg.gate.w_acc * gain
                    scored.append((score, c, cost, gain))
                scored.sort(key=lambda t: t[0])
                run.emit("candidates", round=rnd, proposal=s.agent.last_proposal, items=[
                    {"directive": c.directive, "detail": c.detail, "plan": c.plan.describe(),
                     "by": c.proposed_by, "cost_usd": cost.usd, "latency_ms": cost.latency_ms,
                     "pred_gain": gain, "score": sc,
                     "shortlisted": any(c is x[1] for x in diverse_top(scored, cfg.search.top_m))}
                    for sc, c, cost, gain in scored])
                shortlist = diverse_top(scored, cfg.search.top_m)
                changed = False
                for score, c, cost_c, _ in shortlist:
                    tried.add(c.plan.fingerprint())
                    verdict, entry = self._decide(run, rnd, c, incumbent, cost_c, cost_inc, score, task)
                    run.res.audit.append(entry)
                    if verdict is None:
                        continue
                    if verdict.prefer_candidate:
                        incumbent, cost_inc, changed = c.plan, cost_c, True
                        run.emit("incumbent", round=rnd, plan=incumbent.describe(), directive=c.directive,
                                 yaml=incumbent.to_yaml(), ops=incumbent.to_dict()["operators"])
                retrain = s.memory.maybe_retrain(s.cost_model)
                if retrain:
                    run.emit("retrain", **{k: v for k, v in retrain.items() if k != "gate_calibration"})
                if not changed:
                    break
        except BudgetExhausted as exc:
            run.res.budget_exhausted = True
            run.emit("budget", message=str(exc))
        return run.finish(incumbent)

    # ------------------------------------------------------------------
    def _decide(self, run: _Run, rnd: int, c: Candidate, inc: Pipeline, cost_c, cost_inc, score, task):
        s, cfg = self.s, self.s.settings
        stats, fam = task.stats, s.workload.family
        feats = context_features(inc, c.plan, c.directive, stats)
        sig, vec = signature_of(feats), numeric_vector(stats, c.plan)
        entry = AuditEntry(rnd, c.directive, c.detail, c.plan.describe(), score, cost_c.usd,
                           source="", verdict="", signature=sig)

        hit = (s.memory.lookup(sig, vec, c.directive, feats["parent_shape"], fam, feats)
               if self.use_memory else None)
        if hit is not None and s.memory.usable(hit):
            run.res.n_memory_hits += 1
            s.memory.touch(hit.sig)
            v = Verdict(hit.prefer, source="memory", confidence=hit.confidence,
                        note=f"{hit.match} signature hit, confidence {hit.confidence:.2f} "
                             f"(age {hit.age_days:.1f}d, drift {hit.drift:.2f}, n={hit.n_obs})")
            entry.source, entry.verdict, entry.note, entry.memory = "memory", \
                "prefer" if v.prefer_candidate else "reject", v.note, hit.to_dict()
            run.emit("memory_hit", round=rnd, plan=c.plan.describe(), **hit.to_dict())
            if self.audit_memory and self.use_validation:
                sample = sample_queries(task.opt_queries, c.plan, stats, cfg.sampling, salt=task.task_id)
                fresh = s.validator.validate(c.plan, inc, sample)
                self.memory_audit.append({"task": task.task_id, "directive": c.directive,
                                          "match": hit.match, "confidence": hit.confidence,
                                          "reused": hit.prefer, "fresh": fresh.prefer_candidate})
            return v, entry

        if not self.use_validation:   # cost-only baseline: trust the cost model alone
            v = Verdict(cost_c.usd < cost_inc.usd, source="cost-only",
                        note=f"cost ${cost_c.usd:.6f} vs incumbent ${cost_inc.usd:.6f}")
            entry.source, entry.verdict, entry.note = "cost-only", "prefer" if v.prefer_candidate else "reject", v.note
            return v, entry

        sample = sample_queries(task.opt_queries, c.plan, stats, cfg.sampling, salt=task.task_id)
        judge_p = s.judge.prefer_probability(c.plan, inc, sample[:4]) if (self.use_gate or self.shadow_gate) else None
        crit_n = len(s.validator._criteria or []) or cfg.validation.max_criteria
        gd = s.gate.decide(c.directive, fam, c.plan, inc, stats, cost_c, cost_inc, judge_p,
                           len(sample), crit_n, theta=self.theta)
        entry.gate = gd.to_dict()
        gate_active = self.use_gate and not self.shadow_gate
        log_id = s.memory.log_gate(c.directive, gd.p_raw, gd.p_flip, gd.voi, gd.threshold,
                                   gd.validate, None, s.workload.name)
        run.emit("gate", round=rnd, directive=c.directive, plan=c.plan.describe(), **gd.to_dict(),
                 enforced=gate_active)
        if gate_active and not gd.validate:
            run.res.n_gate_skips += 1
            entry.source, entry.verdict, entry.note = "gate-skip", "skip", gd.rationale
            return None, entry

        run.emit("validating", round=rnd, directive=c.directive, n_items=len(sample))
        v = s.validator.validate(c.plan, inc, sample)
        run.res.n_validations += 1
        flipped = v.prefer_candidate
        s.memory.set_gate_outcome(log_id, flipped)
        s.memory.update_flip(c.directive, flipped)
        if judge_p is not None and abs(judge_p - 0.5) > 0.05:
            s.memory.update_agreement(c.directive, (judge_p > 0.5) == flipped)
        s.memory.write(sig, feats, vec, c.directive, flipped, v.acc_delta, cost_c.usd - cost_inc.usd,
                       s.workload.name)
        self._log_telemetry(c.plan, stats, s.validator.outputs(c.plan, sample))
        self._log_telemetry(inc, stats, s.validator.outputs(inc, sample))
        entry.source, entry.verdict, entry.note = "validation", "prefer" if flipped else "reject", v.note
        entry.validation = v.to_dict()
        run.emit("validation", round=rnd, directive=c.directive, plan=c.plan.describe(), **{
            k: val for k, val in v.to_dict().items() if k != "criteria"}, criteria=[c_["name"] for c_ in v.criteria])
        return v, entry


# ====================================================================== baselines
class NoOptimization:
    """B2: run the user's pipeline as written."""

    name = "no_opt"

    def __init__(self, session: Session) -> None:
        self.s = session

    def optimize(self, task, on_event=None) -> OptimizationResult:
        run = _Run(self.s, self.name, task, on_event)
        return run.finish(run.base)


class RandomRewrite:
    """B3: apply one uniformly random applicable directive, no evaluation."""

    name = "random"

    def __init__(self, session: Session, seed: int = 1) -> None:
        self.s, self.seed = session, seed

    def optimize(self, task, on_event=None) -> OptimizationResult:
        run = _Run(self.s, self.name, task, on_event)
        ctx = self.s.directive_context(task.stats)
        ctx.llm = None   # random choice must not spend tokens
        cands = self.s.directives.expand(run.base, ctx)
        if not cands:
            return run.finish(run.base)
        pick = random.Random(f"{self.seed}:{task.task_id}").choice(cands)
        run.res.n_candidates = len(cands)
        run.res.audit.append(AuditEntry(0, pick.directive, pick.detail, pick.plan.describe(), 0.0, 0.0,
                                        "random", "prefer", "picked uniformly at random"))
        return run.finish(pick.plan)


class DocETLStyle:
    """B1: DocETL's agentic optimizer - validate every candidate, pairwise top-A.

    No cost ranking, no gate, no memory: each round all candidates are run on
    the sample, rated, and the top A are compared pairwise; the winner becomes
    the incumbent for the next round.
    """

    name = "docetl"

    def __init__(self, session: Session) -> None:
        self.s = session

    def optimize(self, task, on_event=None) -> OptimizationResult:
        s, cfg = self.s, self.s.settings
        run = _Run(s, self.name, task, on_event)
        ctx = s.directive_context(task.stats)
        incumbent = run.base
        try:
            for rnd in range(cfg.search.rounds):
                cands = s.agent.propose(incumbent, ctx, cfg.search.max_candidates)
                if not cands:
                    break
                run.res.n_candidates += len(cands)
                sample = sample_queries(task.opt_queries, incumbent, task.stats, cfg.sampling, salt=task.task_id)
                best, table = s.validator.tournament([incumbent] + [c.plan for c in cands], sample)
                run.res.n_validations += len(table)
                for c in cands:
                    run.res.audit.append(AuditEntry(rnd, c.directive, c.detail, c.plan.describe(), 0.0, 0.0,
                                                    "tournament", "prefer" if c.plan is best else "reject"))
                run.emit("tournament", round=rnd, table=table)
                if best is incumbent:
                    break
                incumbent = best
        except BudgetExhausted as exc:
            run.res.budget_exhausted = True
            run.emit("budget", message=str(exc))
        return run.finish(incumbent)


def make_system(name: str, session: Session, **kw):
    """Factory for every system / ablation used in the experiments."""
    table = {
        "no_opt": lambda: NoOptimization(session),
        "random": lambda: RandomRewrite(session),
        "cost_only": lambda: PlannerLLM(session, use_validation=False, use_memory=False, name="cost_only"),
        "docetl": lambda: DocETLStyle(session),
        "plannerllm": lambda: PlannerLLM(session, name="plannerllm", **kw),
        "no_gate": lambda: PlannerLLM(session, use_gate=False, name="no_gate", **kw),
        "no_memory": lambda: PlannerLLM(session, use_memory=False, name="no_memory", **kw),
        "symbolic_only": lambda: PlannerLLM(session, cost_mode="symbolic", name="symbolic_only", **kw),
        "neural_only": lambda: PlannerLLM(session, cost_mode="neural", name="neural_only", **kw),
    }
    if name not in table:
        raise KeyError(f"unknown system '{name}'")
    return table[name]()


SYSTEMS = ["no_opt", "random", "cost_only", "docetl", "plannerllm"]
ABLATIONS = ["plannerllm", "no_gate", "no_memory", "symbolic_only", "neural_only"]
