"""Experiment suite for Section 10 (one function per subsection).

Every experiment writes ``runs/<backend>/<experiment>.json`` so the dashboard
and ``reporting.py`` can render it without re-running anything.  Runs are
resumable for free: every LLM response is cached on disk, so re-running an
interrupted experiment replays finished calls instantly.

    E1 baselines        10.1 / 10.2 table: accuracy, spend, latency per system
    E2 pareto           10.2 accuracy vs optimisation-token budget, theta sweep
    E3 gate             10.3 gate precision / recall / calibration (shadow mode)
    E4 memory           10.4 hit rate, reuse accuracy, decay, perturbed shapes
    E5 cost_model       10.5 calibration error: symbolic / neural / hybrid, zero-shot
    E6 latency          10.6 wall-clock decomposition, ANN family sweep
    E7 ablation         10.7 -gate, -memory, symbolic-only, neural-only
    E8 transcripts      10.8 validation transcripts for accepted rewrites
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.directives import DirectiveContext
from ..core.memory import PlanMemory
from ..core.optimizer import ABLATIONS, SYSTEMS, make_system
from ..core.session import Session, build_session, mean_observed
from ..settings import RUNS_DIR, Settings

PROFILES = {   # groups per workload, optimisation questions, held-out questions
    "smoke": {"n_tasks": 1, "n_opt": 5, "n_test": 5},
    "quick": {"n_tasks": 2, "n_opt": 6, "n_test": 6},
    "full": {"n_tasks": 4, "n_opt": 8, "n_test": 10},
}
WORKLOADS = ["msmarco", "cuad", "spider"]
Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def out_dir(settings: Settings) -> Path:
    tag = "sim" if settings.llm.backend == "sim" else settings.llm.model.replace("/", "_")
    d = RUNS_DIR / f"{tag}_seed{settings.sampling.task_seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(settings: Settings, name: str, payload: dict) -> Path:
    payload = {"experiment": name, "backend": settings.llm.backend,
               "model": settings.llm.model if settings.llm.backend != "sim" else "simulated-extractive",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": settings.sampling.task_seed,
               "sampling_seed": settings.sampling.seed, **payload}
    path = out_dir(settings) / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    return path


def load(settings: Settings, name: str) -> dict | None:
    path = out_dir(settings) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


class LogicalClock:
    """Days advance per optimised task, so decay is reproducible."""

    def __init__(self, days_per_task: float = 1.0) -> None:
        self.day, self.step = 0.0, days_per_task

    def __call__(self) -> float:
        return self.day

    def tick(self) -> None:
        self.day += self.step


def _session(wl: str, settings: Settings, clock: LogicalClock | None = None, **kw) -> Session:
    clock = clock or LogicalClock()
    s = build_session(wl, settings, clock=clock, **kw)
    s.clock = clock   # type: ignore[attr-defined]
    return s


def evaluate_plan(s: Session, plan, queries: list[dict], log_telemetry: bool = True) -> dict:
    results = [s.executor.run(plan, q, purpose="execute") for q in queries]
    sc = s.oracle.score(queries, results)
    usd, ms = mean_observed(s, results)
    br = [r.breakdown() for r in results]
    sc.update(exec_usd=usd, exec_ms=ms,
              db_ms=float(np.mean([b["db_ms"] for b in br])), ann_ms=float(np.mean([b["ann_ms"] for b in br])),
              llm_ms=float(np.mean([b["llm_ms"] for b in br])), exec_tokens=float(np.mean([r.tokens for r in results])))
    if log_telemetry:
        cm = s.cost_model
        stats = s.workload.group_stats(queries)
        sym = cm.symbolic(plan, stats)
        s.memory.add_telemetry(s.workload.name, plan.fingerprint(), cm.features(plan, stats, sym),
                               sym.usd, sym.latency_ms, usd, ms)
    return sc


def run_stream(s: Session, system: str, tasks, progress: Progress = _noop, passes: int = 1,
               on_event=None, **kw) -> list[dict]:
    """Optimise each task in order (memory carries over), then score on held-out queries."""
    sysm = make_system(system, s, **kw)
    rows = []
    for p in range(passes):
        for t in tasks:
            progress(f"{system} | {t.task_id} | pass {p + 1}")
            lat0 = s.ledger.totals(optimization_only=True)["latency_s"]
            r = sysm.optimize(t, on_event=on_event)
            opt_llm_s = s.ledger.totals(optimization_only=True)["latency_s"] - lat0
            sc = evaluate_plan(s, r.plan, t.test_queries)
            clock = getattr(s, "clock", None)
            if clock is not None:
                clock.tick()
            per_query = sc.pop("per_query")
            rows.append({"system": system, "workload": s.workload.name, "task": t.task_id, "pass": p + 1,
                         "seed": s.settings.sampling.task_seed, **r.summary(), **sc, "opt_llm_s": opt_llm_s,
                         "metric": per_query[0].get("metric") if per_query else None, "per_query": per_query,
                         "audit": [a.to_dict() for a in r.audit],
                         "memory_audit": list(getattr(sysm, "memory_audit", []))})
    return rows


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarise(rows: list[dict], keys=("accuracy", "f1", "precision", "recall", "rouge_l", "hallucinated",
                                      "exec_acc", "opt_llm_s",
                                      "recall_at_k", "opt_tokens", "opt_usd", "exec_usd", "exec_ms",
                                      "exec_tokens", "n_validations", "n_memory_hits", "n_gate_skips",
                                      "wall_s", "db_ms", "ann_ms", "llm_ms")) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["workload"], r["system"]), []).append(r)
    out = []
    for (wl, sysname), rs in groups.items():
        row = {"workload": wl, "system": sysname, "n_tasks": len(rs)}
        for k in keys:
            row[k] = _mean(rs, k)
        out.append(row)
    return out


# ====================================================================== E1
def exp_baselines(settings: Settings, profile: str = "quick", workloads=WORKLOADS,
                  systems=SYSTEMS, progress: Progress = _noop) -> dict:
    prof = PROFILES[profile]
    rows = []
    for wl in workloads:
        for system in systems:
            s = _session(wl, settings)
            tasks = s.workload.tasks(**prof, seed=settings.sampling.task_seed)
            rows += run_stream(s, system, tasks, progress)
    # decision quality: agreement with the full-validation (DocETL) choice per task
    ref = {(r["workload"], r["task"]): r["plan"] for r in rows if r["system"] == "docetl"}
    for r in rows:
        if (r["workload"], r["task"]) in ref:
            r["agrees_with_full_validation"] = float(r["plan"] == ref[(r["workload"], r["task"])])
    summary = summarise(rows)
    for srow in summary:
        rs = [r for r in rows if r["workload"] == srow["workload"] and r["system"] == srow["system"]]
        srow["decision_agreement"] = _mean(rs, "agrees_with_full_validation")
    return {"profile": profile, "rows": rows, "summary": summary}


# ====================================================================== E2
def exp_pareto(settings: Settings, profile: str = "quick", workloads=WORKLOADS,
               budgets=(3000, 6000, 12000, 24000, None), thetas=(0.25, 0.5, 1.0, 2.0, 4.0),
               progress: Progress = _noop) -> dict:
    prof = PROFILES[profile]
    points = []
    for wl in workloads:
        for system in ("plannerllm", "docetl", "no_gate"):
            for b in budgets:
                st = settings.copy(**{"budget.optimization_tokens": b})
                s = _session(wl, st)
                rows = run_stream(s, system, s.workload.tasks(**prof, seed=settings.sampling.task_seed), progress)
                points.append({"workload": wl, "system": system, "budget": b, "sweep": "budget",
                               "accuracy": _mean(rows, "accuracy"), "opt_tokens": _mean(rows, "opt_tokens"),
                               "exec_usd": _mean(rows, "exec_usd")})
        for th in thetas:
            st = settings.copy(**{"gate.theta": th})
            s = _session(wl, st)
            rows = run_stream(s, "plannerllm", s.workload.tasks(**prof, seed=settings.sampling.task_seed), progress)
            points.append({"workload": wl, "system": "plannerllm", "theta": th, "sweep": "theta",
                           "accuracy": _mean(rows, "accuracy"), "opt_tokens": _mean(rows, "opt_tokens"),
                           "gate_skips": _mean(rows, "n_gate_skips"), "validations": _mean(rows, "n_validations")})
    return {"profile": profile, "points": points}


# ====================================================================== E3
def gate_scores(log: list[dict], thetas) -> list[dict]:
    """Offline gate P/R: replay logged VoI against theta * validation cost."""
    rows = [g for g in log if g["flipped"] is not None]
    out = []
    for th in thetas:
        admitted = [g for g in rows if g["voi"] > th * g["token_cost"]]
        flips = [g for g in rows if g["flipped"]]
        tp = sum(1 for g in admitted if g["flipped"])
        spend_all = sum(g["token_cost"] for g in rows) or 1.0
        out.append({"theta": th, "n": len(rows), "admitted": len(admitted),
                    "precision": tp / len(admitted) if admitted else None,
                    "recall": tp / len(flips) if flips else None,
                    "flip_rate": len(flips) / len(rows) if rows else None,
                    "spend_saved": 1 - sum(g["token_cost"] for g in admitted) / spend_all})
    return out


def calibration_bins(log: list[dict], n_bins: int = 5) -> list[dict]:
    rows = [g for g in log if g["flipped"] is not None]
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        rs = [g for g in rows if lo <= g["p_used"] < hi or (i == n_bins - 1 and g["p_used"] == 1.0)]
        if rs:
            bins.append({"lo": lo, "hi": hi, "n": len(rs), "predicted": float(np.mean([g["p_used"] for g in rs])),
                         "observed": float(np.mean([g["flipped"] for g in rs]))})
    return bins


def exp_gate(settings: Settings, profile: str = "quick", workloads=WORKLOADS,
             thetas=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0), progress: Progress = _noop) -> dict:
    prof = PROFILES[profile]
    out = {"profile": profile, "per_workload": {}}
    all_log = []
    for wl in workloads:
        s = _session(wl, settings)
        run_stream(s, "plannerllm", s.workload.tasks(**prof, seed=settings.sampling.task_seed), progress, passes=2, shadow_gate=True)
        rep = s.memory.report()
        log = rep["gate_log"]
        all_log += log
        per_dir: dict[str, dict] = {}
        for g in log:
            if g["flipped"] is None:
                continue
            d = per_dir.setdefault(g["directive"], {"n": 0, "flips": 0, "admitted": 0, "p_mean": 0.0})
            d["n"] += 1
            d["flips"] += g["flipped"]
            d["admitted"] += int(g["voi"] > g["token_cost"])
            d["p_mean"] += g["p_used"]
        for d in per_dir.values():
            d["p_mean"] /= max(d["n"], 1)
        out["per_workload"][wl] = {"scores": gate_scores(log, thetas), "calibration": calibration_bins(log),
                                   "per_directive": per_dir, "agreement": rep["agreement"], "log": log}
    out["overall"] = {"scores": gate_scores(all_log, thetas), "calibration": calibration_bins(all_log)}
    return out


# ====================================================================== E4
def _perturb(s: Session, kind: str) -> None:
    """Change the starting pipeline so later queries have a *perturbed shape*."""
    base = s.workload.base_pipeline()
    orig = s.workload.base_pipeline
    if kind == "prompt_edit":
        for o in base.operators:
            if o.prompt:
                o.prompt = o.prompt.rstrip(".") + ". Keep the answer short and precise."
    elif kind == "config_change":
        r = base.find("retrieve")
        if r is not None:
            r.params["top_k"] = int(r.params.get("top_k", 10)) * 3
        g = base.find("sql_generate")
        if g is not None:
            g.params["fewshot_k"] = 1
    s.workload.base_pipeline = lambda: base.clone()  # type: ignore[method-assign]
    s._restore_base = orig   # type: ignore[attr-defined]


def _restore(s: Session) -> None:
    if hasattr(s, "_restore_base"):
        s.workload.base_pipeline = s._restore_base   # type: ignore[method-assign]


def exp_memory(settings: Settings, profile: str = "quick", workloads=WORKLOADS,
               kappas=(0.0, 0.02, 0.1), progress: Progress = _noop) -> dict:
    prof = PROFILES[profile]
    out = {"profile": profile, "streams": [], "decay": [], "reuse": []}
    for wl in workloads:
        # repeated and perturbed shapes, with re-validation of every hit
        s = _session(wl, settings)
        tasks = s.workload.tasks(**prof, seed=settings.sampling.task_seed)
        phases = [("first-seen", None), ("repeated", None), ("prompt_edit", "prompt_edit"),
                  ("config_change", "config_change")]
        for label, pert in phases:
            if pert:
                _perturb(s, pert)
            rows = run_stream(s, "plannerllm", tasks, progress, audit_memory=True)
            _restore(s)
            cands = sum(len(r["audit"]) for r in rows)
            hits = sum(r["n_memory_hits"] for r in rows)
            audit = [m for r in rows for m in r["memory_audit"]]
            out["streams"].append({"workload": wl, "phase": label, "hit_rate": hits / cands if cands else 0.0,
                                   "hits": hits, "candidates": cands, "opt_tokens": _mean(rows, "opt_tokens"),
                                   "accuracy": _mean(rows, "accuracy"),
                                   "reuse_accuracy": (float(np.mean([m["reused"] == m["fresh"] for m in audit]))
                                                      if audit else None), "n_audited": len(audit)})
            out["reuse"] += [{"workload": wl, "phase": label, **m} for m in audit]
        # decay: same stream twice, clock advancing 3 days per task, different kappa
        for kappa in kappas:
            st = settings.copy(**{"memory.kappa": kappa})
            s = _session(wl, st, clock=LogicalClock(3.0))
            rows = run_stream(s, "plannerllm", tasks, progress, passes=2)
            second = [r for r in rows if r["pass"] == 2]
            cands = sum(len(r["audit"]) for r in second)
            out["decay"].append({"workload": wl, "kappa": kappa,
                                 "hit_rate_pass2": sum(r["n_memory_hits"] for r in second) / cands if cands else 0,
                                 "opt_tokens_pass2": _mean(second, "opt_tokens"),
                                 "accuracy_pass2": _mean(second, "accuracy")})
    mem = PlanMemory(kappa=0.02)
    out["decay_curve"] = [{"age_days": a, **{f"kappa={k}": math.exp(-k * a) * (2 / 3) for k in kappas}}
                          for a in range(0, 61, 5)]
    out["reuse_threshold"] = mem.reuse_threshold
    return out


# ====================================================================== E5
def collect_cost_telemetry(s: Session, tasks, n_queries: int = 3, depth: int = 2) -> list[dict]:
    """Execute a spread of rewritten plans to obtain observed costs."""
    rows = []
    for t in tasks:
        ctx: DirectiveContext = s.directive_context(t.stats)
        ctx.llm = None
        frontier = [s.workload.base_pipeline()]
        plans = list(frontier)
        for _ in range(depth):
            nxt = []
            for p in frontier:
                nxt += [c.plan for c in s.directives.expand(p, ctx)]
            random.Random(t.task_id).shuffle(nxt)
            frontier = nxt[:6]
            plans += frontier
        seen = set()
        for plan in plans:
            if plan.fingerprint() in seen:
                continue
            seen.add(plan.fingerprint())
            res = [s.executor.run(plan, q, purpose="execute") for q in t.opt_queries[:n_queries]]
            usd, ms = mean_observed(s, res)
            cm = s.cost_model
            sym = cm.symbolic(plan, t.stats)
            rows.append({"workload": s.workload.name, "task": t.task_id, "plan": plan.describe(),
                         "features": cm.features(plan, t.stats, sym), "sym_usd": sym.usd, "sym_ms": sym.latency_ms,
                         "obs_usd": usd, "obs_ms": ms})
    return rows


def _errors(pred: list[float], obs: list[float]) -> dict:
    p, o = np.maximum(np.array(pred), 1e-12), np.maximum(np.array(obs), 1e-12)
    rel = np.abs(p - o) / o
    q = np.maximum(p / o, o / p)   # q-error, the standard cardinality/cost metric
    return {"mae_log10": float(np.mean(np.abs(np.log10(p) - np.log10(o)))),
            "median_rel": float(np.median(rel)), "p90_rel": float(np.percentile(rel, 90)),
            "median_q": float(np.median(q)), "p90_q": float(np.percentile(q, 90)), "n": len(o)}


def exp_cost_model(settings: Settings, profile: str = "quick", workloads=WORKLOADS,
                   progress: Progress = _noop) -> dict:
    from ..core.cost_model import HybridCostModel
    prof = PROFILES[profile]
    data = []
    sessions = {}
    for wl in workloads:
        progress(f"cost telemetry | {wl}")
        s = _session(wl, settings)
        sessions[wl] = s
        data += collect_cost_telemetry(s, s.workload.tasks(**prof, seed=settings.sampling.task_seed))
    results = []

    def fit_eval(train, test, label, target):
        cm = HybridCostModel(settings.llm.price_in_per_m, settings.llm.price_out_per_m,
                             settings.budget.cpu_usd_per_hour)
        ok = cm.fit([r["features"] for r in train], [r["obs_usd"] for r in train], [r["obs_ms"] for r in train],
                    [r["sym_usd"] for r in train], [r["sym_ms"] for r in train], min_samples=8)
        X = np.array([r["features"] for r in test])
        obs = [r[f"obs_{target}"] for r in test]
        sym = [r[f"sym_{target}"] for r in test]
        res = {"split": label, "target": target, "symbolic": _errors(sym, obs)}
        if ok:
            rm, nm = (cm._res_usd, cm._nn_usd) if target == "usd" else (cm._res_ms, cm._nn_ms)
            res["hybrid"] = _errors(list(np.array(sym) * np.exp(rm.predict(X))), obs)
            res["neural"] = _errors(list(np.exp(nm.predict(X[:, :-2]))), obs)
        return res

    rng = random.Random(0)
    for target in ("usd", "ms"):
        # in-distribution: 5-fold CV over all telemetry
        idx = list(range(len(data)))
        rng.shuffle(idx)
        folds = [idx[i::5] for i in range(5)]
        for f in folds:
            test = [data[i] for i in f]
            train = [data[i] for i in idx if i not in set(f)]
            r = fit_eval(train, test, "cv", target)
            if "hybrid" not in r:
                continue
            results.append(r)
        # zero-shot transfer: train on the other workloads only
        for wl in workloads:
            train = [r for r in data if r["workload"] != wl]
            test = [r for r in data if r["workload"] == wl]
            if train and test:
                results.append(fit_eval(train, test, f"zero-shot:{wl}", target))
    # aggregate CV folds
    agg = []
    for target in ("usd", "ms"):
        for split in sorted({r["split"] for r in results}):
            rs = [r for r in results if r["split"] == split and r["target"] == target]
            if not rs:
                continue
            row = {"split": split, "target": target}
            for model in ("symbolic", "neural", "hybrid"):
                ms = [r[model] for r in rs if model in r]
                if ms:
                    row[model] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            agg.append(row)
    scatter = [{"workload": r["workload"], "plan": r["plan"], "sym_usd": r["sym_usd"], "obs_usd": r["obs_usd"],
                "sym_ms": r["sym_ms"], "obs_ms": r["obs_ms"]} for r in data]
    return {"profile": profile, "summary": agg, "folds": results, "scatter": scatter, "n_samples": len(data)}


# ====================================================================== E6
def exp_latency(settings: Settings, profile: str = "quick", workloads=WORKLOADS,
                progress: Progress = _noop) -> dict:
    out = {"profile": profile, "ann_sweep": [], "probe": [], "validation_path": []}
    for wl in workloads:
        s = _session(wl, settings)
        idx = s.workload.index
        rng = np.random.default_rng(0)
        qs = [q["question"] for q in s.workload.queries]
        picks = rng.choice(len(qs), size=min(40, len(qs)), replace=False)
        qv = s.workload.embedder.encode([qs[i] for i in picks])
        for k in (5, 10, 20, 40):
            exact = [set(idx.search(v, k, "flat")[0]) for v in qv]
            for fam, knobs in (("hnsw", [16, 32, 64, 128]), ("ivf", [1, 2, 4, 8, 16]), ("flat", [0])):
                for knob in knobs:
                    times, recalls = [], []
                    for v, ex in zip(qv, exact):
                        pos, _, ms = idx.search(v, k, fam, ef_search=knob or 64, nprobe=knob or 8)
                        times.append(ms)
                        recalls.append(len(ex & set(pos)) / max(len(ex), 1))
                    out["ann_sweep"].append({"workload": wl, "k": k, "family": fam, "knob": knob,
                                             "ms": float(np.median(times)), "recall": float(np.mean(recalls))})
        # filtered probe vs post-filter at this workload's predicate selectivity
        from ..engine.store import render_where
        for q in s.workload.queries[:30]:
            where = render_where(s.workload.filter_template, q["fields"])
            allowed, db_ms = s.workload.store.positions(where)
            v = s.workload.embedder.encode_query(q["question"])
            truth = set(idx.search(v, 10, "flat", allowed=allowed)[0])
            for fam in ("hnsw", "ivf", "flat"):
                pos, _, ms = idx.search(v, 10, fam, allowed=allowed)
                pos2, _, ms2 = idx.search(v, 30, fam)
                post = [p for p in pos2 if p in set(allowed)][:10]
                out["probe"].append({"workload": wl, "family": fam, "sel": len(allowed) / idx.n,
                                     "filtered_ms": ms + db_ms, "filtered_recall": len(truth & set(pos)) / max(len(truth), 1),
                                     "post_ms": ms2 + db_ms, "post_recall": len(truth & set(post)) / max(len(truth), 1)})
    base = load(settings, "baselines")
    if base:
        for r in base["summary"]:
            out["validation_path"].append({k: r.get(k) for k in ("workload", "system", "db_ms", "ann_ms", "llm_ms",
                                                                 "exec_ms", "opt_tokens", "wall_s")})
    # LLM service time spent on the optimisation path, per task, from the baseline runs
    if base:
        for r in base["rows"]:
            out.setdefault("llm_time", []).append({"workload": r["workload"], "system": r["system"],
                                                   "task": r["task"], "opt_llm_s": r.get("opt_llm_s"),
                                                   "opt_tokens": r.get("opt_tokens")})
    return out


# ====================================================================== E7
def exp_ablation(settings: Settings, profile: str = "quick", workloads=WORKLOADS,
                 budget: int | None = 20000, progress: Progress = _noop) -> dict:
    prof = PROFILES[profile]
    rows = []
    st = settings.copy(**{"budget.optimization_tokens": budget})
    for wl in workloads:
        for system in ABLATIONS:
            s = _session(wl, st)
            # warm the residual model and memory with one pass so the learned parts exist
            tasks = s.workload.tasks(**prof, seed=settings.sampling.task_seed)
            rows += [r for r in run_stream(s, system, tasks, progress, passes=2) if r["pass"] == 2]
    return {"profile": profile, "budget": budget, "rows": rows, "summary": summarise(rows)}


# ====================================================================== E8
def exp_transcripts(settings: Settings, limit: int = 12) -> dict:
    base = load(settings, "baselines")
    picks = []
    if base:
        for r in base["rows"]:
            if r["system"] != "plannerllm":
                continue
            for a in r["audit"]:
                if a["source"] in ("validation", "memory", "gate-skip"):
                    picks.append({"workload": r["workload"], "task": r["task"], **a})
    accepted = [p for p in picks if p["verdict"] == "prefer"]
    skipped = [p for p in picks if p["source"] == "gate-skip"]
    reused = [p for p in picks if p["source"] == "memory"]
    chosen = accepted[: limit // 2] + skipped[: limit // 4] + reused[: limit // 4]
    return {"transcripts": chosen, "counts": {"accepted": len(accepted), "gate_skipped": len(skipped),
                                              "memory_reused": len(reused)}}


# ====================================================================== E9
def exp_judge(settings: Settings, profile: str = "quick", workloads=WORKLOADS, max_validations: int = 20,
              progress: Progress = _noop) -> dict:
    """Independent-judge evaluation of this seed's baseline and ablation runs (see indep_judge.py)."""
    from ..data.workloads import load_workload
    from .indep_judge import IndependentJudge, cohen_kappa
    judge = IndependentJudge(settings)
    out = {"profile": profile, "judge_model": judge.model, "graded": [], "summary": [], "validator": [],
           "validator_summary": []}
    sources = {"baselines": load(settings, "baselines"), "ablation": load(settings, "ablation")}
    for wl in [w for w in workloads if w != "spider"]:      # Spider is scored by execution, no judge needed
        W = load_workload(wl, settings)
        qmap = {q["qid"]: q for q in W.queries}
        cache: dict[tuple, float | None] = {}
        for src, payload in sources.items():
            if not payload:
                continue
            for r in payload["rows"]:
                if r["workload"] != wl:
                    continue
                scores = []
                for pq in r["per_query"]:
                    q = qmap.get(pq["qid"])
                    if q is None or "answer" not in pq:
                        continue
                    key = (pq["qid"], " ".join(str(pq["answer"]).split()))
                    if key not in cache:
                        progress(f"judge grade | {wl} | {len(cache)}")
                        cache[key] = judge.grade(W.task_text, q["question"], q["answers"], pq["answer"])
                    pq["judge"] = cache[key]
                    if cache[key] is not None:
                        scores.append(cache[key])
                        out["graded"].append({"workload": wl, "source": src, "system": r["system"],
                                              "task": r["task"], "qid": pq["qid"], "judge": cache[key],
                                              "metric": pq.get("accuracy"), "f1": pq.get("f1"),
                                              "rouge_l": pq.get("rouge_l")})
                r["judge_acc"] = float(np.mean(scores)) if scores else None
        # validator agreement: re-judge the items the validation agent compared
        rng = random.Random(f"judge:{wl}:{settings.sampling.task_seed}")
        vals = []
        for r in (sources["baselines"] or {}).get("rows", []):
            if r["workload"] != wl or r["system"] not in ("plannerllm", "docetl"):
                continue
            for a in r["audit"]:
                v = a.get("validation") or {}
                if a.get("source") in ("validation", "tournament") and v.get("items"):
                    vals.append((r["system"], r["task"], a))
        rng.shuffle(vals)
        for system, task, a in vals[:max_validations]:
            wins = {"candidate": 0, "incumbent": 0, "tie": 0}
            for it in a["validation"]["items"]:
                q = qmap.get(it["qid"])
                res = judge.compare(W.task_text, q["question"] if q else it["question"], it["candidate"],
                                    it["incumbent"], key=f"{task}:{a['plan']}:{it['qid']}")
                if res:
                    wins[res] += 1
            progress(f"judge compare | {wl} | {a['directive']}")
            out["validator"].append({"workload": wl, "system": system, "task": task, "directive": a["directive"],
                                     "validator_prefers": a["verdict"] == "prefer",
                                     "judge_prefers": wins["candidate"] > wins["incumbent"], **wins})
    # summaries
    for src, payload in sources.items():
        if not payload:
            continue
        groups: dict[tuple, list] = {}
        for r in payload["rows"]:
            if r.get("judge_acc") is not None:
                groups.setdefault((r["workload"], r["system"]), []).append(r["judge_acc"])
        for (wl, system), v in groups.items():
            out["summary"].append({"source": src, "workload": wl, "system": system, "judge_acc": float(np.mean(v)),
                                   "n_tasks": len(v)})
    for wl in sorted({v["workload"] for v in out["validator"]}):
        vs = [v for v in out["validator"] if v["workload"] == wl]
        a_, b_ = [v["validator_prefers"] for v in vs], [v["judge_prefers"] for v in vs]
        out["validator_summary"].append({"workload": wl, "n": len(vs),
                                         "agreement": float(np.mean([x == y for x, y in zip(a_, b_)])),
                                         "kappa": cohen_kappa(a_, b_),
                                         "validator_prefer_rate": float(np.mean(a_)),
                                         "judge_prefer_rate": float(np.mean(b_))})
    allv = out["validator"]
    if allv:
        a_, b_ = [v["validator_prefers"] for v in allv], [v["judge_prefers"] for v in allv]
        out["validator_summary"].append({"workload": "all", "n": len(allv),
                                         "agreement": float(np.mean([x == y for x, y in zip(a_, b_)])),
                                         "kappa": cohen_kappa(a_, b_),
                                         "validator_prefer_rate": float(np.mean(a_)),
                                         "judge_prefer_rate": float(np.mean(b_))})
    out["judge_usage"] = judge.usage()
    return out


EXPERIMENTS = {
    "baselines": exp_baselines, "pareto": exp_pareto, "gate": exp_gate, "memory": exp_memory,
    "cost_model": exp_cost_model, "latency": exp_latency, "ablation": exp_ablation, "judge": exp_judge,
}


def run_experiment(name: str, settings: Settings, profile: str = "quick", workloads=None,
                   progress: Progress = print) -> Path:
    t0 = time.time()
    if name == "transcripts":
        payload = exp_transcripts(settings)
    else:
        payload = EXPERIMENTS[name](settings, profile=profile, workloads=workloads or WORKLOADS, progress=progress)
    payload["elapsed_s"] = time.time() - t0
    return save(settings, name, payload)


ALL = ["baselines", "pareto", "gate", "memory", "cost_model", "latency", "ablation", "transcripts", "judge"]
