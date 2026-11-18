"""Combine the per-seed experiment results into statistics, tables and figures.

Each seed is one full run of the experiments on different query groups, questions and samples
(``runs/<model>_seed<k>/``).  This module reads all of them and writes
``runs/<model>_aggregate/``:

* ``aggregate.json``  every number the report quotes, with mean, standard deviation and a 95%
                      confidence interval over seeds, plus paired significance tests;
* ``fig_*.png``       figures with error bars (one standard deviation over seeds);
* ``tab_*.csv``       the tables behind them.

Statistics used
* mean +- sample standard deviation over seeds; 95% CI = t(0.975, n-1) * sd / sqrt(n);
* paired comparisons on held-out questions (same question, same seed, two systems):
  mean difference with a 10,000-sample paired bootstrap CI and the Wilcoxon signed-rank test;
* paired comparisons of optimisation tokens per task (same task, same seed): relative reduction
  with a bootstrap CI over tasks and the Wilcoxon signed-rank test;
* Holm-Bonferroni correction over the family of baseline comparisons;
* Spearman rank correlation between the automatic metric and the independent judge;
* Cohen's kappa between the validation agent and the independent judge.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

from ..settings import RUNS_DIR

EXPS = ["baselines", "pareto", "gate", "memory", "cost_model", "latency", "ablation", "transcripts", "judge"]
WLS = ["msmarco", "cuad", "spider"]
SYSTEMS = ["no_opt", "random", "cost_only", "docetl", "plannerllm"]
ABLATIONS = ["plannerllm", "no_gate", "no_memory", "symbolic_only", "neural_only"]
METRIC_NAME = {"msmarco": "Rouge-L", "cuad": "token F1", "spider": "execution accuracy"}


# ---------------------------------------------------------------------------- loading
def seed_dirs(tag: str) -> list[tuple[int, Path]]:
    out = []
    for d in RUNS_DIR.glob(f"{tag}_seed*"):
        m = re.fullmatch(re.escape(tag) + r"_seed(\d+)", d.name)
        if m and (d / "baselines.json").exists():
            out.append((int(m.group(1)), d))
    return sorted(out)


def load_all(tag: str) -> dict[str, dict[int, dict]]:
    data: dict[str, dict[int, dict]] = defaultdict(dict)
    for seed, d in seed_dirs(tag):
        for e in EXPS:
            p = d / f"{e}.json"
            if p.exists():
                data[e][seed] = json.loads(p.read_text(encoding="utf-8"))
    return data


# ---------------------------------------------------------------------------- statistics
def msd(values) -> dict:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if n == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0, "values": []}
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
    half = float(sps.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else None
    return {"mean": mean, "std": sd, "ci95": half, "n": n, "values": v}


def bootstrap_ci(fn, n_items: int, n_boot: int = 10000, seed: int = 0) -> tuple[float, float] | None:
    if n_items < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_items, size=(n_boot, n_items))
    vals = np.array([fn(i) for i in idx])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilcoxon_p(a, b) -> float | None:
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0 if len(d) else None
    try:
        return float(sps.wilcoxon(a, b, zero_method="zsplit").pvalue)
    except ValueError:
        return None


def paired_diff(a: list[float], b: list[float]) -> dict:
    """Mean of a - b over paired items, bootstrap CI, Wilcoxon p, wins/ties/losses."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    return {"n": int(len(d)), "mean_a": float(a.mean()) if len(a) else None, "mean_b": float(b.mean()) if len(b) else None,
            "mean_diff": float(d.mean()) if len(d) else None,
            "ci95": bootstrap_ci(lambda i: d[i].mean(), len(d)),
            "p_wilcoxon": wilcoxon_p(a, b),
            "wins": int((d > 1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()), "losses": int((d < -1e-9).sum())}


def paired_reduction(p: list[float], d: list[float]) -> dict:
    """Relative token reduction 1 - sum(p)/sum(d) over paired tasks."""
    p, d = np.asarray(p, float), np.asarray(d, float)
    red = lambda i: 1 - p[i].sum() / d[i].sum() if d[i].sum() > 0 else np.nan
    return {"n": int(len(p)), "reduction": float(1 - p.sum() / d.sum()) if d.sum() else None,
            "ci95": bootstrap_ci(red, len(p)), "p_wilcoxon": wilcoxon_p(p, d),
            "mean_p": float(p.mean()) if len(p) else None, "mean_d": float(d.mean()) if len(d) else None}


def holm(pvals: dict[str, float | None]) -> dict[str, float | None]:
    items = sorted([(k, v) for k, v in pvals.items() if v is not None], key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k, v in pvals.items():
        out.setdefault(k, None if v is None else out.get(k))
    return out


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def _mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


# ---------------------------------------------------------------------------- aggregation
def aggregate(tag: str) -> dict:
    D = load_all(tag)
    seeds = sorted(D["baselines"])
    A: dict = {"tag": tag, "seeds": seeds, "metric_name": METRIC_NAME}
    base = D["baselines"]
    if base:
        any_b = base[seeds[0]]
        A["model"], A["profile"] = any_b.get("model"), any_b.get("profile")
        A["created"] = {s: base[s].get("created") for s in seeds}

    # ---- E1 baselines: per seed means over tasks, then mean +- sd over seeds
    keys = ["accuracy", "opt_tokens", "opt_usd", "n_validations", "n_gate_skips", "n_memory_hits", "exec_ms",
            "exec_tokens", "opt_llm_s", "hallucinated", "f1", "rouge_l", "llm_ms", "db_ms", "ann_ms"]
    rows_all = {s: base[s]["rows"] for s in seeds}
    tab = []
    for wl in WLS:
        for sysname in SYSTEMS:
            entry = {"workload": wl, "system": sysname}
            for k in keys:
                entry[k] = msd([_mean([r for r in rows_all[s] if r["workload"] == wl and r["system"] == sysname], k)
                                for s in seeds])
            # decision agreement with the validate-all choice (per task)
            ag = []
            for s in seeds:
                ref = {r["task"]: r["plan"] for r in rows_all[s] if r["workload"] == wl and r["system"] == "docetl"}
                rs = [r for r in rows_all[s] if r["workload"] == wl and r["system"] == sysname and r["task"] in ref]
                ag.append(float(np.mean([r["plan"] == ref[r["task"]] for r in rs])) if rs else None)
            entry["decision_agreement"] = msd(ag)
            if any(r["workload"] == wl and r["system"] == sysname for s in seeds for r in rows_all[s]):
                tab.append(entry)
    A["baselines"] = tab

    # per-question paired tests, PlannerLLM against each baseline
    tests, pv = [], {}
    for wl in WLS:
        for other in ["docetl", "no_opt", "cost_only", "random"]:
            a, b = [], []
            for s in seeds:
                pq = {(r["task"], q["qid"]): q["accuracy"] for r in rows_all[s]
                      if r["workload"] == wl and r["system"] == "plannerllm" for q in r["per_query"]}
                oq = {(r["task"], q["qid"]): q["accuracy"] for r in rows_all[s]
                      if r["workload"] == wl and r["system"] == other for q in r["per_query"]}
                for k in sorted(set(pq) & set(oq)):
                    a.append(pq[k])
                    b.append(oq[k])
            if a:
                t = {"workload": wl, "other": other, "measure": "accuracy", **paired_diff(a, b)}
                tests.append(t)
                pv[f"{wl}:{other}"] = t["p_wilcoxon"]
    adj = holm(pv)
    for t in tests:
        t["p_holm"] = adj.get(f"{t['workload']}:{t['other']}")
    A["accuracy_tests"] = tests

    # optimisation tokens, PlannerLLM against the validate-all optimizer, paired per task
    red, red_all_p, red_all_d = [], [], []
    for wl in WLS:
        p, d, per_seed = [], [], []
        for s in seeds:
            pr = {r["task"]: r["opt_tokens"] for r in rows_all[s] if r["workload"] == wl and r["system"] == "plannerllm"}
            dr = {r["task"]: r["opt_tokens"] for r in rows_all[s] if r["workload"] == wl and r["system"] == "docetl"}
            ks = sorted(set(pr) & set(dr))
            p += [pr[k] for k in ks]
            d += [dr[k] for k in ks]
            if ks and sum(dr[k] for k in ks):
                per_seed.append(1 - sum(pr[k] for k in ks) / sum(dr[k] for k in ks))
        if p:
            red.append({"workload": wl, **paired_reduction(p, d), "per_seed": msd(per_seed)})
            red_all_p += p
            red_all_d += d
    if red_all_p:
        red.append({"workload": "all", **paired_reduction(red_all_p, red_all_d),
                    "per_seed": msd([np.mean([x["per_seed"]["values"][i] for x in red
                                              if x["workload"] != "all" and len(x["per_seed"]["values"]) > i])
                                     for i in range(len(seeds))])})
    A["token_reduction"] = red

    # how often each optimizer considered and adopted each directive (per task, all seeds)
    adopt = []
    for wl in WLS:
        for sysname in ("docetl", "plannerllm"):
            rs = [r for s in seeds for r in rows_all[s] if r["workload"] == wl and r["system"] == sysname]
            dirs = sorted({a["directive"] for r in rs for a in r["audit"]})
            for d in dirs:
                adopt.append({"workload": wl, "system": sysname, "directive": d, "tasks": len(rs),
                              "considered": sum(any(a["directive"] == d for a in r["audit"]) for r in rs),
                              "adopted": sum(any(a["directive"] == d and a["verdict"] == "prefer" for a in r["audit"])
                                             for r in rs)})
    A["directive_adoption"] = adopt

    # ---- E9 independent judge
    J = D.get("judge", {})
    if J:
        jt = []
        for wl in WLS:
            for sysname in SYSTEMS:
                vals = [next((x["judge_acc"] for x in J[s]["summary"] if x["source"] == "baselines"
                              and x["workload"] == wl and x["system"] == sysname), None) for s in J]
                if any(v is not None for v in vals):
                    jt.append({"workload": wl, "system": sysname, "judge_acc": msd(vals)})
        A["judge_acc"] = jt
        corr = []
        for wl in WLS:
            g = [x for s in J for x in J[s]["graded"] if x["workload"] == wl and x["source"] == "baselines"
                 and x["metric"] is not None]
            if len(g) > 5:
                rho, p = sps.spearmanr([x["metric"] for x in g], [x["judge"] for x in g])
                rho_f1, _ = sps.spearmanr([x["f1"] or 0 for x in g], [x["judge"] for x in g])
                corr.append({"workload": wl, "n": len(g), "spearman_metric": float(rho), "p": float(p),
                             "spearman_f1": float(rho_f1)})
        A["metric_judge_corr"] = corr
        # judge-based paired test PlannerLLM vs DocETL on held-out questions
        jtests = []
        for wl in WLS:
            a, b = [], []
            for s in J:
                g = [x for x in J[s]["graded"] if x["workload"] == wl and x["source"] == "baselines"]
                pq = {(x["task"], x["qid"]): x["judge"] for x in g if x["system"] == "plannerllm"}
                oq = {(x["task"], x["qid"]): x["judge"] for x in g if x["system"] == "docetl"}
                for k in sorted(set(pq) & set(oq)):
                    a.append(pq[k])
                    b.append(oq[k])
            if a:
                jtests.append({"workload": wl, "other": "docetl", "measure": "judge", **paired_diff(a, b)})
        A["judge_tests"] = jtests
        allv = [v for s in J for v in J[s]["validator"]]
        vs = []
        for wl in WLS + ["all"]:
            rs = allv if wl == "all" else [v for v in allv if v["workload"] == wl]
            if not rs:
                continue
            a_, b_ = [v["validator_prefers"] for v in rs], [v["judge_prefers"] for v in rs]
            agree = np.array([x == y for x, y in zip(a_, b_)], float)
            vs.append({"workload": wl, "n": len(rs), "agreement": float(agree.mean()),
                       "ci95": bootstrap_ci(lambda i: agree[i].mean(), len(agree)), "kappa": kappa(a_, b_),
                       "validator_prefer_rate": float(np.mean(a_)), "judge_prefer_rate": float(np.mean(b_))})
        A["validator_agreement"] = vs
        A["judge_model"] = J[sorted(J)[0]].get("judge_model")
        A["judge_tokens"] = int(sum(J[s].get("judge_usage", {}).get("tokens", 0) for s in J))

    # ---- E2 pareto / theta sweep
    P = D.get("pareto", {})
    if P:
        th, bu = defaultdict(list), defaultdict(list)
        for s in P:
            for pt in P[s]["points"]:
                if pt["sweep"] == "theta":
                    th[(pt["workload"], pt["theta"])].append(pt)
                else:
                    bu[(pt["workload"], pt["system"], pt["budget"])].append(pt)
        A["theta_sweep"] = [{"workload": wl, "theta": t, "accuracy": msd([p["accuracy"] for p in ps]),
                             "opt_tokens": msd([p["opt_tokens"] for p in ps]),
                             "validations": msd([p.get("validations") for p in ps]),
                             "gate_skips": msd([p.get("gate_skips") for p in ps])}
                            for (wl, t), ps in sorted(th.items(), key=lambda kv: (WLS.index(kv[0][0]), kv[0][1]))]
        A["budget_sweep"] = [{"workload": wl, "system": sy, "budget": b, "accuracy": msd([p["accuracy"] for p in ps]),
                              "opt_tokens": msd([p["opt_tokens"] for p in ps])}
                             for (wl, sy, b), ps in sorted(bu.items(), key=lambda kv: (WLS.index(kv[0][0]), kv[0][1],
                                                                                         kv[0][2] or 1e12))]

    # ---- E3 gate: per seed replay scores and pooled calibration
    G = D.get("gate", {})
    if G:
        from .experiments import calibration_bins, gate_scores
        thetas = [x["theta"] for x in G[sorted(G)[0]]["overall"]["scores"]]
        per = defaultdict(lambda: defaultdict(list))
        pooled_log = []
        for s in G:
            for x in G[s]["overall"]["scores"]:
                for k in ("precision", "recall", "spend_saved", "flip_rate", "admitted", "n"):
                    per[x["theta"]][k].append(x[k])
            for wl, w in G[s]["per_workload"].items():
                pooled_log += w["log"]
        A["gate"] = [{"theta": t, **{k: msd(v) for k, v in per[t].items()}} for t in thetas]
        A["gate_pooled"] = gate_scores(pooled_log, thetas)
        A["gate_calibration"] = calibration_bins(pooled_log)
        cal = [c for c in A["gate_calibration"] if c["n"]]
        A["gate_ece"] = (sum(c["n"] * abs(c["predicted"] - c["observed"]) for c in cal) / sum(c["n"] for c in cal)
                         if cal else None)
        A["gate_n_decisions"] = len([g for g in pooled_log if g["flipped"] is not None])

    # ---- E4 memory
    M = D.get("memory", {})
    if M:
        ph = defaultdict(list)
        reuse_n, reuse_ok = defaultdict(int), defaultdict(int)
        for s in M:
            for x in M[s]["streams"]:
                ph[(x["workload"], x["phase"])].append(x)
            for x in M[s]["reuse"]:
                reuse_n[x["workload"]] += 1
                reuse_ok[x["workload"]] += int(x["reused"] == x["fresh"])
        order = ["first-seen", "repeated", "prompt_edit", "config_change"]
        A["memory"] = [{"workload": wl, "phase": p, "hit_rate": msd([x["hit_rate"] for x in xs]),
                        "opt_tokens": msd([x["opt_tokens"] for x in xs]), "accuracy": msd([x["accuracy"] for x in xs]),
                        "n_audited": int(sum(x["n_audited"] for x in xs)),
                        "reuse_accuracy": (float(np.mean([x["reuse_accuracy"] for x in xs if x["reuse_accuracy"] is not None]))
                                           if any(x["reuse_accuracy"] is not None for x in xs) else None)}
                       for (wl, p), xs in sorted(ph.items(), key=lambda kv: (WLS.index(kv[0][0]), order.index(kv[0][1])))]
        A["memory_reuse_audit"] = {wl: {"n": reuse_n[wl], "agree": reuse_ok[wl]} for wl in reuse_n}
        rep = [x for x in A["memory"] if x["phase"] == "repeated"]
        A["memory_repeat_hit_rate"] = msd([np.mean([x["hit_rate"]["values"][i] for x in rep
                                                    if len(x["hit_rate"]["values"]) > i]) for i in range(len(M))])
        dec = defaultdict(list)
        for s in M:
            for x in M[s]["decay"]:
                dec[(x["workload"], x["kappa"])].append(x)
        A["memory_decay"] = [{"workload": wl, "kappa": k, "hit_rate_pass2": msd([x["hit_rate_pass2"] for x in xs]),
                              "opt_tokens_pass2": msd([x["opt_tokens_pass2"] for x in xs])}
                             for (wl, k), xs in sorted(dec.items(), key=lambda kv: (WLS.index(kv[0][0]), kv[0][1]))]
        A["decay_curve"] = M[sorted(M)[0]]["decay_curve"]
        A["reuse_threshold"] = M[sorted(M)[0]]["reuse_threshold"]

    # ---- E5 cost model
    C = D.get("cost_model", {})
    if C:
        cm = defaultdict(lambda: defaultdict(list))
        for s in C:
            for row in C[s]["summary"]:
                for model in ("symbolic", "neural", "hybrid"):
                    if model in row:
                        for k in ("median_q", "p90_q", "mae_log10"):
                            cm[(row["split"], row["target"], model)][k].append(row[model][k])
        A["cost_model"] = [{"split": sp, "target": tg, "model": mo, **{k: msd(v) for k, v in d.items()}}
                           for (sp, tg, mo), d in sorted(cm.items())]
        A["cost_samples"] = msd([C[s]["n_samples"] for s in C])
        A["cost_scatter"] = [dict(x, seed=s) for s in C for x in C[s]["scatter"]]

    # ---- E6 latency
    L = D.get("latency", {})
    if L:
        pr = defaultdict(list)
        for s in L:
            for x in L[s]["probe"]:
                pr[(x["workload"], x["family"])].append(x)
        A["probe"] = [{"workload": wl, "family": fam, "sel": float(np.mean([x["sel"] for x in xs])),
                       "filtered_recall": float(np.mean([x["filtered_recall"] for x in xs])),
                       "post_recall": float(np.mean([x["post_recall"] for x in xs])),
                       "filtered_ms": float(np.median([x["filtered_ms"] for x in xs])),
                       "post_ms": float(np.median([x["post_ms"] for x in xs])), "n": len(xs)}
                      for (wl, fam), xs in sorted(pr.items(), key=lambda kv: (WLS.index(kv[0][0]), kv[0][1]))]
        sw = defaultdict(list)
        for s in L:
            for x in L[s]["ann_sweep"]:
                sw[(x["workload"], x["k"], x["family"], x["knob"])].append(x)
        A["ann_sweep"] = [{"workload": k[0], "k": k[1], "family": k[2], "knob": k[3],
                           "ms": float(np.median([x["ms"] for x in xs])), "recall": float(np.mean([x["recall"] for x in xs]))}
                          for k, xs in sorted(sw.items(), key=lambda kv: (WLS.index(kv[0][0]), kv[0][1:]))]
        lt = defaultdict(lambda: defaultdict(list))
        for s in L:
            for x in L[s].get("llm_time", []):
                if x.get("opt_llm_s") is not None:
                    lt[(x["workload"], x["system"])][s].append(x["opt_llm_s"])
        A["llm_time"] = [{"workload": wl, "system": sy, "opt_llm_s": msd([np.mean(v) for v in d.values()])}
                         for (wl, sy), d in sorted(lt.items(), key=lambda kv: (WLS.index(kv[0][0]), kv[0][1]))]

    # ---- E7 ablation
    AB = D.get("ablation", {})
    if AB:
        ab, abt = [], []
        for wl in WLS:
            for v in ABLATIONS:
                e = {"workload": wl, "variant": v}
                for k in ("accuracy", "opt_tokens", "n_validations", "n_memory_hits", "n_gate_skips"):
                    e[k] = msd([_mean([r for r in AB[s]["rows"] if r["workload"] == wl and r["system"] == v], k)
                                for s in AB])
                if e["accuracy"]["n"]:
                    ab.append(e)
                if v != "plannerllm":
                    a_tok, b_tok, a_acc, b_acc = [], [], [], []
                    for s in AB:
                        full = {r["task"]: r for r in AB[s]["rows"] if r["workload"] == wl and r["system"] == "plannerllm"}
                        var = {r["task"]: r for r in AB[s]["rows"] if r["workload"] == wl and r["system"] == v}
                        for k in sorted(set(full) & set(var)):
                            a_tok.append(full[k]["opt_tokens"])
                            b_tok.append(var[k]["opt_tokens"])
                            a_acc.append(full[k]["accuracy"])
                            b_acc.append(var[k]["accuracy"])
                    if a_tok:
                        abt.append({"workload": wl, "variant": v, "tokens": paired_diff(a_tok, b_tok),
                                    "accuracy": paired_diff(a_acc, b_acc)})
        A["ablation"] = ab
        A["ablation_tests"] = abt
        A["ablation_budget"] = AB[sorted(AB)[0]].get("budget")

    # ---- E8 transcripts (from the first seed)
    T = D.get("transcripts", {})
    if T:
        A["transcripts"] = T[sorted(T)[0]]["transcripts"]
    return A


# ---------------------------------------------------------------------------- output
def write(A: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    p = out / "aggregate.json"
    p.write_text(json.dumps(A, indent=1, default=default), encoding="utf-8")
    return p


def fmt_ms(x: dict, d: int = 3) -> str:
    if not x or x.get("mean") is None:
        return "-"
    if x["n"] > 1:
        return f"{x['mean']:.{d}f} ± {x['std']:.{d}f}"
    return f"{x['mean']:.{d}f}"


def write_tables(A: dict, out: Path) -> list[str]:
    files = []

    def tab(name, header, rows):
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        files.append(f"{name}.csv")
    tab("tab_baselines", ["workload", "system", "accuracy_mean", "accuracy_sd", "opt_tokens_mean", "opt_tokens_sd",
                          "validations_mean", "gate_skips_mean", "memory_hits_mean", "exec_ms_mean", "n_seeds"],
        [[r["workload"], r["system"], r["accuracy"]["mean"], r["accuracy"]["std"], r["opt_tokens"]["mean"],
          r["opt_tokens"]["std"], r["n_validations"]["mean"], r["n_gate_skips"]["mean"], r["n_memory_hits"]["mean"],
          r["exec_ms"]["mean"], r["accuracy"]["n"]] for r in A.get("baselines", [])])
    tab("tab_accuracy_tests", ["workload", "vs", "n_questions", "mean_diff", "ci_lo", "ci_hi", "p_wilcoxon", "p_holm",
                               "wins", "ties", "losses"],
        [[t["workload"], t["other"], t["n"], t["mean_diff"], *(t["ci95"] or [None, None]), t["p_wilcoxon"],
          t["p_holm"], t["wins"], t["ties"], t["losses"]] for t in A.get("accuracy_tests", [])])
    tab("tab_token_reduction", ["workload", "n_tasks", "reduction", "ci_lo", "ci_hi", "p_wilcoxon"],
        [[t["workload"], t["n"], t["reduction"], *(t["ci95"] or [None, None]), t["p_wilcoxon"]]
         for t in A.get("token_reduction", [])])
    if A.get("judge_acc"):
        tab("tab_judge", ["workload", "system", "judge_acc_mean", "judge_acc_sd", "n_seeds"],
            [[r["workload"], r["system"], r["judge_acc"]["mean"], r["judge_acc"]["std"], r["judge_acc"]["n"]]
             for r in A["judge_acc"]])
        tab("tab_validator_agreement", ["workload", "n", "agreement", "ci_lo", "ci_hi", "kappa"],
            [[r["workload"], r["n"], r["agreement"], *(r["ci95"] or [None, None]), r["kappa"]]
             for r in A["validator_agreement"]])
    if A.get("gate"):
        tab("tab_gate", ["theta", "precision_mean", "precision_sd", "recall_mean", "recall_sd", "spend_saved_mean",
                         "spend_saved_sd"],
            [[g["theta"], g["precision"]["mean"], g["precision"]["std"], g["recall"]["mean"], g["recall"]["std"],
              g["spend_saved"]["mean"], g["spend_saved"]["std"]] for g in A["gate"]])
    if A.get("memory"):
        tab("tab_memory", ["workload", "phase", "hit_rate_mean", "hit_rate_sd", "opt_tokens_mean", "accuracy_mean",
                           "n_audited", "reuse_accuracy"],
            [[m["workload"], m["phase"], m["hit_rate"]["mean"], m["hit_rate"]["std"], m["opt_tokens"]["mean"],
              m["accuracy"]["mean"], m["n_audited"], m["reuse_accuracy"]] for m in A["memory"]])
    if A.get("cost_model"):
        tab("tab_cost_model", ["split", "target", "model", "median_q_mean", "median_q_sd", "p90_q_mean"],
            [[c["split"], c["target"], c["model"], c["median_q"]["mean"], c["median_q"]["std"], c["p90_q"]["mean"]]
             for c in A["cost_model"]])
    if A.get("ablation"):
        tab("tab_ablation", ["workload", "variant", "accuracy_mean", "accuracy_sd", "opt_tokens_mean",
                             "opt_tokens_sd", "validations_mean", "memory_hits_mean", "gate_skips_mean"],
            [[a["workload"], a["variant"], a["accuracy"]["mean"], a["accuracy"]["std"], a["opt_tokens"]["mean"],
              a["opt_tokens"]["std"], a["n_validations"]["mean"], a["n_memory_hits"]["mean"],
              a["n_gate_skips"]["mean"]] for a in A["ablation"]])
    return files


def build(tag: str = "openai_gpt-oss-20b") -> Path:
    A = aggregate(tag)
    out = RUNS_DIR / f"{tag}_aggregate"
    p = write(A, out)
    write_tables(A, out)
    from .figures import write_figures
    write_figures(A, out)
    return p
