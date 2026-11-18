"""Figures for the seed-aggregated results (error bars = one standard deviation over seeds)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .reporting import INK, LABEL, MUTED, SERIES, SYSTEM_COLOR, WL_LABEL, _save, _style, plt

WLS = ["msmarco", "cuad", "spider"]


def _m(x):
    return None if not x or x.get("mean") is None else x["mean"]


def _s(x):
    return 0.0 if not x or not x.get("std") else x["std"]


def _bars(ax, groups, series, get, fmt="{:.2f}", label=None):
    n = len(series)
    width = 0.8 / max(n, 1)
    x = np.arange(len(groups))
    for i, sname in enumerate(series):
        vals = [get(g, sname) for g in groups]
        ys = [_m(v) or 0 for v in vals]
        es = [_s(v) for v in vals]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, ys, width * 0.92, yerr=es, capsize=2.5,
                      error_kw={"elinewidth": 0.9, "ecolor": INK, "capthick": 0.9},
                      color=SYSTEM_COLOR.get(sname, SERIES[i % 8]), label=(label or LABEL).get(sname, sname),
                      edgecolor="white", linewidth=0.8)
        if n <= 5:
            for b, v, e in zip(bars, vals, es):
                if _m(v) is not None:
                    ax.annotate(fmt.format(_m(v)), (b.get_x() + b.get_width() / 2, b.get_height() + e),
                                ha="center", va="bottom", fontsize=7.5, color=MUTED, xytext=(0, 1.5),
                                textcoords="offset points")
    ax.set_xticks(x, [WL_LABEL.get(g, g) for g in groups])
    ax.grid(axis="x", visible=False)


def write_figures(A: dict, out: Path) -> list[str]:
    _style()
    files = []
    B = A.get("baselines", [])
    get = lambda key: (lambda w, s: next((r[key] for r in B if r["workload"] == w and r["system"] == s), None))
    wls = [w for w in WLS if any(r["workload"] == w for r in B)]
    systems = [s for s in ["no_opt", "random", "cost_only", "docetl", "plannerllm"] if any(r["system"] == s for r in B)]
    if B:
        fig, ax = plt.subplots(figsize=(6.3, 3.0))
        _bars(ax, wls, systems, get("accuracy"))
        ax.set_ylabel("held-out score\n(Rouge-L / F1 / EX)")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
        files.append(_save(fig, out, "fig_baselines_accuracy"))
        fig, ax = plt.subplots(figsize=(6.3, 3.0))
        _bars(ax, wls, [s for s in systems if s in ("cost_only", "docetl", "plannerllm")], get("opt_tokens"),
              fmt="{:,.0f}")
        ax.set_ylabel("optimisation tokens per task")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
        files.append(_save(fig, out, "fig_baselines_tokens"))
    J = A.get("judge_acc", [])
    if J:
        jw = [w for w in WLS if any(r["workload"] == w for r in J)]
        jget = lambda w, s: next((r["judge_acc"] for r in J if r["workload"] == w and r["system"] == s), None)
        fig, ax = plt.subplots(figsize=(6.3, 2.9))
        _bars(ax, jw, systems, jget)
        ax.set_ylabel("independent judge score")
        ax.set_ylim(0, 1.05)
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
        files.append(_save(fig, out, "fig_judge_accuracy"))
    V = [v for v in A.get("validator_agreement", []) if v["workload"] != "all"]
    if V:
        fig, ax = plt.subplots(figsize=(4.6, 2.7))
        xs = np.arange(len(V))
        ys = [v["agreement"] for v in V]
        lo = [v["agreement"] - (v["ci95"] or [v["agreement"]] * 2)[0] for v in V]
        hi = [(v["ci95"] or [v["agreement"]] * 2)[1] - v["agreement"] for v in V]
        ax.bar(xs, ys, 0.55, yerr=[lo, hi], capsize=3, color=SERIES[0], edgecolor="white",
               error_kw={"elinewidth": 0.9, "ecolor": INK})
        for x, v in zip(xs, V):
            ax.annotate(f"{v['agreement']:.2f}\nn = {v['n']}", (x, 0.04), ha="center", va="bottom",
                        fontsize=8.5, color="white")
        ax.axhline(0.5, ls="--", lw=0.8, color=MUTED)
        ax.set_xticks(xs, [WL_LABEL.get(v["workload"], v["workload"]) for v in V])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("agreement with\nvalidation agent")
        ax.grid(axis="x", visible=False)
        files.append(_save(fig, out, "fig_validator_agreement"))
    TH = A.get("theta_sweep", [])
    if TH:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
        for i, wl in enumerate(WLS):
            pts = [p for p in TH if p["workload"] == wl]
            if not pts:
                continue
            t = [p["theta"] for p in pts]
            for ax, key in zip(axes, ("accuracy", "opt_tokens")):
                ax.errorbar(t, [_m(p[key]) for p in pts], yerr=[_s(p[key]) for p in pts], marker="o", ms=3.5,
                            lw=1.6, capsize=2, color=SERIES[i], label=WL_LABEL[wl])
        axes[0].set_ylabel("held-out score")
        axes[1].set_ylabel("optimisation tokens / task")
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel(r"gate threshold $\theta$")
        axes[1].legend(fontsize=8)
        files.append(_save(fig, out, "fig_theta_sweep"))
    BU = A.get("budget_sweep", [])
    if BU:
        bw = [w for w in WLS if any(p["workload"] == w for p in BU)]
        fig, axes = plt.subplots(1, len(bw), figsize=(6.3, 2.6))
        axes = np.atleast_1d(axes)
        for ax, wl in zip(axes, bw):
            for sy in ("docetl", "no_gate", "plannerllm"):
                pts = [p for p in BU if p["workload"] == wl and p["system"] == sy]
                if pts:
                    ax.errorbar([_m(p["opt_tokens"]) for p in pts], [_m(p["accuracy"]) for p in pts],
                                yerr=[_s(p["accuracy"]) for p in pts], marker="o", ms=3, lw=1.4, capsize=2,
                                color=SYSTEM_COLOR[sy], label=LABEL[sy])
            ax.set_title(WL_LABEL[wl], loc="left", fontsize=10, color=INK)
            ax.set_xlabel("optimisation tokens / task")
        axes[0].set_ylabel("held-out score")
        axes[-1].legend(fontsize=7.5)
        files.append(_save(fig, out, "fig_pareto_budget"))
    G = A.get("gate", [])
    if G:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
        t = [g["theta"] for g in G]
        for key, c, mk in (("precision", SERIES[0], "o"), ("recall", SERIES[1], "s")):
            axes[0].errorbar(t, [_m(g[key]) for g in G], yerr=[_s(g[key]) for g in G], marker=mk, ms=3.5, lw=1.6,
                             capsize=2, color=c, label=key)
        axes[1].errorbar(t, [_m(g["spend_saved"]) for g in G], yerr=[_s(g["spend_saved"]) for g in G], marker="o",
                         ms=3.5, lw=1.6, capsize=2, color=SERIES[2])
        axes[0].set_ylabel("score")
        axes[1].set_ylabel("validation spend avoided")
        axes[0].legend(fontsize=8)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel(r"$\theta$")
            ax.set_ylim(0, 1.05)
        files.append(_save(fig, out, "fig_gate_pr"))
    cal = [c for c in A.get("gate_calibration", []) if c["n"]]
    if cal:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="perfect calibration")
        ax.plot([c["predicted"] for c in cal], [c["observed"] for c in cal], marker="o", ms=5, lw=2, color=SERIES[0],
                label="P(flip)")
        for c in cal:
            ax.annotate(f"n={c['n']}", (c["predicted"], c["observed"]), fontsize=8, color=MUTED, xytext=(4, -9),
                        textcoords="offset points")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted flip probability")
        ax.set_ylabel("observed flip frequency")
        ax.legend(fontsize=8, loc="upper left")
        files.append(_save(fig, out, "fig_gate_calibration"))
    MEM = A.get("memory", [])
    if MEM:
        phases = ["first-seen", "repeated", "prompt_edit", "config_change"]
        short = {"first-seen": "first", "repeated": "repeat", "prompt_edit": "prompt", "config_change": "config"}
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
        w = 0.26
        for i, wl in enumerate(WLS):
            rows = {m["phase"]: m for m in MEM if m["workload"] == wl}
            if not rows:
                continue
            xs = np.arange(len(phases)) + (i - 1) * w
            for ax, key in zip(axes, ("hit_rate", "opt_tokens")):
                ax.bar(xs, [_m(rows[p][key]) or 0 for p in phases], w * 0.92,
                       yerr=[_s(rows[p][key]) for p in phases], capsize=2, color=SERIES[i], label=WL_LABEL[wl],
                       edgecolor="white", error_kw={"elinewidth": 0.8, "ecolor": INK})
        for ax in axes:
            ax.set_xticks(np.arange(len(phases)), [short[p] for p in phases])
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("signature hit rate")
        axes[1].set_ylabel("optimisation tokens / task")
        axes[0].legend(fontsize=8)
        files.append(_save(fig, out, "fig_memory_hits"))
    curve = A.get("decay_curve")
    if curve:
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        keys = [k for k in curve[0] if k.startswith("kappa=")]
        for i, k in enumerate(keys):
            ax.plot([c["age_days"] for c in curve], [c[k] for c in curve], lw=2, color=SERIES[i],
                    label=k.replace("kappa", r"$\kappa$"))
        ax.axhline(A.get("reuse_threshold", 0.55), ls="--", lw=1, color=MUTED)
        ax.set_xlabel("verdict age (days)")
        ax.set_ylabel(r"confidence $\delta(t)$, n=2")
        ax.legend(fontsize=8)
        files.append(_save(fig, out, "fig_memory_decay"))
    CM = A.get("cost_model", [])
    if CM:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
        for ax, target, title in zip(axes, ("usd", "ms"), ("USD", "latency")):
            splits = sorted({c["split"] for c in CM if c["target"] == target}, key=lambda s: (s != "cv", s))
            for i, model in enumerate(("symbolic", "neural", "hybrid")):
                rows = [next((c for c in CM if c["split"] == sp and c["target"] == target and c["model"] == model), None)
                        for sp in splits]
                xs = np.arange(len(splits)) + (i - 1) * 0.26
                ax.bar(xs, [_m(r["median_q"]) if r else 0 for r in rows], 0.24,
                       yerr=[_s(r["median_q"]) if r else 0 for r in rows], capsize=2, color=SERIES[i], label=model,
                       edgecolor="white", error_kw={"elinewidth": 0.8, "ecolor": INK})
            ax.set_xticks(np.arange(len(splits)), [s.replace("zero-shot:", "zs ") for s in splits], fontsize=8)
            ax.axhline(1.0, color=MUTED, lw=0.8)
            ax.set_title(title, loc="left", fontsize=10, color=INK)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("median q-error")
        axes[0].legend(fontsize=8)
        files.append(_save(fig, out, "fig_cost_qerror"))
    SC = A.get("cost_scatter", [])
    if SC:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
        for i, wl in enumerate(WLS):
            pts = [p for p in SC if p["workload"] == wl]
            for ax, key in zip(axes, ("usd", "ms")):
                ax.scatter([p[f"sym_{key}"] for p in pts], [p[f"obs_{key}"] for p in pts], s=9, color=SERIES[i],
                           label=WL_LABEL[wl], alpha=0.8, edgecolors="white", linewidths=0.4)
        for ax, key, lab in zip(axes, ("usd", "ms"), ("USD per run", "latency ms per run")):
            vals = [p[f"sym_{key}"] for p in SC] + [p[f"obs_{key}"] for p in SC]
            vals = [v for v in vals if v and v > 0]
            if vals:
                lo, hi = min(vals) * 0.7, max(vals) * 1.4
                ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color=MUTED)
            ax.set_xscale("log")
            ax.set_yscale("log")
            from matplotlib.ticker import LogLocator, NullFormatter
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(LogLocator(base=10, numticks=6))
                axis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(f"symbolic estimate ({lab})")
            ax.set_ylabel(f"observed ({lab})")
        axes[0].legend(fontsize=8)
        files.append(_save(fig, out, "fig_cost_scatter"))
    if B:
        fig, axes = plt.subplots(1, len(wls), figsize=(6.3, 2.7), sharey=False)
        axes = np.atleast_1d(axes)
        sysl = [s for s in ["no_opt", "cost_only", "docetl", "plannerllm"] if s in systems]
        for ax, wl in zip(axes, wls):
            bottom = np.zeros(len(sysl))
            for part, c, lab in (("db_ms", SERIES[2], "DB operators"), ("ann_ms", SERIES[0], "ANN + rerank"),
                                 ("llm_ms", SERIES[1], "LLM")):
                vals = np.array([_m(get(part)(wl, s)) or 0 for s in sysl])
                ax.bar(range(len(sysl)), vals, 0.6, bottom=bottom, color=c, label=lab, edgecolor="white")
                bottom += vals
            ax.set_xticks(range(len(sysl)), [{"no_opt": "No opt."}.get(s, LABEL[s]) for s in sysl], rotation=35, fontsize=8)
            ax.set_title(WL_LABEL[wl], loc="left", fontsize=10, color=INK)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("ms per query (execution)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
        files.append(_save(fig, out, "fig_latency_breakdown"))
    SW = [p for p in A.get("ann_sweep", []) if p["k"] == 10]
    if SW:
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6))
        for ax, wl in zip(axes, WLS):
            for i, fam in enumerate(("hnsw", "ivf", "flat")):
                pts = sorted([p for p in SW if p["workload"] == wl and p["family"] == fam], key=lambda p: p["ms"])
                if pts:
                    ax.plot([p["ms"] for p in pts], [p["recall"] for p in pts], marker="o", ms=3, lw=1.5,
                            color=SERIES[i], label=fam.upper())
            ax.set_title(WL_LABEL[wl], loc="left", fontsize=10, color=INK)
            ax.set_xlabel("median probe ms")
            ax.set_ylim(0, 1.05)
        axes[0].set_ylabel("recall@10 vs exact")
        axes[-1].legend(fontsize=7.5)
        files.append(_save(fig, out, "fig_ann_sweep"))
    AB = A.get("ablation", [])
    if AB:
        abl = ["plannerllm", "no_gate", "no_memory", "symbolic_only", "neural_only"]
        aw = [w for w in WLS if any(a["workload"] == w for a in AB)]
        aget = lambda key: (lambda w, v: next((a[key] for a in AB if a["workload"] == w and a["variant"] == v), None))
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
        _bars(axes[0], aw, abl, aget("accuracy"), fmt="")
        _bars(axes[1], aw, abl, aget("opt_tokens"), fmt="")
        axes[0].set_ylabel("held-out score")
        axes[1].set_ylabel("optimisation tokens / task")
        axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.1, -0.12), fontsize=7.5)
        files.append(_save(fig, out, "fig_ablation"))
    return files
