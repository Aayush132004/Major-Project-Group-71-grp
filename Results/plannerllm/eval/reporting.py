"""Turn experiment JSON into report-ready figures, LaTeX tables and CSVs.

Output: ``runs/<model>/report/{fig_*.png, tab_*.tex, tab_*.csv, transcripts.md}``.
Colours follow a fixed categorical order so a system keeps its colour in
every figure; figures never use a second y-axis (two measures -> two panels).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..settings import Settings  # noqa: E402
from .experiments import load, out_dir  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SYSTEM_COLOR = {"plannerllm": SERIES[0], "docetl": SERIES[1], "cost_only": SERIES[2], "random": SERIES[3],
                "no_opt": SERIES[4], "no_gate": SERIES[6], "no_memory": SERIES[5],
                "symbolic_only": SERIES[7], "neural_only": SERIES[3]}
LABEL = {"plannerllm": "PlannerLLM", "docetl": "DocETL-style", "cost_only": "Cost-only", "random": "Random",
         "no_opt": "No optimisation", "no_gate": "- VoI gate", "no_memory": "- plan memory",
         "symbolic_only": "symbolic cost only", "neural_only": "neural cost only"}
WL_LABEL = {"msmarco": "MS MARCO", "cuad": "CUAD", "spider": "Spider", "bird": "BIRD"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10, "axes.labelsize": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 10, "figure.dpi": 110, "mathtext.fontset": "stix", "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _kfmt(v, _pos=None) -> str:
    a = abs(v)
    return f"{v / 1e6:.1f}M" if a >= 1e6 else f"{v / 1e3:.0f}k" if a >= 1e4 else f"{v / 1e3:.1f}k" if a >= 1e3 else f"{v:g}"


def _save(fig, out: Path, name: str) -> str:
    from matplotlib.ticker import FuncFormatter
    for ax in fig.axes:   # thousands as 12k, never 12000 overlapping its neighbour
        for axis in (ax.xaxis, ax.yaxis):
            locs = np.abs(np.asarray(axis.get_majorticklocs(), dtype=float))
            if axis.get_scale() == "linear" and locs.size and locs.max() >= 1000:
                axis.set_major_formatter(FuncFormatter(_kfmt))
    if fig._suptitle is not None:   # the report caption carries the title
        fig._suptitle.set_visible(False)
    if len(fig.axes) == 1:
        fig.axes[0].set_title("", loc="left")
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _grouped_bars(ax, groups: list[str], series: list[str], value, fmt="{:.2f}", log=False) -> None:
    n = len(series)
    width = 0.8 / max(n, 1)
    x = np.arange(len(groups))
    for i, sname in enumerate(series):
        vals = [value(g, sname) for g in groups]
        ys = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, ys, width * 0.92, color=SYSTEM_COLOR.get(sname, SERIES[i % 8]),
                      label=LABEL.get(sname, sname), edgecolor="white", linewidth=0.8)
        if n <= 5:
            for b, v in zip(bars, vals):
                if v is not None:
                    ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                                ha="center", va="bottom", fontsize=8.5, color=MUTED, xytext=(0, 1.5),
                                textcoords="offset points")
    ax.set_xticks(x, [WL_LABEL.get(g, g) for g in groups])
    if log:
        ax.set_yscale("log")
    ax.grid(axis="x", visible=False)


def _write_table(out: Path, name: str, header: list[str], rows: list[list], caption: str) -> None:
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    fmt = lambda v: ("--" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v).replace("_", r"\_"))
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}",
             r"\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(fmt(v) for v in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------- figures
def fig_baselines(d: dict, out: Path) -> list[str]:
    summ = d["summary"]
    wls = sorted({r["workload"] for r in summ}, key=lambda w: list(WL_LABEL).index(w) if w in WL_LABEL else 9)
    systems = [s for s in ["no_opt", "random", "cost_only", "docetl", "plannerllm"] if any(r["system"] == s for r in summ)]
    get = lambda key: (lambda w, s: next((r[key] for r in summ if r["workload"] == w and r["system"] == s), None))
    files = []
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    _grouped_bars(ax, wls, systems, get("accuracy"))
    ax.set_ylabel("held-out accuracy (F1 / EX)")
    ax.set_title("Accuracy of the chosen plan", loc="left", fontsize=10, color=INK)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    files.append(_save(fig, out, "fig_baselines_accuracy"))
    opt_sys = [s for s in systems if s in ("docetl", "plannerllm", "cost_only")]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    _grouped_bars(ax, wls, opt_sys, get("opt_tokens"), fmt="{:,.0f}")
    ax.set_ylabel("optimisation tokens per task")
    ax.set_title("Optimisation-time LLM spend", loc="left", fontsize=10, color=INK)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    files.append(_save(fig, out, "fig_baselines_tokens"))
    rows = [[WL_LABEL.get(r["workload"], r["workload"]), LABEL.get(r["system"], r["system"]),
             r.get("accuracy"), r.get("hallucinated"), r.get("opt_tokens"), r.get("exec_tokens"),
             r.get("exec_ms"), r.get("n_validations"), r.get("n_memory_hits"), r.get("n_gate_skips"),
             r.get("decision_agreement")] for r in summ]
    _write_table(out, "tab_baselines", ["Workload", "System", "Acc", "Halluc.", "Opt tok", "Exec tok",
                                        "Exec ms", "Valid.", "Mem hits", "Gate skips", "Agree"], rows,
                 "Baselines vs PlannerLLM (means per task).")
    return files


def fig_pareto(d: dict, out: Path) -> list[str]:
    pts = d["points"]
    wls = [w for w in WL_LABEL if any(p["workload"] == w for p in pts)]
    files = []
    fig, axes = plt.subplots(1, len(wls), figsize=(2.1 * len(wls), 2.6), squeeze=False)
    for ax, wl in zip(axes[0], wls):
        for sysname in ("docetl", "no_gate", "plannerllm"):
            ps = sorted([p for p in pts if p["workload"] == wl and p["system"] == sysname and p["sweep"] == "budget"
                         and p["accuracy"] is not None], key=lambda p: p["opt_tokens"] or 0)
            if ps:
                ax.plot([p["opt_tokens"] for p in ps], [p["accuracy"] for p in ps], marker="o", ms=4, lw=2,
                        color=SYSTEM_COLOR[sysname], label=LABEL[sysname])
        ax.set_title(WL_LABEL[wl], loc="left", fontsize=11)
        ax.set_xlabel("optimisation tokens / task")
        ax.set_ylim(0, max(0.1, ax.get_ylim()[1]))
    axes[0][0].set_ylabel("held-out accuracy")
    axes[0][-1].legend(fontsize=10, loc="lower right")
    fig.suptitle("Accuracy vs token budget", x=0.01, ha="left", fontsize=10)
    files.append(_save(fig, out, "fig_pareto_budget"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for i, wl in enumerate(wls):
        ps = sorted([p for p in pts if p["workload"] == wl and p.get("sweep") == "theta"], key=lambda p: p["theta"])
        if not ps:
            continue
        th = [p["theta"] for p in ps]
        axes[0].plot(th, [p["accuracy"] for p in ps], marker="o", ms=4, lw=2, color=SERIES[i], label=WL_LABEL[wl])
        axes[1].plot(th, [p["opt_tokens"] for p in ps], marker="o", ms=4, lw=2, color=SERIES[i], label=WL_LABEL[wl])
    for ax, lab in zip(axes, ("held-out accuracy", "optimisation tokens / task")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"gate threshold $\theta$")
        ax.set_ylabel(lab)
    axes[1].legend(fontsize=10)
    fig.suptitle(r"$\theta$ sweep: the accuracy / spend dial", x=0.01, ha="left", fontsize=10)
    files.append(_save(fig, out, "fig_theta_sweep"))
    return files


def fig_gate(d: dict, out: Path) -> list[str]:
    files = []
    sc = [s for s in d["overall"]["scores"] if s["precision"] is not None]
    if sc:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
        th = [s["theta"] for s in sc]
        axes[0].plot(th, [s["precision"] for s in sc], marker="o", ms=4, lw=2, color=SERIES[0], label="precision")
        axes[0].plot(th, [s["recall"] for s in sc], marker="s", ms=4, lw=2, color=SERIES[1], label="recall")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("score")
        axes[0].legend(fontsize=10)
        axes[1].plot(th, [s["spend_saved"] for s in sc], marker="o", ms=4, lw=2, color=SERIES[2])
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("validation spend avoided")
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel(r"$\theta$")
        fig.suptitle("VoI gate as a flip predictor (shadow mode)", x=0.01, ha="left", fontsize=10)
        files.append(_save(fig, out, "fig_gate_pr"))
        _write_table(out, "tab_gate", ["theta", "n", "admitted", "precision", "recall", "flip rate", "spend saved"],
                     [[s["theta"], s["n"], s["admitted"], s["precision"], s["recall"], s["flip_rate"],
                       s["spend_saved"]] for s in d["overall"]["scores"]],
                     "Gate precision/recall across $\\theta$.")
    cal = d["overall"]["calibration"]
    if cal:
        fig, ax = plt.subplots(figsize=(3.6, 3.3))
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="perfect calibration")
        ax.plot([c["predicted"] for c in cal], [c["observed"] for c in cal], marker="o", ms=5, lw=2,
                color=SERIES[0], label="P(flip)")
        for c in cal:
            ax.annotate(f"n={c['n']}", (c["predicted"], c["observed"]), fontsize=8.5, color=MUTED,
                        xytext=(4, -9), textcoords="offset points")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted flip probability")
        ax.set_ylabel("observed flip frequency")
        ax.legend(fontsize=10, loc="upper left")
        ax.set_title("Gate calibration", loc="left", fontsize=10)
        files.append(_save(fig, out, "fig_gate_calibration"))
    return files


def fig_memory(d: dict, out: Path) -> list[str]:
    files = []
    st = d["streams"]
    phases = ["first-seen", "repeated", "prompt_edit", "config_change"]
    wls = [w for w in WL_LABEL if any(s["workload"] == w for s in st)]
    if st:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
        x = np.arange(len(phases))
        w = 0.8 / max(len(wls), 1)
        for i, wl in enumerate(wls):
            hr = [next((s["hit_rate"] for s in st if s["workload"] == wl and s["phase"] == p), 0) for p in phases]
            tk = [next((s["opt_tokens"] or 0 for s in st if s["workload"] == wl and s["phase"] == p), 0) for p in phases]
            off = (i - (len(wls) - 1) / 2) * w
            axes[0].bar(x + off, hr, w * 0.92, color=SERIES[i], label=WL_LABEL[wl], edgecolor="white")
            axes[1].bar(x + off, tk, w * 0.92, color=SERIES[i], label=WL_LABEL[wl], edgecolor="white")
        for ax, lab in zip(axes, ("signature hit rate", "optimisation tokens / task")):
            ax.set_xticks(x, ["first", "repeat", "prompt", "config"], fontsize=10)
            ax.set_ylabel(lab)
            ax.grid(axis="x", visible=False)
        axes[0].legend(fontsize=10)
        fig.suptitle("Plan memory over repeated and perturbed query shapes", x=0.01, ha="left", fontsize=10)
        files.append(_save(fig, out, "fig_memory_hits"))
    curve = d.get("decay_curve") or []
    if curve:
        fig, ax = plt.subplots(figsize=(4.2, 2.9))
        keys = [k for k in curve[0] if k.startswith("kappa")]
        for i, k in enumerate(keys):
            ax.plot([c["age_days"] for c in curve], [c[k] for c in curve], lw=2, color=SERIES[i], label=k.replace("kappa", "κ"))
        ax.axhline(d.get("reuse_threshold", 0.55), ls="--", lw=1, color=MUTED)
        ax.annotate("reuse threshold", (60, d.get("reuse_threshold", 0.55)), ha="right", va="bottom", fontsize=7, color=MUTED)
        ax.set_xlabel("verdict age (days)")
        ax.set_ylabel("confidence δ(t), n=2")
        ax.legend(fontsize=10)
        ax.set_title("Confidence decay", loc="left", fontsize=10)
        files.append(_save(fig, out, "fig_memory_decay"))
    rows = [[WL_LABEL.get(s["workload"]), s["phase"], s["hit_rate"], s["reuse_accuracy"], s["n_audited"],
             s["opt_tokens"], s["accuracy"]] for s in st]
    if rows:
        _write_table(out, "tab_memory", ["Workload", "Phase", "Hit rate", "Reuse acc", "Audited", "Opt tok", "Acc"],
                     rows, "Plan-memory hit rate and verdict-reuse accuracy.")
    return files


def fig_cost(d: dict, out: Path) -> list[str]:
    files = []
    sc = d.get("scatter") or []
    if sc:
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
        for i, wl in enumerate([w for w in WL_LABEL if any(s["workload"] == w for s in sc)]):
            pts = [s for s in sc if s["workload"] == wl]
            axes[0].scatter([p["sym_usd"] for p in pts], [p["obs_usd"] for p in pts], s=16, color=SERIES[i],
                            label=WL_LABEL[wl], edgecolor="white", linewidth=0.6)
            axes[1].scatter([p["sym_ms"] for p in pts], [p["obs_ms"] for p in pts], s=16, color=SERIES[i],
                            label=WL_LABEL[wl], edgecolor="white", linewidth=0.6)
        for ax, lab in zip(axes, ("USD per run", "latency ms per run")):
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot([max(lo, 1e-9), hi], [max(lo, 1e-9), hi], ls="--", lw=1, color=MUTED)
            ax.set_xlabel(f"symbolic estimate ({lab})")
            ax.set_ylabel(f"observed ({lab})")
        axes[0].legend(fontsize=10)
        fig.suptitle("Symbolic backbone vs observed cost", x=0.01, ha="left", fontsize=10)
        files.append(_save(fig, out, "fig_cost_scatter"))
    summ = d.get("summary") or []
    if summ:
        splits = [s for s in summ if s["target"] == "usd"]
        fig, ax = plt.subplots(figsize=(6.3, 2.8))
        models = ["symbolic", "neural", "hybrid"]
        x = np.arange(len(splits))
        for i, m in enumerate(models):
            vals = [s.get(m, {}).get("median_q") for s in splits]
            ax.bar(x + (i - 1) * 0.26, [v or 0 for v in vals], 0.24, color=SERIES[[0, 1, 2][i]], label=m,
                   edgecolor="white")
        ax.set_xticks(x, [s["split"].replace("zero-shot:", "zero-shot\n") for s in splits], fontsize=10)
        ax.set_ylabel("median q-error (USD)")
        ax.axhline(1, color=MUTED, lw=1)
        ax.legend(fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_title("Cost-model calibration error (lower is better, 1 = exact)", loc="left", fontsize=10)
        files.append(_save(fig, out, "fig_cost_qerror"))
        rows = []
        for s in summ:
            for m in ("symbolic", "neural", "hybrid"):
                if m in s:
                    e = s[m]
                    rows.append([s["split"], s["target"], m, e["mae_log10"], e["median_rel"], e["p90_rel"],
                                 e["median_q"], e["p90_q"]])
        _write_table(out, "tab_cost_model", ["Split", "Target", "Model", "MAE log10", "Med rel", "P90 rel",
                                             "Med q", "P90 q"], rows, "Cost-model calibration error.")
    return files


def fig_latency(d: dict, out: Path) -> list[str]:
    files = []
    vp = [r for r in d.get("validation_path", []) if r.get("exec_ms")]
    if vp:
        wls = [w for w in WL_LABEL if any(r["workload"] == w for r in vp)]
        systems = [s for s in ["no_opt", "cost_only", "docetl", "plannerllm"] if any(r["system"] == s for r in vp)]
        short = {"no_opt": "none", "cost_only": "cost", "docetl": "DocETL", "plannerllm": "PlannerLLM"}
        comps = [("db_ms", "DB operators", SERIES[2]), ("ann_ms", "ANN + rerank", SERIES[0]), ("llm_ms", "LLM", SERIES[1])]
        fig, axes = plt.subplots(1, len(wls), figsize=(2.1 * len(wls), 2.7), squeeze=False, sharey=True)
        for ax, wl in zip(axes[0], wls):
            for i, sname in enumerate(systems):
                r = next((r for r in vp if r["workload"] == wl and r["system"] == sname), None)
                if not r:
                    continue
                bottom = 0.0
                for key, lab, col in comps:
                    v = r.get(key) or 0.0
                    ax.bar(i, v, 0.7, bottom=bottom, color=col, edgecolor="white", linewidth=1,
                           label=lab if (wl == wls[0] and i == 0) else None)
                    bottom += v
            ax.set_xticks(range(len(systems)), [short[s] for s in systems], fontsize=9, rotation=35, ha="right")
            ax.set_title(WL_LABEL[wl], loc="left", fontsize=11)
            ax.grid(axis="x", visible=False)
        axes[0][0].set_ylabel("ms per query (execution)")
        fig.legend(loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.2))
        fig.suptitle("Wall-clock decomposition of the chosen plans", x=0.01, ha="left", fontsize=10)
        files.append(_save(fig, out, "fig_latency_breakdown"))
    sweep = d.get("ann_sweep") or []
    if sweep:
        wls = [w for w in WL_LABEL if any(r["workload"] == w for r in sweep)]
        fig, axes = plt.subplots(1, len(wls), figsize=(2.1 * len(wls), 2.6), squeeze=False)
        for ax, wl in zip(axes[0], wls):
            for i, fam in enumerate(("hnsw", "ivf", "flat")):
                ps = sorted([r for r in sweep if r["workload"] == wl and r["family"] == fam and r["k"] == 10],
                            key=lambda r: r["ms"])
                ax.plot([p["ms"] for p in ps], [p["recall"] for p in ps], marker="o", ms=4, lw=2,
                        color=SERIES[i], label=fam.upper())
            ax.set_title(WL_LABEL[wl], loc="left", fontsize=11)
            ax.set_xlabel("median probe latency (ms)")
            ax.set_ylim(0, 1.05)
        axes[0][0].set_ylabel("recall@10 vs exact")
        axes[0][-1].legend(fontsize=10, loc="lower right")
        fig.suptitle("ANN families: time / recall trade-off (k = 10)", x=0.01, ha="left", fontsize=10)
        files.append(_save(fig, out, "fig_ann_sweep"))
    probe = d.get("probe") or []
    if probe:
        rows = []
        for wl in [w for w in WL_LABEL if any(r["workload"] == w for r in probe)]:
            for fam in ("hnsw", "ivf", "flat"):
                ps = [r for r in probe if r["workload"] == wl and r["family"] == fam]
                if ps:
                    rows.append([WL_LABEL[wl], fam, float(np.mean([p["sel"] for p in ps])),
                                 float(np.mean([p["filtered_recall"] for p in ps])), float(np.mean([p["post_recall"] for p in ps])),
                                 float(np.mean([p["filtered_ms"] for p in ps])), float(np.mean([p["post_ms"] for p in ps]))])
        _write_table(out, "tab_filtered_probe", ["Workload", "Index", "Selectivity", "Filtered recall",
                                                 "Post-filter recall", "Filtered ms", "Post-filter ms"], rows,
                     "Filtered vector probe vs post-filtering at the workload's predicate selectivity.")
    return files



def fig_ablation(d: dict, out: Path) -> list[str]:
    summ = d["summary"]
    wls = [w for w in WL_LABEL if any(r["workload"] == w for r in summ)]
    variants = [v for v in ["plannerllm", "no_gate", "no_memory", "symbolic_only", "neural_only"]
                if any(r["system"] == v for r in summ)]
    get = lambda key: (lambda w, s: next((r[key] for r in summ if r["workload"] == w and r["system"] == s), None))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    _grouped_bars(axes[0], wls, variants, get("accuracy"), fmt="{:.2f}")
    _grouped_bars(axes[1], wls, variants, get("opt_tokens"), fmt="{:,.0f}")
    axes[0].set_ylabel("held-out accuracy")
    axes[1].set_ylabel("optimisation tokens / task")
    for ax in axes:
        for t in list(ax.texts):   # five series per group: values go in the table instead
            t.remove()
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.1, -0.12), fontsize=10)
    fig.suptitle(f"Ablations (same budget: {d.get('budget') or 'unlimited'} tokens)", x=0.01, ha="left", fontsize=10)
    rows = [[WL_LABEL.get(r["workload"]), LABEL.get(r["system"], r["system"]), r.get("accuracy"),
             r.get("opt_tokens"), r.get("n_validations"), r.get("n_memory_hits"), r.get("n_gate_skips")] for r in summ]
    _write_table(out, "tab_ablation", ["Workload", "Variant", "Acc", "Opt tok", "Valid.", "Mem hits", "Gate skips"],
                 rows, "Ablation study under an identical token budget.")
    return [_save(fig, out, "fig_ablation")]


def transcripts_md(d: dict, out: Path) -> str:
    lines = ["# Validation transcripts (Section 10.8)", "",
             f"accepted: {d['counts']['accepted']}, gate-skipped: {d['counts']['gate_skipped']}, "
             f"memory-reused: {d['counts']['memory_reused']}", ""]
    for t in d["transcripts"]:
        lines += [f"## {WL_LABEL.get(t['workload'], t['workload'])} / {t['task']} - {t['directive']} "
                  f"({t['source']}, {t['verdict']})", "", f"* rewrite: {t['detail']}", f"* candidate plan: `{t['plan']}`"]
        if t.get("gate"):
            lines.append(f"* gate: {t['gate'].get('rationale', '')}")
        if t.get("memory"):
            m = t["memory"]
            lines.append(f"* memory: {m['match']} hit, confidence {m['confidence']:.2f}, age {m['age_days']:.1f} d")
        v = t.get("validation")
        if v:
            lines.append(f"* criteria: {', '.join(c['name'] for c in v.get('criteria', []))}")
            lines.append(f"* ratings: candidate {v['score_c']:.2f} vs incumbent {v['score_p']:.2f}"
                         + (f", pairwise win-rate {v['pairwise_win_rate']:.2f}" if v.get("pairwise_win_rate") is not None else ""))
            for it in v.get("items", [])[:2]:
                lines += ["", f"> **Q:** {it['question']}", f"> **candidate ({it['score_c']:.1f}):** {it['candidate'][:300]}",
                          f"> **incumbent ({it['score_p']:.1f}):** {it['incumbent'][:300]}"]
        lines.append("")
    path = out / "transcripts.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.name


def build_report(settings: Settings) -> Path:
    _style()
    out = out_dir(settings) / "report"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in [("baselines", fig_baselines), ("pareto", fig_pareto), ("gate", fig_gate),
                     ("memory", fig_memory), ("cost_model", fig_cost), ("latency", fig_latency),
                     ("ablation", fig_ablation)]:
        d = load(settings, name)
        if d:
            try:
                made += fn(d, out)
            except Exception as exc:   # one broken figure must not hide the rest
                print(f"[report] {name}: {exc}")
    d = load(settings, "transcripts")
    if d:
        made.append(transcripts_md(d, out))
    (out / "index.json").write_text(json.dumps(made, indent=1), encoding="utf-8")
    return out
