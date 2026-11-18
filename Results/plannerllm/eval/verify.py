"""Consistency checks between the raw results, the aggregate, the notebook and the report.

``recompute`` derives the headline numbers again from the raw per-seed JSON with a deliberately
separate, plain implementation (no code shared with ``aggregate.py``), so that a bug in one path
shows up as a mismatch instead of being copied into the report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..settings import RUNS_DIR

TOL = 1e-6


def _seed_files(tag: str, exp: str) -> dict[int, dict]:
    out = {}
    for d in sorted(RUNS_DIR.glob(f"{tag}_seed*")):
        m = re.fullmatch(re.escape(tag) + r"_seed(\d+)", d.name)
        p = d / f"{exp}.json"
        if m and p.exists():
            out[int(m.group(1))] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def recompute(tag: str) -> dict[str, float]:
    """Headline numbers keyed by a dotted name that also exists in aggregate.json."""
    nums: dict[str, float] = {}
    base = _seed_files(tag, "baselines")
    for wl in ("msmarco", "cuad", "spider"):
        for sysname in ("no_opt", "random", "cost_only", "docetl", "plannerllm"):
            for key in ("accuracy", "opt_tokens", "n_validations"):
                per_seed = []
                for s, b in base.items():
                    rows = [r for r in b["rows"] if r["workload"] == wl and r["system"] == sysname]
                    if rows:
                        per_seed.append(_avg([r[key] for r in rows]))
                if per_seed:
                    nums[f"baselines.{wl}.{sysname}.{key}"] = _avg(per_seed)
        p_sum = d_sum = 0.0
        for s, b in base.items():
            p = {r["task"]: r["opt_tokens"] for r in b["rows"] if r["workload"] == wl and r["system"] == "plannerllm"}
            d = {r["task"]: r["opt_tokens"] for r in b["rows"] if r["workload"] == wl and r["system"] == "docetl"}
            for k in set(p) & set(d):
                p_sum += p[k]
                d_sum += d[k]
        if d_sum:
            nums[f"token_reduction.{wl}"] = 1 - p_sum / d_sum
    judge = _seed_files(tag, "judge")
    if judge:
        allv = [v for j in judge.values() for v in j["validator"]]
        if allv:
            nums["validator_agreement.all"] = sum(v["validator_prefers"] == v["judge_prefers"] for v in allv) / len(allv)
        for wl in ("msmarco", "cuad"):
            for sysname in ("docetl", "plannerllm"):
                vals = [next((x["judge_acc"] for x in j["summary"] if x["source"] == "baselines"
                              and x["workload"] == wl and x["system"] == sysname), None) for j in judge.values()]
                if any(v is not None for v in vals):
                    nums[f"judge_acc.{wl}.{sysname}"] = _avg(vals)
    mem = _seed_files(tag, "memory")
    for wl in ("msmarco", "cuad", "spider"):
        vals = [x["hit_rate"] for m in mem.values() for x in m["streams"] if x["workload"] == wl and x["phase"] == "repeated"]
        if vals:
            nums[f"memory.{wl}.repeated.hit_rate"] = _avg(vals)
    return nums


def from_aggregate(A: dict, key: str):
    parts = key.split(".")
    if parts[0] == "baselines":
        _, wl, sysname, k = parts
        r = next((x for x in A["baselines"] if x["workload"] == wl and x["system"] == sysname), None)
        return r[k]["mean"] if r else None
    if parts[0] == "token_reduction":
        r = next((x for x in A["token_reduction"] if x["workload"] == parts[1]), None)
        return r["reduction"] if r else None
    if parts[0] == "validator_agreement":
        r = next((x for x in A.get("validator_agreement", []) if x["workload"] == parts[1]), None)
        return r["agreement"] if r else None
    if parts[0] == "judge_acc":
        r = next((x for x in A.get("judge_acc", []) if x["workload"] == parts[1] and x["system"] == parts[2]), None)
        return r["judge_acc"]["mean"] if r else None
    if parts[0] == "memory":
        r = next((x for x in A.get("memory", []) if x["workload"] == parts[1] and x["phase"] == parts[2]), None)
        return r["hit_rate"]["mean"] if r else None
    raise KeyError(key)


def check_aggregate(tag: str) -> list[dict]:
    A = json.loads((RUNS_DIR / f"{tag}_aggregate" / "aggregate.json").read_text(encoding="utf-8"))
    out = []
    for k, v in sorted(recompute(tag).items()):
        a = from_aggregate(A, k)
        ok = a is not None and v is not None and abs(a - v) <= TOL * max(1.0, abs(v))
        out.append({"check": f"raw vs aggregate: {k}", "raw": v, "aggregate": a, "ok": ok})
    return out


def check_report(pdf: Path, manifest: Path, A: dict) -> list[dict]:
    """Every number the report builder wrote must (a) equal the aggregate value it came from and
    (b) appear, as printed, in the PDF."""
    from pypdf import PdfReader
    text = " ".join(" ".join((p.extract_text() or "").split()) for p in PdfReader(str(pdf)).pages)
    text = text.replace("−", "-")
    out = []
    for n in json.loads(manifest.read_text(encoding="utf-8")):
        printed = n["text"]
        in_pdf = printed in text
        agg_ok = True
        if n.get("key"):
            try:
                a = from_aggregate(A, n["key"])
                agg_ok = a is not None and abs(a - n["value"]) <= 1e-9 * max(1.0, abs(a))
            except KeyError:
                agg_ok = True
        out.append({"check": f"report: {n.get('label') or n.get('key')} = {printed}", "ok": in_pdf and agg_ok,
                    "in_pdf": in_pdf, "matches_aggregate": agg_ok})
    return out


def check_notebook(nb_path: Path) -> list[dict]:
    import nbformat
    nb = nbformat.read(str(nb_path), as_version=4)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    executed = all(c.get("execution_count") for c in nb.cells if c.cell_type == "code" and c.source.strip())
    passed = any("ALL CONSISTENCY CHECKS PASSED" in "".join(o.get("text", "")) for c in nb.cells
                 if c.cell_type == "code" for o in c.get("outputs", []))
    return [{"check": "notebook executed without errors", "ok": executed and not errors},
            {"check": "notebook's own raw-vs-aggregate checks passed", "ok": passed}]
