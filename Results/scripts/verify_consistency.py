"""Check that the raw results, the aggregate, the notebook and the report agree.

    python scripts/verify_consistency.py
    python scripts/verify_consistency.py --report path/to/report.pdf --manifest path/to/numbers.json

Checks
1. every headline number recomputed from the raw per-seed JSON (separate code path) equals the
   value in aggregate.json;
2. notebooks/results_analysis.ipynb has been executed without errors and its own checks passed;
3. if a report PDF and its number manifest are given, every number the report builder inserted
   equals the aggregate value and appears in the PDF as printed.
Exit code 1 if any check fails.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plannerllm.eval.verify import check_aggregate, check_notebook, check_report  # noqa: E402
from plannerllm.settings import RUNS_DIR  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="openai_gpt-oss-20b")
    ap.add_argument("--notebook", default=str(ROOT / "notebooks" / "results_analysis.ipynb"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    checks = check_aggregate(args.tag)
    if Path(args.notebook).exists():
        checks += check_notebook(Path(args.notebook))
    else:
        checks.append({"check": "notebook exists", "ok": False})
    if args.report and args.manifest:
        A = json.loads((RUNS_DIR / f"{args.tag}_aggregate" / "aggregate.json").read_text(encoding="utf-8"))
        checks += check_report(Path(args.report), Path(args.manifest), A)
    bad = [c for c in checks if not c["ok"]]
    for c in checks:
        print(("ok    " if c["ok"] else "FAIL  ") + c["check"])
    print(f"\n{len(checks) - len(bad)} of {len(checks)} checks passed")
    sys.exit(1 if bad else 0)
