"""Combine the per-seed runs into statistics, tables and figures.

    python scripts/aggregate_results.py                 # runs/openai_gpt-oss-20b_seed*/ -> ..._aggregate/
    python scripts/aggregate_results.py --tag <model tag>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plannerllm.eval.aggregate import build, seed_dirs  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="openai_gpt-oss-20b")
    args = ap.parse_args()
    seeds = [s for s, _ in seed_dirs(args.tag)]
    if not seeds:
        raise SystemExit(f"no runs found for runs/{args.tag}_seed*/ (run scripts/run_experiments.py --seed k first)")
    print(f"seeds: {seeds}")
    print(f"written: {build(args.tag)}")
