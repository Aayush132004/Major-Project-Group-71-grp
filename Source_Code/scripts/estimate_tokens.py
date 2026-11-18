"""Estimate how many *unique* Groq tokens an experiment profile will need.

Runs the suite on the offline simulator with an empty response cache; every
distinct prompt is stored once, exactly as the Groq cache would store it, so
the cache total approximates the API tokens the real run will be billed for.

    python scripts/estimate_tokens.py --profile quick
"""
import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plannerllm.eval.experiments import ALL, PROFILES, run_experiment  # noqa: E402
from plannerllm.settings import Settings  # noqa: E402

FREE_TIER_TPD = 500_000   # llama-3.1-8b-instant free tier, tokens per day (check console.groq.com)


def unique_tokens(path: str) -> int:
    con = sqlite3.connect(path)
    n = con.execute("SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM responses").fetchone()[0]
    print(f"      ({con.execute('SELECT COUNT(*) FROM responses').fetchone()[0]} unique calls so far)")
    con.close()
    return int(n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--only", nargs="*", default=None, choices=ALL)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp()) / "estimate_cache.sqlite"
    s = Settings.load()
    s.llm.backend = "sim"
    s.llm.sim_cache_path = str(tmp)
    import plannerllm.eval.experiments as E
    E.RUNS_DIR = Path(tempfile.mkdtemp())          # do not overwrite real sim results
    total_prev = 0
    print(f"profile {args.profile}: {PROFILES[args.profile]}")
    for name in args.only or ALL:
        run_experiment(name, s, args.profile, progress=lambda m: None)
        total = unique_tokens(str(tmp)) if tmp.exists() else 0
        print(f"  {name:12s} +{total - total_prev:>10,} unique tokens   (running total {total:,})")
        total_prev = total
    print(f"\n~{total_prev:,} tokens  =  ~{total_prev / FREE_TIER_TPD:.1f} days of the Groq free tier "
          f"(~${total_prev * 0.06 / 1e6:.2f} on the paid dev tier)")
