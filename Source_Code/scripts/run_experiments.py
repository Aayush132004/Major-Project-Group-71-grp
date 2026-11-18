"""Run the Section-10 experiment suite and build the report assets.

    python scripts/run_experiments.py --backend sim --profile smoke          # offline check
    python scripts/run_experiments.py --profile quick                        # Groq, ~1-2 h
    python scripts/run_experiments.py --profile full --only baselines gate   # selected parts

Every Groq response is cached in data/cache, so an interrupted run resumes
instantly.  Figures, LaTeX tables and CSVs land in runs/<model>/report/.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plannerllm.env import load_env  # noqa: E402
from plannerllm.eval.experiments import ALL, PROFILES, WORKLOADS, run_experiment  # noqa: E402
from plannerllm.settings import Settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["groq", "sim"], default=None)
    ap.add_argument("--model", default=None, help="Groq model id (default from config)")
    ap.add_argument("--profile", choices=list(PROFILES), default="quick")
    ap.add_argument("--only", nargs="*", default=None, choices=ALL)
    ap.add_argument("--workloads", nargs="*", default=None, choices=WORKLOADS + ["bird"])
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="run seed: picks the query groups, questions and samples")
    args = ap.parse_args()

    load_env()
    settings = Settings.load()
    if args.backend:
        settings.llm.backend = args.backend
    if args.model:
        settings.llm.model = args.model
    settings.sampling.task_seed = args.seed
    settings.sampling.seed = 13 + args.seed
    t0 = time.time()
    for name in args.only or ALL:
        print(f"\n=== {name} ({args.profile}, {settings.llm.backend}) ===", flush=True)
        path = run_experiment(name, settings, args.profile, args.workloads,
                              progress=lambda m: print("  ..", m, flush=True))
        print(f"  saved {path}")
    if not args.no_report:
        from plannerllm.eval.reporting import build_report
        out = build_report(settings)
        print(f"\nreport assets -> {out}")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
