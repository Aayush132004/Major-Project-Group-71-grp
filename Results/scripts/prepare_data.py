"""Download and pre-process the benchmark corpora (run once).

    python scripts/prepare_data.py                 # MS MARCO + CUAD + Spider
    python scripts/prepare_data.py --bird D:/bird/dev_databases
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plannerllm.data.prepare import prepare_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="msmarco cuad spider")
    ap.add_argument("--bird", default=None, help="path to BIRD dev_databases folder")
    args = ap.parse_args()
    for name, path in prepare_all(args.only, args.bird).items():
        print(f"[ok] {name:8s} -> {path}")
