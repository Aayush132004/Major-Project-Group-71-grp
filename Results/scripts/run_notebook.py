"""Execute notebooks/results_analysis.ipynb in place, so the saved notebook shows its outputs.

    python scripts/run_notebook.py
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    path = ROOT / "notebooks" / "results_analysis.ipynb"
    nb = nbformat.read(str(path), as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, str(path))
    print(f"executed {path}")
    sys.exit(0)
