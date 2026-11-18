# PlannerLLM evaluation

Open this folder in VS Code or Jupyter with a Python 3.11 kernel.

- `01_implementation_execution.ipynb` shows the implementation and runs the nine experiments for each of the three seeds.
- `02_results_analysis.ipynb` generates the tables, figures and consistency checks from the original runs.
- `runs/` holds the original results. `reproduced_runs/recorded_groq/` holds the notebook rerun. `original_execution_logs/` includes the original terminal logs. Per-query predictions and intermediate decisions are in the JSON files, with CSV exports in `raw_results/`.

## Run the notebooks

Create an environment from PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select the `.venv` kernel in notebook 01, then choose **Restart Kernel** and **Run All**. Repeat in a separate kernel for notebook 02. Check the last cell of notebook 01 for completed experiments and notebook 02 for **ALL CONSISTENCY CHECKS PASSED**. The notebooks already contain saved outputs for review.

Notebook 01 automatically uses bundled Groq replies for cache hits. To let it request missing replies, place your own `GROQ_API_KEY` in a local `.env` in this folder, or set it in your environment. Restart the kernel and select **Run All**. No mode switch is needed. API requests use your Groq quota. Without a key, the notebook logs cache misses as incomplete.

New results go to `notebook_reruns/`. The saved rerun fetched 41 Groq replies and matched 537 of 540 original baseline predictions and scores. See report Section 9.12 for the comparison. Set `PLANNER_NOTEBOOK_MODE=live` only to start with an empty response cache.
