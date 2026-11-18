# PlannerLLM source code

Aayush Sharma (23U02006), IIIT Bhopal. This folder contains the Python optimizer and evaluation code, FastAPI backend, React source and built frontend, tests and configuration. Keep it next to Results inside PLLM. Prepared data, model snapshots and the notebooks are in Results.

## Windows setup

Open PowerShell in Source_Code. Install Python 3.11, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On the original development PC, the existing Python at C:\Users\aayus\plannerllm_venv\Scripts\python.exe can be used in place of the new venv.

## Run the application

```powershell
$env:PLANNER_DATA_DIR = (Resolve-Path ../Results/data).Path
$env:HF_HOME = (Resolve-Path ../Results/model_cache).Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
.\.venv\Scripts\python.exe scripts/serve.py
```

Open http://127.0.0.1:8000. The app uses simulator mode for local demos without a key. For live Groq execution, copy .env.example to .env, set GROQ_API_KEY locally and restart the server. The app also exposes /api/health. Use the separate Results notebooks for the academic Groq evaluation. They include the recorded model responses and distinguish them from model calls made during the rerun.

## Test and run experiments

After setting the data and model paths above, run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

For a new CLI experiment with a Groq key in .env, set a dedicated run and cache folder so the packaged evidence stays intact:
```powershell
$env:PLANNER_RUNS_DIR = Join-Path (Get-Location) 'live_runs'
.\.venv\Scripts\python.exe scripts/run_experiments.py --backend groq --profile smoke --seed 0 --only baselines --no-report
```

The CLI reuses the original cache under PLANNER_DATA_DIR. To use an empty cache without changing the packaged Results, copy Results/data into a separate directory, omit the SQLite response caches in the copy and point PLANNER_DATA_DIR to that directory. See Results/README.md for notebook reproduction steps.

To rebuild the frontend after editing it, run npm ci and npm run build from web/. The supplied web/dist already serves the console.
