"""HTTP API for the PlannerLLM console (FastAPI).

    uvicorn server.app:app --port 8000          (or: python scripts/serve.py)

Optimisation runs execute on a worker thread per request and stream their
decision events to the browser over Server-Sent Events, so the UI can draw
Algorithm 1 as it happens.  Interactive runs share one persistent plan memory
per (backend, workload), which is what lets a user watch the memory learn.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from plannerllm.core.directives import DirectiveStore
from plannerllm.core.optimizer import make_system
from plannerllm.core.pipeline import Pipeline, PipelineError
from plannerllm.core.session import Session, build_session
from plannerllm.data.workloads import available_workloads
from plannerllm.env import load_env
from plannerllm.eval.experiments import ALL as EXPERIMENT_NAMES
from plannerllm.eval.experiments import PROFILES, run_experiment
from plannerllm.settings import DATA_DIR, ROOT, RUNS_DIR, Settings

load_env()
app = FastAPI(title="PlannerLLM", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: dict = {"settings": Settings.load()}
if not os.environ.get("GROQ_API_KEY") and STATE["settings"].llm.backend == "groq":
    STATE["settings"].llm.backend = "sim"
SESSIONS: dict[tuple, Session] = {}
SESSION_LOCKS: dict[tuple, threading.Lock] = {}
RUNS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def settings() -> Settings:
    return STATE["settings"]


def backend_tag() -> str:
    s = settings()
    return "sim" if s.llm.backend == "sim" else s.llm.model.replace("/", "_")


def session_for(workload: str) -> tuple[Session, threading.Lock]:
    if workload not in available_workloads():
        raise HTTPException(404, f"workload '{workload}' is not prepared")
    key = (backend_tag(), workload)
    with _LOCK:
        if key not in SESSIONS:
            mem_path = DATA_DIR / "memory" / f"{key[0]}_{workload}.sqlite"
            SESSIONS[key] = build_session(workload, settings(), memory_path=str(mem_path))
            SESSION_LOCKS[key] = threading.Lock()
        return SESSIONS[key], SESSION_LOCKS[key]


# ------------------------------------------------------------------ status / config
@app.get("/api/health")
def health():
    s = settings()
    return {"backend": s.llm.backend, "model": s.llm.model if s.llm.backend != "sim" else "simulated-extractive",
            "has_key": bool(os.environ.get("GROQ_API_KEY")), "workloads": available_workloads(),
            "settings": s.to_dict(), "time": time.time()}


class ConfigIn(BaseModel):
    backend: str | None = None
    model: str | None = None
    overrides: dict = {}


@app.post("/api/config")
def set_config(cfg: ConfigIn):
    s = settings().copy()
    if cfg.backend:
        if cfg.backend == "groq" and not os.environ.get("GROQ_API_KEY"):
            raise HTTPException(400, "GROQ_API_KEY is not set; add it to .env and restart")
        s.llm.backend = cfg.backend
    if cfg.model:
        s.llm.model = cfg.model
    try:
        s = s.copy(**cfg.overrides)
    except (KeyError, AttributeError) as exc:
        raise HTTPException(400, str(exc))
    STATE["settings"] = s
    SESSIONS.clear()
    return health()


# ------------------------------------------------------------------ workloads & pipelines
@app.get("/api/workloads")
def workloads():
    out = []
    for name in available_workloads():
        meta = json.loads((DATA_DIR / "processed" / name / "meta.json").read_text(encoding="utf-8"))
        sess, _ = session_for(name)
        wl = sess.workload
        groups: dict[str, int] = {}
        for q in wl.queries:
            groups[q["group"]] = groups.get(q["group"], 0) + 1
        base = wl.base_pipeline()
        out.append({**meta, "n_records": len(wl.records), "n_queries": len(wl.queries), "groups": groups,
                    "base_yaml": base.to_yaml(), "base_plan": base.describe(), "base_ops": base.to_dict()["operators"],
                    "calibration": vars(sess.calibration), "avg_doc_tokens": wl.avg_doc_tokens})
    return out


@app.get("/api/workloads/{workload}/tasks")
def tasks(workload: str, n: int = 8):
    sess, _ = session_for(workload)
    return [t.summary() | {"sample_questions": [q["question"] for q in t.opt_queries[:3]]}
            for t in sess.workload.tasks(n_tasks=n, n_opt=8, n_test=8)]


@app.get("/api/directives")
def directives():
    return DirectiveStore().catalog()


class PlanIn(BaseModel):
    workload: str
    yaml: str
    group: str | None = None


def _task(sess: Session, group: str | None):
    ts = sess.workload.tasks(n_tasks=50, n_opt=8, n_test=8)
    if group:
        ts = [t for t in ts if t.group == group] or ts
    if not ts:
        raise HTTPException(400, "no task available")
    return ts[0]


@app.post("/api/cost")
def cost(p: PlanIn):
    sess, _ = session_for(p.workload)
    try:
        plan = Pipeline.from_yaml(p.yaml)
    except (PipelineError, Exception) as exc:
        raise HTTPException(400, f"invalid pipeline: {exc}")
    t = _task(sess, p.group)
    out = {"symbolic": sess.cost_model.predict(plan, t.stats, "symbolic").to_dict(), "stats": t.stats,
           "plan": plan.describe(), "ops": plan.to_dict()["operators"],
           "residual_fitted": sess.cost_model.fitted}
    if sess.cost_model.fitted:
        out["hybrid"] = sess.cost_model.predict(plan, t.stats, "hybrid").to_dict()
    return out


class ExecIn(PlanIn):
    qid: str | None = None


@app.post("/api/execute")
def execute(p: ExecIn):
    sess, lock = session_for(p.workload)
    try:
        plan = Pipeline.from_yaml(p.yaml)
    except Exception as exc:
        raise HTTPException(400, f"invalid pipeline: {exc}")
    t = _task(sess, p.group)
    q = next((q for q in t.test_queries + t.opt_queries if q["qid"] == p.qid), t.test_queries[0])
    with lock:
        res = sess.executor.run(plan, q, purpose="execute")
    score = sess.oracle.score_one(q, res)
    return {"query": {k: v for k, v in q.items() if k != "contract_text"}, "result": res.to_dict(),
            "score": score, "breakdown": res.breakdown()}


# ------------------------------------------------------------------ optimisation runs
class OptimizeIn(BaseModel):
    workload: str
    group: str | None = None
    system: str = "plannerllm"
    overrides: dict = {}


@app.post("/api/optimize")
def optimize(req: OptimizeIn):
    sess, lock = session_for(req.workload)
    run_id = uuid.uuid4().hex[:10]
    RUNS[run_id] = {"id": run_id, "status": "queued", "events": [], "result": None, "error": None,
                    "request": req.model_dump(), "created": time.time()}

    def work():
        run = RUNS[run_id]
        with lock:
            run["status"] = "running"
            old = sess.settings
            try:
                if req.overrides:
                    sess.settings = old.copy(**req.overrides)
                    sess.gate.cfg = sess.settings.gate
                task = _task(sess, req.group)
                system = make_system(req.system, sess)
                r = system.optimize(task, on_event=lambda ev: run["events"].append(ev))
                res = [sess.executor.run(r.plan, q, purpose="execute") for q in task.test_queries]
                base_res = [sess.executor.run(r.base, q, purpose="execute") for q in task.test_queries]
                sc, sb = sess.oracle.score(task.test_queries, res), sess.oracle.score(task.test_queries, base_res)
                run["result"] = {**r.summary(), "audit": [a.to_dict() for a in r.audit],
                                 "test": {k: v for k, v in sc.items() if k != "per_query"},
                                 "test_base": {k: v for k, v in sb.items() if k != "per_query"},
                                 "examples": [{"question": q["question"], "answer": x.answer[:400],
                                               "baseline": y.answer[:400], "reference": q["answers"][:2],
                                               "sql": x.sql, "score": pq["accuracy"]}
                                              for q, x, y, pq in zip(task.test_queries[:6], res, base_res,
                                                                     sc["per_query"])],
                                 "task": task.summary()}
                run["status"] = "done"
            except Exception as exc:  # surface the failure to the UI
                run["status"] = "error"
                run["error"] = f"{type(exc).__name__}: {exc}"
                run["trace"] = traceback.format_exc()[-2000:]
            finally:
                sess.settings = old
                sess.gate.cfg = old.gate
                run["events"].append({"type": "closed", "status": run["status"]})

    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/runs")
def runs():
    return [{k: v for k, v in r.items() if k not in ("events", "result")} | {"n_events": len(r["events"])}
            for r in sorted(RUNS.values(), key=lambda r: -r["created"])]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")
    return RUNS[run_id]


@app.get("/api/runs/{run_id}/stream")
async def run_stream(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")

    async def gen():
        i = 0
        while True:
            evs = RUNS[run_id]["events"]
            while i < len(evs):
                yield f"data: {json.dumps(evs[i], default=str)}\n\n"
                if evs[i]["type"] == "closed":
                    return
                i += 1
            await asyncio.sleep(0.15)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ memory
@app.get("/api/memory/{workload}")
def memory(workload: str):
    sess, _ = session_for(workload)
    rep = sess.memory.report()
    rep["settings"] = vars(sess.settings.memory)
    return rep


@app.delete("/api/memory/{workload}")
def reset_memory(workload: str):
    key = (backend_tag(), workload)
    with _LOCK:
        SESSIONS.pop(key, None)
    path = DATA_DIR / "memory" / f"{key[0]}_{workload}.sqlite"
    if path.exists():
        path.unlink()
    return {"ok": True}


# ------------------------------------------------------------------ experiments & report
@app.get("/api/experiments")
def experiments():
    found = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            if d.is_dir():
                for f in sorted(d.glob("*.json")):
                    try:
                        head = json.loads(f.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        continue
                    found.append({"backend": d.name, "name": f.stem, "created": head.get("created"),
                                  "profile": head.get("profile"), "elapsed_s": head.get("elapsed_s"),
                                  "model": head.get("model")})
    return {"available": found, "names": EXPERIMENT_NAMES, "profiles": PROFILES}


@app.get("/api/experiments/{backend}/{name}")
def experiment(backend: str, name: str):
    path = RUNS_DIR / backend / f"{name}.json"
    if not path.exists() or ".." in backend or ".." in name:
        raise HTTPException(404, "not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if name in ("baselines", "ablation"):   # trim bulky per-query rows for the browser
        for r in data.get("rows", []):
            r.pop("per_query", None)
    return data


class JobIn(BaseModel):
    names: list[str]
    profile: str = "smoke"


@app.post("/api/experiments/run")
def run_experiments(job: JobIn):
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"id": job_id, "status": "running", "log": [], "request": job.model_dump(), "created": time.time()}

    def work():
        j = JOBS[job_id]
        try:
            for name in job.names:
                j["log"].append(f"== {name}")
                run_experiment(name, settings(), job.profile, progress=lambda m: j["log"].append(m))
            from plannerllm.eval.reporting import build_report
            build_report(settings())
            j["status"] = "done"
        except Exception as exc:
            j["status"] = "error"
            j["log"].append(f"ERROR {type(exc).__name__}: {exc}")

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    j = JOBS[job_id]
    return {**j, "log": j["log"][-200:]}


@app.get("/api/report/{backend}")
def report_files(backend: str):
    d = RUNS_DIR / backend / "report"
    if not d.exists():
        return []
    return sorted(f.name for f in d.iterdir() if f.suffix in (".png", ".csv", ".tex", ".md"))


@app.get("/api/report/{backend}/{fname}")
def report_file(backend: str, fname: str):
    path = (RUNS_DIR / backend / "report" / fname).resolve()
    if not str(path).startswith(str(RUNS_DIR.resolve())) or not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path)


# ------------------------------------------------------------------ static frontend
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = DIST / path
        if path and target.is_file():
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
