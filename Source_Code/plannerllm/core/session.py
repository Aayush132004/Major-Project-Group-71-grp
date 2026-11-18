"""Wires every component for one workload (the six services of Section 6)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..data.workloads import Workload, load_workload
from ..engine.executor import Executor
from ..eval.oracle import Oracle
from ..llm.base import TokenLedger
from ..llm.client import build_client
from ..settings import Settings
from .cost_model import Calibration, HybridCostModel
from .directives import DirectiveContext, DirectiveStore
from .gate import VoIGate
from .generation import GenerationAgent
from .judge import LocalJudge
from .memory import PlanMemory
from .validation import ValidationAgent


def calibrate(wl: Workload, settings: Settings) -> Calibration:
    """Microbenchmarks that pin the cost model's constants to this machine."""
    cal = Calibration()
    cal.ann_ms = {fam: wl.ann_latency_ms(fam) for fam in ("hnsw", "ivf", "flat")}
    # cross-encoder cost per pair, measured on fresh pairs
    texts = [r["text"] for r in wl.records[:12]]
    wl.reranker.score("warm-up", texts[:1])          # load weights outside the timer
    t0 = time.perf_counter()
    wl.reranker.score(f"calibration probe {time.time()}", texts)
    cal.ce_ms_per_pair = max(0.05, (time.perf_counter() - t0) * 1000 / len(texts))
    t0 = time.perf_counter()
    wl.store.positions("1 = 1")
    cal.db_ms_per_krow = max(0.01, (time.perf_counter() - t0) * 1000 / max(wl.store.n_rows / 1000, 1))
    return cal


@dataclass
class Session:
    settings: Settings
    workload: Workload
    llm: object
    executor: Executor
    cost_model: HybridCostModel
    memory: PlanMemory
    directives: DirectiveStore
    judge: LocalJudge
    validator: ValidationAgent
    gate: VoIGate
    agent: GenerationAgent
    oracle: Oracle
    calibration: Calibration

    @property
    def ledger(self) -> TokenLedger:
        return self.llm.ledger

    def directive_context(self, stats: dict) -> DirectiveContext:
        wl = self.workload
        return DirectiveContext(wl.name, wl.family, wl.task_text, wl.filter_template, stats, self.llm)

    def fresh_validator(self) -> ValidationAgent:
        """New per-run output/rating cache (tokens are charged again, as they would be)."""
        base = self.workload.base_pipeline()
        final_kind = base.topological()[-1].kind
        self.validator = ValidationAgent(self.llm, self.executor, self.settings, self.workload.task_text,
                                         "sql_generate" if final_kind == "sql_execute" else final_kind)
        return self.validator


_CALIB: dict[str, Calibration] = {}


def build_session(workload: str, settings: Settings | None = None, *, llm=None,
                  memory: PlanMemory | None = None, memory_path: str | None = None,
                  offline: bool = False, cost_mode: str = "hybrid", clock=None) -> Session:
    settings = settings or Settings.load()
    wl = load_workload(workload, settings, offline=offline)
    if llm is None:
        ledger = TokenLedger(settings.llm.price_in_per_m, settings.llm.price_out_per_m,
                             settings.budget.optimization_tokens)
        llm = build_client(settings, ledger)
    if workload not in _CALIB:
        _CALIB[workload] = calibrate(wl, settings)
    cal = _CALIB[workload]
    cost = HybridCostModel(settings.llm.price_in_per_m, settings.llm.price_out_per_m,
                           settings.budget.cpu_usd_per_hour, cal, mode=cost_mode)
    if memory is None:
        memory = PlanMemory.from_settings(settings.memory, path=memory_path or ":memory:", clock=clock)
    ex = Executor(wl, llm, settings.budget.cpu_usd_per_hour)
    store = DirectiveStore()
    s = Session(settings=settings, workload=wl, llm=llm, executor=ex, cost_model=cost, memory=memory,
                directives=store, judge=LocalJudge(wl, ex), validator=None,  # type: ignore[arg-type]
                gate=VoIGate(settings.gate, memory, cost),
                agent=GenerationAgent(llm, store, settings.search.use_generation_agent),
                oracle=Oracle(wl), calibration=cal)
    s.fresh_validator()
    return s


def mean_observed(session: Session, results) -> tuple[float, float]:
    usd = float(np.mean([session.executor.observed_usd(r) for r in results])) if results else 0.0
    ms = float(np.mean([r.latency_ms for r in results])) if results else 0.0
    return usd, ms
