"""Run-time configuration for PlannerLLM.

All tunable knobs live in one dataclass tree so that an experiment can be
reproduced from a single YAML file (see ``configs/default.yaml``).  Values
that come from the methodology (gate threshold theta, blend alpha, decay
kappa, sample sizes, budget caps) are grouped the same way the paper groups
them in Section 5 so the code and the report read alike.
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PLANNER_DATA_DIR", ROOT / "data"))
RUNS_DIR = Path(os.environ.get("PLANNER_RUNS_DIR", ROOT / "runs"))


@dataclass
class LLMSettings:
    backend: str = "groq"              # groq | sim
    model: str = "openai/gpt-oss-20b"
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.0
    requests_per_minute: int = 28      # stay just under the free-tier ceiling
    tokens_per_minute: int = 7500
    extra_body: dict = field(default_factory=lambda: {"reasoning_effort": "low", "include_reasoning": False})
    reasoning_tokens: int = 200        # completion budget reserved for a reasoning model's hidden tokens
    max_retries: int = 6
    timeout_s: float = 60.0
    cache_path: str = str(DATA_DIR / "cache" / "llm_cache.sqlite")
    sim_cache_path: str = str(DATA_DIR / "cache" / "sim_cache.sqlite")
    # USD per 1M tokens, used by the token-cost term of the cost model
    price_in_per_m: float = 0.075     # Groq list price, openai/gpt-oss-20b
    price_out_per_m: float = 0.30
    # independent judge: a model from a different family that grades held-out answers and
    # re-checks the validation agent's verdicts (never used by the optimizer itself)
    judge_model: str = "qwen/qwen3.8-27b"


@dataclass
class SearchSettings:
    top_m: int = 5                     # shortlist size after cost ranking (Alg.1 l.3)
    max_candidates: int = 8            # cap on directive applications per round
    rounds: int = 2                    # opportunistic top-down re-expansion rounds
    lambda_gain: float = 1.0           # weight of predicted gain in score (Alg.1 l.2)
    use_generation_agent: bool = True  # ask the LLM which directives to try


@dataclass
class GateSettings:
    alpha: float = 0.6                 # history vs. judge-agreement blend
    theta: float = 1.0                 # validate iff VoI > theta * tokenCost
    w_acc: float = 0.5                 # USD value of +1.0 accuracy over the reuse horizon
                                       # (theta sweeps rescale the trade-off)
    reuse_runs: int = 200              # executions a chosen plan is expected to serve
    prior_gain: float = 0.04           # optimistic prior on accuracy gain of a directive
    min_agreement_samples: int = 8     # before this, the agreement term is uninformative
    calibrate: bool = True             # Platt-calibrate P(flip) once history exists


@dataclass
class MemorySettings:
    path: str = str(DATA_DIR / "memory" / "plan_memory.sqlite")
    kappa: float = 0.02                # per-day exponential decay of verdict confidence
    drift_penalty: float = 3.0         # decay per unit of context drift
    reuse_threshold: float = 0.55      # min confidence to reuse a stored verdict
    soft_match_threshold: float = 0.90 # cosine similarity for a near-signature hit
    retrain_every_days: float = 7.0
    days_per_query: float = 0.25       # logical clock used by experiments


@dataclass
class SamplingSettings:
    frac: float = 0.05
    min_items: int = 4
    max_items: int = 5
    seed: int = 13
    task_seed: int = 0                 # which query groups and questions an experiment draws (one per run seed)


@dataclass
class ValidationSettings:
    top_a: int = 4                     # plans entering pairwise comparison (DocETL uses 6)
    pairwise_margin: float = 0.6       # rating gap below which pairwise is needed
    pairwise_items: int = 2
    max_criteria: int = 4


@dataclass
class BudgetSettings:
    optimization_tokens: int | None = None   # None = unlimited
    cpu_usd_per_hour: float = 0.04           # converts DB/ANN wall-clock to USD


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    gate: GateSettings = field(default_factory=GateSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    budget: BudgetSettings = field(default_factory=BudgetSettings)
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self, **overrides: Any) -> "Settings":
        """Deep copy with dotted overrides, e.g. ``copy(**{"gate.theta": 2.0})``."""
        new = copy.deepcopy(self)
        for dotted, value in overrides.items():
            _set_dotted(new, dotted, value)
        return new

    @classmethod
    def from_dict(cls, raw: dict | None) -> "Settings":
        s = cls()
        for key, value in (raw or {}).items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    _set_dotted(s, f"{key}.{sub_key}", sub_val)
            else:
                _set_dotted(s, key, value)
        return s

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Settings":
        if path is None:
            path = ROOT / "configs" / "default.yaml"
        path = Path(path)
        if not path.exists():
            s = cls()
        else:
            import yaml
            s = cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
        if os.environ.get("PLANNER_LLM_BACKEND"):
            s.llm.backend = os.environ["PLANNER_LLM_BACKEND"]
        if os.environ.get("PLANNER_LLM_MODEL"):
            s.llm.model = os.environ["PLANNER_LLM_MODEL"]
        return s


def _set_dotted(obj: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown setting '{dotted}'")
    current = getattr(obj, leaf)
    if is_dataclass(current) and isinstance(value, dict):
        for k, v in value.items():
            _set_dotted(current, k, v)
        return
    setattr(obj, leaf, value)


def field_names(dc: Any) -> list[str]:
    return [f.name for f in fields(dc)]
