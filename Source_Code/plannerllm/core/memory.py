"""Plan memory (Sections 5.2 and 5.5).

Persistent SQLite store that turns one-shot agentic optimisation into a
learning optimiser.  It keeps

(i)   validated rewrite verdicts keyed by applicability signature,
(ii)  per-directive flip statistics  h(dir),
(iii) local-judge vs LLM-validator agreement, per directive,
(iv)  observed-cost telemetry for the residual cost model,
(v)   a gate log used to calibrate P(flip) and to score the gate offline.

Verdict confidence decays with age and with context drift:

    delta(t) = exp(-kappa * age_days) * exp(-drift_penalty * drift) * n/(n + 1)

With the default reuse threshold a verdict must be confirmed by two
independent validations before it is reused at all (n/(n+1) = 0.5 -> 0.67).

so a pattern confirmed several times recently is trusted, a stale one or one
validated under different statistics is re-validated instead of reused.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .signature import drift as ctx_drift

SCHEMA = """
CREATE TABLE IF NOT EXISTS rewrites (
  sig TEXT PRIMARY KEY, directive TEXT, family TEXT, parent_shape TEXT,
  features TEXT, vec TEXT, verdict TEXT, acc_delta REAL, cost_delta REAL,
  n_obs INT, n_agree INT, created REAL, updated REAL, workload TEXT);
CREATE TABLE IF NOT EXISTS flip_stats (directive TEXT PRIMARY KEY, flips INT, trials INT);
CREATE TABLE IF NOT EXISTS agreement (directive TEXT PRIMARY KEY, agree INT, total INT);
CREATE TABLE IF NOT EXISTS gate_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, directive TEXT, p_raw REAL, p_used REAL,
  voi REAL, token_cost REAL, validated INT, flipped INT, ts REAL, workload TEXT);
CREATE TABLE IF NOT EXISTS telemetry (
  id INTEGER PRIMARY KEY AUTOINCREMENT, workload TEXT, plan_fp TEXT, features TEXT,
  sym_usd REAL, sym_ms REAL, obs_usd REAL, obs_ms REAL, ts REAL);
CREATE TABLE IF NOT EXISTS retrain_log (ts REAL, n_telemetry INT, n_gate INT, note TEXT);
"""


STAT_KEYS = {"rows_b", "sel_b", "doclen_b", "tables_b"}


def same_structure(a: dict, b: dict) -> bool:
    """Equal on every signature feature except the bucketed data statistics."""
    keys = (set(a) | set(b)) - STAT_KEYS
    return all(str(a.get(k)) == str(b.get(k)) for k in keys)


@dataclass
class MemoryHit:
    sig: str
    directive: str
    prefer: bool
    confidence: float
    acc_delta: float
    match: str          # exact | soft
    age_days: float
    drift: float
    n_obs: int

    def to_dict(self) -> dict:
        return dict(vars(self))


class PlanMemory:
    def __init__(self, path: str | Path = ":memory:", kappa: float = 0.02, drift_penalty: float = 3.0,
                 reuse_threshold: float = 0.55, soft_match_threshold: float = 0.90,
                 retrain_every_days: float = 7.0, clock=None) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()
        self.kappa, self.drift_penalty = kappa, drift_penalty
        self.reuse_threshold, self.soft_threshold = reuse_threshold, soft_match_threshold
        self.retrain_every = retrain_every_days
        self.clock = clock or (lambda: time.time() / 86400.0)
        self._last_retrain = self.clock()
        self._calib: tuple[float, float, int] | None = None

    @classmethod
    def from_settings(cls, ms, path: str | None = None, clock=None) -> "PlanMemory":
        return cls(path or ms.path, ms.kappa, ms.drift_penalty, ms.reuse_threshold,
                   ms.soft_match_threshold, ms.retrain_every_days, clock)

    def _q(self, sql: str, args: tuple = ()):
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _x(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self._db.execute(sql, args)
            self._db.commit()

    # ------------------------------------------------------------ verdicts
    def confidence(self, age_days: float, drift: float, n_obs: int) -> float:
        return (math.exp(-self.kappa * max(age_days, 0.0))
                * math.exp(-self.drift_penalty * drift)
                * (n_obs / (n_obs + 1.0)))

    def lookup(self, sig: str, vec: np.ndarray, directive: str, parent_shape: str,
               family: str, features: dict | None = None) -> MemoryHit | None:
        """Exact signature first; otherwise the closest context that has the same
        plan structure and differs only in data statistics (a *soft* hit)."""
        now = self.clock()
        row = self._q("SELECT sig, directive, verdict, acc_delta, n_obs, updated, vec "
                      "FROM rewrites WHERE sig=?", (sig,))
        best: MemoryHit | None = None
        if row:
            s, d, v, acc, n, upd, vjs = row[0]
            dr = ctx_drift(np.array(json.loads(vjs)), vec)
            best = MemoryHit(s, d, v == "prefer", self.confidence(now - upd, dr, n), acc,
                             "exact", now - upd, dr, n)
        else:
            rows = self._q("SELECT sig, directive, verdict, acc_delta, n_obs, updated, vec, features "
                           "FROM rewrites WHERE directive=? AND parent_shape=? AND family=?",
                           (directive, parent_shape, family))
            for s, d, v, acc, n, upd, vjs, fjs in rows:
                if features is not None and not same_structure(json.loads(fjs), features):
                    continue
                dr = ctx_drift(np.array(json.loads(vjs)), vec)
                sim = math.exp(-dr)
                if sim < self.soft_threshold:
                    continue
                hit = MemoryHit(s, d, v == "prefer", self.confidence(now - upd, dr, n) * sim, acc,
                                "soft", now - upd, dr, n)
                if best is None or hit.confidence > best.confidence:
                    best = hit
        return best

    def usable(self, hit: MemoryHit | None) -> bool:
        return hit is not None and hit.confidence >= self.reuse_threshold

    def write(self, sig: str, features: dict, vec: np.ndarray, directive: str, prefer: bool,
              acc_delta: float, cost_delta: float, workload: str) -> None:
        now = self.clock()
        verdict = "prefer" if prefer else "reject"
        row = self._q("SELECT verdict, acc_delta, n_obs, n_agree FROM rewrites WHERE sig=?", (sig,))
        if row:
            old_v, old_acc, n, agree = row[0]
            if old_v == verdict:
                n, agree = n + 1, agree + 1
                acc = 0.6 * old_acc + 0.4 * acc_delta
            else:   # contradicted: the newest evidence wins, confidence restarts
                n, agree, acc = 1, 0, acc_delta
            self._x("UPDATE rewrites SET verdict=?, acc_delta=?, cost_delta=?, n_obs=?, n_agree=?, "
                    "updated=?, vec=? WHERE sig=?",
                    (verdict, acc, cost_delta, n, agree, now, json.dumps(vec.tolist()), sig))
        else:
            self._x("INSERT INTO rewrites VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sig, directive, features.get("family"), features.get("parent_shape"),
                     json.dumps(features, default=str), json.dumps(vec.tolist()), verdict,
                     acc_delta, cost_delta, 1, 1, now, now, workload))

    def touch(self, sig: str) -> None:
        """A reused verdict counts as a confirmation of recency (not of correctness)."""
        self._x("UPDATE rewrites SET updated=? WHERE sig=?", (self.clock(), sig))

    # ------------------------------------------------------------ gate statistics
    def update_flip(self, directive: str, flipped: bool) -> None:
        row = self._q("SELECT flips, trials FROM flip_stats WHERE directive=?", (directive,))
        f, t = row[0] if row else (0, 0)
        self._x("INSERT OR REPLACE INTO flip_stats VALUES (?,?,?)", (directive, f + int(flipped), t + 1))

    def flip_rate(self, directive: str) -> tuple[float, int]:
        """Laplace-smoothed historical flip rate h(dir) and its sample size."""
        row = self._q("SELECT flips, trials FROM flip_stats WHERE directive=?", (directive,))
        f, t = row[0] if row else (0, 0)
        return (f + 1) / (t + 2), t

    def update_agreement(self, directive: str, agreed: bool) -> None:
        row = self._q("SELECT agree, total FROM agreement WHERE directive=?", (directive,))
        a, t = row[0] if row else (0, 0)
        self._x("INSERT OR REPLACE INTO agreement VALUES (?,?,?)", (directive, a + int(agreed), t + 1))

    def agreement_rate(self, directive: str) -> tuple[float, int]:
        """Judge/validator concordance for a directive, pooled when data is thin."""
        row = self._q("SELECT agree, total FROM agreement WHERE directive=?", (directive,))
        a, t = row[0] if row else (0, 0)
        pooled = self._q("SELECT COALESCE(SUM(agree),0), COALESCE(SUM(total),0) FROM agreement")[0]
        w = t / (t + 5.0)
        own = (a + 1) / (t + 2)
        pool = (pooled[0] + 1) / (pooled[1] + 2)
        return w * own + (1 - w) * pool, int(pooled[1])

    def predicted_gain(self, directive: str, family: str, prior: float, n0: float = 2.0) -> tuple[float, int]:
        """Confidence-weighted mean accuracy delta of a directive, shrunk to ``prior``."""
        now = self.clock()
        rows = self._q("SELECT acc_delta, n_obs, updated FROM rewrites WHERE directive=? AND family=?",
                       (directive, family))
        num, den = n0 * prior, n0
        for acc, n, upd in rows:
            w = self.confidence(now - upd, 0.0, n)
            num += w * acc
            den += w
        return num / den, len(rows)

    def log_gate(self, directive: str, p_raw: float, p_used: float, voi: float, token_cost: float,
                 validated: bool, flipped: bool | None, workload: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO gate_log (directive, p_raw, p_used, voi, token_cost, validated, flipped, ts, workload)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (directive, p_raw, p_used, voi, token_cost, int(validated),
                 None if flipped is None else int(flipped), self.clock(), workload))
            self._db.commit()
            return cur.lastrowid

    def set_gate_outcome(self, log_id: int, flipped: bool) -> None:
        self._x("UPDATE gate_log SET flipped=? WHERE id=?", (int(flipped), log_id))

    def calibrate_gate(self, min_samples: int = 12) -> tuple[float, float, int] | None:
        """Platt scaling of logit(p_raw) -> realised flip, fitted on validated rows."""
        rows = self._q("SELECT p_raw, flipped FROM gate_log WHERE flipped IS NOT NULL")
        if len(rows) < min_samples or len({r[1] for r in rows}) < 2:
            return self._calib
        from sklearn.linear_model import LogisticRegression
        p = np.clip(np.array([r[0] for r in rows]), 1e-3, 1 - 1e-3)
        X = np.log(p / (1 - p)).reshape(-1, 1)
        y = np.array([r[1] for r in rows])
        lr = LogisticRegression(C=1.0).fit(X, y)
        self._calib = (float(lr.coef_[0][0]), float(lr.intercept_[0]), len(rows))
        return self._calib

    def calibrated(self, p_raw: float) -> float:
        if self._calib is None:
            return p_raw
        a, b, n = self._calib
        p = min(max(p_raw, 1e-3), 1 - 1e-3)
        cal = 1 / (1 + math.exp(-(a * math.log(p / (1 - p)) + b)))
        w = n / (n + 30.0)          # trust the calibration map as evidence accumulates
        return w * cal + (1 - w) * p_raw

    # ------------------------------------------------------------ telemetry
    def add_telemetry(self, workload: str, plan_fp: str, features: list[float], sym_usd: float,
                      sym_ms: float, obs_usd: float, obs_ms: float) -> None:
        self._x("INSERT INTO telemetry (workload, plan_fp, features, sym_usd, sym_ms, obs_usd, obs_ms, ts)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (workload, plan_fp, json.dumps(features), sym_usd, sym_ms, obs_usd, obs_ms, self.clock()))

    def telemetry(self, workloads: list[str] | None = None) -> dict:
        rows = self._q("SELECT workload, features, sym_usd, sym_ms, obs_usd, obs_ms FROM telemetry")
        if workloads:
            rows = [r for r in rows if r[0] in workloads]
        return {"workload": [r[0] for r in rows], "X": [json.loads(r[1]) for r in rows],
                "sym_usd": [r[2] for r in rows], "sym_ms": [r[3] for r in rows],
                "obs_usd": [r[4] for r in rows], "obs_ms": [r[5] for r in rows]}

    def maybe_retrain(self, cost_model, force: bool = False) -> dict | None:
        """The 'weekly CPU retraining' hook: residual cost model + gate calibration."""
        now = self.clock()
        if not force and now - self._last_retrain < self.retrain_every:
            return None
        t = self.telemetry()
        fitted = cost_model.fit(t["X"], t["obs_usd"], t["obs_ms"], t["sym_usd"], t["sym_ms"])
        calib = self.calibrate_gate()
        self._last_retrain = now
        n_gate = self._q("SELECT COUNT(*) FROM gate_log WHERE flipped IS NOT NULL")[0][0]
        self._x("INSERT INTO retrain_log VALUES (?,?,?,?)",
                (now, len(t["X"]), n_gate, f"residual={'yes' if fitted else 'no'} calib={calib}"))
        return {"residual_fitted": fitted, "n_telemetry": len(t["X"]), "gate_calibration": calib}

    # ------------------------------------------------------------ reporting
    def report(self) -> dict:
        rw = self._q("SELECT sig, directive, family, verdict, acc_delta, n_obs, updated, workload, features "
                     "FROM rewrites ORDER BY updated DESC")
        now = self.clock()
        return {
            "rewrites": [{"sig": r[0], "directive": r[1], "family": r[2], "verdict": r[3],
                          "acc_delta": r[4], "n_obs": r[5], "age_days": now - r[6], "workload": r[7],
                          "confidence": self.confidence(now - r[6], 0.0, r[5]),
                          "features": json.loads(r[8])} for r in rw],
            "flip_stats": [{"directive": d, "flips": f, "trials": t, "h": (f + 1) / (t + 2)}
                           for d, f, t in self._q("SELECT * FROM flip_stats ORDER BY directive")],
            "agreement": [{"directive": d, "agree": a, "total": t, "rate": (a + 1) / (t + 2)}
                          for d, a, t in self._q("SELECT * FROM agreement ORDER BY directive")],
            "gate_log": [dict(zip(("directive", "p_raw", "p_used", "voi", "token_cost", "validated",
                                   "flipped", "ts", "workload"), r))
                         for r in self._q("SELECT directive, p_raw, p_used, voi, token_cost, validated, "
                                          "flipped, ts, workload FROM gate_log ORDER BY id")],
            "telemetry_rows": self._q("SELECT COUNT(*) FROM telemetry")[0][0],
            "retrains": [dict(zip(("ts", "n_telemetry", "n_gate", "note"), r))
                         for r in self._q("SELECT * FROM retrain_log ORDER BY ts")],
            "calibration": self._calib,
        }
