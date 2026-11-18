"""Persistent response cache for deterministic (temperature 0) calls.

The free Groq tier allows only a few thousand tokens per minute, so every
experiment would otherwise take hours.  Keying on the exact request makes a
re-run free and lets an interrupted experiment resume where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


class ResponseCache:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT PRIMARY KEY, model TEXT, text TEXT,"
                " prompt_tokens INT, completion_tokens INT, created REAL, latency REAL)")
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(responses)")}
            if "lat_kind" not in cols:
                # 'service' = server-side compute time reported by the API; NULL = legacy rows whose
                # latency was wall time including rate-limit waits (replaced by a fitted estimate)
                self._db.execute("ALTER TABLE responses ADD COLUMN lat_kind TEXT")
            self._db.commit()
        self._lat_model: tuple | None = None
        self._lat_rows = -1

    def _latency_model(self) -> tuple:
        """Least-squares fit service_s = a + b*prompt_tokens + c*completion_tokens on clean rows."""
        rows = self._db.execute("SELECT prompt_tokens, completion_tokens, latency FROM responses "
                                "WHERE lat_kind='service' AND latency > 0").fetchall()
        if len(rows) == self._lat_rows and self._lat_model is not None:
            return self._lat_model
        self._lat_rows = len(rows)
        if len(rows) < 20:
            self._lat_model = (0.05, 2e-5, 1.5e-3)      # provisional until enough clean rows exist
            return self._lat_model
        import numpy as np
        X = np.array([[1.0, r[0], r[1]] for r in rows])
        y = np.array([r[2] for r in rows])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self._lat_model = tuple(float(max(c, 0.0)) for c in coef)
        return self._lat_model

    def estimate_latency(self, pt: int, ct: int) -> float:
        with self._lock:
            a, b, c = self._latency_model()
        return a + b * pt + c * ct

    @staticmethod
    def make_key(model: str, system: str | None, prompt: str, max_tokens: int,
                 json_mode: bool, temperature: float) -> str:
        blob = json.dumps([model, system or "", prompt, max_tokens, json_mode, temperature])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[str, int, int, float] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT text, prompt_tokens, completion_tokens, COALESCE(latency, 0), lat_kind "
                "FROM responses WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        text, pt, ct, lat, kind = row
        if kind != "service":
            lat = self.estimate_latency(pt, ct)
        return text, pt, ct, lat

    def put(self, key: str, model: str, text: str, pt: int, ct: int, latency: float = 0.0,
            kind: str | None = "service") -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, model, text, prompt_tokens, completion_tokens, created, "
                "latency, lat_kind) VALUES (?,?,?,?,?,?,?,?)",
                (key, model, text, pt, ct, time.time(), latency, kind))
            self._db.commit()

    def size(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
