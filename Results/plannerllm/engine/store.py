"""Corpus metadata store backed by DuckDB.

Structured predicates (``filter`` operators and the ``where`` clause of a
filtered vector probe) are evaluated here.  The store also exposes the two
selectivity estimators the cost model needs:

* ``estimate`` - DuckDB's own optimizer estimate read from ``EXPLAIN``.  This
  is the *symbolic* signal: it is what a classical planner would believe.
* ``true_selectivity`` - an exact count, used only to log telemetry so the
  neural residual can learn where the symbolic estimate goes wrong.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import duckdb

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def render_where(template: str, fields: dict) -> str:
    """Fill ``{field}`` placeholders from the query, quoting strings safely."""
    def sub(m: re.Match) -> str:
        val = fields.get(m.group(1))
        if val is None:
            raise KeyError(f"query has no field '{m.group(1)}' for predicate '{template}'")
        if isinstance(val, (int, float)):
            return str(val)
        return "'" + str(val).replace("'", "''") + "'"
    return _PLACEHOLDER.sub(sub, template)


class MetadataStore:
    def __init__(self, corpus_path: Path) -> None:
        self._con = duckdb.connect()
        self._lock = threading.Lock()
        path = Path(corpus_path).as_posix()
        self._con.execute(
            f"CREATE TABLE docs AS SELECT row_number() OVER () - 1 AS pos, * "
            f"FROM read_json_auto('{path}', format='newline_delimited')")
        self.n_rows = self._con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        self.columns = [r[0] for r in self._con.execute("DESCRIBE docs").fetchall()]
        self._est_cache: dict[str, float] = {}

    def positions(self, where: str, subset: list[int] | None = None) -> tuple[list[int], float]:
        """Row positions satisfying ``where`` (optionally within ``subset``)."""
        t0 = time.perf_counter()
        sql = f"SELECT pos FROM docs WHERE ({where})"
        with self._lock:
            if subset is not None:
                if not subset:
                    return [], 0.0
                self._con.execute("CREATE OR REPLACE TEMP TABLE _subset(pos BIGINT)")
                self._con.executemany("INSERT INTO _subset VALUES (?)", [(int(p),) for p in subset])
                sql += " AND pos IN (SELECT pos FROM _subset)"
            rows = self._con.execute(sql + " ORDER BY pos").fetchall()
        return [r[0] for r in rows], (time.perf_counter() - t0) * 1000

    def estimate(self, where: str) -> float:
        """Optimizer-estimated selectivity (symbolic, from EXPLAIN)."""
        if where in self._est_cache:
            return self._est_cache[where]
        with self._lock:
            plan = self._con.execute(f"EXPLAIN (FORMAT json) SELECT pos FROM docs WHERE ({where})").fetchall()
        card = _find_cardinality(json.loads(plan[0][1]))
        sel = min(1.0, max(1.0 / max(self.n_rows, 1), (card or self.n_rows) / max(self.n_rows, 1)))
        self._est_cache[where] = sel
        return sel

    def true_selectivity(self, where: str) -> float:
        with self._lock:
            n = self._con.execute(f"SELECT COUNT(*) FROM docs WHERE ({where})").fetchone()[0]
        return n / max(self.n_rows, 1)

    def scalar(self, sql: str):
        with self._lock:
            return self._con.execute(sql).fetchone()[0]

    def avg_tokens(self) -> float:
        if "n_tokens" not in self.columns:
            return 100.0
        return float(self.scalar("SELECT AVG(n_tokens) FROM docs") or 100.0)


def _find_cardinality(node) -> float | None:
    if isinstance(node, list):
        for n in node:
            v = _find_cardinality(n)
            if v is not None:
                return v
        return None
    if isinstance(node, dict):
        info = node.get("extra_info") or {}
        if isinstance(info, dict) and "Estimated Cardinality" in info:
            try:
                return float(info["Estimated Cardinality"])
            except (TypeError, ValueError):
                pass
        for child in node.get("children", []):
            v = _find_cardinality(child)
            if v is not None:
                return v
    return None
