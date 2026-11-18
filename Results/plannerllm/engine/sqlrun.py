"""SQLite execution for the SQL-heavy workloads (Spider, BIRD)."""
from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path


class SQLRunner:
    def __init__(self, db_root: Path, timeout_s: float = 4.0, max_rows: int = 2000) -> None:
        self.db_root = Path(db_root)
        self.timeout_s = timeout_s
        self.max_rows = max_rows

    def db_path(self, db_id: str) -> Path:
        return self.db_root / db_id / f"{db_id}.sqlite"

    def run(self, db_id: str, sql: str) -> tuple[list[tuple] | None, str | None, float]:
        """Execute read-only; returns (rows, error, ms).  Aborts slow queries."""
        sql = clean_sql(sql)
        if not sql:
            return None, "empty SQL", 0.0
        if re.match(r"^\s*(insert|update|delete|drop|alter|create|attach|pragma)\b", sql, re.I):
            return None, "only SELECT statements are allowed", 0.0
        uri = f"file:{self.db_path(db_id).as_posix()}?mode=ro"
        t0 = time.perf_counter()
        con = sqlite3.connect(uri, uri=True, check_same_thread=False)
        con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        deadline = t0 + self.timeout_s
        con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
        try:
            rows = con.execute(sql).fetchmany(self.max_rows)
            return rows, None, (time.perf_counter() - t0) * 1000
        except Exception as exc:  # sqlite3.Error, OperationalError on interrupt, ...
            return None, str(exc)[:200], (time.perf_counter() - t0) * 1000
        finally:
            con.close()

    @lru_cache(maxsize=64)
    def schema(self, db_id: str) -> dict[str, list[str]]:
        con = sqlite3.connect(f"file:{self.db_path(db_id).as_posix()}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            return {t: [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')] for t in tables}
        finally:
            con.close()


def schema_text(schema: dict[str, list[str]], tables: list[str] | None = None) -> str:
    keep = tables or list(schema)
    return "\n".join(f"{t}({', '.join(schema[t])})" for t in keep if t in schema)


def clean_sql(text: str) -> str:
    """Pull a single statement out of an LLM reply (fences, prose, trailing ;)."""
    if not text:
        return ""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    m = re.search(r"\b(SELECT|WITH)\b.*", text, re.S | re.I)
    if m:
        text = m.group(0)
    text = text.strip().split(";")[0]
    return " ".join(text.split())


def results_match(pred: list[tuple] | None, gold: list[tuple] | None, ordered: bool) -> bool:
    """Execution accuracy: same multiset of rows (same order if ORDER BY)."""
    if pred is None or gold is None:
        return False
    norm = lambda rows: [tuple(_norm_cell(c) for c in r) for r in rows]
    p, g = norm(pred), norm(gold)
    if ordered:
        return p == g
    return Counter(p) == Counter(g)


def _norm_cell(c):
    if isinstance(c, float):
        return round(c, 4)
    if isinstance(c, str):
        return c.strip().lower()
    return c
