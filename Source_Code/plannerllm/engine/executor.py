"""Execution engine: runs one pipeline for one query and records telemetry.

Telemetry is split the way Section 10.6 decomposes wall-clock time:
DB operator time (DuckDB / SQLite), retrieval time (ANN probe), and LLM time,
plus the tokens and dollars each operator consumed.  The same numbers feed
the observed-cost log that the neural residual is trained on.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.pipeline import Operator, Pipeline
from . import prompts
from .sqlrun import clean_sql, schema_text
from .store import render_where

MAX_DOC_CHARS = 600


@dataclass
class OpTelemetry:
    op: str
    kind: str
    rows_in: int
    rows_out: int
    ms: float
    db_ms: float = 0.0
    ann_ms: float = 0.0
    llm_ms: float = 0.0
    tokens: int = 0
    usd: float = 0.0
    llm_wall_ms: float = 0.0


@dataclass
class ExecResult:
    qid: str
    answer: str
    context: list[str] = field(default_factory=list)
    sql: str | None = None
    rows: list | None = None
    error: str | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    telemetry: list[OpTelemetry] = field(default_factory=list)

    # aggregate views ---------------------------------------------------
    @property
    def tokens(self) -> int:
        return sum(t.tokens for t in self.telemetry)

    @property
    def llm_usd(self) -> float:
        return sum(t.usd for t in self.telemetry)

    @property
    def latency_ms(self) -> float:
        return sum(t.ms for t in self.telemetry)

    def breakdown(self) -> dict:
        return {"db_ms": sum(t.db_ms for t in self.telemetry),
                "ann_ms": sum(t.ann_ms for t in self.telemetry),
                "llm_ms": sum(t.llm_ms for t in self.telemetry),
                "other_ms": sum(max(0.0, t.ms - t.db_ms - t.ann_ms - t.llm_ms) for t in self.telemetry)}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens"], d["latency_ms"] = self.tokens, self.latency_ms
        d["rows"] = [list(map(str, r)) for r in (self.rows or [])[:10]]
        return d


class _State:
    """Data flowing between operators for one query."""

    def __init__(self) -> None:
        self.positions: list[int] | None = None   # None == whole corpus
        self.extracts: dict[int, str] = {}
        self.answer: str = ""
        self.sql: str | None = None
        self.rows: list | None = None
        self.error: str | None = None
        self.context: list[str] = []


class Executor:
    def __init__(self, workload, llm, cpu_usd_per_hour: float = 0.04) -> None:
        self.wl = workload
        self.llm = llm
        self.cpu_usd_per_ms = cpu_usd_per_hour / 3.6e6

    # ------------------------------------------------------------------
    def run(self, plan: Pipeline, query: dict, purpose: str = "execute") -> ExecResult:
        st = _State()
        res = ExecResult(qid=query["qid"], answer="")
        for op in plan.topological():
            t0 = time.perf_counter()
            rows_in = len(st.positions) if st.positions is not None else self.wl.store.n_rows
            tel = OpTelemetry(op=op.name, kind=op.kind, rows_in=rows_in, rows_out=0, ms=0.0)
            try:
                getattr(self, f"_op_{op.kind}")(op, query, st, tel, purpose)
            except KeyError as exc:  # missing query field for a predicate
                st.error = f"{op.name}: {exc}"
            # wall time of the operator with the LLM's wall time (which on a cache hit is ~0 and on a
            # fresh call includes rate-limit waits) replaced by the API's service time
            tel.ms = (time.perf_counter() - t0) * 1000 - tel.llm_wall_ms + tel.llm_ms
            tel.rows_out = len(st.positions) if st.positions is not None else tel.rows_in
            res.telemetry.append(tel)
        res.answer, res.sql, res.rows, res.error = st.answer, st.sql, st.rows, st.error
        res.context = st.context
        if st.positions is not None:
            res.retrieved_ids = [self.wl.records[p]["id"] for p in st.positions]
        return res

    def observed_usd(self, res: ExecResult) -> float:
        cpu_ms = sum(t.db_ms + t.ann_ms for t in res.telemetry)
        return res.llm_usd + cpu_ms * self.cpu_usd_per_ms

    # ------------------------------------------------------------ helpers
    def _llm(self, tel: OpTelemetry, prompt: str, *, purpose: str, task: str,
             payload: dict, max_tokens: int, system: str = prompts.SYSTEM_EXEC) -> str:
        t0 = time.perf_counter()
        r = self.llm.complete(prompt, purpose=purpose, task=task, payload=payload,
                              system=system, max_tokens=max_tokens)
        tel.llm_wall_ms += (time.perf_counter() - t0) * 1000
        tel.llm_ms += r.latency_s * 1000
        tel.tokens += r.total_tokens
        tel.usd += self.llm.ledger.usd_for(r.prompt_tokens, r.completion_tokens)
        return r.text.strip()

    def _texts(self, st: _State, limit: int) -> list[str]:
        pos = st.positions if st.positions is not None else []
        out = []
        for p in pos[:limit]:
            out.append(st.extracts.get(p) or self.wl.records[p]["text"][:MAX_DOC_CHARS])
        return out

    # ------------------------------------------------------------ operators
    def _op_filter(self, op: Operator, q: dict, st: _State, tel: OpTelemetry, purpose: str) -> None:
        where = render_where(op.params["where"], q.get("fields", {}))
        keep, ms = self.wl.store.positions(where, subset=st.positions)
        tel.db_ms += ms
        if st.positions is None:
            st.positions = keep
        else:
            allowed = set(keep)
            st.positions = [p for p in st.positions if p in allowed]

    def _op_retrieve(self, op: Operator, q: dict, st: _State, tel: OpTelemetry, purpose: str) -> None:
        p = op.params
        allowed = st.positions
        if p.get("where"):
            where = render_where(p["where"], q.get("fields", {}))
            allowed, ms = self.wl.store.positions(where, subset=allowed)
            tel.db_ms += ms
        qvec = self.wl.embedder.encode_query(q["question"])
        pos, _, ms = self.wl.index.search(qvec, int(p.get("top_k", 10)), p.get("index", "hnsw"),
                                          allowed=allowed, ef_search=int(p.get("ef_search", 64)),
                                          nprobe=int(p.get("nprobe", 8)))
        tel.ann_ms += ms
        st.positions = pos

    def _op_rerank(self, op: Operator, q: dict, st: _State, tel: OpTelemetry, purpose: str) -> None:
        pos = st.positions or []
        if not pos:
            return
        t0 = time.perf_counter()
        scores = self.wl.reranker.score(q["question"], [self.wl.records[i]["text"] for i in pos])
        tel.ann_ms += (time.perf_counter() - t0) * 1000   # local model time, CPU
        order = sorted(range(len(pos)), key=lambda i: -scores[i])
        st.positions = [pos[i] for i in order[: int(op.params.get("top_k", 5))]]

    def _op_map(self, op: Operator, q: dict, st: _State, tel: OpTelemetry, purpose: str) -> None:
        kept = []
        cap = op.params.get("max_records")
        for i in (st.positions or [])[: int(cap) if cap else None]:
            text = self.wl.records[i]["text"][:1200]
            prompt = prompts.MAP.format(instruction=op.prompt, question=q["question"], text=text)
            out = self._llm(tel, prompt, purpose=purpose, task="extract", max_tokens=160,
                            payload={"question": q["question"], "instruction": op.prompt, "text": text})
            if out and not out.strip().upper().startswith("NONE"):
                st.extracts[i] = out[:MAX_DOC_CHARS]
                kept.append(i)
        st.positions = kept

    def _op_generate(self, op: Operator, q: dict, st: _State, tel: OpTelemetry, purpose: str) -> None:
        docs = self._texts(st, int(op.params.get("max_docs", 6)))
        st.context = docs
        ctx = "\n\n".join(f"[{k + 1}] {d}" for k, d in enumerate(docs)) or "(no context retrieved)"
        prompt = prompts.GENERATE.format(instruction=op.prompt, question=q["question"], context=ctx)
        st.answer = self._llm(tel, prompt, purpose=purpose, task="answer", max_tokens=160,
                              payload={"question": q["question"], "context": docs})

    def _schema_for(self, q: dict, st: _State) -> dict[str, list[str]]:
        full = self.wl.sql.schema(q["db_id"])
        if st.positions is None:
            return full
        tables = [self.wl.records[p].get("table_name") for p in st.positions]
        picked = {t: full[t] for t in tables if t in full}
        return picked or full

    def _op_sql_generate(self, op: Operator, q: dict, st: _State, tel: OpTelemetry, purpose: str) -> None:
        schema = self._schema_for(q, st)
        st.context = [schema_text(schema)]
        shots = ""
        k = int(op.params.get("fewshot_k", 0))
        if k and self.wl.examples is not None:
            ex = self.wl.nearest_examples(q["question"], k)
            shots = "\nExamples:\n" + "\n".join(f"Q: {e['text']}\nSQL: {e['sql']}" for e in ex) + "\n"
        prompt = prompts.SQL_GENERATE.format(instruction=op.prompt, schema=schema_text(schema),
                                             examples=shots, question=q["question"])
        out = self._llm(tel, prompt, purpose=purpose, task="sql", max_tokens=220,
                        payload={"question": q["question"], "schema": schema})
        st.sql = clean_sql(out)

    def _op_sql_repair(self, op: Operator, q: dict, st: _State, tel: OpTelemetry, purpose: str) -> None:
        rows, err, ms = self.wl.sql.run(q["db_id"], st.sql or "")
        tel.db_ms += ms
        if err is None and rows:
            return
        schema = self._schema_for(q, st)
        prompt = prompts.SQL_REPAIR.format(schema=schema_text(schema), question=q["question"],
                                           sql=st.sql or "", problem=err or "query returned no rows")
        out = self._llm(tel, prompt, purpose=purpose, task="sql_repair", max_tokens=220,
                        payload={"question": q["question"], "schema": schema})
        fixed = clean_sql(out)
        if fixed:
            st.sql = fixed

    def _op_sql_execute(self, op: Operator, q: dict, st: _State, tel: OpTelemetry, purpose: str) -> None:
        rows, err, ms = self.wl.sql.run(q["db_id"], st.sql or "")
        tel.db_ms += ms
        st.rows, st.error = rows, err
        st.answer = st.sql or ""


def run_many(executor: Executor, plan: Pipeline, queries: list[dict], purpose: str = "execute") -> list[ExecResult]:
    return [executor.run(plan, q, purpose) for q in queries]


def as_jsonable(x: Any) -> Any:
    if hasattr(x, "to_dict"):
        return x.to_dict()
    return x
