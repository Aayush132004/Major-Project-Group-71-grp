"""Workload runtime: corpus + indexes + starting pipeline + task streams.

A *task* is one group of related queries (an MS MARCO query type, a CUAD
clause category, a Spider database).  The optimizer runs once per task on an
optimisation split and the chosen plan is then scored on a held-out split,
which mirrors how DocETL optimises on a sample and executes on the rest.
Groups recur across a stream with similar statistics, which is what lets the
plan memory reuse verdicts across queries (Section 5.2).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.pipeline import Operator, Pipeline
from ..engine.sqlrun import SQLRunner
from ..engine.store import MetadataStore, render_where
from ..engine.vectors import Embedder, Reranker, VectorIndex
from ..settings import DATA_DIR, Settings

PROCESSED = DATA_DIR / "processed"

GEN_PROMPTS = {
    "msmarco": "Answer the question using only the numbered passages. Reply in one or two sentences.",
    "cuad": "Using only the contract excerpts, quote the clause text that answers the question. "
            "If the excerpts do not contain it, reply NONE.",
    "bird": "Write one SQLite query that answers the question.",
    "spider": "Write one SQLite query that answers the question.",
}
MAP_PROMPT = "Copy the sentence(s) from the text that help answer the question, word for word. If none do, reply NONE."
FILTER_TEMPLATES = {"msmarco": "category = {category}", "cuad": "contract_id = {contract_id}",
                    "spider": "db_id = {db_id}", "bird": "db_id = {db_id}"}


def available_workloads() -> list[str]:
    return sorted(p.name for p in PROCESSED.glob("*") if (p / "meta.json").exists())


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class Task:
    task_id: str
    workload: str
    group: str
    opt_queries: list[dict]
    test_queries: list[dict]
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"task_id": self.task_id, "workload": self.workload, "group": self.group,
                "n_opt": len(self.opt_queries), "n_test": len(self.test_queries), "stats": self.stats}


class Workload:
    def __init__(self, name: str, settings: Settings | None = None, offline: bool = False) -> None:
        settings = settings or Settings()
        self.name = name
        self.dir = PROCESSED / name
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"workload '{name}' not prepared; run scripts/prepare_data.py")
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.family = self.meta["family"]
        self.metric = self.meta["metric"]
        self.task_text = self.meta["task"]
        self.records = _read_jsonl(self.dir / "corpus.jsonl")
        self.id_to_pos = {r["id"]: i for i, r in enumerate(self.records)}
        self.queries = _read_jsonl(self.dir / "queries.jsonl")
        self.store = MetadataStore(self.dir / "corpus.jsonl")
        self.embedder = Embedder(settings.embedder, use_hashing=offline)
        vecs = self.embedder.encode_corpus([r["text"] for r in self.records],
                                           DATA_DIR / "cache" / f"emb_{name}")
        self.index = VectorIndex(vecs)
        self.reranker = Reranker(settings.cross_encoder, use_lexical=offline)
        self.sql = SQLRunner(self.dir / "database") if (self.dir / "database").exists() else None
        self.examples = None
        if (self.dir / "examples.jsonl").exists():
            self.examples = _read_jsonl(self.dir / "examples.jsonl")
            ev = self.embedder.encode_corpus([e["text"] for e in self.examples],
                                             DATA_DIR / "cache" / f"emb_{name}_examples")
            self._example_index = VectorIndex(ev)
        # pre-encode every question so no system pays the encoder in its latency
        qtexts = [q["question"] for q in self.queries]
        qv = self.embedder.encode_corpus(qtexts, DATA_DIR / "cache" / f"emb_{name}_questions")
        self.embedder._query_cache.update(zip(qtexts, qv))
        self.avg_doc_tokens = self.store.avg_tokens()
        self._ann_ms: dict[str, float] = {}

    # ------------------------------------------------------------ pipelines
    @property
    def filter_template(self) -> str:
        return FILTER_TEMPLATES[self.name]

    def base_pipeline(self) -> Pipeline:
        """The hand-written starting plan a user would submit (Fig. 'User Pipeline YAML')."""
        n = self.name
        if n == "msmarco":
            ops = [Operator("retrieve_passages", "retrieve", {"top_k": 10, "index": "hnsw", "ef_search": 64}),
                   Operator("answer", "generate", {"max_docs": 5}, prompt=GEN_PROMPTS[n])]
        elif n == "cuad":
            ops = [Operator("retrieve_chunks", "retrieve", {"top_k": 8, "index": "hnsw", "ef_search": 64}),
                   Operator("same_contract", "filter", {"where": FILTER_TEMPLATES[n]}),
                   Operator("answer", "generate", {"max_docs": 4}, prompt=GEN_PROMPTS[n])]
        else:  # spider / bird
            ops = [Operator("write_sql", "sql_generate", {"fewshot_k": 0}, prompt=GEN_PROMPTS[n]),
                   Operator("run_sql", "sql_execute")]
        p = Pipeline(name=f"{n}_baseline", workload=n, operators=ops)
        p.validate()
        return p

    def nearest_examples(self, question: str, k: int) -> list[dict]:
        if not self.examples:
            return []
        qv = self.embedder.encode_query(question)
        pos, _, _ = self._example_index.search(qv, k, "flat")
        return [self.examples[p] for p in pos]

    # ------------------------------------------------------------ statistics
    def ann_latency_ms(self, family: str) -> float:
        """Measured median probe latency on this corpus (cost-model calibration)."""
        if family not in self._ann_ms:
            self._ann_ms[family] = self.index.microbenchmark(family)
        return self._ann_ms[family]

    def group_stats(self, queries: list[dict]) -> dict:
        q0 = queries[0]
        where = render_where(self.filter_template, q0.get("fields", {}))
        stats = {
            "workload": self.name, "family": self.family,
            "n_rows": self.store.n_rows, "d": self.index.dim,
            "avg_doc_tokens": round(self.avg_doc_tokens, 1),
            "sel_est": round(self.store.estimate(where), 5),
            "sel_true": round(self.store.true_selectivity(where), 5),
            "n_predicates": where.upper().count(" AND ") + 1,
            "avg_query_tokens": round(float(np.mean([len(q["question"]) / 4 for q in queries])), 1),
            "n_queries": len(queries),
        }
        if self.sql is not None and "db_id" in q0:
            schema = self.sql.schema(q0["db_id"])
            stats["n_tables"] = len(schema)
            stats["schema_tokens"] = sum(len(t) + sum(len(c) + 2 for c in cols) for t, cols in schema.items()) // 4
        return stats

    def tasks(self, n_tasks: int = 4, n_opt: int = 8, n_test: int = 8, seed: int = 0,
              groups: list[str] | None = None) -> list[Task]:
        by_group: dict[str, list[dict]] = {}
        for q in self.queries:
            by_group.setdefault(q["group"], []).append(q)
        rng = random.Random(seed)
        names = sorted(by_group)
        if groups:
            names = [g for g in names if g in groups]
        rng.shuffle(names)
        out = []
        for g in names[:n_tasks]:
            pool = list(by_group[g])
            rng.shuffle(pool)
            if self.name == "cuad":
                # one clause category per task; queries span many contracts
                pool = sorted(pool, key=lambda q: q["answers"] == ["NONE"])[: n_opt + n_test]
                rng.shuffle(pool)
            opt, test = pool[:n_opt], pool[n_opt:n_opt + n_test]
            if not opt or not test:
                continue
            out.append(Task(task_id=f"{self.name}:{g}", workload=self.name, group=g,
                            opt_queries=opt, test_queries=test, stats=self.group_stats(opt)))
        return out


_CACHE: dict[tuple, Workload] = {}


def load_workload(name: str, settings: Settings | None = None, offline: bool = False) -> Workload:
    key = (name, offline)
    if key not in _CACHE:
        _CACHE[key] = Workload(name, settings, offline)
    return _CACHE[key]


def ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b))
