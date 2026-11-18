"""Download and normalise the benchmark corpora used in Section 7.

Every workload is written to ``data/processed/<name>/`` in the same shape:

* ``corpus.jsonl``   one record per retrievable unit (passage, chunk, table)
* ``queries.jsonl``  one record per question, with reference answers
* ``meta.json``      workload description, filterable columns, task prompt

Keeping one layout means the executor, the cost model and the optimizer never
special-case a dataset; only the starting pipeline differs (``workloads.py``).
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sqlite3
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from ..settings import DATA_DIR

RAW = DATA_DIR / "raw"
OUT = DATA_DIR / "processed"

SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"

CUAD_CATEGORIES = [
    "Governing Law", "Agreement Date", "Effective Date", "Expiration Date",
    "Parties", "Renewal Term", "Termination For Convenience", "Cap On Liability",
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------- MS MARCO
def prepare_msmarco(n_queries: int = 300, seed: int = 7) -> Path:
    """Retrieval-heavy family: MS MARCO v1.1 dev queries + their passage pools."""
    import duckdb
    from huggingface_hub import hf_hub_download

    pq = hf_hub_download("microsoft/ms_marco", "v1.1/validation-00000-of-00001.parquet",
                         repo_type="dataset", cache_dir=str(RAW / "hf"))
    con = duckdb.connect()
    rows = con.execute(
        "SELECT query_id, query, query_type, answers, passages "
        f"FROM read_parquet('{Path(pq).as_posix()}')").fetchall()

    by_type: dict[str, list] = defaultdict(list)
    for qid, query, qtype, answers, passages in rows:
        answers = [a for a in (answers or []) if a and a != "No Answer Present."]
        if not answers or not passages or 1 not in (passages.get("is_selected") or []):
            continue
        by_type[str(qtype).lower()].append((qid, query, answers, passages))

    rng = random.Random(seed)
    per_type = max(1, n_queries // max(1, len(by_type)))
    corpus, queries = [], []
    for qtype in sorted(by_type):
        pool = by_type[qtype]
        rng.shuffle(pool)
        for qid, query, answers, passages in pool[:per_type]:
            gold = []
            for j, (sel, text, url) in enumerate(zip(passages["is_selected"],
                                                     passages["passage_text"], passages["url"])):
                pid = f"m{qid}_{j}"
                corpus.append({
                    "id": pid, "text": text, "category": qtype,
                    "domain": urlparse(url).netloc.replace("www.", "")[:60],
                    "n_tokens": _approx_tokens(text),
                })
                if sel:
                    gold.append(pid)
            queries.append({
                "qid": f"m{qid}", "question": query, "answers": answers, "gold_ids": gold,
                "group": qtype, "fields": {"category": qtype},
            })
    out = OUT / "msmarco"
    _write_jsonl(out / "corpus.jsonl", corpus)
    _write_jsonl(out / "queries.jsonl", queries)
    (out / "meta.json").write_text(json.dumps({
        "name": "msmarco", "family": "retrieval-heavy", "metric": "rouge_l",
        "task": "Answer a web search question in one or two sentences using the retrieved passages.",
        "filterable": ["category"], "group_field": "category",
        "source": "MS MARCO v1.1 validation (microsoft/ms_marco on Hugging Face)",
    }, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------- CUAD
def _chunk(text: str, size: int = 1100, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > 0 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def prepare_cuad(n_contracts: int = 60, seed: int = 11, max_chars: int = 60000) -> Path:
    """Mixed family: CUAD legal contracts, chunked, with contract-id metadata."""
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "cuad_data.zip"
    if not zpath.exists():
        urllib.request.urlretrieve(CUAD_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        name = next(n for n in zf.namelist() if n.endswith("CUADv1.json"))
        data = json.loads(zf.read(name))["data"]

    rng = random.Random(seed)
    docs = [d for d in data if len(d["paragraphs"][0]["context"]) <= max_chars]
    rng.shuffle(docs)
    docs = docs[:n_contracts]

    corpus, queries = [], []
    for cid, doc in enumerate(docs):
        para = doc["paragraphs"][0]
        title = doc["title"]
        doc_type = re.sub(r"[_\-]+", " ", title.split("_")[-1] if "_" in title else title).strip()[:60]
        for k, chunk in enumerate(_chunk(para["context"])):
            corpus.append({"id": f"c{cid}_{k}", "text": chunk, "contract_id": cid,
                           "doc_type": doc_type.lower(), "chunk_no": k,
                           "n_tokens": _approx_tokens(chunk)})
        for qa in para["qas"]:
            m = re.search(r'related to "([^"]+)"', qa["question"])
            cat = m.group(1) if m else ""
            if cat not in CUAD_CATEGORIES:
                continue
            details = qa["question"].split("Details:")[-1].strip()
            answers = sorted({a["text"].strip() for a in qa["answers"] if a["text"].strip()})
            queries.append({
                "qid": f"c{cid}_{cat.replace(' ', '_').lower()}",
                "question": f"{cat}: find the text of this contract that states it. ({details})",
                "answers": answers or ["NONE"], "group": cat.lower(),
                "fields": {"contract_id": cid}, "contract_text": para["context"][:max_chars],
            })
    out = OUT / "cuad"
    _write_jsonl(out / "corpus.jsonl", corpus)
    _write_jsonl(out / "queries.jsonl", queries)
    (out / "meta.json").write_text(json.dumps({
        "name": "cuad", "family": "mixed", "metric": "token_f1",
        "task": "Extract the exact clause text from a legal contract that answers the question; "
                "reply NONE if the contract has no such clause.",
        "filterable": ["contract_id"], "group_field": "group",
        "source": "CUAD v1 (The Atticus Project), contracts chunked to ~1.1k chars",
    }, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------- Spider
def _spider_schema(db_path: Path) -> dict[str, list[str]]:
    con = sqlite3.connect(str(db_path))
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    schema = {t: [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')] for t in tables}
    con.close()
    return schema


def prepare_spider(n_queries: int = 240, n_examples: int = 1500, seed: int = 5) -> Path:
    """SQL-heavy family: Spider dev questions over their SQLite databases."""
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "spider.zip"
    if not zpath.exists():
        import gdown
        gdown.download(id=SPIDER_DRIVE_ID, output=str(zpath), quiet=False)
    out = OUT / "spider"
    db_root = out / "database"
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        prefix = next(n for n in names if n.endswith("dev.json")).rsplit("dev.json", 1)[0]
        dev = json.loads(zf.read(prefix + "dev.json"))
        train = json.loads(zf.read(prefix + "train_spider.json"))
        dev_dbs = sorted({q["db_id"] for q in dev})
        for db in dev_dbs:
            member = f"{prefix}database/{db}/{db}.sqlite"
            target = db_root / db / f"{db}.sqlite"
            if not target.exists() and member in names:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    rng = random.Random(seed)
    by_db: dict[str, list] = defaultdict(list)
    for i, q in enumerate(dev):
        by_db[q["db_id"]].append((i, q))
    per_db = max(1, n_queries // len(by_db))
    queries, corpus = [], []
    for db in dev_dbs:
        path = db_root / db / f"{db}.sqlite"
        if not path.exists():
            continue
        schema = _spider_schema(path)
        for t, cols in schema.items():
            text = f"Table {t.replace('_', ' ')} ({t}) with columns: " + ", ".join(cols)
            corpus.append({"id": f"{db}.{t}", "text": text, "db_id": db, "table_name": t,
                           "n_cols": len(cols), "n_tokens": _approx_tokens(text)})
        pool = by_db[db]
        rng.shuffle(pool)
        for i, q in pool[:per_db]:
            queries.append({"qid": f"s{i}", "question": q["question"], "answers": [q["query"]],
                            "gold_sql": q["query"], "db_id": db, "group": db,
                            "fields": {"db_id": db}})
    rng.shuffle(train)
    examples = [{"id": f"t{i}", "text": q["question"], "sql": q["query"], "db_id": q["db_id"]}
                for i, q in enumerate(train[:n_examples])]
    _write_jsonl(out / "corpus.jsonl", corpus)
    _write_jsonl(out / "queries.jsonl", queries)
    _write_jsonl(out / "examples.jsonl", examples)
    (out / "meta.json").write_text(json.dumps({
        "name": "spider", "family": "sql-heavy", "metric": "execution_accuracy",
        "task": "Translate a natural-language question into one SQLite query over the given schema.",
        "filterable": ["db_id"], "group_field": "db_id",
        "source": "Spider 1.0 dev split (Yu et al.), SQLite databases",
    }, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------- BIRD (optional)
def prepare_bird(db_dir: str | None = None, n_queries: int = 150, seed: int = 3) -> Path | None:
    """SQL-heavy (harder) family.  Questions come from Hugging Face; the
    SQLite databases are large, so the user points ``db_dir`` at an unpacked
    BIRD mini-dev ``dev_databases`` folder.  Skipped when not available."""
    if not db_dir or not Path(db_dir).exists():
        return None
    from huggingface_hub import hf_hub_download
    js = hf_hub_download("birdsql/bird_mini_dev",
                         "archive/bird_mini_dev_2025-07-04/data/mini_dev_sqlite-00000-of-00001.json",
                         repo_type="dataset", cache_dir=str(RAW / "hf"))
    data = json.loads(Path(js).read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(data)
    out = OUT / "bird"
    db_root = out / "database"
    queries, corpus, seen = [], [], set()
    for i, q in enumerate(data[:n_queries]):
        db = q["db_id"]
        src = Path(db_dir) / db / f"{db}.sqlite"
        if not src.exists():
            continue
        dst = db_root / db / f"{db}.sqlite"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        if db not in seen:
            seen.add(db)
            for t, cols in _spider_schema(dst).items():
                text = f"Table {t} with columns: " + ", ".join(cols)
                corpus.append({"id": f"{db}.{t}", "text": text, "db_id": db, "table_name": t,
                               "n_cols": len(cols), "n_tokens": _approx_tokens(text)})
        question = q["question"] + (f" (Hint: {q['evidence']})" if q.get("evidence") else "")
        queries.append({"qid": f"b{i}", "question": question, "answers": [q["SQL"]],
                        "gold_sql": q["SQL"], "db_id": db, "group": db, "fields": {"db_id": db}})
    _write_jsonl(out / "corpus.jsonl", corpus)
    _write_jsonl(out / "queries.jsonl", queries)
    (out / "meta.json").write_text(json.dumps({
        "name": "bird", "family": "sql-heavy", "metric": "execution_accuracy",
        "task": "Translate a natural-language question into one SQLite query over the given schema.",
        "filterable": ["db_id"], "group_field": "db_id", "source": "BIRD mini-dev (SQLite)",
    }, indent=2), encoding="utf-8")
    return out


def prepare_all(include: list[str] | None = None, bird_db_dir: str | None = None) -> dict:
    include = include or ["msmarco", "cuad", "spider"]
    done = {}
    for name in include:
        fn = {"msmarco": prepare_msmarco, "cuad": prepare_cuad, "spider": prepare_spider}.get(name)
        if fn:
            done[name] = str(fn())
    if bird_db_dir:
        p = prepare_bird(bird_db_dir)
        if p:
            done["bird"] = str(p)
    return done
