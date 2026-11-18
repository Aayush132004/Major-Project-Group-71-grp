"""Dense retrieval substrate: embeddings, ANN indexes and the cross-encoder.

Three ANN families are built over the same normalised embeddings so a
SWAP_INDEX rewrite changes only the physical access path:

* ``hnsw`` - proximity graph (Malkov & Yashunin), the family that dominates
  the time/accuracy trade-off on public benchmarks
* ``ivf``  - inverted-file quantisation with ``nprobe`` lists
* ``flat`` - exact brute force (the recall reference)

A *filtered vector probe* passes an ``IDSelectorBatch`` into the search, so
the predicate is applied inside graph/list traversal rather than after it
(the ACORN-style hybrid pattern).  The post-filter alternative over-fetches
and drops rows, which loses recall at low selectivity - exactly the case the
FILTERED_PROBE directive targets.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from pathlib import Path

import numpy as np

_MODELS: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


def _load(kind: str, name: str):
    key = f"{kind}:{name}"
    with _MODEL_LOCK:
        if key not in _MODELS:
            import os
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import CrossEncoder, SentenceTransformer
            _MODELS[key] = (SentenceTransformer(name, device="cpu") if kind == "bi"
                            else CrossEncoder(name, device="cpu"))
        return _MODELS[key]


class Embedder:
    """Sentence-transformer encoder with a hashing fallback for offline tests."""

    def __init__(self, model_name: str, use_hashing: bool = False) -> None:
        self.model_name = model_name
        self.use_hashing = use_hashing
        self.dim = 256 if use_hashing else 384
        self._query_cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.use_hashing:
            return np.stack([_hash_embed(t, self.dim) for t in texts]).astype(np.float32)
        model = _load("bi", self.model_name)
        vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        if text not in self._query_cache:
            self._query_cache[text] = self.encode([text])[0]
        return self._query_cache[text]

    def encode_corpus(self, texts: list[str], cache_file: Path) -> np.ndarray:
        tag = "hash" if self.use_hashing else self.model_name.replace("/", "_")
        path = cache_file.with_name(f"{cache_file.stem}.{tag}.npy")
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                return arr
        arr = self.encode(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        return arr


def _hash_embed(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    toks = text.lower().split()
    for a, b in zip(toks, toks[1:] + [""]):
        for tok in (a, a + "_" + b):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v[h % dim] += 1.0 if (h >> 9) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class Reranker:
    """Cross-encoder relevance model; doubles as the local judge (<=3B, CPU)."""

    def __init__(self, model_name: str, use_lexical: bool = False) -> None:
        self.model_name = model_name
        self.use_lexical = use_lexical
        self._cache: dict[tuple[str, str], float] = {}

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if (query, t) not in self._cache]
        if missing:
            if self.use_lexical:
                q = set(query.lower().split())
                vals = [len(q & set(t.lower().split())) / (1 + len(q)) * 10 - 5 for t in missing]
            else:
                model = _load("cross", self.model_name)
                vals = model.predict([(query, t[:1500]) for t in missing],
                                     show_progress_bar=False, batch_size=32)
            for t, v in zip(missing, vals):
                self._cache[(query, t)] = float(v)
        return np.array([self._cache[(query, t)] for t in texts], dtype=np.float32)


class VectorIndex:
    """One embedding matrix, lazily-built FAISS indexes per family."""

    def __init__(self, vectors: np.ndarray) -> None:
        import faiss
        self.faiss = faiss
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n, self.dim = self.vectors.shape
        self._indexes: dict[str, object] = {}
        self._quantizers: list = []
        self._lock = threading.Lock()
        self.build_ms: dict[str, float] = {}

    def _get(self, family: str):
        with self._lock:
            if family in self._indexes:
                return self._indexes[family]
            f = self.faiss
            t0 = time.perf_counter()
            if family == "hnsw":
                idx = f.IndexHNSWFlat(self.dim, 32, f.METRIC_INNER_PRODUCT)
                idx.hnsw.efConstruction = 80
                idx.add(self.vectors)
            elif family == "ivf":
                nlist = max(4, int(math.sqrt(self.n)))
                quant = f.IndexFlatIP(self.dim)
                idx = f.IndexIVFFlat(quant, self.dim, nlist, f.METRIC_INNER_PRODUCT)
                idx.train(self.vectors)
                idx.add(self.vectors)
                self._quantizers.append(quant)  # keep the quantizer alive
            else:
                idx = f.IndexFlatIP(self.dim)
                idx.add(self.vectors)
            self._indexes[family] = idx
            self.build_ms[family] = (time.perf_counter() - t0) * 1000
            return idx

    def search(self, qvec: np.ndarray, k: int, family: str = "hnsw", *,
               allowed: list[int] | None = None, ef_search: int = 64,
               nprobe: int = 8) -> tuple[list[int], list[float], float]:
        """Return (positions, scores, ms).  ``allowed`` restricts the search space."""
        f = self.faiss
        idx = self._get(family)
        if allowed is not None and len(allowed) == 0:
            return [], [], 0.0
        k = max(1, min(k, self.n if allowed is None else len(allowed)))
        sel = f.IDSelectorBatch(np.asarray(allowed, dtype=np.int64)) if allowed is not None else None
        if family == "hnsw":
            params = f.SearchParametersHNSW()
            params.efSearch = max(ef_search, k)
        elif family == "ivf":
            params = f.SearchParametersIVF()
            params.nprobe = nprobe if allowed is None else max(nprobe, 4 * nprobe)
        else:
            params = f.SearchParameters()
        if sel is not None:
            params.sel = sel
        t0 = time.perf_counter()
        D, I = idx.search(qvec.reshape(1, -1).astype(np.float32), k, params=params)
        ms = (time.perf_counter() - t0) * 1000
        pos = [int(i) for i in I[0] if i >= 0]
        scores = [float(d) for d, i in zip(D[0], I[0]) if i >= 0]
        return pos, scores, ms

    def microbenchmark(self, family: str, k: int = 10, trials: int = 20,
                       ef_search: int = 64, nprobe: int = 8) -> float:
        """Median per-query latency (ms); calibrates the ANN term of the cost model."""
        rng = np.random.default_rng(0)
        picks = rng.choice(self.n, size=min(trials, self.n), replace=False)
        times = [self.search(self.vectors[p], k, family, ef_search=ef_search, nprobe=nprobe)[2]
                 for p in picks]
        return float(np.median(times))
