"""Accuracy metrics (Section 10.1 primary criteria).

* token-level precision / recall / F1 (SQuAD normalisation, max over refs)
* Rouge-L F-measure from the longest common *subsequence*
* hallucination rate: share of answers whose content words are mostly absent
  from the source the pipeline was allowed to read
* execution accuracy for SQL (result-set equality)
* retrieval recall@k for MS MARCO gold passages
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)
_STOP = set("a an the of to in and or is are was were be been for on with by as at from that this "
            "it its into than then there their they he she his her which who what when where how".split())


_CITE = re.compile(r"【[^】]*】|\[\d+(?:[,\s]*\d+)*\]")   # citation markers such as [2]


def normalize(text: str) -> str:
    text = _CITE.sub(" ", str(text)).lower()
    # SQuAD normalisation drops punctuation; Unicode punctuation (curly quotes, CJK brackets) is dropped too
    text = "".join(ch for ch in text if ch not in _PUNCT and not unicodedata.category(ch).startswith("P"))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def is_none_answer(text: str) -> bool:
    t = normalize(text)
    return t in {"", "none", "not found", "no answer", "n a"} or t.startswith("none ")


def prf(pred: str, ref: str) -> tuple[float, float, float]:
    if is_none_answer(ref) or is_none_answer(pred):
        hit = float(is_none_answer(ref) and is_none_answer(pred))
        return hit, hit, hit
    p, r = tokens(pred), tokens(ref)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0, 0.0, 0.0
    precision, recall = same / len(p), same / len(r)
    return precision, recall, 2 * precision * recall / (precision + recall)


def best_prf(pred: str, refs: list[str]) -> tuple[float, float, float]:
    """For multi-span references (CUAD) the spans are also joined into one ref."""
    cands = list(refs) + ([" ".join(refs)] if len(refs) > 1 else [])
    return max((prf(pred, r) for r in cands), key=lambda t: t[2], default=(0.0, 0.0, 0.0))


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str, beta: float = 1.2) -> float:
    p, r = tokens(pred), tokens(ref)
    lcs = _lcs(p, r)
    if lcs == 0:
        return float(is_none_answer(pred) and is_none_answer(ref))
    prec, rec = lcs / len(p), lcs / len(r)
    return (1 + beta ** 2) * prec * rec / (rec + beta ** 2 * prec)


def best_rouge(pred: str, refs: list[str]) -> float:
    return max((rouge_l(pred, r) for r in refs), default=0.0)


def hallucinated(pred: str, source: str, threshold: float = 0.6) -> bool:
    """True when fewer than ``threshold`` of the answer's content words occur in the source."""
    if is_none_answer(pred):
        return False
    words = [w for w in tokens(pred) if w not in _STOP and len(w) > 2]
    if not words:
        return False
    src = set(tokens(source))
    grounded = sum(1 for w in words if w in src)
    return grounded / len(words) < threshold


def recall_at_k(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return float(any(g in retrieved for g in gold))
