"""Unit tests for the pieces that carry the paper's logic."""
import math

import numpy as np
import pytest

from plannerllm.core.cost_model import HybridCostModel
from plannerllm.core.directives import DirectiveContext, DirectiveStore
from plannerllm.core.gate import VoIGate
from plannerllm.core.memory import PlanMemory
from plannerllm.core.pipeline import Operator, Pipeline, PipelineError
from plannerllm.core.signature import context_features, numeric_vector, signature_of
from plannerllm.eval import metrics as M
from plannerllm.settings import GateSettings, Settings

STATS = {"workload": "cuad", "family": "mixed", "n_rows": 1500, "d": 384, "avg_doc_tokens": 260,
         "sel_est": 0.04, "sel_true": 0.01, "n_predicates": 1, "avg_query_tokens": 30, "n_queries": 6}


def cuad_plan() -> Pipeline:
    return Pipeline("p", "cuad", [
        Operator("retrieve_chunks", "retrieve", {"top_k": 8, "index": "hnsw"}),
        Operator("same_contract", "filter", {"where": "contract_id = {contract_id}"}),
        Operator("answer", "generate", {"max_docs": 4}, prompt="Quote the clause. Reply NONE if absent."),
    ])


def ctx() -> DirectiveContext:
    return DirectiveContext("cuad", "mixed", "extract clauses", "contract_id = {contract_id}", STATS, llm=None)


# ------------------------------------------------------------------ pipeline
def test_yaml_roundtrip_and_validation():
    p = cuad_plan()
    q = Pipeline.from_yaml(p.to_yaml())
    assert q.fingerprint() == p.fingerprint()
    assert q.kinds() == ["retrieve", "filter", "generate"]
    bad = cuad_plan()
    bad.operators.append(Operator("answer", "generate"))
    with pytest.raises(PipelineError):
        bad.validate()


def test_cycle_is_rejected():
    p = Pipeline("c", "x", [Operator("a", "retrieve", inputs=["b"]), Operator("b", "generate", inputs=["a"])])
    with pytest.raises(PipelineError):
        p.topological()


# ------------------------------------------------------------------ directives
def test_filtered_probe_and_its_inverse():
    store = DirectiveStore()
    cands = store.get("FILTERED_PROBE").apply(cuad_plan(), ctx())
    assert len(cands) == 1
    pushed = cands[0].plan
    assert pushed.find("filter") is None and pushed.find("retrieve").params["where"]
    back = store.get("POST_FILTER").apply(pushed, ctx())[0].plan
    assert back.find("filter") is not None and "where" not in back.find("retrieve").params
    assert back.find("retrieve").params["top_k"] == 24          # over-fetch x3


def test_rerank_inserted_before_llm_and_lineage_tracked():
    c = DirectiveStore().get("ADD_RERANK").apply(cuad_plan(), ctx())[0]
    assert c.plan.kinds() == ["retrieve", "filter", "rerank", "generate"]
    assert c.plan.lineage == ["ADD_RERANK"]


def test_expand_dedups_and_respects_cap():
    store = DirectiveStore()
    cands = store.expand(cuad_plan(), ctx(), cap=3)
    assert len(cands) == 3
    assert len({c.plan.fingerprint() for c in cands}) == 3


# ------------------------------------------------------------------ signatures & memory
def test_signature_is_stable_and_stat_sensitive():
    base, cand = cuad_plan(), DirectiveStore().get("FILTERED_PROBE").apply(cuad_plan(), ctx())[0].plan
    f1 = context_features(base, cand, "FILTERED_PROBE", STATS)
    assert signature_of(f1) == signature_of(dict(f1))
    f2 = context_features(base, cand, "FILTERED_PROBE", {**STATS, "sel_est": 0.5})
    assert signature_of(f1) != signature_of(f2)


def test_memory_requires_confirmation_and_decays():
    day = [0.0]
    mem = PlanMemory(":memory:", kappa=0.05, clock=lambda: day[0])
    base, cand = cuad_plan(), DirectiveStore().get("FILTERED_PROBE").apply(cuad_plan(), ctx())[0].plan
    feats = context_features(base, cand, "FILTERED_PROBE", STATS)
    sig, vec = signature_of(feats), numeric_vector(STATS, cand)
    mem.write(sig, feats, vec, "FILTERED_PROBE", True, 0.2, 0.0, "cuad")
    hit = mem.lookup(sig, vec, "FILTERED_PROBE", feats["parent_shape"], "mixed", feats)
    assert hit is not None and not mem.usable(hit)               # one observation is not enough
    mem.write(sig, feats, vec, "FILTERED_PROBE", True, 0.2, 0.0, "cuad")
    hit = mem.lookup(sig, vec, "FILTERED_PROBE", feats["parent_shape"], "mixed", feats)
    assert mem.usable(hit) and hit.prefer and hit.match == "exact"
    day[0] = 30.0                                                 # a month later the verdict expires
    hit = mem.lookup(sig, vec, "FILTERED_PROBE", feats["parent_shape"], "mixed", feats)
    assert not mem.usable(hit)
    assert math.isclose(hit.confidence, math.exp(-0.05 * 30) * 2 / 3, rel_tol=1e-6)


def test_contradiction_resets_confidence():
    mem = PlanMemory(":memory:", clock=lambda: 0.0)
    feats = {"directive": "X", "family": "mixed", "parent_shape": "a"}
    vec = np.zeros(6)
    for _ in range(3):
        mem.write("s", feats, vec, "X", True, 0.1, 0.0, "w")
    mem.write("s", feats, vec, "X", False, -0.1, 0.0, "w")
    hit = mem.lookup("s", vec, "X", "a", "mixed")
    assert hit.prefer is False and hit.n_obs == 1


def test_flip_rate_is_laplace_smoothed():
    mem = PlanMemory(":memory:")
    assert mem.flip_rate("D") == (0.5, 0)
    for f in (True, True, False):
        mem.update_flip("D", f)
    assert mem.flip_rate("D") == ((2 + 1) / (3 + 2), 3)


# ------------------------------------------------------------------ cost model
def test_symbolic_cost_tracks_context_size():
    cm = HybridCostModel(0.05, 0.08, 0.04)
    plan = cuad_plan()
    pushed = DirectiveStore().get("FILTERED_PROBE").apply(plan, ctx())[0].plan
    a, b = cm.symbolic(plan, STATS), cm.symbolic(pushed, STATS)
    # post-filter keeps 8 * 0.04 rows under independence; the filtered probe hands 4 docs to the LLM
    assert b.tokens > a.tokens and b.usd > a.usd
    assert a.llm_calls == 1


def test_residual_learns_a_constant_bias():
    cm = HybridCostModel(0.05, 0.08, 0.04)
    rng = np.random.default_rng(0)
    X, obs, sym = [], [], []
    for _ in range(60):
        st = {**STATS, "n_rows": int(rng.integers(500, 5000)), "sel_est": float(rng.uniform(0.01, 0.5))}
        s = cm.symbolic(cuad_plan(), st)
        X.append(cm.features(cuad_plan(), st, s))
        sym.append(s.usd)
        obs.append(s.usd * 3.0)                      # the backbone is uniformly 3x too low
    assert cm.fit(X, obs, [1.0] * 60, sym, [1.0] * 60)
    pred = cm.predict(cuad_plan(), STATS, "hybrid").usd
    assert pred / cm.symbolic(cuad_plan(), STATS).usd == pytest.approx(3.0, rel=0.05)


# ------------------------------------------------------------------ gate
def test_gate_math_and_threshold():
    mem = PlanMemory(":memory:")
    cm = HybridCostModel(0.05, 0.08, 0.04)
    gs = GateSettings(alpha=0.6, theta=1.0, w_acc=0.05, reuse_runs=200, prior_gain=0.04, calibrate=False)
    gate = VoIGate(gs, mem, cm)
    plan = cuad_plan()
    cand = DirectiveStore().get("FILTERED_PROBE").apply(plan, ctx())[0].plan
    cc, cp = cm.predict(cand, STATS), cm.predict(plan, STATS)
    d = gate.decide("FILTERED_PROBE", "mixed", cand, plan, STATS, cc, cp, None, 5, 3)
    assert d.p_flip == pytest.approx(0.6 * 0.5 + 0.4 * 0.5)
    expected = d.p_flip * abs(0.05 * 0.04 - abs(200 * (cc.usd - cp.usd)))
    assert d.voi == pytest.approx(expected)
    assert d.validate == (d.voi > d.threshold)
    hi = gate.decide("FILTERED_PROBE", "mixed", cand, plan, STATS, cc, cp, None, 5, 3, theta=1e6)
    assert not hi.validate


def test_judge_term_needs_agreement_history():
    mem = PlanMemory(":memory:")
    gate = VoIGate(GateSettings(alpha=0.0, min_agreement_samples=3, calibrate=False), mem, HybridCostModel(0.05, 0.08, 0.04))
    assert gate.flip_probability("D", 0.9)[0] == pytest.approx(0.5)      # no history: uninformative
    for _ in range(10):
        mem.update_agreement("D", True)
    p = gate.flip_probability("D", 0.9)[0]
    assert p > 0.75                                                       # trusted judge says "flip"


# ------------------------------------------------------------------ metrics
def test_metrics():
    assert M.prf("the cat sat", "cat sat")[2] == pytest.approx(1.0)
    assert M.prf("NONE", "NONE") == (1.0, 1.0, 1.0)
    assert M.prf("something", "NONE")[2] == 0.0
    assert M.rouge_l("a b c d", "a x c d") > M.rouge_l("d c b a", "a x c d")   # LCS, not bag of words
    assert M.hallucinated("purple elephants dance tonight", "the contract is governed by Delaware law")
    assert not M.hallucinated("governed by Delaware law", "the contract is governed by Delaware law")


def test_settings_overrides():
    s = Settings().copy(**{"gate.theta": 2.5, "budget.optimization_tokens": 100})
    assert s.gate.theta == 2.5 and s.budget.optimization_tokens == 100
    with pytest.raises(KeyError):
        Settings().copy(**{"gate.nope": 1})
