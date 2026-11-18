"""End-to-end runs on the real (prepared) datasets with the offline simulator."""
import pytest

from plannerllm.core.optimizer import SYSTEMS, make_system
from plannerllm.core.session import build_session
from plannerllm.data.workloads import available_workloads
from plannerllm.settings import Settings

pytestmark = pytest.mark.skipif(not available_workloads(), reason="run scripts/prepare_data.py first")


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    s = Settings()
    s.llm.backend = "sim"
    s.llm.sim_cache_path = str(tmp_path_factory.mktemp("c") / "sim.sqlite")
    return s


@pytest.mark.parametrize("workload", ["msmarco", "cuad", "spider"])
def test_every_system_runs(settings, workload):
    if workload not in available_workloads():
        pytest.skip("workload not prepared")
    s = build_session(workload, settings)
    task = s.workload.tasks(n_tasks=1, n_opt=5, n_test=3)[0]
    for name in SYSTEMS:
        r = make_system(name, s).optimize(task)
        r.plan.validate()
        res = [s.executor.run(r.plan, q) for q in task.test_queries]
        score = s.oracle.score(task.test_queries, res)
        assert 0.0 <= score["accuracy"] <= 1.0
        if name in ("no_opt", "random"):
            assert r.opt_tokens == 0


def test_memory_reuse_cuts_tokens_on_repeat(settings):
    s = build_session("cuad", settings)
    tasks = s.workload.tasks(n_tasks=3, n_opt=5, n_test=3)
    sysm = make_system("plannerllm", s)
    # a verdict needs two agreeing validations before reuse, so compare the first and third pass
    passes = [[sysm.optimize(t) for t in tasks] for _ in range(3)]
    first, third = passes[0], passes[2]
    assert sum(r.n_memory_hits for r in third) > sum(r.n_memory_hits for r in first)
    assert sum(r.opt_tokens for r in third) < sum(r.opt_tokens for r in first)


def test_budget_caps_optimisation_spend(settings):
    s = build_session("msmarco", settings.copy(**{"budget.optimization_tokens": 3000}))
    task = s.workload.tasks(n_tasks=1, n_opt=5, n_test=3)[0]
    r = make_system("docetl", s).optimize(task)
    assert r.opt_tokens <= 3000 + 1500       # one in-flight call may finish past the cap


def test_filtered_probe_restricts_to_predicate(settings):
    s = build_session("cuad", settings)
    q = s.workload.tasks(n_tasks=1, n_opt=2, n_test=1)[0].opt_queries[0]
    plan = s.workload.base_pipeline()
    from plannerllm.core.directives import DirectiveContext
    ctx = DirectiveContext("cuad", "mixed", "", s.workload.filter_template, {}, None)
    probe = s.directives.get("FILTERED_PROBE").apply(plan, ctx)[0].plan
    res = s.executor.run(probe, q)
    cid = q["fields"]["contract_id"]
    assert res.retrieved_ids and all(s.workload.records[s.workload.id_to_pos[i]]["contract_id"] == cid
                                     for i in res.retrieved_ids)


def test_api_smoke(settings, monkeypatch):
    from fastapi.testclient import TestClient
    import server.app as app_mod
    app_mod.STATE["settings"] = settings
    client = TestClient(app_mod.app)
    assert client.get("/api/health").json()["backend"] == "sim"
    wl = client.get("/api/workloads").json()
    assert {w["name"] for w in wl} >= {"cuad"}
    cuad = next(w for w in wl if w["name"] == "cuad")
    c = client.post("/api/cost", json={"workload": "cuad", "yaml": cuad["base_yaml"]}).json()
    assert c["symbolic"]["usd"] > 0 and len(c["ops"]) == 3
    bad = client.post("/api/cost", json={"workload": "cuad", "yaml": "operators: [{name: x, kind: nope}]"})
    assert bad.status_code == 400
