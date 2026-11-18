import { useEffect, useRef, useState } from "react";
import { api, fmt, stream, SYS_LABEL, WL_LABEL } from "../api";
import { PlanDiagram } from "../components/Plan";
import { Ribbon } from "../components/Ribbon";

const SYSTEMS = ["plannerllm", "docetl", "no_gate", "no_memory", "cost_only", "random", "no_opt"];

export default function Optimize({ health }) {
  const [wls, setWls] = useState([]);
  const [wl, setWl] = useState("cuad");
  const [tasks, setTasks] = useState([]);
  const [group, setGroup] = useState("");
  const [system, setSystem] = useState("plannerllm");
  const [theta, setTheta] = useState(health.settings.gate.theta);
  const [alpha, setAlpha] = useState(health.settings.gate.alpha);
  const [events, setEvents] = useState([]);
  const [run, setRun] = useState(null);
  const [err, setErr] = useState(null);
  const stop = useRef(null);

  useEffect(() => { api.workloads().then(setWls); return () => stop.current?.(); }, []);
  const viewGroup = useRef(null);
  useEffect(() => { api.tasks(wl).then((t) => { setTasks(t); setGroup(viewGroup.current || t[0]?.group || ""); }); }, [wl]);
  // "#optimize?run=cuad" starts a run on load (used for demos and report screenshots)
  const auto = useRef(new URLSearchParams(location.hash.split("?")[1] || "").get("run"));
  // "#optimize?view=<run id>" reopens a finished run from this server session
  useEffect(() => {
    const id = new URLSearchParams(location.hash.split("?")[1] || "").get("view");
    if (!id) return;
    api.run(id).then((d) => {
      viewGroup.current = d.request.group;
      setWl(d.request.workload);
      setGroup(d.request.group || "");
      setEvents(d.events || []);
      setRun(d);
    }).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => {
    if (auto.current && auto.current !== wl) { setWl(auto.current); return; }
    if (auto.current && group && !run) { auto.current = null; start(); }
  }, [wl, group]);

  const start = async () => {
    stop.current?.();
    setEvents([]); setRun({ status: "running" }); setErr(null);
    try {
      const { run_id } = await api.optimize({ workload: wl, group, system,
        overrides: { "gate.theta": Number(theta), "gate.alpha": Number(alpha) } });
      stop.current = stream(run_id, (ev) => {
        setEvents((prev) => [...prev, ev]);
        if (ev.type === "closed") api.run(run_id).then(setRun);
      });
    } catch (e) { setErr(e.message); setRun(null); }
  };

  const startEv = events.find((e) => e.type === "start");
  const lastInc = [...events].reverse().find((e) => e.type === "incumbent");
  const res = run?.result;
  const running = run?.status === "running" || run?.status === "queued";
  const spent = events.filter((e) => e.type === "validation").reduce((a, e) => a + (e.tokens || 0), 0);

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Optimize a pipeline</h2>
          <p>Each shortlisted rewrite is either answered from plan memory, skipped by the value-of-information gate, or
             validated by the LLM agent on a sample. The plan memory persists between runs, so repeat a run to see reuse.</p>
        </div>
      </div>
      {err && <div className="notice error" style={{ marginBottom: 12 }}>{err}</div>}

      <div className="split">
        <section className="panel stack">
          <h3>Run setup</h3>
          <label className="field">Workload
            <select value={wl} onChange={(e) => setWl(e.target.value)} disabled={running}>
              {wls.map((w) => <option key={w.name} value={w.name}>{WL_LABEL[w.name] || w.name} ({w.family})</option>)}
            </select>
          </label>
          <label className="field">Query group
            <select value={group} onChange={(e) => setGroup(e.target.value)} disabled={running}>
              {tasks.map((t) => <option key={t.group} value={t.group}>{t.group} ({t.n_opt} + {t.n_test} questions)</option>)}
            </select>
          </label>
          <label className="field">Optimizer
            <select value={system} onChange={(e) => setSystem(e.target.value)} disabled={running}>
              {SYSTEMS.map((s) => <option key={s} value={s}>{SYS_LABEL[s]}</option>)}
            </select>
          </label>
          <label className="field">Gate threshold θ: {theta}
            <input type="range" min="0.1" max="8" step="0.1" value={theta} onChange={(e) => setTheta(e.target.value)} disabled={running} />
            <span className="small muted">Higher θ demands more expected value before spending tokens on validation.</span>
          </label>
          <label className="field">History weight α: {alpha}
            <input type="range" min="0" max="1" step="0.05" value={alpha} onChange={(e) => setAlpha(e.target.value)} disabled={running} />
            <span className="small muted">Blend of a directive's past flip rate and the local judge's opinion.</span>
          </label>
          <button className="btn primary" onClick={start} disabled={running || !group}>
            {running ? "Optimizing…" : "Start optimization"}
          </button>
        </section>

        <div className="stack">
          {res && (
            <section className="panel">
              <h3>Outcome on {res.test.n} held-out questions</h3>
              <div className="grid cols-3">
                <div><div className="big num">{fmt.n(res.test_base.accuracy)} <span className="muted" style={{ fontSize: 18 }}>to</span> {fmt.n(res.test.accuracy)}</div>
                  <div className="small muted">accuracy, starting plan to chosen plan</div></div>
                <div><div className="big num">{fmt.int(res.opt_tokens)}</div><div className="small muted">optimization tokens ({fmt.usd(res.opt_usd)})</div></div>
                <div><div className="big num">{res.n_validations} / {res.n_memory_hits} / {res.n_gate_skips}</div>
                  <div className="small muted">validated, reused from memory, skipped by gate</div></div>
              </div>
              {res.lineage.length > 0 && <p className="small" style={{ marginBottom: 0 }}>Applied rewrites: <b>{res.lineage.join(", ")}</b></p>}
            </section>
          )}
          {run?.status === "error" && <div className="notice error">{run.error}</div>}

          <section className="panel">
            <h3>Decision trace {running && <span className="pulse" style={{ marginLeft: 8 }} />}
              <small>{events.filter((e) => e.type === "validation").length} validations, {fmt.int(spent)} validation tokens so far</small></h3>
            <Ribbon events={events} />
            {events.filter((e) => e.type === "tournament").map((e, i) => (
              <div key={i} className="stack" style={{ marginTop: 8 }}>
                <div className="round-head">Round {e.round + 1}: every candidate validated, pairwise among the top plans</div>
                <table className="data"><thead><tr><th>Plan</th><th className="n">Rating</th><th className="n">Pairwise wins</th></tr></thead>
                  <tbody>{e.table.map((r, j) => <tr key={j}><td className="mono">{r.plan}</td><td className="n">{fmt.n(r.score)}</td><td className="n">{r.wins}</td></tr>)}</tbody></table>
              </div>
            ))}
          </section>

          {startEv && (
            <div className="grid cols-2">
              <section className="panel"><h3>Starting plan</h3><PlanDiagram ops={startEv.ops} /></section>
              <section className="panel"><h3>{res ? "Chosen plan" : "Current incumbent"}</h3>
                <PlanDiagram ops={res ? res.plan_ops : (lastInc?.ops || startEv.ops)} compareTo={startEv.ops} /></section>
            </div>
          )}

          {res?.examples?.length > 0 && (
            <section className="panel scroll-x">
              <h3>Held-out answers</h3>
              <table className="data">
                <thead><tr><th>Question</th><th>Starting plan</th><th>Chosen plan</th><th>Reference</th><th className="n">Score</th></tr></thead>
                <tbody>{res.examples.map((x, i) => (
                  <tr key={i}><td>{x.question}</td><td className="small">{x.baseline}</td><td className="small">{x.sql || x.answer}</td>
                    <td className="small muted">{x.reference.join(" | ").slice(0, 200)}</td><td className="n">{fmt.n(x.score)}</td></tr>))}
                </tbody>
              </table>
            </section>
          )}
        </div>
      </div>
    </>
  );
}
