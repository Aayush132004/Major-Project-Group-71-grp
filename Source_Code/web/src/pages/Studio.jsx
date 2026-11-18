import { useEffect, useState } from "react";
import { api, fmt, WL_LABEL } from "../api";
import { PlanDiagram } from "../components/Plan";

export default function Studio() {
  const [wls, setWls] = useState([]);
  const [wl, setWl] = useState("cuad");
  const [yaml, setYaml] = useState("");
  const [tasks, setTasks] = useState([]);
  const [group, setGroup] = useState("");
  const [cost, setCost] = useState(null);
  const [exec, setExec] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [directives, setDirectives] = useState([]);

  useEffect(() => { api.workloads().then(setWls); api.directives().then(setDirectives); }, []);
  useEffect(() => {
    const w = wls.find((x) => x.name === wl);
    if (w) setYaml(w.base_yaml);
    setExec(null);
    api.tasks(wl).then((t) => { setTasks(t); setGroup(t[0]?.group || ""); });
  }, [wl, wls]);
  useEffect(() => {
    if (!yaml) return;
    const h = setTimeout(() => {
      api.cost({ workload: wl, yaml, group }).then((c) => { setCost(c); setErr(null); }).catch((e) => setErr(e.message));
    }, 350);
    return () => clearTimeout(h);
  }, [yaml, group, wl]);

  const run = async () => {
    setBusy(true); setExec(null);
    try { setExec(await api.execute({ workload: wl, yaml, group })); setErr(null); } catch (e) { setErr(e.message); }
    setBusy(false);
  };
  const est = cost?.hybrid || cost?.symbolic;
  const task = tasks.find((t) => t.group === group);

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Pipeline studio</h2>
          <p>Edit a pipeline in the YAML language, see what the cost model expects before anything runs, then execute it on one held-out question.</p>
        </div>
        <div className="row">
          <label className="field">Workload
            <select value={wl} onChange={(e) => setWl(e.target.value)}>
              {wls.map((w) => <option key={w.name} value={w.name}>{WL_LABEL[w.name] || w.name}</option>)}
            </select>
          </label>
          <label className="field">Query group
            <select value={group} onChange={(e) => setGroup(e.target.value)}>
              {tasks.map((t) => <option key={t.group} value={t.group}>{t.group}</option>)}
            </select>
          </label>
        </div>
      </div>
      {err && <div className="notice error" style={{ marginBottom: 12 }}>{err}</div>}

      <div className="grid cols-2">
        <section className="panel stack">
          <h3>Pipeline YAML</h3>
          <textarea className="yaml" spellCheck={false} value={yaml} onChange={(e) => setYaml(e.target.value)} aria-label="Pipeline YAML" />
          <div className="row">
            <button className="btn primary" onClick={run} disabled={busy}>{busy ? "Running…" : "Run on one question"}</button>
            <button className="btn" onClick={() => setYaml(wls.find((x) => x.name === wl)?.base_yaml || "")}>Reset to starting pipeline</button>
          </div>
          <details>
            <summary className="small">Rewrite directives the optimizer can apply</summary>
            <table className="data" style={{ marginTop: 8 }}>
              <tbody>{directives.map((d) => <tr key={d.name}><td><b>{d.name}</b></td><td className="muted">{d.family}</td><td>{d.description}</td></tr>)}</tbody>
            </table>
          </details>
        </section>

        <section className="panel stack">
          <h3>Plan and cost estimate <small>{cost?.residual_fitted ? "hybrid (symbolic + learned residual)" : "symbolic backbone"}</small></h3>
          {cost && <PlanDiagram ops={cost.ops} costs={est?.breakdown?.per_op} />}
          {est && (
            <dl className="kv">
              <dt>Per run</dt><dd>{fmt.usd(est.usd)}, {fmt.ms(est.latency_ms)}, {fmt.int(est.tokens)} tokens, {fmt.n(est.llm_calls, 1)} LLM calls</dd>
              <dt>Time split</dt><dd>DB {fmt.ms(est.breakdown.db_ms)}, ANN {fmt.ms(est.breakdown.ann_ms)}, rerank {fmt.ms(est.breakdown.rerank_ms)}, LLM {fmt.ms(est.breakdown.llm_ms)}</dd>
              <dt>Statistics used</dt><dd>{fmt.int(cost.stats.n_rows)} rows, selectivity estimate {fmt.n(cost.stats.sel_est, 4)} (true {fmt.n(cost.stats.sel_true, 4)})</dd>
            </dl>
          )}
          {task && <p className="small muted" style={{ margin: 0 }}>Example question: {task.sample_questions[0]}</p>}
        </section>
      </div>

      {exec && (
        <section className="panel" style={{ marginTop: 16 }}>
          <h3>Execution on one held-out question</h3>
          <div className="grid cols-2">
            <div className="stack">
              <div><span className="muted small">Question</span><div>{exec.query.question}</div></div>
              <div><span className="muted small">Pipeline answer</span><pre className="block">{exec.result.sql || exec.result.answer || "(empty)"}</pre></div>
              {exec.result.rows?.length > 0 && <div><span className="muted small">First result rows</span><pre className="block">{exec.result.rows.slice(0, 5).map((r) => r.join(" | ")).join("\n")}</pre></div>}
              <div><span className="muted small">Reference</span><pre className="block">{exec.query.answers.slice(0, 3).join("\n")}</pre></div>
            </div>
            <div className="stack">
              <dl className="kv">
                <dt>Score</dt><dd><b>{fmt.n(exec.score.accuracy)}</b> {exec.score.precision != null && `(P ${fmt.n(exec.score.precision)}, R ${fmt.n(exec.score.recall)}, Rouge-L ${fmt.n(exec.score.rouge_l)})`}</dd>
                <dt>Tokens</dt><dd>{fmt.int(exec.result.tokens)}</dd>
                <dt>Wall clock</dt><dd>{fmt.ms(exec.result.latency_ms)}</dd>
              </dl>
              <table className="data">
                <thead><tr><th>Operator</th><th className="n">Rows in</th><th className="n">Rows out</th><th className="n">ms</th><th className="n">Tokens</th></tr></thead>
                <tbody>{exec.result.telemetry.map((t) => (
                  <tr key={t.op}><td>{t.kind} <span className="muted small">{t.op}</span></td><td className="n">{fmt.int(t.rows_in)}</td>
                    <td className="n">{fmt.int(t.rows_out)}</td><td className="n">{fmt.n(t.ms, 1)}</td><td className="n">{fmt.int(t.tokens)}</td></tr>))}
                </tbody>
              </table>
              {exec.result.context?.length > 0 && (
                <details><summary className="small">Context the LLM saw ({exec.result.context.length} passages)</summary>
                  <ol className="small">{exec.result.context.map((c, i) => <li key={i}>{c.slice(0, 300)}</li>)}</ol></details>
              )}
            </div>
          </div>
        </section>
      )}
    </>
  );
}
