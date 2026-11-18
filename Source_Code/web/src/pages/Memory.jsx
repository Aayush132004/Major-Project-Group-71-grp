import { useEffect, useState } from "react";
import { api, fmt, WL_LABEL } from "../api";
import { GroupedBars, Scatter } from "../components/Charts";

export default function Memory() {
  const [wls, setWls] = useState([]);
  const [wl, setWl] = useState("cuad");
  const [mem, setMem] = useState(null);
  const [err, setErr] = useState(null);

  const load = () => api.memory(wl).then((m) => { setMem(m); setErr(null); }).catch((e) => setErr(e.message));
  useEffect(() => { api.workloads().then(setWls); }, []);
  useEffect(() => { load(); }, [wl]);

  const reset = async () => {
    if (!confirm(`Clear the plan memory for ${WL_LABEL[wl] || wl}? Stored verdicts and gate statistics are deleted.`)) return;
    await api.resetMemory(wl);
    load();
  };

  const flips = mem?.flip_stats || [];
  const gate = (mem?.gate_log || []).filter((g) => g.voi > 0 && g.token_cost > 0);
  return (
    <>
      <div className="page-head">
        <div>
          <h2>Plan memory</h2>
          <p>Verdicts of validated rewrites, keyed by context signature. Confidence decays with age and with drift in the
             data statistics; a verdict is reused only above the threshold.</p>
        </div>
        <div className="row">
          <label className="field">Workload
            <select value={wl} onChange={(e) => setWl(e.target.value)}>
              {wls.map((w) => <option key={w.name} value={w.name}>{WL_LABEL[w.name] || w.name}</option>)}
            </select>
          </label>
          <button className="btn" onClick={load}>Refresh</button>
          <button className="btn" onClick={reset}>Clear memory</button>
        </div>
      </div>
      {err && <div className="notice error">{err}</div>}
      {mem && (
        <>
          <div className="grid cols-3" style={{ marginBottom: 16 }}>
            <section className="panel"><div className="big num">{mem.rewrites.length}</div><div className="small muted">stored verdicts</div></section>
            <section className="panel"><div className="big num">{mem.gate_log.filter((g) => g.flipped != null).length}</div>
              <div className="small muted">validations logged for gate calibration</div></section>
            <section className="panel"><div className="big num">{mem.telemetry_rows}</div><div className="small muted">observed-cost rows for the residual model</div></section>
          </div>

          <div className="grid cols-2">
            <section className="panel">
              <h3>Historical flip rate h(dir) <small>Laplace-smoothed</small></h3>
              {flips.length ? (
                <GroupedBars groups={flips.map((f) => ({ key: f.directive, label: f.directive.replace(/_/g, " ").toLowerCase() }))}
                  series={[{ key: "h", label: "flip rate", color: "var(--validate)" }]}
                  value={(g) => flips.find((f) => f.directive === g)?.h} format={(v) => v.toFixed(2)} height={230} />
              ) : <p className="muted">No validations yet. Run the optimizer once.</p>}
            </section>
            <section className="panel">
              <h3>Gate decisions <small>VoI against θ × validation cost</small></h3>
              {gate.length ? (
                <Scatter points={gate.map((g) => ({ x: g.token_cost, y: g.voi,
                  color: g.flipped == null ? "var(--skip)" : g.flipped ? "var(--prefer)" : "var(--reject)",
                  label: `${g.directive}: ${g.flipped == null ? "skipped" : g.flipped ? "validated, flipped" : "validated, kept incumbent"}` }))}
                  xLabel="θ × validation cost (USD)" yLabel="VoI (USD)"
                  legend={[{ key: "f", label: "validated and flipped", color: "var(--prefer)" },
                           { key: "k", label: "validated, incumbent kept", color: "var(--reject)" },
                           { key: "s", label: "skipped", color: "var(--skip)" }]} />
              ) : <p className="muted">No gate decisions logged yet.</p>}
              <p className="small muted" style={{ margin: 0 }}>Points above the dashed line were admitted to validation.</p>
            </section>
          </div>

          <section className="panel scroll-x" style={{ marginTop: 16 }}>
            <h3>Stored verdicts <small>reuse threshold {mem.settings.reuse_threshold}, κ = {mem.settings.kappa}/day</small></h3>
            <table className="data">
              <thead><tr><th>Directive</th><th>Context</th><th>Verdict</th><th className="n">Accuracy Δ</th><th className="n">Confirmations</th><th className="n">Age (days)</th><th>Confidence</th></tr></thead>
              <tbody>
                {mem.rewrites.map((r) => (
                  <tr key={r.sig}>
                    <td><b>{r.directive}</b><div className="small muted mono">{r.sig}</div></td>
                    <td className="small">{r.features.parent_shape}<div className="muted">selectivity bucket {r.features.sel_b}, top-k bucket {r.features.topk_b}, {r.features.index || "no index"}</div></td>
                    <td><span className={`tag ${r.verdict}`}>{r.verdict}</span></td>
                    <td className="n">{fmt.n(r.acc_delta, 3)}</td>
                    <td className="n">{r.n_obs}</td>
                    <td className="n">{fmt.n(r.age_days, 1)}</td>
                    <td style={{ minWidth: 130 }}>
                      <div className="bar-track"><div className="bar-fill" style={{ width: `${r.confidence * 100}%`,
                        background: r.confidence >= mem.settings.reuse_threshold ? "var(--memory)" : "var(--rule-strong)" }} /></div>
                      <span className="small">{fmt.n(r.confidence)} {r.confidence >= mem.settings.reuse_threshold ? "reusable" : "re-validate"}</span>
                    </td>
                  </tr>
                ))}
                {!mem.rewrites.length && <tr><td colSpan="7" className="muted">Empty. Each validation writes one verdict here.</td></tr>}
              </tbody>
            </table>
          </section>

          {mem.agreement.length > 0 && (
            <section className="panel" style={{ marginTop: 16 }}>
              <h3>Local judge agreement with the LLM validator</h3>
              <table className="data"><thead><tr><th>Directive</th><th className="n">Agreed</th><th className="n">Compared</th><th className="n">Rate</th></tr></thead>
                <tbody>{mem.agreement.map((a) => <tr key={a.directive}><td>{a.directive}</td><td className="n">{a.agree}</td><td className="n">{a.total}</td><td className="n">{fmt.n(a.rate)}</td></tr>)}</tbody></table>
            </section>
          )}
        </>
      )}
    </>
  );
}
