import { useEffect, useState } from "react";
import { api } from "../api";

const CAPTIONS = {
  fig_baselines_accuracy: "Held-out accuracy of the plan each system chose",
  fig_baselines_tokens: "Optimization-time token spend per task",
  fig_pareto_budget: "Accuracy against the optimization token budget",
  fig_theta_sweep: "Gate threshold θ as an accuracy and spend dial",
  fig_gate_pr: "Gate precision, recall and avoided spend across θ",
  fig_gate_calibration: "Calibration of the predicted flip probability",
  fig_memory_hits: "Plan-memory hit rate on repeated and perturbed shapes",
  fig_memory_decay: "Verdict confidence decay for three values of κ",
  fig_cost_scatter: "Symbolic cost estimates against observed cost",
  fig_cost_qerror: "Calibration error of symbolic, neural and hybrid cost models",
  fig_latency_breakdown: "Execution wall-clock split into DB, ANN and LLM time",
  fig_ann_sweep: "Time/recall trade-off of the three ANN index families",
  fig_ablation: "Ablations under an identical token budget",
};

export default function Report() {
  const [backends, setBackends] = useState([]);
  const [backend, setBackend] = useState(null);
  const [files, setFiles] = useState([]);

  useEffect(() => {
    api.experiments().then((ix) => {
      const b = [...new Set(ix.available.map((a) => a.backend))].filter((b) => b !== "sim");
      setBackends(b);
      setBackend(b.find((x) => x !== "sim") || b[0] || null);
    });
  }, []);
  useEffect(() => { if (backend) api.reportFiles(backend).then(setFiles); }, [backend]);

  const figs = files.filter((f) => f.endsWith(".png"));
  const others = files.filter((f) => !f.endsWith(".png"));
  return (
    <>
      <div className="page-head">
        <div>
          <h2>Report assets</h2>
          <p>300 dpi figures, LaTeX tables and CSVs generated from the experiment results, ready to drop into the report.
             They are written to <code>runs/{backend || "<model>"}/report/</code>.</p>
        </div>
        {backends.length > 0 && (
          <label className="field">Results from
            <select value={backend || ""} onChange={(e) => setBackend(e.target.value)}>
              {backends.map((b) => <option key={b} value={b}>{b.replace("_", "/")}</option>)}
            </select>
          </label>
        )}
      </div>
      {!files.length && <p className="muted">No report assets yet. Run <code>python scripts/run_experiments.py</code>; the figures are built at the end.</p>}
      {others.length > 0 && (
        <section className="panel" style={{ marginBottom: 16 }}>
          <h3>Tables and transcripts</h3>
          <div className="row">{others.map((f) => <a key={f} className="btn" href={`/api/report/${backend}/${f}`} download>{f}</a>)}</div>
        </section>
      )}
      <div className="figs">
        {figs.map((f) => (
          <figure key={f}>
            <img src={`/api/report/${backend}/${f}`} alt={CAPTIONS[f.replace(".png", "")] || f} loading="lazy" />
            <figcaption><span>{CAPTIONS[f.replace(".png", "")] || f}</span><a href={`/api/report/${backend}/${f}`} download>{f}</a></figcaption>
          </figure>
        ))}
      </div>
    </>
  );
}
