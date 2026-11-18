import { useEffect, useRef, useState } from "react";
import { api, fmt, SYS_COLOR, SYS_LABEL, WL_LABEL } from "../api";
import { GroupedBars, Lines } from "../components/Charts";

const WL_ORDER = ["msmarco", "cuad", "spider", "bird"];
const wlGroups = (rows) => WL_ORDER.filter((w) => rows.some((r) => r.workload === w)).map((w) => ({ key: w, label: WL_LABEL[w] }));
const sysSeries = (rows, order) => order.filter((s) => rows.some((r) => r.system === s)).map((s) => ({ key: s, label: SYS_LABEL[s], color: SYS_COLOR[s] }));
const pick = (rows, key) => (w, s) => rows.find((r) => r.workload === w && r.system === s)?.[key] ?? null;
const SECTIONS = [
  ["baselines", "10.2 Baselines"], ["pareto", "10.2 Budget and θ"], ["gate", "10.3 VoI gate"], ["memory", "10.4 Plan memory"],
  ["cost_model", "10.5 Cost model"], ["latency", "10.6 Latency"], ["ablation", "10.7 Ablations"], ["transcripts", "10.8 Transcripts"],
];

export default function Results({ health }) {
  const [index, setIndex] = useState(null);
  const [backend, setBackend] = useState(null);
  const [data, setData] = useState({});
  const [tab, setTab] = useState("baselines");

  const refresh = () => api.experiments().then((ix) => {
    setIndex(ix);
    const backends = [...new Set(ix.available.map((a) => a.backend))].filter((b) => b !== "sim");
    setBackend((b) => b && backends.includes(b) ? b : backends.find((x) => x !== "sim") || backends[0] || null);
  });
  useEffect(() => { refresh(); }, []);
  useEffect(() => {
    if (!backend || !index) return;
    setData({});
    index.available.filter((a) => a.backend === backend).forEach((a) =>
      api.experiment(backend, a.name).then((d) => setData((prev) => ({ ...prev, [a.name]: d }))));
  }, [backend, index]);

  const backends = index ? [...new Set(index.available.map((a) => a.backend))] : [];
  const d = data[tab];
  return (
    <>
      <div className="page-head">
        <div>
          <h2>Experiments</h2>
          <p>Results of the Section 10 protocol. Every number is computed on held-out questions that the optimizer never saw;
             reference answers are read only by the scorer.</p>
        </div>
        {backends.length > 0 && (
          <label className="field">Results from
            <select value={backend || ""} onChange={(e) => setBackend(e.target.value)}>
              {backends.map((b) => <option key={b} value={b}>{b.replace("_", "/")}</option>)}
            </select>
          </label>
        )}
      </div>

      <RunPanel health={health} index={index} onDone={refresh} />

      <div className="seg" role="tablist" style={{ margin: "16px 0", flexWrap: "wrap" }}>
        {SECTIONS.map(([k, label]) => (
          <button key={k} role="tab" aria-pressed={tab === k} onClick={() => setTab(k)} disabled={!data[k]}>{label}</button>
        ))}
      </div>
      {!d && <p className="muted">{backends.length ? "Pick a section with results." : "No experiment results yet. Run the suite above."}</p>}
      {d && tab === "baselines" && <Baselines d={d} />}
      {d && tab === "pareto" && <Pareto d={d} />}
      {d && tab === "gate" && <Gate d={d} />}
      {d && tab === "memory" && <MemoryExp d={d} />}
      {d && tab === "cost_model" && <CostModel d={d} />}
      {d && tab === "latency" && <Latency d={d} />}
      {d && tab === "ablation" && <Ablation d={d} />}
      {d && tab === "transcripts" && <Transcripts d={d} />}
      {d && <p className="small muted">Profile {d.profile || "–"}, model {d.model}, finished {d.created}{d.elapsed_s ? `, took ${(d.elapsed_s / 60).toFixed(1)} min` : ""}.</p>}
    </>
  );
}

function RunPanel({ health, index, onDone }) {
  const [names, setNames] = useState(["baselines"]);
  const [profile, setProfile] = useState("smoke");
  const [job, setJob] = useState(null);
  const timer = useRef(null);
  useEffect(() => () => clearInterval(timer.current), []);
  const start = async () => {
    const { job_id } = await api.runExperiments({ names, profile });
    timer.current = setInterval(async () => {
      const j = await api.job(job_id);
      setJob(j);
      if (j.status !== "running") { clearInterval(timer.current); onDone(); }
    }, 1500);
  };
  if (!index) return null;
  return (
    <details className="panel">
      <summary><b>Run experiments from here</b> <span className="small muted">on {health.model}</span></summary>
      <div className="stack" style={{ marginTop: 10 }}>
        <div className="row">
          {index.names.map((n) => (
            <label key={n} className="row small" style={{ gap: 4 }}>
              <input type="checkbox" checked={names.includes(n)} onChange={(e) => setNames(e.target.checked ? [...names, n] : names.filter((x) => x !== n))} />{n}
            </label>
          ))}
        </div>
        <div className="row">
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            {Object.entries(index.profiles).map(([k, v]) => <option key={k} value={k}>{k}: {v.n_tasks} groups, {v.n_opt} + {v.n_test} questions each</option>)}
          </select>
          <button className="btn primary" onClick={start} disabled={!names.length || job?.status === "running"}>Run selected</button>
          {job && <span className="small">{job.status === "running" ? <><span className="pulse" /> running</> : job.status}</span>}
        </div>
        {job && <pre className="block small" style={{ maxHeight: 180 }}>{job.log.slice(-14).join("\n")}</pre>}
        <p className="small muted" style={{ margin: 0 }}>Long runs are better started from a terminal: <code>python scripts/run_experiments.py --profile quick</code>.</p>
      </div>
    </details>
  );
}

function Baselines({ d }) {
  const s = d.summary;
  const order = ["no_opt", "random", "cost_only", "docetl", "plannerllm"];
  return (
    <div className="stack">
      <div className="grid cols-2">
        <section className="panel"><h3>Held-out accuracy <small>F1 for text, execution accuracy for SQL</small></h3>
          <GroupedBars groups={wlGroups(s)} series={sysSeries(s, order)} value={pick(s, "accuracy")} format={(v) => v.toFixed(2)} /></section>
        <section className="panel"><h3>Optimization tokens per task</h3>
          <GroupedBars groups={wlGroups(s)} series={sysSeries(s, ["cost_only", "docetl", "plannerllm"])} value={pick(s, "opt_tokens")} /></section>
      </div>
      <SummaryTable rows={s} cols={[["accuracy", "Accuracy", 3], ["hallucinated", "Halluc.", 3], ["opt_tokens", "Opt. tokens", 0],
        ["exec_tokens", "Exec. tokens / q", 0], ["exec_ms", "Exec. ms / q", 1], ["n_validations", "Validations", 1],
        ["n_memory_hits", "Memory hits", 1], ["n_gate_skips", "Gate skips", 1], ["decision_agreement", "Agrees with full validation", 2]]} />
    </div>
  );
}

function SummaryTable({ rows, cols }) {
  return (
    <section className="panel scroll-x">
      <table className="data">
        <thead><tr><th>Workload</th><th>System</th>{cols.map(([k, l]) => <th key={k} className="n">{l}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i} className={r.system === "plannerllm" ? "hl" : ""}><td>{WL_LABEL[r.workload]}</td><td>{SYS_LABEL[r.system] || r.system}</td>
            {cols.map(([k, , dp]) => <td key={k} className="n">{r[k] == null ? "–" : dp === 0 ? fmt.int(r[k]) : fmt.n(r[k], dp)}</td>)}</tr>))}
        </tbody>
      </table>
    </section>
  );
}

function Pareto({ d }) {
  const wls = WL_ORDER.filter((w) => d.points.some((p) => p.workload === w));
  return (
    <div className="stack">
      <div className="grid cols-3">
        {wls.map((w) => (
          <section key={w} className="panel"><h3>{WL_LABEL[w]} <small>accuracy against spend</small></h3>
            <Lines height={240} xLabel="optimization tokens / task" yLabel="accuracy" format={(v) => v.toFixed(2)}
              series={["docetl", "no_gate", "plannerllm"].map((s) => ({ key: s, label: SYS_LABEL[s], color: SYS_COLOR[s],
                points: d.points.filter((p) => p.workload === w && p.system === s && p.sweep === "budget" && p.accuracy != null)
                  .map((p) => ({ x: p.opt_tokens, y: p.accuracy, label: `budget ${p.budget ?? "unlimited"}: ${fmt.int(p.opt_tokens)} tokens, accuracy ${fmt.n(p.accuracy)}` })) }))} />
          </section>
        ))}
      </div>
      <div className="grid cols-2">
        <section className="panel"><h3>θ sweep: accuracy</h3>
          <Lines logX xLabel="θ" yLabel="accuracy" format={(v) => v.toFixed(2)} xFormat={(v) => `${v}`}
            series={wls.map((w, i) => ({ key: w, label: WL_LABEL[w], color: `var(--s${i + 1})`,
              points: d.points.filter((p) => p.workload === w && p.sweep === "theta").map((p) => ({ x: p.theta, y: p.accuracy })) }))} /></section>
        <section className="panel"><h3>θ sweep: optimization tokens</h3>
          <Lines logX xLabel="θ" yLabel="tokens / task" xFormat={(v) => `${v}`}
            series={wls.map((w, i) => ({ key: w, label: WL_LABEL[w], color: `var(--s${i + 1})`,
              points: d.points.filter((p) => p.workload === w && p.sweep === "theta").map((p) => ({ x: p.theta, y: p.opt_tokens })) }))} /></section>
      </div>
    </div>
  );
}

function Gate({ d }) {
  const sc = d.overall.scores.filter((s) => s.precision != null);
  return (
    <div className="stack">
      <div className="grid cols-2">
        <section className="panel"><h3>Gate as a flip predictor <small>shadow mode: every candidate validated, gate decision replayed</small></h3>
          <Lines logX yMin={0} yMax={1} xLabel="θ" yLabel="score" format={(v) => v.toFixed(1)} xFormat={(v) => `${v}`}
            series={[{ key: "p", label: "precision", color: "var(--s1)", points: sc.map((s) => ({ x: s.theta, y: s.precision })) },
                     { key: "r", label: "recall", color: "var(--s2)", points: sc.map((s) => ({ x: s.theta, y: s.recall })) },
                     { key: "s", label: "validation spend avoided", color: "var(--s3)", points: sc.map((s) => ({ x: s.theta, y: s.spend_saved })) }]} />
        </section>
        <section className="panel"><h3>Calibration <small>predicted P(flip) against observed flip frequency</small></h3>
          <Lines yMin={0} yMax={1} xLabel="predicted" yLabel="observed" format={(v) => v.toFixed(1)} xFormat={(v) => v.toFixed(1)}
            series={[{ key: "ideal", label: "perfect calibration", color: "var(--rule-strong)", points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] },
                     { key: "cal", label: "P(flip)", color: "var(--s1)", points: d.overall.calibration.map((c) => ({ x: c.predicted, y: c.observed, label: `${c.n} decisions: predicted ${fmt.n(c.predicted)}, observed ${fmt.n(c.observed)}` })) }]} />
        </section>
      </div>
      <section className="panel scroll-x"><h3>Per workload</h3>
        <table className="data"><thead><tr><th>Workload</th><th className="n">θ</th><th className="n">Decisions</th><th className="n">Admitted</th><th className="n">Precision</th><th className="n">Recall</th><th className="n">Flip rate</th><th className="n">Spend avoided</th></tr></thead>
          <tbody>{Object.entries(d.per_workload).flatMap(([w, v]) => v.scores.map((s) => (
            <tr key={w + s.theta} className={s.theta === 1 ? "hl" : ""}><td>{WL_LABEL[w]}</td><td className="n">{s.theta}</td><td className="n">{s.n}</td><td className="n">{s.admitted}</td>
              <td className="n">{fmt.n(s.precision)}</td><td className="n">{fmt.n(s.recall)}</td><td className="n">{fmt.n(s.flip_rate)}</td><td className="n">{fmt.pct(s.spend_saved)}</td></tr>)))}
          </tbody></table>
      </section>
    </div>
  );
}

function MemoryExp({ d }) {
  const phases = ["first-seen", "repeated", "prompt_edit", "config_change"];
  const wls = WL_ORDER.filter((w) => d.streams.some((s) => s.workload === w));
  const series = wls.map((w, i) => ({ key: w, label: WL_LABEL[w], color: `var(--s${i + 1})` }));
  const groups = phases.map((p) => ({ key: p, label: p.replace("_", " ") }));
  const v = (key) => (p, w) => d.streams.find((s) => s.workload === w && s.phase === p)?.[key] ?? null;
  return (
    <div className="stack">
      <div className="grid cols-2">
        <section className="panel"><h3>Signature hit rate</h3><GroupedBars groups={groups} series={series} value={v("hit_rate")} format={(x) => x.toFixed(2)} /></section>
        <section className="panel"><h3>Optimization tokens per task</h3><GroupedBars groups={groups} series={series} value={v("opt_tokens")} /></section>
      </div>
      <section className="panel scroll-x"><h3>Reuse accuracy <small>memory hits re-validated in shadow; agreement of the reused verdict with a fresh one</small></h3>
        <table className="data"><thead><tr><th>Workload</th><th>Phase</th><th className="n">Hit rate</th><th className="n">Hits audited</th><th className="n">Reuse accuracy</th><th className="n">Accuracy</th></tr></thead>
          <tbody>{d.streams.map((s, i) => <tr key={i}><td>{WL_LABEL[s.workload]}</td><td>{s.phase}</td><td className="n">{fmt.n(s.hit_rate)}</td><td className="n">{s.n_audited}</td><td className="n">{fmt.n(s.reuse_accuracy)}</td><td className="n">{fmt.n(s.accuracy)}</td></tr>)}</tbody></table>
      </section>
      <section className="panel scroll-x"><h3>Decay: second pass over the same stream, 3 days per task</h3>
        <table className="data"><thead><tr><th>Workload</th><th className="n">κ per day</th><th className="n">Hit rate</th><th className="n">Opt. tokens</th><th className="n">Accuracy</th></tr></thead>
          <tbody>{d.decay.map((s, i) => <tr key={i}><td>{WL_LABEL[s.workload]}</td><td className="n">{s.kappa}</td><td className="n">{fmt.n(s.hit_rate_pass2)}</td><td className="n">{fmt.int(s.opt_tokens_pass2)}</td><td className="n">{fmt.n(s.accuracy_pass2)}</td></tr>)}</tbody></table>
      </section>
    </div>
  );
}

function CostModel({ d }) {
  const usd = d.summary.filter((s) => s.target === "usd");
  const groups = usd.map((s) => ({ key: s.split, label: s.split.replace("zero-shot:", "zero-shot ") }));
  const models = [["symbolic", "var(--s1)"], ["neural", "var(--s2)"], ["hybrid", "var(--s3)"]];
  return (
    <div className="stack">
      <section className="panel"><h3>Median q-error of the USD estimate <small>1.0 is exact; {d.n_samples} executed plans</small></h3>
        <GroupedBars groups={groups} series={models.map(([k, c]) => ({ key: k, label: k, color: c }))}
          value={(g, m) => usd.find((s) => s.split === g)?.[m]?.median_q ?? null} format={(v) => v.toFixed(2)} /></section>
      <section className="panel scroll-x"><table className="data">
        <thead><tr><th>Split</th><th>Target</th><th>Model</th><th className="n">MAE (log10)</th><th className="n">Median rel. error</th><th className="n">P90 rel. error</th><th className="n">Median q</th><th className="n">P90 q</th></tr></thead>
        <tbody>{d.summary.flatMap((s) => models.filter(([m]) => s[m]).map(([m]) => (
          <tr key={s.split + s.target + m} className={m === "hybrid" ? "hl" : ""}><td>{s.split}</td><td>{s.target}</td><td>{m}</td>
            {["mae_log10", "median_rel", "p90_rel", "median_q", "p90_q"].map((k) => <td key={k} className="n">{fmt.n(s[m][k], 3)}</td>)}</tr>)))}
        </tbody></table></section>
    </div>
  );
}

function Latency({ d }) {
  const wls = WL_ORDER.filter((w) => d.ann_sweep.some((r) => r.workload === w));
  return (
    <div className="stack">
      <div className="grid cols-3">
        {wls.map((w) => (
          <section key={w} className="panel"><h3>{WL_LABEL[w]} <small>recall@10 against probe time</small></h3>
            <Lines height={230} yMin={0} yMax={1.02} xLabel="median ms per probe" yLabel="recall@10" format={(v) => v.toFixed(1)} xFormat={(v) => v.toFixed(2)}
              series={["hnsw", "ivf", "flat"].map((f, i) => ({ key: f, label: f.toUpperCase(), color: `var(--s${i + 1})`,
                points: d.ann_sweep.filter((r) => r.workload === w && r.family === f && r.k === 10).map((r) => ({ x: r.ms, y: r.recall, label: `${f} knob ${r.knob}: ${r.ms.toFixed(3)} ms, recall ${r.recall.toFixed(2)}` })) }))} />
          </section>
        ))}
      </div>
      <section className="panel scroll-x"><h3>Filtered vector probe against post-filtering</h3>
        <table className="data"><thead><tr><th>Workload</th><th>Index</th><th className="n">Selectivity</th><th className="n">Filtered recall</th><th className="n">Post-filter recall</th><th className="n">Filtered ms</th><th className="n">Post-filter ms</th></tr></thead>
          <tbody>{wls.flatMap((w) => ["hnsw", "ivf", "flat"].map((f) => {
            const ps = d.probe.filter((p) => p.workload === w && p.family === f);
            if (!ps.length) return null;
            const m = (k) => ps.reduce((a, p) => a + p[k], 0) / ps.length;
            return <tr key={w + f}><td>{WL_LABEL[w]}</td><td>{f}</td><td className="n">{fmt.n(m("sel"), 4)}</td><td className="n">{fmt.n(m("filtered_recall"))}</td><td className="n">{fmt.n(m("post_recall"))}</td><td className="n">{fmt.n(m("filtered_ms"), 2)}</td><td className="n">{fmt.n(m("post_ms"), 2)}</td></tr>;
          }))}</tbody></table>
      </section>
      {d.llm_time && (
        <section className="panel scroll-x"><h3>Where the LLM time goes <small>smoke-sized stream, per purpose</small></h3>
          <table className="data"><thead><tr><th>Workload</th><th>System</th><th>Purpose</th><th className="n">Calls</th><th className="n">Tokens</th><th className="n">LLM seconds</th></tr></thead>
            <tbody>{d.llm_time.flatMap((r) => Object.entries(r).filter(([k]) => !["workload", "system"].includes(k)).map(([p, v]) => (
              <tr key={r.workload + r.system + p}><td>{WL_LABEL[r.workload]}</td><td>{SYS_LABEL[r.system]}</td><td>{p}</td><td className="n">{v.calls}</td><td className="n">{fmt.int(v.tokens)}</td><td className="n">{fmt.n(v.latency_s, 1)}</td></tr>)))}</tbody></table>
        </section>
      )}
    </div>
  );
}

function Ablation({ d }) {
  const s = d.summary;
  const order = ["plannerllm", "no_gate", "no_memory", "symbolic_only", "neural_only"];
  return (
    <div className="stack">
      <div className="grid cols-2">
        <section className="panel"><h3>Accuracy <small>budget {d.budget ? fmt.int(d.budget) + " tokens" : "unlimited"}, second pass</small></h3>
          <GroupedBars groups={wlGroups(s)} series={sysSeries(s, order)} value={pick(s, "accuracy")} format={(v) => v.toFixed(2)} /></section>
        <section className="panel"><h3>Optimization tokens per task</h3>
          <GroupedBars groups={wlGroups(s)} series={sysSeries(s, order)} value={pick(s, "opt_tokens")} /></section>
      </div>
      <SummaryTable rows={s} cols={[["accuracy", "Accuracy", 3], ["opt_tokens", "Opt. tokens", 0], ["n_validations", "Validations", 1],
        ["n_memory_hits", "Memory hits", 1], ["n_gate_skips", "Gate skips", 1]]} />
    </div>
  );
}

function Transcripts({ d }) {
  return (
    <div className="stack">
      <p className="muted" style={{ margin: 0 }}>{d.counts.accepted} accepted rewrites, {d.counts.gate_skipped} gate skips and {d.counts.memory_reused} memory reuses in the baseline run.</p>
      {d.transcripts.map((t, i) => (
        <section key={i} className="panel stack">
          <div className="row"><b>{t.directive}</b><span className={`tag ${t.source}`}>{t.source}</span><span className={`tag ${t.verdict}`}>{t.verdict}</span>
            <span className="small muted">{WL_LABEL[t.workload]}, {t.task}</span></div>
          <div className="small">{t.detail}</div>
          <code className="small">{t.plan}</code>
          {t.gate && <div className="small"><span className="muted">Gate: </span>{t.gate.rationale}</div>}
          {t.memory && <div className="small"><span className="muted">Memory: </span>{t.note}</div>}
          {t.validation && (
            <div className="stack">
              <div className="small"><span className="muted">Criteria: </span>{t.validation.criteria.map((c) => c.name).join(", ")}.
                <span className="muted"> Ratings: </span>{fmt.n(t.validation.score_c)} against {fmt.n(t.validation.score_p)}</div>
              <table className="data"><thead><tr><th>Question</th><th>Candidate</th><th>Incumbent</th></tr></thead>
                <tbody>{t.validation.items.slice(0, 3).map((it, j) => <tr key={j}><td className="small">{it.question}</td>
                  <td className="small">{it.candidate.slice(0, 220)} <b>({fmt.n(it.score_c, 1)})</b></td><td className="small">{it.incumbent.slice(0, 220)} <b>({fmt.n(it.score_p, 1)})</b></td></tr>)}</tbody></table>
            </div>
          )}
        </section>
      ))}
    </div>
  );
}
