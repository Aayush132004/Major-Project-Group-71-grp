import { useEffect, useState } from "react";
import { api, fmt, WL_LABEL } from "../api";
import { PlanDiagram } from "../components/Plan";

// The six services of the architecture, in the order a request travels.
const FLOW = [
  { name: "Directive store", note: "13 rewrites in 4 families", page: "studio" },
  { name: "Generation agent", note: "LLM nominates rewrites", page: "optimize" },
  { name: "Hybrid cost model", note: "DuckDB stats + ANN + tokens + residual", page: "studio", core: true },
  { name: "Plan memory", note: "signature index with decay", page: "memory", core: true },
  { name: "VoI gate", note: "P(flip) × |ΔV| > θ × tokenCost", page: "optimize", core: true },
  { name: "Validation agent", note: "1–5 ratings, pairwise", page: "optimize" },
];

export default function Console({ health, setHealth, go }) {
  const [wls, setWls] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api.workloads().then(setWls).catch((e) => setErr(e.message)); }, [health.backend]);

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Optimizer console</h2>
          <p>PlannerLLM rewrites RAG pipelines (retrieve, filter, rerank, generate, text-to-SQL) and decides,
             candidate by candidate, whether an LLM validation is worth its token cost.</p>
        </div>
        <dl className="kv">
          <dt>Model</dt><dd>{health.model}</dd>
          <dt>Provider</dt><dd>{health.backend === "groq" ? "Groq API" : "offline test mode"}</dd>
        </dl>
      </div>

      {err && <div className="notice error">{err}</div>}
      {!health.has_key && (
        <div className="notice" style={{ marginBottom: 16 }}>
          No Groq API key is configured. Add <code>GROQ_API_KEY</code> to <code>.env</code> and restart the server.
        </div>
      )}

      <section className="panel" style={{ marginBottom: 16 }}>
        <h3>How a query travels through the system</h3>
        <div className="flow">
          {FLOW.map((f) => (
            <button key={f.name} className={`node ${f.core ? "core" : ""}`} onClick={() => go(f.page)}>
              <b>{f.name}</b><span>{f.note}</span>
            </button>
          ))}
        </div>
        <p className="small muted" style={{ margin: "10px 0 0" }}>
          Outlined in blue: the three modules this work adds to the DocETL-style agent loop. The chosen plan then runs on the
          execution engine (DuckDB + FAISS + SQLite), and its telemetry flows back into the plan memory and the cost model.
        </p>
      </section>

      <div className="grid cols-3">
        {(wls || []).map((w) => (
          <section key={w.name} className="panel stack">
            <div>
              <h3 style={{ marginBottom: 2 }}>{WL_LABEL[w.name] || w.name} <small>{w.family}</small></h3>
              <p className="small muted" style={{ margin: 0 }}>{w.task}</p>
            </div>
            <dl className="kv">
              <dt>Records</dt><dd>{fmt.int(w.n_records)} ({fmt.n(w.avg_doc_tokens, 0)} tokens avg.)</dd>
              <dt>Questions</dt><dd>{fmt.int(w.n_queries)} in {Object.keys(w.groups).length} groups</dd>
              <dt>Metric</dt><dd>{({ rouge_l: "Rouge-L", token_f1: "token F1", execution_accuracy: "execution accuracy" })[w.metric] || w.metric}</dd>
              <dt>ANN probe</dt><dd>HNSW {fmt.n(w.calibration.ann_ms.hnsw, 3)} ms, IVF {fmt.n(w.calibration.ann_ms.ivf, 3)} ms, flat {fmt.n(w.calibration.ann_ms.flat, 3)} ms</dd>
              <dt>Source</dt><dd className="small">{w.source}</dd>
            </dl>
            <div>
              <div className="small muted" style={{ marginBottom: 4 }}>Starting pipeline</div>
              <PlanDiagram ops={w.base_ops} />
            </div>
          </section>
        ))}
        {!wls && !err && <p className="muted">Loading workloads and building indexes (first load can take ~30 s)…</p>}
      </div>
    </>
  );
}
