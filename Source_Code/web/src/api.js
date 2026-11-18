// Thin wrapper over the FastAPI endpoints.
async function call(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* plain text error */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const api = {
  health: () => call("GET", "/api/health"),
  config: (body) => call("POST", "/api/config", body),
  workloads: () => call("GET", "/api/workloads"),
  tasks: (wl) => call("GET", `/api/workloads/${wl}/tasks?n=12`),
  directives: () => call("GET", "/api/directives"),
  cost: (body) => call("POST", "/api/cost", body),
  execute: (body) => call("POST", "/api/execute", body),
  optimize: (body) => call("POST", "/api/optimize", body),
  run: (id) => call("GET", `/api/runs/${id}`),
  runs: () => call("GET", "/api/runs"),
  memory: (wl) => call("GET", `/api/memory/${wl}`),
  resetMemory: (wl) => call("DELETE", `/api/memory/${wl}`),
  experiments: () => call("GET", "/api/experiments"),
  experiment: (backend, name) => call("GET", `/api/experiments/${backend}/${name}`),
  runExperiments: (body) => call("POST", "/api/experiments/run", body),
  job: (id) => call("GET", `/api/jobs/${id}`),
  reportFiles: (backend) => call("GET", `/api/report/${backend}`),
};

export function stream(runId, onEvent) {
  const es = new EventSource(`/api/runs/${runId}/stream`);
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    onEvent(ev);
    if (ev.type === "closed") es.close();
  };
  es.onerror = () => es.close();
  return () => es.close();
}

export const WL_LABEL = { msmarco: "MS MARCO", cuad: "CUAD", spider: "Spider", bird: "BIRD" };
export const SYS_LABEL = {
  plannerllm: "PlannerLLM", docetl: "DocETL-style", cost_only: "Cost-only", random: "Random",
  no_opt: "No optimisation", no_gate: "without VoI gate", no_memory: "without plan memory",
  symbolic_only: "symbolic cost only", neural_only: "neural cost only",
};
export const SYS_COLOR = {
  plannerllm: "var(--s1)", docetl: "var(--s2)", cost_only: "var(--s3)", random: "var(--s4)",
  no_opt: "var(--s5)", no_gate: "var(--s7)", no_memory: "var(--s6)", symbolic_only: "var(--s8)", neural_only: "var(--s4)",
};

export const fmt = {
  n: (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d)),
  int: (v) => (v == null ? "–" : Math.round(v).toLocaleString()),
  usd: (v) => (v == null ? "–" : v < 0.01 ? `$${Number(v).toExponential(2)}` : `$${Number(v).toFixed(4)}`),
  pct: (v) => (v == null ? "–" : `${(v * 100).toFixed(0)}%`),
  ms: (v) => (v == null ? "–" : v > 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(1)} ms`),
};
