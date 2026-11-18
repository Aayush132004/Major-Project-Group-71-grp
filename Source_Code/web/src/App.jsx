import { useEffect, useState } from "react";
import { api } from "./api";
import Console from "./pages/Console";
import Studio from "./pages/Studio";
import Optimize from "./pages/Optimize";
import Memory from "./pages/Memory";
import Results from "./pages/Results";
import Report from "./pages/Report";

const PAGES = [
  { id: "console", label: "Console", hint: "system and datasets", el: Console },
  { id: "studio", label: "Pipeline studio", hint: "edit, cost and run a plan", el: Studio },
  { id: "optimize", label: "Optimize", hint: "watch Algorithm 1 live", el: Optimize },
  { id: "memory", label: "Plan memory", hint: "what the optimizer learned", el: Memory },
  { id: "results", label: "Experiments", hint: "Section 10 results", el: Results },
  { id: "report", label: "Report assets", hint: "figures and tables", el: Report },
];

export default function App() {
  const pageOf = () => location.hash.slice(1).split("?")[0] || "console";
  const [page, setPage] = useState(pageOf);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "auto");

  useEffect(() => {
    const onHash = () => setPage(pageOf());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => { api.health().then(setHealth).catch((e) => setError(e.message)); }, []);
  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch { /* private mode */ }
  }, [theme]);

  const go = (id) => { location.hash = id; };
  const Page = (PAGES.find((p) => p.id === page) || PAGES[0]).el;
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <div className="brand-mark" aria-hidden>
            <span style={{ background: "var(--primary)" }} /><span style={{ background: "var(--validate)" }} />
            <span style={{ background: "var(--memory)" }} /><span style={{ background: "var(--skip)" }} />
          </div>
          <div><h1>PlannerLLM</h1><p>cost-gated query rewriting</p></div>
        </div>
        <nav className="nav">
          {PAGES.map((p) => (
            <button key={p.id} aria-current={page === p.id ? "page" : undefined} onClick={() => go(p.id)}>
              {p.label}<small>{p.hint}</small>
            </button>
          ))}
        </nav>
        <div className="side-foot">
          {health && (
            <span>LLM <b>{health.model}</b>{health.backend === "groq" ? " on Groq" : ""}</span>
          )}
          <div className="seg" role="group" aria-label="Theme">
            {["auto", "light", "dark"].map((t) => (
              <button key={t} aria-pressed={theme === t} onClick={() => setTheme(t)}>{t}</button>
            ))}
          </div>
        </div>
      </aside>
      <main className="main">
        {error && <div className="notice error">The API is not reachable ({error}). Start it with <code>python scripts/serve.py</code>.</div>}
        {health && <Page health={health} setHealth={setHealth} go={go} />}
      </main>
    </div>
  );
}
