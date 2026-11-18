// The decision ribbon: every shortlisted candidate as one row moving through the
// four stages of Algorithm 1 (cost rank, plan memory, VoI gate, validation).
import { fmt } from "../api";

export function buildRows(events) {
  const rounds = [];
  let cur = null;
  const byKey = new Map();
  for (const ev of events) {
    if (ev.type === "candidates") {
      cur = { round: ev.round, proposal: ev.proposal || [], items: ev.items, rows: [] };
      rounds.push(cur);
      for (const it of ev.items.filter((i) => i.shortlisted)) {
        const row = { ...it, stage: "ranked" };
        cur.rows.push(row);
        byKey.set(`${ev.round}|${it.directive}|${it.plan}`, row);
      }
    }
    const row = byKey.get(`${ev.round}|${ev.directive}|${ev.plan}`);
    if (!row) continue;
    if (ev.type === "memory_hit") Object.assign(row, { stage: "memory", memory: ev });
    if (ev.type === "gate") Object.assign(row, { stage: ev.validate || !ev.enforced ? "gate-pass" : "gate-skip", gate: ev });
    if (ev.type === "validating") Object.assign(row, { stage: "validating" });
    if (ev.type === "validation") Object.assign(row, { stage: "validated", validation: ev });
    if (ev.type === "incumbent") row.won = true;
  }
  return rounds;
}

function GateCell({ gate }) {
  if (!gate) return <span className="muted">not reached</span>;
  const scale = Math.max(gate.voi, gate.threshold) * 1.25 || 1;
  return (
    <>
      <span className={`tag ${gate.validate ? "validate" : "skip"}`}>{gate.validate ? "validate" : "skip"}</span>
      <div className="voi" title={gate.rationale}>
        <div className="voi-line">
          <span className="v" style={{ width: `${(gate.voi / scale) * 100}%`, background: gate.validate ? "var(--validate)" : "var(--skip)" }} />
          <span className="t" style={{ left: `${(gate.threshold / scale) * 100}%` }} />
        </div>
        <span className="small muted">P(flip) {fmt.n(gate.p_flip)}, VoI {fmt.usd(gate.voi)} vs {fmt.usd(gate.threshold)}</span>
      </div>
    </>
  );
}

export function Ribbon({ events }) {
  const rounds = buildRows(events);
  if (!rounds.length) return <p className="muted">Start a run to watch the optimizer decide, candidate by candidate.</p>;
  return (
    <div className="ribbon">
      {rounds.map((r) => (
        <div key={r.round} className="stack">
          <div className="round-head">
            Round {r.round + 1}: {r.items.length} candidates generated, {r.rows.length} shortlisted by cost and predicted gain
            {r.proposal.length > 0 && <>; the agent nominated {r.proposal.map((p) => p.name).join(", ")}</>}
          </div>
          {r.rows.map((row) => {
            const v = row.validation;
            const mem = row.memory;
            const final = mem ? mem.prefer : v ? v.prefer_candidate : null;
            return (
              <div key={row.directive + row.plan} className="cand enter">
                <div>
                  <div className="dname">{row.directive}</div>
                  <div className="small muted">{row.detail}</div>
                </div>
                <div className="stage">
                  <span className="stage-label">Cost model</span>
                  {fmt.usd(row.cost_usd)} per run
                  <div className="small muted">gain prior {fmt.n(row.pred_gain, 3)}, score {fmt.n(row.score, 4)}</div>
                </div>
                <div className={`stage ${mem ? "hit" : "off"}`}>
                  <span className="stage-label">Plan memory</span>
                  {mem ? (<><span className="tag memory">reused</span>
                    <div className="small">{mem.match} hit, confidence {fmt.n(mem.confidence)}</div></>)
                    : <span className="small">no reusable verdict</span>}
                </div>
                <div className={`stage ${row.gate ? (row.stage === "gate-skip" ? "skip" : "go") : "off"}`}>
                  <span className="stage-label">VoI gate</span>
                  <GateCell gate={row.gate} />
                </div>
                <div className={`stage ${final == null ? "off" : final ? "win" : "lose"}`}>
                  <span className="stage-label">Validation agent</span>
                  {row.stage === "validating" && <span><span className="pulse" /> rating sample outputs</span>}
                  {v && (<>
                    <span className={`tag ${v.prefer_candidate ? "prefer" : "reject"}`}>{v.prefer_candidate ? "prefer" : "reject"}</span>
                    <div className="small">rating {fmt.n(v.score_c)} vs {fmt.n(v.score_p)}
                      {v.pairwise_win_rate != null && `, pairwise ${fmt.pct(v.pairwise_win_rate)}`}</div>
                    <div className="small muted">{fmt.int(v.tokens)} tokens</div>
                  </>)}
                  {mem && <span className={`tag ${mem.prefer ? "prefer" : "reject"}`}>{mem.prefer ? "prefer" : "reject"} (no LLM call)</span>}
                  {!v && !mem && row.stage === "gate-skip" && <span className="small muted">skipped, incumbent kept</span>}
                  {row.won && <div className="small" style={{ color: "var(--prefer)", fontWeight: 600 }}>became the incumbent</div>}
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
