// EXPLAIN-style vertical plan: one node per operator, LLM operators filled,
// operators that differ from a reference plan marked as changed.
const KIND_NOTE = {
  retrieve: "ANN probe", filter: "SQL predicate (DuckDB)", rerank: "cross-encoder", map: "LLM per record",
  generate: "LLM answer", sql_generate: "LLM text-to-SQL", sql_repair: "execute + LLM fix", sql_execute: "SQLite",
};
const LLM = new Set(["map", "generate", "sql_generate", "sql_repair"]);

function params(op) {
  const p = op.params || {};
  const bits = [];
  if (p.top_k != null) bits.push(`k = ${p.top_k}`);
  if (p.index) bits.push(p.index.toUpperCase());
  if (p.where) bits.push(op.kind === "retrieve" ? `probe filtered by ${p.where}` : `where ${p.where}`);
  if (p.fewshot_k) bits.push(`${p.fewshot_k} demonstrations`);
  if (p.max_docs) bits.push(`≤ ${p.max_docs} docs in context`);
  return bits.join(", ");
}

export function PlanDiagram({ ops = [], compareTo, costs }) {
  const ref = new Map((compareTo || []).map((o) => [o.name, JSON.stringify(o)]));
  return (
    <div className="plan">
      {ops.map((op) => {
        const changed = compareTo && ref.get(op.name) !== JSON.stringify(op);
        const c = costs?.find((x) => x.op === op.name);
        return (
          <div key={op.name} className={`op ${LLM.has(op.kind) ? "llm" : ""} ${changed ? "changed" : ""}`}>
            <div className="spine"><span className="dot" /></div>
            <div className="body">
              <div className="title">
                <b>{op.kind}</b><code>{op.name}</code>
                <span className="small muted">{KIND_NOTE[op.kind]}</span>
                {changed && <span className="tag validate">changed</span>}
              </div>
              {params(op) && <div className="params">{params(op)}</div>}
              {op.prompt && <div className="params muted" title={op.prompt}>“{op.prompt.length > 110 ? op.prompt.slice(0, 110) + "…" : op.prompt}”</div>}
              {c && (
                <div className="cost">
                  est. {c.rows_out != null ? `${(+c.rows_out).toFixed(1)} rows out, ` : ""}{c.ms.toFixed(1)} ms
                  {c.tokens > 0 ? `, ${Math.round(c.tokens)} tokens` : ""}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
