// Small SVG chart kit: grouped bars, lines and a scatter, each with hover tooltips.
// One y-axis per chart; colours come from CSS variables so dark mode follows.
import { useMemo, useRef, useState } from "react";

const W = 640;
const PAD = { l: 52, r: 12, t: 10, b: 34 };

function ticks(min, max, n = 4) {
  if (max === min) return [min];
  const step = niceStep((max - min) / n);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-12; v += step) out.push(+v.toFixed(10));
  return out;
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}
const short = (v) => {
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  if (a > 0 && a < 0.01) return v.toExponential(0);
  return `${+v.toFixed(2)}`;
};

function useTip() {
  const [tip, setTip] = useState(null);
  const ref = useRef(null);
  const show = (evt, html) => {
    const box = ref.current.getBoundingClientRect();
    setTip({ x: evt.clientX - box.left, y: evt.clientY - box.top, html });
  };
  return { ref, tip, show, hide: () => setTip(null) };
}

export function Legend({ items }) {
  if (!items || items.length < 2) return null;
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.key}><i style={{ background: it.color }} />{it.label}</span>
      ))}
    </div>
  );
}

/** groups: [{key,label}], series: [{key,label,color}], value(groupKey, seriesKey) -> number|null */
export function GroupedBars({ groups, series, value, height = 220, format = short, yLabel }) {
  const t = useTip();
  const H = height;
  const vals = groups.flatMap((g) => series.map((s) => value(g.key, s.key) ?? 0));
  const max = Math.max(1e-9, ...vals) * 1.08;
  const ys = ticks(0, max);
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
  const gw = iw / groups.length;
  const bw = Math.min(28, (gw * 0.78) / series.length);
  const y = (v) => PAD.t + ih - (v / max) * ih;
  return (
    <div className="chart" ref={t.ref}>
      <Legend items={series} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={yLabel || "bar chart"}>
        <g className="grid">{ys.map((v) => <line key={v} x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />)}</g>
        <g className="axis">
          {ys.map((v) => <text key={v} x={PAD.l - 6} y={y(v) + 4} textAnchor="end">{format(v)}</text>)}
          {groups.map((g, i) => <text key={g.key} x={PAD.l + gw * i + gw / 2} y={H - 10} textAnchor="middle">{g.label}</text>)}
          <line x1={PAD.l} x2={W - PAD.r} y1={PAD.t + ih} y2={PAD.t + ih} />
          {yLabel && <text x={12} y={PAD.t + ih / 2} transform={`rotate(-90 12 ${PAD.t + ih / 2})`} textAnchor="middle">{yLabel}</text>}
        </g>
        {groups.map((g, gi) => {
          const x0 = PAD.l + gw * gi + gw / 2 - (bw * series.length) / 2;
          return series.map((s, si) => {
            const v = value(g.key, s.key);
            if (v == null) return null;
            const top = y(Math.max(v, 0));
            const h = Math.max(1, PAD.t + ih - top);
            const x = x0 + si * bw;
            return (
              <g key={g.key + s.key}>
                <path d={roundedTop(x + 1, top, bw - 2, h, 3)} fill={s.color}
                      onMouseMove={(e) => t.show(e, `${g.label}, ${s.label}: ${format(v)}`)} onMouseLeave={t.hide} />
                {series.length <= 5 && <text className="lbl" x={x + bw / 2} y={top - 4} textAnchor="middle">{format(v)}</text>}
              </g>
            );
          });
        })}
      </svg>
      {t.tip && <div className="tip" style={{ left: t.tip.x, top: t.tip.y }}>{t.tip.html}</div>}
    </div>
  );
}

function roundedTop(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h} Z`;
}

/** series: [{key,label,color,points:[{x,y,label?}]}] */
export function Lines({ series, height = 220, xLabel, yLabel, logX = false, yMin = null, yMax = null, format = short, xFormat = short }) {
  const t = useTip();
  const H = height;
  const pts = series.flatMap((s) => s.points);
  const tx = (v) => (logX ? Math.log2(Math.max(v, 1e-9)) : v);
  const xs = pts.map((p) => tx(p.x)), yv = pts.map((p) => p.y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const lo = yMin ?? Math.min(0, ...yv), hi = yMax ?? Math.max(...yv) * 1.08;
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
  const X = (v) => PAD.l + (x1 === x0 ? iw / 2 : ((tx(v) - x0) / (x1 - x0)) * iw);
  const Y = (v) => PAD.t + ih - ((v - lo) / (hi - lo || 1)) * ih;
  const xt = logX ? [...new Set(pts.map((p) => p.x))].sort((a, b) => a - b) : ticks(Math.min(...pts.map((p) => p.x)), Math.max(...pts.map((p) => p.x)));
  if (!pts.length) return <p className="muted small">No data yet.</p>;
  return (
    <div className="chart" ref={t.ref}>
      <Legend items={series} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={yLabel || "line chart"}>
        <g className="grid">{ticks(lo, hi).map((v) => <line key={v} x1={PAD.l} x2={W - PAD.r} y1={Y(v)} y2={Y(v)} />)}</g>
        <g className="axis">
          {ticks(lo, hi).map((v) => <text key={v} x={PAD.l - 6} y={Y(v) + 4} textAnchor="end">{format(v)}</text>)}
          {xt.map((v) => <text key={v} x={X(v)} y={H - 16} textAnchor="middle">{xFormat(v)}</text>)}
          {xLabel && <text x={PAD.l + iw / 2} y={H - 1} textAnchor="middle">{xLabel}</text>}
          {yLabel && <text x={12} y={PAD.t + ih / 2} transform={`rotate(-90 12 ${PAD.t + ih / 2})`} textAnchor="middle">{yLabel}</text>}
        </g>
        {series.map((s) => {
          const ps = [...s.points].sort((a, b) => a.x - b.x);
          return (
            <g key={s.key}>
              <path d={ps.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y)}`).join(" ")} fill="none" stroke={s.color} strokeWidth="2" />
              {ps.map((p, i) => (
                <circle key={i} cx={X(p.x)} cy={Y(p.y)} r="4.5" fill={s.color} stroke="var(--surface)" strokeWidth="2"
                        onMouseMove={(e) => t.show(e, `${s.label}: ${p.label || `${xFormat(p.x)}: ${format(p.y)}`}`)} onMouseLeave={t.hide} />
              ))}
            </g>
          );
        })}
      </svg>
      {t.tip && <div className="tip" style={{ left: t.tip.x, top: t.tip.y }}>{t.tip.html}</div>}
    </div>
  );
}

/** points: [{x,y,color,label}], log axes, optional y=x diagonal */
export function Scatter({ points, height = 260, xLabel, yLabel, diagonal = true, legend }) {
  const t = useTip();
  const H = height;
  const L = (v) => Math.log10(Math.max(v, 1e-12));
  const all = points.flatMap((p) => [L(p.x), L(p.y)]);
  const lo = Math.floor(Math.min(...all)), hi = Math.ceil(Math.max(...all));
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
  const X = (v) => PAD.l + ((L(v) - lo) / (hi - lo || 1)) * iw;
  const Y = (v) => PAD.t + ih - ((L(v) - lo) / (hi - lo || 1)) * ih;
  const decades = useMemo(() => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i), [lo, hi]);
  if (!points.length) return <p className="muted small">No data yet.</p>;
  return (
    <div className="chart" ref={t.ref}>
      <Legend items={legend} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="scatter">
        <g className="grid">{decades.map((d) => <line key={d} x1={PAD.l} x2={W - PAD.r} y1={Y(10 ** d)} y2={Y(10 ** d)} />)}</g>
        <g className="axis">
          {decades.map((d) => <text key={`y${d}`} x={PAD.l - 6} y={Y(10 ** d) + 4} textAnchor="end">1e{d}</text>)}
          {decades.map((d) => <text key={`x${d}`} x={X(10 ** d)} y={H - 16} textAnchor="middle">1e{d}</text>)}
          {xLabel && <text x={PAD.l + iw / 2} y={H - 1} textAnchor="middle">{xLabel}</text>}
          {yLabel && <text x={12} y={PAD.t + ih / 2} transform={`rotate(-90 12 ${PAD.t + ih / 2})`} textAnchor="middle">{yLabel}</text>}
        </g>
        {diagonal && <line x1={X(10 ** lo)} y1={Y(10 ** lo)} x2={X(10 ** hi)} y2={Y(10 ** hi)} stroke="var(--muted)" strokeDasharray="4 4" />}
        {points.map((p, i) => (
          <circle key={i} cx={X(p.x)} cy={Y(p.y)} r="5" fill={p.color} stroke="var(--surface)" strokeWidth="1.5" opacity="0.9"
                  onMouseMove={(e) => t.show(e, p.label)} onMouseLeave={t.hide} />
        ))}
      </svg>
      {t.tip && <div className="tip" style={{ left: t.tip.x, top: t.tip.y }}>{t.tip.html}</div>}
    </div>
  );
}
