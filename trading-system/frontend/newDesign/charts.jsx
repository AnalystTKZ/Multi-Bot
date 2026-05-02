(function(){
// charts.jsx — small SVG chart primitives (no external chart lib)

const { useMemo } = React;

// Sparkline — tiny line chart for KPI cards
function Sparkline({ data, color = "currentColor", w = 120, h = 32, fill = false }) {
  const path = useMemo(() => {
    if (!data || data.length < 2) return { line: "", area: "" };
    const min = Math.min(...data);
    const max = Math.max(...data);
    const r = max - min || 1;
    const stepX = w / (data.length - 1);
    const points = data.map((v, i) => [i * stepX, h - ((v - min) / r) * h]);
    const line = points.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
    const area = line + ` L${w},${h} L0,${h} Z`;
    return { line, area };
  }, [data, w, h]);

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      {fill && <path d={path.area} fill={color} fillOpacity="0.12" />}
      <path d={path.line} stroke={color} strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Equity curve — bigger area chart with gradient
function EquityChart({ data, height = 280, color = "var(--gain)" }) {
  const w = 1000; // viewBox width — scaled via CSS
  const h = height;
  const padL = 44, padR = 12, padT = 16, padB = 28;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const values = data.map((d) => d.value);
  const min = Math.min(...values) * 0.995;
  const max = Math.max(...values) * 1.003;
  const r = max - min || 1;

  const sx = (i) => padL + (i / (data.length - 1)) * innerW;
  const sy = (v) => padT + innerH - ((v - min) / r) * innerH;

  const linePath = data.map((d, i) => (i === 0 ? "M" : "L") + sx(i).toFixed(1) + "," + sy(d.value).toFixed(1)).join(" ");
  const areaPath = linePath + ` L${(padL + innerW).toFixed(1)},${(padT + innerH).toFixed(1)} L${padL.toFixed(1)},${(padT + innerH).toFixed(1)} Z`;

  // Y gridlines (4)
  const yTicks = [];
  for (let i = 0; i <= 4; i++) {
    const v = min + (r * i) / 4;
    yTicks.push({ y: sy(v), v });
  }
  // X labels every ~ 12 points
  const xTicks = [];
  for (let i = 0; i < data.length; i += Math.floor(data.length / 6)) {
    xTicks.push({ x: sx(i), d: data[i].date });
  }

  const gid = "eqgrad-" + Math.random().toString(36).slice(2, 8);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
      <defs>
        <linearGradient id={gid} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.32" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={padL} x2={w - padR} y1={t.y} y2={t.y} stroke="var(--line-soft)" strokeWidth="1" strokeDasharray="2 4" />
          <text x={padL - 8} y={t.y + 4} fontSize="10" fill="var(--text-2)" textAnchor="end" fontFamily="var(--mono)">
            {window.DB.fmtCompact(t.v)}
          </text>
        </g>
      ))}
      {xTicks.map((t, i) => (
        <text key={i} x={t.x} y={h - 8} fontSize="10" fill="var(--text-2)" textAnchor="middle" fontFamily="var(--mono)">
          {t.d.toLocaleDateString("en-US", { month: "short", day: "numeric" })}
        </text>
      ))}
      <path d={areaPath} fill={`url(#${gid})`} />
      <path d={linePath} stroke={color} strokeWidth="1.6" fill="none" />
      {/* end-of-line dot */}
      <circle cx={sx(data.length - 1)} cy={sy(data[data.length - 1].value)} r="3" fill={color} />
      <circle cx={sx(data.length - 1)} cy={sy(data[data.length - 1].value)} r="6" fill={color} fillOpacity="0.2" />
    </svg>
  );
}

// Bar chart — horizontal label, vertical bars (monthly returns)
function BarChart({ data, height = 200, valueKey = "ret", labelKey = "month" }) {
  const w = 1000;
  const h = height;
  const padL = 36, padR = 8, padT = 16, padB = 24;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const values = data.map((d) => d[valueKey]);
  const max = Math.max(...values, 0);
  const min = Math.min(...values, 0);
  const r = max - min || 1;
  const zero = padT + (max / r) * innerH;
  const bw = innerW / data.length;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
      <line x1={padL} x2={w - padR} y1={zero} y2={zero} stroke="var(--line)" strokeWidth="1" />
      {data.map((d, i) => {
        const v = d[valueKey];
        const barH = Math.abs((v / r) * innerH);
        const x = padL + i * bw + bw * 0.15;
        const y = v >= 0 ? zero - barH : zero;
        const color = v >= 0 ? "var(--gain)" : "var(--loss)";
        return (
          <g key={i}>
            <rect x={x} y={y} width={bw * 0.7} height={barH} fill={color} fillOpacity="0.85" rx="1" />
            <text x={x + bw * 0.35} y={h - 6} fontSize="10" fill="var(--text-2)" textAnchor="middle" fontFamily="var(--mono)">
              {d[labelKey]}
            </text>
            <text
              x={x + bw * 0.35}
              y={v >= 0 ? y - 4 : y + barH + 11}
              fontSize="9.5"
              fill={color}
              textAnchor="middle"
              fontFamily="var(--mono)"
            >
              {(v >= 0 ? "+" : "") + v.toFixed(1)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// Donut — composition chart
function Donut({ data, size = 140, thickness = 22 }) {
  const total = data.reduce((s, d) => s + d.value, 0);
  const r = size / 2 - thickness / 2;
  const c = size / 2;
  const circ = 2 * Math.PI * r;
  let offset = 0;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={c} cy={c} r={r} stroke="var(--bg-2)" strokeWidth={thickness} fill="none" />
      {data.map((d, i) => {
        const len = (d.value / total) * circ;
        const dasharray = `${len} ${circ}`;
        const el = (
          <circle
            key={i}
            cx={c} cy={c} r={r}
            stroke={d.color}
            strokeWidth={thickness}
            fill="none"
            strokeDasharray={dasharray}
            strokeDashoffset={-offset}
            transform={`rotate(-90 ${c} ${c})`}
            strokeLinecap="butt"
          />
        );
        offset += len;
        return el;
      })}
    </svg>
  );
}

// Mini orderbook-style price column chart
function PriceColumns({ data, height = 80 }) {
  const max = Math.max(...data.map(d => Math.abs(d)));
  const w = 100;
  const bw = w / data.length;
  return (
    <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
      {data.map((v, i) => {
        const h = (Math.abs(v) / max) * height * 0.9;
        const color = v >= 0 ? "var(--gain)" : "var(--loss)";
        return <rect key={i} x={i * bw + bw * 0.1} y={height - h} width={bw * 0.8} height={h} fill={color} fillOpacity="0.85" />;
      })}
    </svg>
  );
}

// Heatmap — grid of returns
function Heatmap({ data, cols = 7, rows = 8 }) {
  // data: array of values (return %)
  const max = Math.max(...data.map(Math.abs));
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: `repeat(${cols}, 1fr)`,
      gap: 3,
    }}>
      {data.slice(0, cols * rows).map((v, i) => {
        const norm = max ? Math.abs(v) / max : 0;
        const bg = v >= 0
          ? `oklch(0.78 0.15 155 / ${0.15 + norm * 0.6})`
          : `oklch(0.68 0.19 25 / ${0.15 + norm * 0.6})`;
        return (
          <div key={i} className="heat" style={{ background: bg }}>
            {v.toFixed(1)}
          </div>
        );
      })}
    </div>
  );
}

// Multi-line chart — for ML loss curves etc
function LineChart({ series, height = 200, yLabel = "" }) {
  // series: [{ name, color, data: number[] }]
  const w = 1000;
  const h = height;
  const padL = 40, padR = 12, padT = 12, padB = 26;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const allValues = series.flatMap(s => s.data);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const r = max - min || 1;
  const len = Math.max(...series.map(s => s.data.length));
  const sx = (i) => padL + (i / (len - 1)) * innerW;
  const sy = (v) => padT + innerH - ((v - min) / r) * innerH;

  const yTicks = [];
  for (let i = 0; i <= 3; i++) {
    yTicks.push({ y: padT + (innerH * i) / 3, v: max - (r * i) / 3 });
  }
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={padL} x2={w - padR} y1={t.y} y2={t.y} stroke="var(--line-soft)" strokeDasharray="2 4" />
          <text x={padL - 6} y={t.y + 3} fontSize="10" fill="var(--text-2)" textAnchor="end" fontFamily="var(--mono)">
            {t.v.toFixed(t.v < 1 ? 3 : 1)}
          </text>
        </g>
      ))}
      {series.map((s, si) => {
        const path = s.data.map((v, i) => (i === 0 ? "M" : "L") + sx(i).toFixed(1) + "," + sy(v).toFixed(1)).join(" ");
        return <path key={si} d={path} stroke={s.color} strokeWidth="1.6" fill="none" />;
      })}
    </svg>
  );
}

// Neural network diagram — for ML page model viz
function NetworkDiagram({ layers, color = "var(--info)" }) {
  // layers: [{ size: number, label: string }]
  const w = 720;
  const h = 280;
  const padX = 60;
  const stepX = (w - padX * 2) / (layers.length - 1);

  const nodes = [];
  layers.forEach((l, li) => {
    const visible = Math.min(l.size, 6);
    const yStart = h / 2 - (visible - 1) * 22;
    for (let n = 0; n < visible; n++) {
      nodes.push({ x: padX + li * stepX, y: yStart + n * 44, layer: li, idx: n });
    }
  });

  // edges between adjacent layers (sample)
  const edges = [];
  for (let li = 0; li < layers.length - 1; li++) {
    const left = nodes.filter(n => n.layer === li);
    const right = nodes.filter(n => n.layer === li + 1);
    left.forEach(a => right.forEach(b => edges.push({ a, b, w: 0.2 + Math.random() * 0.7 })));
  }

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {edges.map((e, i) => (
        <line key={i} x1={e.a.x} y1={e.a.y} x2={e.b.x} y2={e.b.y} stroke={color} strokeOpacity={e.w * 0.45} strokeWidth="1" />
      ))}
      {nodes.map((n, i) => (
        <circle key={i} cx={n.x} cy={n.y} r="9" fill="var(--bg-1)" stroke={color} strokeWidth="1.4" />
      ))}
      {layers.map((l, li) => (
        <g key={li}>
          <text x={padX + li * stepX} y={h - 14} fontSize="11" fill="var(--text-1)" textAnchor="middle" fontFamily="var(--sans)">
            {l.label}
          </text>
          <text x={padX + li * stepX} y={h - 0} fontSize="9.5" fill="var(--text-3)" textAnchor="middle" fontFamily="var(--mono)">
            {l.size}{l.size > 6 ? " (truncated)" : ""}
          </text>
        </g>
      ))}
    </svg>
  );
}

// Attention heatmap — colored grid for transformer/attention viz
function AttentionGrid({ rows, cols, data, rowLabels = [], colLabels = [], color = "var(--info)" }) {
  const cw = 26, ch = 22, padL = 80, padT = 28;
  const w = padL + cols * cw + 12;
  const h = padT + rows * ch + 12;
  const max = Math.max(...data.flat());
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {colLabels.map((l, i) => (
        <text key={i} x={padL + i * cw + cw / 2} y={padT - 8} fontSize="10" fill="var(--text-2)" textAnchor="middle" fontFamily="var(--mono)">{l}</text>
      ))}
      {rowLabels.map((l, i) => (
        <text key={i} x={padL - 8} y={padT + i * ch + ch / 2 + 3} fontSize="10" fill="var(--text-2)" textAnchor="end" fontFamily="var(--mono)">{l}</text>
      ))}
      {data.map((row, ri) => row.map((v, ci) => {
        const op = max ? v / max : 0;
        return (
          <rect
            key={`${ri}-${ci}`}
            x={padL + ci * cw + 1}
            y={padT + ri * ch + 1}
            width={cw - 2} height={ch - 2}
            fill={color} fillOpacity={0.08 + op * 0.85} rx="2"
          />
        );
      }))}
    </svg>
  );
}

// Confusion matrix
function ConfusionMatrix({ matrix, labels }) {
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  const max = Math.max(...matrix.flat());
  return (
    <div style={{ display: "grid", gridTemplateColumns: `auto repeat(${labels.length}, 1fr)`, gap: 4, fontFamily: "var(--mono)", fontSize: 11 }}>
      <div></div>
      {labels.map(l => <div key={l} style={{ textAlign: "center", color: "var(--text-2)", fontSize: 10 }}>{l}</div>)}
      {matrix.map((row, ri) => (
        <React.Fragment key={ri}>
          <div style={{ color: "var(--text-2)", fontSize: 10, alignSelf: "center", textAlign: "right", paddingRight: 6 }}>{labels[ri]}</div>
          {row.map((v, ci) => {
            const isDiag = ri === ci;
            const norm = max ? v / max : 0;
            const color = isDiag ? "var(--gain)" : "var(--loss)";
            return (
              <div key={ci} style={{
                background: isDiag ? `oklch(0.78 0.15 155 / ${0.1 + norm * 0.6})` : `oklch(0.68 0.19 25 / ${0.05 + norm * 0.5})`,
                color: "var(--text-0)",
                padding: "10px 4px",
                textAlign: "center",
                borderRadius: 3,
                fontWeight: isDiag ? 600 : 400,
              }}>
                {((v / total) * 100).toFixed(1)}%
              </div>
            );
          })}
        </React.Fragment>
      ))}
    </div>
  );
}

// Feature importance — horizontal bar chart
function FeatureBars({ items, color = "var(--info)" }) {
  const max = Math.max(...items.map(i => i.value));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {items.map((it, i) => (
        <div key={i} style={{ display: "grid", gridTemplateColumns: "120px 1fr 50px", gap: 10, alignItems: "center", fontSize: 12 }}>
          <div style={{ color: "var(--text-1)", fontFamily: "var(--mono)", fontSize: 11 }}>{it.label}</div>
          <div style={{ height: 14, background: "var(--bg-2)", borderRadius: 2, position: "relative", overflow: "hidden" }}>
            <div style={{ width: `${(it.value / max) * 100}%`, height: "100%", background: color, opacity: 0.85, borderRadius: 2 }} />
          </div>
          <div style={{ fontFamily: "var(--mono)", color: "var(--text-1)", fontSize: 11, textAlign: "right" }}>{it.value.toFixed(3)}</div>
        </div>
      ))}
    </div>
  );
}

window.Charts = { Sparkline, EquityChart, BarChart, Donut, PriceColumns, Heatmap, LineChart, NetworkDiagram, AttentionGrid, ConfusionMatrix, FeatureBars };

})();