(function(){
// pages.jsx — all page content for Dashbot

const fmtUSD = window.DB.fmtUSD, fmtPct = window.DB.fmtPct, fmtNum = window.DB.fmtNum, fmtCompact = window.DB.fmtCompact, timeAgo = window.DB.timeAgo, mulberry32 = window.DB.mulberry32;
const KPI = window.DB.KPI, ML_TRADER = window.DB.ML_TRADER, POSITIONS = window.DB.POSITIONS, SIGNALS = window.DB.SIGNALS, TRADES = window.DB.TRADES, ALERTS = window.DB.ALERTS, EQUITY = window.DB.EQUITY, MONTHLY = window.DB.MONTHLY, SYM_BREAKDOWN = window.DB.SYM_BREAKDOWN, ML_MODELS = window.DB.ML_MODELS, RL_AGENT = window.DB.RL_AGENT, BACKTESTS = window.DB.BACKTESTS, RISK = window.DB.RISK, MONITOR_FEED = window.DB.MONITOR_FEED, Icon = window.DB.Icon;
const Sparkline = window.Charts.Sparkline, EquityChart = window.Charts.EquityChart, BarChart = window.Charts.BarChart, Donut = window.Charts.Donut, PriceColumns = window.Charts.PriceColumns, Heatmap = window.Charts.Heatmap, LineChart = window.Charts.LineChart, NetworkDiagram = window.Charts.NetworkDiagram, AttentionGrid = window.Charts.AttentionGrid, ConfusionMatrix = window.Charts.ConfusionMatrix, FeatureBars = window.Charts.FeatureBars;
const useState = React.useState, useEffect = React.useEffect, useMemo = React.useMemo, useRef = React.useRef;

/* ───────── Dashboard ───────── */
function DashboardPage({ tickers }) {
  const sparkData = useMemo(() => {
    const r = mulberry32(99);
    const out = {};
    ["balance", "monthly", "winrate", "drawdown", "active", "trades30d"].forEach((k, i) => {
      const arr = [];
      let v = 50;
      for (let j = 0; j < 30; j++) { v += (r() - (k === "drawdown" ? 0.7 : 0.3)) * 8; arr.push(v); }
      out[k] = arr;
    });
    return out;
  }, []);

  return (
    <div className="page" data-screen-label="01 Dashboard">
      <div className="page-head">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <div className="page-sub">System health · positions · signals · risk · {new Date().toLocaleString("en-US", { weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</div>
        </div>
        <div className="row">
          <button className="btn ghost"><Icon name="refresh" size={14} /> Refresh</button>
          <button className="btn"><Icon name="download" size={14} /> Export</button>
        </div>
      </div>

      {/* KPI bar */}
      <div className="kpi-grid">
        <KPICard label="Balance" value={fmtUSD(KPI.balance)} delta={fmtUSD(KPI.balanceDelta) + " today"} positive={KPI.balanceDelta >= 0} spark={sparkData.balance} color="var(--gain)" />
        <KPICard label="Monthly P&L" value={fmtUSD(KPI.monthlyPnl)} delta={fmtPct(KPI.monthlyPnlPct)} positive={KPI.monthlyPnl >= 0} spark={sparkData.monthly} color="var(--gain)" />
        <KPICard label="Win Rate" value={KPI.winRate.toFixed(1) + "%"} delta={fmtPct(KPI.winRateDelta) + " vs 30d"} positive={KPI.winRateDelta >= 0} spark={sparkData.winrate} color="var(--info)" />
        <KPICard label="Max Drawdown" value={KPI.maxDrawdown.toFixed(1) + "%"} delta={fmtPct(KPI.maxDrawdownDelta) + " vs 30d"} positive={KPI.maxDrawdownDelta >= 0} spark={sparkData.drawdown} color="var(--warn)" inverse />
        <KPICard label="Active Trades" value={String(KPI.activeTrades)} delta="3 long · 4 short" positive={null} spark={sparkData.active} color="var(--info)" />
        <KPICard label="Trades · 30d" value={fmtNum(KPI.totalTrades30d)} delta="7.1 / day avg" positive={null} spark={sparkData.trades30d} color="var(--text-1)" />
      </div>

      <div className="dashboard-cols">
        {/* Left col */}
        <div className="col" style={{ gap: 14 }}>
          <Card title="Open Positions" right={<span className="muted" style={{ fontSize: 11 }}>{POSITIONS.length} open · {fmtUSD(POSITIONS.reduce((s, p) => s + p.pnl, 0))}</span>}>
            <div className="card-body flush">
              {POSITIONS.map(p => (
                <div className="pos-row" key={p.id}>
                  <div>
                    <div className="pos-meta">
                      <span style={{ fontWeight: 600, fontFamily: "var(--mono)" }}>{p.sym}</span>
                      <span className={"chip " + (p.side === "long" ? "gain" : "loss")}>{p.side}</span>
                      <span className="muted" style={{ fontSize: 11, fontFamily: "var(--mono)" }}>{p.qty} @ {p.entry}</span>
                    </div>
                    <div className="pos-sub">opened {timeAgo(p.opened)} ago · mark <span style={{ fontFamily: "var(--mono)" }}>{p.mark}</span></div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div className={p.pnl >= 0 ? "gain" : "loss"} style={{ fontFamily: "var(--mono)", fontWeight: 600 }}>{fmtUSD(p.pnl)}</div>
                    <div className={p.pnl >= 0 ? "gain" : "loss"} style={{ fontFamily: "var(--mono)", fontSize: 11, opacity: 0.85 }}>{fmtPct(p.pnlPct)}</div>
                  </div>
                  <button className="btn sm">Close</button>
                </div>
              ))}
            </div>
          </Card>

          <Card title="ML Trader · Status" right={<span className="chip gain"><span className="chip-dot" />running</span>}>
            <div className="card-body">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, fontSize: 12 }}>
                <KV label="Strategy" value={ML_TRADER.strategy} />
                <KV label="Session" value={ML_TRADER.session} />
                <KV label="Model" value={ML_TRADER.modelVersion} mono />
                <KV label="Last train" value={timeAgo(ML_TRADER.lastTrained) + " ago"} />
                <KV label="Sharpe" value={ML_TRADER.sharpe.toFixed(2)} mono />
                <KV label="Sortino" value={ML_TRADER.sortino.toFixed(2)} mono />
              </div>
              <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 4 }}>
                {ML_TRADER.symbols.map(s => <span key={s} className="chip neutral">{s}</span>)}
              </div>
              <div style={{ marginTop: 14, display: "flex", gap: 6 }}>
                <button className="btn"><Icon name="pause" size={12} /> Pause</button>
                <button className="btn danger"><Icon name="stop" size={12} /> Stop</button>
                <div className="spacer" />
                <button className="btn ghost">Open detail →</button>
              </div>
            </div>
          </Card>

          <Card title="Signal Alerts" right={<button className="btn sm ghost">View all</button>}>
            <div className="card-body flush">
              {SIGNALS.slice(0, 6).map(s => (
                <div className="sig-row" key={s.id}>
                  <div className="sig-time">{timeAgo(s.time)}</div>
                  <div>
                    <div className="row" style={{ gap: 6, alignItems: "baseline" }}>
                      <div className={"sev " + s.severity} />
                      <span style={{ fontFamily: "var(--mono)", fontWeight: 600 }}>{s.sym}</span>
                      <span className="sig-msg">{s.msg}</span>
                    </div>
                    <div className="sig-detail">{s.detail}</div>
                  </div>
                  <span className={"chip " + (s.kind === "entry" ? "gain" : s.kind === "exit" ? "info" : s.kind === "alert" ? "warn" : "neutral")}>{s.kind}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right col */}
        <div className="col" style={{ gap: 14 }}>
          <Card title="Risk Overview" right={<span className="chip gain"><span className="chip-dot" />within limits</span>}>
            <div className="card-body">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: 24, alignItems: "center" }}>
                <div>
                  <RiskRow label="Net exposure" value={(RISK.exposure * 100).toFixed(0) + "%"} pct={RISK.exposure / RISK.exposureLimit} />
                  <RiskRow label="Daily drawdown" value={(RISK.dailyDD * 100).toFixed(2) + "%"} pct={RISK.dailyDD / RISK.dailyDDLimit} accent="var(--warn)" />
                  <RiskRow label="Long / Short" value={(RISK.longShort.long * 100).toFixed(0) + " / " + (RISK.longShort.short * 100).toFixed(0)} pct={RISK.longShort.long} accent="var(--info)" />
                  <RiskRow label="VaR (1d, 95%)" value={fmtUSD(RISK.varDay)} pct={0.32} accent="var(--loss)" />
                  <RiskRow label="Beta vs SPY" value={RISK.beta.toFixed(2)} pct={RISK.beta / 2} accent="var(--text-1)" />
                </div>
                <div className="donut-wrap" style={{ flexDirection: "column", gap: 10, alignItems: "center" }}>
                  <Donut data={RISK.byClass.map(c => ({ value: c.value, color: c.color }))} size={130} thickness={20} />
                  <div className="donut-legend" style={{ flexDirection: "column" }}>
                    {RISK.byClass.map(c => (
                      <div key={c.name}>
                        <span className="dot" style={{ background: c.color }} />
                        {c.name}
                        <span className="muted" style={{ marginLeft: 6, fontFamily: "var(--mono)" }}>{(c.value * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Equity Curve · 90d" right={
            <div className="row" style={{ gap: 4 }}>
              <span className="chip gain">+{((EQUITY[EQUITY.length-1].value/EQUITY[0].value-1)*100).toFixed(1)}%</span>
              <span className="muted" style={{ fontSize: 11, marginLeft: 4 }}>{fmtUSD(EQUITY[0].value)} → {fmtUSD(EQUITY[EQUITY.length-1].value)}</span>
            </div>}>
            <div className="card-body" style={{ paddingBottom: 6 }}>
              <EquityChart data={EQUITY} height={220} />
            </div>
          </Card>

          <Card title="Recent Trades" right={<button className="btn sm ghost">Full history →</button>}>
            <div className="card-body flush" style={{ maxHeight: 320, overflowY: "auto" }}>
              <table className="tbl">
                <thead><tr>
                  <th>Time</th><th>Symbol</th><th>Side</th><th className="num">Qty</th><th className="num">Entry</th><th className="num">Exit</th><th className="num">P&L</th><th className="num">R:R</th><th>Reason</th>
                </tr></thead>
                <tbody>
                  {TRADES.slice(0, 8).map(t => (
                    <tr key={t.id}>
                      <td className="muted">{timeAgo(t.time)} ago</td>
                      <td className="sym">{t.sym}</td>
                      <td><span className={"chip " + (t.side === "long" ? "gain" : "loss")}>{t.side}</span></td>
                      <td className="num">{t.qty}</td>
                      <td className="num">{t.entry}</td>
                      <td className="num">{t.exit}</td>
                      <td className={"num " + (t.pnl >= 0 ? "gain" : "loss")}>{fmtUSD(t.pnl)}</td>
                      <td className="num muted">{t.rr.toFixed(2)}</td>
                      <td className="muted">{t.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function KPICard({ label, value, delta, positive, spark, color, inverse }) {
  const tone = positive === null ? "muted" : (inverse ? !positive : positive) ? "gain" : "loss";
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      <div className={"kpi-delta " + tone}>
        {positive !== null && <span style={{ fontSize: 9 }}>{(inverse ? !positive : positive) ? "▲" : "▼"}</span>}
        {delta}
      </div>
      <div className="kpi-spark"><Sparkline data={spark} color={color} w={140} h={36} fill /></div>
    </div>
  );
}

function Card({ title, right, children }) {
  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">{title}</div>
        {right}
      </div>
      {children}
    </div>
  );
}

function KV({ label, value, mono }) {
  return (
    <div>
      <div style={{ color: "var(--text-2)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600 }}>{label}</div>
      <div style={{ marginTop: 2, fontFamily: mono ? "var(--mono)" : undefined }}>{value}</div>
    </div>
  );
}

function RiskRow({ label, value, pct, accent = "var(--gain)" }) {
  return (
    <div className="risk-row">
      <div className="risk-label">{label}</div>
      <div className="meter"><span style={{ width: Math.min(pct * 100, 100) + "%", background: accent }} /></div>
      <div className="risk-val">{value}</div>
    </div>
  );
}

/* ───────── Trader page ───────── */
function TraderPage() {
  const equity = useMemo(() => EQUITY.slice(-30), []);
  const heatData = useMemo(() => {
    const r = mulberry32(31);
    return Array.from({ length: 56 }, () => +(((r() - 0.4) * 6).toFixed(1)));
  }, []);
  return (
    <div className="page" data-screen-label="02 Trader">
      <div className="page-head">
        <div>
          <h1 className="page-title">ML Trader</h1>
          <div className="page-sub">{ML_TRADER.strategy} · {ML_TRADER.session}</div>
        </div>
        <div className="row">
          <button className="btn"><Icon name="pause" size={12} /> Pause</button>
          <button className="btn danger"><Icon name="stop" size={12} /> Stop</button>
          <button className="btn primary"><Icon name="lightning" size={12} /> Force re-evaluate</button>
        </div>
      </div>

      <div className="kpi-grid" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
        <SmallStat label="Net P&L · 30d" value={fmtUSD(ML_TRADER.pnl)} tone="gain" />
        <SmallStat label="Win rate" value={ML_TRADER.winRate.toFixed(1) + "%"} />
        <SmallStat label="Avg R:R" value={ML_TRADER.avgRR.toFixed(2)} />
        <SmallStat label="Sharpe" value={ML_TRADER.sharpe.toFixed(2)} tone="info" />
        <SmallStat label="Trades · 30d" value={String(ML_TRADER.trades)} />
      </div>

      <div className="grid-2">
        <Card title="Equity · 30d">
          <div className="card-body"><EquityChart data={equity} height={240} /></div>
        </Card>
        <Card title="Daily Returns · 8 weeks (M–S)" right={<span className="muted" style={{ fontSize: 11 }}>by week, top→bottom</span>}>
          <div className="card-body">
            <Heatmap data={heatData} cols={7} rows={8} />
          </div>
        </Card>
      </div>

      <div className="grid-2" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card title="Recent Signals">
          <div className="card-body flush">
            {SIGNALS.map(s => (
              <div className="sig-row" key={s.id}>
                <div className="sig-time">{timeAgo(s.time)}</div>
                <div>
                  <div className="row" style={{ gap: 6, alignItems: "baseline" }}>
                    <div className={"sev " + s.severity} />
                    <span style={{ fontFamily: "var(--mono)", fontWeight: 600 }}>{s.sym}</span>
                    <span className="sig-msg">{s.msg}</span>
                  </div>
                  <div className="sig-detail">{s.detail}</div>
                </div>
                <span className={"chip " + (s.kind === "entry" ? "gain" : s.kind === "exit" ? "info" : s.kind === "alert" ? "warn" : "neutral")}>{s.kind}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Symbol Breakdown">
          <div className="card-body flush">
            <table className="tbl">
              <thead><tr><th>Symbol</th><th className="num">Trades</th><th className="num">Win Rate</th><th className="num">P&L</th><th>Share</th></tr></thead>
              <tbody>
                {SYM_BREAKDOWN.map(s => (
                  <tr key={s.sym}>
                    <td className="sym">{s.sym}</td>
                    <td className="num">{s.trades}</td>
                    <td className="num">{s.winRate.toFixed(1)}%</td>
                    <td className={"num " + (s.pnl >= 0 ? "gain" : "loss")}>{fmtUSD(s.pnl)}</td>
                    <td><div className="meter" style={{ width: 100 }}><span style={{ width: (s.share * 100 * 3) + "%" }} /></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
function SmallStat({ label, value, tone = "" }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className={"kpi-value " + tone}>{value}</div>
    </div>
  );
}

/* ───────── Monitors page ───────── */
function MonitorsPage({ feedItems }) {
  return (
    <div className="page" data-screen-label="03 Monitors">
      <div className="page-head">
        <div><h1 className="page-title">Monitors</h1><div className="page-sub">Real-time feeds from the trading engine</div></div>
        <div className="row"><span className="conn-status"><span className="conn-dot" /> ws://localhost:3000/ws</span></div>
      </div>

      <div className="grid-4">
        <MonitorStat label="Market events / min" value="142" tone="info" />
        <MonitorStat label="Signals (last 1h)" value="38" tone="gain" />
        <MonitorStat label="WS latency" value="42ms" tone="muted" />
        <MonitorStat label="System health" value="OK" tone="gain" />
      </div>

      <div className="grid-2">
        <Card title="Live event stream" right={<span className="chip gain"><span className="chip-dot" />streaming</span>}>
          <div className="card-body flush" style={{ maxHeight: 380, overflowY: "auto" }}>
            {feedItems.map((e, i) => (
              <div key={i} style={{ padding: "7px 14px", borderBottom: "1px solid var(--line-soft)", display: "grid", gridTemplateColumns: "60px 70px 1fr", gap: 10, fontFamily: "var(--mono)", fontSize: 11.5 }}>
                <span className="muted">{new Date(e.t).toLocaleTimeString("en-US", { hour12: false })}</span>
                <span className={"chip " + ({MARKET:"info", SIGNAL:"gain", EXEC:"warn", RISK:"neutral", NEWS:"info", ML:"info"}[e.src] || "neutral")} style={{ justifyContent: "center" }}>{e.src}</span>
                <span style={{ color: "var(--text-0)" }}>{e.msg}</span>
              </div>
            ))}
          </div>
        </Card>

        <div className="col" style={{ gap: 14 }}>
          <Card title="Chart Monitor · BTC-USD">
            <div className="card-body" style={{ paddingBottom: 6 }}>
              <EquityChart data={EQUITY.slice(-40).map((d,i)=>({...d, value: 65000 + (d.value-150000)*0.1}))} height={170} color="var(--info)" />
            </div>
          </Card>
          <Card title="System Health">
            <div className="card-body">
              <HealthRow label="Engine" value="running" tone="ok" />
              <HealthRow label="WebSocket" value="connected · 42ms" tone="ok" />
              <HealthRow label="Database (Postgres)" value="healthy · 2.3GB" tone="ok" />
              <HealthRow label="Redis (state)" value="connected · 412 keys" tone="ok" />
              <HealthRow label="ML inference" value="degraded · 124ms p95" tone="warn" />
              <HealthRow label="News provider" value="connected" tone="ok" />
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
function MonitorStat({ label, value, tone }) {
  return <div className="kpi"><div className="kpi-label">{label}</div><div className={"kpi-value " + tone}>{value}</div></div>;
}
function HealthRow({ label, value, tone }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", padding: "8px 0", borderBottom: "1px solid var(--line-soft)", alignItems: "center", fontSize: 12.5 }}>
      <span className="muted">{label}</span>
      <span className="row" style={{ gap: 6 }}>
        <span className={"sev " + tone} style={{ marginTop: 0 }} />
        <span className={tone === "ok" ? "gain" : tone === "warn" ? "warn" : "loss"} style={{ fontFamily: "var(--mono)", fontSize: 11.5 }}>{value}</span>
      </span>
    </div>
  );
}

/* ───────── Analytics page ───────── */
function AnalyticsPage() {
  const [period, setPeriod] = useState("30d");
  return (
    <div className="page" data-screen-label="04 Analytics">
      <div className="page-head">
        <div><h1 className="page-title">Analytics</h1><div className="page-sub">Performance · returns · attribution</div></div>
        <div className="seg">
          {["7d","30d","90d","all"].map(p => (
            <button key={p} className={period===p?"active":""} onClick={()=>setPeriod(p)}>{p}</button>
          ))}
        </div>
      </div>

      <div className="kpi-grid">
        <SmallStat label="Net P&L" value={fmtUSD(KPI.monthlyPnl)} tone="gain" />
        <SmallStat label="Win Rate" value={KPI.winRate.toFixed(1) + "%"} tone="info" />
        <SmallStat label="Total Trades" value={fmtNum(KPI.totalTrades30d)} />
        <SmallStat label="Avg R:R" value={ML_TRADER.avgRR.toFixed(2)} />
        <SmallStat label="Max Drawdown" value={KPI.maxDrawdown.toFixed(1) + "%"} tone="warn" />
        <SmallStat label="Profit Factor" value="2.14" tone="gain" />
      </div>

      <Card title={"Equity Curve · " + period} right={<span className="muted" style={{ fontSize: 11 }}>vs SPY benchmark · 0.62 beta</span>}>
        <div className="card-body" style={{ paddingBottom: 6 }}><EquityChart data={EQUITY} height={280} /></div>
      </Card>

      <div className="grid-2">
        <Card title="Monthly Returns · last 12 months">
          <div className="card-body"><BarChart data={MONTHLY} height={220} /></div>
        </Card>
        <Card title="P&L Attribution by Symbol">
          <div className="card-body flush">
            <table className="tbl">
              <thead><tr><th>Symbol</th><th className="num">Trades</th><th className="num">Win %</th><th className="num">P&L</th><th>Contribution</th></tr></thead>
              <tbody>
                {SYM_BREAKDOWN.map(s => (
                  <tr key={s.sym}>
                    <td className="sym">{s.sym}</td>
                    <td className="num">{s.trades}</td>
                    <td className="num">{s.winRate.toFixed(1)}%</td>
                    <td className={"num " + (s.pnl >= 0 ? "gain" : "loss")}>{fmtUSD(s.pnl)}</td>
                    <td><div className="meter" style={{ width: 110 }}><span style={{ width: (s.share * 100 * 3) + "%", background: s.pnl >= 0 ? "var(--gain)" : "var(--loss)" }} /></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ───────── Trade History ───────── */
function HistoryPage() {
  const [filter, setFilter] = useState("");
  const [side, setSide] = useState("all");
  const [sortBy, setSortBy] = useState("time");
  const [page, setPage] = useState(0);
  const pageSize = 20;
  const filtered = TRADES.filter(t => {
    if (side !== "all" && t.side !== side) return false;
    if (filter && !t.sym.toLowerCase().includes(filter.toLowerCase())) return false;
    return true;
  }).sort((a, b) => sortBy === "pnl" ? b.pnl - a.pnl : sortBy === "sym" ? a.sym.localeCompare(b.sym) : b.time - a.time);
  const pageData = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const totalPnl = filtered.reduce((s, t) => s + t.pnl, 0);

  return (
    <div className="page" data-screen-label="05 History">
      <div className="page-head">
        <div><h1 className="page-title">Trade History</h1><div className="page-sub">{filtered.length} trades · net <span className={totalPnl>=0?"gain":"loss"}>{fmtUSD(totalPnl)}</span></div></div>
        <div className="row">
          <input className="btn" style={{ background: "var(--bg-2)", padding: "6px 10px", outline: "none", border: "1px solid var(--line)", color: "var(--text-0)" }} placeholder="Filter symbol…" value={filter} onChange={(e)=>{setFilter(e.target.value); setPage(0);}} />
          <div className="seg">
            {["all","long","short"].map(s => <button key={s} className={side===s?"active":""} onClick={()=>{setSide(s); setPage(0);}}>{s}</button>)}
          </div>
          <button className="btn"><Icon name="download" size={14}/> Export CSV</button>
        </div>
      </div>

      <Card title={`Trades · ${filtered.length}`} right={
        <div className="seg">
          <button className={sortBy==="time"?"active":""} onClick={()=>setSortBy("time")}>time</button>
          <button className={sortBy==="pnl"?"active":""} onClick={()=>setSortBy("pnl")}>pnl</button>
          <button className={sortBy==="sym"?"active":""} onClick={()=>setSortBy("sym")}>sym</button>
        </div>
      }>
        <div className="card-body flush">
          <table className="tbl">
            <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th className="num">Qty</th><th className="num">Entry</th><th className="num">Exit</th><th className="num">P&L</th><th className="num">R:R</th><th>Reason</th></tr></thead>
            <tbody>
              {pageData.map(t => (
                <tr key={t.id}>
                  <td className="muted">{new Date(t.time).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</td>
                  <td className="sym">{t.sym}</td>
                  <td><span className={"chip " + (t.side === "long" ? "gain" : "loss")}>{t.side}</span></td>
                  <td className="num">{t.qty}</td>
                  <td className="num">{t.entry}</td>
                  <td className="num">{t.exit}</td>
                  <td className={"num " + (t.pnl >= 0 ? "gain" : "loss")}>{fmtUSD(t.pnl)}</td>
                  <td className="num muted">{t.rr.toFixed(2)}</td>
                  <td className="muted">{t.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ display: "flex", padding: 12, justifyContent: "space-between", alignItems: "center", borderTop: "1px solid var(--line-soft)" }}>
            <span className="muted" style={{ fontSize: 11 }}>Page {page+1} of {Math.max(1, Math.ceil(filtered.length / pageSize))}</span>
            <div className="row">
              <button className="btn sm" disabled={page===0} onClick={()=>setPage(p=>Math.max(0,p-1))}>← Prev</button>
              <button className="btn sm" disabled={(page+1)*pageSize>=filtered.length} onClick={()=>setPage(p=>p+1)}>Next →</button>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

/* ───────── Alerts ───────── */
function AlertsPage({ onAck }) {
  const [filter, setFilter] = useState("all");
  const [acked, setAcked] = useState(new Set());
  const filtered = ALERTS.filter(a => filter === "all" || a.severity === filter);
  return (
    <div className="page" data-screen-label="06 Alerts">
      <div className="page-head">
        <div><h1 className="page-title">Alerts</h1><div className="page-sub">{ALERTS.length} total · {ALERTS.filter(a=>!acked.has(a.id) && a.severity !== "ok").length} unread</div></div>
        <div className="seg">
          {["all","high","med","low","ok"].map(s => <button key={s} className={filter===s?"active":""} onClick={()=>setFilter(s)}>{s}</button>)}
        </div>
      </div>
      <Card title="Alerts feed">
        <div className="card-body flush">
          {filtered.map(a => (
            <div key={a.id} style={{ display: "grid", gridTemplateColumns: "auto 1fr auto auto", gap: 12, padding: "12px 16px", borderBottom: "1px solid var(--line-soft)", opacity: acked.has(a.id) ? 0.5 : 1 }}>
              <div className={"sev " + a.severity} style={{ marginTop: 6 }} />
              <div>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{a.title}</div>
                <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>{a.msg}</div>
                <div className="muted" style={{ fontSize: 11, marginTop: 4, fontFamily: "var(--mono)" }}>{a.source} · {timeAgo(a.time)} ago</div>
              </div>
              <span className={"chip " + (a.severity === "high" ? "loss" : a.severity === "med" ? "warn" : a.severity === "low" ? "info" : "gain")}>{a.severity}</span>
              {!acked.has(a.id) ? <button className="btn sm" onClick={()=>{ const ns = new Set(acked); ns.add(a.id); setAcked(ns); onAck && onAck(); }}>Ack</button> : <span className="muted" style={{ fontSize: 11, alignSelf: "center" }}>acked</span>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ───────── Backtesting ───────── */
function BacktestPage() {
  const [symbols, setSymbols] = useState("BTC-USD,ETH-USD,NVDA,AAPL");
  const [capital, setCapital] = useState(100000);
  const [commission, setCommission] = useState(0.001);
  const [slippage, setSlippage] = useState(0.0005);
  const [period, setPeriod] = useState("90d");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(BACKTESTS);

  function runBacktest() {
    setRunning(true);
    setProgress(0);
    const t = setInterval(() => {
      setProgress(p => {
        if (p >= 100) {
          clearInterval(t);
          setRunning(false);
          const newId = "2026-04-02-" + String.fromCharCode(67 + results.length - BACKTESTS.length);
          setResults([{ id: newId, date: "2026-04-02", symbols: symbols.split(",").length, period, capital, finalPnl: +(capital*0.08*(0.5+Math.random())).toFixed(0), winRate: 55+Math.random()*15, maxDD: -3-Math.random()*8, sharpe: 0.8+Math.random()*1.8, status: "complete" }, ...results]);
          return 100;
        }
        return p + 4;
      });
    }, 80);
  }

  return (
    <div className="page" data-screen-label="07 Backtest">
      <div className="page-head"><div><h1 className="page-title">Backtesting</h1><div className="page-sub">Run strategy backtests against historical data</div></div></div>
      <div className="grid-2" style={{ gridTemplateColumns: "400px 1fr" }}>
        <Card title="New backtest">
          <div className="card-body">
            <FormRow label="Symbols (comma-separated)"><input className="form-field" value={symbols} onChange={e=>setSymbols(e.target.value)} /></FormRow>
            <FormRow label="Period">
              <div className="seg">{["7d","30d","90d","180d","365d"].map(p=><button key={p} className={period===p?"active":""} onClick={()=>setPeriod(p)}>{p}</button>)}</div>
            </FormRow>
            <FormRow label="Starting capital"><input className="form-field" type="number" value={capital} onChange={e=>setCapital(+e.target.value)} /></FormRow>
            <FormRow label="Commission (per trade)"><input className="form-field" type="number" step="0.0001" value={commission} onChange={e=>setCommission(+e.target.value)} /></FormRow>
            <FormRow label="Slippage"><input className="form-field" type="number" step="0.0001" value={slippage} onChange={e=>setSlippage(+e.target.value)} /></FormRow>
            <button className="btn primary" disabled={running} onClick={runBacktest} style={{ width: "100%", justifyContent: "center", marginTop: 4 }}>
              {running ? `Running… ${progress}%` : <><Icon name="play" size={12}/> Run backtest</>}
            </button>
            {running && <div className="meter" style={{ marginTop: 10 }}><span style={{ width: progress + "%" }} /></div>}
          </div>
        </Card>
        <Card title={`Results · ${results.length}`}>
          <div className="card-body flush">
            <table className="tbl">
              <thead><tr><th>ID</th><th>Date</th><th>Period</th><th className="num">Capital</th><th className="num">P&L</th><th className="num">Win %</th><th className="num">Max DD</th><th className="num">Sharpe</th></tr></thead>
              <tbody>
                {results.map(r => (
                  <tr key={r.id}>
                    <td className="sym">{r.id}</td>
                    <td className="muted">{r.date}</td>
                    <td>{r.period}</td>
                    <td className="num">{fmtUSD(r.capital, 0)}</td>
                    <td className={"num " + (r.finalPnl >= 0 ? "gain" : "loss")}>{fmtUSD(r.finalPnl)}</td>
                    <td className="num">{r.winRate.toFixed(1)}%</td>
                    <td className="num warn">{r.maxDD.toFixed(1)}%</td>
                    <td className={"num " + (r.sharpe >= 1 ? "gain" : "loss")}>{r.sharpe.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

function FormRow({ label, children }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ color: "var(--text-2)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", fontWeight: 600, marginBottom: 5 }}>{label}</div>
      {children}
      <style>{`.form-field{width:100%;padding:7px 10px;background:var(--bg-2);color:var(--text-0);border:1px solid var(--line);border-radius:6px;font-family:var(--mono);font-size:12px;outline:none}.form-field:focus{border-color:var(--accent)}`}</style>
    </div>
  );
}

/* ───────── Training ───────── */
function TrainingPage() {
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState([
    "[2026-04-02 04:12:03] starting nightly training run",
    "[2026-04-02 04:12:04] loaded 14,283 episodes from replay buffer",
    "[2026-04-02 04:12:09] LSTM: epoch 1/12 · loss 0.0421 · val_loss 0.0398",
    "[2026-04-02 04:14:22] LSTM: epoch 6/12 · loss 0.0258 · val_loss 0.0244",
    "[2026-04-02 04:18:51] LSTM: epoch 12/12 · loss 0.0231 · val_loss 0.0231",
    "[2026-04-02 04:18:51] LSTM: ✓ saved lstm_price_v0.7.3.pt",
    "[2026-04-02 04:19:02] PPO: rollout 1/8 · avg_reward 0.318",
    "[2026-04-02 04:24:18] PPO: rollout 8/8 · avg_reward 0.412",
    "[2026-04-02 04:24:18] PPO: ✓ saved ppo_policy_v0.7.3.pt",
    "[2026-04-02 04:24:19] training run complete · 12m 16s",
  ]);
  return (
    <div className="page" data-screen-label="08 Training">
      <div className="page-head"><div><h1 className="page-title">Training</h1><div className="page-sub">Pipeline · datasets · jobs</div></div>
        <button className="btn primary" disabled={running} onClick={()=>{setRunning(true); setTimeout(()=>setRunning(false),3000);}}><Icon name="play" size={12}/> Start training job</button>
      </div>
      <div className="grid-3">
        <Card title="Last run"><div className="card-body" style={{display:"flex",flexDirection:"column",gap:6}}>
          <KV label="Status" value={<span className="chip gain"><span className="chip-dot"/>complete</span>} />
          <KV label="Duration" value="12m 16s" mono />
          <KV label="Val loss" value="0.0231 (-3.4%)" mono />
        </div></Card>
        <Card title="Replay buffer"><div className="card-body" style={{display:"flex",flexDirection:"column",gap:6}}>
          <KV label="Episodes" value={fmtNum(RL_AGENT.episodes)} mono />
          <KV label="Capacity" value="50,000" mono />
          <div className="meter"><span style={{ width: (RL_AGENT.episodes/50000*100).toFixed(0)+"%" }} /></div>
        </div></Card>
        <Card title="Datasets"><div className="card-body" style={{display:"flex",flexDirection:"column",gap:6}}>
          <KV label="Price (1m)" value="384.2 GB · 6 yrs" mono />
          <KV label="News corpus" value="2.1 GB · 18 mo" mono />
          <KV label="Sentiment labels" value="124k · gold" mono />
        </div></Card>
      </div>
      <Card title="Training log" right={<span className="chip gain"><span className="chip-dot"/>complete</span>}>
        <div className="card-body"><div className="code">{logs.join("\n")}</div></div>
      </Card>
    </div>
  );
}

/* ───────── ML / AI page ───────── */
function MLPage() {
  const [tab, setTab] = useState("overview");
  const lossSeries = useMemo(() => {
    const r = mulberry32(7);
    const train = []; const val = [];
    let t = 0.08; let v = 0.082;
    for (let i = 0; i < 50; i++) { t = Math.max(0.018, t * (0.96 + r() * 0.03)); v = Math.max(0.022, v * (0.965 + r() * 0.04)); train.push(t); val.push(v); }
    return [{ name: "Train loss", color: "var(--info)", data: train }, { name: "Val loss", color: "var(--gain)", data: val }];
  }, []);
  const rewardSeries = useMemo(() => {
    const r = mulberry32(13);
    const arr = []; let v = 0.05;
    for (let i = 0; i < 80; i++) { v += (r() - 0.42) * 0.04; arr.push(v); }
    return [{ name: "Avg reward (per 1k eps)", color: "var(--gain)", data: arr }];
  }, []);

  return (
    <div className="page" data-screen-label="09 ML/AI">
      <div className="page-head">
        <div><h1 className="page-title">ML / AI</h1><div className="page-sub">5 models · ensemble inference · what the trader is modeling</div></div>
        <div className="row"><button className="btn"><Icon name="refresh" size={12}/> Refresh weights</button><button className="btn primary"><Icon name="brain" size={12}/> Retrain all</button></div>
      </div>

      <div className="tabs">
        {["overview","architecture","attention","features","rl"].map(t => <button key={t} className={"tab "+(tab===t?"active":"")} onClick={()=>setTab(t)}>{t === "rl" ? "RL agent" : t.charAt(0).toUpperCase()+t.slice(1)}</button>)}
      </div>

      {tab === "overview" && <>
        <div className="grid-3">
          {ML_MODELS.map(m => <ModelCard key={m.id} m={m} />)}
        </div>
        <Card title="Ensemble inference flow" right={<span className="muted" style={{ fontSize: 11 }}>realtime · 38ms p50 · 124ms p95</span>}>
          <div className="card-body"><EnsembleFlow /></div>
        </Card>
      </>}

      {tab === "architecture" && <>
        <Card title="Price LSTM · architecture">
          <div className="card-body">
            <NetworkDiagram color="var(--info)" layers={[
              { size: 60, label: "Input · OHLCV+indicators" },
              { size: 128, label: "LSTM 1" },
              { size: 128, label: "LSTM 2" },
              { size: 64, label: "Dense · ReLU" },
              { size: 3, label: "Output · ↑ → ↓" },
            ]} />
            <div className="muted" style={{ fontSize: 11.5, fontFamily: "var(--mono)", marginTop: 12 }}>
              60-step lookback · 14 indicators (RSI, MACD, ATR, BB, OBV…) · 128-unit stacked LSTM → 64 ReLU → 3-class softmax
            </div>
          </div>
        </Card>
        <Card title="PPO Policy · architecture">
          <div className="card-body">
            <NetworkDiagram color="var(--gain)" layers={[
              { size: 80, label: "State · price+pos+risk" },
              { size: 256, label: "Shared FC" },
              { size: 128, label: "Actor head" },
              { size: 128, label: "Critic head" },
              { size: 5, label: "Action · 5 disc." },
            ]} />
            <div className="muted" style={{ fontSize: 11.5, fontFamily: "var(--mono)", marginTop: 12 }}>
              State 80-d (last 12 ticks + position + risk metrics) · actor-critic shared trunk · 5 actions: {"{ buy, sell, hold, scale_up, scale_down }"}
            </div>
          </div>
        </Card>
      </>}

      {tab === "attention" && <>
        <Card title="Attention weights · NVDA, last 24 ticks → forecast" right={<span className="muted" style={{ fontSize: 11 }}>Self-attention head 4 · brighter = more weight</span>}>
          <div className="card-body">
            <AttentionGrid
              rows={6}
              cols={24}
              rowLabels={["t-5m","t-4m","t-3m","t-2m","t-1m","t-0"]}
              colLabels={Array.from({length:24}, (_,i)=>"t-"+(23-i))}
              data={Array.from({length:6}, (_, ri) => Array.from({length:24}, (_, ci) => {
                const dist = Math.abs(ri*4 - ci);
                return Math.exp(-dist/6) * (0.6 + Math.random() * 0.4);
              }))}
              color="var(--info)"
            />
            <div className="muted" style={{ fontSize: 11.5, marginTop: 14, lineHeight: 1.55 }}>
              The model attends most strongly to the <strong style={{ color: "var(--text-0)" }}>last 4–6 minutes</strong> when forecasting the next tick — a short-horizon momentum bias consistent with the
              <strong style={{ color: "var(--text-0)" }}> mean-reversion</strong> regime detected by the regime classifier today. Older context contributes mostly through the LSTM's hidden state.
            </div>
          </div>
        </Card>

        <div className="grid-2">
          <Card title="Confusion matrix · regime classifier">
            <div className="card-body">
              <ConfusionMatrix
                labels={["trend↑","range","trend↓","vol-spike"]}
                matrix={[
                  [142, 12, 4, 6],
                  [18, 196, 14, 8],
                  [5, 16, 128, 9],
                  [3, 6, 5, 78],
                ]}
              />
              <div className="muted" style={{ fontSize: 11.5, marginTop: 10 }}>4-class accuracy: 81.2% · weakest cell: range ↔ trend↑ (often mid-transition)</div>
            </div>
          </Card>
          <Card title="Loss curves · last 50 epochs">
            <div className="card-body">
              <LineChart series={lossSeries} height={240} />
              <div className="row" style={{ gap: 14, marginTop: 8, fontSize: 12 }}>
                <span><span className="dot" style={{ display: "inline-block", width: 10, height: 2, background: "var(--info)", marginRight: 6, verticalAlign: "middle" }} /> Train</span>
                <span><span className="dot" style={{ display: "inline-block", width: 10, height: 2, background: "var(--gain)", marginRight: 6, verticalAlign: "middle" }} /> Val</span>
              </div>
            </div>
          </Card>
        </div>
      </>}

      {tab === "features" && <>
        <Card title="Feature importance · what drives entry decisions" right={<span className="muted" style={{ fontSize: 11 }}>SHAP values, normalized · last 30d</span>}>
          <div className="card-body">
            <FeatureBars items={[
              { label: "RSI(14)", value: 0.182 },
              { label: "ATR(14)", value: 0.158 },
              { label: "Volume Δ", value: 0.144 },
              { label: "MACD hist", value: 0.121 },
              { label: "BB width", value: 0.098 },
              { label: "Sentiment", value: 0.087 },
              { label: "OBV slope", value: 0.072 },
              { label: "Regime · prob", value: 0.061 },
              { label: "Time of day", value: 0.041 },
              { label: "Btc-corr 30m", value: 0.036 },
            ]} color="var(--info)" />
          </div>
        </Card>

        <div className="grid-2">
          <Card title="Predicted distribution · BTC-USD next 5m">
            <div className="card-body">
              <PriceColumns data={Array.from({length: 21}, (_, i) => {
                const x = (i - 10) / 4;
                const v = Math.exp(-x*x) * (1 + (i-10)*0.04);
                return v;
              })} height={140} />
              <div className="row" style={{ justifyContent: "space-between", marginTop: 8, fontSize: 11, fontFamily: "var(--mono)", color: "var(--text-2)" }}>
                <span>-1.5%</span><span>μ = +0.18%</span><span>+1.5%</span>
              </div>
            </div>
          </Card>
          <Card title="Sentiment · NVDA · last 24h">
            <div className="card-body">
              <PriceColumns data={Array.from({length: 24}, (_, i) => Math.sin(i / 4) * 0.6 + (Math.random() - 0.5) * 0.4)} height={140} />
              <div className="muted" style={{ fontSize: 11.5, marginTop: 8 }}>BERT-derived sentiment per hour · 24h mean +0.12 · 4 high-impact items detected</div>
            </div>
          </Card>
        </div>
      </>}

      {tab === "rl" && <>
        <div className="kpi-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
          <SmallStat label="Episodes" value={fmtNum(RL_AGENT.episodes)} />
          <SmallStat label="Avg reward" value={RL_AGENT.avgReward.toFixed(3)} tone="info" />
          <SmallStat label="Recent (1k)" value={RL_AGENT.recentReward.toFixed(3)} tone="gain" />
          <SmallStat label="Exploration ε" value={RL_AGENT.exploration.toFixed(2)} />
        </div>
        <Card title="Reward over training · per 1k episodes">
          <div className="card-body"><LineChart series={rewardSeries} height={260} /></div>
        </Card>
        <Card title="Action distribution · last 1,000 trading steps">
          <div className="card-body">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
              {[
                { name: "BUY", v: 18, color: "var(--gain)" },
                { name: "SELL", v: 14, color: "var(--loss)" },
                { name: "HOLD", v: 58, color: "var(--text-2)" },
                { name: "SCALE↑", v: 6, color: "var(--info)" },
                { name: "SCALE↓", v: 4, color: "var(--warn)" },
              ].map(a => (
                <div key={a.name} style={{ background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 6, padding: 12, textAlign: "center" }}>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{a.name}</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 24, fontWeight: 600, color: a.color, marginTop: 4 }}>{a.v}%</div>
                  <div className="meter" style={{ marginTop: 8 }}><span style={{ width: a.v * 1.5 + "%", background: a.color }} /></div>
                </div>
              ))}
            </div>
            <div className="muted" style={{ fontSize: 11.5, marginTop: 14 }}>The policy currently <strong style={{ color: "var(--text-0)" }}>holds</strong> ~58% of steps — consistent with a low-conviction regime where the value head suppresses trade frequency to preserve sharpe.</div>
          </div>
        </Card>
      </>}
    </div>
  );
}

function ModelCard({ m }) {
  return (
    <div className="card">
      <div className="card-head">
        <div className="row" style={{ gap: 8 }}>
          <Icon name="brain" size={14} />
          <div className="card-title" style={{ textTransform: "none", letterSpacing: 0, fontSize: 13 }}>{m.name}</div>
        </div>
        <span className={"chip " + (m.status === "active" ? "gain" : m.status === "warming" ? "warn" : "neutral")}><span className="chip-dot" />{m.status}</span>
      </div>
      <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div className="muted" style={{ fontSize: 11.5 }}>{m.role}</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 4, fontSize: 12 }}>
          <KV label="Accuracy" value={(m.accuracy*100).toFixed(1)+"%"} mono />
          <KV label="Loss" value={m.loss.toFixed(4)} mono />
          <KV label="Params" value={m.params} mono />
          <KV label="Last train" value={timeAgo(m.lastTrain)+" ago"} />
        </div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--text-3)", padding: "6px 8px", background: "var(--bg-2)", borderRadius: 4, marginTop: 4 }}>{m.weights}</div>
        <div className="row" style={{ marginTop: 4 }}>
          <span className="muted" style={{ fontSize: 11 }}>Enabled</span>
          <div className={"tgl " + (m.enabled ? "on" : "")} />
          <div className="spacer" />
          <button className="btn sm">Retrain</button>
        </div>
      </div>
    </div>
  );
}

function EnsembleFlow() {
  // Visual diagram of how data flows: MARKET -> [LSTM, BERT, GARCH, Regime] -> RL Policy -> Action
  return (
    <svg viewBox="0 0 920 260" style={{ width: "100%", height: "auto", display: "block" }}>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" fill="var(--text-2)"/>
        </marker>
      </defs>
      {/* Inputs */}
      {[
        { x: 30, y: 20, w: 120, h: 38, label: "Market data", sub: "OHLCV · 1m" },
        { x: 30, y: 70, w: 120, h: 38, label: "News stream", sub: "FT · Bloomberg · WSJ" },
        { x: 30, y: 120, w: 120, h: 38, label: "Position state", sub: "Redis cache" },
        { x: 30, y: 170, w: 120, h: 38, label: "Risk metrics", sub: "VaR · DD · β" },
      ].map((b, i) => <FlowBox key={i} {...b} fill="var(--bg-2)" />)}
      {/* Models */}
      {[
        { x: 240, y: 14, w: 150, h: 50, label: "Price LSTM", sub: "↑→↓ probability", color: "var(--info)" },
        { x: 240, y: 78, w: 150, h: 50, label: "Sentiment BERT", sub: "score [-1, +1]", color: "var(--info)" },
        { x: 240, y: 142, w: 150, h: 50, label: "GARCH volatility", sub: "σ forecast", color: "var(--info)" },
        { x: 240, y: 206, w: 150, h: 50, label: "Regime classifier", sub: "trend / range / vol", color: "var(--info)" },
      ].map((b, i) => <FlowBox key={i} {...b} />)}
      {/* RL Policy */}
      <FlowBox x={490} y={90} w={170} h={80} label="PPO Policy" sub="actor-critic · discrete" color="var(--gain)" big />
      {/* Action */}
      <FlowBox x={730} y={70} w={150} h={50} label="Action" sub="buy/sell/hold/scale" color="var(--warn)" />
      <FlowBox x={730} y={140} w={150} h={50} label="Confidence" sub="0 → 1" color="var(--text-1)" />

      {/* Arrows */}
      {[ [150,39, 240,39], [150,89, 240,103], [150,139, 240,167], [150,189, 240,231],
         [390,39, 490,110], [390,103, 490,120], [390,167, 490,140], [390,231, 490,150],
         [660,130, 730,95], [660,140, 730,165] ].map((c, i) => (
        <line key={i} x1={c[0]} y1={c[1]} x2={c[2]} y2={c[3]} stroke="var(--text-3)" strokeWidth="1" markerEnd="url(#arr)" />
      ))}
    </svg>
  );
}

function FlowBox({ x, y, w, h, label, sub, color = "var(--text-1)", fill, big }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="6" fill={fill || "var(--bg-1)"} stroke={color} strokeWidth={big ? 1.4 : 1} />
      <text x={x + 10} y={y + 18} fontSize={big ? 14 : 12} fontWeight="600" fill="var(--text-0)" fontFamily="var(--sans)">{label}</text>
      <text x={x + 10} y={y + (big ? 38 : 32)} fontSize="10.5" fill="var(--text-2)" fontFamily="var(--mono)">{sub}</text>
      {big && <text x={x + 10} y={y + 60} fontSize="10" fill={color} fontFamily="var(--mono)">action = π(state | θ)</text>}
    </g>
  );
}

/* ───────── Settings ───────── */
function SettingsPage({ tweaks, setTweak }) {
  return (
    <div className="page" data-screen-label="10 Settings">
      <div className="page-head"><div><h1 className="page-title">Settings</h1><div className="page-sub">User preferences · system config</div></div></div>
      <div className="grid-2">
        <Card title="Profile">
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <SettingRow label="Display name"><input className="form-field" defaultValue="Trader" /></SettingRow>
            <SettingRow label="Email"><input className="form-field" defaultValue="admin@admin.com" /></SettingRow>
            <SettingRow label="Time zone"><select className="form-field" defaultValue="UTC"><option>UTC</option><option>America/New_York</option><option>Europe/London</option><option>Asia/Tokyo</option></select></SettingRow>
            <SettingRow label="Default landing page"><select className="form-field"><option>Dashboard</option><option>Trader</option><option>Monitors</option></select></SettingRow>
          </div>
        </Card>
        <Card title="Notifications">
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <ToggleRow label="High-severity alerts" defaultOn />
            <ToggleRow label="Trade fills" defaultOn />
            <ToggleRow label="Daily P&L summary" defaultOn />
            <ToggleRow label="Training pipeline events" />
            <ToggleRow label="Backtest completion" defaultOn />
          </div>
        </Card>
        <Card title="Risk thresholds">
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <SettingRow label="Daily DD limit"><input className="form-field" defaultValue="5.0%" /></SettingRow>
            <SettingRow label="Max exposure"><input className="form-field" defaultValue="70%" /></SettingRow>
            <SettingRow label="Max position size"><input className="form-field" defaultValue="$25,000" /></SettingRow>
            <SettingRow label="Auto-pause trigger"><input className="form-field" defaultValue="3 consecutive stops" /></SettingRow>
          </div>
        </Card>
        <Card title="Display">
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <SettingRow label="Theme">
              <div className="seg">{["navy","black","charcoal"].map(t=><button key={t} className={tweaks.theme===t?"active":""} onClick={()=>setTweak("theme", t)}>{t}</button>)}</div>
            </SettingRow>
            <SettingRow label="Density">
              <div className="seg">{["compact","regular","comfy"].map(t=><button key={t} className={tweaks.density===t?"active":""} onClick={()=>setTweak("density", t)}>{t}</button>)}</div>
            </SettingRow>
          </div>
        </Card>
      </div>
    </div>
  );
}
function SettingRow({ label, children }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", gap: 14, alignItems: "center" }}>
      <span className="muted" style={{ fontSize: 12 }}>{label}</span>
      <div>{children}</div>
    </div>
  );
}
function ToggleRow({ label, defaultOn }) {
  const [on, setOn] = useState(!!defaultOn);
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: 12.5 }}>{label}</span>
      <div className={"tgl " + (on ? "on" : "")} onClick={()=>setOn(!on)} />
    </div>
  );
}

window.Pages = { DashboardPage, TraderPage, MonitorsPage, AnalyticsPage, HistoryPage, AlertsPage, BacktestPage, TrainingPage, MLPage, SettingsPage };

})();