(function(){
// app.jsx — root shell, sidebar, header, ticker, mode dialog

const useS = React.useState, useE = React.useEffect, useM = React.useMemo, useR = React.useRef;
const Icon2 = window.DB.Icon;
const fmtUSD2 = window.DB.fmtUSD;

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: "dashboard" },
  { id: "trader",    label: "ML Trader", icon: "trader" },
  { id: "monitors",  label: "Monitors",  icon: "monitor" },
  { id: "analytics", label: "Analytics", icon: "analytics" },
  { id: "history",   label: "Trade History", icon: "history" },
  { id: "alerts",    label: "Alerts",    icon: "alerts" },
  { id: "_div_",     section: "ML / Research" },
  { id: "ml",        label: "ML / AI",   icon: "ml" },
  { id: "backtest",  label: "Backtesting", icon: "backtest" },
  { id: "training",  label: "Training",  icon: "training" },
  { id: "_div2_",    section: "System" },
  { id: "settings",  label: "Settings",  icon: "settings" },
];

const TICKER_INIT = [
  { sym: "BTC-USD", price: 68512.30, chg: 1.82 },
  { sym: "ETH-USD", price: 3502.70, chg: -0.41 },
  { sym: "SPY",     price: 554.12,  chg: 0.21 },
  { sym: "QQQ",     price: 484.60,  chg: 0.74 },
  { sym: "NVDA",    price: 911.60,  chg: 1.05 },
  { sym: "AAPL",    price: 209.18,  chg: -1.01 },
  { sym: "TSLA",    price: 181.42,  chg: -1.72 },
  { sym: "MSFT",    price: 432.18,  chg: 0.62 },
];

function App() {
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);
  const [page, setPage]       = useS("dashboard");
  const [collapsed, setColl]  = useS(false);
  const [mode, setMode]       = useS(t.mode || "paper");
  const [pendingMode, setPM]  = useS(null);
  const [unread, setUnread]   = useS(3);
  const [tickers, setTickers] = useS(TICKER_INIT);
  const [feed, setFeed]       = useS(window.DB.MONITOR_FEED);

  // Apply theme + density to <html>
  useE(() => {
    document.documentElement.setAttribute("data-theme", t.theme);
    document.documentElement.setAttribute("data-density", t.density);
  }, [t.theme, t.density]);

  // Live tickers — speed driven by tweak
  useE(() => {
    if (t.activitySpeed === "off") return;
    const ms = t.activitySpeed === "fast" ? 700 : t.activitySpeed === "slow" ? 3500 : 1600;
    const id = setInterval(() => {
      setTickers((cur) =>
        cur.map((tk) => {
          const drift = (Math.random() - 0.5) * 0.0025 * tk.price;
          const newPrice = +(tk.price + drift).toFixed(2);
          return { ...tk, price: newPrice, chg: +(tk.chg + (Math.random() - 0.5) * 0.05).toFixed(2) };
        })
      );
    }, ms);
    return () => clearInterval(id);
  }, [t.activitySpeed]);

  // Live feed pushes new lines on monitors page
  useE(() => {
    if (t.activitySpeed === "off") return;
    const ms = t.activitySpeed === "fast" ? 1500 : t.activitySpeed === "slow" ? 6000 : 3000;
    const sources = ["MARKET","SIGNAL","EXEC","RISK","NEWS","ML"];
    const msgs = {
      MARKET: ["BTC-USD tick {p} (vol+{v}%)", "ETH-USD tick {p} (vol-{v}%)", "SPY tick {p} (vol+{v}%)", "NVDA tick {p} (vol+{v}%)"],
      SIGNAL: ["NVDA: long-entry confidence {c}", "TSLA: short-entry confidence {c}", "AAPL: hold (low conviction)"],
      EXEC: ["Filled 80 NVDA @ {p}", "Closed 22 MSFT @ {p}", "Partial fill 0.42 BTC @ {p}"],
      RISK: ["Exposure 42% / limit 70%", "Daily DD 1.8% / limit 5.0%"],
      NEWS: ["AAPL: -0.71 sentiment (FT)", "NVDA: +0.42 sentiment (Bloomberg)"],
      ML: ["LSTM forecast batch done ({v}ms)", "Regime classifier: trend↑ p={c}"],
    };
    const id = setInterval(() => {
      const src = sources[Math.floor(Math.random() * sources.length)];
      const tmpl = msgs[src][Math.floor(Math.random() * msgs[src].length)];
      const msg = tmpl
        .replaceAll("{p}", (50 + Math.random()*900).toFixed(2))
        .replaceAll("{v}", (Math.random()*3).toFixed(1))
        .replaceAll("{c}", (0.5 + Math.random()*0.45).toFixed(2));
      setFeed((cur) => [{ t: Date.now(), src, msg }, ...cur].slice(0, 80));
    }, ms);
    return () => clearInterval(id);
  }, [t.activitySpeed]);

  function tryMode(next) {
    if (next === "live") { setPM("live"); }
    else { setMode("paper"); setTweak("mode", "paper"); }
  }
  function confirmLive() { setMode("live"); setTweak("mode", "live"); setPM(null); }

  // Auto-collapse on small screens
  useE(() => {
    const onResize = () => { if (window.innerWidth < 1280) setColl(true); };
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const Pages = window.Pages;
  let content;
  switch (page) {
    case "dashboard": content = <Pages.DashboardPage tickers={tickers} />; break;
    case "trader":    content = <Pages.TraderPage />; break;
    case "monitors":  content = <Pages.MonitorsPage feedItems={feed} />; break;
    case "analytics": content = <Pages.AnalyticsPage />; break;
    case "history":   content = <Pages.HistoryPage />; break;
    case "alerts":    content = <Pages.AlertsPage onAck={()=>setUnread(u=>Math.max(0,u-1))} />; break;
    case "ml":        content = <Pages.MLPage />; break;
    case "backtest":  content = <Pages.BacktestPage />; break;
    case "training":  content = <Pages.TrainingPage />; break;
    case "settings":  content = <Pages.SettingsPage tweaks={t} setTweak={setTweak} />; break;
    default:          content = <Pages.DashboardPage tickers={tickers} />;
  }

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <div className="brand" style={{ width: collapsed ? "auto" : undefined, marginRight: collapsed ? 8 : -16 }}>
          <div className="brand-mark" />
          {!collapsed && <span>Dashbot</span>}
        </div>
        <div className="header-divider" />

        {/* Mode selector */}
        <div className="mode-selector" title="Trading mode">
          <button className={"paper " + (mode === "paper" ? "active" : "")} onClick={()=>tryMode("paper")}>● Paper</button>
          <button className={"live "  + (mode === "live"  ? "active" : "")} onClick={()=>tryMode("live")}>● Live</button>
        </div>

        <div className="conn-status">
          <span className="conn-dot" />
          <span style={{ color: "var(--text-1)" }}>WS connected</span>
          <span className="muted">· 42ms</span>
        </div>

        <div className="ticker-strip">
          {tickers.map(tk => (
            <span key={tk.sym} className="ticker-item">
              <span className="ticker-sym">{tk.sym}</span>
              <span style={{ fontVariantNumeric: "tabular-nums" }}>{tk.price.toFixed(2)}</span>
              <span className={tk.chg >= 0 ? "gain" : "loss"} style={{ fontSize: 10.5 }}>
                {tk.chg >= 0 ? "▲" : "▼"} {Math.abs(tk.chg).toFixed(2)}%
              </span>
            </span>
          ))}
        </div>

        <div className="header-right">
          <button className="icon-btn" title="Search"><Icon2 name="search" size={15} /></button>
          <button className="icon-btn" title="Alerts" onClick={()=>setPage("alerts")}>
            <Icon2 name="alerts" size={15} />
            {unread > 0 && <span className="badge">{unread}</span>}
          </button>
          <div className="user-avatar">T</div>
        </div>
      </div>

      {/* Sidebar */}
      <div className={"sidebar" + (collapsed ? " collapsed" : "")}>
        <div className="nav-list">
          {NAV.map((n, i) => {
            if (n.section) return collapsed ? <div key={i} style={{ height: 8 }} /> : <div key={i} className="nav-section">{n.section}</div>;
            return (
              <button key={n.id} className={"nav-item" + (page === n.id ? " active" : "")} onClick={()=>setPage(n.id)} title={collapsed ? n.label : undefined}>
                <span className="icon"><Icon2 name={n.icon} size={15} /></span>
                <span>{n.label}</span>
                {n.id === "alerts" && unread > 0 && !collapsed && (
                  <span style={{ marginLeft: "auto", background: "var(--loss)", color: "#fff", fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 8 }}>{unread}</span>
                )}
              </button>
            );
          })}
        </div>
        <div className="sidebar-footer">
          <div className="user-avatar" style={{ width: 22, height: 22, fontSize: 9 }}>T</div>
          {!collapsed && <div className="sidebar-footer-text"><div style={{ fontSize: 11.5 }}>Trader</div><div className="muted" style={{ fontSize: 10.5 }}>admin@admin.com</div></div>}
          <button className="sidebar-footer-toggle" onClick={()=>setColl(!collapsed)} title={collapsed ? "Expand" : "Collapse"}>
            <Icon2 name={collapsed ? "chevron-r" : "chevron-l"} size={13} />
          </button>
        </div>
      </div>

      {content}

      {/* Live mode confirmation */}
      {pendingMode === "live" && (
        <div className="modal-back" onClick={()=>setPM(null)}>
          <div className="modal" onClick={(e)=>e.stopPropagation()}>
            <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="sev high" style={{ marginTop: 0, width: 10, height: 10 }} />
              Switch to LIVE trading?
            </h3>
            <p>You are about to put real capital at risk. The ML trader will execute orders against your connected brokerage account using current model weights ({window.DB.ML_TRADER.modelVersion}).</p>
            <div style={{ background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 6, padding: 10, marginBottom: 14, fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--text-1)" }}>
              <div>Account: <span style={{ color: "var(--text-0)" }}>•••••4821</span></div>
              <div>Available: <span style={{ color: "var(--text-0)" }}>{fmtUSD2(184_273.48)}</span></div>
              <div>Daily DD limit: <span style={{ color: "var(--warn)" }}>5.0% ({fmtUSD2(9_213.67)})</span></div>
            </div>
            <div className="modal-actions">
              <button className="btn ghost" onClick={()=>setPM(null)}>Cancel</button>
              <button className="btn" style={{ background: "var(--loss)", borderColor: "var(--loss)", color: "#fff" }} onClick={confirmLive}>I understand · Switch to LIVE</button>
            </div>
          </div>
        </div>
      )}

      {/* Tweaks panel */}
      <TweaksPanel>
        <TweakSection label="Theme" />
        <TweakRadio label="Palette" value={t.theme} options={["navy","black","charcoal"]} onChange={(v)=>setTweak("theme", v)} />
        <TweakRadio label="Density" value={t.density} options={["compact","regular","comfy"]} onChange={(v)=>setTweak("density", v)} />

        <TweakSection label="Activity" />
        <TweakRadio label="Live updates" value={t.activitySpeed} options={["off","slow","normal","fast"]} onChange={(v)=>setTweak("activitySpeed", v)} />

        <TweakSection label="System" />
        <TweakRadio label="Mode" value={mode} options={["paper","live"]} onChange={(v)=>tryMode(v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

})();