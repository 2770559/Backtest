import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import uuid
import json
from datetime import datetime, timedelta

# --- Version ---
APP_VERSION = "1.1.0"  # semver: major.minor.patch
APP_BUILD_DATE = "2026-03-11"

# --- 1. Page Config ---
st.set_page_config(page_title="Portfolio Backtest", layout="wide", page_icon="📊")

# --- Custom CSS: KPI Dashboard + Responsive Design ---
st.markdown("""
<style>
/* ===== Global Typography & Spacing (responsive-design skill) ===== */
section.main > div { max-width: 1400px; margin: 0 auto; }

h1, h2, h3, h4 { letter-spacing: -0.02em; }

/* ===== Header Bar ===== */
.header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 1.25rem 1.75rem;
    border-radius: 0.75rem;
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.header-bar h2 { margin: 0; font-size: clamp(1.1rem, 2.5vw, 1.5rem); font-weight: 700; }
.header-bar .subtitle { opacity: 0.7; font-size: 0.85rem; }
.header-bar .version-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 1rem;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    white-space: nowrap;
}

/* ===== KPI Metric Cards (kpi-dashboard-design skill) ===== */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
@media (max-width: 900px) {
    .kpi-row { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 600px) {
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
}

.kpi-card {
    background: #ffffff;
    border-radius: 0.6rem;
    padding: 0.875rem 1rem;
    border-left: 4px solid #e0e0e0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: transform 0.15s, box-shadow 0.15s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.kpi-card .label {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
    font-weight: 600;
}
.kpi-card .value {
    font-size: clamp(1rem, 2vw, 1.35rem);
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.2;
}

/* Border color variants */
.kpi-card.green  { border-left-color: #10b981; }
.kpi-card.red    { border-left-color: #ef4444; }
.kpi-card.blue   { border-left-color: #3b82f6; }
.kpi-card.purple { border-left-color: #8b5cf6; }
.kpi-card.amber  { border-left-color: #f59e0b; }
.kpi-card.gray   { border-left-color: #6b7280; }

/* ===== Portfolio Group Title ===== */
.port-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1a1a2e;
    padding: 0.5rem 0.75rem;
    background: #f1f5f9;
    border-radius: 0.4rem;
    margin: 0.75rem 0 0.5rem 0;
    display: inline-block;
}
.port-title.bench { background: #e0e7ff; color: #3730a3; }

/* ===== Comparison Table ===== */
.cmp-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85rem;
    margin: 0.75rem 0 1.25rem 0;
    border-radius: 0.5rem;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.cmp-table th {
    background: #f8fafc;
    padding: 0.6rem 1rem;
    text-align: left;
    font-weight: 600;
    color: #475569;
    border-bottom: 2px solid #e2e8f0;
}
.cmp-table td {
    padding: 0.55rem 1rem;
    border-bottom: 1px solid #f1f5f9;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table tr:hover td { background: #f8fafc; }
.cmp-table .best { color: #10b981; font-weight: 700; }
.cmp-table .worst { color: #ef4444; }

/* ===== Portfolio Config Cards ===== */
.config-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.6rem;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

/* ===== Section Divider ===== */
.section-gap { margin: 1.5rem 0 0.75rem 0; }

/* ===== Hide default Streamlit metric styling for cleaner look ===== */
[data-testid="stMetric"] {
    background: #f8fafc;
    border-radius: 0.5rem;
    padding: 0.75rem;
    border-left: 3px solid #3b82f6;
}

/* ===== Button Styling ===== */
/* Delete button: compact red circle */
button[kind="secondary"][data-testid="stBaseButton-secondary"]:has(p) {
    /* fallback handled by .del-btn class below */
}

/* Primary action button (Analyze) — match secondary button size */
button[kind="primary"] {
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
    font-size: inherit !important;
    letter-spacing: 0.02em;
    padding: 0.5rem 1.5rem !important;
    min-height: 0 !important;
    line-height: normal !important;
    transition: transform 0.1s, box-shadow 0.15s !important;
}
button[kind="primary"] div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.35rem !important;
    flex-direction: row !important;
}
button[kind="primary"] p {
    display: inline !important;
    margin: 0 !important;
}
button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.35) !important;
}

/* Secondary buttons (Add Portfolio, etc.) */
button[kind="secondary"] {
    border-radius: 0.5rem !important;
    border: 1.5px solid #e2e8f0 !important;
    transition: all 0.15s !important;
}
button[kind="secondary"]:hover {
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
    background: #eff6ff !important;
}

/* ===== Sidebar refinements ===== */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
<div class="header-bar">
    <div>
        <h2>Portfolio Backtest & Rebalance Analyzer</h2>
        <div class="subtitle">Multi-portfolio backtesting with rebalancing strategies</div>
    </div>
    <div class="version-badge">v{APP_VERSION}</div>
</div>
""", unsafe_allow_html=True)

# Ticker short-name mapping (for CN-listed ETFs display)
TICKER_TO_NAME = {
    "159941.SZ": "NQ ETF",    # NASDAQ ETF
    "513500.SS": "SP500",      # S&P 500 ETF
    "512890.SS": "DivLV",      # Dividend Low-Vol
    "512400.SS": "Metal",      # Non-ferrous Metals
    "515220.SS": "Coal",       # Coal ETF
    "588080.SS": "STAR50",     # STAR Market 50
    "518880.SS": "Gold",       # Gold ETF
    "510300.SS": "CSI300",     # CSI 300
    "511130.SS": "30YBd"       # 30-Year Treasury Bond
}

if 'run_backtest' not in st.session_state:
    st.session_state.run_backtest = False

# --- Init Session State ---
if 'bi' not in st.session_state: st.session_state['bi'] = "SPY"
if 'sd' not in st.session_state: st.session_state['sd'] = datetime(2020, 1, 1)
if 'init_funds' not in st.session_state: st.session_state['init_funds'] = 10000

# Strategy name constants
STRAT_BH       = "Buy & Hold"
STRAT_ANNUAL   = "Periodic (Annual)"       # Rebalance every 365 days
STRAT_SEMI     = "Periodic (Semi-Annual)"  # Rebalance every 180 days
STRAT_RD_LOCAL = "RelDiff Local"           # Relative-diff local rebalance
STRAT_RD_MIXED = "RelDiff Mixed"           # Relative-diff mixed rebalance
STRAT_RD_FULL  = "RelDiff Full"            # Relative-diff global rebalance
STRAT_ASYM     = "Asymmetric RelDiff"      # Asymmetric relative-diff rebalance

if 'portfolios_list' not in st.session_state:
    st.session_state.portfolios_list = [
        {
            "id": str(uuid.uuid4()),
            "name": "Port A",
            "tickers": "QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, ETH-USD",
            "weights": "0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05",
            "strat": STRAT_ASYM,
            "thr": 38
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Port B",
            "tickers": "159941.SZ, 512890.SS, 515220.SS, 588080.SS, 518880.SS, 511130.SS",
            "weights": "0.35, 0.15, 0.10, 0.05, 0.15, 0.20",
            "strat": STRAT_RD_MIXED,
            "thr": 38
        }
    ]

def delete_portfolio(idx):
    if 0 <= idx < len(st.session_state.portfolios_list):
        st.session_state.portfolios_list.pop(idx)

# --- 2. Core Algorithms ---
def clean_ticker(t):
    t = t.strip().upper()
    mapping = {"BRK.B": "BRK-B", "ETHUSD": "ETH-USD", "BTCUSD": "BTC-USD"}
    return mapping.get(t, t)

def calculate_metrics(nav_series, rebalance_count, risk_free_rate=0.02):
    empty = {"final_nav": "-", "total_ret": "-", "ann_ret": "-", "max_dd": "-", "sharpe": "-", "rebal_cnt": "-",
             "_total_ret": 0, "_ann_ret": 0, "_max_dd": 0, "_sharpe": 0}
    if nav_series is None or nav_series.empty or len(nav_series) < 2:
        return empty
    try:
        nav = nav_series.dropna()
        if len(nav) < 2: return empty
        total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
        days = (nav.index[-1] - nav.index[0]).days
        if days <= 0: return empty
        years = days / 365.25
        ann_return = (1 + total_return) ** (1 / years) - 1
        rolling_max = nav.cummax()
        max_dd = ((nav - rolling_max) / rolling_max).min()
        daily_ret = nav.pct_change().dropna()
        median_gap = np.median(np.diff(nav.index).astype('timedelta64[D]').astype(int))
        if median_gap <= 5:
            ann_factor = 252
        elif median_gap <= 10:
            ann_factor = 52
        else:
            ann_factor = 12
        ann_vol = daily_ret.std() * np.sqrt(ann_factor)
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        return {
            "final_nav": f"{nav.iloc[-1]:,.2f}",
            "total_ret": f"{total_return:.2%}",
            "ann_ret": f"{ann_return:.2%}",
            "max_dd": f"{max_dd:.2%}",
            "sharpe": f"{sharpe:.2f}",
            "rebal_cnt": int(rebalance_count),
            "_total_ret": total_return,
            "_ann_ret": ann_return,
            "_max_dd": max_dd,
            "_sharpe": sharpe,
        }
    except Exception:
        return {"final_nav": "Err", "total_ret": "Err", "ann_ret": "Err", "max_dd": "Err", "sharpe": "Err", "rebal_cnt": "Err",
                "_total_ret": 0, "_ann_ret": 0, "_max_dd": 0, "_sharpe": 0}

def apply_local_rebalance(asset_values, target_weights, threshold):
    total_val = asset_values.sum()
    current_vals = asset_values.copy()
    reset_indices = []
    safe_targets = target_weights.replace(0, 1e-9)
    for _ in range(10):
        current_weights = current_vals / total_val
        rel_diffs = np.abs(current_weights - target_weights) / safe_targets
        to_trigger = (rel_diffs > threshold) & (~current_vals.index.isin(reset_indices))
        if not to_trigger.any(): break
        triggered_indices = to_trigger.index[to_trigger].tolist()
        reset_indices.extend(triggered_indices)
        for idx in triggered_indices: current_vals[idx] = target_weights[idx] * total_val
        rem_indices = [i for i in current_vals.index if i not in reset_indices]
        if not rem_indices: return total_val * target_weights
        rem_cash = total_val - current_vals[reset_indices].sum()
        current_rem_sum = asset_values[rem_indices].sum()
        if current_rem_sum > 0: ratios = asset_values[rem_indices] / current_rem_sum
        else: ratios = target_weights[rem_indices] / target_weights[rem_indices].sum()
        current_vals[rem_indices] = ratios * rem_cash
    return current_vals

def run_detailed_backtest(strategy_name, price_df, target_weights, initial_cap, threshold):
    tickers = price_df.columns
    if price_df.empty: return pd.DataFrame(), 0, {}
    start_prices = price_df.iloc[0]
    if start_prices.isna().any():
        start_prices = price_df.bfill().iloc[0]
        if start_prices.isna().any(): return pd.DataFrame(), 0, {}

    current_shares = (initial_cap * target_weights) / start_prices
    history = []
    last_rebalance_date = price_df.index[0]
    rebalance_count = 0
    price_df_filled = price_df.ffill()

    cumulative_pnl = pd.Series(0.0, index=tickers)
    prev_prices = start_prices

    for i in range(len(price_df)):
        current_date = price_df.index[i]
        current_prices = price_df_filled.iloc[i]

        if i > 0: cumulative_pnl += current_shares * (current_prices - prev_prices)
        prev_prices = current_prices

        asset_values = current_shares * current_prices
        total_val = asset_values.sum()
        if total_val == 0 or np.isnan(total_val): continue
        current_weights = asset_values / total_val

        if i == 0:
            rec = {"Date": current_date, "Type": "Init", "NAV": total_val}
            rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec); continue

        do_rebalance = False
        new_values = asset_values.copy()

        if strategy_name == STRAT_ANNUAL:
            if (current_date - last_rebalance_date).days >= 365:
                new_values, do_rebalance = total_val * target_weights, True
        elif strategy_name == STRAT_SEMI:
            if (current_date - last_rebalance_date).days >= 180:
                new_values, do_rebalance = total_val * target_weights, True

        elif strategy_name == STRAT_ASYM:
            diff_ratio = (current_weights - target_weights) / target_weights.replace(0, 1e-9)
            mask_major = target_weights >= 0.06
            mask_minor = target_weights < 0.06

            trigger_major = mask_major & (np.abs(diff_ratio) > threshold)
            trigger_minor_up = mask_minor & (diff_ratio > threshold * 2.5)
            trigger_minor_down = mask_minor & (diff_ratio < -threshold * 1.25)

            if trigger_major.any() or trigger_minor_up.any() or trigger_minor_down.any():
                new_values = total_val * target_weights
                do_rebalance = True

        elif "RelDiff" in strategy_name:
            rel_diffs = np.abs(current_weights - target_weights) / target_weights.replace(0, 1e-9)
            if rel_diffs.max() > threshold:
                if strategy_name == STRAT_RD_FULL:
                    new_values = total_val * target_weights
                    do_rebalance = True
                elif strategy_name == STRAT_RD_MIXED:
                    if ((target_weights >= 0.1) & (rel_diffs > threshold)).any():
                        new_values = total_val * target_weights
                    else:
                        new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                    do_rebalance = True
                elif strategy_name == STRAT_RD_LOCAL:
                    new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                    do_rebalance = True

        if do_rebalance:
            rebalance_count += 1
            pre_rec = {"Date": current_date, "Type": "Pre-Rebal", "NAV": total_val}
            pre_rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(pre_rec)
            current_shares, last_rebalance_date = new_values / current_prices, current_date
            post_rec = {"Date": current_date, "Type": "Post-Rebal", "NAV": total_val}
            post_rec.update({f"{t}": f"{(new_values/total_val)[t]:.2%}" for t in tickers})
            history.append(post_rec)
        else:
            rec = {"Date": current_date, "Type": "Hold", "NAV": total_val}
            rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec)

    total_pnl = cumulative_pnl.sum()
    pct_pnl = cumulative_pnl / total_pnl if total_pnl != 0 else cumulative_pnl * 0
    pnl_rec = {"Date": "Overall", "Type": "PnL Contrib%", "NAV": float(total_pnl)}
    pnl_rec.update({f"{t}": f"{pct_pnl[t]:.2%}" for t in tickers})

    return pd.DataFrame(history), rebalance_count, pnl_rec

@st.cache_data(ttl=86400)
def fetch_cpi_data():
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
        cpi = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date')
        cpi.columns = ['CPI']
        return cpi
    except Exception:
        return None

def render_kpi_cards(name, m, is_bench=False):
    """Render a row of styled KPI cards (kpi-dashboard-design pattern)"""
    title_cls = "bench" if is_bench else ""
    st.markdown(f'<div class="port-title {title_cls}">{name}</div>', unsafe_allow_html=True)

    # Determine card colors based on values
    ret_color = "green" if isinstance(m.get("_total_ret"), (int, float)) and m["_total_ret"] > 0 else "red"
    ann_color = "green" if isinstance(m.get("_ann_ret"), (int, float)) and m["_ann_ret"] > 0 else "red"
    dd_color = "red" if isinstance(m.get("_max_dd"), (int, float)) and m["_max_dd"] < -0.2 else "amber"
    sharpe_color = "green" if isinstance(m.get("_sharpe"), (int, float)) and m["_sharpe"] > 1 else ("amber" if isinstance(m.get("_sharpe"), (int, float)) and m["_sharpe"] > 0 else "red")

    cards_html = f"""
    <div class="kpi-row">
        <div class="kpi-card blue">
            <div class="label">Final Value</div>
            <div class="value">${m['final_nav']}</div>
        </div>
        <div class="kpi-card {ret_color}">
            <div class="label">Total Return</div>
            <div class="value">{m['total_ret']}</div>
        </div>
        <div class="kpi-card {ann_color}">
            <div class="label">Ann. Return</div>
            <div class="value">{m['ann_ret']}</div>
        </div>
        <div class="kpi-card {dd_color}">
            <div class="label">Max Drawdown</div>
            <div class="value">{m['max_dd']}</div>
        </div>
        <div class="kpi-card {sharpe_color}">
            <div class="label">Sharpe Ratio</div>
            <div class="value">{m['sharpe']}</div>
        </div>
        <div class="kpi-card gray">
            <div class="label">Rebalances</div>
            <div class="value">{m['rebal_cnt']}</div>
        </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

def render_comparison_table(metrics):
    """Render a side-by-side comparison table (kpi-dashboard-design: executive summary pattern)"""
    if len(metrics) < 2:
        return

    fields = [("Total Return", "total_ret", "_total_ret", True),
              ("Ann. Return", "ann_ret", "_ann_ret", True),
              ("Max Drawdown", "max_dd", "_max_dd", False),
              ("Sharpe Ratio", "sharpe", "_sharpe", True),
              ("Rebalances", "rebal_cnt", None, None)]

    header = "<tr><th>Metric</th>"
    for m in metrics:
        header += f"<th>{m['name']}</th>"
    header += "</tr>"

    rows = ""
    for label, key, raw_key, higher_better in fields:
        rows += f"<tr><td><strong>{label}</strong></td>"
        raw_vals = []
        for m in metrics:
            if raw_key and isinstance(m.get(raw_key), (int, float)):
                raw_vals.append(m[raw_key])
            else:
                raw_vals.append(None)

        for i, m in enumerate(metrics):
            val = m[key]
            cls = ""
            if higher_better is not None and all(v is not None for v in raw_vals) and len(raw_vals) >= 2:
                if higher_better:
                    if raw_vals[i] == max(raw_vals): cls = "best"
                    elif raw_vals[i] == min(raw_vals): cls = "worst"
                else:
                    if raw_vals[i] == max(raw_vals): cls = "best"
                    elif raw_vals[i] == min(raw_vals): cls = "worst"
            rows += f'<td class="{cls}">{val}</td>'
        rows += "</tr>"

    st.markdown(f'<table class="cmp-table">{header}{rows}</table>', unsafe_allow_html=True)


# --- 3. Sidebar: Global Settings ---
with st.sidebar:
    st.markdown("### Settings")

    bench_in = st.text_input("Benchmark Ticker", value=st.session_state['bi'])
    start_d = st.date_input(
        "Start Date",
        value=st.session_state['sd'],
        min_value=datetime(1970, 1, 1).date(),
        max_value=datetime.today().date()
    )
    init_f = st.number_input("Initial Investment ($)", value=st.session_state['init_funds'], step=1000)

    st.session_state['bi'] = bench_in
    st.session_state['sd'] = start_d
    st.session_state['init_funds'] = init_f

    st.divider()
    st.markdown("### Inflation")
    inf_adj = st.checkbox("Enable CPI Adjustment", value=False)

    st.divider()
    st.markdown("### Config I/O")
    current_config = {
        "benchmark": st.session_state['bi'],
        "start_date": str(st.session_state['sd']),
        "initial_funds": st.session_state['init_funds'],
        "portfolios": st.session_state.portfolios_list
    }
    json_str = json.dumps(current_config, indent=2, ensure_ascii=False)
    col_io1, col_io2 = st.columns(2)
    with col_io1:
        st.download_button(label=":material/download: Export", data=json_str, file_name="backtest_config.json", mime="application/json", use_container_width=True)

    uploaded_file = st.file_uploader("Import Config", type=["json"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            loaded_config = json.load(uploaded_file)
            if st.button(":material/upload: Apply Config", use_container_width=True):
                st.session_state.portfolios_list = loaded_config.get("portfolios", [])
                strat_legacy_map = {
                    "买入持有": STRAT_BH,
                    "定期再平衡(年)": STRAT_ANNUAL,
                    "定期再平衡(半年)": STRAT_SEMI,
                    "相对差局部再平衡": STRAT_RD_LOCAL,
                    "相对差混合再平衡": STRAT_RD_MIXED,
                    "相对差全局再平衡": STRAT_RD_FULL,
                    "不对称相对差再平衡": STRAT_ASYM,
                }
                for p in st.session_state.portfolios_list:
                    if 'id' not in p: p['id'] = str(uuid.uuid4())
                    if p.get('strat') in strat_legacy_map:
                        p['strat'] = strat_legacy_map[p['strat']]
                st.session_state['bi'] = loaded_config.get("benchmark", "SPY")
                st.session_state['sd'] = pd.to_datetime(loaded_config.get("start_date", "2020-01-01")).date()
                st.session_state['init_funds'] = int(loaded_config.get("initial_funds", 10000))
                st.rerun()
        except Exception as e:
            st.error(f"Parse error: {e}")

# --- 4. Main Area: Portfolio Config ---
strategy_options = [
    STRAT_BH, STRAT_ANNUAL, STRAT_SEMI,
    STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_RD_FULL, STRAT_ASYM
]

# Column headers
hdr = st.columns([0.8, 3, 2, 2, 0.8, 0.4])
with hdr[0]: st.caption("Name")
with hdr[1]: st.caption("Tickers")
with hdr[2]: st.caption("Weights")
with hdr[3]: st.caption("Strategy")
with hdr[4]: st.caption("Thr%")

total_portfolios = len(st.session_state.portfolios_list)
for i, port in enumerate(st.session_state.portfolios_list):
    if 'id' not in port: port['id'] = str(uuid.uuid4())

    cols = st.columns([0.8, 3, 2, 2, 0.8, 0.4])
    with cols[0]: st.markdown(f"**{port['name']}**")
    with cols[1]: port['tickers'] = st.text_input("Tickers", port['tickers'], key=f"t_{port['id']}", label_visibility="collapsed")
    with cols[2]: port['weights'] = st.text_input("Weights", port['weights'], key=f"w_{port['id']}", label_visibility="collapsed")
    with cols[3]: port['strat'] = st.selectbox("Strategy", strategy_options, index=strategy_options.index(port['strat']), key=f"s_{port['id']}", label_visibility="collapsed")
    with cols[4]: port['thr'] = st.number_input("Thr%", 1, 200, port['thr'], key=f"tr_{port['id']}", label_visibility="collapsed")
    with cols[5]:
        if total_portfolios > 1:
            st.button(":material/delete:", key=f"del_{port['id']}", on_click=delete_portfolio, args=(i,))

btn_cols = st.columns([1, 1.5, 5.5])
with btn_cols[0]:
    if st.button(":material/add_circle: Add", use_container_width=True):
        existing_names = set(p["name"] for p in st.session_state.portfolios_list)
        new_char_code = 65
        while f"Port {chr(new_char_code)}" in existing_names: new_char_code += 1
        last_port = st.session_state.portfolios_list[-1]
        st.session_state.portfolios_list.append({
            "id": str(uuid.uuid4()), "name": f"Port {chr(new_char_code)}",
            "tickers": last_port["tickers"], "weights": last_port["weights"],
            "strat": STRAT_ASYM, "thr": 38
        })
        st.rerun()
with btn_cols[1]:
    run_clicked = st.button(":material/play_arrow: Analyze", type="primary", use_container_width=True)
    if run_clicked:
        validation_pass = True
        error_msgs = []
        for p in st.session_state.portfolios_list:
            t_str = p['tickers'].replace("\uff0c", ",")  # fullwidth comma
            w_str = p['weights'].replace("\uff0c", ",")
            t_list = [x.strip() for x in t_str.split(',') if x.strip()]
            w_list = [x.strip() for x in w_str.split(',') if x.strip()]
            if len(t_list) != len(w_list):
                validation_pass = False
                error_msgs.append(f"**{p['name']}**: {len(t_list)} tickers vs {len(w_list)} weights")
                continue
            try:
                w_floats = [float(w) for w in w_list]
                total_w = sum(w_floats)
                if abs(total_w - 1.0) > 0.01:
                    validation_pass = False
                    error_msgs.append(f"**{p['name']}**: weights sum = {total_w:.2f}, should be 1.0")
            except ValueError:
                validation_pass = False
                error_msgs.append(f"**{p['name']}**: invalid weight format")
        if validation_pass:
            st.session_state.run_backtest = True
        else:
            for msg in error_msgs: st.error(msg)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

# --- 5. Results ---
if st.session_state.run_backtest:
    with st.spinner('Fetching data & running backtest...'):
        all_tks = list(set([clean_ticker(bench_in)] + [clean_ticker(t) for p in st.session_state.portfolios_list for t in p['tickers'].replace("\uff0c", ",").split(",")]))
        try:
            df_raw = yf.download(all_tks, start=start_d - timedelta(days=20), progress=False)
        except Exception as e:
            st.error(f"Download error: {e}"); st.stop()
        if df_raw.empty: st.error("Download failed."); st.stop()

        if 'Adj Close' in df_raw.columns.get_level_values(0): price_data = df_raw['Adj Close']
        elif 'Close' in df_raw.columns.get_level_values(0): price_data = df_raw['Close']
        else: price_data = df_raw
        for tk in all_tks:
            if tk not in price_data.columns: price_data[tk] = np.nan

        bench_tk = clean_ticker(bench_in)
        if bench_tk not in price_data.columns: st.error(f"Benchmark {bench_tk} not found."); st.stop()

        bench_valid_days = price_data[bench_tk].dropna().index
        future_days = bench_valid_days[bench_valid_days >= pd.Timestamp(start_d)]
        if future_days.empty: st.error(f"No data after {start_d}."); st.stop()
        market_start_day = future_days[0]

        # ffill on full calendar first so weekend crypto prices carry to next trading day
        price_data_prefilled = price_data.ffill()
        df_aligned = price_data_prefilled.reindex(bench_valid_days)
        df_aligned = df_aligned[df_aligned.index >= market_start_day]
        df_filled = df_aligned.ffill().bfill()

        all_port_tks = [clean_ticker(t) for p in st.session_state.portfolios_list for t in p['tickers'].replace("\uff0c", ",").split(",")]
        raw_aligned = df_aligned[all_port_tks]
        first_valid_idx = raw_aligned.apply(lambda x: x.first_valid_index())
        bottleneck_date = first_valid_idx.max()
        bottleneck_ticker = first_valid_idx.idxmax()
        actual_start_day = market_start_day

        days_diff_bench = (market_start_day.date() - pd.Timestamp(start_d).date()).days
        if days_diff_bench > 7:
            st.warning(f"Benchmark **{bench_tk}** not listed until {market_start_day.date()}, start date adjusted.")
        elif days_diff_bench > 0:
            if pd.notna(bottleneck_date) and (bottleneck_date - market_start_day).days > 7:
                actual_start_day = bottleneck_date
                st.warning(f"**{bottleneck_ticker}** listed late ({bottleneck_date.date()}), start date aligned.")
            else:
                st.info(f"Aligned to next trading day: {market_start_day.date()}")
        else:
            if pd.notna(bottleneck_date) and (bottleneck_date - market_start_day).days > 7:
                actual_start_day = bottleneck_date
                st.warning(f"**{bottleneck_ticker}** listed late ({bottleneck_date.date()}), start date adjusted.")

        final_data = df_filled[df_filled.index >= actual_start_day]
        if final_data.empty: st.error("Insufficient data."); st.stop()
        days_span = (final_data.index[-1] - final_data.index[0]).days

        if days_span < 90:
            price_df = final_data.copy()
        else:
            first_row = final_data.iloc[[0]]
            monthly_rows = final_data.resample('ME').last()
            price_df = pd.concat([first_row, monthly_rows]).sort_index()
            price_df = price_df[~price_df.index.duplicated(keep='first')]

        comp_df = pd.DataFrame(index=price_df.index)
        bench_nav = (price_df[bench_tk] / price_df[bench_tk].iloc[0]) * init_f
        comp_df[f"Benchmark({bench_in})"] = bench_nav

        res_list = {}
        valid_ports_meta = {}

        def clean_col(c):
            target = c.strip()
            for tk, name in TICKER_TO_NAME.items():
                if tk in target: return name
            return target

        for p in st.session_state.portfolios_list:
            t_str = p['tickers'].replace("\uff0c", ",")
            w_str = p['weights'].replace("\uff0c", ",")
            p_tks = [clean_ticker(t) for t in t_str.split(",")]
            p_wts = [float(w) for w in w_str.split(",")]
            valid_p_tks = [t for t in p_tks if t in price_df.columns and not price_df[t].isna().all()]
            if not valid_p_tks: continue
            if len(valid_p_tks) < len(p_tks):
                w_series = pd.Series(p_wts[:len(p_tks)], index=p_tks)[valid_p_tks]
                w_series = w_series / w_series.sum()
            else: w_series = pd.Series(p_wts, index=p_tks)

            res_df, cnt, pnl_rec = run_detailed_backtest(p['strat'], price_df[valid_p_tks], w_series, init_f, p['thr']/100.0)
            if not res_df.empty:
                df_chart = res_df.drop_duplicates(subset='Date', keep='last').copy()
                df_chart['Date'] = pd.to_datetime(df_chart['Date'])
                df_chart = df_chart.set_index('Date')
                comp_df[p['name']] = df_chart['NAV']
                valid_ports_meta[p['name']] = cnt
                translated_pnl = {}
                for k, v in pnl_rec.items(): translated_pnl[clean_col(k)] = v
                pnl_df = pd.DataFrame([translated_pnl])
                df_history = res_df.iloc[::-1].rename(columns=clean_col).reset_index(drop=True)
                df_history['Date'] = pd.to_datetime(df_history['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                res_list[p['name']] = pd.concat([pnl_df, df_history], ignore_index=True)

        # --- Inflation Adjustment ---
        if inf_adj:
            start_ts = pd.Timestamp(actual_start_day)
            end_ts = comp_df.index[-1]
            cpi_raw = fetch_cpi_data()
            if cpi_raw is not None:
                cpi_yearly = cpi_raw['CPI'].resample('YS').first()
                yearly_inf = []
                for y in range(start_ts.year, end_ts.year + 1):
                    ts_curr = pd.Timestamp(f"{y}-01-01")
                    ts_next = pd.Timestamp(f"{y+1}-01-01")
                    if ts_curr in cpi_yearly.index and ts_next in cpi_yearly.index:
                        rate = (cpi_yearly[ts_next] / cpi_yearly[ts_curr] - 1) * 100
                    else:
                        rate = 3.0
                    yearly_inf.append({"Year": y, "Inflation(%)": round(rate, 2)})
                inf_df = pd.DataFrame(yearly_inf)
                with st.expander("CPI Inflation Rates (editable)", expanded=False):
                    edited_inf = st.data_editor(inf_df, hide_index=True, use_container_width=True)
                rate_map = dict(zip(edited_inf['Year'].astype(int), edited_inf['Inflation(%)'] / 100))
                discount_factors = pd.Series(1.0, index=comp_df.index)
                for i, date in enumerate(comp_df.index):
                    factor = 1.0
                    for y in range(start_ts.year, date.year + 1):
                        rate = rate_map.get(y, 0.03)
                        y_begin = max(pd.Timestamp(f"{y}-01-01"), start_ts)
                        y_end = min(pd.Timestamp(f"{y}-12-31"), date)
                        if y_begin > date or y_end < start_ts: continue
                        frac = (y_end - y_begin).days / 365.25
                        factor *= (1 + rate) ** frac
                    discount_factors.iloc[i] = factor
                for col in comp_df.columns:
                    comp_df[col] = comp_df[col] / discount_factors
            else:
                inf_rate = st.sidebar.number_input("Fixed Rate (%)", value=3.0, step=0.1, format="%.1f") / 100.0
                st.warning("CPI fetch failed, using fixed rate.")
                days_diff = (comp_df.index - start_ts).days
                discount_factors = (1 + inf_rate) ** (days_diff / 365.25)
                for col in comp_df.columns:
                    comp_df[col] = comp_df[col] / discount_factors

        # --- Metrics Calculation ---
        metrics = []
        bench_m = calculate_metrics(comp_df[f"Benchmark({bench_in})"], 0)
        bench_m["name"] = f"Benchmark({bench_in})"
        metrics.append(bench_m)
        for p_name, cnt in valid_ports_meta.items():
            m = calculate_metrics(comp_df[p_name], cnt)
            m["name"] = p_name
            metrics.append(m)

        # --- KPI Cards (kpi-dashboard-design: Executive Summary pattern) ---
        for i_m, m in enumerate(metrics):
            render_kpi_cards(m["name"], m, is_bench=(i_m == 0))

        # --- Comparison Table ---
        if len(metrics) >= 2:
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            with st.expander("Performance Comparison", expanded=True):
                render_comparison_table(metrics)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        # --- Chart ---
        comp_df.index.name = 'Date'
        chart_df = comp_df / comp_df.iloc[0] - 1
        x_axis_format = '%Y-%m-%d' if days_span < 90 else '%Y-%m'
        label_angle = -45 if days_span < 90 else 0

        # Color palette
        palette = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4']

        chart_data = chart_df.reset_index().melt('Date', var_name='Portfolio', value_name='Return')
        rule_data = chart_df.reset_index()
        nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Date'], empty=False)

        line = alt.Chart(chart_data).mark_line(strokeWidth=2.5).encode(
            x=alt.X('Date:T', axis=alt.Axis(format=x_axis_format, title=None, labelAngle=label_angle, grid=False)),
            y=alt.Y('Return:Q', axis=alt.Axis(format='.1%', title='Cumulative Return' + (' (CPI adj.)' if inf_adj else ''), grid=True, gridDash=[3,3], gridColor='#e5e7eb')),
            color=alt.Color('Portfolio:N', legend=alt.Legend(orient='top', title=None, labelFontSize=12), scale=alt.Scale(range=palette))
        )

        # Zero baseline
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#94a3b8', strokeDash=[4,4], strokeWidth=1).encode(y='y:Q')

        tooltips = [alt.Tooltip('Date:T', format='%Y-%m-%d', title='Date')]
        for col in chart_df.columns:
            tooltips.append(alt.Tooltip(field=col, type='quantitative', format='.2%', title=col))

        selectors = alt.Chart(rule_data).mark_rule(opacity=0.001, strokeWidth=40).encode(
            x='Date:T', tooltip=tooltips
        ).add_params(nearest)

        rules = alt.Chart(rule_data).mark_rule(color='#94a3b8', strokeDash=[3,3]).encode(
            x='Date:T', tooltip=tooltips
        ).transform_filter(nearest)

        points = line.mark_point(size=60, filled=True).encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        base_chart = alt.layer(zero_line, line, selectors, rules, points).properties(
            height=440
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(base_chart, use_container_width=True)

        # --- Detail Tabs ---
        if res_list:
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            tab_names = list(res_list.keys())
            tabs = st.tabs(tab_names)
            def style_row(row):
                if row['Type'] == 'PnL Contrib%': return ['background-color: #fce4ec; color: #d81b60; font-weight: bold'] * len(row)
                if row['Type'] == 'Init': return ['background-color: #e3f2fd; font-weight: bold'] * len(row)
                if row['Type'] == 'Pre-Rebal': return ['background-color: #fff3e0'] * len(row)
                if row['Type'] == 'Post-Rebal': return ['background-color: #e8f5e9'] * len(row)
                return [''] * len(row)
            for tab, lbl in zip(tabs, tab_names):
                with tab:
                    st.caption(f"Start: {actual_start_day.date()}")
                    st.dataframe(res_list[lbl].style.apply(style_row, axis=1).format({"NAV": "{:,.2f}"}), use_container_width=True)
