import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path

from backtest_core import (
    STRAT_BH, STRAT_ANNUAL, STRAT_SEMI,
    STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_RD_FULL, STRAT_ASYM, STRAT_ASYM_LOCAL,
    STRAT_ASYM_LOCAL_EQ, STRAT_ASYM_LOCAL_PROP,
    STRAT_LEGACY_MAP,
    clean_ticker, parse_portfolio, calculate_metrics,
    run_detailed_backtest, compute_annual_returns,
    scrub_leading_glitches, scrub_isolated_spikes, sample_monthly,
)

# --- Version ---
APP_VERSION = "1.5.0"  # semver: major.minor.patch
APP_BUILD_DATE = "2026-07-06"

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
.cmp-table .cagr-row td { background: #f8fafc; font-weight: 600; border-top: 2px solid #e2e8f0; }

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

if 'portfolios_list' not in st.session_state:
    st.session_state.portfolios_list = [
        {
            # Crypto sleeve split into a composite (ETH-USD, MSTR): same 5% slot ->
            # 2.5% / 2.5%. Rebalanced with RelDiff Mixed at a 40% trigger band.
            "id": str(uuid.uuid4()),
            "name": "Port A",
            "tickers": "QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, (ETH-USD, MSTR)",
            "weights": "0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05",
            "strat": STRAT_RD_MIXED,
            "thr": 40
        },
        {
            # Port A with the crypto sleeve as a plain ETH-USD slot (5%), rebalanced
            # with Asymmetric RelDiff at a 38% trigger band. Weights identical to Port A.
            "id": str(uuid.uuid4()),
            "name": "Port B",
            "tickers": "QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, ETH-USD",
            "weights": "0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05",
            "strat": STRAT_ASYM,
            "thr": 38
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Port C",
            "tickers": "159941.SZ, 512890.SS, 515220.SS, 588080.SS, 518880.SS, 511130.SS",
            "weights": "0.35, 0.15, 0.10, 0.05, 0.15, 0.20",
            "strat": STRAT_RD_MIXED,
            "thr": 38
        }
    ]

def delete_portfolio(idx):
    if 0 <= idx < len(st.session_state.portfolios_list):
        st.session_state.portfolios_list.pop(idx)

# --- 2. Data Fetch & Validation Helpers ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(tickers, start):
    """Cached wrapper around yf.download so widget interactions don't re-hit Yahoo."""
    return yf.download(list(tickers), start=start, progress=False)

def validate_inputs(portfolios, benchmark):
    """Validate benchmark + all portfolio configs. Returns list of error messages."""
    errors = []
    if not str(benchmark).strip():
        errors.append("Benchmark ticker is empty")
    names = [p['name'] for p in portfolios]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    if dup_names:
        errors.append("Duplicate portfolio names: " + ", ".join(dup_names))
    for p in portfolios:
        _, _, perrs, _ = parse_portfolio(p)
        errors.extend(f"**{p['name']}**: {e}" for e in perrs)
    return errors

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


def render_annual_returns_table(comp_df, metrics):
    """HTML table: rows = calendar years (newest first) + CAGR; cols = each series in comp_df."""
    rows_data = compute_annual_returns(comp_df)
    if not rows_data:
        return
    cols = list(comp_df.columns)

    header = "<tr><th>Year</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"

    body = ""
    for r in sorted(rows_data, key=lambda x: -x["year"]):
        raw_vals = [r["returns"][c] for c in cols]
        valid = [v for v in raw_vals if v is not None]
        hi, lo = (max(valid), min(valid)) if len(valid) >= 2 else (None, None)
        year_lbl = f"{r['year']}*" if r["partial"] else str(r["year"])
        body += f"<tr><td><strong>{year_lbl}</strong></td>"
        for v in raw_vals:
            if v is None:
                body += "<td>-</td>"
            else:
                cls = "best" if v == hi else ("worst" if v == lo else "")
                body += f'<td class="{cls}">{v:+.2%}</td>'
        body += "</tr>"

    # CAGR row — reuse _ann_ret from metrics so it matches KPI cards byte-for-byte
    name_to_ann = {m["name"]: m.get("_ann_ret") for m in metrics}
    cagr_vals = [name_to_ann.get(c) for c in cols]
    valid = [v for v in cagr_vals if isinstance(v, (int, float))]
    hi, lo = (max(valid), min(valid)) if len(valid) >= 2 else (None, None)
    body += '<tr class="cagr-row"><td><strong>CAGR</strong></td>'
    for v in cagr_vals:
        if not isinstance(v, (int, float)):
            body += "<td>-</td>"
        else:
            cls = "best" if v == hi else ("worst" if v == lo else "")
            body += f'<td class="{cls}">{v:+.2%}</td>'
    body += "</tr>"

    st.markdown(f'<table class="cmp-table">{header}{body}</table>', unsafe_allow_html=True)

    partial_rows = [r for r in rows_data if r["partial"]]
    if partial_rows:
        parts = [f"{r['year']} = {r['start_date'].date()}→{r['end_date'].date()}" for r in partial_rows]
        st.caption("*Partial year. Coverage: " + "; ".join(parts) + ".")


# --- 3. Sidebar: Global Settings ---
SAVED_CONFIG_DIR = Path(__file__).parent / "Backtest"
VALID_STRATS = {STRAT_BH, STRAT_ANNUAL, STRAT_SEMI, STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_RD_FULL,
                STRAT_ASYM, STRAT_ASYM_LOCAL, STRAT_ASYM_LOCAL_EQ, STRAT_ASYM_LOCAL_PROP}

def apply_config(loaded_config):
    """Apply an imported/saved config dict to session state and rerun."""
    st.session_state.portfolios_list = loaded_config.get("portfolios", [])
    for p in st.session_state.portfolios_list:
        if 'id' not in p: p['id'] = str(uuid.uuid4())
        p.setdefault('name', 'Port ?')
        p.setdefault('tickers', '')
        p.setdefault('weights', '')
        p.setdefault('thr', 38)
        if p.get('strat') in STRAT_LEGACY_MAP:
            p['strat'] = STRAT_LEGACY_MAP[p['strat']]
        if p.get('strat') not in VALID_STRATS:
            p['strat'] = STRAT_ASYM
    st.session_state['bi'] = loaded_config.get("benchmark", "SPY")
    st.session_state['sd'] = pd.to_datetime(loaded_config.get("start_date", "2020-01-01")).date()
    st.session_state['init_funds'] = int(loaded_config.get("initial_funds", 10000))
    st.session_state.run_backtest = False  # require explicit Analyze on new config
    st.rerun()

with st.sidebar:
    st.markdown("### Settings")

    bench_in = st.text_input("Benchmark Ticker", value=st.session_state['bi'])
    start_d = st.date_input(
        "Start Date",
        value=st.session_state['sd'],
        min_value=datetime(1970, 1, 1).date(),
        max_value=datetime.today().date()
    )
    init_f = st.number_input("Initial Investment ($)", min_value=100, value=st.session_state['init_funds'], step=1000)
    rf_pct = st.number_input("Risk-free Rate (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.25, format="%.2f")
    rf_rate = rf_pct / 100.0

    st.session_state['bi'] = bench_in
    st.session_state['sd'] = start_d
    st.session_state['init_funds'] = init_f

    st.divider()
    st.markdown("### Inflation")
    inf_adj = st.checkbox("Enable CPI Adjustment", value=False)

    st.divider()
    st.markdown("### Config I/O")

    saved_files = sorted(SAVED_CONFIG_DIR.glob("*.json")) if SAVED_CONFIG_DIR.is_dir() else []
    if saved_files:
        sel_saved = st.selectbox("Saved Configs", saved_files, format_func=lambda p: p.stem)
        if st.button(":material/folder_open: Load Saved Config", width="stretch"):
            try:
                apply_config(json.loads(sel_saved.read_text(encoding="utf-8")))
            except Exception as e:
                st.error(f"Load error: {e}")

    current_config = {
        "benchmark": st.session_state['bi'],
        "start_date": str(st.session_state['sd']),
        "initial_funds": st.session_state['init_funds'],
        "portfolios": st.session_state.portfolios_list
    }
    json_str = json.dumps(current_config, indent=2, ensure_ascii=False)
    st.download_button(label=":material/download: Export", data=json_str, file_name="backtest_config.json", mime="application/json", width="stretch")

    uploaded_file = st.file_uploader("Import Config", type=["json"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            loaded_config = json.load(uploaded_file)
            if st.button(":material/upload: Apply Config", width="stretch"):
                apply_config(loaded_config)
        except Exception as e:
            st.error(f"Parse error: {e}")

# --- 4. Main Area: Portfolio Config ---
strategy_options = [
    STRAT_BH, STRAT_ANNUAL, STRAT_SEMI,
    STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_RD_FULL, STRAT_ASYM,
    STRAT_ASYM_LOCAL, STRAT_ASYM_LOCAL_EQ, STRAT_ASYM_LOCAL_PROP
]

# Column headers
hdr = st.columns([0.8, 3, 2, 2, 0.8, 0.4])
with hdr[0]: st.caption("Name")
with hdr[1]: st.caption("Tickers  ·  group with ( ) for composites")
with hdr[2]: st.caption("Weights")
with hdr[3]: st.caption("Strategy")
with hdr[4]: st.caption("Thr%")

total_portfolios = len(st.session_state.portfolios_list)
for i, port in enumerate(st.session_state.portfolios_list):
    if 'id' not in port: port['id'] = str(uuid.uuid4())

    cols = st.columns([0.8, 3, 2, 2, 0.8, 0.4])
    with cols[0]: st.markdown(f"**{port['name']}**")
    with cols[1]: port['tickers'] = st.text_input(
        "Tickers", port['tickers'], key=f"t_{port['id']}", label_visibility="collapsed",
        help='Composite slot: wrap elements in parentheses, e.g. "QQQM, GLDM, (DBMF, KMLM)". '
             "The slot takes ONE weight, split equally across its elements, and rebalances as one block.")
    with cols[2]: port['weights'] = st.text_input("Weights", port['weights'], key=f"w_{port['id']}", label_visibility="collapsed")
    with cols[3]: port['strat'] = st.selectbox("Strategy", strategy_options, index=strategy_options.index(port['strat']), key=f"s_{port['id']}", label_visibility="collapsed")
    with cols[4]: port['thr'] = st.number_input("Thr%", 1, 200, port['thr'], key=f"tr_{port['id']}", label_visibility="collapsed")
    with cols[5]:
        if total_portfolios > 1:
            st.button(":material/delete:", key=f"del_{port['id']}", on_click=delete_portfolio, args=(i,))

btn_cols = st.columns([1, 1.5, 5.5])
with btn_cols[0]:
    if st.button(":material/add_circle: Add", width="stretch"):
        existing_names = set(p["name"] for p in st.session_state.portfolios_list)
        new_char_code = 65
        while f"Port {chr(new_char_code)}" in existing_names: new_char_code += 1
        last_port = st.session_state.portfolios_list[-1]
        st.session_state.portfolios_list.append({
            "id": str(uuid.uuid4()), "name": f"Port {chr(new_char_code)}",
            "tickers": last_port["tickers"], "weights": last_port["weights"],
            "strat": STRAT_RD_MIXED, "thr": 40
        })
        st.rerun()
with btn_cols[1]:
    run_clicked = st.button(":material/play_arrow: Analyze", type="primary", width="stretch")
    if run_clicked:
        error_msgs = validate_inputs(st.session_state.portfolios_list, bench_in)
        if not error_msgs:
            st.session_state.run_backtest = True
        else:
            st.session_state.run_backtest = False
            for msg in error_msgs: st.error(msg)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

# --- 5. Results ---
if st.session_state.run_backtest:
    # Re-validate on every rerun: once run_backtest is set, any widget edit reruns
    # this block with current (possibly invalid) inputs, bypassing the button gate.
    revalidate_errs = validate_inputs(st.session_state.portfolios_list, bench_in)
    if revalidate_errs:
        for msg in revalidate_errs: st.error(msg)
        st.stop()

    parsed_ports = []
    for p in st.session_state.portfolios_list:
        p_tks, p_wts, _, p_comp = parse_portfolio(p)
        parsed_ports.append((p, p_tks, p_wts, p_comp))

    with st.spinner('Fetching data & running backtest...'):
        all_tks = sorted(set([clean_ticker(bench_in)] + [t for _, p_tks, _, _ in parsed_ports for t in p_tks]))
        try:
            df_raw = fetch_price_history(tuple(all_tks), str(start_d - timedelta(days=20)))
        except Exception as e:
            st.error(f"Download error: {e}"); st.stop()
        if df_raw is None or df_raw.empty: st.error("Download failed."); st.stop()

        if 'Adj Close' in df_raw.columns.get_level_values(0): price_data = df_raw['Adj Close'].copy()
        elif 'Close' in df_raw.columns.get_level_values(0): price_data = df_raw['Close'].copy()
        else: price_data = df_raw.copy()
        for tk in all_tks:
            if tk not in price_data.columns: price_data[tk] = np.nan

        # Scrub data glitches: corrupted listing-day prints (wrong scale on day 1)
        # and isolated mid-series spikes that revert on the next print. Genuine
        # multi-print moves are never touched. See backtest_core for details.
        scrubbed = scrub_leading_glitches(price_data)
        spiked = scrub_isolated_spikes(price_data)
        if scrubbed:
            st.warning("Dropped corrupted listing-day price(s): " + "; ".join(scrubbed))
        if spiked:
            st.warning("Dropped isolated mid-series price glitch(es): " + "; ".join(spiked))

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

        all_port_tks = sorted({t for _, p_tks, _, _ in parsed_ports for t in p_tks})
        raw_aligned = df_aligned[all_port_tks]
        first_valid_idx = raw_aligned.apply(lambda x: x.first_valid_index())
        bottleneck_date = first_valid_idx.max()
        bottleneck_ticker = first_valid_idx.idxmax()
        actual_start_day = market_start_day

        days_diff_bench = (market_start_day.date() - pd.Timestamp(start_d).date()).days
        if days_diff_bench > 7:
            st.warning(f"Benchmark **{bench_tk}** not listed until {market_start_day.date()}, start date adjusted.")
        elif days_diff_bench > 0:
            st.info(f"Aligned to next trading day: {market_start_day.date()}")

        # Must run regardless of how late the benchmark listed: any portfolio asset
        # listing after the start would otherwise be bfilled into flat fabricated
        # prices for the whole pre-listing stretch.
        if pd.notna(bottleneck_date) and (bottleneck_date - market_start_day).days > 7:
            actual_start_day = bottleneck_date
            st.warning(f"**{bottleneck_ticker}** listed late ({bottleneck_date.date()}), start date adjusted.")

        final_data = df_filled[df_filled.index >= actual_start_day]
        if final_data.empty: st.error("Insufficient data."); st.stop()
        days_span = (final_data.index[-1] - final_data.index[0]).days

        if days_span < 90:
            price_df = final_data.copy()
        else:
            price_df = sample_monthly(final_data)

        comp_df = pd.DataFrame(index=price_df.index)
        bench_nav = (price_df[bench_tk] / price_df[bench_tk].iloc[0]) * init_f
        comp_df[f"Benchmark({bench_in})"] = bench_nav

        res_list = {}
        valid_ports_meta = {}

        def clean_col(c):
            target = str(c).strip()
            for tk, name in TICKER_TO_NAME.items():
                if tk in target: return name
            return target

        def _pct_to_float(s):
            try: return float(str(s).rstrip('%')) / 100.0
            except (ValueError, TypeError): return np.nan

        for p, p_tks, p_wts, p_comp in parsed_ports:
            has_data = lambda t: t in price_df.columns and not price_df[t].isna().all()

            # Slot view: composite metadata if present, else one singleton per element.
            if p_comp:
                slots = list(zip(p_comp["slot_labels"], p_comp["slot_members"], p_comp["slot_targets"]))
            else:
                slots = [(t, [t], w) for t, w in zip(p_tks, p_wts)]

            # Drop dataless elements: re-split a slot among survivors; drop a slot
            # only if ALL its elements are dataless.
            valid_p_tks, elem_weights, slot_survivors = [], {}, {}
            dropped_elems, dropped_slots = [], []
            for si, (lbl, members, st_w) in enumerate(slots):
                live = [t for t in members if has_data(t)]
                dead = [t for t in members if not has_data(t)]
                if not live:
                    dropped_slots.append(lbl); continue
                if dead: dropped_elems.extend(dead)
                per = st_w / len(live)                 # equal split across survivors
                for t in live:
                    elem_weights[t] = per; valid_p_tks.append(t)
                slot_survivors[si] = live
            if not valid_p_tks:
                st.error(f"**{p['name']}**: no usable data for any ticker ({', '.join(p_tks)}) \u2014 portfolio skipped.")
                continue

            w_series = pd.Series(elem_weights).reindex(valid_p_tks)
            # Renormalize ONLY when something was dropped (no-drop path stays
            # byte-identical to the legacy un-normalized weights). One-shot
            # normalization handles intra-slot re-split + cross-slot drop together.
            if dropped_elems or dropped_slots:
                w_series = w_series / w_series.sum()
            if dropped_elems:
                st.warning(
                    f"**{p['name']}**: no data for **{', '.join(dropped_elems)}** \u2014 "
                    "dropped from their composite; slot re-split among remaining elements.")
            if dropped_slots:
                st.warning(
                    f"**{p['name']}**: no data for **{', '.join(dropped_slots)}** \u2014 "
                    "dropped; remaining weights renormalized: "
                    + ", ".join(f"{t} {w:.1%}" for t, w in w_series.items()))

            # Engine groups: element -> synthetic slot-id, only for composite slots
            # that still have >1 surviving element. Empty => None => legacy path.
            groups = {}
            for si, live in slot_survivors.items():
                if len(live) > 1:
                    for t in live: groups[t] = f"__slot{si}"
            groups = groups or None

            res_df, cnt, pnl_rec = run_detailed_backtest(
                p['strat'], price_df[valid_p_tks], w_series, init_f, p['thr']/100.0, groups=groups)
            if not res_df.empty:
                # For each surviving composite slot, add an aggregate weight column.
                # Element columns keep their OWN ticker name (no slot prefix \u2014 keeps
                # headers narrow). The slot's members are translated separately only in
                # the "(slot)" header so CN tickers don't collapse under substring match.
                comp_slots = {si: live for si, live in slot_survivors.items() if len(live) > 1}
                for si, live in comp_slots.items():
                    agg_name = "+".join(live) + " (slot)"
                    res_df[agg_name] = res_df[live].apply(
                        lambda row: sum(_pct_to_float(x) for x in row), axis=1
                    ).map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
                    pnl_rec[agg_name] = f"{sum(_pct_to_float(pnl_rec[m]) for m in live):.2%}"

                def clean_col_p(c):
                    raw = str(c).strip()
                    if raw.endswith(" (slot)"):
                        body = raw[:-len(" (slot)")]
                        return "+".join(clean_col(m) for m in body.split("+")) + " (slot)"
                    return clean_col(raw)

                df_chart = res_df.drop_duplicates(subset='Date', keep='last').copy()
                df_chart['Date'] = pd.to_datetime(df_chart['Date'])
                df_chart = df_chart.set_index('Date')
                comp_df[p['name']] = df_chart['NAV']
                valid_ports_meta[p['name']] = cnt
                translated_pnl = {}
                for k, v in pnl_rec.items(): translated_pnl[clean_col_p(k)] = v
                pnl_df = pd.DataFrame([translated_pnl])
                df_history = res_df.iloc[::-1].rename(columns=clean_col_p).reset_index(drop=True)
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
                    edited_inf = st.data_editor(inf_df, hide_index=True, width="stretch")
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
        bench_m = calculate_metrics(comp_df[f"Benchmark({bench_in})"], 0, risk_free_rate=rf_rate)
        bench_m["name"] = f"Benchmark({bench_in})"
        metrics.append(bench_m)
        for p_name, cnt in valid_ports_meta.items():
            m = calculate_metrics(comp_df[p_name], cnt, risk_free_rate=rf_rate)
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
        st.altair_chart(base_chart, width="stretch")

        # --- Drawdown Chart ---
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        with st.expander("Drawdown", expanded=True):
            dd_df = comp_df / comp_df.cummax() - 1
            dd_long = dd_df.reset_index().melt('Date', var_name='Portfolio', value_name='Drawdown')
            dd_wide = dd_df.reset_index()
            dd_nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Date'], empty=False)

            dd_line = alt.Chart(dd_long).mark_line(strokeWidth=2).encode(
                x=alt.X('Date:T', axis=alt.Axis(format=x_axis_format, title=None, labelAngle=label_angle, grid=False)),
                y=alt.Y('Drawdown:Q', axis=alt.Axis(format='.0%', title='Drawdown', grid=True, gridDash=[3,3], gridColor='#e5e7eb')),
                color=alt.Color('Portfolio:N', legend=alt.Legend(orient='top', title=None, labelFontSize=12), scale=alt.Scale(range=palette))
            )
            dd_tooltips = [alt.Tooltip('Date:T', format='%Y-%m-%d', title='Date')]
            for col in dd_df.columns:
                dd_tooltips.append(alt.Tooltip(field=col, type='quantitative', format='.2%', title=col))
            dd_selectors = alt.Chart(dd_wide).mark_rule(opacity=0.001, strokeWidth=40).encode(
                x='Date:T', tooltip=dd_tooltips
            ).add_params(dd_nearest)
            dd_rules = alt.Chart(dd_wide).mark_rule(color='#94a3b8', strokeDash=[3,3]).encode(
                x='Date:T', tooltip=dd_tooltips
            ).transform_filter(dd_nearest)
            dd_points = dd_line.mark_point(size=60, filled=True).encode(
                opacity=alt.condition(dd_nearest, alt.value(1), alt.value(0))
            )
            dd_chart = alt.layer(dd_line, dd_selectors, dd_rules, dd_points).properties(
                height=260
            ).configure_view(strokeWidth=0)
            st.altair_chart(dd_chart, width="stretch")

        # --- NAV CSV Export ---
        st.download_button(
            ":material/download: Export NAV Series (CSV)",
            data=comp_df.to_csv(date_format="%Y-%m-%d").encode("utf-8-sig"),
            file_name=f"backtest_nav_{actual_start_day.date()}.csv",
            mime="text/csv",
        )

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
                    st.dataframe(res_list[lbl].style.apply(style_row, axis=1).format({"NAV": "{:,.2f}"}), width="stretch")

        # --- Annual Returns by Calendar Year ---
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        with st.expander("Annual Returns by Calendar Year", expanded=True):
            render_annual_returns_table(comp_df, metrics)
