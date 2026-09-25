import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import requests
import re
import threading
import time
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    from streamlit_searchbox import st_searchbox
    HAS_SEARCHBOX = True
except ImportError:            # optional dependency: matrix still accepts typed tickers
    HAS_SEARCHBOX = False

try:
    from streamlit_js_eval import streamlit_js_eval
    HAS_JS_EVAL = True
except ImportError:            # optional dependency: browser-persisted default disabled
    HAS_JS_EVAL = False

def _fresh_core():
    """Streamlit Cloud hot-reloads this script on a push but can keep the old
    backtest_core in sys.modules until the app is rebooted: the import below
    then fails on names added in this version (seen after the v2.5.0 push).
    Reload the module when those names are missing (v2.6.0)."""
    import importlib
    import backtest_core
    if not all(hasattr(backtest_core, n) for n in ("align_price_data", "prepare_portfolio", "band_trigger_weights")):
        importlib.reload(backtest_core)


_fresh_core()

from backtest_core import (
    STRAT_BH, STRAT_ANNUAL, STRAT_SEMI,
    STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_RD_FULL, STRAT_ASYM,
    STRAT_LEGACY_MAP,
    clean_ticker, parse_portfolio, calculate_metrics,
    run_detailed_backtest, compute_annual_returns,
    scrub_leading_glitches, scrub_isolated_spikes, sample_monthly,
    _split_top_level, slot_labels, normalize_slot_bands, build_band_thresholds,
    BAND_MODE_REL, BAND_MODE_RATIO, BAND_MODES,
    band_trigger_weights, align_price_data, prepare_portfolio,
)

# Band mode (v2.5.2) as shown in the portfolio row: what Down % / Up % measure.
BAND_MODE_LABELS = {BAND_MODE_REL: "Δ vs target", BAND_MODE_RATIO: "Leg vs rest"}

# --- Version ---
APP_VERSION = "2.6.0"  # semver: major.minor.patch
APP_BUILD_DATE = "2026-09-24"

# --- 1. Page Config ---
st.set_page_config(page_title="Portfolio Backtest", layout="wide", page_icon="📊")

# --- Theme & series palette ---
def _theme_type():
    """Active Streamlit theme type; safe fallback for AppTest / bare mode."""
    try:
        t = st.context.theme.type
    except Exception:
        t = None
    return t if t in ("light", "dark") else "light"

THEME = _theme_type()

# Categorical series palette (dataviz six-checks validated: light set on #ffffff,
# dark set on #0e1117). Slot ORDER is the CVD-safety mechanism — never re-sort.
# Color follows the ENTITY: each portfolio keeps its slot from the editor row
# index across charts, cards and tables, even if another portfolio is dropped.
CAT_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
CAT_DARK  = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"]
SERIES_COLORS = CAT_DARK if THEME == "dark" else CAT_LIGHT
BENCH_COLOR = "#898781"   # benchmark = neutral context line (dashed), not a competing identity

_T = {
    "light": {
        "card": "#ffffff", "subtle": "#f6f8fa", "ink1": "#1a1a2e", "ink2": "#52514e",
        "muted": "#898781", "border": "rgba(11,11,11,0.10)", "grid": "#e5e7eb",
        "good": "#006300", "bad": "#d03b3b",
        "shadow": "0 1px 4px rgba(15,23,42,0.06)", "hover": "0 6px 16px rgba(15,23,42,0.12)",
    },
    "dark": {
        "card": "#1b1f27", "subtle": "#262b36", "ink1": "#fafafa", "ink2": "#c3c2b7",
        "muted": "#898781", "border": "rgba(255,255,255,0.12)", "grid": "#2c2f36",
        "good": "#0ca30c", "bad": "#e66767",
        "shadow": "0 1px 4px rgba(0,0,0,0.35)", "hover": "0 6px 16px rgba(0,0,0,0.45)",
    },
}[THEME]

st.markdown(
    "<style>:root{"
    + "".join(f"--{k}:{v};" for k, v in _T.items())
    + "}</style>",
    unsafe_allow_html=True,
)

# --- Custom CSS (all colors via :root design tokens -> theme-aware) ---
st.markdown("""
<style>
/* ===== Global ===== */
section.main > div { max-width: 1400px; margin: 0 auto; }
h1, h2, h3, h4 { letter-spacing: -0.02em; }

/* ===== Header Bar ===== */
.header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 1.1rem 1.6rem;
    border-radius: 0.75rem;
    margin-bottom: 1.1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.header-bar h2 { margin: 0; font-size: clamp(1.05rem, 2.5vw, 1.4rem); font-weight: 700; color: #fff; }
.header-bar .subtitle { opacity: 0.65; font-size: 0.83rem; }
.header-bar .version-badge {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 1rem;
    padding: 0.2rem 0.75rem;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    white-space: nowrap;
}

/* ===== Section label ===== */
.sec-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink2);
    margin: 0.25rem 0 0.35rem 0;
}

/* ===== Portfolio editor ===== */
.col-cap {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.rb-legend {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0.4rem 0 0.3rem;
    line-height: 1.6;
}
.rb-legend b { color: inherit; font-weight: 600; }
.rb-sw {
    display: inline-block;
    width: 0.8rem; height: 0.8rem;
    border-radius: 2px;
    border: 1px solid rgba(0,0,0,0.18);
    vertical-align: -0.12rem;
    margin-right: 0.15rem;
}
.row-dot {
    display: inline-block;
    width: 11px; height: 11px;
    border-radius: 50%;
    box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 18%, transparent);
}

/* ===== Summary cards (one per series; hero = annualized return) ===== */
.sum-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
    gap: 0.75rem;
    margin: 0.25rem 0 0.75rem 0;
}
.sum-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--pc);
    border-radius: 0.65rem;
    padding: 0.8rem 0.95rem 0.7rem 0.95rem;
    box-shadow: var(--shadow);
    transition: transform 0.12s, box-shadow 0.15s;
}
.sum-card:hover { transform: translateY(-2px); box-shadow: var(--hover); }
.sum-head {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.82rem; font-weight: 700; color: var(--ink1);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sum-head .dot {
    flex: none; width: 9px; height: 9px; border-radius: 50%;
    background: var(--pc);
}
.sum-hero {
    font-size: clamp(1.35rem, 2.2vw, 1.7rem);
    font-weight: 750;
    line-height: 1.15;
    margin: 0.35rem 0 0.45rem 0;
    color: var(--ink1);
}
.sum-hero.pos { color: var(--good); }
.sum-hero.neg { color: var(--bad); }
.sum-hero .per { font-size: 0.75rem; font-weight: 600; color: var(--muted); }
.sum-minis {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.2rem 0.75rem;
    padding-top: 0.45rem;
    border-top: 1px solid var(--border);
}
.sum-minis > div { display: flex; justify-content: space-between; gap: 0.5rem; }
.sum-minis > div.wide { grid-column: 1 / -1; }
.sum-minis .k { font-size: 0.7rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
.sum-minis .v { font-size: 0.78rem; color: var(--ink1); font-weight: 600; font-variant-numeric: tabular-nums; }
.sum-foot {
    margin-top: 0.4rem;
    font-size: 0.72rem;
    color: var(--ink2);
    font-variant-numeric: tabular-nums;
}

/* ===== Comparison / annual tables ===== */
.cmp-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85rem;
    margin: 0.5rem 0 0.75rem 0;
    border-radius: 0.5rem;
    overflow: hidden;
    border: 1px solid var(--border);
}
.cmp-table th {
    background: var(--subtle);
    padding: 0.55rem 1rem;
    text-align: left;
    font-weight: 600;
    color: var(--ink2);
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
}
.cmp-table th .th-dot {
    display: inline-block; width: 9px; height: 9px; border-radius: 50%;
    margin-right: 0.45rem; vertical-align: baseline;
}
.cmp-table td {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--ink1);
    font-variant-numeric: tabular-nums;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table tr:hover td { background: var(--subtle); }
.cmp-table .best { color: var(--good); font-weight: 700; }
.cmp-table .worst { color: var(--bad); }
.cmp-table .cagr-row td { background: var(--subtle); font-weight: 600; border-top: 2px solid var(--border); }

/* ===== Allocation sums line ===== */
.alloc-sums {
    font-size: 0.78rem;
    margin: 0.15rem 0 0.5rem 0.2rem;
    font-variant-numeric: tabular-nums;
    color: var(--ink2);
}

/* ===== Empty state ===== */
.empty-state {
    border: 1.5px dashed var(--border);
    border-radius: 0.75rem;
    padding: 2.2rem 1.5rem;
    text-align: center;
    color: var(--ink2);
    margin-top: 0.75rem;
}
.empty-state .es-icon { font-size: 1.9rem; margin-bottom: 0.4rem; }
.empty-state .es-title { font-weight: 700; color: var(--ink1); margin-bottom: 0.3rem; }
.empty-state .es-body { font-size: 0.86rem; max-width: 560px; margin: 0 auto; line-height: 1.55; }

/* ===== Section divider ===== */
.section-gap { margin: 1.25rem 0 0.6rem 0; }

/* ===== Buttons ===== */
button[kind="primary"] {
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: transform 0.1s, box-shadow 0.15s !important;
}
button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--hover) !important;
}
button[kind="secondary"] {
    border-radius: 0.5rem !important;
    transition: all 0.15s !important;
}

/* ===== Invisible utility components (localStorage bridge) ===== */
div[data-testid="stElementContainer"]:has(iframe[title="streamlit_js_eval.streamlit_js_eval"]) {
    display: none;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
section[data-testid="stSidebar"] h3 {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink2);
}
</style>
""", unsafe_allow_html=True)

# --- Mode (v2.6.0): Backtest | Live Portfolio ---
# Always rendered first in the sidebar in both modes, so it never shifts the
# blocks below it (see the v2.5.1 note on transient elements).
APP_MODE = st.sidebar.radio("Mode", ["Backtest", "Live Portfolio"], horizontal=True, key="app_mode",
                            label_visibility="collapsed")
_HDR = ({"Backtest": ("Portfolio Backtest & Rebalance Analyzer", "Multi-portfolio backtesting with rebalancing strategies"),
         "Live Portfolio": ("Live Portfolio Desk", "Share-level orders that keep the real account on the live rule's backtest")}
        [APP_MODE])

# --- Header ---
st.markdown(f"""
<div class="header-bar">
    <div>
        <h2>{_HDR[0]}</h2>
        <div class="subtitle">{_HDR[1]}</div>
    </div>
    <div class="version-badge">v{APP_VERSION} · {APP_BUILD_DATE}</div>
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

# --- Config persistence (sidebar Config I/O + Save Default button) ---
SAVED_CONFIG_DIR = Path(__file__).parent / "Backtest"
DEFAULT_CONFIG_PATH = SAVED_CONFIG_DIR / "_default.json"
VALID_STRATS = {STRAT_BH, STRAT_ANNUAL, STRAT_SEMI, STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_RD_FULL,
                STRAT_ASYM}

def _norm_band_pct(v, fallback):
    """Band in percent as an int inside the UI's 1..200 range; None / non-numeric -> fallback."""
    if v is None:
        return fallback
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        return fallback
    return min(200, max(1, iv))


def _norm_band_mode(v):
    """Band mode from a config value: "ratio" (any case / padding) or "rel";
    anything else or missing -> "rel", the original rule, so pre-2.5.2 configs
    run unchanged."""
    return BAND_MODE_RATIO if str(v or "").strip().lower() == BAND_MODE_RATIO else BAND_MODE_REL


# Rebalance-row colours in the detail table (v2.5.4): (Pre-Rebal, Post-Rebal).
# Global = every slot back to target (orange / green, the pre-2.5.4 colours);
# Local = only the breached minor slots reset, the rest scaled pro rata (purple).
# Pastel fills with explicit dark ink, so rows stay readable in the dark theme.
REBAL_COLORS = {"global": ("#fff3e0", "#e8f5e9"), "local": ("#f3e5f5", "#e1bee7")}
_INK = "color: #1a1a2e"


def rebal_row_style(row, scope=None):
    """Styler row function for the rebalance detail table. `scope` is the
    engine's "global" / "local" for the row's rebalance (None when unknown,
    e.g. a cached pre-2.5.4 engine -> the global colours, as before)."""
    n = len(row)
    kind = row.get('Type')
    if kind == 'PnL Contrib%': return ['background-color: #fce4ec; color: #d81b60; font-weight: bold'] * n
    if kind == 'Init': return [f'background-color: #e3f2fd; {_INK}; font-weight: bold'] * n
    pre, post = REBAL_COLORS["local" if scope == "local" else "global"]
    if kind == 'Pre-Rebal': return [f'background-color: {pre}; {_INK}'] * n
    if kind == 'Post-Rebal': return [f'background-color: {post}; {_INK}'] * n
    return [''] * n


def rebal_legend_html(n_global, n_local):
    """Legend line above the detail table: colour swatches + counts per scope."""
    sw = lambda c: f'<span class="rb-sw" style="background:{c}"></span>'
    g, l = REBAL_COLORS["global"], REBAL_COLORS["local"]
    return ('<div class="rb-legend">Rebalance rows · '
            f'{sw(g[0])}{sw(g[1])}<b>Global {n_global}</b>: every slot back to target '
            '(RelDiff Mixed: a major slot, target ≥ 10%, breached its band) · '
            f'{sw(l[0])}{sw(l[1])}<b>Local {n_local}</b>: only minor slots breached · they are reset, '
            'the other slots scaled pro rata · each pair = Pre-Rebal row, then Post-Rebal row</div>')


def _apply_config_state(loaded_config):
    """Normalize a config dict into session state (no rerun).

    Band fields (v2.5.0): `thr` = DOWN band (legacy name kept), `thr_up` = UP
    band (missing/null -> equal to thr, i.e. symmetric), `slot_bands` = per-slot
    overrides {slot_label: {"down": pct|None, "up": pct|None}}. v2.5.2 adds
    `band_mode` ("rel" | "ratio", missing -> "rel"). Old configs therefore load
    unchanged and run on the engine's legacy scalar path."""
    st.session_state.portfolios_list = loaded_config.get("portfolios", [])
    band_warns = []
    for p in st.session_state.portfolios_list:
        if 'id' not in p: p['id'] = str(uuid.uuid4())
        p.setdefault('name', 'Port ?')
        p.setdefault('tickers', '')
        p.setdefault('weights', '')
        p['thr'] = _norm_band_pct(p.get('thr'), 38)
        p['thr_up'] = _norm_band_pct(p.get('thr_up'), p['thr'])
        p['slot_bands'] = normalize_slot_bands(p.get('slot_bands'))
        p['band_mode'] = _norm_band_mode(p.get('band_mode'))
        known = set(slot_labels(p['tickers']))
        unknown = [k for k in p['slot_bands'] if k not in known]
        if unknown:
            band_warns.append(
                f"**{p['name']}**: per-slot band(s) for unknown slot(s) **{', '.join(unknown)}** "
                "— kept in the config but ignored until a matching slot exists (labels: the "
                "ticker, or a composite's members joined with '+'). Blank them in "
                "*Per-slot bands* to drop them.")
        if p.get('strat') in STRAT_LEGACY_MAP:
            p['strat'] = STRAT_LEGACY_MAP[p['strat']]
        if p.get('strat') not in VALID_STRATS:
            p['strat'] = STRAT_ASYM
    if band_warns:
        st.session_state['_flash_warn'] = band_warns
    st.session_state['bi'] = loaded_config.get("benchmark", "SPY")
    st.session_state['sd'] = pd.to_datetime(loaded_config.get("start_date", "2020-01-01")).date()
    st.session_state['init_funds'] = int(loaded_config.get("initial_funds", 10000))
    st.session_state.run_backtest = False
    # Discard allocation-editor scaffolding: stale in-flight edits must not be
    # flushed over a freshly loaded config on the next matrix rebuild.
    _old_key = st.session_state.pop('_alloc_key', None)
    if _old_key:
        st.session_state.pop(_old_key, None)
    st.session_state.pop('_alloc_base', None)
    st.session_state.pop('_alloc_pending', None)
    st.session_state.pop('_alloc_pending_seen', None)
    for _k in (st.session_state.pop('_sb_keys', None) or {}).values():
        st.session_state.pop(_k, None)      # per-slot band editors: same staleness rule
    st.session_state.pop('_sb_base', None)
    st.session_state.pop('_sb_expanded', None)
    _sb_state = st.session_state.get("asset_search")
    if isinstance(_sb_state, dict):  # a stale searchbox pick must not leak in
        _sb_state["result"] = None

if 'portfolios_list' not in st.session_state:
    # New session: a saved default (Save Default button) replaces the built-in
    # config below. Delete Backtest/_default.json to restore the built-ins.
    _default_cfg = None
    if DEFAULT_CONFIG_PATH.is_file():
        try:
            _default_cfg = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            _default_cfg = None
    if _default_cfg and _default_cfg.get("portfolios"):
        _apply_config_state(_default_cfg)

if 'portfolios_list' not in st.session_state:
    st.session_state.portfolios_list = [
        {
            # Crypto sleeve split into a composite (ETH-USD, MSTR): same 5% slot ->
            # 2.5% / 2.5%. Rebalanced with RelDiff Mixed at a 40% trigger band.
            "id": str(uuid.uuid4()),
            "name": "AV-US",
            "tickers": "QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, (ETH-USD, MSTR)",
            "weights": "0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05",
            "strat": STRAT_RD_MIXED,
            "thr": 40, "thr_up": 40, "slot_bands": {}, "band_mode": BAND_MODE_REL
        },
        {
            # AV-US with the crypto sleeve as a plain ETH-USD slot (5%), rebalanced
            # with Asymmetric RelDiff at a 38% trigger band. Weights identical to AV-US.
            "id": str(uuid.uuid4()),
            "name": "Port B",
            "tickers": "QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, ETH-USD",
            "weights": "0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05",
            "strat": STRAT_ASYM,
            "thr": 38, "thr_up": 38, "slot_bands": {}, "band_mode": BAND_MODE_REL
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Port C",
            "tickers": "159941.SZ, 512890.SS, 515220.SS, 588080.SS, 518880.SS, 511130.SS",
            "weights": "0.35, 0.15, 0.10, 0.05, 0.15, 0.20",
            "strat": STRAT_RD_MIXED,
            "thr": 38, "thr_up": 38, "slot_bands": {}, "band_mode": BAND_MODE_REL
        }
    ]

# --- Browser-persisted startup default -------------------------------------
# Streamlit Cloud rebuilds the container on every deploy/reboot, wiping
# Backtest/_default.json. Save Default therefore ALSO stores the config in
# the browser's localStorage; on session start, when no file default exists,
# it is restored from there. The file (local workflow) always wins.
LS_DEFAULT_KEY = "backtest_default_config"

LS_GET_EXPR = f"localStorage.getItem({json.dumps(LS_DEFAULT_KEY)}) ?? '__none__'"

if DEFAULT_CONFIG_PATH.is_file():
    st.session_state['_ls_checked'] = True     # file default already applied
    st.session_state['_ls_default_present'] = True
if HAS_JS_EVAL and not st.session_state.get('_ls_checked'):
    # The reader component itself is rendered at the END of the script (see
    # the bottom of the file); its answer arrives here through session state
    # on the following run. Rendering it at the top and dropping it once
    # answered shifted every main-area element down by one position on the
    # first user-triggered rerun — the frontend keys elements by position, so
    # all of them re-mounted and any digits being typed at that moment were
    # lost (the reported "number jumps away" in the Down % / Up % inputs).
    _ls_raw = st.session_state.get('_ls_get')
    if _ls_raw == '__none__':
        st.session_state['_ls_checked'] = True
    elif isinstance(_ls_raw, str) and _ls_raw:
        st.session_state['_ls_checked'] = True
        st.session_state['_ls_default_present'] = True
        # Apply only while the session is pristine (no results on screen).
        if not st.session_state.run_backtest:
            try:
                _ls_cfg = json.loads(_ls_raw)
                if _ls_cfg.get("portfolios"):
                    _apply_config_state(_ls_cfg)
                    st.rerun()
            except Exception:
                pass

def delete_portfolio(idx):
    if 0 <= idx < len(st.session_state.portfolios_list):
        st.session_state.portfolios_list.pop(idx)

def _n_to_letters(n):
    """1 -> A ... 26 -> Z, 27 -> AA (spreadsheet-column order)."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def next_port_name(ports):
    """Continue the letter sequence after the highest existing "Port X"
    (last is Port C -> Port D); custom names like AV-US are ignored."""
    existing = {str(p.get("name", "")).strip() for p in ports}
    last = 0
    for name in existing:
        m = re.fullmatch(r"(?i)port\s+([a-z]+)", name)
        if m:
            n = 0
            for ch in m.group(1).upper():
                n = n * 26 + ord(ch) - 64
            last = max(last, n)
    n = last + 1
    while f"Port {_n_to_letters(n)}" in existing:
        n += 1
    return f"Port {_n_to_letters(n)}"

# --- 2. Data Fetch & Validation Helpers ---
# Prices are cached PER TICKER, not per ticker set: editing a portfolio only
# requests the tickers not already held, and one ticker's failure can never
# poison another's data.
#
# Before v2.4.1 the whole yf.download frame was cached per ticker set, and the
# price level was chosen for the whole frame ('Adj Close' if present, else
# 'Close'). yfinance never raises for a subset failure: a ticker whose request
# failed (Yahoo rate limit, network hiccup, unknown symbol) comes back as an
# EMPTY placeholder column -- and that placeholder carries an 'Adj Close'
# level which auto-adjusted real data lacks. So one failed ticker made the app
# select a frame holding nothing but that empty column, every ticker looked
# dataless ("No data after <start>"), and the partial frame was served from
# cache for the full hour. Deleting a portfolio triggered it: the changed set
# was a cache miss, and the fresh batch hit Yahoo's rate limit for one ticker.
PRICE_CACHE_TTL = 3600       # seconds a fetched series is reused
PRICE_CACHE_MISS_TTL = 60    # seconds a failed ticker is not re-requested by widget reruns
PRICE_CACHE_MAX = 256        # distinct tickers kept (a few thousand floats each); oldest evicted
PRICE_RETRY_PAUSE = 1.0      # seconds before the single retry of a non-rate-limited failure
_RATE_LIMIT_MARKERS = ("rate limit", "too many requests")


@st.cache_resource(show_spinner=False)
def _price_cache():
    """Process-wide singleton (same object on every rerun, for every session):
    {"lock": Lock, "entries": {ticker: {expires, start, series|None, reason}}}.
    The lock also serializes yf.download calls, whose per-ticker results go
    through yfinance's module-global state (yf.shared) and would otherwise be
    corrupted by two sessions downloading at once."""
    return {"lock": threading.Lock(), "entries": {}}


def clear_price_cache():
    _price_cache.clear()


def forget_failed_prices():
    """Drop the negative entries so an explicit Analyze re-requests every
    ticker that had no data, whatever is left of PRICE_CACHE_MISS_TTL."""
    cache = _price_cache()
    with cache["lock"]:
        for tk in [k for k, e in cache["entries"].items() if e["series"] is None]:
            del cache["entries"][tk]


def _is_rate_limited(reason):
    return any(m in str(reason).lower() for m in _RATE_LIMIT_MARKERS)


def _tidy_reason(reason):
    """yf.shared._ERRORS holds repr(exception): "YFRateLimitError('Too Many
    Requests. ...')" -> "Too Many Requests. ..."."""
    m = re.fullmatch(r"\w+\((['\"])(.*)\1\)", str(reason).strip(), flags=re.S)
    return m.group(2) if m else str(reason).strip()


def _extract_close(df, tk, single=False):
    """Adjusted-close Series for ONE ticker out of a yf.download frame, or None
    when the ticker has no price at all. Selection is per ticker on purpose: a
    failed ticker's placeholder must never decide the price level used for the
    others (see the note above). `single`: frame of a lone ticker without a
    ticker level (older yfinance), whose columns are the price fields."""
    if df is None or df.empty:
        return None
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        candidates = [(lvl, tk) for lvl in ("Adj Close", "Close") if (lvl, tk) in cols]
    elif single:
        candidates = [lvl for lvl in ("Adj Close", "Close") if lvl in cols]
    else:
        candidates = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            return s.rename(tk)
    return None


def _batch_close(tickers, start, download):
    """One yf.download batch -> ({ticker: close Series}, {ticker: reason}).
    Every requested ticker is validated here, since a subset failure is
    silent (empty placeholder column + reason in yf.shared._ERRORS)."""
    yf.shared._ERRORS = {}
    df = download(list(tickers), start=start, auto_adjust=True, progress=False)
    reasons = {str(k).upper(): _tidy_reason(v) for k, v in (yf.shared._ERRORS or {}).items()}
    good, bad = {}, {}
    for tk in tickers:
        s = _extract_close(df, tk, single=len(tickers) == 1)
        if s is None:
            bad[tk] = reasons.get(tk.upper(), "Yahoo returned no prices")
        else:
            good[tk] = s
    return good, bad


def fetch_price_history(tickers, start, download=None, now=None):
    """Adjusted-close history from `start`, one column per ticker, through
    the per-ticker cache. Returns (prices, failures): `failures` maps each
    ticker that has no price at all to Yahoo's reason (its column is NaN).

    Only tickers without a live entry are requested, in one threaded
    yf.download batch, so add/delete/rename never re-downloads what an earlier
    run fetched. A ticker that came back empty is retried once -- never when
    rate limited, that only extends the block -- then negatively cached for
    PRICE_CACHE_MISS_TTL so widget reruns don't re-hit Yahoo (Analyze clears
    it, see forget_failed_prices). Raises when the download itself fails, so
    nothing is cached. `download`/`now` are injection points for tests."""
    tickers = list(dict.fromkeys(tickers))
    download = download or yf.download
    now = time.time() if now is None else now
    start_ts = pd.Timestamp(start)
    cache = _price_cache()
    with cache["lock"]:
        entries = cache["entries"]

        def live(tk):
            e = entries.get(tk)
            return e is not None and e["expires"] > now and e["start"] <= start_ts

        missing = [tk for tk in tickers if not live(tk)]
        if missing:
            good, bad = _batch_close(missing, start, download)
            retry = [tk for tk, why in bad.items() if not _is_rate_limited(why)]
            if retry:
                time.sleep(PRICE_RETRY_PAUSE)
                good2, bad2 = _batch_close(retry, start, download)
                good.update(good2)
                bad = {tk: why for tk, why in bad.items() if tk not in good2}
                bad.update(bad2)
            for tk, s in good.items():
                entries[tk] = {"expires": now + PRICE_CACHE_TTL, "start": start_ts,
                               "series": s, "reason": None}
            for tk, why in bad.items():
                entries[tk] = {"expires": now + PRICE_CACHE_MISS_TTL, "start": start_ts,
                               "series": None, "reason": why}
            if len(entries) > PRICE_CACHE_MAX:
                oldest = sorted(entries, key=lambda k: entries[k]["expires"])
                for tk in oldest[:len(entries) - PRICE_CACHE_MAX]:
                    del entries[tk]
        frames, failures = [], {}
        for tk in tickers:
            e = entries[tk]
            if e["series"] is None:
                failures[tk] = e["reason"]
            else:
                frames.append(e["series"])
    prices = pd.concat(frames, axis=1, sort=True) if frames else pd.DataFrame()
    if len(prices):
        prices = prices[prices.index >= start_ts]
    for tk in tickers:
        if tk not in prices.columns:
            prices[tk] = np.nan
    return prices[tickers].copy(), failures

# Series names that collide with chart/table plumbing (index/melt names) or
# the allocation matrix's token column ("Asset").
RESERVED_SERIES_NAMES = {"Date", "Return", "Drawdown", "Portfolio", "Asset"}

def validate_inputs(portfolios, benchmark):
    """Validate benchmark + all portfolio configs. Returns list of error messages."""
    errors = []
    if not str(benchmark).strip():
        errors.append("Benchmark ticker is empty")
    if not portfolios:
        errors.append("No portfolios configured — add one first")
    names = [p['name'] for p in portfolios]
    if any(not str(n).strip() for n in names):
        errors.append("Portfolio name is empty")
    dup_names = sorted({n for n in names if names.count(n) > 1})
    if dup_names:
        errors.append("Duplicate portfolio names: " + ", ".join(dup_names))
    reserved = sorted({str(n).strip() for n in names} & RESERVED_SERIES_NAMES)
    if reserved:
        errors.append("Reserved name(s) not allowed for portfolios: " + ", ".join(reserved))
    bench_label = f"Benchmark({benchmark})"
    if bench_label in names:
        errors.append(f'Portfolio name "{bench_label}" collides with the benchmark series')
    for p in portfolios:
        _, _, perrs, _ = parse_portfolio(p)
        errors.extend(f"**{p['name']}**: {e}" for e in perrs)
    return errors

# --- Allocation matrix (Portfolio-Visualizer style editor) ------------------
# The matrix is a VIEW over the stored per-portfolio tickers/weights strings:
# rows = slot tokens (a ticker or a "(A, B)" composite group), one weight column
# per portfolio in PERCENT, blank/0 = not held. Storage, JSON export/import and
# saved configs keep the legacy string format unchanged.

def _slot_tokens(tickers_str):
    """Split a tickers string into slot tokens, respecting ( ) groups."""
    s = str(tickers_str or "").replace("，", ",").replace("（", "(").replace("）", ")")
    tokens, err = _split_top_level(s, ",")
    if err:
        tokens = s.split(",")   # unbalanced parens: degrade gracefully, validated on Analyze
    return [t.strip() for t in tokens if t.strip()]


def build_alloc_df(ports):
    """portfolios -> DataFrame(Asset | <name>% per portfolio), union of slots."""
    slots, weights = [], {}
    for p in ports:
        tokens = _slot_tokens(p.get("tickers", ""))
        w_raw = [w.strip() for w in str(p.get("weights", "")).replace("，", ",").split(",") if w.strip()]
        for i, tok in enumerate(tokens):
            if tok not in weights:
                weights[tok] = {}
                slots.append(tok)
            try:
                weights[tok][p["id"]] = float(w_raw[i]) * 100 if i < len(w_raw) else None
            except ValueError:
                weights[tok][p["id"]] = None
    data = {"Asset": slots}
    for p in ports:
        data[p["name"]] = [weights[tok].get(p["id"]) for tok in slots]
    return pd.DataFrame(data)


def sync_alloc(df, ports):
    """Write the edited matrix back into each portfolio's tickers/weights strings."""
    for p in ports:
        if p["name"] not in df.columns:
            continue
        tks, wts = [], []
        for _, row in df.iterrows():
            tok = _strip_asset_label(row["Asset"] if pd.notna(row["Asset"]) else "")
            w = row[p["name"]]
            if not tok or pd.isna(w) or w == 0:
                continue
            tks.append(tok)
            wts.append(f"{w / 100:g}")
        p["tickers"] = ", ".join(tks)
        p["weights"] = ", ".join(wts)


def _alloc_struct_key(ports):
    """Editor key: changes when portfolios are added/removed/renamed (or an
    asset is added via search — see _alloc_nonce), which re-anchors the
    editor's edit-state on a freshly built base DataFrame."""
    nonce = st.session_state.get("_alloc_nonce", 0)
    return f"alloc_{abs(hash(tuple((p['id'], p['name']) for p in ports)))}_{nonce}"


# --- Asset display names (CN-listed ETFs/stocks) ---------------------------
# The matrix's Asset column shows CN-listed codes with their exchange Chinese
# short name ("511010.SS - 国债ETF国泰"). Live names come from Tencent's batch
# quote endpoint; CN_NAME_SEED keeps the common set labeled when offline.
# Display-layer only: the stored tickers/weights strings, JSON export/import
# and the engine always carry bare codes (sync_alloc strips the label).
CN_NAME_SEED = {
    "159915.SZ": "创业板ETF易方达", "159941.SZ": "纳指ETF广发", "159985.SZ": "豆粕ETF华夏",
    "510300.SS": "沪深300ETF华泰柏瑞", "511010.SS": "国债ETF国泰", "511130.SS": "30年国债ETF博时",
    "512400.SS": "有色金属ETF南方", "512890.SS": "红利低波ETF华泰柏瑞", "513100.SS": "纳指ETF国泰",
    "513500.SS": "标普500ETF博时", "515100.SS": "红利低波100ETF景顺", "515220.SS": "煤炭ETF国泰",
    "518880.SS": "黄金ETF华安", "588080.SS": "科创50ETF易方达",
}
NAME_SEP = " - "


# Process-lifetime circuit breaker: from hosts that can't reach Tencent's CDN
# (e.g. Streamlit Cloud), DNS resolution can HANG — requests' timeout does not
# cover getaddrinfo — wedging the script thread until the platform kills the
# app. After 2 straight failures the endpoint is never tried again in this
# process and labels come from CN_NAME_SEED only.
_cn_name_failures = {"n": 0}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_cn_names(symbols):
    """Batch-resolve .SS/.SZ tickers to Chinese short names via Tencent's
    quote API (GBK payload, no auth). The HTTP call runs on a daemon thread
    with a hard 4s deadline so even a hung DNS lookup can't block the script.
    Failures fall back to CN_NAME_SEED; tickers absent from the result simply
    display as the bare code."""
    codes = {}
    for tk in symbols:
        tk = str(tk).strip().upper()
        if tk.endswith(".SS") and tk[:-3].isdigit():
            codes["sh" + tk[:-3]] = tk
        elif tk.endswith(".SZ") and tk[:-3].isdigit():
            codes["sz" + tk[:-3]] = tk
    out = dict(CN_NAME_SEED)
    if not codes or _cn_name_failures["n"] >= 2:
        return out
    box = {}

    def _worker():
        try:
            r = requests.get("https://qt.gtimg.cn/q=" + ",".join(codes),
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
            box["payload"] = r.content.decode("gbk", errors="replace")
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(4.0)
    payload = box.get("payload")
    if payload is None:
        _cn_name_failures["n"] += 1
        return out
    _cn_name_failures["n"] = 0
    for m in re.finditer(r'v_(s[hz]\d+)="([^"]*)"', payload):
        parts = m.group(2).split("~")
        if len(parts) > 2 and parts[1] and m.group(1) in codes:
            out[codes[m.group(1)]] = parts[1]
    return out


def _label_asset(tok, names):
    """Slot token -> display label; composite members are labeled individually."""
    tok = str(tok).strip()
    if tok.startswith("(") and tok.endswith(")"):
        inner = [_label_asset(m, names) for m in tok[1:-1].split(",") if m.strip()]
        return "(" + ", ".join(inner) + ")"
    name = names.get(clean_ticker(tok)) if tok else None
    return f"{tok}{NAME_SEP}{name}" if name else tok


def _strip_asset_label(label):
    """Inverse of _label_asset: drop the display name, keep the raw token."""
    s = str(label or "").strip()
    if s.startswith("(") and s.endswith(")"):
        inner = [_strip_asset_label(m) for m in s[1:-1].split(",")]
        return "(" + ", ".join(t for t in inner if t) + ")"
    return s.split(NAME_SEP)[0].strip()


def label_alloc_assets(assets):
    """Label a whole Asset column with one batched name fetch for CN tickers.
    Never raises: on any failure the column falls back to bare codes."""
    toks = [str(a).strip() for a in assets]
    try:
        cn = sorted({clean_ticker(m)
                     for t in toks
                     for m in (t[1:-1].split(",") if t.startswith("(") and t.endswith(")") else [t])
                     if m.strip() and clean_ticker(m).endswith((".SS", ".SZ"))})
        if not cn:
            return toks
        names = fetch_cn_names(tuple(cn))
        return [_label_asset(t, names) for t in toks]
    except Exception:
        return toks


def _merge_editor_state(base_df, state):
    """Apply a data_editor edit-state dict ({edited_rows, added_rows,
    deleted_rows}) onto its base DataFrame. Deletions reference base row
    positions, so they are applied before additions."""
    df = base_df.copy()
    for ridx, changes in (state.get("edited_rows") or {}).items():
        r = int(ridx)
        if 0 <= r < len(df) and isinstance(changes, dict):
            for col, val in changes.items():
                if col in df.columns:
                    df.iloc[r, df.columns.get_loc(col)] = val
    drop = [int(i) for i in (state.get("deleted_rows") or []) if 0 <= int(i) < len(df)]
    if drop:
        df = df.drop(index=drop).reset_index(drop=True)
    add = [row for row in (state.get("added_rows") or []) if isinstance(row, dict)]
    if add:
        pad = pd.DataFrame([{c: row.get(c) for c in df.columns} for row in add])
        try:  # match base dtypes so all-NA pad columns don't warn on concat
            pad = pad.astype({c: str(df[c].dtype) for c in df.columns})
        except Exception:
            pass
        df = pd.concat([df, pad], ignore_index=True)
    return df


def flush_alloc_edits():
    """Fold the allocation editor's in-flight edit state into the stored
    tickers/weights strings; returns the merged frame (or None).

    The editor's edits live in per-widget state keyed by the editor key. Any
    action that changes that key (add/delete/rename portfolio, searchbox
    asset add) creates a fresh editor and discards that state — and because
    those controls render ABOVE the editor, an edit delivered in the same
    browser event would be lost before sync_alloc ever saw it. Call this
    before such mutations (the matrix-rebuild block does it for all
    key-changing paths)."""
    key = st.session_state.get('_alloc_key')
    base = st.session_state.get('_alloc_base')
    state = st.session_state.get(key) if key else None
    if base is None or not isinstance(state, dict):
        return None
    if not any(state.get(k) for k in ("edited_rows", "added_rows", "deleted_rows")):
        # No diffs in flight. NEVER sync in this situation: if the engine just
        # wiped the editor's state (early rerun above it), base is stale and
        # syncing would overwrite the strings with old values.
        return None
    try:
        merged = _merge_editor_state(base, state)
        sync_alloc(merged, st.session_state.portfolios_list)
        # Prune pending rows the user deleted in the editor, HERE while the
        # edit state is still alive: the engine may clean widget state at the
        # next rerun boundary, so the rebuild block cannot do this reliably.
        # Only rows previously shown in the editor (seen) are eligible — a
        # just-picked token is absent from this frame but must survive.
        seen = set(st.session_state.get('_alloc_pending_seen', []))
        if seen:
            still = {_strip_asset_label(a) for a in merged["Asset"]}
            st.session_state['_alloc_pending'] = [
                t for t in st.session_state.get('_alloc_pending', [])
                if t not in seen or t in still]
        return merged
    except Exception:
        return None  # editor-state format drift: skip rather than corrupt


# --- Per-slot bands (v2.5.0) -------------------------------------------------
# Portfolio-level overrides of the Down / Up band for individual slots, stored
# as port['slot_bands'] = {slot_label: {"down": pct|None, "up": pct|None}}.
# Each editor is a VIEW over that dict: rows = the portfolio's current slots
# (plus any stored label that no longer matches a slot, flagged so the user can
# blank it), synced straight back on every run. The frame the editor is
# anchored on is a SNAPSHOT kept while edits are in flight: st.data_editor
# hashes its data into the widget identity, so re-deriving the frame from the
# freshly synced dict after every cell edit would re-create the editor and
# drop a second edit typed before that rerun landed. The allocation matrix's
# Asset column is untouched: the matrix is a cross-portfolio slot view, whereas
# a band is a per-portfolio attribute.
ORPHAN_MARK = " ⚠ not in portfolio"


def slot_band_rows(port):
    """Base frame for a portfolio's per-slot band editor (blank = inherit)."""
    labels = slot_labels(port.get("tickers", ""))
    bands = port.get("slot_bands") or {}
    rows = [(l, bands.get(l) or {}) for l in labels]
    rows += [(l + ORPHAN_MARK, b or {}) for l, b in bands.items() if l not in labels]
    return pd.DataFrame({
        "Slot": [r[0] for r in rows],
        "Down %": pd.Series([r[1].get("down") for r in rows], dtype="float64"),
        "Up %": pd.Series([r[1].get("up") for r in rows], dtype="float64"),
    })


def sync_slot_bands(df, port):
    """Editor frame -> port['slot_bands'] in the canonical {"down": x, "up": y}
    form (None = inherit, same as normalize_slot_bands); a row with both sides
    blank is dropped."""
    out = {}
    for _, row in df.iterrows():
        lbl = str(row["Slot"]).replace(ORPHAN_MARK, "").strip()
        entry = {side: (int(row[col]) if pd.notna(row[col]) else None)
                 for col, side in (("Down %", "down"), ("Up %", "up"))}
        if lbl and (entry["down"] is not None or entry["up"] is not None):
            out[lbl] = entry
    port["slot_bands"] = out


def flush_slot_band_edits():
    """Fold in-flight per-slot band edits into the port dicts. Same rationale
    as flush_alloc_edits: controls rendered ABOVE the editors (Add, Save
    Default, the searchbox) would otherwise discard an edit delivered in the
    same browser event before the editor ever syncs it."""
    keys = st.session_state.get('_sb_keys') or {}
    for p in st.session_state.get('portfolios_list', []):
        key = keys.get(p.get('id'))
        state = st.session_state.get(key) if key else None
        if not isinstance(state, dict) or not state.get("edited_rows"):
            continue
        try:
            sync_slot_bands(_merge_editor_state(slot_band_rows(p), state), p)
        except Exception:
            pass  # editor-state format drift: skip rather than corrupt


def render_slot_band_editors(ports):
    keys = {}
    bases = st.session_state.get('_sb_base') or {}
    for p in ports:
        fresh = slot_band_rows(p)
        if fresh.empty:
            continue
        # Key changes with the slot set, so a structural change re-anchors the
        # editor on a fresh base (edits are synced every run, nothing is lost).
        key = f"sb_{p['id']}_{abs(hash(tuple(fresh['Slot'])))}"
        keys[p['id']] = key
        # Keep the anchored frame while the editor holds diffs (its identity
        # must not move between two quick edits); re-derive it from the dict
        # otherwise — idempotent, the dict already carries every synced edit.
        state = st.session_state.get(key)
        in_flight = isinstance(state, dict) and bool(state.get("edited_rows"))
        prev = bases.get(p['id'])
        if prev is None or prev[0] != key or not in_flight:
            prev = (key, fresh)
        bases[p['id']] = prev
        base = prev[1]
        # `expanded` is decided once per session and portfolio: flipping it after
        # the first override is synced would re-render the block around the
        # editor mid-edit. The user's own toggling is client-side and sticks.
        _exp = st.session_state.setdefault('_sb_expanded', {})
        with st.expander(f"Per-slot bands · {p['name']}",
                         expanded=_exp.setdefault(p['id'], bool(p.get('slot_bands')))):
            st.caption(
                "Override the Down / Up band for individual slots — e.g. keep a volatile "
                "5% crypto sleeve on a tight 40 / 40 while the core runs 60 / 100. Blank = "
                "inherit the portfolio band. Composite slots are keyed by their members "
                "joined with '+'. Read in the portfolio's band mode (Δ vs target / Leg vs "
                "rest). Applies to the RelDiff strategies only.")
            edited = st.data_editor(
                base, key=key, hide_index=True, width="stretch", num_rows="fixed",
                column_config={
                    "Slot": st.column_config.TextColumn("Slot", disabled=True, width="medium"),
                    "Down %": st.column_config.NumberColumn(
                        "Down %", min_value=1, max_value=200, step=1, format="%d",
                        help="Δ vs target: trigger when the slot falls below target × (1 − Down %). "
                             "Leg vs rest: when it has lost Down % against the rest of the "
                             "portfolio since its last reset."),
                    "Up %": st.column_config.NumberColumn(
                        "Up %", min_value=1, max_value=200, step=1, format="%d",
                        help="Δ vs target: trigger when the slot rises above target × (1 + Up %). "
                             "Leg vs rest: when it has gained Up % against the rest of the "
                             "portfolio since its last reset."),
                })
            sync_slot_bands(edited, p)
    st.session_state['_sb_keys'] = keys
    st.session_state['_sb_base'] = bases


@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_symbol_search(query):
    """Ticker/fund-name typeahead via Yahoo's symbol-search endpoint.
    Returns [(label, symbol), ...]; [] on any failure (search is best-effort)."""
    q = str(query or "").strip()
    if len(q) < 2:
        return []
    try:
        r = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": q, "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        quotes = r.json().get("quotes", [])
    except Exception:
        return []
    out = []
    for it in quotes:
        sym = it.get("symbol")
        if not sym:
            continue
        name = it.get("shortname") or it.get("longname") or ""
        tail = " · ".join(x for x in (it.get("quoteType"), it.get("exchange")) if x)
        out.append((f"{name} ({sym})" + (f" · {tail}" if tail else ""), sym))
    return out


@st.cache_data(ttl=86400)
def fetch_cpi_data():
    """Raises on failure so a transient FRED outage is NOT cached for 24h
    (exceptions bypass st.cache_data); the caller falls back to fixed-rate."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    cpi = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date')
    cpi.columns = ['CPI']
    return cpi

def render_summary_cards(metrics, color_map):
    """One compact card per series: identity dot + name, hero = annualized
    return (sign-colored), mini stats grid, final value footer. Replaces the
    old per-portfolio 6-card KPI rows so the chart lands above the fold."""
    cards = []
    for m in metrics:
        pc = color_map.get(m["name"], BENCH_COLOR)
        ann = m.get("_ann_ret")
        if m["ann_ret"] in ("-", "Err"):
            hero, pol, per = m["ann_ret"], "", ""
        else:
            hero = f"{ann:+.2%}"
            pol = "pos" if ann >= 0 else "neg"
            per = '<span class="per"> /yr</span>'
        total = f"{m['_total_ret']:+.2%}" if m["total_ret"] not in ("-", "Err") else m["total_ret"]
        final = f"Final ${m['final_nav']}" if m["final_nav"] not in ("-", "Err") else "—"
        cards.append(
            f'<div class="sum-card" style="--pc:{pc}">'
            f'<div class="sum-head"><span class="dot"></span>{m["name"]}</div>'
            f'<div class="sum-hero {pol}">{hero}{per}</div>'
            f'<div class="sum-minis">'
            f'<div><span class="k">Total</span><span class="v">{total}</span></div>'
            f'<div><span class="k">Max DD</span><span class="v">{m["max_dd"]}</span></div>'
            f'<div><span class="k">Sharpe</span><span class="v">{m["sharpe"]}</span></div>'
            f'<div><span class="k">Rebal</span><span class="v">{m["rebal_cnt"]}</span></div>'
            f'<div class="wide"><span class="k">Turnover</span><span class="v">{m.get("turnover", "-")}</span></div>'
            f'</div>'
            f'<div class="sum-foot">{final}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="sum-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _series_th(name, color_map):
    """Table header cell with the series' identity dot."""
    c = color_map.get(name)
    dot = f'<span class="th-dot" style="background:{c}"></span>' if c else ""
    return f"<th>{dot}{name}</th>"


def render_comparison_table(metrics, color_map):
    """Render a side-by-side comparison table (kpi-dashboard-design: executive summary pattern)"""
    if len(metrics) < 2:
        return

    # (label, display key, raw key, higher_is_better). Max drawdown is stored as a
    # negative fraction, so "higher" (closer to 0) IS better; turnover is a tax
    # proxy (California taxable account, HIFO) — lower is better.
    fields = [("Total Return", "total_ret", "_total_ret", True),
              ("Ann. Return", "ann_ret", "_ann_ret", True),
              ("Max Drawdown", "max_dd", "_max_dd", True),
              ("Sharpe Ratio", "sharpe", "_sharpe", True),
              ("Rebalances", "rebal_cnt", None, None),
              ("Turnover /yr", "turnover", "_turnover", False)]

    header = "<tr><th>Metric</th>"
    for m in metrics:
        header += _series_th(m["name"], color_map)
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

        # Best / worst among the series that HAVE a value (the benchmark shows
        # "-" for turnover; an errored series must not blank the whole row).
        avail = [v for v in raw_vals if v is not None]
        best = worst = None
        if higher_better is not None and len(avail) >= 2:
            best, worst = (max(avail), min(avail)) if higher_better else (min(avail), max(avail))

        for i, m in enumerate(metrics):
            val = m.get(key, "-")
            cls = ""
            if raw_vals[i] is not None and best is not None:
                if raw_vals[i] == best: cls = "best"
                elif raw_vals[i] == worst: cls = "worst"
            rows += f'<td class="{cls}">{val}</td>'
        rows += "</tr>"

    st.markdown(f'<table class="cmp-table">{header}{rows}</table>', unsafe_allow_html=True)


def render_annual_returns_table(comp_df, metrics, color_map):
    """HTML table: rows = calendar years (newest first) + CAGR; cols = each series in comp_df."""
    rows_data = compute_annual_returns(comp_df)
    if not rows_data:
        return
    cols = list(comp_df.columns)

    header = "<tr><th>Year</th>" + "".join(_series_th(c, color_map) for c in cols) + "</tr>"

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
def apply_config(loaded_config):
    """Apply an imported/saved config dict to session state and rerun."""
    _apply_config_state(loaded_config)  # leaves run_backtest False: explicit Analyze required
    st.rerun()


@st.dialog("Set startup default")
def _confirm_save_default():
    """Confirmation gate for Save Default (accidental clicks would silently
    replace the persisted setup). Also hosts the reset-to-built-ins action."""
    ports = st.session_state.portfolios_list
    st.markdown(
        "New sessions will open with **" + ", ".join(p['name'] for p in ports) + "** · "
        f"benchmark `{st.session_state['bi']}` · start {st.session_state['sd']} · "
        f"${st.session_state['init_funds']:,}")
    has_saved = DEFAULT_CONFIG_PATH.is_file() or st.session_state.get('_ls_default_present')
    if has_saved:
        st.warning("A saved default already exists — Confirm will **replace** it.")
    c1, c2 = st.columns(2)
    if c1.button("Confirm & Save", type="primary", width="stretch"):
        payload = json.dumps({
            "benchmark": st.session_state['bi'],
            "start_date": str(st.session_state['sd']),
            "initial_funds": st.session_state['init_funds'],
            "portfolios": st.session_state.portfolios_list,
        }, indent=2, ensure_ascii=False)
        try:
            SAVED_CONFIG_DIR.mkdir(exist_ok=True)
            DEFAULT_CONFIG_PATH.write_text(payload, encoding="utf-8")
        except Exception as e:      # cloud FS quirks: browser copy still proceeds
            st.toast(f"File save failed: {e}", icon="⚠️")
        if HAS_JS_EVAL:
            st.session_state['_ls_payload'] = payload      # -> localStorage setItem
            st.session_state['_ls_nonce'] = str(uuid.uuid4())[:8]
        st.session_state['_ls_default_present'] = True
        st.session_state['_flash_toast'] = "Saved — new sessions now open with this setup"
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()
    if has_saved:
        st.divider()
        if st.button(":material/restart_alt: Reset default to built-ins", width="stretch",
                     help="Delete the saved default (file + this browser's copy); new "
                          "sessions open with the built-in config again."):
            DEFAULT_CONFIG_PATH.unlink(missing_ok=True)
            if HAS_JS_EVAL:
                st.session_state['_ls_payload'] = ""       # "" -> localStorage removeItem
                st.session_state['_ls_nonce'] = str(uuid.uuid4())[:8]
            st.session_state['_ls_default_present'] = False
            st.session_state['_flash_toast'] = "Default reset — new sessions open with built-ins"
            st.rerun()

# Live Portfolio mode: the desk replaces the whole Backtest UI. It reuses this
# page's price cache (fetch_price_history) and saved configs; the Backtest
# settings keep their values in session state while the other mode is shown.
if APP_MODE == "Live Portfolio":
    import live_page
    live_page.render_live_desk(fetch_price_history, SAVED_CONFIG_DIR)
    st.stop()

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

    # Placeholder only: the sidebar renders BEFORE the main section's write-backs
    # (row edits, matrix sync). The actual download button is rendered into this
    # slot at the end of the portfolio section so the payload is this-run fresh.
    export_slot = st.container()

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
    STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_RD_FULL, STRAT_ASYM
]

ROW_SPEC = [0.28, 1.9, 2.2, 0.8, 0.8, 1.25, 0.5]

for _w in st.session_state.pop('_flash_warn', None) or []:
    st.warning(_w)

st.markdown('<div class="sec-label">Portfolios</div>', unsafe_allow_html=True)
with st.container(border=True):
    hdr = st.columns(ROW_SPEC, vertical_alignment="center")
    for c, lbl in zip(hdr, ["", "Name", "Strategy", "Down %", "Up %", "Band", ""]):
        with c:
            if lbl: st.markdown(f'<div class="col-cap">{lbl}</div>', unsafe_allow_html=True)

    total_portfolios = len(st.session_state.portfolios_list)
    for i, port in enumerate(st.session_state.portfolios_list):
        if 'id' not in port: port['id'] = str(uuid.uuid4())
        port.setdefault('thr_up', port['thr'])      # rows injected without the v2.5.0 fields
        port.setdefault('slot_bands', {})
        port.setdefault('band_mode', BAND_MODE_REL)  # ... or the v2.5.2 field

        cols = st.columns(ROW_SPEC, vertical_alignment="center")
        with cols[0]: st.markdown(
            f'<span class="row-dot" style="background:{SERIES_COLORS[i % len(SERIES_COLORS)]};'
            f'color:{SERIES_COLORS[i % len(SERIES_COLORS)]}"></span>', unsafe_allow_html=True)
        with cols[1]: port['name'] = st.text_input(
            "Name", port['name'], key=f"n_{port['id']}", label_visibility="collapsed")
        with cols[2]: port['strat'] = st.selectbox(
            "Strategy", strategy_options, index=strategy_options.index(port['strat']),
            key=f"s_{port['id']}", label_visibility="collapsed",
            help="RelDiff Full = any breach resets all · Mixed = major (≥10%) breach resets all, "
                 "else local · Local = only breached slots reset · Asymmetric = minors (<6%) "
                 "trigger at 2.5×↑ / 1.25×↓ the band.")
        with cols[3]: port['thr'] = st.number_input(
            "Down %", 1, 200, port['thr'], key=f"tr_{port['id']}", label_visibility="collapsed",
            help="DOWN band D. Δ vs target: a slot triggers when its relative deviation vs "
                 "target drops below −D%, i.e. its weight falls under target × (1 − D); e.g. 60 "
                 "→ a 10% slot triggers below 4%. Leg vs rest: it triggers once it has lost D% "
                 "against the rest of the portfolio since its last reset (50 → it halved). "
                 "Asymmetric RelDiff uses this as its single band; Periodic / Buy & Hold "
                 "ignore it.")
        # Up % starts equal to Down % when a portfolio is created or loaded and
        # is independent from then on. It is a plain numeric input exactly like
        # Down %: the server never writes into it and it is never empty. Two
        # earlier variants lost digits typed here right after a Down % commit
        # (a rerun landing mid-typing): pushing the Down value into this widget
        # through session state replaced them, and a nullable "empty = mirror
        # Down" input was cleared by the frontend on every rerun while empty.
        with cols[4]: port['thr_up'] = st.number_input(
            "Up %", 1, 200, port['thr_up'], key=f"tu_{port['id']}", label_visibility="collapsed",
            help="UP band U. Δ vs target: a slot triggers when its relative deviation vs "
                 "target rises above +U%, i.e. its weight exceeds target × (1 + U); e.g. 100 → "
                 "a 10% slot triggers above 20%. Leg vs rest: it triggers once it has gained "
                 "U% against the rest of the portfolio since its last reset (100 → it "
                 "doubled). Starts equal to Down % and is set separately from then on (equal "
                 "values = the symmetric band). Ignored by Asymmetric RelDiff / Periodic / "
                 "Buy & Hold.")
        # Band mode (v2.5.2): what the two bands measure. A plain selectbox like
        # Strategy — the server never writes into it.
        with cols[5]: port['band_mode'] = st.selectbox(
            "Band", list(BAND_MODES), index=list(BAND_MODES).index(port['band_mode']),
            format_func=BAND_MODE_LABELS.get, key=f"bm_{port['id']}", label_visibility="collapsed",
            help="What Down % / Up % measure. **Δ vs target** — the original rule: the slot's "
                 "relative weight deviation (w − target) / target. It is size-biased: a 35% slot "
                 "must beat the rest of the portfolio by +136% to reach U = 60, a 10% slot only "
                 "by +71%. **Leg vs rest** (v2.5.2, size-neutral): the slot's cumulative return "
                 "relative to the rest of the portfolio since its last reset, g = (w/t) / "
                 "((1−w)/(1−t)) − 1; U = 100 → the slot doubled against the rest, D = 50 → it "
                 "halved, whatever its size. Per-slot bands are read in the same mode. RelDiff "
                 "strategies only; Asymmetric RelDiff / Periodic / Buy & Hold ignore it.")
        with cols[6]:
            if total_portfolios > 1:
                st.button(":material/delete:", key=f"del_{port['id']}",
                          on_click=delete_portfolio, args=(i,), help="Remove this portfolio")

    # --- Actions: add portfolio / persist current setup as startup default ---
    act_cols = st.columns([1.4, 2.2, 4.4])
    with act_cols[0]:
        if st.button(":material/add_circle: Add", width="stretch",
                     help="Add a portfolio (copies the last one's allocation)."):
            flush_alloc_edits()  # copy must see edits delivered in this same event
            flush_slot_band_edits()
            ports_now = st.session_state.portfolios_list
            last_port = ports_now[-1] if ports_now else {"tickers": "", "weights": ""}
            ports_now.append({
                "id": str(uuid.uuid4()),
                "name": next_port_name(ports_now),
                "tickers": last_port["tickers"], "weights": last_port["weights"],
                "strat": STRAT_RD_MIXED, "thr": 40, "thr_up": 40, "slot_bands": {}, "band_mode": BAND_MODE_REL
            })
            st.rerun()
    with act_cols[1]:
        if st.button(":material/bookmark_add: Save Default", width="stretch",
                     help="Save the current setup (portfolios, benchmark, start date, initial "
                          "funds) as the startup default — new sessions open with it. Stored "
                          "in Backtest/_default.json AND this browser (the browser copy "
                          "survives cloud redeploys). A confirmation dialog guards against "
                          "accidental clicks and offers reset-to-built-ins."):
            flush_alloc_edits()  # save must include edits delivered in this same event
            flush_slot_band_edits()
            _errs = validate_inputs(st.session_state.portfolios_list, st.session_state['bi'])
            if _errs:
                st.error("Default NOT saved — fix first: " + " · ".join(_errs))
            else:
                _confirm_save_default()

    # --- Allocation matrix: one row per slot, one % column per portfolio ---
    st.markdown(
        '<div class="col-cap" style="margin-top:0.5rem">Allocation · weights in % · '
        'blank or 0 = not held · wrap "(A, B)" for a composite slot · '
        'add rows below for new assets</div>', unsafe_allow_html=True)

    ports = st.session_state.portfolios_list
    port_names = [p['name'] for p in ports]
    name_clash = len(set(port_names)) != len(port_names)
    asset_clash = any(str(n).strip() == "Asset" for n in port_names)
    if name_clash or asset_clash:
        # No editor renders in this state, so its widget state is destroyed at
        # end-of-run. Drop the anchor: on recovery the same names reproduce the
        # same key, and reusing it would re-anchor the editor on a stale base —
        # silently reverting every edit since the last rebuild. ("Asset" would
        # additionally overwrite the matrix's token column in build_alloc_df.)
        st.session_state.pop('_alloc_key', None)
        st.error("Duplicate portfolio names — rename them above before editing allocations."
                 if name_clash else
                 '"Asset" is reserved for the matrix\'s first column — rename that portfolio.')
    else:
        # Surface storage the matrix cannot faithfully represent BEFORE the
        # first sync rewrites it: ragged or unparseable weights in imported /
        # hand-edited configs would otherwise be dropped with no trace.
        _lossy = []
        for p in ports:
            _toks = _slot_tokens(p.get("tickers", ""))
            _wr = [w.strip() for w in str(p.get("weights", "")).replace("，", ",").split(",") if w.strip()]
            _bad = False
            for w in _wr:
                try:
                    float(w)
                except ValueError:
                    _bad = True
            if _toks and (_bad or len(_wr) != len(_toks)):
                _lossy.append(p['name'])
        if _lossy:
            st.warning("**" + ", ".join(_lossy) + "**: weights don't align with tickers "
                       "(count mismatch or unparseable value). Misaligned tickers show a "
                       "blank weight below and drop from the stored config when the matrix "
                       "syncs — fill their weights now or re-import a corrected JSON.")

        def _build_alloc_base():
            """Fresh editor base from the stored strings: union of slots, plus
            pending (searchbox-added, not yet weighted) rows, CN labels last.
            Pending rows the user weighted (now in the strings) leave the
            list; deleted-row pruning happens inside flush_alloc_edits while
            the editor state is still alive."""
            base = build_alloc_df(ports)
            existing = set(str(a).strip() for a in base["Asset"])
            pending = [t for t in st.session_state.get('_alloc_pending', [])
                       if t not in existing]
            st.session_state['_alloc_pending'] = pending
            st.session_state['_alloc_pending_seen'] = list(pending)
            if pending:
                pad = pd.DataFrame({"Asset": pending,
                                    **{p['name']: [None] * len(pending) for p in ports}})
                base = pd.concat([base, pad], ignore_index=True)
            base["Asset"] = label_alloc_assets(base["Asset"])
            return base

        alloc_key = _alloc_struct_key(ports)
        if st.session_state.get('_alloc_key') != alloc_key:
            # The key changed this run (port added/deleted/renamed or searchbox
            # add): fold the outgoing editor's in-flight edits into the strings
            # FIRST, or edits delivered in this same browser event are lost.
            flush_alloc_edits()
            st.session_state['_alloc_key'] = alloc_key
            st.session_state['_alloc_base'] = _build_alloc_base()
        else:
            # Same key. The engine DROPS the editor's accumulated diffs on any
            # run that ends before instantiating it — st_searchbox fires an
            # internal rerun per search keystroke, above the editor. With the
            # diffs gone the editor would re-anchor on this stale base and
            # sync_alloc would write the OLD values back over the strings.
            # Whenever no diffs are in flight, re-derive the base from the
            # strings instead of trusting the snapshot (idempotent when
            # nothing changed).
            _ed_state = st.session_state.get(alloc_key)
            _has_diffs = isinstance(_ed_state, dict) and any(
                _ed_state.get(k) for k in ("edited_rows", "added_rows", "deleted_rows"))
            if not _has_diffs:
                st.session_state['_alloc_base'] = _build_alloc_base()

        # Typeahead add-asset search (Yahoo symbol search, like Portfolio
        # Visualizer's ticker box). Optional: without the package the matrix
        # still accepts hand-typed tickers in new rows.
        if HAS_SEARCHBOX:
            # Capture any in-flight edits BEFORE the component runs: its
            # per-keystroke internal rerun would otherwise drop them unseen.
            flush_alloc_edits()
            flush_slot_band_edits()
            sb_cols = st.columns([2.8, 4.2])
            with sb_cols[0]:
                picked = st_searchbox(
                    yahoo_symbol_search, key="asset_search",
                    placeholder="Add asset — search ticker or fund name…",
                    clear_on_submit=True, debounce=250)
            if picked:
                tok = clean_ticker(str(picked))
                rows_now = set(_strip_asset_label(a) for a in st.session_state['_alloc_base']["Asset"])
                pend = st.session_state.setdefault('_alloc_pending', [])
                if tok not in rows_now and tok not in pend:
                    pend.append(tok)
                    st.session_state['_alloc_nonce'] = st.session_state.get('_alloc_nonce', 0) + 1
                    # Consume the pick: st_searchbox returns the last result on
                    # every rerun otherwise, resurrecting removed assets on each
                    # rebuild and leaking into freshly loaded configs.
                    _sb_state = st.session_state.get("asset_search")
                    if isinstance(_sb_state, dict):
                        _sb_state["result"] = None
                    st.rerun()

        col_cfg = {"Asset": st.column_config.TextColumn(
            "Asset", width="medium",
            help='Ticker (e.g. QQQM, 0700.HK) or a composite slot "(DBMF, KMLM)": '
                 'one weight, split equally inside, rebalanced as one block. '
                 'CN-listed codes (.SS/.SZ) show their Chinese short name after the '
                 'code — type just the code when adding; the label is display-only.')}
        for p in ports:
            col_cfg[p['name']] = st.column_config.NumberColumn(
                p['name'], min_value=0.0, max_value=100.0, step=0.5, format="%.2f%%",
                help=f"Target weight of each asset in {p['name']}, in percent.")

        edited_alloc = st.data_editor(
            st.session_state['_alloc_base'], key=alloc_key, num_rows="dynamic",
            column_config=col_cfg, hide_index=True, width="stretch")
        sync_alloc(edited_alloc, ports)

        sums = []
        for p in ports:
            total = float(pd.to_numeric(edited_alloc[p['name']], errors='coerce').fillna(0).sum())
            ok = abs(total - 100) < 0.01
            color, mark = ('var(--good)', '✓') if ok else ('var(--bad)', '≠ 100%')
            sums.append(f'<span style="color:{color};font-weight:600">{p["name"]}: {total:.4g}% {mark}</span>')
        st.markdown('<div class="alloc-sums">' + ' &nbsp;·&nbsp; '.join(sums) + '</div>',
                    unsafe_allow_html=True)

        # --- Per-slot bands (v2.5.0): one collapsible editor per portfolio ---
        render_slot_band_editors(ports)

    btn_cols = st.columns([2, 6])
    with btn_cols[0]:
        run_clicked = st.button(":material/play_arrow: Analyze", type="primary", width="stretch")
        if run_clicked:
            error_msgs = validate_inputs(st.session_state.portfolios_list, bench_in)
            if not error_msgs:
                forget_failed_prices()   # an explicit Analyze re-requests what had no data
                st.session_state.run_backtest = True
            else:
                st.session_state.run_backtest = False
                for msg in error_msgs: st.error(msg)

if _t := st.session_state.pop('_flash_toast', None):
    st.toast(_t, icon="✅")

# Rendered into the sidebar slot AFTER the portfolio section so the exported
# JSON reflects this run's row edits and matrix sync (not last run's state).
with export_slot:
    st.download_button(
        label=":material/download: Export",
        data=json.dumps({
            "benchmark": st.session_state['bi'],
            "start_date": str(st.session_state['sd']),
            "initial_funds": st.session_state['init_funds'],
            "portfolios": st.session_state.portfolios_list,
        }, indent=2, ensure_ascii=False),
        file_name="backtest_config.json", mime="application/json", width="stretch")

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
        bench_tk = clean_ticker(bench_in)
        all_tks = sorted(set([bench_tk] + [t for _, p_tks, _, _ in parsed_ports for t in p_tks]))
        try:
            price_data, fetch_failures = fetch_price_history(
                tuple(all_tks), str(start_d - timedelta(days=20)))
        except Exception as e:
            # Nothing was cached; show once and let the next Analyze retry
            # instead of re-hitting Yahoo on every widget rerun.
            st.session_state.run_backtest = False
            st.error(f"Download error: {e}"); st.stop()
        if bench_tk in fetch_failures:
            st.session_state.run_backtest = False
            st.error(f"Benchmark **{bench_tk}** could not be downloaded: {fetch_failures[bench_tk]}. "
                     "Nothing was cached \u2014 wait a moment and click Analyze again.")
            st.stop()
        if fetch_failures:
            st.warning("Yahoo returned no prices for "
                       + "; ".join(f"**{tk}** ({why})" for tk, why in fetch_failures.items())
                       + " \u2014 excluded from this run; click Analyze again to retry.")

        # Scrub data glitches: corrupted listing-day prints (wrong scale on day 1)
        # and isolated mid-series spikes that revert on the next print. Genuine
        # multi-print moves are never touched. See backtest_core for details.
        scrubbed = scrub_leading_glitches(price_data)
        spiked = scrub_isolated_spikes(price_data)
        if scrubbed:
            st.warning("Dropped corrupted listing-day price(s): " + "; ".join(scrubbed))
        if spiked:
            st.warning("Dropped isolated mid-series price glitch(es): " + "; ".join(spiked))

        # Calendar alignment, late-listing start and month-end sampling live in
        # backtest_core.align_price_data (v2.6.0) so the live desk runs on the
        # identical frame; messages and their order are unchanged.
        all_port_tks = sorted({t for _, p_tks, _, _ in parsed_ports for t in p_tks})
        aligned = align_price_data(price_data, bench_tk, start_d, all_port_tks)
        if aligned["notice"]:
            (st.warning if aligned["notice"][0] == "warning" else st.info)(aligned["notice"][1])
        if aligned["error"]:
            st.error(aligned["error"]); st.stop()
        actual_start_day = aligned["actual_start_day"]
        final_data, price_df = aligned["final_data"], aligned["price_df"]
        days_span = (final_data.index[-1] - final_data.index[0]).days

        comp_df = pd.DataFrame(index=price_df.index)
        bench_nav = (price_df[bench_tk] / price_df[bench_tk].iloc[0]) * init_f
        comp_df[f"Benchmark({bench_in})"] = bench_nav

        res_list = {}
        valid_ports_meta = {}
        port_stats, drift_tables, rebal_scope = {}, {}, {}

        def clean_col(c):
            target = str(c).strip()
            for tk, name in TICKER_TO_NAME.items():
                if tk in target: return name
            return target

        def _pct_to_float(s):
            try: return float(str(s).rstrip('%')) / 100.0
            except (ValueError, TypeError): return np.nan

        for p, p_tks, p_wts, p_comp in parsed_ports:
            # Dataless elements, renormalisation, composite groups and per-slot
            # bands: backtest_core.prepare_portfolio (v2.6.0), shared with the
            # live desk; messages unchanged.
            prep = prepare_portfolio(p, p_tks, p_wts, p_comp, price_df)
            if prep["error"]:
                st.error(prep["error"]); continue
            for _lvl, _msg in prep["notices"]:
                (st.warning if _lvl == "warning" else st.info)(_msg)
            valid_p_tks, w_series, groups = prep["valid_tks"], prep["w_series"], prep["groups"]
            slots, slot_survivors = prep["slots"], prep["slot_survivors"]
            thr_dn, thr_up = prep["thr_dn"], prep["thr_up"]

            res_df, cnt, pnl_rec, bt_stats = run_detailed_backtest(
                p['strat'], price_df[valid_p_tks], w_series, init_f, thr_dn, groups=groups,
                threshold_up=thr_up, return_stats=True,
                band_mode=p.get('band_mode', BAND_MODE_REL))
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
                # Rebalance scope per date for the row colours (v2.5.4). .get():
                # a Cloud container still running a cached pre-2.5.4 engine has
                # no event log -> uniform colours instead of a crash.
                rebal_scope[p['name']] = {pd.Timestamp(e["date"]).strftime('%Y-%m-%d'): e["scope"]
                                          for e in (bt_stats.get("rebal_events") or [])}

                # Turnover + per-slot weight drift (v2.5.0). The band column shows
                # the EFFECTIVE Down/Up band per slot for the RelDiff strategies
                # (Asymmetric RelDiff has its own major/minor rule; Periodic and
                # Buy & Hold have none); v2.5.3 adds the weight levels at which
                # that band is crossed (and, for Leg vs rest, what they amount to
                # as a Δ vs target band) so Min / Max can be read against them.
                port_stats[p['name']] = bt_stats
                _band_of = lambda b, sid: (b.get(sid, b.get("*")) if isinstance(b, dict) else b)
                show_band = p['strat'] in (STRAT_RD_FULL, STRAT_RD_MIXED, STRAT_RD_LOCAL)
                p_mode = p.get('band_mode', BAND_MODE_REL)
                band_col = "Band · " + BAND_MODE_LABELS[p_mode]
                _lvl = lambda w: "never" if w is None else f"{w:.2%}"
                _dlt = lambda w, t: "—" if w is None else f"{w / t - 1:+.0%}"
                drift_rows = []
                for sid in bt_stats["slot_ids"]:
                    row = {"Slot": "+".join(clean_col(m) for m in bt_stats["slot_members"][sid]),
                           "Target": bt_stats["slot_target"][sid],
                           "Min": bt_stats["weight_min"].get(sid),
                           "Max": bt_stats["weight_max"].get(sid)}
                    if show_band:
                        d_eff = _band_of(thr_dn, sid)
                        u_eff = _band_of(thr_dn if thr_up is None else thr_up, sid)
                        row[band_col] = f"−{d_eff:.0%} / +{u_eff:.0%}"
                        w_dn, w_up = band_trigger_weights(row["Target"], d_eff, u_eff, p_mode)
                        row["Trigger <"] = _lvl(w_dn)
                        row["Trigger >"] = _lvl(w_up)
                        if p_mode == BAND_MODE_RATIO:
                            row["Δ equiv."] = f"{_dlt(w_dn, row['Target'])} / {_dlt(w_up, row['Target'])}"
                    drift_rows.append(row)
                drift_tables[p['name']] = pd.DataFrame(drift_rows)

        # --- Inflation Adjustment ---
        if inf_adj:
            start_ts = pd.Timestamp(actual_start_day)
            end_ts = comp_df.index[-1]
            try:
                cpi_raw = fetch_cpi_data()
            except Exception:
                cpi_raw = None
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
                    edited_inf = st.data_editor(
                        inf_df, hide_index=True, width="stretch",
                        column_config={"Year": st.column_config.NumberColumn(
                            "Year", disabled=True, format="%d")})
                rate_map = dict(zip(edited_inf['Year'].astype(int), edited_inf['Inflation(%)'].fillna(3.0) / 100))
                discount_factors = pd.Series(1.0, index=comp_df.index)
                for i, date in enumerate(comp_df.index):
                    factor = 1.0
                    for y in range(start_ts.year, date.year + 1):
                        rate = rate_map.get(y, 0.03)
                        y_begin = max(pd.Timestamp(f"{y}-01-01"), start_ts)
                        # Exclusive upper bound (Jan 1 of y+1): a full calendar
                        # year discounts as ~365/365.25, and the Dec31->Jan1
                        # boundary day is counted exactly once.
                        y_end = min(pd.Timestamp(f"{y + 1}-01-01"), date)
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
        bench_m["turnover"], bench_m["_turnover"] = "-", None
        metrics.append(bench_m)
        for p_name, cnt in valid_ports_meta.items():
            m = calculate_metrics(comp_df[p_name], cnt, risk_free_rate=rf_rate)
            m["name"] = p_name
            _s = port_stats.get(p_name)
            m["_turnover"] = _s["turnover_yr"] if _s else None
            m["turnover"] = f"{_s['turnover_yr']:.1%}/yr" if _s else "-"
            metrics.append(m)

        # --- Identity colors: keyed to editor ROW INDEX so a series keeps its
        # color across charts, cards and tables even if another one is dropped.
        color_map = {f"Benchmark({bench_in})": BENCH_COLOR}
        for i_p, p in enumerate(st.session_state.portfolios_list):
            color_map.setdefault(p['name'], SERIES_COLORS[i_p % len(SERIES_COLORS)])

        # --- Summary cards (one per series) ---
        render_summary_cards(metrics, color_map)

        # --- Comparison Table ---
        if len(metrics) >= 2:
            with st.expander("Performance Comparison", expanded=True):
                render_comparison_table(metrics, color_map)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        # --- Chart ---
        comp_df.index.name = 'Date'
        chart_df = comp_df / comp_df.iloc[0] - 1
        x_axis_format = '%Y-%m-%d' if days_span < 90 else '%Y-%m'
        label_angle = -45 if days_span < 90 else 0

        # Series color scale: identity colors, benchmark drawn as a dashed
        # neutral context line (secondary encoding on top of the gray hue).
        bench_col = f"Benchmark({bench_in})"
        series_domain = list(chart_df.columns)
        series_range = [color_map.get(c, BENCH_COLOR) for c in series_domain]
        series_scale = alt.Scale(domain=series_domain, range=series_range)
        bench_dash = alt.condition(alt.datum.Portfolio == bench_col,
                                   alt.value([5, 4]), alt.value([1, 0]))

        chart_data = chart_df.reset_index().melt('Date', var_name='Portfolio', value_name='Return')
        rule_data = chart_df.reset_index()
        nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Date'], empty=False)

        line = alt.Chart(chart_data).mark_line(strokeWidth=2).encode(
            x=alt.X('Date:T', axis=alt.Axis(format=x_axis_format, title=None, labelAngle=label_angle, grid=False)),
            y=alt.Y('Return:Q', axis=alt.Axis(format='.1%', title='Cumulative Return' + (' (CPI adj.)' if inf_adj else ''), grid=True, gridDash=[3,3], gridColor=_T["grid"])),
            color=alt.Color('Portfolio:N', legend=alt.Legend(orient='top', title=None, labelFontSize=12), scale=series_scale),
            strokeDash=bench_dash
        )

        # Zero baseline
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color=_T["muted"], strokeDash=[4,4], strokeWidth=1).encode(y='y:Q')

        tooltips = [alt.Tooltip('Date:T', format='%Y-%m-%d', title='Date')]
        for col in chart_df.columns:
            tooltips.append(alt.Tooltip(field=col, type='quantitative', format='.2%', title=col))

        selectors = alt.Chart(rule_data).mark_rule(opacity=0.001, strokeWidth=40).encode(
            x='Date:T', tooltip=tooltips
        ).add_params(nearest)

        rules = alt.Chart(rule_data).mark_rule(color=_T["muted"], strokeDash=[3,3]).encode(
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
                y=alt.Y('Drawdown:Q', axis=alt.Axis(format='.0%', title='Drawdown', grid=True, gridDash=[3,3], gridColor=_T["grid"])),
                color=alt.Color('Portfolio:N', legend=alt.Legend(orient='top', title=None, labelFontSize=12), scale=series_scale),
                strokeDash=bench_dash
            )
            dd_tooltips = [alt.Tooltip('Date:T', format='%Y-%m-%d', title='Date')]
            for col in dd_df.columns:
                dd_tooltips.append(alt.Tooltip(field=col, type='quantitative', format='.2%', title=col))
            dd_selectors = alt.Chart(dd_wide).mark_rule(opacity=0.001, strokeWidth=40).encode(
                x='Date:T', tooltip=dd_tooltips
            ).add_params(dd_nearest)
            dd_rules = alt.Chart(dd_wide).mark_rule(color=_T["muted"], strokeDash=[3,3]).encode(
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
            for tab, lbl in zip(tabs, tab_names):
                with tab:
                    _s = port_stats.get(lbl)
                    cap = f"Start: {actual_start_day.date()}"
                    if _s:
                        # Dollar signs are escaped: st.caption renders markdown, where a
                        # "$...$" pair would be typeset as LaTeX.
                        cap += (f" · Turnover {_s['turnover_yr']:.1%}/yr = sold \\${_s['sold_total']:,.0f} "
                                f"÷ mean NAV \\${_s['nav_mean']:,.0f} ÷ {_s['years']:.2f} yrs")
                    st.caption(cap)
                    if lbl in drift_tables and not drift_tables[lbl].empty:
                        _dt = drift_tables[lbl]
                        _has_trig = "Trigger <" in _dt.columns
                        st.markdown('<div class="col-cap">Slot weight range · weights carried between '
                                    'bars (Init / Hold / Post-Rebal)'
                                    + (' · Trigger = weight at which the band in force is crossed'
                                       if _has_trig else '') + '</div>', unsafe_allow_html=True)
                        _pct_col = lambda name: st.column_config.NumberColumn(name, format="%.2f%%")
                        _cfg = {"Target": _pct_col("Target"), "Min": _pct_col("Min"), "Max": _pct_col("Max")}
                        if _has_trig:
                            _cfg["Trigger <"] = st.column_config.TextColumn(
                                "Trigger <", help="The slot triggers once its weight falls below this "
                                                  "level (never: a DOWN band of 100% or more cannot be "
                                                  "crossed). Compare with Min.")
                            _cfg["Trigger >"] = st.column_config.TextColumn(
                                "Trigger >", help="The slot triggers once its weight rises above this "
                                                  "level. Compare with Max.")
                        if "Δ equiv." in _dt.columns:
                            _cfg["Δ equiv."] = st.column_config.TextColumn(
                                "Δ equiv.", help="The same trigger levels expressed as a Δ vs target "
                                                     "band (relative weight deviation, down / up) — what this "
                                                     "slot's Leg-vs-rest band amounts to under the original rule.")
                        st.dataframe(
                            _dt.assign(**{c: _dt[c] * 100 for c in ("Target", "Min", "Max")}),
                            hide_index=True, width="content", column_config=_cfg)
                    _scope = rebal_scope.get(lbl, {})
                    if _s and _s.get("rebal_events"):
                        st.markdown(rebal_legend_html(_s.get("rebal_global", 0), _s.get("rebal_local", 0)),
                                    unsafe_allow_html=True)
                    st.dataframe(res_list[lbl].style.apply(lambda r, _sc=_scope: rebal_row_style(r, _sc.get(r.get('Date'))), axis=1)
                                 .format({"NAV": "{:,.2f}"}), width="stretch")

        # --- Annual Returns by Calendar Year ---
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        with st.expander("Annual Returns by Calendar Year", expanded=True):
            render_annual_returns_table(comp_df, metrics, color_map)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="es-icon">📊</div>
        <div class="es-title">Ready when you are</div>
        <div class="es-body">Configure the portfolios above — or load a saved config from the
        sidebar — then hit <b>Analyze</b>. Prices come from Yahoo Finance with dividends
        reinvested; windows ≥ 90 days are sampled at month-end by design.</div>
    </div>
    """, unsafe_allow_html=True)

# --- localStorage bridge components (rendered LAST on purpose) ---------------
# Both are invisible (CSS above) and transient: the reader shows until the
# browser has answered, the writer while a Save/Reset payload is pending. An
# element that appears or disappears shifts the position of everything after
# it inside the same block, and the frontend re-mounts widgets by position —
# dropping any input being typed at that moment. At the end of the main area
# nothing comes after them, so mounting or unmounting them shifts nothing.
if HAS_JS_EVAL and not st.session_state.get('_ls_checked'):
    streamlit_js_eval(js_expressions=LS_GET_EXPR, key="_ls_get")   # read at the top of the next run

if HAS_JS_EVAL and st.session_state.get('_ls_payload') is not None:
    # Unique nonce per action, so a stale component value can never clear a
    # newer payload prematurely.
    _ls_ok = "OK" + st.session_state.get('_ls_nonce', '')
    if st.session_state['_ls_payload'] == "":
        _ls_expr = (f"(localStorage.removeItem({json.dumps(LS_DEFAULT_KEY)}), "
                    f"{json.dumps(_ls_ok)})")
    else:
        _ls_expr = (f"(localStorage.setItem({json.dumps(LS_DEFAULT_KEY)}, "
                    f"{json.dumps(st.session_state['_ls_payload'])}), {json.dumps(_ls_ok)})")
    if streamlit_js_eval(js_expressions=_ls_expr, key="_ls_set") == _ls_ok:
        st.session_state.pop('_ls_payload', None)
