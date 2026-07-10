import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import requests
import re
import threading
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    from streamlit_searchbox import st_searchbox
    HAS_SEARCHBOX = True
except ImportError:            # optional dependency: matrix still accepts typed tickers
    HAS_SEARCHBOX = False

from backtest_core import (
    STRAT_BH, STRAT_ANNUAL, STRAT_SEMI,
    STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_RD_FULL, STRAT_ASYM,
    STRAT_LEGACY_MAP,
    clean_ticker, parse_portfolio, calculate_metrics,
    run_detailed_backtest, compute_annual_returns,
    scrub_leading_glitches, scrub_isolated_spikes, sample_monthly,
    _split_top_level,
)

# --- Version ---
APP_VERSION = "2.3.5"  # semver: major.minor.patch
APP_BUILD_DATE = "2026-07-10"

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

# --- Header ---
st.markdown(f"""
<div class="header-bar">
    <div>
        <h2>Portfolio Backtest & Rebalance Analyzer</h2>
        <div class="subtitle">Multi-portfolio backtesting with rebalancing strategies</div>
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

def _apply_config_state(loaded_config):
    """Normalize a config dict into session state (no rerun)."""
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
    st.session_state.run_backtest = False
    # Discard allocation-editor scaffolding: stale in-flight edits must not be
    # flushed over a freshly loaded config on the next matrix rebuild.
    _old_key = st.session_state.pop('_alloc_key', None)
    if _old_key:
        st.session_state.pop(_old_key, None)
    st.session_state.pop('_alloc_base', None)
    st.session_state.pop('_alloc_pending', None)
    st.session_state.pop('_alloc_pending_seen', None)
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
            "thr": 40
        },
        {
            # AV-US with the crypto sleeve as a plain ETH-USD slot (5%), rebalanced
            # with Asymmetric RelDiff at a 38% trigger band. Weights identical to AV-US.
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
@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def fetch_price_history(tickers, start):
    """Cached wrapper around yf.download so widget interactions don't re-hit Yahoo.
    max_entries bounds memory on small cloud containers: every distinct ticker
    set caches its own download for the TTL. Raises on an empty result so a
    transient Yahoo failure is NOT cached for the full TTL (exceptions bypass
    st.cache_data); the caller already handles download exceptions."""
    df = yf.download(list(tickers), start=start, progress=False)
    if df is None or df.empty:
        raise RuntimeError("Yahoo returned no data (transient failure or rate limit)")
    return df

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

    fields = [("Total Return", "total_ret", "_total_ret", True),
              ("Ann. Return", "ann_ret", "_ann_ret", True),
              ("Max Drawdown", "max_dd", "_max_dd", False),
              ("Sharpe Ratio", "sharpe", "_sharpe", True),
              ("Rebalances", "rebal_cnt", None, None)]

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

ROW_SPEC = [0.28, 2.2, 2.6, 1.0, 0.5]

st.markdown('<div class="sec-label">Portfolios</div>', unsafe_allow_html=True)
with st.container(border=True):
    hdr = st.columns(ROW_SPEC, vertical_alignment="center")
    for c, lbl in zip(hdr, ["", "Name", "Strategy", "Band %", ""]):
        with c:
            if lbl: st.markdown(f'<div class="col-cap">{lbl}</div>', unsafe_allow_html=True)

    total_portfolios = len(st.session_state.portfolios_list)
    for i, port in enumerate(st.session_state.portfolios_list):
        if 'id' not in port: port['id'] = str(uuid.uuid4())

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
            "Band %", 1, 200, port['thr'], key=f"tr_{port['id']}", label_visibility="collapsed",
            help="Rebalance trigger band: relative deviation vs target weight, in %. "
                 "E.g. 40 → a 10% slot triggers beyond 6%–14%.")
        with cols[4]:
            if total_portfolios > 1:
                st.button(":material/delete:", key=f"del_{port['id']}",
                          on_click=delete_portfolio, args=(i,), help="Remove this portfolio")

    # --- Actions: add portfolio / persist current setup as startup default ---
    act_cols = st.columns([1.4, 2.2, 4.4])
    with act_cols[0]:
        if st.button(":material/add_circle: Add", width="stretch",
                     help="Add a portfolio (copies the last one's allocation)."):
            flush_alloc_edits()  # copy must see edits delivered in this same event
            ports_now = st.session_state.portfolios_list
            last_port = ports_now[-1] if ports_now else {"tickers": "", "weights": ""}
            ports_now.append({
                "id": str(uuid.uuid4()),
                "name": next_port_name(ports_now),
                "tickers": last_port["tickers"], "weights": last_port["weights"],
                "strat": STRAT_RD_MIXED, "thr": 40
            })
            st.rerun()
    with act_cols[1]:
        if st.button(":material/bookmark_add: Save Default", width="stretch",
                     help="Save the current setup (portfolios, benchmark, start date, initial "
                          "funds) as the startup default — new sessions open with it. Delete "
                          "Backtest/_default.json to restore the built-in config."):
            try:
                flush_alloc_edits()  # save must include edits delivered in this same event
                _errs = validate_inputs(st.session_state.portfolios_list, st.session_state['bi'])
                if _errs:
                    st.error("Default NOT saved — fix first: " + " · ".join(_errs))
                else:
                    SAVED_CONFIG_DIR.mkdir(exist_ok=True)
                    DEFAULT_CONFIG_PATH.write_text(json.dumps({
                        "benchmark": st.session_state['bi'],
                        "start_date": str(st.session_state['sd']),
                        "initial_funds": st.session_state['init_funds'],
                        "portfolios": st.session_state.portfolios_list,
                    }, indent=2, ensure_ascii=False), encoding="utf-8")
                    st.toast("Saved — new sessions now open with this setup", icon="✅")
            except Exception as e:
                st.error(f"Save default failed: {e}")

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

    btn_cols = st.columns([2, 6])
    with btn_cols[0]:
        run_clicked = st.button(":material/play_arrow: Analyze", type="primary", width="stretch")
        if run_clicked:
            error_msgs = validate_inputs(st.session_state.portfolios_list, bench_in)
            if not error_msgs:
                st.session_state.run_backtest = True
            else:
                st.session_state.run_backtest = False
                for msg in error_msgs: st.error(msg)

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
        # Guard idxmax: empty (no portfolio tickers) or all-None (every ticker
        # dataless) raises / is deprecated in pandas. Downstream per-portfolio
        # "no usable data" errors handle the degenerate cases.
        if first_valid_idx.notna().any():
            bottleneck_date = first_valid_idx.max()
            bottleneck_ticker = first_valid_idx.dropna().idxmax()
        else:
            bottleneck_date, bottleneck_ticker = pd.NaT, None
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
                if w_series.sum() <= 0:
                    # e.g. every nonzero-weight slot was dataless: 0/0 -> NaN
                    # weights would silently produce an empty backtest.
                    st.error(f"**{p['name']}**: surviving tickers carry no positive weight "
                             f"after dropping dataless ones — portfolio skipped.")
                    continue
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
        metrics.append(bench_m)
        for p_name, cnt in valid_ports_meta.items():
            m = calculate_metrics(comp_df[p_name], cnt, risk_free_rate=rf_rate)
            m["name"] = p_name
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
            def style_row(row):
                # Explicit dark ink on the pastel fills so rows stay readable in dark theme
                if row['Type'] == 'PnL Contrib%': return ['background-color: #fce4ec; color: #d81b60; font-weight: bold'] * len(row)
                if row['Type'] == 'Init': return ['background-color: #e3f2fd; color: #1a1a2e; font-weight: bold'] * len(row)
                if row['Type'] == 'Pre-Rebal': return ['background-color: #fff3e0; color: #1a1a2e'] * len(row)
                if row['Type'] == 'Post-Rebal': return ['background-color: #e8f5e9; color: #1a1a2e'] * len(row)
                return [''] * len(row)
            for tab, lbl in zip(tabs, tab_names):
                with tab:
                    st.caption(f"Start: {actual_start_day.date()}")
                    st.dataframe(res_list[lbl].style.apply(style_row, axis=1).format({"NAV": "{:,.2f}"}), width="stretch")

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
