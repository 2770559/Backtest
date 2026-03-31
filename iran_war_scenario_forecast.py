import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="伊朗战争情景预测对比", layout="wide", page_icon="⚔️")

# --- Custom CSS ---
st.markdown("""
<style>
section.main > div { max-width: 1400px; margin: 0 auto; }
.header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white; padding: 1.25rem 1.75rem; border-radius: 0.75rem;
    margin-bottom: 1.25rem;
}
.header-bar h2 { margin: 0; font-size: 1.4rem; font-weight: 700; }
.header-bar .subtitle { opacity: 0.7; font-size: 0.85rem; margin-top: 0.3rem; }
.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; margin-bottom: 1rem; }
@media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(3, 1fr); } }
.kpi-card {
    background: #ffffff; border-radius: 0.6rem; padding: 0.875rem 1rem;
    border-left: 4px solid #e0e0e0; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.kpi-card .label { font-size: 0.72rem; color: #666; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 0.25rem; font-weight: 600; }
.kpi-card .value { font-size: 1.2rem; font-weight: 700; color: #1a1a2e; }
.kpi-card.green  { border-left-color: #10b981; }
.kpi-card.red    { border-left-color: #ef4444; }
.kpi-card.blue   { border-left-color: #3b82f6; }
.kpi-card.purple { border-left-color: #8b5cf6; }
.kpi-card.amber  { border-left-color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA: All scenario prediction data extracted from
# iran_war_scenario_analysis.md (2026-03-22)
# ============================================================

ASSETS = ["QQQM", "BRK-B", "GLDM", "XLE", "DBMF", "KMLM", "BTC"]
WEIGHTS = {"QQQM": 0.35, "BRK-B": 0.15, "GLDM": 0.15, "XLE": 0.10,
           "DBMF": 0.10, "KMLM": 0.10, "BTC": 0.05}

SCENARIO_META = {
    "S1": {"name": "S1: 短期降级", "prob": "22%", "desc": "4-8周内停火，海峡4-6月恢复，Brent回落$80-90"},
    "S2": {"name": "S2: 夺岛→谈判", "prob": "17%", "desc": "海军陆战队夺哈尔格岛→经济施压→谈判→框架协议"},
    "S3": {"name": "S3: 持久低烈度（最可能）", "prob": "38%", "desc": "无决定性胜利/停火，滞胀→Fed转向→缓慢正常化"},
    "S4": {"name": "S4: 升级+地区蔓延", "prob": "18%", "desc": "多战场升级→流动性危机→央行紧急干预→衰退→恢复"},
    "S5": {"name": "S5: 政权更迭", "prob": "5%", "desc": "军事+经济压力→IRGC分裂→新政权→制裁解除→石油重返"},
}

def _calc_portfolio(row):
    """Calculate weighted portfolio return from asset returns."""
    return sum(row.get(a, 0) * WEIGHTS[a] for a in ASSETS)

def _build_scenario_data():
    """Build all scenario data as list of dicts with month, asset returns, and portfolio."""
    scenarios = {}

    # ---- S1: 短期降级 ----
    s1 = [
        {"month": 0,    "label": "开战基准(2/28)",   "QQQM": 0,    "BRK-B": 0,   "GLDM": 0,    "XLE": 0,     "DBMF": 0,   "KMLM": 0,   "BTC": 0},
        {"month": 0.8,  "label": "当前(3/22)",       "QQQM": -4.4, "BRK-B": -3,  "GLDM": -14,  "XLE": 21.6,  "DBMF": 3,   "KMLM": 2,   "BTC": 11},
        {"month": 1.2,  "label": "最大压力(4月初)",   "QQQM": -7,   "BRK-B": -4,  "GLDM": -18,  "XLE": 28,    "DBMF": 5,   "KMLM": 4,   "BTC": 5},
        {"month": 2.5,  "label": "停火信号(4-5月)",   "QQQM": -2,   "BRK-B": -1,  "GLDM": -16,  "XLE": 12,    "DBMF": 2,   "KMLM": 1,   "BTC": 15},
        {"month": 5,    "label": "海峡恢复50%(6-8月)","QQQM": 6,    "BRK-B": 4,   "GLDM": -10,  "XLE": 2,     "DBMF": -1,  "KMLM": -1,  "BTC": 20},
        {"month": 12,   "label": "正常化(12个月)",    "QQQM": 12,   "BRK-B": 8,   "GLDM": -5,   "XLE": -5,    "DBMF": 0,   "KMLM": -1,  "BTC": 25},
    ]
    for r in s1: r["组合"] = round(_calc_portfolio(r), 2)
    scenarios["S1"] = s1

    # ---- S2: 夺岛→谈判 ----
    s2 = [
        {"month": 0,    "label": "开战基准(2/28)",     "QQQM": 0,    "BRK-B": 0,   "GLDM": 0,    "XLE": 0,     "DBMF": 0,   "KMLM": 0,   "BTC": 0},
        {"month": 0.8,  "label": "当前(3/22)",         "QQQM": -4.4, "BRK-B": -3,  "GLDM": -14,  "XLE": 21.6,  "DBMF": 3,   "KMLM": 2,   "BTC": 11},
        {"month": 1.2,  "label": "空袭强化(4月初)",     "QQQM": -8,   "BRK-B": -5,  "GLDM": -16,  "XLE": 32,    "DBMF": 7,   "KMLM": 5,   "BTC": 3},
        {"month": 1.5,  "label": "夺岛D-Day(4月中)",   "QQQM": -14,  "BRK-B": -8,  "GLDM": -12,  "XLE": 45,    "DBMF": 12,  "KMLM": 9,   "BTC": -5},
        {"month": 2,    "label": "美军伤亡报道(4月下旬)","QQQM": -18,  "BRK-B": -10, "GLDM": -8,   "XLE": 48,    "DBMF": 15,  "KMLM": 11,  "BTC": -12},
        {"month": 3.5,  "label": "谈判启动(5-6月)",     "QQQM": -10,  "BRK-B": -6,  "GLDM": -10,  "XLE": 30,    "DBMF": 10,  "KMLM": 7,   "BTC": 5},
        {"month": 5.5,  "label": "框架协议(7-8月)",     "QQQM": 2,    "BRK-B": 2,   "GLDM": -12,  "XLE": 8,     "DBMF": 3,   "KMLM": 2,   "BTC": 18},
        {"month": 9,    "label": "海峡恢复(10-12月)",   "QQQM": 10,   "BRK-B": 6,   "GLDM": -8,   "XLE": -2,    "DBMF": 0,   "KMLM": -1,  "BTC": 22},
        {"month": 12,   "label": "12个月终态",          "QQQM": 13,   "BRK-B": 7,   "GLDM": -6,   "XLE": -4,    "DBMF": -1,  "KMLM": -2,  "BTC": 25},
    ]
    for r in s2: r["组合"] = round(_calc_portfolio(r), 2)
    scenarios["S2"] = s2

    # ---- S3: 持久低烈度（最可能） ----
    s3 = [
        {"month": 0,    "label": "开战基准(2/28)",          "QQQM": 0,    "BRK-B": 0,   "GLDM": 0,    "XLE": 0,     "DBMF": 0,   "KMLM": 0,   "BTC": 0},
        {"month": 0.8,  "label": "当前(3/22)",              "QQQM": -4.4, "BRK-B": -3,  "GLDM": -14,  "XLE": 21.6,  "DBMF": 3,   "KMLM": 2,   "BTC": 11},
        {"month": 2.5,  "label": "油价维持高位(4-5月)",      "QQQM": -8,   "BRK-B": -5,  "GLDM": -17,  "XLE": 28,    "DBMF": 6,   "KMLM": 5,   "BTC": 5},
        {"month": 3.5,  "label": "通胀数据恶化(6月)",        "QQQM": -12,  "BRK-B": -6,  "GLDM": -20,  "XLE": 25,    "DBMF": 8,   "KMLM": 7,   "BTC": 0},
        {"month": 5.5,  "label": "GDP放缓<1%(7-9月)",       "QQQM": -16,  "BRK-B": -8,  "GLDM": -18,  "XLE": 15,    "DBMF": 12,  "KMLM": 10,  "BTC": -5},
        {"month": 7,    "label": "就业恶化/开始降息(9-10月)", "QQQM": -20,  "BRK-B": -10, "GLDM": -15,  "XLE": 10,    "DBMF": 15,  "KMLM": 13,  "BTC": -10},
        {"month": 8.5,  "label": "首次降息50bp(10-11月)",    "QQQM": -15,  "BRK-B": -7,  "GLDM": -5,   "XLE": 8,     "DBMF": 13,  "KMLM": 11,  "BTC": 5},
        {"month": 11,   "label": "连续降息+QE暗示(12-3月)",  "QQQM": -8,   "BRK-B": -3,  "GLDM": 8,    "XLE": 2,     "DBMF": 10,  "KMLM": 8,   "BTC": 15},
        {"month": 18,   "label": "冲突降温(18个月)",         "QQQM": 2,    "BRK-B": 5,   "GLDM": 18,   "XLE": -3,    "DBMF": 6,   "KMLM": 4,   "BTC": 22},
        {"month": 24,   "label": "24个月终态",               "QQQM": 8,    "BRK-B": 7,   "GLDM": 25,   "XLE": -8,    "DBMF": 4,   "KMLM": 2,   "BTC": 28},
    ]
    for r in s3: r["组合"] = round(_calc_portfolio(r), 2)
    scenarios["S3"] = s3

    # ---- S4: 升级+地区蔓延 ----
    s4 = [
        {"month": 0,    "label": "开战基准(2/28)",          "QQQM": 0,    "BRK-B": 0,    "GLDM": 0,    "XLE": 0,     "DBMF": 0,   "KMLM": 0,    "BTC": 0},
        {"month": 0.8,  "label": "当前(3/22)",              "QQQM": -4.4, "BRK-B": -3,   "GLDM": -14,  "XLE": 21.6,  "DBMF": 3,   "KMLM": 2,    "BTC": 11},
        {"month": 0.9,  "label": "伊朗电厂被炸(3/24-25)",   "QQQM": -10,  "BRK-B": -6,   "GLDM": -16,  "XLE": 35,    "DBMF": 8,   "KMLM": 6,    "BTC": 3},
        {"month": 1.2,  "label": "伊朗攻击沙特Abqaiq(4月初)","QQQM": -18,  "BRK-B": -10,  "GLDM": -22,  "XLE": 55,    "DBMF": 14,  "KMLM": 11,   "BTC": -8},
        {"month": 1.5,  "label": "全球Margin Call(4月中)",   "QQQM": -28,  "BRK-B": -15,  "GLDM": -25,  "XLE": 50,    "DBMF": 18,  "KMLM": 14,   "BTC": -25},
        {"month": 2,    "label": "红海封锁(4月下旬)",        "QQQM": -32,  "BRK-B": -18,  "GLDM": -20,  "XLE": 45,    "DBMF": 22,  "KMLM": 18,   "BTC": -20},
        {"month": 2.5,  "label": "Fed紧急降息+QE(5月)",      "QQQM": -25,  "BRK-B": -13,  "GLDM": -8,   "XLE": 35,    "DBMF": 20,  "KMLM": 16,   "BTC": -10},
        {"month": 3.5,  "label": "G7联合干预(6月)",          "QQQM": -18,  "BRK-B": -8,   "GLDM": 5,    "XLE": 22,    "DBMF": 16,  "KMLM": 13,   "BTC": 5},
        {"month": 5.5,  "label": "全球GDP负增长(7-9月)",     "QQQM": -22,  "BRK-B": -10,  "GLDM": 15,   "XLE": 5,     "DBMF": 12,  "KMLM": 9,    "BTC": 10},
        {"month": 9,    "label": "油价回落/衰退加深(10-12月)","QQQM": -15,  "BRK-B": -5,   "GLDM": 25,   "XLE": -5,    "DBMF": 8,   "KMLM": 5,    "BTC": 18},
        {"month": 18,   "label": "冲突逐步降温(18个月)",     "QQQM": -5,   "BRK-B": 2,    "GLDM": 32,   "XLE": -10,   "DBMF": 4,   "KMLM": 2,    "BTC": 25},
        {"month": 24,   "label": "24个月终态",               "QQQM": 5,    "BRK-B": 6,    "GLDM": 35,   "XLE": -12,   "DBMF": 2,   "KMLM": 0,    "BTC": 30},
    ]
    for r in s4: r["组合"] = round(_calc_portfolio(r), 2)
    scenarios["S4"] = s4

    # ---- S5: 政权更迭 ----
    s5 = [
        {"month": 0,    "label": "开战基准(2/28)",          "QQQM": 0,    "BRK-B": 0,   "GLDM": 0,    "XLE": 0,     "DBMF": 0,   "KMLM": 0,   "BTC": 0},
        {"month": 0.8,  "label": "当前(3/22)",              "QQQM": -4.4, "BRK-B": -3,  "GLDM": -14,  "XLE": 21.6,  "DBMF": 3,   "KMLM": 2,   "BTC": 11},
        {"month": 3.5,  "label": "滞胀深化(4-6月)",         "QQQM": -12,  "BRK-B": -6,  "GLDM": -20,  "XLE": 20,    "DBMF": 8,   "KMLM": 7,   "BTC": 0},
        {"month": 5.5,  "label": "经济放缓(7-9月)",         "QQQM": -18,  "BRK-B": -9,  "GLDM": -18,  "XLE": 10,    "DBMF": 12,  "KMLM": 10,  "BTC": -8},
        {"month": 8,    "label": "IRGC分裂/起义前(10-12月)", "QQQM": -10,  "BRK-B": -4,  "GLDM": -15,  "XLE": 5,     "DBMF": 10,  "KMLM": 8,   "BTC": 5},
        {"month": 8.5,  "label": "政权更迭事件(10-12月)",    "QQQM": 5,    "BRK-B": 3,   "GLDM": -18,  "XLE": -15,   "DBMF": 2,   "KMLM": 1,   "BTC": 20},
        {"month": 11,   "label": "新政权确认(12-15月)",      "QQQM": 15,   "BRK-B": 8,   "GLDM": -22,  "XLE": -25,   "DBMF": -3,  "KMLM": -4,  "BTC": 30},
        {"month": 18,   "label": "制裁逐步解除(18个月)",     "QQQM": 20,   "BRK-B": 10,  "GLDM": -15,  "XLE": -20,   "DBMF": -2,  "KMLM": -3,  "BTC": 35},
        {"month": 24,   "label": "24个月终态",               "QQQM": 22,   "BRK-B": 12,  "GLDM": -12,  "XLE": -22,   "DBMF": -1,  "KMLM": -2,  "BTC": 38},
    ]
    for r in s5: r["组合"] = round(_calc_portfolio(r), 2)
    scenarios["S5"] = s5

    return scenarios

SCENARIOS = _build_scenario_data()
ALL_SERIES = ASSETS + ["组合"]

# War start date: 2/28/2026
WAR_START = datetime(2026, 2, 28)
# Prediction baseline: the last trading day when predictions were made (fixed, never moves)
PREDICTION_BASELINE = datetime(2026, 3, 20)

def month_to_date(m):
    """Convert month offset to real date string (YYYY-MM-DD).
    If m matches a known exact date (baseline or latest actual), return that date precisely."""
    if BASELINE_MONTH is not None and abs(m - BASELINE_MONTH) < 0.001:
        return PREDICTION_BASELINE.strftime("%Y-%m-%d")
    # Match latest actual trading day exactly
    if ACTUAL_DATA is not None and not ACTUAL_DATA.empty:
        last_days = (ACTUAL_DATA.index[-1] - pd.Timestamp(WAR_START)).days
        last_month = round(last_days / 30.44, 4)
        if abs(m - last_month) < 0.001:
            return ACTUAL_DATA.index[-1].strftime("%Y-%m-%d")
    return (WAR_START + timedelta(days=round(m * 30.44))).strftime("%Y-%m-%d")

# ============================================================
# ACTUAL DATA: Fetch real prices from yfinance
# ============================================================
TICKER_MAP = {
    "QQQM": "QQQM", "BRK-B": "BRK-B", "GLDM": "GLDM",
    "XLE": "XLE", "DBMF": "DBMF", "KMLM": "KMLM", "BTC": "BTC",
}
ACTUAL_LABEL = "实际数据"
ACTUAL_COLOR = "#000000"  # black

@st.cache_data(ttl=3600)
def fetch_actual_data():
    """Fetch actual daily prices from yfinance and compute cumulative returns (%) from war-start baseline."""
    baseline_start = WAR_START - timedelta(days=5)  # fetch a few days before to get 2/27 close
    tickers = list(TICKER_MAP.values())
    raw = None
    for attempt in range(3):
        try:
            raw = yf.download(tickers, start=baseline_start.strftime("%Y-%m-%d"),
                              end=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                              auto_adjust=True, progress=False)
            if raw is not None and not raw.empty:
                break
        except Exception:
            import time
            time.sleep(2 * (attempt + 1))
    try:
        if raw is None or raw.empty:
            return None
        # Handle yfinance column structure (varies by version)
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0).unique().tolist()
            if "Close" in level0:
                close = raw["Close"]
            elif "Price" in level0:
                close = raw["Price"]
            else:
                close = raw.xs("Close", level=1, axis=1) if "Close" in raw.columns.get_level_values(1).unique() else raw
        else:
            close = raw
        # Rename columns back to our asset names
        rename = {v: k for k, v in TICKER_MAP.items()}
        close = close.rename(columns=rename)
        # Find baseline: last trading day on or before 2/27
        baseline_mask = close.index <= pd.Timestamp(WAR_START - timedelta(days=1))
        if not baseline_mask.any():
            baseline_prices = close.iloc[0]
        else:
            baseline_prices = close.loc[baseline_mask].iloc[-1]
        # Keep US trading days only (all tickers are now ARCA-listed)
        close = close.dropna(how="all")
        close = close.ffill()
        # Cumulative return (%) from baseline
        returns_pct = ((close / baseline_prices) - 1) * 100
        # Filter to war period only (>= 2/28)
        returns_pct = returns_pct[returns_pct.index >= pd.Timestamp(WAR_START)]
        returns_pct = returns_pct.ffill()
        available = [a for a in ASSETS if a in returns_pct.columns]
        returns_pct["组合"] = sum(returns_pct[a] * WEIGHTS[a] for a in available)
        returns_pct = returns_pct.dropna(subset=["组合"])
        return returns_pct
    except Exception as e:
        st.warning(f"获取实际数据失败: {e}")
        return None

ACTUAL_DATA = fetch_actual_data()

def build_actual_long_df(series_list, color_field="情景"):
    """Build long-format DataFrame for actual data, compatible with prediction data."""
    if ACTUAL_DATA is None or ACTUAL_DATA.empty:
        return pd.DataFrame()
    rows = []
    available = [s for s in series_list if s in ACTUAL_DATA.columns]
    for date, row in ACTUAL_DATA.iterrows():
        for s in available:
            if pd.notna(row[s]):
                entry = {"日期": date, "节点": "实际", "资产": s, "收益率(%)": round(row[s], 2)}
                if color_field == "情景":
                    entry["情景"] = ACTUAL_LABEL
                    entry["scenario_key"] = "actual"
                else:
                    entry["情景"] = ACTUAL_LABEL
                entry[color_field] = entry.get(color_field, ACTUAL_LABEL)
                rows.append(entry)
    return pd.DataFrame(rows)

def get_actual_wide_at_dates(dates, series_list):
    """Get actual data values at specific dates for tooltip. Future dates return None."""
    if ACTUAL_DATA is None or ACTUAL_DATA.empty:
        return {}
    available = [s for s in series_list if s in ACTUAL_DATA.columns]
    last_actual_date = ACTUAL_DATA.index[-1]
    result = {}
    for d in dates:
        ts = pd.Timestamp(d)
        if ts > last_actual_date:
            # Future date: no actual data
            result[d] = {s: None for s in available}
        else:
            idx = ACTUAL_DATA.index.get_indexer([ts], method="nearest")[0]
            if 0 <= idx < len(ACTUAL_DATA):
                row = ACTUAL_DATA.iloc[idx]
                result[d] = {s: round(row[s], 2) if pd.notna(row.get(s)) else None for s in available}
    return result

# Color palette
COLORS = {
    "QQQM":  "#3b82f6",  # blue
    "BRK-B": "#6366f1",  # indigo
    "GLDM":  "#f59e0b",  # amber/gold
    "XLE":   "#10b981",  # green
    "DBMF":  "#8b5cf6",  # purple
    "KMLM":  "#ec4899",  # pink
    "BTC":   "#f97316",  # orange
    "组合":   "#ef4444",  # red (portfolio)
}

SCENARIO_COLORS = {
    "S1": "#10b981",  # green
    "S2": "#3b82f6",  # blue
    "S3": "#f59e0b",  # amber
    "S4": "#ef4444",  # red
    "S5": "#8b5cf6",  # purple
}

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-bar">
    <div>
        <h2>⚔️ 2026美伊战争 — AV-US组合情景预测对比系统</h2>
        <div class="subtitle">基于 iran_war_scenario_analysis.md (2026-03-22) · 五情景×七资产全过程推演 · 实际数据自动更新(yfinance)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- KPI Cards ----
kpi_html = '<div class="kpi-row">'
for sid, meta in SCENARIO_META.items():
    color_cls = {"S1": "green", "S2": "blue", "S3": "amber", "S4": "red", "S5": "purple"}[sid]
    kpi_html += f'''<div class="kpi-card {color_cls}">
        <div class="label">{meta["name"]}</div>
        <div class="value">概率 {meta["prob"]}</div>
    </div>'''
kpi_html += '</div>'
st.markdown(kpi_html, unsafe_allow_html=True)

# ---- Actual data status ----
if ACTUAL_DATA is not None and not ACTUAL_DATA.empty:
    last_date = ACTUAL_DATA.index[-1].strftime("%Y-%m-%d")
    last_portfolio = ACTUAL_DATA["组合"].dropna().iloc[-1]
    col_status, col_refresh = st.columns([0.85, 0.15])
    with col_status:
        st.caption(f"📡 实际数据最后交易日: **{last_date}** | 组合实际累计收益: **{last_portfolio:+.2f}%** | 数据缓存5分钟自动刷新")
    with col_refresh:
        if st.button("🔄 刷新数据"):
            st.cache_data.clear()
            st.rerun()
else:
    st.warning("⚠️ 未能获取实际市场数据，图表仅显示预测数据")

# ============================================================
# HELPER: Build long-format DataFrame for Altair
# ============================================================
# Fixed baseline month offset (prediction baseline date, never moves)
BASELINE_MONTH = round((PREDICTION_BASELINE - WAR_START).days / 30.44, 4)

def _interpolate_scenario(data, all_months, series_list, scenario_name):
    """Interpolate a scenario's asset values at all unified month points."""
    existing_months = [p["month"] for p in data]
    existing_labels = {p["month"]: p["label"] for p in data}
    asset_vals = {s: {p["month"]: p[s] for p in data} for s in series_list}

    result = []
    for m in sorted(all_months):
        if m in existing_months:
            point = next(p for p in data if p["month"] == m)
            label = point["label"]
            vals = {s: point[s] for s in series_list}
        else:
            lower = [em for em in existing_months if em <= m]
            upper = [em for em in existing_months if em >= m]
            if not lower or not upper:
                continue
            m0, m1 = max(lower), min(upper)
            if m0 == m1:
                vals = {s: asset_vals[s][m0] for s in series_list}
            else:
                t = (m - m0) / (m1 - m0)
                vals = {s: round(asset_vals[s][m0] * (1 - t) + asset_vals[s][m1] * t, 2)
                        for s in series_list}
            if BASELINE_MONTH is not None and abs(m - BASELINE_MONTH) < 0.001:
                label = f"预测基准日({PREDICTION_BASELINE.strftime('%m/%d')})"
            else:
                label = f"(插值 ≈{m:.1f}月)"
        result.append({"month": m, "label": label, **vals})
    return result

def build_long_df(scenario_keys, series_list):
    """Build a long-format DataFrame with interpolated values at all time points."""
    all_months = set()
    for sk in scenario_keys:
        for point in SCENARIOS[sk]:
            all_months.add(point["month"])
    # Add fixed baseline so predictions are interpolated there
    if BASELINE_MONTH is not None:
        all_months.add(BASELINE_MONTH)
    # Add latest actual trading day so crosshair tooltip works on it
    if ACTUAL_DATA is not None and not ACTUAL_DATA.empty:
        last_actual_days = (ACTUAL_DATA.index[-1] - pd.Timestamp(WAR_START)).days
        last_actual_month = round(last_actual_days / 30.44, 4)
        all_months.add(last_actual_month)
    all_months = sorted(all_months)

    rows = []
    for sk in scenario_keys:
        meta = SCENARIO_META[sk]
        interpolated = _interpolate_scenario(SCENARIOS[sk], all_months, series_list, sk)
        for point in interpolated:
            date_str = month_to_date(point["month"])
            for s in series_list:
                rows.append({
                    "情景": meta["name"],
                    "scenario_key": sk,
                    "日期": date_str,
                    "月份": point["month"],
                    "节点": point["label"],
                    "资产": s,
                    "收益率(%)": point[s],
                })
    df = pd.DataFrame(rows)
    df["日期"] = pd.to_datetime(df["日期"])

    # Rebase predictions: use fixed PREDICTION_BASELINE as the anchor point
    baseline_ts = pd.Timestamp(PREDICTION_BASELINE)
    if ACTUAL_DATA is not None and not ACTUAL_DATA.empty:
        # Get actual values at the fixed baseline date (3/20)
        baseline_idx = ACTUAL_DATA.index.get_indexer([baseline_ts], method="nearest")[0]
        baseline_actual_vals = ACTUAL_DATA.iloc[baseline_idx] if 0 <= baseline_idx < len(ACTUAL_DATA) else None

        if baseline_actual_vals is not None:
            # Step 1: Calculate offset at FIXED baseline for each (scenario, asset)
            offsets = {}
            for sk in scenario_keys:
                for s in series_list:
                    mask_transition = (
                        (df["scenario_key"] == sk) & (df["资产"] == s) & (df["日期"] == baseline_ts)
                    )
                    pred_rows = df[mask_transition]
                    if not pred_rows.empty and s in ACTUAL_DATA.columns:
                        predicted_val = pred_rows.iloc[0]["收益率(%)"]
                        actual_val = baseline_actual_vals.get(s)
                        if pd.notna(actual_val) and pd.notna(predicted_val):
                            offsets[(sk, s)] = round(actual_val - predicted_val, 4)

            # Step 2: Apply offset to all dates AFTER baseline (rebase from actual at 3/20)
            future_mask = df["日期"] > baseline_ts
            for idx in df[future_mask].index:
                sk = df.at[idx, "scenario_key"]
                asset = df.at[idx, "资产"]
                key = (sk, asset)
                if key in offsets:
                    df.at[idx, "收益率(%)"] = round(df.at[idx, "收益率(%)"] + offsets[key], 2)

            # Step 3: Replace dates ON or BEFORE baseline with actual data
            past_mask = df["日期"] <= baseline_ts
            for idx in df[past_mask].index:
                asset = df.at[idx, "资产"]
                ts = df.at[idx, "日期"]
                if asset in ACTUAL_DATA.columns:
                    ai = ACTUAL_DATA.index.get_indexer([ts], method="nearest")[0]
                    if 0 <= ai < len(ACTUAL_DATA):
                        val = ACTUAL_DATA.iloc[ai][asset]
                        if pd.notna(val):
                            df.at[idx, "收益率(%)"] = round(val, 2)
    return df

def build_chart(df, color_field, color_scale, title, height=500, zero_line=True,
                actual_series=None):
    """Build an Altair layered line+point chart with crosshair tooltip and actual data overlay.

    actual_series: list of series names to draw actual data for (e.g. ["组合"] or ASSETS+["组合"]).
                   If None, no actual data is drawn.
    """

    # --- Determine the pivot column (the one used for color) and value columns ---
    pivot_col = color_field  # "资产" or "情景"
    pivot_values = list(df[pivot_col].unique())

    # Build wide-format for crosshair tooltip (one column per series)
    wide_df = df.pivot_table(index="日期", columns=pivot_col, values="收益率(%)", aggfunc="first").reset_index()
    wide_df = wide_df.sort_values("日期")

    # --- Add actual data columns to tooltip wide_df ---
    has_actual = (actual_series is not None and ACTUAL_DATA is not None and not ACTUAL_DATA.empty)
    actual_tooltip_cols = []
    if has_actual:
        actual_at_dates = get_actual_wide_at_dates(wide_df["日期"].tolist(), actual_series)
        for s in actual_series:
            col_name = f"实际:{s}"
            actual_tooltip_cols.append(col_name)
            wide_df[col_name] = wide_df["日期"].map(
                lambda d, _s=s: actual_at_dates.get(d, {}).get(_s))

    # --- Nearest selection for crosshair ---
    nearest = alt.selection_point(nearest=True, on="mouseover", fields=["日期"], empty=False)

    # --- Line chart (long format, predictions) ---
    line = alt.Chart(df).mark_line(strokeWidth=2.5, opacity=0.85).encode(
        x=alt.X("日期:T", title="日期", axis=alt.Axis(format="%Y-%m", labelAngle=-45)),
        y=alt.Y("收益率(%):Q", title="累计收益率 (%)"),
        color=alt.Color(f"{pivot_col}:N", scale=color_scale, title=pivot_col),
    )

    # --- Tooltip: show all prediction series + actual values ---
    tooltips = [alt.Tooltip("日期:T", format="%Y-%m-%d", title="日期")]
    for col in pivot_values:
        tooltips.append(alt.Tooltip(field=col, type="quantitative", format=".1f", title=f"{col} (%)"))
    for col in actual_tooltip_cols:
        tooltips.append(alt.Tooltip(field=col, type="quantitative", format=".1f", title=f"{col} (%)"))

    # Invisible thick rule for mouse capture
    selectors = alt.Chart(wide_df).mark_rule(opacity=0.001, strokeWidth=40).encode(
        x="日期:T", tooltip=tooltips
    ).add_params(nearest)

    # Visible vertical rule at hovered date
    rules = alt.Chart(wide_df).mark_rule(color="#94a3b8", strokeDash=[3, 3]).encode(
        x="日期:T", tooltip=tooltips
    ).transform_filter(nearest)

    # Highlight points at hovered date (predictions only)
    points = alt.Chart(df).mark_point(size=60, filled=True).encode(
        x="日期:T",
        y="收益率(%):Q",
        color=alt.Color(f"{pivot_col}:N", scale=color_scale, title=pivot_col),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
    )

    layers = [line, selectors, rules, points]

    # --- Actual data overlay lines ---
    if has_actual:
        for s in actual_series:
            if s not in ACTUAL_DATA.columns:
                continue
            adf = ACTUAL_DATA[[s]].dropna().reset_index()
            adf.columns = ["日期", "收益率(%)"]
            adf["系列"] = f"实际:{s}"

            # Actual data always uses black solid line for clear distinction
            line_color = ACTUAL_COLOR
            dash = [0]
            width = 3

            actual_line = alt.Chart(adf).mark_line(
                strokeWidth=width, strokeDash=dash, opacity=0.9
            ).encode(
                x="日期:T",
                y="收益率(%):Q",
                color=alt.value(line_color),
            )
            layers.append(actual_line)

    if zero_line:
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            strokeDash=[4, 4], color="#888", strokeWidth=1
        ).encode(y="y:Q")
        layers.insert(0, zero)

    chart = alt.layer(*layers).properties(
        title=alt.Title(title, fontSize=16, anchor="start"),
        height=height,
    ).configure_legend(
        orient="bottom", columns=4, titleFontSize=12, labelFontSize=11
    ).configure_view(strokeWidth=0)

    return chart

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 全景总览（所有情景×所有资产）",
    "🎯 单情景查看（选择S1-S5）",
    "📈 单资产跨情景对比"
])

# ---- TAB 1: 全景总览 ----
with tab1:
    st.markdown("### 全部五种情景下 AV-US 组合走势对比")
    st.caption("虚线为组合加权收益率，实线为各成分资产。基准：2/28开战前=0%")

    col_overview_1, col_overview_2 = st.columns([1, 1])

    with col_overview_1:
        st.markdown("#### 组合走势对比（五情景）")
        df_portfolio = build_long_df(list(SCENARIOS.keys()), ["组合"])
        scale_p = alt.Scale(domain=[m["name"] for m in SCENARIO_META.values()],
                           range=[SCENARIO_COLORS[k] for k in SCENARIO_META])
        chart_p = build_chart(df_portfolio, "情景", scale_p, "AV-US 组合：五情景走势 (黑线=实际)",
                              height=420, actual_series=["组合"])
        st.altair_chart(chart_p, use_container_width=True)

    with col_overview_2:
        st.markdown("#### 五情景终态对比")
        # Build terminal state comparison table
        terminal_rows = []
        for sk, meta in SCENARIO_META.items():
            last = SCENARIOS[sk][-1]
            terminal_rows.append({
                "情景": meta["name"],
                "概率": meta["prob"],
                "终态日期": month_to_date(last["month"]),
                **{a: f"{last[a]:+.1f}%" for a in ASSETS},
                "组合": f"{last['组合']:+.1f}%",
            })
        st.dataframe(pd.DataFrame(terminal_rows), use_container_width=True, hide_index=True)

        # Max drawdown comparison
        st.markdown("#### 组合最大回撤对比")
        dd_data = {
            "S1: 短期降级": {"最大回撤": "-2.8%", "时点": "4月初", "12个月回报": "+4.8%"},
            "S2: 夺岛谈判": {"最大回撤": "-4.2%", "时点": "4月中下", "12个月回报": "+5.5%"},
            "S3: 持久低烈度": {"最大回撤": "-6.8%", "时点": "9-10月", "12个月回报": "-1.0%"},
            "S4: 升级蔓延": {"最大回撤": "-9.2%", "时点": "4月中下", "12个月回报": "-1.0%"},
            "S5: 政权更迭": {"最大回撤": "-6.5%", "时点": "7-9月", "12个月回报": "+3.5%"},
            "概率加权": {"最大回撤": "-5.7%", "时点": "—", "12个月回报": "+1.5%"},
        }
        dd_df = pd.DataFrame(dd_data).T.reset_index().rename(columns={"index": "情景"})
        st.dataframe(dd_df, use_container_width=True, hide_index=True)

# ---- TAB 2: 单情景查看 ----
with tab2:
    st.markdown("### 选择情景，查看该情景下所有资产的全过程走势")

    selected_scenario = st.radio(
        "选择情景：",
        options=list(SCENARIO_META.keys()),
        format_func=lambda x: f"{SCENARIO_META[x]['name']} (概率{SCENARIO_META[x]['prob']})",
        horizontal=True,
        key="scenario_radio",
    )

    meta = SCENARIO_META[selected_scenario]
    st.info(f"**{meta['name']}** — {meta['desc']}")

    # Build chart for this single scenario, all assets + portfolio
    df_single = build_long_df([selected_scenario], ALL_SERIES)
    scale_a = alt.Scale(domain=ALL_SERIES, range=[COLORS[a] for a in ALL_SERIES])
    chart_single = build_chart(
        df_single, "资产", scale_a,
        f"{meta['name']} — 各资产全过程走势 (黑线=实际AV-US)",
        height=500, actual_series=["组合"],
    )
    st.altair_chart(chart_single, use_container_width=True)

    # Data table for this scenario
    with st.expander("📋 查看详细数据表"):
        table_data = SCENARIOS[selected_scenario]
        display_rows = []
        for p in table_data:
            row = {"节点": p["label"], "日期": month_to_date(p["month"])}
            for a in ALL_SERIES:
                row[a] = f"{p[a]:+.1f}%"
            display_rows.append(row)
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

# ---- TAB 3: 单资产跨情景 ----
with tab3:
    st.markdown("### 选择资产，查看该资产在五种情景下的走势对比")

    selected_asset = st.radio(
        "选择资产：",
        options=ALL_SERIES,
        format_func=lambda x: f"{x} ({WEIGHTS.get(x, '')})" if x in WEIGHTS else f"{x} (加权组合)",
        horizontal=True,
        key="asset_radio",
    )

    df_asset = build_long_df(list(SCENARIOS.keys()), [selected_asset])
    scale_s = alt.Scale(domain=[m["name"] for m in SCENARIO_META.values()],
                       range=[SCENARIO_COLORS[k] for k in SCENARIO_META])

    asset_weight_info = f"权重 {WEIGHTS[selected_asset]*100:.0f}%" if selected_asset in WEIGHTS else "加权组合"
    chart_asset = build_chart(
        df_asset, "情景", scale_s,
        f"{selected_asset} ({asset_weight_info}) — 五情景走势对比 (黑线=实际)",
        height=500, actual_series=[selected_asset],
    )
    st.altair_chart(chart_asset, use_container_width=True)

    # Summary table for this asset across scenarios
    with st.expander("📋 查看详细数据表"):
        asset_table_rows = []
        for sk, meta in SCENARIO_META.items():
            data = SCENARIOS[sk]
            for p in data:
                asset_table_rows.append({
                    "情景": meta["name"],
                    "节点": p["label"],
                    "日期": month_to_date(p["month"]),
                    f"{selected_asset} 收益率": f"{p[selected_asset]:+.1f}%",
                })
        st.dataframe(pd.DataFrame(asset_table_rows), use_container_width=True, hide_index=True)

# ---- Footer ----
st.divider()
st.caption("⚠️ 本系统所有数据为情景推演预测，非投资建议。数据来源：iran_war_scenario_analysis.md (2026-03-22 v3)")
st.caption("AV-US 成分权重：QQQM 35% | BRK-B 15% | GLDM 15% | XLE 10% | DBMF 10% | KMLM 10% | BTC 5%")
