"""2026 美伊战争 — AV-US 组合情景预测追踪（独立 Streamlit 应用，wf-2770559.streamlit.app）。

预测部分是 iran_war_scenario_analysis.md（2026-03-22 v3）的事前快照：五情景 × 七资产的
节点路径与情景概率在这里冻结、永不回改；随时间更新的只有实际数据（yfinance）。

两种预测口径：
- 原文：3/22 发布的数值，基准 = 开战前 2/27 收盘 = 0%。
- 平移校准（默认）：每条路径都从同一个「当前(3/22)」快照出发，那是分析者对 3/20 收盘的
  记录，但其中几项是估计（原文 BRK-B/DBMF/KMLM 标注「估」），XLE 的 +21.6% 实为年初至今。
  校准把每条路径整体平移，使快照恰好落在 3/20 的实际值上：保留原文预测的变化量，只替换起点。

UI 全部在 main() 里（Streamlit 以 __main__ 执行本脚本），所以测试可以直接 import 纯函数。
"""
import re
import threading
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# FORECAST: frozen snapshot of iran_war_scenario_analysis.md (2026-03-22 v3)
# ============================================================

ASSETS = ["QQQM", "BRK-B", "GLDM", "XLE", "DBMF", "KMLM", "BTC"]
WEIGHTS = {"QQQM": 0.35, "BRK-B": 0.15, "GLDM": 0.15, "XLE": 0.10,
           "DBMF": 0.10, "KMLM": 0.10, "BTC": 0.05}
PORT = "组合"
ALL_SERIES = ASSETS + [PORT]

WAR_START = pd.Timestamp("2026-02-28")
# The analysis was written on Sunday 3/22; its "当前" row records the 3/20 close.
# Every prediction is anchored here (fixed, never moves with new data).
PREDICTION_BASELINE = pd.Timestamp("2026-03-20")
DAYS_PER_MONTH = 30.44
BASELINE_MONTH = (PREDICTION_BASELINE - WAR_START).days / DAYS_PER_MONTH

SCENARIO_META = {
    "S1": {"name": "S1: 短期降级", "prob": 0.22, "desc": "4-8周内停火，海峡4-6月恢复，Brent回落$80-90"},
    "S2": {"name": "S2: 夺岛→谈判", "prob": 0.17, "desc": "海军陆战队夺哈尔格岛→经济施压→谈判→框架协议"},
    "S3": {"name": "S3: 持久低烈度（最可能）", "prob": 0.38, "desc": "无决定性胜利/停火，滞胀→Fed转向→缓慢正常化"},
    "S4": {"name": "S4: 升级+地区蔓延", "prob": 0.18, "desc": "多战场升级→流动性危机→央行紧急干预→衰退→恢复"},
    "S5": {"name": "S5: 政权更迭", "prob": 0.05, "desc": "军事+经济压力→IRGC分裂→新政权→制裁解除→石油重返"},
}


def portfolio_return(row):
    """Buy-and-hold portfolio return (%) from per-asset cumulative returns."""
    return sum(row[a] * WEIGHTS[a] for a in ASSETS)


START_NODE = {"month": 0, "label": "开战基准(2/28)",
              "QQQM": 0, "BRK-B": 0, "GLDM": 0, "XLE": 0, "DBMF": 0, "KMLM": 0, "BTC": 0}
# The analysis's "当前(3/22)" row, shared by every scenario. It sits exactly on the
# 3/20 baseline: it was the author's record of that close, not a forecast for later.
SNAPSHOT_NODE = {"month": BASELINE_MONTH, "label": "当前(3/22)",
                 "QQQM": -4.4, "BRK-B": -3, "GLDM": -14, "XLE": 21.6, "DBMF": 3, "KMLM": 2, "BTC": 11}


def _scenario(*nodes):
    rows = [dict(START_NODE), dict(SNAPSHOT_NODE), *(dict(n) for n in nodes)]
    for r in rows:
        r[PORT] = portfolio_return(r)
    return rows


SCENARIOS = {
    # ---- S1: 短期降级 ----
    "S1": _scenario(
        {"month": 1.2,  "label": "最大压力(4月初)",   "QQQM": -7,   "BRK-B": -4,  "GLDM": -18,  "XLE": 28,    "DBMF": 5,   "KMLM": 4,   "BTC": 5},
        {"month": 2.5,  "label": "停火信号(4-5月)",   "QQQM": -2,   "BRK-B": -1,  "GLDM": -16,  "XLE": 12,    "DBMF": 2,   "KMLM": 1,   "BTC": 15},
        {"month": 5,    "label": "海峡恢复50%(6-8月)","QQQM": 6,    "BRK-B": 4,   "GLDM": -10,  "XLE": 2,     "DBMF": -1,  "KMLM": -1,  "BTC": 20},
        {"month": 12,   "label": "正常化(12个月)",    "QQQM": 12,   "BRK-B": 8,   "GLDM": -5,   "XLE": -5,    "DBMF": 0,   "KMLM": -1,  "BTC": 25},
    ),
    # ---- S2: 夺岛→谈判 ----
    "S2": _scenario(
        {"month": 1.2,  "label": "空袭强化(4月初)",     "QQQM": -8,   "BRK-B": -5,  "GLDM": -16,  "XLE": 32,    "DBMF": 7,   "KMLM": 5,   "BTC": 3},
        {"month": 1.5,  "label": "夺岛D-Day(4月中)",   "QQQM": -14,  "BRK-B": -8,  "GLDM": -12,  "XLE": 45,    "DBMF": 12,  "KMLM": 9,   "BTC": -5},
        {"month": 2,    "label": "美军伤亡报道(4月下旬)","QQQM": -18,  "BRK-B": -10, "GLDM": -8,   "XLE": 48,    "DBMF": 15,  "KMLM": 11,  "BTC": -12},
        {"month": 3.5,  "label": "谈判启动(5-6月)",     "QQQM": -10,  "BRK-B": -6,  "GLDM": -10,  "XLE": 30,    "DBMF": 10,  "KMLM": 7,   "BTC": 5},
        {"month": 5.5,  "label": "框架协议(7-8月)",     "QQQM": 2,    "BRK-B": 2,   "GLDM": -12,  "XLE": 8,     "DBMF": 3,   "KMLM": 2,   "BTC": 18},
        {"month": 9,    "label": "海峡恢复(10-12月)",   "QQQM": 10,   "BRK-B": 6,   "GLDM": -8,   "XLE": -2,    "DBMF": 0,   "KMLM": -1,  "BTC": 22},
        {"month": 12,   "label": "12个月终态",          "QQQM": 13,   "BRK-B": 7,   "GLDM": -6,   "XLE": -4,    "DBMF": -1,  "KMLM": -2,  "BTC": 25},
    ),
    # ---- S3: 持久低烈度（最可能） ----
    "S3": _scenario(
        {"month": 2.5,  "label": "油价维持高位(4-5月)",      "QQQM": -8,   "BRK-B": -5,  "GLDM": -17,  "XLE": 28,    "DBMF": 6,   "KMLM": 5,   "BTC": 5},
        {"month": 3.5,  "label": "通胀数据恶化(6月)",        "QQQM": -12,  "BRK-B": -6,  "GLDM": -20,  "XLE": 25,    "DBMF": 8,   "KMLM": 7,   "BTC": 0},
        {"month": 5.5,  "label": "GDP放缓<1%(7-9月)",       "QQQM": -16,  "BRK-B": -8,  "GLDM": -18,  "XLE": 15,    "DBMF": 12,  "KMLM": 10,  "BTC": -5},
        {"month": 7,    "label": "就业恶化/开始降息(9-10月)", "QQQM": -20,  "BRK-B": -10, "GLDM": -15,  "XLE": 10,    "DBMF": 15,  "KMLM": 13,  "BTC": -10},
        {"month": 8.5,  "label": "首次降息50bp(10-11月)",    "QQQM": -15,  "BRK-B": -7,  "GLDM": -5,   "XLE": 8,     "DBMF": 13,  "KMLM": 11,  "BTC": 5},
        {"month": 11,   "label": "连续降息+QE暗示(12-3月)",  "QQQM": -8,   "BRK-B": -3,  "GLDM": 8,    "XLE": 2,     "DBMF": 10,  "KMLM": 8,   "BTC": 15},
        {"month": 18,   "label": "冲突降温(18个月)",         "QQQM": 2,    "BRK-B": 5,   "GLDM": 18,   "XLE": -3,    "DBMF": 6,   "KMLM": 4,   "BTC": 22},
        {"month": 24,   "label": "24个月终态",               "QQQM": 8,    "BRK-B": 7,   "GLDM": 25,   "XLE": -8,    "DBMF": 4,   "KMLM": 2,   "BTC": 28},
    ),
    # ---- S4: 升级+地区蔓延 ----
    "S4": _scenario(
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
    ),
    # ---- S5: 政权更迭 ----
    "S5": _scenario(
        {"month": 3.5,  "label": "滞胀深化(4-6月)",         "QQQM": -12,  "BRK-B": -6,  "GLDM": -20,  "XLE": 20,    "DBMF": 8,   "KMLM": 7,   "BTC": 0},
        {"month": 5.5,  "label": "经济放缓(7-9月)",         "QQQM": -18,  "BRK-B": -9,  "GLDM": -18,  "XLE": 10,    "DBMF": 12,  "KMLM": 10,  "BTC": -8},
        {"month": 8,    "label": "IRGC分裂/起义前(10-12月)", "QQQM": -10,  "BRK-B": -4,  "GLDM": -15,  "XLE": 5,     "DBMF": 10,  "KMLM": 8,   "BTC": 5},
        {"month": 8.5,  "label": "政权更迭事件(10-12月)",    "QQQM": 5,    "BRK-B": 3,   "GLDM": -18,  "XLE": -15,   "DBMF": 2,   "KMLM": 1,   "BTC": 20},
        {"month": 11,   "label": "新政权确认(12-15月)",      "QQQM": 15,   "BRK-B": 8,   "GLDM": -22,  "XLE": -25,   "DBMF": -3,  "KMLM": -4,  "BTC": 30},
        {"month": 18,   "label": "制裁逐步解除(18个月)",     "QQQM": 20,   "BRK-B": 10,  "GLDM": -15,  "XLE": -20,   "DBMF": -2,  "KMLM": -3,  "BTC": 35},
        {"month": 24,   "label": "24个月终态",               "QQQM": 22,   "BRK-B": 12,  "GLDM": -12,  "XLE": -22,   "DBMF": -1,  "KMLM": -2,  "BTC": 38},
    ),
}


def month_to_ts(m):
    """Scenario month offset -> date. The snapshot sits exactly on the baseline."""
    if abs(m - BASELINE_MONTH) < 1e-9:
        return PREDICTION_BASELINE
    return WAR_START + pd.Timedelta(days=round(m * DAYS_PER_MONTH))


def _day_no(dates):
    return np.asarray((pd.DatetimeIndex(dates) - WAR_START).days, dtype=float)


_BASE_DAY = float((PREDICTION_BASELINE - WAR_START).days)


def path_values(sk, dates, series, offsets=None):
    """Scenario `sk` evaluated at `dates` for each of `series`: linear between its
    nodes, NaN outside its span. `offsets` ({series: pp}, the calibrated basis) is
    phased in over the pre-baseline stretch and applied in full from the baseline
    on, so a calibrated path runs 0 (2/28) -> 3/20 actual -> the original moves."""
    nodes = SCENARIOS[sk]
    xs = _day_no([month_to_ts(n["month"]) for n in nodes])
    xd = _day_no(dates)
    ramp = np.clip(xd / _BASE_DAY, 0.0, 1.0)
    out = {}
    for s in series:
        v = np.interp(xd, xs, [n[s] for n in nodes], left=np.nan, right=np.nan)
        if offsets and s in offsets:
            v = v + offsets[s] * ramp
        out[s] = v
    return pd.DataFrame(out, index=pd.DatetimeIndex(dates))


def snapshot_values():
    return {**{a: SNAPSHOT_NODE[a] for a in ASSETS}, PORT: portfolio_return(SNAPSHOT_NODE)}


def calibration_offsets(actual):
    """pp to add to each series so the 3/22 snapshot lands on the actual close of
    the baseline day (as-of: the last print on or before 3/20). {} without data."""
    if actual is None or actual.empty:
        return {}
    i = actual.index.get_indexer([PREDICTION_BASELINE], method="pad")[0]
    if i < 0:
        return {}
    row = actual.iloc[i]
    snap = snapshot_values()
    return {s: float(row[s]) - snap[s]
            for s in ALL_SERIES if s in row.index and pd.notna(row[s])}


def _rmse(err):
    err = np.asarray(err, dtype=float)
    err = err[~np.isnan(err)]
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan


def scorecard(actual, offsets):
    """Portfolio forecast vs actual on the latest actual day, per scenario and
    probability-weighted. Tracking error = RMSE of (actual - forecast) over every
    trading day after the baseline. None when nothing was traded after 3/20."""
    if actual is None or PORT not in actual.columns:
        return None
    days = actual.index[actual.index > PREDICTION_BASELINE]
    if not len(days):
        return None
    act = actual.loc[days, PORT]
    rows = []
    weighted = pd.Series(0.0, index=days)
    for sk, meta in SCENARIO_META.items():
        pred = path_values(sk, days, [PORT], offsets)[PORT]
        weighted = weighted + meta["prob"] * pred
        rows.append({"key": sk, "name": meta["name"], "prob": meta["prob"],
                     "pred": float(pred.iloc[-1]), "rmse": _rmse(act - pred)})
    rows.append({"key": "W", "name": "概率加权", "prob": 1.0,
                 "pred": float(weighted.iloc[-1]), "rmse": _rmse(act - weighted)})
    return {"day": days[-1], "actual": float(act.iloc[-1]), "rows": rows}


def weighted_forecast(day, series, offsets):
    """Probability-weighted forecast of every series on `day`."""
    total = {s: 0.0 for s in series}
    for sk, meta in SCENARIO_META.items():
        vals = path_values(sk, [day], series, offsets).iloc[0]
        for s in series:
            total[s] += meta["prob"] * vals[s]
    return total


def portfolio_path_stats(sk, offsets):
    """Lowest portfolio node (label/date/value) and the 12/24-month values of one
    scenario, on the same basis as the charts. NaN past the scenario's horizon."""
    nodes = SCENARIOS[sk]
    dates = [month_to_ts(n["month"]) for n in nodes]
    vals = path_values(sk, dates, [PORT], offsets)[PORT].to_numpy()
    i = int(np.nanargmin(vals))
    at = path_values(sk, [month_to_ts(12), month_to_ts(24)], [PORT], offsets)[PORT].to_numpy()
    return {"low": float(vals[i]), "low_label": nodes[i]["label"], "low_date": dates[i],
            "m12": float(at[0]), "m24": float(at[1])}


# ============================================================
# ACTUAL DATA (yfinance, adjusted closes)
# ============================================================
# Yahoo symbols equal the display names; BTC = Grayscale Bitcoin Mini Trust (NYSE Arca),
# so every leg trades on US days.
TICKERS = list(ASSETS)
FETCH_START = (WAR_START - pd.Timedelta(days=10)).strftime("%Y-%m-%d")  # covers the 2/27 base close
ACTUAL_TTL = 3600         # a complete download is reused for an hour
ACTUAL_RETRY_TTL = 120    # an incomplete one is retried after two minutes
RETRY_PAUSE = 1.0         # seconds before the one immediate retry of a non-rate-limited failure
_RATE_LIMIT_MARKERS = ("rate limit", "too many requests")


@st.cache_resource(show_spinner=False)
def _actual_store():
    """Process-wide singleton: {"lock", "entry": {expires, closes, failures, stale}}.
    The lock also serializes yf.download, whose per-ticker errors land in
    module-global state (yf.shared._ERRORS)."""
    return {"lock": threading.Lock(), "entry": None}


def _tidy_reason(reason):
    """"YFRateLimitError('Too Many Requests...')" -> "Too Many Requests..."."""
    m = re.fullmatch(r"\w+\((['\"])(.*)\1\)", str(reason).strip(), flags=re.S)
    return m.group(2) if m else str(reason).strip()


def _is_rate_limited(reason):
    return any(m in str(reason).lower() for m in _RATE_LIMIT_MARKERS)


def _close_series(raw, tk):
    """One ticker's close out of a yf.download frame, or None when it has no price.
    Picked per ticker: a failed ticker comes back as an all-NaN placeholder column."""
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return None
    for lvl in ("Close", "Adj Close"):
        if (lvl, tk) in raw.columns:
            s = pd.to_numeric(raw[(lvl, tk)], errors="coerce").dropna()
            if not s.empty:
                return s.rename(tk)
    return None


def _download_closes(tickers, download):
    """One yf.download batch -> ({ticker: close}, {ticker: reason})."""
    yf.shared._ERRORS = {}
    raw = download(list(tickers), start=FETCH_START, auto_adjust=True, progress=False)
    reasons = {str(k).upper(): _tidy_reason(v)
               for k, v in (getattr(yf.shared, "_ERRORS", None) or {}).items()}
    good, bad = {}, {}
    for tk in tickers:
        s = _close_series(raw, tk)
        if s is None:
            bad[tk] = reasons.get(tk.upper(), "Yahoo 未返回价格")
        else:
            good[tk] = s
    return good, bad


def load_actual(force=False, now=None, download=None):
    """Closes for every ticker -> (closes, failures, stale).

    A complete download is served for ACTUAL_TTL; an incomplete one only for
    ACTUAL_RETRY_TTL, so a transient Yahoo failure heals itself instead of hiding
    the actual line for an hour. A ticker that fails keeps its last good series
    (listed in `stale`). `download`/`now` are injection points for tests."""
    download = download or yf.download
    now = time.time() if now is None else now
    store = _actual_store()
    with store["lock"]:
        e = store["entry"]
        if force or e is None or e["expires"] <= now:
            try:
                good, bad = _download_closes(TICKERS, download)
                retry = [tk for tk, why in bad.items() if not _is_rate_limited(why)]
                if retry:
                    time.sleep(RETRY_PAUSE)
                    good2, bad2 = _download_closes(retry, download)
                    good.update(good2)
                    bad = {tk: why for tk, why in {**bad, **bad2}.items() if tk not in good}
            except Exception as exc:  # network down, yfinance internals, ...
                good, bad = {}, {tk: _tidy_reason(exc) for tk in TICKERS}
            prev = e["closes"] if e else {}
            stale = [tk for tk in bad if tk in prev]
            closes = {**{tk: prev[tk] for tk in stale}, **good}
            failures = {tk: why for tk, why in bad.items() if tk not in closes}
            ttl = ACTUAL_TTL if not bad else ACTUAL_RETRY_TTL
            e = store["entry"] = {"expires": now + ttl, "closes": closes,
                                  "failures": failures, "stale": stale}
        return dict(e["closes"]), dict(e["failures"]), list(e["stale"])


def cumulative_returns(closes):
    """Cumulative % return vs the last close before the war (2/27) on every trading
    day from 2/28 on, plus the buy-and-hold 组合 over the assets that have data
    (weights renormalized). -> (frame or None, assets without data)."""
    if not closes:
        return None, list(ASSETS)
    close = pd.concat([closes[a] for a in ASSETS if a in closes], axis=1, sort=True)
    # ffill BEFORE picking the base row: one missing print on 2/27 would otherwise
    # make that asset's whole return series (and thus 组合) NaN.
    close = close.dropna(how="all").ffill()
    pre = close.index < WAR_START
    if not pre.any():
        return None, list(ASSETS)
    ret = (close / close.loc[pre].iloc[-1] - 1) * 100
    ret = ret.loc[ret.index >= WAR_START]
    available = [a for a in ASSETS if a in ret.columns and ret[a].notna().any()]
    missing = [a for a in ASSETS if a not in available]
    if not available or ret.empty:
        return None, list(ASSETS)
    wsum = sum(WEIGHTS[a] for a in available)
    ret[PORT] = sum(ret[a] * (WEIGHTS[a] / wsum) for a in available)
    ret = ret.dropna(subset=[PORT])
    if ret.empty:
        return None, list(ASSETS)
    return ret[available + [PORT]], missing


def value_asof(series, day):
    """Last value on or before `day` (never a later print); None outside the data."""
    if series is None or series.empty or day > series.index[-1]:
        return None
    i = series.index.get_indexer([day], method="pad")[0]
    return None if i < 0 or pd.isna(series.iloc[i]) else round(float(series.iloc[i]), 2)


# ============================================================
# PRESENTATION
# ============================================================
# Categorical palettes, dataviz-validated on the adjacent pairlist (light vs
# #ffffff, dark vs #0e1117 = Streamlit's surfaces). Slot order is the CVD-safety
# mechanism - never re-sort. Assets in ALL_SERIES order (GLDM gold, 组合 red).
_ASSET_HEX = {
    "light": ["#008300", "#e87ba4", "#eda100", "#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6", "#e34948"],
    "dark":  ["#008300", "#d55181", "#c98500", "#199e70", "#9085e9", "#d95926", "#3987e5", "#e66767"],
}
# S1..S5. Five crossing lines cannot clear the all-pairs floor (S2/S5 sit close in
# dark), so scenario charts also label every line end directly.
_SCEN_HEX = {
    "light": ["#008300", "#2a78d6", "#eda100", "#e87ba4", "#4a3aa7"],
    "dark":  ["#008300", "#3987e5", "#c98500", "#d55181", "#9085e9"],
}
_TOKENS = {
    "light": {"card": "#ffffff", "ink1": "#1a1a2e", "ink2": "#52514e", "muted": "#6b6a66",
              "border": "rgba(11,11,11,0.10)", "shadow": "0 1px 4px rgba(15,23,42,0.06)",
              "actual": "#0b0b0b", "rule": "#94a3b8", "zero": "#898781"},
    "dark":  {"card": "#1b1f27", "ink1": "#fafafa", "ink2": "#c3c2b7", "muted": "#a3a29c",
              "border": "rgba(255,255,255,0.12)", "shadow": "0 1px 4px rgba(0,0,0,0.35)",
              "actual": "#fafafa", "rule": "#64748b", "zero": "#898781"},
}
BASIS_LABELS = {"calibrated": "平移校准（锚定3/20实际）", "raw": "原文数值（3/22发布）"}


def _theme_type():
    """Active Streamlit theme type; safe fallback for AppTest / bare mode."""
    try:
        t = st.context.theme.type
    except Exception:
        t = None
    return t if t in ("light", "dark") else "light"


def _pct(v, digits=1):
    return "—" if v is None or pd.isna(v) else f"{v:+.{digits}f}%"


def chart_frame(scenario_keys, series, offsets, extra_dates=()):
    """Long frame (日期/情景/key/资产/收益率) on the union of every node date plus
    `extra_dates` (hover stops); points outside a scenario's span are dropped."""
    grid = sorted({month_to_ts(n["month"]) for sk in scenario_keys for n in SCENARIOS[sk]}
                  | {pd.Timestamp(d) for d in extra_dates})
    rows = []
    for sk in scenario_keys:
        vals = path_values(sk, grid, series, offsets)
        for day, row in vals.iterrows():
            for s in series:
                if pd.notna(row[s]):
                    rows.append({"日期": day, "情景": SCENARIO_META[sk]["name"], "key": sk,
                                 "资产": s, "收益率(%)": round(float(row[s]), 2)})
    return pd.DataFrame(rows)


def _merge_close_labels(ends, min_gap):
    """Direct labels that would print on top of each other (same end date, values
    closer than `min_gap`) become one label, e.g. "S1·S2"."""
    out = []
    for day, grp in ends.groupby("日期"):
        cluster = []
        for _, r in grp.sort_values("收益率(%)").iterrows():
            if cluster and r["收益率(%)"] - cluster[-1]["收益率(%)"] >= min_gap:
                out.append(cluster)
                cluster = []
            cluster.append(r)
        out.append(cluster)
    return pd.DataFrame([{"日期": c[0]["日期"],
                          "收益率(%)": float(np.mean([r["收益率(%)"] for r in c])),
                          "lbl": "·".join(sorted(r["lbl"] for r in c))} for c in out])


def build_chart(df, field, domain, colors, title, tokens, actual=None, actual_name=None,
                end_labels=None, height=460):
    """Prediction lines (colored by `field`) + crosshair tooltip, with the actual
    series drawn in the ink color and labeled at its last print.
    end_labels: {domain value: short text} placed at each line's last point."""
    span_months = (df["日期"].max() - df["日期"].min()).days / DAYS_PER_MONTH
    step = 1 if span_months <= 13 else 2 if span_months <= 26 else 3
    x = alt.X("日期:T", title=None, axis=alt.Axis(
        format="%Y-%m", labelAngle=-45, tickCount={"interval": "month", "step": step}))
    y = alt.Y("收益率(%):Q", title="累计收益率 (%)")
    color = alt.Color(f"{field}:N", title=field,
                      scale=alt.Scale(domain=domain, range=[colors[d] for d in domain]))

    # Wide frame for the crosshair tooltip: one column per series (+ actual).
    wide = (df.pivot_table(index="日期", columns=field, values="收益率(%)", aggfunc="first")
            .reset_index().sort_values("日期"))
    tooltips = [alt.Tooltip("日期:T", format="%Y-%m-%d", title="日期")]
    for col in domain:
        if col in wide.columns:
            tooltips.append(alt.Tooltip(field=col, type="quantitative", format="+.1f", title=f"{col} (%)"))
    if actual is not None and not actual.empty:
        wide[actual_name] = [value_asof(actual, d) for d in wide["日期"]]
        tooltips.append(alt.Tooltip(field=actual_name, type="quantitative", format="+.1f",
                                    title=f"{actual_name} (%)"))

    nearest = alt.selection_point(nearest=True, on="mouseover", fields=["日期"], empty=False)
    layers = [
        alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            strokeDash=[4, 4], color=tokens["zero"], strokeWidth=1).encode(y="y:Q"),
        alt.Chart(df).mark_line(strokeWidth=2.5, opacity=0.9).encode(x=x, y=y, color=color),
        # Invisible thick rule for mouse capture
        alt.Chart(wide).mark_rule(opacity=0.001, strokeWidth=40).encode(
            x="日期:T", tooltip=tooltips).add_params(nearest),
        alt.Chart(wide).mark_rule(color=tokens["rule"], strokeDash=[3, 3]).encode(
            x="日期:T", tooltip=tooltips).transform_filter(nearest),
        alt.Chart(df).mark_point(size=60, filled=True).encode(
            x="日期:T", y="收益率(%):Q", color=color,
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))),
    ]
    if end_labels:
        ends = df.sort_values("日期").groupby(field, as_index=False).last()
        ends["lbl"] = ends[field].map(end_labels)
        values = df["收益率(%)"]
        if actual is not None and not actual.empty:
            values = pd.concat([values, actual.dropna()])
        ends = _merge_close_labels(ends, 0.045 * max(values.max() - values.min(), 1.0))
        layers.append(alt.Chart(ends).mark_text(
            align="left", dx=6, fontSize=11, fontWeight="bold", color=tokens["ink2"]).encode(
            x="日期:T", y="收益率(%):Q", text="lbl:N"))
    if actual is not None and not actual.empty:
        adf = pd.DataFrame({"日期": actual.index, "收益率(%)": actual.to_numpy()}).dropna()
        last = adf.iloc[[-1]].assign(lbl=f"实际 {adf['收益率(%)'].iloc[-1]:+.1f}%")
        layers += [
            alt.Chart(adf).mark_line(strokeWidth=3, color=tokens["actual"]).encode(
                x="日期:T", y="收益率(%):Q"),
            alt.Chart(last).mark_point(size=50, filled=True, color=tokens["actual"]).encode(
                x="日期:T", y="收益率(%):Q"),
            alt.Chart(last).mark_text(align="left", dx=7, dy=-9, fontSize=12, fontWeight="bold",
                                      color=tokens["actual"]).encode(
                x="日期:T", y="收益率(%):Q", text="lbl:N"),
        ]
    return alt.layer(*layers).properties(
        title=alt.Title(title, fontSize=15, anchor="start"),
        height=height,
        padding={"left": 5, "top": 5, "right": 40, "bottom": 5},
    ).configure_legend(
        orient="bottom", columns=4, titleFontSize=12, labelFontSize=11
    ).configure_view(strokeWidth=0)


def _request_refresh():
    st.session_state["_force_refresh"] = True


def main():
    st.set_page_config(page_title="伊朗战争情景预测对比", layout="wide", page_icon="⚔️")

    theme = _theme_type()
    tok = _TOKENS[theme]
    asset_colors = dict(zip(ALL_SERIES, _ASSET_HEX[theme]))
    scen_colors = dict(zip(SCENARIO_META, _SCEN_HEX[theme]))
    actual_word = "黑线" if theme == "light" else "白线"
    scen_names = [m["name"] for m in SCENARIO_META.values()]
    scen_name_colors = {m["name"]: scen_colors[k] for k, m in SCENARIO_META.items()}
    scen_end_labels = {m["name"]: k for k, m in SCENARIO_META.items()}

    st.markdown("<style>:root{" + "".join(f"--{k}:{v};" for k, v in tok.items()) + "}</style>",
                unsafe_allow_html=True)
    st.markdown("""
<style>
section.main > div { max-width: 1400px; margin: 0 auto; }
.header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white; padding: 1.25rem 1.75rem; border-radius: 0.75rem;
    margin-bottom: 1.25rem;
}
.header-bar h2 { margin: 0; font-size: 1.4rem; font-weight: 700; color: #fff; }
.header-bar .subtitle { opacity: 0.7; font-size: 0.85rem; margin-top: 0.3rem; }
.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; margin-bottom: 1rem; }
@media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
.kpi-card {
    background: var(--card); border: 1px solid var(--border); border-left: 4px solid var(--border);
    border-radius: 0.6rem; padding: 0.875rem 1rem; box-shadow: var(--shadow);
}
.kpi-card .label { font-size: 0.72rem; color: var(--muted);
    letter-spacing: 0.05em; margin-bottom: 0.25rem; font-weight: 600; }
.kpi-card .value { font-size: 1.2rem; font-weight: 700; color: var(--ink1); }
.kpi-card .value .unit { font-size: 0.72rem; font-weight: 600; color: var(--muted); margin-right: 0.3rem; }
</style>
""", unsafe_allow_html=True)

    # ---- Header ----
    st.markdown("""
<div class="header-bar">
    <div>
        <h2>⚔️ 2026美伊战争 — AV-US组合情景预测对比系统</h2>
        <div class="subtitle">基于 iran_war_scenario_analysis.md (2026-03-22 v3) · 五情景×七资产事前推演（冻结，不回改）· 实际数据自动更新(yfinance)</div>
    </div>
</div>
""", unsafe_allow_html=True)

    kpi_html = '<div class="kpi-row">'
    for sk, meta in SCENARIO_META.items():
        kpi_html += (f'<div class="kpi-card" style="border-left-color:{scen_colors[sk]}">'
                     f'<div class="label">{meta["name"]}</div>'
                     f'<div class="value"><span class="unit">3/22 概率</span>{meta["prob"]:.0%}</div></div>')
    st.markdown(kpi_html + '</div>', unsafe_allow_html=True)

    # ---- Actual data ----
    with st.spinner("获取实际行情…"):
        closes, failures, stale = load_actual(force=st.session_state.pop("_force_refresh", False))
    actual, missing = cumulative_returns(closes)

    col_status, col_refresh = st.columns([0.85, 0.15])
    with col_status:
        if actual is not None:
            st.caption(f"📡 实际数据最后交易日: **{actual.index[-1]:%Y-%m-%d}** | "
                       f"组合实际累计收益: **{actual[PORT].iloc[-1]:+.2f}%**（2/27 收盘买入持有）| "
                       f"数据缓存1小时，可点击刷新")
        else:
            st.warning("⚠️ 未能获取实际市场数据，图表仅显示预测数据")
    with col_refresh:
        st.button("🔄 刷新数据", on_click=_request_refresh)
    if actual is not None and missing:
        why = "；".join(f"{a}: {failures[a]}" for a in missing if a in failures)
        st.warning("实际数据缺少资产: " + ", ".join(missing) + "（组合线已按剩余权重归一化）"
                   + (f" — {why}" if why else ""))
    if stale:
        st.caption("ℹ️ 本次刷新未取到 " + ", ".join(stale) + "，暂用上次成功的数据，2 分钟后自动重试")

    basis = st.radio(
        "预测口径", list(BASIS_LABELS), format_func=BASIS_LABELS.get, horizontal=True, key="basis",
        help="原文：3/22 发布的数值。平移校准：每条路径整体平移，使「当前(3/22)」快照落在 3/20 的实际值上"
             "——保留原文预测的变化量，只替换起点（原文快照有估计值，XLE 用的是年初至今）。")
    offsets = calibration_offsets(actual) if basis == "calibrated" else None
    if basis == "calibrated" and not offsets:
        st.caption("没有 3/20 的实际数据可供校准，预测线按原文数值显示。")

    last_day = actual.index[-1] if actual is not None else None
    # Hover stops along the tracked stretch: every month's last trading day + the latest.
    hover_days = []
    if actual is not None:
        tracked = actual.index[actual.index > PREDICTION_BASELINE]
        hover_days = list(pd.Series(tracked, index=tracked).groupby(tracked.to_period("M")).max())

    tab1, tab2, tab3 = st.tabs([
        "📊 全景总览（所有情景×所有资产）",
        "🎯 单情景查看（选择S1-S5）",
        "📈 单资产跨情景对比",
    ])

    # ---- TAB 1: 全景总览 ----
    with tab1:
        st.markdown("### 全部五种情景下 AV-US 组合走势对比")
        st.caption(f"彩色线 = 各情景的组合预测（{BASIS_LABELS[basis]}）；{actual_word} = 实际组合。"
                   "基准：2/27 收盘（开战前）= 0%")

        col_chart, col_score = st.columns([1.15, 1])
        with col_chart:
            df_p = chart_frame(list(SCENARIOS), [PORT], offsets, hover_days)
            st.altair_chart(build_chart(
                df_p, "情景", scen_names, scen_name_colors,
                f"AV-US 组合：五情景走势（{actual_word}=实际）", tok,
                actual=actual[PORT] if actual is not None else None, actual_name=f"实际:{PORT}",
                end_labels=scen_end_labels, height=440), width="stretch")

        with col_score:
            card = scorecard(actual, offsets)
            if card is None:
                st.info("暂无 3/20 之后的实际数据，无法记分。")
            else:
                st.markdown(f"#### 预测记分 · 截至 {card['day']:%Y-%m-%d}")
                st.markdown(f"实际组合 **{card['actual']:+.2f}%**")
                best = min((r for r in card["rows"] if r["key"] != "W"), key=lambda r: r["rmse"])
                st.dataframe(pd.DataFrame([{
                    "情景": ("★ " if r is best else "") + r["name"],
                    "概率": "—" if r["key"] == "W" else f"{r['prob']:.0%}",
                    "预测组合": _pct(r["pred"]),
                    "偏差(实际−预测)": _pct(card["actual"] - r["pred"]),
                    "跟踪误差": f"{r['rmse']:.1f}pp",
                } for r in card["rows"]]), width="stretch", hide_index=True)
                st.caption("偏差为正 = 实际好于预测。跟踪误差 = 3/20 之后逐交易日偏差的均方根，"
                           "★ = 跟踪误差最小（整条路径最贴近实际）。概率为 3/22 的判断。")

                st.markdown("#### 概率加权预测 vs 实际（分资产）")
                present = [s for s in ALL_SERIES if s in actual.columns]
                wf = weighted_forecast(card["day"], present, offsets)
                st.dataframe(pd.DataFrame([{
                    "资产": s,
                    "权重": f"{WEIGHTS[s]:.0%}" if s in WEIGHTS else "加权组合",
                    "概率加权预测": _pct(wf[s]),
                    "实际": _pct(actual.loc[card["day"], s]),
                    "偏差(实际−预测)": _pct(actual.loc[card["day"], s] - wf[s]),
                } for s in present]), width="stretch", hide_index=True)

        col_term, col_dd = st.columns(2)
        with col_term:
            st.markdown("#### 五情景终态对比")
            rows = []
            for sk, meta in SCENARIO_META.items():
                end = month_to_ts(SCENARIOS[sk][-1]["month"])
                vals = path_values(sk, [end], ALL_SERIES, offsets).iloc[0]
                rows.append({"情景": meta["name"], "概率": f"{meta['prob']:.0%}",
                             "终态日期": f"{end:%Y-%m-%d}", **{s: _pct(vals[s]) for s in ALL_SERIES}})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        with col_dd:
            st.markdown("#### 组合路径：最低点与 12/24 个月")
            rows, w12 = [], 0.0
            for sk, meta in SCENARIO_META.items():
                ps = portfolio_path_stats(sk, offsets)
                w12 += meta["prob"] * ps["m12"]
                rows.append({"情景": meta["name"], "最低点": _pct(ps["low"]),
                             "时点": f"{ps['low_label']} · {ps['low_date']:%m/%d}",
                             "12个月": _pct(ps["m12"]), "24个月": _pct(ps["m24"])})
            rows.append({"情景": "概率加权", "最低点": "—", "时点": "—", "12个月": _pct(w12), "24个月": "—"})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("由情景路径节点计算，与图同一口径（原文汇总表的组合数字与其自身路径不一致，已不再使用）。"
                       "S1/S2 只推演到 12 个月，故无 24 个月值。")

        with st.expander("🔧 平移校准说明：原文快照 vs 3/20 实际"):
            base = calibration_offsets(actual)
            if not base:
                st.caption("没有 3/20 的实际数据。")
            else:
                snap = snapshot_values()
                st.dataframe(pd.DataFrame([{
                    "资产": s, "原文快照(3/22)": _pct(snap[s]),
                    "3/20 实际": _pct(snap[s] + base[s]), "平移量": f"{base[s]:+.1f}pp",
                } for s in ALL_SERIES if s in base]), width="stretch", hide_index=True)
                st.caption("校准口径下，每条预测路径在 3/20 之后整体加上该资产的平移量；3/20 之前从 0% 线性过渡。"
                           "原文 XLE +21.6% 为年初至今涨幅（自 2/27 起实际约 +6%），BRK-B/DBMF/KMLM 为估计值。")

    # ---- TAB 2: 单情景查看 ----
    with tab2:
        st.markdown("### 选择情景，查看该情景下所有资产的全过程走势")
        sel = st.radio("选择情景：", list(SCENARIO_META),
                       format_func=lambda k: f"{SCENARIO_META[k]['name']} (3/22概率{SCENARIO_META[k]['prob']:.0%})",
                       horizontal=True, key="scenario_radio")
        meta = SCENARIO_META[sel]
        st.info(f"**{meta['name']}** — {meta['desc']}")

        df_s = chart_frame([sel], ALL_SERIES, offsets, hover_days)
        st.altair_chart(build_chart(
            df_s, "资产", ALL_SERIES, asset_colors,
            f"{meta['name']} — 各资产全过程走势（{actual_word}=实际组合）", tok,
            actual=actual[PORT] if actual is not None else None, actual_name=f"实际:{PORT}",
            height=500), width="stretch")

        with st.expander("📋 查看详细数据表"):
            nodes = SCENARIOS[sel]
            dates = [month_to_ts(n["month"]) for n in nodes]
            vals = path_values(sel, dates, ALL_SERIES, offsets)
            st.dataframe(pd.DataFrame([{
                "节点": n["label"], "日期": f"{d:%Y-%m-%d}", **{s: _pct(vals.iloc[i][s]) for s in ALL_SERIES},
            } for i, (n, d) in enumerate(zip(nodes, dates))]), width="stretch", hide_index=True)
            st.caption(f"口径：{BASIS_LABELS[basis]}")

    # ---- TAB 3: 单资产跨情景 ----
    with tab3:
        st.markdown("### 选择资产，查看该资产在五种情景下的走势对比")
        asset = st.radio("选择资产：", ALL_SERIES,
                         format_func=lambda a: f"{a} ({WEIGHTS[a]:.0%})" if a in WEIGHTS else f"{a} (加权组合)",
                         horizontal=True, key="asset_radio")
        df_a = chart_frame(list(SCENARIOS), [asset], offsets, hover_days)
        weight_info = f"权重 {WEIGHTS[asset]:.0%}" if asset in WEIGHTS else "加权组合"
        has_actual = actual is not None and asset in actual.columns
        st.altair_chart(build_chart(
            df_a, "情景", scen_names, scen_name_colors,
            f"{asset} ({weight_info}) — 五情景走势对比（{actual_word}=实际）", tok,
            actual=actual[asset] if has_actual else None, actual_name=f"实际:{asset}",
            end_labels=scen_end_labels, height=500), width="stretch")

        with st.expander("📋 查看详细数据表"):
            rows = []
            for sk, meta in SCENARIO_META.items():
                nodes = SCENARIOS[sk]
                dates = [month_to_ts(n["month"]) for n in nodes]
                vals = path_values(sk, dates, [asset], offsets)[asset]
                rows += [{"情景": meta["name"], "节点": n["label"], "日期": f"{d:%Y-%m-%d}",
                          f"{asset} 收益率": _pct(v)} for n, d, v in zip(nodes, dates, vals)]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(f"口径：{BASIS_LABELS[basis]}")

    # ---- Footer ----
    st.divider()
    st.caption("⚠️ 本系统所有数据为情景推演预测，非投资建议。数据来源：iran_war_scenario_analysis.md (2026-03-22 v3)")
    st.caption("AV-US 成分权重（3/22）：" + " | ".join(f"{a} {WEIGHTS[a]:.0%}" for a in ASSETS))


if __name__ == "__main__":
    main()
