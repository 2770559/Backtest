"""Core backtest algorithms and parsing helpers.

Pure pandas/numpy functions with no Streamlit dependency, so they can be
unit-tested and reused outside the app.
"""
import numpy as np
import pandas as pd

# Strategy name constants
STRAT_BH       = "Buy & Hold"
STRAT_ANNUAL   = "Periodic (Annual)"       # Rebalance every 365 days
STRAT_SEMI     = "Periodic (Semi-Annual)"  # Rebalance every 180 days
STRAT_RD_LOCAL = "RelDiff Local"           # Relative-diff local rebalance
STRAT_RD_MIXED = "RelDiff Mixed"           # Relative-diff mixed rebalance
STRAT_RD_FULL  = "RelDiff Full"            # Relative-diff global rebalance
STRAT_ASYM     = "Asymmetric RelDiff"      # Asymmetric relative-diff rebalance

# Legacy (Chinese) strategy names from configs exported by v1.0.x
STRAT_LEGACY_MAP = {
    "买入持有": STRAT_BH,
    "定期再平衡(年)": STRAT_ANNUAL,
    "定期再平衡(半年)": STRAT_SEMI,
    "相对差局部再平衡": STRAT_RD_LOCAL,
    "相对差混合再平衡": STRAT_RD_MIXED,
    "相对差全局再平衡": STRAT_RD_FULL,
    "不对称相对差再平衡": STRAT_ASYM,
}


def clean_ticker(t):
    t = t.strip().upper()
    mapping = {"BRK.B": "BRK-B", "ETHUSD": "ETH-USD", "BTCUSD": "BTC-USD"}
    return mapping.get(t, t)


def parse_portfolio(port):
    """Parse one portfolio config dict into (tickers, weights, errors).

    Tickers are cleaned/normalized; weights are floats. errors is a list of
    human-readable strings — empty list means the portfolio is valid.
    Both fields accept fullwidth commas.
    """
    errors = []
    t_str = str(port.get('tickers', '')).replace("，", ",")
    w_str = str(port.get('weights', '')).replace("，", ",")
    t_list = [clean_ticker(x) for x in t_str.split(',') if x.strip()]
    w_raw = [x.strip() for x in w_str.split(',') if x.strip()]

    if len(t_list) != len(w_raw):
        errors.append(f"{len(t_list)} tickers vs {len(w_raw)} weights")
        return t_list, [], errors

    dupes = sorted({t for t in t_list if t_list.count(t) > 1})
    if dupes:
        errors.append("duplicate tickers: " + ", ".join(dupes))

    try:
        w_list = [float(w) for w in w_raw]
    except ValueError:
        errors.append("invalid weight format")
        return t_list, [], errors

    total_w = sum(w_list)
    if abs(total_w - 1.0) > 0.01:
        errors.append(f"weights sum = {total_w:.2f}, should be 1.0")

    return t_list, w_list, errors


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


def scrub_leading_glitches(price_data, max_drops=3, jump=5.0):
    """Drop corrupted leading prints (IPO / listing-day scale glitches).

    Some sources (Yahoo) record a security's first trading day at the wrong
    scale, e.g. 511130.SS listing day = 0.97 vs ~97 thereafter (a 100x error).
    If the initial buy is anchored on such a print, the next day's correction
    fabricates a ~100x gain that blows up portfolio NAV. Drops any leading
    print whose step to the next valid print is physically impossible for
    these instruments (>= jump x up or <= 1/jump down in one observation).

    Mutates price_data in place; returns list of human-readable notes.
    """
    notes = []
    for tk in price_data.columns:
        valid = price_data[tk].dropna()
        guard = 0
        while len(valid) >= 2 and guard < max_drops:
            p1, p2 = valid.iloc[0], valid.iloc[1]
            if p1 > 0 and (p2 / p1 >= jump or p2 / p1 <= 1 / jump):
                price_data.loc[valid.index[0], tk] = np.nan
                notes.append(f"{tk} {valid.index[0].date()} ({p1:.4g}->{p2:.4g})")
                valid = valid.iloc[1:]
                guard += 1
            else:
                break
    return notes


def scrub_isolated_spikes(price_data, jump=5.0):
    """Drop isolated mid-series price glitches (single-print scale errors).

    A glitch print jumps >= jump x away from the previous print AND reverts
    on the very next print. Genuine crashes/rallies persist across multiple
    prints, so they are never touched.

    Mutates price_data in place; returns list of human-readable notes.
    """
    notes = []
    for tk in price_data.columns:
        valid = price_data[tk].dropna()
        if len(valid) < 3:
            continue
        v = valid.values
        for i in range(1, len(valid) - 1):
            p0, p1, p2 = v[i - 1], v[i], v[i + 1]
            if p0 <= 0 or p1 <= 0:
                continue
            spike_up = p1 / p0 >= jump and p2 / p1 <= 1 / jump
            spike_down = p1 / p0 <= 1 / jump and p2 / p1 >= jump
            if spike_up or spike_down:
                price_data.loc[valid.index[i], tk] = np.nan
                notes.append(f"{tk} {valid.index[i].date()} ({p0:.4g}->{p1:.4g}->{p2:.4g})")
    return notes


def compute_annual_returns(comp_df):
    """For each calendar year present in comp_df, compute return per column.
    Returns list of dicts: {year, returns: {col_name: pct}, partial: bool, start_date, end_date}.
    """
    if comp_df is None or comp_df.empty:
        return []
    first_date, last_date = comp_df.index[0], comp_df.index[-1]
    yearly_last = comp_df.resample('YE').last()
    years = sorted({d.year for d in comp_df.index})
    rows = []
    for y in years:
        if y == years[-1]:
            end_row = comp_df.iloc[-1]
            end_date = last_date
        else:
            mask = yearly_last.index.year == y
            if not mask.any():
                continue
            end_row = yearly_last.loc[mask].iloc[0]
            end_date = yearly_last.index[mask][0]

        if y == years[0]:
            start_row = comp_df.iloc[0]
            start_date = first_date
            partial_first = not (first_date.month == 1 and first_date.day <= 7)
        else:
            prev_mask = yearly_last.index.year == (y - 1)
            if not prev_mask.any():
                continue
            start_row = yearly_last.loc[prev_mask].iloc[0]
            start_date = yearly_last.index[prev_mask][0]
            partial_first = False

        partial_last = (y == years[-1]) and not (last_date.month == 12 and last_date.day >= 25)
        partial = partial_first or partial_last

        rets = {}
        for col in comp_df.columns:
            s = start_row[col]
            e = end_row[col]
            if pd.isna(s) or pd.isna(e) or s == 0:
                rets[col] = None
            else:
                rets[col] = (e / s) - 1
        rows.append({"year": y, "returns": rets, "partial": partial,
                     "start_date": start_date, "end_date": end_date})
    return rows
