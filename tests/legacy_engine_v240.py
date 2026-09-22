"""Verbatim copy of the v2.4.0 engine (backtest_core.py @ fd32620), frozen as a
reference for bit-identity regression tests.

Do NOT edit. tests/test_bands.py runs random price matrices through both this
copy and the live engine and asserts byte-identical histories / counts / PnL
whenever the up band equals the down band (threshold_up=None or == threshold).
"""
import numpy as np
import pandas as pd

from backtest_core import (
    STRAT_ANNUAL, STRAT_SEMI, STRAT_ASYM, STRAT_RD_FULL, STRAT_RD_MIXED, STRAT_RD_LOCAL,
)


def apply_local_rebalance(asset_values, target_weights, threshold, return_resets=False):
    """Local relative-diff rebalance.

    Triggered indices are reset to their target weight; the remainder is
    redistributed proportionally to preserve its internal drift.

    When return_resets=True, also returns the set of indices that were RESET to
    target (vs only proportionally scaled) — the composite engine uses it to
    decide which slots restore their default equal split. The math is byte-for-byte
    the original; the only addition is threading `reset_indices` out. Default
    False keeps the original single-value return so existing callers are unaffected.
    """
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
        if not rem_indices:
            out = total_val * target_weights
            return (out, set(current_vals.index)) if return_resets else out
        rem_cash = total_val - current_vals[reset_indices].sum()
        current_rem_sum = asset_values[rem_indices].sum()
        if current_rem_sum > 0: ratios = asset_values[rem_indices] / current_rem_sum
        else: ratios = target_weights[rem_indices] / target_weights[rem_indices].sum()
        current_vals[rem_indices] = ratios * rem_cash
    return (current_vals, set(reset_indices)) if return_resets else current_vals


def run_detailed_backtest(strategy_name, price_df, target_weights, initial_cap,
                          threshold, groups=None):
    """Backtest one portfolio under one rebalance strategy.

    groups: optional element-ticker -> slot-id mapping (dict[str, str]). Elements
    sharing a slot-id form a COMPOSITE: rebalance decisions are made on the slot's
    AGGREGATE weight (elements never trigger individually), but each element keeps
    its own price series so NAV = sum over elements. On a reset the slot value is
    restored to an EQUAL split among its elements ("再平衡时恢复默认份额"); between
    resets the elements drift and the slot moves "as one block".

    groups=None  ==  every column is its own slot  ==  original behaviour,
    bit-identical (singletons short-circuit the unfold; see below).

    threshold: float, or dict {slot_id: thr} for PER-SLOT bands (slot_id = the
    ticker for singleton slots / the groups value for composites). "*" supplies
    the default for unlisted slots and is required if any slot is missing.
    A scalar is broadcast — decisions are bit-identical to the original code.
    """
    tickers = price_df.columns
    if price_df.empty: return pd.DataFrame(), 0, {}
    start_prices = price_df.iloc[0]
    if start_prices.isna().any():
        start_prices = price_df.bfill().iloc[0]
        if start_prices.isna().any(): return pd.DataFrame(), 0, {}

    # Force alignment so groupby/elementwise ops are well-defined (no-op for valid
    # callers, where target_weights is already indexed by price_df.columns).
    target_weights = target_weights.reindex(tickers)

    # ---- Slot grouping (identity when groups is None => default path) ----
    gmap = groups or {}
    slot_of = pd.Series({t: gmap.get(t, t) for t in tickers}, index=tickers)
    # Stable slot order = first appearance across columns. For the identity map
    # this is exactly price_df.columns, which guarantees bit-identity.
    slot_ids = list(dict.fromkeys(slot_of.tolist()))
    slot_members = {s: [t for t in tickers if slot_of[t] == s] for s in slot_ids}

    # Slot-level targets = SUM of member element targets (singleton -> itself).
    slot_targets = target_weights.groupby(slot_of).sum().reindex(slot_ids)

    # Normalize threshold: scalar stays scalar (broadcasts bit-identically);
    # a dict becomes a Series aligned to slot_ids ("*" = default band).
    if isinstance(threshold, dict):
        missing = [s for s in slot_ids if s not in threshold]
        if missing and "*" not in threshold:
            raise ValueError(f"threshold dict missing slots {missing} and no '*' default")
        threshold = pd.Series({s: float(threshold.get(s, threshold.get("*"))) for s in slot_ids})

    # Per-element initial weights = slot target split EQUALLY among its members.
    member_counts = slot_of.map(slot_of.value_counts())            # aligned to columns
    elem_init_weights = target_weights.groupby(slot_of).transform('sum') / member_counts
    current_shares = (initial_cap * elem_init_weights) / start_prices

    history = []
    last_rebalance_date = price_df.index[0]
    rebalance_count = 0
    price_df_filled = price_df.ffill()

    cumulative_pnl = pd.Series(0.0, index=tickers)                 # PnL stays element-level
    prev_prices = start_prices

    for i in range(len(price_df)):
        current_date = price_df.index[i]
        current_prices = price_df_filled.iloc[i]

        # fillna: with leading-NaN price series (asset not yet listed) the NaN
        # diff would otherwise poison that asset's cumulative PnL permanently.
        if i > 0: cumulative_pnl += (current_shares * (current_prices - prev_prices)).fillna(0.0)
        prev_prices = current_prices

        asset_values = current_shares * current_prices            # element-level $
        total_val = asset_values.sum()
        if total_val == 0 or np.isnan(total_val): continue
        current_weights = asset_values / total_val                # element-level weights

        if i == 0:
            rec = {"Date": current_date, "Type": "Init", "NAV": total_val}
            rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec); continue

        # ---- FOLD: aggregate elements up to slots for the decision ----
        slot_values = asset_values.groupby(slot_of).sum().reindex(slot_ids)
        slot_weights = slot_values / total_val

        do_rebalance = False
        new_slot_values = slot_values.copy()
        reset_slots = set()                                       # slots restored to default split

        # ---- DECISION: EXISTING trigger math, now on slot Series ----
        if strategy_name == STRAT_ANNUAL:
            if (current_date - last_rebalance_date).days >= 365:
                new_slot_values, reset_slots, do_rebalance = total_val * slot_targets, set(slot_ids), True
        elif strategy_name == STRAT_SEMI:
            if (current_date - last_rebalance_date).days >= 180:
                new_slot_values, reset_slots, do_rebalance = total_val * slot_targets, set(slot_ids), True

        elif strategy_name == STRAT_ASYM:
            diff_ratio = (slot_weights - slot_targets) / slot_targets.replace(0, 1e-9)
            mask_major = slot_targets >= 0.06
            mask_minor = slot_targets < 0.06

            trigger_major = mask_major & (np.abs(diff_ratio) > threshold)
            trigger_minor_up = mask_minor & (diff_ratio > threshold * 2.5)
            trigger_minor_down = mask_minor & (diff_ratio < -threshold * 1.25)

            if trigger_major.any() or trigger_minor_up.any() or trigger_minor_down.any():
                new_slot_values, reset_slots, do_rebalance = total_val * slot_targets, set(slot_ids), True

        elif "RelDiff" in strategy_name:
            rel_diffs = np.abs(slot_weights - slot_targets) / slot_targets.replace(0, 1e-9)
            # (rel_diffs > threshold).any() == rel_diffs.max() > threshold for a
            # scalar; the elementwise form also supports per-slot bands.
            if (rel_diffs > threshold).any():
                if strategy_name == STRAT_RD_FULL:
                    new_slot_values, reset_slots, do_rebalance = total_val * slot_targets, set(slot_ids), True
                elif strategy_name == STRAT_RD_MIXED:
                    if ((slot_targets >= 0.1) & (rel_diffs > threshold)).any():
                        new_slot_values, reset_slots = total_val * slot_targets, set(slot_ids)
                    else:
                        new_slot_values, reset_slots = apply_local_rebalance(
                            slot_values, slot_targets, threshold, return_resets=True)
                    do_rebalance = True
                elif strategy_name == STRAT_RD_LOCAL:
                    new_slot_values, reset_slots = apply_local_rebalance(
                        slot_values, slot_targets, threshold, return_resets=True)
                    do_rebalance = True

        if do_rebalance:
            rebalance_count += 1
            pre_rec = {"Date": current_date, "Type": "Pre-Rebal", "NAV": total_val}
            pre_rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(pre_rec)

            # ---- UNFOLD: expand slot values back to element values ----
            # Singletons assign directly (no float round-trip) => legacy path is
            # byte-identical. Composites: reset -> equal default split; otherwise
            # scale each element as one block, preserving internal drift.
            new_values = asset_values.copy()
            for s in slot_ids:
                members = slot_members[s]
                nsv = new_slot_values[s]
                if len(members) == 1:
                    new_values[members[0]] = nsv
                elif s in reset_slots:
                    share = nsv / len(members)
                    for m in members: new_values[m] = share
                else:
                    osv = slot_values[s]
                    if osv > 0:
                        scale = nsv / osv
                        for m in members: new_values[m] = asset_values[m] * scale
                    else:
                        share = nsv / len(members)
                        for m in members: new_values[m] = share

            current_shares, last_rebalance_date = new_values / current_prices, current_date
            post_weights = new_values / total_val
            post_rec = {"Date": current_date, "Type": "Post-Rebal", "NAV": total_val}
            post_rec.update({f"{t}": f"{post_weights[t]:.2%}" for t in tickers})
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
