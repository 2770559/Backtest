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
STRAT_ASYM_LOCAL = "Asymmetric RelDiff Local"  # Asymmetric trigger; minor breach handled locally
STRAT_ASYM_LOCAL_EQ = "Asymmetric RelDiff Local (Equal)"    # minor breach: trade vs majors, equal dollar split
STRAT_ASYM_LOCAL_PROP = "Asymmetric RelDiff Local (Prop)"   # minor breach: trade vs majors, pro-rata to current value

# Distribution mode for the "Asymmetric RelDiff Local" family: how a triggered
# minor slot's correction is split across the major (>= 6%) counterparties.
ASYM_LOCAL_MODES = {
    STRAT_ASYM_LOCAL: "rank",           # most-deviated major first, capped at its target
    STRAT_ASYM_LOCAL_EQ: "equal",       # equal dollar split across ALL majors, no target cap
    STRAT_ASYM_LOCAL_PROP: "prop",      # split across ALL majors pro-rata to current value, no cap
}

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
    if t in mapping:
        return mapping[t]
    # Normalize Hong Kong codes to Yahoo's canonical 4-digit form:
    # 00700.HK -> 0700.HK, 09992.HK -> 9992.HK. Yahoo 404s on the 5-digit,
    # leading-zero-padded codes many brokers/data feeds use. int() strips
    # leading zeros; :04d re-pads to a minimum of 4 digits, so genuine
    # 5-digit codes (e.g. 80737.HK) are left untouched.
    if t.endswith(".HK"):
        code = t[:-3]
        if code.isdigit():
            return f"{int(code):04d}.HK"
    return t


def _split_top_level(s, sep=','):
    """Split `s` on `sep`, ignoring separators inside parentheses.

    Returns (tokens, err); err is None on success else a human-readable string.
    Nested parens are rejected. Used for the tickers field so a composite group
    "(DBMF, KMLM)" stays one token.
    """
    tokens, buf, depth = [], [], 0
    for ch in s:
        if ch == '(':
            depth += 1
            if depth > 1:
                return None, "nested parentheses are not allowed"
            buf.append(ch)
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return None, "unbalanced ')' in tickers"
            buf.append(ch)
        elif ch == sep and depth == 0:
            tokens.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    if depth != 0:
        return None, "unbalanced '(' in tickers"
    tokens.append(''.join(buf))
    return tokens, None


def parse_portfolio(port):
    """Parse one portfolio config dict into (tickers, weights, errors, composite).

    tickers   : list[str]    FLAT element tickers (price-fetch order; cleaned)
    weights   : list[float]  per-ELEMENT target weight (equal split inside a slot)
    errors    : list[str]    human-readable; [] means valid
    composite : dict | None  slot metadata; None when no (...) group is present

    A weight token maps to one SLOT. A parenthesised group "(A, B)" is one slot
    whose target weight is split equally across its elements ("默认平均分配份额").
    Backward-compatible: a portfolio with no parentheses yields exactly today's
    tickers/weights/errors and composite=None. Both fields accept fullwidth commas.
    """
    errors = []
    t_str = str(port.get('tickers', '')).replace("，", ",").replace("（", "(").replace("）", ")")
    w_str = str(port.get('weights', '')).replace("，", ",")

    t_tokens, paren_err = _split_top_level(t_str, ',')
    if paren_err:
        return [], [], [paren_err], None

    # Build slots: each token is a single ticker or a parenthesised composite group.
    slot_labels, slot_members = [], []
    has_composite = False
    for tok in t_tokens:
        tok = tok.strip()
        if not tok:
            continue  # trailing/empty token ignored (matches today's behaviour)
        if tok.startswith('(') and tok.endswith(')'):
            has_composite = True
            members = [clean_ticker(x) for x in tok[1:-1].split(',') if x.strip()]
            if len(members) == 0:
                errors.append(f"empty composite '{tok}'")
                continue
            if len(members) == 1:
                errors.append(f"composite '{tok}' needs >= 2 elements")
            slot_labels.append("+".join(members))
            slot_members.append(members)
        else:
            ck = clean_ticker(tok)
            slot_labels.append(ck)
            slot_members.append([ck])

    w_raw = [x.strip() for x in w_str.split(',') if x.strip()]
    flat_now = [t for grp in slot_members for t in grp]

    # Count check is SLOT-level (one weight per slot, not per element).
    if len(slot_labels) != len(w_raw):
        unit = "slots" if has_composite else "tickers"   # preserve legacy wording
        errors.append(f"{len(slot_labels)} {unit} vs {len(w_raw)} weights")
        return flat_now, [], errors, None

    try:
        slot_targets = [float(w) for w in w_raw]
    except ValueError:
        errors.append("invalid weight format")
        return flat_now, [], errors, None

    total_w = sum(slot_targets)
    if abs(total_w - 1.0) > 0.01:
        errors.append(f"weights sum = {total_w:.2f}, should be 1.0")

    # Duplicate detection across ALL elements (flat, cross-slot).
    dupes = sorted({t for t in flat_now if flat_now.count(t) > 1})
    if dupes:
        errors.append("duplicate tickers: " + ", ".join(dupes))

    # Expand slots -> per-element flat list + equal-split per-element weights.
    elem_tickers, elem_weights, element_slot = [], [], []
    for si, (members, st_w) in enumerate(zip(slot_members, slot_targets)):
        per = st_w / len(members)
        for m in members:
            elem_tickers.append(m)
            elem_weights.append(per)
            element_slot.append(si)

    composite = None
    if has_composite:
        composite = {
            "has_composite": True,
            "slot_labels": slot_labels,
            "slot_targets": slot_targets,
            "slot_members": slot_members,
            "element_slot": element_slot,
        }

    return elem_tickers, elem_weights, errors, composite


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


def _transfer_minor_vs_majors(work, target_vals, majors, s, need, mode):
    """Trade a triggered MINOR slot back toward its target against the MAJOR slots.

    Mutates `work` (slot-id -> $ value Series) in place, moving up to abs(need)
    dollars between minor slot `s` and the majors; every transfer conserves NAV
    and minors never trade with other minors. Returns dollars actually moved.

    mode="rank"  : counterparties ranked most-deviated first, each move capped so
                   the major never crosses its OWN target; if the eligible majors
                   can't fund/absorb it all, the minor is only PARTIALLY corrected.
    mode="equal" : split across ALL majors in equal dollar shares, no target cap
                   (majors may cross their own target). On the pull side a major
                   can't go below zero; any shortfall cascades to the others.
    mode="prop"  : split across ALL majors pro-rata to their CURRENT values, no
                   target cap. Pro-rata pulls can never turn a major negative.
    """
    moved = 0.0
    if need > 0:                                      # minor under target: pull from majors
        if mode == "rank":                            # over-weight majors only, most-deviated first
            pool = [j for j in majors if work[j] > target_vals[j]]
            pool.sort(key=lambda j: (work[j] - target_vals[j]) / target_vals[j], reverse=True)
            for j in pool:
                if need <= 1e-9: break
                take = min(need, work[j] - target_vals[j])   # cap: don't cross target
                work[j] -= take; work[s] += take
                need -= take; moved += take
        elif mode == "equal":
            pool = [j for j in majors if work[j] > 0]
            while need > 1e-9 and pool:
                share = need / len(pool)
                survivors = []
                for j in pool:
                    take = min(share, work[j])        # a major can't fund past zero
                    work[j] -= take; work[s] += take
                    need -= take; moved += take
                    if work[j] > 1e-9: survivors.append(j)
                pool = survivors
        else:                                         # prop
            tot = sum(work[j] for j in majors if work[j] > 0)
            if tot > 0:
                amt = min(need, tot)
                for j in majors:
                    if work[j] <= 0: continue
                    take = amt * (work[j] / tot)
                    work[j] -= take; work[s] += take
                    moved += take
    elif need < 0:                                    # minor over target: push excess to majors
        excess = -need
        if mode == "rank":                            # under-weight majors only, most-deviated first
            pool = [j for j in majors if work[j] < target_vals[j]]
            pool.sort(key=lambda j: (work[j] - target_vals[j]) / target_vals[j])
            for j in pool:
                if excess <= 1e-9: break
                give = min(excess, target_vals[j] - work[j])  # cap: don't cross target
                work[j] += give; work[s] -= give
                excess -= give; moved += give
        elif mode == "equal":
            if majors:
                share = excess / len(majors)
                for j in majors:
                    work[j] += share; work[s] -= share
                    moved += share
        else:                                         # prop
            tot = sum(work[j] for j in majors if work[j] > 0)
            if tot > 0:
                for j in majors:
                    if work[j] <= 0: continue
                    give = excess * (work[j] / tot)
                    work[j] += give; work[s] -= give
                    moved += give
            elif majors:                              # degenerate: all majors at $0 -> equal split
                share = excess / len(majors)
                for j in majors:
                    work[j] += share; work[s] -= share
                    moved += share
    return moved


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

        elif strategy_name in ASYM_LOCAL_MODES:
            # Same asymmetric trigger geometry as STRAT_ASYM, but a MINOR-only
            # breach is corrected LOCALLY instead of forcing a global reset. A
            # major breach still resets globally (bit-identical to STRAT_ASYM).
            #
            # For a minor-only breach, each TRIGGERED minor slot is nudged back
            # toward its target by trading ONLY against the MAJOR (>= 6%) slots;
            # other minor slots are left untouched unless they triggered too.
            # HOW the correction is split across the majors is the only difference
            # between the three variants — see _transfer_minor_vs_majors:
            #   rank (STRAT_ASYM_LOCAL):        most-deviated major first, each move
            #        capped at that major's own target ("恢复到基础比例优先");
            #        insufficient majors => the minor is only PARTIALLY corrected.
            #   equal (STRAT_ASYM_LOCAL_EQ):    equal dollar split across all majors.
            #   prop (STRAT_ASYM_LOCAL_PROP):   pro-rata to majors' current values.
            # Transfers read/mutate the live `work` state so multiple minors in one
            # bar compound correctly. Triggered minor slots restore their composite
            # default split (reset); major counterparties scale, keeping drift.
            diff_ratio = (slot_weights - slot_targets) / slot_targets.replace(0, 1e-9)
            mask_major = slot_targets >= 0.06
            mask_minor = slot_targets < 0.06

            trigger_major = mask_major & (np.abs(diff_ratio) > threshold)
            trigger_minor_up = mask_minor & (diff_ratio > threshold * 2.5)
            trigger_minor_down = mask_minor & (diff_ratio < -threshold * 1.25)

            if trigger_major.any():
                new_slot_values, reset_slots, do_rebalance = total_val * slot_targets, set(slot_ids), True
            elif trigger_minor_up.any() or trigger_minor_down.any():
                work = slot_values.copy()
                target_vals = total_val * slot_targets
                majors = [j for j in slot_ids if mask_major[j]]       # counterparty pool: majors only
                mode = ASYM_LOCAL_MODES[strategy_name]
                for s in slot_ids:
                    if not (trigger_minor_up[s] or trigger_minor_down[s]):
                        continue
                    need = target_vals[s] - work[s]                   # >0 补足 / <0 削减
                    moved = _transfer_minor_vs_majors(work, target_vals, majors, s, need, mode)
                    if moved > 0:
                        reset_slots.add(s)
                if reset_slots:
                    new_slot_values, do_rebalance = work, True

        elif "RelDiff" in strategy_name:
            rel_diffs = np.abs(slot_weights - slot_targets) / slot_targets.replace(0, 1e-9)
            if rel_diffs.max() > threshold:
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


def sample_monthly(final_data):
    """Month-end sampling used by the app for windows >= 90 days.

    Keeps the actual first row, then the last available row of each calendar
    month (stamped at the calendar month-end label, the historical convention)
    — EXCEPT the final bar: a partial last month keeps its real last data date
    instead of a future month-end label. Stamping the last bar in the future
    inflated the day-count in CAGR (up to ~29 days; material on short windows)
    and plotted a not-yet-reached date on the charts.
    """
    first_row = final_data.iloc[[0]]
    monthly_rows = final_data.resample('ME').last()
    if len(monthly_rows) and monthly_rows.index[-1] > final_data.index[-1]:
        monthly_rows = monthly_rows.rename(index={monthly_rows.index[-1]: final_data.index[-1]})
    out = pd.concat([first_row, monthly_rows]).sort_index()
    return out[~out.index.duplicated(keep='first')]


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
