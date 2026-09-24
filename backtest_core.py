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

# Band modes (v2.5.2): how a slot's distance from target is measured before it
# is compared with the Down / Up band.
#   "rel"   — relative deviation of the WEIGHT, d = (w − t)/t. The original rule.
#             Size-biased: a large slot needs a much bigger relative move against
#             the rest of the portfolio to reach the same d than a small slot does
#             (a 35% slot must beat the rest by +136% to hit U = 60%, a 10% slot
#             by +71%).
#   "ratio" — the slot's cumulative return RELATIVE TO THE REST of the portfolio
#             since it last stood at target, g = (w/t) / ((1 − w)/(1 − t)), and
#             the band variable is g − 1. A slot that beats the rest by r ends at
#             w = t(1+r)/(1+t·r), so g − 1 == r for every target size: the band
#             means the same thing for a 35% slot and a 10% slot.
BAND_MODE_REL   = "rel"
BAND_MODE_RATIO = "ratio"
BAND_MODES      = (BAND_MODE_REL, BAND_MODE_RATIO)


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


def _slot_label(tok):
    """Slot token -> canonical slot label: the cleaned ticker for a singleton,
    or the cleaned members joined with '+' for a composite "(A, B)". Mirrors
    the labels parse_portfolio() emits in composite['slot_labels'] and is the
    key space of per-slot bands (config `slot_bands`)."""
    tok = str(tok).strip()
    if tok.startswith('(') and tok.endswith(')'):
        return "+".join(clean_ticker(x) for x in tok[1:-1].split(',') if x.strip())
    return clean_ticker(tok)


def slot_labels(tickers_str):
    """Slot labels of a tickers string, in order, independent of the weights
    field (works for rows whose weights are still invalid). Returns [] when
    the parentheses are unbalanced."""
    s = str(tickers_str or '').replace("，", ",").replace("（", "(").replace("）", ")")
    tokens, err = _split_top_level(s, ',')
    if err:
        return []
    return [_slot_label(t) for t in tokens if t.strip()]


def normalize_slot_bands(raw):
    """Coerce a config `slot_bands` value into {label: {"down": int|None, "up": int|None}}.

    Percent integers; None (or a missing key) means "inherit the portfolio
    band" for that side. Malformed entries degrade to inherit instead of
    raising, and entries with neither side set are dropped, so a hand-edited
    or legacy config can never break loading.
    """
    out = {}
    if not isinstance(raw, dict):
        return out
    for label, band in raw.items():
        if not isinstance(band, dict):
            continue
        entry = {}
        for side in ("down", "up"):
            v = band.get(side)
            try:
                entry[side] = None if v is None or v == "" else int(round(float(v)))
            except (TypeError, ValueError):
                entry[side] = None
        if entry["down"] is None and entry["up"] is None:
            continue
        out[str(label).strip()] = entry
    return out


def build_band_thresholds(thr_pct, thr_up_pct, slot_bands, label_to_id):
    """Translate a portfolio's band config (percent integers) into the engine's
    (threshold, threshold_up) arguments (fractions).

    thr_pct / thr_up_pct : portfolio-wide DOWN / UP bands in percent
                           (thr_up_pct None -> same as thr_pct)
    slot_bands           : {slot_label: {"down": pct|None, "up": pct|None}}
    label_to_id          : {slot_label: engine slot id} for the slots in this
                           run (the ticker for singletons, the `groups` value
                           for composites). Labels absent here are ignored.

    Scalars are returned when no per-slot override applies (the legacy scalar
    path); otherwise dicts with a "*" default. threshold_up is None whenever it
    would equal threshold, so a symmetric config takes the pre-2.5.0 code path
    exactly.
    """
    down = float(thr_pct) / 100.0
    up = float(thr_pct if thr_up_pct is None else thr_up_pct) / 100.0
    d_over, u_over = {}, {}
    for label, band in (slot_bands or {}).items():
        sid = label_to_id.get(label)
        if sid is None or not isinstance(band, dict):
            continue
        if band.get("down") is not None:
            d_over[sid] = float(band["down"]) / 100.0
        if band.get("up") is not None:
            u_over[sid] = float(band["up"]) / 100.0
    threshold = {"*": down, **d_over} if d_over else down
    threshold_up = {"*": up, **u_over} if u_over else up
    if threshold_up == threshold:
        threshold_up = None
    return threshold, threshold_up


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


def _check_band_mode(band_mode):
    if band_mode not in BAND_MODES:
        raise ValueError(f"band_mode must be one of {BAND_MODES}, got {band_mode!r}")
    return band_mode


def ratio_deviation(weights, targets):
    """Size-neutral band variable (band_mode="ratio"), per slot.

    g = (w / t) / ((1 − w) / (1 − t)) is the slot's cumulative return relative to
    the REST of the portfolio since its weight last stood at target: if the slot
    beats the rest by r its weight becomes w = t(1+r)/(1+t·r), and that gives
    g − 1 == r whatever t is. The returned g − 1 is compared with +U / −D exactly
    like the relative deviation d of the "rel" mode.

    Edge cases: a slot with target ≥ 1 has no rest to measure against and gets
    0 (never triggers); a slot that has swallowed the whole portfolio while its
    target is < 1 gets +inf (triggers upward for any U); a slot at zero value
    gets −1 (triggers for any D < 1), the same as d in "rel" mode.
    """
    t = targets.astype(float)
    w = weights.astype(float)
    rest_t = 1.0 - t
    rest_w = 1.0 - w
    g = (w / t.replace(0, 1e-9)) / (rest_w / rest_t.replace(0, 1e-9))
    g = g.where(rest_t > 1e-12, 1.0)
    return g - 1.0


def apply_local_rebalance(asset_values, target_weights, threshold, return_resets=False,
                          threshold_up=None, band_mode=BAND_MODE_REL):
    """Local relative-diff rebalance.

    Triggered indices are reset to their target weight; the remainder is
    redistributed proportionally to preserve its internal drift.

    threshold    : DOWN band D — scalar, or Series aligned to asset_values.index
    threshold_up : UP band U, same shapes; None -> U = D (symmetric band).
    band_mode    : "rel" (default) — an index triggers when its signed relative
                   deviation d = (w - target)/target is > U or < -D. With U == D
                   this is exactly the pre-2.5.0 test |d| > D: IEEE-754 abs() and
                   negation are exact and division rounds symmetrically about
                   zero, so the symmetric path is bit-identical to the old abs()
                   code. "ratio" — the same test on the size-neutral variable
                   g − 1 of ratio_deviation() (v2.5.2); the "rel" code path is
                   untouched.

    When return_resets=True, also returns the set of indices that were RESET to
    target (vs only proportionally scaled) — the composite engine uses it to
    decide which slots restore their default equal split. Default False keeps
    the original single-value return so existing callers are unaffected.
    """
    if threshold_up is None:
        threshold_up = threshold
    _check_band_mode(band_mode)
    total_val = asset_values.sum()
    current_vals = asset_values.copy()
    reset_indices = []
    safe_targets = target_weights.replace(0, 1e-9)
    for _ in range(10):
        current_weights = current_vals / total_val
        if band_mode == BAND_MODE_RATIO:
            signed = ratio_deviation(current_weights, target_weights)
        else:
            signed = (current_weights - target_weights) / safe_targets
        breach = (signed > threshold_up) | (signed < -threshold)
        to_trigger = breach & (~current_vals.index.isin(reset_indices))
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


def _empty_stats():
    return {"sold_total": 0.0, "nav_mean": 0.0, "years": 0.0, "turnover_yr": 0.0,
            "slot_ids": [], "slot_members": {}, "slot_target": {},
            "weight_min": {}, "weight_max": {}}


def run_detailed_backtest(strategy_name, price_df, target_weights, initial_cap,
                          threshold, groups=None, threshold_up=None, return_stats=False,
                          band_mode=BAND_MODE_REL):
    """Backtest one portfolio under one rebalance strategy.

    groups: optional element-ticker -> slot-id mapping (dict[str, str]). Elements
    sharing a slot-id form a COMPOSITE: rebalance decisions are made on the slot's
    AGGREGATE weight (elements never trigger individually), but each element keeps
    its own price series so NAV = sum over elements. On a reset the slot value is
    restored to an EQUAL split among its elements ("再平衡时恢复默认份额"); between
    resets the elements drift and the slot moves "as one block".

    groups=None  ==  every column is its own slot  ==  original behaviour,
    bit-identical (singletons short-circuit the unfold; see below).

    threshold: DOWN band D — float, or dict {slot_id: D, "*": default} for
    PER-SLOT bands (slot_id = the ticker for singleton slots / the groups value
    for composites). "*" supplies the default for unlisted slots and is required
    if any slot is missing. A scalar is broadcast — decisions are bit-identical
    to the original code. (Keeps its pre-2.5.0 name for compatibility.)

    threshold_up: UP band U, same shapes as threshold; None -> U = D. The RelDiff
    strategies (Full / Mixed / Local) trigger a slot when its signed relative
    deviation d = (w - target)/target is > U or < -D. Asymmetric RelDiff keeps its
    own major/minor multiplier rule on `threshold` alone and ignores threshold_up;
    Periodic and Buy & Hold ignore both.

    band_mode (v2.5.2): "rel" (default) measures d as above — the original,
    bit-identical rule. "ratio" replaces d by the size-neutral g − 1 of
    ratio_deviation(): the slot's cumulative return relative to the rest of the
    portfolio since its last reset, so U / D mean the same for every slot size
    (U = 100% = "the slot doubled against the rest", D = 50% = "it halved").
    Per-slot bands are read in the same mode. Only the RelDiff strategies use it;
    Asymmetric RelDiff, Periodic and Buy & Hold ignore it.

    return_stats: when True a 4th value is returned, a dict with turnover and
    drift statistics:
      sold_total  : Σ over rebalances of Σ_elements max(0, pre − post value)
      nav_mean    : mean NAV over the processed bars
      years       : span of processed bars in years (365.25-day)
      turnover_yr : sold_total / nav_mean / years  (0 when undefined)
      slot_ids, slot_members, slot_target : the slot view used for decisions
      weight_min / weight_max : per-slot aggregate weight extremes over the
        Init / Hold / Post-Rebal states — i.e. the weights actually carried
        from one bar to the next (Pre-Rebal breach snapshots are excluded).
    """
    _check_band_mode(band_mode)
    tickers = price_df.columns
    if price_df.empty:
        return (pd.DataFrame(), 0, {}, _empty_stats()) if return_stats else (pd.DataFrame(), 0, {})
    start_prices = price_df.iloc[0]
    if start_prices.isna().any():
        start_prices = price_df.bfill().iloc[0]
        if start_prices.isna().any():
            return (pd.DataFrame(), 0, {}, _empty_stats()) if return_stats else (pd.DataFrame(), 0, {})

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

    # Normalize bands: scalar stays scalar (broadcasts bit-identically);
    # a dict becomes a Series aligned to slot_ids ("*" = default band).
    def _band(v, name):
        if isinstance(v, dict):
            missing = [s for s in slot_ids if s not in v]
            if missing and "*" not in v:
                raise ValueError(f"{name} dict missing slots {missing} and no '*' default")
            return pd.Series({s: float(v.get(s, v.get("*"))) for s in slot_ids})
        return v
    threshold = _band(threshold, "threshold")
    threshold_up = threshold if threshold_up is None else _band(threshold_up, "threshold_up")

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

    # ---- Stats (read-only observers; they never feed back into the path) ----
    sold_total = 0.0
    nav_sum, nav_n = 0.0, 0
    first_date = last_date = None
    w_ext = {"min": None, "max": None}

    def _track(slot_w):
        w_ext["min"] = slot_w if w_ext["min"] is None else np.minimum(w_ext["min"], slot_w)
        w_ext["max"] = slot_w if w_ext["max"] is None else np.maximum(w_ext["max"], slot_w)

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

        nav_sum += float(total_val); nav_n += 1
        if first_date is None: first_date = current_date
        last_date = current_date

        if i == 0:
            rec = {"Date": current_date, "Type": "Init", "NAV": total_val}
            rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec)
            _track(asset_values.groupby(slot_of).sum().reindex(slot_ids) / total_val)
            continue

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
            # Signed relative deviation against an UP / DOWN band. With U == D
            # this is exactly the old |d| > thr test (abs/negation are exact in
            # IEEE-754 and division rounds symmetrically), so symmetric configs
            # stay bit-identical; the elementwise form also carries per-slot bands.
            # band_mode="ratio" swaps d for the size-neutral g − 1 (v2.5.2).
            if band_mode == BAND_MODE_RATIO:
                signed = ratio_deviation(slot_weights, slot_targets)
            else:
                signed = (slot_weights - slot_targets) / slot_targets.replace(0, 1e-9)
            breach = (signed > threshold_up) | (signed < -threshold)
            if breach.any():
                if strategy_name == STRAT_RD_FULL:
                    new_slot_values, reset_slots, do_rebalance = total_val * slot_targets, set(slot_ids), True
                elif strategy_name == STRAT_RD_MIXED:
                    if ((slot_targets >= 0.1) & breach).any():
                        new_slot_values, reset_slots = total_val * slot_targets, set(slot_ids)
                    else:
                        new_slot_values, reset_slots = apply_local_rebalance(
                            slot_values, slot_targets, threshold, return_resets=True,
                            threshold_up=threshold_up, band_mode=band_mode)
                    do_rebalance = True
                elif strategy_name == STRAT_RD_LOCAL:
                    new_slot_values, reset_slots = apply_local_rebalance(
                        slot_values, slot_targets, threshold, return_resets=True,
                        threshold_up=threshold_up, band_mode=band_mode)
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

            # Turnover = money SOLD this bar (element level; buys are the mirror).
            sold_total += float((asset_values - new_values).clip(lower=0).sum())
            _track(new_slot_values / total_val)

            current_shares, last_rebalance_date = new_values / current_prices, current_date
            post_weights = new_values / total_val
            post_rec = {"Date": current_date, "Type": "Post-Rebal", "NAV": total_val}
            post_rec.update({f"{t}": f"{post_weights[t]:.2%}" for t in tickers})
            history.append(post_rec)
        else:
            rec = {"Date": current_date, "Type": "Hold", "NAV": total_val}
            rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec)
            _track(slot_weights)

    total_pnl = cumulative_pnl.sum()
    pct_pnl = cumulative_pnl / total_pnl if total_pnl != 0 else cumulative_pnl * 0
    pnl_rec = {"Date": "Overall", "Type": "PnL Contrib%", "NAV": float(total_pnl)}
    pnl_rec.update({f"{t}": f"{pct_pnl[t]:.2%}" for t in tickers})

    if not return_stats:
        return pd.DataFrame(history), rebalance_count, pnl_rec

    nav_mean = nav_sum / nav_n if nav_n else 0.0
    years = (last_date - first_date).days / 365.25 if (first_date is not None and last_date is not None) else 0.0
    stats = {
        "sold_total": sold_total,
        "nav_mean": nav_mean,
        "years": years,
        "turnover_yr": (sold_total / nav_mean / years) if (nav_mean > 0 and years > 0) else 0.0,
        "slot_ids": list(slot_ids),
        "slot_members": {s: list(slot_members[s]) for s in slot_ids},
        "slot_target": {s: float(slot_targets[s]) for s in slot_ids},
        "weight_min": {s: float(w_ext["min"][s]) for s in slot_ids} if w_ext["min"] is not None else {},
        "weight_max": {s: float(w_ext["max"][s]) for s in slot_ids} if w_ext["max"] is not None else {},
    }
    return pd.DataFrame(history), rebalance_count, pnl_rec, stats


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
