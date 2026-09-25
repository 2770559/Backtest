"""Live-account desk (v2.6.0): pure functions that turn the app's backtest of the
live rule (the "shadow" portfolio) into share-level orders for the real account.

No Streamlit here; the page lives in live_page.py and the tests in
tests/test_live_core.py.

Concepts
  shadow   the live rule's backtest (same data pipeline and engine as the
           Backtest page) run to the latest price. It alone decides WHEN to
           rebalance and WHAT each element should weigh.
  element  a backtest ticker (QQQM ... ETH-USD, MSTR). A slot holds one element
           or, for a composite, several.
  mapping  shadow element -> live tickers, primary first. The live account may
           hold a different instrument for an element (BMNR for ETH-USD): its
           target weight is the element's shadow weight and it never triggers
           anything itself.

Every operation is "trade the account to target weights x target total"; the
cases differ only in the target weights and the trade directions allowed:
  flow  only the cash flow moves: a deposit buys the elements that sit below
        their shadow weight (in proportion to the shortfall), a withdrawal sells
        the ones above it (in proportion to the excess). When the account
        already matches the shadow this is plain pro-rata.
  sync  every element is traded to its shadow weight x (value + cash flow).
A shadow rebalance at a month-end bar is a sync: after it the shadow's current
weights already are the post-reset weights.
"""
import io
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from backtest_core import (
    STRAT_RD_FULL, STRAT_RD_LOCAL, STRAT_RD_MIXED, BAND_MODE_RATIO, BAND_MODE_REL,
    run_detailed_backtest, band_trigger_weights,
)

SYMBOL_ALIASES = {"BRK B": "BRK-B", "BRK.B": "BRK-B", "BRK/B": "BRK-B", "BRKB": "BRK-B"}
# Primary first: buys go to the primary, sells come out of whatever is held.
DEFAULT_MAPPING = {"ETH-USD": ["BMNR", "ETHW"], "MSTR": ["MSTR", "IBIT", "FBTC", "BTC"]}
RELDIFF = (STRAT_RD_FULL, STRAT_RD_MIXED, STRAT_RD_LOCAL)
CRYPTO_ELEMENTS = {"ETH-USD", "BTC-USD", "MSTR", "BMNR", "ETHW", "IBIT", "FBTC", "BTC"}


def norm_symbol(sym):
    """Upper-case, strip, collapse spaces and apply the broker aliases (BRK B -> BRK-B)."""
    s = re.sub(r"\s+", " ", str(sym or "").strip().upper())
    return SYMBOL_ALIASES.get(s, s)


def mapping_for(elements, mapping=None):
    """Complete element -> [live tickers] map for `elements`: entries of `mapping`
    (default DEFAULT_MAPPING) where given, the element itself otherwise."""
    mapping = DEFAULT_MAPPING if mapping is None else mapping
    out = {}
    for e in elements:
        lst = [norm_symbol(x) for x in (mapping.get(e) or []) if str(x).strip()]
        out[e] = lst or [norm_symbol(e)]
    return out


def symbol_to_element(emap):
    """Invert an element -> [live tickers] map; a live ticker belongs to one element."""
    inv = {}
    for e, syms in emap.items():
        for s in syms:
            inv.setdefault(s, e)
    return inv


# --------------------------------------------------------------------------- #
# Shadow state
# --------------------------------------------------------------------------- #
def month_complete(day):
    """True when `day` is the last business day of its month (the next business
    day falls in the next month). Exchange holidays are not modelled: when the
    true last session precedes a holiday, the month counts as complete one
    business day later."""
    d = pd.Timestamp(day)
    return (d + pd.offsets.BDay(1)).month != d.month


def next_month_end(as_of, complete):
    """Last business day of the month whose close is the next month-end check."""
    d = pd.Timestamp(as_of)
    me = d + pd.offsets.BMonthEnd(0)
    return me + pd.offsets.BMonthEnd(1) if (complete and me.normalize() == d.normalize()) else me


def last_reset_date(events, slot):
    """Date of the shadow's last reset that included `slot` (a global reset, or a
    local one it triggered); None when it never reset."""
    for e in reversed(events):
        if e["scope"] == "global" or slot in e["trigger"]:
            return pd.Timestamp(e["date"])
    return None


def ibkr_commission(orders, per_share=0.005, minimum=1.0, max_pct=0.01):
    """IBKR Pro fixed-rate estimate: $0.005 per share, $1 minimum, capped at 1% of
    the trade value (exchange / regulatory fees not included)."""
    if orders is None or len(orders) == 0:
        return 0.0
    return float(sum(min(max(minimum, per_share * q), max_pct * a) for q, a in zip(orders["shares"], orders["amount"])))


def _band_of(band, sid):
    if band is None:
        return None
    return float(band.get(sid, band.get("*"))) if isinstance(band, dict) else float(band)


def _g(w, t):
    """Leg-vs-rest multiple of a slot at weight w with target t."""
    if w <= 0:
        return 0.0
    if w >= 1:
        return math.inf
    return (w / t) / ((1 - w) / (1 - t))


def shadow_state(strategy, price_df, w_series, thr_dn, thr_up=None, groups=None, band_mode=BAND_MODE_REL, now=None):
    """The shadow portfolio as of the last row of `price_df` (the engine's frame:
    month-end bars plus the real last date).

    Month-end semantics (the live protocol trades only on month-end bars):
      * last row is a month's last business day -> it IS a month-end bar; the
        shadow's current state is the engine's state after that bar, including
        any rebalance on it.
      * otherwise the last row is a partial month: the state is the one carried
        out of the previous month-end bar, drifted to the latest prices; a
        breach on the partial bar is only reported as a preview.
      * `now` (New York local time, optional): on a month's last business day
        the bar only counts as a month-end bar once that session has closed
        (16:30 or later) or the date has passed — intraday prints are not final.

    Returns a dict:
      as_of, month_complete, ref_date (month-end bar the state comes from),
      elem_weights (Series), slot_table (DataFrame), ref_event (rebalance on
      ref_date or None), preview_event (breach on a partial last bar or None),
      events (all rebalances), nav (engine NAV series)
    """
    tks = list(w_series.index)
    kw = dict(groups=groups, threshold_up=thr_up, return_stats=True, band_mode=band_mode)
    res, cnt, _, st = run_detailed_backtest(strategy, price_df[tks], w_series, 10000, thr_dn, **kw)
    if res.empty:
        raise ValueError("the shadow backtest produced no rows")
    as_of = pd.Timestamp(price_df.index[-1])
    closed = now is None or pd.Timestamp(now).date() > as_of.date() or \
        (pd.Timestamp(now).hour * 60 + pd.Timestamp(now).minute) >= 16 * 60 + 30
    complete = (month_complete(as_of) and closed) or len(price_df) < 2
    preview = None
    if complete:
        ref_date, vals = as_of, pd.Series(st["last_values"])
        events = st["rebal_events"]
    else:
        res0, cnt0, _, st0 = run_detailed_backtest(strategy, price_df[tks].iloc[:-1], w_series, 10000, thr_dn, **kw)
        ref_date = pd.Timestamp(price_df.index[-2])
        p_ref, p_now = price_df[tks].ffill().iloc[-2], price_df[tks].ffill().iloc[-1]
        vals = pd.Series(st0["last_values"]) * (p_now / p_ref)
        events = st0["rebal_events"]
        last_ev = st["rebal_events"][-1] if st["rebal_events"] else None
        if last_ev is not None and pd.Timestamp(last_ev["date"]) == as_of:
            preview = last_ev
    ref_event = next((e for e in events if pd.Timestamp(e["date"]) == ref_date), None)
    elem_w = (vals / vals.sum()).reindex(tks)

    rows = []
    for sid in st["slot_ids"]:
        members = st["slot_members"][sid]
        t = st["slot_target"][sid]
        w = float(elem_w[members].sum())
        row = {"slot": sid, "label": "+".join(members), "members": members, "target": t, "weight": w,
               "trig_low": None, "trig_high": None, "move_to_low": None, "move_to_high": None}
        if strategy in RELDIFF:
            d = _band_of(thr_dn, sid)
            u = _band_of(thr_up if thr_up is not None else thr_dn, sid)
            lo, hi = band_trigger_weights(t, d, u, band_mode)
            row.update(trig_low=lo, trig_high=hi)
            gn = _g(w, t)
            if lo is not None and gn > 0:
                row["move_to_low"] = _g(lo, t) / gn - 1
            if hi is not None and gn > 0 and math.isfinite(gn):
                row["move_to_high"] = _g(hi, t) / gn - 1
        rows.append(row)
    return {"as_of": as_of, "month_complete": complete, "ref_date": ref_date, "elem_weights": elem_w,
            "slot_table": pd.DataFrame(rows), "ref_event": ref_event, "preview_event": preview,
            "events": events, "nav": res.drop_duplicates(subset="Date", keep="last").set_index("Date")["NAV"]}


# --------------------------------------------------------------------------- #
# Holdings
# --------------------------------------------------------------------------- #
def holdings_by_element(holdings, prices, emap):
    """holdings {live symbol: shares}, prices {live symbol: price}, emap element ->
    [live tickers]. Returns (elem_values Series over the elements, detail
    DataFrame per held symbol, unmapped symbols, symbols without a price)."""
    inv = symbol_to_element(emap)
    vals = pd.Series(0.0, index=list(emap))
    rows, unmapped, no_price = [], [], []
    for sym, sh in holdings.items():
        s = norm_symbol(sym)
        if s not in inv:
            unmapped.append(s)
            continue
        p = prices.get(s)
        if p is None or not np.isfinite(p) or p <= 0:
            no_price.append(s)
            continue
        v = float(sh) * float(p)
        vals[inv[s]] += v
        rows.append({"symbol": s, "element": inv[s], "shares": float(sh), "price": float(p), "value": v})
    return vals, pd.DataFrame(rows, columns=["symbol", "element", "shares", "price", "value"]), unmapped, no_price


def slot_weights(elem_values, slot_table):
    """Aggregate element values to the shadow's slots -> weight Series by slot id."""
    tot = float(elem_values.sum())
    return pd.Series({r["slot"]: (float(elem_values[r["members"]].sum()) / tot if tot > 0 else 0.0)
                      for _, r in slot_table.iterrows()})


# --------------------------------------------------------------------------- #
# Order planning
# --------------------------------------------------------------------------- #
def element_trades(elem_values, target_weights, flow=0.0, mode="sync", min_trade=0.0):
    """Dollar trade per element (+ buy / − sell), before share rounding.

    mode "flow": only the cash flow moves (deposit -> buy-only toward the
    shadow, withdrawal -> sell-only); "sync": every element to target x (V + F).
    Trades smaller than `min_trade` are dropped; in flow mode their money goes
    to the remaining elements (at least one always trades), in sync mode the
    buy or sell side is scaled so the orders still net to exactly `flow`."""
    v = elem_values.astype(float)
    w = target_weights.reindex(v.index).fillna(0.0).astype(float)
    w = w / w.sum()
    total = float(v.sum()) + float(flow)
    if total < 0:
        raise ValueError("withdrawal larger than the portfolio")
    desired = w * total
    if mode == "flow":
        if flow == 0:
            return pd.Series(0.0, index=v.index)
        gap = (desired - v).clip(lower=0) if flow > 0 else (v - desired).clip(lower=0)
        if gap.sum() <= 0:
            gap = w.copy() if flow > 0 else v.copy()
        keep = gap > 0
        while True:
            x = gap.where(keep, 0.0) / gap.where(keep, 0.0).sum() * abs(flow)
            small = keep & (x < min_trade)
            if not small.any() or small.sum() == keep.sum():
                if small.sum() == keep.sum() and keep.sum() > 1:     # everything small: one order carries the flow
                    top = gap.where(keep, -1).idxmax()
                    x = pd.Series(0.0, index=v.index); x[top] = abs(flow)
                break
            keep &= ~small
        return x if flow > 0 else -x
    if mode != "sync":
        raise ValueError(f"unknown mode {mode!r}")
    x = desired - v
    x = x.where(x.abs() >= min_trade, 0.0)
    buys, sells = x.clip(lower=0).sum(), -x.clip(upper=0).sum()
    if buys - sells > flow + 1e-9 and buys > 0:              # dropped sells: fund fewer buys
        x = x.where(x <= 0, x * max(0.0, sells + flow) / buys)
    elif buys - sells < flow - 1e-9 and sells > 0:           # dropped buys: raise less
        x = x.where(x >= 0, x * (buys - flow) / sells)
    return x


def plan_orders(holdings, prices, emap, target_weights, flow=0.0, mode="sync", whole_shares=True,
                min_trade=200.0, frac_digits=4):
    """Share-level orders for the live account.

    holdings {symbol: shares}, prices {symbol: price} (every symbol that may be
    traded needs one, including an element's primary ticker not yet held),
    emap element -> [live tickers], target_weights element -> shadow weight.

    Buys of an element go to its primary ticker; sells come out of its held
    tickers in proportion to their value. Whole shares: sells round UP (a
    withdrawal is always fully funded, a sale never exceeds what is held),
    buys round DOWN, and the cash left over buys one share at a time of the
    element furthest below its target while a share is affordable.

    Returns {"orders": DataFrame[symbol, element, side, shares, price, amount],
             "elements": DataFrame per element (before / after / target),
             "cash_left": cash not spent (deposit leftovers or excess proceeds),
             "flow": flow, "warnings": [...]}"""
    elem_vals, detail, unmapped, no_price = holdings_by_element(holdings, prices, emap)
    warnings = []
    if unmapped:
        warnings.append("Not part of the shadow (ignored): " + ", ".join(sorted(set(unmapped))))
    if no_price:
        raise ValueError("no price for held symbol(s): " + ", ".join(sorted(set(no_price))))
    x = element_trades(elem_vals, target_weights, flow, mode, min_trade)
    inv = symbol_to_element(emap)
    orders = {}                                               # symbol -> signed shares

    def price_of(sym):
        p = prices.get(sym)
        if p is None or not np.isfinite(p) or p <= 0:
            raise ValueError(f"no price for {sym}")
        return float(p)

    rnd_sell = (lambda q: math.ceil(q - 1e-9)) if whole_shares else (lambda q: math.ceil(q * 10 ** frac_digits - 1e-9) / 10 ** frac_digits)
    rnd_buy = (lambda q: math.floor(q + 1e-9)) if whole_shares else (lambda q: math.floor(q * 10 ** frac_digits + 1e-9) / 10 ** frac_digits)
    for e, amt in x.items():
        if amt < 0:
            rows = detail[detail["element"] == e]
            tot = rows["value"].sum()
            for _, r in rows.iterrows():
                want = -amt * r["value"] / tot / r["price"]
                q = min(rnd_sell(want), r["shares"])
                if q > 0:
                    orders[r["symbol"]] = orders.get(r["symbol"], 0.0) - q
        elif amt > 0:
            sym = emap[e][0]
            q = rnd_buy(amt / price_of(sym))
            if q > 0:
                orders[sym] = orders.get(sym, 0.0) + q
    # cash bookkeeping, then spend leftovers one share at a time
    proceeds = sum(-q * price_of(s) for s, q in orders.items() if q < 0)
    cost = sum(q * price_of(s) for s, q in orders.items() if q > 0)
    cash = float(flow) + proceeds - cost
    after = elem_vals.copy()
    for s, q in orders.items():
        after[inv[s]] += q * price_of(s)
    total_after = float(elem_vals.sum()) + float(flow)
    desired = target_weights.reindex(elem_vals.index).fillna(0.0) / target_weights.reindex(elem_vals.index).fillna(0.0).sum() * total_after
    may_buy = set(x.index[x > 0]) if mode == "flow" else set(elem_vals.index)
    if whole_shares and cash > 0 and (mode == "sync" or flow > 0):
        while True:
            short = (desired - after)[[e for e in after.index if e in may_buy]]
            short = short[short > 0].sort_values(ascending=False)
            pick = next((e for e in short.index if price_of(emap[e][0]) <= cash + 1e-9), None)
            if pick is None:
                break
            sym = emap[pick][0]
            orders[sym] = orders.get(sym, 0.0) + 1
            after[pick] += price_of(sym)
            cash -= price_of(sym)
    ords = []
    for s, q in orders.items():
        if abs(q) < 1e-12:
            continue
        p = price_of(s)
        ords.append({"symbol": s, "element": inv.get(s, s), "side": "BUY" if q > 0 else "SELL",
                     "shares": abs(q), "price": p, "amount": abs(q) * p})
    ords = pd.DataFrame(ords, columns=["symbol", "element", "side", "shares", "price", "amount"])
    if len(ords):
        ords = ords.sort_values(["side", "amount"], ascending=[False, False]).reset_index(drop=True)
    tb, ta = float(elem_vals.sum()), float(after.sum())
    tw = target_weights.reindex(elem_vals.index).fillna(0.0)
    tw = tw / tw.sum()
    elements = pd.DataFrame({"element": elem_vals.index, "value_before": elem_vals.values, "value_after": after.values,
                             "weight_before": (elem_vals / tb).values if tb > 0 else 0.0,
                             "weight_after": (after / ta).values if ta > 0 else 0.0, "target": tw.values})
    elements["dev_after"] = elements["weight_after"] - elements["target"]
    if flow < 0 and cash < -1e-6:
        warnings.append(f"Proceeds fall short of the withdrawal by ${-cash:,.2f}")
    return {"orders": ords, "elements": elements, "cash_left": cash, "flow": float(flow), "warnings": warnings}


def crypto_slot_ids(slot_table):
    """Slots holding a crypto element (their relative tolerance applies)."""
    return [r["slot"] for _, r in slot_table.iterrows() if set(r["members"]) & CRYPTO_ELEMENTS]


# --------------------------------------------------------------------------- #
# Sync advice (default tolerances agreed 2026-09-24)
# --------------------------------------------------------------------------- #
def sync_advice(state, live_elem_values, tol_abs=0.01, crypto_rel=0.30, crypto_slots=(), rebal_eps=0.0025,
                window_days=7):
    """What the live account should do now, from the shadow state and the live values.

    rebalance : the shadow rebalanced on its reference month-end bar and the
                account still differs from it (any slot off by more than rebal_eps)
    sync      : at a month-end check (the reference month-end is at most
                `window_days` old), a slot is more than tol_abs off its shadow
                weight, or a crypto slot more than crypto_rel off in relative terms
    watch     : the same deviations mid-month -> sync at the next month-end
    ok        : nothing to do
    Returns {"level", "reasons": [...], "table": DataFrame per slot}."""
    st = state["slot_table"]
    live_w = slot_weights(live_elem_values, st)
    tab = pd.DataFrame({"slot": st["slot"], "label": st["label"], "shadow": st["weight"].values,
                        "live": live_w.reindex(st["slot"]).values})
    tab["dev"] = tab["live"] - tab["shadow"]
    tab["rel_dev"] = tab["dev"] / tab["shadow"].where(tab["shadow"] > 0)
    over = tab[(tab["dev"].abs() > tol_abs) | (tab["slot"].isin(crypto_slots) & (tab["rel_dev"].abs() > crypto_rel))]
    reasons = [f"{r.label}: live {r.live:.2%} vs shadow {r.shadow:.2%}" for r in over.itertuples()]
    ev = state.get("ref_event")
    at_check = state["month_complete"] or (state["as_of"] - state["ref_date"]).days <= window_days
    if ev is not None and tab["dev"].abs().max() > rebal_eps:
        kind = "global" if ev["scope"] == "global" else "local"
        return {"level": "rebalance", "table": tab,
                "reasons": [f"shadow rebalanced ({kind}, trigger {'+'.join(ev['trigger'])}) on "
                            f"{pd.Timestamp(ev['date']).date()}"] + reasons}
    if len(over):
        return {"level": "sync" if at_check else "watch", "table": tab, "reasons": reasons}
    return {"level": "ok", "table": tab, "reasons": []}


# --------------------------------------------------------------------------- #
# Holdings import
# --------------------------------------------------------------------------- #
def parse_ib_flex_positions(xml_text):
    """Stock / ETF positions from an IB Flex statement (OpenPosition rows).
    Options and other asset classes are skipped; when lot-level rows are present
    only SUMMARY rows are used. Returns (DataFrame[symbol, shares, price, value],
    meta {account, report_date})."""
    root = ET.fromstring(xml_text)
    stmt = root.find(".//FlexStatement")
    meta = {"account": stmt.get("accountId") if stmt is not None else None, "report_date": None}
    rows = []
    for el in root.iter("OpenPosition"):
        lvl = el.get("levelOfDetail")
        if lvl and lvl.upper() != "SUMMARY":
            continue
        cat = (el.get("assetCategory") or "STK").upper()
        if cat not in ("STK", "ETF", "FUND"):
            continue
        try:
            q = float(el.get("position") or 0)
        except ValueError:
            continue
        if q == 0:
            continue
        fx = float(el.get("fxRateToBase") or 1) or 1.0
        mark = el.get("markPrice")
        pv = el.get("positionValue")
        price = float(mark) * fx if mark not in (None, "") else (float(pv) * fx / q if pv not in (None, "") else float("nan"))
        rows.append({"symbol": norm_symbol(el.get("symbol")), "shares": q, "price": price, "value": q * price})
        meta["report_date"] = meta["report_date"] or el.get("reportDate")
    df = pd.DataFrame(rows, columns=["symbol", "shares", "price", "value"])
    if len(df):
        df = df.groupby("symbol", as_index=False).agg(shares=("shares", "sum"), value=("value", "sum"))
        df["price"] = df["value"] / df["shares"]
        df = df[["symbol", "shares", "price", "value"]]
    return df, meta


_COLS = {"symbol": ("symbol", "ticker", "financial instrument", "instrument", "code"),
         "shares": ("shares", "quantity", "position", "qty", "units"),
         "price": ("price", "last", "last price", "mark", "mark price", "close", "close price", "market price")}


def parse_positions_csv(text):
    """A holdings CSV with a symbol column and a share-count column (any of the
    usual header names, case-insensitive); an optional price column is kept.
    Returns DataFrame[symbol, shares, price]."""
    df = pd.read_csv(io.StringIO(text))
    low = {c: str(c).strip().lower() for c in df.columns}
    pick = {}
    for k, names in _COLS.items():
        pick[k] = next((c for c, l in low.items() if l in names), None)
    if pick["symbol"] is None or pick["shares"] is None:
        raise ValueError("CSV needs a symbol column and a shares / quantity column")
    out = pd.DataFrame({"symbol": df[pick["symbol"]].map(norm_symbol),
                        "shares": pd.to_numeric(df[pick["shares"]].astype(str).str.replace(",", ""), errors="coerce")})
    out["price"] = pd.to_numeric(df[pick["price"]].astype(str).str.replace(",", ""), errors="coerce") if pick["price"] else np.nan
    out = out[out["symbol"].astype(bool) & out["shares"].notna() & (out["shares"] != 0)]
    return out.groupby("symbol", as_index=False).agg(shares=("shares", "sum"), price=("price", "last"))


def orders_csv(orders):
    """Order list as CSV (Symbol, Action, Quantity, RefPrice, Amount)."""
    df = pd.DataFrame({"Symbol": orders["symbol"], "Action": orders["side"], "Quantity": orders["shares"],
                       "RefPrice": orders["price"].round(4), "Amount": orders["amount"].round(2)})
    return df.to_csv(index=False)
