"""Live Portfolio desk (v2.6.0): the Streamlit page, rendered by backtest_app.py
when the sidebar mode is "Live Portfolio". All arithmetic lives in live_core.py.

Privacy: holdings only live in this browser session (st.session_state). Nothing
is written to disk; the local IB Flex snapshot is read, never copied.
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from backtest_core import (
    STRAT_LEGACY_MAP, STRAT_ASYM, STRAT_BH, STRAT_ANNUAL, STRAT_SEMI, STRAT_RD_FULL, STRAT_RD_LOCAL, STRAT_RD_MIXED,
    BAND_MODE_RATIO, BAND_MODE_REL, parse_portfolio, clean_ticker, normalize_slot_bands,
    scrub_leading_glitches, scrub_isolated_spikes, align_price_data, prepare_portfolio,
)
import live_core as lc

FO_SYNC_DIR = Path.home() / "Work-Space" / "My-Claude-Project" / "wall-street-analyst" / "family_office" / "ib_sync"
IB_FLEX_PATH = FO_SYNC_DIR / "last_flex.xml"
FO_CONFIG_PATH = FO_SYNC_DIR / "config.yaml"
BAND_LABELS = {BAND_MODE_REL: "Δ vs target", BAND_MODE_RATIO: "Leg vs rest"}
VALID_STRATS = {STRAT_BH, STRAT_ANNUAL, STRAT_SEMI, STRAT_RD_FULL, STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_ASYM}
SRC_IB, SRC_CSV, SRC_MANUAL = "IB Flex snapshot (local)", "Upload CSV", "Manual entry"
OPS = {"Cash flow only": "flow", "Sync to shadow": "sync", "Sync + cash flow": "sync"}


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _band_pct(v, fallback):
    try:
        return min(200, max(1, int(round(float(v)))))
    except (TypeError, ValueError):
        return fallback


def normalize_port(p):
    """Same normalisation the Backtest page applies to a loaded config."""
    q = dict(p)
    q["thr"] = _band_pct(q.get("thr"), 38)
    q["thr_up"] = _band_pct(q.get("thr_up"), q["thr"])
    q["slot_bands"] = normalize_slot_bands(q.get("slot_bands"))
    q["band_mode"] = BAND_MODE_RATIO if str(q.get("band_mode") or "").strip().lower() == BAND_MODE_RATIO else BAND_MODE_REL
    strat = STRAT_LEGACY_MAP.get(q.get("strat"), q.get("strat"))
    q["strat"] = strat if strat in VALID_STRATS else STRAT_ASYM
    return q


def compute_shadow(cfg, port_idx, fetch_price_history):
    """Run the chosen portfolio exactly like the Backtest page (same fetch cache,
    scrubbing, alignment over ALL the config's portfolios, preparation, engine)
    and derive the live-desk shadow state."""
    ports = [normalize_port(p) for p in cfg.get("portfolios", [])]
    p = ports[port_idx]
    parsed = [parse_portfolio(q) for q in ports]
    bench_tk = clean_ticker(str(cfg.get("benchmark") or "SPY"))
    start_d = pd.Timestamp(cfg.get("start_date") or "2020-01-01").date()
    all_tks = sorted(set([bench_tk] + [t for tks, *_ in parsed for t in tks]))
    price_data, failures = fetch_price_history(tuple(all_tks), str(start_d - timedelta(days=20)))
    if bench_tk in failures:
        raise RuntimeError(f"Benchmark {bench_tk} could not be downloaded: {failures[bench_tk]}")
    scrub_leading_glitches(price_data)
    scrub_isolated_spikes(price_data)
    aligned = align_price_data(price_data, bench_tk, start_d, sorted({t for tks, *_ in parsed for t in tks}))
    if aligned["error"]:
        raise RuntimeError(aligned["error"])
    tks, wts, errs, comp = parsed[port_idx]
    if errs:
        raise RuntimeError("; ".join(errs))
    prep = prepare_portfolio(p, tks, wts, comp, aligned["price_df"])
    if prep["error"]:
        raise RuntimeError(prep["error"])
    now_ny = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    state = lc.shadow_state(p["strat"], aligned["price_df"], prep["w_series"], prep["thr_dn"], prep["thr_up"],
                            prep["groups"], p["band_mode"], now=now_ny)
    notices = ([aligned["notice"]] if aligned["notice"] else []) + prep["notices"]
    return {"state": state, "port": p, "notices": notices, "failures": failures,
            "elements": list(prep["w_series"].index), "computed": pd.Timestamp.now()}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_quotes(symbols):
    """Latest price per symbol (today's bar during the session, else the last
    close) -> {symbol: (price, date)}. Yahoo symbols: BRK-B, not BRK.B."""
    symbols = tuple(symbols)
    if not symbols:
        return {}
    df = yf.download(list(symbols), period="5d", interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return {}
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df[["Close"]].rename(columns={"Close": symbols[0]})
    out = {}
    for s in symbols:
        if s in close.columns:
            ser = pd.to_numeric(close[s], errors="coerce").dropna()
            if len(ser):
                out[s] = (float(ser.iloc[-1]), str(pd.Timestamp(ser.index[-1]).date()))
    return out


def fo_avus_components():
    """AV-US component + substitute tickers from the family-office sync config (local only)."""
    try:
        import yaml
        c = yaml.safe_load(FO_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return {lc.norm_symbol(x) for x in (c.get("avus_components") or []) + (c.get("avus_crypto_substitutes") or [])}
    except Exception:
        return set()


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def _pct(v, digits=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v * 100:.{digits}f}%"


def _move(v):
    return "never" if v is None else f"{v * 100:+.0f}%"


def _usd(v):
    """Dollar amount for MARKDOWN text: '$' is escaped, a pair of them would be typeset as LaTeX."""
    return f"\\${v:,.2f}" if v >= 0 else f"−\\${-v:,.2f}"


def _qty(q):
    return f"{q:,.4f}".rstrip("0").rstrip(".")


def render_live_desk(fetch_price_history, saved_config_dir):
    ss = st.session_state
    st.caption("The **shadow** is this app's backtest of the live rule, run to the latest price with the same data "
               "pipeline and engine as the Backtest page. It alone decides when to rebalance and what each holding "
               "should weigh; this page turns it into share counts for the real account. Holdings stay in this "
               "browser session and are never written to the server.")

    # ---- 1. Rule ----------------------------------------------------------
    cfgs = sorted(Path(saved_config_dir).glob("*.json")) if Path(saved_config_dir).is_dir() else []
    cfgs = [c for c in cfgs if c.name != "_default.json"]
    if not cfgs:
        st.error("No saved configs in Backtest/ — save the live rule as a config first.")
        return
    default_i = next((i for i, c in enumerate(cfgs) if "实盘" in c.stem), 0)
    c1, c2 = st.columns([3, 2])
    cfg_path = c1.selectbox("Live rule config", cfgs, index=default_i, format_func=lambda c: c.stem, key="live_cfg")
    try:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Cannot read {Path(cfg_path).name}: {e}")
        return
    ports = cfg.get("portfolios") or []
    if not ports:
        st.error("That config has no portfolios.")
        return
    pi = c2.selectbox("Portfolio", list(range(len(ports))), format_func=lambda i: ports[i].get("name", f"#{i + 1}"),
                      key=f"live_port_{Path(cfg_path).stem}")
    p = normalize_port(ports[pi])
    sb = "; ".join(f"{k} {v.get('down') or '·'}/{v.get('up') or '·'}" for k, v in p["slot_bands"].items())
    st.caption(f"**{p.get('name')}** · {p.get('tickers')} · weights {p.get('weights')} · {p['strat']} · "
               f"{BAND_LABELS[p['band_mode']]} {p['thr']}/{p['thr_up']}" + (f" · per-slot {sb}" if sb else "")
               + f" · start {cfg.get('start_date')} · benchmark {cfg.get('benchmark', 'SPY')}")

    rule_key = (str(cfg_path), pi)
    if ss.get("live_rule_key") != rule_key:           # a different rule invalidates the shadow and the plan
        ss["live_rule_key"] = rule_key
        for k in ("live_shadow", "live_plan"):
            ss.pop(k, None)

    tks, _, _, _ = parse_portfolio(p)
    with st.expander("Instrument mapping · shadow element → live tickers (primary first; buys go to the primary)"):
        mcols = st.columns(4)
        emap_in = {}
        for i, e in enumerate(tks):
            default = ", ".join(lc.mapping_for([e])[e])
            txt = mcols[i % 4].text_input(e, value=default, key=f"live_map_{e}")
            emap_in[e] = [x for x in (t.strip() for t in txt.split(",")) if x]
    emap = lc.mapping_for(tks, emap_in)
    inv = lc.symbol_to_element(emap)

    # ---- 2. Shadow --------------------------------------------------------
    b1, b2 = st.columns([1.4, 4])
    if b1.button(":material/refresh: Refresh shadow", type="primary", width="stretch",
                 help="Downloads prices (shared cache with the Backtest page) and runs the live rule to today."):
        try:
            with st.spinner("Running the shadow backtest…"):
                ss["live_shadow"] = compute_shadow(cfg, pi, fetch_price_history)
            ss.pop("live_plan", None)
        except Exception as e:
            ss.pop("live_shadow", None)
            st.error(f"Shadow not available: {e}")
    sh = ss.get("live_shadow")
    if not sh:
        b2.info("Click **Refresh shadow** to run the live rule to today.")
    else:
        stt = sh["state"]
        for lvl, msg in sh["notices"]:
            (st.warning if lvl == "warning" else st.info)(msg)
        label_of = {r["slot"]: r["label"] for _, r in stt["slot_table"].iterrows()}
        lines = [f"Prices through **{stt['as_of'].date()}** · state taken from the month-end bar of "
                 f"**{stt['ref_date'].date()}**" + ("" if stt["month_complete"] else " (month in progress: drifted to the latest prices)")]
        ev = stt["ref_event"]
        if ev:
            lines.append(f"**Rebalanced on that month-end bar** — {ev['scope']} reset, triggered by "
                         f"{', '.join(label_of.get(t, t) for t in ev['trigger'])}.")
        else:
            lines.append("No rebalance on that month-end bar.")
        pv = stt["preview_event"]
        if pv:
            lines.append(f"Preview: if the month ended today, a **{pv['scope']}** rebalance would trigger "
                         f"({', '.join(label_of.get(t, t) for t in pv['trigger'])}). The live protocol waits for month-end.")
        lines.append(f"Next month-end check: **{lc.next_month_end(stt['as_of'], stt['month_complete']).date()}** close.")
        st.markdown("  \n".join(lines))
        t = stt["slot_table"]
        st.dataframe(pd.DataFrame({
            "Slot": t["label"], "Target": t["target"].map(_pct), "Shadow weight": t["weight"].map(_pct),
            "Trigger <": t["trig_low"].map(lambda v: "never" if v is None else _pct(v)),
            "Trigger >": t["trig_high"].map(lambda v: "never" if v is None else _pct(v)),
            "Move vs rest to lower": t["move_to_low"].map(_move), "Move vs rest to upper": t["move_to_high"].map(_move)}),
            hide_index=True, width="stretch")

    # ---- 3. Holdings ------------------------------------------------------
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Live holdings (AV-US)</div>', unsafe_allow_html=True)
    sources = ([SRC_IB] if IB_FLEX_PATH.is_file() else []) + [SRC_CSV, SRC_MANUAL]
    src = st.radio("Holdings source", sources, horizontal=True, key="live_src", label_visibility="collapsed")
    ib_stamp = IB_FLEX_PATH.stat().st_mtime if (src == SRC_IB and IB_FLEX_PATH.is_file()) else None
    load_key = (src, rule_key, ib_stamp)
    if src != SRC_CSV and ss.get("live_src_loaded") != load_key:
        base, fallback, meta = None, {}, {}
        if src == SRC_IB:
            try:
                df, meta = lc.parse_ib_flex_positions(IB_FLEX_PATH.read_text(encoding="utf-8"))
                comps = fo_avus_components()
                av = df[df["symbol"].isin(inv.keys())]
                ss["live_unmapped_av"] = sorted((set(df["symbol"]) & comps) - set(inv))
                base = pd.DataFrame({"Symbol": av["symbol"], "Shares": av["shares"], "Price override": np.nan})
                fallback = dict(zip(av["symbol"], av["price"]))
            except Exception as e:
                st.error(f"Cannot read the IB snapshot: {e}")
        elif src == SRC_MANUAL:
            base = pd.DataFrame({"Symbol": [emap[e][0] for e in tks], "Shares": 0.0, "Price override": np.nan})
        if base is not None:
            ss["live_hold_base"] = base.reset_index(drop=True)
            ss["live_fallback_px"] = fallback
            ss["live_meta"] = meta
            ss["live_src_loaded"] = load_key
            ss["live_ed_n"] = ss.get("live_ed_n", 0) + 1
            ss.pop("live_plan", None)
    if src == SRC_CSV:
        up = st.file_uploader("Holdings CSV (symbol + shares columns; price optional)", type=["csv"], key="live_csv")
        if up is not None and ss.get("live_csv_name") != (up.name, up.size, rule_key):
            try:
                df = lc.parse_positions_csv(up.getvalue().decode("utf-8-sig"))
                av = df[df["symbol"].isin(inv.keys())]
                ss["live_hold_base"] = pd.DataFrame({"Symbol": av["symbol"], "Shares": av["shares"], "Price override": np.nan}).reset_index(drop=True)
                ss["live_fallback_px"] = {s: p for s, p in zip(av["symbol"], av["price"]) if np.isfinite(p)}
                ss["live_meta"] = {}
                ss["live_unmapped_av"] = sorted(set(df["symbol"]) - set(inv))
                ss["live_csv_name"] = (up.name, up.size, rule_key)
                ss["live_src_loaded"] = load_key
                ss["live_ed_n"] = ss.get("live_ed_n", 0) + 1
                ss.pop("live_plan", None)
            except Exception as e:
                st.error(f"CSV not read: {e}")
    base = ss.get("live_hold_base") if ss.get("live_src_loaded") == load_key else None
    if base is None:
        st.info("Upload a holdings CSV to continue." if src == SRC_CSV else "No holdings loaded.")
        return
    meta = ss.get("live_meta") or {}
    if meta.get("report_date"):
        rd = pd.Timestamp(meta["report_date"])
        stale = (pd.Timestamp.now().normalize() - rd).days > 3
        (st.warning if stale else st.caption)(f"IB snapshot of {rd.date()}" + (" — older than 3 days: run `family_office.py ibsync` first." if stale else "")
                                              + " Shares are read from it; edit below if trades happened since.")
    if ss.get("live_unmapped_av"):
        st.warning("Held but not mapped to a shadow element (left out): **" + ", ".join(ss["live_unmapped_av"])
                   + "**. Add them to the mapping above if they belong to a slot.")
    edited = st.data_editor(base, key=f"live_ed_{ss.get('live_ed_n', 0)}", num_rows="dynamic", hide_index=True,
                            width="stretch", column_config={
                                "Symbol": st.column_config.TextColumn("Symbol", help="Broker symbol, e.g. BRK.B or BRK-B"),
                                "Shares": st.column_config.NumberColumn("Shares", format="%.4f", min_value=0.0),
                                "Price override": st.column_config.NumberColumn(
                                    "Price override", format="%.4f", min_value=0.0,
                                    help="Optional: price to use instead of the quote (e.g. your limit price)")})
    hold = {}
    overrides = {}
    for _, r in edited.iterrows():
        s = lc.norm_symbol(r.get("Symbol"))
        if not s or pd.isna(r.get("Shares")):
            continue
        hold[s] = hold.get(s, 0.0) + float(r["Shares"])
        if pd.notna(r.get("Price override")) and float(r["Price override"]) > 0:
            overrides[s] = float(r["Price override"])
    need = sorted(set(hold) | {emap[e][0] for e in tks})
    q1, q2 = st.columns([1.4, 4])
    if q1.button(":material/sell: Fetch live quotes", width="stretch",
                 help="Latest Yahoo prices for the held symbols and each element's primary ticker (cached 60 s)."):
        try:
            ss["live_quotes"] = fetch_quotes(tuple(need))
        except Exception as e:
            st.error(f"Quotes not available: {e}")
    quotes = ss.get("live_quotes") or {}
    prices, px_src = {}, {}
    for s in need:
        if s in overrides:
            prices[s], px_src[s] = overrides[s], "override"
        elif s in quotes:
            prices[s], px_src[s] = quotes[s][0], f"quote {quotes[s][1]}"
        elif s in (ss.get("live_fallback_px") or {}):
            prices[s], px_src[s] = float(ss["live_fallback_px"][s]), "snapshot"
    q2.caption("Prices: " + " · ".join(f"{s} {prices[s]:,.2f} ({px_src[s]})" for s in need if s in prices)
               + (" · missing: **" + ", ".join(s for s in need if s not in prices) + "**" if any(s not in prices for s in need) else ""))

    if not sh:
        return
    elem_vals, detail, unmapped, no_price = lc.holdings_by_element(hold, prices, emap)
    if no_price:
        st.warning("No price for **" + ", ".join(no_price) + "** — fetch quotes or enter a price override.")
        return
    total = float(elem_vals.sum())
    if total <= 0:
        st.info("Enter share counts to compare the account with the shadow.")
        return
    stt = sh["state"]
    adv = lc.sync_advice(stt, elem_vals, crypto_slots=lc.crypto_slot_ids(stt["slot_table"]))
    tab = adv["table"]
    st.markdown(f"AV-US value **{_usd(total)}**")
    crypto = set(lc.crypto_slot_ids(stt["slot_table"]))
    beyond = [(abs(d) > 0.01) or (sid in crypto and pd.notna(r) and abs(r) > 0.30)
              for sid, d, r in zip(tab["slot"], tab["dev"], tab["rel_dev"])]
    dev_df = pd.DataFrame({"Slot": tab["label"], "Shadow": tab["shadow"].map(_pct), "Live": tab["live"].map(_pct),
                           "Deviation": tab["dev"].map(lambda v: f"{v * 100:+.2f} pt"),
                           "Relative": tab["rel_dev"].map(lambda v: "–" if pd.isna(v) else f"{v * 100:+.0f}%")})
    st.dataframe(dev_df.style.apply(lambda row: ["background-color: #fdecea; color: #1a1a2e" if beyond[row.name] else ""] * len(row),
                                    axis=1), hide_index=True, width="stretch")
    # Substituted instruments: how far the live ticker has run from the shadow element since that slot's last reset
    subs = [(e, emap[e][0]) for e in tks if emap[e][0] != lc.norm_symbol(e)]
    if subs:
        notes = []
        for e, live_sym in subs:
            sid = next((r["slot"] for _, r in stt["slot_table"].iterrows() if e in r["members"]), None)
            since = lc.last_reset_date(stt["events"], sid) if sid else None
            if since is None:
                continue
            try:
                px, fails = fetch_price_history((e, live_sym), str((since - pd.Timedelta(days=7)).date()))
                a, b = px[e].dropna(), px[live_sym].dropna()
                ra = a.iloc[-1] / a[a.index <= since].iloc[-1] - 1
                rb = b.iloc[-1] / b[b.index <= since].iloc[-1] - 1
                notes.append(f"{live_sym} for {e} since that slot's last reset ({since.date()}): {live_sym} {rb * 100:+.1f}% vs "
                             f"{e} {ra * 100:+.1f}% — tracking gap {(rb - ra) * 100:+.1f} pts")
            except Exception:
                notes.append(f"{live_sym} vs {e}: no price history for the tracking gap")
        if notes:
            st.caption("  \n".join(notes))
    msg = {"rebalance": ("error", "**Rebalance due** — sync the account to the shadow. "),
           "sync": ("warning", "**Sync recommended** at this month-end check (beyond tolerance: any slot > 1 pt, crypto > 30% relative). "),
           "watch": ("info", "Beyond tolerance mid-month — **sync at the next month-end** if it persists. "),
           "ok": ("success", "Within tolerance — no sync needed. Cash flows still steer toward the shadow.")}[adv["level"]]
    getattr(st, msg[0])(msg[1] + ("  \n" + "  \n".join(adv["reasons"]) if adv["reasons"] else ""))

    # ---- 4. Orders --------------------------------------------------------
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Orders</div>', unsafe_allow_html=True)
    default_op = 1 if adv["level"] in ("rebalance", "sync") else 0
    o1, o2, o3, o4 = st.columns([2.6, 1.6, 1.1, 1.1], vertical_alignment="bottom")
    op = o1.radio("Operation", list(OPS), index=default_op, horizontal=True, key="live_op",
                  help="Cash flow only: a deposit buys what sits below the shadow, a withdrawal sells what sits above it "
                       "(pro rata when the account matches). Sync: trade every holding to its shadow weight — this is "
                       "also how a shadow rebalance is executed.")
    flow = 0.0
    if op != "Sync to shadow":
        flow = o2.number_input("Cash flow ($, − = withdrawal)", value=0.0, step=1000.0, format="%.2f", key="live_flow")
    whole = o3.checkbox("Whole shares", value=True, key="live_whole")
    min_trade = o4.number_input("Min trade ($)", value=200.0, min_value=0.0, step=50.0, key="live_min",
                                help="Orders smaller than the larger of this and 0.05% of the portfolio are skipped.")
    if st.button(":material/calculate: Calculate orders", type="primary"):
        if OPS[op] == "flow" and flow == 0:
            st.info("Enter a cash flow amount.")
        else:
            try:
                eff_min = max(float(min_trade), 0.0005 * total)        # default: the larger of $200 and 0.05% of the portfolio
                ss["live_plan"] = lc.plan_orders(hold, prices, emap, stt["elem_weights"], flow=flow, mode=OPS[op],
                                                 whole_shares=whole, min_trade=eff_min)
                ss["live_plan_desc"] = f"{op} · cash flow {_usd(flow)} · minimum trade {_usd(eff_min)}"
            except Exception as e:
                ss.pop("live_plan", None)
                st.error(f"Orders not computed: {e}")
    plan = ss.get("live_plan")
    if plan:
        st.caption(ss.get("live_plan_desc", ""))
        for w in plan["warnings"]:
            st.warning(w.replace("$", "\\$"))
        o = plan["orders"]
        if o.empty:
            st.success("No orders needed.")
        else:
            st.dataframe(pd.DataFrame({"Symbol": o["symbol"], "Shadow element": o["element"], "Side": o["side"],
                                       "Shares": o["shares"].map(_qty), "Ref price": o["price"].map(lambda v: f"${v:,.2f}"),
                                       "Amount": o["amount"].map(lambda v: f"${v:,.2f}")}),
                         hide_index=True, width="stretch")
            buys = o.loc[o["side"] == "BUY", "amount"].sum()
            sells = o.loc[o["side"] == "SELL", "amount"].sum()
            st.markdown(f"Buys **{_usd(buys)}** · Sells **{_usd(sells)}** · Cash flow **{_usd(plan['flow'])}** · "
                        f"Cash left **{_usd(plan['cash_left'])}** · Est. commission **{_usd(lc.ibkr_commission(o))}** (IBKR Pro fixed)")
            st.download_button(":material/download: Orders CSV", data=lc.orders_csv(o), file_name="avus_orders.csv",
                               mime="text/csv")
        e = plan["elements"]
        st.dataframe(pd.DataFrame({"Element": e["element"], "Before": e["weight_before"].map(_pct),
                                   "After": e["weight_after"].map(_pct), "Shadow": e["target"].map(_pct),
                                   "After − shadow": e["dev_after"].map(lambda v: f"{v * 100:+.2f} pt")}),
                     hide_index=True, width="stretch")
