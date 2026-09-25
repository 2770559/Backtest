"""v2.6.0 live desk: live_core (orders, shadow state, advice, imports) and the
alignment / preparation helpers moved from the app into backtest_core.

Run with:  python3 -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from backtest_core import (  # noqa: E402
    STRAT_RD_MIXED, BAND_MODE_RATIO, align_price_data, prepare_portfolio, parse_portfolio,
    run_detailed_backtest, sample_monthly,
)
import live_core as lc  # noqa: E402


def S(d):
    return pd.Series(d, dtype=float)


# --------------------------------------------------------------------------- #
# Symbols and mapping
# --------------------------------------------------------------------------- #
class MappingTest(unittest.TestCase):
    def test_norm_symbol_aliases(self):
        for raw in ("BRK B", "brk.b", " BRK/B ", "BRKB"):
            self.assertEqual(lc.norm_symbol(raw), "BRK-B")
        self.assertEqual(lc.norm_symbol(" qqqm "), "QQQM")

    def test_mapping_defaults_and_inverse(self):
        emap = lc.mapping_for(["QQQM", "ETH-USD", "MSTR"])
        self.assertEqual(emap["QQQM"], ["QQQM"])
        self.assertEqual(emap["ETH-USD"][0], "BMNR")
        inv = lc.symbol_to_element(emap)
        self.assertEqual((inv["BMNR"], inv["ETHW"], inv["IBIT"], inv["QQQM"]), ("ETH-USD", "ETH-USD", "MSTR", "QQQM"))

    def test_custom_mapping(self):
        emap = lc.mapping_for(["ETH-USD"], {"ETH-USD": ["ethw", ""]})
        self.assertEqual(emap["ETH-USD"], ["ETHW"])


class MonthCompleteTest(unittest.TestCase):
    def test_last_business_day(self):
        self.assertTrue(lc.month_complete("2026-09-30"))         # Wednesday
        self.assertFalse(lc.month_complete("2026-09-29"))
        self.assertTrue(lc.month_complete("2026-10-30"))         # Friday, 31st is Saturday
        self.assertTrue(lc.month_complete("2026-05-29"))         # Friday, 31st is Sunday


# --------------------------------------------------------------------------- #
# Element-level trades
# --------------------------------------------------------------------------- #
class ElementTradesTest(unittest.TestCase):
    def test_flow_deposit_buys_only_below_shadow(self):
        x = lc.element_trades(S({"A": 60, "B": 40}), S({"A": .5, "B": .5}), flow=20, mode="flow")
        self.assertAlmostEqual(x["A"], 0.0)
        self.assertAlmostEqual(x["B"], 20.0)

    def test_flow_is_pro_rata_when_matched(self):
        x = lc.element_trades(S({"A": 50, "B": 50}), S({"A": .5, "B": .5}), flow=20, mode="flow")
        self.assertAlmostEqual(x["A"], 10.0)
        self.assertAlmostEqual(x["B"], 10.0)

    def test_flow_withdrawal_sells_only_above_shadow(self):
        x = lc.element_trades(S({"A": 70, "B": 30}), S({"A": .5, "B": .5}), flow=-20, mode="flow")
        self.assertAlmostEqual(x["A"], -20.0)
        self.assertAlmostEqual(x["B"], 0.0)

    def test_flow_min_trade_redistributes(self):
        v = S({"A": 40, "B": 39, "C": 19})
        x = lc.element_trades(v, S({"A": .4, "B": .4, "C": .2}), flow=500, mode="flow", min_trade=200)
        self.assertAlmostEqual(x.sum(), 500.0)
        self.assertTrue(((x == 0) | (x >= 200)).all(), x)

    def test_flow_all_small_goes_to_largest_gap(self):
        x = lc.element_trades(S({"A": 50, "B": 50}), S({"A": .6, "B": .4}), flow=10, mode="flow", min_trade=200)
        self.assertEqual((x > 0).sum(), 1)
        self.assertAlmostEqual(x.sum(), 10.0)

    def test_sync_with_and_without_flow(self):
        x = lc.element_trades(S({"A": 70, "B": 30}), S({"A": .5, "B": .5}), flow=0, mode="sync")
        self.assertAlmostEqual(x["A"], -20.0)
        self.assertAlmostEqual(x["B"], 20.0)
        x = lc.element_trades(S({"A": 70, "B": 30}), S({"A": .5, "B": .5}), flow=10, mode="sync")
        self.assertAlmostEqual(x["A"], -15.0)
        self.assertAlmostEqual(x["B"], 25.0)
        self.assertAlmostEqual(x.sum(), 10.0)

    def test_sync_small_trades_dropped_and_rebalanced(self):
        v = S({"A": 1000, "B": 1000, "C": 1000})
        x = lc.element_trades(v, S({"A": .4, "B": .3, "C": .3}), flow=0, mode="sync", min_trade=150)
        self.assertTrue((x == 0).all(), x)                          # sells dropped -> nothing funds the buy
        x = lc.element_trades(v, S({"A": .5, "B": .3, "C": .2}), flow=0, mode="sync", min_trade=150)
        # x = +500 / -100 / -400: B's -100 is dropped, so the +500 buy is scaled to the 400 still sold
        self.assertEqual(x["B"], 0.0)
        self.assertAlmostEqual(x["A"], 400.0)
        self.assertAlmostEqual(x["C"], -400.0)

    def test_withdrawal_larger_than_portfolio_raises(self):
        with self.assertRaises(ValueError):
            lc.element_trades(S({"A": 10}), S({"A": 1.0}), flow=-20, mode="sync")


# --------------------------------------------------------------------------- #
# Share-level orders
# --------------------------------------------------------------------------- #
class PlanOrdersTest(unittest.TestCase):
    EMAP = {"A": ["A"], "B": ["B"]}

    def test_deposit_whole_shares_spends_leftover(self):
        r = lc.plan_orders({"A": 10, "B": 5}, {"A": 10.0, "B": 20.0}, self.EMAP, S({"A": .5, "B": .5}),
                           flow=50, mode="flow", min_trade=0)
        o = r["orders"].set_index("symbol")
        self.assertEqual((o.loc["A", "side"], o.loc["A", "shares"]), ("BUY", 3))
        self.assertEqual((o.loc["B", "side"], o.loc["B", "shares"]), ("BUY", 1))
        self.assertAlmostEqual(r["cash_left"], 0.0)

    def test_withdrawal_rounds_sells_up_and_is_fully_funded(self):
        r = lc.plan_orders({"A": 10, "B": 5}, {"A": 10.0, "B": 20.0}, self.EMAP, S({"A": .5, "B": .5}),
                           flow=-30, mode="flow", min_trade=0)
        o = r["orders"].set_index("symbol")
        self.assertEqual((o.loc["A", "shares"], o.loc["B", "shares"]), (2, 1))
        self.assertTrue((r["orders"]["side"] == "SELL").all())
        self.assertGreaterEqual(r["cash_left"], 0.0)
        self.assertAlmostEqual(r["cash_left"], 10.0)

    def test_buys_go_to_primary_of_mapped_element(self):
        emap = {"QQQM": ["QQQM"], "ETH-USD": ["BMNR", "ETHW"]}
        r = lc.plan_orders({"QQQM": 10, "ETHW": 5}, {"QQQM": 100.0, "BMNR": 40.0, "ETHW": 20.0}, emap,
                           S({"QQQM": .5, "ETH-USD": .5}), flow=0, mode="sync", min_trade=0)
        o = r["orders"].set_index("symbol")
        self.assertIn("BMNR", o.index)                               # buy the primary, not the held ETHW
        self.assertEqual(o.loc["BMNR", "side"], "BUY")
        self.assertEqual(o.loc["QQQM", "side"], "SELL")

    def test_group_sells_pro_rata_across_held_tickers(self):
        emap = {"MSTR": ["MSTR", "IBIT"], "A": ["A"]}
        r = lc.plan_orders({"MSTR": 10, "IBIT": 50}, {"MSTR": 100.0, "IBIT": 20.0, "A": 10.0}, emap,
                           S({"MSTR": .5, "A": .5}), flow=0, mode="sync", min_trade=0)
        o = r["orders"].set_index("symbol")
        self.assertEqual((o.loc["MSTR", "shares"], o.loc["IBIT", "shares"], o.loc["A", "shares"]), (5, 25, 100))
        self.assertAlmostEqual(r["cash_left"], 0.0)

    def test_sell_never_exceeds_fractional_holding(self):
        r = lc.plan_orders({"A": 2.5}, {"A": 100.0, "B": 10.0}, self.EMAP, S({"A": 0.0, "B": 1.0}),
                           flow=0, mode="sync", min_trade=0)
        o = r["orders"].set_index("symbol")
        self.assertAlmostEqual(o.loc["A", "shares"], 2.5)
        self.assertEqual(o.loc["B", "shares"], 25)

    def test_fractional_mode(self):
        r = lc.plan_orders({"A": 10, "B": 5}, {"A": 10.0, "B": 20.0}, self.EMAP, S({"A": .5, "B": .5}),
                           flow=50, mode="flow", whole_shares=False, min_trade=0)
        o = r["orders"].set_index("symbol")
        self.assertAlmostEqual(o.loc["A", "shares"], 2.5)
        self.assertAlmostEqual(o.loc["B", "shares"], 1.25)
        self.assertAlmostEqual(r["cash_left"], 0.0, places=6)

    def test_post_trade_close_to_shadow_and_unmapped_warned(self):
        emap = {"A": ["A"], "B": ["B"], "C": ["C"]}
        r = lc.plan_orders({"A": 700, "B": 200, "C": 100, "ZZZ": 3}, {"A": 10.0, "B": 10.0, "C": 10.0, "ZZZ": 5.0},
                           emap, S({"A": .5, "B": .3, "C": .2}), flow=0, mode="sync", min_trade=0)
        self.assertTrue(any("ZZZ" in w for w in r["warnings"]))
        self.assertLess(r["elements"]["dev_after"].abs().max(), 0.001)

    def test_orders_csv(self):
        r = lc.plan_orders({"A": 10, "B": 5}, {"A": 10.0, "B": 20.0}, self.EMAP, S({"A": .5, "B": .5}),
                           flow=50, mode="flow", min_trade=0)
        csv = lc.orders_csv(r["orders"])
        self.assertTrue(csv.startswith("Symbol,Action,Quantity,RefPrice,Amount"))
        self.assertIn("A,BUY,3", csv)


# --------------------------------------------------------------------------- #
# Shadow state
# --------------------------------------------------------------------------- #
def _daily(end, jump=None):
    idx = pd.bdate_range("2024-01-02", end)
    n = len(idx)
    a = 100 * np.exp(np.linspace(0, 0.8, n))                       # A trends up
    b = 100 * np.exp(np.linspace(0, -0.2, n))
    c = 100 * np.ones(n)
    df = pd.DataFrame({"A": a, "B": b, "C": c, "SPY": 100 * np.ones(n)}, index=idx)
    if jump:
        df.iloc[-1, df.columns.get_loc("A")] *= jump
    return df


class ShadowStateTest(unittest.TestCase):
    W = pd.Series({"A": .5, "B": .3, "C": .2})

    def _state(self, daily, now=None):
        price_df = sample_monthly(daily)
        return lc.shadow_state(STRAT_RD_MIXED, price_df, self.W, 0.8, 1.25, band_mode=BAND_MODE_RATIO, now=now), price_df

    def test_complete_month_uses_engine_last_row(self):
        st, price_df = self._state(_daily("2026-09-30"))
        self.assertTrue(st["month_complete"])
        self.assertEqual(st["ref_date"], st["as_of"])
        res, *_ = run_detailed_backtest(STRAT_RD_MIXED, price_df[list(self.W.index)], self.W, 10000, 0.8,
                                        threshold_up=1.25, band_mode=BAND_MODE_RATIO)
        last = res.iloc[-1]
        for t in self.W.index:
            self.assertAlmostEqual(st["elem_weights"][t], float(str(last[t]).rstrip("%")) / 100, places=4)

    def test_last_business_day_before_close_is_not_complete(self):
        st, _ = self._state(_daily("2026-09-30"), now=pd.Timestamp("2026-09-30 11:00"))
        self.assertFalse(st["month_complete"])
        st, _ = self._state(_daily("2026-09-30"), now=pd.Timestamp("2026-09-30 17:00"))
        self.assertTrue(st["month_complete"])

    def test_partial_month_drifts_previous_month_end(self):
        daily = _daily("2026-09-15")
        st, price_df = self._state(daily)
        self.assertFalse(st["month_complete"])
        self.assertEqual(st["ref_date"], price_df.index[-2])
        # manual drift of the month-end state
        res, cnt, _, s0 = run_detailed_backtest(STRAT_RD_MIXED, price_df[list(self.W.index)].iloc[:-1], self.W, 10000,
                                                0.8, threshold_up=1.25, band_mode=BAND_MODE_RATIO, return_stats=True)
        v = pd.Series(s0["last_values"]) * price_df[list(self.W.index)].iloc[-1] / price_df[list(self.W.index)].iloc[-2]
        pd.testing.assert_series_equal(st["elem_weights"], (v / v.sum()).reindex(self.W.index), check_names=False)

    def test_partial_bar_breach_is_only_a_preview(self):
        daily = _daily("2026-09-15", jump=4.0)                      # A quadruples on the last (mid-month) day
        st, price_df = self._state(daily)
        self.assertIsNotNone(st["preview_event"])
        self.assertEqual(pd.Timestamp(st["preview_event"]["date"]), st["as_of"])
        self.assertTrue(all(pd.Timestamp(e["date"]) <= st["ref_date"] for e in st["events"]))   # the preview is not an event
        self.assertGreater(st["elem_weights"]["A"], 0.6)            # not reset: the account waits for month-end

    def test_slot_table_trigger_lines(self):
        st, _ = self._state(_daily("2026-09-30"))
        row = st["slot_table"].set_index("slot").loc["A"]
        self.assertAlmostEqual(row["trig_high"], .5 * 2.25 / (1 + .5 * 1.25), places=10)
        self.assertAlmostEqual(row["trig_low"], .5 * .2 / (1 - .5 * .8), places=10)


# --------------------------------------------------------------------------- #
# Advice
# --------------------------------------------------------------------------- #
class AdviceTest(unittest.TestCase):
    def _state(self, ref_event=None, complete=True, days=0):
        tab = pd.DataFrame([{"slot": "A", "label": "A", "members": ["A"], "weight": .5},
                            {"slot": "__c", "label": "ETH-USD+MSTR", "members": ["ETH-USD", "MSTR"], "weight": .05},
                            {"slot": "B", "label": "B", "members": ["B"], "weight": .45}])
        as_of = pd.Timestamp("2026-10-01") + pd.Timedelta(days=days)
        return {"slot_table": tab, "ref_event": ref_event, "month_complete": complete, "as_of": as_of,
                "ref_date": pd.Timestamp("2026-09-30")}

    def test_ok_when_close(self):
        v = S({"A": 50.2, "ETH-USD": 2.5, "MSTR": 2.5, "B": 44.8})
        self.assertEqual(lc.sync_advice(self._state(), v, crypto_slots=["__c"])["level"], "ok")

    def test_rebalance_when_shadow_reset(self):
        v = S({"A": 55, "ETH-USD": 2.5, "MSTR": 2.5, "B": 40})
        ev = {"date": pd.Timestamp("2026-09-30"), "scope": "global", "trigger": ["A"]}
        r = lc.sync_advice(self._state(ref_event=ev), v, crypto_slots=["__c"])
        self.assertEqual(r["level"], "rebalance")

    def test_crypto_relative_tolerance(self):
        v = S({"A": 49, "ETH-USD": 4.0, "MSTR": 3.0, "B": 44})    # crypto 7% vs 5% shadow: +40% relative
        self.assertEqual(lc.sync_advice(self._state(), v, crypto_slots=["__c"])["level"], "sync")
        self.assertEqual(lc.sync_advice(self._state(complete=False, days=14), v, crypto_slots=["__c"])["level"], "watch")

    def test_crypto_slot_ids(self):
        self.assertEqual(lc.crypto_slot_ids(self._state()["slot_table"]), ["__c"])


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
FLEX = """<FlexQueryResponse><FlexStatements count="1"><FlexStatement accountId="U123" fromDate="20260921" toDate="20260921">
<OpenPositions>
<OpenPosition symbol="QQQM" assetCategory="STK" position="100" markPrice="250.5" positionValue="25050" fxRateToBase="1" reportDate="20260921" levelOfDetail="SUMMARY"/>
<OpenPosition symbol="QQQM" assetCategory="STK" position="100" markPrice="250.5" positionValue="25050" fxRateToBase="1" reportDate="20260921" levelOfDetail="LOT"/>
<OpenPosition symbol="BRK B" assetCategory="STK" position="10.5" markPrice="480" positionValue="5040" fxRateToBase="1" reportDate="20260921" levelOfDetail="SUMMARY"/>
<OpenPosition symbol="MSTR  261218C00500000" assetCategory="OPT" position="1" markPrice="12" positionValue="1200" fxRateToBase="1" reportDate="20260921" levelOfDetail="SUMMARY"/>
</OpenPositions></FlexStatement></FlexStatements></FlexQueryResponse>"""


class ImportTest(unittest.TestCase):
    def test_flex_positions(self):
        df, meta = lc.parse_ib_flex_positions(FLEX)
        self.assertEqual(meta["account"], "U123")
        d = df.set_index("symbol")
        self.assertEqual(sorted(d.index), ["BRK-B", "QQQM"])
        self.assertEqual(d.loc["QQQM", "shares"], 100)               # the LOT row is not double counted
        self.assertAlmostEqual(d.loc["BRK-B", "price"], 480.0)

    def test_csv_positions_header_variants(self):
        df = lc.parse_positions_csv("Financial Instrument,Position,Last\nQQQM,\"1,200\",250.5\nBRK.B,10,480\nX,0,1\n")
        d = df.set_index("symbol")
        self.assertEqual(d.loc["QQQM", "shares"], 1200)
        self.assertEqual(d.loc["BRK-B", "price"], 480)
        self.assertNotIn("X", d.index)
        with self.assertRaises(ValueError):
            lc.parse_positions_csv("foo,bar\n1,2\n")


# --------------------------------------------------------------------------- #
# Core helpers shared with the app (moved in v2.6.0)
# --------------------------------------------------------------------------- #
def _prices(start="2020-01-01", n=400, late=None):
    idx = pd.date_range(start, periods=n, freq="D")                # calendar days (crypto-like)
    df = pd.DataFrame({"SPY": np.linspace(100, 150, n), "A": np.linspace(50, 80, n), "B": np.linspace(20, 10, n),
                       "C": np.linspace(10, 30, n)}, index=idx)
    df.loc[df.index.dayofweek >= 5, "SPY"] = np.nan              # benchmark trades on weekdays only
    if late:
        df.loc[df.index < pd.Timestamp(late), "C"] = np.nan
    return df


class AlignPrepareTest(unittest.TestCase):
    def test_align_late_listing_moves_start(self):
        out = align_price_data(_prices(late="2020-03-02"), "SPY", "2020-01-01", ["A", "B", "C"])
        self.assertIsNone(out["error"])
        self.assertEqual(out["actual_start_day"], pd.Timestamp("2020-03-02"))
        self.assertEqual(out["notice"][0], "warning")
        self.assertIn("**C** listed late", out["notice"][1])
        self.assertEqual(out["price_df"].index[0], pd.Timestamp("2020-03-02"))

    def test_align_weekend_start_and_errors(self):
        out = align_price_data(_prices(), "SPY", "2020-01-04", ["A"])       # Saturday
        self.assertEqual(out["notice"], ("info", "Aligned to next trading day: 2020-01-06"))
        out = align_price_data(_prices(), "SPY", "2030-01-01", ["A"])
        self.assertIn("has no prices on or after", out["error"])

    def test_prepare_composite_with_a_dataless_member(self):
        df = _prices()
        df["D"] = np.nan
        price_df = align_price_data(df, "SPY", "2020-01-01", ["A", "B", "C", "D"])["price_df"]
        p = {"name": "P", "tickers": "A, (B, D), C", "weights": "0.5, 0.3, 0.2", "thr": 60, "thr_up": 100,
             "slot_bands": {"B+D": {"down": 40, "up": 40}}}
        tks, wts, errs, comp = parse_portfolio(p)
        out = prepare_portfolio(p, tks, wts, comp, price_df)
        self.assertIsNone(out["error"])
        self.assertEqual(out["valid_tks"], ["A", "B", "C"])
        self.assertAlmostEqual(out["w_series"]["B"], 0.3)           # the slot's 30% goes to the survivor
        self.assertIsNone(out["groups"])                            # one survivor -> singleton
        self.assertTrue(any("dropped from their composite" in m for _, m in out["notices"]))
        self.assertEqual(out["label_to_id"]["B+D"], "B")
        self.assertEqual(out["thr_dn"], {"*": 0.6, "B": 0.4})

    def test_prepare_dropped_slot_renormalises(self):
        df = _prices()
        df["D"] = np.nan
        price_df = align_price_data(df, "SPY", "2020-01-01", ["A", "D"])["price_df"]
        p = {"name": "P", "tickers": "A, D", "weights": "0.6, 0.4", "thr": 40}
        tks, wts, errs, comp = parse_portfolio(p)
        out = prepare_portfolio(p, tks, wts, comp, price_df)
        self.assertAlmostEqual(out["w_series"]["A"], 1.0)
        self.assertTrue(any("remaining weights renormalized" in m for _, m in out["notices"]))
        p2 = {"name": "Q", "tickers": "D", "weights": "1.0", "thr": 40}
        tks, wts, errs, comp = parse_portfolio(p2)
        self.assertIn("no usable data", prepare_portfolio(p2, tks, wts, comp, price_df)["error"])


if __name__ == "__main__":
    unittest.main()


class SmallHelpersTest(unittest.TestCase):
    def test_next_month_end(self):
        self.assertEqual(lc.next_month_end("2026-09-24", False), pd.Timestamp("2026-09-30"))
        self.assertEqual(lc.next_month_end("2026-09-30", True), pd.Timestamp("2026-10-30"))
        self.assertEqual(lc.next_month_end("2026-09-30", False), pd.Timestamp("2026-09-30"))

    def test_last_reset_date(self):
        ev = [{"date": pd.Timestamp("2022-10-31"), "scope": "global", "trigger": ["XLE"]},
              {"date": pd.Timestamp("2026-02-27"), "scope": "local", "trigger": ["__c"]},
              {"date": pd.Timestamp("2026-05-29"), "scope": "local", "trigger": ["__d"]}]
        self.assertEqual(lc.last_reset_date(ev, "__c"), pd.Timestamp("2026-02-27"))
        self.assertEqual(lc.last_reset_date(ev, "QQQM"), pd.Timestamp("2022-10-31"))
        self.assertIsNone(lc.last_reset_date([], "QQQM"))

    def test_ibkr_commission(self):
        o = pd.DataFrame({"shares": [10, 1000, 1], "amount": [5000.0, 30000.0, 50.0]})
        self.assertAlmostEqual(lc.ibkr_commission(o), 1.0 + 5.0 + 0.5)       # $1 min, 1000*0.005, 1% cap on $50
        self.assertEqual(lc.ibkr_commission(o.iloc[0:0]), 0.0)
