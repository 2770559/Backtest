"""Unit tests for backtest_core. Run with:  python3 -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_core import (
    STRAT_BH, STRAT_ANNUAL, STRAT_RD_FULL, STRAT_ASYM, STRAT_ASYM_LOCAL,
    STRAT_ASYM_LOCAL_EQ, STRAT_ASYM_LOCAL_PROP,
    clean_ticker, parse_portfolio, calculate_metrics,
    run_detailed_backtest, compute_annual_returns,
    scrub_leading_glitches, scrub_isolated_spikes, sample_monthly,
)


class TestCleanTicker(unittest.TestCase):
    def test_mapping_and_normalization(self):
        self.assertEqual(clean_ticker(" brk.b "), "BRK-B")
        self.assertEqual(clean_ticker("ethusd"), "ETH-USD")
        self.assertEqual(clean_ticker("btcusd"), "BTC-USD")
        self.assertEqual(clean_ticker("qqqm"), "QQQM")
        self.assertEqual(clean_ticker("159941.sz"), "159941.SZ")

    def test_hk_code_normalization(self):
        # 5-digit / leading-zero HK codes -> Yahoo's canonical 4-digit form
        self.assertEqual(clean_ticker("00700.HK"), "0700.HK")
        self.assertEqual(clean_ticker("09992.hk"), "9992.HK")
        self.assertEqual(clean_ticker("700.HK"), "0700.HK")
        self.assertEqual(clean_ticker("0700.HK"), "0700.HK")   # already canonical
        self.assertEqual(clean_ticker("9988.HK"), "9988.HK")
        self.assertEqual(clean_ticker("80737.HK"), "80737.HK")  # genuine 5-digit untouched
        self.assertEqual(clean_ticker("600519.SS"), "600519.SS")  # non-HK unaffected


class TestParsePortfolio(unittest.TestCase):
    def test_valid(self):
        tks, wts, errs, _ = parse_portfolio({"tickers": "QQQM, SPY", "weights": "0.6, 0.4"})
        self.assertEqual(tks, ["QQQM", "SPY"])
        self.assertEqual(wts, [0.6, 0.4])
        self.assertEqual(errs, [])

    def test_fullwidth_comma(self):
        tks, wts, errs, _ = parse_portfolio({"tickers": "QQQM，SPY", "weights": "0.5，0.5"})
        self.assertEqual(tks, ["QQQM", "SPY"])
        self.assertEqual(errs, [])

    def test_trailing_comma_ignored(self):
        tks, wts, errs, _ = parse_portfolio({"tickers": "QQQM, SPY,", "weights": "0.5, 0.5"})
        self.assertEqual(tks, ["QQQM", "SPY"])
        self.assertEqual(errs, [])

    def test_count_mismatch(self):
        _, _, errs, _ = parse_portfolio({"tickers": "QQQM, SPY, GLD", "weights": "0.5, 0.5"})
        self.assertTrue(any("3 tickers vs 2 weights" in e for e in errs))

    def test_invalid_float(self):
        _, _, errs, _ = parse_portfolio({"tickers": "QQQM, SPY", "weights": "0.5, abc"})
        self.assertTrue(any("invalid weight format" in e for e in errs))

    def test_weight_sum(self):
        _, _, errs, _ = parse_portfolio({"tickers": "QQQM, SPY", "weights": "0.5, 0.6"})
        self.assertTrue(any("should be 1.0" in e for e in errs))

    def test_duplicate_tickers(self):
        _, _, errs, _ = parse_portfolio({"tickers": "SPY, spy", "weights": "0.5, 0.5"})
        self.assertTrue(any("duplicate tickers" in e for e in errs))


class TestCalculateMetrics(unittest.TestCase):
    def test_empty(self):
        m = calculate_metrics(pd.Series(dtype=float), 0)
        self.assertEqual(m["total_ret"], "-")

    def test_doubling_one_year(self):
        idx = pd.date_range("2020-01-01", "2021-01-01", freq="D")
        nav = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
        m = calculate_metrics(nav, 0)
        self.assertAlmostEqual(m["_total_ret"], 1.0, places=6)
        # 366 days elapsed -> annualized slightly under 100%
        self.assertAlmostEqual(m["_ann_ret"], 2 ** (365.25 / 366) - 1, places=4)
        self.assertEqual(m["_max_dd"], 0)

    def test_max_drawdown(self):
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        nav = pd.Series([100, 120, 90, 130], index=idx)
        m = calculate_metrics(nav, 0)
        self.assertAlmostEqual(m["_max_dd"], (90 - 120) / 120, places=6)


def _make_price_df(prices_a, prices_b, freq="ME", start="2020-01-31"):
    idx = pd.date_range(start, periods=len(prices_a), freq=freq)
    return pd.DataFrame({"A": prices_a, "B": prices_b}, index=idx)


class TestRunDetailedBacktest(unittest.TestCase):
    def setUp(self):
        self.weights = pd.Series([0.5, 0.5], index=["A", "B"])

    def test_buy_and_hold(self):
        price_df = _make_price_df([100, 150, 200], [100, 100, 100])
        hist, cnt, pnl = run_detailed_backtest(STRAT_BH, price_df, self.weights, 10000, 0.38)
        self.assertEqual(cnt, 0)
        # 50 shares A + 50 shares B -> 50*200 + 50*100
        self.assertAlmostEqual(hist.iloc[-1]["NAV"], 15000, places=6)
        self.assertAlmostEqual(pnl["NAV"], 5000, places=6)

    def test_periodic_annual_triggers_on_constant_prices(self):
        n = 25  # monthly bars spanning two years
        price_df = _make_price_df([100] * n, [100] * n)
        _, cnt, _ = run_detailed_backtest(STRAT_ANNUAL, price_df, self.weights, 10000, 0.38)
        self.assertEqual(cnt, 2)

    def test_reldiff_full_triggers_and_restores_targets(self):
        # A quadruples -> weights 0.8/0.2 -> rel_diff 0.6 > 0.5 threshold
        price_df = _make_price_df([100, 400, 400], [100, 100, 100])
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, price_df, self.weights, 10000, 0.5)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], "50.00%")
        self.assertEqual(post["B"], "50.00%")

    def test_reldiff_full_no_trigger_below_threshold(self):
        # A doubles -> weights 2/3 vs 1/3 -> rel_diff 0.333 < 0.5
        price_df = _make_price_df([100, 200, 200], [100, 100, 100])
        _, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, price_df, self.weights, 10000, 0.5)
        self.assertEqual(cnt, 0)


class TestAsymLocal(unittest.TestCase):
    """Asymmetric RelDiff Local: a minor-only breach trades ONLY against the
    major (>= 6%) slots, never pushing a major past its own target (cascading
    down the deviation ranking); non-triggered minors are untouched. A major
    breach still resets globally, bit-identical to Asymmetric RelDiff."""

    @staticmethod
    def _df(cols, start="2020-01-31"):
        n = len(next(iter(cols.values())))
        idx = pd.date_range(start, periods=n, freq="ME")
        return pd.DataFrame(cols, index=idx)

    @staticmethod
    def _pct(cell):
        return float(cell.rstrip("%"))

    def test_minor_trim_cascades_and_caps_at_major_targets(self):
        # A .60 / B .30 / C .05(minor) / D .05(minor). C moons (minor-up -> trim).
        # Its excess is pushed to the under-weight majors, most-deviated first,
        # each capped at its target: A fills to 60% then B to 30% (cascade), and
        # C is only PARTIALLY trimmed (5.93%, not 5%) because the majors fill up.
        # D (a non-triggered minor) is left completely untouched.
        w = pd.Series([0.60, 0.30, 0.05, 0.05], index=["A", "B", "C", "D"])
        price = self._df({"A": [100, 100], "B": [100, 110],
                          "C": [100, 500], "D": [100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_ASYM_LOCAL, price, w, 10000, 0.5)
        self.assertEqual(cnt, 1)
        pre = hist[hist["Type"] == "Pre-Rebal"].iloc[0]
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], "60.00%")              # capped at target, not past
        self.assertEqual(post["B"], "30.00%")              # cascade filled B too, capped
        self.assertEqual(post["D"], pre["D"])              # non-triggered minor untouched
        self.assertGreater(self._pct(post["C"]), 5.0)      # only partially trimmed
        self.assertLess(self._pct(post["C"]), self._pct(pre["C"]))
        self.assertAlmostEqual(pre["NAV"], post["NAV"], places=6)

    def test_minor_topup_cascades_and_caps_at_major_targets(self):
        # A .60 / B .30 / C .05(minor) / D .05(minor). C crashes (minor-down ->
        # top-up). Cash is pulled from the over-weight majors, most-deviated
        # first, each capped so it does not fall below target: A drops to 60%
        # then B to 30% (cascade), and C is only PARTIALLY topped up (4.76%).
        # D is over-weight but a non-triggered minor -> it is NOT used as a
        # source and stays put (proves minors only trade with majors).
        w = pd.Series([0.60, 0.30, 0.05, 0.05], index=["A", "B", "C", "D"])
        price = self._df({"A": [100, 100], "B": [100, 100],
                          "C": [100, 10], "D": [100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_ASYM_LOCAL, price, w, 10000, 0.5)
        self.assertEqual(cnt, 1)
        pre = hist[hist["Type"] == "Pre-Rebal"].iloc[0]
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], "60.00%")              # capped at target, not below
        self.assertEqual(post["B"], "30.00%")              # cascade drained B too, capped
        self.assertEqual(post["D"], pre["D"])              # over-weight minor NOT tapped
        self.assertLess(self._pct(post["C"]), 5.0)         # only partially topped up
        self.assertGreater(self._pct(post["C"]), self._pct(pre["C"]))
        self.assertAlmostEqual(pre["NAV"], post["NAV"], places=6)

    def test_two_minors_do_not_trade_with_each_other(self):
        # A .80 / B .10 / C .05 / D .05. C moons (trim), D crashes (top-up) in the
        # same bar. Under the old design they would net against each other; now C
        # can ONLY offload to majors (fills A->80%, B->10%, stuck at 9.10%) and D
        # can ONLY source from majors (none over-weight left -> D gets nothing,
        # stays 0.90%). The minors never touch each other.
        w = pd.Series([0.80, 0.10, 0.05, 0.05], index=["A", "B", "C", "D"])
        price = self._df({"A": [100, 100], "B": [100, 100],
                          "C": [100, 400], "D": [100, 20]})
        hist, cnt, _ = run_detailed_backtest(STRAT_ASYM_LOCAL, price, w, 10000, 0.5)
        self.assertEqual(cnt, 1)
        pre = hist[hist["Type"] == "Pre-Rebal"].iloc[0]
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], "80.00%")              # majors filled to target
        self.assertEqual(post["B"], "10.00%")
        self.assertEqual(post["D"], pre["D"])              # under-weight minor got nothing
        self.assertGreater(self._pct(post["C"]), 5.0)      # over-weight minor still stuck > target
        self.assertAlmostEqual(pre["NAV"], post["NAV"], places=6)

    def test_minor_fully_corrected_without_global_reset(self):
        # A .70 / B .25 / C .05. A drifts over-weight, B under-weight, C moons.
        # C fully trims to 5% into the sole under-weight major B, which stays
        # BELOW its 25% target (room to spare, cap not hit). The over-weight major
        # A is never a trim recipient -> untouched, proving this is not a reset.
        w = pd.Series([0.70, 0.25, 0.05], index=["A", "B", "C"])
        price = self._df({"A": [100, 105], "B": [100, 95], "C": [100, 150]})
        hist, cnt, _ = run_detailed_backtest(STRAT_ASYM_LOCAL, price, w, 10000, 0.1)
        self.assertEqual(cnt, 1)
        pre = hist[hist["Type"] == "Pre-Rebal"].iloc[0]
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["C"], "5.00%")               # fully trimmed
        self.assertEqual(post["A"], pre["A"])              # over-weight major untouched
        self.assertGreater(self._pct(post["B"]), self._pct(pre["B"]))  # B received it
        self.assertLess(self._pct(post["B"]), 25.0)        # ... but did NOT overshoot target
        self.assertAlmostEqual(pre["NAV"], post["NAV"], places=6)

    def test_major_breach_falls_back_to_global_reset(self):
        # A (major) breaches its symmetric band -> global reset, identical to
        # Asymmetric RelDiff. Post weights snap to target and the two strategies
        # produce the same history.
        w = pd.Series([0.50, 0.45, 0.05], index=["A", "B", "C"])
        price = self._df({"A": [100, 400], "B": [100, 100], "C": [100, 100]})
        loc = run_detailed_backtest(STRAT_ASYM_LOCAL, price, w, 10000, 0.3)
        asym = run_detailed_backtest(STRAT_ASYM, price, w, 10000, 0.3)
        pd.testing.assert_frame_equal(loc[0], asym[0], check_exact=True)
        post = loc[0][loc[0]["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"], post["C"]), ("50.00%", "45.00%", "5.00%"))

    def test_in_band_does_not_trigger(self):
        # Nothing breaches its band -> no rebalance at all.
        w = pd.Series([0.70, 0.25, 0.05], index=["A", "B", "C"])
        price = self._df({"A": [100, 105], "B": [100, 98], "C": [100, 103]})
        _, cnt, _ = run_detailed_backtest(STRAT_ASYM_LOCAL, price, w, 10000, 0.5)
        self.assertEqual(cnt, 0)


class TestAsymLocalVariants(unittest.TestCase):
    """Equal / Prop distribution variants of Asymmetric RelDiff Local.

    Same trigger geometry; they differ ONLY in how a triggered minor's
    correction is split across the majors: equal dollar shares vs pro-rata to
    current value — in both cases across ALL majors with NO target cap, so the
    minor is always FULLY corrected (contrast: rank mode caps each major at its
    own target and may leave the minor partially corrected)."""

    W = ["A", "B", "C", "D"]  # A .60 / B .30 majors; C, D .05 minors

    @staticmethod
    def _df(cols, start="2020-01-31"):
        n = len(next(iter(cols.values())))
        idx = pd.date_range(start, periods=n, freq="ME")
        return pd.DataFrame(cols, index=idx)

    @staticmethod
    def _pct(cell):
        return float(cell.rstrip("%"))

    def _run(self, strat, c_path):
        w = pd.Series([0.60, 0.30, 0.05, 0.05], index=self.W)
        price = self._df({"A": [100, 100], "B": [100, 100],
                          "C": c_path, "D": [100, 100]})
        hist, cnt, _ = run_detailed_backtest(strat, price, w, 10000, 0.5)
        self.assertEqual(cnt, 1)
        pre = hist[hist["Type"] == "Pre-Rebal"].iloc[0]
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertAlmostEqual(pre["NAV"], post["NAV"], places=6)
        self.assertEqual(post["D"], pre["D"])   # non-triggered minor untouched
        return pre, post

    def test_equal_trim_splits_evenly_and_ignores_target_cap(self):
        # C moons 100->500: values A 6000 / B 3000 / C 2500 / D 500, total 12000.
        # Excess 1900 pushed 950/950 to A and B. B crosses its 30% target
        # (32.92% > 30%) — rank mode would cap B at 30% and leave C partial.
        _, post = self._run(STRAT_ASYM_LOCAL_EQ, [100, 500])
        self.assertEqual(post["C"], "5.00%")               # fully trimmed
        self.assertEqual(post["A"], "57.92%")              # 6950/12000
        self.assertEqual(post["B"], "32.92%")              # 3950/12000 — past target
        self.assertGreater(self._pct(post["B"]), 30.0)

    def test_prop_trim_splits_pro_rata_no_cap(self):
        # Same breach; 1900 split 2:1 by current value (A 6000 : B 3000).
        # A crosses its own 60% target (60.56%) — impossible under rank mode.
        _, post = self._run(STRAT_ASYM_LOCAL_PROP, [100, 500])
        self.assertEqual(post["C"], "5.00%")
        self.assertEqual(post["A"], "60.56%")              # 7266.67/12000
        self.assertEqual(post["B"], "30.28%")              # 3633.33/12000
        self.assertGreater(self._pct(post["A"]), 60.0)

    def test_equal_topup_pulls_evenly_even_below_target(self):
        # C crashes 100->10: values A 6000 / B 3000 / C 50 / D 500, total 9550.
        # Shortfall 427.5 pulled 213.75 from each; B ends BELOW its 30% target
        # (29.18%) — rank mode would stop at target and top C up only partially.
        _, post = self._run(STRAT_ASYM_LOCAL_EQ, [100, 10])
        self.assertEqual(post["C"], "5.00%")               # fully topped up
        self.assertEqual(post["A"], "60.59%")              # 5786.25/9550
        self.assertEqual(post["B"], "29.18%")              # 2786.25/9550 — below target
        self.assertLess(self._pct(post["B"]), 30.0)

    def test_prop_topup_pulls_pro_rata(self):
        # Same crash; 427.5 pulled 2:1 (A 285, B 142.5).
        _, post = self._run(STRAT_ASYM_LOCAL_PROP, [100, 10])
        self.assertEqual(post["C"], "5.00%")
        self.assertEqual(post["A"], "59.84%")              # 5715/9550
        self.assertEqual(post["B"], "29.92%")              # 2857.5/9550

    def test_major_breach_still_resets_globally(self):
        # A major breach must behave exactly like Asymmetric RelDiff for all
        # three local variants.
        w = pd.Series([0.50, 0.45, 0.05], index=["A", "B", "C"])
        price = self._df({"A": [100, 400], "B": [100, 100], "C": [100, 100]})
        ref = run_detailed_backtest(STRAT_ASYM, price, w, 10000, 0.3)
        for strat in (STRAT_ASYM_LOCAL_EQ, STRAT_ASYM_LOCAL_PROP):
            got = run_detailed_backtest(strat, price, w, 10000, 0.3)
            pd.testing.assert_frame_equal(got[0], ref[0], check_exact=True)

    def test_in_band_does_not_trigger(self):
        w = pd.Series([0.70, 0.25, 0.05], index=["A", "B", "C"])
        price = self._df({"A": [100, 105], "B": [100, 98], "C": [100, 103]})
        for strat in (STRAT_ASYM_LOCAL_EQ, STRAT_ASYM_LOCAL_PROP):
            _, cnt, _ = run_detailed_backtest(strat, price, w, 10000, 0.5)
            self.assertEqual(cnt, 0)


class TestSampleMonthly(unittest.TestCase):
    def test_final_partial_month_keeps_real_last_date(self):
        idx = pd.date_range("2020-01-02", "2020-07-02", freq="B")
        df = pd.DataFrame({"A": np.arange(len(idx), dtype=float)}, index=idx)
        out = sample_monthly(df)
        self.assertEqual(out.index[-1], idx[-1])           # NOT future 2020-07-31
        self.assertEqual(out.iloc[-1]["A"], df.iloc[-1]["A"])
        self.assertEqual(out.index[0], idx[0])             # first real row kept
        self.assertIn(pd.Timestamp("2020-03-31"), out.index)  # interior EOM labels kept
        self.assertFalse(out.index.duplicated().any())

    def test_data_ending_exactly_on_month_end_unchanged(self):
        idx = pd.date_range("2020-01-02", "2020-06-30", freq="B")
        df = pd.DataFrame({"A": np.arange(len(idx), dtype=float)}, index=idx)
        out = sample_monthly(df)
        self.assertEqual(out.index[-1], pd.Timestamp("2020-06-30"))
        self.assertFalse(out.index.duplicated().any())


class TestPnlNanHardening(unittest.TestCase):
    def test_leading_nan_prices_do_not_poison_pnl(self):
        # B lists two bars late (leading NaNs). Its NaN price diffs must not
        # turn the cumulative PnL (and thus every contribution pct) into NaN.
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        df = pd.DataFrame({"A": [100.0, 110.0, 120.0, 130.0],
                           "B": [np.nan, np.nan, 100.0, 110.0]}, index=idx)
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        _, _, pnl = run_detailed_backtest(STRAT_BH, df, w, 10000, 0.5)
        self.assertFalse(np.isnan(pnl["NAV"]))
        self.assertNotIn("nan", str(pnl["A"]) + str(pnl["B"]))


class TestScrubbing(unittest.TestCase):
    def test_leading_glitch_dropped(self):
        idx = pd.date_range("2023-08-01", periods=4, freq="D")
        df = pd.DataFrame({"X": [0.97, 97.0, 98.0, 99.0]}, index=idx)
        notes = scrub_leading_glitches(df)
        self.assertEqual(len(notes), 1)
        self.assertTrue(np.isnan(df["X"].iloc[0]))
        self.assertEqual(df["X"].dropna().iloc[0], 97.0)

    def test_clean_series_untouched(self):
        idx = pd.date_range("2023-08-01", periods=4, freq="D")
        df = pd.DataFrame({"X": [100.0, 102.0, 101.0, 103.0]}, index=idx)
        self.assertEqual(scrub_leading_glitches(df), [])
        self.assertEqual(scrub_isolated_spikes(df), [])
        self.assertFalse(df["X"].isna().any())

    def test_isolated_spike_dropped(self):
        idx = pd.date_range("2023-08-01", periods=5, freq="D")
        df = pd.DataFrame({"X": [100.0, 101.0, 10000.0, 102.0, 103.0]}, index=idx)
        notes = scrub_isolated_spikes(df)
        self.assertEqual(len(notes), 1)
        self.assertTrue(np.isnan(df["X"].iloc[2]))

    def test_genuine_crash_untouched(self):
        # A real crash persists across prints — must not be scrubbed
        idx = pd.date_range("2023-08-01", periods=5, freq="D")
        df = pd.DataFrame({"X": [100.0, 100.0, 15.0, 14.0, 15.0]}, index=idx)
        self.assertEqual(scrub_isolated_spikes(df), [])
        self.assertFalse(df["X"].isna().any())


class TestComputeAnnualReturns(unittest.TestCase):
    def test_two_years_with_partial_last(self):
        idx = pd.date_range("2020-01-02", "2021-06-30", freq="D")
        comp = pd.DataFrame({"P": np.linspace(100, 200, len(idx))}, index=idx)
        rows = compute_annual_returns(comp)
        self.assertEqual([r["year"] for r in rows], [2020, 2021])
        self.assertFalse(rows[0]["partial"])  # starts Jan 2 -> full year
        self.assertTrue(rows[1]["partial"])   # ends June 30 -> partial
        # chained yearly returns must reproduce the total return
        total = (1 + rows[0]["returns"]["P"]) * (1 + rows[1]["returns"]["P"]) - 1
        self.assertAlmostEqual(total, 1.0, places=6)


class TestAppSmoke(unittest.TestCase):
    """Headless render of the Streamlit app (no network: backtest not triggered)."""

    APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtest_app.py")

    def test_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(self.APP)
        at.run(timeout=60)
        self.assertFalse(at.exception)

    def test_invalid_input_after_run_shows_error_not_crash(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(self.APP)
        at.run(timeout=60)
        # Simulate: user already ran a backtest, then edits weights to garbage.
        # Re-validation must stop with st.error instead of raising on float().
        pid = at.session_state["portfolios_list"][0]["id"]
        at.text_input(key=f"w_{pid}").set_value("0.5, abc")
        at.session_state["run_backtest"] = True
        at.run(timeout=60)
        self.assertFalse(at.exception)
        self.assertGreater(len(at.error), 0)


if __name__ == "__main__":
    unittest.main()
