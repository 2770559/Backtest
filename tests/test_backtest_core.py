"""Unit tests for backtest_core. Run with:  python3 -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_core import (
    STRAT_BH, STRAT_ANNUAL, STRAT_RD_FULL,
    clean_ticker, parse_portfolio, calculate_metrics,
    run_detailed_backtest, compute_annual_returns,
    scrub_leading_glitches, scrub_isolated_spikes,
)


class TestCleanTicker(unittest.TestCase):
    def test_mapping_and_normalization(self):
        self.assertEqual(clean_ticker(" brk.b "), "BRK-B")
        self.assertEqual(clean_ticker("ethusd"), "ETH-USD")
        self.assertEqual(clean_ticker("btcusd"), "BTC-USD")
        self.assertEqual(clean_ticker("qqqm"), "QQQM")
        self.assertEqual(clean_ticker("159941.sz"), "159941.SZ")


class TestParsePortfolio(unittest.TestCase):
    def test_valid(self):
        tks, wts, errs = parse_portfolio({"tickers": "QQQM, SPY", "weights": "0.6, 0.4"})
        self.assertEqual(tks, ["QQQM", "SPY"])
        self.assertEqual(wts, [0.6, 0.4])
        self.assertEqual(errs, [])

    def test_fullwidth_comma(self):
        tks, wts, errs = parse_portfolio({"tickers": "QQQM，SPY", "weights": "0.5，0.5"})
        self.assertEqual(tks, ["QQQM", "SPY"])
        self.assertEqual(errs, [])

    def test_trailing_comma_ignored(self):
        tks, wts, errs = parse_portfolio({"tickers": "QQQM, SPY,", "weights": "0.5, 0.5"})
        self.assertEqual(tks, ["QQQM", "SPY"])
        self.assertEqual(errs, [])

    def test_count_mismatch(self):
        _, _, errs = parse_portfolio({"tickers": "QQQM, SPY, GLD", "weights": "0.5, 0.5"})
        self.assertTrue(any("3 tickers vs 2 weights" in e for e in errs))

    def test_invalid_float(self):
        _, _, errs = parse_portfolio({"tickers": "QQQM, SPY", "weights": "0.5, abc"})
        self.assertTrue(any("invalid weight format" in e for e in errs))

    def test_weight_sum(self):
        _, _, errs = parse_portfolio({"tickers": "QQQM, SPY", "weights": "0.5, 0.6"})
        self.assertTrue(any("should be 1.0" in e for e in errs))

    def test_duplicate_tickers(self):
        _, _, errs = parse_portfolio({"tickers": "SPY, spy", "weights": "0.5, 0.5"})
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
