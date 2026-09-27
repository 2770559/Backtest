"""Unit tests for backtest_core. Run with:  python3 -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from backtest_core import (
    align_price_data, prepare_portfolio,
    STRAT_BH, STRAT_ANNUAL, STRAT_RD_FULL, STRAT_ASYM, STRAT_RD_MIXED,
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

    def test_negative_weight_rejected(self):
        # Sums to 1.0, so the sum check alone let a short position through.
        _, _, errs, _ = parse_portfolio({"tickers": "QQQM, SPY", "weights": "-0.2, 1.2"})
        self.assertIn("negative weight(s): QQQM", errs)
        _, _, errs, _ = parse_portfolio({"tickers": "(A, B), SPY", "weights": "-0.1, 1.1"})
        self.assertIn("negative weight(s): A+B", errs)

    def test_non_finite_weight_rejected(self):
        # float("nan") parses, and a NaN sum passed the |sum - 1| check.
        for w in ("nan, 1.0", "inf, 0"):
            _, _, errs, _ = parse_portfolio({"tickers": "QQQM, SPY", "weights": w})
            self.assertIn("invalid weight format", errs, w)


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

    @staticmethod
    def _alternating_nav(mean_ann, vol_ann, periods_per_year, n, freq, start):
        """NAV whose periodic returns alternate mean ± vol: an exact mean and volatility."""
        m, s = mean_ann / periods_per_year, vol_ann / np.sqrt(periods_per_year)
        r = np.tile([m + s, m - s], n // 2)
        idx = pd.date_range(start, periods=n + 1, freq=freq)
        return pd.Series(10000 * np.concatenate([[1.0], np.cumprod(1 + r)]), index=idx)

    def test_sharpe_is_mean_excess_return_over_volatility(self):
        """Conventional Sharpe = (mean periodic return x periods/yr - rf) / annualised vol.
        The CAGR numerator it replaced fell with volatility: at 40% vol a 10%/yr mean
        scored 0.00 instead of ~0.20."""
        for vol in (0.10, 0.40):
            nav = self._alternating_nav(0.10, vol, 12, 240, "ME", "2005-12-31")
            m = calculate_metrics(nav, 0, risk_free_rate=0.02)
            r = nav.pct_change().dropna()
            expect = (r.mean() * 12 - 0.02) / (r.std() * np.sqrt(12))
            self.assertAlmostEqual(m["_sharpe"], expect, places=12, msg=vol)
            self.assertAlmostEqual(m["_sharpe"], 0.08 / vol, places=2, msg=vol)
            self.assertEqual(m["sharpe"], f"{expect:.2f}")
        cagr_based = (m["_ann_ret"] - 0.02) / (r.std() * np.sqrt(12))
        self.assertLess(cagr_based, 0.05)          # what the old formula reported at 40% vol

    def test_sharpe_uses_the_periodicity_of_the_volatility(self):
        nav = self._alternating_nav(0.12, 0.20, 252, 200, "B", "2020-01-01")   # daily bars (< 90-day path)
        m = calculate_metrics(nav, 0, risk_free_rate=0.02)
        r = nav.pct_change().dropna()
        self.assertAlmostEqual(m["_sharpe"], (r.mean() * 252 - 0.02) / (r.std() * np.sqrt(252)), places=12)

    def test_sharpe_zero_without_volatility(self):
        idx = pd.date_range("2020-01-31", periods=13, freq="ME")
        m = calculate_metrics(pd.Series(100.0, index=idx), 0)
        self.assertEqual(m["_sharpe"], 0)


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


class TestPerSlotThreshold(unittest.TestCase):
    """threshold may be a dict {slot_id: thr, '*': default} for per-slot bands."""

    @staticmethod
    def _df(cols, start="2020-01-31"):
        n = len(next(iter(cols.values())))
        idx = pd.date_range(start, periods=n, freq="ME")
        return pd.DataFrame(cols, index=idx)

    def test_scalar_equals_uniform_dict_bit_identical(self):
        w = pd.Series([0.6, 0.3, 0.1], index=["A", "B", "C"])
        price = self._df({"A": [100, 130, 90, 120], "B": [100, 95, 105, 80],
                          "C": [100, 160, 60, 130]})
        from backtest_core import STRAT_RD_LOCAL
        for strat in (STRAT_RD_FULL, STRAT_RD_MIXED, STRAT_RD_LOCAL, STRAT_ASYM):
            a = run_detailed_backtest(strat, price, w, 10000, 0.4)
            b = run_detailed_backtest(strat, price, w, 10000, {"*": 0.4})
            pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
            self.assertEqual(a[1], b[1])

    def test_tight_band_on_one_slot_triggers_alone(self):
        # C +30%: inside the global 40% band, outside its own 20% band.
        # With {'C': .2, '*': .4} the portfolio must rebalance; with scalar .4 not.
        w = pd.Series([0.6, 0.3, 0.1], index=["A", "B", "C"])
        price = self._df({"A": [100, 100], "B": [100, 100], "C": [100, 130]})
        _, cnt_scalar, _ = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.4)
        _, cnt_dict, _ = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, {"C": 0.2, "*": 0.4})
        self.assertEqual(cnt_scalar, 0)
        self.assertEqual(cnt_dict, 1)

    def test_missing_slot_without_default_raises(self):
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        price = self._df({"A": [100, 100], "B": [100, 100]})
        with self.assertRaises(ValueError):
            run_detailed_backtest(STRAT_RD_FULL, price, w, 10000, {"A": 0.4})


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
        # Weight-FORMAT garbage can no longer be typed (the allocation matrix is
        # a typed editor); an unparseable weight in a loaded config degrades to
        # "not held", leaving a weights-sum violation. The re-validation gate
        # must stop with st.error instead of crashing the results block.
        at.session_state["portfolios_list"] = [{
            "id": "smoke1", "name": "P1", "tickers": "QQQM, SPY",
            "weights": "0.5, abc", "strat": "RelDiff Mixed", "thr": 40,
        }]
        at.session_state["run_backtest"] = True
        at.run(timeout=60)
        self.assertFalse(at.exception)
        self.assertGreater(len(at.error), 0)


# --------------------------------------------------------------------------- #
# Alignment and per-portfolio preparation (moved from the app in v2.6.0)
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

    def test_prepare_scales_weights_to_100(self):
        price_df = align_price_data(_prices(), "SPY", "2020-01-01", ["A", "B", "C"])["price_df"]
        p = {"name": "P", "tickers": "A, B, C", "weights": "0.335, 0.335, 0.335", "thr": 40}   # 100.5%: accepted
        tks, wts, errs, comp = parse_portfolio(p)
        self.assertEqual(errs, [])
        out = prepare_portfolio(p, tks, wts, comp, price_df)
        self.assertAlmostEqual(out["w_series"].sum(), 1.0, places=12)   # a rebalance keeps the portfolio's value
        self.assertTrue(any("scaled to 100%" in m for _, m in out["notices"]))
        p = {"name": "P", "tickers": "A, B, C", "weights": "0.4, 0.3, 0.3", "thr": 40}
        tks, wts, errs, comp = parse_portfolio(p)
        out = prepare_portfolio(p, tks, wts, comp, price_df)
        self.assertEqual(list(out["w_series"]), [0.4, 0.3, 0.3])        # exactly 100%: untouched (legacy engine)
        self.assertFalse(any("scaled to 100%" in m for _, m in out["notices"]))

    def test_a_sum_the_matrix_shows_as_100_is_scaled_silently(self):
        """The notice appears exactly when the allocation matrix does not show "✓ 100%" (0.01 pp tolerance):
        33.333333% x 3 was scaled with "weights add up to 100.00% — scaled to 100%". Scaling itself is
        unchanged: every sum more than float noise off 100% is scaled."""
        price_df = align_price_data(_prices(), "SPY", "2020-01-01", ["A", "B", "C"])["price_df"]
        for weights, shown in (("0.333333, 0.333333, 0.333333", None), ("0.33335, 0.33335, 0.33335", None),
                               ("0.3332, 0.3332, 0.3332", "99.96%"), ("0.33337, 0.33337, 0.33337", "100.01%")):
            p = {"name": "P", "tickers": "A, B, C", "weights": weights, "thr": 40}
            tks, wts, errs, comp = parse_portfolio(p)
            out = prepare_portfolio(p, tks, wts, comp, price_df)
            self.assertAlmostEqual(out["w_series"].sum(), 1.0, places=12, msg=weights)
            msgs = [m for _, m in out["notices"] if "scaled to 100%" in m]
            matrix_ok = abs(sum(float(w) * 100 for w in weights.split(",")) - 100) < 0.01   # the matrix's ✓
            self.assertEqual(bool(msgs), not matrix_ok, (weights, msgs))
            if shown:
                self.assertIn(f"weights add up to {shown}", msgs[0])

    def test_asymmetric_ignores_per_slot_bands_and_says_so(self):
        price_df = align_price_data(_prices(), "SPY", "2020-01-01", ["A", "B", "C"])["price_df"]
        base = {"name": "P", "tickers": "A, B, C", "weights": "0.5, 0.3, 0.2", "thr": 60, "thr_up": 100,
                "slot_bands": {"B": {"down": 40, "up": 40}}}
        tks, wts, errs, comp = parse_portfolio(base)
        out = prepare_portfolio({**base, "strat": STRAT_ASYM}, tks, wts, comp, price_df)
        self.assertEqual((out["thr_dn"], out["thr_up"]), (0.6, 1.0))           # the portfolio band only
        self.assertTrue(any("per-slot bands apply to the RelDiff strategies only" in m and STRAT_ASYM in m
                            for _, m in out["notices"]))
        for strat in (STRAT_RD_FULL, STRAT_RD_MIXED, None):                    # None: an old config, RelDiff
            p = dict(base, strat=strat) if strat else dict(base)
            out = prepare_portfolio(p, tks, wts, comp, price_df)
            self.assertEqual(out["thr_dn"], {"*": 0.6, "B": 0.4}, strat)
            self.assertFalse(any("per-slot bands" in m for _, m in out["notices"]), strat)
        out = prepare_portfolio({**base, "strat": STRAT_ASYM, "slot_bands": {}}, tks, wts, comp, price_df)
        self.assertFalse(any("per-slot bands" in m for _, m in out["notices"]))  # nothing to ignore: no notice

    def test_size_weights_are_the_weights_as_entered(self):
        price_df = align_price_data(_prices(), "SPY", "2020-01-01", ["A", "B", "C"])["price_df"]
        p = {"name": "P", "tickers": "A, B, C", "weights": "0.1, 0.5, 0.405", "thr": 40}     # 100.5%
        tks, wts, errs, comp = parse_portfolio(p)
        out = prepare_portfolio(p, tks, wts, comp, price_df)
        self.assertEqual(list(out["size_weights"]), [0.1, 0.5, 0.405])
        self.assertAlmostEqual(out["w_series"]["A"], 0.1 / 1.005)                # < 10%: scaled below the cut-off
        p = {"name": "P", "tickers": "A, B, C", "weights": "0.1, 0.5, 0.4", "thr": 40}
        tks, wts, errs, comp = parse_portfolio(p)
        self.assertIsNone(prepare_portfolio(p, tks, wts, comp, price_df)["size_weights"])   # 100%: the targets

    def test_size_cut_offs_read_the_weights_as_entered(self):
        """Mixed resets every slot when a slot of 10% or more breaches, Asymmetric treats slots under 6% as
        minor: a sum of 100.5% scaled a 10% / 6% slot just under the cut-off and changed the rebalance."""
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        price = pd.DataFrame({"A": [100, 200, 200], "B": [100, 100, 100], "C": [100, 100, 100]}, index=idx)
        entered = pd.Series([0.1, 0.5, 0.405], index=price.columns)
        run = lambda strat, w, **kw: run_detailed_backtest(strat, price, w, 10000, 0.4, return_stats=True, **kw)
        scope = lambda st: [e["scope"] for e in st["rebal_events"]]
        self.assertEqual(scope(run(STRAT_RD_MIXED, entered / entered.sum())[3]), ["local"])     # before: minor
        self.assertEqual(scope(run(STRAT_RD_MIXED, entered / entered.sum(), size_weights=entered)[3]), ["global"])
        price = pd.DataFrame({"A": [100, 160, 160], "B": [100, 100, 100], "C": [100, 100, 100]}, index=idx)
        entered = pd.Series([0.06, 0.5, 0.445], index=price.columns)
        self.assertEqual(run(STRAT_ASYM, entered / entered.sum())[1], 0)           # minor: needs +100% to trigger
        self.assertEqual(run(STRAT_ASYM, entered / entered.sum(), size_weights=entered)[1], 1)   # major: +40%
        h1 = run(STRAT_RD_MIXED, entered / entered.sum())[0]                       # None: the targets, as before
        h2 = run(STRAT_RD_MIXED, entered / entered.sum(), size_weights=None)[0]
        pd.testing.assert_frame_equal(h1, h2)

    def test_nav_is_a_reserved_ticker(self):
        for tickers, weights in (("nav, A", "0.5, 0.5"), ("A, (NAV, B)", "0.5, 0.5")):
            tks, wts, errs, comp = parse_portfolio({"name": "P", "tickers": tickers, "weights": weights, "thr": 40})
            self.assertTrue(any("NAV is the name of a column of the results table" in e for e in errs), errs)
        tks, wts, errs, comp = parse_portfolio({"name": "P", "tickers": "NAVI, A", "weights": "0.5, 0.5", "thr": 40})
        self.assertEqual(errs, [])

    def test_per_slot_band_keys_match_like_tickers(self):
        from backtest_core import build_band_thresholds
        ids = {"BRK-B": "BRK-B", "A": "A", "ETH-USD+MSTR": "__slot2"}
        dn, up = build_band_thresholds(60, 100, {"brk.b": {"down": 40}, " a ": {"up": 80},
                                                 "mstr+eth-usd": {"down": 30, "up": 50}}, ids)
        self.assertEqual(dn, {"*": 0.6, "BRK-B": 0.4, "__slot2": 0.3})
        self.assertEqual(up, {"*": 1.0, "A": 0.8, "__slot2": 0.5})
        dn, up = build_band_thresholds(60, 100, {"XYZ": {"down": 40}}, ids)      # still unknown: ignored
        self.assertEqual((dn, up), (0.6, 1.0))

    def test_prices_that_end_early_are_reported(self):
        df = _prices()
        df.loc[df.index > pd.Timestamp("2020-10-01"), "B"] = np.nan                # delisted in October
        df.loc[df.index > pd.Timestamp(df.index[-5]), "C"] = np.nan                  # a few days: a holiday
        out = align_price_data(df, "SPY", "2020-01-01", ["A", "B", "C"])
        self.assertEqual(out["ended"], {"B": pd.Timestamp("2020-10-01")})
        self.assertTrue(np.isfinite(out["price_df"]["B"].iloc[-1]))                   # still carried flat
        self.assertEqual(align_price_data(_prices(), "SPY", "2020-01-01", ["A", "B"])["ended"], {})

    def test_app_reloads_a_stale_engine(self):
        import backtest_core
        app = (Path(__file__).resolve().parent.parent / "backtest_app.py").read_text(encoding="utf-8")
        self.assertIn(f'EXPECTED_CORE = "{backtest_core.CORE_VERSION}"', app)   # bump both with an engine change

if __name__ == "__main__":
    unittest.main()
