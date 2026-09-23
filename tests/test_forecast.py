"""Regressions for iran_war_scenario_forecast.py (the Iran-war prediction tracker).

1. Anchor: the analysis's "当前(3/22)" row is its record of the 3/20 close. It
   used to sit at month 0.8 (= 3/24), so the value "predicted" for the 3/20
   baseline was an interpolation (~82% of the snapshot) and every calibrated
   path was shifted by the missing ~18% (XLE +3.9pp, GLDM -2.5pp, BTC +2.0pp)
   with a jump right after 3/20. The snapshot now sits exactly on the baseline:
   a calibrated path passes through 3/20's actual close and keeps the
   original moves from there on.
2. Summary tables are computed from the same paths as the charts. The old
   drawdown / 12-month table was hard-coded from the analysis text and did
   not match its own paths (S1 low -2.8% vs -1.8% on the path).
3. Actual-data fetch: an incomplete download is only kept for two minutes and a
   failed ticker keeps its last good series, instead of the whole result (or
   a None) being cached for an hour.

No network: yf.download is replaced by a fake in yfinance's frame layout.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
APP = str(APP_DIR / "iran_war_scenario_forecast.py")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import iran_war_scenario_forecast as fc  # noqa: E402  (UI lives in main(); import is side-effect free)

RATE_LIMITED = "YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')"
DAYS = pd.bdate_range("2026-02-16", "2026-09-21", name="Date")


def _series(tk, days=DAYS):
    """Deterministic positive price path, distinct per ticker."""
    drift = 0.0003 + (sum(map(ord, tk)) % 7) * 0.0002
    wiggle = 0.01 * np.sin(np.arange(len(days)) / (3 + len(tk)))
    return pd.Series(100.0 * np.cumprod(1 + drift + wiggle / 10), index=days, name=tk)


def _yf_frame(tickers, failed=()):
    """yf.download(auto_adjust=True) layout: (Price, Ticker) MultiIndex columns;
    a failed ticker is an all-NaN placeholder column."""
    parts = {}
    for tk in tickers:
        if tk in failed:
            parts[tk] = pd.DataFrame(index=DAYS, data={"Open": np.nan, "High": np.nan, "Low": np.nan,
                                                       "Close": np.nan, "Adj Close": np.nan,
                                                       "Volume": np.nan})
        else:
            c = _series(tk)
            parts[tk] = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0})
    df = pd.concat(parts.values(), axis=1, keys=parts.keys(), names=["Ticker", "Price"])
    df.columns = df.columns.swaplevel(0, 1)
    return df.sort_index(level=0, axis=1)


class FakeYahoo:
    """Stand-in for yf.download. `failing`: {ticker: reason}, or a callable
    returning it per call (to change behaviour between calls)."""

    def __init__(self, failing=None, raise_exc=None):
        self.failing = failing or {}
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, tickers, **kw):
        self.calls.append(list(tickers))
        if self.raise_exc:
            raise self.raise_exc
        failing = self.failing(len(self.calls)) if callable(self.failing) else self.failing
        failed = [tk for tk in tickers if tk in failing]
        yf.shared._ERRORS = {tk: failing[tk] for tk in failed}
        return _yf_frame(tickers, failed)


def _actual_from_path(sk, basis_offsets=None):
    """An 'actual' frame that follows scenario sk exactly on trading days."""
    days = pd.bdate_range("2026-03-02", "2026-09-21")
    return fc.path_values(sk, days, fc.ALL_SERIES, basis_offsets)


class SnapshotAnchorTest(unittest.TestCase):
    def test_snapshot_sits_on_the_prediction_baseline(self):
        self.assertEqual(fc.month_to_ts(fc.BASELINE_MONTH), pd.Timestamp("2026-03-20"))
        for sk, nodes in fc.SCENARIOS.items():
            snap = nodes[1]
            self.assertEqual(snap["label"], "当前(3/22)", sk)
            self.assertEqual(fc.month_to_ts(snap["month"]), fc.PREDICTION_BASELINE, sk)
            for a in fc.ASSETS:
                self.assertEqual(snap[a], fc.SNAPSHOT_NODE[a], (sk, a))

    def test_node_dates_strictly_increase(self):
        for sk, nodes in fc.SCENARIOS.items():
            dates = [fc.month_to_ts(n["month"]) for n in nodes]
            self.assertTrue(all(a < b for a, b in zip(dates, dates[1:])), sk)

    def test_raw_path_reproduces_every_node(self):
        for sk, nodes in fc.SCENARIOS.items():
            dates = [fc.month_to_ts(n["month"]) for n in nodes]
            vals = fc.path_values(sk, dates, fc.ALL_SERIES)
            for i, n in enumerate(nodes):
                for s in fc.ALL_SERIES:
                    self.assertAlmostEqual(vals.iloc[i][s], n[s], places=9, msg=(sk, n["label"], s))

    def test_portfolio_node_is_the_weighted_sum(self):
        self.assertAlmostEqual(sum(fc.WEIGHTS.values()), 1.0)
        snap = fc.SCENARIOS["S1"][1]
        self.assertAlmostEqual(snap[fc.PORT], -0.88, places=9)  # the analysis's "组合当前位置 -0.9%"

    def test_nan_outside_a_scenarios_horizon(self):
        # S1 ends at 12 months; asking for month 24 must not extrapolate
        v = fc.path_values("S1", [fc.month_to_ts(24)], [fc.PORT])[fc.PORT].iloc[0]
        self.assertTrue(np.isnan(v))


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        days = pd.bdate_range("2026-03-02", "2026-09-21")
        # an actual frame whose 3/20 values differ from the snapshot by known amounts
        self.shift = {s: (i + 1) * 1.5 * (-1) ** i for i, s in enumerate(fc.ALL_SERIES)}
        snap = fc.snapshot_values()
        self.actual = pd.DataFrame({s: snap[s] + self.shift[s] for s in fc.ALL_SERIES}, index=days)

    def test_offsets_are_actual_minus_snapshot(self):
        off = fc.calibration_offsets(self.actual)
        for s in fc.ALL_SERIES:
            self.assertAlmostEqual(off[s], self.shift[s], places=9)

    def test_calibrated_path_passes_through_the_baseline_close(self):
        """The bug: the first calibrated point after 3/20 jumped by ~18% of the snapshot."""
        off = fc.calibration_offsets(self.actual)
        for sk in fc.SCENARIOS:
            at_base = fc.path_values(sk, [fc.PREDICTION_BASELINE], fc.ALL_SERIES, off).iloc[0]
            for s in fc.ALL_SERIES:
                self.assertAlmostEqual(at_base[s], self.actual.loc["2026-03-20", s], places=9,
                                       msg=(sk, s))

    def test_calibration_keeps_the_original_moves(self):
        off = fc.calibration_offsets(self.actual)
        for sk, nodes in fc.SCENARIOS.items():
            later = [fc.month_to_ts(n["month"]) for n in nodes[2:]]
            raw = fc.path_values(sk, later, fc.ALL_SERIES)
            cal = fc.path_values(sk, later, fc.ALL_SERIES, off)
            for s in fc.ALL_SERIES:
                np.testing.assert_allclose(cal[s] - raw[s], off[s], atol=1e-9, err_msg=f"{sk} {s}")

    def test_war_start_stays_at_zero(self):
        off = fc.calibration_offsets(self.actual)
        v = fc.path_values("S3", [fc.WAR_START], fc.ALL_SERIES, off).iloc[0]
        self.assertTrue((v.abs() < 1e-12).all())

    def test_missing_baseline_day_uses_the_prior_print_not_a_later_one(self):
        actual = self.actual.drop(pd.Timestamp("2026-03-20"))
        actual.loc["2026-03-19"] = 1.0
        actual.loc["2026-03-23"] = 99.0
        off = fc.calibration_offsets(actual)
        self.assertAlmostEqual(off["QQQM"], 1.0 - fc.SNAPSHOT_NODE["QQQM"])

    def test_no_actual_data_means_no_calibration(self):
        self.assertEqual(fc.calibration_offsets(None), {})
        self.assertEqual(fc.calibration_offsets(self.actual.loc["2026-04-01":]), {})


class SummaryTablesTest(unittest.TestCase):
    def test_path_stats_come_from_the_paths(self):
        s1 = fc.portfolio_path_stats("S1", None)
        self.assertAlmostEqual(s1["low"], -1.8, places=9)  # the analysis text said -2.8%
        self.assertEqual(s1["low_label"], "最大压力(4月初)")
        self.assertAlmostEqual(s1["m12"], 5.3, places=9)   # the analysis text said +4.8%
        self.assertTrue(np.isnan(s1["m24"]))
        s3 = fc.portfolio_path_stats("S3", None)
        self.assertEqual(s3["low_label"], "就业恶化/开始降息(9-10月)")
        self.assertAlmostEqual(s3["m24"], 8.8, places=9)

    def test_path_stats_follow_the_basis(self):
        off = {s: 2.0 for s in fc.ALL_SERIES}
        for sk in fc.SCENARIOS:
            raw, cal = fc.portfolio_path_stats(sk, None), fc.portfolio_path_stats(sk, off)
            self.assertAlmostEqual(cal["m12"] - raw["m12"], 2.0, places=9)

    def test_scorecard_finds_the_scenario_reality_followed(self):
        actual = _actual_from_path("S2")
        card = fc.scorecard(actual, fc.calibration_offsets(actual))
        rows = {r["key"]: r for r in card["rows"]}
        self.assertAlmostEqual(rows["S2"]["rmse"], 0.0, places=9)
        self.assertEqual(min((r for r in card["rows"] if r["key"] != "W"), key=lambda r: r["rmse"])["key"], "S2")
        self.assertAlmostEqual(card["actual"], rows["S2"]["pred"], places=9)
        weighted = sum(fc.SCENARIO_META[k]["prob"] * rows[k]["pred"] for k in fc.SCENARIO_META)
        self.assertAlmostEqual(rows["W"]["pred"], weighted, places=9)

    def test_weighted_forecast_matches_the_scenarios(self):
        day = pd.Timestamp("2026-06-30")
        wf = fc.weighted_forecast(day, fc.ALL_SERIES, None)
        for s in fc.ALL_SERIES:
            expect = sum(m["prob"] * fc.path_values(k, [day], [s]).iloc[0][s]
                         for k, m in fc.SCENARIO_META.items())
            self.assertAlmostEqual(wf[s], expect, places=9)

    def test_scorecard_needs_data_after_the_baseline(self):
        actual = _actual_from_path("S1").loc[:"2026-03-20"]
        self.assertIsNone(fc.scorecard(actual, {}))
        self.assertIsNone(fc.scorecard(None, {}))


class ChartLabelTest(unittest.TestCase):
    def test_overlapping_end_labels_merge(self):
        """S1/S2 both end on 2027-02-28 at +5.3/+5.25 -> one "S1·S2" label, not two on top."""
        d1, d2 = fc.month_to_ts(12), fc.month_to_ts(24)
        ends = pd.DataFrame({"日期": [d1, d1, d2, d2, d2], "收益率(%)": [5.3, 5.25, 8.8, 8.4, 2.0],
                             "lbl": ["S1", "S2", "S3", "S4", "S5"]})
        out = fc._merge_close_labels(ends, 1.0)
        self.assertEqual(sorted(out["lbl"]), ["S1·S2", "S3·S4", "S5"])
        self.assertAlmostEqual(out.set_index("lbl").loc["S1·S2", "收益率(%)"], 5.275)

    def test_distinct_end_labels_stay_apart(self):
        d = fc.month_to_ts(24)
        ends = pd.DataFrame({"日期": [d, d], "收益率(%)": [1.0, 5.0], "lbl": ["S3", "S4"]})
        self.assertEqual(sorted(fc._merge_close_labels(ends, 1.0)["lbl"]), ["S3", "S4"])


class CumulativeReturnsTest(unittest.TestCase):
    def closes(self, drop=()):
        return {tk: _series(tk) for tk in fc.TICKERS if tk not in drop}

    def test_base_is_the_last_prewar_close(self):
        actual, missing = fc.cumulative_returns(self.closes())
        self.assertEqual(missing, [])
        self.assertEqual(actual.index[0], pd.Timestamp("2026-03-02"))
        q = _series("QQQM")
        self.assertAlmostEqual(actual.loc["2026-03-02", "QQQM"],
                               (q["2026-03-02"] / q["2026-02-27"] - 1) * 100, places=9)
        port = sum(actual[a] * fc.WEIGHTS[a] for a in fc.ASSETS)
        np.testing.assert_allclose(actual[fc.PORT], port, atol=1e-9)

    def test_missing_ticker_renormalizes_the_portfolio(self):
        actual, missing = fc.cumulative_returns(self.closes(drop=("XLE",)))
        self.assertEqual(missing, ["XLE"])
        w = {a: fc.WEIGHTS[a] / 0.9 for a in fc.ASSETS if a != "XLE"}
        np.testing.assert_allclose(actual[fc.PORT], sum(actual[a] * w[a] for a in w), atol=1e-9)

    def test_missing_print_on_the_base_day_is_filled_forward(self):
        closes = self.closes()
        closes["GLDM"] = closes["GLDM"].drop(pd.Timestamp("2026-02-27"))
        actual, missing = fc.cumulative_returns(closes)
        self.assertEqual(missing, [])
        self.assertFalse(actual[fc.PORT].isna().any())

    def test_nothing_usable(self):
        self.assertEqual(fc.cumulative_returns({}), (None, fc.ASSETS))


class LoadActualTest(unittest.TestCase):
    def setUp(self):
        fc._actual_store.clear()
        self._pause = fc.RETRY_PAUSE
        fc.RETRY_PAUSE = 0

    def tearDown(self):
        fc.RETRY_PAUSE = self._pause
        fc._actual_store.clear()

    def test_complete_download_is_reused_for_an_hour(self):
        fake = FakeYahoo()
        closes, failures, stale = fc.load_actual(now=0, download=fake)
        self.assertEqual(sorted(closes), sorted(fc.TICKERS))
        self.assertEqual((failures, stale), ({}, []))
        fc.load_actual(now=fc.ACTUAL_TTL - 1, download=fake)
        self.assertEqual(len(fake.calls), 1)
        fc.load_actual(now=fc.ACTUAL_TTL + 1, download=fake)
        self.assertEqual(len(fake.calls), 2)

    def test_rate_limited_ticker_is_retried_after_two_minutes_not_an_hour(self):
        fake = FakeYahoo(failing={"BTC": RATE_LIMITED})
        closes, failures, _ = fc.load_actual(now=0, download=fake)
        self.assertNotIn("BTC", closes)
        self.assertIn("Too Many Requests", failures["BTC"])
        self.assertEqual(len(fake.calls), 1)  # no immediate retry while rate limited
        fc.load_actual(now=fc.ACTUAL_RETRY_TTL - 1, download=fake)
        self.assertEqual(len(fake.calls), 1)
        fc.load_actual(now=fc.ACTUAL_RETRY_TTL + 1, download=fake)
        self.assertEqual(len(fake.calls), 2)

    def test_transient_failure_retried_once_immediately(self):
        fake = FakeYahoo(failing=lambda n: {"KMLM": "YFTzMissingError('$KMLM: no timezone')"} if n == 1 else {})
        closes, failures, _ = fc.load_actual(now=0, download=fake)
        self.assertIn("KMLM", closes)
        self.assertEqual(failures, {})
        self.assertEqual(fake.calls[1], ["KMLM"])

    def test_failed_refresh_keeps_the_last_good_series(self):
        fc.load_actual(now=0, download=FakeYahoo())
        closes, failures, stale = fc.load_actual(force=True, now=10, download=FakeYahoo(failing={"GLDM": RATE_LIMITED}))
        self.assertIn("GLDM", closes)
        self.assertEqual(stale, ["GLDM"])
        self.assertEqual(failures, {})
        fake = FakeYahoo()
        fc.load_actual(now=10 + fc.ACTUAL_RETRY_TTL + 1, download=fake)
        self.assertEqual(len(fake.calls), 1)  # stale data is retried soon

    def test_download_exception_is_not_kept_for_an_hour(self):
        _, failures, _ = fc.load_actual(now=0, download=FakeYahoo(raise_exc=ConnectionError("offline")))
        self.assertEqual(sorted(failures), sorted(fc.TICKERS))
        fake = FakeYahoo()
        closes, failures, _ = fc.load_actual(now=fc.ACTUAL_RETRY_TTL + 1, download=fake)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(failures, {})


class AppSmokeTest(unittest.TestCase):
    def run_app(self, fake):
        st.cache_resource.clear()
        at = AppTest.from_file(APP, default_timeout=60)
        with patch.object(yf, "download", fake):
            at.run()
        return at

    def test_full_data_renders_scorecard_and_both_bases(self):
        fake = FakeYahoo()
        at = self.run_app(fake)
        self.assertFalse(at.exception, at.exception)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("预测记分", text)
        self.assertIn("概率加权预测 vs 实际", text)
        self.assertGreaterEqual(len(at.dataframe), 5)
        with patch.object(yf, "download", fake):
            at.radio(key="basis").set_value("raw").run()
            self.assertFalse(at.exception, at.exception)
            at.radio(key="scenario_radio").set_value("S4").run()
            at.radio(key="asset_radio").set_value("XLE").run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(fake.calls), 1)  # widget reruns never re-hit Yahoo

    def test_partial_failure_warns_and_still_renders(self):
        at = self.run_app(FakeYahoo(failing={"DBMF": RATE_LIMITED}))
        self.assertFalse(at.exception, at.exception)
        warnings = " ".join(w.value for w in at.warning)
        self.assertIn("DBMF", warnings)
        self.assertIn("Too Many Requests", warnings)

    def test_total_failure_shows_predictions_only(self):
        at = self.run_app(FakeYahoo(raise_exc=ConnectionError("offline")))
        self.assertFalse(at.exception, at.exception)
        self.assertIn("未能获取实际市场数据", " ".join(w.value for w in at.warning))


if __name__ == "__main__":
    unittest.main()
