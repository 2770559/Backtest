"""Price-fetch regressions (backtest_app.fetch_price_history, v2.4.1).

The reported bug: with the three built-in portfolios Analyze worked, but
after deleting Port C every run ended in "No data after 2020-01-01".

Root cause, reproduced against yfinance 1.1.0's real download path: a ticker
whose request fails (Yahoo rate limit, network hiccup, unknown symbol) is
returned as an EMPTY placeholder column that carries an 'Adj Close' level
which auto-adjusted real data lacks. The app selected the price level for the
whole frame ('Adj Close' when present), so one failed ticker left a frame
holding nothing but that empty column: every ticker, the benchmark included,
looked dataless -- and the partial frame was cached per ticker set for an
hour. Deleting a portfolio changed the set (cache miss), the fresh batch hit
the rate limit for one ticker, and the poisoned frame was served on every
retry.

No network: yf.download is replaced by a fake that builds frames in
yfinance's exact layout (placeholder included) and fills yf.shared._ERRORS.

StartDateNoticeTest (v2.4.2) reuses the same fake: exactly one notice about
the effective start day, naming whatever finally decided it.
"""
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
APP = str(APP_DIR / "backtest_app.py")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import backtest_app as app  # noqa: E402  (bare-mode import: warnings are harmless)

RATE_LIMITED = "YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')"
TRANSIENT = "YFTzMissingError('$QQQM: possibly delisted; no timezone found')"

THREE_PORT = ('159941.SZ', '511130.SS', '512890.SS', '515220.SS', '518880.SS', '588080.SS',
              'BRK-B', 'DBMF', 'ETH-USD', 'GLDM', 'KMLM', 'MSTR', 'QQQM', 'SPY', 'XLE')
TWO_PORT = ('BRK-B', 'DBMF', 'ETH-USD', 'GLDM', 'KMLM', 'MSTR', 'QQQM', 'SPY', 'XLE')
START = "2019-12-12"


def _series(tk, start="2019-12-01", end="2021-06-30", holidays=()):
    """Deterministic, glitch-free positive price path (no >=5x steps)."""
    idx = pd.bdate_range(start, end, name="Date")
    if holidays:
        idx = idx[~idx.isin(pd.to_datetime(list(holidays)))]
    drift = 0.0002 + (sum(map(ord, tk)) % 7) * 0.0001
    return pd.Series(100.0 * np.cumprod(np.full(len(idx), 1 + drift)), index=idx, name=tk)


def _yf_frame(good, failed=()):
    """Mimic yf.download(auto_adjust=True): (Price, Ticker) MultiIndex columns.
    A failed ticker is yfinance's placeholder (yfinance.utils.empty_df):
    zero rows, all-NaN, WITH an 'Adj Close' column."""
    parts = {}
    for tk, close in good.items():
        parts[tk] = pd.DataFrame({"Open": close, "High": close, "Low": close,
                                  "Close": close, "Volume": 1.0})
    for tk in failed:
        parts[tk] = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"), data={
            'Open': np.nan, 'High': np.nan, 'Low': np.nan,
            'Close': np.nan, 'Adj Close': np.nan, 'Volume': np.nan})
    df = pd.concat(parts.values(), axis=1, sort=True, keys=parts.keys(),
                   names=['Ticker', 'Price'])
    df.columns = df.columns.swaplevel(0, 1)
    df.sort_index(level=0, axis=1, inplace=True)
    return df


class FakeYahoo:
    """Stand-in for yf.download: serves `failing` tickers as placeholders,
    lists each ticker from `starts[tk]` (default 2019-12-01), skips `holidays`
    for every ticker, and records every call's ticker list."""

    def __init__(self, failing=None, starts=None, holidays=()):
        self.failing = dict(failing or {})
        self.starts = dict(starts or {})
        self.holidays = tuple(holidays)
        self.calls = []

    def __call__(self, tickers, start=None, **kw):
        tickers = list(tickers)
        self.calls.append(tickers)
        good = {}
        for tk in tickers:
            if tk in self.failing:
                continue
            s = _series(tk, start=self.starts.get(tk, "2019-12-01"), holidays=self.holidays)
            good[tk] = s[s.index >= pd.Timestamp(start)]
        failed = [tk for tk in tickers if tk in self.failing]
        yf.shared._ERRORS = {tk: self.failing[tk] for tk in failed}
        return _yf_frame(good, failed)


class ExtractCloseTest(unittest.TestCase):
    def test_placeholder_never_decides_price_level(self):
        df = _yf_frame({"SPY": _series("SPY")}, failed=["KMLM"])
        # Whole-frame selection (the old code) would have picked 'Adj Close'.
        self.assertIn("Adj Close", df.columns.get_level_values(0))
        spy = app._extract_close(df, "SPY")
        self.assertIsNotNone(spy)
        self.assertGreater(spy.notna().sum(), 300)
        self.assertIsNone(app._extract_close(df, "KMLM"))

    def test_adj_close_preferred_when_it_holds_data(self):
        s = _series("SPY")
        df = _yf_frame({"SPY": s})
        df[("Adj Close", "SPY")] = s * 0.9
        pd.testing.assert_series_equal(app._extract_close(df, "SPY"), (s * 0.9).rename("SPY"))

    def test_single_level_lone_ticker_frame(self):
        s = _series("SPY")
        df = pd.DataFrame({"Open": s, "Close": s})
        self.assertIsNotNone(app._extract_close(df, "SPY", single=True))
        self.assertIsNone(app._extract_close(df, "SPY", single=False))


class FetchPriceHistoryTest(unittest.TestCase):
    def setUp(self):
        app.clear_price_cache()
        self._pause = app.PRICE_RETRY_PAUSE
        app.PRICE_RETRY_PAUSE = 0

    def tearDown(self):
        app.PRICE_RETRY_PAUSE = self._pause
        app.clear_price_cache()

    def test_deleting_a_portfolio_reuses_cached_tickers(self):
        """The reported flow: 3-portfolio run, then the 2-portfolio subset.
        The subset must not touch Yahoo at all."""
        fake = FakeYahoo()
        prices, failures = app.fetch_price_history(THREE_PORT, START, download=fake, now=0)
        self.assertEqual(failures, {})
        self.assertEqual(list(prices.columns), list(THREE_PORT))
        self.assertEqual(len(fake.calls), 1)

        prices2, failures2 = app.fetch_price_history(TWO_PORT, START, download=fake, now=10)
        self.assertEqual(len(fake.calls), 1, "subset re-downloaded")
        self.assertEqual(failures2, {})
        self.assertEqual(list(prices2.columns), list(TWO_PORT))
        self.assertGreater(prices2["SPY"].dropna().shape[0], 300)

    def test_one_rate_limited_ticker_does_not_poison_the_rest(self):
        fake = FakeYahoo(failing={"KMLM": RATE_LIMITED})
        prices, failures = app.fetch_price_history(TWO_PORT, START, download=fake, now=0)
        self.assertEqual(list(failures), ["KMLM"])
        self.assertIn("Too Many Requests", failures["KMLM"])
        self.assertNotIn("YFRateLimitError", failures["KMLM"])
        self.assertTrue(prices["KMLM"].isna().all())
        for tk in TWO_PORT:
            if tk != "KMLM":
                self.assertGreater(prices[tk].dropna().shape[0], 300, tk)
        # A rate limit is never retried immediately (that only extends the block).
        self.assertEqual(fake.calls, [list(TWO_PORT)])

    def test_failure_is_not_served_after_analyze_clears_it(self):
        fake = FakeYahoo(failing={"SPY": RATE_LIMITED})
        _, failures = app.fetch_price_history(TWO_PORT, START, download=fake, now=0)
        self.assertIn("SPY", failures)
        # Widget reruns inside the miss TTL: no new request.
        _, failures = app.fetch_price_history(TWO_PORT, START, download=fake, now=30)
        self.assertIn("SPY", failures)
        self.assertEqual(len(fake.calls), 1)
        # An explicit Analyze forgets the failure; Yahoo recovered meanwhile.
        app.forget_failed_prices()
        good = FakeYahoo()
        prices, failures = app.fetch_price_history(TWO_PORT, START, download=good, now=31)
        self.assertEqual(failures, {})
        self.assertEqual(good.calls, [["SPY"]], "only the failed ticker is re-requested")
        self.assertGreater(prices["SPY"].dropna().shape[0], 300)

    def test_negative_entry_expires_on_its_own(self):
        fake = FakeYahoo(failing={"SPY": RATE_LIMITED})
        app.fetch_price_history(("SPY", "QQQM"), START, download=fake, now=0)
        app.fetch_price_history(("SPY", "QQQM"), START, download=fake, now=app.PRICE_CACHE_MISS_TTL - 1)
        self.assertEqual(len(fake.calls), 1)
        app.fetch_price_history(("SPY", "QQQM"), START, download=fake, now=app.PRICE_CACHE_MISS_TTL + 1)
        self.assertEqual(fake.calls[-1], ["SPY"])

    def test_transient_failure_retried_once(self):
        class Flaky(FakeYahoo):
            def __call__(self, tickers, start=None, **kw):
                if not self.calls:
                    self.failing = {"QQQM": TRANSIENT}
                else:
                    self.failing = {}
                return super().__call__(tickers, start=start, **kw)
        fake = Flaky()
        prices, failures = app.fetch_price_history(TWO_PORT, START, download=fake, now=0)
        self.assertEqual(failures, {})
        self.assertEqual(fake.calls, [list(TWO_PORT), ["QQQM"]])
        self.assertGreater(prices["QQQM"].dropna().shape[0], 300)

    def test_persistent_unknown_symbol_reported_with_reason(self):
        fake = FakeYahoo(failing={"NOPE": TRANSIENT})
        prices, failures = app.fetch_price_history(("SPY", "NOPE"), START, download=fake, now=0)
        self.assertEqual(fake.calls, [["SPY", "NOPE"], ["NOPE"]])
        self.assertIn("possibly delisted", failures["NOPE"])
        self.assertTrue(prices["NOPE"].isna().all())
        self.assertFalse(prices["SPY"].isna().any())

    def test_later_start_served_from_cache_earlier_start_refetched(self):
        fake = FakeYahoo()
        app.fetch_price_history(("SPY",), START, download=fake, now=0)
        prices, _ = app.fetch_price_history(("SPY",), "2020-06-01", download=fake, now=1)
        self.assertEqual(len(fake.calls), 1)
        self.assertGreaterEqual(prices.index[0], pd.Timestamp("2020-06-01"))
        app.fetch_price_history(("SPY",), "2019-01-01", download=fake, now=2)
        self.assertEqual(len(fake.calls), 2)

    def test_download_exception_caches_nothing(self):
        def boom(tickers, **kw):
            raise RuntimeError("Yahoo down")
        with self.assertRaises(RuntimeError):
            app.fetch_price_history(("SPY",), START, download=boom, now=0)
        fake = FakeYahoo()
        _, failures = app.fetch_price_history(("SPY",), START, download=fake, now=1)
        self.assertEqual(failures, {})
        self.assertEqual(fake.calls, [["SPY"]])

    def test_returned_frame_is_detached_from_the_cache(self):
        fake = FakeYahoo()
        prices, _ = app.fetch_price_history(("SPY",), START, download=fake, now=0)
        prices.iloc[0, 0] = np.nan            # what the scrubbers do in place
        again, _ = app.fetch_price_history(("SPY",), START, download=fake, now=1)
        self.assertFalse(again["SPY"].isna().any())


def _two_port_config():
    return [
        {"id": "a", "name": "AV-US", "strat": "RelDiff Mixed", "thr": 40,
         "tickers": "QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, (ETH-USD, MSTR)",
         "weights": "0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05"},
        {"id": "b", "name": "Port B", "strat": "Asymmetric RelDiff", "thr": 38,
         "tickers": "QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, ETH-USD",
         "weights": "0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05"},
    ]


class AppRegressionTest(unittest.TestCase):
    """End-to-end: the exact reported configuration (Port C deleted) run
    against a batch in which one ticker was rate limited."""

    def setUp(self):
        st.cache_resource.clear()   # the script's own price cache singleton

    def tearDown(self):
        st.cache_resource.clear()

    def _run(self, fake):
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["portfolios_list"] = _two_port_config()
        at.session_state["run_backtest"] = True
        with patch.object(yf, "download", fake):
            at.run()
        return at

    def test_rate_limited_portfolio_ticker_no_longer_kills_the_run(self):
        fake = FakeYahoo(failing={"KMLM": RATE_LIMITED})
        at = self._run(fake)
        self.assertFalse(at.exception)
        self.assertEqual([e.value for e in at.error], [])
        self.assertEqual(sorted(fake.calls[0]), sorted(TWO_PORT))
        warnings = " | ".join(w.value for w in at.warning)
        self.assertIn("KMLM", warnings)
        self.assertIn("Too Many Requests", warnings)
        # Results rendered: the summary cards only exist in the results block.
        body = " ".join(m.value for m in at.markdown)
        self.assertIn("sum-card", body)
        self.assertIn("Port B", body)

    def test_delete_port_c_after_a_good_run_never_touches_yahoo(self):
        """The literal report: Analyze with the three built-ins, delete Port C,
        and the rerun must come entirely from the per-ticker cache. Yahoo is
        rate limiting everything by then, so any request would have failed."""
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["run_backtest"] = True
        first = FakeYahoo()
        with patch.object(yf, "download", first):
            at.run()
        self.assertFalse(at.exception)
        self.assertEqual([e.value for e in at.error], [])
        self.assertEqual(len(first.calls), 1)
        self.assertIn("511130.SS", first.calls[0])

        port_c = next(p for p in at.session_state["portfolios_list"] if p["name"] == "Port C")
        blocked = FakeYahoo(failing={tk: RATE_LIMITED for tk in THREE_PORT})
        with patch.object(yf, "download", blocked):
            at.button(key=f"del_{port_c['id']}").click().run()
        self.assertFalse(at.exception)
        self.assertEqual([p["name"] for p in at.session_state["portfolios_list"]], ["AV-US", "Port B"])
        self.assertEqual(blocked.calls, [], "deleting a portfolio re-downloaded")
        self.assertEqual([e.value for e in at.error], [])
        body = " ".join(m.value for m in at.markdown)
        self.assertIn("sum-card", body)
        self.assertNotIn("Port C", " ".join(w.value for w in at.warning))

    def test_rate_limited_benchmark_reports_the_cause(self):
        fake = FakeYahoo(failing={"SPY": RATE_LIMITED})
        at = self._run(fake)
        self.assertFalse(at.exception)
        errors = " | ".join(e.value for e in at.error)
        self.assertIn("SPY", errors)
        self.assertIn("Too Many Requests", errors)
        self.assertNotIn("No data after", errors)
        # Shown once; the next Analyze retries instead of every widget rerun.
        self.assertFalse(at.session_state["run_backtest"])


class StartDateNoticeTest(unittest.TestCase):
    """v2.4.2: one notice about the effective start day, naming what decided
    it. Before, every intermediate step reported itself, so a holiday start
    pushed to 2020-01-02 AND KMLM's late listing pushing it to 2020-12-02
    showed up together although only the latter is where the backtest starts."""

    HOLIDAY = ("2020-01-01",)   # the requested start is not a trading day in the fake calendar

    def setUp(self):
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_resource.clear()

    def _notices(self, fake):
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["portfolios_list"] = _two_port_config()
        at.session_state["sd"] = date(2020, 1, 1)
        at.session_state["run_backtest"] = True
        with patch.object(yf, "download", fake):
            at.run()
        self.assertFalse(at.exception)
        self.assertEqual([e.value for e in at.error], [])
        return ([w.value for w in at.warning if "listed" in w.value or "backtest starts" in w.value]
                + [i.value for i in at.info if "Aligned" in i.value])

    def test_late_ticker_is_the_only_notice(self):
        notices = self._notices(FakeYahoo(starts={"KMLM": "2020-12-02"}, holidays=self.HOLIDAY))
        self.assertEqual(len(notices), 1, notices)
        self.assertIn("KMLM", notices[0])
        self.assertIn("2020-12-02", notices[0])
        self.assertIn("2020-01-01", notices[0])
        self.assertNotIn("Aligned", notices[0])

    def test_late_ticker_beats_late_benchmark(self):
        notices = self._notices(FakeYahoo(starts={"SPY": "2020-06-01", "KMLM": "2020-12-02"},
                                          holidays=self.HOLIDAY))
        self.assertEqual(len(notices), 1, notices)
        self.assertIn("KMLM", notices[0])
        self.assertNotIn("SPY", notices[0])

    def test_late_benchmark_alone(self):
        notices = self._notices(FakeYahoo(starts={"SPY": "2020-06-01"}, holidays=self.HOLIDAY))
        self.assertEqual(len(notices), 1, notices)
        self.assertIn("SPY", notices[0])
        self.assertIn("2020-06-01", notices[0])
        self.assertIn("2020-01-01", notices[0])

    def test_holiday_alignment_alone(self):
        notices = self._notices(FakeYahoo(holidays=self.HOLIDAY))
        self.assertEqual(notices, ["Aligned to next trading day: 2020-01-02"])

    def test_trading_day_start_with_full_data_is_silent(self):
        # 2020-01-01 is a weekday, and no holiday in this fake calendar.
        self.assertEqual(self._notices(FakeYahoo()), [])


if __name__ == "__main__":
    unittest.main()
