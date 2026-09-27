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
import json
import socket
import sys
import threading
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests
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

    def __init__(self, failing=None, starts=None, holidays=(), ends=None):
        self.failing = dict(failing or {})
        self.starts = dict(starts or {})
        self.ends = dict(ends or {})                      # ticker -> its last price date (delisted)
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
            s = s[s.index <= pd.Timestamp(self.ends.get(tk, s.index[-1]))]
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

    def test_eviction_never_drops_a_ticker_of_the_request(self):
        """Once the process-wide cache was full, evicting the earliest-expiring entry
        could remove a still-live ticker of this very request, read right after:
        KeyError, shown as "Download error: 'A'"."""
        fake = FakeYahoo()
        app.fetch_price_history(("A",), START, download=fake, now=-10)
        app.fetch_price_history(tuple(f"T{i:03d}" for i in range(app.PRICE_CACHE_MAX - 1)), START,
                                download=fake, now=0)
        prices, failures = app.fetch_price_history(("A", "NEW"), START, download=fake, now=100)
        self.assertEqual(failures, {})
        self.assertEqual(list(prices.columns), ["A", "NEW"])
        entries = app._price_cache()["entries"]
        self.assertEqual(len(entries), app.PRICE_CACHE_MAX)
        self.assertIn("A", entries)

    def test_eviction_drops_expired_entries_first(self):
        fake = FakeYahoo()
        old = ("X0", "X1", "X2")
        app.fetch_price_history(old, START, download=fake, now=0)                  # expire at TTL
        live = tuple(f"L{i:03d}" for i in range(app.PRICE_CACHE_MAX - len(old)))
        app.fetch_price_history(live, START, download=fake, now=app.PRICE_CACHE_TTL - 600)
        app.fetch_price_history(("N1", "N2"), START, download=fake, now=app.PRICE_CACHE_TTL + 1)
        entries = app._price_cache()["entries"]
        self.assertEqual(len(entries), app.PRICE_CACHE_MAX)
        self.assertTrue(all(tk in entries for tk in live + ("N1", "N2")))
        self.assertEqual(sum(tk in entries for tk in old), 1)


class PriceCacheLockTest(unittest.TestCase):
    """The cache lock used to be held for the whole download, so every widget rerun of a session with
    results on screen (a pure cache hit) waited for whatever another visitor was downloading. Downloads
    still run one at a time (yfinance's shared state) and never fetch a ticker twice."""

    def setUp(self):
        app.clear_price_cache()
        self._pause = app.PRICE_RETRY_PAUSE
        app.PRICE_RETRY_PAUSE = 0

    def tearDown(self):
        app.PRICE_RETRY_PAUSE = self._pause
        app.clear_price_cache()

    @staticmethod
    def _gated(fake):
        """-> (download, entered, release): the download blocks inside until `release` is set."""
        entered, release = threading.Event(), threading.Event()

        def download(tickers, **kw):
            entered.set()
            release.wait(10)
            return fake(tickers, **kw)
        return download, entered, release

    @staticmethod
    def _call(out, key, *args, **kw):
        t = threading.Thread(target=lambda: out.update({key: app.fetch_price_history(*args, **kw)}), daemon=True)
        t.start()
        return t

    def test_a_cache_hit_does_not_wait_for_another_sessions_download(self):
        fake = FakeYahoo()
        app.fetch_price_history(("SPY", "TLT"), START, download=fake, now=0)
        slow, entered, release = self._gated(fake)
        out = {}
        other = self._call(out, "other", ("NEW1", "NEW2"), START, download=slow, now=1)
        try:
            self.assertTrue(entered.wait(5))                  # the other session is inside its download
            self._call(out, "hit", ("SPY", "TLT"), START, download=fake, now=2).join(5)
            self.assertIn("hit", out, "a cache hit waited for another session's download")
            prices, failures = out["hit"]
            self.assertEqual((list(prices.columns), failures), (["SPY", "TLT"], {}))
        finally:
            release.set()
            other.join(10)
        self.assertEqual(sorted(map(sorted, fake.calls)), [["NEW1", "NEW2"], ["SPY", "TLT"]])

    def test_a_ticker_being_downloaded_is_not_fetched_twice(self):
        fake = FakeYahoo()
        slow, entered, release = self._gated(fake)
        out = {}
        first = self._call(out, "a", ("X",), START, download=slow, now=0)
        try:
            self.assertTrue(entered.wait(5))
            second = self._call(out, "b", ("X", "Y"), START, download=fake, now=1)
            second.join(0.3)
            self.assertTrue(second.is_alive())                # waits for the download in flight
        finally:
            release.set()
            first.join(10)
        second.join(10)
        self.assertEqual(fake.calls, [["X"], ["Y"]])          # X once (first session), then only Y
        pd.testing.assert_series_equal(out["a"][0]["X"], out["b"][0]["X"])


class _Resp:
    def __init__(self, text="", content=b""):
        self.text, self.content = text, content

    def raise_for_status(self):
        pass


class _HttpResp:
    """requests.Response stand-in with a status code (raise_for_status) and an optional JSON payload."""

    def __init__(self, status=200, content=b"", payload=None):
        self.status_code, self.content, self._payload = status, content, payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class CpiFetchTest(unittest.TestCase):
    """FRED CPI for the inflation adjustment: pd.read_csv(url) had no timeout, so a
    stalled response hung the run; the request now carries one."""
    CSV = "observation_date,CPIAUCSL\n2020-01-01,258.7\n2021-01-01,262.2\n"

    def setUp(self):
        app.fetch_cpi_data.clear()

    def tearDown(self):
        app.fetch_cpi_data.clear()

    def test_request_carries_a_timeout_and_parses_the_csv(self):
        calls = []

        def fake_get(url, **kw):
            calls.append(kw)
            return _Resp(text=self.CSV)
        with patch.object(app.requests, "get", fake_get):
            cpi = app.fetch_cpi_data()
        self.assertEqual(list(cpi.columns), ["CPI"])
        self.assertAlmostEqual(cpi.loc["2021-01-01", "CPI"], 262.2)
        self.assertTrue(calls[0].get("timeout"))

    def test_stalled_server_gives_up_and_is_not_cached(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)                                        # accepts, never answers
        real_get = requests.get
        stalled = lambda url, **kw: real_get(f"http://127.0.0.1:{srv.getsockname()[1]}/cpi.csv", **kw)
        outcome = {}

        def run():
            try:
                app.fetch_cpi_data()
                outcome["r"] = "returned"
            except requests.RequestException as e:
                outcome["r"] = type(e).__name__
        try:
            with patch.object(app, "CPI_TIMEOUT", 0.5), patch.object(app.requests, "get", stalled):
                t = threading.Thread(target=run, daemon=True)
                t.start()
                t.join(10)
        finally:
            srv.close()
        self.assertEqual(outcome.get("r"), "ReadTimeout", "still blocked on a stalled FRED")
        with patch.object(app.requests, "get", lambda url, **kw: _Resp(text=self.CSV)):
            self.assertEqual(len(app.fetch_cpi_data()), 2)   # the failure was not cached for 24 h


class CnNameLookupTest(unittest.TestCase):
    """Tencent name lookup: the failure counter used to be a module-level dict,
    re-created by every rerun (the breaker never tripped), and a failed lookup
    was cached for a day."""

    def setUp(self):
        app._cn_name_breaker.clear()
        app._tencent_names.clear()

    def tearDown(self):
        app._cn_name_breaker.clear()
        app._tencent_names.clear()

    def test_failures_are_retried_until_the_breaker_trips(self):
        calls = []

        def down(url, **kw):
            calls.append(url)
            raise requests.ConnectionError("unreachable")
        with patch.object(app.requests, "get", down):
            for _ in range(app.CN_NAME_MAX_FAILURES):
                names = app.fetch_cn_names(("511010.SS",))
                self.assertEqual(names["511010.SS"], app.CN_NAME_SEED["511010.SS"])   # seed fallback
            self.assertEqual(len(calls), app.CN_NAME_MAX_FAILURES)                     # none cached
            app.fetch_cn_names(("511010.SS",))
            app.fetch_cn_names(("600519.SS",))
        self.assertEqual(len(calls), app.CN_NAME_MAX_FAILURES)                         # tripped: no more tries

    def test_answer_is_cached_and_resets_the_breaker(self):
        calls = []

        def ok(url, **kw):
            calls.append(url)
            return _Resp(content='v_sh600519="1~贵州茅台~600519~1700.00";'.encode("gbk"))
        app._cn_name_breaker()["n"] = 1
        with patch.object(app.requests, "get", ok):
            self.assertEqual(app.fetch_cn_names(("600519.SS",))["600519.SS"], "贵州茅台")
            self.assertEqual(app.fetch_cn_names(("600519.SS",))["600519.SS"], "贵州茅台")
        self.assertEqual(len(calls), 1)                      # answered once, then served from the day cache
        self.assertEqual(app._cn_name_breaker()["n"], 0)

    def test_http_error_or_non_quote_body_is_not_cached(self):
        """Any HTTP answer used to count: a 403 / 429 / error page reset the breaker and was cached for a
        day as "no names", so the labels stayed missing after Tencent recovered."""
        good = _HttpResp(200, 'v_sh600519="1~贵州茅台~600519~1700.00";'.encode("gbk"))
        for bad in (_HttpResp(403, b"<html>Forbidden</html>"), _HttpResp(429, b""),
                    _HttpResp(200, b"<html>captive portal</html>"), _HttpResp(200, b"")):
            app._cn_name_breaker.clear()
            app._tencent_names.clear()
            with patch.object(app.requests, "get", lambda url, **kw: bad):
                self.assertNotIn("600519.SS", app.fetch_cn_names(("600519.SS",)), bad.content)
            self.assertEqual(app._cn_name_breaker()["n"], 1, bad.content)      # counted as a failure
            with patch.object(app.requests, "get", lambda url, **kw: good):
                self.assertEqual(app.fetch_cn_names(("600519.SS",))["600519.SS"], "贵州茅台", bad.content)
            self.assertEqual(app._cn_name_breaker()["n"], 0)

    def test_no_such_code_is_an_answer_not_a_failure(self):
        """Tencent's whole answer is v_pv_none_match when none of the codes exists: cached like any answer,
        and it resets the breaker — a visitor's mistyped code must not trip it for every visitor."""
        calls = []

        def get(url, **kw):
            calls.append(url)
            return _HttpResp(200, b'v_pv_none_match="1";\n')
        app._cn_name_breaker()["n"] = 1
        with patch.object(app.requests, "get", get):
            for _ in range(3):
                self.assertNotIn("999999.SS", app.fetch_cn_names(("999999.SS",)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(app._cn_name_breaker()["n"], 0)


class SymbolSearchTest(unittest.TestCase):
    """The typeahead's answers are cached for an hour; its failures were cached too (as []), so a Yahoo rate
    limit or network blip blanked the search for that query for an hour, for every visitor."""
    ANSWER = {"quotes": [{"symbol": "QQQ", "shortname": "Invesco QQQ", "quoteType": "ETF", "exchange": "NMS"}]}
    RESULT = [("Invesco QQQ (QQQ) · ETF · NMS", "QQQ")]

    def setUp(self):
        app._yahoo_search.clear()

    def tearDown(self):
        app._yahoo_search.clear()

    def test_failures_are_not_cached(self):
        for failure in (requests.ConnectionError("blip"), _HttpResp(429, b"Too Many Requests"),
                        _HttpResp(200, b"<html>", payload=None), _HttpResp(200, b"{}", payload={"finance": {}})):
            def get(url, **kw):
                if isinstance(failure, Exception):
                    raise failure
                return failure
            with patch.object(app.requests, "get", get):
                self.assertEqual(app.yahoo_symbol_search("QQQ"), [], failure)
            with patch.object(app.requests, "get", lambda url, **kw: _HttpResp(200, b"{}", payload=self.ANSWER)):
                self.assertEqual(app.yahoo_symbol_search("QQQ"), self.RESULT, failure)
            app._yahoo_search.clear()

    def test_answers_are_cached(self):
        calls = []

        def get(url, **kw):
            calls.append(kw["params"]["q"])
            return _HttpResp(200, b"{}", payload=self.ANSWER)
        with patch.object(app.requests, "get", get):
            self.assertEqual(app.yahoo_symbol_search("QQQ"), self.RESULT)
            self.assertEqual(app.yahoo_symbol_search(" QQQ "), self.RESULT)
            self.assertEqual(app.yahoo_symbol_search("Q"), [])       # too short: never requested
        self.assertEqual(calls, ["QQQ"])


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

    def test_prices_that_end_early_are_reported(self):
        """A delisted or renamed ticker's last price was carried flat to the end of the backtest in silence."""
        at = self._run(FakeYahoo(ends={"DBMF": "2021-02-26"}))
        self.assertFalse(at.exception)
        warnings = " | ".join(w.value for w in at.warning)
        self.assertIn("Prices end early for **DBMF** (2021-02-26)", warnings)
        self.assertIn("sum-card", " ".join(m.value for m in at.markdown))       # the run itself goes on
        st.cache_resource.clear()                                                  # the cut series is cached
        at = self._run(FakeYahoo())
        self.assertNotIn("Prices end early", " | ".join(w.value for w in at.warning))

    def test_dollar_names_and_bands_asymmetric_ignores(self):
        """Review 3 online low-2: two "$" of a portfolio name were typeset as LaTeX in the tabs and the per-slot band
        expander. Review 4: Asymmetric RelDiff ignores per-slot bands, and says so."""
        ports = _two_port_config()
        ports[0]["name"] = "US$ $Plan"
        ports[1]["slot_bands"] = {"ETH-USD": {"down": 40, "up": 40}}
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["portfolios_list"] = ports
        at.session_state["run_backtest"] = True
        with patch.object(yf, "download", FakeYahoo()):
            at.run()
        self.assertFalse(at.exception)
        self.assertIn("US\\$ \\$Plan", [t.label for t in at.tabs])
        self.assertIn("Per-slot bands · US\\$ \\$Plan", [e.label for e in at.expander])
        infos = " | ".join(i.value for i in at.info)
        self.assertIn("**Port B**: per-slot bands apply to the RelDiff strategies only", infos)

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


def _vega_access_path(p):
    """Python port of vega-util's splitAccessPath, the parser Vega-Lite runs on every field name (checked
    against the copy bundled in Streamlit's frontend): "." and "[ ]" split the path, a quote at the start
    of a segment opens a quoted segment, "\\" escapes the next character. Raises like the browser does."""
    path, n = [], len(p)
    q, b, s, i, j = None, 0, "", 0, 0

    def push():
        nonlocal s, i
        path.append(s + p[i:j])
        s, i = "", j + 1
    while j < n:
        c = p[j]
        if c == "\\":
            s += p[i:j]
            j += 1
            i = j
        elif c == q:
            push()
            q, b = None, -1
        elif q:
            pass
        elif i == b and c in "\"'":
            i, q = j + 1, c
        elif c == "." and not b:
            if j > i:
                push()
            else:
                i = j + 1
        elif c == "[":
            if j > i:
                push()
            b = i = j + 1
        elif c == "]":
            if not b:
                raise ValueError("Access path missing open bracket: " + p)
            if b > 0:
                push()
            b, i = 0, j + 1
        j += 1
    if b:
        raise ValueError("Access path missing closing bracket: " + p)
    if q:
        raise ValueError("Access path missing closing quote: " + p)
    if j > i:
        j += 1
        push()
    return path


def _tooltip_fields(spec):
    """Every tooltip field of a Vega-Lite spec (any layer depth)."""
    out = set()
    if isinstance(spec, dict):
        if isinstance(spec.get("tooltip"), list):
            out |= {t.get("field") for t in spec["tooltip"] if isinstance(t, dict)}
        for v in spec.values():
            out |= _tooltip_fields(v)
    elif isinstance(spec, list):
        for v in spec:
            out |= _tooltip_fields(v)
    return out


class OutsideTextTest(unittest.TestCase):
    """Names and reasons are data, not markup: a dotted series name is escaped as a
    Vega-Lite field (its tooltip row was blank), "$" in Yahoo's reasons is escaped
    for markdown (a "$…$" pair was typeset as LaTeX)."""

    def setUp(self):
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_resource.clear()

    def _run(self, fake, bench, ports=None):
        at = AppTest.from_file(APP, default_timeout=120)
        at.session_state["portfolios_list"] = ports or _two_port_config()
        at.session_state["bi"] = bench
        at.session_state["run_backtest"] = True
        with patch.object(yf, "download", fake):
            at.run()
        self.assertFalse(at.exception)
        return at

    def test_quoted_names_keep_both_charts(self):
        """A quote opens a quoted segment in a Vega field path: a portfolio named "Dad's" failed with
        "Access path missing closing quote" and both charts were replaced by an error. Every tooltip field
        must resolve, under Vega's own parsing rules, to exactly its series name."""
        names = ["Dad's", "Mom's 60/40", '"Growth" mix']
        base = _two_port_config()
        ports = [dict(base[i % 2], id=f"q{i}", name=n) for i, n in enumerate(names)]
        at = self._run(FakeYahoo(), "BRK.B", ports)
        charts = at.get("arrow_vega_lite_chart")
        self.assertEqual(len(charts), 2)                     # cumulative return + drawdown
        for chart in charts:
            paths = [_vega_access_path(f) for f in _tooltip_fields(json.loads(chart.proto.spec)) - {"Date"}]
            self.assertTrue(all(len(p) == 1 for p in paths), paths)
            self.assertEqual({p[0] for p in paths}, set(names) | {"Benchmark(BRK.B)"})

    def test_dotted_benchmark_tooltip_field_is_escaped(self):
        at = self._run(FakeYahoo(), "BRK.B")
        charts = at.get("arrow_vega_lite_chart")
        self.assertEqual(len(charts), 2)                     # cumulative return + drawdown
        for chart in charts:
            fields = _tooltip_fields(json.loads(chart.proto.spec))
            self.assertIn("Benchmark(BRK\\.B)", fields)
            self.assertNotIn("Benchmark(BRK.B)", fields)

    def test_dollar_in_yahoo_reasons_is_escaped(self):
        reason = "YFTzMissingError('${}: possibly delisted; no timezone found')"
        at = self._run(FakeYahoo(failing={tk: reason.format(tk) for tk in ("DBMF", "KMLM")}), "SPY")
        warn = next(str(w.value) for w in at.warning if "Yahoo returned no prices" in str(w.value))
        self.assertIn("\\$DBMF: possibly delisted", warn)
        self.assertNotRegex(warn, r"(?<!\\)\$")              # every "$" escaped


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
