"""v2.6.0 Live Portfolio mode (AppTest, offline) and the page's shadow wiring.

- The app opens in Backtest mode; switching to Live renders the desk without
  touching the network (nothing is fetched until Refresh shadow is clicked);
  switching back keeps the Backtest settings and portfolios.
- live_page.compute_shadow runs a config through the same fetch → scrub → align
  → prepare → engine chain as the Backtest page (checked with a fake fetch).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
APP = str(APP_DIR / "backtest_app.py")

from streamlit.testing.v1 import AppTest  # noqa: E402

import live_page  # noqa: E402
import live_core as lc  # noqa: E402
from backtest_core import (  # noqa: E402
    STRAT_RD_MIXED, BAND_MODE_RATIO, align_price_data, prepare_portfolio, parse_portfolio, run_detailed_backtest,
)


class ModeSwitchTest(unittest.TestCase):
    def test_backtest_is_default_and_live_renders_offline(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio(key="app_mode").value, "Backtest")
        self.assertTrue(any("Portfolio Backtest" in str(m.value) for m in at.markdown))
        before = [dict(p) for p in at.session_state["portfolios_list"]]
        bi, sd = at.session_state["bi"], at.session_state["sd"]

        at.radio(key="app_mode").set_value("Live Portfolio").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Live Portfolio Desk" in str(m.value) for m in at.markdown))
        self.assertTrue(any(s.label == "Live rule config" for s in at.selectbox))
        self.assertTrue(any("Refresh shadow" in str(b.label) for b in at.button))
        self.assertNotIn("live_shadow", at.session_state)          # nothing fetched on entry

        at.radio(key="app_mode").set_value("Backtest").run()
        self.assertFalse(at.exception)
        self.assertEqual([p["tickers"] for p in at.session_state["portfolios_list"]], [p["tickers"] for p in before])
        self.assertEqual((at.session_state["bi"], at.session_state["sd"]), (bi, sd))


def _fake_fetch(tickers, start, **_):
    idx = pd.date_range("2020-11-01", "2026-09-15", freq="D")
    n = len(idx)
    rng = np.random.default_rng(3)
    cols = {}
    for k, t in enumerate(tickers):
        cols[t] = 100 * np.exp(np.cumsum(rng.normal(0.0004 * (k + 1), 0.01, n)))
    df = pd.DataFrame(cols, index=idx)
    if "SPY" in df:
        df.loc[df.index.dayofweek >= 5, "SPY"] = np.nan
    return df[list(tickers)].copy(), {}


class ComputeShadowTest(unittest.TestCase):
    CFG = {"benchmark": "SPY", "start_date": "2020-12-02", "portfolios": [
        {"name": "Live", "tickers": "QQQM, BRK.B, (ETH-USD, MSTR)", "weights": "0.6, 0.35, 0.05", "strat": "RelDiff Mixed",
         "thr": 80, "thr_up": 125, "slot_bands": {"ETH-USD+MSTR": {"down": 60, "up": 90}}, "band_mode": "ratio"}]}

    def test_matches_the_backtest_pipeline(self):
        sh = live_page.compute_shadow(self.CFG, 0, _fake_fetch)
        st = sh["state"]
        # independent run of the same chain
        p = live_page.normalize_port(self.CFG["portfolios"][0])
        tks, wts, errs, comp = parse_portfolio(p)
        px, _ = _fake_fetch(tuple(sorted(set(["SPY"] + tks))), "2020-11-12")
        aligned = align_price_data(px, "SPY", pd.Timestamp("2020-12-02").date(), sorted(tks))
        prep = prepare_portfolio(p, tks, wts, comp, aligned["price_df"])
        self.assertEqual(prep["groups"], {"ETH-USD": "__slot2", "MSTR": "__slot2"})
        self.assertEqual(prep["thr_dn"], {"*": 0.8, "__slot2": 0.6})
        ref = lc.shadow_state(STRAT_RD_MIXED, aligned["price_df"], prep["w_series"], prep["thr_dn"], prep["thr_up"],
                              prep["groups"], BAND_MODE_RATIO)
        pd.testing.assert_series_equal(st["elem_weights"], ref["elem_weights"])
        self.assertEqual(st["ref_date"], ref["ref_date"])
        self.assertEqual(list(st["slot_table"]["label"]), ["QQQM", "BRK-B", "ETH-USD+MSTR"])
        crypto = st["slot_table"].set_index("label").loc["ETH-USD+MSTR"]
        self.assertAlmostEqual(crypto["trig_low"], .05 * .4 / (1 - .05 * .6), places=10)
        self.assertAlmostEqual(crypto["trig_high"], .05 * 1.9 / (1 + .05 * .9), places=10)

    def test_orders_from_the_shadow(self):
        sh = live_page.compute_shadow(self.CFG, 0, _fake_fetch)
        emap = lc.mapping_for(sh["elements"])
        prices = {"QQQM": 500.0, "BRK-B": 450.0, "BMNR": 40.0, "MSTR": 300.0}
        hold = {"QQQM": 700, "BRK.B": 300, "BMNR": 120, "MSTR": 10}
        plan = lc.plan_orders({lc.norm_symbol(k): v for k, v in hold.items()}, prices, emap, sh["state"]["elem_weights"],
                              flow=0, mode="sync", min_trade=0)
        self.assertLess(plan["elements"]["dev_after"].abs().max(), 0.002)
        self.assertGreaterEqual(plan["cash_left"], 0)


if __name__ == "__main__":
    unittest.main()


class StaleCoreGuardTest(unittest.TestCase):
    def test_reloads_a_stale_backtest_core(self):
        import backtest_core
        import backtest_app as app
        saved = backtest_core.prepare_portfolio
        del backtest_core.prepare_portfolio                        # what a cached pre-2.6.0 module looks like
        try:
            app._fresh_core()
            self.assertTrue(hasattr(backtest_core, "prepare_portfolio"))
        finally:
            if not hasattr(backtest_core, "prepare_portfolio"):
                backtest_core.prepare_portfolio = saved
