"""v2.5.4 rebalance scope log + detail-table colours.

Engine (backtest_core.run_detailed_backtest, return_stats=True):
  stats["rebal_events"] = [{"date", "scope": "global" | "local", "trigger": [slot ids]}]
  global = every slot reset to target; local = only some slots reset, the rest scaled.
  The history frame and all decisions are unchanged (bit-identity vs the frozen
  v2.4.0 engine is covered in test_bands.py).
App (backtest_app): rebal_row_style colours Pre/Post-Rebal rows by scope;
rebal_legend_html shows the swatches and counts.

Run with:  python3 -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from backtest_core import (  # noqa: E402
    STRAT_BH, STRAT_ANNUAL, STRAT_RD_FULL, STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_ASYM, BAND_MODE_RATIO,
    run_detailed_backtest,
)
import backtest_app as app  # noqa: E402  (bare-mode import: warnings are harmless)


def _df(cols, start="2020-01-31", freq="ME"):
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=pd.date_range(start, periods=n, freq=freq))


def _run(strat, price, w, thr=0.4, **kw):
    return run_detailed_backtest(strat, price, pd.Series(w, index=price.columns), 10000, thr, return_stats=True, **kw)


class ScopeClassificationTest(unittest.TestCase):
    def test_mixed_minor_only_breach_is_local(self):
        # A (5%) quadruples: only the minor slot breaches -> local reset of A.
        hist, cnt, _, st = _run(STRAT_RD_MIXED, _df({"A": [100, 400, 400], "B": [100, 110, 110], "C": [100, 100, 100]}), [.05, .60, .35])
        self.assertEqual(cnt, 1)
        self.assertEqual(st["rebal_events"][0]["scope"], "local")
        self.assertEqual(st["rebal_events"][0]["trigger"], ["A"])
        self.assertEqual((st["rebal_global"], st["rebal_local"]), (0, 1))

    def test_mixed_major_breach_is_global(self):
        hist, cnt, _, st = _run(STRAT_RD_MIXED, _df({"A": [100, 300, 300], "B": [100, 100, 100], "C": [100, 100, 100]}), [.35, .60, .05])
        self.assertEqual(cnt, 1)
        ev = st["rebal_events"][0]
        self.assertEqual(ev["scope"], "global")
        self.assertIn("A", ev["trigger"])
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"], post["C"]), ("35.00%", "60.00%", "5.00%"))

    def test_full_is_global_even_for_a_minor_breach(self):
        hist, cnt, _, st = _run(STRAT_RD_FULL, _df({"A": [100, 400, 400], "B": [100, 110, 110], "C": [100, 100, 100]}), [.05, .60, .35])
        self.assertEqual([e["scope"] for e in st["rebal_events"]], ["global"])
        self.assertEqual(st["rebal_events"][0]["trigger"], ["A"])

    def test_local_strategy_single_breach_is_local(self):
        # A +80%: only A breaches the 40% band; B and C are scaled, not reset.
        hist, cnt, _, st = _run(STRAT_RD_LOCAL, _df({"A": [100, 180, 180], "B": [100, 100, 100], "C": [100, 100, 100]}), [.35, .60, .05])
        self.assertEqual(cnt, 1)
        self.assertEqual(st["rebal_events"][0]["scope"], "local")
        self.assertEqual(st["rebal_events"][0]["trigger"], ["A"])

    def test_local_cascade_that_resets_every_slot_is_global(self):
        # Two slots, both breach: the local reset touches every slot -> global in effect.
        hist, cnt, _, st = _run(STRAT_RD_LOCAL, _df({"A": [100, 200, 200], "B": [100, 100, 100]}), [.5, .5], thr=0.3)
        self.assertEqual(cnt, 1)
        self.assertEqual(st["rebal_events"][0]["scope"], "global")
        self.assertEqual(sorted(st["rebal_events"][0]["trigger"]), ["A", "B"])

    def test_periodic_is_global_with_no_trigger(self):
        price = _df({"A": np.linspace(100, 200, 30), "B": np.linspace(100, 90, 30)})
        hist, cnt, _, st = _run(STRAT_ANNUAL, price, [.5, .5])
        self.assertGreater(cnt, 0)
        self.assertTrue(all(e["scope"] == "global" and e["trigger"] == [] for e in st["rebal_events"]))

    def test_asymmetric_is_global_with_trigger(self):
        hist, cnt, _, st = _run(STRAT_ASYM, _df({"A": [100, 300, 300], "B": [100, 100, 100]}), [.4, .6], thr=0.38)
        self.assertEqual(cnt, 1)
        self.assertEqual(st["rebal_events"][0]["scope"], "global")
        self.assertIn("A", st["rebal_events"][0]["trigger"])

    def test_buy_and_hold_has_no_events(self):
        hist, cnt, _, st = _run(STRAT_BH, _df({"A": [100, 400, 50], "B": [100, 100, 100]}), [.5, .5])
        self.assertEqual((cnt, st["rebal_events"], st["rebal_global"], st["rebal_local"]), (0, [], 0, 0))

    def test_composite_minor_slot_is_local_and_named_by_slot_id(self):
        # (C1, C2) is one 5% slot; both elements triple -> the slot breaches alone.
        price = _df({"A": [100, 100, 100], "B": [100, 100, 100], "C1": [100, 300, 300], "C2": [100, 300, 300]})
        res, cnt, _, st = run_detailed_backtest(STRAT_RD_MIXED, price, pd.Series([.55, .40, .025, .025], index=price.columns), 10000, 0.4,
                                                groups={"C1": "__c", "C2": "__c"}, return_stats=True)
        self.assertEqual(cnt, 1)
        self.assertEqual(st["rebal_events"][0], {"date": price.index[1], "scope": "local", "trigger": ["__c"]})

    def test_ratio_mode_scope(self):
        # Same geometry as test_mixed_minor_only_breach_is_local, read in leg-vs-rest terms.
        hist, cnt, _, st = _run(STRAT_RD_MIXED, _df({"A": [100, 400, 400], "B": [100, 110, 110], "C": [100, 100, 100]}), [.05, .60, .35],
                                thr=0.5, threshold_up=2.5, band_mode=BAND_MODE_RATIO)
        self.assertEqual([e["scope"] for e in st["rebal_events"]], ["local"])


class EventLogConsistencyTest(unittest.TestCase):
    def _random_case(self, seed):
        rng = np.random.default_rng(seed)
        n_assets, n_bars = int(rng.integers(3, 7)), int(rng.integers(36, 120))
        names = [chr(65 + k) for k in range(n_assets)]
        rets = rng.normal(rng.uniform(-0.01, 0.02, n_assets), rng.uniform(0.03, 0.25, n_assets), size=(n_bars, n_assets))
        price = _df({n: 100 * np.exp(np.cumsum(rets[:, k])) for k, n in enumerate(names)}, start="2005-01-31")
        w = rng.dirichlet(np.ones(n_assets) * 2)
        groups = {names[0]: "__s0", names[1]: "__s0"} if n_assets >= 4 and seed % 2 == 0 else None
        return price, pd.Series(w / w.sum(), index=names), groups

    def test_events_match_history_and_counts(self):
        seen = set()
        for seed in range(12):
            price, w, groups = self._random_case(seed)
            for strat in (STRAT_RD_MIXED, STRAT_RD_LOCAL, STRAT_RD_FULL, STRAT_ASYM, STRAT_ANNUAL):
                hist, cnt, pnl, st = run_detailed_backtest(strat, price, w, 10000, 0.3, groups=groups, return_stats=True)
                ev = st["rebal_events"]
                self.assertEqual(len(ev), cnt)
                self.assertEqual(st["rebal_global"] + st["rebal_local"], cnt)
                pre_dates = list(hist.loc[hist["Type"] == "Pre-Rebal", "Date"])
                self.assertEqual([e["date"] for e in ev], pre_dates)
                seen.update(e["scope"] for e in ev)
                # the 3-tuple path returns the identical history
                h3, c3, p3 = run_detailed_backtest(strat, price, w, 10000, 0.3, groups=groups)
                pd.testing.assert_frame_equal(hist, h3, check_exact=True)
                self.assertEqual((cnt, pnl), (c3, p3))
                if strat in (STRAT_RD_FULL, STRAT_ASYM, STRAT_ANNUAL):
                    self.assertTrue(all(e["scope"] == "global" for e in ev), strat)
        self.assertEqual(seen, {"global", "local"}, "random cases must exercise both scopes")


class AppStylingTest(unittest.TestCase):
    ROW = lambda self, kind: pd.Series({"Date": "2026-08-31", "Type": kind, "NAV": 1.0, "QQQM": "35.00%"})

    def test_rebalance_rows_coloured_by_scope(self):
        g_pre, g_post = app.REBAL_COLORS["global"]
        l_pre, l_post = app.REBAL_COLORS["local"]
        self.assertEqual({g_pre, g_post} & {l_pre, l_post}, set())          # the two scopes share no colour
        self.assertTrue(all(l_pre in c for c in app.rebal_row_style(self.ROW("Pre-Rebal"), "local")))
        self.assertTrue(all(l_post in c for c in app.rebal_row_style(self.ROW("Post-Rebal"), "local")))
        self.assertTrue(all(g_pre in c for c in app.rebal_row_style(self.ROW("Pre-Rebal"), "global")))
        self.assertTrue(all(g_post in c for c in app.rebal_row_style(self.ROW("Post-Rebal"), "global")))
        # unknown scope (cached pre-2.5.4 engine) -> the global colours, as before
        self.assertTrue(all(g_post in c for c in app.rebal_row_style(self.ROW("Post-Rebal"), None)))
        self.assertEqual(len(app.rebal_row_style(self.ROW("Post-Rebal"), "local")), 4)

    def test_other_rows_unchanged(self):
        self.assertEqual(app.rebal_row_style(self.ROW("Hold"), "local"), [""] * 4)
        self.assertIn("#e3f2fd", app.rebal_row_style(self.ROW("Init"))[0])
        self.assertIn("#fce4ec", app.rebal_row_style(self.ROW("PnL Contrib%"))[0])

    def test_legend_has_counts_and_swatches(self):
        html = app.rebal_legend_html(1, 9)
        self.assertIn("Global 1", html)
        self.assertIn("Local 9", html)
        for c in app.REBAL_COLORS["global"] + app.REBAL_COLORS["local"]:
            self.assertIn(c, html)


if __name__ == "__main__":
    unittest.main()
