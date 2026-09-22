"""v2.5.0 asymmetric / per-slot bands + turnover & drift stats.

Run with:  python3 -m unittest discover -s tests

Semantics under test (backtest_core):
  d = (w - target) / target                  signed relative deviation per slot
  trigger  <=>  d > U  or  d < -D            U = threshold_up, D = threshold
  threshold_up=None  ==>  U = D  ==>  bit-identical to the pre-2.5.0 |d| > D test
  (tests/legacy_engine_v240.py is a verbatim copy of that engine).
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
    STRAT_BH, STRAT_ANNUAL, STRAT_SEMI, STRAT_RD_FULL, STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_ASYM,
    apply_local_rebalance, run_detailed_backtest, parse_portfolio,
    slot_labels, normalize_slot_bands, build_band_thresholds,
)
import legacy_engine_v240 as legacy  # noqa: E402


def _df(cols, start="2020-01-31", freq="ME"):
    n = len(next(iter(cols.values())))
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(cols, index=idx)


def _pct(s):
    return float(str(s).rstrip("%")) / 100.0


# --------------------------------------------------------------------------- #
# Config-model helpers
# --------------------------------------------------------------------------- #
class TestSlotLabels(unittest.TestCase):
    def test_matches_parse_portfolio_labels(self):
        port = {"tickers": "QQQM, brk.b, (ETH-USD, MSTR), 00700.HK", "weights": "0.4, 0.3, 0.2, 0.1"}
        _, _, errs, comp = parse_portfolio(port)
        self.assertEqual(errs, [])
        self.assertEqual(slot_labels(port["tickers"]), comp["slot_labels"])
        self.assertEqual(slot_labels(port["tickers"]), ["QQQM", "BRK-B", "ETH-USD+MSTR", "0700.HK"])

    def test_plain_and_fullwidth(self):
        self.assertEqual(slot_labels("QQQM， SPY,"), ["QQQM", "SPY"])
        self.assertEqual(slot_labels("（DBMF，KMLM）, SPY"), ["DBMF+KMLM", "SPY"])
        self.assertEqual(slot_labels(""), [])
        self.assertEqual(slot_labels("(A, B"), [])          # unbalanced -> no labels


class TestNormalizeSlotBands(unittest.TestCase):
    def test_coercion(self):
        raw = {"ETH-USD+MSTR": {"down": "40", "up": 40.0}, " X ": {"down": None, "up": 30},
               "junk": "nope", "empty": {}, "bad": {"down": "abc"}}
        out = normalize_slot_bands(raw)
        self.assertEqual(out, {"ETH-USD+MSTR": {"down": 40, "up": 40},
                               "X": {"down": None, "up": 30}})

    def test_non_dict(self):
        self.assertEqual(normalize_slot_bands(None), {})
        self.assertEqual(normalize_slot_bands([1, 2]), {})


class TestBuildBandThresholds(unittest.TestCase):
    L2I = {"QQQM": "QQQM", "ETH-USD+MSTR": "__slot2", "GLDM": "GLDM"}

    def test_legacy_config_takes_scalar_path(self):
        self.assertEqual(build_band_thresholds(40, None, {}, self.L2I), (0.4, None))
        self.assertEqual(build_band_thresholds(40, 40, None, self.L2I), (0.4, None))

    def test_asymmetric_scalars(self):
        self.assertEqual(build_band_thresholds(60, 100, {}, self.L2I), (0.6, 1.0))

    def test_per_slot_overrides_map_labels_to_engine_ids(self):
        d, u = build_band_thresholds(60, 100, {"ETH-USD+MSTR": {"down": 40, "up": 40}}, self.L2I)
        self.assertEqual(d, {"*": 0.6, "__slot2": 0.4})
        self.assertEqual(u, {"*": 1.0, "__slot2": 0.4})

    def test_one_sided_override_keeps_other_side_scalar(self):
        d, u = build_band_thresholds(60, 60, {"GLDM": {"down": 30, "up": None}}, self.L2I)
        self.assertEqual(d, {"*": 0.6, "GLDM": 0.3})
        self.assertEqual(u, 0.6)

    def test_unknown_labels_ignored(self):
        self.assertEqual(build_band_thresholds(40, 40, {"NOPE": {"down": 10, "up": 10}}, self.L2I), (0.4, None))

    def test_identical_dicts_collapse_to_none(self):
        d, u = build_band_thresholds(40, 40, {"GLDM": {"down": 20, "up": 20}}, self.L2I)
        self.assertEqual(d, {"*": 0.4, "GLDM": 0.2})
        self.assertIsNone(u)


# --------------------------------------------------------------------------- #
# Hand-calculated trigger cases
# --------------------------------------------------------------------------- #
class TestAsymmetricTriggerSingleton(unittest.TestCase):
    """A halves: A -44.4% below target, B/C +11.1% above (3 bars; bar 3 flat)."""

    def setUp(self):
        self.w = pd.Series([0.2, 0.4, 0.4], index=["A", "B", "C"])
        self.price = _df({"A": [100, 50, 50], "B": [100, 100, 100], "C": [100, 100, 100]})

    def _cnt(self, D, U, strat=STRAT_RD_FULL):
        return run_detailed_backtest(strat, self.price, self.w, 10000, D, threshold_up=U)[1]

    def test_neither_side_breached(self):
        self.assertEqual(self._cnt(0.6, 0.3), 0)

    def test_down_breach_only(self):
        self.assertEqual(self._cnt(0.4, 0.3), 1)          # -0.444 < -0.4, +0.111 < 0.3

    def test_up_breach_only(self):
        self.assertEqual(self._cnt(0.6, 0.1), 1)          # +0.111 > 0.1, -0.444 > -0.6

    def test_symmetric_arg_matches_legacy_trigger(self):
        # |−0.444| > 0.3 -> the symmetric 30% band triggers; (D=0.6, U=0.3) does not.
        self.assertEqual(self._cnt(0.3, None), 1)
        self.assertEqual(self._cnt(0.3, 0.3), 1)

    def test_global_reset_restores_targets_on_up_breach(self):
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, self.price, self.w, 10000, 0.6, threshold_up=0.1)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"], post["C"]), ("20.00%", "40.00%", "40.00%"))


class TestAsymmetricTriggerComposite(unittest.TestCase):
    """Same geometry with the 20% slot as a composite (A, B): both halve."""

    def setUp(self):
        self.w = pd.Series([0.1, 0.1, 0.4, 0.4], index=["A", "B", "C", "D"])
        self.groups = {"A": "S", "B": "S"}
        self.price = _df({"A": [100, 50, 50], "B": [100, 50, 50], "C": [100, 100, 100], "D": [100, 100, 100]})

    def _cnt(self, D, U):
        return run_detailed_backtest(STRAT_RD_FULL, self.price, self.w, 10000, D,
                                     groups=self.groups, threshold_up=U)[1]

    def test_neither(self):
        self.assertEqual(self._cnt(0.6, 0.3), 0)

    def test_down_only_slot_aggregate(self):
        self.assertEqual(self._cnt(0.4, 0.3), 1)

    def test_up_only_other_slots(self):
        self.assertEqual(self._cnt(0.6, 0.1), 1)

    def test_internal_drift_never_triggers(self):
        # A doubles, B halves: slot aggregate ~ flat -> no trigger even with tight bands.
        price = _df({"A": [100, 200], "B": [100, 50], "C": [100, 100], "D": [100, 100]})
        cnt = run_detailed_backtest(STRAT_RD_FULL, price, self.w, 10000, 0.2,
                                    groups=self.groups, threshold_up=0.2)[1]
        self.assertEqual(cnt, 0)

    def test_reset_restores_equal_split(self):
        price = _df({"A": [100, 800], "B": [100, 100], "C": [100, 100], "D": [100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, price, self.w, 10000, 0.9,
                                             groups=self.groups, threshold_up=0.3)
        self.assertEqual(cnt, 1)                           # S = 900/2700 = 33% vs 20% -> +67% > U
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], post["B"])
        self.assertEqual(post["A"], "10.00%")


class TestMixedRuleWithAsymmetricBands(unittest.TestCase):
    def test_minor_up_breach_is_local_reset(self):
        # A (5%) +50% -> d = +0.463 > U=0.4 while B/C are -2.4% (inside).
        # Mixed: only a MINOR slot breached -> local reset: A back to 5%, B/C
        # share the remainder pro-rata (1:1) -> 47.5% each.
        w = pd.Series([0.05, 0.475, 0.475], index=["A", "B", "C"])
        price = _df({"A": [100, 150], "B": [100, 100], "C": [100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.9, threshold_up=0.4)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"], post["C"]), ("5.00%", "47.50%", "47.50%"))

    def test_major_up_breach_is_global_reset(self):
        w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        price = _df({"A": [100, 200], "B": [100, 100], "C": [100, 100]})   # A: 66.7% -> d=+0.333
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.9, threshold_up=0.3)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"], post["C"]), ("50.00%", "30.00%", "20.00%"))


class TestLocalRebalanceAsymmetric(unittest.TestCase):
    def test_wide_down_band_does_not_trigger_on_deep_underweight(self):
        # A (5%) -60% -> d = -0.588: inside D=0.9, and B/C at +3.1% inside U=0.2.
        # The old symmetric 20% test WOULD have fired on |−0.588|.
        w = pd.Series([0.05, 0.475, 0.475], index=["A", "B", "C"])
        price = _df({"A": [100, 40], "B": [100, 100], "C": [100, 100]})
        for strat in (STRAT_RD_LOCAL, STRAT_RD_MIXED):
            cnt_asym = run_detailed_backtest(strat, price, w, 10000, 0.9, threshold_up=0.2)[1]
            cnt_sym = run_detailed_backtest(strat, price, w, 10000, 0.2)[1]
            self.assertEqual(cnt_asym, 0, strat)
            self.assertEqual(cnt_sym, 1, strat)

    def test_apply_local_rebalance_direct(self):
        vals = pd.Series([750.0, 4750.0, 4750.0], index=["A", "B", "C"])
        tgt = pd.Series([0.05, 0.475, 0.475], index=["A", "B", "C"])
        out, resets = apply_local_rebalance(vals, tgt, 0.9, return_resets=True, threshold_up=0.4)
        self.assertEqual(resets, {"A"})
        self.assertAlmostEqual(out["A"], 0.05 * 10250.0, places=9)
        self.assertAlmostEqual(out["B"], (10250.0 - 512.5) / 2, places=9)
        self.assertAlmostEqual(out["C"], out["B"], places=9)
        # Series-shaped bands are accepted (per-slot): tight U only on A.
        U = pd.Series([0.4, 5.0, 5.0], index=["A", "B", "C"])
        out2, resets2 = apply_local_rebalance(vals, tgt, 0.9, return_resets=True, threshold_up=U)
        pd.testing.assert_series_equal(out, out2, check_exact=True)
        self.assertEqual(resets2, {"A"})

    def test_cascade_uses_asymmetric_test_on_remainder(self):
        # After the first reset the remainder is re-scaled and re-tested with
        # the SAME asymmetric bands: D wide, U tight -> a remainder slot pushed
        # above U by the rescale must trigger; below -D it must not.
        vals = pd.Series([2000.0, 1000.0, 7000.0], index=["A", "B", "C"])
        tgt = pd.Series([0.1, 0.2, 0.7], index=["A", "B", "C"])
        # A at 20% (d=+1.0) triggers on U=0.5; B (d=-0.5) inside D=0.9.
        out, resets = apply_local_rebalance(vals, tgt, 0.9, return_resets=True, threshold_up=0.5)
        self.assertIn("A", resets)
        self.assertNotIn("B", resets)
        self.assertAlmostEqual(out.sum(), 10000.0, places=6)


# --------------------------------------------------------------------------- #
# Bit-identity vs the frozen v2.4.0 engine
# --------------------------------------------------------------------------- #
def _random_case(seed):
    rng = np.random.default_rng(seed)
    n_assets = int(rng.integers(3, 7))
    n_bars = int(rng.integers(36, 120))
    names = [chr(65 + k) for k in range(n_assets)]
    vol = rng.uniform(0.03, 0.25, size=n_assets)
    drift = rng.uniform(-0.01, 0.02, size=n_assets)
    rets = rng.normal(drift, vol, size=(n_bars, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    price = _df({n: prices[:, k] for k, n in enumerate(names)}, start="2005-01-31")
    w = rng.dirichlet(np.ones(n_assets) * 2)
    w = pd.Series(w / w.sum(), index=names)
    groups = None
    if n_assets >= 4 and seed % 2 == 0:
        groups = {names[0]: "__s0", names[1]: "__s0"}
    return price, w, groups


class TestBitIdentityWithLegacyEngine(unittest.TestCase):
    STRATS = (STRAT_RD_FULL, STRAT_RD_MIXED, STRAT_RD_LOCAL, STRAT_ASYM, STRAT_ANNUAL, STRAT_SEMI, STRAT_BH)

    def _assert_same(self, a, b):
        pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], b[2])            # exact float compare on pnl_rec['NAV']

    def test_symmetric_paths_bit_identical(self):
        n_rebal = 0
        for seed in range(10):
            price, w, groups = _random_case(seed)
            for strat in self.STRATS:
                for thr in (0.2, 0.4):
                    ref = legacy.run_detailed_backtest(strat, price, w, 10000, thr, groups=groups)
                    n_rebal += ref[1]
                    self._assert_same(ref, run_detailed_backtest(strat, price, w, 10000, thr, groups=groups))
                    self._assert_same(ref, run_detailed_backtest(strat, price, w, 10000, thr, groups=groups,
                                                                 threshold_up=thr))
                    self._assert_same(ref, run_detailed_backtest(strat, price, w, 10000, {"*": thr}, groups=groups,
                                                                 threshold_up={"*": thr}))
                    # 4-tuple form: first three elements unchanged
                    four = run_detailed_backtest(strat, price, w, 10000, thr, groups=groups, return_stats=True)
                    self.assertEqual(len(four), 4)
                    self._assert_same(ref, four[:3])
        self.assertGreater(n_rebal, 200, "random cases must actually exercise rebalances")

    def test_apply_local_rebalance_bit_identical(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            n = int(rng.integers(2, 7))
            idx = [chr(65 + k) for k in range(n)]
            vals = pd.Series(rng.uniform(50, 5000, size=n), index=idx)
            tgt = rng.dirichlet(np.ones(n))
            tgt = pd.Series(tgt / tgt.sum(), index=idx)
            thr = float(rng.choice([0.1, 0.2, 0.3, 0.5]))
            ref_v, ref_r = legacy.apply_local_rebalance(vals, tgt, thr, return_resets=True)
            for up in (None, thr):
                v, r = apply_local_rebalance(vals, tgt, thr, return_resets=True, threshold_up=up)
                pd.testing.assert_series_equal(ref_v, v, check_exact=True)
                self.assertEqual(ref_r, r)
            pd.testing.assert_series_equal(legacy.apply_local_rebalance(vals, tgt, thr),
                                           apply_local_rebalance(vals, tgt, thr), check_exact=True)

    def test_asymmetric_differs_when_it_should(self):
        # Sanity: U != D must change SOME decision across the random set.
        diffs = 0
        for seed in range(12):
            price, w, groups = _random_case(seed)
            a = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.3, groups=groups)[1]
            b = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.3, groups=groups, threshold_up=1.5)[1]
            diffs += int(a != b)
        self.assertGreater(diffs, 0)


# --------------------------------------------------------------------------- #
# Dict bands
# --------------------------------------------------------------------------- #
class TestBandDicts(unittest.TestCase):
    def setUp(self):
        self.w = pd.Series([0.6, 0.3, 0.1], index=["A", "B", "C"])

    def test_up_dict_with_default(self):
        # C +30%: inside the global U=0.4, outside its own U=0.2.
        price = _df({"A": [100, 100], "B": [100, 100], "C": [100, 130]})
        cnt_scalar = run_detailed_backtest(STRAT_RD_MIXED, price, self.w, 10000, 0.9, threshold_up=0.4)[1]
        cnt_dict = run_detailed_backtest(STRAT_RD_MIXED, price, self.w, 10000, 0.9,
                                         threshold_up={"C": 0.2, "*": 0.4})[1]
        self.assertEqual((cnt_scalar, cnt_dict), (0, 1))

    def test_down_dict_with_default(self):
        # C -30%: inside the global D=0.4, outside its own D=0.2; U wide.
        price = _df({"A": [100, 100], "B": [100, 100], "C": [100, 70]})
        cnt_scalar = run_detailed_backtest(STRAT_RD_MIXED, price, self.w, 10000, 0.4, threshold_up=0.9)[1]
        cnt_dict = run_detailed_backtest(STRAT_RD_MIXED, price, self.w, 10000, {"C": 0.2, "*": 0.4},
                                         threshold_up=0.9)[1]
        self.assertEqual((cnt_scalar, cnt_dict), (0, 1))

    def test_missing_slot_without_default_raises(self):
        price = _df({"A": [100, 100], "B": [100, 100], "C": [100, 100]})
        with self.assertRaises(ValueError):
            run_detailed_backtest(STRAT_RD_FULL, price, self.w, 10000, 0.4, threshold_up={"A": 0.4})
        with self.assertRaises(ValueError):
            run_detailed_backtest(STRAT_RD_FULL, price, self.w, 10000, {"A": 0.4}, threshold_up=0.4)

    def test_composite_slot_id_in_up_dict(self):
        w = pd.Series([0.05, 0.05, 0.9], index=["A", "B", "C"])
        groups = {"A": "__c", "B": "__c"}
        price = _df({"A": [100, 140], "B": [100, 140], "C": [100, 100]})   # slot +36% -> d ≈ +0.36
        wide = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.6, groups=groups,
                                     threshold_up={"*": 1.0})[1]
        tight = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.6, groups=groups,
                                      threshold_up={"*": 1.0, "__c": 0.3})[1]
        self.assertEqual((wide, tight), (0, 1))


# --------------------------------------------------------------------------- #
# Turnover & drift stats
# --------------------------------------------------------------------------- #
class TestTurnoverAndDrift(unittest.TestCase):
    def test_default_return_arity_unchanged(self):
        price = _df({"A": [100, 110], "B": [100, 100]})
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        self.assertEqual(len(run_detailed_backtest(STRAT_BH, price, w, 10000, 0.4)), 3)

    def test_known_global_reset_sold_amount(self):
        # A x4: $20,000 vs $5,000 -> RD_FULL reset to 12,500/12,500 -> sold 7,500.
        price = _df({"A": [100, 400, 400], "B": [100, 100, 100]})
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        _, cnt, _, s = run_detailed_backtest(STRAT_RD_FULL, price, w, 10000, 0.5, return_stats=True)
        self.assertEqual(cnt, 1)
        self.assertEqual(s["sold_total"], 7500.0)
        self.assertEqual(s["nav_mean"], (10000 + 25000 + 25000) / 3)
        days = (price.index[-1] - price.index[0]).days
        self.assertAlmostEqual(s["years"], days / 365.25, places=12)
        self.assertAlmostEqual(s["turnover_yr"], 7500.0 / 20000.0 / (days / 365.25), places=12)
        # Range uses Init / Post-Rebal / Hold states (NOT the 80% Pre-Rebal snapshot).
        self.assertEqual((s["weight_min"]["A"], s["weight_max"]["A"]), (0.5, 0.5))
        self.assertEqual(s["slot_target"], {"A": 0.5, "B": 0.5})
        self.assertEqual(s["slot_ids"], ["A", "B"])

    def test_buy_and_hold_zero_turnover_and_drift_range(self):
        price = _df({"A": [100, 200, 400], "B": [100, 100, 100]})
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        _, _, _, s = run_detailed_backtest(STRAT_BH, price, w, 10000, 0.5, return_stats=True)
        self.assertEqual(s["sold_total"], 0.0)
        self.assertEqual(s["turnover_yr"], 0.0)
        self.assertAlmostEqual(s["weight_min"]["A"], 0.5, places=12)
        self.assertAlmostEqual(s["weight_max"]["A"], 0.8, places=12)   # 20000 / 25000
        self.assertAlmostEqual(s["weight_min"]["B"], 0.2, places=12)

    def test_local_reset_sells_only_triggered_slot(self):
        # Minor A resets down to target; B/C are scaled UP (buys) -> sold == A's excess.
        w = pd.Series([0.05, 0.475, 0.475], index=["A", "B", "C"])
        price = _df({"A": [100, 150], "B": [100, 100], "C": [100, 100]})
        _, cnt, _, s = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.9, threshold_up=0.4,
                                             return_stats=True)
        self.assertEqual(cnt, 1)
        self.assertAlmostEqual(s["sold_total"], 750.0 - 0.05 * 10250.0, places=9)

    def test_composite_slot_range_is_aggregate(self):
        w = pd.Series([0.25, 0.25, 0.5], index=["A", "B", "C"])
        groups = {"A": "S", "B": "S"}
        price = _df({"A": [100, 200], "B": [100, 50], "C": [100, 100]})   # slot: 2500+... = 5000+1250 = 6250? see below
        _, _, _, s = run_detailed_backtest(STRAT_BH, price, w, 10000, 0.5, groups=groups, return_stats=True)
        # bar 2: A=5000, B=1250, C=5000 -> S = 6250/11250
        self.assertEqual(s["slot_ids"], ["S", "C"])
        self.assertEqual(s["slot_members"]["S"], ["A", "B"])
        self.assertAlmostEqual(s["weight_max"]["S"], 6250 / 11250, places=12)
        self.assertAlmostEqual(s["weight_min"]["S"], 0.5, places=12)

    def test_stats_agree_with_history_rows(self):
        """Post-hoc computation from the Pre-/Post-Rebal rows (the research
        scripts' method) must agree with the engine's exact figures up to the
        2-decimal percent rounding of the history strings."""
        for seed in range(10):
            price, w, groups = _random_case(seed)
            for strat in (STRAT_RD_MIXED, STRAT_RD_LOCAL, STRAT_RD_FULL, STRAT_ANNUAL):
                hist, cnt, _, s = run_detailed_backtest(strat, price, w, 10000, 0.3, groups=groups,
                                                        threshold_up=0.8, return_stats=True)
                elems = list(price.columns)
                pre = hist[hist["Type"] == "Pre-Rebal"].reset_index(drop=True)
                post = hist[hist["Type"] == "Post-Rebal"].reset_index(drop=True)
                self.assertEqual(len(pre), cnt); self.assertEqual(len(post), cnt)
                sold = 0.0
                for k in range(cnt):
                    nav = float(pre.loc[k, "NAV"])
                    sold += sum(max(0.0, (_pct(pre.loc[k, t]) - _pct(post.loc[k, t])) * nav) for t in elems)
                tol = cnt * len(elems) * 0.00005 * float(hist["NAV"].max()) + 1e-6
                self.assertAlmostEqual(sold, s["sold_total"], delta=tol)
                # NAV mean / years from the deduplicated per-date rows
                dedup = hist.drop_duplicates(subset="Date", keep="last")
                self.assertAlmostEqual(float(dedup["NAV"].mean()), s["nav_mean"], places=6)
                yrs = (pd.Timestamp(dedup["Date"].iloc[-1]) - pd.Timestamp(dedup["Date"].iloc[0])).days / 365.25
                self.assertAlmostEqual(yrs, s["years"], places=12)
                # Weight range from Init/Hold/Post-Rebal rows, slot-aggregated
                held = hist[hist["Type"].isin(["Init", "Hold", "Post-Rebal"])]
                for sid in s["slot_ids"]:
                    agg = held[s["slot_members"][sid]].map(_pct).sum(axis=1)
                    self.assertAlmostEqual(float(agg.min()), s["weight_min"][sid], delta=len(elems) * 0.00005 + 1e-9)
                    self.assertAlmostEqual(float(agg.max()), s["weight_max"][sid], delta=len(elems) * 0.00005 + 1e-9)

    def test_empty_price_df_returns_empty_stats(self):
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        out = run_detailed_backtest(STRAT_BH, pd.DataFrame(columns=["A", "B"]), w, 10000, 0.4, return_stats=True)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[3]["turnover_yr"], 0.0)


if __name__ == "__main__":
    unittest.main()
