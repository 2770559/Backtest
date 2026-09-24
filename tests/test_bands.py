"""v2.5.0 asymmetric / per-slot bands + turnover & drift stats.

Run with:  python3 -m unittest discover -s tests

Semantics under test (backtest_core):
  d = (w - target) / target                  signed relative deviation per slot
  trigger  <=>  d > U  or  d < -D            U = threshold_up, D = threshold
  threshold_up=None  ==>  U = D  ==>  bit-identical to the pre-2.5.0 |d| > D test
  (tests/legacy_engine_v240.py is a verbatim copy of that engine).
  band_mode="ratio" (v2.5.2): d is replaced by g − 1, g = (w/t) / ((1−w)/(1−t)),
  the slot's cumulative return relative to the rest since its last reset — the
  same U / D mean the same thing for every slot size. band_mode="rel" (default)
  is the untouched original path.
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
    BAND_MODE_REL, BAND_MODE_RATIO, BAND_MODES,
    apply_local_rebalance, run_detailed_backtest, parse_portfolio, ratio_deviation,
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


# --------------------------------------------------------------------------- #
# v2.5.2 band_mode="ratio" (size-neutral: leg return vs the rest of the portfolio)
# --------------------------------------------------------------------------- #
def _w_after(t, r):
    """Weight of a slot with target t after it beat the rest of the portfolio by r."""
    return t * (1 + r) / (1 + t * r)


class TestRatioDeviation(unittest.TestCase):
    def test_hand_computed(self):
        tgt = pd.Series([0.35, 0.65], index=["A", "B"])
        out = ratio_deviation(pd.Series([0.5, 0.5], index=["A", "B"]), tgt)
        # A: (0.5/0.35) / (0.5/0.65) = 0.65/0.35 = 1.857...; B is its reciprocal.
        self.assertAlmostEqual(out["A"], 0.65 / 0.35 - 1, places=12)
        self.assertAlmostEqual(out["B"], 0.35 / 0.65 - 1, places=12)
        at_target = ratio_deviation(tgt, tgt)
        self.assertAlmostEqual(at_target["A"], 0.0, places=12)
        self.assertAlmostEqual(at_target["B"], 0.0, places=12)

    def test_size_neutral_identity_g_minus_1_equals_r(self):
        # Whatever the target, a slot that beat the rest by r reads exactly r,
        # while the relative deviation d = (w − t)/t depends on t.
        for t in (0.05, 0.10, 0.15, 0.35, 0.65, 0.90):
            for r in (-0.8, -0.5, -0.2, 0.3, 0.5, 1.0, 3.0):
                w = _w_after(t, r)
                got = ratio_deviation(pd.Series([w, 1 - w]), pd.Series([t, 1 - t]))
                self.assertAlmostEqual(got.iloc[0], r, places=10, msg=(t, r))
                self.assertAlmostEqual(got.iloc[1], 1 / (1 + r) - 1, places=10, msg=(t, r))
                d = (w - t) / t
                if abs(r) > 0.05 and t > 0.1:
                    self.assertNotAlmostEqual(d, r, places=2, msg=(t, r))

    def test_rel_band_is_size_biased_ratio_band_is_not(self):
        # The research table: U = 60% under "rel" needs +136% vs the rest for a 35%
        # slot but only +71% for a 10% slot; under "ratio" both need exactly +60%.
        def r_needed_rel(t, d):
            return d / ((1 - t) - t * d)
        self.assertAlmostEqual(r_needed_rel(0.35, 0.6), 1.3636, places=3)
        self.assertAlmostEqual(r_needed_rel(0.10, 0.6), 0.7143, places=3)
        for t in (0.35, 0.10):
            w = _w_after(t, 0.6)
            self.assertAlmostEqual(ratio_deviation(pd.Series([w, 1 - w]), pd.Series([t, 1 - t])).iloc[0],
                                   0.6, places=10)

    def test_degenerate_cases(self):
        # target 1: no rest -> 0 (never triggers); w = 1 with t < 1 -> +inf; w = 0 -> −1.
        self.assertEqual(ratio_deviation(pd.Series([1.0]), pd.Series([1.0])).iloc[0], 0.0)
        out = ratio_deviation(pd.Series([1.0, 0.0], index=["A", "B"]), pd.Series([0.35, 0.65], index=["A", "B"]))
        self.assertTrue(np.isposinf(out["A"]))
        self.assertEqual(out["B"], -1.0)
        self.assertFalse(out.isna().any())


class TestRatioModeTriggerSingleton(unittest.TestCase):
    """A doubles against a flat rest (r = +1): ratio reads exactly 1.0 for a 35%
    slot AND a 10% slot; the relative deviation reads 0.481 and 0.818."""

    def _cnt(self, t, D, U, mode, strat=STRAT_RD_FULL, a=(100, 200, 200)):
        w = pd.Series([t, 1 - t], index=["A", "B"])
        price = _df({"A": list(a), "B": [100] * len(a)})
        return run_detailed_backtest(strat, price, w, 10000, D, threshold_up=U, band_mode=mode)[1]

    def test_up_trigger_sits_exactly_at_r(self):
        for t in (0.35, 0.10):
            self.assertEqual(self._cnt(t, 0.99, 0.99, BAND_MODE_RATIO), 1, t)   # 1.0 > 0.99
            self.assertEqual(self._cnt(t, 0.99, 1.01, BAND_MODE_RATIO), 0, t)   # 1.0 < 1.01

    def test_rel_mode_same_inputs_is_size_biased(self):
        # U = 60%: the 10% slot triggers (d = 0.818), the 35% slot does not (d = 0.481).
        self.assertEqual(self._cnt(0.10, 0.99, 0.6, BAND_MODE_REL), 1)
        self.assertEqual(self._cnt(0.35, 0.99, 0.6, BAND_MODE_REL), 0)
        # ... while under "ratio" both read 1.0 and both trigger.
        self.assertEqual(self._cnt(0.10, 0.99, 0.6, BAND_MODE_RATIO), 1)
        self.assertEqual(self._cnt(0.35, 0.99, 0.6, BAND_MODE_RATIO), 1)

    def test_down_trigger_sits_exactly_at_r(self):
        # A halves against the rest (r = −0.5): ratio −0.5; rel d = −0.394 for t = 0.35.
        for t in (0.35, 0.10):
            self.assertEqual(self._cnt(t, 0.49, 5.0, BAND_MODE_RATIO, a=(100, 50, 50)), 1, t)
            self.assertEqual(self._cnt(t, 0.51, 5.0, BAND_MODE_RATIO, a=(100, 50, 50)), 0, t)
        self.assertEqual(self._cnt(0.35, 0.49, 5.0, BAND_MODE_REL, a=(100, 50, 50)), 0)

    def test_down_band_at_or_above_100_never_triggers_downward(self):
        # g >= 0, so g − 1 >= −1: D = 100% can only be reached, never crossed —
        # the same property as the relative deviation. (A −99% collapse is the
        # OTHER slot's +9900% vs the rest, so U is parked far away to isolate D.)
        self.assertEqual(self._cnt(0.35, 1.0, 1e6, BAND_MODE_RATIO, a=(100, 1, 1)), 0)
        self.assertEqual(self._cnt(0.35, 1.0, 1e6, BAND_MODE_REL, a=(100, 1, 1)), 0)
        # ... and with U back in range that collapse IS the rest's up-breach.
        self.assertEqual(self._cnt(0.35, 1.0, 5.0, BAND_MODE_RATIO, a=(100, 1, 1)), 1)

    def test_global_reset_restores_targets(self):
        w = pd.Series([0.35, 0.65], index=["A", "B"])
        price = _df({"A": [100, 200, 200], "B": [100, 100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, price, w, 10000, 0.99, threshold_up=0.99,
                                             band_mode=BAND_MODE_RATIO)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"]), ("35.00%", "65.00%"))

    def test_single_slot_portfolio_never_triggers(self):
        w = pd.Series([1.0], index=["A"])
        price = _df({"A": [100, 300, 20, 50]})
        for mode in BAND_MODES:
            hist, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, price, w, 10000, 0.2, band_mode=mode)
            self.assertEqual(cnt, 0, mode)
            self.assertFalse(hist.empty)

    def test_invalid_mode_raises(self):
        w = pd.Series([0.35, 0.65], index=["A", "B"])
        price = _df({"A": [100, 200], "B": [100, 100]})
        with self.assertRaises(ValueError):
            run_detailed_backtest(STRAT_RD_FULL, price, w, 10000, 0.4, band_mode="abs")
        with self.assertRaises(ValueError):
            apply_local_rebalance(pd.Series([1.0, 1.0]), pd.Series([0.5, 0.5]), 0.4, band_mode="")


class TestRatioModeComposite(unittest.TestCase):
    """(A, B) is one 35% slot next to C 65%. Decisions use the slot AGGREGATE."""

    def setUp(self):
        self.w = pd.Series([0.175, 0.175, 0.65], index=["A", "B", "C"])
        self.groups = {"A": "AB", "B": "AB"}

    def _run(self, a, b, U, strat=STRAT_RD_FULL):
        price = _df({"A": a, "B": b, "C": [100] * len(a)})
        return run_detailed_backtest(strat, price, self.w, 10000, 0.99, groups=self.groups,
                                     threshold_up=U, band_mode=BAND_MODE_RATIO)

    def test_slot_aggregate_doubling_triggers(self):
        self.assertEqual(self._run([100, 200, 200], [100, 200, 200], 0.99)[1], 1)
        self.assertEqual(self._run([100, 200, 200], [100, 200, 200], 1.01)[1], 0)

    def test_internal_drift_never_triggers(self):
        # A doubles, B halves: the slot is 0.4375 vs 0.35 -> r = +0.25 only.
        self.assertEqual(self._run([100, 200, 200], [100, 50, 50], 0.26)[1], 0)
        self.assertEqual(self._run([100, 200, 200], [100, 50, 50], 0.24)[1], 1)

    def test_reset_restores_equal_split(self):
        hist, cnt, _ = self._run([100, 400, 400], [100, 100, 100], 0.99)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"], post["C"]), ("17.50%", "17.50%", "65.00%"))


class TestRatioModeMixedAndLocal(unittest.TestCase):
    """A 5% / B 60% / C 35%. A quadruples, B +10%, C flat -> values 0.2 / 0.66 / 0.35.
    Ratio readings: A +2.76 (rel: +2.31), B −0.20 (rel: −0.09), C −0.24 (rel: −0.17)."""

    def setUp(self):
        self.w = pd.Series([0.05, 0.60, 0.35], index=["A", "B", "C"])
        self.price = _df({"A": [100, 400, 400], "B": [100, 110, 110], "C": [100, 100, 100]})
        self.vals = pd.Series([200.0, 660.0, 350.0], index=["A", "B", "C"])

    def test_readings(self):
        got = ratio_deviation(self.vals / self.vals.sum(), self.w)
        self.assertAlmostEqual(got["A"], 4 / (1.01 / 0.95) - 1, places=12)      # A ×4 vs rest ×1.0632
        self.assertAlmostEqual(got["B"], 1.1 / (0.55 / 0.40) - 1, places=12)    # B ×1.1 vs rest ×1.375
        self.assertAlmostEqual(got["C"], 1.0 / (0.86 / 0.65) - 1, places=12)    # C ×1.0 vs rest ×1.323

    def test_minor_breach_is_local_reset_under_mixed(self):
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_MIXED, self.price, self.w, 10000, 0.5,
                                             threshold_up=2.5, band_mode=BAND_MODE_RATIO)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        # A back to 5%; B and C share the remainder in their pre-reset proportion
        # (660 : 350), NOT reset to 60 / 35.
        total = 1210.0
        b_exp = (total - 0.05 * total) * 660 / 1010 / total
        self.assertEqual(post["A"], "5.00%")
        self.assertAlmostEqual(_pct(post["B"]), b_exp, places=4)
        self.assertAlmostEqual(_pct(post["C"]), 0.95 - b_exp, places=4)
        # The same U = 2.5 does NOT trigger under "rel" (A reads +2.31 there).
        self.assertEqual(run_detailed_backtest(STRAT_RD_MIXED, self.price, self.w, 10000, 0.5,
                                               threshold_up=2.5, band_mode=BAND_MODE_REL)[1], 0)

    def test_major_breach_is_global_reset_under_mixed(self):
        w = pd.Series([0.35, 0.60, 0.05], index=["A", "B", "C"])
        price = _df({"A": [100, 300, 300], "B": [100, 100, 100], "C": [100, 100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.5, threshold_up=1.5,
                                             band_mode=BAND_MODE_RATIO)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"], post["C"]), ("35.00%", "60.00%", "5.00%"))

    def test_apply_local_rebalance_ratio_direct(self):
        out, resets = apply_local_rebalance(self.vals, self.w, 0.5, return_resets=True, threshold_up=2.5,
                                            band_mode=BAND_MODE_RATIO)
        self.assertEqual(resets, {"A"})
        self.assertAlmostEqual(out["A"], 0.05 * 1210.0, places=9)
        self.assertAlmostEqual(out["B"], (1210.0 - 60.5) * 660 / 1010, places=9)
        self.assertAlmostEqual(out["C"], (1210.0 - 60.5) * 350 / 1010, places=9)
        self.assertAlmostEqual(out.sum(), 1210.0, places=9)
        # Under "rel" the same bands leave everything untouched.
        out_rel, resets_rel = apply_local_rebalance(self.vals, self.w, 0.5, return_resets=True,
                                                    threshold_up=2.5, band_mode=BAND_MODE_REL)
        self.assertEqual(resets_rel, set())
        pd.testing.assert_series_equal(out_rel, self.vals, check_exact=True)

    def test_local_cascade_retests_remainder_in_ratio_terms(self):
        # A 35% / B 20% / C 45%; A ×2.05, B ×0.2, C ×1.5 -> values 71.75 / 4 / 67.5.
        # Pass 1 readings: A +0.864 (breach at U = 0.8), B −0.885 (inside D = 0.95),
        # C +0.089. Resetting A hands its excess to B and C (×1.302): C now reads
        # +0.941 and breaches on the SECOND pass; B (−0.849) still does not. Under
        # "rel" nothing breaches at all (A reads +0.431, C +0.047).
        w = pd.Series([0.35, 0.20, 0.45], index=["A", "B", "C"])
        vals = pd.Series([71.75, 4.0, 67.5], index=["A", "B", "C"])
        first = ratio_deviation(vals / vals.sum(), w)
        self.assertGreater(first["A"], 0.8)
        self.assertLess(first["C"], 0.8)                       # C is inside the band before the reset
        self.assertGreater(first["B"], -0.95)
        out, resets = apply_local_rebalance(vals, w, 0.95, return_resets=True, threshold_up=0.8,
                                            band_mode=BAND_MODE_RATIO)
        self.assertEqual(resets, {"A", "C"})
        total = 143.25
        self.assertAlmostEqual(out["A"], 0.35 * total, places=9)
        self.assertAlmostEqual(out["C"], 0.45 * total, places=9)
        self.assertAlmostEqual(out["B"], 0.20 * total, places=9)   # last one standing takes the rest
        out_rel, resets_rel = apply_local_rebalance(vals, w, 0.95, return_resets=True, threshold_up=0.8,
                                                    band_mode=BAND_MODE_REL)
        self.assertEqual(resets_rel, set())
        pd.testing.assert_series_equal(out_rel, vals, check_exact=True)
        # Same geometry through the engine (Local strategy): one rebalance, all at target.
        price = _df({"A": [100, 205, 205], "B": [100, 20, 20], "C": [100, 150, 150]})
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_LOCAL, price, w, 10000, 0.95, threshold_up=0.8,
                                             band_mode=BAND_MODE_RATIO)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual((post["A"], post["B"], post["C"]), ("35.00%", "20.00%", "45.00%"))
        self.assertEqual(run_detailed_backtest(STRAT_RD_LOCAL, price, w, 10000, 0.95, threshold_up=0.8,
                                               band_mode=BAND_MODE_REL)[1], 0)

    def test_per_slot_dict_bands_in_ratio_mode(self):
        # Tight U only on A (a "crypto sleeve" style override), wide elsewhere.
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_MIXED, self.price, self.w, 10000, {"*": 0.9},
                                             threshold_up={"*": 5.0, "A": 2.5}, band_mode=BAND_MODE_RATIO)
        self.assertEqual(cnt, 1)
        self.assertEqual(run_detailed_backtest(STRAT_RD_MIXED, self.price, self.w, 10000, {"*": 0.9},
                                               threshold_up={"*": 5.0, "A": 3.0},
                                               band_mode=BAND_MODE_RATIO)[1], 0)


class TestRatioModeBitIdentityAndScope(unittest.TestCase):
    def _assert_same(self, a, b):
        pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], b[2])

    def test_explicit_rel_mode_is_the_legacy_engine(self):
        for seed in range(8):
            price, w, groups = _random_case(seed)
            for strat in (STRAT_RD_FULL, STRAT_RD_MIXED, STRAT_RD_LOCAL, STRAT_ASYM):
                for thr in (0.2, 0.4):
                    ref = legacy.run_detailed_backtest(strat, price, w, 10000, thr, groups=groups)
                    self._assert_same(ref, run_detailed_backtest(strat, price, w, 10000, thr, groups=groups,
                                                                 band_mode=BAND_MODE_REL))

    def test_apply_local_rebalance_explicit_rel_bit_identical(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            n = int(rng.integers(2, 7))
            idx = [chr(65 + k) for k in range(n)]
            vals = pd.Series(rng.uniform(50, 5000, size=n), index=idx)
            tgt = rng.dirichlet(np.ones(n))
            tgt = pd.Series(tgt / tgt.sum(), index=idx)
            thr = float(rng.choice([0.1, 0.2, 0.3, 0.5]))
            ref_v, ref_r = legacy.apply_local_rebalance(vals, tgt, thr, return_resets=True)
            v, r = apply_local_rebalance(vals, tgt, thr, return_resets=True, band_mode=BAND_MODE_REL)
            pd.testing.assert_series_equal(ref_v, v, check_exact=True)
            self.assertEqual(ref_r, r)

    def test_ratio_mode_changes_some_decisions(self):
        diffs = 0
        for seed in range(12):
            price, w, groups = _random_case(seed)
            a = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.3, groups=groups)[1]
            b = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.3, groups=groups,
                                      band_mode=BAND_MODE_RATIO)[1]
            diffs += int(a != b)
        self.assertGreater(diffs, 0)

    def test_non_reldiff_strategies_ignore_band_mode(self):
        for seed in range(6):
            price, w, groups = _random_case(seed)
            for strat in (STRAT_ASYM, STRAT_ANNUAL, STRAT_SEMI, STRAT_BH):
                self._assert_same(
                    run_detailed_backtest(strat, price, w, 10000, 0.3, groups=groups, band_mode=BAND_MODE_REL),
                    run_detailed_backtest(strat, price, w, 10000, 0.3, groups=groups, band_mode=BAND_MODE_RATIO))

    def test_stats_arity_unchanged_in_ratio_mode(self):
        price, w, groups = _random_case(3)
        four = run_detailed_backtest(STRAT_RD_MIXED, price, w, 10000, 0.3, groups=groups, return_stats=True,
                                     band_mode=BAND_MODE_RATIO)
        self.assertEqual(len(four), 4)
        self.assertIn("turnover_yr", four[3])


if __name__ == "__main__":
    unittest.main()
