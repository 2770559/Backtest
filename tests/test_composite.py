"""Composite (聚合标的) tests + no-composite regression locks.

Run with:  python3 -m unittest discover -s tests

A "composite" is a single weight slot (e.g. 5%) made of N elements that default
to an EQUAL dollar split (2 -> 2.5%/2.5%; 4 -> 1.25% each). Syntax: wrap the
elements in parentheses inside the tickers field, "(DBMF, KMLM)" — the slot takes
one weight, split equally. The rebalance TRIGGER is decided on the slot's
AGGREGATE weight; elements never trigger individually. On a reset the slot is
restored to its equal default split; between resets the elements drift and the
slot moves "as one block".

Contract:
  parse_portfolio(port) -> (tickers, weights, errors, composite)   # 4-tuple
  run_detailed_backtest(..., groups=None)                          # optional dict element->slot-id
  groups=None == every column is its own slot == legacy behaviour, bit-identical.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_core import (
    STRAT_BH, STRAT_RD_FULL, STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_ASYM,
    parse_portfolio, run_detailed_backtest,
)


def _price_df(cols_prices, freq="ME", start="2020-01-31"):
    n = len(next(iter(cols_prices.values())))
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(cols_prices, index=idx)


class TestParseComposite(unittest.TestCase):
    def test_two_element_equal_split(self):
        tks, wts, errs, comp = parse_portfolio(
            {"tickers": "QQQM, (DBMF, KMLM), SPY", "weights": "0.5, 0.3, 0.2"})
        self.assertEqual(tks, ["QQQM", "DBMF", "KMLM", "SPY"])
        self.assertEqual(wts, [0.5, 0.15, 0.15, 0.2])     # 0.3 split equally
        self.assertEqual(errs, [])
        self.assertEqual(comp["slot_members"], [["QQQM"], ["DBMF", "KMLM"], ["SPY"]])
        self.assertEqual(comp["slot_targets"], [0.5, 0.3, 0.2])
        self.assertEqual(comp["element_slot"], [0, 1, 1, 2])

    def test_four_element_split_1_25(self):
        _, wts, errs, comp = parse_portfolio(
            {"tickers": "(A, B, C, D), SPY", "weights": "0.05, 0.95"})
        self.assertEqual(wts[:4], [0.0125, 0.0125, 0.0125, 0.0125])
        self.assertEqual(errs, [])
        self.assertEqual(comp["slot_labels"], ["A+B+C+D", "SPY"])

    def test_count_is_slot_level(self):
        # 2 slots (one composite) vs 2 weights -> OK
        _, _, errs, _ = parse_portfolio({"tickers": "(A, B), SPY", "weights": "0.5, 0.5"})
        self.assertEqual([e for e in errs if "vs" in e], [])

    def test_count_mismatch_uses_slot_count(self):
        _, _, errs, _ = parse_portfolio({"tickers": "(A, B), SPY", "weights": "0.5, 0.3, 0.2"})
        self.assertTrue(any("2 slots vs 3 weights" in e for e in errs))

    def test_weight_sum_slot_level(self):
        _, _, errs, _ = parse_portfolio({"tickers": "(A, B), SPY", "weights": "0.5, 0.6"})
        self.assertTrue(any("should be 1.0" in e for e in errs))

    def test_empty_composite_rejected(self):
        _, _, errs, _ = parse_portfolio({"tickers": "(), SPY", "weights": "0.5, 0.5"})
        self.assertTrue(any("empty composite" in e for e in errs))

    def test_single_element_composite_flagged(self):
        _, _, errs, _ = parse_portfolio({"tickers": "(A), SPY", "weights": "0.5, 0.5"})
        self.assertTrue(any(">= 2 elements" in e for e in errs))

    def test_nested_rejected(self):
        _, _, errs, comp = parse_portfolio({"tickers": "(A, (B, C)), SPY", "weights": "0.5, 0.5"})
        self.assertTrue(any("nested parentheses" in e for e in errs))
        self.assertIsNone(comp)

    def test_unbalanced_rejected(self):
        _, _, errs, _ = parse_portfolio({"tickers": "(A, B, SPY", "weights": "0.5, 0.5"})
        self.assertTrue(any("unbalanced" in e for e in errs))

    def test_duplicate_across_slots(self):
        _, _, errs, _ = parse_portfolio({"tickers": "(DBMF, KMLM), KMLM", "weights": "0.5, 0.5"})
        self.assertTrue(any("duplicate tickers" in e and "KMLM" in e for e in errs))

    def test_fullwidth_parens(self):
        tks, _, errs, comp = parse_portfolio(
            {"tickers": "（DBMF，KMLM）, SPY", "weights": "0.5, 0.5"})
        self.assertEqual(tks, ["DBMF", "KMLM", "SPY"])
        self.assertEqual(errs, [])
        self.assertIsNotNone(comp)


class TestEngineComposite(unittest.TestCase):
    def setUp(self):
        self.w = pd.Series([0.25, 0.25, 0.5], index=["A", "B", "C"])
        self.groups = {"A": "S", "B": "S"}   # (A, B) is one slot at 0.5

    def test_internal_drift_does_not_trigger_when_aggregate_in_band(self):
        # A doubles, B halves -> slot aggregate stays ~0.5; no element acts alone.
        # Aggregate rel-diff small -> 0 rebalances even at a tight-ish threshold.
        price = _price_df({"A": [100, 200, 200], "B": [100, 50, 50], "C": [100, 100, 100]})
        _, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, price, self.w, 10000, 0.5, groups=self.groups)
        self.assertEqual(cnt, 0)

    def test_aggregate_breach_resets_to_equal_split(self):
        # Both slot elements x4 -> slot aggregate balloons -> RD_FULL global reset,
        # elements restored to equal split (each = slot/2 = 25%).
        price = _price_df({"A": [100, 400, 400], "B": [100, 400, 400], "C": [100, 100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, price, self.w, 10000, 0.5, groups=self.groups)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], "25.00%")
        self.assertEqual(post["B"], "25.00%")
        self.assertEqual(post["C"], "50.00%")

    def test_unequal_drift_resets_to_equal(self):
        # A x8, B flat inside slot; on reset both snap back to equal 25%/25%.
        price = _price_df({"A": [100, 800, 800], "B": [100, 100, 100], "C": [100, 100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, price, self.w, 10000, 0.5, groups=self.groups)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], "25.00%")
        self.assertEqual(post["B"], "25.00%")

    def test_nav_is_element_sum(self):
        price = _price_df({"A": [100, 150, 200], "B": [100, 100, 100], "C": [100, 100, 100]})
        hist, _, _ = run_detailed_backtest(STRAT_BH, price, self.w, 10000, 0.5, groups=self.groups)
        # 25 sh A*200 + 25 sh B*100 + 50 sh C*100 = 5000 + 2500 + 5000 = 12500
        self.assertAlmostEqual(hist.iloc[-1]["NAV"], 12500, places=6)

    def test_slot_participates_via_aggregate(self):
        # The composite slot (target 0.5) breaching its band drives a global reset,
        # i.e. the slot acts as the trigger via its aggregate weight.
        price = _price_df({"A": [100, 1000, 1000], "B": [100, 1000, 1000], "C": [100, 100, 100]})
        _, cnt, _ = run_detailed_backtest(STRAT_ASYM, price, self.w, 10000, 0.4, groups=self.groups)
        self.assertGreaterEqual(cnt, 1)

    def test_local_resets_only_triggered_slot_to_equal(self):
        # RD_LOCAL: triggered composite slot resets to equal split.
        price = _price_df({"A": [100, 400, 400], "B": [100, 400, 400], "C": [100, 100, 100]})
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_LOCAL, price, self.w, 10000, 0.5, groups=self.groups)
        self.assertGreaterEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], post["B"])    # slot reset -> equal split


class TestNoCompositeRegression(unittest.TestCase):
    """Locks: the composite machinery must NOT perturb the legacy path."""

    def setUp(self):
        self.w = pd.Series([0.5, 0.5], index=["A", "B"])
        self.price = _price_df({"A": [100, 400, 400], "B": [100, 100, 100]})

    def test_parse_arity_is_four_and_composite_none(self):
        out = parse_portfolio({"tickers": "QQQM, SPY", "weights": "0.6, 0.4"})
        self.assertEqual(len(out), 4)
        self.assertIsNone(out[3])

    def test_groups_none_equals_no_arg_bit_identical(self):
        for strat in (STRAT_BH, STRAT_RD_FULL, STRAT_RD_LOCAL, STRAT_RD_MIXED, STRAT_ASYM):
            a = run_detailed_backtest(strat, self.price, self.w, 10000, 0.5)
            b = run_detailed_backtest(strat, self.price, self.w, 10000, 0.5, groups=None)
            pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
            self.assertEqual(a[1], b[1])
            self.assertEqual(a[2], b[2])    # exact float compare on pnl_rec['NAV']

    def test_empty_groups_dict_equals_none(self):
        a = run_detailed_backtest(STRAT_RD_FULL, self.price, self.w, 10000, 0.5, groups=None)
        b = run_detailed_backtest(STRAT_RD_FULL, self.price, self.w, 10000, 0.5, groups={})
        pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
        self.assertEqual(a[2], b[2])

    def test_local_scale_branch_bit_identical(self):
        # 3-asset RD_LOCAL exercises the untriggered-slot SCALE branch — the path
        # the singleton short-circuit protects from IEEE-754 drift.
        w = pd.Series([0.34, 0.33, 0.33], index=["A", "B", "C"])
        price = _price_df({"A": [100, 400, 420, 410], "B": [100, 110, 90, 130],
                           "C": [100, 95, 105, 100]})
        for strat in (STRAT_RD_LOCAL, STRAT_RD_MIXED):
            a = run_detailed_backtest(strat, price, w, 10000, 0.3)
            b = run_detailed_backtest(strat, price, w, 10000, 0.3, groups=None)
            pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
            self.assertEqual(a[2]["NAV"], b[2]["NAV"])

    def test_rd_full_golden_post_weights(self):
        hist, cnt, _ = run_detailed_backtest(STRAT_RD_FULL, self.price, self.w, 10000, 0.5)
        self.assertEqual(cnt, 1)
        post = hist[hist["Type"] == "Post-Rebal"].iloc[0]
        self.assertEqual(post["A"], "50.00%")
        self.assertEqual(post["B"], "50.00%")

    def test_positional_five_arg_still_works(self):
        hist, cnt, pnl = run_detailed_backtest(STRAT_BH, self.price, self.w, 10000, 0.5)
        self.assertEqual(cnt, 0)


if __name__ == "__main__":
    unittest.main()
