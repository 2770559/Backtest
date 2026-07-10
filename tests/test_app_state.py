"""Allocation-editor state-machine regressions (backtest_app).

Covers the v2.3.4 fixes:
- In-flight editor edits (delivered in the same browser event as a
  structure-changing action) must survive the matrix rebuild.
- _merge_editor_state applies edited/deleted/added rows correctly.
- Loading a config discards stale editor state instead of flushing it
  over the freshly loaded values.

AppTest runs execute the real app script; network lookups inside it are
best-effort with offline fallbacks, so these tests pass without internet.
"""
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
APP = str(APP_DIR / "backtest_app.py")

from streamlit.testing.v1 import AppTest  # noqa: E402

import backtest_app as app  # noqa: E402  (bare-mode import: warnings are harmless)


class MergeEditorStateTest(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame({
            "Asset": ["QQQM", "159941.SZ - 纳指ETF广发", "513500.SS"],
            "Port C": [None, 35.0, None],
            "Port D": [10.0, 20.0, None],
        })

    def test_edit_delete_add(self):
        state = {
            "edited_rows": {"1": {"Port C": 20.0}, "2": {"Port D": 15.0}},
            "deleted_rows": [0],
            "added_rows": [{"Asset": "GLDM", "Port D": 5.0}],
        }
        merged = app._merge_editor_state(self.base, state)
        self.assertEqual(list(merged["Asset"]),
                         ["159941.SZ - 纳指ETF广发", "513500.SS", "GLDM"])
        self.assertEqual(merged.loc[0, "Port C"], 20.0)
        self.assertEqual(merged.loc[1, "Port D"], 15.0)
        self.assertEqual(merged.loc[2, "Port D"], 5.0)

    def test_empty_state_is_identity(self):
        self.assertTrue(app._merge_editor_state(self.base, {}).equals(self.base))

    def test_out_of_range_indices_ignored(self):
        state = {"edited_rows": {"99": {"Port C": 1.0}}, "deleted_rows": [99]}
        self.assertTrue(app._merge_editor_state(self.base, state).equals(self.base))

    def test_sync_strips_labels(self):
        state = {"edited_rows": {"1": {"Port C": 20.0}}}
        merged = app._merge_editor_state(self.base, state)
        ports = [{"id": "c", "name": "Port C", "tickers": "159941.SZ", "weights": "0.35"}]
        app.sync_alloc(merged, ports)
        self.assertEqual(ports[0]["tickers"], "159941.SZ")
        self.assertEqual(ports[0]["weights"], "0.2")


class InFlightEditSurvivalTest(unittest.TestCase):
    """The reported bug: edits delivered in the same event as a
    structure-changing action were silently reverted."""

    def _port(self, at, name):
        return next(p for p in at.session_state["portfolios_list"] if p["name"] == name)

    def test_edit_survives_add_portfolio(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertFalse(at.exception)

        alloc_key = at.session_state["_alloc_key"]
        base = at.session_state["_alloc_base"]
        row = next(i for i, a in enumerate(base["Asset"]) if str(a).startswith("159941.SZ"))
        at.session_state[alloc_key] = {
            "edited_rows": {str(row): {"Port C": 20.0}},
            "added_rows": [], "deleted_rows": [],
        }

        add = [b for b in at.button if "Add" in str(b.label)][0]
        add.click()
        at.run()
        self.assertFalse(at.exception)

        port_c = self._port(at, "Port C")
        weights = dict(zip([t.strip() for t in port_c["tickers"].split(",")],
                           [w.strip() for w in port_c["weights"].split(",")]))
        self.assertEqual(weights["159941.SZ"], "0.2", "in-flight edit was lost")
        # The new portfolio copies the flushed strings.
        self.assertEqual(self._port(at, "Port D")["weights"], port_c["weights"])


def _port(at, name):
    return next(p for p in at.session_state["portfolios_list"] if p["name"] == name)


def _name_input(at, pname):
    return at.text_input(key=f"n_{_port(at, pname)['id']}")


def _simulate_committed_edit(at):
    """Real post-edit state: strings updated (sync ran), editor widget state
    holds the diff, _alloc_base stale (it only rebuilds on key change)."""
    alloc_key = at.session_state["_alloc_key"]
    base = at.session_state["_alloc_base"]
    row = next(i for i, a in enumerate(base["Asset"]) if str(a).strip() == "QQQM")
    at.session_state[alloc_key] = {
        "edited_rows": {str(row): {"AV-US": 20.0}},
        "added_rows": [], "deleted_rows": [],
    }
    p = _port(at, "AV-US")
    w = [x.strip() for x in p["weights"].split(",")]
    w[0] = "0.2"
    p["weights"] = ", ".join(w)


class DupNameRoundTripTest(unittest.TestCase):
    """Renaming into a duplicate and back must not revert committed edits:
    the dup run renders no editor (widget state destroyed), and restoring the
    names reproduces the same key — recovery must rebuild from the strings."""

    def test_dup_roundtrip_preserves_edits(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        _simulate_committed_edit(at)

        _name_input(at, "Port B").set_value("Port C")   # duplicate
        at.run()
        self.assertTrue(any("Duplicate" in str(e.value) for e in at.error))

        dups = [p for p in at.session_state["portfolios_list"] if p["name"] == "Port C"]
        renamed = next(p for p in dups if p["tickers"].startswith("QQQM"))
        at.text_input(key=f"n_{renamed['id']}").set_value("Port B")
        at.run()
        self.assertFalse(at.exception)

        weights = _port(at, "AV-US")["weights"]
        self.assertEqual(weights.split(",")[0].strip(), "0.2",
                         f"dup round-trip reverted committed edit: {weights}")


class AssetNameGuardTest(unittest.TestCase):
    def test_rename_to_asset_is_blocked(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        before = {p["name"]: p["tickers"] for p in at.session_state["portfolios_list"]}
        _name_input(at, "Port B").set_value("Asset")
        at.run()
        self.assertFalse(at.exception, "renaming to 'Asset' must not crash")
        self.assertTrue(any("Asset" in str(e.value) for e in at.error))
        for p in at.session_state["portfolios_list"]:
            orig = "Port B" if p["name"] == "Asset" else p["name"]
            self.assertEqual(p["tickers"], before[orig],
                             f"tickers corrupted for {p['name']}")


class SearchboxPickLifecycleTest(unittest.TestCase):
    def test_pick_consumed_and_deleted_row_stays_deleted(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        # Simulate a searchbox pick (custom component: plain session dict;
        # options_js/key_react are accessed unconditionally by st_searchbox).
        at.session_state["asset_search"] = {
            "result": "TLT", "search": "",
            "options_js": [], "key_react": "asset_search_react_test",
        }
        at.run()
        self.assertIn("TLT", at.session_state["_alloc_pending"])
        self.assertIsNone(at.session_state["asset_search"]["result"],
                          "pick must be consumed, not returned forever")
        base = at.session_state["_alloc_base"]
        row = next(i for i, a in enumerate(base["Asset"]) if str(a).strip() == "TLT")

        # User deletes the pending row, then triggers a rebuild via Add:
        # the row must NOT resurrect.
        alloc_key = at.session_state["_alloc_key"]
        at.session_state[alloc_key] = {
            "edited_rows": {}, "added_rows": [], "deleted_rows": [row],
        }
        add = [b for b in at.button if "Add" in str(b.label)][0]
        add.click()
        at.run()
        self.assertFalse(at.exception)
        self.assertNotIn("TLT", at.session_state["_alloc_pending"],
                         "deleted pending row resurrected")
        self.assertFalse(any(str(a).strip() == "TLT"
                             for a in at.session_state["_alloc_base"]["Asset"]))


class SaveDefaultValidationTest(unittest.TestCase):
    def test_invalid_config_not_saved(self):
        default_path = APP_DIR / "Backtest" / "_default.json"
        self.assertFalse(default_path.exists(), "pre-existing default would taint this test")
        at = AppTest.from_file(APP, default_timeout=120).run()
        _name_input(at, "Port B").set_value("Port C")   # duplicate -> invalid
        at.run()
        save = [b for b in at.button if "Save Default" in str(b.label)][0]
        save.click()
        at.run()
        try:
            self.assertTrue(any("NOT saved" in str(e.value) for e in at.error))
            self.assertFalse(default_path.exists(), "invalid config was persisted")
        finally:
            default_path.unlink(missing_ok=True)


class ApplyConfigDiscardsStaleEditsTest(unittest.TestCase):
    def test_load_saved_config_ignores_inflight_edits(self):
        # Named to sort FIRST so the sidebar selectbox defaults to it (AppTest
        # cannot drive that selectbox: its options are Path objects). Contains
        # a "Port C" so a stale-edit bleed on name collision would be visible.
        cfg_path = APP_DIR / "Backtest" / "0000_apptest_tmp.json"
        cfg_path.write_text(json.dumps({
            "benchmark": "SPY", "start_date": "2021-01-01", "initial_funds": 10000,
            "portfolios": [{"name": "Port C", "tickers": "159941.SZ, 511130.SS",
                            "weights": "0.35, 0.65", "strat": "RelDiff Mixed", "thr": 38}],
        }, ensure_ascii=False), encoding="utf-8")
        try:
            at = AppTest.from_file(APP, default_timeout=120).run()
            # Stale in-flight edit on the built-in Port C, then load the file.
            alloc_key = at.session_state["_alloc_key"]
            base = at.session_state["_alloc_base"]
            row = next(i for i, a in enumerate(base["Asset"]) if str(a).startswith("159941.SZ"))
            at.session_state[alloc_key] = {
                "edited_rows": {str(row): {"Port C": 20.0}},
                "added_rows": [], "deleted_rows": [],
            }
            load = [b for b in at.button if "Load Saved Config" in str(b.label)][0]
            load.click()
            at.run()
            self.assertFalse(at.exception)

            ports = at.session_state["portfolios_list"]
            self.assertEqual([p["name"] for p in ports], ["Port C"])
            self.assertEqual(ports[0]["weights"], "0.35, 0.65",
                             "stale in-flight edit bled into the loaded config")
        finally:
            cfg_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
