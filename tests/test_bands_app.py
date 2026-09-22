"""v2.5.0 band UI / config-model regressions (backtest_app, AppTest).

- Old configs (no thr_up / slot_bands) load with thr_up == thr and no overrides.
- Unknown slot labels in slot_bands are kept, flagged once, and never crash.
- Up % follows Down % while symmetric; an independently set Up % stays put.
- Per-slot band editors sync into port['slot_bands'] and survive an Add click
  delivered in the same event (flush_slot_band_edits).
- Every stored portfolio carries the new fields (what Export / Save Default write).

No network: the backtest itself is never triggered here.
"""
import json
import sys
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
APP = str(APP_DIR / "backtest_app.py")

from streamlit.testing.v1 import AppTest  # noqa: E402

import backtest_app as app  # noqa: E402  (bare-mode import: warnings are harmless)

TMP_CFG = APP_DIR / "Backtest" / "0000_apptest_bands_tmp.json"   # sorts first -> default selection


def _port(at, name):
    return next(p for p in at.session_state["portfolios_list"] if p["name"] == name)


def _load_cfg(cfg):
    TMP_CFG.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    at = AppTest.from_file(APP, default_timeout=120).run()
    load = [b for b in at.button if "Load Saved Config" in str(b.label)][0]
    load.click()
    at.run()
    return at


class NormBandPctTest(unittest.TestCase):
    def test_coercion_and_clamp(self):
        self.assertEqual(app._norm_band_pct(40, 38), 40)
        self.assertEqual(app._norm_band_pct("60", 38), 60)
        self.assertEqual(app._norm_band_pct(40.4, 38), 40)
        self.assertEqual(app._norm_band_pct(None, 38), 38)
        self.assertEqual(app._norm_band_pct("abc", 38), 38)
        self.assertEqual(app._norm_band_pct(999, 38), 200)
        self.assertEqual(app._norm_band_pct(0, 38), 1)


class OldConfigLoadsSymmetricTest(unittest.TestCase):
    def tearDown(self):
        TMP_CFG.unlink(missing_ok=True)

    def test_v240_json_gets_symmetric_band_and_no_overrides(self):
        at = _load_cfg({
            "benchmark": "SPY", "start_date": "2007-01-01", "initial_funds": 10000,
            "portfolios": [{"name": "Old", "tickers": "QQQ, BRK.B, GLD, XLE, RYMTX",
                            "weights": "0.3684, 0.1579, 0.1579, 0.1053, 0.2105",
                            "strat": "RelDiff Mixed", "thr": 40}],
        })
        self.assertFalse(at.exception)
        p = _port(at, "Old")
        self.assertEqual((p["thr"], p["thr_up"], p["slot_bands"]), (40, 40, {}))
        self.assertFalse(any("per-slot band" in str(w.value) for w in at.warning))
        # Up widget shows the inherited value.
        self.assertEqual(at.number_input(key=f"tu_{p['id']}").value, 40)


class UnknownSlotLabelTest(unittest.TestCase):
    def tearDown(self):
        TMP_CFG.unlink(missing_ok=True)

    def test_unknown_label_is_flagged_kept_and_harmless(self):
        at = _load_cfg({
            "benchmark": "SPY", "start_date": "2020-12-02", "initial_funds": 10000,
            "portfolios": [{"name": "New", "strat": "RelDiff Mixed", "thr": 60, "thr_up": 100,
                            "tickers": "QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, (ETH-USD, MSTR)",
                            "weights": "0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05",
                            "slot_bands": {"ETH-USD+MSTR": {"down": "40", "up": 40},
                                           "NOPE": {"down": 10}, "junk": "x"}}],
        })
        self.assertFalse(at.exception)
        p = _port(at, "New")
        self.assertEqual((p["thr"], p["thr_up"]), (60, 100))
        self.assertEqual(p["slot_bands"], {"ETH-USD+MSTR": {"down": 40, "up": 40},
                                           "NOPE": {"down": 10, "up": None}})
        warns = [str(w.value) for w in at.warning]
        self.assertTrue(any("NOPE" in w for w in warns), warns)
        # The orphan shows up in the editor, flagged, so it can be blanked.
        key = at.session_state["_sb_keys"][p["id"]]
        base = app.slot_band_rows(p)
        self.assertIn("NOPE" + app.ORPHAN_MARK, list(base["Slot"]))
        # Blank it -> dropped from the config on sync.
        row = list(base["Slot"]).index("NOPE" + app.ORPHAN_MARK)
        at.session_state[key] = {"edited_rows": {str(row): {"Down %": None, "Up %": None}},
                                 "added_rows": [], "deleted_rows": []}
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(_port(at, "New")["slot_bands"], {"ETH-USD+MSTR": {"down": 40, "up": 40}})


class UpFollowsDownTest(unittest.TestCase):
    def test_up_tracks_down_until_set_independently(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        p = _port(at, "AV-US")
        self.assertEqual((p["thr"], p["thr_up"]), (40, 40))
        tr, tu = f"tr_{p['id']}", f"tu_{p['id']}"

        at.number_input(key=tr).set_value(60)
        at.run()
        self.assertFalse(at.exception)
        p = _port(at, "AV-US")
        self.assertEqual((p["thr"], p["thr_up"]), (60, 60), "Up must follow Down while symmetric")
        self.assertEqual(at.number_input(key=tu).value, 60)
        self.assertFalse(any("Session State API" in str(w.value) for w in at.warning),
                         "programmatic Up update must not raise the duplication warning")

        at.number_input(key=tu).set_value(100)
        at.run()
        self.assertEqual(_port(at, "AV-US")["thr_up"], 100)

        at.number_input(key=tr).set_value(50)
        at.run()
        p = _port(at, "AV-US")
        self.assertEqual((p["thr"], p["thr_up"]), (50, 100), "independent Up must not be overwritten")

        # Setting Up back to Down re-links them.
        at.number_input(key=tu).set_value(50)
        at.run()
        at.number_input(key=tr).set_value(70)
        at.run()
        p = _port(at, "AV-US")
        self.assertEqual((p["thr"], p["thr_up"]), (70, 70))


class SlotBandEditorTest(unittest.TestCase):
    def _edit(self, at, pname, label, down, up):
        p = _port(at, pname)
        key = at.session_state["_sb_keys"][p["id"]]
        base = app.slot_band_rows(p)
        row = list(base["Slot"]).index(label)
        at.session_state[key] = {"edited_rows": {str(row): {"Down %": down, "Up %": up}},
                                 "added_rows": [], "deleted_rows": []}
        return key

    def test_edit_syncs_and_blank_clears(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self._edit(at, "AV-US", "ETH-USD+MSTR", 40, 40)
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(_port(at, "AV-US")["slot_bands"], {"ETH-USD+MSTR": {"down": 40, "up": 40}})
        # Other portfolios untouched.
        self.assertEqual(_port(at, "Port B")["slot_bands"], {})

        self._edit(at, "AV-US", "ETH-USD+MSTR", 30, None)     # one-sided override
        at.run()
        self.assertEqual(_port(at, "AV-US")["slot_bands"], {"ETH-USD+MSTR": {"down": 30, "up": None}})

        self._edit(at, "AV-US", "ETH-USD+MSTR", None, None)
        at.run()
        self.assertEqual(_port(at, "AV-US")["slot_bands"], {})

    def test_edit_survives_add_portfolio(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self._edit(at, "AV-US", "GLDM", 25, 80)
        add = [b for b in at.button if "Add" in str(b.label)][0]
        add.click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(_port(at, "AV-US")["slot_bands"], {"GLDM": {"down": 25, "up": 80}},
                         "in-flight per-slot edit was lost on Add")
        new = _port(at, "Port D")
        self.assertEqual((new["thr"], new["thr_up"], new["slot_bands"]), (40, 40, {}))

    def test_slot_change_reanchors_editor(self):
        # Removing a slot from the matrix drops its row; the stored override for
        # it is kept (flagged) rather than silently lost.
        at = AppTest.from_file(APP, default_timeout=120).run()
        self._edit(at, "AV-US", "XLE", 20, 20)
        at.run()
        self.assertEqual(_port(at, "AV-US")["slot_bands"], {"XLE": {"down": 20, "up": 20}})
        alloc_key = at.session_state["_alloc_key"]
        base = at.session_state["_alloc_base"]
        row = next(i for i, a in enumerate(base["Asset"]) if str(a).strip() == "XLE")
        at.session_state[alloc_key] = {"edited_rows": {str(row): {"AV-US": None}},
                                       "added_rows": [], "deleted_rows": []}
        at.run()
        self.assertFalse(at.exception)
        p = _port(at, "AV-US")
        self.assertNotIn("XLE", p["tickers"].split(", "))
        self.assertEqual(p["slot_bands"], {"XLE": {"down": 20, "up": 20}})
        self.assertIn("XLE" + app.ORPHAN_MARK, list(app.slot_band_rows(p)["Slot"]))


class StoredFieldsTest(unittest.TestCase):
    def test_every_port_carries_band_fields(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        for p in at.session_state["portfolios_list"]:
            self.assertIn("thr_up", p)
            self.assertIn("slot_bands", p)
            self.assertEqual(p["thr_up"], p["thr"])      # built-ins are symmetric
        # Export / Save Default serialize this list verbatim.
        payload = json.loads(json.dumps(at.session_state["portfolios_list"]))
        self.assertTrue(all("thr_up" in p and "slot_bands" in p for p in payload))


if __name__ == "__main__":
    unittest.main()
