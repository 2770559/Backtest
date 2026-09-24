"""v2.5.0 band UI / config-model regressions (backtest_app, AppTest).

- Old configs (no thr_up / slot_bands) load with thr_up == thr and no overrides.
- Unknown slot labels in slot_bands are kept, flagged once, and never crash.
- Up % starts equal to Down % and is independent from then on; the server never
  writes into the Up widget. The earlier "push Down into Up" follow logic
  landed while the user was already typing in Up and replaced the digits
  (reported 2026-09-22); a nullable "empty = mirror" variant was cleared by
  the frontend on every rerun while empty.
- Per-slot band editors sync into port['slot_bands'], keep their anchored frame
  while edits are in flight (a rebuilt frame changes st.data_editor's identity
  and drops a second quick edit), and survive an Add click delivered in the
  same event (flush_slot_band_edits).
- Every stored portfolio carries the new fields (what Export / Save Default write).
- v2.5.2 band mode: `band_mode` loads as "ratio" or falls back to "rel" (missing
  or unknown values), the row selectbox shows and edits it, built-ins are "rel".

No network: the backtest itself is never triggered here.
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
        self.assertIsNone(app._norm_band_pct(None, None))
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
        # Up widget shows the inherited value; the engine gets the symmetric legacy path.
        self.assertEqual(at.number_input(key=f"tu_{p['id']}").value, 40)
        self.assertEqual(app.build_band_thresholds(p["thr"], p["thr_up"], p["slot_bands"], {}), (0.4, None))

    def test_null_thr_up_loads_as_symmetric(self):
        at = _load_cfg({
            "benchmark": "SPY", "start_date": "2020-12-02", "initial_funds": 10000,
            "portfolios": [{"name": "Nul", "tickers": "QQQM, SPY", "weights": "0.5, 0.5",
                            "strat": "RelDiff Mixed", "thr": 60, "thr_up": None, "slot_bands": {}},
                           {"name": "Asym", "tickers": "QQQM, SPY", "weights": "0.5, 0.5",
                            "strat": "RelDiff Mixed", "thr": 60, "thr_up": 100}],
        })
        self.assertFalse(at.exception)
        self.assertEqual(_port(at, "Nul")["thr_up"], 60)
        self.assertEqual(_port(at, "Asym")["thr_up"], 100)
        self.assertEqual(at.number_input(key=f"tu_{_port(at, 'Asym')['id']}").value, 100)


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


class UpIndependentOfDownTest(unittest.TestCase):
    def test_down_edits_never_touch_the_up_widget(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        p = _port(at, "AV-US")
        self.assertEqual((p["thr"], p["thr_up"]), (40, 40))
        tr, tu = f"tr_{p['id']}", f"tu_{p['id']}"
        self.assertEqual(at.number_input(key=tu).value, 40)
        up_proto = at.number_input(key=tu).proto.SerializeToString()

        # Changing Down must leave the Up widget untouched: no value pushed
        # into it (that push replaced digits being typed into Up during the
        # rerun) and an unchanged proto (a changed proto re-creates the input).
        at.number_input(key=tr).set_value(60)
        at.run()
        self.assertFalse(at.exception)
        p = _port(at, "AV-US")
        self.assertEqual((p["thr"], p["thr_up"]), (60, 40))
        self.assertEqual(at.number_input(key=tu).value, 40)
        self.assertEqual(at.number_input(key=tu).proto.SerializeToString(), up_proto)
        self.assertFalse(any("Session State API" in str(w.value) for w in at.warning))
        self.assertEqual(app.build_band_thresholds(60, 40, {}, {}), (0.6, 0.4))

        at.number_input(key=tu).set_value(100)
        at.run()
        self.assertEqual(_port(at, "AV-US")["thr_up"], 100)

        at.number_input(key=tr).set_value(50)
        at.run()
        p = _port(at, "AV-US")
        self.assertEqual((p["thr"], p["thr_up"]), (50, 100), "independent Up must not be overwritten")
        self.assertEqual(app.build_band_thresholds(50, 100, {}, {}), (0.5, 1.0))

        # Equal values are the symmetric band -> legacy engine path.
        at.number_input(key=tu).set_value(50)
        at.run()
        p = _port(at, "AV-US")
        self.assertEqual((p["thr"], p["thr_up"]), (50, 50))
        self.assertEqual(app.build_band_thresholds(50, 50, {}, {}), (0.5, None))


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

    def test_anchor_frame_is_kept_while_edits_are_in_flight(self):
        """st.data_editor hashes its data into the widget identity. After the
        first edit is synced into the dict, re-deriving the frame would move
        the identity and drop a second edit typed before that rerun landed —
        so the anchored frame must stay the pre-edit snapshot while the
        editor holds diffs, and be re-derived only once the diffs are gone."""
        at = AppTest.from_file(APP, default_timeout=120).run()
        p = _port(at, "AV-US")
        key0 = at.session_state["_sb_keys"][p["id"]]
        anchor0 = at.session_state["_sb_base"][p["id"]][1]
        row = list(anchor0["Slot"]).index("ETH-USD+MSTR")
        self.assertTrue(pd.isna(anchor0.loc[row, "Down %"]))

        # First edit committed and synced.
        at.session_state[key0] = {"edited_rows": {str(row): {"Down %": 40}},
                                  "added_rows": [], "deleted_rows": []}
        at.run()
        self.assertEqual(_port(at, "AV-US")["slot_bands"], {"ETH-USD+MSTR": {"down": 40, "up": None}})
        key1, anchor1 = at.session_state["_sb_base"][p["id"]]
        self.assertEqual(key1, key0)
        self.assertTrue(pd.isna(anchor1.loc[row, "Down %"]), "anchor frame was re-derived mid-edit")
        self.assertTrue(anchor1.equals(anchor0))

        # Second edit arrives on the SAME editor, accumulated with the first.
        at.session_state[key0] = {"edited_rows": {str(row): {"Down %": 40, "Up %": 40}},
                                  "added_rows": [], "deleted_rows": []}
        at.run()
        self.assertEqual(_port(at, "AV-US")["slot_bands"], {"ETH-USD+MSTR": {"down": 40, "up": 40}})
        self.assertTrue(at.session_state["_sb_base"][p["id"]][1].equals(anchor0))

        # Diffs gone (e.g. the searchbox's internal rerun wiped them): the
        # frame is re-derived from the dict, which already holds both edits.
        at.session_state[key0] = {"edited_rows": {}, "added_rows": [], "deleted_rows": []}
        at.run()
        self.assertFalse(at.exception)
        anchor2 = at.session_state["_sb_base"][p["id"]][1]
        self.assertEqual((anchor2.loc[row, "Down %"], anchor2.loc[row, "Up %"]), (40.0, 40.0))
        self.assertEqual(_port(at, "AV-US")["slot_bands"], {"ETH-USD+MSTR": {"down": 40, "up": 40}})

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


class MainAreaStructureTest(unittest.TestCase):
    """The localStorage bridge components (streamlit_js_eval) are transient:
    the reader unmounts once the browser has answered. Streamlit keys BLOCKS
    by their index inside the parent, so a transient element ABOVE the
    portfolio container re-keys — and re-mounts — that whole container when
    it disappears, wiping any digits being typed at that moment (the
    reported first-edit "number jumps away"). They must therefore be the
    last children of the main area, after every block that holds widgets."""

    def test_bridge_components_render_after_every_block(self):
        if not app.HAS_JS_EVAL:
            self.skipTest("streamlit-js-eval not installed")
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertFalse(at.exception)
        kinds = [c.type for c in at.main.children.values()]
        comp_idx = [i for i, k in enumerate(kinds) if k == "component_instance"]
        block_idx = [i for i, k in enumerate(kinds) if k in ("vertical_block", "flex_container")]
        self.assertTrue(comp_idx, f"reader component not rendered on a fresh session: {kinds}")
        self.assertTrue(block_idx, kinds)
        self.assertGreater(min(comp_idx), max(block_idx),
                           f"bridge component rendered above a widget block: {kinds}")
        self.assertEqual(comp_idx[-1], len(kinds) - 1, kinds)


class StoredFieldsTest(unittest.TestCase):
    def test_every_port_carries_band_fields(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        for p in at.session_state["portfolios_list"]:
            self.assertIn("thr_up", p)
            self.assertIn("slot_bands", p)
            self.assertEqual(p["thr_up"], p["thr"])      # built-ins are symmetric
            self.assertEqual(p["band_mode"], "rel")      # built-ins run the original rule
            self.assertEqual(at.selectbox(key=f"bm_{p['id']}").value, "rel")
        # Export / Save Default serialize this list verbatim.
        payload = json.loads(json.dumps(at.session_state["portfolios_list"]))
        self.assertTrue(all("thr_up" in p and "slot_bands" in p and p["band_mode"] == "rel"
                            for p in payload))


class NormBandModeTest(unittest.TestCase):
    def test_coercion(self):
        self.assertEqual(app._norm_band_mode("ratio"), "ratio")
        self.assertEqual(app._norm_band_mode(" RATIO "), "ratio")
        for v in (None, "", "rel", "REL", "garbage", 0, 1, "abs", {"x": 1}):
            self.assertEqual(app._norm_band_mode(v), "rel", v)


class BandModeConfigTest(unittest.TestCase):
    """band_mode round-trips through Load / the row selectbox / the stored dict."""

    CFG = {
        "benchmark": "SPY", "start_date": "2020-12-02", "initial_funds": 10000,
        "portfolios": [
            {"name": "Ratio", "tickers": "QQQM, BRK.B, (ETH-USD, MSTR)", "weights": "0.6, 0.35, 0.05",
             "strat": "RelDiff Mixed", "thr": 100, "thr_up": 100,
             "slot_bands": {"ETH-USD+MSTR": {"down": 40, "up": 40}}, "band_mode": "ratio"},
            {"name": "Bad", "tickers": "QQQM, SPY", "weights": "0.5, 0.5",
             "strat": "RelDiff Full", "thr": 40, "band_mode": "abs"},
            {"name": "Old", "tickers": "QQQM, SPY", "weights": "0.5, 0.5",
             "strat": "RelDiff Full", "thr": 40},
        ],
    }

    def tearDown(self):
        TMP_CFG.unlink(missing_ok=True)

    def test_load_shows_and_keeps_mode(self):
        at = _load_cfg(self.CFG)
        self.assertFalse(at.exception)
        ratio, bad, old = (_port(at, n) for n in ("Ratio", "Bad", "Old"))
        self.assertEqual(ratio["band_mode"], "ratio")
        self.assertEqual(bad["band_mode"], "rel")           # unknown value -> original rule
        self.assertEqual(old["band_mode"], "rel")           # pre-2.5.2 config -> original rule
        self.assertEqual(at.selectbox(key=f"bm_{ratio['id']}").value, "ratio")
        self.assertEqual(at.selectbox(key=f"bm_{old['id']}").value, "rel")
        # The row selectbox shows the human labels, in engine order.
        self.assertEqual(at.selectbox(key=f"bm_{ratio['id']}").options, ["Δ vs target", "Leg vs rest"])
        # The per-slot override is untouched by the mode.
        self.assertEqual(ratio["slot_bands"], {"ETH-USD+MSTR": {"down": 40, "up": 40}})
        # Export serializes the field.
        payload = json.loads(json.dumps(at.session_state["portfolios_list"]))
        self.assertEqual([p["band_mode"] for p in payload], ["ratio", "rel", "rel"])

    def test_selectbox_edits_the_stored_dict(self):
        at = _load_cfg(self.CFG)
        old = _port(at, "Old")
        at.selectbox(key=f"bm_{old['id']}").select("ratio")
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(_port(at, "Old")["band_mode"], "ratio")
        ratio = _port(at, "Ratio")
        at.selectbox(key=f"bm_{ratio['id']}").select("rel")
        at.run()
        self.assertEqual(_port(at, "Ratio")["band_mode"], "rel")
        # Down / Up rows are untouched by the mode switch.
        self.assertEqual((_port(at, "Ratio")["thr"], _port(at, "Ratio")["thr_up"]), (100, 100))


if __name__ == "__main__":
    unittest.main()
