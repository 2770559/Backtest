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
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class EditorStateWipeTest(unittest.TestCase):
    """The engine drops the editor's accumulated diffs on any run that ends
    before instantiating it — st_searchbox fires one such internal rerun per
    search keystroke. The editor then re-anchors on the stale base with empty
    diffs, and sync must NOT write the old values back over the strings."""

    def test_wiped_editor_state_does_not_revert_strings(self):
        at = AppTest.from_file(APP, default_timeout=120).run()

        # Committed edit: delta in widget state, editor renders, sync ran.
        alloc_key = at.session_state["_alloc_key"]
        base = at.session_state["_alloc_base"]
        row = next(i for i, a in enumerate(base["Asset"]) if str(a).strip() == "QQQM")
        at.session_state[alloc_key] = {
            "edited_rows": {str(row): {"AV-US": 20.0}},
            "added_rows": [], "deleted_rows": [],
        }
        at.run()
        self.assertTrue(_port(at, "AV-US")["weights"].startswith("0.2"))

        # Engine wipe: diffs cleared, key unchanged, base snapshot stale.
        at.session_state[alloc_key] = {
            "edited_rows": {}, "added_rows": [], "deleted_rows": [],
        }
        at.run()
        self.assertFalse(at.exception)

        weights = _port(at, "AV-US")["weights"]
        self.assertTrue(weights.startswith("0.2"),
                        f"wiped editor state reverted the strings: {weights}")
        # And the rebuilt base must show the true (string) values.
        base2 = at.session_state["_alloc_base"]
        row2 = next(i for i, a in enumerate(base2["Asset"]) if str(a).strip() == "QQQM")
        self.assertEqual(float(base2.loc[row2, "AV-US"]), 20.0)


class SaveDefaultDialogTest(unittest.TestCase):
    """Save Default must not write anything until confirmed in the dialog.

    AppTest limitation: st.dialog interactions are fragment reruns, which
    AppTest does not simulate — clicks on dialog-internal buttons are never
    observed by the dialog function. The confirm/reset paths are therefore
    covered by real-browser E2E; here we lock in the click-side contract."""

    def setUp(self):
        self.default_path = APP_DIR / "Backtest" / "_default.json"
        self.assertFalse(self.default_path.exists(),
                         "pre-existing default would taint this test")

    def tearDown(self):
        self.default_path.unlink(missing_ok=True)

    def test_click_alone_saves_nothing_and_opens_dialog(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        save = [b for b in at.button if "Save Default" in str(b.label)][0]
        save.click()
        at.run()
        self.assertFalse(at.exception)
        self.assertFalse(self.default_path.exists(),
                         "file written without confirmation")
        labels = [str(b.label) for b in at.button]
        self.assertTrue(any("Confirm & Save" in l for l in labels),
                        "confirmation dialog did not open")
        self.assertTrue(any(l.strip() == "Cancel" for l in labels))


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


class SharedHostTest(unittest.TestCase):
    """On a hosted server one Backtest/_default.json would be every visitor's default: it is ignored there; and user
    text never reaches the page as raw HTML."""

    def test_file_default_only_on_your_own_machine(self):
        import os
        path = APP_DIR / "Backtest" / "_default.json"
        self.assertFalse(path.exists(), "pre-existing default would taint this test")
        path.write_text(json.dumps({"benchmark": "SPY", "start_date": "2021-01-01", "initial_funds": 10000,
                                    "portfolios": [{"name": "Shared X", "tickers": "SPY", "weights": "1.0",
                                                    "strat": "Buy & Hold", "thr": 40}]}), encoding="utf-8")
        try:
            at = AppTest.from_file(APP, default_timeout=120).run()
            self.assertIn("Shared X", [p["name"] for p in at.session_state.portfolios_list])     # local checkout
            os.environ["BACKTEST_SHARED_HOST"] = "1"
            at = AppTest.from_file(APP, default_timeout=120).run()
            self.assertFalse(at.exception)
            self.assertNotIn("Shared X", [p["name"] for p in at.session_state.portfolios_list])  # hosted: ignored
        finally:
            os.environ.pop("BACKTEST_SHARED_HOST", None)
            path.unlink(missing_ok=True)

    def test_names_are_escaped_in_html(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        _name_input(at, "Port B").set_value('<img src=x onerror=1>Y')
        at.run()
        sums = [str(m.value) for m in at.markdown if 'class="alloc-sums"' in str(m.value)]
        self.assertTrue(sums)
        self.assertIn("&lt;img src=x onerror=1&gt;Y", sums[0])
        self.assertNotIn("<img", sums[0])


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


def _cfg(**top):
    cfg = {"benchmark": "SPY", "start_date": "2021-01-01", "initial_funds": 10000,
           "portfolios": [{"id": "a", "name": "P", "tickers": "SPY, TLT", "weights": "0.6, 0.4",
                           "strat": "RelDiff Full", "thr": 40}]}
    cfg.update(top)
    return cfg


class ParseConfigTest(unittest.TestCase):
    """_parse_config: values the widgets would reject are brought into range and reported (a start
    date outside the picker's range or funds below its minimum raised on every rerun of the session);
    a file that is not a config raises before anything is applied. Pure: no session state touched."""

    def test_valid_legacy_config_loads_without_warnings(self):
        cfg, warns = app._parse_config(_cfg())
        self.assertEqual(warns, [])
        p = cfg["portfolios"][0]
        self.assertEqual((p["id"], p["thr"], p["thr_up"], p["slot_bands"], p["band_mode"]),
                         ("a", 40, 40, {}, "rel"))
        self.assertEqual((cfg["bi"], cfg["sd"], cfg["init_funds"]), ("SPY", date(2021, 1, 1), 10000))

    def test_legacy_chinese_strategy_still_maps_silently(self):
        port = dict(_cfg()["portfolios"][0], strat="相对差混合再平衡")
        cfg, warns = app._parse_config(_cfg(portfolios=[port]))
        self.assertEqual((cfg["portfolios"][0]["strat"], warns), (app.STRAT_RD_MIXED, []))

    def test_funds_brought_into_the_inputs_range(self):
        for raw, kept in ((50, 100), (10 ** 16, 10 ** 12), ("10k", 10000), (float("nan"), 10000)):
            cfg, warns = app._parse_config(_cfg(initial_funds=raw))
            self.assertEqual(cfg["init_funds"], kept, raw)
            self.assertTrue(any("Initial investment" in w for w in warns), (raw, warns))
            self.assertTrue(all("\\$" in w for w in warns if "Initial investment" in w))   # no LaTeX pair
        cfg, warns = app._parse_config(_cfg(initial_funds="2500.9"))
        self.assertEqual((cfg["init_funds"], warns), (2500, []))

    def test_start_date_brought_into_the_pickers_range(self):
        today = datetime.today().date()
        for raw, kept in (("1965-01-01", date(1970, 1, 1)), ((today + timedelta(days=30)).isoformat(), today),
                          ("not a date", date(2020, 1, 1)), (None, date(2020, 1, 1))):
            cfg, warns = app._parse_config(_cfg(start_date=raw))
            self.assertEqual(cfg["sd"], kept, raw)
            self.assertTrue(any("Start date" in w for w in warns), (raw, warns))

    def test_bands_clamped_to_the_editors_range(self):
        port = {"id": "a", "name": "P", "tickers": "SPY, (ETH-USD, MSTR)", "weights": "0.9, 0.1",
                "strat": "RelDiff Mixed", "thr": 0, "thr_up": 250,
                "slot_bands": {"SPY": {"down": 0, "up": 500}, "ETH-USD+MSTR": {"down": -5, "up": None}}}
        cfg, warns = app._parse_config(_cfg(portfolios=[port]))
        p = cfg["portfolios"][0]
        self.assertEqual((p["thr"], p["thr_up"]), (1, 200))
        self.assertEqual(p["slot_bands"], {"SPY": {"down": 1, "up": 200},
                                           "ETH-USD+MSTR": {"down": 1, "up": None}})
        self.assertEqual(len(warns), 5, warns)          # Down, Up, SPY down / up, crypto down

    def test_negative_weights_warned_at_load_and_refused_by_analyze(self):
        port = dict(_cfg()["portfolios"][0], weights="-0.2, 1.2")
        cfg, warns = app._parse_config(_cfg(portfolios=[port]))
        self.assertTrue(any("negative weight(s) -0.2" in w for w in warns), warns)
        self.assertEqual(cfg["portfolios"][0]["weights"], "-0.2, 1.2")   # kept: a 0 would drop the asset
        errs = app.validate_inputs(cfg["portfolios"], "SPY")
        self.assertTrue(any("negative weight(s): SPY" in e for e in errs), errs)

    def test_unknown_strategy_and_band_mode_are_reported(self):
        port = dict(_cfg()["portfolios"][0], strat="Momentum", band_mode="Leg vs rest")
        cfg, warns = app._parse_config(_cfg(portfolios=[port]))
        self.assertEqual((cfg["portfolios"][0]["strat"], cfg["portfolios"][0]["band_mode"]),
                         (app.STRAT_ASYM, "rel"))
        self.assertTrue(any("unknown strategy" in w for w in warns), warns)
        self.assertTrue(any("unknown band mode" in w for w in warns), warns)

    def test_duplicate_or_missing_ids_get_fresh_ones(self):
        base = _cfg()["portfolios"][0]
        ports = [dict(base, id="same", name="A"), dict(base, id="same", name="B"),
                 {k: v for k, v in base.items() if k != "id"}]
        cfg, _ = app._parse_config(_cfg(portfolios=ports))
        ids = [p["id"] for p in cfg["portfolios"]]
        self.assertEqual(ids[0], "same")
        self.assertEqual(len(set(ids)), 3)

    def test_not_a_config_raises(self):
        for bad in ([1, 2], {"portfolios": {"a": 1}}, {"portfolios": [_cfg()["portfolios"][0], "oops"]}):
            with self.assertRaises(ValueError, msg=bad):
                app._parse_config(bad)

    def test_input_is_not_mutated(self):
        raw = _cfg()
        snapshot = json.dumps(raw)
        app._parse_config(raw)
        self.assertEqual(json.dumps(raw), snapshot)


class _SavedConfig:
    """A temporary Backtest/ config named to sort FIRST, so the sidebar selectbox
    (which AppTest cannot drive: its options are Path objects) defaults to it."""

    def __init__(self, cfg, stem):
        self.path = APP_DIR / "Backtest" / f"0000_{stem}.json"
        self.cfg = cfg

    def __enter__(self):
        self.path.write_text(json.dumps(self.cfg, ensure_ascii=False), encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _click_load(at):
    [b for b in at.button if "Load Saved Config" in str(b.label)][0].click()
    at.run()


class ConfigLoadAppTest(unittest.TestCase):
    PORT_C = {"name": "Port C", "tickers": "159941.SZ, 511130.SS", "weights": "0.35, 0.65",
              "strat": "RelDiff Mixed", "thr": 38}

    def test_out_of_range_values_no_longer_break_the_session(self):
        cfg = _cfg(start_date="1965-01-01", initial_funds=50)
        with _SavedConfig(cfg, "cfgload_range"):
            at = AppTest.from_file(APP, default_timeout=120).run()
            _click_load(at)
            self.assertFalse(at.exception)
            self.assertEqual((at.session_state["init_funds"], at.session_state["sd"]), (100, date(1970, 1, 1)))
            warns = " ".join(str(w.value) for w in at.warning)
            self.assertIn("Initial investment", warns)
            self.assertIn("Start date", warns)
            at.run()                                     # the flash warnings are gone; still no exception
            self.assertFalse(at.exception)

    def test_broken_config_is_not_half_applied(self):
        """A value failing halfway (int("10k")) used to leave the portfolios replaced, the editor
        clean-up skipped and stale edits flushed into them. A file that cannot load now changes nothing."""
        broken = _cfg(benchmark="QQQ", portfolios=[self.PORT_C, "not a portfolio"])
        with _SavedConfig(broken, "cfgload_broken"):
            at = AppTest.from_file(APP, default_timeout=120).run()
            before = [(p["name"], p["weights"]) for p in at.session_state["portfolios_list"]]
            _click_load(at)
            self.assertFalse(at.exception)
            self.assertTrue(any("Load error" in str(e.value) for e in at.error))
            self.assertEqual([(p["name"], p["weights"]) for p in at.session_state["portfolios_list"]], before)
            self.assertEqual(at.session_state["bi"], "SPY")

    def test_bad_value_loads_completely_without_stale_edits(self):
        """The review's case: initial_funds "10k" with an in-flight matrix edit on the built-in Port C."""
        with _SavedConfig(_cfg(initial_funds="10k", portfolios=[self.PORT_C]), "cfgload_10k"):
            at = AppTest.from_file(APP, default_timeout=120).run()
            alloc_key, base = at.session_state["_alloc_key"], at.session_state["_alloc_base"]
            row = next(i for i, a in enumerate(base["Asset"]) if str(a).startswith("159941.SZ"))
            at.session_state[alloc_key] = {"edited_rows": {str(row): {"Port C": 20.0}},
                                           "added_rows": [], "deleted_rows": []}
            _click_load(at)
            self.assertFalse(at.exception)
            self.assertEqual([e.value for e in at.error], [])
            self.assertEqual([(p["name"], p["tickers"], p["weights"]) for p in at.session_state["portfolios_list"]],
                             [("Port C", "159941.SZ, 511130.SS", "0.35, 0.65")])
            self.assertEqual(at.session_state["init_funds"], 10000)
            self.assertTrue(any("Initial investment" in str(w.value) for w in at.warning))

    def test_copy_pasted_portfolio_ids_no_longer_crash(self):
        # Two portfolios with the same "id" collided as widget keys (StreamlitDuplicateElementKey).
        ports = [dict(self.PORT_C, id="same", name="A"), dict(self.PORT_C, id="same", name="B")]
        with _SavedConfig(_cfg(portfolios=ports), "cfgload_dupid"):
            at = AppTest.from_file(APP, default_timeout=120).run()
            _click_load(at)
            self.assertFalse(at.exception)
            self.assertEqual(len({p["id"] for p in at.session_state["portfolios_list"]}), 2)


class DeleteByIdTest(unittest.TestCase):
    def test_second_delete_of_the_same_row_is_a_no_op(self):
        """The callback used to get the row index: a second click on the same row, handled before the
        rerun re-rendered the rows, removed the portfolio that had moved into that position."""
        state = SimpleNamespace(portfolios_list=[{"id": "a"}, {"id": "b"}, {"id": "c"}])
        with patch.object(app.st, "session_state", state):
            app.delete_portfolio("b")
            app.delete_portfolio("b")
        self.assertEqual([p["id"] for p in state.portfolios_list], ["a", "c"])

    def test_delete_button_removes_that_portfolio(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        at.button(key=f"del_{_port(at, 'Port B')['id']}").click().run()
        self.assertFalse(at.exception)
        self.assertEqual([p["name"] for p in at.session_state["portfolios_list"]], ["AV-US", "Port C"])


class TickerCapTest(unittest.TestCase):
    @staticmethod
    def _ports(n):
        return [{"name": "P", "tickers": ", ".join(f"T{i:02d}" for i in range(n)),
                 "weights": ", ".join(["0"] * (n - 1) + ["1"])}]

    def test_distinct_tickers_per_run_are_capped(self):
        cap = app.MAX_TICKERS_PER_RUN
        capped = lambda errs: any("distinct tickers" in e for e in errs)
        self.assertFalse(capped(app.validate_inputs(self._ports(cap - 1), "SPY")))    # + benchmark = cap
        errs = app.validate_inputs(self._ports(cap), "SPY")                            # cap + 1
        self.assertTrue(any(f"{cap + 1} distinct tickers" in e for e in errs), errs)
        self.assertFalse(capped(app.validate_inputs(self._ports(cap), "T00")))        # benchmark among them


class VegaFieldEscapeTest(unittest.TestCase):
    def test_nested_access_characters_are_escaped(self):
        self.assertEqual(app._vl_field("Benchmark(510300.SS)"), "Benchmark(510300\\.SS)")
        self.assertEqual(app._vl_field("Port [A]"), "Port \\[A\\]")
        self.assertEqual(app._vl_field("a\\b"), "a\\\\b")
        self.assertEqual(app._vl_field("AV-US 按腿80/125 · 加密60/90"), "AV-US 按腿80/125 · 加密60/90")


if __name__ == "__main__":
    unittest.main()
