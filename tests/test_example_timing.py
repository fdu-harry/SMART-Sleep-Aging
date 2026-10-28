from pathlib import Path
import copy
import importlib.util
import json
import re
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smart_demo", ROOT / "examples/run_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def _record(name):
    return json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))


def _value(fields, key):
    return float(re.match(r"^[0-9.]+", fields[key])[0])


def test_continuity_scenario_closes_time_balance_and_audits_waso():
    original = _record("02_sleep_continuity")
    before = copy.deepcopy(original)
    changed, audit = demo.scenario(original, "TST", 360)
    fields = changed["Sleep study measurements"]
    assert original == before
    assert _value(fields, "Total recording time") == 420
    assert _value(fields, "Total sleep time") == 360
    assert _value(fields, "Sleep onset latency") == 30
    assert _value(fields, "Wake after sleep onset") == 30
    assert _value(fields, "Sleep efficiency") == pytest.approx(100 * 360 / 420, abs=0.05)
    assert {row["field"] for row in audit} == {"TST", "SE", "WASO"}
    waso = next(row for row in audit if row["field"] == "WASO")
    assert waso == dict(field="WASO", baseline=90.0, counterfactual=30.0)
    text = demo.render_scn(changed)
    assert "Wake after sleep onset: 30 min" in text
    assert "Total sleep time: 360 min" in text


def test_tst_scenario_preserves_positive_terminal_wake():
    original = _record("02_sleep_continuity")
    original["Sleep study measurements"]["Total recording time"] = "430 min"
    original["Sleep study measurements"]["Sleep efficiency"] = "69.77%"
    changed, _ = demo.scenario(original, "TST", 360)
    fields = changed["Sleep study measurements"]
    terminal = (_value(fields, "Total recording time")
                - _value(fields, "Total sleep time")
                - _value(fields, "Sleep onset latency")
                - _value(fields, "Wake after sleep onset"))
    assert terminal == 10
    assert _value(fields, "Wake after sleep onset") == 30


def test_tst_scenario_rejects_negative_waso():
    with pytest.raises(ValueError, match="insufficient time"):
        demo.scenario(_record("02_sleep_continuity"), "TST", 400)


def test_ahi_scenario_preserves_sleep_timing():
    original = _record("01_respiratory_context")
    changed, audit = demo.scenario(original, "AHI", 5)
    fields = changed["Sleep study measurements"]
    assert _value(fields, "Wake after sleep onset") == 60
    assert all(row["field"] != "WASO" for row in audit)
    assert sum(_value(fields, key) for key in ["Obstructive apnea index", "Central apnea index", "Mixed apnea index", "Hypopnea index"]) == 5
