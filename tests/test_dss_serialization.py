import json

from tm_ecg.dss.discretization import fit_wedd_discretization
from tm_ecg.dss.rules import build_decision_system


def test_decision_system_and_plan_serialize(tmp_path):
    rows = [
        {"qrs_dur_med_ms": 90.0, "record_id": "a"},
        {"qrs_dur_med_ms": 140.0, "record_id": "b"},
    ]
    labels = ["Normal", "RBBB spectrum"]
    plan = fit_wedd_discretization(rows, labels, features=["qrs_dur_med_ms"], min_support=1)
    system = build_decision_system(rows, labels, plan, object_ids=["a", "b"])
    plan_path = plan.to_json(tmp_path / "plan.json")
    system_path = system.to_json(tmp_path / "system.json")
    assert json.loads(plan_path.read_text())["orientation"] == "B_hat = A @ T"
    assert json.loads(system_path.read_text())["universe"] == ["a", "b"]
