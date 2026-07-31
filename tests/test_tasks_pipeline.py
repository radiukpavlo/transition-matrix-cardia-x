"""Ordering tests for the public Invoke reproduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from invoke import Exit

import tasks


@dataclass(frozen=True)
class _Result:
    ok: bool = True
    exited: int = 0


class _RecordingContext:
    def __init__(self, *, dss_exit: int = 0, report_exit: int = 0) -> None:
        self.dss_exit = dss_exit
        self.report_exit = report_exit
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, command: str, **kwargs: Any) -> _Result:
        self.calls.append((command, kwargs))
        if " dss " in command:
            return _Result(ok=self.dss_exit == 0, exited=self.dss_exit)
        if " report " in command:
            return _Result(ok=self.report_exit == 0, exited=self.report_exit)
        return _Result()

    @property
    def commands(self) -> list[str]:
        return [command for command, _ in self.calls]


def test_b1_pipeline_orders_model_transition_and_reporting_stages() -> None:
    context = _RecordingContext()

    tasks.pipeline.body(context, dataset="b1")

    classifier_index = next(
        index for index, command in enumerate(context.commands) if " train-classifier " in command
    )
    transition_index = next(
        index for index, command in enumerate(context.commands) if " fit-transition " in command
    )
    dss_index = next(
        index for index, command in enumerate(context.commands) if " dss " in command
    )
    report_index = next(
        index for index, command in enumerate(context.commands) if " report " in command
    )
    assert classifier_index < transition_index < dss_index < report_index
    assert report_index == len(context.commands) - 1
    assert all(call[1]["env"] == {"PYTHONPATH": "src"} for call in context.calls)
    assert context.calls[dss_index][1]["warn"] is True
    assert context.calls[report_index][1]["warn"] is True


def test_pipeline_writes_report_before_propagating_rule_gate_failure() -> None:
    context = _RecordingContext(dss_exit=8)
    try:
        tasks.pipeline.body(context, dataset="b1")
    except Exit as error:
        assert error.code == 8
    else:
        raise AssertionError("failed rule gate must propagate a nonzero status")
    dss_index = next(
        index for index, command in enumerate(context.commands) if " dss " in command
    )
    report_index = next(
        index for index, command in enumerate(context.commands) if " report " in command
    )
    assert dss_index < report_index


def test_pipeline_rejects_unknown_dataset_before_running_commands() -> None:
    context = _RecordingContext()
    try:
        tasks.pipeline.body(context, dataset="unknown")
    except Exit as error:
        assert error.code == 2
    else:
        raise AssertionError("unknown dataset must fail closed")
    assert context.commands == []
