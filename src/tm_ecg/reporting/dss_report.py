"""DSS report generation logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _condition_text(condition: Any) -> str:
    atom = f"{condition.feature}={condition.state}"
    return f"NOT {atom}" if getattr(condition, "negated", False) else atom


def write_dss_markdown_report(
    path: Path,
    dataset: str,
    selected_features: list[str],
    labels: list[str],
    rules: list[Any],
    min_support: int,
) -> None:
    by_label: dict[str, list[Any]] = {}
    for rule in rules:
        by_label.setdefault(rule.target_label, []).append(rule)
    lines = [
        f"# DSS production-rule report for {dataset.upper()}",
        "",
        "This research artifact converts discretized clinician-understandable B-matrix rows into rough-set production rules.",
        "It is not an autonomous diagnostic medical device; final interpretation remains a clinician responsibility.",
        "",
        "## Orientation and transition convention",
        "",
        "The repository convention is `B_hat = A @ T`, where `A` is `m x k`, `B` is `m x l`, and `T` is `k x l`.",
        "",
        "## Input summary",
        "",
        f"- Training objects: {len(labels)}",
        f"- Distinct labels: {', '.join(sorted(set(labels)))}",
        f"- Selected B features: {len(selected_features)}",
        f"- Minimum support for deterministic rules: {min_support}",
        f"- Induced deterministic rules: {len(rules)}",
        "",
        "## Top rules by label",
    ]
    for label, label_rules in sorted(by_label.items()):
        lines.extend(["", f"### {label}"])
        for rule in sorted(
            label_rules,
            key=lambda item: (
                -float(getattr(item, "predicate_similarity", {}).get("score", 0.0)),
                -item.support_count,
                len(item.antecedents),
            ),
        )[:5]:
            atoms = ", ".join(_condition_text(cond) for cond in rule.antecedents[:8])
            if len(rule.antecedents) > 8:
                atoms += ", ..."
            sim = getattr(rule, "predicate_similarity", {}).get("score")
            sim_text = f", predicate_similarity={sim:.3f}" if sim is not None else ""
            physician = getattr(rule, "physician_predicate_label", None)
            physician_text = f", closest_physician_predicate={physician}" if physician else ""
            lines.append(
                f"- `{rule.rule_id}`: IF {atoms} THEN {label}; support={rule.support_count}, "
                f"confidence={rule.confidence:.3f}, reduced_conditions={rule.reduced_from_conditions}"
                f"{sim_text}{physician_text}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
