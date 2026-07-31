"""Human-readable reporting for audited clinical-validation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


def _format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "not estimable"
    return f"{float(str(value)):.{digits}f}"


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def render_validation_report(
    *,
    run_manifest: Mapping[str, object],
    import_audit: Mapping[str, object],
    scenario_results: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
    repeatability: Mapping[str, object],
    b2_summary: Mapping[str, object],
    abstention_concordance: Mapping[str, object] | None = None,
    cardia_x_track: Mapping[str, object] | None = None,
    assisted_review: Mapping[str, object] | None = None,
    failure_classification: Mapping[str, object] | None = None,
    baseline: Mapping[str, object] | None = None,
    rule_review: Mapping[str, object] | None = None,
    output_path: str | Path,
) -> Path:
    lines = [
        "# CARDIA-X Audited Clinical-Validation Report",
        "",
        "## Acceptance conclusion",
        "",
        (
            "**PASS:** every required, estimable scenario met its pre-registered point-kappa threshold."
            if gate.get("passed")
            else "**FAIL:** one or more required scenarios did not meet the pre-registered acceptance contract."
        ),
        "",
        f"Run ID: `{run_manifest.get('run_id', 'unknown')}`  ",
        f"Ontology hash: `{run_manifest.get('ontology_hash', 'unknown')}`  ",
        f"Scenario-registry hash: `{run_manifest.get('scenario_registry_hash', 'unknown')}`",
        "",
        "This is an internal, project-only alignment audit of a fixed reader study. It is not independent external clinical validation, and no software change can retroactively alter the physician's recorded unassisted response.",
        "",
        "## Population reconciliation",
        "",
        f"The completed workbook contained **{import_audit.get('total_rows')} rows**: **{import_audit.get('b1_rows')} B1 rows**, including **{import_audit.get('b1_unique')} unique B1 cases** and **{import_audit.get('b1_repeats')} concealed repeat rows**, plus **{import_audit.get('b2_rows')} B2 morphology-audit rows**.",
        "",
        f"Unique B1 route counts were `{json.dumps(import_audit.get('unique_b1_route_counts', {}), sort_keys=True)}`. Case-index, case-sheet, response-export, and provenance reconciliation passed before coding.",
        "",
        "## Pre-registered agreement scenarios",
        "",
        "All confidence intervals use cluster bootstrap resampling by original case identity. Point estimates, not lower confidence limits, define this release gate.",
        "",
        "| Scenario | Requirement | n | Status | Observed agreement | Cohen's kappa | 95% CI | Threshold | Gate |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in scenario_results:
        scenario = _object_dict(item.get("scenario", {}))
        result = _object_dict(item.get("result", {}))
        ci = result.get("confidence_interval", [None, None])
        if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None:
            ci_text = f"[{_format_number(ci[0])}, {_format_number(ci[1])}]"
        else:
            ci_text = "not estimable"
        passes = result.get("passes_point_threshold")
        gate_text = "pass" if passes else "fail" if passes is False else "n/a"
        lines.append(
            "| {scenario_id} | {requirement} | {n} | {status} | {po} | {kappa} | {ci} | {threshold} | {gate} |".format(
                scenario_id=scenario.get("scenario_id", "unknown"),
                requirement=scenario.get("requirement", ""),
                n=result.get("sample_size", 0),
                status=result.get("status", ""),
                po=_format_number(result.get("observed_agreement")),
                kappa=_format_number(result.get("kappa")),
                ci=ci_text,
                threshold=_format_number(scenario.get("minimum_kappa", 0.70), 2),
                gate=gate_text,
            )
        )
    lines.extend(
        [
            "",
            "## Failure decomposition",
            "",
            f"Required scenarios below threshold: `{json.dumps(gate.get('below_threshold', []))}`.",
            "",
            f"Required scenarios reported as non-estimable under policy: `{json.dumps(gate.get('required_not_estimable', []))}`.",
            "",
            "Confusion matrices, both class margins, maximum attainable kappa under the observed margins, and the fixed-margin approximation of additional agreements needed for 0.70 are stored in the machine-readable scenario bundle. The disagreement ledger preserves both independently generated coding traces and must not be used to hand-edit case labels.",
            "",
            "## Reader repeatability",
            "",
            f"The 10 concealed B1 repeat pairs produced exact projected agreement of `{_format_number(repeatability.get('observed_agreement'))}` and Cohen's kappa of `{_format_number(repeatability.get('kappa'))}` (`{repeatability.get('status', 'unknown')}`). Repeat rows were excluded from all 100-case primary scenarios and retained only for reliability and clustered uncertainty.",
            "",
        ]
    )
    if abstention_concordance:
        endpoint = _object_dict(abstention_concordance.get("concordance_of_ambiguity", {}))
        interval = endpoint.get("wilson_ci_95", [None, None])
        interval_text = (
            f"[{_format_number(interval[0])}, {_format_number(interval[1])}]"
            if isinstance(interval, (list, tuple))
            and len(interval) == 2
            and interval[0] is not None
            else "not estimable"
        )
        proportion = endpoint.get("proportion")
        lines.extend(
            [
                "## Structural-abstention concordance",
                "",
                f"Using the predeclared composite—{abstention_concordance.get('composite_definition')}—the physician independently corroborated ambiguity in **{endpoint.get('count')}/{endpoint.get('n')} cases ({_format_number(float(str(proportion)) * 100 if proportion is not None else None, 1)}%)**, with a 95% Wilson interval of `{interval_text}`.",
                "",
                "This is descriptive evidence about alignment of deferral with difficult cases. It is not evidence that abstention improves clinical outcomes, and it does not convert the internal reader study into external validation.",
                "",
            ]
        )
    if cardia_x_track:
        results = _object_dict(cardia_x_track.get("results", {}))
        model_acceptance = _object_dict(cardia_x_track.get("acceptance", {}))
        lines.extend(
            [
                "## Frozen CARDIA-X output versus benchmark",
                "",
                (
                    "**CARDIA-X GATE: PASS.**"
                    if model_acceptance.get("passed")
                    else "**CARDIA-X GATE: FAIL.** Required all-case model projections did not all reach point kappa 0.70."
                ),
                "",
                f"This track is separate from physician-versus-benchmark agreement. CARDIA-X issued diagnoses for **{cardia_x_track.get('diagnosed_n')}/{cardia_x_track.get('n')} cases** (coverage `{_format_number(cardia_x_track.get('coverage'))}`) and operationally abstained for **{cardia_x_track.get('abstained_n')}**. In abstention cases, the displayed internal candidate was retained only for audit and was not counted as an issued diagnosis.",
                "",
                "| Projection | Covered n | Covered agreement | Covered kappa | Intention-to-diagnose n | ITD agreement | ITD kappa |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for projection in ("exact", "five_family", "binary", "dominant"):
            projection_results = _object_dict(results.get(projection, {}))
            covered = _object_dict(projection_results.get("covered_cases", {}))
            itd = _object_dict(projection_results.get("intention_to_diagnose", {}))
            lines.append(
                f"| {projection} | {covered.get('sample_size', 0)} | "
                f"{_format_number(covered.get('observed_agreement'))} | "
                f"{_format_number(covered.get('kappa'))} | "
                f"{itd.get('sample_size', 0)} | "
                f"{_format_number(itd.get('observed_agreement'))} | "
                f"{_format_number(itd.get('kappa'))} |"
            )
        lines.extend(
            [
                "",
                "The CARDIA-X track is diagnostic-only and cannot override or repair the preregistered physician agreement gate.",
                "",
            ]
        )
    if assisted_review:
        overall = _object_dict(assisted_review.get("overall", {}))
        abstention_rating = _object_dict(
            assisted_review.get("structural_abstention_appropriate", {})
        )
        conflict_rating = _object_dict(
            assisted_review.get("conflict_region_plausible", {})
        )
        lines.extend(
            [
                "## Post-assistance explanatory utility",
                "",
                "Post-assistance fields were analyzed only after the unassisted coding pass and never entered the physician-versus-benchmark labels.",
                "",
                f"Across 100 unique B1 cases, mean utility was `{_format_number(_object_dict(overall.get('utility', {})).get('mean'))}`, mean soundness `{_format_number(_object_dict(overall.get('soundness', {})).get('mean'))}`, mean safety `{_format_number(_object_dict(overall.get('safety', {})).get('mean'))}`, and mean clarity `{_format_number(_object_dict(overall.get('clarity', {})).get('mean'))}` on the recorded ordinal scales.",
                "",
                f"The physician marked abstention appropriate in `{abstention_rating.get('yes')}/{abstention_rating.get('n')}` structural-abstention cases and conflict plausible in `{conflict_rating.get('yes')}/{conflict_rating.get('n')}` conflict-region cases.",
                "",
            ]
        )
    lines.extend(
        [
            "## B2 morphology and feature plausibility",
            "",
            "B2 is not treated as a physician-versus-benchmark diagnostic kappa task. Its outputs are proportions and ordinal summaries because a degenerate benchmark margin cannot support Cohen's kappa.",
            "",
            "```json",
            json.dumps(b2_summary, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Interpretation limits",
            "",
            "- Physician coding used only the unassisted diagnosis, unassisted rationale, pre-assistance evidence checkboxes, confidence, ambiguity, quality, and additional-information fields.",
            "- Benchmark coding ran independently from project provenance and source metadata. Routes and CARDIA-X outputs were unavailable to the physician coder; they entered only separately labeled audits.",
            "- Unknown or unsupported benchmark statements remained visible as residual abnormal findings; they were not silently converted to normal.",
            "- A failed 0.70 gate is a valid empirical outcome. Unsupported synthetic agreement is prohibited.",
            "- Optimizing core CARDIA-X components against project training/validation folds does not change historical physician-versus-benchmark agreement and does not establish external clinical generalization.",
            "",
        ]
    )
    lines.extend(
        [
            "## Global rule-soundness review",
            "",
            (
                f"Rule-review status is `{rule_review.get('status')}`. "
                f"Complete definitions: `{rule_review.get('complete_rule_definitions')}/10`; "
                f"fully scored rules: `{rule_review.get('fully_scored_rules')}/10`; "
                f"mean Likert: `{_format_number(rule_review.get('mean_likert'))}`."
                if rule_review
                else "No rule-review summary was available."
            ),
            "",
            (
                str(rule_review.get("reason", ""))
                if rule_review
                else "Rule soundness was not estimated."
            ),
            "",
        ]
    )
    if failure_classification:
        lines.extend(
            [
                "## Failure-cause classification",
                "",
                f"All **{failure_classification.get('failed_required_scenario_count')}** failed or non-estimable required scenarios were classified under the locked correction policy. Remaining below-threshold results are recorded as empirical physician–benchmark disagreement after generalizable semantic corrections; degenerate margins remain explicitly non-estimable. The complete scenario-level classifications are in `failure_classification.json`, and case evidence remains in `disagreement_ledger.csv`.",
                "",
            ]
        )
    if baseline:
        lines.extend(
            [
                "## Baseline reproduction",
                "",
                "Strict primary-diagnosis-only and broader unassisted baselines were recomputed in this run. Full margins and version-drift deltas are stored in `baseline_reference.json`.",
                "",
                "| Track | Exact κ | Five-family κ | Binary κ | Dominant κ |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for track in ("strict", "broader"):
            values = _object_dict(baseline.get(track, {}))
            lines.append(
                f"| {track} | {_format_number(_object_dict(values.get('exact', {})).get('kappa'))} | "
                f"{_format_number(_object_dict(values.get('five_family', {})).get('kappa'))} | "
                f"{_format_number(_object_dict(values.get('binary', {})).get('kappa'))} | "
                f"{_format_number(_object_dict(values.get('dominant', {})).get('kappa'))} |"
            )
        lines.append("")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
