"""Single-record inference: raw ECG -> class prediction + clinician explanation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from tm_ecg.config import ProjectConfig
from tm_ecg.dss.eligibility import (
    EligibilityCertificateError,
    verify_rulebook_eligibility_certificate,
)
from tm_ecg.dss.models import (
    AttributeDomain,
    DiscretizationPlan,
    IntervalBin,
    MedicalPredicate,
    ProductionRule,
    RuleCondition,
    ThresholdRecord,
)
from tm_ecg.dss.rules import infer_with_rules
from tm_ecg.io.common import read_json, sha256_file
from tm_ecg.transition.ridge import (
    apply_transition_package,
    load_operator_package,
)
from tm_ecg.transition.typed_transforms import inverse_rows
from tm_ecg.types import TransformBundle, TransformColumnStats, TypedAbstention
from tm_ecg.modeling.calibration import (
    apply_temperature,
    compatibility_from_calibrated_axes,
)
from tm_ecg.modeling.label_contract import DEFAULT_COMPATIBILITY_CONTRACT_V4

logger = logging.getLogger(__name__)

_MEASURED_DSS_QUALITY_FEATURES = (
    "lead_quality_min_db",
    "delineation_confidence",
    "analyzable_duration_s",
    "rhythm_valid_beat_fraction",
    "atrial_valid_beat_fraction",
    "qrs_valid_beat_fraction",
    "st_t_valid_beat_fraction",
    "atrial_lead_coverage",
    "qrs_lead_coverage",
    "st_t_lead_coverage",
    "detector_agreement",
)


def _load_bundle(path: Path) -> TransformBundle:
    payload = read_json(path)
    return TransformBundle(
        dataset=str(payload["dataset"]),
        fit_columns=list(payload["fit_columns"]),
        dropped_columns=list(payload.get("dropped_columns", [])),
        stats=[TransformColumnStats(**item) for item in payload["stats"]],
    )


def _load_rulebook(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DSS rulebook must be a JSON object")
    return payload


def _condition_from_dict(payload: Mapping[str, Any]) -> RuleCondition:
    allowed = RuleCondition.__dataclass_fields__
    return RuleCondition(**{key: value for key, value in payload.items() if key in allowed})


def _plan_from_dict(payload: Mapping[str, Any]) -> DiscretizationPlan:
    domains: dict[str, AttributeDomain] = {}
    for name, raw_domain in dict(payload.get("feature_domains", {})).items():
        domain_payload = dict(raw_domain)
        bins = [IntervalBin(**dict(item)) for item in domain_payload.pop("bins", [])]
        domains[str(name)] = AttributeDomain(bins=bins, **domain_payload)
    thresholds = [ThresholdRecord(**dict(item)) for item in payload.get("thresholds", [])]
    return DiscretizationPlan(
        feature_domains=domains,
        thresholds=thresholds,
        class_labels=[str(value) for value in payload.get("class_labels", [])],
        alpha=float(payload.get("alpha", 0.0)),
        min_support=int(payload.get("min_support", 0)),
        max_depth=int(payload.get("max_depth", 0)),
        orientation=str(payload.get("orientation", "B_hat = A @ T")),
    )


def _rules_from_dict(payloads: list[Mapping[str, Any]]) -> list[ProductionRule]:
    rules: list[ProductionRule] = []
    allowed = ProductionRule.__dataclass_fields__
    for raw in payloads:
        row = {key: value for key, value in dict(raw).items() if key in allowed}
        row["antecedents"] = [
            _condition_from_dict(dict(item)) for item in raw.get("antecedents", [])
        ]
        rules.append(ProductionRule(**row))
    return rules


def _predicates_from_dict(
    payload: Mapping[str, Any],
) -> dict[str, MedicalPredicate]:
    predicates: dict[str, MedicalPredicate] = {}
    for label, raw in payload.items():
        row = dict(raw)
        predicates[str(label)] = MedicalPredicate(
            label=str(row.get("label", label)),
            source_labels={
                str(dataset): [str(value) for value in values]
                for dataset, values in dict(row.get("source_labels", {})).items()
            },
            required=[
                _condition_from_dict(dict(item)) for item in row.get("required", [])
            ],
            supportive=[
                _condition_from_dict(dict(item)) for item in row.get("supportive", [])
            ],
            contraindications=[
                _condition_from_dict(dict(item))
                for item in row.get("contraindications", [])
            ],
            references=[str(value) for value in row.get("references", [])],
            explanation=str(row.get("explanation", "")),
            weak_signature=bool(row.get("weak_signature", False)),
        )
    return predicates


def _abstained_dss_result(reason: str, detail: str) -> dict[str, Any]:
    return {
        "predicted_label": None,
        "ranked_scores": [],
        "activated_rules": [],
        "uncertainty_flags": [reason],
        "explanation": detail,
        "abstention": TypedAbstention(
            reason=reason,
            failed_requirement=detail,
            needed_features=(),
        ).to_dict(),
        "score_decomposition": {},
    }


def _compatibility_set_from_scores(
    class_scores: Mapping[str, float],
    thresholds: Mapping[str, object] | list[object] | None,
) -> list[str]:
    """Apply frozen per-label thresholds and the v4 set-validity contract."""

    from tm_ecg.constants import PROJECT_LABELS

    if isinstance(thresholds, Mapping):
        threshold_by_label = {
            label: float(thresholds.get(label, 0.5)) for label in PROJECT_LABELS
        }
    elif isinstance(thresholds, list) and len(thresholds) == len(PROJECT_LABELS):
        threshold_by_label = {
            label: float(value)
            for label, value in zip(PROJECT_LABELS, thresholds, strict=True)
        }
    else:
        threshold_by_label = {label: 0.5 for label in PROJECT_LABELS}
    selected = [
        label
        for label in PROJECT_LABELS
        if float(class_scores.get(label, 0.0)) >= threshold_by_label[label]
    ]
    return list(
        DEFAULT_COMPATIBILITY_CONTRACT_V4.normalize(
            selected,
            empty_policy="residual",
        )
    )


def _validate_executable_rulebook(payload: Mapping[str, Any]) -> None:
    """Reject ambiguous, empty, or ontology-external executable content."""

    from tm_ecg.constants import PROJECT_LABELS

    selected_raw = payload.get("selected_features")
    if not isinstance(selected_raw, list) or not selected_raw:
        raise ValueError("eligible rulebook lacks selected_features")
    if any(
        not isinstance(feature, str) or not feature or feature.strip() != feature
        for feature in selected_raw
    ):
        raise ValueError("selected_features must contain non-empty canonical strings")
    selected_features = set(selected_raw)
    if len(selected_features) != len(selected_raw):
        raise ValueError("selected_features contains duplicates")

    raw_plan = payload.get("discretization_plan")
    if not isinstance(raw_plan, Mapping):
        raise ValueError("eligible rulebook lacks a discretization_plan object")
    raw_domains = raw_plan.get("feature_domains")
    if not isinstance(raw_domains, Mapping) or not raw_domains:
        raise ValueError("eligible rulebook lacks executable feature domains")
    if any(not isinstance(name, str) or not name for name in raw_domains):
        raise ValueError("feature domain names must be non-empty strings")
    if set(raw_domains) != selected_features:
        raise ValueError(
            "selected_features must exactly match discretization feature domains"
        )

    raw_class_labels = raw_plan.get("class_labels")
    if not isinstance(raw_class_labels, list) or not raw_class_labels:
        raise ValueError("eligible rulebook lacks discretization class labels")
    if any(not isinstance(label, str) or not label for label in raw_class_labels):
        raise ValueError("discretization class labels must be non-empty strings")
    class_labels = set(raw_class_labels)
    if len(class_labels) != len(raw_class_labels):
        raise ValueError("discretization class labels contain duplicates")
    known_targets = set(PROJECT_LABELS)
    unknown_class_labels = class_labels - known_targets
    if unknown_class_labels:
        raise ValueError(
            f"discretization plan contains unknown targets: {sorted(unknown_class_labels)}"
        )

    raw_predicates = payload.get("medical_predicates")
    if not isinstance(raw_predicates, Mapping) or not raw_predicates:
        raise ValueError("eligible rulebook lacks medical predicates")
    predicate_labels = set(raw_predicates)
    if any(not isinstance(label, str) or not label for label in predicate_labels):
        raise ValueError("medical predicate labels must be non-empty strings")
    unknown_predicates = predicate_labels - known_targets
    if unknown_predicates:
        raise ValueError(
            f"medical predicate library contains unknown targets: {sorted(unknown_predicates)}"
        )
    for label, raw_predicate in raw_predicates.items():
        if not isinstance(raw_predicate, Mapping):
            raise ValueError(f"medical predicate {label!r} must be an object")
        predicate_label = raw_predicate.get("label", label)
        if predicate_label != label:
            raise ValueError(f"medical predicate key/label mismatch for {label!r}")
        condition_count = 0
        for field in ("required", "supportive", "contraindications"):
            conditions = raw_predicate.get(field, [])
            if not isinstance(conditions, list):
                raise ValueError(f"medical predicate {label!r}.{field} must be an array")
            condition_count += len(conditions)
        if condition_count == 0:
            raise ValueError(f"medical predicate {label!r} has no executable conditions")

    raw_rules = payload.get("production_rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("eligible rulebook lacks production rules")
    rule_ids: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"production rule at index {index} must be an object")
        rule_id = raw_rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError(f"production rule at index {index} lacks rule_id")
        if rule_id in rule_ids:
            raise ValueError(f"duplicate production rule_id {rule_id!r}")
        rule_ids.add(rule_id)
        target = raw_rule.get("target_label")
        if not isinstance(target, str) or not target:
            raise ValueError(f"production rule {rule_id!r} lacks target_label")
        if target not in known_targets:
            raise ValueError(f"production rule {rule_id!r} has unknown target {target!r}")
        if target not in class_labels or target not in predicate_labels:
            raise ValueError(
                f"production rule {rule_id!r} target is not bound to its plan and predicate"
            )
        antecedents = raw_rule.get("antecedents")
        if not isinstance(antecedents, list) or not antecedents:
            raise ValueError(f"production rule {rule_id!r} has no antecedents")
        for condition_index, raw_condition in enumerate(antecedents):
            if not isinstance(raw_condition, Mapping):
                raise ValueError(
                    f"production rule {rule_id!r} antecedent {condition_index} must be an object"
                )
            feature = raw_condition.get("feature")
            if not isinstance(feature, str) or not feature:
                raise ValueError(
                    f"production rule {rule_id!r} antecedent {condition_index} lacks feature"
                )
            if feature not in selected_features:
                raise ValueError(
                    f"production rule {rule_id!r} references unselected feature {feature!r}"
                )


def _infer_with_rulebook(
    feature_row: Mapping[str, object],
    payload: Mapping[str, Any],
    config: ProjectConfig | SimpleNamespace,
) -> dict[str, Any]:
    """Execute a schema-compatible rulebook or fail closed with typed abstention."""

    if payload.get("status") != "eligible" or payload.get("rules_eligible") is not True:
        return _abstained_dss_result(
            "dss_rulebook_ineligible",
            "The strict DSS eligibility gate has not authorized a production rulebook.",
        )
    if str(payload.get("ontology_version", "")) != str(config.ontology_version):
        return _abstained_dss_result(
            "dss_ontology_mismatch",
            "The DSS rulebook ontology version does not match the active inference configuration.",
        )
    try:
        verify_rulebook_eligibility_certificate(payload, config)
    except (EligibilityCertificateError, TypeError, ValueError) as exc:
        return _abstained_dss_result(
            "dss_eligibility_certificate_invalid",
            f"The DSS rulebook authorization certificate is invalid: {exc}",
        )
    try:
        _validate_executable_rulebook(payload)
        plan = _plan_from_dict(dict(payload["discretization_plan"]))
        rules = _rules_from_dict(
            [dict(item) for item in payload.get("production_rules", [])]
        )
        predicates = _predicates_from_dict(
            dict(payload.get("medical_predicates", {}))
        )
        if not rules or not plan.feature_domains:
            raise ValueError("eligible rulebook lacks executable rules or domains")
    except (KeyError, TypeError, ValueError) as exc:
        return _abstained_dss_result(
            "invalid_dss_rulebook",
            f"The production rulebook is incomplete or malformed: {exc}",
        )
    policy = dict(getattr(config, "dss", {}))
    return infer_with_rules(
        feature_row,
        rules,
        plan,
        predicates=predicates,
        soft_match_min_fraction=float(policy.get("soft_match_min_fraction", 0.60)),
        top_k=int(policy.get("top_k", 5)),
        close_vote_ratio=float(policy.get("close_vote_ratio", 1.10)),
        low_strength_threshold=float(policy.get("low_strength_threshold", 0.20)),
        contradiction_penalty=float(policy.get("contradiction_penalty", 0.25)),
        hard_veto_criticality=float(policy.get("hard_veto_criticality", 1.5)),
    ).to_dict()


def _measure_dss_quality_features(
    signal: Any,
    fs: float,
    sig_names: list[str],
    config: ProjectConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    """Measure safety guards from the current waveform or return explicit missing values."""

    from tm_ecg.features.formulas import (
        BeatMeasurement,
        RecordMeasurements,
        compute_record_features,
    )
    from tm_ecg.real_data import _one_record_measurements

    missing = {feature: None for feature in _MEASURED_DSS_QUALITY_FEATURES}
    try:
        measurements, _triads, ratios, quality_by_lead, provenance = (
            _one_record_measurements(
                signal,
                fs,
                sig_names,
                "inference",
                config,
            )
        )
        record = RecordMeasurements(
            record_id="inference",
            beats=[BeatMeasurement(**dict(beat)) for beat in measurements],
            tq_power_ratios=list(ratios),
            sampling_rate_hz=float(fs),
            lead_quality_by_lead_db={
                str(key): float(value) for key, value in quality_by_lead.items()
            },
            analyzable_duration_s=float(
                provenance.get("analyzable_duration_s", 0.0)
            ),
        )
        computed = compute_record_features(record, config.thresholds)
        measured = {
            feature: computed.get(feature)
            for feature in _MEASURED_DSS_QUALITY_FEATURES
        }
        measured["lead_quality_min_db"] = (
            min(quality_by_lead.values()) if quality_by_lead else None
        )
        return measured, {
            "status": "measured_from_current_waveform",
            "method": provenance.get("fiducial_method"),
            "accepted_morphology_beat_count": provenance.get(
                "accepted_morphology_beat_count", 0
            ),
            "accepted_quality_valid_beat_count": provenance.get(
                "accepted_quality_valid_beat_count", 0
            ),
            "r_detector_agreement_f1": provenance.get(
                "r_detector_agreement_f1"
            ),
        }
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning("DSS quality measurement failed closed: %s", exc)
        return missing, {
            "status": "unavailable_fail_closed",
            "reason": str(exc),
        }


def _metadata_path_for_operator(operator_path: Path) -> Path:
    name = operator_path.name
    if name.endswith("_T_ridge.npz"):
        return operator_path.with_name(name.replace("_T_ridge.npz", "_operator_metadata.json"))
    if name.endswith("_T_ridge.json"):
        return operator_path.with_name(name.replace("_T_ridge.json", "_operator_metadata.json"))
    return operator_path.with_name("operator_metadata.json")


def _validate_transition_artifacts(
    config: ProjectConfig,
    operator_path: Path,
    bundle_path: Path,
) -> dict[str, object]:
    """Fail closed unless inference artifacts match strict transition metadata."""

    metadata_path = _metadata_path_for_operator(operator_path)
    if not metadata_path.exists():
        raise RuntimeError("Transition operator lacks strict provenance metadata")
    metadata = read_json(metadata_path)
    if metadata.get("artifact_version") != 2:
        raise RuntimeError("Transition metadata uses an unsupported provenance contract")
    if metadata.get("ontology_version") != config.ontology_version:
        raise RuntimeError("Transition metadata ontology does not match inference config")
    operator_hash_key = (
        "operator_sha256" if operator_path.suffix == ".npz" else "legacy_operator_sha256"
    )
    if (
        not operator_path.exists()
        or metadata.get(operator_hash_key) != sha256_file(operator_path)
    ):
        raise RuntimeError("Transition operator hash mismatch")
    if (
        not bundle_path.exists()
        or metadata.get("transform_bundle_sha256") != sha256_file(bundle_path)
    ):
        raise RuntimeError("Transition transform-bundle hash mismatch")
    a_bundle_path = Path(str(metadata.get("a_preprocess_bundle", "")))
    if (
        not a_bundle_path.exists()
        or metadata.get("a_preprocess_bundle_sha256") != sha256_file(a_bundle_path)
    ):
        raise RuntimeError("Transition latent-preprocess bundle hash mismatch")
    return metadata


def _maybe_reduce_latent(latent_vector: Any, operator_path: Path, operator: list[list[float]]):
    if not operator or len(operator) == len(latent_vector):
        return latent_vector
    metadata_path = _metadata_path_for_operator(operator_path)
    if not metadata_path.exists():
        return latent_vector
    metadata = read_json(metadata_path)
    a_bundle_path = metadata.get("a_preprocess_bundle")
    if not a_bundle_path:
        return latent_vector

    import numpy as np  # type: ignore

    from tm_ecg.transition.a_preprocess import apply_a_preprocess_bundle, read_a_preprocess_bundle

    latent_row: dict[str, object] = {"record_id": "inference", "split": "inference"}
    for idx, value in enumerate(latent_vector.tolist()):
        latent_row[f"latent_{idx:04d}"] = float(value)
    reduced = apply_a_preprocess_bundle([latent_row], read_a_preprocess_bundle(Path(str(a_bundle_path))))
    values = [value for key, value in sorted(reduced[0].items()) if key.startswith("a_red_")]
    return np.asarray(values, dtype=np.float32)


def predict_and_explain(
    signal: Any,
    fs: int,
    sig_names: list[str],
    config: ProjectConfig,
    checkpoint_path: str | Path,
    operator_path: str | Path,
    bundle_path: str | Path,
    rulebook_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run raw-signal inference through classifier, transition map, and DSS rules."""

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError("Inference requires numpy and torch. Install with: pip install 'tm-ecg[train]'") from exc

    from tm_ecg.constants import PROJECT_LABELS
    from tm_ecg.modeling.classifier import build_model
    from tm_ecg.real_data import representative_triad_tensor

    triad_result = representative_triad_tensor(signal, fs, sig_names, config)
    if triad_result is None:
        return {
            "predicted_label": None,
            "compatibility_label_set": [],
            "semantic_measurements": {},
            "fired_rules": [],
            "rule_conflicts": [],
            "route": "structural_abstention",
            "confidence": None,
            "calibration_status": "not_evaluated",
            "quality_status": "triad_unavailable",
            "abstention_reason": "could_not_construct_valid_triad",
            "model_and_contract_hashes": {},
            "class_scores": {},
            "predicted_features": {},
            "activated_rules": [],
            "error": "Could not construct a valid triad from the input signal.",
        }
    triad_tensor, _triad_peaks = triad_result

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model = build_model(
        in_leads=len(sig_names),
        triad_length=3,
        samples_per_beat=int(config.training["samples_per_beat"]),
        latent_dim=int(config.latents["penultimate_dim"]),
        num_classes=len(PROJECT_LABELS),
        axis_classes={
            str(axis): len(values)
            for axis, values in dict(checkpoint.get("axis_labels", {})).items()
        },
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    input_tensor = torch.tensor(triad_tensor[None, ...], dtype=torch.float32)
    with torch.no_grad():
        if checkpoint.get("multiaxial"):
            logits, axis_logits, latent = model.forward_multiaxial(input_tensor)
        else:
            logits, latent = model(input_tensor)
            axis_logits = {}
        calibration = dict(checkpoint.get("calibration", {}))
        compatibility_calibration = dict(
            calibration.get("compatibility_head", {})
        )
        probabilities = torch.sigmoid(
            apply_temperature(logits, compatibility_calibration)
        ).squeeze(0).numpy()
    class_scores = {label: float(probabilities[idx]) for idx, label in enumerate(PROJECT_LABELS)}
    model_compatibility_label_set = _compatibility_set_from_scores(
        class_scores,
        checkpoint.get("compatibility_thresholds"),
    )
    legacy_predicted_label = max(class_scores, key=class_scores.get)
    axis_calibration = dict(calibration.get("axes", {}))
    axis_scores = {
        axis: {
            label: float(value)
            for label, value in zip(
                checkpoint.get("axis_labels", {}).get(axis, []),
                torch.sigmoid(
                    apply_temperature(values, dict(axis_calibration.get(axis, {})))
                ).squeeze(0).numpy(),
                strict=True,
            )
        }
        for axis, values in axis_logits.items()
    }
    if axis_scores and axis_calibration:
        predicted_label, compatibility_trace = compatibility_from_calibrated_axes(
            axis_scores, calibration
        )
    else:
        predicted_label = legacy_predicted_label
        compatibility_trace = {
            "source": "legacy_head_uncalibrated_axis_fallback",
            "reason": "checkpoint lacks a calibrated multiaxial artifact",
        }

    resolved_operator_path = Path(operator_path)
    resolved_bundle_path = Path(bundle_path)
    _validate_transition_artifacts(
        config,
        resolved_operator_path,
        resolved_bundle_path,
    )
    operator_payload = load_operator_package(resolved_operator_path)
    operator = operator_payload["operator"]
    latent_vector = _maybe_reduce_latent(
        latent.squeeze(0).cpu().numpy().astype(np.float32),
        resolved_operator_path,
        operator,
    )
    predicted_fit = apply_transition_package(
        [latent_vector.tolist()],
        operator_payload,
    )

    bundle = _load_bundle(resolved_bundle_path)
    fit_row: dict[str, object] = {"record_id": "inference"}
    for column, value in zip(bundle.fit_columns, predicted_fit[0], strict=False):
        fit_row[column] = value
    predicted_features = {
        key: value
        for key, value in inverse_rows([fit_row], bundle)[0].items()
        if key != "record_id"
    }

    model_predicted_label = predicted_label
    activated_rules: list[dict[str, Any]] = []
    dss_result: dict[str, Any] | None = None
    rulebook_status: str | None = None
    dss_feature_provenance: dict[str, object] | None = None
    if rulebook_path is not None:
        rulebook = _load_rulebook(Path(rulebook_path))
        rulebook_status = str(rulebook.get("status", "invalid"))
        measured_quality, dss_feature_provenance = _measure_dss_quality_features(
            signal,
            fs,
            sig_names,
            config,
        )
        # Safety-critical quality guards must describe this waveform. Transition-
        # predicted quality values are never permitted to authorize a DSS rule.
        predicted_features.update(measured_quality)
        dss_result = _infer_with_rulebook(predicted_features, rulebook, config)
        activated_rules = list(dss_result.get("activated_rules", []))
        # When a DSS artifact is supplied, only its authorized, quality-aware
        # result may become the public decision.  The model-only output remains
        # available under an explicitly separate research field.
        predicted_label = dss_result.get("predicted_label")

    logger.info("Inference complete: predicted=%s, rules_activated=%d", predicted_label, len(activated_rules))
    uncertainty_flags = (
        [
            str(value)
            for value in dss_result.get("uncertainty_flags", [])
        ]
        if isinstance(dss_result, Mapping)
        else []
    )
    rule_conflicts = [
        value
        for value in uncertainty_flags
        if "conflict" in value or "contradict" in value
    ]
    abstention_payload = (
        dss_result.get("abstention")
        if isinstance(dss_result, Mapping)
        else None
    )
    if abstention_payload:
        route = "structural_abstention"
    elif rule_conflicts:
        route = "conflict_region"
    elif activated_rules and not uncertainty_flags:
        route = "exact_match"
    elif activated_rules:
        route = "soft_match"
    else:
        route = "structural_abstention"
    top_rule_score = None
    if (
        isinstance(dss_result, Mapping)
        and isinstance(dss_result.get("ranked_scores"), list)
        and dss_result.get("ranked_scores")
    ):
        first_score = dss_result.get("ranked_scores", [])[0]
        if isinstance(first_score, Mapping):
            top_rule_score = first_score.get("score")
        elif isinstance(first_score, (list, tuple)) and len(first_score) >= 2:
            top_rule_score = first_score[1]
    confidence = (
        float(top_rule_score)
        if top_rule_score is not None
        else max(class_scores.values(), default=0.0)
    )
    resolved_checkpoint_path = Path(checkpoint_path)
    contract_path = config.paths.root / "configs/compatibility_label_contract_v4.yaml"
    artifact_hashes = {
        "checkpoint_sha256": sha256_file(resolved_checkpoint_path),
        "transition_operator_sha256": sha256_file(resolved_operator_path),
        "transform_bundle_sha256": sha256_file(resolved_bundle_path),
        "compatibility_label_contract_sha256": (
            sha256_file(contract_path) if contract_path.exists() else None
        ),
        "rulebook_sha256": (
            sha256_file(Path(rulebook_path))
            if rulebook_path is not None and Path(rulebook_path).exists()
            else None
        ),
    }
    compatibility_label_set = (
        [str(predicted_label)]
        if rulebook_path is not None and predicted_label
        else (
            []
            if rulebook_path is not None
            else model_compatibility_label_set
        )
    )
    return {
        "predicted_label": predicted_label,
        "compatibility_label_set": compatibility_label_set,
        "semantic_measurements": predicted_features,
        "fired_rules": activated_rules,
        "rule_conflicts": rule_conflicts,
        "route": route,
        "confidence": confidence,
        "calibration_status": (
            "calibrated"
            if compatibility_calibration
            else "uncalibrated_legacy_fallback"
        ),
        "quality_status": (
            str(dss_feature_provenance.get("status"))
            if isinstance(dss_feature_provenance, Mapping)
            else "not_measured_model_only"
        ),
        "abstention_reason": (
            abstention_payload.get("reason")
            if isinstance(abstention_payload, Mapping)
            else None
        ),
        "model_and_contract_hashes": artifact_hashes,
        "decision_mode": "strict_dss" if rulebook_path is not None else "model_only_research",
        "model_predicted_label": model_predicted_label,
        "legacy_head_predicted_label": legacy_predicted_label,
        "compatibility_projection_trace": compatibility_trace,
        "class_scores": class_scores,
        "axis_scores": axis_scores,
        "predicted_features": predicted_features,
        "activated_rules": activated_rules,
        "dss_rulebook_status": rulebook_status,
        "dss_feature_provenance": dss_feature_provenance,
        "dss_result": dss_result,
    }
