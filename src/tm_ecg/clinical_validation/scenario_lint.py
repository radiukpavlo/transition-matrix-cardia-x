"""Pre-registration and hash lint for the CARDIA-X v3 scenario system."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from tm_ecg.clinical_validation.audit import read_json, sha256_file, write_json
from tm_ecg.clinical_validation.ontology import load_json_yaml
from tm_ecg.clinical_validation.scenarios import load_scenario_registry


REQUIRED_HARMONIZED_IDS = frozenset(
    {
        "harmonized_b1_unique_exact",
        "harmonized_b1_unique_five_family",
        "harmonized_b1_unique_binary",
        "harmonized_b1_unique_dominant",
        "harmonized_b1_axis_rhythm_exact",
        "harmonized_b1_axis_ectopy_exact",
        "harmonized_b1_axis_conduction_exact",
        "harmonized_b1_axis_repolarization_exact",
        "harmonized_b1_axis_rhythm_presence",
        "harmonized_b1_axis_ectopy_presence",
        "harmonized_b1_axis_conduction_presence",
        "harmonized_b1_axis_repolarization_presence",
    }
)
REQUIRED_RAW_IDS = frozenset(
    {
        "raw_b1_unique_exact",
        "raw_b1_unique_five_family",
        "raw_b1_unique_binary",
        "raw_b1_unique_dominant",
        "raw_b1_axis_rhythm",
        "raw_b1_axis_ectopy",
        "raw_b1_axis_conduction",
        "raw_b1_axis_repolarization",
        "raw_b1_axis_pacing",
    }
)
REQUIRED_GROUPS = frozenset(
    {
        "raw_audit",
        "harmonized_required",
        "route_source_slices",
        "diagnostic_sensitivity",
    }
)


def _frozen_hash(path: Path) -> str:
    hash_path = path.with_suffix(".sha256")
    if not hash_path.exists():
        return ""
    return hash_path.read_text(encoding="utf-8").split()[0].strip()


def lint_scenario_registry(
    *,
    registry_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
) -> tuple[int, dict[str, object]]:
    registry_source = Path(registry_path)
    policy_source = Path(policy_path)
    registry_payload, scenarios = load_scenario_registry(registry_source)
    policy = load_json_yaml(policy_source)
    root = policy_source.resolve().parents[2]
    ids = {item.scenario_id for item in scenarios}
    required_ids = {
        item.scenario_id for item in scenarios if item.requirement == "required"
    }
    raw_ids = {
        item.scenario_id for item in scenarios if item.requirement == "audit"
    }
    groups = {item.scenario_group for item in scenarios}
    actual_registry_hash = sha256_file(registry_source)
    recorded_registry_hash = _frozen_hash(registry_source)

    migration_path = root / str(registry_payload.get("migration_record", ""))
    migration = read_json(migration_path) if migration_path.exists() else {}
    predecessor_path = root / str(registry_payload.get("predecessor", ""))
    predecessor_hash_ok = bool(
        predecessor_path.exists()
        and migration.get("from_registry_sha256")
        == sha256_file(predecessor_path)
    )
    migrated = migration.get("required_scenario_migrations", {})
    migrated_map = migrated if isinstance(migrated, Mapping) else {}
    predecessor_payload = (
        load_json_yaml(predecessor_path) if predecessor_path.exists() else {}
    )
    predecessor_scenarios = predecessor_payload.get("scenarios", [])
    predecessor_required = {
        str(item.get("scenario_id"))
        for item in predecessor_scenarios
        if isinstance(item, Mapping) and item.get("requirement") == "required"
    }
    unmigrated_required = sorted(predecessor_required - set(migrated_map))

    expected_hashes = {
        "raw_immutable_v2": {
            "projection_contract_hash": sha256_file(
                root / str(policy["raw_ontology"])
            ),
            "physician_coder_hash": sha256_file(
                root / str(policy["raw_physician_rules"])
            ),
            "benchmark_mapping_hash": sha256_file(
                root / str(policy["raw_benchmark_mapping"])
            ),
        },
        "harmonized_v3": {
            "projection_contract_hash": sha256_file(
                root / str(policy["ontology"])
            ),
            "physician_coder_hash": sha256_file(
                root / str(policy["physician_rules"])
            ),
            "benchmark_mapping_hash": sha256_file(
                root / str(policy["benchmark_mapping"])
            ),
        },
    }
    contract_hash_mismatches: list[dict[str, str]] = []
    for scenario in scenarios:
        expected = expected_hashes.get(scenario.endpoint_version, {})
        for field, expected_hash in expected.items():
            actual_hash = str(getattr(scenario, field))
            if actual_hash != expected_hash:
                contract_hash_mismatches.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "field": field,
                        "expected": expected_hash,
                        "actual": actual_hash,
                    }
                )

    primary = [
        item
        for item in scenarios
        if item.requirement in {"required", "audit"}
        and item.population == "b1_unique"
        and item.route is None
    ]
    primary_hashes = {item.required_case_ids_hash for item in primary}
    primary_case_contract_ok = (
        len(primary_hashes) == 1
        and None not in primary_hashes
        and all(item.required_case_count == 100 for item in primary)
        and all(item.minimum_sample_size == 100 for item in primary)
    )
    threshold_errors = sorted(
        item.scenario_id
        for item in scenarios
        if item.requirement == "required"
        and (
            (
                item.scenario_id == "harmonized_b1_unique_exact"
                and item.minimum_kappa != 0.7
            )
            or (
                item.scenario_id != "harmonized_b1_unique_exact"
                and item.minimum_kappa != 0.6
            )
        )
    )

    checks = {
        "registry_declares_frozen": registry_payload.get("frozen") is True,
        "registry_hash_matches_frozen_record": (
            bool(recorded_registry_hash)
            and recorded_registry_hash == actual_registry_hash
        ),
        "all_required_harmonized_ids_present": (
            required_ids == REQUIRED_HARMONIZED_IDS
        ),
        "all_raw_audit_ids_present": raw_ids == REQUIRED_RAW_IDS,
        "all_four_scenario_groups_present": REQUIRED_GROUPS.issubset(groups),
        "predecessor_hash_matches_migration": predecessor_hash_ok,
        "no_required_predecessor_scenario_unmigrated": not unmigrated_required,
        "contract_hashes_match_files": not contract_hash_mismatches,
        "primary_case_identity_contract_is_uniform": primary_case_contract_ok,
        "required_thresholds_are_consistent": not threshold_errors,
        "scenario_ids_are_unique": len(ids) == len(scenarios),
        "required_non_estimable_policy_is_fail": (
            policy.get("non_estimable_required_scenario") == "fail"
        ),
    }
    report = {
        "version": 1,
        "registry": str(registry_source),
        "registry_sha256": actual_registry_hash,
        "recorded_registry_sha256": recorded_registry_hash,
        "policy": str(policy_source),
        "policy_sha256": sha256_file(policy_source),
        "scenario_count": len(scenarios),
        "required_ids": sorted(required_ids),
        "raw_audit_ids": sorted(raw_ids),
        "scenario_groups": sorted(groups),
        "unmigrated_predecessor_required_ids": unmigrated_required,
        "contract_hash_mismatches": contract_hash_mismatches,
        "threshold_errors": threshold_errors,
        "checks": checks,
        "passed": all(checks.values()),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = write_json(destination / "scenario_registry_gate.json", report)
    return (0 if report["passed"] else 19), {
        "report": str(output_path),
        "passed": report["passed"],
    }

