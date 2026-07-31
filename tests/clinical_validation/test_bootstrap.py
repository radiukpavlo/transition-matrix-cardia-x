from __future__ import annotations

from tm_ecg.clinical_validation.bootstrap import cluster_bootstrap_kappa


def test_cluster_bootstrap_is_seed_deterministic() -> None:
    reference = ["a", "a", "b", "b", "a", "b"]
    comparison = ["a", "b", "b", "b", "a", "a"]
    cases = [f"C{i}" for i in range(6)]
    clusters = ["1", "1", "2", "2", "3", "3"]
    first, dist_a = cluster_bootstrap_kappa(
        reference, comparison, cases, clusters, replicates=40, seed=2026
    )
    second, dist_b = cluster_bootstrap_kappa(
        reference, comparison, cases, clusters, replicates=40, seed=2026
    )
    assert dist_a == dist_b
    assert first.confidence_interval == second.confidence_interval
    assert first.bootstrap_replicates == 40

