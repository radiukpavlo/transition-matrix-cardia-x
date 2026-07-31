# Reported result snapshot

`reported_metrics.json` is the curated aggregate snapshot associated with the CARDIA-X manuscript:

> Pavlo Radiuk, Oleksander Barmak, Liliana Klymenko, and Iurii Krak. “CARDIA-X: Global Semantic Transition and Rough-Set Rules for Auditable Post-hoc Electrocardiographic Explainability.” Unpublished manuscript, 2026.

It is intended for deterministic numerical verification, figure regeneration, and comparison with clean reruns of the public-dataset pipelines. The snapshot preserves favorable, unfavorable, and non-estimable findings without turning the repository into a clinical product.

## Verification

Run these commands from the repository root after installing the locked environment:

```bash
tm-ecg-verify-manifest
tm-ecg-verify-reported
```

The first command verifies the public file ledger. The second independently recalculates the deterministic aggregate quantities and reports the registered PTB-XL, reader-audit, transition, and LUDB counts.

Cluster-bootstrap confidence limits are retained as reported values. Recomputing them requires the corresponding row-level predictions or timing errors produced by the full public-dataset workflows.

The snapshot does not contain raw ECG waveforms, patient identifiers, reader identities, case-level reader responses, free-text clinical comments, trained models, development experiments, or unpublished intermediate results. See the root [README](../README.md) for the complete evidence boundary and reproduction protocol.
