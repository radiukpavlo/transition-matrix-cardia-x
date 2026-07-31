# Reported result snapshot

`reported_metrics.json` contains only aggregate quantities reported in the CARDIA-X study. It is intended for numerical verification, figure regeneration, and comparison with clean reruns of the public-dataset pipelines.

The snapshot does not contain raw ECG waveforms, patient identifiers, reader identities, case-level reader responses, free-text clinical comments, trained models, development experiments, or unpublished intermediate results.

Run the verifier from the repository root:

```bash
tm-ecg-verify-reported
```

Cluster-bootstrap confidence limits are retained as reported values. Recomputing them requires the corresponding row-level predictions or timing errors produced by the full public-dataset workflows.
