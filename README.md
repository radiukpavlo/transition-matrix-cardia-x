# CARDIA-X: transition-matrix-ecg

This repository contains the source code and curated reported-result snapshot for CARDIA-X, a research framework that maps frozen electrocardiogram (ECG) representations to clinician-facing semantic features and conflict-aware rough-set rules.

It is designed to reproduce the computational results reported in the study **“CARDIA-X: Global Semantic Transition and Rough-Set Rules for Auditable Post-hoc Electrocardiographic Explainability.”** The implementation covers PTB-XL/PTB-XL+ compatibility classification, ECG preprocessing and fiducial analysis, semantic transition estimation, WEDD discretization, rough-set rule induction, route-aware explanation, and aggregate reader-audit calculations.

## Evidence boundary

The repository deliberately contains:

- the computational implementation used by the reported analyses;
- locked configuration and ontology contracts;
- unit and regression tests;
- a machine-readable snapshot of the final aggregate results reported in the study; and
- a verifier that independently recalculates deterministic quantities from the reported counts.

It deliberately excludes manuscript source files and PDFs, raw ECG datasets, trained model binaries, reader workbooks, case-level reader responses, exploratory model comparisons, temporary outputs, and publication build records.

PTB-XL, PTB-XL+, and LUDB must be obtained from PhysioNet under their respective terms:

- [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL+ v1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- [LUDB v1.0.1](https://physionet.org/content/ludb/1.0.1/)

## Reported findings

The public snapshot preserves favorable and unfavorable findings without changing endpoints:

- On 2,198 patient-disjoint PTB-XL test records, exact multilabel subset accuracy was 0.741128 (95% patient-cluster bootstrap CI, 0.722420–0.760074), micro-F1 was 0.813226 (0.799528–0.827875), and per-label bitwise accuracy was 0.952027 (0.948460–0.955887).
- Performance was heterogeneous. Atrial fibrillation F1 was 0.875000, while PVC, APB, and AFL F1 values were 0.170543, 0.240000, and 0.250000, respectively.
- Strict observed-target masked B1 semantic-reconstruction MAE was 0.571700 on the test split (0.562215–0.581335).
- At a 100 ms tolerance, LUDB R-peak detection produced 1,761 true positives, 267 false positives, and 56 false negatives; F1 was 0.915995 (0.904811–0.925566).
- The historical formula-linked endpoint in the retrospective 100-case, single-physician audit yielded Cohen’s kappa of 0.683067 (0.558895–0.796751).
- No assisted diagnosis changed among 99 eligible cases. None of 40 structural abstentions was endorsed as clinically appropriate, six of ten reviewed rule artifacts received a medical-soundness score of 3/5, and reconstructed QRS morphology was judged plausible in 0/40 B2 cases.

These findings support use of CARDIA-X as a research audit layer. They do not establish prospective clinical benefit, safe autonomous deferral, multicenter generalizability, or readiness for deployment.

## Quick start

The reference environment uses Python 3.13.6. Python 3.12 and 3.13 are supported by the package metadata.

```bash
python -m venv .venv
```

Activate the environment, then install the locked dependencies and package:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

Verify the curated reported results:

```bash
tm-ecg-verify-reported
```

Run the regression suite:

```bash
python -m pytest
```

## Dataset setup

Place the three unmodified PhysioNet archives in `data/archives/` using the filenames configured in `configs/defaults.toml`:

```text
data/archives/
├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
├── ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip
└── lobachevsky-university-electrocardiography-database-1.0.1.zip
```

The pipeline uses the recommended PTB-XL folds: 1–8 for training, 9 for validation, and 10 for testing. Dataset archives are never modified.

## Reproducing the computational analyses

Run the PTB-XL/PTB-XL+ branch:

```bash
invoke pipeline --dataset=b1
```

Run the LUDB branch:

```bash
invoke pipeline --dataset=b2
```

The two workflows execute ingestion, indexing, patient-aware splits, filtering, R-peak detection, fiducial delineation, triad extraction, semantic-feature construction, transition fitting, explanation generation, and final metric reporting. The B1 workflow also trains and evaluates the compatibility head; the B2 workflow performs LUDB fiducial validation.

The complete runs are computationally intensive and require local storage for the PhysioNet archives and derived Parquet artifacts. Intermediate run products are written below `data/` and `artifacts/` and are ignored by Git.

A workflow may finish its evidence-producing stages and still return a nonzero status when a registered scientific eligibility gate fails. This behavior preserves negative findings and prevents a failed rulebook or reporting gate from being mistaken for a passing release.

## Reported-result verification

`results/reported_metrics.json` is the only bundled result artifact. It contains the aggregate quantities shown in the study, including denominators, confidence intervals, class-specific confusion counts, semantic-reconstruction coverage, LUDB timing results, reader-audit endpoints, and the five prespecified public PTB-XL visualization records.

`tm-ecg-verify-reported` independently recalculates:

- exact subset accuracy, micro-F1, bitwise accuracy, and per-class precision/recall/F1;
- Cohen’s kappa, observed agreement, expected agreement, and the maximum attainable kappa from the aggregate contingency counts;
- Wilson score intervals for count-based endpoints;
- semantic-target coverage;
- LUDB R-peak sensitivity, positive predictive value, and F1;
- ordinal-rating means; and
- the aggregate rule-review score.

Patient- and record-cluster bootstrap intervals require row-level predictions or timing errors and are reproduced by the full public-dataset workflows. The bundled verifier checks their registered values and bounds but does not reconstruct them from insufficient aggregate statistics.

## Repository structure

```text
clinical_validation/config/  Versioned reader-audit coding and scenario contracts
configs/                     Dataset, ontology, compatibility, and semantic-target contracts
results/                     Curated aggregate results reported in the study
schemas/                     Machine-readable output schemas
src/tm_ecg/                  Pipeline, modeling, transition, rule, and audit implementation
tests/                       Unit and regression tests
tasks.py                     Reproduction task entry points
```

## Research-use statement

CARDIA-X is research software. It is not a medical device, does not provide medical advice, and must not be used for diagnosis, treatment, triage, or autonomous clinical decision-making. The retrospective reader audit involved one physician and did not evaluate a live clinical workflow or patient outcomes.

## License and citation

The source code is released under the MIT License. Dataset files remain governed by their original PhysioNet licenses and terms. Citation metadata are provided in `CITATION.cff`.
